"""Testes unitários do adapter Codex e de seu contrato opcional."""

from __future__ import annotations

import contextlib
import dataclasses
import errno
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from datetime import date, datetime, timedelta
from datetime import time as time_type
from ipaddress import IPv4Address, IPv6Address
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Annotated, Any, Literal
from unittest.mock import MagicMock, call, patch
from uuid import UUID

import pandas as pd
import pytest
from pydantic import UUID4, AnyUrl, BaseModel, ConfigDict, Field, RootModel
from pydantic.errors import PydanticInvalidForJsonSchema

from dataframeit import codex as codex_module
from dataframeit.codex import (
    _CATALOG_TOOL_FIELDS,
    _READER_THREAD_JOIN,
    CodexBackend,
    _AppServerStop,
    _build_schema,
    _close_client,
    _raise_classified_sdk_error,
    _to_strict_json_schema,
    _toml_string,
    _turn_timeout,
    _validate_config,
    open_codex_backend,
)
from dataframeit.core import dataframeit
from dataframeit.errors import (
    CODEX_FILE_AUTH_LOGIN_COMMAND,
    ProviderAbortError,
    ProviderConfigurationError,
    ProviderError,
    ProviderOutputError,
    ProviderOverloadedError,
    ProviderTransientError,
    ProviderUsageLimitError,
    get_friendly_error_message,
    is_rate_limit_error,
    is_recoverable_error,
    validate_provider_dependencies,
)
from dataframeit.llm import LLMConfig

try:
    from openai_codex.types import ReasoningEffort as SdkReasoningEffort
except ImportError:  # sem o extra codex, os casos gerados pelo enum ficam vazios
    SdkReasoningEffort = None

DECLARED_EFFORTS = list(SdkReasoningEffort) if SdkReasoningEffort is not None else []


class SampleModel(BaseModel):
    sentimento: str
    confianca: float


class NestedModel(BaseModel):
    label: str


class ModelWithRefSibling(BaseModel):
    nested: Annotated[NestedModel, Field(description="Nested value")]


class ModelWithArray(BaseModel):
    items: list[NestedModel]


class ModelWithAnyOf(BaseModel):
    value: str | int
    note: str | None = None


class ModelWithDefault(BaseModel):
    label: str = "fallback"


class CatModel(BaseModel):
    kind: Literal["cat"]
    lives: int


class DogModel(BaseModel):
    kind: Literal["dog"]
    barks: bool


class ModelWithDiscriminatedUnion(BaseModel):
    animal: Annotated[CatModel | DogModel, Field(discriminator="kind")]


class ModelWithDynamicKeys(BaseModel):
    values: dict[str, str]


class ModelWithFixedTuple(BaseModel):
    pair: tuple[str, int]


class ModelWithSet(BaseModel):
    tags: set[str]


class ModelWithAny(BaseModel):
    value: Any


class ModelWithListAny(BaseModel):
    values: list[Any]


class ModelWithCallable(BaseModel):
    callback: Callable


class ListRootModel(RootModel[list[str]]):
    pass


class RecursiveModel(BaseModel):
    name: str
    child: RecursiveModel | None = None


def make_config(**overrides) -> LLMConfig:
    values: dict[str, Any] = {
        "model": "gpt-6-luna",
        "provider": "codex",
        "api_key": None,
        "max_retries": 2,
        "base_delay": 0,
        "max_delay": 0,
        "rate_limit_delay": 0,
        "model_kwargs": {},
        "search_config": None,
    }
    values.update(overrides)
    return LLMConfig(**values)


def auth_lock_is_available(lock_path: Path) -> bool:
    """Consulta o lock em outro processo, onde o estado do SO é independente."""
    probe = subprocess.run(  # noqa: S603 (roda o próprio interpretador com código fixo do teste)
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from filelock import FileLock, Timeout; "
                "lock = FileLock(sys.argv[1], timeout=0); "
                "\ntry:\n lock.acquire()\nexcept Timeout:\n raise SystemExit(73)\n"
                "else:\n lock.release()"
            ),
            os.fspath(lock_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert probe.returncode in (0, 73), probe.stderr
    return probe.returncode == 0


@pytest.fixture
def codex_sdk():
    """Carrega o SDK real apenas nos testes que exercitam sua fronteira.

    O filelock é importado aqui, antes de qualquer patch do teste, porque a sua
    importação roda uma sonda que usa tempfile.TemporaryDirectory e os.link; os
    patches desses nomes valem para o processo inteiro e capturariam a sonda.
    """
    pytest.importorskip("filelock")
    sdk = pytest.importorskip("openai_codex")
    sdk_types = pytest.importorskip("openai_codex.types")
    generated = pytest.importorskip("openai_codex.generated.v2_all")
    return sdk, sdk_types, generated


def make_result(  # noqa: PLR0913 (um parâmetro por parte do evento simulado)
    codex_sdk,
    response: str | None = '{"sentimento": "positivo", "confianca": 0.9}',
    *,
    status=None,
    usage: bool = True,
    error_info=None,
    message: str = "falhou",
    turn_id: str = "turn-1",
):
    """Eventos do stream de um turno, na ordem em que o app-server os envia.

    Com `error_info`, o turno termina `failed` com o erro tipado que o runtime
    manda em `turn/completed`.
    """
    _, sdk_types, generated = codex_sdk
    from openai_codex.models import Notification  # noqa: PLC0415 (SDK carregado pelo fixture)

    events = []
    if response is not None:
        item = generated.ThreadItem(
            root=generated.AgentMessageThreadItem(
                id="item-1",
                text=response,
                type="agentMessage",
                phase=generated.MessagePhase.final_answer,
            )
        )
        events.append(
            Notification(
                method="item/completed",
                payload=generated.ItemCompletedNotification(
                    completedAtMs=1, item=item, threadId="thread-1", turnId=turn_id
                ),
            )
        )
    if usage:
        token_usage = generated.TokenUsageBreakdown(
            inputTokens=100,
            cachedInputTokens=40,
            outputTokens=30,
            reasoningOutputTokens=10,
            totalTokens=130,
        )
        events.append(
            Notification(
                method="thread/tokenUsage/updated",
                payload=generated.ThreadTokenUsageUpdatedNotification(
                    threadId="thread-1",
                    turnId=turn_id,
                    tokenUsage=sdk_types.ThreadTokenUsage(last=token_usage, total=token_usage),
                ),
            )
        )
    if status is None:
        status = sdk_types.TurnStatus.failed if error_info else sdk_types.TurnStatus.completed
    error = (
        sdk_types.TurnError(message=message, codexErrorInfo=error_info)
        if error_info is not None
        else None
    )
    events.append(
        Notification(
            method="turn/completed",
            payload=generated.TurnCompletedNotification(
                threadId="thread-1",
                turn=sdk_types.Turn(id=turn_id, items=[], status=status, error=error),
            ),
        )
    )
    return events


def as_stream(events):
    """Gerador, como `TurnHandle.stream`, para que o backend possa fechá-lo."""
    yield from events


def threads_descarregadas(client) -> list[str]:
    """Ids das threads cujo `thread/unsubscribe` o backend pediu, na ordem."""
    return [
        chamada.args[1]["threadId"]
        for chamada in client._client.request.call_args_list
        if chamada.args[0] == "thread/unsubscribe"
    ]


def aguardar_descargas() -> None:
    """Espera as threads daemon que pedem `thread/unsubscribe` terminarem."""
    for pedido in threading.enumerate():
        if "_unsubscribe_quietly" in pedido.name:
            pedido.join(5)


def initialized_backend(tmp_path, codex_sdk, result=None, turn_timeout=None):
    sdk, sdk_types, _ = codex_sdk
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    events = result if result is not None else make_result(codex_sdk)
    turn = MagicMock(spec=sdk.TurnHandle)
    turn.id = "turn-1"
    turn.stream.side_effect = lambda: as_stream(events)
    thread = MagicMock(spec=sdk.Thread)
    thread.id = "thread-1"
    thread.turn.return_value = turn
    client = MagicMock(spec=sdk.Codex)
    client._client = MagicMock()
    client.thread_start.return_value = thread
    config = make_config()
    backend = CodexBackend(
        config=config,
        _pydantic_model=SampleModel,
        _user_prompt="Analise: {texto}",
        _schema=_build_schema(SampleModel),
        _effort=sdk_types.ReasoningEffort.medium,
        _client=client,
        _workspace=workspace,
        _turn_timeout=turn_timeout,
    )
    return backend, client, thread, turn


def as_context_manager(client, resolved_model="gpt-6-luna"):
    """Configura o mock com o mesmo contrato de contexto do SDK real.

    O cliente de protocolo responde à thread de sonda com o modelo resolvido,
    que por padrão é o de `make_config`.
    """
    from openai_codex.generated.v2_all import (  # noqa: PLC0415 (SDK carregado pelo fixture)
        Thread,
        ThreadStartResponse,
    )

    client.__enter__.return_value = client
    client._client = MagicMock()
    client._client.thread_start.return_value = ThreadStartResponse.model_construct(
        model=resolved_model, thread=Thread.model_construct(id="thread-sonda")
    )

    def close_without_suppressing(*_):
        client.close()
        return False

    client.__exit__.side_effect = close_without_suppressing
    return client


class TestProviderDependency:
    def test_codex_auth_hint_uses_file_backed_login_command(self):
        message = get_friendly_error_message(RuntimeError("AuthenticationError"), "codex")

        assert CODEX_FILE_AUTH_LOGIN_COMMAND in message

    def test_missing_sdk_reports_only_codex_extra(self):

        with (
            patch("importlib.import_module", side_effect=ImportError("missing")),
            pytest.raises(ImportError) as exc_info,
        ):
            validate_provider_dependencies("codex")

        message = str(exc_info.value)
        assert "dataframeit[codex]" in message
        assert "dataframeit[all]" not in message

    def test_langchain_provider_keeps_all_extra_as_alternative(self):

        def import_module(name):
            if name == "langchain_google_genai":
                msg = "missing"
                raise ImportError(msg)
            return MagicMock()

        with (
            patch("importlib.import_module", side_effect=import_module),
            pytest.raises(ImportError) as exc_info,
        ):
            validate_provider_dependencies("google_genai")

        message = str(exc_info.value)
        assert "langchain-google-genai" in message
        assert "dataframeit[all]" in message

    def test_sdk_provider_skips_langchain_validation(self):

        imported = []

        def import_module(name):
            imported.append(name)
            return MagicMock()

        with patch("importlib.import_module", side_effect=import_module):
            validate_provider_dependencies("codex")

        assert imported == ["openai_codex"]


class TestStrictPydanticSchema:
    def test_refs_with_sibling_metadata_are_expanded_and_strict(self):
        schema = _to_strict_json_schema(ModelWithRefSibling.model_json_schema())

        assert schema["additionalProperties"] is False
        assert schema["required"] == ["nested"]
        assert schema["$defs"]["NestedModel"]["additionalProperties"] is False
        nested = schema["properties"]["nested"]
        assert "$ref" not in nested
        assert nested["description"] == "Nested value"
        assert nested["additionalProperties"] is False
        assert nested["required"] == ["label"]

    def test_arrays_keep_internal_refs_and_make_definitions_strict(self):
        schema = _to_strict_json_schema(ModelWithArray.model_json_schema())

        item = schema["properties"]["items"]["items"]
        assert item == {"$ref": "#/$defs/NestedModel"}
        assert schema["$defs"]["NestedModel"]["additionalProperties"] is False
        assert schema["$defs"]["NestedModel"]["required"] == ["label"]

    def test_any_of_nullable_removes_default_and_requires_every_property(self):
        schema = _to_strict_json_schema(ModelWithAnyOf.model_json_schema())

        assert schema["required"] == ["value", "note"]
        assert schema["properties"]["value"]["anyOf"] == [
            {"type": "string"},
            {"type": "integer"},
        ]
        note = schema["properties"]["note"]
        assert "default" not in note
        assert note["anyOf"] == [{"type": "string"}, {"type": "null"}]

    def test_non_null_default_is_removed_and_property_becomes_required(self):
        schema = _to_strict_json_schema(ModelWithDefault.model_json_schema())

        assert schema["required"] == ["label"]
        assert "default" not in schema["properties"]["label"]

    def test_discriminated_one_of_becomes_supported_any_of(self):
        schema = _to_strict_json_schema(ModelWithDiscriminatedUnion.model_json_schema())

        animal = schema["properties"]["animal"]
        assert "oneOf" not in animal
        assert "discriminator" not in animal
        assert animal["anyOf"] == [
            {"$ref": "#/$defs/CatModel"},
            {"$ref": "#/$defs/DogModel"},
        ]
        assert schema["$defs"]["CatModel"]["additionalProperties"] is False
        assert schema["$defs"]["DogModel"]["additionalProperties"] is False

    def test_dynamic_dict_is_rejected_from_real_pydantic_schema(self):
        with pytest.raises(ProviderConfigurationError, match="chaves dinâmicas"):
            _to_strict_json_schema(ModelWithDynamicKeys.model_json_schema())

    @pytest.mark.parametrize(
        ("model", "keyword"),
        [
            (ModelWithFixedTuple, "prefixItems"),
            (ModelWithSet, "uniqueItems"),
        ],
    )
    def test_unsupported_pydantic_keywords_are_rejected(self, model, keyword):
        with pytest.raises(ProviderConfigurationError, match=keyword):
            _to_strict_json_schema(model.model_json_schema())

    def test_anotacoes_de_json_schema_extra_sao_descartadas_com_aviso(self):
        """Chave fora do vocabulário do JSON Schema não valida nada e não bloqueia."""

        class Anotado(BaseModel):
            q1: Literal["Sim", "Não"] = Field(
                description="Pergunta",
                json_schema_extra={
                    "help_text": "ajuda",
                    "condition": {"field": "q0", "equals": "Sim"},
                    "target": "all",
                },
            )
            q2: NestedModel = Field(description="Aninhado", json_schema_extra={"allowOther": True})

        with pytest.warns(UserWarning, match="allowOther, condition, help_text, target"):
            schema = _to_strict_json_schema(Anotado.model_json_schema())

        q1 = schema["properties"]["q1"]
        assert q1["enum"] == ["Sim", "Não"]
        assert q1["description"] == "Pergunta"
        assert not {"help_text", "condition", "target"} & set(q1)
        q2 = schema["properties"]["q2"]
        assert "allowOther" not in q2
        assert q2["required"] == ["label"]

    def test_metadados_do_vocabulario_sao_descartados_com_aviso(self):
        class ComMetadados(BaseModel):
            valor: str = Field(examples=["a"], deprecated=True)

        with pytest.warns(UserWarning, match="deprecated, examples"):
            schema = _to_strict_json_schema(ComMetadados.model_json_schema())

        assert schema["properties"]["valor"] == {"type": "string", "title": "Valor"}

    def test_schema_sem_anotacao_nao_avisa(self, recwarn):
        _to_strict_json_schema(SampleModel.model_json_schema())

        assert not [w for w in recwarn if "descartadas" in str(w.message)]

    @pytest.mark.parametrize(
        ("campo", "keyword"),
        [
            (Field(max_length=10), "maxLength"),
            (Field(min_length=1), "minLength"),
        ],
    )
    def test_keyword_de_validacao_nao_suportada_continua_recusada(self, campo, keyword):
        """Descartar uma restrição afrouxaria o contrato que o modelo declara."""

        class Restrito(BaseModel):
            valor: str = campo

        with pytest.raises(ProviderConfigurationError, match=keyword):
            _to_strict_json_schema(Restrito.model_json_schema())

    @pytest.mark.parametrize(
        ("tipo", "formato"),
        [(bytes, "binary"), (Path, "path"), (AnyUrl, "uri"), (UUID4, "uuid4")],
    )
    def test_format_fora_do_subconjunto_e_recusado(self, tipo, formato):
        """Sem a conferência, cada linha falharia no turno, e não no preflight."""

        class Modelo(BaseModel):
            valor: tipo

        schema = Modelo.model_json_schema()
        schema["properties"]["valor"].pop("minLength", None)
        with pytest.raises(ProviderConfigurationError, match=f"format '{formato}'"):
            _to_strict_json_schema(schema)

    @pytest.mark.parametrize(
        ("tipo", "formato"),
        [
            (date, "date"),
            (datetime, "date-time"),
            (time_type, "time"),
            (timedelta, "duration"),
            (UUID, "uuid"),
            (IPv4Address, "ipv4"),
            (IPv6Address, "ipv6"),
        ],
    )
    def test_format_do_subconjunto_e_mantido(self, tipo, formato):
        class Modelo(BaseModel):
            valor: tipo

        schema = _to_strict_json_schema(Modelo.model_json_schema())

        assert schema["properties"]["valor"]["format"] == formato

    @pytest.mark.parametrize("formato", [["date"], None, 1])
    def test_format_que_nao_e_texto_e_recusado(self, formato):
        schema = {
            "type": "object",
            "properties": {"valor": {"type": "string", "format": formato}},
        }

        with pytest.raises(ProviderConfigurationError, match="não é suportado"):
            _to_strict_json_schema(schema)

    def test_pattern_do_decimal_e_recusado_pelo_lookahead(self):
        # O pattern que o pydantic emite para Decimal até a 2.13, escrito aqui
        # porque da 2.14 em diante o schema do tipo não traz pattern.
        schema = {
            "type": "object",
            "properties": {
                "valor": {
                    "anyOf": [
                        {"type": "number"},
                        {"type": "string", "pattern": r"^(?!^[-+.]*$)[+-]?0*\d*\.?\d*$"},
                    ]
                }
            },
        }

        with pytest.raises(ProviderConfigurationError, match="lookahead"):
            _to_strict_json_schema(schema)

    @pytest.mark.parametrize(
        ("pattern", "construcao"),
        [
            (r"^(?!x)\w+$", "lookahead"),
            (r"^\w+(?=x)", "lookahead"),
            (r"^.*(?<=x)$", "lookbehind"),
            (r"^.*(?<!x)$", "lookbehind"),
            (r"^(a)\1$", "referência a grupo"),
            (r"^(?P<n>a)(?P=n)$", "referência a grupo"),
            (r"^(?<n>a)\k<n>$", "referência a grupo"),
        ],
    )
    def test_pattern_com_construcao_nao_suportada_e_recusado(self, pattern, construcao):
        schema = {
            "type": "object",
            "properties": {"valor": {"type": "string", "pattern": pattern}},
        }

        with pytest.raises(ProviderConfigurationError, match=construcao):
            _to_strict_json_schema(schema)

    def test_pattern_de_field_com_motor_python_re_e_conferido(self):
        """O motor padrão do pydantic-core já recusa lookaround; o `re` aceita."""

        class Modelo(BaseModel):
            model_config = ConfigDict(regex_engine="python-re")
            valor: str = Field(pattern=r"^(?!x)\w+$")

        with pytest.raises(ProviderConfigurationError, match="lookahead"):
            _to_strict_json_schema(Modelo.model_json_schema())

    @pytest.mark.parametrize(
        "pattern",
        [
            r"^[0-9]{3}-[a-z]+$",
            r"^\(?=x$",
            r"^[(?=]+$",
            r"^[]\1(?<=]+$",
            r"^[^](?!]+$",
            r"^(?:ab)+\0?$",
            r"^(?<nome>a)b$",
            "^a\\\\$",
        ],
    )
    def test_pattern_sem_construcao_recusada_e_mantido(self, pattern):
        """Escape e classe de caracteres tornam literal o que pareceria lookaround."""
        schema = {
            "type": "object",
            "properties": {"valor": {"type": "string", "pattern": pattern}},
        }

        strict_schema = _to_strict_json_schema(schema)

        assert strict_schema["properties"]["valor"]["pattern"] == pattern

    def test_pattern_que_nao_e_texto_e_recusado(self):
        schema = {
            "type": "object",
            "properties": {"valor": {"type": "string", "pattern": 1}},
        }

        with pytest.raises(ProviderConfigurationError, match="pattern inválido"):
            _to_strict_json_schema(schema)

    def test_one_of_without_discriminator_is_converted_to_any_of(self):
        schema = {
            "type": "object",
            "properties": {"value": {"oneOf": [{"type": "string"}, {"type": "integer"}]}},
        }

        strict_schema = _to_strict_json_schema(schema)

        assert strict_schema["properties"]["value"]["anyOf"] == [
            {"type": "string"},
            {"type": "integer"},
        ]

    def test_all_of_is_rejected_instead_of_forwarded_to_runtime(self):
        schema = {
            "type": "object",
            "properties": {"value": {"allOf": [{"type": "string"}]}},
        }

        with pytest.raises(ProviderConfigurationError, match="allOf"):
            _to_strict_json_schema(schema)

    @pytest.mark.parametrize("model", [ModelWithAny, ModelWithListAny])
    def test_untyped_any_schema_is_rejected(self, model):
        with pytest.raises(ProviderConfigurationError, match="Any não é suportado"):
            _to_strict_json_schema(model.model_json_schema())

    def test_root_model_is_rejected_before_processing(self):
        with pytest.raises(ProviderConfigurationError, match="RootModel não é suportado"):
            _to_strict_json_schema(ListRootModel.model_json_schema())

    def test_recursive_pydantic_schema_remains_finite_and_strict(self):
        schema = _to_strict_json_schema(RecursiveModel.model_json_schema())

        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert schema["required"] == ["name", "child"]
        child_ref = schema["properties"]["child"]["anyOf"][0]
        assert child_ref == {"$ref": "#/$defs/RecursiveModel"}
        recursive_definition = schema["$defs"]["RecursiveModel"]
        assert recursive_definition["additionalProperties"] is False
        assert recursive_definition["properties"]["child"]["anyOf"][0] == child_ref


class TestBackendConfiguration:
    def test_invalid_pydantic_json_schema_is_configuration_error(self):
        with pytest.raises(
            ProviderConfigurationError,
            match="Não foi possível gerar JSON Schema",
        ) as exc_info:
            _build_schema(ModelWithCallable)

        assert isinstance(exc_info.value.__cause__, PydanticInvalidForJsonSchema)

    def test_effort_defaults_to_real_medium_enum(self, codex_sdk):
        _, sdk_types, _ = codex_sdk

        effort = _validate_config(make_config())

        assert effort is sdk_types.ReasoningEffort.medium

    @pytest.mark.parametrize("member", DECLARED_EFFORTS, ids=lambda member: member.value)
    @pytest.mark.parametrize("as_text", [True, False], ids=["texto", "membro"])
    def test_every_declared_effort_is_accepted(self, codex_sdk, member, as_text):
        value = member.value if as_text else member

        effort = _validate_config(make_config(model_kwargs={"effort": value}))

        assert effort is member

    def test_effort_member_created_by_open_enum_is_rejected(self, codex_sdk):
        _, sdk_types, _ = codex_sdk
        bogus = sdk_types.ReasoningEffort("bogus")
        assert bogus.value not in {member.value for member in sdk_types.ReasoningEffort}

        with pytest.raises(ProviderConfigurationError, match="effort inválido"):
            _validate_config(make_config(model_kwargs={"effort": bogus}))

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"api_key": "secret"}, "não passe api_key"),
            ({"model_kwargs": {"temperature": 0}}, "temperature"),
            ({"model_kwargs": {"codex_bin": "/some/codex"}}, "codex_bin"),
            ({"model_kwargs": {"effort": "maximum"}}, "effort inválido"),
            ({"model_kwargs": {"effort": "LOW"}}, "effort inválido"),
            ({"model_kwargs": {"effort": None}}, "effort inválido"),
        ],
    )
    def test_invalid_config_fails_before_client_start(self, codex_sdk, overrides, message):

        with (
            patch.object(codex_module, "_open_client") as codex,
            pytest.raises(ProviderConfigurationError, match=message),
        ):
            _validate_config(make_config(**overrides))

        codex.assert_not_called()


@pytest.mark.parametrize(
    "value",
    ["/srv/codex", 'C:\\Users\\Zé\\a"b', "/srv/códex 😀", "/srv/a\x7fb", "/srv/a\x01b"],
)
def test_toml_string_volta_ao_mesmo_valor(value):
    tomllib = pytest.importorskip("tomllib")

    assert tomllib.loads(f"k = {_toml_string(value)}")["k"] == value


class TestBackendLifecycle:
    def test_uses_bundled_runtime_in_isolated_home_and_cleans_up(
        self, codex_sdk, monkeypatch, tmp_path
    ):
        sdk, sdk_types, _ = codex_sdk
        source_home = tmp_path / "source-home"
        source_home.mkdir()
        source_auth = source_home / "auth.json"
        source_auth.write_text("{}")
        (source_home / "config.toml").write_text('[mcp_servers.unsafe]\ncommand="unsafe"\n')
        monkeypatch.setenv("CODEX_HOME", str(source_home))

        client = as_context_manager(MagicMock(spec=sdk.Codex))
        client.account.return_value = sdk_types.GetAccountResponse(requiresOpenaiAuth=False)

        with (
            patch("dataframeit.codex.os.link", wraps=os.link) as hard_link,
            patch.object(
                Path,
                "symlink_to",
                side_effect=AssertionError("symlink não deve ser usado"),
            ) as symlink,
            patch.object(codex_module, "_open_client", return_value=client) as codex,
        ):
            with open_codex_backend(make_config(), SampleModel, "{texto}") as backend:
                launch_config = codex.call_args.args[0]
                assert isinstance(launch_config, sdk.CodexConfig)
                assert launch_config.codex_bin is None
                workspace = Path(launch_config.cwd)
                isolated_home = Path(launch_config.env["CODEX_HOME"])
                isolated_auth = isolated_home / "auth.json"
                assert isolated_home.parent == workspace.parent
                assert workspace.parent.parent == source_home
                assert launch_config.env["CODEX_SQLITE_HOME"] == str(isolated_home)
                assert launch_config.env["HOME"] == str(isolated_home)
                assert isolated_home != source_home
                assert not isolated_auth.is_symlink()
                assert Path(isolated_auth).samefile(source_auth)
                lock_path = source_home / "auth.json.dataframeit.lock"
                assert not auth_lock_is_available(lock_path)

                with (
                    pytest.raises(ProviderConfigurationError, match="Outra execução"),
                    open_codex_backend(make_config(), SampleModel, "{texto}"),
                ):
                    pass
                assert codex.call_count == 1

                def close_while_lock_is_held():
                    assert not auth_lock_is_available(lock_path)

                client.close.side_effect = close_while_lock_is_held
                isolated_auth.write_text('{"updated": true}')
                assert source_auth.read_text() == '{"updated": true}'
                assert not (isolated_home / "config.toml").exists()
                assert 'cli_auth_credentials_store="file"' in launch_config.config_overrides
                assert "project_doc_max_bytes=0" in launch_config.config_overrides
                assert "mcp_servers={}" in launch_config.config_overrides
                assert "features.shell_tool=false" in launch_config.config_overrides
                assert not any(
                    "model_reasoning_effort" in item for item in launch_config.config_overrides
                )
                assert backend._client is client

            # O patch troca `os.link` no módulo `os`, compartilhado pelo processo, e
            # o filelock 4.x cria um hard link de sonda ao ser importado dentro do
            # backend. Conta-se só o link que aponta para o home isolado.
            links_para_home_isolado = [
                chamada
                for chamada in hard_link.call_args_list
                if Path(chamada.args[1]).parent == isolated_home
            ]
            assert links_para_home_isolado == [call(source_auth.resolve(), isolated_auth)]
            symlink.assert_not_called()

        client.close.assert_called_once_with()
        assert auth_lock_is_available(lock_path)
        assert not workspace.parent.exists()

    def test_app_server_le_o_catalogo_gravado_no_runtime(self, codex_sdk, monkeypatch, tmp_path):
        sdk, sdk_types, _ = codex_sdk
        source_home = tmp_path / "source-home"
        source_home.mkdir()
        (source_home / "auth.json").write_text("{}")
        monkeypatch.setenv("CODEX_HOME", str(source_home))
        client = as_context_manager(MagicMock(spec=sdk.Codex))
        client.account.return_value = sdk_types.GetAccountResponse(requiresOpenaiAuth=False)

        with (
            patch.object(codex_module, "_open_client", return_value=client) as codex,
            open_codex_backend(make_config(), SampleModel, "{texto}") as backend,
        ):
            catalog_override = codex.call_args.args[0].config_overrides[-1]
            assert catalog_override.startswith("model_catalog_json=")
            catalog_path = Path(json.loads(catalog_override.removeprefix("model_catalog_json=")))
            assert catalog_path.parent == backend._workspace.parent
            models = json.loads(catalog_path.read_text(encoding="utf-8"))["models"]

        assert models
        for model in models:
            for field, value in _CATALOG_TOOL_FIELDS.items():
                assert model[field] == value
        assert not catalog_path.exists()

    def test_catalogo_e_lido_sem_codex_api_key(self, codex_sdk, monkeypatch, tmp_path):
        # O app-server do turno ignora CODEX_API_KEY; com ela, `debug models` leria
        # o catálogo da API com essa chave, e não com a credencial dos turnos.
        sdk, sdk_types, _ = codex_sdk
        source_home = tmp_path / "source-home"
        source_home.mkdir()
        (source_home / "auth.json").write_text("{}")
        monkeypatch.setenv("CODEX_HOME", str(source_home))
        monkeypatch.setenv("CODEX_API_KEY", "sk-falsa")
        monkeypatch.setenv("DATAFRAMEIT_VARIAVEL_DO_USUARIO", "mantida")
        client = as_context_manager(MagicMock(spec=sdk.Codex))
        client.account.return_value = sdk_types.GetAccountResponse(requiresOpenaiAuth=False)
        listing = subprocess.CompletedProcess(
            args=[], returncode=0, stdout='{"models": []}', stderr=""
        )

        with (
            patch("dataframeit.codex.subprocess.run", return_value=listing) as run,
            patch.object(codex_module, "_open_client", return_value=client),
            open_codex_backend(make_config(), SampleModel, "{texto}") as backend,
        ):
            isolated_home = backend._workspace.parent / "home"

        ambiente = run.call_args.kwargs["env"]
        assert "CODEX_API_KEY" not in ambiente
        assert ambiente["DATAFRAMEIT_VARIAVEL_DO_USUARIO"] == "mantida"
        assert ambiente["CODEX_HOME"] == os.fspath(isolated_home)
        assert run.call_args.args[0][-2:] == ["debug", "models"]

    @pytest.mark.parametrize(
        ("stdout", "message"),
        [
            ("não é json", "Expecting value"),
            ('{"modelos": []}', "'models'"),
            ('{"models": ["gpt-6-luna"]}', "update"),
        ],
    )
    def test_catalogo_ilegivel_falha_antes_do_app_server(
        self, codex_sdk, monkeypatch, tmp_path, stdout, message
    ):
        source_home = tmp_path / "source-home"
        source_home.mkdir()
        (source_home / "auth.json").write_text("{}")
        monkeypatch.setenv("CODEX_HOME", str(source_home))
        listing = subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")

        with (
            patch("dataframeit.codex.subprocess.run", return_value=listing),
            patch.object(codex_module, "_open_client") as codex,
            pytest.raises(ProviderConfigurationError, match=rf"catálogo de modelos.*{message}"),
            open_codex_backend(make_config(), SampleModel, "{texto}"),
        ):
            pass

        codex.assert_not_called()

    @pytest.mark.parametrize(
        ("run", "message"),
        [
            ({"side_effect": subprocess.TimeoutExpired(["codex"], 60)}, "timed out"),
            ({"side_effect": FileNotFoundError("codex ausente")}, "codex ausente"),
            (
                {
                    "return_value": subprocess.CompletedProcess(
                        args=[], returncode=0, stdout='{"models": []}', stderr=""
                    )
                },
                "disco cheio",
            ),
        ],
    )
    def test_falha_ao_obter_ou_gravar_o_catalogo_falha_antes_do_app_server(
        self, codex_sdk, monkeypatch, tmp_path, run, message
    ):
        source_home = tmp_path / "source-home"
        source_home.mkdir()
        (source_home / "auth.json").write_text("{}")
        monkeypatch.setenv("CODEX_HOME", str(source_home))

        with (
            patch("dataframeit.codex.subprocess.run", **run),
            patch.object(Path, "write_text", side_effect=OSError("disco cheio")),
            patch.object(codex_module, "_open_client") as codex,
            pytest.raises(ProviderConfigurationError, match=rf"catálogo de modelos.*{message}"),
            open_codex_backend(make_config(), SampleModel, "{texto}"),
        ):
            pass

        codex.assert_not_called()

    def test_falha_do_runtime_ao_ler_o_catalogo_traz_o_stderr(
        self, codex_sdk, monkeypatch, tmp_path
    ):
        source_home = tmp_path / "source-home"
        source_home.mkdir()
        (source_home / "auth.json").write_text("{}")
        monkeypatch.setenv("CODEX_HOME", str(source_home))
        failure = subprocess.CalledProcessError(1, ["codex"], output="", stderr="  sem rede\n")

        with (
            patch("dataframeit.codex.subprocess.run", side_effect=failure),
            patch.object(codex_module, "_open_client") as codex,
            pytest.raises(
                ProviderConfigurationError,
                match=r"catálogo de modelos do Codex \(código 1\): sem rede$",
            ),
            open_codex_backend(make_config(), SampleModel, "{texto}"),
        ):
            pass

        codex.assert_not_called()

    def _open_with_resolved_model(self, codex_sdk, monkeypatch, tmp_path, resolved, **config):
        sdk, sdk_types, generated = codex_sdk
        source_home = tmp_path / "source-home"
        source_home.mkdir()
        (source_home / "auth.json").write_text("{}")
        monkeypatch.setenv("CODEX_HOME", str(source_home))
        client = as_context_manager(MagicMock(spec=sdk.Codex), resolved_model=resolved)
        client.account.return_value = sdk_types.GetAccountResponse(requiresOpenaiAuth=False)
        with (
            patch.object(codex_module, "_open_client", return_value=client),
            open_codex_backend(make_config(**config), SampleModel, "{texto}") as backend,
        ):
            params = client._client.thread_start.call_args.args[0]
            assert isinstance(params, generated.ThreadStartParams)
            assert params.ephemeral is True
            assert params.cwd == str(backend._workspace)
            return backend, params, client

    def test_sonda_confirma_o_modelo_pedido_sem_abrir_turno(self, codex_sdk, monkeypatch, tmp_path):
        backend, params, client = self._open_with_resolved_model(
            codex_sdk, monkeypatch, tmp_path, "gpt-6-luna"
        )

        assert params.model == "gpt-6-luna"
        assert client._client.thread_start.call_count == 1
        client.thread_start.assert_not_called()
        assert backend.config.model == "gpt-6-luna"
        assert threads_descarregadas(client) == ["thread-sonda"]

    def test_falha_ao_descarregar_a_sonda_nao_impede_a_abertura(
        self, codex_sdk, monkeypatch, tmp_path
    ):
        sdk, sdk_types, _ = codex_sdk
        source_home = tmp_path / "source-home"
        source_home.mkdir()
        (source_home / "auth.json").write_text("{}")
        monkeypatch.setenv("CODEX_HOME", str(source_home))
        client = as_context_manager(MagicMock(spec=sdk.Codex))
        client.account.return_value = sdk_types.GetAccountResponse(requiresOpenaiAuth=False)
        client._client.request.side_effect = RuntimeError("app-server recusou")

        with (
            patch.object(codex_module, "_open_client", return_value=client),
            open_codex_backend(make_config(), SampleModel, "{texto}") as backend,
        ):
            assert backend.config.model == "gpt-6-luna"

        assert threads_descarregadas(client) == ["thread-sonda"]

    def _client_whose_close_raises(self, codex_sdk, monkeypatch, tmp_path, erro):
        sdk, sdk_types, _ = codex_sdk
        source_home = tmp_path / "source-home"
        source_home.mkdir()
        (source_home / "auth.json").write_text("{}")
        monkeypatch.setenv("CODEX_HOME", str(source_home))
        client = as_context_manager(MagicMock(spec=sdk.Codex))
        client.account.return_value = sdk_types.GetAccountResponse(requiresOpenaiAuth=False)
        client.close.side_effect = erro
        return client

    @pytest.mark.parametrize(
        "erro",
        [BrokenPipeError(errno.EPIPE, "Broken pipe"), OSError(errno.EINVAL, "Invalid argument")],
        ids=["broken-pipe", "einval"],
    )
    @pytest.mark.parametrize(
        "saida", [None, ProviderError("falha da linha")], ids=["normal", "erro-do-bloco"]
    )
    def test_falha_de_e_s_ao_fechar_nao_substitui_a_saida_do_backend(
        self, codex_sdk, monkeypatch, tmp_path, erro, saida
    ):
        """O fechamento que falha na escrita do stdin não troca o que sai do bloco.

        A escrita num pipe sem leitor falha com `BrokenPipeError`, ou com
        `OSError(EINVAL)` no Windows, quando o app-server morreu com um pedido
        no buffer. Propagar a falha do fechamento faria `dataframeit()` perder
        as linhas processadas, ou trocaria o erro do bloco pelo do fechamento.
        """
        client = self._client_whose_close_raises(codex_sdk, monkeypatch, tmp_path, erro)
        esperado = pytest.raises(ProviderError) if saida is not None else contextlib.nullcontext()

        with (
            esperado as excinfo,
            patch.object(codex_module, "_open_client", return_value=client),
            open_codex_backend(make_config(), SampleModel, "{texto}"),
        ):
            if saida is not None:
                raise saida

        client.close.assert_called_once_with()
        if saida is not None:
            assert excinfo is not None
            assert excinfo.value is saida

    @pytest.mark.parametrize(
        "erro",
        [
            pytest.param(OSError(errno.EIO, "Input/output error"), id="eio"),
            pytest.param(PermissionError(errno.EACCES, "Permission denied"), id="permissao"),
            pytest.param(ValueError("I/O operation on closed file"), id="valor"),
            pytest.param(
                AttributeError("'Codex' object has no attribute '_client'"), id="atributo"
            ),
        ],
    )
    def test_falha_do_fechamento_fora_do_pipe_sem_leitor_continua_aparecendo(
        self, codex_sdk, monkeypatch, tmp_path, erro
    ):
        """Só a falha da escrita num pipe sem leitor é suprimida; as outras sobem.

        O conjunto é o que o `subprocess` do CPython engole ao fechar o stdin:
        `BrokenPipeError` e `OSError` com `errno.EINVAL`.
        """
        client = self._client_whose_close_raises(codex_sdk, monkeypatch, tmp_path, erro)

        with (
            pytest.raises(type(erro)) as excinfo,
            patch.object(codex_module, "_open_client", return_value=client),
            open_codex_backend(make_config(), SampleModel, "{texto}"),
        ):
            pass

        assert excinfo.value is erro

    @pytest.mark.parametrize(
        ("model_kwargs", "prazo"), [({}, 600), ({"timeout": 30}, 30), ({"timeout": None}, None)]
    )
    def test_backend_recebe_o_prazo_configurado(
        self, codex_sdk, monkeypatch, tmp_path, model_kwargs, prazo
    ):
        backend, _, _ = self._open_with_resolved_model(
            codex_sdk, monkeypatch, tmp_path, "gpt-6-luna", model_kwargs=model_kwargs
        )

        assert backend._turn_timeout == prazo

    def test_prazo_invalido_falha_antes_de_subir_o_cliente(self, codex_sdk):

        with (
            patch.object(codex_module, "_open_client") as codex,
            pytest.raises(ProviderConfigurationError, match="timeout inválido"),
            open_codex_backend(make_config(model_kwargs={"timeout": 0}), SampleModel, "{texto}"),
        ):
            pass

        codex.assert_not_called()

    def test_modelo_resolvido_diferente_do_pedido_e_recusado(
        self, codex_sdk, monkeypatch, tmp_path
    ):
        with pytest.raises(ProviderConfigurationError, match=r"'outro-modelo'.*'gpt-6-luna'"):
            self._open_with_resolved_model(codex_sdk, monkeypatch, tmp_path, "outro-modelo")

    def test_sem_model_avisa_qual_o_codex_resolveu(self, codex_sdk, monkeypatch, tmp_path):
        with pytest.warns(UserWarning, match="sem model.*'modelo-padrao'"):
            _, params, _ = self._open_with_resolved_model(
                codex_sdk, monkeypatch, tmp_path, "modelo-padrao", model=None
            )

        assert params.model is None

    def test_missing_auth_fails_before_runtime_or_client(self, codex_sdk, monkeypatch, tmp_path):
        source_home = tmp_path / "source-home"
        source_home.mkdir()
        monkeypatch.setenv("CODEX_HOME", str(source_home))

        with (
            patch("dataframeit.codex.tempfile.TemporaryDirectory") as temporary_directory,
            patch.object(codex_module, "_open_client") as codex,
            pytest.raises(ProviderConfigurationError) as exc_info,
            open_codex_backend(make_config(), SampleModel, "{texto}"),
        ):
            pass

        assert CODEX_FILE_AUTH_LOGIN_COMMAND in str(exc_info.value)
        temporary_directory.assert_not_called()
        codex.assert_not_called()

    def test_distinct_auth_files_do_not_contend(self, codex_sdk, monkeypatch, tmp_path):
        sdk, sdk_types, _ = codex_sdk
        homes = [tmp_path / "home-a", tmp_path / "home-b"]
        clients = []
        for home in homes:
            home.mkdir()
            (home / "auth.json").write_text("{}")
            client = as_context_manager(MagicMock(spec=sdk.Codex))
            client.account.return_value = sdk_types.GetAccountResponse(requiresOpenaiAuth=False)
            clients.append(client)

        monkeypatch.setenv("CODEX_HOME", str(homes[0]))
        with (
            patch.object(codex_module, "_open_client", side_effect=clients) as codex,
            open_codex_backend(make_config(), SampleModel, "{texto}"),
        ):
            monkeypatch.setenv("CODEX_HOME", str(homes[1]))
            with open_codex_backend(make_config(), SampleModel, "{texto}"):
                assert not auth_lock_is_available(homes[0] / "auth.json.dataframeit.lock")
                assert not auth_lock_is_available(homes[1] / "auth.json.dataframeit.lock")
                assert codex.call_count == 2

        for client in clients:
            client.close.assert_called_once_with()

    def test_hard_link_failure_is_explicit_and_cleans_runtime(
        self, codex_sdk, monkeypatch, tmp_path
    ):
        source_home = tmp_path / "source-home"
        source_home.mkdir()
        source_auth = source_home / "auth.json"
        source_auth.write_text("{}")
        monkeypatch.setenv("CODEX_HOME", str(source_home))

        with (
            patch("dataframeit.codex.os.link", side_effect=OSError("unsupported")),
            patch.object(Path, "symlink_to") as symlink,
            patch.object(codex_module, "_open_client") as codex,
            pytest.raises(ProviderConfigurationError, match="hard link"),
            open_codex_backend(make_config(), SampleModel, "{texto}"),
        ):
            pass

        codex.assert_not_called()
        symlink.assert_not_called()
        assert auth_lock_is_available(source_home / "auth.json.dataframeit.lock")
        assert list(source_home.glob("dataframeit-codex-*")) == []

    @pytest.mark.parametrize("failure_stage", ["constructor", "account"])
    def test_client_start_failure_releases_auth_lock(
        self, codex_sdk, monkeypatch, tmp_path, failure_stage
    ):
        sdk, _, _ = codex_sdk
        source_home = tmp_path / "source-home"
        source_home.mkdir()
        (source_home / "auth.json").write_text("{}")
        monkeypatch.setenv("CODEX_HOME", str(source_home))
        client = as_context_manager(MagicMock(spec=sdk.Codex))
        client.account.side_effect = RuntimeError("account failed")
        codex_result = (
            RuntimeError("constructor failed") if failure_stage == "constructor" else client
        )

        with (
            patch.object(codex_module, "_open_client", side_effect=[codex_result]),
            pytest.raises(RuntimeError, match="failed"),
            open_codex_backend(make_config(), SampleModel, "{texto}"),
        ):
            pass

        if failure_stage == "account":
            client.close.assert_called_once_with()
        assert auth_lock_is_available(source_home / "auth.json.dataframeit.lock")
        assert list(source_home.glob("dataframeit-codex-*")) == []

    def test_client_close_failure_still_releases_auth_lock(self, codex_sdk, monkeypatch, tmp_path):
        sdk, sdk_types, _ = codex_sdk
        source_home = tmp_path / "source-home"
        source_home.mkdir()
        (source_home / "auth.json").write_text("{}")
        monkeypatch.setenv("CODEX_HOME", str(source_home))
        client = as_context_manager(MagicMock(spec=sdk.Codex))
        client.account.return_value = sdk_types.GetAccountResponse(requiresOpenaiAuth=False)
        client.close.side_effect = RuntimeError("close failed")

        with (
            patch.object(codex_module, "_open_client", return_value=client),
            pytest.raises(RuntimeError, match="close failed"),
            open_codex_backend(make_config(), SampleModel, "{texto}"),
        ):
            pass

        assert auth_lock_is_available(source_home / "auth.json.dataframeit.lock")
        assert list(source_home.glob("dataframeit-codex-*")) == []

    def test_runtime_directory_failure_has_accurate_error_and_cleans_up(
        self, codex_sdk, monkeypatch, tmp_path
    ):
        source_home = tmp_path / "source-home"
        source_home.mkdir()
        (source_home / "auth.json").write_text("{}")
        monkeypatch.setenv("CODEX_HOME", str(source_home))
        original_mkdir = Path.mkdir

        def fail_runtime_directory(path, *args, **kwargs):
            if path.name in {"workspace", "home"}:
                msg = "read only"
                raise OSError(msg)
            return original_mkdir(path, *args, **kwargs)

        with (
            patch.object(Path, "mkdir", fail_runtime_directory),
            patch.object(codex_module, "_open_client") as codex,
            pytest.raises(ProviderConfigurationError, match="diretórios do runtime"),
            open_codex_backend(make_config(), SampleModel, "{texto}"),
        ):
            pass

        codex.assert_not_called()
        assert auth_lock_is_available(source_home / "auth.json.dataframeit.lock")
        assert list(source_home.glob("dataframeit-codex-*")) == []

    @pytest.mark.parametrize("error_type", [OSError, NotImplementedError])
    def test_auth_lock_failure_is_explicit_before_runtime_creation(
        self, codex_sdk, monkeypatch, tmp_path, error_type
    ):
        source_home = tmp_path / "source-home"
        source_home.mkdir()
        (source_home / "auth.json").write_text("{}")
        monkeypatch.setenv("CODEX_HOME", str(source_home))

        with (
            patch("filelock.FileLock.acquire", side_effect=error_type("unsupported")),
            patch.object(codex_module, "_open_client") as codex,
            pytest.raises(ProviderConfigurationError, match="acesso exclusivo"),
            open_codex_backend(make_config(), SampleModel, "{texto}"),
        ):
            pass

        codex.assert_not_called()
        assert list(source_home.glob("dataframeit-codex-*")) == []


class TestCodexInvocation:
    def test_thread_owns_execution_config_and_turn_only_owns_output_config(
        self, codex_sdk, tmp_path
    ):
        sdk, sdk_types, _ = codex_sdk
        backend, client, thread, _ = initialized_backend(tmp_path, codex_sdk)

        result = backend.invoke("texto")

        assert result["data"] == {"sentimento": "positivo", "confianca": 0.9}
        assert result["usage"] == {
            "input_tokens": 100,
            "cached_input_tokens": 40,
            "output_tokens": 30,
            "reasoning_tokens": 10,
            "total_tokens": 130,
        }
        start_kwargs = client.thread_start.call_args.kwargs
        assert set(start_kwargs) == {
            "approval_mode",
            "cwd",
            "developer_instructions",
            "ephemeral",
            "model",
            "sandbox",
        }
        assert start_kwargs["approval_mode"] is sdk.ApprovalMode.deny_all
        assert start_kwargs["cwd"] == str(backend._workspace)
        assert start_kwargs["ephemeral"] is True
        assert start_kwargs["model"] == "gpt-6-luna"
        assert start_kwargs["sandbox"] is sdk.Sandbox.read_only
        assert "untrusted data" in start_kwargs["developer_instructions"]
        turn_args = thread.turn.call_args
        assert turn_args.args == ("Analise: texto",)
        assert set(turn_args.kwargs) == {"effort", "output_schema"}
        assert turn_args.kwargs["effort"] is sdk_types.ReasoningEffort.medium
        assert turn_args.kwargs["output_schema"] == backend._schema

    def test_valid_output_without_usage_is_preserved(self, codex_sdk, tmp_path):
        result_without_usage = make_result(codex_sdk, usage=False)
        backend, _, _, _ = initialized_backend(tmp_path, codex_sdk, result_without_usage)

        result = backend.invoke("texto")

        assert result["data"] == {"sentimento": "positivo", "confianca": 0.9}
        assert result["usage"] is None

    @pytest.mark.parametrize(
        "response",
        [
            "not-json",
            '{"sentimento": "positivo"}',
            '{"sentimento": 42, "confianca": 0.9}',
        ],
    )
    def test_final_response_is_validated_directly_by_pydantic_json(
        self, codex_sdk, tmp_path, response
    ):
        backend, client, _, _ = initialized_backend(
            tmp_path, codex_sdk, make_result(codex_sdk, response=response)
        )

        with (
            pytest.warns(UserWarning, match="não-recuperável"),
            pytest.raises(ProviderOutputError, match="não corresponde ao schema"),
        ):
            backend.invoke("texto")

        assert client.thread_start.call_count == 1

    def test_number_in_rejected_response_is_not_a_credential_error(self, codex_sdk, tmp_path):
        response = '{"sentimento": "Súmula 403 do STJ", "confianca": "art. 401"}'
        backend, _, _, _ = initialized_backend(
            tmp_path, codex_sdk, make_result(codex_sdk, response=response)
        )

        with (
            pytest.warns(UserWarning, match="não-recuperável"),
            pytest.raises(ProviderOutputError) as error,
        ):
            backend.invoke("texto")

        message = str(error.value)
        assert "Súmula" not in message
        assert "art. 401" not in message
        assert "confianca: Input should be a valid number" in message
        friendly = get_friendly_error_message(error.value, "codex")
        assert "AUTENTICAÇÃO" not in friendly
        assert "PERMISSÃO" not in friendly
        assert "login" not in friendly

    @pytest.mark.parametrize(
        ("response", "status", "message"),
        [
            ("", None, "resposta vazia"),
            (None, None, "resposta vazia"),
            (
                '{"sentimento": "positivo", "confianca": 0.9}',
                "interrupted",
                "interrupted",
            ),
        ],
    )
    def test_empty_or_incomplete_turn_is_output_error(
        self, codex_sdk, tmp_path, response, status, message
    ):
        _, sdk_types, _ = codex_sdk
        turn_status = sdk_types.TurnStatus(status) if status else None
        backend, client, _, _ = initialized_backend(
            tmp_path,
            codex_sdk,
            make_result(codex_sdk, response=response, status=turn_status),
        )

        with (
            pytest.warns(UserWarning, match="não-recuperável"),
            pytest.raises(ProviderOutputError, match=message),
        ):
            backend.invoke("texto")

        assert client.thread_start.call_count == 1

    def test_retry_uses_real_sdk_overload_classification(self, codex_sdk, tmp_path):
        sdk, _, _ = codex_sdk
        backend, client, thread, _ = initialized_backend(tmp_path, codex_sdk)
        busy = sdk.ServerBusyError(
            -32000,
            "server busy",
            {"codexErrorInfo": "server_overloaded"},
        )
        client.thread_start.side_effect = [busy, thread]

        with pytest.warns(UserWarning, match="Tentativa 1/2"):
            result = backend.invoke("texto")

        assert result["_retry_info"]["retries"] == 1
        assert client.thread_start.call_count == 2

    def test_failed_turn_overload_uses_real_protocol_error(self, codex_sdk, tmp_path):
        _, _, generated = codex_sdk
        backend, client, thread, turn = initialized_backend(tmp_path, codex_sdk)
        overloaded = generated.CodexErrorInfo(root=generated.CodexErrorInfoValue.server_overloaded)
        turn.stream.side_effect = [
            as_stream(make_result(codex_sdk, error_info=overloaded, message="overloaded")),
            as_stream(make_result(codex_sdk)),
        ]

        with pytest.warns(UserWarning, match="Tentativa 1/2"):
            result = backend.invoke("texto")

        assert result["_retry_info"]["retries"] == 1
        assert client.thread_start.call_count == 2
        thread.read.assert_not_called()

    @pytest.mark.parametrize("code", ["internal_server_error", "thread_rollback_failed"])
    def test_failed_turn_transient_code_retries_without_rate_limit(self, codex_sdk, tmp_path, code):
        _, _, generated = codex_sdk
        backend, client, _, turn = initialized_backend(tmp_path, codex_sdk)
        info = generated.CodexErrorInfo(root=generated.CodexErrorInfoValue[code])
        turn.stream.side_effect = lambda: as_stream(
            make_result(codex_sdk, error_info=info, message="internal failure")
        )

        with (
            pytest.warns(UserWarning, match="Tentativa 1/2"),
            pytest.raises(ProviderTransientError, match="internal failure") as exc_info,
        ):
            backend.invoke("texto")

        assert client.thread_start.call_count == 2
        assert is_rate_limit_error(exc_info.value) is False

    def test_failed_turn_http_429_is_overload_and_retries(self, codex_sdk, tmp_path):
        _, _, generated = codex_sdk
        backend, client, _, turn = initialized_backend(tmp_path, codex_sdk)
        too_many = generated.CodexErrorInfo(
            root=generated.HttpConnectionFailedCodexErrorInfo(
                httpConnectionFailed=generated.HttpConnectionFailed(httpStatusCode=429)
            )
        )
        turn.stream.side_effect = lambda: as_stream(
            make_result(codex_sdk, error_info=too_many, message="too many requests")
        )

        with (
            pytest.warns(UserWarning, match="Tentativa 1/2"),
            pytest.raises(ProviderOverloadedError, match="too many requests"),
        ):
            backend.invoke("texto")

        assert client.thread_start.call_count == 2

    @pytest.mark.parametrize("code", ["rate_limit_exceeded", "flex_unavailable"])
    def test_failed_turn_rate_limit_and_flex_capacity_are_overload(self, codex_sdk, tmp_path, code):
        _, _, generated = codex_sdk
        backend, client, _, turn = initialized_backend(tmp_path, codex_sdk)
        info = generated.CodexErrorInfo(root=generated.CodexErrorInfoValue[code])
        turn.stream.side_effect = lambda: as_stream(
            make_result(codex_sdk, error_info=info, message="try again later")
        )

        with (
            pytest.warns(UserWarning, match="Tentativa 1/2"),
            pytest.raises(ProviderOverloadedError, match="try again later"),
        ):
            backend.invoke("texto")

        assert client.thread_start.call_count == 2

    @pytest.mark.parametrize(
        "code", ["misalignment_policy_violation", "too_many_denials", "session_budget_exceeded"]
    )
    def test_failed_turn_row_scoped_codes_fail_only_the_row(self, codex_sdk, tmp_path, code):
        _, _, generated = codex_sdk
        backend, client, _, turn = initialized_backend(tmp_path, codex_sdk)
        info = generated.CodexErrorInfo(root=generated.CodexErrorInfoValue[code])
        turn.stream.side_effect = lambda: as_stream(
            make_result(codex_sdk, error_info=info, message="blocked")
        )

        with (
            pytest.warns(UserWarning, match="não-recuperável"),
            pytest.raises(ProviderError, match="blocked") as exc_info,
        ):
            backend.invoke("texto")

        assert not isinstance(exc_info.value, (ProviderTransientError, ProviderAbortError))
        assert client.thread_start.call_count == 1

    # O SDK aceita código e variante fora do protocolo que conhece: o código vira
    # membro novo de CodexErrorInfoValue, e a variante fica como dict cru.
    @pytest.mark.parametrize(
        "codigo", ["codigoNovo", {"variantNova": {}}], ids=["codigo", "variante"]
    )
    def test_codigo_de_erro_desconhecido_falha_so_a_linha(self, codex_sdk, tmp_path, codigo):
        _, _, generated = codex_sdk
        backend, client, _, turn = initialized_backend(tmp_path, codex_sdk)
        info = generated.CodexErrorInfo.model_validate(codigo)
        turn.stream.side_effect = lambda: as_stream(
            make_result(codex_sdk, error_info=info, message="novidade")
        )

        with (
            pytest.warns(UserWarning, match="não-recuperável"),
            pytest.raises(ProviderError, match="novidade") as exc_info,
        ):
            backend.invoke("texto")

        assert not isinstance(exc_info.value, (ProviderTransientError, ProviderAbortError))
        assert client.thread_start.call_count == 1

    def test_failed_turn_http_401_is_definitive(self, codex_sdk, tmp_path):
        _, _, generated = codex_sdk
        backend, client, _, turn = initialized_backend(tmp_path, codex_sdk)
        unauthorized = generated.CodexErrorInfo(
            root=generated.ResponseStreamConnectionFailedCodexErrorInfo(
                responseStreamConnectionFailed=(
                    generated.ResponseStreamConnectionFailed(httpStatusCode=401)
                )
            )
        )
        turn.stream.side_effect = lambda: as_stream(
            make_result(codex_sdk, error_info=unauthorized, message="unauthorized")
        )

        with (
            pytest.warns(UserWarning, match="não-recuperável"),
            pytest.raises(ProviderError, match="unauthorized") as exc_info,
        ):
            backend.invoke("texto")

        assert not isinstance(exc_info.value, ProviderTransientError)
        assert client.thread_start.call_count == 1

    def test_unknown_sdk_error_is_provider_error_without_retry(self, codex_sdk, tmp_path):
        backend, client, _, _ = initialized_backend(tmp_path, codex_sdk)
        client.thread_start.side_effect = RuntimeError("unexpected")

        with (
            pytest.warns(UserWarning, match="não-recuperável"),
            pytest.raises(ProviderError, match="RuntimeError: unexpected"),
        ):
            backend.invoke("texto")

        assert client.thread_start.call_count == 1

    def test_each_row_gets_an_ephemeral_thread(self, codex_sdk, tmp_path):
        sdk, _, _ = codex_sdk
        backend, client, _, _ = initialized_backend(tmp_path, codex_sdk)
        threads = []
        for response in ("primeiro", "segundo"):
            result = make_result(
                codex_sdk,
                response=('{"sentimento": "' + response + '", "confianca": 1.0}'),
            )
            turn = MagicMock(spec=sdk.TurnHandle)
            turn.id = "turn-1"
            turn.stream.side_effect = lambda events=result: as_stream(events)
            thread = MagicMock(spec=sdk.Thread)
            thread.turn.return_value = turn
            threads.append(thread)
        client.thread_start.side_effect = threads

        first = backend.invoke("a")
        second = backend.invoke("b")

        assert first["data"]["sentimento"] == "primeiro"
        assert second["data"]["sentimento"] == "segundo"
        assert client.thread_start.call_count == 2
        assert all(call.kwargs["ephemeral"] is True for call in client.thread_start.call_args_list)


class TestProviderErrorClassification:
    def test_typed_overload_drives_retry_and_worker_reduction(self):
        error = ProviderOverloadedError("server overloaded")

        assert is_recoverable_error(error) is True
        assert is_rate_limit_error(error) is True

    def test_typed_transient_error_retries_without_worker_reduction(self):
        error = ProviderTransientError("internal server error after HTTP 429")

        assert is_recoverable_error(error) is True
        assert is_rate_limit_error(error) is False

    @pytest.mark.parametrize(
        "error",
        [
            ProviderError("definitive"),
            ProviderConfigurationError("bad config"),
            ProviderOutputError("bad output"),
        ],
    )
    def test_other_typed_provider_errors_are_not_recoverable(self, error):
        assert is_recoverable_error(error) is False


# O Pydantic não gera esses schemas sozinho, mas json_schema_extra e um
# model_json_schema sobrescrito deixam o usuário entregar qualquer dicionário,
# e o structured output do Codex recusa o que não sabe converter.
class TestSchemaMalformado:
    @pytest.mark.parametrize(
        ("propriedade", "mensagem"),
        [
            ({"$ref": "#/definitions/Externo"}, "Referência não suportada"),
            ({"$ref": "#/$defs/Inexistente"}, "Referência inválida"),
            ({"$ref": "#/$defs/Item/required"}, "Referência inválida"),
            (True, "representados por objetos"),
            ({"oneOf": {"type": "string"}}, "oneOf inválido"),
            ({"type": "string", "discriminator": {"propertyName": "tipo"}}, "sem oneOf"),
        ],
    )
    def test_propriedade_malformada_e_erro_de_configuracao(self, propriedade, mensagem):
        schema = {
            "type": "object",
            "$defs": {
                "Item": {
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                    "required": ["x"],
                }
            },
            "properties": {"valor": propriedade},
        }

        with pytest.raises(ProviderConfigurationError, match=mensagem):
            _to_strict_json_schema(schema)

    def test_defs_que_nao_e_objeto_e_erro_de_configuracao(self):
        schema = {"type": "object", "$defs": [], "properties": {"x": {"type": "string"}}}

        with pytest.raises(ProviderConfigurationError, match=r"\$defs inválido"):
            _to_strict_json_schema(schema)

    def test_recursao_com_metadados_ao_lado_da_referencia_e_recusada(self):
        """Expandir a referência com a descrição ao lado nunca terminaria."""

        class No(BaseModel):
            nome: str
            filho: No = Field(description="Nó filho")

        with pytest.raises(ProviderConfigurationError, match="recursivos com metadados"):
            _build_schema(No)

    @pytest.mark.parametrize("erro", [AttributeError, TypeError])
    def test_falha_do_schema_personalizado_e_erro_de_configuracao(self, erro):
        def json_schema_extra(schema):
            msg = "hook quebrou"
            raise erro(msg)

        class Modelo(BaseModel):
            model_config = ConfigDict(json_schema_extra=json_schema_extra)
            valor: str

        mensagem = "A geração do JSON Schema do modelo Pydantic falhou: hook quebrou"
        with pytest.raises(ProviderConfigurationError, match=f"^{mensagem}$") as exc_info:
            _build_schema(Modelo)

        assert isinstance(exc_info.value.__cause__, erro)

    def test_schema_personalizado_que_nao_e_objeto_e_recusado(self):

        class Modelo(BaseModel):
            valor: str

            @classmethod
            def model_json_schema(cls, *args, **kwargs):
                return ["não", "é", "objeto"]

        with pytest.raises(ProviderConfigurationError, match="deve retornar um objeto"):
            _build_schema(Modelo)


class TestFalhasDoRuntime:
    def test_falha_ao_criar_o_runtime_temporario_libera_o_lock(
        self, codex_sdk, monkeypatch, tmp_path
    ):
        source_home = tmp_path / "source-home"
        source_home.mkdir()
        (source_home / "auth.json").write_text("{}")
        monkeypatch.setenv("CODEX_HOME", str(source_home))

        with (
            patch(
                "dataframeit.codex.tempfile.TemporaryDirectory",
                side_effect=OSError("disco cheio"),
            ),
            patch.object(codex_module, "_open_client") as codex,
            pytest.raises(ProviderConfigurationError, match="runtime temporário") as exc_info,
            open_codex_backend(make_config(), SampleModel, "{texto}"),
        ):
            pass

        assert isinstance(exc_info.value.__cause__, OSError)
        codex.assert_not_called()
        assert auth_lock_is_available(source_home / "auth.json.dataframeit.lock")

    def test_conta_que_exige_login_para_antes_do_backend(self, codex_sdk, monkeypatch, tmp_path):
        sdk, sdk_types, _ = codex_sdk
        source_home = tmp_path / "source-home"
        source_home.mkdir()
        (source_home / "auth.json").write_text("{}")
        monkeypatch.setenv("CODEX_HOME", str(source_home))
        client = as_context_manager(MagicMock(spec=sdk.Codex))
        client.account.return_value = sdk_types.GetAccountResponse(requiresOpenaiAuth=True)

        with (
            patch.object(codex_module, "_open_client", return_value=client),
            pytest.raises(ProviderConfigurationError) as exc_info,
            open_codex_backend(make_config(), SampleModel, "{texto}"),
        ):
            pass

        assert CODEX_FILE_AUTH_LOGIN_COMMAND in str(exc_info.value)
        client.close.assert_called_once_with()
        assert auth_lock_is_available(source_home / "auth.json.dataframeit.lock")


class TestClassificacaoDoTurnoQueFalhou:
    def test_turno_falho_sem_detalhe_e_definitivo(self, codex_sdk, tmp_path):
        _, sdk_types, _ = codex_sdk
        backend, client, _, _ = initialized_backend(
            tmp_path, codex_sdk, make_result(codex_sdk, status=sdk_types.TurnStatus.failed)
        )

        with (
            pytest.warns(UserWarning, match="não-recuperável"),
            pytest.raises(ProviderError, match="sem detalhe") as exc_info,
        ):
            backend.invoke("texto")

        assert not isinstance(exc_info.value, ProviderTransientError)
        assert client.thread_start.call_count == 1

    @pytest.mark.parametrize("status", [None, 503])
    def test_falha_http_sem_status_ou_do_servidor_e_transitoria(self, codex_sdk, tmp_path, status):
        _, _, generated = codex_sdk
        backend, client, _, turn = initialized_backend(tmp_path, codex_sdk)
        caiu = generated.CodexErrorInfo(
            root=generated.ResponseStreamDisconnectedCodexErrorInfo(
                responseStreamDisconnected=generated.ResponseStreamDisconnected(
                    httpStatusCode=status
                )
            )
        )
        turn.stream.side_effect = lambda: as_stream(
            make_result(codex_sdk, error_info=caiu, message="stream caiu")
        )

        with (
            pytest.warns(UserWarning, match="Tentativa 1/2"),
            pytest.raises(ProviderTransientError, match="stream caiu") as exc_info,
        ):
            backend.invoke("texto")

        assert client.thread_start.call_count == 2
        assert is_rate_limit_error(exc_info.value) is False

    def test_codigo_de_erro_fora_dos_transitorios_e_definitivo(self, codex_sdk, tmp_path):
        _, _, generated = codex_sdk
        janela = generated.CodexErrorInfo(
            root=generated.CodexErrorInfoValue.context_window_exceeded
        )
        backend, client, _, _ = initialized_backend(
            tmp_path,
            codex_sdk,
            make_result(codex_sdk, error_info=janela, message="janela de contexto"),
        )

        with (
            pytest.warns(UserWarning, match="não-recuperável"),
            pytest.raises(ProviderError, match="janela de contexto") as exc_info,
        ):
            backend.invoke("texto")

        assert not isinstance(exc_info.value, ProviderTransientError)
        assert client.thread_start.call_count == 1

    def test_limite_de_uso_interrompe_sem_nova_tentativa(self, codex_sdk, tmp_path):
        _, _, generated = codex_sdk
        limite = generated.CodexErrorInfo(root=generated.CodexErrorInfoValue.usage_limit_exceeded)
        backend, client, _, _ = initialized_backend(
            tmp_path,
            codex_sdk,
            make_result(codex_sdk, error_info=limite, message="usage limit reached"),
        )

        with (
            pytest.warns(UserWarning, match="não-recuperável"),
            pytest.raises(ProviderUsageLimitError, match="usage limit reached"),
        ):
            backend.invoke("texto")

        assert client.thread_start.call_count == 1

    @pytest.mark.parametrize("onde", ["thread_start", "stream"])
    @pytest.mark.parametrize(
        "erro",
        [
            pytest.param("transport", id="transport-closed"),
            pytest.param(BrokenPipeError("pipe"), id="broken-pipe"),
        ],
    )
    def test_app_server_encerrado_interrompe_a_execucao(self, codex_sdk, tmp_path, onde, erro):
        from openai_codex.errors import (  # noqa: PLC0415 (SDK carregado pelo fixture)
            TransportClosedError,
        )

        if erro == "transport":
            erro = TransportClosedError("Codex process is not running")
        backend, client, _, turn = initialized_backend(tmp_path, codex_sdk)
        if onde == "thread_start":
            client.thread_start.side_effect = erro
        else:
            turn.stream.side_effect = erro

        with (
            pytest.warns(UserWarning, match="não-recuperável"),
            pytest.raises(ProviderAbortError, match="app-server do Codex encerrou"),
        ):
            backend.invoke("texto")

        assert client.thread_start.call_count == 1

    # O app-server recusa com o código JSON-RPC -32001, sem `data`, o pedido que
    # chega com a fila de entrada cheia. A classificação lê o código, e não a
    # mensagem: o -32000 com o mesmo texto continua definitivo.
    def test_fila_cheia_do_app_server_e_sobrecarga(self, codex_sdk):
        from openai_codex.errors import (  # noqa: PLC0415 (SDK carregado pelo fixture)
            map_jsonrpc_error,
        )

        erro = map_jsonrpc_error(-32001, "Server overloaded; retry later.", None)

        with pytest.raises(ProviderOverloadedError, match="Server overloaded") as exc_info:
            _raise_classified_sdk_error(erro)

        assert is_rate_limit_error(exc_info.value) is True

    @pytest.mark.parametrize("codigo", [-32000, -32002])
    def test_outro_codigo_json_rpc_com_texto_de_sobrecarga_e_definitivo(self, codex_sdk, codigo):
        from openai_codex.errors import (  # noqa: PLC0415 (SDK carregado pelo fixture)
            map_jsonrpc_error,
        )

        erro = map_jsonrpc_error(codigo, "Server overloaded; retry later.", None)

        with pytest.raises(ProviderError) as exc_info:
            _raise_classified_sdk_error(erro)

        assert not isinstance(exc_info.value, ProviderTransientError)
        assert is_rate_limit_error(exc_info.value) is False

    def test_codigo_fora_de_erro_json_rpc_nao_e_lido(self, codex_sdk):
        class ErroComCodigo(RuntimeError):
            code = -32001

        with pytest.raises(ProviderError) as exc_info:
            _raise_classified_sdk_error(ErroComCodigo("falhou"))

        assert not isinstance(exc_info.value, ProviderTransientError)

    def test_fila_cheia_reduz_workers_e_repete_a_linha(self, codex_sdk, tmp_path, monkeypatch):
        from openai_codex.errors import (  # noqa: PLC0415 (SDK carregado pelo fixture)
            map_jsonrpc_error,
        )

        backend, client, _, _ = initialized_backend(tmp_path, codex_sdk)
        pedidos = []
        trava = threading.Lock()

        def fila_cheia(**_):
            with trava:
                pedidos.append(1)
            raise map_jsonrpc_error(-32001, "Server overloaded; retry later.", None)

        client.thread_start.side_effect = fila_cheia

        @contextlib.contextmanager
        def backend_falso(*_args):
            yield backend

        monkeypatch.setattr("dataframeit.codex.open_codex_backend", backend_falso)
        # O evento de rate limit é limpo na hora, sem o timer de 5 s.
        monkeypatch.setattr(
            "dataframeit.core.threading.Timer",
            lambda _intervalo, funcao: SimpleNamespace(start=funcao),
        )
        entrada = pd.DataFrame({"texto": ["a", "b", "c", "d"]})

        with pytest.warns(UserWarning, match="Reduzindo workers de 4 para 2") as avisos:
            saida = dataframeit(
                entrada, SampleModel, "Analise: {texto}", provider="codex", parallel_requests=4
            )

        mensagens = [str(aviso.message) for aviso in avisos]
        assert [m for m in mensagens if m.startswith("Rate limit detectado")] == [
            "Rate limit detectado! Reduzindo workers de 4 para 2.",
            "Rate limit detectado! Reduzindo workers de 2 para 1.",
        ]
        # make_config dá duas tentativas a cada linha.
        assert (
            sum(m.startswith("Tentativa 1/2 falhou (ProviderOverloadedError)") for m in mensagens)
            == 4
        )
        assert len(pedidos) == 8
        assert saida["_dataframeit_status"].tolist() == ["error"] * 4
        assert all("ProviderOverloadedError" in detalhe for detalhe in saida["_error_details"])

    def test_stream_sem_conclusao_e_transitorio(self, codex_sdk, tmp_path):
        backend, client, _, turn = initialized_backend(tmp_path, codex_sdk)
        sem_conclusao = make_result(codex_sdk)[:-1]
        turn.stream.side_effect = [as_stream(sem_conclusao), as_stream(make_result(codex_sdk))]

        with pytest.warns(UserWarning, match=r"Tentativa 1/2 falhou \(ProviderTransientError\)"):
            result = backend.invoke("texto")

        assert result["_retry_info"]["retries"] == 1
        assert client.thread_start.call_count == 2

    def test_eventos_de_outro_turno_sao_ignorados(self, codex_sdk, tmp_path):
        outro = make_result(codex_sdk, response='{"sentimento": "x", "confianca": 0}', turn_id="t2")
        backend, _, _, _ = initialized_backend(
            tmp_path, codex_sdk, outro[:-1] + make_result(codex_sdk)
        )

        result = backend.invoke("texto")

        assert result["data"] == {"sentimento": "positivo", "confianca": 0.9}
        assert result["usage"]["input_tokens"] == 100

    def test_reroteamento_de_modelo_interrompe_o_turno_e_falha_a_linha(self, codex_sdk, tmp_path):
        _, _, generated = codex_sdk
        from openai_codex.models import Notification  # noqa: PLC0415 (SDK carregado pelo fixture)

        rerouted = Notification(
            method="model/rerouted",
            payload=generated.ModelReroutedNotification(
                fromModel="gpt-6-luna",
                toModel="outro",
                reason=generated.ModelRerouteReason("highRiskCyberActivity"),
                threadId="thread-1",
                turnId="turn-1",
            ),
        )
        backend, client, _, turn = initialized_backend(
            tmp_path, codex_sdk, [rerouted, *make_result(codex_sdk)]
        )
        interrompido = threading.Event()
        turn.interrupt.side_effect = interrompido.set

        with (
            pytest.warns(UserWarning, match="não-recuperável"),
            pytest.raises(ProviderError, match=r"'gpt-6-luna' para 'outro'.*highRiskCyberActivity"),
        ):
            backend.invoke("texto")

        assert interrompido.wait(5)
        turn.interrupt.assert_called_once_with()
        assert client.thread_start.call_count == 1

    def test_interrupt_do_reroteamento_que_trava_nao_prende_a_linha(self, codex_sdk, tmp_path):
        _, _, generated = codex_sdk
        from openai_codex.models import Notification  # noqa: PLC0415 (SDK carregado pelo fixture)

        rerouted = Notification(
            method="model/rerouted",
            payload=generated.ModelReroutedNotification(
                fromModel="gpt-6-luna",
                toModel="outro",
                reason=generated.ModelRerouteReason("highRiskCyberActivity"),
                threadId="thread-1",
                turnId="turn-1",
            ),
        )
        backend, _, _, turn = initialized_backend(tmp_path, codex_sdk, [rerouted])
        liberar = threading.Event()
        turn.interrupt.side_effect = liberar.wait

        inicio = time.monotonic()
        try:
            with (
                pytest.warns(UserWarning, match="não-recuperável"),
                pytest.raises(ProviderError, match="'gpt-6-luna' para 'outro'"),
            ):
                backend.invoke("texto")
        finally:
            liberar.set()

        assert time.monotonic() - inicio < 5

    def test_tokens_das_tentativas_que_falharam_sao_somados(self, codex_sdk, tmp_path):
        _, _, generated = codex_sdk
        backend, _, _, turn = initialized_backend(tmp_path, codex_sdk)
        backend = dataclasses.replace(backend, config=make_config(max_retries=3))
        overloaded = generated.CodexErrorInfo(root=generated.CodexErrorInfoValue.server_overloaded)
        internal = generated.CodexErrorInfo(
            root=generated.CodexErrorInfoValue.internal_server_error
        )
        turn.stream.side_effect = [
            as_stream(make_result(codex_sdk, response=None, error_info=overloaded)),
            as_stream(make_result(codex_sdk, response=None, error_info=internal)),
            as_stream(make_result(codex_sdk)),
        ]

        with pytest.warns(UserWarning, match="Tentativa 2/3"):
            result = backend.invoke("texto")

        assert result["usage"] == {
            "input_tokens": 300,
            "cached_input_tokens": 120,
            "output_tokens": 90,
            "reasoning_tokens": 30,
            "total_tokens": 390,
        }

    def test_turno_sem_uso_informado_nao_inventa_tokens(self, codex_sdk, tmp_path):
        _, _, generated = codex_sdk
        backend, _, _, turn = initialized_backend(tmp_path, codex_sdk)
        overloaded = generated.CodexErrorInfo(root=generated.CodexErrorInfoValue.server_overloaded)
        turn.stream.side_effect = [
            as_stream(make_result(codex_sdk, error_info=overloaded, usage=False)),
            as_stream(make_result(codex_sdk)),
        ]

        with pytest.warns(UserWarning, match="Tentativa 1/2"):
            result = backend.invoke("texto")

        assert result["usage"]["input_tokens"] == 100


def stuck_turn(codex_sdk, events=()):
    """Turno real do SDK cujo `turn/completed` nunca chega.

    A assinatura vem do `MessageRouter` do SDK, e o `next()` dela espera em
    `Condition.wait()` sem timeout, como num turno perdido de verdade. Os
    `events` entram no roteador antes, como se o app-server os tivesse enviado.
    """
    sdk, _, _ = codex_sdk
    from openai_codex._message_router import (  # noqa: PLC0415 (SDK carregado pelo fixture)
        MessageRouter,
    )

    router = MessageRouter()
    subscription = router.subscribe_turn("turn-1")
    for event in events:
        router.route_notification(event)
    protocol_client = MagicMock()
    turn = sdk.TurnHandle(protocol_client, "thread-1", "turn-1", _subscription=subscription)
    return turn, protocol_client


class _RelogioDoStream:
    """`threading.Timer` que só começa a contar quando a linha passa o turno ao stream.

    O prazo real começa antes do `thread/start`, e uma pausa do runner antes de
    o turno abrir faria o estouro cair na fase dos pedidos, que aborta em vez
    de repetir a linha. Os testes do stream esperam o estouro dentro dele.
    """

    def __init__(self, interval, function):
        self.interval = interval
        self.function = function
        self.daemon = True
        self._cancelado = threading.Event()
        self.thread = threading.Thread(target=self._conta, daemon=True)

    def start(self):
        self.thread.start()

    def cancel(self):
        self._cancelado.set()

    def _conta(self):
        prazo = self.function.__self__
        while prazo._turn is None:
            if self._cancelado.wait(0.001):
                return
        if not self._cancelado.wait(self.interval):
            self.function()


@pytest.fixture
def relogio_do_stream():
    """Troca o relógio da tentativa pelo `_RelogioDoStream` e devolve os criados."""
    criados = []

    def cria(interval, function):
        relogio = _RelogioDoStream(interval, function)
        criados.append(relogio)
        return relogio

    with patch("dataframeit.codex.threading.Timer", cria):
        yield criados


class TestPrazoDoTurno:
    @pytest.mark.usefixtures("relogio_do_stream")
    def test_turno_sem_conclusao_estoura_o_prazo_e_a_linha_e_tentada_de_novo(
        self, codex_sdk, tmp_path
    ):
        backend, client, thread, normal = initialized_backend(
            tmp_path, codex_sdk, turn_timeout=0.05
        )
        travado, protocol_client = stuck_turn(codex_sdk)
        interrompido = threading.Event()
        protocol_client.turn_interrupt.side_effect = lambda *_: interrompido.set()
        thread.turn.side_effect = [travado, normal]

        with pytest.warns(UserWarning, match=r"Tentativa 1/2 falhou \(ProviderTransientError\)"):
            result = backend.invoke("texto")

        assert result["data"] == {"sentimento": "positivo", "confianca": 0.9}
        assert result["_retry_info"]["retries"] == 1
        assert client.thread_start.call_count == 2
        assert interrompido.wait(5)
        protocol_client.turn_interrupt.assert_called_once_with("thread-1", "turn-1")

    @pytest.mark.usefixtures("relogio_do_stream")
    def test_prazo_esgotado_em_todas_as_tentativas_falha_a_linha_sem_abortar(
        self, codex_sdk, tmp_path
    ):
        backend, _, thread, _ = initialized_backend(tmp_path, codex_sdk, turn_timeout=0.05)
        thread.turn.side_effect = lambda *_, **__: stuck_turn(codex_sdk)[0]

        with (
            pytest.warns(UserWarning, match="Tentativa 1/2"),
            pytest.raises(ProviderTransientError, match=r"prazo de 0\.05 s"),
        ):
            backend.invoke("texto")

        assert thread.turn.call_count == 2

    @pytest.mark.usefixtures("relogio_do_stream")
    def test_interrupt_que_trava_nao_prende_a_linha(self, codex_sdk, tmp_path):
        backend, _, thread, normal = initialized_backend(tmp_path, codex_sdk, turn_timeout=0.05)
        travado, protocol_client = stuck_turn(codex_sdk)
        liberar = threading.Event()
        protocol_client.turn_interrupt.side_effect = lambda *_: liberar.wait()
        thread.turn.side_effect = [travado, normal]

        inicio = time.monotonic()
        try:
            with pytest.warns(UserWarning, match="Tentativa 1/2"):
                result = backend.invoke("texto")
        finally:
            liberar.set()

        assert time.monotonic() - inicio < 5
        assert result["data"] == {"sentimento": "positivo", "confianca": 0.9}

    @pytest.mark.usefixtures("relogio_do_stream")
    def test_uso_do_turno_que_estourou_o_prazo_e_somado(self, codex_sdk, tmp_path):
        backend, _, thread, normal = initialized_backend(tmp_path, codex_sdk, turn_timeout=0.05)
        so_uso = make_result(codex_sdk, response=None)[:-1]
        thread.turn.side_effect = [stuck_turn(codex_sdk, so_uso)[0], normal]

        with pytest.warns(UserWarning, match="Tentativa 1/2"):
            result = backend.invoke("texto")

        assert result["usage"]["input_tokens"] == 200
        assert result["usage"]["total_tokens"] == 260

    def test_uso_do_stream_sem_conclusao_e_somado(self, codex_sdk, tmp_path):
        backend, _, _, turn = initialized_backend(tmp_path, codex_sdk)
        sem_conclusao = make_result(codex_sdk)[:-1]
        turn.stream.side_effect = [as_stream(sem_conclusao), as_stream(make_result(codex_sdk))]

        with pytest.warns(UserWarning, match="Tentativa 1/2"):
            result = backend.invoke("texto")

        assert result["usage"]["input_tokens"] == 200

    def test_turno_concluido_dentro_do_prazo_cancela_o_relogio(self, codex_sdk, tmp_path):
        backend, _, _, turn = initialized_backend(tmp_path, codex_sdk, turn_timeout=60)

        with patch("dataframeit.codex.threading.Timer") as timer:
            result = backend.invoke("texto")

        assert result["data"] == {"sentimento": "positivo", "confianca": 0.9}
        timer.assert_called_once()
        assert timer.call_args.args[0] == 60
        timer.return_value.start.assert_called_once_with()
        timer.return_value.cancel.assert_called_once_with()
        turn.interrupt.assert_not_called()

    def test_sem_prazo_nao_arma_relogio(self, codex_sdk, tmp_path):
        backend, _, _, _ = initialized_backend(tmp_path, codex_sdk, turn_timeout=None)

        with patch("dataframeit.codex.threading.Timer") as timer:
            result = backend.invoke("texto")

        assert result["data"] == {"sentimento": "positivo", "confianca": 0.9}
        timer.assert_not_called()

    def test_transporte_fechado_sem_estouro_continua_abortando(self, codex_sdk, tmp_path):
        from openai_codex.errors import (  # noqa: PLC0415 (SDK carregado pelo fixture)
            TransportClosedError,
        )

        def morre_no_meio():
            yield from make_result(codex_sdk)[:1]
            msg = "Codex process is not running"
            raise TransportClosedError(msg)

        backend, _, _, turn = initialized_backend(tmp_path, codex_sdk, turn_timeout=60)
        turn.stream.side_effect = morre_no_meio

        with (
            pytest.warns(UserWarning, match="não-recuperável"),
            pytest.raises(ProviderAbortError, match="app-server do Codex encerrou"),
        ):
            backend.invoke("texto")


def unanswered_request(client):
    """Pedido ao app-server que espera a resposta como o SDK, e o `close` que o acorda.

    A espera é a fila do `MessageRouter` do SDK, sem timeout, como em
    `CodexClient._request_raw`. O `close` do cliente faz o que o leitor do stdout
    faz quando o processo encerra: chama `fail_all`, que entrega o erro de
    transporte a todo pedido em voo.
    """
    from openai_codex._message_router import (  # noqa: PLC0415 (SDK carregado pelo fixture)
        MessageRouter,
    )
    from openai_codex.errors import (  # noqa: PLC0415 (SDK carregado pelo fixture)
        TransportClosedError,
    )

    router = MessageRouter()
    closed = threading.Event()

    def close():
        closed.set()
        router.fail_all(TransportClosedError("Codex process closed stdout"))

    client.close.side_effect = close

    def wait_for_response(*_, **__):
        waiter = router.create_response_waiter("pedido-1")
        # Com o cliente já fechado, o SDK real recusa escrever o pedido; sem
        # esta conferência, a fila criada depois do `fail_all` não acordaria.
        if closed.is_set():
            msg = "Codex process is not running"
            raise TransportClosedError(msg)
        # O limite só existe para que uma regressão falhe o teste em vez de
        # prender a suíte.
        item = waiter.get(timeout=10)
        if isinstance(item, BaseException):
            raise item
        return item

    return wait_for_response


class TestPrazoDosPedidos:
    def test_thread_start_sem_resposta_encerra_o_app_server_e_aborta(self, codex_sdk, tmp_path):
        backend, client, _, _ = initialized_backend(tmp_path, codex_sdk, turn_timeout=0.05)
        client.thread_start.side_effect = unanswered_request(client)

        inicio = time.monotonic()
        with (
            pytest.warns(UserWarning, match="não-recuperável"),
            pytest.raises(
                ProviderAbortError,
                match=r"não respondeu a thread/start no prazo de 0\.05 s e foi encerrado",
            ),
        ):
            backend.invoke("texto")

        assert time.monotonic() - inicio < 5
        client.close.assert_called_once_with()
        assert client.thread_start.call_count == 1

    def test_turn_start_sem_resposta_encerra_o_app_server_e_aborta(self, codex_sdk, tmp_path):
        backend, client, thread, turn = initialized_backend(tmp_path, codex_sdk, turn_timeout=0.05)
        thread.turn.side_effect = unanswered_request(client)

        with (
            pytest.warns(UserWarning, match="não-recuperável"),
            pytest.raises(ProviderAbortError, match="não respondeu a turn/start"),
        ):
            backend.invoke("texto")

        client.close.assert_called_once_with()
        turn.stream.assert_not_called()

    def test_resposta_do_turn_start_depois_do_estouro_nao_entra_no_stream(
        self, codex_sdk, tmp_path
    ):
        """O turno aberto depois de o relógio derrubar o app-server aborta a linha.

        A resposta chegou, mas o app-server já foi encerrado, e a linha aborta
        com a causa do estouro em vez de esperar o stream de um turno morto.
        """
        backend, client, thread, turn = initialized_backend(tmp_path, codex_sdk, turn_timeout=0.05)
        encerrado = threading.Event()
        client.close.side_effect = encerrado.set

        def responde_tarde(*_, **__):
            assert encerrado.wait(5)
            return turn

        thread.turn.side_effect = responde_tarde

        with (
            pytest.warns(UserWarning, match="não-recuperável"),
            pytest.raises(ProviderAbortError, match="não respondeu a turn/start"),
        ):
            backend.invoke("texto")

        turn.stream.assert_not_called()
        turn.interrupt.assert_not_called()

    def test_relogio_e_armado_antes_do_thread_start(self, codex_sdk, tmp_path):
        backend, client, _, _ = initialized_backend(tmp_path, codex_sdk, turn_timeout=60)
        ordem = []
        thread = client.thread_start.return_value

        def abre_thread(**_):
            ordem.append("thread/start")
            return thread

        client.thread_start.side_effect = abre_thread

        with patch("dataframeit.codex.threading.Timer") as timer:
            timer.return_value.start.side_effect = lambda: ordem.append("relógio")
            backend.invoke("texto")

        assert ordem == ["relógio", "thread/start"]
        timer.return_value.cancel.assert_called_once_with()

    @pytest.mark.usefixtures("relogio_do_stream")
    def test_estouro_no_stream_nao_encerra_o_app_server(self, codex_sdk, tmp_path):
        backend, client, thread, normal = initialized_backend(
            tmp_path, codex_sdk, turn_timeout=0.05
        )
        thread.turn.side_effect = [stuck_turn(codex_sdk)[0], normal]

        with pytest.warns(UserWarning, match="Tentativa 1/2"):
            backend.invoke("texto")

        client.close.assert_not_called()

    def test_sem_prazo_o_pedido_nao_e_encerrado(self, codex_sdk, tmp_path):
        backend, client, _, _ = initialized_backend(tmp_path, codex_sdk, turn_timeout=None)
        espera = unanswered_request(client)
        liberado = threading.Event()

        def responde_depois(**_):
            # Sem prazo, só o fim do teste encerra o cliente e solta o pedido.
            liberado.set()
            return espera()

        client.thread_start.side_effect = responde_depois
        resultado = {}

        def processa():
            try:
                backend.invoke("texto")
            except Exception as err:  # noqa: BLE001 (o erro é conferido fora da thread)
                resultado["erro"] = err

        linha = threading.Thread(target=processa, daemon=True)
        linha.start()
        assert liberado.wait(5)
        linha.join(0.3)
        assert linha.is_alive()
        client.close.assert_not_called()
        client.close()
        linha.join(5)

        assert not linha.is_alive()
        assert isinstance(resultado["erro"], ProviderAbortError)
        assert "app-server do Codex encerrou" in str(resultado["erro"])

    def test_processo_e_morto_antes_de_fechar_o_cliente(self, codex_sdk, tmp_path):
        """O `close` do SDK fecha o stdin antes de sinalizar o processo.

        Com uma escrita bloqueada no pipe de um app-server parado, o fechamento
        esperaria a trava do buffer para sempre; matar antes quebra o pipe.
        """
        backend, client, _, _ = initialized_backend(tmp_path, codex_sdk, turn_timeout=0.05)
        espera = unanswered_request(client)
        ordem = []
        client._client = MagicMock()
        client._client._proc.kill.side_effect = lambda: ordem.append("kill")
        fecha = client.close.side_effect
        client.close.side_effect = lambda: ordem.append("close") or fecha()
        client.thread_start.side_effect = espera

        with (
            pytest.warns(UserWarning, match="não-recuperável"),
            pytest.raises(ProviderAbortError, match="thread/start"),
        ):
            backend.invoke("texto")

        assert ordem == ["kill", "close"]

    @pytest.mark.parametrize(
        ("leitor_do_stdout_vivo", "leitor_do_stderr_vivo", "esperado"),
        [
            (False, False, ["close", "stdout", "stderr"]),
            (True, False, ["close", "stderr"]),
            (False, True, ["close", "stdout"]),
        ],
    )
    def test_pipe_fecha_depois_do_close_e_so_sem_leitor_vivo(
        self, leitor_do_stdout_vivo, leitor_do_stderr_vivo, esperado
    ):
        """Os pipes de saída fecham depois do `close` do SDK, que espera o processo.

        O pipe cuja thread de leitura ainda roda fica aberto: fechá-lo esperaria
        a leitura em curso, e a thread do stderr levantaria `ValueError` ao ler
        a linha seguinte do arquivo fechado.
        """
        ordem = []
        cliente = MagicMock()
        processo = cliente._client._proc
        cliente.close.side_effect = lambda: ordem.append("close")
        processo.stdout.close.side_effect = lambda: ordem.append("stdout")
        processo.stderr.close.side_effect = lambda: ordem.append("stderr")
        cliente._client._reader_thread.is_alive.return_value = leitor_do_stdout_vivo
        cliente._client._stderr_thread.is_alive.return_value = leitor_do_stderr_vivo

        _close_client(cliente)

        assert ordem == esperado

    def test_pipes_fecham_mesmo_quando_o_close_do_sdk_levanta(self):
        """O `close` que levanta não deixa os pipes de saída abertos.

        Com o app-server morto e um pedido no buffer do stdin, a descarga do
        `close` levanta antes do `terminate`. As threads de leitura já viram o
        fim do processo, e os dois pipes fecham; a exceção continua subindo.
        """
        erro = BrokenPipeError(32, "Broken pipe")
        cliente = MagicMock()
        processo = cliente._client._proc
        cliente.close.side_effect = erro
        cliente._client._reader_thread.is_alive.return_value = False
        cliente._client._stderr_thread.is_alive.return_value = False

        with pytest.raises(BrokenPipeError) as excinfo:
            _close_client(cliente)

        assert excinfo.value is erro
        processo.stdout.close.assert_called_once_with()
        processo.stderr.close.assert_called_once_with()

    @pytest.mark.parametrize("returncode", [None, 0], ids=["sem_returncode", "ja_recolhido"])
    def test_pipe_fecha_quando_o_leitor_termina_durante_a_espera(self, returncode):
        """A thread de leitura que ainda roda depois do `close` é esperada antes do pipe fechar.

        Depois de um Ctrl+C no `wait` do SDK, ou do `close` que levanta ao
        descarregar o stdin, as threads de leitura não foram esperadas, e o fim
        do processo ainda não chegou a elas. A espera deixa cada uma ver o EOF
        e sair, e o pipe dela fecha em seguida, em vez de ficar para o coletor
        de lixo.
        """
        cliente = MagicMock()
        processo = cliente._client._proc
        processo.returncode = returncode
        leitores = (_LeitorQueTerminaNaEspera(), _LeitorQueTerminaNaEspera())
        cliente._client._reader_thread, cliente._client._stderr_thread = leitores

        _close_client(cliente)

        processo.stdout.close.assert_called_once_with()
        processo.stderr.close.assert_called_once_with()
        assert [leitor.esperas for leitor in leitores] == [[_READER_THREAD_JOIN]] * 2
        assert processo.kill.call_count == (1 if returncode is None else 0)

    def test_encerramento_pelo_prazo_nao_levanta_quando_o_close_falha(self):
        """O relógio que derruba o app-server termina sem exceção.

        Depois do `kill`, o `close` pode levantar ao descarregar o stdin do
        processo morto. O encerramento roda na thread do relógio, e a falha não
        tem quem a receba: as linhas já abortam pela causa registrada.
        """
        parada = _AppServerStop()
        cliente = MagicMock()
        processo = cliente._client._proc
        cliente.close.side_effect = BrokenPipeError(32, "Broken pipe")
        cliente._client._reader_thread.is_alive.return_value = False
        cliente._client._stderr_thread.is_alive.return_value = False

        parada.stop(cliente, "causa")

        assert str(parada.error()) == "causa"
        processo.kill.assert_called_once_with()
        processo.stdout.close.assert_called_once_with()
        processo.stderr.close.assert_called_once_with()

    def test_dois_estouros_mantem_a_primeira_causa(self):
        """Linhas que estouram juntas abortam todas com a causa de quem chegou primeiro."""
        parada = _AppServerStop()
        cliente = MagicMock()
        processo = cliente._client._proc

        def fecha():
            # Como o SDK: o `close` solta o processo, e o segundo encerramento
            # encontra o cliente já sem ele.
            cliente._client._proc = None

        cliente.close.side_effect = fecha

        parada.stop(cliente, "primeira causa")
        parada.stop(cliente, "segunda causa")

        assert str(parada.error()) == "primeira causa"
        processo.kill.assert_called_once_with()
        assert cliente.close.call_count == 2

    @pytest.mark.parametrize(
        "erro",
        [
            AttributeError("'NoneType' object has no attribute 'stdin'"),
            ValueError("I/O operation on closed file."),
            BrokenPipeError(32, "Broken pipe"),
        ],
    )
    def test_falha_de_outra_linha_depois_do_encerramento_aborta_com_a_causa(
        self, codex_sdk, tmp_path, erro
    ):
        """A linha em voo que pega o cliente no meio do fechamento aborta pela causa.

        Sem o estado compartilhado, o erro da escrita interrompida viraria falha
        definitiva da linha, gravada como 'error' e não retomada.
        """
        backend, client, _, _ = initialized_backend(tmp_path, codex_sdk, turn_timeout=0.05)
        espera = unanswered_request(client)
        tentativas = []

        def abre_thread(**_):
            tentativas.append(1)
            if len(tentativas) == 1:
                return espera()
            raise erro

        client.thread_start.side_effect = abre_thread

        with pytest.warns(UserWarning, match="não-recuperável"), pytest.raises(ProviderAbortError):
            backend.invoke("primeira")
        with (
            pytest.warns(UserWarning, match="não-recuperável"),
            pytest.raises(ProviderAbortError, match="não respondeu a thread/start") as excinfo,
        ):
            backend.invoke("segunda")

        assert excinfo.value.__cause__ is erro


class _LeitorQueTerminaNaEspera:
    """Thread de leitura viva que vê o fim do processo durante o `join`."""

    def __init__(self) -> None:
        self.vivo = True
        self.esperas: list[float | None] = []

    def is_alive(self) -> bool:
        return self.vivo

    def join(self, timeout: float | None = None) -> None:
        self.esperas.append(timeout)
        self.vivo = False


class _ClienteDoProtocolo:
    """Cliente de protocolo com o `close` da classe, como o do SDK."""

    _proc = None

    def close(self) -> None:
        pass


class TestAberturaDoCliente:
    """`_open_client` sobre um `Codex` falso, no lugar do construtor do SDK."""

    @staticmethod
    def _sdk_com(monkeypatch, construtor):
        modulo = ModuleType("openai_codex")
        vars(modulo)["Codex"] = type("Codex", (), {"__init__": construtor})
        monkeypatch.setitem(sys.modules, "openai_codex", modulo)

    @pytest.mark.parametrize(
        "atribui",
        [
            lambda codex, cliente: codex.__dict__.__setitem__("_client", cliente),
            lambda codex, cliente: setattr(codex, "_rpc", cliente),
        ],
        ids=["client_pelo_dict", "client_com_outro_nome"],
    )
    def test_abertura_segue_quando_o_close_nao_e_trocado(self, monkeypatch, atribui):
        """O SDK que não atribui `_client` pelo `__setattr__` abre como abriria sem a subclasse.

        Sem a troca do `close`, a subclasse não tem o que desfazer depois da
        abertura, e o cliente sai dela com o `close` da classe.
        """
        cliente = _ClienteDoProtocolo()

        def construtor(codex, config):
            atribui(codex, cliente)

        self._sdk_com(monkeypatch, construtor)

        aberto = codex_module._open_client(MagicMock())

        assert cliente in vars(aberto).values()
        assert "close" not in vars(cliente)

    def test_close_trocado_sem_processo_em_proc_fecha_como_o_do_sdk(self, monkeypatch):
        """Com o processo fora de `_proc`, o `close` trocado chama o do SDK e não recolhe nada."""
        fechamentos = []

        class ClienteSemProc:
            def close(self) -> None:
                fechamentos.append(self)

        cliente = ClienteSemProc()

        def construtor(codex, config):
            codex._client = cliente

        self._sdk_com(monkeypatch, construtor)
        aberto = codex_module._open_client(MagicMock())

        cliente.close()

        assert fechamentos == [cliente]
        assert aberto._client is cliente

    def test_ctrl_c_na_abertura_sobe_quando_o_close_do_sdk_levanta(self, monkeypatch):
        """O `close` que levanta durante o Ctrl+C da abertura não troca o `KeyboardInterrupt`.

        O `close` do SDK descarrega o stdin, e a descarga levanta
        `BrokenPipeError` quando o app-server já não lê o pipe. O processo é
        recolhido mesmo assim, e o que sobe é a interrupção.
        """
        interrupcao = KeyboardInterrupt()
        cliente = MagicMock()
        processo = cliente._proc
        close_do_sdk = cliente.close
        close_do_sdk.side_effect = BrokenPipeError(32, "Broken pipe")
        cliente._reader_thread.is_alive.return_value = False
        cliente._stderr_thread.is_alive.return_value = False

        def construtor(codex, config):
            codex._client = cliente
            raise interrupcao

        self._sdk_com(monkeypatch, construtor)

        with pytest.raises(KeyboardInterrupt) as excinfo:
            codex_module._open_client(MagicMock())

        assert excinfo.value is interrupcao
        close_do_sdk.assert_called_once_with()
        processo.stdout.close.assert_called_once_with()
        processo.stderr.close.assert_called_once_with()


class TestConfiguracaoDoPrazo:
    def test_prazo_padrao_e_de_600_segundos(self):
        assert _turn_timeout(make_config()) == 600

    @pytest.mark.parametrize("valor", [1, 0.5, 3600, threading.TIMEOUT_MAX])
    def test_prazo_positivo_e_aceito(self, valor):
        assert _turn_timeout(make_config(model_kwargs={"timeout": valor})) == valor

    def test_none_desliga_o_prazo(self):
        assert _turn_timeout(make_config(model_kwargs={"timeout": None})) is None

    @pytest.mark.parametrize(
        "valor",
        [0, -1, "10", True, float("inf"), float("nan"), threading.TIMEOUT_MAX + 1, 10**400],
        ids=["zero", "negativo", "texto", "bool", "infinito", "nan", "acima-do-teto", "gigante"],
    )
    def test_prazo_invalido_e_recusado(self, valor):
        with pytest.raises(ProviderConfigurationError, match="timeout inválido"):
            _turn_timeout(make_config(model_kwargs={"timeout": valor}))

    def test_timeout_e_chave_aceita_em_model_kwargs(self, codex_sdk):
        _, sdk_types, _ = codex_sdk

        effort = _validate_config(make_config(model_kwargs={"timeout": 30, "effort": "low"}))

        assert effort is sdk_types.ReasoningEffort.low


class TestMensagensEAtrasos:
    def test_stream_sem_conclusao_em_todas_as_tentativas_diz_o_motivo(self, codex_sdk, tmp_path):
        backend, _, _, turn = initialized_backend(tmp_path, codex_sdk)
        turn.stream.side_effect = lambda: as_stream(make_result(codex_sdk)[:-1])

        with (
            pytest.warns(UserWarning, match="Tentativa 1/2"),
            pytest.raises(
                ProviderTransientError,
                match=r"^O stream do turno Codex terminou sem o evento de conclusão$",
            ),
        ):
            backend.invoke("texto")

    def test_resposta_vazia_diz_o_motivo(self, codex_sdk, tmp_path):
        backend, _, _, _ = initialized_backend(
            tmp_path, codex_sdk, make_result(codex_sdk, response="  ")
        )

        with (
            pytest.warns(UserWarning, match="não-recuperável"),
            pytest.raises(ProviderOutputError, match=r"^Codex retornou resposta vazia$"),
        ):
            backend.invoke("texto")

    def test_prazo_invalido_diz_o_que_aceitar(self):
        with pytest.raises(
            ProviderConfigurationError,
            match=r"Use um número de segundos maior que zero, ou None para não ter prazo\.$",
        ):
            _turn_timeout(make_config(model_kwargs={"timeout": 0}))

    def test_atraso_entre_tentativas_segue_base_e_teto_da_configuracao(self, codex_sdk, tmp_path):
        _, _, generated = codex_sdk
        backend, _, _, turn = initialized_backend(tmp_path, codex_sdk)
        backend = dataclasses.replace(
            backend, config=make_config(max_retries=3, base_delay=4, max_delay=5)
        )
        overloaded = generated.CodexErrorInfo(root=generated.CodexErrorInfoValue.server_overloaded)
        turn.stream.side_effect = [
            as_stream(make_result(codex_sdk, response=None, error_info=overloaded)),
            as_stream(make_result(codex_sdk, response=None, error_info=overloaded)),
            as_stream(make_result(codex_sdk)),
        ]

        with (
            patch("dataframeit.errors.time.sleep") as sleep,
            pytest.warns(UserWarning, match="Tentativa 2/3"),
        ):
            backend.invoke("texto")

        primeiro, segundo = (chamada.args[0] for chamada in sleep.call_args_list)
        assert 4 <= primeiro <= 4.4
        assert 5 <= segundo <= 5.5

    def test_falha_do_interrupt_nao_escapa_da_thread(self, codex_sdk, tmp_path, relogio_do_stream):
        backend, _, thread, normal = initialized_backend(tmp_path, codex_sdk, turn_timeout=0.05)
        travado, protocol_client = stuck_turn(codex_sdk)
        tentou = threading.Event()

        def interrupt_que_falha(*_):
            tentou.set()
            msg = "app-server recusou"
            raise RuntimeError(msg)

        protocol_client.turn_interrupt.side_effect = interrupt_que_falha
        thread.turn.side_effect = [travado, normal]
        escapadas = []

        with patch.object(threading, "excepthook", escapadas.append):
            with pytest.warns(UserWarning, match="Tentativa 1/2"):
                backend.invoke("texto")
            assert tentou.wait(5)
            for relogio in relogio_do_stream:
                relogio.thread.join(5)

        assert escapadas == []

    def test_thread_do_interrupt_do_reroteamento_e_daemon(self, codex_sdk, tmp_path):
        _, _, generated = codex_sdk
        from openai_codex.models import Notification  # noqa: PLC0415 (SDK carregado pelo fixture)

        rerouted = Notification(
            method="model/rerouted",
            payload=generated.ModelReroutedNotification(
                fromModel="gpt-6-luna",
                toModel="outro",
                reason=generated.ModelRerouteReason("highRiskCyberActivity"),
                threadId="thread-1",
                turnId="turn-1",
            ),
        )
        backend, _, _, turn = initialized_backend(tmp_path, codex_sdk, [rerouted])
        liberar = threading.Event()
        turn.interrupt.side_effect = liberar.wait

        try:
            with pytest.warns(UserWarning, match="não-recuperável"), pytest.raises(ProviderError):
                backend.invoke("texto")
            presas = [t for t in threading.enumerate() if "_interrupt_quietly" in t.name]
            assert presas
            assert all(t.daemon for t in presas)
        finally:
            liberar.set()


class TestDescargaDaThread:
    """Cada tentativa pede ao app-server que descarregue a sua thread efêmera."""

    def test_thread_da_linha_e_descarregada_depois_do_turno(self, codex_sdk, tmp_path):
        from openai_codex.generated.v2_all import (  # noqa: PLC0415 (SDK carregado pelo fixture)
            ThreadUnsubscribeResponse,
        )

        backend, client, _, turn = initialized_backend(tmp_path, codex_sdk)
        ordem = []

        def stream_do_turno():
            ordem.append("turno")
            return as_stream(make_result(codex_sdk))

        turn.stream.side_effect = stream_do_turno
        client._client.request.side_effect = lambda *_, **__: ordem.append("descarga")

        result = backend.invoke("texto")
        aguardar_descargas()

        assert result["data"] == {"sentimento": "positivo", "confianca": 0.9}
        assert ordem == ["turno", "descarga"]
        client._client.request.assert_called_once_with(
            "thread/unsubscribe",
            {"threadId": "thread-1"},
            response_model=ThreadUnsubscribeResponse,
        )

    def test_cada_tentativa_descarrega_a_propria_thread(self, codex_sdk, tmp_path):
        sdk, _, generated = codex_sdk
        backend, client, primeira, _ = initialized_backend(
            tmp_path,
            codex_sdk,
            make_result(codex_sdk, error_info=generated.CodexErrorInfoValue.server_overloaded),
        )
        segunda = MagicMock(spec=sdk.Thread)
        segunda.id = "thread-2"
        turno_ok = MagicMock(spec=sdk.TurnHandle)
        turno_ok.id = "turn-1"
        turno_ok.stream.side_effect = lambda: as_stream(make_result(codex_sdk))
        segunda.turn.return_value = turno_ok
        client.thread_start.side_effect = [primeira, segunda]

        with pytest.warns(UserWarning, match="Tentativa 1/2"):
            result = backend.invoke("texto")
        aguardar_descargas()

        assert result["_retry_info"]["retries"] == 1
        assert sorted(threads_descarregadas(client)) == ["thread-1", "thread-2"]

    def test_thread_e_descarregada_quando_o_turno_nem_comeca(self, codex_sdk, tmp_path):
        backend, client, thread, _ = initialized_backend(tmp_path, codex_sdk)
        thread.turn.side_effect = RuntimeError("turn/start recusado")

        with (
            pytest.warns(UserWarning, match="não-recuperável"),
            pytest.raises(ProviderError, match="turn/start recusado"),
        ):
            backend.invoke("texto")
        aguardar_descargas()

        assert threads_descarregadas(client) == ["thread-1"]

    def test_sem_thread_aberta_nada_e_descarregado(self, codex_sdk, tmp_path):
        backend, client, _, _ = initialized_backend(tmp_path, codex_sdk)
        client.thread_start.side_effect = RuntimeError("thread/start recusado")

        with pytest.warns(UserWarning, match="não-recuperável"), pytest.raises(ProviderError):
            backend.invoke("texto")
        aguardar_descargas()

        assert threads_descarregadas(client) == []

    @pytest.mark.usefixtures("relogio_do_stream")
    def test_descarga_que_trava_nao_prende_a_linha(self, codex_sdk, tmp_path):
        backend, client, thread, normal = initialized_backend(
            tmp_path, codex_sdk, turn_timeout=0.05
        )
        thread.turn.side_effect = [stuck_turn(codex_sdk)[0], normal]
        liberar = threading.Event()
        client._client.request.side_effect = lambda *_, **__: liberar.wait()

        inicio = time.monotonic()
        try:
            with pytest.warns(UserWarning, match="Tentativa 1/2"):
                result = backend.invoke("texto")
            presas = [t for t in threading.enumerate() if "_unsubscribe_quietly" in t.name]
            assert presas
            assert all(t.daemon for t in presas)
        finally:
            liberar.set()
        aguardar_descargas()

        assert time.monotonic() - inicio < 5
        assert result["data"] == {"sentimento": "positivo", "confianca": 0.9}
        # As duas tentativas usam a mesma thread simulada.
        assert threads_descarregadas(client) == ["thread-1", "thread-1"]

    def test_fechamento_solta_a_descarga_escrita_depois_do_fim_do_leitor(self, codex_sdk):
        """A descarga que escreveu depois do `fail_all` do leitor termina no fechamento.

        O app-server morto ainda pode aceitar a escrita enquanto termina, e o SDK
        registra a espera da resposta sem conferir que o leitor do stdout já
        acordou os pedidos em voo e saiu. Ninguém responde nem acorda essa
        espera, e a thread da descarga ficaria presa para sempre.
        """
        from openai_codex.client import CodexClient  # noqa: PLC0415 (SDK carregado pelo fixture)
        from openai_codex.errors import (  # noqa: PLC0415 (SDK carregado pelo fixture)
            TransportClosedError,
        )

        cliente_do_protocolo = CodexClient()
        processo = MagicMock()
        processo.returncode = -9
        escrita = threading.Event()
        processo.stdin.write.side_effect = lambda _linha: escrita.set()
        leitor_encerrado = threading.Thread(target=lambda: None)
        leitor_encerrado.start()
        leitor_encerrado.join()
        cliente_do_protocolo._proc = processo
        cliente_do_protocolo._reader_thread = leitor_encerrado
        cliente_do_protocolo._stderr_thread = leitor_encerrado
        cliente_do_protocolo._router.fail_all(TransportClosedError("stdout fechado"))
        cliente = MagicMock()
        cliente._client = cliente_do_protocolo
        cliente.close.side_effect = cliente_do_protocolo.close

        descarga = threading.Thread(
            target=codex_module._unsubscribe_quietly, args=(cliente, "thread-1"), daemon=True
        )
        descarga.start()
        assert escrita.wait(5)
        descarga.join(0.2)
        assert descarga.is_alive(), "sem o fechamento, nada acorda a espera da resposta"

        _close_client(cliente)
        descarga.join(5)

        assert not descarga.is_alive()

    def test_turno_reroteado_tambem_descarrega_a_thread(self, codex_sdk, tmp_path):
        _, _, generated = codex_sdk
        from openai_codex.models import Notification  # noqa: PLC0415 (SDK carregado pelo fixture)

        rerouted = Notification(
            method="model/rerouted",
            payload=generated.ModelReroutedNotification(
                fromModel="gpt-6-luna",
                toModel="outro",
                reason=generated.ModelRerouteReason("highRiskCyberActivity"),
                threadId="thread-1",
                turnId="turn-1",
            ),
        )
        backend, client, _, _ = initialized_backend(tmp_path, codex_sdk, [rerouted])

        with pytest.warns(UserWarning, match="não-recuperável"), pytest.raises(ProviderError):
            backend.invoke("texto")
        aguardar_descargas()

        assert threads_descarregadas(client) == ["thread-1"]

    def test_falha_da_descarga_nao_escapa_nem_muda_o_resultado(self, codex_sdk, tmp_path):
        backend, client, _, _ = initialized_backend(tmp_path, codex_sdk)
        client._client.request.side_effect = RuntimeError("app-server recusou")
        escapadas = []

        with patch.object(threading, "excepthook", escapadas.append):
            result = backend.invoke("texto")
            aguardar_descargas()

        assert result["data"] == {"sentimento": "positivo", "confianca": 0.9}
        assert threads_descarregadas(client) == ["thread-1"]
        assert escapadas == []
