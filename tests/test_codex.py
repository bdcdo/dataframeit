"""Testes unitários do adapter Codex e de seu contrato opcional."""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from datetime import date, datetime, timedelta
from datetime import time as time_type
from decimal import Decimal
from ipaddress import IPv4Address, IPv6Address
from pathlib import Path
from typing import Annotated, Any, Literal
from unittest.mock import MagicMock, call, patch
from uuid import UUID

import pytest
from pydantic import UUID4, AnyUrl, BaseModel, ConfigDict, Field, RootModel
from pydantic.errors import PydanticInvalidForJsonSchema

from dataframeit.codex import (
    CodexBackend,
    _build_schema,
    _to_strict_json_schema,
    _turn_timeout,
    _validate_config,
    open_codex_backend,
)
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
        class Modelo(BaseModel):
            valor: Decimal

        with pytest.raises(ProviderConfigurationError, match="lookahead"):
            _to_strict_json_schema(Modelo.model_json_schema())

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
        sdk, _, _ = codex_sdk

        with (
            patch.object(sdk, "Codex") as codex,
            pytest.raises(ProviderConfigurationError, match=message),
        ):
            _validate_config(make_config(**overrides))

        codex.assert_not_called()


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
            patch.object(sdk, "Codex", return_value=client) as codex,
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
            patch.object(sdk, "Codex", return_value=client) as codex,
            open_codex_backend(make_config(), SampleModel, "{texto}") as backend,
        ):
            catalog_override = codex.call_args.args[0].config_overrides[-1]
            assert catalog_override.startswith("model_catalog_json=")
            catalog_path = Path(json.loads(catalog_override.removeprefix("model_catalog_json=")))
            assert catalog_path.parent == backend._workspace.parent
            models = json.loads(catalog_path.read_text(encoding="utf-8"))["models"]

        assert models
        assert all(model["tool_mode"] is None for model in models)
        assert not catalog_path.exists()

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
        sdk, _, _ = codex_sdk
        source_home = tmp_path / "source-home"
        source_home.mkdir()
        (source_home / "auth.json").write_text("{}")
        monkeypatch.setenv("CODEX_HOME", str(source_home))
        listing = subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")

        with (
            patch("dataframeit.codex.subprocess.run", return_value=listing),
            patch.object(sdk, "Codex") as codex,
            pytest.raises(ProviderConfigurationError, match=rf"catálogo de modelos.*{message}"),
            open_codex_backend(make_config(), SampleModel, "{texto}"),
        ):
            pass

        codex.assert_not_called()

    def test_falha_do_runtime_ao_ler_o_catalogo_traz_o_stderr(
        self, codex_sdk, monkeypatch, tmp_path
    ):
        sdk, _, _ = codex_sdk
        source_home = tmp_path / "source-home"
        source_home.mkdir()
        (source_home / "auth.json").write_text("{}")
        monkeypatch.setenv("CODEX_HOME", str(source_home))
        failure = subprocess.CalledProcessError(1, ["codex"], output="", stderr="  sem rede\n")

        with (
            patch("dataframeit.codex.subprocess.run", side_effect=failure),
            patch.object(sdk, "Codex") as codex,
            pytest.raises(
                ProviderConfigurationError, match=r"catálogo de modelos do Codex: sem rede$"
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
            patch.object(sdk, "Codex", return_value=client),
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
            patch.object(sdk, "Codex", return_value=client),
            open_codex_backend(make_config(), SampleModel, "{texto}") as backend,
        ):
            assert backend.config.model == "gpt-6-luna"

        assert threads_descarregadas(client) == ["thread-sonda"]

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
        sdk, _, _ = codex_sdk

        with (
            patch.object(sdk, "Codex") as codex,
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
        sdk, _, _ = codex_sdk
        source_home = tmp_path / "source-home"
        source_home.mkdir()
        monkeypatch.setenv("CODEX_HOME", str(source_home))

        with (
            patch("dataframeit.codex.tempfile.TemporaryDirectory") as temporary_directory,
            patch.object(sdk, "Codex") as codex,
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
            patch.object(sdk, "Codex", side_effect=clients) as codex,
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
        sdk, _, _ = codex_sdk
        source_home = tmp_path / "source-home"
        source_home.mkdir()
        source_auth = source_home / "auth.json"
        source_auth.write_text("{}")
        monkeypatch.setenv("CODEX_HOME", str(source_home))

        with (
            patch("dataframeit.codex.os.link", side_effect=OSError("unsupported")),
            patch.object(Path, "symlink_to") as symlink,
            patch.object(sdk, "Codex") as codex,
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
            patch.object(sdk, "Codex", side_effect=[codex_result]),
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
            patch.object(sdk, "Codex", return_value=client),
            pytest.raises(RuntimeError, match="close failed"),
            open_codex_backend(make_config(), SampleModel, "{texto}"),
        ):
            pass

        assert auth_lock_is_available(source_home / "auth.json.dataframeit.lock")
        assert list(source_home.glob("dataframeit-codex-*")) == []

    def test_runtime_directory_failure_has_accurate_error_and_cleans_up(
        self, codex_sdk, monkeypatch, tmp_path
    ):
        sdk, _, _ = codex_sdk
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
            patch.object(sdk, "Codex") as codex,
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
        sdk, _, _ = codex_sdk
        source_home = tmp_path / "source-home"
        source_home.mkdir()
        (source_home / "auth.json").write_text("{}")
        monkeypatch.setenv("CODEX_HOME", str(source_home))

        with (
            patch("filelock.FileLock.acquire", side_effect=error_type("unsupported")),
            patch.object(sdk, "Codex") as codex,
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

    def test_failed_turn_internal_server_error_retries_without_rate_limit(
        self, codex_sdk, tmp_path
    ):
        _, _, generated = codex_sdk
        backend, client, _, turn = initialized_backend(tmp_path, codex_sdk)
        internal = generated.CodexErrorInfo(
            root=generated.CodexErrorInfoValue.internal_server_error
        )
        turn.stream.side_effect = [
            as_stream(make_result(codex_sdk, error_info=internal, message="internal failure")),
            as_stream(make_result(codex_sdk)),
        ]

        with pytest.warns(UserWarning, match="Tentativa 1/2"):
            result = backend.invoke("texto")

        assert result["_retry_info"]["retries"] == 1
        assert client.thread_start.call_count == 2

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
        sdk, _, _ = codex_sdk
        source_home = tmp_path / "source-home"
        source_home.mkdir()
        (source_home / "auth.json").write_text("{}")
        monkeypatch.setenv("CODEX_HOME", str(source_home))

        with (
            patch(
                "dataframeit.codex.tempfile.TemporaryDirectory",
                side_effect=OSError("disco cheio"),
            ),
            patch.object(sdk, "Codex") as codex,
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
            patch.object(sdk, "Codex", return_value=client),
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
        turn.stream.side_effect = [
            as_stream(make_result(codex_sdk, error_info=caiu, message="stream caiu")),
            as_stream(make_result(codex_sdk)),
        ]

        with pytest.warns(UserWarning, match="Tentativa 1/2"):
            result = backend.invoke("texto")

        assert result["_retry_info"]["retries"] == 1
        assert client.thread_start.call_count == 2

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


class TestPrazoDoTurno:
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

    def test_falha_do_interrupt_nao_escapa_da_thread(self, codex_sdk, tmp_path):
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
            for relogio in threading.enumerate():
                if isinstance(relogio, threading.Timer):
                    relogio.join(5)

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
