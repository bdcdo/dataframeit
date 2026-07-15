"""Testes unitários do adapter Codex e de seu contrato opcional."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from dataframeit.codex import CodexBackend, _to_strict_json_schema
from dataframeit.errors import (
    ProviderConfigurationError,
    ProviderError,
    ProviderOutputError,
    ProviderOverloadedError,
    is_rate_limit_error,
    is_recoverable_error,
)
from dataframeit.llm import LLMConfig


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


class RecursiveModel(BaseModel):
    name: str
    child: RecursiveModel | None = None


def make_config(**overrides) -> LLMConfig:
    values = {
        "model": "gpt-5.4",
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


@pytest.fixture
def codex_sdk():
    """Carrega o SDK real apenas nos testes que exercitam sua fronteira."""
    sdk = pytest.importorskip("openai_codex")
    sdk_types = pytest.importorskip("openai_codex.types")
    generated = pytest.importorskip("openai_codex.generated.v2_all")
    return sdk, sdk_types, generated


def make_result(
    codex_sdk,
    response: str | None = '{"sentimento": "positivo", "confianca": 0.9}',
    *,
    status=None,
    usage: bool = True,
):
    sdk, sdk_types, generated = codex_sdk
    token_usage = generated.TokenUsageBreakdown(
        inputTokens=100,
        cachedInputTokens=40,
        outputTokens=30,
        reasoningOutputTokens=10,
        totalTokens=130,
    )
    thread_usage = (
        sdk_types.ThreadTokenUsage(last=token_usage, total=token_usage) if usage else None
    )
    return sdk.TurnResult(
        id="turn-1",
        status=status or sdk_types.TurnStatus.completed,
        error=None,
        started_at=1,
        completed_at=2,
        duration_ms=1,
        final_response=response,
        items=[],
        usage=thread_usage,
    )


def initialized_backend(tmp_path, codex_sdk, result=None):
    sdk, _, _ = codex_sdk
    backend = CodexBackend(make_config(), SampleModel, "Analise: {texto}")
    backend._workspace = tmp_path / "workspace"
    backend._workspace.mkdir()

    turn = MagicMock(spec=sdk.TurnHandle)
    turn.id = "turn-1"
    turn.run.return_value = result or make_result(codex_sdk)
    thread = MagicMock(spec=sdk.Thread)
    thread.turn.return_value = turn
    client = MagicMock(spec=sdk.Codex)
    client.thread_start.return_value = thread
    backend._client = client
    return backend, client, thread, turn


class TestProviderDependency:
    def test_missing_sdk_reports_only_codex_extra(self):
        from dataframeit.errors import validate_provider_dependencies

        with patch("importlib.import_module", side_effect=ImportError("missing")):
            with pytest.raises(ImportError) as exc_info:
                validate_provider_dependencies("codex")

        message = str(exc_info.value)
        assert "dataframeit[codex]" in message
        assert "dataframeit[all]" not in message

    def test_langchain_provider_keeps_all_extra_as_alternative(self):
        from dataframeit.errors import validate_provider_dependencies

        def import_module(name):
            if name == "langchain_google_genai":
                raise ImportError("missing")
            return MagicMock()

        with patch("importlib.import_module", side_effect=import_module):
            with pytest.raises(ImportError) as exc_info:
                validate_provider_dependencies("google_genai")

        message = str(exc_info.value)
        assert "langchain-google-genai" in message
        assert "dataframeit[all]" in message

    def test_sdk_provider_skips_langchain_validation(self):
        from dataframeit.errors import validate_provider_dependencies

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

    def test_discriminated_one_of_preserves_mapping_and_strict_variants(self):
        schema = _to_strict_json_schema(ModelWithDiscriminatedUnion.model_json_schema())

        animal = schema["properties"]["animal"]
        assert animal["oneOf"] == [
            {"$ref": "#/$defs/CatModel"},
            {"$ref": "#/$defs/DogModel"},
        ]
        assert animal["discriminator"]["propertyName"] == "kind"
        assert animal["discriminator"]["mapping"] == {
            "cat": "#/$defs/CatModel",
            "dog": "#/$defs/DogModel",
        }
        assert schema["$defs"]["CatModel"]["additionalProperties"] is False
        assert schema["$defs"]["DogModel"]["additionalProperties"] is False

    def test_dynamic_dict_is_rejected_from_real_pydantic_schema(self):
        with pytest.raises(ProviderConfigurationError, match="chaves dinâmicas"):
            _to_strict_json_schema(ModelWithDynamicKeys.model_json_schema())

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
    def test_effort_defaults_to_real_medium_enum(self, codex_sdk):
        _, sdk_types, _ = codex_sdk

        backend = CodexBackend(make_config(), SampleModel, "{texto}")

        assert backend._effort is sdk_types.ReasoningEffort.medium

    def test_effort_is_the_only_supported_model_kwarg(self, codex_sdk):
        _, sdk_types, _ = codex_sdk

        backend = CodexBackend(make_config(model_kwargs={"effort": "high"}), SampleModel, "{texto}")

        assert backend._effort is sdk_types.ReasoningEffort.high

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"api_key": "secret"}, "não passe api_key"),
            ({"model_kwargs": {"temperature": 0}}, "temperature"),
            ({"model_kwargs": {"codex_bin": "/some/codex"}}, "codex_bin"),
            ({"model_kwargs": {"effort": "maximum"}}, "effort inválido"),
        ],
    )
    def test_invalid_config_fails_before_client_start(self, codex_sdk, overrides, message):
        sdk, _, _ = codex_sdk

        with patch.object(sdk, "Codex") as codex:
            with pytest.raises(ProviderConfigurationError, match=message):
                CodexBackend(make_config(**overrides), SampleModel, "{texto}")

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

        client = MagicMock(spec=sdk.Codex)
        client.account.return_value = sdk_types.GetAccountResponse(requiresOpenaiAuth=False)

        with patch.object(sdk, "Codex", return_value=client) as codex:
            with CodexBackend(make_config(), SampleModel, "{texto}") as backend:
                launch_config = codex.call_args.args[0]
                assert isinstance(launch_config, sdk.CodexConfig)
                assert launch_config.codex_bin is None
                workspace = Path(launch_config.cwd)
                isolated_home = Path(launch_config.env["CODEX_HOME"])
                assert isolated_home.parent == workspace.parent
                assert launch_config.env["CODEX_SQLITE_HOME"] == str(isolated_home)
                assert isolated_home != source_home
                assert (isolated_home / "auth.json").is_symlink()
                assert (isolated_home / "auth.json").resolve() == source_auth.resolve()
                assert not (isolated_home / "config.toml").exists()
                assert "project_doc_max_bytes=0" in launch_config.config_overrides
                assert "mcp_servers={}" in launch_config.config_overrides
                assert "features.shell_tool=false" in launch_config.config_overrides
                assert not any(
                    "model_reasoning_effort" in item for item in launch_config.config_overrides
                )
                assert backend._client is client

        client.close.assert_called_once_with()
        assert not workspace.parent.exists()

    def test_missing_auth_closes_client_and_removes_runtime(self, codex_sdk, monkeypatch, tmp_path):
        sdk, sdk_types, _ = codex_sdk
        source_home = tmp_path / "source-home"
        source_home.mkdir()
        monkeypatch.setenv("CODEX_HOME", str(source_home))
        client = MagicMock(spec=sdk.Codex)
        client.account.return_value = sdk_types.GetAccountResponse(requiresOpenaiAuth=True)
        backend = CodexBackend(make_config(), SampleModel, "{texto}")

        with patch.object(sdk, "Codex", return_value=client) as codex:
            with pytest.raises(ProviderConfigurationError, match="codex login"):
                with backend:
                    pass

        launch_config = codex.call_args.args[0]
        runtime_root = Path(launch_config.cwd).parent
        client.close.assert_called_once_with()
        assert backend._client is None
        assert backend._runtime is None
        assert not runtime_root.exists()


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
        assert start_kwargs["model"] == "gpt-5.4"
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

        with pytest.warns(UserWarning, match="não-recuperável"):
            with pytest.raises(ProviderOutputError, match="não corresponde ao schema"):
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

        with pytest.warns(UserWarning, match="não-recuperável"):
            with pytest.raises(ProviderOutputError, match=message):
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
        _, sdk_types, generated = codex_sdk
        backend, client, thread, turn = initialized_backend(tmp_path, codex_sdk)
        turn.run.side_effect = [RuntimeError("overloaded"), make_result(codex_sdk)]
        failed_turn = sdk_types.Turn(
            id="turn-1",
            items=[],
            status=sdk_types.TurnStatus.failed,
            error=sdk_types.TurnError(
                message="overloaded",
                codexErrorInfo=generated.CodexErrorInfo(
                    root=generated.CodexErrorInfoValue.server_overloaded
                ),
            ),
        )
        protocol_thread = generated.Thread.model_construct(turns=[failed_turn])
        thread.read.return_value = sdk_types.ThreadReadResponse.model_construct(
            thread=protocol_thread
        )

        with pytest.warns(UserWarning, match="Tentativa 1/2"):
            result = backend.invoke("texto")

        assert result["_retry_info"]["retries"] == 1
        assert client.thread_start.call_count == 2
        thread.read.assert_called_once_with(include_turns=True)

    def test_unknown_sdk_error_is_provider_error_without_retry(self, codex_sdk, tmp_path):
        backend, client, _, _ = initialized_backend(tmp_path, codex_sdk)
        client.thread_start.side_effect = RuntimeError("unexpected")

        with pytest.warns(UserWarning, match="não-recuperável"):
            with pytest.raises(ProviderError, match="RuntimeError: unexpected"):
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
            turn.id = f"turn-{response}"
            turn.run.return_value = result
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
