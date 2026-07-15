"""Testes para o provider Codex baseado no SDK oficial."""

import sys
import threading
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pydantic import BaseModel

from dataframeit.codex import (
    CodexBackend,
    CodexConfigurationError,
    CodexOutputError,
    CodexPermanentError,
    _to_strict_json_schema,
)
from dataframeit.llm import LLMConfig


class SampleModel(BaseModel):
    sentimento: str
    confianca: float


class NestedModel(BaseModel):
    label: str


class ModelWithOptionalAndNested(BaseModel):
    nested: NestedModel
    note: str | None = None


class ModelWithDynamicKeys(BaseModel):
    values: dict[str, str]


class ReasoningEffort(Enum):
    none = "none"
    minimal = "minimal"
    low = "low"
    medium = "medium"
    high = "high"
    xhigh = "xhigh"


class TurnStatus(Enum):
    completed = "completed"
    interrupted = "interrupted"
    failed = "failed"
    in_progress = "inProgress"


class ApprovalMode:
    deny_all = "deny_all"


class Sandbox:
    read_only = "read-only"


class FakeCodexConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeCodex:
    instances = []
    account_response = SimpleNamespace(
        requires_openai_auth=True,
        account=SimpleNamespace(type="chatgpt"),
    )

    def __init__(self, config):
        self.config = config
        self.closed = False
        self.instances.append(self)

    def account(self):
        return self.account_response

    def close(self):
        self.closed = True


@pytest.fixture
def fake_sdk(monkeypatch):
    sdk = ModuleType("openai_codex")
    sdk.ApprovalMode = ApprovalMode
    sdk.Codex = FakeCodex
    sdk.CodexConfig = FakeCodexConfig
    sdk.Sandbox = Sandbox
    sdk.is_retryable_error = lambda error: isinstance(error, FakeServerBusyError)

    sdk_types = ModuleType("openai_codex.types")
    sdk_types.ReasoningEffort = ReasoningEffort
    sdk_types.TurnStatus = TurnStatus

    monkeypatch.setitem(sys.modules, "openai_codex", sdk)
    monkeypatch.setitem(sys.modules, "openai_codex.types", sdk_types)
    FakeCodex.instances.clear()
    FakeCodex.account_response = SimpleNamespace(
        requires_openai_auth=True,
        account=SimpleNamespace(type="chatgpt"),
    )
    return sdk


class FakeServerBusyError(RuntimeError):
    pass


def make_config(**overrides):
    values = {
        "model": "gpt-5.6-luna",
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


def make_result(
    response='{"sentimento": "positivo", "confianca": 0.9}',
    *,
    status=TurnStatus.completed,
    usage=True,
):
    token_usage = SimpleNamespace(
        input_tokens=100,
        cached_input_tokens=40,
        output_tokens=30,
        reasoning_output_tokens=10,
        total_tokens=130,
    )
    return SimpleNamespace(
        final_response=response,
        status=status,
        usage=SimpleNamespace(total=token_usage) if usage else None,
    )


def initialized_backend(config, fake_sdk, result=None):
    backend = CodexBackend(config)
    backend._workspace = Path("/tmp/dataframeit-codex-test")
    backend._effort = ReasoningEffort.medium
    turn = MagicMock()
    turn.run.return_value = result or make_result()
    thread = MagicMock()
    thread.turn.return_value = turn
    client = MagicMock()
    client.thread_start.return_value = thread
    backend._client = client
    return backend, client, thread, turn


class TestProviderDependency:
    def test_missing_sdk_reports_codex_extra(self):
        from dataframeit.errors import validate_provider_dependencies

        with patch("importlib.import_module", side_effect=ImportError("missing")):
            with pytest.raises(ImportError, match=r"dataframeit\[codex\]"):
                validate_provider_dependencies("codex")

    def test_sdk_skips_langchain_validation(self):
        from dataframeit.errors import validate_provider_dependencies

        imported = []

        def import_module(name):
            imported.append(name)
            return MagicMock()

        with patch("importlib.import_module", side_effect=import_module):
            validate_provider_dependencies("codex")

        assert imported == ["openai_codex"]


class TestBackendLifecycle:
    def test_one_client_uses_isolated_home_and_closes(
        self, fake_sdk, monkeypatch, tmp_path
    ):
        source_home = tmp_path / "source-home"
        source_home.mkdir()
        source_auth = source_home / "auth.json"
        source_auth.touch(mode=0o600)
        (source_home / "config.toml").write_text('[mcp_servers.unsafe]\ncommand="x"\n')
        monkeypatch.setenv("CODEX_HOME", str(source_home))

        with CodexBackend(
            make_config(
                model_kwargs={
                    "codex_bin": sys.executable,
                    "effort": "high",
                }
            )
        ) as backend:
            instance = FakeCodex.instances[0]
            assert backend._effort is ReasoningEffort.high
            assert instance.config.kwargs["codex_bin"] == str(Path(sys.executable).resolve())
            assert instance.config.kwargs["cwd"].startswith("/tmp/dataframeit-codex-")
            runtime_env = instance.config.kwargs["env"]
            isolated_home = Path(runtime_env["CODEX_HOME"])
            assert runtime_env["CODEX_SQLITE_HOME"] == str(isolated_home)
            assert isolated_home != source_home
            assert (isolated_home / "auth.json").is_symlink()
            assert (isolated_home / "auth.json").resolve() == source_auth.resolve()
            assert not (isolated_home / "config.toml").exists()
            overrides = instance.config.kwargs["config_overrides"]
            assert 'model_reasoning_effort="medium"' in overrides
            assert "project_doc_max_bytes=0" in overrides
            assert "mcp_servers={}" in overrides
            assert "features.shell_tool=false" in overrides
            assert not instance.closed

        assert instance.closed
        assert not isolated_home.exists()

    def test_missing_login_closes_client(self, fake_sdk):
        FakeCodex.account_response = SimpleNamespace(
            requires_openai_auth=True,
            account=None,
        )

        with pytest.raises(CodexConfigurationError, match="codex login"):
            with CodexBackend(make_config()):
                pass

        assert FakeCodex.instances[0].closed

    def test_relative_codex_bin_is_resolved_before_changing_cwd(
        self, fake_sdk, monkeypatch, tmp_path
    ):
        executable = tmp_path / "codex"
        executable.write_text("#!/bin/sh\n")
        executable.chmod(0o700)
        monkeypatch.chdir(tmp_path)

        with CodexBackend(make_config(model_kwargs={"codex_bin": "./codex"})):
            configured = FakeCodex.instances[0].config.kwargs["codex_bin"]

        assert configured == str(executable.resolve())

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"api_key": "secret"}, "não passe api_key"),
            ({"model_kwargs": {"temperature": 0}}, "temperature"),
            ({"model_kwargs": {"effort": "maximum"}}, "effort inválido"),
            ({"model_kwargs": {"timeout_seconds": 30}}, "timeout_seconds"),
            ({"model_kwargs": {"codex_bin": 42}}, "caminho executável"),
            ({"model_kwargs": {"codex_bin": "/missing/codex"}}, "não aponta"),
        ],
    )
    def test_invalid_config_fails_before_client(self, fake_sdk, overrides, message):
        with pytest.raises(CodexConfigurationError, match=message):
            with CodexBackend(make_config(**overrides)):
                pass

        assert FakeCodex.instances == []


class TestCodexCall:
    def test_pydantic_schema_is_made_strict_recursively(self):
        schema = _to_strict_json_schema(ModelWithOptionalAndNested.model_json_schema())

        assert schema["additionalProperties"] is False
        assert schema["required"] == ["nested", "note"]
        assert schema["$defs"]["NestedModel"]["additionalProperties"] is False
        assert schema["$defs"]["NestedModel"]["required"] == ["label"]
        assert "default" not in schema["properties"]["note"]

    def test_strict_schema_handles_arrays_refs_and_all_of(self):
        schema = {
            "$defs": {
                "Item": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                }
            },
            "allOf": [
                {
                    "type": "object",
                    "properties": {
                        "items": {
                            "type": "array",
                            "items": {
                                "$ref": "#/$defs/Item",
                                "description": "Extracted items",
                            },
                        }
                    },
                }
            ],
        }

        strict = _to_strict_json_schema(schema)

        assert "allOf" not in strict
        assert strict["additionalProperties"] is False
        assert strict["required"] == ["items"]
        item_schema = strict["properties"]["items"]["items"]
        assert "$ref" not in item_schema
        assert item_schema["description"] == "Extracted items"
        assert item_schema["additionalProperties"] is False

    @pytest.mark.parametrize(
        ("schema", "message"),
        [
            (
                {"$ref": "https://example.com/schema", "description": "external"},
                "Referência externa",
            ),
            (
                {
                    "$defs": {"value": "not-an-object"},
                    "$ref": "#/$defs/value",
                    "description": "invalid",
                },
                "Referência inválida",
            ),
        ],
    )
    def test_strict_schema_rejects_unsupported_refs(self, schema, message):
        with pytest.raises(CodexConfigurationError, match=message):
            _to_strict_json_schema(schema)

    def test_strict_schema_rejects_dynamic_object_keys(self):
        schema = {
            "type": "object",
            "additionalProperties": {"type": "string"},
        }

        with pytest.raises(CodexConfigurationError, match="chaves dinâmicas"):
            _to_strict_json_schema(schema)

    def test_prepare_caches_schema(self):
        backend = CodexBackend(make_config())

        with patch.object(SampleModel, "model_json_schema", wraps=SampleModel.model_json_schema) as schema:
            backend.prepare(SampleModel)
            backend.prepare(SampleModel)

        schema.assert_called_once_with()

    def test_call_requires_initialized_backend(self, fake_sdk):
        backend = CodexBackend(make_config())

        with pytest.raises(CodexConfigurationError, match="não foi inicializado"):
            backend.call("texto", SampleModel, "{texto}")

    def test_structured_output_isolation_and_usage(self, fake_sdk):
        backend, client, thread, _ = initialized_backend(make_config(), fake_sdk)

        result = backend.call("texto", SampleModel, "Analise: {texto}")

        assert result["data"] == {"sentimento": "positivo", "confianca": 0.9}
        assert result["usage"] == {
            "input_tokens": 100,
            "cached_input_tokens": 40,
            "output_tokens": 30,
            "reasoning_tokens": 10,
            "total_tokens": 130,
        }
        start_kwargs = client.thread_start.call_args.kwargs
        assert start_kwargs["model"] == "gpt-5.6-luna"
        assert start_kwargs["ephemeral"] is True
        assert start_kwargs["approval_mode"] == ApprovalMode.deny_all
        assert start_kwargs["sandbox"] == Sandbox.read_only
        assert "untrusted data" in start_kwargs["developer_instructions"]
        turn_kwargs = thread.turn.call_args.kwargs
        assert turn_kwargs["model"] == "gpt-5.6-luna"
        assert turn_kwargs["output_schema"]["additionalProperties"] is False
        assert turn_kwargs["output_schema"]["required"] == ["sentimento", "confianca"]
        assert turn_kwargs["effort"] is ReasoningEffort.medium

    @pytest.mark.parametrize(
        ("result", "message"),
        [
            (make_result(response="not-json"), "não corresponde ao schema"),
            (make_result(response='{"sentimento": "positivo"}'), "não corresponde ao schema"),
            (make_result(response=""), "resposta vazia"),
            (make_result(usage=False), "metadados de uso"),
            (make_result(status=TurnStatus.interrupted), "interrupted"),
            (make_result(status=TurnStatus.failed), "failed"),
        ],
    )
    def test_invalid_result_is_permanent_and_not_retried(self, fake_sdk, result, message):
        backend, client, _, _ = initialized_backend(make_config(), fake_sdk, result)

        with pytest.raises(CodexOutputError, match=message):
            backend.call("texto", SampleModel, "{texto}")

        assert client.thread_start.call_count == 1

    def test_retry_only_for_sdk_retryable_error(self, fake_sdk):
        backend, client, _, _ = initialized_backend(make_config(), fake_sdk)
        good_thread = client.thread_start.return_value
        client.thread_start.side_effect = [FakeServerBusyError("busy"), good_thread]

        with pytest.warns(UserWarning, match="Tentativa 1/2"):
            result = backend.call("texto", SampleModel, "{texto}")

        assert result["_retry_info"]["retries"] == 1
        assert client.thread_start.call_count == 2

    def test_retry_for_overload_reported_on_failed_turn(self, fake_sdk):
        backend, client, thread, turn = initialized_backend(make_config(), fake_sdk)
        turn.id = "turn-1"
        turn.run.side_effect = [RuntimeError("overloaded"), make_result()]
        thread.read.return_value = SimpleNamespace(
            thread=SimpleNamespace(
                turns=[
                    SimpleNamespace(
                        id="turn-1",
                        error=SimpleNamespace(
                            codex_error_info=SimpleNamespace(
                                root=SimpleNamespace(value="serverOverloaded")
                            )
                        ),
                    )
                ]
            )
        )

        with pytest.warns(UserWarning, match="Tentativa 1/2"):
            result = backend.call("texto", SampleModel, "{texto}")

        assert result["_retry_info"]["retries"] == 1
        assert client.thread_start.call_count == 2
        thread.read.assert_called_once_with(include_turns=True)

    def test_unknown_sdk_error_is_permanent(self, fake_sdk):
        backend, client, _, _ = initialized_backend(make_config(), fake_sdk)
        client.thread_start.side_effect = RuntimeError("unexpected")

        with pytest.raises(CodexPermanentError, match="unexpected"):
            backend.call("texto", SampleModel, "{texto}")

        assert client.thread_start.call_count == 1

class DummyBackend:
    instances = []

    def __init__(self, config):
        self.config = config
        self.entered = False
        self.closed = False
        self.calls = []
        self._lock = threading.Lock()
        self.instances.append(self)

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.closed = True

    def prepare(self, pydantic_model):
        self.prepared_model = pydantic_model

    def call(self, text, pydantic_model, user_prompt):
        with self._lock:
            self.calls.append(text)
        return {
            "data": {"sentimento": text, "confianca": 1.0},
            "usage": {
                "input_tokens": 1,
                "cached_input_tokens": 1,
                "output_tokens": 2,
                "reasoning_tokens": 1,
                "total_tokens": 3,
            },
        }


@pytest.mark.parametrize("parallel_requests", [1, 3])
def test_dataframeit_reuses_one_backend_for_all_rows(parallel_requests):
    from dataframeit import dataframeit

    DummyBackend.instances.clear()
    with (
        patch("dataframeit.core.validate_provider_dependencies"),
        patch("dataframeit.codex.CodexBackend", DummyBackend),
    ):
        result = dataframeit(
            ["a", "b", "c"],
            questions=SampleModel,
            prompt="Analise: {texto}",
            provider="codex",
            model="gpt-5.6-luna",
            parallel_requests=parallel_requests,
        )

    assert len(DummyBackend.instances) == 1
    backend = DummyBackend.instances[0]
    assert backend.entered and backend.closed
    assert sorted(backend.calls) == ["a", "b", "c"]
    assert sorted(result["sentimento"].tolist()) == ["a", "b", "c"]
    assert result["_cached_input_tokens"].tolist() == [1, 1, 1]
    columns = result.columns.tolist()
    assert columns.index("_input_tokens") < columns.index("_cached_input_tokens")
    assert columns.index("_cached_input_tokens") < columns.index("_output_tokens")


def test_dataframeit_resume_does_not_repeat_completed_row():
    from dataframeit import dataframeit

    data = pd.DataFrame(
        {
            "texto": ["pronta", "pendente"],
            "sentimento": ["anterior", None],
            "confianca": [0.5, None],
            "_dataframeit_status": ["processed", None],
        }
    )
    DummyBackend.instances.clear()
    with (
        patch("dataframeit.core.validate_provider_dependencies"),
        patch("dataframeit.codex.CodexBackend", DummyBackend),
    ):
        result = dataframeit(
            data,
            questions=SampleModel,
            prompt="{texto}",
            provider="codex",
            model="gpt-5.6-luna",
            text_column="texto",
            resume=True,
        )

    assert DummyBackend.instances[0].calls == ["pendente"]
    assert result.loc[0, "sentimento"] == "anterior"
    assert result.loc[1, "sentimento"] == "pendente"


def test_dataframeit_resume_without_null_status_does_not_open_backend():
    from dataframeit import dataframeit

    data = pd.DataFrame(
        {
            "texto": ["pronta", "erro preservado"],
            "sentimento": ["anterior", None],
            "confianca": [0.5, None],
            "_dataframeit_status": ["processed", "error"],
        }
    )
    DummyBackend.instances.clear()
    validate_dependencies = MagicMock()
    with (
        patch(
            "dataframeit.core.validate_provider_dependencies",
            validate_dependencies,
        ),
        patch("dataframeit.codex.CodexBackend", DummyBackend),
    ):
        result = dataframeit(
            data,
            questions=SampleModel,
            prompt="{texto}",
            provider="codex",
            model="gpt-5.6-luna",
            text_column="texto",
            resume=True,
        )

    assert DummyBackend.instances == []
    validate_dependencies.assert_not_called()
    assert result.loc[0, "sentimento"] == "anterior"
    assert result.loc[1, "_dataframeit_status"] == "error"


def test_dataframeit_rejects_invalid_schema_before_opening_client(fake_sdk):
    from dataframeit import dataframeit

    with patch("dataframeit.core.validate_provider_dependencies"):
        with pytest.raises(CodexConfigurationError, match="chaves dinâmicas"):
            dataframeit(
                ["texto"],
                questions=ModelWithDynamicKeys,
                prompt="{texto}",
                provider="codex",
                model="gpt-5.6-luna",
            )

    assert FakeCodex.instances == []


def test_codex_rejects_search_before_opening_backend():
    from dataframeit import dataframeit

    DummyBackend.instances.clear()
    with patch("dataframeit.core.validate_provider_dependencies"):
        with pytest.raises(ValueError, match=r"use_search.*provider='codex'"):
            dataframeit(
                ["texto"],
                questions=SampleModel,
                prompt="{texto}",
                provider="codex",
                model="gpt-5.6-luna",
                use_search=True,
            )

    assert DummyBackend.instances == []


def test_retry_with_backoff_honors_provider_predicate():
    from dataframeit.errors import retry_with_backoff

    attempts = 0

    def fail():
        nonlocal attempts
        attempts += 1
        raise RuntimeError("definitive")

    with pytest.raises(RuntimeError, match="definitive"):
        retry_with_backoff(fail, max_retries=3, should_retry=lambda error: False)

    assert attempts == 1
