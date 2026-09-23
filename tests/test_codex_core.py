"""Contratos do core compartilhados pelo provider Codex."""

from __future__ import annotations

import importlib
import threading
from contextlib import contextmanager
from unittest.mock import Mock

import pandas as pd
import pytest
from pydantic import BaseModel, ConfigDict, Field

import dataframeit.core as core
from dataframeit.llm import LLMConfig, SearchConfig, SearchGroupConfig


class ResultModel(BaseModel):
    value: str


class ExpandedResultModel(BaseModel):
    value: list[str]
    new_value: str


class OptionalExpandedResultModel(BaseModel):
    value: list[str]
    new_value: str | None = None


class DefaultExpandedResultModel(BaseModel):
    value: list[str]
    new_value: str = "default"


class DerivedDefaultExpandedResultModel(BaseModel):
    value: list[str]
    new_value: str = Field(default_factory=lambda data: data["value"][0])


class AliasedExpandedResultModel(BaseModel):
    model_config = ConfigDict(validate_by_name=False, validate_by_alias=True)

    value: list[str] = Field(validation_alias="VALUE")
    new_value: str


class RequiredAndDefaultExpandedResultModel(BaseModel):
    value: list[str]
    new_value: str
    default_value: str = "default"


class TwiceExpandedResultModel(BaseModel):
    value: list[str]
    first_new_value: str
    second_new_value: str


def make_config(
    provider: str = "codex",
    search_config: SearchConfig | None = None,
) -> LLMConfig:
    return LLMConfig(
        model="gpt-5.4",
        provider=provider,
        api_key=None,
        max_retries=1,
        base_delay=0,
        max_delay=0,
        rate_limit_delay=0,
        search_config=search_config,
    )


class RecordingCodexBackend:
    instances: list[RecordingCodexBackend] = []

    def __init__(self, config, pydantic_model, user_prompt):
        self.config = config
        self.pydantic_model = pydantic_model
        self.user_prompt = user_prompt
        self.calls: list[str] = []
        self._lock = threading.Lock()
        self.instances.append(self)

    def invoke(self, text: str) -> dict:
        with self._lock:
            self.calls.append(text)
        return {"data": {"value": text}, "usage": None}


def install_recording_codex(monkeypatch) -> Mock:
    dependencies = Mock()
    codex_module = importlib.import_module("dataframeit.codex")
    monkeypatch.setattr(core, "validate_provider_dependencies", dependencies)

    @contextmanager
    def recording_backend(config, pydantic_model, user_prompt):
        yield RecordingCodexBackend(config, pydantic_model, user_prompt)

    monkeypatch.setattr(codex_module, "open_codex_backend", recording_backend)
    RecordingCodexBackend.instances.clear()
    return dependencies


@pytest.mark.parametrize("parallel_requests", [1, 3])
def test_codex_backend_is_created_once_for_all_rows(monkeypatch, parallel_requests):
    dependencies = install_recording_codex(monkeypatch)

    result = core.dataframeit(
        ["a", "b", "c"],
        questions=ResultModel,
        prompt="Extract: {texto}",
        provider="codex",
        model="gpt-5.4",
        parallel_requests=parallel_requests,
        track_tokens=False,
    )

    dependencies.assert_called_once_with("codex")
    assert len(RecordingCodexBackend.instances) == 1
    backend = RecordingCodexBackend.instances[0]
    assert backend.config.provider == "codex"
    assert backend.pydantic_model is ResultModel
    assert backend.user_prompt == "Extract: {texto}"
    assert sorted(backend.calls) == ["a", "b", "c"]
    assert sorted(result["value"].tolist()) == ["a", "b", "c"]


def test_resume_only_invokes_backend_for_pending_rows(monkeypatch):
    install_recording_codex(monkeypatch)
    data = pd.DataFrame(
        {
            "text": ["ready", "pending"],
            "value": ["previous", None],
            "_dataframeit_status": ["processed", None],
        }
    )

    result = core.dataframeit(
        data,
        questions=ResultModel,
        prompt="{texto}",
        provider="codex",
        model="gpt-5.4",
        resume=True,
        track_tokens=False,
    )

    assert RecordingCodexBackend.instances[0].calls == ["pending"]
    assert result["value"].tolist() == ["previous", "pending"]


def test_empty_dataframe_adds_result_columns_without_provider(monkeypatch):
    dependencies = Mock(side_effect=AssertionError("dependency preflight must not run"))
    backend_factory = Mock(side_effect=AssertionError("backend must not open"))
    monkeypatch.setattr(core, "validate_provider_dependencies", dependencies)
    monkeypatch.setattr(core, "_provider_backend", backend_factory)
    data = pd.DataFrame({"text": pd.Series(dtype=str)})

    result = core.dataframeit(
        data,
        questions=ResultModel,
        prompt="{texto}",
        provider="codex",
        model="gpt-5.4",
    )

    dependencies.assert_not_called()
    backend_factory.assert_not_called()
    assert result.empty
    assert result.columns.tolist() == [
        "text",
        "value",
        "_input_tokens",
        "_cached_input_tokens",
        "_output_tokens",
        "_reasoning_tokens",
    ]


def test_completed_checkpoint_rejects_new_model_field_without_reprocessing(monkeypatch):
    dependencies = Mock(side_effect=AssertionError("dependency preflight must not run"))
    backend_factory = Mock(side_effect=AssertionError("backend must not open"))
    monkeypatch.setattr(core, "validate_provider_dependencies", dependencies)
    monkeypatch.setattr(core, "_provider_backend", backend_factory)
    data = pd.DataFrame(
        {
            "text": ["ready"],
            "value": ['["previous"]'],
            "_dataframeit_status": ["processed"],
        }
    )

    original = data.copy(deep=True)

    with pytest.raises(ValueError, match=r"reprocess_columns=\['new_value'\]"):
        core.dataframeit(
            data,
            questions=ExpandedResultModel,
            prompt="{texto}",
            provider="codex",
            model="gpt-5.4",
            resume=True,
            track_tokens=False,
        )

    dependencies.assert_not_called()
    backend_factory.assert_not_called()
    pd.testing.assert_frame_equal(data, original)


def test_completed_checkpoint_rejects_required_null_field_without_mutation(monkeypatch):
    dependencies = Mock(side_effect=AssertionError("dependency preflight must not run"))
    backend_factory = Mock(side_effect=AssertionError("backend must not open"))
    monkeypatch.setattr(core, "validate_provider_dependencies", dependencies)
    monkeypatch.setattr(core, "_provider_backend", backend_factory)
    data = pd.DataFrame(
        {
            "text": ["ready"],
            "value": ['["previous"]'],
            "new_value": [None],
            "_dataframeit_status": ["processed"],
        }
    )
    original = data.copy(deep=True)

    with pytest.raises(ValueError, match=r"reprocess_columns=\['new_value'\]"):
        core.dataframeit(
            data,
            questions=ExpandedResultModel,
            prompt="{texto}",
            provider="codex",
            model="gpt-5.4",
            resume=True,
            track_tokens=False,
        )

    dependencies.assert_not_called()
    backend_factory.assert_not_called()
    pd.testing.assert_frame_equal(data, original)


def test_completed_checkpoint_accepts_optional_null_field_without_provider(monkeypatch):
    dependencies = Mock(side_effect=AssertionError("dependency preflight must not run"))
    backend_factory = Mock(side_effect=AssertionError("backend must not open"))
    monkeypatch.setattr(core, "validate_provider_dependencies", dependencies)
    monkeypatch.setattr(core, "_provider_backend", backend_factory)
    data = pd.DataFrame(
        {
            "text": ["ready"],
            "value": ['["previous"]'],
            "new_value": [None],
            "_dataframeit_status": ["processed"],
        }
    )

    result = core.dataframeit(
        data,
        questions=OptionalExpandedResultModel,
        prompt="{texto}",
        provider="codex",
        model="gpt-5.4",
        resume=True,
        track_tokens=False,
    )

    dependencies.assert_not_called()
    backend_factory.assert_not_called()
    assert result["value"].tolist() == [["previous"]]
    assert result["new_value"].isna().all()


def test_completed_checkpoint_fills_absent_model_default_without_provider(monkeypatch):
    dependencies = Mock(side_effect=AssertionError("dependency preflight must not run"))
    backend_factory = Mock(side_effect=AssertionError("backend must not open"))
    monkeypatch.setattr(core, "validate_provider_dependencies", dependencies)
    monkeypatch.setattr(core, "_provider_backend", backend_factory)
    data = pd.DataFrame(
        {
            "text": ["ready"],
            "value": ['["previous"]'],
            "_dataframeit_status": ["processed"],
        }
    )

    result = core.dataframeit(
        data,
        questions=DefaultExpandedResultModel,
        prompt="{texto}",
        provider="codex",
        model="gpt-5.4",
        resume=True,
        track_tokens=False,
    )

    dependencies.assert_not_called()
    backend_factory.assert_not_called()
    assert result["value"].tolist() == [["previous"]]
    assert result["new_value"].tolist() == ["default"]


def test_completed_checkpoint_fills_default_factory_using_validated_data(monkeypatch):
    dependencies = Mock(side_effect=AssertionError("dependency preflight must not run"))
    backend_factory = Mock(side_effect=AssertionError("backend must not open"))
    monkeypatch.setattr(core, "validate_provider_dependencies", dependencies)
    monkeypatch.setattr(core, "_provider_backend", backend_factory)
    data = pd.DataFrame(
        {
            "text": ["ready"],
            "value": ['["previous"]'],
            "_dataframeit_status": ["processed"],
        }
    )

    result = core.dataframeit(
        data,
        questions=DerivedDefaultExpandedResultModel,
        prompt="{texto}",
        provider="codex",
        model="gpt-5.4",
        resume=True,
        track_tokens=False,
    )

    dependencies.assert_not_called()
    backend_factory.assert_not_called()
    assert result["value"].tolist() == [["previous"]]
    assert result["new_value"].tolist() == ["previous"]


def test_completed_checkpoint_accepts_canonical_name_with_validation_alias(monkeypatch):
    dependencies = Mock(side_effect=AssertionError("dependency preflight must not run"))
    backend_factory = Mock(side_effect=AssertionError("backend must not open"))
    monkeypatch.setattr(core, "validate_provider_dependencies", dependencies)
    monkeypatch.setattr(core, "_provider_backend", backend_factory)
    data = pd.DataFrame(
        {
            "text": ["ready"],
            "value": ['["previous"]'],
            "new_value": ["kept"],
            "_dataframeit_status": ["processed"],
        }
    )

    result = core.dataframeit(
        data,
        questions=AliasedExpandedResultModel,
        prompt="{texto}",
        provider="codex",
        model="gpt-5.4",
        resume=True,
        track_tokens=False,
    )

    dependencies.assert_not_called()
    backend_factory.assert_not_called()
    assert result["value"].tolist() == [["previous"]]
    assert result["new_value"].tolist() == ["kept"]


def test_completed_checkpoint_fills_defaults_by_position_with_duplicate_index(monkeypatch):
    dependencies = Mock(side_effect=AssertionError("dependency preflight must not run"))
    backend_factory = Mock(side_effect=AssertionError("backend must not open"))
    monkeypatch.setattr(core, "validate_provider_dependencies", dependencies)
    monkeypatch.setattr(core, "_provider_backend", backend_factory)
    data = pd.DataFrame(
        {
            "text": ["first", "second"],
            "value": ['["a"]', '["b"]'],
            "_dataframeit_status": ["processed", "processed"],
        },
        index=[0, 0],
    )

    result = core.dataframeit(
        data,
        questions=DerivedDefaultExpandedResultModel,
        prompt="{texto}",
        provider="codex",
        model="gpt-5.4",
        resume=True,
        track_tokens=False,
    )

    dependencies.assert_not_called()
    backend_factory.assert_not_called()
    assert result["value"].tolist() == [["a"], ["b"]]
    assert result["new_value"].tolist() == ["a", "b"]


def test_completed_compatible_checkpoint_normalizes_without_provider(monkeypatch):
    dependencies = Mock(side_effect=AssertionError("dependency preflight must not run"))
    backend_factory = Mock(side_effect=AssertionError("backend must not open"))
    monkeypatch.setattr(core, "validate_provider_dependencies", dependencies)
    monkeypatch.setattr(core, "_provider_backend", backend_factory)
    data = pd.DataFrame(
        {
            "text": ["ready"],
            "value": ['["previous"]'],
            "new_value": ["kept"],
            "_dataframeit_status": ["processed"],
        }
    )

    result = core.dataframeit(
        data,
        questions=ExpandedResultModel,
        prompt="{texto}",
        provider="codex",
        model="gpt-5.4",
        resume=True,
        track_tokens=False,
    )

    dependencies.assert_not_called()
    backend_factory.assert_not_called()
    assert result["value"].tolist() == [["previous"]]
    assert result["new_value"].tolist() == ["kept"]


def test_partial_checkpoint_rejects_new_model_field_without_reprocessing(monkeypatch):
    dependencies = Mock(side_effect=AssertionError("dependency preflight must not run"))
    backend_factory = Mock(side_effect=AssertionError("backend must not open"))
    monkeypatch.setattr(core, "validate_provider_dependencies", dependencies)
    monkeypatch.setattr(core, "_provider_backend", backend_factory)
    data = pd.DataFrame(
        {
            "text": ["ready", "pending"],
            "value": ['["previous"]', None],
            "_dataframeit_status": ["processed", None],
        }
    )
    original = data.copy(deep=True)

    with pytest.raises(ValueError, match=r"reprocess_columns=\['new_value'\]"):
        core.dataframeit(
            data,
            questions=ExpandedResultModel,
            prompt="{texto}",
            provider="codex",
            model="gpt-5.4",
            resume=True,
            track_tokens=False,
        )

    dependencies.assert_not_called()
    backend_factory.assert_not_called()
    pd.testing.assert_frame_equal(data, original)


def test_reprocessing_new_field_updates_processed_and_pending_rows(monkeypatch):
    @contextmanager
    def expanded_backend(*args):
        yield core.ProviderBackend(
            label="codex",
            invoke=lambda text: {
                "data": {"value": [text], "new_value": f"new:{text}"},
                "usage": None,
            },
        )

    monkeypatch.setattr(core, "validate_provider_dependencies", Mock())
    monkeypatch.setattr(core, "_provider_backend", expanded_backend)
    data = pd.DataFrame(
        {
            "text": ["ready", "pending"],
            "value": [["previous"], None],
            "_dataframeit_status": ["processed", None],
        }
    )

    result = core.dataframeit(
        data,
        questions=ExpandedResultModel,
        prompt="{texto}",
        provider="codex",
        model="gpt-5.4",
        resume=True,
        reprocess_columns=["new_value"],
        track_tokens=False,
    )

    assert result["value"].tolist() == [["previous"], ["pending"]]
    assert result["new_value"].tolist() == ["new:ready", "new:pending"]


def test_reprocessing_covers_required_null_field(monkeypatch):
    @contextmanager
    def expanded_backend(*args):
        yield core.ProviderBackend(
            label="codex",
            invoke=lambda text: {
                "data": {"value": [text], "new_value": f"new:{text}"},
                "usage": None,
            },
        )

    monkeypatch.setattr(core, "validate_provider_dependencies", Mock())
    monkeypatch.setattr(core, "_provider_backend", expanded_backend)
    data = pd.DataFrame(
        {
            "text": ["ready"],
            "value": [["previous"]],
            "new_value": [None],
            "_dataframeit_status": ["processed"],
        }
    )

    result = core.dataframeit(
        data,
        questions=ExpandedResultModel,
        prompt="{texto}",
        provider="codex",
        model="gpt-5.4",
        resume=True,
        reprocess_columns=["new_value"],
        track_tokens=False,
    )

    assert result["value"].tolist() == [["previous"]]
    assert result["new_value"].tolist() == ["new:ready"]


def test_reprocessing_null_field_also_fills_absent_default(monkeypatch):
    @contextmanager
    def expanded_backend(*args):
        yield core.ProviderBackend(
            label="codex",
            invoke=lambda text: {
                "data": {
                    "value": [text],
                    "new_value": f"new:{text}",
                    "default_value": "default",
                },
                "usage": None,
            },
        )

    monkeypatch.setattr(core, "validate_provider_dependencies", Mock())
    monkeypatch.setattr(core, "_provider_backend", expanded_backend)
    data = pd.DataFrame(
        {
            "text": ["ready"],
            "value": [["previous"]],
            "new_value": [None],
            "_dataframeit_status": ["processed"],
        }
    )

    result = core.dataframeit(
        data,
        questions=RequiredAndDefaultExpandedResultModel,
        prompt="{texto}",
        provider="codex",
        model="gpt-5.4",
        resume=True,
        reprocess_columns=["new_value"],
        track_tokens=False,
    )

    assert result["value"].tolist() == [["previous"]]
    assert result["new_value"].tolist() == ["new:ready"]
    assert result["default_value"].tolist() == ["default"]


def test_reprocessing_must_cover_every_new_model_field(monkeypatch):
    dependencies = Mock(side_effect=AssertionError("dependency preflight must not run"))
    backend_factory = Mock(side_effect=AssertionError("backend must not open"))
    monkeypatch.setattr(core, "validate_provider_dependencies", dependencies)
    monkeypatch.setattr(core, "_provider_backend", backend_factory)
    data = pd.DataFrame(
        {
            "text": ["ready"],
            "value": [["previous"]],
            "_dataframeit_status": ["processed"],
        }
    )
    original = data.copy(deep=True)

    with pytest.raises(ValueError, match="second_new_value"):
        core.dataframeit(
            data,
            questions=TwiceExpandedResultModel,
            prompt="{texto}",
            provider="codex",
            model="gpt-5.4",
            reprocess_columns=["first_new_value"],
            track_tokens=False,
        )

    dependencies.assert_not_called()
    backend_factory.assert_not_called()
    pd.testing.assert_frame_equal(data, original)


def test_completed_checkpoint_adds_missing_cached_token_column_without_provider(
    monkeypatch,
):
    dependencies = Mock(side_effect=AssertionError("dependency preflight must not run"))
    backend_factory = Mock(side_effect=AssertionError("backend must not open"))
    monkeypatch.setattr(core, "validate_provider_dependencies", dependencies)
    monkeypatch.setattr(core, "_provider_backend", backend_factory)
    data = pd.DataFrame(
        {
            "text": ["ready"],
            "value": ["previous"],
            "_input_tokens": [10],
            "_output_tokens": [5],
            "_reasoning_tokens": [2],
            "_dataframeit_status": ["processed"],
        }
    )

    result = core.dataframeit(
        data,
        questions=ResultModel,
        prompt="{texto}",
        provider="google_genai",
        resume=True,
    )

    dependencies.assert_not_called()
    backend_factory.assert_not_called()
    assert result["value"].tolist() == ["previous"]
    assert result["_cached_input_tokens"].isna().all()


def test_codex_preflight_failure_does_not_mutate_dataframe(monkeypatch):
    @contextmanager
    def failing_backend(*args):
        raise ValueError("invalid schema, configuration or authentication")
        yield

    codex_module = importlib.import_module("dataframeit.codex")
    monkeypatch.setattr(core, "validate_provider_dependencies", Mock())
    monkeypatch.setattr(codex_module, "open_codex_backend", failing_backend)
    data = pd.DataFrame({"text": ["pending"]})
    original = data.copy(deep=True)

    with pytest.raises(ValueError):
        core.dataframeit(
            data,
            questions=ResultModel,
            prompt="{texto}",
            provider="codex",
            model="gpt-5.4",
        )

    pd.testing.assert_frame_equal(data, original)


def test_langchain_backend_invokes_selected_provider(monkeypatch):
    selected_call = Mock(return_value={"data": {"value": "first"}})
    monkeypatch.setattr(core, "call_langchain", selected_call)
    config = make_config(provider="google_genai")

    with core._provider_backend(config, ResultModel, "{texto}", None) as backend:
        result = backend.invoke("row")

    assert backend.label == "langchain"
    assert result["data"]["value"] == "first"
    selected_call.assert_called_once_with("row", ResultModel, "{texto}", config)


def test_claude_backend_invokes_selected_provider(monkeypatch):
    claude_module = importlib.import_module("dataframeit.claude_code")
    selected_call = Mock(return_value={"data": {"value": "first"}})
    monkeypatch.setattr(claude_module, "call_claude_code", selected_call)
    config = make_config(provider="claude_code")

    with core._provider_backend(config, ResultModel, "{texto}", None) as backend:
        result = backend.invoke("row")

    assert backend.label == "claude_code"
    assert result["data"]["value"] == "first"
    selected_call.assert_called_once_with("row", ResultModel, "{texto}", config)


@pytest.mark.parametrize(
    ("per_field", "groups", "selected_name"),
    [
        (False, None, "call_agent"),
        (True, None, "call_agent_per_field"),
        (
            True,
            {"main": SearchGroupConfig(fields=["value"])},
            "call_agent_per_group",
        ),
    ],
)
def test_search_backend_invokes_selected_mode(monkeypatch, per_field, groups, selected_name):
    agent_module = importlib.import_module("dataframeit.agent")
    calls = {
        name: Mock(return_value={"data": {"value": name}})
        for name in ("call_agent", "call_agent_per_field", "call_agent_per_group")
    }
    for name, call in calls.items():
        monkeypatch.setattr(agent_module, name, call)

    search_config = SearchConfig(enabled=True, per_field=per_field, groups=groups)
    config = make_config(provider="google_genai", search_config=search_config)
    with core._provider_backend(config, ResultModel, "{texto}", "minimal") as backend:
        first = backend.invoke("one")
        second = backend.invoke("two")

    assert backend.label == "langchain"
    assert first["data"]["value"] == selected_name
    assert second["data"]["value"] == selected_name
    assert calls[selected_name].call_count == 2
    for name, call in calls.items():
        if name != selected_name:
            call.assert_not_called()


@pytest.mark.parametrize("parallel_requests", [1, 2])
def test_malformed_backend_result_is_recorded_as_row_error(monkeypatch, parallel_requests):
    @contextmanager
    def malformed_backend(*args):
        yield core.ProviderBackend(
            label="codex",
            invoke=lambda text: {"usage": None},
        )

    monkeypatch.setattr(core, "validate_provider_dependencies", Mock())
    monkeypatch.setattr(core, "_provider_backend", malformed_backend)
    data = pd.DataFrame({"text": ["row"]})

    with pytest.warns(UserWarning, match="Falha ao processar linha"):
        result = core.dataframeit(
            data,
            questions=ResultModel,
            prompt="{texto}",
            provider="codex",
            model="gpt-5.4",
            parallel_requests=parallel_requests,
            track_tokens=False,
        )

    assert result["_dataframeit_status"].tolist() == ["error"]
    assert "KeyError: 'data'" in result["_error_details"].iloc[0]
    assert result["value"].isna().all()
