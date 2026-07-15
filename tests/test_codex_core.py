"""Contratos do core compartilhados pelo provider Codex."""

from __future__ import annotations

import importlib
import threading
from unittest.mock import Mock

import pandas as pd
import pytest
from pydantic import BaseModel

import dataframeit.core as core
from dataframeit.llm import LLMConfig, SearchConfig, SearchGroupConfig


class ResultModel(BaseModel):
    value: str


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
        self.entered = False
        self.closed = False
        self._lock = threading.Lock()
        self.instances.append(self)

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.closed = True

    def invoke(self, text: str) -> dict:
        with self._lock:
            self.calls.append(text)
        return {"data": {"value": text}, "usage": None}


def install_recording_codex(monkeypatch) -> Mock:
    dependencies = Mock()
    codex_module = importlib.import_module("dataframeit.codex")
    monkeypatch.setattr(core, "validate_provider_dependencies", dependencies)
    monkeypatch.setattr(codex_module, "CodexBackend", RecordingCodexBackend)
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
    assert backend.entered and backend.closed
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


def test_completed_checkpoint_does_not_open_provider(monkeypatch):
    dependencies = Mock(side_effect=AssertionError("dependency preflight must not run"))
    backend_factory = Mock(side_effect=AssertionError("backend must not open"))
    monkeypatch.setattr(core, "validate_provider_dependencies", dependencies)
    monkeypatch.setattr(core, "_provider_backend", backend_factory)
    data = pd.DataFrame(
        {
            "text": ["ready"],
            "value": ["previous"],
            "_dataframeit_status": ["processed"],
        }
    )

    result = core.dataframeit(
        data,
        questions=ResultModel,
        prompt="{texto}",
        provider="codex",
        model="gpt-5.4",
        resume=True,
    )

    dependencies.assert_not_called()
    backend_factory.assert_not_called()
    assert result["value"].tolist() == ["previous"]


@pytest.mark.parametrize("failure_stage", ["constructor", "enter"])
def test_codex_preflight_failure_does_not_mutate_dataframe(monkeypatch, failure_stage):
    class FailingCodexBackend:
        def __init__(self, config, pydantic_model, user_prompt):
            if failure_stage == "constructor":
                raise ValueError("invalid schema or configuration")

        def __enter__(self):
            raise ValueError("authentication failed")

        def __exit__(self, exc_type, exc, traceback):
            return None

    codex_module = importlib.import_module("dataframeit.codex")
    monkeypatch.setattr(core, "validate_provider_dependencies", Mock())
    monkeypatch.setattr(codex_module, "CodexBackend", FailingCodexBackend)
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


def test_langchain_callable_is_bound_when_backend_is_selected(monkeypatch):
    selected_call = Mock(return_value={"data": {"value": "first"}})
    late_replacement = Mock(return_value={"data": {"value": "late"}})
    monkeypatch.setattr(core, "call_langchain", selected_call)
    config = make_config(provider="google_genai")
    backend_context = core._provider_backend(config, ResultModel, "{texto}", None)
    monkeypatch.setattr(core, "call_langchain", late_replacement)

    with backend_context as backend:
        result = backend.invoke("row")

    assert result["data"]["value"] == "first"
    selected_call.assert_called_once_with("row", ResultModel, "{texto}", config)
    late_replacement.assert_not_called()


def test_claude_callable_is_bound_when_backend_is_selected(monkeypatch):
    claude_module = importlib.import_module("dataframeit.claude_code")
    selected_call = Mock(return_value={"data": {"value": "first"}})
    late_replacement = Mock(return_value={"data": {"value": "late"}})
    monkeypatch.setattr(claude_module, "call_claude_code", selected_call)
    config = make_config(provider="claude_code")
    backend_context = core._provider_backend(config, ResultModel, "{texto}", None)
    monkeypatch.setattr(claude_module, "call_claude_code", late_replacement)

    with backend_context as backend:
        result = backend.invoke("row")

    assert result["data"]["value"] == "first"
    selected_call.assert_called_once_with("row", ResultModel, "{texto}", config)
    late_replacement.assert_not_called()


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
def test_search_dispatch_is_bound_once(monkeypatch, per_field, groups, selected_name):
    agent_module = importlib.import_module("dataframeit.agent")
    calls = {
        name: Mock(return_value={"data": {"value": name}})
        for name in ("call_agent", "call_agent_per_field", "call_agent_per_group")
    }
    for name, call in calls.items():
        monkeypatch.setattr(agent_module, name, call)

    search_config = SearchConfig(enabled=True, per_field=per_field, groups=groups)
    config = make_config(provider="google_genai", search_config=search_config)
    backend_context = core._provider_backend(
        config,
        ResultModel,
        "{texto}",
        "minimal",
    )
    late_replacement = Mock(return_value={"data": {"value": "late"}})
    monkeypatch.setattr(agent_module, selected_name, late_replacement)

    with backend_context as backend:
        first = backend.invoke("one")
        second = backend.invoke("two")

    assert first["data"]["value"] == selected_name
    assert second["data"]["value"] == selected_name
    assert calls[selected_name].call_count == 2
    late_replacement.assert_not_called()
    for name, call in calls.items():
        if name != selected_name:
            call.assert_not_called()


def test_row_processing_has_no_late_dispatch_helper():
    assert not hasattr(core, "_call_row_model")
