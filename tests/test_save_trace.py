"""Testes para a funcionalidade save_trace."""

import json
import pandas as pd
import pytest
from pydantic import BaseModel
from unittest.mock import patch, MagicMock

from dataframeit.core import dataframeit
from dataframeit.agent import _extract_trace


class SimpleModel(BaseModel):
    campo1: str
    campo2: str


# ============================================================================
# Testes de validação
# ============================================================================


def test_save_trace_requires_use_search():
    """Testa que save_trace sem use_search levanta ValueError."""
    df = pd.DataFrame({"texto": ["a"]})

    with patch("dataframeit.core.validate_provider_dependencies"):
        with pytest.raises(ValueError) as exc_info:
            dataframeit(
                df,
                questions=SimpleModel,
                prompt="Teste {texto}",
                save_trace=True,
                use_search=False,  # save_trace requer use_search=True
            )

        assert "save_trace requer use_search=True" in str(exc_info.value)


def test_save_trace_invalid_value():
    """Testa que valores inválidos para save_trace levantam ValueError."""
    df = pd.DataFrame({"texto": ["a"]})

    with patch("dataframeit.core.validate_provider_dependencies"):
        with patch("dataframeit.core.validate_search_dependencies"):
            with pytest.raises(ValueError) as exc_info:
                dataframeit(
                    df,
                    questions=SimpleModel,
                    prompt="Teste {texto}",
                    save_trace="invalid",
                    use_search=True,
                )

            assert "save_trace deve ser True, 'full' ou 'minimal'" in str(exc_info.value)


def test_save_trace_normalizes_true_to_full():
    """Testa que save_trace=True é normalizado para 'full'."""
    df = pd.DataFrame({"texto": ["a"]})

    mock_result = {
        "data": {"campo1": "valor1", "campo2": "valor2"},
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        "trace": {
            "messages": [],
            "search_queries": ["query1"],
            "total_tool_calls": 1,
            "duration_seconds": 1.5,
            "model": "test-model",
        },
    }

    with patch("dataframeit.core.validate_provider_dependencies"):
        with patch("dataframeit.core.validate_search_dependencies"):
            with patch("dataframeit.agent.call_agent", return_value=mock_result):
                result = dataframeit(
                    df,
                    questions=SimpleModel,
                    prompt="Teste {texto}",
                    save_trace=True,  # Deve funcionar como "full"
                    use_search=True,
                )

                assert "_trace" in result.columns


# ============================================================================
# Testes de _extract_trace
# ============================================================================


def test_extract_trace_full_mode():
    """Testa extração de trace no modo full."""
    # Mock de mensagens LangChain
    human_msg = MagicMock()
    human_msg.type = "human"
    human_msg.content = "Analise: test input"

    ai_msg = MagicMock()
    ai_msg.type = "ai"
    ai_msg.content = ""
    ai_msg.tool_calls = [
        {"name": "tavily_search", "args": {"query": "test query"}, "id": "call_123", "type": "tool_call"}
    ]

    tool_msg = MagicMock()
    tool_msg.type = "tool"
    tool_msg.content = "Search results: lots of text here..."
    tool_msg.tool_call_id = "call_123"
    del tool_msg.tool_calls  # ToolMessage não tem tool_calls

    agent_result = {"messages": [human_msg, ai_msg, tool_msg]}

    trace = _extract_trace(agent_result, "gemini-2.0-flash", 2.5, "full")

    assert trace["model"] == "gemini-2.0-flash"
    assert trace["duration_seconds"] == 2.5
    assert trace["search_queries"] == ["test query"]
    assert trace["total_tool_calls"] == 1
    assert len(trace["messages"]) == 3
    # No modo full, o conteúdo do tool message é preservado
    assert trace["messages"][2]["content"] == "Search results: lots of text here..."


def test_extract_trace_minimal_mode():
    """Testa extração de trace no modo minimal."""
    human_msg = MagicMock()
    human_msg.type = "human"
    human_msg.content = "Analise: test input"

    ai_msg = MagicMock()
    ai_msg.type = "ai"
    ai_msg.content = "Based on results..."
    ai_msg.tool_calls = []

    tool_msg = MagicMock()
    tool_msg.type = "tool"
    tool_msg.content = "Search results: lots of text here..."
    tool_msg.tool_call_id = "call_123"
    del tool_msg.tool_calls

    agent_result = {"messages": [human_msg, ai_msg, tool_msg]}

    trace = _extract_trace(agent_result, "gemini-2.0-flash", 2.5, "minimal")

    # No modo minimal, o conteúdo do tool message é omitido
    assert trace["messages"][2]["content"] == "[omitted]"
    # Mas o conteúdo das mensagens AI é preservado
    assert trace["messages"][1]["content"] == "Based on results..."


def test_extract_trace_extracts_search_queries():
    """Testa que search queries são extraídas dos tool calls."""
    ai_msg = MagicMock()
    ai_msg.type = "ai"
    ai_msg.content = ""
    ai_msg.tool_calls = [
        {"name": "tavily_search", "args": {"query": "query 1"}, "id": "call_1", "type": "tool_call"},
        {"name": "tavily_search", "args": {"query": "query 2"}, "id": "call_2", "type": "tool_call"},
    ]

    agent_result = {"messages": [ai_msg]}

    trace = _extract_trace(agent_result, "model", 1.0, "full")

    assert trace["search_queries"] == ["query 1", "query 2"]
    assert trace["total_tool_calls"] == 2


# ============================================================================
# Testes de integração com dataframeit
# ============================================================================


def test_save_trace_creates_trace_column():
    """Testa que save_trace cria coluna _trace no resultado."""
    df = pd.DataFrame({"texto": ["a"]})

    mock_result = {
        "data": {"campo1": "valor1", "campo2": "valor2"},
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        "trace": {
            "messages": [{"type": "human", "content": "test"}],
            "search_queries": ["query1"],
            "total_tool_calls": 1,
            "duration_seconds": 1.5,
            "model": "test-model",
        },
    }

    with patch("dataframeit.core.validate_provider_dependencies"):
        with patch("dataframeit.core.validate_search_dependencies"):
            with patch("dataframeit.agent.call_agent", return_value=mock_result):
                result = dataframeit(
                    df,
                    questions=SimpleModel,
                    prompt="Teste {texto}",
                    save_trace="full",
                    use_search=True,
                )

                assert "_trace" in result.columns
                trace_json = result["_trace"].iloc[0]
                assert trace_json is not None
                trace = json.loads(trace_json)
                assert trace["model"] == "test-model"
                assert trace["search_queries"] == ["query1"]


def test_save_trace_per_field_creates_multiple_columns():
    """Testa que save_trace com per_field cria uma coluna por campo."""
    df = pd.DataFrame({"texto": ["a"]})

    mock_result = {
        "data": {"campo1": "valor1", "campo2": "valor2"},
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        "traces": {
            "campo1": {
                "messages": [],
                "search_queries": ["query campo1"],
                "total_tool_calls": 1,
                "duration_seconds": 1.0,
                "model": "test-model",
            },
            "campo2": {
                "messages": [],
                "search_queries": ["query campo2"],
                "total_tool_calls": 1,
                "duration_seconds": 1.2,
                "model": "test-model",
            },
        },
    }

    with patch("dataframeit.core.validate_provider_dependencies"):
        with patch("dataframeit.core.validate_search_dependencies"):
            with patch("dataframeit.agent.call_agent_per_field", return_value=mock_result):
                result = dataframeit(
                    df,
                    questions=SimpleModel,
                    prompt="Teste {texto}",
                    save_trace="full",
                    use_search=True,
                    search_per_field=True,
                )

                assert "_trace_campo1" in result.columns
                assert "_trace_campo2" in result.columns

                trace1 = json.loads(result["_trace_campo1"].iloc[0])
                trace2 = json.loads(result["_trace_campo2"].iloc[0])

                assert trace1["search_queries"] == ["query campo1"]
                assert trace2["search_queries"] == ["query campo2"]


def test_save_trace_disabled_by_default():
    """Testa que save_trace está desabilitado por padrão."""
    df = pd.DataFrame({"texto": ["a"]})

    mock_result = {
        "data": {"campo1": "valor1", "campo2": "valor2"},
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }

    with patch("dataframeit.core.validate_provider_dependencies"):
        with patch("dataframeit.core.validate_search_dependencies"):
            with patch("dataframeit.agent.call_agent", return_value=mock_result):
                result = dataframeit(
                    df,
                    questions=SimpleModel,
                    prompt="Teste {texto}",
                    use_search=True,
                    # save_trace não especificado (default None)
                )

                assert "_trace" not in result.columns


def test_save_trace_json_is_valid():
    """Testa que o trace salvo é JSON válido."""
    df = pd.DataFrame({"texto": ["a", "b"]})

    call_count = 0

    def mock_call(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return {
            "data": {"campo1": f"valor{call_count}", "campo2": f"outro{call_count}"},
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            "trace": {
                "messages": [{"type": "human", "content": f"test {call_count}"}],
                "search_queries": [f"query {call_count}"],
                "total_tool_calls": 1,
                "duration_seconds": 1.5,
                "model": "test-model",
            },
        }

    with patch("dataframeit.core.validate_provider_dependencies"):
        with patch("dataframeit.core.validate_search_dependencies"):
            with patch("dataframeit.agent.call_agent", side_effect=mock_call):
                result = dataframeit(
                    df,
                    questions=SimpleModel,
                    prompt="Teste {texto}",
                    save_trace="full",
                    use_search=True,
                )

                # Verifica que todos os traces são JSON válidos
                for trace_json in result["_trace"]:
                    trace = json.loads(trace_json)
                    assert "messages" in trace
                    assert "search_queries" in trace
                    assert "model" in trace
