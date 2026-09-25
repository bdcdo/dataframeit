"""Testes para a funcionalidade save_trace."""

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel, Field

from dataframeit.agent import _extract_trace
from dataframeit.core import dataframeit


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
        with pytest.raises(ValueError, match="save_trace requer use_search=True") as exc_info:
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

    with (
        patch("dataframeit.core.validate_provider_dependencies"),
        patch("dataframeit.core.validate_search_dependencies"),
    ):
        with pytest.raises(
            ValueError, match="save_trace deve ser True, 'full' ou 'minimal'"
        ) as exc_info:
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

    with (
        patch("dataframeit.core.validate_provider_dependencies"),
        patch("dataframeit.core.validate_search_dependencies"),
        patch("dataframeit.agent.call_agent", return_value=mock_result),
    ):
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
        {
            "name": "tavily_search",
            "args": {"query": "test query"},
            "id": "call_123",
            "type": "tool_call",
        }
    ]

    tool_msg = MagicMock()
    tool_msg.type = "tool"
    tool_msg.content = "Search results: lots of text here..."
    tool_msg.tool_call_id = "call_123"
    del tool_msg.tool_calls  # ToolMessage não tem tool_calls

    agent_result = {"messages": [human_msg, ai_msg, tool_msg]}

    trace = _extract_trace(
        agent_result, "gemini-2.0-flash", 2.5, "full", search_tool_name="tavily_search"
    )

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
        {
            "name": "tavily_search",
            "args": {"query": "query 1"},
            "id": "call_1",
            "type": "tool_call",
        },
        {
            "name": "tavily_search",
            "args": {"query": "query 2"},
            "id": "call_2",
            "type": "tool_call",
        },
    ]

    agent_result = {"messages": [ai_msg]}

    trace = _extract_trace(agent_result, "model", 1.0, "full", search_tool_name="tavily_search")

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

    with (
        patch("dataframeit.core.validate_provider_dependencies"),
        patch("dataframeit.core.validate_search_dependencies"),
        patch("dataframeit.agent.call_agent", return_value=mock_result),
    ):
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

    with (
        patch("dataframeit.core.validate_provider_dependencies"),
        patch("dataframeit.core.validate_search_dependencies"),
        patch("dataframeit.agent.call_agent_per_field", return_value=mock_result),
    ):
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

    with (
        patch("dataframeit.core.validate_provider_dependencies"),
        patch("dataframeit.core.validate_search_dependencies"),
        patch("dataframeit.agent.call_agent", return_value=mock_result),
    ):
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

    with (
        patch("dataframeit.core.validate_provider_dependencies"),
        patch("dataframeit.core.validate_search_dependencies"),
        patch("dataframeit.agent.call_agent", side_effect=mock_call),
    ):
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


# ============================================================================
# Traces das buscas aninhadas e por item, com o agente de verdade
# ============================================================================


class Endereco(BaseModel):
    cidade: str
    cep: str | None = Field(default=None, json_schema_extra={"prompt": "Busque o CEP"})


class Pedido(BaseModel):
    item: str
    registro: str | None = Field(default=None, json_schema_extra={"prompt": "Busque o registro"})


class Etiqueta(BaseModel):
    rotulo: str


class Ficha(BaseModel):
    nome: str
    endereco: Endereco
    pedidos: list[Pedido]
    etiquetas: list[Etiqueta]


# Valor que o agente falso devolve para cada campo, pelo nome.
_RESPOSTAS = {
    "nome": "Ana",
    "endereco": {"cidade": "Recife"},
    "cep": "50000-000",
    "pedidos": [{"item": "dipirona"}, {"item": "insulina"}],
    "registro": "reg-1",
    "etiquetas": [{"rotulo": "urgente"}],
}


def _executar_ficha(monkeypatch):
    """Roda call_agent de verdade; só o agente LangChain e a busca são falsos.

    Cada invocação responde o modelo montado para a chamada e registra uma
    busca cuja consulta é o nome desse modelo, para que o trace de cada
    chamada seja reconhecível no resultado.
    """
    esquemas = []

    def create_agent(*, model, tools, response_format, middleware):
        esquema = response_format.schema
        esquemas.append(esquema.__name__)

        class AgenteFalso:
            def invoke(self, payload, config=None):
                resposta = esquema(**{campo: _RESPOSTAS[campo] for campo in esquema.model_fields})
                chamada = {"name": "busca_web", "args": {"query": esquema.__name__}, "id": "1"}
                return {
                    "structured_response": resposta,
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[chamada],
                            usage_metadata={
                                "input_tokens": 10,
                                "output_tokens": 2,
                                "total_tokens": 12,
                            },
                        ),
                        ToolMessage(content="resultado", tool_call_id="1", name="busca_web"),
                    ],
                }

        return AgenteFalso()

    class FerramentaFalsa:
        name = "busca_web"

    provider = MagicMock()
    provider.name = "tavily"
    provider.create_tool = lambda **kwargs: FerramentaFalsa()
    provider.calculate_credits.side_effect = lambda search_count, **kw: search_count

    monkeypatch.setattr("dataframeit.llm._create_langchain_llm", lambda *a, **k: object())
    monkeypatch.setattr("langchain.agents.create_agent", create_agent)
    with (
        patch("dataframeit.core.validate_provider_dependencies"),
        patch("dataframeit.core.validate_search_dependencies"),
        patch("dataframeit.agent.get_provider", return_value=provider),
    ):
        resultado = dataframeit(
            pd.DataFrame({"texto": ["ficha da Ana"]}),
            questions=Ficha,
            prompt="Extraia de {texto}",
            use_search=True,
            search_per_field=True,
            save_trace="full",
        )
    return resultado, esquemas


def test_trace_por_campo_inclui_buscas_aninhadas_e_por_item(monkeypatch):
    """Campo aninhado, item de lista e campo de primeiro nível têm cada um o seu trace."""
    resultado, _ = _executar_ficha(monkeypatch)
    linha = resultado.iloc[0]

    assert json.loads(linha["_trace_nome"])["search_queries"] == ["Ficha_nome"]
    assert json.loads(linha["_trace_endereco.cep"])["search_queries"] == [
        "NestedSearch_endereco_cep"
    ]
    traces_dos_itens = json.loads(linha["_trace_pedidos_items"])
    assert [trace["registro"]["search_queries"] for trace in traces_dos_itens] == [
        ["ItemSearch_0_registro"],
        ["ItemSearch_1_registro"],
    ]


def test_busca_por_item_grava_o_valor_em_cada_item_e_soma_o_uso(monkeypatch):
    resultado, _ = _executar_ficha(monkeypatch)
    linha = resultado.iloc[0]

    assert linha["pedidos"] == [
        {"item": "dipirona", "registro": "reg-1"},
        {"item": "insulina", "registro": "reg-1"},
    ]
    # Quatro campos de primeiro nível, a busca aninhada do CEP e uma busca por
    # pedido: sete chamadas, cada uma com uma busca, 10 tokens de entrada e 2
    # de saída.
    assert linha["_search_credits"] == 7
    assert (linha["_input_tokens"], linha["_output_tokens"]) == (70, 14)


def test_lista_de_modelo_sem_campo_configurado_nao_ganha_busca_por_item(monkeypatch):
    resultado, esquemas = _executar_ficha(monkeypatch)

    assert resultado.iloc[0]["etiquetas"] == [{"rotulo": "urgente"}]
    assert "_trace_etiquetas_items" not in resultado.columns
    assert not [nome for nome in esquemas if nome.startswith("ItemSearch") and "rotulo" in nome]
