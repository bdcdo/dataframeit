"""Ferramentas de busca (Exa, Tavily), teto de buscas por execução e contagem."""

import asyncio
import itertools
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel, Field

from dataframeit import dataframeit
from dataframeit.agent import _blocked_tool_call_ids, _extract_trace, _recursion_limit, call_agent
from dataframeit.llm import LLMConfig, SearchConfig, SearchGroupConfig
from dataframeit.search import get_provider


class Resposta(BaseModel):
    campo: str


def _config(**busca):
    return LLMConfig(
        model="m",
        provider="openai",
        api_key=None,
        max_retries=1,
        base_delay=0.0,
        max_delay=0.0,
        rate_limit_delay=0.0,
        search_config=SearchConfig(enabled=True, provider="tavily", **busca),
    )


# =============================================================================
# Exa
# =============================================================================


@pytest.fixture
def exa_falso(monkeypatch):
    pytest.importorskip("langchain_exa")
    import exa_py  # noqa: PLC0415 (o langchain_exa, pulado acima quando ausente, traz o exa_py)

    monkeypatch.setenv("EXA_API_KEY", "chave-de-teste")
    chamadas = []

    def search_and_contents(self, query, **kwargs):
        chamadas.append((query, kwargs))
        return "resultados"

    monkeypatch.setattr(exa_py.Exa, "search_and_contents", search_and_contents)
    return chamadas


def test_exa_respeita_max_results_e_limite_de_texto(exa_falso):

    ferramenta = get_provider("exa").create_tool(max_results=3)
    ferramenta.invoke({"query": "dipirona anvisa"})

    assert exa_falso == [("dipirona anvisa", {"num_results": 3, "text": {"max_characters": 1000}})]


def test_exa_expoe_so_a_consulta_ao_modelo(exa_falso):

    ferramenta = get_provider("exa").create_tool(max_results=3)
    assert set(ferramenta.args) == {"query"}


def test_exa_levanta_o_erro_do_provider(monkeypatch):
    pytest.importorskip("langchain_exa")
    import exa_py  # noqa: PLC0415 (o langchain_exa, pulado acima quando ausente, traz o exa_py)

    monkeypatch.setenv("EXA_API_KEY", "chave-de-teste")

    def falha(self, query, **kwargs):
        msg = "401 Unauthorized: invalid api key"
        raise RuntimeError(msg)

    monkeypatch.setattr(exa_py.Exa, "search_and_contents", falha)

    ferramenta = get_provider("exa").create_tool(max_results=3)
    with pytest.raises(RuntimeError, match="401"):
        ferramenta.invoke({"query": "x"})


# =============================================================================
# Tavily
# =============================================================================


@pytest.fixture
def tavily_disponivel(monkeypatch):
    pytest.importorskip("langchain_tavily")
    monkeypatch.setenv("TAVILY_API_KEY", "chave-de-teste")
    # Import direto, e não importorskip: se o módulo interno mudar de lugar numa versão
    # nova, o teste deve quebrar, e não pular.
    from langchain_tavily import _utilities  # noqa: PLC0415

    return _utilities.TavilySearchAPIWrapper


@pytest.mark.parametrize(
    "mensagem",
    [
        "Error 432: This request exceeds your plan's set usage limit.",
        "Error 401: Unauthorized: missing or invalid API key.",
        "Error 429: Too many requests.",
        "Error 500: Internal server error.",
    ],
)
def test_tavily_levanta_o_erro_do_provider(tavily_disponivel, monkeypatch, mensagem):
    # O wrapper real levanta ValueError com o status só no texto.
    def falha(self, **kwargs):
        raise ValueError(mensagem)

    monkeypatch.setattr(tavily_disponivel, "raw_results", falha)

    ferramenta = get_provider("tavily").create_tool(max_results=3)
    with pytest.raises(ValueError, match=mensagem[:9]):
        ferramenta.invoke({"query": "x"})


@pytest.mark.parametrize(
    "mensagem",
    [
        "Error 400: Query is too long. Max query length is 400 characters.",
        "Error 422: Invalid time_range.",
    ],
)
def test_tavily_devolve_ao_modelo_o_erro_de_argumento(tavily_disponivel, monkeypatch, mensagem):
    def falha(self, **kwargs):
        raise ValueError(mensagem)

    monkeypatch.setattr(tavily_disponivel, "raw_results", falha)

    ferramenta = get_provider("tavily").create_tool(max_results=3)
    assert mensagem in str(ferramenta.invoke({"query": "x"}))


def test_tavily_assincrono_tambem_levanta(tavily_disponivel, monkeypatch):

    async def falha(self, **kwargs):
        msg = "Error 432: usage limit."
        raise ValueError(msg)

    monkeypatch.setattr(tavily_disponivel, "raw_results_async", falha)

    ferramenta = get_provider("tavily").create_tool(max_results=3)
    with pytest.raises(ValueError, match="Error 432"):
        asyncio.run(ferramenta.ainvoke({"query": "x"}))


def test_tavily_sem_resultado_continua_sendo_mensagem_ao_modelo(tavily_disponivel, monkeypatch):
    monkeypatch.setattr(tavily_disponivel, "raw_results", lambda self, **kwargs: {"results": []})

    ferramenta = get_provider("tavily").create_tool(max_results=3)
    assert "No search results" in str(ferramenta.invoke({"query": "x"}))


# =============================================================================
# Teto de buscas por execução
# =============================================================================


def _chamar_agente(monkeypatch, config, mensagens=()):

    capturado = {}

    class AgenteFalso:
        def invoke(self, _payload, config=None):
            capturado["config"] = config
            return {"structured_response": Resposta(campo="ok"), "messages": list(mensagens)}

    def create_agent(**kwargs):
        capturado.update(kwargs)
        return AgenteFalso()

    class FerramentaFalsa:
        name = "busca_web"

    monkeypatch.setattr("dataframeit.llm._create_langchain_llm", lambda *a, **k: object())
    monkeypatch.setattr("langchain.agents.create_agent", create_agent)
    with patch("dataframeit.agent.get_provider") as get_provider:
        provider = MagicMock()
        provider.name = "tavily"
        provider.create_tool = lambda **kwargs: FerramentaFalsa()
        provider.calculate_credits.side_effect = lambda search_count, **kw: search_count
        get_provider.return_value = provider
        resultado = call_agent("t", Resposta, "Responda {texto}", config, save_trace="full")
    return capturado, resultado


def test_agente_recebe_o_teto_de_buscas(monkeypatch):

    capturado, _ = _chamar_agente(monkeypatch, _config(max_search_calls=4))
    assert capturado["config"] == {"recursion_limit": 3 * 4 + 20}

    limites = [m for m in capturado["middleware"] if isinstance(m, ToolCallLimitMiddleware)]
    assert len(limites) == 1
    assert limites[0].run_limit == 4
    assert limites[0].tool_name == "busca_web"
    assert limites[0].exit_behavior == "continue"


def test_busca_bloqueada_pelo_teto_nao_conta(monkeypatch):

    mensagens = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "busca_web", "args": {"query": "a"}, "id": "1"},
                {"name": "busca_web", "args": {"query": "b"}, "id": "2"},
            ],
        ),
        ToolMessage(content="resultado", tool_call_id="1", name="busca_web"),
        ToolMessage(
            content="Tool call limit exceeded. Do not call 'busca_web' again.",
            tool_call_id="2",
            name="busca_web",
            status="error",
        ),
        AIMessage(
            content="", tool_calls=[{"name": "Resposta", "args": {"campo": "ok"}, "id": "3"}]
        ),
    ]
    _, resultado = _chamar_agente(monkeypatch, _config(), mensagens)

    assert resultado["usage"]["search_count"] == 1
    assert resultado["trace"]["search_queries"] == ["a"]
    assert resultado["trace"]["total_tool_calls"] == 3


# =============================================================================
# max_search_calls na API
# =============================================================================


def _executar(questions, **opcoes):

    with (
        patch("dataframeit.core.validate_provider_dependencies"),
        patch("dataframeit.core.validate_search_dependencies"),
        patch("dataframeit.agent.call_agent") as call_agent,
    ):
        call_agent.return_value = {"data": {"campo": "x"}, "usage": {}}
        dataframeit(
            pd.DataFrame({"texto": ["a"]}),
            questions=questions,
            prompt="{texto}",
            use_search=True,
            track_tokens=False,
            **opcoes,
        )
    return call_agent


@pytest.mark.parametrize("valor", [0, True, 2.5])
def test_max_search_calls_invalido(valor):
    with pytest.raises(ValueError, match="max_search_calls"):
        _executar(Resposta, max_search_calls=valor)


def test_max_search_calls_chega_ao_agente():
    call_agent = _executar(Resposta, max_search_calls=3)
    config = call_agent.call_args.args[3]
    assert config.search_config.max_search_calls == 3


def test_max_search_calls_por_campo_e_por_grupo():
    class Modelo(BaseModel):
        a: str = Field(json_schema_extra={"max_search_calls": 2})
        b: str = ""
        c: str = ""

    limites = {}

    def falso(text, model, prompt, config, save_trace=None):
        limites[tuple(model.model_fields)] = config.search_config.max_search_calls
        return {"data": dict.fromkeys(model.model_fields, "x"), "usage": {}}

    with (
        patch("dataframeit.core.validate_provider_dependencies"),
        patch("dataframeit.core.validate_search_dependencies"),
        patch("dataframeit.agent.call_agent", side_effect=falso),
    ):
        dataframeit(
            pd.DataFrame({"texto": ["t"]}),
            questions=Modelo,
            prompt="{texto}",
            use_search=True,
            search_per_field=True,
            track_tokens=False,
            search_groups={"g": {"fields": ["b", "c"], "max_search_calls": 5}},
        )

    assert limites == {("a",): 2, ("b", "c"): 5}


def test_max_search_calls_invalido_por_campo():
    class Modelo(BaseModel):
        a: str = Field(json_schema_extra={"max_search_calls": 0})

    with pytest.raises(ValueError, match="Campo 'a': max_search_calls"):
        _executar(Modelo, search_per_field=True)


def test_group_config_aceita_max_search_calls():
    assert SearchGroupConfig(fields=["a"], max_search_calls=2).max_search_calls == 2


def test_trace_nao_confunde_structured_output_com_busca():

    mensagens = [
        SimpleNamespace(
            type="ai",
            content="",
            tool_calls=[
                {"name": "ResearchResult", "args": {"query": "nao e busca"}, "id": "9"},
            ],
        ),
    ]
    trace = _extract_trace({"messages": mensagens}, "m", 0.1, "full", None, "busca_web")
    assert trace["search_queries"] == []
    assert trace["total_tool_calls"] == 1


def test_max_search_calls_invalido_por_grupo():
    class Modelo(BaseModel):
        b: str = ""
        c: str = ""

    with pytest.raises(ValueError, match="max_search_calls"):
        _executar(
            Modelo,
            search_per_field=True,
            search_groups={"g": {"fields": ["b", "c"], "max_search_calls": 0}},
        )


def test_max_search_calls_padrao_e_10():
    assert SearchConfig().max_search_calls == 10
    config = _executar(Resposta).call_args.args[3]
    assert config.search_config.max_search_calls == 10


def test_max_search_calls_aceita_inteiro_numpy():

    config = _executar(Resposta, max_search_calls=np.int64(3)).call_args.args[3]
    assert config.search_config.max_search_calls == 3


def test_sem_resultados_nao_conta_como_busca_bloqueada():
    """ "Sem resultados" é ToolMessage com status 'error', mas a busca aconteceu."""

    mensagens = [
        AIMessage(
            content="", tool_calls=[{"name": "busca_web", "args": {"query": "a"}, "id": "1"}]
        ),
        ToolMessage(
            content="No search results found for 'a'.",
            tool_call_id="1",
            name="busca_web",
            status="error",
        ),
        ToolMessage(
            content="Tool call limit exceeded. Do not call 'busca_web' again.",
            tool_call_id="2",
            name="busca_web",
            status="success",
        ),
    ]
    assert _blocked_tool_call_ids(mensagens) == set()


def test_recursion_limit_interrompe_modelo_que_insiste_em_buscar():
    """Com o teto e o limite de passos, um modelo que ignora o bloqueio para cedo."""

    chamadas = []

    class ModeloTeimoso(GenericFakeChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    def mensagens():
        for i in itertools.count():
            chamadas.append(i)
            yield AIMessage(
                content="",
                tool_calls=[
                    {"name": "busca", "args": {"query": str(i)}, "id": f"c{i}", "type": "tool_call"}
                ],
            )

    def busca(query: str) -> str:
        """Busca."""
        return "x"

    agente = create_agent(
        model=ModeloTeimoso(messages=mensagens()),
        tools=[StructuredTool.from_function(busca, name="busca")],
        response_format=ToolStrategy(Resposta),
        middleware=[
            ToolCallLimitMiddleware(tool_name="busca", run_limit=2, exit_behavior="continue")
        ],
    )
    with pytest.raises(GraphRecursionError):
        agente.invoke(
            {"messages": [("user", "q")]}, config={"recursion_limit": _recursion_limit(2)}
        )
    assert len(chamadas) < 20


def test_chamada_da_busca_sem_consulta_conta_mas_nao_entra_no_trace(monkeypatch):
    """O modelo pode chamar a busca sem o argumento query; não há consulta a registrar."""

    mensagens = [
        AIMessage(content="", tool_calls=[{"name": "busca_web", "args": {}, "id": "1"}]),
        ToolMessage(content="resultado", tool_call_id="1", name="busca_web"),
    ]
    _, resultado = _chamar_agente(monkeypatch, _config(), mensagens)

    assert resultado["trace"]["search_queries"] == []
    assert resultado["trace"]["total_tool_calls"] == 1
    assert resultado["usage"]["search_count"] == 1


def test_agente_sem_resposta_estruturada_vira_erro_da_linha(monkeypatch):

    class AgenteSemResposta:
        def invoke(self, _payload, config=None):
            return {"messages": []}

    class FerramentaFalsa:
        name = "busca_web"

    monkeypatch.setattr("dataframeit.llm._create_langchain_llm", lambda *a, **k: object())
    monkeypatch.setattr("langchain.agents.create_agent", lambda **kwargs: AgenteSemResposta())
    provider = MagicMock()
    provider.create_tool = lambda **kwargs: FerramentaFalsa()
    with (
        patch("dataframeit.core.validate_provider_dependencies"),
        patch("dataframeit.core.validate_search_dependencies"),
        patch("dataframeit.agent.get_provider", return_value=provider),
        pytest.warns(UserWarning, match="Falha ao processar linha 0"),
    ):
        resultado = dataframeit(
            pd.DataFrame({"texto": ["a"]}),
            questions=Resposta,
            prompt="{texto}",
            use_search=True,
            max_retries=1,
            track_tokens=False,
        )

    assert resultado["_dataframeit_status"].tolist() == ["error"]
    assert "Agente não retornou resposta estruturada" in resultado["_error_details"].iloc[0]
