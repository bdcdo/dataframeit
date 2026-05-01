"""Testes para helpers internos de dataframeit.agent (sem chamadas a LLM/rede)."""

from types import SimpleNamespace
from typing import List, Optional
from unittest.mock import MagicMock

from pydantic import BaseModel, Field


def _make_config(**overrides):
    """Helper: monta LLMConfig com SearchConfig basico."""
    from dataframeit.llm import LLMConfig, SearchConfig

    cfg = LLMConfig(
        model="m", provider="p", api_key=None,
        max_retries=1, base_delay=0.0, max_delay=0.0, rate_limit_delay=0.0,
        search_config=SearchConfig(enabled=True, provider="tavily", max_results=5, search_depth="basic"),
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# =============================================================================
# _build_field_prompt
# =============================================================================

class TestBuildFieldPrompt:
    def test_prompt_replace_substitui_completamente(self):
        from dataframeit.agent import _build_field_prompt

        result = _build_field_prompt(
            "prompt original", "campo_x", "descricao",
            {"prompt": "novo prompt completo"},
        )
        assert result == "novo prompt completo"

    def test_sem_overrides_inclui_nome_e_descricao(self):
        from dataframeit.agent import _build_field_prompt

        result = _build_field_prompt(
            "Analise: {texto}", "categoria", "tipo do documento",
            {},
        )
        assert "Analise: {texto}" in result
        assert "categoria" in result
        assert "tipo do documento" in result

    def test_prompt_append_concatena(self):
        from dataframeit.agent import _build_field_prompt

        result = _build_field_prompt(
            "base", "f", None,
            {"prompt_append": "instrucao extra"},
        )
        assert "base" in result
        assert "instrucao extra" in result

    def test_sem_descricao_omite_parenteses(self):
        from dataframeit.agent import _build_field_prompt

        result = _build_field_prompt("base", "f", None, {})
        assert "(" not in result.split("campo:")[1]


# =============================================================================
# _apply_field_overrides
# =============================================================================

class TestApplyFieldOverrides:
    def test_sem_overrides_retorna_config_original(self):
        from dataframeit.agent import _apply_field_overrides

        cfg = _make_config()
        result = _apply_field_overrides(cfg, {})
        assert result is cfg

    def test_search_depth_override(self):
        from dataframeit.agent import _apply_field_overrides

        cfg = _make_config()
        result = _apply_field_overrides(cfg, {"search_depth": "advanced"})
        assert result is not cfg
        assert result.search_config.search_depth == "advanced"
        # max_results preservado
        assert result.search_config.max_results == 5

    def test_max_results_override(self):
        from dataframeit.agent import _apply_field_overrides

        cfg = _make_config()
        result = _apply_field_overrides(cfg, {"max_results": 20})
        assert result.search_config.max_results == 20
        assert result.search_config.search_depth == "basic"

    def test_ambos_overrides(self):
        from dataframeit.agent import _apply_field_overrides

        cfg = _make_config()
        result = _apply_field_overrides(
            cfg, {"search_depth": "advanced", "max_results": 10}
        )
        assert result.search_config.search_depth == "advanced"
        assert result.search_config.max_results == 10

    def test_nao_muta_config_original(self):
        from dataframeit.agent import _apply_field_overrides

        cfg = _make_config()
        _apply_field_overrides(cfg, {"max_results": 99})
        assert cfg.search_config.max_results == 5


# =============================================================================
# _apply_group_overrides
# =============================================================================

class TestApplyGroupOverrides:
    def test_sem_overrides_retorna_config_original(self):
        from dataframeit.agent import _apply_group_overrides
        from dataframeit.llm import SearchGroupConfig

        cfg = _make_config()
        group = SearchGroupConfig(fields=["a"])
        assert _apply_group_overrides(cfg, group) is cfg

    def test_aplica_search_depth_e_max_results(self):
        from dataframeit.agent import _apply_group_overrides
        from dataframeit.llm import SearchGroupConfig

        cfg = _make_config()
        group = SearchGroupConfig(fields=["a"], search_depth="advanced", max_results=15)
        result = _apply_group_overrides(cfg, group)
        assert result.search_config.search_depth == "advanced"
        assert result.search_config.max_results == 15
        # original intacto
        assert cfg.search_config.search_depth == "basic"


# =============================================================================
# _set_nested_value
# =============================================================================

class TestSetNestedValue:
    def test_path_simples(self):
        from dataframeit.agent import _set_nested_value

        obj = {}
        _set_nested_value(obj, "a", 1)
        assert obj == {"a": 1}

    def test_path_aninhado(self):
        from dataframeit.agent import _set_nested_value

        obj = {}
        _set_nested_value(obj, "a.b.c", "v")
        assert obj == {"a": {"b": {"c": "v"}}}

    def test_preserva_chaves_existentes(self):
        from dataframeit.agent import _set_nested_value

        obj = {"a": {"existing": 1}}
        _set_nested_value(obj, "a.new", 2)
        assert obj == {"a": {"existing": 1, "new": 2}}

    def test_substitui_valor_nao_dict_no_caminho(self):
        from dataframeit.agent import _set_nested_value

        obj = {"a": "string"}
        _set_nested_value(obj, "a.b", 5)
        assert obj == {"a": {"b": 5}}


# =============================================================================
# _build_item_context
# =============================================================================

class _ItemModel(BaseModel):
    nome: str
    quantidade: int
    obs: Optional[str] = None


class TestBuildItemContext:
    def test_inclui_primeiros_campos_simples(self):
        from dataframeit.agent import _build_item_context

        ctx = _build_item_context(
            {"nome": "X", "quantidade": 3, "obs": "ok"}, _ItemModel,
        )
        assert "nome: X" in ctx
        assert "quantidade: 3" in ctx

    def test_pula_dict_e_list(self):
        from dataframeit.agent import _build_item_context

        class M(BaseModel):
            simples: str
            complexo: list

        ctx = _build_item_context(
            {"simples": "ok", "complexo": [1, 2]}, M,
        )
        assert "simples: ok" in ctx
        assert "complexo" not in ctx

    def test_dict_vazio_retorna_item(self):
        from dataframeit.agent import _build_item_context

        assert _build_item_context({}, _ItemModel) == "item"

    def test_limita_a_tres_campos(self):
        from dataframeit.agent import _build_item_context

        class M(BaseModel):
            a: str
            b: str
            c: str
            d: str

        ctx = _build_item_context({"a": "1", "b": "2", "c": "3", "d": "4"}, M)
        assert "d: 4" not in ctx


# =============================================================================
# _collect_configured_fields
# =============================================================================

class TestCollectConfiguredFields:
    def test_modelo_simples_sem_config(self):
        from dataframeit.agent import _collect_configured_fields

        class M(BaseModel):
            a: str
            b: int

        assert _collect_configured_fields(M) == []

    def test_campo_com_prompt(self):
        from dataframeit.agent import _collect_configured_fields

        class M(BaseModel):
            a: str = Field(json_schema_extra={"prompt": "busque a"})
            b: int

        result = _collect_configured_fields(M)
        assert len(result) == 1
        path, name, _info, parent, has_config = result[0]
        assert path == "a"
        assert name == "a"
        assert parent is M
        assert has_config is True

    def test_modelo_aninhado(self):
        from dataframeit.agent import _collect_configured_fields

        class Inner(BaseModel):
            campo: str = Field(json_schema_extra={"max_results": 10})

        class Outer(BaseModel):
            interno: Inner

        result = _collect_configured_fields(Outer)
        paths = [r[0] for r in result]
        assert "interno.campo" in paths

    def test_lista_de_modelos_aninhados(self):
        from dataframeit.agent import _collect_configured_fields

        class Item(BaseModel):
            descricao: str = Field(json_schema_extra={"search_depth": "advanced"})

        class Container(BaseModel):
            items: List[Item]

        result = _collect_configured_fields(Container)
        paths = [r[0] for r in result]
        assert "items.descricao" in paths

    def test_nao_loop_em_modelo_visitado(self):
        """O parametro _visited evita recursao infinita."""
        from dataframeit.agent import _collect_configured_fields

        class A(BaseModel):
            x: str = Field(json_schema_extra={"prompt": "p"})

        # Chamar duas vezes com mesmo _visited nao duplica
        visited = set()
        first = _collect_configured_fields(A, _visited=visited)
        second = _collect_configured_fields(A, _visited=visited)
        assert len(first) == 1
        assert second == []  # ja visitado


# =============================================================================
# _extract_usage
# =============================================================================

def _msg_with_usage(input_tokens, output_tokens, total_tokens, reasoning=0):
    """Cria mensagem mockada com usage_metadata."""
    return SimpleNamespace(
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "output_token_details": {"reasoning": reasoning},
        },
        type="ai",
    )


def _msg_with_tool_calls(tool_names):
    return SimpleNamespace(
        tool_calls=[{"name": n, "args": {}, "id": "id"} for n in tool_names],
        type="ai",
    )


def _make_provider(name="tavily", pattern="tavily", credits_fn=None):
    provider = MagicMock()
    provider.name = name
    provider.get_tool_name_pattern.return_value = pattern
    provider.calculate_credits.side_effect = (
        credits_fn if credits_fn else lambda search_count, **kw: search_count
    )
    return provider


class TestExtractUsage:
    def test_soma_tokens_de_multiplas_mensagens(self):
        from dataframeit.agent import _extract_usage
        from dataframeit.llm import SearchConfig

        result = {
            "messages": [
                _msg_with_usage(10, 5, 15),
                _msg_with_usage(20, 10, 30),
            ],
        }
        provider = _make_provider()
        usage = _extract_usage(result, provider, SearchConfig(provider="tavily"))
        assert usage["input_tokens"] == 30
        assert usage["output_tokens"] == 15
        assert usage["total_tokens"] == 45

    def test_acumula_reasoning_tokens(self):
        from dataframeit.agent import _extract_usage
        from dataframeit.llm import SearchConfig

        result = {"messages": [_msg_with_usage(0, 0, 0, reasoning=8)]}
        usage = _extract_usage(result, _make_provider(), SearchConfig(provider="tavily"))
        assert usage["reasoning_tokens"] == 8

    def test_usage_metadata_como_objeto(self):
        from dataframeit.agent import _extract_usage
        from dataframeit.llm import SearchConfig

        meta = SimpleNamespace(
            input_tokens=1, output_tokens=2, total_tokens=3,
            output_token_details=SimpleNamespace(reasoning=4),
        )
        msg = SimpleNamespace(usage_metadata=meta, type="ai")
        usage = _extract_usage({"messages": [msg]}, _make_provider(), SearchConfig(provider="tavily"))
        assert usage["input_tokens"] == 1
        assert usage["reasoning_tokens"] == 4

    def test_search_count_via_padrao_do_provider(self):
        from dataframeit.agent import _extract_usage
        from dataframeit.llm import SearchConfig

        result = {
            "messages": [
                _msg_with_tool_calls(["tavily_search", "tavily_search", "outra_tool"]),
            ],
        }
        provider = _make_provider(pattern="tavily")
        usage = _extract_usage(result, provider, SearchConfig(provider="tavily"))
        assert usage["search_count"] == 2

    def test_search_count_pelo_substring_search(self):
        from dataframeit.agent import _extract_usage
        from dataframeit.llm import SearchConfig

        result = {"messages": [_msg_with_tool_calls(["custom_search_tool"])]}
        provider = _make_provider(pattern="naoexiste")
        usage = _extract_usage(result, provider, SearchConfig(provider="tavily"))
        assert usage["search_count"] == 1

    def test_search_credits_calculado_pelo_provider(self):
        from dataframeit.agent import _extract_usage
        from dataframeit.llm import SearchConfig

        result = {"messages": [_msg_with_tool_calls(["tavily_search"])]}
        provider = _make_provider(
            pattern="tavily",
            credits_fn=lambda search_count, **kw: search_count * 2,
        )
        usage = _extract_usage(result, provider, SearchConfig(provider="tavily", search_depth="advanced"))
        assert usage["search_credits"] == 2
        assert usage["search_provider"] == "tavily"

    def test_resultado_sem_messages(self):
        from dataframeit.agent import _extract_usage
        from dataframeit.llm import SearchConfig

        usage = _extract_usage({}, _make_provider(), SearchConfig(provider="tavily"))
        assert usage["input_tokens"] == 0
        assert usage["search_count"] == 0
        assert usage["search_credits"] == 0


# =============================================================================
# _extract_trace
# =============================================================================

def _trace_msg(msg_type, content, tool_calls=None, tool_call_id=None):
    msg = SimpleNamespace(type=msg_type, content=content)
    if tool_calls is not None:
        msg.tool_calls = tool_calls
    if tool_call_id is not None:
        msg.tool_call_id = tool_call_id
    return msg


class TestExtractTrace:
    def test_trace_basico(self):
        from dataframeit.agent import _extract_trace

        result = {"messages": [_trace_msg("human", "ola")]}
        trace = _extract_trace(result, "modelo-x", 1.234, "full")
        assert trace["model"] == "modelo-x"
        assert trace["duration_seconds"] == 1.234
        assert trace["search_provider"] is None
        assert trace["messages"][0]["type"] == "human"
        assert trace["messages"][0]["content"] == "ola"

    def test_minimal_omite_content_de_tool(self):
        from dataframeit.agent import _extract_trace

        result = {
            "messages": [
                _trace_msg("tool", "resultado verboso", tool_call_id="abc"),
            ],
        }
        trace = _extract_trace(result, "m", 0.1, "minimal")
        assert trace["messages"][0]["content"] == "[omitted]"
        assert trace["messages"][0]["tool_call_id"] == "abc"

    def test_full_preserva_content_de_tool(self):
        from dataframeit.agent import _extract_trace

        result = {"messages": [_trace_msg("tool", "verboso", tool_call_id="abc")]}
        trace = _extract_trace(result, "m", 0.1, "full")
        assert trace["messages"][0]["content"] == "verboso"

    def test_extrai_search_queries_de_tool_calls(self):
        from dataframeit.agent import _extract_trace

        result = {
            "messages": [
                _trace_msg("ai", "", tool_calls=[
                    {"name": "tavily_search", "args": {"query": "q1"}, "id": "1"},
                    {"name": "outra_tool", "args": {}, "id": "2"},
                    {"name": "tavily_search", "args": {"query": "q2"}, "id": "3"},
                ]),
            ],
        }
        trace = _extract_trace(result, "m", 0.0, "full")
        assert trace["search_queries"] == ["q1", "q2"]
        assert trace["total_tool_calls"] == 2

    def test_provider_preenche_search_provider(self):
        from dataframeit.agent import _extract_trace

        provider = SimpleNamespace(name="exa")
        trace = _extract_trace({"messages": []}, "m", 0.0, "full", provider=provider)
        assert trace["search_provider"] == "exa"

    def test_arredonda_duration(self):
        from dataframeit.agent import _extract_trace

        trace = _extract_trace({"messages": []}, "m", 1.23456789, "full")
        assert trace["duration_seconds"] == 1.235
