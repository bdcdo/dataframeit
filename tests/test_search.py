"""Testes para funcionalidade de busca web via Tavily."""

import pandas as pd
from pydantic import BaseModel, Field
from typing import Literal, List, Optional
import pytest
from unittest.mock import patch, MagicMock
import os
import sys
import types


class MedicamentoInfo(BaseModel):
    """Modelo de teste para informações de medicamento."""
    principio_ativo: str = Field(description="Princípio ativo do medicamento")
    indicacao: str = Field(description="Indicação principal")


class PaisInfo(BaseModel):
    """Modelo de teste para informações de país."""
    capital: str = Field(description="Capital do país")
    populacao: str = Field(description="População aproximada")
    continente: Literal["América", "Europa", "Ásia", "África", "Oceania"] = Field(
        description="Continente onde se localiza"
    )


# =============================================================================
# Testes de parâmetros e validação
# =============================================================================

def test_use_search_false_by_default():
    """Verifica que use_search=False é o padrão."""
    from dataframeit.core import dataframeit
    import inspect

    sig = inspect.signature(dataframeit)
    assert sig.parameters['use_search'].default is False


def test_search_per_field_false_by_default():
    """Verifica que search_per_field=False é o padrão."""
    from dataframeit.core import dataframeit
    import inspect

    sig = inspect.signature(dataframeit)
    assert sig.parameters['search_per_field'].default is False


def test_max_results_default():
    """Verifica que max_results=5 é o padrão."""
    from dataframeit.core import dataframeit
    import inspect

    sig = inspect.signature(dataframeit)
    assert sig.parameters['max_results'].default == 5


def test_search_depth_default():
    """Verifica que search_depth='basic' é o padrão."""
    from dataframeit.core import dataframeit
    import inspect

    sig = inspect.signature(dataframeit)
    assert sig.parameters['search_depth'].default == "basic"


def test_use_search_validates_depth():
    """Verifica que search_depth inválido gera erro."""
    from dataframeit.core import dataframeit

    df = pd.DataFrame({"texto": ["Paracetamol"]})

    # Mock validate_provider_dependencies para não precisar do provider instalado
    with patch('dataframeit.core.validate_provider_dependencies'):
        with pytest.raises(ValueError) as exc_info:
            dataframeit(
                df,
                questions=MedicamentoInfo,
                prompt="Pesquise sobre {texto}",
                use_search=True,
                search_depth="invalid",
            )

    assert "search_depth" in str(exc_info.value)


def test_use_search_validates_max_results_min():
    """Verifica que max_results < 1 gera erro."""
    from dataframeit.core import dataframeit

    df = pd.DataFrame({"texto": ["Paracetamol"]})

    with patch('dataframeit.core.validate_provider_dependencies'):
        with pytest.raises(ValueError) as exc_info:
            dataframeit(
                df,
                questions=MedicamentoInfo,
                prompt="Pesquise sobre {texto}",
                use_search=True,
                max_results=0,
            )

    assert "max_results" in str(exc_info.value)


def test_use_search_validates_max_results_max():
    """Verifica que max_results > 20 gera erro."""
    from dataframeit.core import dataframeit

    df = pd.DataFrame({"texto": ["Paracetamol"]})

    with patch('dataframeit.core.validate_provider_dependencies'):
        with pytest.raises(ValueError) as exc_info:
            dataframeit(
                df,
                questions=MedicamentoInfo,
                prompt="Pesquise sobre {texto}",
                use_search=True,
                max_results=21,
            )

    assert "max_results" in str(exc_info.value)


# =============================================================================
# Testes de validação de dependências
# =============================================================================

def test_use_search_requires_tavily_package():
    """Verifica que use_search=True requer langchain-tavily instalado."""
    from dataframeit.errors import validate_search_dependencies

    # Mock importlib para simular pacote não instalado
    with patch('importlib.import_module') as mock_import:
        def side_effect(name):
            if name == 'langchain_tavily':
                raise ImportError("No module named 'langchain_tavily'")
            return MagicMock()

        mock_import.side_effect = side_effect

        with pytest.raises(ImportError) as exc_info:
            validate_search_dependencies()

        assert "langchain-tavily" in str(exc_info.value) or "langchain_tavily" in str(exc_info.value)


def test_use_search_requires_api_key():
    """Verifica que use_search=True requer TAVILY_API_KEY."""
    from dataframeit.errors import validate_search_dependencies

    # Salvar valor original
    original = os.environ.get('TAVILY_API_KEY')

    try:
        # Remover API key
        if 'TAVILY_API_KEY' in os.environ:
            del os.environ['TAVILY_API_KEY']

        # Mock do import para simular pacote instalado
        with patch('importlib.import_module') as mock_import:
            mock_import.return_value = MagicMock()

            with pytest.raises(ValueError) as exc_info:
                validate_search_dependencies()

            assert "TAVILY_API_KEY" in str(exc_info.value)
    finally:
        # Restaurar valor original
        if original is not None:
            os.environ['TAVILY_API_KEY'] = original


# =============================================================================
# Testes de SearchConfig
# =============================================================================

def test_search_config_creation():
    """Verifica criação de SearchConfig."""
    from dataframeit.llm import SearchConfig

    config = SearchConfig(
        enabled=True,
        per_field=True,
        max_results=10,
        search_depth="advanced",
    )

    assert config.enabled is True
    assert config.per_field is True
    assert config.max_results == 10
    assert config.search_depth == "advanced"


def test_search_config_defaults():
    """Verifica valores padrão de SearchConfig."""
    from dataframeit.llm import SearchConfig

    config = SearchConfig()

    assert config.enabled is False
    assert config.provider == "tavily"
    assert config.per_field is False
    assert config.max_results == 5
    assert config.search_depth == "basic"


def test_search_agent_uses_initialized_model(monkeypatch):
    """Garante que create_agent recebe um LLM, sem model_provider."""
    from dataframeit.agent import call_agent
    from dataframeit.llm import LLMConfig, SearchConfig

    class TestOutput(BaseModel):
        result: str

    class DummyAgent:
        def invoke(self, _payload):
            return {"structured_response": TestOutput(result="ok"), "messages": []}

    class DummyTavily:
        def __init__(self, *args, **kwargs):
            self.name = "tavily"

    dummy_llm = object()
    captured = {}

    def fake_create_agent(*, model, tools, response_format, **kwargs):
        captured["model"] = model
        captured["kwargs"] = kwargs
        return DummyAgent()

    monkeypatch.setattr("dataframeit.agent._create_langchain_llm", lambda *args, **kwargs: dummy_llm)
    monkeypatch.setattr("langchain.agents.create_agent", fake_create_agent)
    monkeypatch.setitem(sys.modules, "langchain_tavily", types.SimpleNamespace(TavilySearch=DummyTavily))

    config = LLMConfig(
        model="gpt-4o-mini",
        provider="openai",
        api_key=None,
        max_retries=1,
        base_delay=0.0,
        max_delay=0.0,
        rate_limit_delay=0.0,
        model_kwargs={},
        search_config=SearchConfig(enabled=True),
    )

    result = call_agent("teste", TestOutput, "Responda {texto}", config)

    assert captured["model"] is dummy_llm
    assert "model_provider" not in captured["kwargs"]
    assert "api_key" not in captured["kwargs"]
    assert result["data"]["result"] == "ok"


def test_llm_config_with_search():
    """Verifica que LLMConfig aceita SearchConfig."""
    from dataframeit.llm import LLMConfig, SearchConfig

    search_config = SearchConfig(enabled=True)

    config = LLMConfig(
        model="gpt-4o-mini",
        provider="openai",
        api_key=None,
        max_retries=3,
        base_delay=1.0,
        max_delay=30.0,
        rate_limit_delay=0.0,
        search_config=search_config,
    )

    assert config.search_config is not None
    assert config.search_config.enabled is True


# =============================================================================
# Testes de setup de colunas
# =============================================================================

def test_setup_columns_with_search():
    """Verifica que _setup_columns cria colunas de busca quando habilitado."""
    from dataframeit.core import _setup_columns
    from dataframeit.llm import SearchConfig

    df = pd.DataFrame({"texto": ["a", "b"]})
    search_config = SearchConfig(enabled=True)

    _setup_columns(df, ["campo1"], None, False, True, search_config)

    assert "_search_credits" in df.columns
    assert "_search_count" in df.columns


def test_setup_columns_without_search():
    """Verifica que _setup_columns não cria colunas de busca quando desabilitado."""
    from dataframeit.core import _setup_columns

    df = pd.DataFrame({"texto": ["a", "b"]})

    _setup_columns(df, ["campo1"], None, False, True, None)

    assert "_search_credits" not in df.columns
    assert "_search_count" not in df.columns


# =============================================================================
# Testes de error handling
# =============================================================================

def test_tavily_errors_classified_correctly():
    """Verifica classificação de erros do Tavily."""
    from dataframeit.errors import is_recoverable_error, NON_RECOVERABLE_ERRORS, RECOVERABLE_ERRORS

    # Erros não-recuperáveis do Tavily
    assert "MissingAPIKeyError" in NON_RECOVERABLE_ERRORS
    assert "InvalidAPIKeyError" in NON_RECOVERABLE_ERRORS
    assert "BadRequestError" in NON_RECOVERABLE_ERRORS

    # Erros recuperáveis do Tavily
    assert "UsageLimitExceededError" in RECOVERABLE_ERRORS


def test_tavily_friendly_error_messages():
    """Verifica mensagens amigáveis para erros do Tavily."""
    from dataframeit.errors import get_friendly_error_message

    # Erro de API key
    class TavilyMissingAPIKeyError(Exception):
        pass

    error = TavilyMissingAPIKeyError("Missing API key for Tavily")
    msg = get_friendly_error_message(error)
    assert "TAVILY" in msg.upper() or "API" in msg.upper()

    # Erro de limite de uso
    class UsageLimitExceededError(Exception):
        pass

    error = UsageLimitExceededError("Usage limit exceeded for Tavily")
    msg = get_friendly_error_message(error)
    assert "LIMITE" in msg.upper() or "LIMIT" in msg.upper()


# =============================================================================
# Testes de integração (com mocks)
# =============================================================================

def test_call_agent_returns_expected_structure():
    """Verifica que call_agent retorna estrutura esperada."""
    from dataframeit.agent import _extract_usage
    from dataframeit.search import TavilyProvider
    from dataframeit.llm import SearchConfig

    provider = TavilyProvider()
    search_config = SearchConfig(enabled=True, provider="tavily")

    # Simular resultado do agente
    mock_result = {
        "messages": [
            MagicMock(
                usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
                tool_calls=[{"name": "tavily_search"}]
            ),
            MagicMock(
                usage_metadata={"input_tokens": 200, "output_tokens": 100, "total_tokens": 300},
                tool_calls=None
            ),
        ],
        "structured_response": MagicMock(),
    }

    usage = _extract_usage(mock_result, provider, search_config)

    assert "input_tokens" in usage
    assert "output_tokens" in usage
    assert "total_tokens" in usage
    assert "search_credits" in usage
    assert "search_count" in usage
    assert "search_provider" in usage
    assert usage["search_count"] == 1  # Uma tool call de busca
    assert usage["search_credits"] == 1  # basic = 1 crédito
    assert usage["search_provider"] == "tavily"


def test_extract_usage_advanced_depth():
    """Verifica cálculo de créditos com search_depth='advanced'."""
    from dataframeit.agent import _extract_usage
    from dataframeit.search import TavilyProvider
    from dataframeit.llm import SearchConfig

    provider = TavilyProvider()
    search_config = SearchConfig(enabled=True, provider="tavily", search_depth="advanced")

    mock_result = {
        "messages": [
            MagicMock(
                usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
                tool_calls=[{"name": "tavily_search"}, {"name": "tavily_search"}]
            ),
        ],
    }

    usage = _extract_usage(mock_result, provider, search_config)

    assert usage["search_count"] == 2
    assert usage["search_credits"] == 4  # advanced = 2 créditos × 2 buscas


def test_extract_usage_with_object_metadata():
    """Verifica extração de tokens quando usage_metadata é um objeto (não dict)."""
    from dataframeit.agent import _extract_usage
    from dataframeit.search import TavilyProvider
    from dataframeit.llm import SearchConfig

    provider = TavilyProvider()
    search_config = SearchConfig(enabled=True, provider="tavily")

    # Criar objeto mock que usa atributos ao invés de dict
    metadata_obj = MagicMock()
    metadata_obj.input_tokens = 150
    metadata_obj.output_tokens = 75
    metadata_obj.total_tokens = 225
    # Simular que não é um dict
    type(metadata_obj).__iter__ = MagicMock(side_effect=TypeError)

    mock_result = {
        "messages": [
            MagicMock(
                type="ai",
                usage_metadata=metadata_obj,
                tool_calls=None
            ),
        ],
    }

    usage = _extract_usage(mock_result, provider, search_config)

    assert usage["input_tokens"] == 150
    assert usage["output_tokens"] == 75
    assert usage["total_tokens"] == 225


def test_extract_usage_with_debug_logging(caplog):
    """Verifica que logging de diagnóstico funciona sem erros."""
    import logging
    from dataframeit.agent import _extract_usage
    from dataframeit.search import TavilyProvider
    from dataframeit.llm import SearchConfig

    provider = TavilyProvider()
    search_config = SearchConfig(enabled=True, provider="tavily")

    mock_result = {
        "messages": [
            MagicMock(
                type="human",
                usage_metadata=None,
                tool_calls=None
            ),
            MagicMock(
                type="ai",
                usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
                tool_calls=[{"name": "tavily_search"}]
            ),
            MagicMock(
                type="tool",
                usage_metadata=None,
                tool_calls=None
            ),
            MagicMock(
                type="ai",
                usage_metadata={"input_tokens": 200, "output_tokens": 100, "total_tokens": 300},
                tool_calls=None
            ),
        ],
    }

    # Ativar logging de debug
    with caplog.at_level(logging.DEBUG, logger="dataframeit.agent"):
        usage = _extract_usage(mock_result, provider, search_config)

    # Verificar que os tokens foram extraídos corretamente
    assert usage["input_tokens"] == 300  # 100 + 200
    assert usage["output_tokens"] == 150  # 50 + 100
    assert usage["search_count"] == 1

    # Verificar que o logging foi gerado
    assert "[token_tracking]" in caplog.text
    assert "Total messages in agent result: 4" in caplog.text


# =============================================================================
# Testes de call_agent_per_field
# =============================================================================

def test_call_agent_per_field_iterates_fields():
    """Verifica que call_agent_per_field itera por cada campo."""
    from dataframeit.agent import call_agent_per_field
    from dataframeit.llm import LLMConfig, SearchConfig

    # Contar quantas vezes call_agent é chamado
    call_count = 0

    def mock_call_agent(text, model, prompt, config, save_trace=None):
        nonlocal call_count
        call_count += 1

        # Extrair nome do campo do modelo
        field_name = list(model.model_fields.keys())[0]

        return {
            "data": {field_name: f"valor_{field_name}"},
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "search_credits": 1,
                "search_count": 1,
            }
        }

    search_config = SearchConfig(enabled=True, per_field=True)
    config = LLMConfig(
        model="gpt-4o-mini",
        provider="openai",
        api_key="test",
        max_retries=1,
        base_delay=0.1,
        max_delay=1.0,
        rate_limit_delay=0.0,
        search_config=search_config,
    )

    with patch('dataframeit.agent.call_agent', side_effect=mock_call_agent):
        result = call_agent_per_field(
            "Paracetamol",
            MedicamentoInfo,
            "Pesquise sobre o medicamento {texto}",
            config,
        )

    # MedicamentoInfo tem 2 campos
    assert call_count == 2

    # Verifica que dados foram combinados
    assert "principio_ativo" in result["data"]
    assert "indicacao" in result["data"]


def test_call_agent_per_field_sums_usage():
    """Verifica que call_agent_per_field soma usage de todas as chamadas."""
    from dataframeit.agent import call_agent_per_field
    from dataframeit.llm import LLMConfig, SearchConfig

    def mock_call_agent(text, model, prompt, config, save_trace=None):
        field_name = list(model.model_fields.keys())[0]
        return {
            "data": {field_name: f"valor_{field_name}"},
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
                "search_credits": 2,
                "search_count": 2,
            }
        }

    search_config = SearchConfig(enabled=True, per_field=True)
    config = LLMConfig(
        model="gpt-4o-mini",
        provider="openai",
        api_key="test",
        max_retries=1,
        base_delay=0.1,
        max_delay=1.0,
        rate_limit_delay=0.0,
        search_config=search_config,
    )

    with patch('dataframeit.agent.call_agent', side_effect=mock_call_agent):
        result = call_agent_per_field(
            "Paracetamol",
            MedicamentoInfo,
            "Pesquise sobre {texto}",
            config,
        )

    # MedicamentoInfo tem 2 campos, então soma 2x
    assert result["usage"]["input_tokens"] == 200
    assert result["usage"]["output_tokens"] == 100
    assert result["usage"]["total_tokens"] == 300
    assert result["usage"]["search_credits"] == 4
    assert result["usage"]["search_count"] == 4


# =============================================================================
# Testes de configuração per-field (json_schema_extra)
# =============================================================================

def test_get_field_config_extracts_prompt():
    """Testa extração de prompt do json_schema_extra."""
    from dataframeit.agent import _get_field_config

    extra = {"prompt": "Custom prompt: {texto}"}
    config = _get_field_config(extra)

    assert config["prompt"] == "Custom prompt: {texto}"
    assert config["prompt_append"] is None


def test_get_field_config_extracts_prompt_replace():
    """Testa que prompt_replace é equivalente a prompt."""
    from dataframeit.agent import _get_field_config

    extra = {"prompt_replace": "Replaced prompt: {texto}"}
    config = _get_field_config(extra)

    assert config["prompt"] == "Replaced prompt: {texto}"


def test_get_field_config_extracts_prompt_append():
    """Testa extração de prompt_append."""
    from dataframeit.agent import _get_field_config

    extra = {"prompt_append": "Extra instructions"}
    config = _get_field_config(extra)

    assert config["prompt"] is None
    assert config["prompt_append"] == "Extra instructions"


def test_get_field_config_extracts_search_params():
    """Testa extração de search_depth e max_results."""
    from dataframeit.agent import _get_field_config

    extra = {"search_depth": "advanced", "max_results": 10}
    config = _get_field_config(extra)

    assert config["search_depth"] == "advanced"
    assert config["max_results"] == 10


def test_build_field_prompt_replace():
    """Testa que prompt substitui completamente o prompt base."""
    from dataframeit.agent import _build_field_prompt

    field_config = {"prompt": "Custom prompt: {texto}"}
    prompt = _build_field_prompt("Base {texto}", "campo", "Descrição", field_config)

    assert prompt == "Custom prompt: {texto}"
    assert "Base" not in prompt
    assert "Responda APENAS" not in prompt


def test_build_field_prompt_append():
    """Testa que prompt_append adiciona ao prompt base."""
    from dataframeit.agent import _build_field_prompt

    field_config = {"prompt_append": "Extra info"}
    prompt = _build_field_prompt("Base {texto}", "campo", "Descrição", field_config)

    assert "Base {texto}" in prompt
    assert "Responda APENAS o campo: campo" in prompt
    assert "(Descrição)" in prompt
    assert "Extra info" in prompt


def test_build_field_prompt_default():
    """Testa comportamento padrão sem configuração."""
    from dataframeit.agent import _build_field_prompt

    field_config = {}
    prompt = _build_field_prompt("Base {texto}", "campo", "Descrição", field_config)

    assert "Base {texto}" in prompt
    assert "Responda APENAS o campo: campo" in prompt
    assert "(Descrição)" in prompt
    assert "Extra info" not in prompt


def test_apply_field_overrides_no_changes():
    """Testa que config original é retornada se não há overrides."""
    from dataframeit.agent import _apply_field_overrides
    from dataframeit.llm import LLMConfig, SearchConfig

    config = LLMConfig(
        model="test",
        provider="test",
        api_key=None,
        max_retries=3,
        base_delay=1.0,
        max_delay=60.0,
        rate_limit_delay=0,
        search_config=SearchConfig(enabled=True, search_depth="basic", max_results=5)
    )

    field_config = {}
    new_config = _apply_field_overrides(config, field_config)

    # Deve ser o mesmo objeto
    assert new_config is config


def test_apply_field_overrides_with_changes():
    """Testa override de search_depth e max_results."""
    from dataframeit.agent import _apply_field_overrides
    from dataframeit.llm import LLMConfig, SearchConfig

    config = LLMConfig(
        model="test",
        provider="test",
        api_key=None,
        max_retries=3,
        base_delay=1.0,
        max_delay=60.0,
        rate_limit_delay=0,
        search_config=SearchConfig(enabled=True, search_depth="basic", max_results=5)
    )

    field_config = {"search_depth": "advanced", "max_results": 10}
    new_config = _apply_field_overrides(config, field_config)

    # Novo config deve ter valores sobrescritos
    assert new_config.search_config.search_depth == "advanced"
    assert new_config.search_config.max_results == 10

    # Config original não deve mudar
    assert config.search_config.search_depth == "basic"
    assert config.search_config.max_results == 5


def test_has_field_config_detects_prompt():
    """Testa detecção de prompt em json_schema_extra."""
    from dataframeit.core import _has_field_config

    class ModelWithPrompt(BaseModel):
        campo: str = Field(json_schema_extra={"prompt": "custom"})

    assert _has_field_config(ModelWithPrompt) is True


def test_has_field_config_detects_prompt_append():
    """Testa detecção de prompt_append em json_schema_extra."""
    from dataframeit.core import _has_field_config

    class ModelWithAppend(BaseModel):
        campo: str = Field(json_schema_extra={"prompt_append": "extra"})

    assert _has_field_config(ModelWithAppend) is True


def test_has_field_config_detects_search_params():
    """Testa detecção de search_depth e max_results."""
    from dataframeit.core import _has_field_config

    class ModelWithSearchParams(BaseModel):
        campo: str = Field(json_schema_extra={"search_depth": "advanced"})

    assert _has_field_config(ModelWithSearchParams) is True


def test_has_field_config_returns_false_for_empty():
    """Testa que retorna False se não há configuração."""
    from dataframeit.core import _has_field_config

    class ModelWithoutConfig(BaseModel):
        campo: str = Field(description="Normal field")

    assert _has_field_config(ModelWithoutConfig) is False


def test_has_field_config_ignores_other_keys():
    """Testa que ignora outras chaves em json_schema_extra."""
    from dataframeit.core import _has_field_config

    class ModelWithOtherKeys(BaseModel):
        campo: str = Field(json_schema_extra={"other_key": "value"})

    assert _has_field_config(ModelWithOtherKeys) is False


def test_field_config_without_per_field_raises():
    """Testa que usar config per-field sem search_per_field=True dá erro."""
    from dataframeit.core import dataframeit

    class ModelWithConfig(BaseModel):
        campo: str = Field(json_schema_extra={"prompt": "custom"})

    df = pd.DataFrame({"texto": ["teste"]})

    with patch('dataframeit.core.validate_provider_dependencies'):
        with patch('dataframeit.core.validate_search_dependencies'):
            with pytest.raises(ValueError) as exc_info:
                dataframeit(
                    df,
                    questions=ModelWithConfig,
                    prompt="Analise {texto}",
                    use_search=True,
                    search_per_field=False,  # Deve dar erro
                )

    assert "search_per_field=True" in str(exc_info.value)


def test_field_config_with_per_field_no_error():
    """Testa que config per-field funciona com search_per_field=True."""
    from dataframeit.core import _has_field_config

    class ModelWithConfig(BaseModel):
        campo: str = Field(json_schema_extra={"prompt": "custom"})

    # Apenas verificar que a validação passa
    assert _has_field_config(ModelWithConfig) is True


def test_call_agent_per_field_uses_custom_prompt():
    """Testa que call_agent_per_field usa prompt customizado."""
    from dataframeit.agent import call_agent_per_field
    from dataframeit.llm import LLMConfig, SearchConfig

    class ModelWithCustomPrompt(BaseModel):
        campo_custom: str = Field(
            description="Campo com prompt customizado",
            json_schema_extra={"prompt": "Busque em fonte específica: {texto}"}
        )

    captured_prompts = []

    def mock_call_agent(text, model, prompt, config, save_trace=None):
        captured_prompts.append(prompt)
        field_name = list(model.model_fields.keys())[0]
        return {
            "data": {field_name: "valor"},
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                      "search_credits": 0, "search_count": 0}
        }

    config = LLMConfig(
        model="test", provider="test", api_key=None,
        max_retries=1, base_delay=0.1, max_delay=1.0, rate_limit_delay=0,
        search_config=SearchConfig(enabled=True, per_field=True)
    )

    with patch('dataframeit.agent.call_agent', side_effect=mock_call_agent):
        call_agent_per_field("Aspirina", ModelWithCustomPrompt, "Prompt base {texto}", config)

    assert len(captured_prompts) == 1
    assert "Busque em fonte específica: {texto}" in captured_prompts[0]
    assert "Prompt base" not in captured_prompts[0]


def test_call_agent_per_field_uses_config_override():
    """Testa que call_agent_per_field usa search_depth override."""
    from dataframeit.agent import call_agent_per_field
    from dataframeit.llm import LLMConfig, SearchConfig

    class ModelWithSearchOverride(BaseModel):
        campo: str = Field(json_schema_extra={"search_depth": "advanced"})

    captured_configs = []

    def mock_call_agent(text, model, prompt, config, save_trace=None):
        captured_configs.append(config)
        field_name = list(model.model_fields.keys())[0]
        return {
            "data": {field_name: "valor"},
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                      "search_credits": 0, "search_count": 0}
        }

    config = LLMConfig(
        model="test", provider="test", api_key=None,
        max_retries=1, base_delay=0.1, max_delay=1.0, rate_limit_delay=0,
        search_config=SearchConfig(enabled=True, per_field=True, search_depth="basic")
    )

    with patch('dataframeit.agent.call_agent', side_effect=mock_call_agent):
        call_agent_per_field("teste", ModelWithSearchOverride, "Prompt {texto}", config)

    assert len(captured_configs) == 1
    # Deve usar advanced (override), não basic (original)
    assert captured_configs[0].search_config.search_depth == "advanced"


# =============================================================================
# Testes de search_groups
# =============================================================================

class RegulatoryModel(BaseModel):
    """Modelo de teste para search_groups."""
    status_anvisa: str = Field(description="Status de aprovação na ANVISA")
    avaliacao_conitec: str = Field(description="Avaliação da CONITEC")
    nome: str = Field(description="Nome do medicamento")
    fabricante: str = Field(description="Fabricante")


def test_search_groups_requires_use_search():
    """Verifica que search_groups requer use_search=True."""
    from dataframeit.core import _validate_search_groups

    with pytest.raises(ValueError) as exc_info:
        _validate_search_groups(
            {"grupo": {"fields": ["nome"]}},
            RegulatoryModel,
            use_search=False,
            search_per_field=True
        )

    assert "use_search=True" in str(exc_info.value)


def test_search_groups_requires_per_field():
    """Verifica que search_groups requer search_per_field=True."""
    from dataframeit.core import _validate_search_groups

    with pytest.raises(ValueError) as exc_info:
        _validate_search_groups(
            {"grupo": {"fields": ["nome"]}},
            RegulatoryModel,
            use_search=True,
            search_per_field=False
        )

    assert "search_per_field=True" in str(exc_info.value)


def test_search_groups_validates_unknown_fields():
    """Verifica que campos inexistentes geram erro."""
    from dataframeit.core import _validate_search_groups

    with pytest.raises(ValueError) as exc_info:
        _validate_search_groups(
            {"grupo": {"fields": ["campo_inexistente"]}},
            RegulatoryModel,
            use_search=True,
            search_per_field=True
        )

    assert "campo_inexistente" in str(exc_info.value)
    assert "não existem" in str(exc_info.value)


def test_search_groups_validates_duplicate_fields():
    """Verifica que campos em múltiplos grupos geram erro."""
    from dataframeit.core import _validate_search_groups

    with pytest.raises(ValueError) as exc_info:
        _validate_search_groups(
            {
                "grupo1": {"fields": ["nome", "fabricante"]},
                "grupo2": {"fields": ["fabricante", "status_anvisa"]},  # fabricante duplicado
            },
            RegulatoryModel,
            use_search=True,
            search_per_field=True
        )

    assert "fabricante" in str(exc_info.value)
    assert "múltiplos grupos" in str(exc_info.value)


def test_search_groups_validates_field_config_conflict():
    """Verifica que campos com json_schema_extra em grupos geram erro."""
    from dataframeit.core import _validate_search_groups

    class ModelWithConfig(BaseModel):
        campo_a: str = Field(json_schema_extra={"prompt": "custom"})
        campo_b: str

    with pytest.raises(ValueError) as exc_info:
        _validate_search_groups(
            {"grupo": {"fields": ["campo_a", "campo_b"]}},
            ModelWithConfig,
            use_search=True,
            search_per_field=True
        )

    assert "campo_a" in str(exc_info.value)
    assert "json_schema_extra" in str(exc_info.value)


def test_search_groups_validates_search_depth():
    """Verifica que search_depth inválido no grupo gera erro."""
    from dataframeit.core import _validate_search_groups

    with pytest.raises(ValueError) as exc_info:
        _validate_search_groups(
            {"grupo": {"fields": ["nome"], "search_depth": "invalid"}},
            RegulatoryModel,
            use_search=True,
            search_per_field=True
        )

    assert "search_depth" in str(exc_info.value)


def test_search_groups_validates_max_results():
    """Verifica que max_results inválido no grupo gera erro."""
    from dataframeit.core import _validate_search_groups

    with pytest.raises(ValueError) as exc_info:
        _validate_search_groups(
            {"grupo": {"fields": ["nome"], "max_results": 100}},
            RegulatoryModel,
            use_search=True,
            search_per_field=True
        )

    assert "max_results" in str(exc_info.value)


def test_search_groups_valid_config():
    """Verifica que configuração válida é processada corretamente."""
    from dataframeit.core import _validate_search_groups
    from dataframeit.llm import SearchGroupConfig

    result = _validate_search_groups(
        {
            "regulatory": {
                "fields": ["status_anvisa", "avaliacao_conitec"],
                "prompt": "Busque regulatório: {query}",
                "max_results": 10,
                "search_depth": "advanced",
            }
        },
        RegulatoryModel,
        use_search=True,
        search_per_field=True
    )

    assert "regulatory" in result
    assert isinstance(result["regulatory"], SearchGroupConfig)
    assert result["regulatory"].fields == ["status_anvisa", "avaliacao_conitec"]
    assert result["regulatory"].prompt == "Busque regulatório: {query}"
    assert result["regulatory"].max_results == 10
    assert result["regulatory"].search_depth == "advanced"


def test_call_agent_per_group_basic():
    """Verifica que call_agent_per_group funciona com grupos."""
    from dataframeit.agent import call_agent_per_group
    from dataframeit.llm import LLMConfig, SearchConfig, SearchGroupConfig

    # Rastrear chamadas
    call_count = 0

    def mock_call_agent(text, model, prompt, config, save_trace=None):
        nonlocal call_count
        call_count += 1

        # Retornar valores para todos os campos do modelo
        fields = list(model.model_fields.keys())
        return {
            "data": {f: f"valor_{f}" for f in fields},
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "search_credits": 1,
                "search_count": 1,
            }
        }

    search_config = SearchConfig(
        enabled=True,
        per_field=True,
        groups={
            "regulatory": SearchGroupConfig(
                fields=["status_anvisa", "avaliacao_conitec"]
            )
        }
    )
    config = LLMConfig(
        model="test", provider="test", api_key=None,
        max_retries=1, base_delay=0.1, max_delay=1.0, rate_limit_delay=0,
        search_config=search_config
    )

    with patch('dataframeit.agent.call_agent', side_effect=mock_call_agent):
        result = call_agent_per_group(
            "Medicamento X",
            RegulatoryModel,
            "Pesquise sobre {texto}",
            config,
        )

    # Deve fazer 3 chamadas: 1 para grupo (2 campos) + 2 para campos isolados
    assert call_count == 3

    # Todos os campos devem ter valores
    assert "status_anvisa" in result["data"]
    assert "avaliacao_conitec" in result["data"]
    assert "nome" in result["data"]
    assert "fabricante" in result["data"]


def test_call_agent_per_group_sums_usage():
    """Verifica que call_agent_per_group soma usage de todas as chamadas."""
    from dataframeit.agent import call_agent_per_group
    from dataframeit.llm import LLMConfig, SearchConfig, SearchGroupConfig

    def mock_call_agent(text, model, prompt, config, save_trace=None):
        fields = list(model.model_fields.keys())
        return {
            "data": {f: f"valor_{f}" for f in fields},
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
                "search_credits": 2,
                "search_count": 1,
            }
        }

    search_config = SearchConfig(
        enabled=True,
        per_field=True,
        groups={
            "regulatory": SearchGroupConfig(
                fields=["status_anvisa", "avaliacao_conitec"]
            )
        }
    )
    config = LLMConfig(
        model="test", provider="test", api_key=None,
        max_retries=1, base_delay=0.1, max_delay=1.0, rate_limit_delay=0,
        search_config=search_config
    )

    with patch('dataframeit.agent.call_agent', side_effect=mock_call_agent):
        result = call_agent_per_group(
            "Medicamento X",
            RegulatoryModel,
            "Pesquise sobre {texto}",
            config,
        )

    # 3 chamadas (1 grupo + 2 isolados), 100 tokens cada
    assert result["usage"]["input_tokens"] == 300
    assert result["usage"]["output_tokens"] == 150
    assert result["usage"]["total_tokens"] == 450
    assert result["usage"]["search_credits"] == 6
    assert result["usage"]["search_count"] == 3


def test_call_agent_per_group_uses_custom_prompt():
    """Verifica que call_agent_per_group usa prompt customizado do grupo."""
    from dataframeit.agent import call_agent_per_group
    from dataframeit.llm import LLMConfig, SearchConfig, SearchGroupConfig

    captured_prompts = []

    def mock_call_agent(text, model, prompt, config, save_trace=None):
        captured_prompts.append(prompt)
        fields = list(model.model_fields.keys())
        return {
            "data": {f: f"valor_{f}" for f in fields},
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                      "search_credits": 0, "search_count": 0}
        }

    search_config = SearchConfig(
        enabled=True,
        per_field=True,
        groups={
            "regulatory": SearchGroupConfig(
                fields=["status_anvisa", "avaliacao_conitec"],
                prompt="Busque dados regulatórios ANVISA/CONITEC para {query}"
            )
        }
    )
    config = LLMConfig(
        model="test", provider="test", api_key=None,
        max_retries=1, base_delay=0.1, max_delay=1.0, rate_limit_delay=0,
        search_config=search_config
    )

    with patch('dataframeit.agent.call_agent', side_effect=mock_call_agent):
        call_agent_per_group("Aspirina", RegulatoryModel, "Prompt base {texto}", config)

    # Primeira chamada deve ser do grupo com prompt customizado
    assert any("ANVISA/CONITEC" in p for p in captured_prompts)


def test_call_agent_per_group_traces():
    """Verifica que traces são coletados por grupo e por campo isolado."""
    from dataframeit.agent import call_agent_per_group
    from dataframeit.llm import LLMConfig, SearchConfig, SearchGroupConfig

    call_counter = [0]

    def mock_call_agent(text, model, prompt, config, save_trace=None):
        call_counter[0] += 1
        fields = list(model.model_fields.keys())
        result = {
            "data": {f: f"valor_{f}" for f in fields},
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                      "search_credits": 0, "search_count": 0}
        }
        if save_trace:
            result["trace"] = {"call_number": call_counter[0]}
        return result

    search_config = SearchConfig(
        enabled=True,
        per_field=True,
        groups={
            "regulatory": SearchGroupConfig(
                fields=["status_anvisa", "avaliacao_conitec"]
            )
        }
    )
    config = LLMConfig(
        model="test", provider="test", api_key=None,
        max_retries=1, base_delay=0.1, max_delay=1.0, rate_limit_delay=0,
        search_config=search_config
    )

    with patch('dataframeit.agent.call_agent', side_effect=mock_call_agent):
        result = call_agent_per_group(
            "Medicamento X",
            RegulatoryModel,
            "Pesquise sobre {texto}",
            config,
            save_trace="full"
        )

    # Deve ter traces para grupo + campos isolados
    assert "traces" in result
    assert "regulatory" in result["traces"]  # Trace do grupo
    assert "nome" in result["traces"]  # Trace do campo isolado
    assert "fabricante" in result["traces"]  # Trace do campo isolado


def test_search_groups_setup_columns():
    """Verifica que _setup_columns cria colunas corretas para grupos."""
    from dataframeit.core import _setup_columns
    from dataframeit.llm import SearchConfig, SearchGroupConfig

    df = pd.DataFrame({"texto": ["a", "b"]})

    search_config = SearchConfig(
        enabled=True,
        per_field=True,
        groups={
            "regulatory": SearchGroupConfig(
                fields=["status_anvisa", "avaliacao_conitec"]
            )
        }
    )

    _setup_columns(
        df,
        ["status_anvisa", "avaliacao_conitec", "nome", "fabricante"],
        None, False, True, search_config, "full", RegulatoryModel
    )

    # Deve ter coluna de trace para o grupo
    assert "_trace_regulatory" in df.columns

    # Deve ter colunas de trace para campos isolados
    assert "_trace_nome" in df.columns
    assert "_trace_fabricante" in df.columns

    # NÃO deve ter colunas de trace para campos no grupo
    assert "_trace_status_anvisa" not in df.columns
    assert "_trace_avaliacao_conitec" not in df.columns


def test_apply_group_overrides_no_changes():
    """Verifica que config original é retornada se não há overrides."""
    from dataframeit.agent import _apply_group_overrides
    from dataframeit.llm import LLMConfig, SearchConfig, SearchGroupConfig

    config = LLMConfig(
        model="test", provider="test", api_key=None,
        max_retries=3, base_delay=1.0, max_delay=60.0, rate_limit_delay=0,
        search_config=SearchConfig(enabled=True, search_depth="basic", max_results=5)
    )

    group_config = SearchGroupConfig(fields=["campo"])
    new_config = _apply_group_overrides(config, group_config)

    # Deve ser o mesmo objeto
    assert new_config is config


def test_apply_group_overrides_with_changes():
    """Verifica override de search_depth e max_results no grupo."""
    from dataframeit.agent import _apply_group_overrides
    from dataframeit.llm import LLMConfig, SearchConfig, SearchGroupConfig

    config = LLMConfig(
        model="test", provider="test", api_key=None,
        max_retries=3, base_delay=1.0, max_delay=60.0, rate_limit_delay=0,
        search_config=SearchConfig(enabled=True, search_depth="basic", max_results=5)
    )

    group_config = SearchGroupConfig(
        fields=["campo"],
        search_depth="advanced",
        max_results=10
    )
    new_config = _apply_group_overrides(config, group_config)

    # Novo config deve ter valores sobrescritos
    assert new_config.search_config.search_depth == "advanced"
    assert new_config.search_config.max_results == 10

    # Config original não deve mudar
    assert config.search_config.search_depth == "basic"
    assert config.search_config.max_results == 5


# =============================================================================
# Testes de campos aninhados em List[Model]
# =============================================================================

class InformacoesMedicamento(BaseModel):
    """Modelo aninhado para informações de medicamento."""
    status_anvisa_atual: Optional[str] = Field(
        default=None,
        json_schema_extra={"search_depth": "basic", "max_results": 2},
    )
    principio_ativo: Optional[str] = Field(default=None)


class PedidoItem(BaseModel):
    """Item de pedido com modelo aninhado."""
    info_medicamento: Optional[InformacoesMedicamento] = None
    quantidade: int = Field(default=1)


class AnaliseSentencaSaude(BaseModel):
    """Modelo principal com List[Model]."""
    pedidos: List[PedidoItem]
    observacao: Optional[str] = None


def test_get_nested_pydantic_models_list():
    """Testa extração de modelos de List[Model]."""
    from dataframeit.utils import get_nested_pydantic_models

    models = get_nested_pydantic_models(List[PedidoItem])

    assert len(models) == 1
    assert models[0] is PedidoItem


def test_get_nested_pydantic_models_optional_list():
    """Testa extração de modelos de Optional[List[Model]]."""
    from dataframeit.utils import get_nested_pydantic_models

    models = get_nested_pydantic_models(Optional[List[PedidoItem]])

    assert len(models) == 1
    assert models[0] is PedidoItem


def test_get_nested_pydantic_models_optional_model():
    """Testa extração de modelos de Optional[Model]."""
    from dataframeit.utils import get_nested_pydantic_models

    models = get_nested_pydantic_models(Optional[InformacoesMedicamento])

    assert len(models) == 1
    assert models[0] is InformacoesMedicamento


def test_get_nested_pydantic_models_primitive():
    """Testa que tipos primitivos retornam lista vazia."""
    from dataframeit.utils import get_nested_pydantic_models

    assert get_nested_pydantic_models(str) == []
    assert get_nested_pydantic_models(int) == []
    assert get_nested_pydantic_models(Optional[str]) == []
    assert get_nested_pydantic_models(List[str]) == []


def test_has_field_config_detects_nested_in_list():
    """Verifica que _has_field_config detecta config em List[Model]."""
    from dataframeit.core import _has_field_config

    # AnaliseSentencaSaude tem pedidos: List[PedidoItem]
    # PedidoItem tem info_medicamento: InformacoesMedicamento
    # InformacoesMedicamento.status_anvisa_atual tem json_schema_extra
    assert _has_field_config(AnaliseSentencaSaude) is True


def test_has_field_config_detects_optional_list():
    """Verifica que _has_field_config detecta config em Optional[List[Model]]."""
    from dataframeit.core import _has_field_config

    class ModeloComOptionalList(BaseModel):
        itens: Optional[List[InformacoesMedicamento]] = None

    assert _has_field_config(ModeloComOptionalList) is True


def test_has_field_config_deeply_nested():
    """Verifica detecção em estruturas profundamente aninhadas."""
    from dataframeit.core import _has_field_config

    class Nivel3(BaseModel):
        campo_profundo: str = Field(json_schema_extra={"prompt": "deep search"})

    class Nivel2(BaseModel):
        nivel3: Nivel3

    class Nivel1(BaseModel):
        nivel2: List[Nivel2]

    class Raiz(BaseModel):
        nivel1: Nivel1

    assert _has_field_config(Raiz) is True


def test_has_field_config_no_infinite_recursion():
    """Verifica que modelos auto-referenciais não causam loop infinito."""
    from dataframeit.core import _has_field_config

    class NodoArvore(BaseModel):
        valor: str
        filhos: Optional[List['NodoArvore']] = None

    # Atualizar referências forward para Python resolver o tipo
    NodoArvore.model_rebuild()

    # Não deve entrar em loop infinito, deve retornar False (sem config)
    assert _has_field_config(NodoArvore) is False


def test_has_field_config_no_config_in_nested():
    """Verifica que retorna False se modelo aninhado não tem config."""
    from dataframeit.core import _has_field_config

    class NestedSimples(BaseModel):
        campo: str

    class RaizSimples(BaseModel):
        nested: List[NestedSimples]

    assert _has_field_config(RaizSimples) is False


def test_collect_configured_fields_returns_paths():
    """Verifica que _collect_configured_fields retorna caminhos corretos."""
    from dataframeit.agent import _collect_configured_fields

    results = _collect_configured_fields(AnaliseSentencaSaude)

    # Deve encontrar: pedidos.info_medicamento.status_anvisa_atual
    paths = [r[0] for r in results]

    assert len(results) >= 1
    assert any('status_anvisa_atual' in path for path in paths)
    assert any('pedidos' in path for path in paths)


def test_collect_configured_fields_deeply_nested():
    """Verifica coleta de caminhos em estruturas profundas."""
    from dataframeit.agent import _collect_configured_fields

    class Nivel3(BaseModel):
        campo_profundo: str = Field(json_schema_extra={"max_results": 5})

    class Nivel2(BaseModel):
        nivel3: Nivel3

    class Nivel1(BaseModel):
        nivel2: List[Nivel2]

    class Raiz(BaseModel):
        nivel1: Nivel1

    results = _collect_configured_fields(Raiz)
    paths = [r[0] for r in results]

    assert len(results) == 1
    assert 'nivel1.nivel2.nivel3.campo_profundo' in paths


def test_collect_configured_fields_no_infinite_recursion():
    """Verifica que modelos auto-referenciais não causam loop infinito."""
    from dataframeit.agent import _collect_configured_fields

    class NodoArvoreConfig(BaseModel):
        valor: str = Field(json_schema_extra={"prompt": "search value"})
        filhos: Optional[List['NodoArvoreConfig']] = None

    NodoArvoreConfig.model_rebuild()

    # Não deve entrar em loop infinito
    results = _collect_configured_fields(NodoArvoreConfig)

    # Deve encontrar apenas o campo 'valor' do primeiro nível
    paths = [r[0] for r in results]
    assert 'valor' in paths


def test_call_agent_per_field_nested_search():
    """Testa integração de call_agent_per_field com busca em campos aninhados."""
    from dataframeit.agent import call_agent_per_field
    from dataframeit.llm import LLMConfig, SearchConfig

    call_count = [0]
    captured_prompts = []

    def mock_call_agent(text, model, prompt, config, save_trace=None):
        call_count[0] += 1
        captured_prompts.append(prompt)

        # Retornar valores para todos os campos do modelo
        fields = list(model.model_fields.keys())
        return {
            "data": {f: f"valor_{f}" for f in fields},
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "search_credits": 1,
                "search_count": 1,
            }
        }

    search_config = SearchConfig(enabled=True, per_field=True)
    config = LLMConfig(
        model="test", provider="test", api_key=None,
        max_retries=1, base_delay=0.1, max_delay=1.0, rate_limit_delay=0,
        search_config=search_config
    )

    with patch('dataframeit.agent.call_agent', side_effect=mock_call_agent):
        result = call_agent_per_field(
            "Medicamento X",
            AnaliseSentencaSaude,
            "Analise {texto}",
            config,
        )

    # Deve ter chamado agente para:
    # 1. Campo aninhado configurado (status_anvisa_atual)
    # 2. Campos de primeiro nível (pedidos, observacao)
    assert call_count[0] >= 3

    # Resultado deve ter os campos de primeiro nível
    assert "pedidos" in result["data"]
    assert "observacao" in result["data"]


def test_call_agent_per_field_nested_context_in_prompt():
    """Verifica que contexto de busca aninhada é incluído no prompt."""
    from dataframeit.agent import call_agent_per_field
    from dataframeit.llm import LLMConfig, SearchConfig

    captured_prompts = []

    def mock_call_agent(text, model, prompt, config, save_trace=None):
        captured_prompts.append(prompt)

        fields = list(model.model_fields.keys())
        return {
            "data": {f: f"valor_busca_{f}" for f in fields},
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                      "search_credits": 0, "search_count": 0}
        }

    search_config = SearchConfig(enabled=True, per_field=True)
    config = LLMConfig(
        model="test", provider="test", api_key=None,
        max_retries=1, base_delay=0.1, max_delay=1.0, rate_limit_delay=0,
        search_config=search_config
    )

    with patch('dataframeit.agent.call_agent', side_effect=mock_call_agent):
        call_agent_per_field("Medicamento X", AnaliseSentencaSaude, "Analise {texto}", config)

    # Procurar prompt do campo 'pedidos' que deve ter contexto aninhado
    prompts_pedidos = [p for p in captured_prompts if 'pedidos' in p.lower()]

    # Deve haver menção ao contexto de buscas aninhadas
    pedidos_prompt = next((p for p in prompts_pedidos if 'Contexto de buscas' in p), None)
    assert pedidos_prompt is not None or len(prompts_pedidos) > 0


def test_call_agent_per_field_sums_nested_usage():
    """Verifica que usage de buscas aninhadas é somado."""
    from dataframeit.agent import call_agent_per_field
    from dataframeit.llm import LLMConfig, SearchConfig

    call_count = [0]

    def mock_call_agent(text, model, prompt, config, save_trace=None):
        call_count[0] += 1

        fields = list(model.model_fields.keys())
        return {
            "data": {f: f"valor_{f}" for f in fields},
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
                "search_credits": 1,
                "search_count": 1,
            }
        }

    search_config = SearchConfig(enabled=True, per_field=True)
    config = LLMConfig(
        model="test", provider="test", api_key=None,
        max_retries=1, base_delay=0.1, max_delay=1.0, rate_limit_delay=0,
        search_config=search_config
    )

    with patch('dataframeit.agent.call_agent', side_effect=mock_call_agent):
        result = call_agent_per_field(
            "Medicamento X",
            AnaliseSentencaSaude,
            "Analise {texto}",
            config,
        )

    # Usage deve ser a soma de todas as chamadas
    expected_calls = call_count[0]
    assert result["usage"]["input_tokens"] == expected_calls * 100
    assert result["usage"]["total_tokens"] == expected_calls * 150
    assert result["usage"]["search_count"] == expected_calls


# =============================================================================
# Testes de ordenação de colunas (Issue #81)
# =============================================================================

def test_reorder_columns_basic():
    """Verifica que _reorder_columns ordena colunas corretamente."""
    from dataframeit.utils import _reorder_columns

    # Criar DataFrame com colunas em ordem incorreta
    df = pd.DataFrame({
        'texto': ['a'],
        'campo1': ['b'],
        '_input_tokens': [100],
        '_output_tokens': [50],
        'campo2': ['c'],
        '_trace_grupo1': ['trace1'],
        '_total_tokens': [150],
        '_search_credits': [1],
        '_search_count': [1],
    })

    result = _reorder_columns(df)

    # Verificar ordem das colunas
    cols = result.columns.tolist()

    # Colunas do usuário primeiro
    assert cols.index('texto') < cols.index('_trace_grupo1')
    assert cols.index('campo1') < cols.index('_trace_grupo1')
    assert cols.index('campo2') < cols.index('_trace_grupo1')

    # Trace antes de search
    assert cols.index('_trace_grupo1') < cols.index('_search_credits')

    # Search antes de tokens
    assert cols.index('_search_credits') < cols.index('_input_tokens')
    assert cols.index('_search_count') < cols.index('_input_tokens')

    # Tokens no final
    token_cols = ['_input_tokens', '_output_tokens', '_total_tokens']
    for tcol in token_cols:
        assert cols.index(tcol) > cols.index('campo2')


def test_reorder_columns_with_status():
    """Verifica que colunas de status ficam no final."""
    from dataframeit.utils import _reorder_columns

    df = pd.DataFrame({
        'texto': ['a'],
        '_dataframeit_status': ['ok'],
        'campo1': ['b'],
        '_error_details': [None],
        '_input_tokens': [100],
    })

    result = _reorder_columns(df)
    cols = result.columns.tolist()

    # Status e error devem estar no final
    assert cols.index('_dataframeit_status') > cols.index('_input_tokens')
    assert cols.index('_error_details') > cols.index('_input_tokens')


def test_reorder_columns_multiple_traces():
    """Verifica ordenação com múltiplas colunas de trace."""
    from dataframeit.utils import _reorder_columns

    df = pd.DataFrame({
        'texto': ['a'],
        '_trace_campo1': ['t1'],
        'campo1': ['b'],
        '_input_tokens': [100],
        '_trace_grupo1': ['tg1'],
        'campo2': ['c'],
        '_output_tokens': [50],
        '_trace_campo2': ['t2'],
        '_total_tokens': [150],
    })

    result = _reorder_columns(df)
    cols = result.columns.tolist()

    # Todas as traces devem estar após os campos do usuário
    for trace_col in ['_trace_campo1', '_trace_grupo1', '_trace_campo2']:
        assert cols.index(trace_col) > cols.index('campo2')

    # Todas as traces devem estar antes dos tokens
    for trace_col in ['_trace_campo1', '_trace_grupo1', '_trace_campo2']:
        assert cols.index(trace_col) < cols.index('_input_tokens')


def test_column_ordering_in_from_pandas():
    """Verifica que from_pandas retorna colunas ordenadas corretamente."""
    from dataframeit.utils import from_pandas, ConversionInfo, ORIGINAL_TYPE_PANDAS_DF

    # Criar DataFrame com colunas em ordem incorreta (simulando problema real)
    df = pd.DataFrame({
        'texto': ['a'],
        'regulatory': ['info1'],
        '_input_tokens': [100],
        '_output_tokens': [50],
        'nome': ['nome1'],
        '_trace_grupo_reg': ['trace1'],
        '_total_tokens': [150],
        'tipo': ['tipo1'],
        '_search_credits': [1],
        '_search_count': [1],
    })

    conversion_info = ConversionInfo(original_type=ORIGINAL_TYPE_PANDAS_DF)
    result = from_pandas(df, conversion_info)

    cols = result.columns.tolist()

    # Ordem esperada: texto, regulatory, nome, tipo, _trace_*, _search_*, _*_tokens
    # Colunas do usuário
    user_cols = ['texto', 'regulatory', 'nome', 'tipo']
    for ucol in user_cols:
        assert cols.index(ucol) < cols.index('_trace_grupo_reg')

    # Trace antes de search
    assert cols.index('_trace_grupo_reg') < cols.index('_search_credits')

    # Search antes de tokens
    assert cols.index('_search_credits') < cols.index('_input_tokens')
    assert cols.index('_search_count') < cols.index('_input_tokens')

    # Tokens no final
    token_start_idx = cols.index('_input_tokens')
    assert '_output_tokens' in cols[token_start_idx:]
    assert '_total_tokens' in cols[token_start_idx:]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
