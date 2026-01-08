"""Testes para funcionalidade de busca web via Tavily."""

import pandas as pd
from pydantic import BaseModel, Field
from typing import Literal
import pytest
from unittest.mock import patch, MagicMock
import os


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
    assert config.per_field is False
    assert config.max_results == 5
    assert config.search_depth == "basic"


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

    usage = _extract_usage(mock_result, "basic")

    assert "input_tokens" in usage
    assert "output_tokens" in usage
    assert "total_tokens" in usage
    assert "search_credits" in usage
    assert "search_count" in usage
    assert usage["search_count"] == 1  # Uma tool call de busca
    assert usage["search_credits"] == 1  # basic = 1 crédito


def test_extract_usage_advanced_depth():
    """Verifica cálculo de créditos com search_depth='advanced'."""
    from dataframeit.agent import _extract_usage

    mock_result = {
        "messages": [
            MagicMock(
                usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
                tool_calls=[{"name": "tavily_search"}, {"name": "tavily_search"}]
            ),
        ],
    }

    usage = _extract_usage(mock_result, "advanced")

    assert usage["search_count"] == 2
    assert usage["search_credits"] == 4  # advanced = 2 créditos × 2 buscas


# =============================================================================
# Testes de call_agent_per_field
# =============================================================================

def test_call_agent_per_field_iterates_fields():
    """Verifica que call_agent_per_field itera por cada campo."""
    from dataframeit.agent import call_agent_per_field
    from dataframeit.llm import LLMConfig, SearchConfig

    # Contar quantas vezes call_agent é chamado
    call_count = 0

    def mock_call_agent(text, model, prompt, config):
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

    def mock_call_agent(text, model, prompt, config):
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
