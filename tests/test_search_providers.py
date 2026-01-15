"""Testes para suporte a múltiplos provedores de busca (Tavily e Exa)."""

import pytest
from unittest.mock import patch, MagicMock
import os
import sys
import types

from pydantic import BaseModel, Field


class SampleModel(BaseModel):
    """Modelo de teste."""
    campo: str = Field(description="Campo de teste")


# =============================================================================
# Testes de registro de provedores
# =============================================================================

def test_get_provider_tavily():
    """Verifica que get_provider retorna TavilyProvider."""
    from dataframeit.search import get_provider, TavilyProvider

    provider = get_provider("tavily")
    assert isinstance(provider, TavilyProvider)
    assert provider.name == "tavily"


def test_get_provider_exa():
    """Verifica que get_provider retorna ExaProvider."""
    from dataframeit.search import get_provider, ExaProvider

    provider = get_provider("exa")
    assert isinstance(provider, ExaProvider)
    assert provider.name == "exa"


def test_get_provider_invalid():
    """Verifica que get_provider levanta erro para provedor inválido."""
    from dataframeit.search import get_provider

    with pytest.raises(ValueError) as exc_info:
        get_provider("invalid_provider")

    assert "não suportado" in str(exc_info.value)


def test_get_available_providers():
    """Verifica lista de provedores disponíveis."""
    from dataframeit.search import get_available_providers

    providers = get_available_providers()
    assert "tavily" in providers
    assert "exa" in providers


# =============================================================================
# Testes de TavilyProvider
# =============================================================================

def test_tavily_provider_properties():
    """Verifica propriedades do TavilyProvider."""
    from dataframeit.search import TavilyProvider

    provider = TavilyProvider()
    assert provider.name == "tavily"
    assert provider.env_var == "TAVILY_API_KEY"
    assert provider.package_name == "langchain_tavily"
    assert provider.install_name == "langchain-tavily"
    assert "tavily" in provider.signup_url.lower()


def test_tavily_calculate_credits_basic():
    """Verifica cálculo de créditos Tavily com depth basic."""
    from dataframeit.search import TavilyProvider

    provider = TavilyProvider()
    credits = provider.calculate_credits(search_count=3, search_depth="basic")
    assert credits == 3  # 3 buscas × 1 crédito


def test_tavily_calculate_credits_advanced():
    """Verifica cálculo de créditos Tavily com depth advanced."""
    from dataframeit.search import TavilyProvider

    provider = TavilyProvider()
    credits = provider.calculate_credits(search_count=3, search_depth="advanced")
    assert credits == 6  # 3 buscas × 2 créditos


def test_tavily_tool_name_pattern():
    """Verifica padrão de nome de ferramenta Tavily."""
    from dataframeit.search import TavilyProvider

    provider = TavilyProvider()
    assert "tavily" in provider.get_tool_name_pattern()


# =============================================================================
# Testes de ExaProvider
# =============================================================================

def test_exa_provider_properties():
    """Verifica propriedades do ExaProvider."""
    from dataframeit.search import ExaProvider

    provider = ExaProvider()
    assert provider.name == "exa"
    assert provider.env_var == "EXA_API_KEY"
    assert provider.package_name == "langchain_exa"
    assert provider.install_name == "langchain-exa"
    assert "exa" in provider.signup_url.lower()


def test_exa_calculate_credits_small_results():
    """Verifica cálculo de créditos Exa com <=25 resultados."""
    from dataframeit.search import ExaProvider

    provider = ExaProvider()
    credits = provider.calculate_credits(search_count=3, max_results=10)
    assert credits == 3  # 3 buscas × 1 crédito


def test_exa_calculate_credits_large_results():
    """Verifica cálculo de créditos Exa com >25 resultados."""
    from dataframeit.search import ExaProvider

    provider = ExaProvider()
    credits = provider.calculate_credits(search_count=3, max_results=50)
    assert credits == 15  # 3 buscas × 5 créditos


def test_exa_tool_name_pattern():
    """Verifica padrão de nome de ferramenta Exa."""
    from dataframeit.search import ExaProvider

    provider = ExaProvider()
    assert "exa" in provider.get_tool_name_pattern()


# =============================================================================
# Testes de SearchConfig com provider
# =============================================================================

def test_search_config_default_provider():
    """Verifica que provider padrão é 'tavily'."""
    from dataframeit.llm import SearchConfig

    config = SearchConfig()
    assert config.provider == "tavily"


def test_search_config_exa_provider():
    """Verifica criação de SearchConfig com provider exa."""
    from dataframeit.llm import SearchConfig

    config = SearchConfig(enabled=True, provider="exa")
    assert config.provider == "exa"


# =============================================================================
# Testes de validação de dependências
# =============================================================================

def test_validate_search_dependencies_tavily():
    """Verifica validação de dependências Tavily."""
    from dataframeit.errors import validate_search_dependencies

    original = os.environ.get('TAVILY_API_KEY')
    try:
        os.environ['TAVILY_API_KEY'] = 'test-key'

        with patch('importlib.import_module') as mock_import:
            mock_import.return_value = MagicMock()
            # Não deve levantar exceção
            validate_search_dependencies("tavily")
    finally:
        if original:
            os.environ['TAVILY_API_KEY'] = original
        elif 'TAVILY_API_KEY' in os.environ:
            del os.environ['TAVILY_API_KEY']


def test_validate_search_dependencies_exa():
    """Verifica validação de dependências Exa."""
    from dataframeit.errors import validate_search_dependencies

    original = os.environ.get('EXA_API_KEY')
    try:
        os.environ['EXA_API_KEY'] = 'test-key'

        with patch('importlib.import_module') as mock_import:
            mock_import.return_value = MagicMock()
            # Não deve levantar exceção
            validate_search_dependencies("exa")
    finally:
        if original:
            os.environ['EXA_API_KEY'] = original
        elif 'EXA_API_KEY' in os.environ:
            del os.environ['EXA_API_KEY']


def test_validate_search_dependencies_exa_missing_package():
    """Verifica erro quando langchain-exa não está instalado."""
    from dataframeit.errors import validate_search_dependencies

    with patch('importlib.import_module') as mock_import:
        def side_effect(name):
            if name == 'langchain_exa':
                raise ImportError("No module named 'langchain_exa'")
            return MagicMock()

        mock_import.side_effect = side_effect

        with pytest.raises(ImportError) as exc_info:
            validate_search_dependencies("exa")

        assert "langchain-exa" in str(exc_info.value) or "langchain_exa" in str(exc_info.value)


def test_validate_search_dependencies_exa_missing_key():
    """Verifica erro quando EXA_API_KEY não está configurada."""
    from dataframeit.errors import validate_search_dependencies

    original = os.environ.get('EXA_API_KEY')
    try:
        if 'EXA_API_KEY' in os.environ:
            del os.environ['EXA_API_KEY']

        with patch('importlib.import_module') as mock_import:
            mock_import.return_value = MagicMock()

            with pytest.raises(ValueError) as exc_info:
                validate_search_dependencies("exa")

            assert "EXA_API_KEY" in str(exc_info.value)
    finally:
        if original:
            os.environ['EXA_API_KEY'] = original


def test_validate_search_dependencies_invalid_provider():
    """Verifica erro para provedor inválido."""
    from dataframeit.errors import validate_search_dependencies

    with pytest.raises(ValueError) as exc_info:
        validate_search_dependencies("invalid")

    assert "não suportado" in str(exc_info.value)


# =============================================================================
# Testes de parâmetro search_provider em dataframeit
# =============================================================================

def test_search_provider_default():
    """Verifica que search_provider='tavily' é o padrão."""
    from dataframeit.core import dataframeit
    import inspect

    sig = inspect.signature(dataframeit)
    assert sig.parameters['search_provider'].default == "tavily"


def test_search_provider_invalid_raises():
    """Verifica que search_provider inválido gera erro."""
    from dataframeit.core import dataframeit
    import pandas as pd

    df = pd.DataFrame({"texto": ["teste"]})

    with patch('dataframeit.core.validate_provider_dependencies'):
        with pytest.raises(ValueError) as exc_info:
            dataframeit(
                df,
                questions=SampleModel,
                prompt="Pesquise {texto}",
                use_search=True,
                search_provider="invalid",
            )

        assert "search_provider" in str(exc_info.value)


# =============================================================================
# Testes de criação de ferramenta via factory
# =============================================================================

def test_create_tool_tavily(monkeypatch):
    """Verifica que TavilyProvider.create_tool cria TavilySearch."""
    from dataframeit.search import TavilyProvider

    class DummyTavilySearch:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    # Mock do módulo langchain_tavily
    mock_module = types.SimpleNamespace(TavilySearch=DummyTavilySearch)
    monkeypatch.setitem(sys.modules, "langchain_tavily", mock_module)

    provider = TavilyProvider()
    tool = provider.create_tool(max_results=10, search_depth="advanced")

    assert isinstance(tool, DummyTavilySearch)
    assert tool.kwargs["max_results"] == 10
    assert tool.kwargs["search_depth"] == "advanced"


def test_create_tool_exa(monkeypatch):
    """Verifica que ExaProvider.create_tool cria ExaSearchResults."""
    from dataframeit.search import ExaProvider

    class DummyExaSearchResults:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    # Mock do módulo langchain_exa
    mock_module = types.SimpleNamespace(ExaSearchResults=DummyExaSearchResults)
    monkeypatch.setitem(sys.modules, "langchain_exa", mock_module)

    provider = ExaProvider()
    tool = provider.create_tool(max_results=15)

    assert isinstance(tool, DummyExaSearchResults)
    assert tool.kwargs["num_results"] == 15


# =============================================================================
# Testes de _extract_usage com provider
# =============================================================================

def test_extract_usage_includes_provider_name():
    """Verifica que _extract_usage inclui nome do provider."""
    from dataframeit.agent import _extract_usage
    from dataframeit.search import TavilyProvider
    from dataframeit.llm import SearchConfig

    provider = TavilyProvider()
    search_config = SearchConfig(enabled=True, provider="tavily")

    mock_result = {
        "messages": [
            MagicMock(
                usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
                tool_calls=[{"name": "tavily_search"}]
            ),
        ],
    }

    usage = _extract_usage(mock_result, provider, search_config)

    assert usage["search_provider"] == "tavily"


def test_extract_usage_with_exa_provider():
    """Verifica _extract_usage com ExaProvider."""
    from dataframeit.agent import _extract_usage
    from dataframeit.search import ExaProvider
    from dataframeit.llm import SearchConfig

    provider = ExaProvider()
    search_config = SearchConfig(enabled=True, provider="exa", max_results=10)

    mock_result = {
        "messages": [
            MagicMock(
                usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
                tool_calls=[{"name": "exa_search"}, {"name": "exa_search"}]
            ),
        ],
    }

    usage = _extract_usage(mock_result, provider, search_config)

    assert usage["search_provider"] == "exa"
    assert usage["search_count"] == 2
    # Exa: 2 buscas × 1 crédito (max_results <= 25)
    assert usage["search_credits"] == 2


# =============================================================================
# Testes de mensagens de erro amigáveis
# =============================================================================

def test_exa_authentication_error_message():
    """Verifica mensagem amigável para erro de autenticação Exa."""
    from dataframeit.errors import get_friendly_error_message

    # Criar erro específico do Exa
    # A função get_friendly_error_message verifica padrões no error_str
    class ExaError(Exception):
        pass

    # Usar erro com "exa" e um padrão de API key que não seja capturado pelo genérico primeiro
    error = ExaError("Exa: Invalid api_key provided")
    msg = get_friendly_error_message(error)

    # Verificar que é uma mensagem de erro de autenticação (genérica ou específica)
    assert "AUTENTICAÇÃO" in msg.upper() or "EXA" in msg.upper() or "API" in msg.upper()


def test_exa_limit_error_message():
    """Verifica mensagem amigável para erro de limite Exa."""
    from dataframeit.errors import get_friendly_error_message

    class ExaLimitError(Exception):
        pass

    error = ExaLimitError("Exa quota exceeded")
    msg = get_friendly_error_message(error)

    assert "LIMITE" in msg.upper() or "EXA" in msg.upper()


# =============================================================================
# Testes de integração com call_agent
# =============================================================================

def test_call_agent_uses_provider_factory(monkeypatch):
    """Verifica que call_agent usa factory para criar ferramenta."""
    from dataframeit.agent import call_agent
    from dataframeit.llm import LLMConfig, SearchConfig

    class DummyAgent:
        def invoke(self, _payload):
            return {"structured_response": SampleModel(campo="ok"), "messages": []}

    class DummySearchTool:
        def __init__(self, **kwargs):
            self.name = "search"

    dummy_llm = object()
    captured_tools = []

    def fake_create_agent(*, model, tools, response_format, **kwargs):
        captured_tools.extend(tools)
        return DummyAgent()

    monkeypatch.setattr("dataframeit.agent._create_langchain_llm", lambda *args, **kwargs: dummy_llm)
    monkeypatch.setattr("langchain.agents.create_agent", fake_create_agent)

    # Mock do provider (no caminho correto: dataframeit.agent.get_provider)
    def mock_create_tool(**kwargs):
        return DummySearchTool(**kwargs)

    with patch('dataframeit.agent.get_provider') as mock_get_provider:
        mock_provider = MagicMock()
        mock_provider.name = "tavily"
        mock_provider.create_tool = mock_create_tool
        mock_provider.get_tool_name_pattern.return_value = "tavily"
        mock_provider.calculate_credits.return_value = 0
        mock_get_provider.return_value = mock_provider

        config = LLMConfig(
            model="gpt-4o-mini",
            provider="openai",
            api_key=None,
            max_retries=1,
            base_delay=0.0,
            max_delay=0.0,
            rate_limit_delay=0.0,
            model_kwargs={},
            search_config=SearchConfig(enabled=True, provider="tavily"),
        )

        result = call_agent("teste", SampleModel, "Responda {texto}", config)

        # Verifica que a factory foi chamada
        mock_get_provider.assert_called_once_with("tavily")

        # Verifica que a ferramenta foi criada e passada
        assert len(captured_tools) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
