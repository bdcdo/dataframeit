"""Testes para funcionalidade de busca web via Tavily."""

import pandas as pd
from pydantic import BaseModel, Field
from typing import Literal
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
# Testes de warning de rate limit (Issue #67)
# =============================================================================

def test_warn_search_rate_limit_triggers_on_high_concurrent():
    """Testa que warning é emitido quando queries concorrentes excedem limite."""
    from dataframeit.core import _warn_search_rate_limit

    with pytest.warns(UserWarning) as record:
        _warn_search_rate_limit(
            num_rows=100,
            num_fields=4,
            parallel_requests=20,
            search_per_field=True,
            rate_limit_delay=0.0,
        )

    assert len(record) == 1
    warning_msg = str(record[0].message)
    assert "rate limit" in warning_msg.lower()
    assert "80" in warning_msg  # 20 * 4 = 80 concurrent queries


def test_warn_search_rate_limit_triggers_on_high_rpm():
    """Testa que warning é emitido quando taxa estimada excede limite Tavily."""
    from dataframeit.core import _warn_search_rate_limit

    with pytest.warns(UserWarning) as record:
        _warn_search_rate_limit(
            num_rows=1000,
            num_fields=1,
            parallel_requests=10,
            search_per_field=False,
            rate_limit_delay=0.0,
        )

    assert len(record) == 1
    warning_msg = str(record[0].message)
    assert "queries/min" in warning_msg.lower() or "taxa estimada" in warning_msg.lower()


def test_warn_search_rate_limit_no_warning_safe_config():
    """Testa que nenhum warning é emitido com configuração segura."""
    from dataframeit.core import _warn_search_rate_limit
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Configuração muito conservadora:
        # - 2 workers * 1 = 2 queries concorrentes (< 10)
        # - Com delay de 2.0s: (60/2.0) * 2 = 60 req/min (< 80)
        _warn_search_rate_limit(
            num_rows=10,
            num_fields=1,
            parallel_requests=2,
            search_per_field=False,
            rate_limit_delay=2.0,
        )

    # Nenhum warning deve ser emitido (queries concorrentes baixo e taxa < 80% do limite)
    rate_limit_warnings = [x for x in w if "rate limit" in str(x.message).lower()]
    assert len(rate_limit_warnings) == 0


def test_warn_search_rate_limit_includes_recommendations():
    """Testa que warning inclui recomendações de parallel_requests e rate_limit_delay."""
    from dataframeit.core import _warn_search_rate_limit

    with pytest.warns(UserWarning) as record:
        _warn_search_rate_limit(
            num_rows=100,
            num_fields=4,
            parallel_requests=20,
            search_per_field=True,
            rate_limit_delay=0.0,
        )

    warning_msg = str(record[0].message)
    assert "parallel_requests=" in warning_msg
    assert "rate_limit_delay=" in warning_msg


def test_warn_search_rate_limit_respects_existing_delay():
    """Testa que warning considera rate_limit_delay existente no cálculo."""
    from dataframeit.core import _warn_search_rate_limit
    import warnings

    # Com alto delay, a taxa estimada é menor
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _warn_search_rate_limit(
            num_rows=100,
            num_fields=1,
            parallel_requests=5,
            search_per_field=False,
            rate_limit_delay=2.0,  # Alto delay = ~30 req/min
        )

    # Com 5 workers e 2s delay = ~150 req/min (5 * 60/2)
    # Isso está acima de 80% de 100, então pode gerar warning
    # mas com baixo concurrent (5), não deve
    rate_limit_warnings = [x for x in w if "concurrent" in str(x.message).lower()]
    # Não deve ter warning de queries concorrentes pois 5 < 10
    assert len(rate_limit_warnings) == 0


def test_dataframeit_calls_warn_with_search_and_parallel():
    """Testa que dataframeit chama warning quando use_search=True e parallel_requests>1."""
    from dataframeit.core import dataframeit

    df = pd.DataFrame({"texto": ["item"] * 100})

    with patch('dataframeit.core.validate_provider_dependencies'), \
         patch('dataframeit.core.validate_search_dependencies'), \
         patch('dataframeit.core._warn_search_rate_limit') as mock_warn, \
         patch('dataframeit.core._process_rows_parallel') as mock_process:

        # Mock _process_rows_parallel para retornar sem fazer nada
        mock_process.return_value = {'total_tokens': 0}

        try:
            dataframeit(
                df,
                questions=MedicamentoInfo,
                prompt="Pesquise sobre {texto}",
                use_search=True,
                search_per_field=True,
                parallel_requests=10,
            )
        except Exception:
            pass  # Ignorar erros de processamento

        # Verificar que warning foi chamado com parâmetros corretos
        mock_warn.assert_called_once()
        call_args = mock_warn.call_args
        assert call_args.kwargs['num_rows'] == 100
        assert call_args.kwargs['num_fields'] == 2  # MedicamentoInfo tem 2 campos
        assert call_args.kwargs['parallel_requests'] == 10
        assert call_args.kwargs['search_per_field'] is True


def test_dataframeit_no_warn_without_search():
    """Testa que dataframeit não chama warning quando use_search=False."""
    from dataframeit.core import dataframeit

    df = pd.DataFrame({"texto": ["item"]})

    with patch('dataframeit.core.validate_provider_dependencies'), \
         patch('dataframeit.core._warn_search_rate_limit') as mock_warn, \
         patch('dataframeit.core._process_rows_parallel') as mock_process:

        mock_process.return_value = {'total_tokens': 0}

        try:
            dataframeit(
                df,
                questions=MedicamentoInfo,
                prompt="Analise {texto}",
                use_search=False,  # Sem busca
                parallel_requests=10,
            )
        except Exception:
            pass

        # Warning não deve ser chamado
        mock_warn.assert_not_called()


def test_dataframeit_no_warn_sequential():
    """Testa que dataframeit não chama warning com parallel_requests=1."""
    from dataframeit.core import dataframeit

    df = pd.DataFrame({"texto": ["item"]})

    with patch('dataframeit.core.validate_provider_dependencies'), \
         patch('dataframeit.core.validate_search_dependencies'), \
         patch('dataframeit.core._warn_search_rate_limit') as mock_warn, \
         patch('dataframeit.core._process_rows') as mock_process:

        mock_process.return_value = {'total_tokens': 0}

        try:
            dataframeit(
                df,
                questions=MedicamentoInfo,
                prompt="Pesquise {texto}",
                use_search=True,
                parallel_requests=1,  # Sequencial
            )
        except Exception:
            pass

        # Warning não deve ser chamado
        mock_warn.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
