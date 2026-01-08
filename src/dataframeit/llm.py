from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from .utils import check_dependency
from .errors import retry_with_backoff


@dataclass
class SearchConfig:
    """Configuração para busca web via Tavily."""
    enabled: bool = False
    per_field: bool = False  # Um agente por campo
    max_results: int = 5
    max_searches: int = 3
    search_depth: str = "basic"  # "basic" ou "advanced"


@dataclass
class LLMConfig:
    """Configuração para chamadas de LLM."""
    model: str
    provider: str
    api_key: Optional[str]
    max_retries: int
    base_delay: float
    max_delay: float
    rate_limit_delay: float
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    search_config: Optional[SearchConfig] = None


def build_prompt(user_prompt: str, text: str) -> str:
    """Substitui {texto} pelo texto a ser analisado.

    Args:
        user_prompt: Template do prompt (já com {texto} incluído).
        text: Texto a ser processado.

    Returns:
        Prompt formatado pronto para envio ao LLM.
    """
    return user_prompt.replace('{texto}', text)


def call_langchain(text: str, pydantic_model, user_prompt: str, config: LLMConfig) -> dict:
    """Processa texto usando LangChain com structured output.

    Args:
        text: Texto a ser processado.
        pydantic_model: Modelo Pydantic para estruturar resposta.
        user_prompt: Template do prompt do usuário.
        config: Configuração do LLM.

    Returns:
        Dicionário com 'data' (dados extraídos) e 'usage' (metadata de uso de tokens).
    """
    check_dependency("langchain", "langchain")
    check_dependency("langchain_core", "langchain-core")

    # Criar LLM base
    llm = _create_langchain_llm(config.model, config.provider, config.api_key, config.model_kwargs)

    # Usar with_structured_output com include_raw=True para manter usage_metadata
    # method="json_schema" é o padrão e mais confiável
    structured_llm = llm.with_structured_output(pydantic_model, include_raw=True)

    def _call():
        prompt = build_prompt(user_prompt, text)
        result = structured_llm.invoke(prompt)

        # Verificar erros de parsing
        if result.get('parsing_error'):
            raise ValueError(f"Falha no parsing do structured output: {result['parsing_error']}")

        # Extrair instância Pydantic parseada e converter para dict
        parsed = result.get('parsed')
        if parsed is None:
            raise ValueError("Structured output retornou None")

        data = parsed.model_dump()

        # Extrair usage_metadata do raw AIMessage
        # Nota: Para Google GenAI, tokens estão em usage_metadata, não response_metadata
        usage = None
        raw_message = result.get('raw')
        if raw_message and hasattr(raw_message, 'usage_metadata') and raw_message.usage_metadata:
            usage = {
                'input_tokens': raw_message.usage_metadata.get('input_tokens', 0),
                'output_tokens': raw_message.usage_metadata.get('output_tokens', 0),
                'total_tokens': raw_message.usage_metadata.get('total_tokens', 0)
            }

        return {'data': data, 'usage': usage}

    return retry_with_backoff(_call, config.max_retries, config.base_delay, config.max_delay)


def _create_langchain_llm(model: str, provider: str, api_key: Optional[str], extra_kwargs: Optional[Dict[str, Any]] = None):
    """Cria instância de LLM do LangChain baseado no provider.

    Args:
        model: Nome do modelo.
        provider: Nome do provider ('google_genai', etc).
        api_key: Chave de API (opcional).
        extra_kwargs: Parâmetros extras para o modelo (reasoning_effort, use_responses_api, etc).

    Returns:
        Instância do LLM configurado.
    """
    try:
        from langchain.chat_models import init_chat_model
    except ImportError:
        try:
            from langchain_core.chat_models import init_chat_model
        except ImportError:
            raise ImportError("LangChain não está disponível. Instale com: pip install langchain langchain-core")

    kwargs = {"model_provider": provider, "temperature": 0}
    if api_key:
        kwargs["api_key"] = api_key

    # Adicionar parâmetros extras do usuário
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    return init_chat_model(model, **kwargs)
