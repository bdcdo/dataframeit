import threading
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, Generic, TypeVar

from .errors import retry_with_backoff
from .utils import check_dependency

T = TypeVar("T")


class _BuildOnce(Generic[T]):
    """Constrói o valor na primeira chamada e o reusa nas seguintes.

    A construção acontece na primeira linha processada, e não na entrada do
    backend: um erro de construção, como chave de API ausente, continua sendo
    erro da linha. Falha não fica em cache, então a linha seguinte tenta de
    novo. O lock impede que threads do modo paralelo construam duas vezes.
    """

    def __init__(self, build: Callable[[], T]):
        self._build = build
        self._lock = threading.Lock()
        self._built = False
        self._value: T | None = None

    def __call__(self) -> T:
        if not self._built:
            with self._lock:
                if not self._built:
                    self._value = self._build()
                    self._built = True
        return self._value


@dataclass
class SearchGroupConfig:
    """Configuração de um grupo de busca.

    Permite agrupar múltiplos campos que compartilham contexto de busca,
    reduzindo chamadas de API redundantes.

    Attributes:
        fields: Lista de nomes dos campos que pertencem a este grupo.
        prompt: Prompt customizado para o grupo. Use {query} para inserir
            o texto de busca. Se None, usa o prompt padrão.
        max_results: Número máximo de resultados por busca (1-20).
            Se None, usa o valor global.
        search_depth: Profundidade da busca ("basic" ou "advanced").
            Se None, usa o valor global.
        max_search_calls: Máximo de buscas por execução do agente do grupo.
            Se None, usa o valor global.
    """
    fields: list[str]
    prompt: str | None = None
    max_results: int | None = None
    search_depth: str | None = None
    max_search_calls: int | None = None


@dataclass
class SearchConfig:
    """Configuração para busca web.

    Suporta múltiplos provedores de busca:
    - tavily: Motor de busca otimizado para IA (default)
    - exa: Motor de busca semântico, mais econômico para alto volume
    """
    enabled: bool = False
    provider: str = "tavily"  # "tavily" ou "exa"
    per_field: bool = False  # Um agente por campo
    max_results: int = 5
    search_depth: str = "basic"  # "basic" ou "advanced" (apenas Tavily)
    # Máximo de buscas por execução do agente; ao atingi-lo, o agente responde
    # com o que já encontrou.
    max_search_calls: int = 10
    groups: dict[str, SearchGroupConfig] | None = None


@dataclass
class LLMConfig:
    """Configuração para chamadas de LLM.

    `model` é None só com providers cujo runtime escolhe o modelo (codex, claude_code).
    """
    model: str | None
    provider: str
    api_key: str | None
    max_retries: int
    base_delay: float
    max_delay: float
    rate_limit_delay: float
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    search_config: SearchConfig | None = None
    # Modelo LangChain da execução, preenchido por _provider_backend. As cópias
    # com overrides de busca por campo ou grupo herdam o mesmo objeto, então
    # todas as linhas e threads reusam um único cliente.
    shared_chat_model: _BuildOnce | None = field(default=None, repr=False, compare=False)


def build_prompt(user_prompt: str, text: str) -> str:
    """Substitui {texto} pelo texto a ser analisado.

    Args:
        user_prompt: Template do prompt (já com {texto} incluído).
        text: Texto a ser processado.

    Returns:
        Prompt formatado pronto para envio ao LLM.
    """
    return user_prompt.replace('{texto}', text)


def _parse_usage_metadata(meta) -> dict[str, int]:
    """Extrai tokens de um usage_metadata dict ou objeto.

    ``cache_read`` representa tokens lidos do cache. ``cache_creation`` não
    entra nessa métrica porque continua sendo consumo de entrada sem cache.
    """
    if isinstance(meta, dict):
        input_tokens = meta.get('input_tokens', 0)
        output_tokens = meta.get('output_tokens', 0)
        total_tokens = meta.get('total_tokens', 0)
        output_details = meta.get('output_token_details') or {}
        input_details = meta.get('input_token_details') or {}
    else:
        input_tokens = getattr(meta, 'input_tokens', 0)
        output_tokens = getattr(meta, 'output_tokens', 0)
        total_tokens = getattr(meta, 'total_tokens', 0)
        output_details = getattr(meta, 'output_token_details', None) or {}
        input_details = getattr(meta, 'input_token_details', None) or {}

    if isinstance(output_details, dict):
        reasoning_tokens = output_details.get('reasoning', 0)
    else:
        reasoning_tokens = getattr(output_details, 'reasoning', 0)

    if isinstance(input_details, dict):
        cached_input_tokens = input_details.get('cache_read', 0)
    else:
        cached_input_tokens = getattr(input_details, 'cache_read', 0)

    return {
        'input_tokens': input_tokens,
        'cached_input_tokens': cached_input_tokens,
        'output_tokens': output_tokens,
        'total_tokens': total_tokens,
        'reasoning_tokens': reasoning_tokens,
    }


def with_shared_chat_model(config: LLMConfig) -> LLMConfig:
    """Cópia da config com um único modelo LangChain para a execução inteira."""
    return replace(config, shared_chat_model=_BuildOnce(
        lambda: _create_langchain_llm(
            config.model, config.provider, config.api_key, config.model_kwargs
        )
    ))


def chat_model(config: LLMConfig):
    """Devolve o modelo LangChain da execução, ou cria um se não há compartilhado."""
    if config.shared_chat_model is not None:
        return config.shared_chat_model()
    return _create_langchain_llm(config.model, config.provider, config.api_key, config.model_kwargs)


def build_structured_llm(pydantic_model, config: LLMConfig):
    """Modelo com structured output para ``pydantic_model``.

    O método (json_schema, tool calling) é o padrão de cada integração
    LangChain. ``include_raw=True`` mantém a mensagem crua, de onde sai o
    ``usage_metadata``.
    """
    check_dependency("langchain", "langchain")
    check_dependency("langchain_core", "langchain-core")
    return chat_model(config).with_structured_output(pydantic_model, include_raw=True)


def call_langchain(
    text: str,
    pydantic_model,
    user_prompt: str,
    config: LLMConfig,
    structured_llm: _BuildOnce | None = None,
) -> dict:
    """Processa texto usando LangChain com structured output.

    Args:
        text: Texto a ser processado.
        pydantic_model: Modelo Pydantic para estruturar resposta.
        user_prompt: Template do prompt do usuário.
        config: Configuração do LLM.
        structured_llm: Modelo estruturado compartilhado pela execução. Se
            None, é construído nesta chamada.

    Returns:
        Dicionário com 'data' (dados extraídos) e 'usage' (metadata de uso de tokens).
    """
    if structured_llm is not None:
        model = structured_llm()
    else:
        model = build_structured_llm(pydantic_model, config)

    def _call():
        prompt = build_prompt(user_prompt, text)
        result = model.invoke(prompt)

        # Verificar erros de parsing
        if result.get('parsing_error'):
            raise ValueError(f"Falha no parsing do structured output: {result['parsing_error']}")

        # Extrair instância Pydantic parseada e converter para dict
        parsed = result.get('parsed')
        if parsed is None:
            raise ValueError("Structured output retornou None")

        data = parsed.model_dump()

        # Tokens de uso (suporta dict ou objeto — provedores variam)
        usage = None
        raw_message = result.get('raw')
        if raw_message and hasattr(raw_message, 'usage_metadata') and raw_message.usage_metadata:
            usage = _parse_usage_metadata(raw_message.usage_metadata)

        return {'data': data, 'usage': usage}

    return retry_with_backoff(_call, config.max_retries, config.base_delay, config.max_delay)


def _create_langchain_llm(model: str, provider: str, api_key: str | None, extra_kwargs: dict[str, Any] | None = None):
    """Cria instância de LLM do LangChain baseado no provider.

    Args:
        model: Nome do modelo.
        provider: Nome do provider ('google_genai', etc).
        api_key: Chave de API (opcional).
        extra_kwargs: Parâmetros extras para o modelo (reasoning_effort, use_responses_api, etc).

    Returns:
        Instância do LLM configurado.
    """
    from langchain.chat_models import init_chat_model

    # Nenhum parâmetro de amostragem é injetado: vários modelos rejeitam
    # `temperature` com erro 400, e a lista muda a cada lançamento. Quem quer
    # determinismo passa `temperature` em `model_kwargs`, nos modelos que aceitam.
    kwargs = {"model_provider": provider}
    if api_key:
        kwargs["api_key"] = api_key

    # Adicionar parâmetros extras do usuário
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    return init_chat_model(model, **kwargs)
