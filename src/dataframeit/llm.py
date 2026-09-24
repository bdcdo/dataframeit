import json
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from .errors import ProviderRejectedOutputError, retry_with_backoff
from .utils import check_dependency


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

    prompt = build_prompt(user_prompt, text)
    # Estado entre tentativas: a tentativa seguinte a uma resposta inválida leva
    # ao modelo a resposta e o erro, porque repetir o mesmo prompt tende a
    # repetir o mesmo erro. O uso soma todas as tentativas, que são cobradas
    # mesmo quando a resposta é recusada.
    feedback: list = []
    usage_total: dict[str, int] = {}

    def _call():
        result = structured_llm.invoke([('human', prompt), *feedback] if feedback else prompt)

        raw_message = result.get('raw')
        if raw_message is not None and getattr(raw_message, 'usage_metadata', None):
            for key, value in _parse_usage_metadata(raw_message.usage_metadata).items():
                usage_total[key] = usage_total.get(key, 0) + (value or 0)

        if result.get('parsing_error'):
            _request_correction(pydantic_model, raw_message, result['parsing_error'], feedback)
        parsed = result.get('parsed')
        if parsed is None:
            raise ValueError("Structured output retornou None")

        return {'data': parsed.model_dump(), 'usage': dict(usage_total) or None}

    return retry_with_backoff(_call, config.max_retries, config.base_delay, config.max_delay)


def _raw_payload(raw_message) -> tuple[dict | None, str]:
    """Resposta bruta do modelo como dict, quando dá, e como texto.

    O structured output chega por tool call (function calling) ou no conteúdo
    da mensagem (json_schema), em texto ou em blocos, conforme o provider.
    """
    if raw_message is None:
        return None, ''
    for call in getattr(raw_message, 'tool_calls', None) or []:
        if isinstance(call.get('args'), dict):
            return call['args'], json.dumps(call['args'], ensure_ascii=False)
    content = getattr(raw_message, 'content', '')
    if isinstance(content, list):
        content = ''.join(
            block.get('text', '') if isinstance(block, dict) else str(block) for block in content
        )
    text = str(content or '')
    try:
        payload = json.loads(text)
    except ValueError:
        return None, text
    return (payload if isinstance(payload, dict) else None), text


def _format_validation_error(error) -> str:
    """Uma linha por erro, com o caminho do campo; texto bruto se não for ValidationError."""
    if not isinstance(error, ValidationError):
        return str(error)[:2000]
    lines = []
    for detail in error.errors()[:20]:
        location = '.'.join(str(part) for part in detail.get('loc', ())) or '(modelo)'
        lines.append(f"- {location}: {detail.get('msg', '')}")
    return '\n'.join(lines)


def _request_correction(pydantic_model, raw_message, parsing_error, feedback: list):
    """Grava em `feedback` a resposta recusada e o erro, e levanta ProviderRejectedOutputError.

    O retry repete a chamada, e a tentativa seguinte leva os dois ao modelo. O
    `parsing_error` do LangChain pode vir como exceção do parser que embrulha a
    validação; nesse caso, validar a resposta bruta recupera o erro por campo.
    """
    payload, raw_text = _raw_payload(raw_message)
    error = parsing_error
    if not isinstance(error, ValidationError) and payload is not None:
        try:
            pydantic_model.model_validate(payload)
        except ValidationError as validation_error:
            error = validation_error
    feedback[:] = [
        ('ai', raw_text or '(resposta vazia)'),
        ('human', _CORRECTION_REQUEST.format(errors=_format_validation_error(error))),
    ]
    raise ProviderRejectedOutputError(f"Falha no parsing do structured output: {parsing_error}")


_CORRECTION_REQUEST = (
    "A resposta anterior não passou na validação do esquema:\n{errors}\n"
    "Responda de novo ao pedido original, com a resposta completa, corrigindo esses pontos."
)


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
