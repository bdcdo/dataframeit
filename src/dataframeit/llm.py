from dataclasses import dataclass
from typing import Optional, Any
from .utils import check_dependency, parse_json, retry_with_backoff

def _extract_response_text(response: Any) -> str:
    """Extrai texto útil do objeto de resposta da API Responses."""
    # Handler direto para propriedade auxiliar introduzida pelo SDK
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text

    # Tentar percorrer a lista `output`
    output = getattr(response, "output", None)
    if output:
        texts = []
        for block in output:
            # Objetos do SDK expõem atributos e/ou model_dump()
            content = getattr(block, "content", None)
            if content is None and hasattr(block, "model_dump"):
                content = block.model_dump().get("content")

            if not content:
                continue

            for item in content:
                text = getattr(item, "text", None)
                if text is None and hasattr(item, "model_dump"):
                    text = item.model_dump().get("text")
                if text:
                    texts.append(text)

        if texts:
            return "\n".join(texts)

    # Fallback: delegar para representação em string (último recurso)
    return str(response)



@dataclass
class LLMConfig:
    """Configuração para chamadas de LLM."""
    model: str
    provider: str
    use_openai: bool
    api_key: Optional[str]
    openai_client: Optional[Any]
    reasoning_effort: str
    verbosity: str
    max_retries: int
    base_delay: float
    max_delay: float
    placeholder: str
    rate_limit_delay: float


def build_prompt(user_prompt: str, text: str, placeholder: str) -> str:
    """Substitui placeholder pelo texto.

    Format instructions não são mais necessárias pois with_structured_output
    gerencia o schema automaticamente.

    Args:
        user_prompt: Template do prompt do usuário.
        text: Texto a ser processado.
        placeholder: Nome do placeholder no template.

    Returns:
        Prompt formatado pronto para envio ao LLM.

    Raises:
        ValueError: Se o placeholder não for encontrado no template.
    """
    import warnings

    placeholder_tag = f"{{{placeholder}}}"

    # Validar presença do placeholder no template
    if placeholder_tag not in user_prompt:
        raise ValueError(
            f"Placeholder '{placeholder_tag}' não encontrado no template do prompt. "
            f"Adicione '{placeholder_tag}' ao seu template para indicar onde o texto será inserido. "
            f"Exemplo: 'Analise o seguinte texto: {placeholder_tag}'"
        )

    # Warning de deprecation se {format} estiver presente
    if '{format}' in user_prompt:
        warnings.warn(
            f"O placeholder {{format}} está deprecated e será ignorado. "
            f"O método with_structured_output() agora gerencia o schema automaticamente. "
            f"Remova '{{format}}' do seu template para evitar este aviso.",
            DeprecationWarning,
            stacklevel=2
        )

    return user_prompt.replace(placeholder_tag, text)


def call_openai(text: str, pydantic_model, user_prompt: str, config: LLMConfig) -> dict:
    """Processa texto usando OpenAI API.

    Args:
        text: Texto a ser processado.
        pydantic_model: Modelo Pydantic para estruturar resposta.
        user_prompt: Template do prompt do usuário.
        config: Configuração do LLM.

    Returns:
        Dicionário com 'data' (dados extraídos) e 'usage' (metadata de uso de tokens).
    """
    check_dependency("openai", "openai")

    # Criar ou usar cliente fornecido
    if config.openai_client is None:
        from openai import OpenAI
        client = OpenAI(api_key=config.api_key) if config.api_key else OpenAI()
    else:
        client = config.openai_client

    def _call():
        prompt = build_prompt(user_prompt, text, config.placeholder)

        request_kwargs = {
            "model": config.model,
            "input": prompt,
        }
        if config.reasoning_effort:
            request_kwargs["reasoning"] = {"effort": config.reasoning_effort}
        if config.verbosity:
            request_kwargs["text"] = {"verbosity": config.verbosity}

        response = client.responses.create(**request_kwargs)

        # Extrair dados e usage metadata (Responses API)
        data = parse_json(_extract_response_text(response))
        usage = None
        if getattr(response, "usage", None):
            u = response.usage
            input_tokens = getattr(u, "input_tokens", None)
            output_tokens = getattr(u, "output_tokens", None)
            total_tokens = getattr(u, "total_tokens", None)
            if any(v is not None for v in (input_tokens, output_tokens, total_tokens)):
                usage = {
                    "input_tokens": input_tokens or 0,
                    "output_tokens": output_tokens or 0,
                    "total_tokens": total_tokens or 0,
                }

        return {'data': data, 'usage': usage}

    return retry_with_backoff(_call, config.max_retries, config.base_delay, config.max_delay)


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
    llm = _create_langchain_llm(config.model, config.provider, config.api_key)

    # Usar with_structured_output com include_raw=True para manter usage_metadata
    # method="json_schema" é o padrão e mais confiável
    structured_llm = llm.with_structured_output(pydantic_model, include_raw=True)

    def _call():
        prompt = build_prompt(user_prompt, text, config.placeholder)
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


def _create_langchain_llm(model: str, provider: str, api_key: Optional[str]):
    """Cria instância de LLM do LangChain baseado no provider.

    Args:
        model: Nome do modelo.
        provider: Nome do provider ('google_genai', etc).
        api_key: Chave de API (opcional).

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

    model_kwargs = {"model_provider": provider, "temperature": 0}
    if api_key:
        model_kwargs["api_key"] = api_key

    return init_chat_model(model, **model_kwargs)
