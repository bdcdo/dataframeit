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

# Imports opcionais
try:
    from langchain_core.output_parsers import PydanticOutputParser
except ImportError:
    try:
        from langchain.output_parsers import PydanticOutputParser
    except ImportError:
        PydanticOutputParser = None


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


def build_prompt(pydantic_model, user_prompt: str, text: str, placeholder: str) -> str:
    """Constrói prompt completo com instruções de formatação.

    Args:
        pydantic_model: Modelo Pydantic para estruturar resposta.
        user_prompt: Template do prompt do usuário.
        text: Texto a ser processado.
        placeholder: Nome do placeholder no template.

    Returns:
        Prompt formatado pronto para envio ao LLM.
    """
    if PydanticOutputParser is None:
        check_dependency("langchain", "langchain")

    parser = PydanticOutputParser(pydantic_object=pydantic_model)
    format_instructions = parser.get_format_instructions()

    # Adicionar instruções de formatação
    full_template = f"{user_prompt}\n\n{format_instructions}"

    # Substituir placeholder pelo texto (evita KeyError com {format})
    return full_template.replace(f"{{{placeholder}}}", text)


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
        prompt = build_prompt(pydantic_model, user_prompt, text, config.placeholder)

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
    """Processa texto usando LangChain.

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

    # Criar LLM baseado no provider
    llm = _create_langchain_llm(config.model, config.provider, config.api_key)

    def _call():
        prompt = build_prompt(pydantic_model, user_prompt, text, config.placeholder)
        response = llm.invoke(prompt)

        # Extrair dados e usage metadata
        data = parse_json(response)
        usage = None
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage = {
                'input_tokens': response.usage_metadata.get('input_tokens', 0),
                'output_tokens': response.usage_metadata.get('output_tokens', 0),
                'total_tokens': response.usage_metadata.get('total_tokens', 0)
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
