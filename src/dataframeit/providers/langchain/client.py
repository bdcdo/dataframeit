from typing import Any, Optional

# Imports opcionais do LangChain
try:
    from langchain.chat_models import init_chat_model
except ImportError:
    init_chat_model = None


def create_langchain_llm(model: str, provider: str, api_key: Optional[str] = None) -> Any:
    """Cria modelo LLM LangChain baseado na configuração.

    Args:
        model: Nome do modelo a ser utilizado.
        provider: Provider do LangChain.
        api_key: Chave API específica (opcional).

    Returns:
        Any: Modelo LLM configurado.

    Raises:
        ImportError: Se o LangChain não estiver disponível.
    """
    # Verificar se os imports estão disponíveis
    if init_chat_model is None:
        raise ImportError("LangChain não está disponível. Instale com: pip install langchain langchain-core")

    model_kwargs = {"model_provider": provider, "temperature": 0}
    if api_key:
        model_kwargs["api_key"] = api_key

    return init_chat_model(model, **model_kwargs)