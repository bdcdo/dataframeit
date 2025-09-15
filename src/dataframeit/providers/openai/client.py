from typing import Any, Optional

# Import opcional de OpenAI
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def create_openai_client(api_key: Optional[str] = None, openai_client: Optional[Any] = None) -> Any:
    """Cria cliente OpenAI baseado na configuração.

    Args:
        api_key: Chave API específica (opcional).
        openai_client: Cliente OpenAI customizado (opcional).

    Returns:
        Any: Instância do cliente OpenAI configurado.

    Raises:
        ImportError: Se a biblioteca OpenAI não estiver instalada.
    """
    if OpenAI is None:
        raise ImportError("OpenAI não está instalado. Instale com: pip install openai")

    if openai_client:
        return openai_client
    elif api_key:
        return OpenAI(api_key=api_key)
    else:
        return OpenAI()