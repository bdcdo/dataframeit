"""Interface abstrata para provedores de busca web."""

from abc import ABC, abstractmethod
from typing import Any


class SearchProvider(ABC):
    """Interface abstrata para provedores de busca web.

    Define o contrato que todos os provedores de busca devem implementar,
    permitindo trocar entre Tavily, Exa e outros provedores de forma transparente.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Nome identificador do provedor (ex: 'tavily', 'exa')."""
        pass

    @property
    @abstractmethod
    def env_var(self) -> str:
        """Nome da variável de ambiente para a API key."""
        pass

    @property
    @abstractmethod
    def package_name(self) -> str:
        """Nome do pacote Python/LangChain (ex: 'langchain_tavily')."""
        pass

    @property
    @abstractmethod
    def install_name(self) -> str:
        """Nome do pacote para pip install (ex: 'langchain-tavily')."""
        pass

    @property
    @abstractmethod
    def signup_url(self) -> str:
        """URL para criar conta e obter API key."""
        pass

    @abstractmethod
    def create_tool(self, max_results: int, **kwargs) -> Any:
        """Cria a ferramenta de busca do LangChain.

        Args:
            max_results: Número máximo de resultados por busca.
            **kwargs: Parâmetros específicos do provedor.

        Returns:
            Instância da ferramenta de busca configurada.
        """
        pass

    @abstractmethod
    def calculate_credits(self, search_count: int, **kwargs) -> int:
        """Calcula créditos/custos consumidos.

        Args:
            search_count: Número de buscas realizadas.
            **kwargs: Parâmetros específicos do provedor (ex: search_depth).

        Returns:
            Número de créditos consumidos (para rastreamento de custos).
        """
        pass

    @abstractmethod
    def get_tool_name_pattern(self) -> str:
        """Retorna padrão para identificar tool calls deste provedor.

        Returns:
            String que aparece no nome da ferramenta nas mensagens do agente.
        """
        pass


# Registry de provedores disponíveis
_PROVIDERS: dict[str, type[SearchProvider]] = {}


def register_provider(cls: type[SearchProvider]) -> type[SearchProvider]:
    """Decorator para registrar um provedor de busca."""
    # Instanciar para obter o nome
    instance = cls()
    _PROVIDERS[instance.name] = cls
    return cls


def get_provider(name: str) -> SearchProvider:
    """Factory para obter instância de provedor de busca.

    Args:
        name: Nome do provedor ('tavily' ou 'exa').

    Returns:
        Instância do provedor de busca.

    Raises:
        ValueError: Se o provedor não for suportado.
    """
    # Importar providers para garantir que estão registrados
    from . import tavily_provider, exa_provider  # noqa: F401

    if name not in _PROVIDERS:
        available = list(_PROVIDERS.keys())
        raise ValueError(
            f"Provedor de busca '{name}' não suportado. "
            f"Provedores disponíveis: {available}"
        )
    return _PROVIDERS[name]()


def get_available_providers() -> list[str]:
    """Retorna lista de provedores de busca disponíveis."""
    # Importar providers para garantir que estão registrados
    from . import tavily_provider, exa_provider  # noqa: F401

    return list(_PROVIDERS.keys())
