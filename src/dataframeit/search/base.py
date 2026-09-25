"""Interface abstrata para provedores de busca web."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool


class SearchProvider(ABC):
    """Interface abstrata para provedores de busca web.

    Define o contrato que todos os provedores de busca devem implementar,
    permitindo trocar entre Tavily, Exa e outros provedores de forma transparente.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Nome identificador do provedor (ex: 'tavily', 'exa')."""

    @property
    @abstractmethod
    def env_var(self) -> str:
        """Nome da variável de ambiente para a API key."""

    @property
    @abstractmethod
    def package_name(self) -> str:
        """Nome do pacote Python/LangChain (ex: 'langchain_tavily')."""

    @property
    @abstractmethod
    def install_name(self) -> str:
        """Nome do pacote para pip install (ex: 'langchain-tavily')."""

    @property
    @abstractmethod
    def signup_url(self) -> str:
        """URL para criar conta e obter API key."""

    @property
    @abstractmethod
    def friendly_name(self) -> str:
        """Nome exibido nas mensagens de erro (ex: 'Tavily Search')."""

    @property
    @abstractmethod
    def free_tier(self) -> str:
        """Descrição curta do plano de entrada, exibida na mensagem de API key ausente."""

    @property
    @abstractmethod
    def requests_per_minute(self) -> int:
        """Limite aproximado de requisições por minuto, usado no aviso de rate limit."""

    @abstractmethod
    def create_tool(self, max_results: int, **kwargs: Any) -> BaseTool:
        """Cria a ferramenta de busca do LangChain.

        Args:
            max_results: Número máximo de resultados por busca.
            **kwargs: Parâmetros específicos do provedor.

        Returns:
            Instância da ferramenta de busca configurada.
        """

    @abstractmethod
    def calculate_credits(self, search_count: int, **kwargs: Any) -> int:
        """Calcula créditos/custos consumidos.

        Args:
            search_count: Número de buscas realizadas.
            **kwargs: Parâmetros específicos do provedor (ex: search_depth).

        Returns:
            Número de créditos consumidos (para rastreamento de custos).
        """


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
    from . import (  # noqa: F401, PLC0415 (registra os provedores; no topo seria import circular)
        exa_provider,
        tavily_provider,
    )

    if name not in _PROVIDERS:
        available = list(_PROVIDERS.keys())
        msg = f"Provedor de busca '{name}' não suportado. Provedores disponíveis: {available}"
        raise ValueError(msg)
    return _PROVIDERS[name]()


def get_available_providers() -> list[str]:
    """Retorna lista de provedores de busca disponíveis."""
    # Importar providers para garantir que estão registrados
    from . import (  # noqa: F401, PLC0415 (registra os provedores; no topo seria import circular)
        exa_provider,
        tavily_provider,
    )

    return list(_PROVIDERS.keys())
