"""Interface abstrata para provedores de busca web."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from typing_extensions import override

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool


class SearchToolError(RuntimeError):
    """Falha do provedor de busca ao executar a ferramenta, e não do provider de LLM.

    A mensagem dos clientes do Tavily e do Exa não nomeia o serviço ("Error 401:
    ..."), e por isso a classe leva o nome do provedor que falhou: é por ele que
    a mensagem amigável aponta a chave e o plano da busca. Os dois clientes só
    põem o status HTTP no texto, e cada provedor o extrai para `status_code`, de
    onde a classificação o lê como o de qualquer erro HTTP. O erro original fica
    em `error` e em `__cause__`, e o tipo dele abre a mensagem.
    """

    def __init__(self, provider: str, error: Exception, status_code: int | None = None) -> None:
        """Guarda o provedor de busca ('tavily', 'exa'), o erro do cliente e o status."""
        # Todos os argumentos em args, para que pickle e copy reconstruam o erro.
        super().__init__(provider, error, status_code)
        self.provider = provider
        self.error = error
        self.status_code = status_code

    @override
    def __str__(self) -> str:
        """Tipo e mensagem do erro do cliente de busca."""
        return f"{type(self.error).__name__}: {self.error}"


def search_failure(
    error: Exception,
    status: int | None,
    *,
    provider: str,
    usage_limit_statuses: tuple[int, ...],
    usage_limit_message: str,
) -> Exception:
    """Exceção a levantar pela falha do cliente de busca.

    O status de limite de uso, que só crédito na conta resolve, dá
    ProviderUsageLimitError, que interrompe a execução; os demais, SearchToolError.
    """
    if status in usage_limit_statuses:
        # Import local: errors importa search no topo do módulo.
        from dataframeit.errors import ProviderUsageLimitError  # noqa: PLC0415

        return ProviderUsageLimitError(
            f"{usage_limit_message} ({type(error).__name__}: {error}); para seguir sem esta "
            "busca, use use_search=False ou outro search_provider"
        )
    return SearchToolError(provider=provider, error=error, status_code=status)


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
