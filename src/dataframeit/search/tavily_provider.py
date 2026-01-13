"""Provedor de busca Tavily.

Tavily é um motor de busca otimizado para IA com:
- 1000 créditos gratuitos/mês
- $0.008 por crédito após isso
- Suporte a search_depth: "basic" (1 crédito) ou "advanced" (2 créditos)

Recomendado para:
- Volume baixo-médio (<2667 buscas/mês)
- Quando precisa de mais de 25 resultados por busca
"""

from typing import Any

from .base import SearchProvider, register_provider


@register_provider
class TavilyProvider(SearchProvider):
    """Implementação do provedor de busca Tavily."""

    @property
    def name(self) -> str:
        return "tavily"

    @property
    def env_var(self) -> str:
        return "TAVILY_API_KEY"

    @property
    def package_name(self) -> str:
        return "langchain_tavily"

    @property
    def install_name(self) -> str:
        return "langchain-tavily"

    @property
    def signup_url(self) -> str:
        return "https://app.tavily.com"

    def create_tool(self, max_results: int, search_depth: str = "basic", **kwargs) -> Any:
        """Cria ferramenta TavilySearch.

        Args:
            max_results: Número de resultados (1-20).
            search_depth: "basic" (1 crédito) ou "advanced" (2 créditos).

        Returns:
            Instância de TavilySearch configurada.
        """
        from langchain_tavily import TavilySearch

        return TavilySearch(
            max_results=max_results,
            search_depth=search_depth,
            include_raw_content=False,
            include_answer=False,
        )

    def calculate_credits(self, search_count: int, search_depth: str = "basic", **kwargs) -> int:
        """Calcula créditos Tavily consumidos.

        Args:
            search_count: Número de buscas realizadas.
            search_depth: "basic" (1 crédito) ou "advanced" (2 créditos).

        Returns:
            Total de créditos consumidos.
        """
        depth_cost = 2 if search_depth == "advanced" else 1
        return search_count * depth_cost

    def get_tool_name_pattern(self) -> str:
        """Padrão para identificar tool calls Tavily."""
        return "tavily"
