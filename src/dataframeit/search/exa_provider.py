"""Provedor de busca Exa.

Exa é um motor de busca semântico com:
- $0.005 por busca (1-25 resultados)
- $0.025 por busca (26-100 resultados)
- Busca semântica avançada com embeddings

Recomendado para:
- Alto volume (>2667 buscas/mês com 1-25 resultados)
- Busca semântica mais precisa
- Quando não precisa de mais de 25 resultados
"""

from typing import Any

from .base import SearchProvider, register_provider


@register_provider
class ExaProvider(SearchProvider):
    """Implementação do provedor de busca Exa."""

    @property
    def name(self) -> str:
        return "exa"

    @property
    def env_var(self) -> str:
        return "EXA_API_KEY"

    @property
    def package_name(self) -> str:
        return "langchain_exa"

    @property
    def install_name(self) -> str:
        return "langchain-exa"

    @property
    def signup_url(self) -> str:
        return "https://exa.ai"

    def create_tool(self, max_results: int, **kwargs) -> Any:
        """Cria ferramenta ExaSearchResults.

        Args:
            max_results: Número de resultados por busca.

        Returns:
            Instância de ExaSearchResults configurada.
        """
        from langchain_exa import ExaSearchResults

        return ExaSearchResults(
            num_results=max_results,
            text_contents_options={"max_characters": 1000},
        )

    def calculate_credits(self, search_count: int, max_results: int = 5, **kwargs) -> int:
        """Calcula créditos Exa consumidos.

        Exa cobra por busca, com preço variando pelo número de resultados:
        - 1-25 resultados: $0.005 (representamos como 1 crédito)
        - 26-100 resultados: $0.025 (representamos como 5 créditos)

        Args:
            search_count: Número de buscas realizadas.
            max_results: Número de resultados por busca.

        Returns:
            Total de créditos consumidos (1 crédito = $0.005).
        """
        # Representamos em unidades de $0.005 para facilitar comparação
        cost_per_search = 1 if max_results <= 25 else 5
        return search_count * cost_per_search

    def get_tool_name_pattern(self) -> str:
        """Padrão para identificar tool calls Exa."""
        return "exa"
