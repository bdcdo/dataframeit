"""Provedor de busca Exa.

Exa é um motor de busca semântico com:
- $0.005 por busca na faixa de 1 a 25 resultados, que cobre o teto de 20 de max_results
- Busca semântica avançada com embeddings

Recomendado para:
- Alto volume (>2667 buscas/mês)
- Busca semântica mais precisa
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

    @property
    def friendly_name(self) -> str:
        return "Exa Search"

    @property
    def free_tier(self) -> str:
        return "plano pago"

    @property
    def requests_per_minute(self) -> int:
        # Plano padrão: ~5 QPS
        return 300

    def create_tool(self, max_results: int, **kwargs) -> Any:
        """Cria ferramenta ExaSearchResults.

        Args:
            max_results: Número de resultados por busca.

        Returns:
            Ferramenta LangChain que recebe só a consulta.
        """
        from langchain_core.tools import StructuredTool
        from langchain_exa import ExaSearchResults

        # ExaSearchResults aceita num_results e text_contents_options no
        # construtor sem usá-los: são argumentos do _run, que o modelo escolhe,
        # e o _run devolve repr(e) em vez de levantar. A ferramenta abaixo usa
        # só o cliente autenticado, fixa os dois limites e deixa o erro subir
        # até o retry.
        exa = ExaSearchResults()

        def exa_search(query: str) -> str:
            """Busca na web e devolve título, URL e trecho de cada resultado."""
            return str(
                exa.client.search_and_contents(
                    query,
                    num_results=max_results,
                    text={"max_characters": 1000},
                )
            )

        return StructuredTool.from_function(
            func=exa_search,
            name=exa.name,
            description=exa.description,
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
