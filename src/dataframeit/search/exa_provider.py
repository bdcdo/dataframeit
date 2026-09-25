"""Provedor de busca Exa.

Exa é um motor de busca semântico com:
- $0.005 por busca na faixa de 1 a 25 resultados, que cobre o teto de 20 de max_results
- Busca semântica avançada com embeddings

Recomendado para:
- Alto volume (>2667 buscas/mês)
- Busca semântica mais precisa
"""

from __future__ import annotations

from langchain_core.tools import BaseTool, StructuredTool

from .base import SearchProvider, register_provider

# Até este número de resultados, a Exa cobra a faixa de preço mais baixa.
_MAX_RESULTS_FAIXA_BASICA = 25


@register_provider
class ExaProvider(SearchProvider):
    """Implementação do provedor de busca Exa."""

    @property
    def name(self) -> str:
        """Identificador do provedor."""
        return "exa"

    @property
    def env_var(self) -> str:
        """Variável de ambiente da API key."""
        return "EXA_API_KEY"

    @property
    def package_name(self) -> str:
        """Módulo Python da integração LangChain."""
        return "langchain_exa"

    @property
    def install_name(self) -> str:
        """Nome do pacote para pip install."""
        return "langchain-exa"

    @property
    def signup_url(self) -> str:
        """Página para criar conta e obter a API key."""
        return "https://exa.ai"

    @property
    def friendly_name(self) -> str:
        """Nome exibido nas mensagens de erro."""
        return "Exa Search"

    @property
    def free_tier(self) -> str:
        """Plano de entrada, exibido na mensagem de API key ausente."""
        return "plano pago"

    @property
    def requests_per_minute(self) -> int:
        """Limite aproximado de requisições por minuto."""
        # Plano padrão: ~5 QPS
        return 300

    def create_tool(self, max_results: int, **kwargs: object) -> BaseTool:
        """Cria ferramenta ExaSearchResults.

        Args:
            max_results: Número de resultados por busca.
            **kwargs: Parâmetros de outros provedores, ignorados pela Exa.

        Returns:
            Ferramenta LangChain que recebe só a consulta.
        """
        from langchain_exa import ExaSearchResults  # noqa: PLC0415 (extra de busca opcional)

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

    def calculate_credits(self, search_count: int, max_results: int = 5, **kwargs: object) -> int:
        """Calcula créditos Exa consumidos.

        Exa cobra por busca, com preço variando pelo número de resultados:
        - 1-25 resultados: $0.005 (representamos como 1 crédito)
        - 26-100 resultados: $0.025 (representamos como 5 créditos)

        Args:
            search_count: Número de buscas realizadas.
            max_results: Número de resultados por busca.
            **kwargs: Parâmetros de outros provedores, ignorados pela Exa.

        Returns:
            Total de créditos consumidos (1 crédito = $0.005).
        """
        # Representamos em unidades de $0.005 para facilitar comparação
        cost_per_search = 1 if max_results <= _MAX_RESULTS_FAIXA_BASICA else 5
        return search_count * cost_per_search
