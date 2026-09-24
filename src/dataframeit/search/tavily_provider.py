"""Provedor de busca Tavily.

Tavily é um motor de busca otimizado para IA com:
- 1000 créditos gratuitos/mês
- $0.008 por crédito após isso
- Suporte a search_depth: "basic" (1 crédito) ou "advanced" (2 créditos)

Recomendado para:
- Volume baixo-médio (<2667 buscas/mês)
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

    @property
    def friendly_name(self) -> str:
        return "Tavily Search"

    @property
    def free_tier(self) -> str:
        return "1000 buscas/mês"

    @property
    def requests_per_minute(self) -> int:
        # Plano gratuito/básico
        return 100

    def create_tool(self, max_results: int, search_depth: str = "basic", **kwargs) -> Any:
        """Cria ferramenta TavilySearch.

        Args:
            max_results: Número de resultados (1-20).
            search_depth: "basic" (1 crédito) ou "advanced" (2 créditos).

        Returns:
            Instância de TavilySearch configurada.
        """
        import warnings

        # langchain_tavily emite UserWarnings sobre "Field name X shadows attribute"
        # em BaseTool na definição de TavilyResearch. São warnings externos e ruidosos;
        # filtramos apenas o módulo upstream para não mascarar warnings nossos.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Field name .* shadows.*",
                category=UserWarning,
                module=r"langchain_tavily\..*",
            )
            from langchain_tavily import TavilySearch

        class _TavilySearchQueLevantaErro(TavilySearch):
            """TavilySearch que levanta o erro do provedor.

            O _run original devolve {"error": e} para quota, chave inválida ou
            falha de rede, e o agente segue sem evidência, com a linha marcada
            como processada. "Sem resultados" continua sendo ToolException,
            que vira mensagem ao modelo para ele tentar outra consulta.
            """

            def _run(self, *args, **kwargs):
                result = super()._run(*args, **kwargs)
                if isinstance(result, dict) and isinstance(result.get("error"), Exception):
                    raise result["error"]
                return result

        return _TavilySearchQueLevantaErro(
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
