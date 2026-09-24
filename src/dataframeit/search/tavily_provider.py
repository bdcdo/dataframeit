"""Provedor de busca Tavily.

Tavily é um motor de busca otimizado para IA com:
- 1000 créditos gratuitos/mês
- $0.008 por crédito após isso
- Suporte a search_depth: "basic" (1 crédito) ou "advanced" (2 créditos)

Recomendado para:
- Volume baixo-médio (<2667 buscas/mês)
"""

import re
from typing import Any

from .base import SearchProvider, register_provider

# Status que indicam argumento inválido escolhido pelo modelo, e não falha da
# conta ou do serviço. O wrapper do langchain_tavily só põe o status no texto:
# "Error 400: ...".
_ERROS_DO_MODELO = (400, 422)
_STATUS_NO_TEXTO = re.compile(r"^Error (\d{3}):")


def _levantar_erro(resultado, tool_exception):
    """Levanta o erro que o TavilySearch devolveu como {"error": e}."""
    if not (isinstance(resultado, dict) and isinstance(resultado.get("error"), Exception)):
        return resultado
    erro = resultado["error"]
    status = _STATUS_NO_TEXTO.match(str(erro))
    if (status and int(status.group(1)) in _ERROS_DO_MODELO) or (
        "can only be set during instantiation" in str(erro)
    ):
        raise tool_exception(str(erro)) from erro
    raise erro


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

        from langchain_core.tools import ToolException

        class _TavilySearchQueLevantaErro(TavilySearch):
            """TavilySearch que separa o erro do modelo do erro do provedor.

            O _run original devolve {"error": e} para qualquer falha, e o agente
            segue sem evidência, com a linha marcada como processada. Erro de
            conta, quota, rede ou servidor sobe ao retry da biblioteca. Erro
            causado pelos argumentos que o modelo escolheu (400, 422 ou
            parâmetro proibido na chamada) vira ToolException, como "sem
            resultados", e volta ao modelo para ele tentar outra consulta.
            """

            def _run(self, *args, **kwargs):
                return _levantar_erro(super()._run(*args, **kwargs), ToolException)

            async def _arun(self, *args, **kwargs):
                return _levantar_erro(await super()._arun(*args, **kwargs), ToolException)

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
