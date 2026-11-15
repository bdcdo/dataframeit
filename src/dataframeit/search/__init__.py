"""Módulo de provedores de busca web.

Suporta múltiplos provedores de busca:
- Tavily: Boa opção para volume baixo-médio (<2667 buscas/mês)
- Exa: Mais econômico para alto volume (>2667 buscas/mês)
"""

# A ordem de import define a ordem do registro de provedores, que aparece em
# get_available_providers() e nas mensagens de erro. Tavily, o padrão, vem primeiro.
from .base import (  # isort: skip
    SearchProvider,
    SearchToolError,
    get_available_providers,
    get_provider,
)
from .tavily_provider import TavilyProvider  # isort: skip
from .exa_provider import ExaProvider  # isort: skip

__all__ = [
    "ExaProvider",
    "SearchProvider",
    "SearchToolError",
    "TavilyProvider",
    "get_available_providers",
    "get_provider",
]
