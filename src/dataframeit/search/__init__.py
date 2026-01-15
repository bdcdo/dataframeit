"""Módulo de provedores de busca web.

Suporta múltiplos provedores de busca:
- Tavily: Boa opção para volume baixo-médio (<2667 buscas/mês)
- Exa: Mais econômico para alto volume (>2667 buscas/mês com 1-25 resultados)
"""

from .base import SearchProvider, get_provider, get_available_providers
from .tavily_provider import TavilyProvider
from .exa_provider import ExaProvider

__all__ = [
    'SearchProvider',
    'get_provider',
    'get_available_providers',
    'TavilyProvider',
    'ExaProvider',
]
