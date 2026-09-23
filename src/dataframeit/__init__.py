from importlib.metadata import version as _version

from .core import dataframeit
from .utils import (
    normalize_value,
    normalize_complex_columns,
    get_complex_fields,
    read_df,
)

# Lida dos metadados do pacote instalado, para que o campo `version` do
# pyproject.toml seja a única fonte do número.
__version__ = _version("dataframeit")

__all__ = [
    'dataframeit',
    'read_df',
    'normalize_value',
    'normalize_complex_columns',
    'get_complex_fields',
]
