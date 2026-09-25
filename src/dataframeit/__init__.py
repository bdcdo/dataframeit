from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from .core import dataframeit
from .errors import (
    ProviderConfigurationError,
    ProviderError,
    ProviderOutputError,
    ProviderOverloadedError,
    ProviderRejectedOutputError,
    ProviderTransientError,
)
from .utils import (
    get_complex_fields,
    normalize_complex_columns,
    normalize_value,
    read_df,
)

# Lida dos metadados do pacote instalado, para que o campo `version` do
# pyproject.toml seja a única fonte do número. Sem metadados (código
# copiado para outro projeto, app congelado sem copy_metadata), o import
# segue funcionando com uma versão local que não se confunde com release.
try:
    __version__ = _version("dataframeit")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "dataframeit",
    "read_df",
    "normalize_value",
    "normalize_complex_columns",
    "get_complex_fields",
    "ProviderError",
    "ProviderTransientError",
    "ProviderOverloadedError",
    "ProviderRejectedOutputError",
    "ProviderConfigurationError",
    "ProviderOutputError",
    "__version__",
]
