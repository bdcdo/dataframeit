import re
import json
import importlib
import time
import random
import warnings
from typing import Tuple, List, Union, Any
import pandas as pd

# Import opcional de Polars
try:
    import polars as pl
except ImportError:
    pl = None


def parse_json(resposta: str) -> dict:
    """Extrai e faz parse de JSON da resposta de um LLM.

    Args:
        resposta: String de resposta do LLM ou objeto com atributo 'content'.

    Returns:
        Dicionário com os dados do JSON.

    Raises:
        ValueError: Se o JSON não puder ser extraído ou decodificado.
    """
    # Extrair conteúdo se for objeto do LangChain
    if hasattr(resposta, 'content'):
        if isinstance(resposta.content, list):
            content = "".join(str(item) for item in resposta.content)
        else:
            content = resposta.content
    else:
        content = str(resposta)

    # Tentar extrair JSON de bloco de código markdown
    match = re.search(r"```json\n(.*?)\n```", content, re.DOTALL)
    if match:
        json_string = match.group(1).strip()
    else:
        # Tentar extrair entre chaves
        start = content.find('{')
        end = content.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_string = content[start:end + 1]
        else:
            json_string = content.strip()

    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError(f"Falha ao decodificar JSON. Erro: {e}. Resposta: '{json_string[:200]}'...")


def check_dependency(package: str, install_name: str = None):
    """Verifica se dependência está instalada.

    Args:
        package: Nome do pacote para importação.
        install_name: Nome do pacote para instalação (padrão: package).

    Raises:
        ImportError: Se a dependência não estiver instalada.
    """
    install_name = install_name or package
    try:
        importlib.import_module(package)
    except ImportError:
        raise ImportError(
            f"'{package}' não instalado. Instale com: pip install {install_name}"
        )


# Erros considerados recuperáveis (transientes)
RECOVERABLE_ERRORS = (
    # Timeouts e deadlines
    'DeadlineExceeded',
    'Timeout',
    'TimeoutError',
    'ReadTimeout',
    'ConnectTimeout',
    # Rate limits
    'RateLimitError',
    'ResourceExhausted',
    'TooManyRequests',
    '429',
    # Erros de servidor temporários
    'ServiceUnavailable',
    'InternalServerError',
    '500',
    '502',
    '503',
    '504',
    # Erros de conexão
    'ConnectionError',
    'ConnectionReset',
    'SSLError',
)

# Erros não-recuperáveis (não adianta tentar novamente)
NON_RECOVERABLE_ERRORS = (
    'AuthenticationError',
    'InvalidAPIKey',
    'PermissionDenied',
    'InvalidArgument',
    'NotFound',
    '401',
    '403',
    '404',
)


def is_recoverable_error(error: Exception) -> bool:
    """Verifica se um erro é recuperável (vale a pena fazer retry).

    Args:
        error: Exceção a ser analisada.

    Returns:
        True se o erro é recuperável, False caso contrário.
    """
    error_str = f"{type(error).__name__}: {error}"

    # Verificar se é explicitamente não-recuperável
    for pattern in NON_RECOVERABLE_ERRORS:
        if pattern.lower() in error_str.lower():
            return False

    # Verificar se é explicitamente recuperável
    for pattern in RECOVERABLE_ERRORS:
        if pattern.lower() in error_str.lower():
            return True

    # Por padrão, tentar recuperar (comportamento original)
    return True


def retry_with_backoff(func, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 30.0) -> dict:
    """Executa função com retry e backoff exponencial.

    Args:
        func: Função a ser executada.
        max_retries: Número máximo de tentativas.
        base_delay: Delay base em segundos.
        max_delay: Delay máximo em segundos.

    Returns:
        Dicionário com 'result' (resultado da função) e 'retry_info' (informações de retry).

    Raises:
        Exception: Última exceção após esgotar tentativas ou erro não-recuperável.
    """
    retry_info = {
        'attempts': 0,
        'retries': 0,
        'errors': [],
    }

    for attempt in range(max_retries):
        retry_info['attempts'] = attempt + 1
        try:
            result = func()
            # Adicionar retry_info ao resultado se for dict
            if isinstance(result, dict):
                result['_retry_info'] = retry_info
            return result
        except Exception as e:
            error_name = type(e).__name__
            error_msg = str(e)
            retry_info['errors'].append(f"{error_name}: {error_msg[:100]}")

            # Verificar se é erro não-recuperável
            if not is_recoverable_error(e):
                warnings.warn(
                    f"Erro não-recuperável detectado ({error_name}). Não será feito retry.",
                    stacklevel=3
                )
                raise

            # Última tentativa - não fazer mais retry
            if attempt == max_retries - 1:
                raise

            # Calcular delay com backoff exponencial
            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = random.uniform(0, 0.1) * delay
            total_delay = delay + jitter

            retry_info['retries'] = attempt + 1

            # Warning informativo sobre o retry
            warnings.warn(
                f"Tentativa {attempt + 1}/{max_retries} falhou ({error_name}). "
                f"Aguardando {total_delay:.1f}s antes de tentar novamente...",
                stacklevel=3
            )

            time.sleep(total_delay)


def to_pandas(df) -> Tuple[pd.DataFrame, bool]:
    """Converte DataFrame para pandas se necessário.

    Args:
        df: DataFrame pandas ou polars.

    Returns:
        Tupla (DataFrame pandas, flag se era polars).

    Raises:
        TypeError: Se não for pandas nem polars DataFrame.
    """
    if pl is not None and isinstance(df, pl.DataFrame):
        return df.to_pandas(), True
    elif isinstance(df, pd.DataFrame):
        return df, False
    else:
        raise TypeError("df deve ser pandas.DataFrame ou polars.DataFrame")


def from_pandas(df: pd.DataFrame, was_polars: bool) -> Union[pd.DataFrame, Any]:
    """Converte de volta para polars se necessário.

    Args:
        df: DataFrame pandas.
        was_polars: Se o DataFrame original era polars.

    Returns:
        DataFrame no formato original.
    """
    if was_polars and pl is not None:
        return pl.from_pandas(df)
    return df
