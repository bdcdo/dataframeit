import re
import json
import importlib
import time
import random
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


def retry_with_backoff(func, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 30.0):
    """Executa função com retry e backoff exponencial.

    Args:
        func: Função a ser executada.
        max_retries: Número máximo de tentativas.
        base_delay: Delay base em segundos.
        max_delay: Delay máximo em segundos.

    Returns:
        Resultado da função.

    Raises:
        Exception: Última exceção após esgotar tentativas.
    """
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise

            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = random.uniform(0, 0.1) * delay
            time.sleep(delay + jitter)


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
