import re
import json
import importlib
import time
import random
import warnings
from functools import wraps
from typing import Tuple, List, Union, Any
import pandas as pd

# Import opcional de Polars
try:
    import polars as pl  # type: ignore
except Exception:  # Polars não instalado
    pl = None  # type: ignore

def parse_json(resposta: str) -> dict:
    """
    Extrai e faz o parse de uma string JSON contida na resposta de um LLM.

    Tenta extrair o JSON de blocos de código (```json) ou encontrando o
    primeiro e último caracter '{' e '}'. Se a extração falhar ou o JSON
    for inválido, levanta um ValueError.

    Args:
        resposta: A string de resposta do LLM.

    Returns:
        Um dicionário com os dados do JSON.

    Raises:
        ValueError: Se o JSON não puder ser extraído ou decodificado.
    """
    # Se a resposta for um objeto com atributo 'content', extrai o conteúdo
    if hasattr(resposta, 'content'):
        if isinstance(resposta.content, list):
            langchain_output_content = "".join(str(item) for item in resposta.content)
        else:
            langchain_output_content = resposta.content
    else:
        # Assume que a resposta já é uma string
        langchain_output_content = str(resposta)


    json_string_extraida = None
    match = re.search(r"```json\n(.*?)\n```", langchain_output_content, re.DOTALL)
    
    if match:
        json_string_extraida = match.group(1).strip()
    else:
        start_brace = langchain_output_content.find('{')
        end_brace = langchain_output_content.rfind('}')
        if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
            json_string_extraida = langchain_output_content[start_brace : end_brace + 1]

    if not json_string_extraida:
        # Se nenhuma das estratégias acima funcionou, tenta usar a string toda
        json_string_extraida = langchain_output_content.strip()

    try:
        return json.loads(json_string_extraida)
    except json.JSONDecodeError as e:
        raise ValueError(f"Falha ao decodificar JSON. Erro: {e}. Resposta recebida: '{json_string_extraida[:200]}'...")

def check_dependency(dependency: str, pip_name: str = None):
    """Verifica se uma dependência está instalada e lança um erro claro se não estiver."""
    pip_name = pip_name or dependency
    try:
        importlib.import_module(dependency)
    except ImportError:
        raise ImportError(
            f"A dependência '{dependency}' não está instalada. "
            f"Por favor, instale-a com: uv pip install '{pip_name}'"
        )


def retry_with_backoff(func):
    """Decorator para implementar retry com backoff exponencial.

    Usa os parâmetros de configuração do DataFrameConfiguration.

    Args:
        func: Função a ser decorada com retry.

    Returns:
        Decorator que implementa retry com backoff exponencial.

    Raises:
        Exception: Relança a última exceção ocorrida após todas as tentativas.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Obter configuração de retry do self.config
        max_retries = getattr(self.config, 'max_retries', 3)
        base_delay = getattr(self.config, 'base_delay', 1.0)
        max_delay = getattr(self.config, 'max_delay', 30.0)
        exponential_base = 2

        retries = 0
        while retries < max_retries:
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                retries += 1
                if retries >= max_retries:
                    # Lançar a exceção em vez de retornar None
                    raise e

                # Calcular delay com backoff exponencial
                delay = min(base_delay * (exponential_base ** (retries - 1)), max_delay)
                # Adicionar jitter para evitar thundering herd
                # O jitter adiciona uma pequena variação aleatória ao delay para evitar
                # que múltiplas requisições ocorram simultaneamente após falhas
                jitter = random.uniform(0, 0.1) * delay
                time.sleep(delay + jitter)
        # Esta linha não será alcançada, mas está aqui para clareza
        return None
    return wrapper


def convert_dataframe_to_pandas(df) -> Tuple[pd.DataFrame, bool]:
    """Converte DataFrame para pandas se necessário.

    Converte DataFrames do Polars para pandas, mantendo os DataFrames pandas inalterados.

    Args:
        df: DataFrame pandas ou polars para conversão.

    Returns:
        Tuple[pd.DataFrame, bool]: Tupla contendo o DataFrame pandas e um booleano indicando
        se a conversão foi realizada (True se era Polars, False se já era pandas).

    Raises:
        TypeError: Se o df não for pandas.DataFrame nem polars.DataFrame.
    """
    if pl is not None and hasattr(pl, 'DataFrame') and isinstance(df, pl.DataFrame):
        return df.to_pandas(), True
    elif isinstance(df, pd.DataFrame):
        return df, False
    else:
        raise TypeError("df must be a pandas.DataFrame or polars.DataFrame")


def convert_dataframe_back(df: pd.DataFrame, was_polars: bool) -> Union[pd.DataFrame, Any]:
    """Converte de volta para Polars se necessário.

    Converte DataFrames pandas de volta para Polars se a conversão original foi feita.

    Args:
        df (pd.DataFrame): DataFrame pandas para possível conversão.
        was_polars (bool): Indica se o DataFrame original era Polars.

    Returns:
        Union[pd.DataFrame, Any]: DataFrame Polars se was_polars é True, caso contrário
        retorna o DataFrame pandas original.
    """
    if was_polars and pl is not None:
        return pl.from_pandas(df)
    return df


def validate_text_column(df: pd.DataFrame, text_column: str) -> None:
    """Valida se coluna de texto existe no DataFrame.

    Args:
        df (pd.DataFrame): DataFrame para validar.
        text_column (str): Nome da coluna de texto esperada.

    Raises:
        ValueError: Se a coluna de texto não existir no DataFrame.
    """
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame.")


def validate_columns_conflict(df: pd.DataFrame, expected_columns: List[str], resume: bool) -> bool:
    """Valida conflitos de colunas existentes.

    Verifica se colunas que serão criadas já existem no DataFrame e emite um aviso
    se for o caso e o modo de retomada não estiver ativado.

    Args:
        df (pd.DataFrame): DataFrame para validar.
        expected_columns (List[str]): Lista de colunas que serão criadas.
        resume (bool): Se True, permite continuar mesmo com colunas existentes.

    Returns:
        bool: True se pode continuar o processamento, False caso contrário.
    """
    existing_result_columns = [col for col in expected_columns if col in df.columns]
    if existing_result_columns and not resume:
        warnings.warn(f"Columns {existing_result_columns} already exist. Use resume=True to continue or rename them.")
        return False  # Não continuar
    return True  # Continuar processamento
