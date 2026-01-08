"""Utilitários gerais para DataFrameIt.

Este módulo contém funções utilitárias para:
- Parse de JSON de respostas de LLM
- Verificação de dependências
- Conversão entre pandas e polars
- Conversão de Series, listas e dicionários
"""
import re
import json
import importlib
from typing import Tuple, Union, Any
from dataclasses import dataclass
import pandas as pd

# Import opcional de Polars
try:
    import polars as pl
except ImportError:
    pl = None


# Tipos de dados originais suportados
ORIGINAL_TYPE_PANDAS_DF = 'pandas_dataframe'
ORIGINAL_TYPE_POLARS_DF = 'polars_dataframe'
ORIGINAL_TYPE_PANDAS_SERIES = 'pandas_series'
ORIGINAL_TYPE_POLARS_SERIES = 'polars_series'
ORIGINAL_TYPE_LIST = 'list'
ORIGINAL_TYPE_DICT = 'dict'

# Coluna padrão usada para dados convertidos
DEFAULT_TEXT_COLUMN = '_texto'


@dataclass
class ConversionInfo:
    """Informações sobre a conversão de dados para pandas DataFrame."""
    original_type: str
    original_index: Any = None  # Guarda índice/chaves originais para reconversão
    series_name: str = None  # Nome original da Series (se aplicável)


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


def to_pandas(data) -> Tuple[pd.DataFrame, ConversionInfo]:
    """Converte dados para pandas DataFrame.

    Suporta:
    - pandas.DataFrame
    - polars.DataFrame
    - pandas.Series
    - polars.Series
    - list (de strings)
    - dict (valores são os textos)

    Args:
        data: Dados a serem convertidos.

    Returns:
        Tupla (DataFrame pandas, ConversionInfo com metadados da conversão).

    Raises:
        TypeError: Se o tipo não for suportado.
    """
    # pandas DataFrame
    if isinstance(data, pd.DataFrame):
        return data, ConversionInfo(original_type=ORIGINAL_TYPE_PANDAS_DF)

    # polars DataFrame
    if pl is not None and isinstance(data, pl.DataFrame):
        return data.to_pandas(), ConversionInfo(original_type=ORIGINAL_TYPE_POLARS_DF)

    # pandas Series
    if isinstance(data, pd.Series):
        df = pd.DataFrame({DEFAULT_TEXT_COLUMN: data})
        return df, ConversionInfo(
            original_type=ORIGINAL_TYPE_PANDAS_SERIES,
            original_index=data.index.copy(),
            series_name=data.name,
        )

    # polars Series
    if pl is not None and isinstance(data, pl.Series):
        df = pd.DataFrame({DEFAULT_TEXT_COLUMN: data.to_list()})
        return df, ConversionInfo(
            original_type=ORIGINAL_TYPE_POLARS_SERIES,
            series_name=data.name,
        )

    # list
    if isinstance(data, list):
        df = pd.DataFrame({DEFAULT_TEXT_COLUMN: data})
        return df, ConversionInfo(original_type=ORIGINAL_TYPE_LIST)

    # dict
    if isinstance(data, dict):
        keys = list(data.keys())
        values = list(data.values())
        df = pd.DataFrame({DEFAULT_TEXT_COLUMN: values}, index=keys)
        return df, ConversionInfo(
            original_type=ORIGINAL_TYPE_DICT,
            original_index=keys,
        )

    raise TypeError(
        f"Tipo não suportado: {type(data).__name__}. "
        "Use pandas.DataFrame, polars.DataFrame, pandas.Series, polars.Series, list ou dict."
    )


def from_pandas(df: pd.DataFrame, conversion_info: Union[ConversionInfo, bool]) -> Any:
    """Converte DataFrame pandas de volta para o formato original.

    Remove automaticamente as colunas internas de controle (_dataframeit_status
    e _error_details) se não houver erros no processamento.

    Args:
        df: DataFrame pandas processado.
        conversion_info: ConversionInfo com metadados da conversão original,
                        ou bool para retrocompatibilidade (True = era polars).

    Returns:
        Dados no formato original (DataFrame, Series, list ou dict).
    """
    # Retrocompatibilidade: aceita bool (was_polars)
    if isinstance(conversion_info, bool):
        was_polars = conversion_info
        conversion_info = ConversionInfo(
            original_type=ORIGINAL_TYPE_POLARS_DF if was_polars else ORIGINAL_TYPE_PANDAS_DF
        )

    # Colunas internas de controle (não usar esses nomes em seus dados!)
    status_col = '_dataframeit_status'
    error_col = '_error_details'

    # Remover colunas de status/erro se não houver erros
    if status_col in df.columns:
        has_errors = (df[status_col] == 'error').any()
        has_error_details = error_col in df.columns and df[error_col].notna().any()

        if not has_errors and not has_error_details:
            cols_to_drop = [c for c in [status_col, error_col] if c in df.columns]
            df = df.drop(columns=cols_to_drop)

    # pandas DataFrame
    if conversion_info.original_type == ORIGINAL_TYPE_PANDAS_DF:
        return df

    # polars DataFrame
    if conversion_info.original_type == ORIGINAL_TYPE_POLARS_DF:
        if pl is not None:
            return pl.from_pandas(df)
        return df

    # pandas Series - retorna todas as colunas extraídas como DataFrame
    # (não faz sentido retornar Series quando temos múltiplas colunas de resultado)
    if conversion_info.original_type == ORIGINAL_TYPE_PANDAS_SERIES:
        # Remove a coluna de texto original se existir
        if DEFAULT_TEXT_COLUMN in df.columns:
            df = df.drop(columns=[DEFAULT_TEXT_COLUMN])
        # Restaurar índice original
        if conversion_info.original_index is not None:
            df.index = conversion_info.original_index
        return df

    # polars Series - similar ao pandas Series
    if conversion_info.original_type == ORIGINAL_TYPE_POLARS_SERIES:
        if DEFAULT_TEXT_COLUMN in df.columns:
            df = df.drop(columns=[DEFAULT_TEXT_COLUMN])
        if pl is not None:
            return pl.from_pandas(df)
        return df

    # list - tratar como Series, retorna DataFrame
    if conversion_info.original_type == ORIGINAL_TYPE_LIST:
        if DEFAULT_TEXT_COLUMN in df.columns:
            df = df.drop(columns=[DEFAULT_TEXT_COLUMN])
        return df

    # dict - tratar como DataFrame, retorna DataFrame com chaves como índice
    if conversion_info.original_type == ORIGINAL_TYPE_DICT:
        if DEFAULT_TEXT_COLUMN in df.columns:
            df = df.drop(columns=[DEFAULT_TEXT_COLUMN])
        return df

    # Fallback
    return df
