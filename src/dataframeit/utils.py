"""Utilitários gerais para DataFrameIt.

Este módulo contém funções utilitárias para:
- Parse de JSON de respostas de LLM
- Verificação de dependências
- Conversão entre pandas e polars
- Conversão de Series, listas e dicionários
- Normalização de estruturas Python (listas, dicionários, tuplas)
"""
import re
import json
import importlib
import types
from typing import Tuple, Union, Any, List, get_origin, get_args
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
        return _reorder_columns(df)

    # polars DataFrame
    if conversion_info.original_type == ORIGINAL_TYPE_POLARS_DF:
        df = _reorder_columns(df)
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
        return _reorder_columns(df)

    # polars Series - similar ao pandas Series
    if conversion_info.original_type == ORIGINAL_TYPE_POLARS_SERIES:
        if DEFAULT_TEXT_COLUMN in df.columns:
            df = df.drop(columns=[DEFAULT_TEXT_COLUMN])
        df = _reorder_columns(df)
        if pl is not None:
            return pl.from_pandas(df)
        return df

    # list - tratar como Series, retorna DataFrame
    if conversion_info.original_type == ORIGINAL_TYPE_LIST:
        if DEFAULT_TEXT_COLUMN in df.columns:
            df = df.drop(columns=[DEFAULT_TEXT_COLUMN])
        return _reorder_columns(df)

    # dict - tratar como DataFrame, retorna DataFrame com chaves como índice
    if conversion_info.original_type == ORIGINAL_TYPE_DICT:
        if DEFAULT_TEXT_COLUMN in df.columns:
            df = df.drop(columns=[DEFAULT_TEXT_COLUMN])
        return _reorder_columns(df)

    # Fallback
    return _reorder_columns(df)


def _reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Reordena colunas para que internas fiquem no final.

    Ordem final das colunas:
    1. Colunas do usuário (originais + campos do modelo)
    2. Colunas de trace (_trace_*)
    3. Colunas de busca (_search_credits, _search_count)
    4. Colunas de tokens (_input_tokens, _output_tokens, _total_tokens)
    5. Colunas de controle (_dataframeit_status, _error_details)

    Args:
        df: DataFrame a reordenar.

    Returns:
        DataFrame com colunas reordenadas.
    """
    # Separar colunas por categoria
    user_cols = []
    trace_cols = []
    search_cols = []
    token_cols = []
    status_cols = []

    for col in df.columns:
        if col.startswith('_trace_'):
            trace_cols.append(col)
        elif col in ['_search_credits', '_search_count']:
            search_cols.append(col)
        elif col in ['_input_tokens', '_output_tokens', '_total_tokens']:
            token_cols.append(col)
        elif col in ['_dataframeit_status', '_error_details']:
            status_cols.append(col)
        else:
            user_cols.append(col)

    # Ordem final: user -> trace -> search -> tokens -> status
    new_order = user_cols + trace_cols + search_cols + token_cols + status_cols

    return df[new_order]


# =============================================================================
# NORMALIZAÇÃO DE ESTRUTURAS PYTHON (listas, dicts, tuples)
# =============================================================================

def is_complex_type(field_type) -> bool:
    """Verifica se um tipo é complexo (list, dict, tuple).

    Args:
        field_type: Tipo a verificar (pode ser tipo simples ou genérico).

    Returns:
        True se o tipo for list, dict ou tuple.
    """
    origin = get_origin(field_type)

    # Tipos genéricos: list[str], dict[str, int], tuple[int, str], etc.
    if origin in (list, dict, tuple):
        return True

    # Union types (Optional, Union) - verificar os argumentos internos
    # typing.Union para sintaxe Union[X, Y] e Optional[X]
    if origin is Union:
        args = get_args(field_type)
        return any(is_complex_type(arg) for arg in args if arg is not type(None))

    # types.UnionType para sintaxe X | Y (Python 3.10+)
    if isinstance(field_type, types.UnionType):
        args = get_args(field_type)
        return any(is_complex_type(arg) for arg in args if arg is not type(None))

    # Tipos diretos
    if field_type in (list, dict, tuple):
        return True

    return False


def get_complex_fields(pydantic_model) -> set:
    """Retorna os nomes dos campos que são tipos complexos (list, dict, tuple).

    Args:
        pydantic_model: Modelo Pydantic a analisar.

    Returns:
        Set com nomes dos campos complexos.
    """
    complex_fields = set()

    for field_name, field_info in pydantic_model.model_fields.items():
        if is_complex_type(field_info.annotation):
            complex_fields.add(field_name)

    return complex_fields


def normalize_value(value: Any) -> Any:
    """Normaliza um valor, convertendo strings JSON para estruturas Python.

    Esta função garante que valores que deveriam ser listas, dicionários ou
    tuplas sejam tratados como tal, mesmo que tenham sido armazenados como
    strings JSON (comum ao salvar/carregar de Excel/CSV).

    Args:
        value: Valor a normalizar.

    Returns:
        Valor normalizado (estrutura Python se era JSON string válido,
        ou o valor original caso contrário).

    Examples:
        >>> normalize_value('[1, 2, 3]')
        [1, 2, 3]
        >>> normalize_value('{"a": 1}')
        {'a': 1}
        >>> normalize_value('texto normal')
        'texto normal'
        >>> normalize_value([1, 2, 3])  # já é lista
        [1, 2, 3]
    """
    # Se já é estrutura Python, retorna como está
    if isinstance(value, (list, dict, tuple)):
        return value

    # Se é None ou não é string, retorna como está
    if value is None or not isinstance(value, str):
        return value

    # Tenta fazer parse de JSON
    stripped = value.strip()
    if not stripped:
        return value

    # Verifica se parece com JSON (começa com [ ou {)
    if stripped.startswith(('[', '{')):
        try:
            parsed = json.loads(stripped)
            return parsed
        except (json.JSONDecodeError, ValueError):
            pass

    return value


def normalize_complex_columns(df: pd.DataFrame, complex_fields: set) -> None:
    """Normaliza colunas complexas no DataFrame, convertendo strings JSON.

    Modifica o DataFrame in-place.

    Args:
        df: DataFrame a normalizar.
        complex_fields: Set com nomes das colunas a normalizar.
    """
    for col in complex_fields:
        if col in df.columns:
            df[col] = df[col].apply(normalize_value)


def read_df(
    path: str,
    model=None,
    normalize: bool = True,
    **kwargs
) -> pd.DataFrame:
    """Carrega um DataFrame de arquivo e normaliza estruturas Python automaticamente.

    Esta função é útil para carregar dados que foram previamente processados
    pelo dataframeit e salvos em arquivo. Ela converte automaticamente strings
    JSON de volta para listas, dicionários e outras estruturas Python.

    Args:
        path: Caminho do arquivo (suporta .xlsx, .xls, .csv, .parquet, .json).
        model: Modelo Pydantic opcional. Se fornecido, apenas as colunas que
               correspondem a campos complexos do modelo serão normalizadas.
        normalize: Se True (padrão), normaliza colunas que parecem conter JSON.
                   Se False, não faz nenhuma normalização.
        **kwargs: Argumentos adicionais passados para a função de leitura do pandas.

    Returns:
        DataFrame com estruturas Python normalizadas.

    Raises:
        ValueError: Se o formato do arquivo não for suportado.
        FileNotFoundError: Se o arquivo não existir.

    Examples:
        >>> # Uso simples - normaliza automaticamente
        >>> df = read_df('resultados.xlsx')
        >>> print(df['lista_itens'][0])  # ['item1', 'item2']

        >>> # Com modelo Pydantic (mais preciso)
        >>> df = read_df('resultados.xlsx', MeuModelo)

        >>> # Sem normalização
        >>> df = read_df('dados.csv', normalize=False)
    """
    import os

    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    # Detectar formato pelo sufixo
    _, ext = os.path.splitext(path.lower())

    # Carregar DataFrame baseado na extensão
    if ext in ('.xlsx', '.xls'):
        df = pd.read_excel(path, **kwargs)
    elif ext == '.csv':
        df = pd.read_csv(path, **kwargs)
    elif ext == '.parquet':
        df = pd.read_parquet(path, **kwargs)
    elif ext == '.json':
        df = pd.read_json(path, **kwargs)
    else:
        raise ValueError(
            f"Formato '{ext}' não suportado. "
            "Use: .xlsx, .xls, .csv, .parquet ou .json"
        )

    # Normalizar colunas
    if not normalize:
        return df

    if model is not None:
        # Usar modelo Pydantic para identificar colunas complexas
        complex_fields = get_complex_fields(model)
        if complex_fields:
            normalize_complex_columns(df, complex_fields)
    else:
        # Normalizar todas as colunas que parecem ter JSON
        _normalize_all_json_columns(df)

    return df


def _normalize_all_json_columns(df: pd.DataFrame) -> None:
    """Normaliza todas as colunas do DataFrame que parecem conter JSON.

    Modifica o DataFrame in-place.

    Args:
        df: DataFrame a normalizar.
    """
    for col in df.columns:
        # Pular colunas não-string
        if df[col].dtype != 'object':
            continue

        # Verificar se algum valor parece JSON
        sample = df[col].dropna().head(10)
        has_json = any(
            isinstance(v, str) and v.strip().startswith(('[', '{'))
            for v in sample
        )

        if has_json:
            df[col] = df[col].apply(normalize_value)


# =============================================================================
# EXTRAÇÃO DE MODELOS PYDANTIC ANINHADOS
# =============================================================================

def get_nested_pydantic_models(field_type) -> List:
    """Extrai todos os modelos Pydantic de uma anotação de tipo.

    Trata List[Model], Optional[List[Model]], Union[Model, None], etc.
    Usado para detectar campos de busca em modelos aninhados.

    Args:
        field_type: Anotação de tipo a analisar.

    Returns:
        Lista de modelos Pydantic encontrados na anotação.

    Examples:
        >>> class Inner(BaseModel):
        ...     x: str
        >>> get_nested_pydantic_models(List[Inner])
        [<class 'Inner'>]
        >>> get_nested_pydantic_models(Optional[List[Inner]])
        [<class 'Inner'>]
        >>> get_nested_pydantic_models(str)
        []
    """
    from pydantic import BaseModel

    models = []
    origin = get_origin(field_type)

    # Caso 1: É diretamente um modelo Pydantic
    if isinstance(field_type, type) and issubclass(field_type, BaseModel):
        models.append(field_type)
        return models

    # Caso 2: Tipos genéricos (List[X], Optional[X], Union[X, Y], etc.)
    if origin is not None:
        args = get_args(field_type)

        for arg in args:
            # Ignorar NoneType em Optional/Union
            if arg is type(None):
                continue

            # Checar se o argumento é um modelo Pydantic
            if isinstance(arg, type) and issubclass(arg, BaseModel):
                models.append(arg)
            else:
                # Recursivamente buscar em tipos aninhados (e.g., List[List[Model]])
                models.extend(get_nested_pydantic_models(arg))

    # Caso 3: types.UnionType para sintaxe X | Y (Python 3.10+)
    if isinstance(field_type, types.UnionType):
        args = get_args(field_type)
        for arg in args:
            if arg is type(None):
                continue
            if isinstance(arg, type) and issubclass(arg, BaseModel):
                models.append(arg)
            else:
                models.extend(get_nested_pydantic_models(arg))

    return models
