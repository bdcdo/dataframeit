"""Utilitários gerais para DataFrameIt.

Este módulo contém funções utilitárias para:
- Parse de JSON de respostas de LLM
- Verificação de dependências
- Conversão entre pandas e polars
- Normalização de estruturas Python (listas, dicionários, tuplas)
"""
import re
import json
import importlib
import types
from typing import Tuple, Union, Any, get_origin, get_args
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

    Remove automaticamente as colunas internas de controle (_dataframeit_status
    e _error_details) se não houver erros no processamento.

    Args:
        df: DataFrame pandas.
        was_polars: Se o DataFrame original era polars.

    Returns:
        DataFrame no formato original.
    """
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

    if was_polars and pl is not None:
        return pl.from_pandas(df)
    return df
