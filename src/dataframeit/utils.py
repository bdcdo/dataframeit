"""Utilitários gerais para DataFrameIt.

Este módulo contém funções utilitárias para:
- Parse de JSON de respostas de LLM
- Verificação de dependências
- Conversão entre pandas e polars
- Conversão de Series, listas e dicionários
- Normalização de estruturas Python (listas, dicionários, tuplas)
"""

from __future__ import annotations

import ast
import datetime
import functools
import json
import operator
import re
import sys
import types
import typing
import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_args, get_origin

import pandas as pd
from pandas.api.types import is_float_dtype, is_integer_dtype, is_string_dtype
from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Hashable, Mapping

    from polars import DataFrame as PolarsDataFrame
    from polars import Series as PolarsSeries

# Import opcional de Polars
try:
    import polars as pl
except ImportError:
    pl = None


# Tipos de dados originais suportados
ORIGINAL_TYPE_PANDAS_DF = "pandas_dataframe"
ORIGINAL_TYPE_POLARS_DF = "polars_dataframe"
ORIGINAL_TYPE_PANDAS_SERIES = "pandas_series"
ORIGINAL_TYPE_POLARS_SERIES = "polars_series"
ORIGINAL_TYPE_LIST = "list"
ORIGINAL_TYPE_DICT = "dict"

# Coluna padrão usada para dados convertidos
DEFAULT_TEXT_COLUMN = "_texto"
TOKEN_COLUMNS = (
    "_input_tokens",
    "_cached_input_tokens",
    "_output_tokens",
    "_reasoning_tokens",
)

# Entradas convertidas para DataFrame com a coluna interna de texto.
_TEXT_COLUMN_INPUTS = (
    ORIGINAL_TYPE_PANDAS_SERIES,
    ORIGINAL_TYPE_POLARS_SERIES,
    ORIGINAL_TYPE_LIST,
    ORIGINAL_TYPE_DICT,
)
_POLARS_INPUTS = (ORIGINAL_TYPE_POLARS_DF, ORIGINAL_TYPE_POLARS_SERIES)

_READERS = {
    ".xlsx": pd.read_excel,
    ".xls": pd.read_excel,
    ".csv": pd.read_csv,
    ".parquet": pd.read_parquet,
    ".json": pd.read_json,
}
# Tipados como chamável genérico: com **kwargs, o ty não escolhe entre as
# sobrecargas de read_csv e read_excel, que variam com a versão do pandas.
_TEXT_READERS: dict[str, Callable[..., pd.DataFrame]] = {
    ".csv": pd.read_csv,
    ".xlsx": pd.read_excel,
    ".xls": pd.read_excel,
}
# Argumentos de leitura com que o usuário já controla tipos ou NA.
_READ_CONTROLS = ("dtype", "converters", "na_values", "keep_default_na", "na_filter", "usecols")


@dataclass
class ConversionInfo:
    """Informações sobre a conversão de dados para pandas DataFrame."""

    original_type: str
    original_index: pd.Index | list | None = None  # Índice/chaves originais para reconversão
    series_name: Hashable | None = None  # Nome original da Series (se aplicável)
    # DataFrame polars de entrada, de onde a saída tira as colunas que a execução não escreve
    polars_frame: PolarsDataFrame | None = None


def parse_json(resposta: str) -> dict:
    """Extrai e faz parse de JSON da resposta de um LLM.

    Args:
        resposta: Texto da resposta do LLM.

    Returns:
        Dicionário com os dados do JSON.

    Raises:
        ValueError: Se o JSON não puder ser extraído ou decodificado.
    """
    content = str(resposta)

    # Tentar extrair JSON de bloco de código markdown
    match = re.search(r"```json\n(.*?)\n```", content, re.DOTALL)
    if match:
        json_string = match.group(1).strip()
    else:
        # Tentar extrair entre chaves
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_string = content[start : end + 1]
        else:
            json_string = content.strip()

    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        msg = f"Falha ao decodificar JSON. Erro: {e}. Resposta: '{json_string[:200]}'..."
        raise ValueError(msg) from e


def to_pandas(data: object) -> tuple[pd.DataFrame, ConversionInfo]:
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
        return data.to_pandas(), ConversionInfo(
            original_type=ORIGINAL_TYPE_POLARS_DF, polars_frame=data
        )

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
        # tupleize_cols=False: com chaves-tupla, o pd.Index padrão criaria um
        # MultiIndex, e a saída mudaria de forma para o usuário.
        df = pd.DataFrame({DEFAULT_TEXT_COLUMN: values}, index=pd.Index(keys, tupleize_cols=False))
        return df, ConversionInfo(
            original_type=ORIGINAL_TYPE_DICT,
            original_index=keys,
        )

    msg = (
        f"Tipo não suportado: {type(data).__name__}. "
        "Use pandas.DataFrame, polars.DataFrame, pandas.Series, polars.Series, list ou dict."
    )
    raise TypeError(msg)


def from_pandas(
    df: pd.DataFrame,
    conversion_info: ConversionInfo | bool,  # noqa: FBT001 (bool aceito por retrocompatibilidade)
    status_col: str = "_dataframeit_status",
    model_columns: Mapping[str, object] = types.MappingProxyType({}),
) -> pd.DataFrame | PolarsDataFrame:
    """Converte DataFrame pandas de volta para o formato original.

    Remove automaticamente as colunas internas de controle (_dataframeit_status
    e _error_details) se não houver erros nem linhas pendentes.

    Args:
        df: DataFrame pandas processado.
        conversion_info: ConversionInfo com metadados da conversão original,
                        ou bool para retrocompatibilidade (True = era polars).
        status_col: Nome da coluna de status (o `status_column` do usuário).
        model_columns: Anotação de cada campo do modelo que a execução gravou, pelo
                        nome. Na saída polars, só eles e as colunas de controle são
                        convertidos do pandas, e a anotação entra em `_to_polars`.

    Returns:
        Dados no formato original (DataFrame, Series, list ou dict).
    """
    # Retrocompatibilidade: aceita bool (was_polars)
    if isinstance(conversion_info, bool):
        was_polars = conversion_info
        conversion_info = ConversionInfo(
            original_type=ORIGINAL_TYPE_POLARS_DF if was_polars else ORIGINAL_TYPE_PANDAS_DF
        )

    df = _drop_idle_control_columns(df, status_col)

    # Series, list e dict viram DataFrame sem a coluna interna de texto; a
    # pandas.Series recupera o índice original.
    original_type = conversion_info.original_type
    if original_type in _TEXT_COLUMN_INPUTS and DEFAULT_TEXT_COLUMN in df.columns:
        df = df.drop(columns=[DEFAULT_TEXT_COLUMN])
    if original_type == ORIGINAL_TYPE_PANDAS_SERIES and conversion_info.original_index is not None:
        df.index = conversion_info.original_index

    df = _reorder_columns(df, status_col)
    if original_type in _POLARS_INPUTS and pl is not None:
        untouched = _untouched_columns(conversion_info.polars_frame, status_col, model_columns)
        columns = []
        as_object = []
        for col in df.columns:
            if col in untouched:
                columns.append(untouched[col])
                continue
            values = _enums_by_value(df[col])
            # As chamadas ao provedor já foram feitas, e a coluna que o polars ou o
            # pyarrow recusam, seja qual for a exceção (ValueError, TypeError,
            # OverflowError, ComputeError), sai Object em vez de derrubar a saída.
            # O pânico do polars, como o do Decimal acima de 38 dígitos, que o
            # pyarrow entrega em 256 bits, herda de BaseException e é capturado
            # pelo nome; capturar BaseException engoliria o KeyboardInterrupt. O
            # polars ainda escreve a mensagem do pânico no stderr.
            try:
                columns.append(_to_polars(values, model_columns.get(col)))
            except (Exception, pl.exceptions.PanicException):  # noqa: BLE001 (a recusa muda de tipo com os valores)
                # Há versões do polars em que o construtor de Series inspeciona os
                # valores mesmo com dtype=pl.Object: quando o primeiro valor não nulo
                # é membro de Enum, troca cada valor pelo `.value`, e levanta erro no
                # que não o tem, como o None. O map_elements guarda o que a função
                # devolve, sem inspecionar.
                items = values.tolist()
                positions = pl.Series(col, range(len(items)), dtype=pl.Int64)
                columns.append(positions.map_elements(items.__getitem__, return_dtype=pl.Object))
                as_object.append(col)
        if as_object:
            warnings.warn(
                f"As colunas {as_object} têm valores de tipos que o polars não reúne numa "
                "coluna tipada e saem com o tipo Object, com os objetos Python que o modelo "
                "devolveu, exceto o Enum fora de lista ou dict, que sai pelo valor. Converta-as "
                "antes de gravar em arquivo.",
                UserWarning,
                stacklevel=3,
            )
        return pl.DataFrame(columns)
    return df


def _drop_idle_control_columns(df: pd.DataFrame, status_col: str) -> pd.DataFrame:
    """Remove as colunas de status e de erro quando não há erro nem linha pendente.

    A linha sem status é a que uma interrupção deixou para `resume=True`, e sem
    a coluna ela não se distingue das processadas.
    """
    error_col = "_error_details"
    if status_col not in df.columns:
        return df
    has_errors = (df[status_col] == "error").any()
    has_pending = df[status_col].isna().any()
    has_error_details = error_col in df.columns and df[error_col].notna().any()
    if has_errors or has_pending or has_error_details:
        return df
    return df.drop(columns=[c for c in [status_col, error_col] if c in df.columns])


def _enums_by_value(values: pd.Series) -> pd.Series:
    """Troca o Enum pelo valor numa coluna de objetos.

    O Arrow não tipa o membro de um Enum, e é pelo valor que o checkpoint o
    grava. O Enum dentro de lista ou dict fica como está, e a coluna sai Object.
    A Series é montada com dtype=object porque `Series.map` voltaria a inferir
    o tipo e levaria o inteiro com None a float.
    """
    if values.dtype != object:
        return values
    return pd.Series(
        [value.value if isinstance(value, Enum) else value for value in values],
        name=values.name,
        dtype=object,
    )


def _json_default(value: object) -> object:
    """O que json.dumps não serializa: numpy pelo tolist, Enum pelo valor, o resto texto.

    O tolist dá lista ao array e número ou bool do Python ao escalar do numpy,
    que pelo texto voltaria "5" ou "True". O Enum vai pelo valor porque é o que
    a validação do campo aceita de volta; o texto dele ("Cor.AZUL") não valida.
    """
    if isinstance(value, Enum):
        return value.value
    tolist = getattr(value, "tolist", None)
    return tolist() if callable(tolist) else str(value)


def _without_optional(annotation: object) -> object:
    """A anotação sem o Optional e sem o Annotated de dentro dele.

    A união de mais de um tipo além de None fica como está.
    """
    args = [arg for arg in get_args(annotation) if arg is not type(None)]
    if get_origin(annotation) in (typing.Union, types.UnionType) and len(args) == 1:
        annotation = args[0]
    if get_origin(annotation) is typing.Annotated:
        annotation = get_args(annotation)[0]
    return annotation


def _to_polars(values: pd.Series, annotation: object) -> PolarsSeries:
    """Converte a coluna para polars; `annotation` é a do campo do modelo, ou None.

    O campo dict sai como texto JSON, como o checkpoint o grava. Como Struct, a
    coluna teria em cada linha as chaves de todas, e a chave ausente não se
    distinguiria da chave com valor nulo. O dict com chave que o JSON não
    aceita, como um Enum, levanta TypeError, e `from_pandas` devolve a coluna
    como Object.

    A coluna sem valor, toda nula ou só com listas vazias, não dá ao polars de
    onde inferir o tipo, e o schema da saída mudaria com as respostas. Ela
    recebe o tipo da anotação quando esta é um dos tipos de `scalars` ou uma
    lista de um deles. A busca em `scalars` é por identidade, e não por chave
    de dict, porque a anotação pode não ter hash, como o item de
    list[Annotated[int, {...}]].
    """
    # Só é chamada com o polars instalado; o Any dispensa conferir o None a cada uso.
    polars: Any = pl
    scalars = (
        (str, polars.String),
        (int, polars.Int64),
        (float, polars.Float64),
        (bool, polars.Boolean),
        (datetime.date, polars.Date),
        (datetime.datetime, polars.Datetime),
    )
    annotation = _without_optional(annotation)
    origin, args = get_origin(annotation), get_args(annotation)
    if (origin or annotation) is dict:
        values = pd.Series(
            [
                json.dumps(value, ensure_ascii=False, default=_json_default)
                if isinstance(value, dict)
                else value
                for value in values
            ],
            name=values.name,
        )
        # A coluna de texto JSON sem nenhuma resposta segue a regra do campo str.
        annotation = str
    series = polars.from_pandas(values)
    item = args[0] if origin is list and args else annotation
    dtype = next((dtype for python_type, dtype in scalars if item is python_type), None)
    if dtype is None:
        return series
    # Da lista vazia o explode tira um nulo ou nada, conforme a versão do polars,
    # e a coluna que não é lista ele devolve como está.
    leaves = series.explode()
    if leaves.null_count() < len(leaves):
        return series
    return series.cast(polars.List(dtype) if origin is list else dtype)


def _untouched_columns(
    source: PolarsDataFrame | None,
    status_col: str,
    model_columns: Collection[str],
) -> dict[str, PolarsSeries]:
    """Colunas da entrada polars que a execução não escreve, pelo nome.

    São as colunas do usuário que não são campo do modelo. A saída as tira de
    `source`, o DataFrame polars de entrada: convertidas de volta do pandas,
    elas trocariam de tipo (Date por Datetime, Enum por Categorical) e trariam
    arredondado o inteiro acima de 2**53 que tem nulo na coluna ou está dentro
    de Struct e List, porque o `to_pandas()` do polars o leva a float. As linhas da saída estão na ordem das de `source`, porque a
    execução grava por rótulo e não remove nem reordena linhas.

    Sem `source` (entrada polars.Series, que só traz o texto), todas as colunas
    são da execução.
    """
    if source is None:
        return {}
    return {
        col: source[col]
        for col in source.columns
        if col not in model_columns and _column_rank(col, status_col) == _USER_COLUMN
    }


# Posição da coluna do usuário em _column_rank; as demais são as que a execução acrescenta.
_USER_COLUMN = (0, 0)


def _column_rank(col: Hashable, status_col: str) -> tuple[int, int]:
    """Família da coluna e posição dentro dela, na ordem em que a saída as traz.

    1. Colunas do usuário (originais + campos do modelo)
    2. Colunas de trace (_trace e _trace_*)
    3. Colunas de busca (_search_credits)
    4. Colunas de tokens, na ordem de TOKEN_COLUMNS
    5. Colunas de controle (status e _error_details)
    """
    if col == "_trace" or str(col).startswith("_trace_"):
        return (1, 0)
    if col == "_search_credits":
        return (2, 0)
    if col in TOKEN_COLUMNS:
        return (3, TOKEN_COLUMNS.index(col))
    if col in (status_col, "_error_details"):
        return (4, 0)
    return _USER_COLUMN


def _reorder_columns(df: pd.DataFrame, status_col: str = "_dataframeit_status") -> pd.DataFrame:
    """Reordena as colunas para que as da execução fiquem no final.

    A ordem é a de `_column_rank`; dentro de cada família, menos a de tokens,
    as colunas ficam na ordem em que estavam. A seleção é por posição, para que
    um rótulo repetido não traga a coluna mais de uma vez.
    """
    order = sorted(range(df.shape[1]), key=lambda i: _column_rank(df.columns[i], status_col))
    return df.iloc[:, order]


# =============================================================================
# NORMALIZAÇÃO DE ESTRUTURAS PYTHON (listas, dicts, tuples)
# =============================================================================


def is_complex_type(field_type: object) -> bool:
    """Verifica se um tipo é complexo (list, dict, tuple ou modelo Pydantic).

    Args:
        field_type: Tipo a verificar (pode ser tipo simples ou genérico).

    Returns:
        True se o tipo for list, dict, tuple ou subclasse de BaseModel, direto
        ou dentro de Optional/Union. Um modelo aninhado conta como complexo
        porque a célula guarda o seu model_dump, um dict.
    """
    origin = get_origin(field_type)

    # O Pydantic tira o Annotated só do nível de cima da anotação; dentro de
    # Optional ele fica, e o tipo que importa é o primeiro argumento.
    if origin is typing.Annotated:
        return is_complex_type(get_args(field_type)[0])

    # Tipos genéricos: list[str], dict[str, int], tuple[int, str], etc.
    if origin in (list, dict, tuple):
        return True

    # Union types (Optional, Union) - verificar os argumentos internos
    # typing.Union para sintaxe Union[X, Y] e Optional[X]
    if origin is typing.Union:
        args = get_args(field_type)
        return any(is_complex_type(arg) for arg in args if arg is not type(None))

    # types.UnionType para sintaxe X | Y
    if isinstance(field_type, types.UnionType):
        args = get_args(field_type)
        return any(is_complex_type(arg) for arg in args if arg is not type(None))

    # Tipos diretos
    if field_type in (list, dict, tuple):
        return True

    # Modelo aninhado: a linha guarda o model_dump, um dict
    return bool(isinstance(field_type, type) and issubclass(field_type, BaseModel))


def get_complex_fields(pydantic_model: type[BaseModel]) -> set:
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


def accepts_only_text(field_type: object) -> bool:
    """True se o tipo só aceita texto: str, Literal ou Enum de textos, ou esses com None."""
    if field_type is str:
        return True
    if isinstance(field_type, type) and issubclass(field_type, Enum):
        return all(isinstance(member.value, str) for member in field_type)
    origin = get_origin(field_type)
    if origin is typing.Annotated:  # dentro de Optional, como em is_complex_type
        return accepts_only_text(get_args(field_type)[0])
    if origin is typing.Literal:
        return all(isinstance(arg, str) for arg in get_args(field_type))
    if origin is typing.Union or isinstance(field_type, types.UnionType):
        args = [arg for arg in get_args(field_type) if arg is not type(None)]
        return bool(args) and all(accepts_only_text(arg) for arg in args)
    return False


def get_text_fields(pydantic_model: type[BaseModel]) -> set:
    """Nomes dos campos que só aceitam texto."""
    return {
        field_name
        for field_name, field_info in pydantic_model.model_fields.items()
        if accepts_only_text(field_info.annotation)
    }


def normalize_value(value: Any) -> Any:  # noqa: ANN401, PLR0911 (célula de qualquer tipo; cada forma sai cedo)
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
    if stripped.startswith(("[", "{")):
        try:
            return json.loads(stripped)
        except (json.JSONDecodeError, ValueError, RecursionError):
            pass
        # Checkpoints textuais antigos gravavam o repr Python ("['a', 'b']").
        # literal_eval só aceita literais, nunca executa código.
        try:
            parsed = ast.literal_eval(stripped)
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            return value
        if isinstance(parsed, (list, dict, tuple)):
            return parsed

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
    model: type[BaseModel] | None = None,
    normalize: bool = True,  # noqa: FBT001, FBT002 (posicional na API pública)
    **kwargs: Any,
) -> pd.DataFrame:
    """Carrega um DataFrame de arquivo e normaliza estruturas Python automaticamente.

    Esta função é útil para carregar dados que foram previamente processados
    pelo dataframeit e salvos em arquivo. Ela converte automaticamente strings
    JSON de volta para listas, dicionários e outras estruturas Python.

    Args:
        path: Caminho do arquivo (suporta .xlsx, .xls, .csv, .parquet, .json).
        model: Modelo Pydantic opcional. Se fornecido, apenas as colunas que
               correspondem a campos complexos do modelo serão normalizadas.
               Com ele, a coluna de campo int ou Optional[int] lida de CSV ou
               parquet volta como object, com o int exato e None na célula
               vazia; dtype, converters, na_values, keep_default_na,
               na_filter, usecols, dtype_backend, index_col, parse_dates e
               engine desligam essa releitura.
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
    if not Path(path).exists():
        msg = f"Arquivo não encontrado: {path}"
        raise FileNotFoundError(msg)

    # Detectar formato pelo sufixo
    ext = Path(path.lower()).suffix

    reader = _READERS.get(ext)
    if reader is None:
        msg = f"Formato '{ext}' não suportado. Use: .xlsx, .xls, .csv, .parquet ou .json"
        raise ValueError(msg)
    df = reader(path, **kwargs)

    if model is not None:
        _reread_text_columns(df, path, ext, model, kwargs)
        _reread_int_columns(df, path, ext, model, kwargs)

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


def _reread_text_columns(
    df: pd.DataFrame, path: str, ext: str, model: type[BaseModel], kwargs: dict
) -> None:
    """Relê como texto cru os campos de texto do modelo, in-place.

    Sem isso, "2023" volta como número e "N/A" ou "NA" viram ausência, e a
    retomada acusa ou troca o valor. Só a célula vazia vira ausência. A
    releitura fica de fora quando o usuário já controla tipos ou NA.
    """
    reader = _TEXT_READERS.get(ext)
    if reader is None or any(k in kwargs for k in _READ_CONTROLS):
        return
    text_columns = [f for f in get_text_fields(model) if f in df.columns]
    if not text_columns:
        return
    raw = reader(path, usecols=text_columns, dtype=str, na_filter=False, **kwargs)
    for col in text_columns:
        df[col] = raw[col].mask(raw[col] == "")


def _accepts_only_int(field_type: object) -> bool:
    """True se o tipo é int, ou int com None."""
    if field_type is int:
        return True
    origin = get_origin(field_type)
    if origin is typing.Annotated:  # dentro de Optional, como em is_complex_type
        return _accepts_only_int(get_args(field_type)[0])
    if origin is typing.Union or isinstance(field_type, types.UnionType):
        args = [arg for arg in get_args(field_type) if arg is not type(None)]
        return bool(args) and all(_accepts_only_int(arg) for arg in args)
    return False


_INT_TEXT = re.compile(r"-?\d+")
# Além dos argumentos com que o usuário controla tipos ou NA, os que mudam o
# que a releitura acha no arquivo: index_col tira a coluna do índice do usecols
# dela, parse_dates pede coluna que ela não lê, e o engine pyarrow ignora o
# na_filter e devolve float na célula vazia.
_INT_REREAD_OFF = (*_READ_CONTROLS, "dtype_backend", "index_col", "parse_dates", "engine")


def _read_as_rereadable(column: pd.Series) -> bool:
    """True se o leitor devolveu a coluna num tipo que não guarda o int exato.

    Float64 é a coluna com int e célula vazia; texto, a do CSV com int acima
    do Int64 e célula vazia; UInt64, a do parquet que o checkpoint grava com
    esse int, que a releitura devolve como int e None, como as demais.
    """
    return (
        is_float_dtype(column)
        or is_string_dtype(column.dtype)
        or isinstance(column.dtype, pd.UInt64Dtype)
    )


def _int_or_read(text: str, read: object) -> object:
    """O int que o texto da célula do CSV guarda, None se ela é ausente, ou o valor lido.

    O texto vazio vale como ausente também quando o leitor não o trocou por
    NaN, como na coluna com int acima do Int64, que ele devolve em texto cru.
    """
    if _INT_TEXT.fullmatch(text):
        return int(text)
    if text == "" or pd.isna(read):
        return None
    return read


def _reread_int_columns(
    df: pd.DataFrame, path: str, ext: str, model: type[BaseModel], kwargs: dict
) -> None:
    """Relê como int exato os campos int do modelo, do CSV e do parquet, in-place.

    O leitor do pandas devolve float64 numa coluna com int e célula vazia, como
    a de um checkpoint com linha pendente, e o float arredonda o int acima de
    2**53; com int acima do Int64, o CSV volta como texto. A coluna relida
    passa a object, com o int que o arquivo guarda como inteiro e None na
    célula que o leitor devolveu ausente, seja vazia ou "NA"; a célula que
    guarda outra coisa, como 7.5, fica como lida. O XLSX guarda número como
    double e fica de fora, porque o int que ele devolve já pode vir
    arredondado. Também ficam de fora os argumentos de _INT_REREAD_OFF e o
    leitor que não devolve DataFrame, como o CSV com chunksize e o XLSX com
    sheet_name=None.
    """
    if (
        ext not in (".csv", ".parquet")
        or not isinstance(df, pd.DataFrame)
        or any(k in kwargs for k in _INT_REREAD_OFF)
    ):
        return
    columns = [
        f
        for f, info in model.model_fields.items()
        if f in df.columns and _accepts_only_int(info.annotation) and _read_as_rereadable(df[f])
    ]
    if not columns:
        return
    if ext == ".parquet":
        # As colunas pedidas pelo usuário dão lugar às desta releitura.
        options = {k: v for k, v in kwargs.items() if k != "columns"}
        raw = pd.read_parquet(path, columns=columns, dtype_backend="numpy_nullable", **options)
        exact = {
            col: [None if pd.isna(v) else int(v) for v in raw[col]]
            for col in columns
            if is_integer_dtype(raw[col])
        }
    else:
        raw = pd.read_csv(path, usecols=columns, dtype=str, na_filter=False, **kwargs)
        exact = {
            col: [_int_or_read(text, read) for text, read in zip(raw[col], df[col], strict=True)]
            for col in columns
        }
    for col, values in exact.items():
        if any(type(v) is int for v in values):
            df[col] = pd.Series(values, index=df.index, dtype=object)


def _normalize_all_json_columns(df: pd.DataFrame) -> None:
    """Normaliza todas as colunas do DataFrame que parecem conter JSON.

    Modifica o DataFrame in-place.

    Args:
        df: DataFrame a normalizar.
    """
    for col in df.columns:
        if not is_string_dtype(df[col].dtype):
            continue

        # Verificar se algum valor parece JSON
        sample = df[col].dropna().head(10)
        has_json = any(isinstance(v, str) and v.strip().startswith(("[", "{")) for v in sample)

        if has_json:
            df[col] = df[col].apply(normalize_value)


# =============================================================================
# EXTRAÇÃO DE MODELOS PYDANTIC ANINHADOS
# =============================================================================


def is_list_of_pydantic_model(field_type: object) -> tuple:
    """Verifica se um tipo é List[Model] ou Optional[List[Model]].

    Args:
        field_type: Anotação de tipo a analisar.

    Returns:
        Tupla (is_list, inner_model):
        - is_list: True se o tipo é List[PydanticModel]
        - inner_model: O modelo Pydantic interno (ou None)

    Examples:
        >>> class Inner(BaseModel):
        ...     x: str
        >>> is_list_of_pydantic_model(List[Inner])
        (True, <class 'Inner'>)
        >>> is_list_of_pydantic_model(Optional[List[Inner]])
        (True, <class 'Inner'>)
        >>> is_list_of_pydantic_model(Inner)
        (False, None)
        >>> is_list_of_pydantic_model(List[str])
        (False, None)
    """
    origin = get_origin(field_type)
    args = get_args(field_type)

    # Caso 1: Diretamente List[Model]
    inner_model = _list_item_model(origin, args)
    if inner_model is not None:
        return (True, inner_model)

    # Caso 2: Optional[List[Model]] ou Union[List[Model], None], com typing.Union
    # ou com types.UnionType (X | None)
    if origin is typing.Union or isinstance(field_type, types.UnionType):
        for arg in args:
            if arg is type(None):
                continue
            inner_model = _list_item_model(get_origin(arg), get_args(arg))
            if inner_model is not None:
                return (True, inner_model)

    return (False, None)


def _list_item_model(origin: object, args: tuple) -> type[BaseModel] | None:
    """Modelo Pydantic dos itens quando origin e args descrevem list[Model]."""
    if origin is list and args:
        inner_type = args[0]
        if isinstance(inner_type, type) and issubclass(inner_type, BaseModel):
            return inner_type
    return None


def _lookup_forward_ref(name: str, owner: type[BaseModel]) -> type | None:
    """Resolve o nome de uma referência adiantada no contexto do modelo dono.

    O Pydantic resolve `List['X']`, mas deixa a string crua em `list['X']`.
    A busca segue a ordem de visibilidade do nome: o próprio modelo
    (auto-referência), o namespace em que ele foi definido (classes locais de
    função) e o módulo. O namespace de definição guarda weakrefs do Pydantic
    para modelos, e só esses são desembrulhados.
    """
    if name == owner.__name__:
        return owner

    namespaces = (
        getattr(owner, "__pydantic_parent_namespace__", None) or {},
        vars(sys.modules[owner.__module__]) if owner.__module__ in sys.modules else {},
    )
    for namespace in namespaces:
        value = namespace.get(name)
        if value is not None and type(value).__name__.endswith("WeakRef"):
            value = value()
        if isinstance(value, type):
            return value
    return None


def resolve_forward_refs(  # noqa: PLR0911 (uma saída por forma de anotação)
    annotation: Any,  # noqa: ANN401 (anotação de tipo arbitrária)
    owner: type[BaseModel],
) -> Any:  # noqa: ANN401
    """Devolve a anotação com as referências adiantadas trocadas pelas classes.

    Referência que não se resolve fica como está. A forma de cada tipo é
    preservada: `typing.Union` continua `typing.Union` e `X | Y` continua
    `types.UnionType`, porque os helpers deste módulo tratam os dois em ramos
    distintos.
    """
    if isinstance(annotation, str):
        return _lookup_forward_ref(annotation, owner) or annotation
    if isinstance(annotation, typing.ForwardRef):
        return _lookup_forward_ref(annotation.__forward_arg__, owner) or annotation

    origin = get_origin(annotation)
    args = get_args(annotation)
    # Em Literal os argumentos são valores, e uma string ali nunca é referência.
    if origin is None or origin is typing.Literal or not args:
        return annotation
    # Em Annotated só o primeiro argumento é tipo; os metadados (Field, texto)
    # voltam intactos, e um texto com o nome de um modelo não vira a classe.
    if origin is typing.Annotated:
        inner = resolve_forward_refs(args[0], owner)
        if inner is args[0]:
            return annotation
        return typing.Annotated[(inner, *annotation.__metadata__)]

    resolved = tuple(resolve_forward_refs(arg, owner) for arg in args)
    if resolved == args:
        return annotation
    if isinstance(annotation, types.UnionType):
        return functools.reduce(operator.or_, resolved)
    if origin is typing.Union:
        return typing.Union[resolved]  # noqa: UP007 (preserva a forma typing.Union)
    if isinstance(annotation, types.GenericAlias):
        return types.GenericAlias(origin, resolved)
    return annotation.copy_with(resolved)


def get_nested_pydantic_models(field_type: object) -> list:
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
    models = []
    origin = get_origin(field_type)

    # Caso 1: É diretamente um modelo Pydantic
    if isinstance(field_type, type) and issubclass(field_type, BaseModel):
        models.append(field_type)
        return models

    # Caso 2: Tipos genéricos (List[X], Optional[X], Union[X, Y], X | Y etc.).
    # get_origin também reconhece types.UnionType, então `X | None` cai aqui.
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

    return models
