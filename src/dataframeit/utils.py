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
import collections.abc
import datetime
import json
import math
import numbers
import re
import sys
import types
import typing
import warnings
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_args, get_origin

import pandas as pd
from pandas.api.types import is_integer_dtype, is_string_dtype
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


def to_pandas(
    data: object,
    model_columns: Mapping[str, object] = types.MappingProxyType({}),
) -> tuple[pd.DataFrame, ConversionInfo]:
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
        model_columns: Anotação de cada campo do modelo, pelo nome. Na entrada
            polars, a coluna de campo que admite `int` chega ao pandas com os
            valores do polars, e a de campo float, com None no nulo, como em
            `_python_values`.

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
        df = _python_values(data, model_columns)
        # O to_pandas entrega a lista de uma coluna List ou Array, e a de dentro
        # de um Struct, como array do numpy, que muda o valor: a data vira
        # datetime64, o datetime com fuso passa a UTC sem o fuso, e o inteiro
        # ao lado de um nulo vira float. A retomada validaria essa forma, e a
        # linha processada de uma saída polars voltaria recusada ou com outra
        # hora. O to_list do polars dá os objetos Python que o modelo devolveu.
        for col, dtype in data.schema.items():
            if isinstance(dtype, (pl.List, pl.Array, pl.Struct)):
                df[col] = pd.Series(data[col].to_list())
        return df, ConversionInfo(original_type=ORIGINAL_TYPE_POLARS_DF, polars_frame=data)

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


def _python_values(data: PolarsDataFrame, model_columns: Mapping[str, object]) -> pd.DataFrame:
    """Converte o DataFrame polars para pandas, sem confundir o nulo com o valor em dois campos.

    O `to_pandas()` do polars leva a coluna inteira com nulo a float64, que
    arredonda o inteiro acima de 2**53. Na coluna de campo que admite `int`
    (`_admits_int`), a execução sem interrupção grava inteiros Python e None, e
    a coluna relida vem assim, em dtype object. A coluna sem nulo também é
    trocada, porque o int64 do numpy não passa num modelo com strict=True, que
    acusaria a linha processada como incompatível. Com outra anotação, o
    inteiro relido não diz o mesmo que o valor validado e fica na coluna: o 1
    de um campo bool e o inteiro grande de um campo float saem numa coluna
    Object ao lado das respostas novas, e o bool com strict=True é acusado como
    incompatível.

    O mesmo `to_pandas()` põe NaN no nulo da coluna de números, e o NaN que o
    modelo devolveu num campo float não se distinguiria da linha pendente. Na
    coluna de campo que guarda o NaN (`_keeps_nan`) e não admite int, o nulo
    volta None, e os demais valores ficam como o `to_pandas()` os dá.
    """
    df = data.to_pandas()
    for col, annotation in model_columns.items():
        if col not in data.columns:
            continue
        if _admits_int(annotation):
            df[col] = pd.Series(data[col].to_list(), dtype=object)
        elif _keeps_nan(annotation):
            df[col] = df[col].astype(object).where(data[col].is_not_null().to_numpy(), None)
    return df


def _members(annotation: object) -> list:
    """Os membros da união, ou a própria anotação, cada um por `_without_optional`."""
    annotation = _without_optional(annotation)
    if get_origin(annotation) in (typing.Union, types.UnionType):
        return [_without_optional(arg) for arg in get_args(annotation)]
    return [annotation]


def _admits_int(annotation: object) -> bool:
    """Se a anotação é `int`, o `Literal` com algum int, ou a união com um deles entre os membros.

    O bool é subclasse de int, e o `Literal[True]` não conta.
    """
    return any(
        member is int
        or (
            get_origin(member) is typing.Literal
            and any(type(arg) is int for arg in get_args(member))
        )
        for member in _members(annotation)
    )


def _keeps_nan(annotation: object) -> bool:
    """Se a coluna do campo guarda o NaN como valor: a anotação admite float e não é estrutura.

    O NaN é um float que o modelo devolve, e ao lado dele o nulo da linha
    pendente ou da resposta None é None, desde que a coluna não passe por
    nenhuma conversão que troque um pelo outro. A entrada polars restaura o
    None (`_python_values`), e a saída polars não troca o NaN por nulo
    (`_to_polars`). A estrutura (lista, dict, tupla ou submodelo) fica de fora,
    porque a retomada normaliza a coluna dela com `Series.apply`, que reinfere
    o tipo e leva o None de uma coluna de números a NaN. O `Any` também fica,
    porque a coluna dele chega da entrada polars com qualquer tipo, e o
    `to_pandas` põe NaN no nulo de uma coluna de texto.
    """
    return any(member is float for member in _members(annotation)) and not is_complex_type(
        annotation
    )


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
    """O que json.dumps não serializa: conjunto e deque como lista, numpy pelo tolist, Enum pelo valor.

    O conjunto vira lista em ordem fixa, a do texto JSON de cada item. A ordem
    em que um set de textos é percorrido muda de um processo para outro, com o
    hash dos textos, e a lista que a retomada relê do checkpoint só diz o mesmo
    que o conjunto validado (`_same_json`) quando os dois saem na mesma ordem.
    O texto JSON também ordena itens de tipos que não se comparam, como 1 e "a".

    A sequência que não é lista nem tupla, como o deque, vira lista na ordem
    dela; os bytes não são coleção para a célula (`_TEXT_CLASSES`) e vão como
    texto.

    O tolist dá lista ao array e número ou bool do Python ao escalar do numpy,
    que pelo texto voltaria "5" ou "True". O Enum vai pelo valor porque é o que
    a validação do campo aceita de volta; o texto dele ("Cor.AZUL") não valida.
    O resto vai como texto.
    """
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=lambda item: json.dumps(item, default=_json_default))
    if isinstance(value, collections.abc.Sequence) and not isinstance(value, _TEXT_CLASSES):
        return list(value)
    if isinstance(value, Enum):
        return value.value
    tolist = getattr(value, "tolist", None)
    return tolist() if callable(tolist) else str(value)


def _resolved(annotation: object) -> object:
    """A anotação sem NewType e sem apelido `type`, pelo tipo a que eles se referem.

    É o ponto único em que os helpers deste módulo que inspecionam anotação
    resolvem os dois, a cada nível, antes de olhar a origem ou os argumentos.
    O NewType guarda o tipo em `__supertype__`, e o apelido, em `__value__`. O
    apelido é reconhecido pelo nome da classe, porque o de `type X = ...` e o
    `TypeAliasType` do typing_extensions são classes diferentes com esse mesmo
    nome, e nenhum é instância da classe do outro. O atributo `__value__` não
    basta, porque uma classe qualquer pode ter um com esse nome.

    O apelido genérico usado com argumentos, como `Opcional[int]` de
    `type Opcional[T] = T | None`, tem o apelido como origem, e o `__value__`
    dele traz os parâmetros por substituir (ver `_alias_applied`).
    """
    while True:
        if isinstance(annotation, typing.NewType):
            annotation = annotation.__supertype__
        elif type(annotation).__name__ == "TypeAliasType":
            annotation = annotation.__value__  # ty: ignore[unresolved-attribute]
        elif type(get_origin(annotation)).__name__ == "TypeAliasType":
            annotation = _alias_applied(get_origin(annotation), get_args(annotation))
        else:
            return annotation


def _alias_applied(alias: object, args: tuple) -> object:
    """O `__value__` do apelido genérico com cada parâmetro trocado pelo argumento.

    Cada parâmetro de `__type_params__` vale o argumento na mesma posição. A
    substituição segue a ordem de `__parameters__` do valor, que é a ordem em
    que os parâmetros aparecem nele e pode não ser a da declaração: em
    `type Par[K, V] = dict[V, K]`, `Par[int, str]` é `dict[str, int]`. O valor
    que é o próprio parâmetro não tem `__parameters__` e é trocado direto; o
    que não usa parâmetro nenhum fica como está.
    """
    value = alias.__value__  # ty: ignore[unresolved-attribute]
    by_param = dict(zip(alias.__type_params__, args, strict=False))  # ty: ignore[unresolved-attribute]
    if isinstance(value, typing.TypeVar):
        return by_param.get(value, value)
    parameters = getattr(value, "__parameters__", ())
    if not parameters:
        return value
    return value[tuple(by_param.get(param, param) for param in parameters)]


def _without_optional(annotation: object) -> object:
    """A anotação sem o Optional e sem o Annotated de dentro dele.

    A união de mais de um tipo além de None fica como está. O NewType e o
    apelido `type` são resolvidos por `_resolved` antes de cada passo e no
    fim, e valem também dentro do Optional e do Annotated.
    """
    annotation = _resolved(annotation)
    args = [arg for arg in get_args(annotation) if arg is not type(None)]
    if get_origin(annotation) in (typing.Union, types.UnionType) and len(args) == 1:
        annotation = _resolved(args[0])
    if get_origin(annotation) is typing.Annotated:
        annotation = _resolved(get_args(annotation)[0])
    return annotation


# Data, hora e duração: o pyarrow converte cada um destes tipos nos outros e o
# número em cada um deles. O datetime depois de um date perde a hora, e o int
# depois de um date ou de um timedelta vira data ou duração.
_TEMPORAL = (datetime.date, datetime.datetime, datetime.time, datetime.timedelta)
# Pares que o pyarrow converte um no outro quando o primeiro vem antes: o True
# depois de um float vira 1.0, e o texto depois de um bytes vira bytes.
_CONVERTED_PAIRS = ((float, bool), (bytes, str))


def _mixes_converted_types(annotation: object) -> bool:
    """Se a anotação junta, na união ou na tupla, tipos que o pyarrow converte uns nos outros.

    São os de `_TEMPORAL` com qualquer outro tipo e os pares de
    `_CONVERTED_PAIRS`. A lista ou a tupla é examinada em cada argumento,
    recursivamente: vale para `tuple[Union[date, int], ...]`, para
    `list[tuple[date, int]]` e para listas aninhadas. O Optional, o Annotated,
    o NewType e o apelido `type` são desembrulhados, na anotação e em cada
    membro. A busca é por identidade, porque o membro pode não ter hash.
    """
    annotation = _without_optional(annotation)
    origin, args = get_origin(annotation), get_args(annotation)
    if origin in (list, tuple) and any(_mixes_converted_types(arg) for arg in args):
        return True
    if origin not in (typing.Union, types.UnionType, tuple):
        return False
    members = [_without_optional(arg) for arg in args if arg is not Ellipsis]
    if any(member in _TEMPORAL for member in members) and any(
        member is not members[0] for member in members
    ):
        return True
    return any(
        all(any(member is kind for member in members) for kind in pair) for pair in _CONVERTED_PAIRS
    )


# Classe que representa, em `_group_kinds`, o datetime com fuso.
_AWARE_DATETIME = "datetime com fuso"
# O bool vem antes do int, e o datetime antes do date, porque cada um é
# subclasse do seguinte.
_KINDS = (
    bool,
    int,
    float,
    Decimal,
    str,
    bytes,
    datetime.datetime,
    datetime.date,
    datetime.time,
    datetime.timedelta,
)
_NUMBERS = (int, float, Decimal)
# A classe de `_KINDS` do número que não é subclasse de nenhuma delas, como o
# np.int64, que não é int, e o np.float32, que não é float, pela ABC de
# `numbers` em que o numpy o registra. O pyarrow os converte como os do
# Python: o np.int64 depois de um date vira data. O bool é Integral, mas
# chega aqui já contado como bool, primeiro de `_KINDS`; o np.bool_ não é
# registrado em nenhuma ABC e fica com a própria classe.
_NUMBER_ABCS = ((numbers.Integral, int), (numbers.Real, float))
_TEMPORAL_KINDS = (*_TEMPORAL, _AWARE_DATETIME)
_SEQUENCES = (list, tuple)


def _group_kinds(group: list, classes: set) -> set | None:
    """As classes de `_KINDS` do grupo, ou None se algum valor nenhum tipo do polars guarda.

    A classe do valor vale pela primeira de `_KINDS` de que é subclasse, e o
    pd.Timestamp conta como datetime; o número do numpy fora delas vale pela
    ABC de `numbers` (`_NUMBER_ABCS`), e a classe restante fica com ela mesma. A
    estrutura não entra, porque `_arrow_alters` a abre em outros grupos. O
    nulo também não, porque é nulo em qualquer tipo: o None, o NaT e o NaN, que
    a conversão da entrada polars para pandas põe no nulo da coluna de datas e
    da de números. Na coluna que guarda o NaN como valor (`_keeps_nan`), o nulo
    volta None, e o NaN que sobra ali só junta float a float. Só quatro
    classes são conferidas valor a valor: o datetime, com e sem fuso, o float,
    pelo NaN, o Decimal NaN, que o pyarrow grava nulo, e o time com fuso, que o
    tipo Time não tem.
    """
    kinds = set()
    for cls in classes:
        if cls is type(None) or cls is type(pd.NaT) or issubclass(cls, (dict, *_SEQUENCES)):
            continue
        members = [value for value in group if type(value) is cls]
        if issubclass(cls, datetime.datetime):
            kinds.update(
                _AWARE_DATETIME if value.tzinfo is not None else datetime.datetime
                for value in members
            )
        elif issubclass(cls, float):
            if not all(math.isnan(value) for value in members):
                kinds.add(float)
        elif (issubclass(cls, Decimal) and any(value.is_nan() for value in members)) or (
            issubclass(cls, datetime.time) and any(value.tzinfo is not None for value in members)
        ):
            return None
        else:
            kind = next((kind for kind in _KINDS if issubclass(cls, kind)), None)
            kinds.add(kind or next((k for abc, k in _NUMBER_ABCS if issubclass(cls, abc)), cls))
    return kinds


def _arrow_alters(values: pd.Series) -> bool:
    """Se o pyarrow mudaria algum dos valores ao pôr a coluna num tipo do polars.

    Os valores são agrupados como o pyarrow os tipa: as células da coluna num
    grupo e cada chave de dict, que vira campo de Struct, num grupo. Os itens
    de lista ou tupla ficam no grupo da própria lista, porque o pyarrow não
    reúne lista e escalar num tipo, e a coluna que os misturasse já sairia
    Object. O grupo que junta classes de `_TEMPORAL_KINDS` entre si ou com
    número, ou os dois tipos de um par de `_CONVERTED_PAIRS`, seria tipado pelo
    primeiro valor, e os outros convertidos: o datetime sem fuso depois de um
    com fuso ganharia o fuso dele. Os dicts de um grupo com conjuntos de chaves
    diferentes também contam, porque o Struct teria a união das chaves, e a
    chave ausente sairia com nulo, como `{}` ao lado de `{"a": 1}` sairia
    `{"a": None}`. O valor que nenhum tipo do polars guarda também conta (ver
    `_group_kinds`).

    O grupo é percorrido inteiro de uma vez, pelas classes dos valores, e só
    as classes que pedem conferência valor a valor descem a cada um: com uma
    chamada por valor, a conferência pesava mais que a própria conversão.
    """
    if values.dtype != object:
        return False
    pending: list[tuple[tuple, list]] = [((), values.tolist())]
    while pending:
        path, group = pending.pop()
        classes = set(map(type, group))
        while any(issubclass(cls, _SEQUENCES) for cls in classes):
            group = [
                item
                for value in group
                for item in (value if isinstance(value, _SEQUENCES) else (value,))
            ]
            classes = set(map(type, group))
        if any(issubclass(cls, dict) for cls in classes):
            dicts = [value for value in group if isinstance(value, dict)]
            if len({frozenset(value) for value in dicts}) > 1:
                return True
            by_key: dict[object, list] = {}
            for value in dicts:
                for key, item in value.items():
                    by_key.setdefault(key, []).append(item)
            pending.extend(((*path, key), items) for key, items in by_key.items())
        kinds = _group_kinds(group, classes)
        if kinds is None or _mixes_converted_kinds(kinds):
            return True
    return False


def _mixes_converted_kinds(kinds: set) -> bool:
    """Se as classes de um grupo de `_arrow_alters` seriam convertidas umas nas outras."""
    temporal = kinds.intersection(_TEMPORAL_KINDS)
    return (
        len(temporal) > 1
        or bool(temporal and kinds.intersection(_NUMBERS))
        or any(kinds.issuperset(pair) for pair in _CONVERTED_PAIRS)
    )


def _to_polars(values: pd.Series, annotation: object) -> PolarsSeries:
    """Converte a coluna para polars; `annotation` é a do campo do modelo, ou None.

    O campo dict sai como texto JSON, como o checkpoint o grava. Como Struct, a
    coluna teria em cada linha as chaves de todas, e a chave ausente não se
    distinguiria da chave com valor nulo. O dict com chave que o JSON não
    aceita, como um Enum, levanta TypeError, e `from_pandas` devolve a coluna
    como Object.

    A anotação que junta tipos que o pyarrow converte uns nos outros, como
    date, datetime, time ou timedelta com outro tipo, float com bool e bytes
    com str, também levanta TypeError, antes de olhar os valores. O pyarrow
    tiparia a coluna pelo primeiro valor e converteria os outros sem erro: o
    datetime depois de um date perderia a hora, o int depois de um date
    viraria data, e o True depois de um float viraria 1.0. Os valores que o
    pyarrow mudaria levantam TypeError pela mesma razão, com qualquer anotação
    (ver `_arrow_alters`).

    Na coluna de campo que guarda o NaN como valor (`_keeps_nan`), o NaN sai
    NaN, e não nulo como o `from_pandas` do polars o grava por padrão: nela, o
    nulo da linha pendente e da resposta None é None.

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
    if _mixes_converted_types(annotation):
        msg = f"A anotação {annotation} junta tipos que o pyarrow converte uns nos outros."
        raise TypeError(msg)
    keeps_nan = _keeps_nan(annotation)
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
    if _arrow_alters(values):
        msg = "O pyarrow mudaria valores da coluna ao tipá-la."
        raise TypeError(msg)
    series = polars.from_pandas(values, nan_to_null=not keeps_nan)
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
    arredondado o inteiro acima de 2**53 que tem nulo na coluna, porque o
    `to_pandas()` do polars o leva a float. As linhas da saída estão na ordem
    das de `source`, porque a execução grava por rótulo e não remove nem
    reordena linhas.

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


# Coleções que a célula guarda como estrutura, reconhecidas pela ABC: a lista,
# a tupla, o NamedTuple, o deque, o set, o frozenset, o dict, o TypedDict e as
# subclasses de dict, além das próprias origens abstratas e das versões
# mutáveis delas. O texto e os bytes também são Sequence para o
# collections.abc e ficam de fora pela subclasse, inclusive o Enum de texto: a
# célula guarda o valor inteiro deles.
_STRUCTURE_ABCS = (collections.abc.Sequence, collections.abc.Set, collections.abc.Mapping)
_TEXT_CLASSES = (str, bytes, bytearray)


def _is_structure_class(cls: type) -> bool:
    """Se a classe é coleção que a célula guarda como estrutura (`_STRUCTURE_ABCS`)."""
    return issubclass(cls, _STRUCTURE_ABCS) and not issubclass(cls, _TEXT_CLASSES)


def is_complex_type(field_type: object) -> bool:
    """Verifica se um tipo é estrutura: lista, tupla, conjunto, dict ou modelo Pydantic.

    Args:
        field_type: Tipo a verificar (pode ser tipo simples ou genérico).

    Returns:
        True se o tipo, direto ou dentro de Optional/Union, ou a origem dele,
        quando genérico, é coleção de `_is_structure_class` ou modelo
        Pydantic, que a célula guarda pelo model_dump, um dict. É o que decide
        quais campos o checkpoint grava em JSON e a retomada normaliza de
        volta.
    """
    field_type = _resolved(field_type)
    origin = get_origin(field_type)

    # O Pydantic tira o Annotated só do nível de cima da anotação; dentro de
    # Optional ele fica, e o tipo que importa é o primeiro argumento.
    if origin is typing.Annotated:
        return is_complex_type(get_args(field_type)[0])

    # Union[X, Y], Optional[X] e X | Y: verificar os argumentos internos.
    # Antes do 3.14, get_origin de X | Y é types.UnionType; a partir dele, os
    # dois nomes são o mesmo objeto, e a tupla cobre as duas versões.
    if origin in (typing.Union, types.UnionType):
        args = get_args(field_type)
        return any(is_complex_type(arg) for arg in args if arg is not type(None))

    container = field_type if origin is None else origin
    return isinstance(container, type) and (
        issubclass(container, BaseModel) or _is_structure_class(container)
    )


def get_complex_fields(pydantic_model: type[BaseModel]) -> set:
    """Retorna os nomes dos campos que são estruturas (ver `is_complex_type`).

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
    """True se o tipo só aceita texto: str, Literal ou Enum de textos, ou esses com None.

    O None vale também dentro do Literal, como em Literal["NA", None], que diz o
    mesmo que Optional[Literal["NA"]].
    """
    field_type = _resolved(field_type)
    if field_type is str:
        return True
    if isinstance(field_type, type) and issubclass(field_type, Enum):
        return all(isinstance(member.value, str) for member in field_type)
    origin = get_origin(field_type)
    if origin is typing.Annotated:  # dentro de Optional, como em is_complex_type
        return accepts_only_text(get_args(field_type)[0])
    if origin is typing.Literal:
        args = [arg for arg in get_args(field_type) if arg is not None]
        return bool(args) and all(isinstance(arg, str) for arg in args)
    if origin is typing.Union or isinstance(field_type, types.UnionType):
        args = [arg for arg in get_args(field_type) if arg is not type(None)]
        return bool(args) and all(accepts_only_text(arg) for arg in args)
    return False


def _admits_none(field_type: object) -> bool:
    """Se o tipo aceita None: o próprio None, a união com ele ou o Literal com ele."""
    field_type = _resolved(field_type)
    if field_type is type(None):
        return True
    origin = get_origin(field_type)
    if origin is typing.Annotated:  # dentro de Optional, como em is_complex_type
        return _admits_none(get_args(field_type)[0])
    if origin is typing.Literal:
        return None in get_args(field_type)
    if origin is typing.Union or isinstance(field_type, types.UnionType):
        return any(_admits_none(arg) for arg in get_args(field_type))
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
               Com ele e sem kwargs, os campos de texto do .csv, do .xlsx e
               do .xls são lidos como texto cru: "2023" não vira número,
               "N/A" não vira ausência, e só a célula vazia vira ausência.
               Com kwargs, o texto fica como o leitor do pandas o devolve.
               Também com ele e sem kwargs, a coluna de campo int, Literal
               só de ints, ou um deles com None, que o CSV ou o parquet não
               devolve em tipo inteiro volta como object, com o int exato e
               None na célula vazia. O NewType e o apelido `type` contam
               pelo tipo a que se referem. A coluna só com células vazias fica como lida: float
               no CSV e, na coluna nula que o checkpoint grava no parquet,
               object com None.
        normalize: Se True (padrão), normaliza colunas que parecem conter JSON.
                   Se False, não faz nenhuma normalização.
        **kwargs: Argumentos adicionais passados para a função de leitura do pandas.
                  Com o argumento que faz o leitor devolver outra coisa que
                  um DataFrame, como chunksize, iterator ou sheet_name=None
                  ou lista, a função devolve o que o pandas devolve, sem
                  normalizar nem reler nada, mesmo com model e normalize=True.

    Returns:
        DataFrame com estruturas Python normalizadas, ou o objeto do leitor do
        pandas quando ele não devolve DataFrame (ver **kwargs).

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
    if not isinstance(df, pd.DataFrame):
        # Leitor de pedaços ou dict de abas: a normalização e as releituras
        # trabalham sobre as colunas de um DataFrame, e o objeto sai como o
        # pandas o devolve.
        return df

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
    retomada acusa ou troca o valor. Só a célula vazia vira ausência.
    Qualquer argumento de leitura desliga a releitura, porque, somado ao
    usecols dela, ele pode mudar as linhas que o leitor devolve: com
    on_bad_lines, a linha malformada deixa de sê-lo quando só as colunas de
    texto são lidas, volta na releitura e desloca o texto das seguintes; com
    index_col, parse_dates ou names, a releitura levanta. A atribuição alinha
    pelo índice e não acusa a diferença de tamanho.
    """
    reader = _TEXT_READERS.get(ext)
    if reader is None or kwargs:
        return
    text_columns = [f for f in get_text_fields(model) if f in df.columns]
    if not text_columns:
        return
    raw = reader(path, usecols=text_columns, dtype=str, na_filter=False)
    for col in text_columns:
        df[col] = raw[col].mask(raw[col] == "")


def _accepts_only_int(field_type: object) -> bool:
    """True se o tipo é int, o Literal só de ints, ou um deles com None."""
    field_type = _resolved(field_type)
    if field_type is int:
        return True
    origin = get_origin(field_type)
    if origin is typing.Annotated:  # dentro de Optional, como em is_complex_type
        return _accepts_only_int(get_args(field_type)[0])
    if origin is typing.Literal:
        return all(type(arg) is int for arg in get_args(field_type))
    if origin is typing.Union or isinstance(field_type, types.UnionType):
        args = [arg for arg in get_args(field_type) if arg is not type(None)]
        return all(_accepts_only_int(arg) for arg in args)
    return False


_INT_TEXT = re.compile(r"-?\d+")


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
    arredondado. Qualquer argumento de leitura também desliga a releitura,
    porque ele muda o que o leitor devolve (on_bad_lines pula linhas, dtype
    fixa tipos, chunksize nem devolve DataFrame), e a releitura sem ele não
    casaria com o que foi lido.
    """
    if ext not in (".csv", ".parquet") or kwargs:
        return
    columns = [
        f
        for f, info in model.model_fields.items()
        if f in df.columns and _accepts_only_int(info.annotation) and not is_integer_dtype(df[f])
    ]
    if not columns:
        return
    if ext == ".parquet":
        raw = pd.read_parquet(path, columns=columns, dtype_backend="numpy_nullable")
        exact = {
            col: [None if pd.isna(v) else int(v) for v in raw[col]]
            for col in columns
            if is_integer_dtype(raw[col])
        }
    else:
        raw = pd.read_csv(path, usecols=columns, dtype=str, na_filter=False)
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


def _unwrapped(annotation: object) -> object:
    """Um nível da anotação sem o `Annotated`, que só acrescenta metadados ao tipo.

    É o ponto único em que os helpers de modelo aninhado abaixo desembrulham
    cada nível da anotação antes de olhar a origem e os argumentos: todos
    passam por `_value_types`, que o chama. O Pydantic tira o `Annotated` só do
    nível de cima da anotação do campo; dentro de Optional, de união ou de
    lista ele fica, e os metadados dele nunca são percorridos como tipo.
    """
    if get_origin(annotation) is typing.Annotated:
        return get_args(annotation)[0]
    return annotation


def _value_types(annotation: object) -> list:
    """Os tipos que o valor pode ter: cada membro da união, menos o None, sem `Annotated`.

    A união dentro de outra, como `Optional[Annotated[A | B, ...]]`, é aberta
    também, e a anotação que não é união volta sozinha na lista.
    """
    annotation = _unwrapped(annotation)
    if get_origin(annotation) in (typing.Union, types.UnionType):
        return [
            value_type
            for arg in get_args(annotation)
            if arg is not type(None)
            for value_type in _value_types(arg)
        ]
    return [annotation]


def _item_types(value_type: object) -> list:
    """Os tipos que o item pode ter quando `value_type` é `list[X]`; vazio para outro tipo.

    É a decisão única de camada de lista: `is_list_of_pydantic_model`, que a
    extração usa para buscar item a item, e `list_layers`, que a validação usa
    para contar as listas em volta de um modelo, reconhecem a lista por aqui.
    """
    args = get_args(value_type)
    return _value_types(args[0]) if get_origin(value_type) is list and args else []


def _is_pydantic_model(annotation: object) -> bool:
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


def is_list_of_pydantic_model(field_type: object) -> tuple:
    """Verifica se o valor do campo é uma lista de modelo Pydantic.

    O Optional, a união e o `Annotated` não contam, em volta da lista e em
    volta do item: `Optional[Annotated[list[Item], ...]]` e `list[Item | None]`
    são listas de `Item`, como `list[Item]`. A lista de listas, o dict e a tupla
    de modelos não são: só a lista cujo item é o modelo tem um dicionário por
    item onde gravar a busca. Também não é a lista cujo item pode ser mais de um
    modelo, como `list[A | B]` ou `list[A] | list[B]`: sem saber de qual modelo
    é cada item, a busca do campo de A iria para o item de B, e a de B não
    seria feita; os campos de busca dos dois seguem a busca aninhada da linha.

    Args:
        field_type: Anotação de tipo a analisar.

    Returns:
        Tupla (is_list, inner_model):
        - is_list: True se o tipo é lista de modelo Pydantic
        - inner_model: O modelo Pydantic dos itens (ou None)

    Examples:
        >>> class Inner(BaseModel):
        ...     x: str
        >>> is_list_of_pydantic_model(List[Inner])
        (True, <class 'Inner'>)
        >>> is_list_of_pydantic_model(Optional[List[Optional[Inner]]])
        (True, <class 'Inner'>)
        >>> is_list_of_pydantic_model(Inner)
        (False, None)
        >>> is_list_of_pydantic_model(List[str])
        (False, None)
    """
    item_models = {
        item_type
        for value_type in _value_types(field_type)
        for item_type in _item_types(value_type)
        if _is_pydantic_model(item_type)
    }
    if len(item_models) == 1:
        return (True, item_models.pop())
    return (False, None)


def list_layers(annotation: object, target: type) -> int | None:
    """Quantas listas envolvem `target` na anotação: list[list[X]] dá 2.

    Só conta os ramos que chegam ao modelo; em Union[list[str], X], a lista
    de strings não envolve X. A camada de lista é a que `_item_types`
    reconhece, a mesma de `is_list_of_pydantic_model`: com uma camada, a
    validação aceita a configuração do item, que a extração busca item a item
    ou, quando o item pode ser mais de um modelo, uma vez para a linha.
    Devolve None quando `target` não aparece.
    """
    depths = []
    for value_type in _value_types(annotation):
        if value_type is target:
            depths.append(0)
            continue
        item_types = _item_types(value_type)
        layer = 1 if item_types else 0
        for inner in item_types or get_args(value_type):
            depth = list_layers(inner, target)
            if depth is not None:
                depths.append(depth + layer)
    return max(depths, default=None)


# Os tipos cujo valor não tem chave: no `model_dump`, eles ficam como estão, e
# `get_nested_value` não segue caminho dentro deles. `datetime.datetime` entra
# por ser subclasse de `datetime.date`, e `bool`, por ser de `int`.
_KEYLESS_TYPES = (
    str,
    bytes,
    int,
    float,
    complex,
    Decimal,
    Enum,
    datetime.date,
    datetime.time,
    datetime.timedelta,
)


def _is_keyless(value_type: object) -> bool:
    """Se o valor desse tipo é escalar, sem chave que um caminho possa seguir."""
    if get_origin(value_type) is typing.Literal:
        return True
    return isinstance(value_type, type) and issubclass(value_type, _KEYLESS_TYPES)


def _is_collection(value_type: object) -> bool:
    """Se o valor desse tipo é coleção sem chave, de qualquer item, com ou sem argumentos.

    A coleção é reconhecida pela ABC: `Iterable` pega a lista, a tupla, o set,
    o deque, `Sequence` e o próprio `Iterable`, de `typing` ou de
    `collections.abc`. O `Mapping` tem chave e fica de fora, como `dict`. O
    modelo Pydantic define `__iter__` e passaria também: `submodels` confere o
    modelo antes. O texto e os bytes passam, mas `_is_keyless` já os põe de
    fora pelo mesmo motivo.
    """
    origin = get_origin(value_type) or value_type
    return (
        isinstance(origin, type)
        and issubclass(origin, collections.abc.Iterable)
        and not issubclass(origin, collections.abc.Mapping)
    )


def submodels(annotation: object) -> list[type[BaseModel]] | None:
    """Os modelos Pydantic em que pode estar a chave seguinte de um caminho no valor.

    `Optional[Endereco]` dá `[Endereco]`, e `Union[A, B]`, `[A, B]`. O tipo
    escalar (`_KEYLESS_TYPES` e `Literal`) não tem chave e fica de fora:
    `Union[Endereco, str]` dá `[Endereco]`, e `str`, a lista vazia, em que
    nenhuma chave existe. A lista, a tupla ou outra coleção (`_is_collection`)
    ficam de fora também, porque `get_nested_value` só segue dict: nem o campo
    do item nem o índice são seguidos, e `list[Endereco]` dá a lista vazia.
    None quando algum tipo do valor pode ter chave que nenhum modelo declara,
    como `dict`, `Any`, `Union[Endereco, dict[str, str]]` ou um tipo que esta
    função não conhece.
    """
    models = []
    for value_type in _value_types(annotation):
        if _is_pydantic_model(value_type):
            models.append(value_type)
        elif not (_is_keyless(value_type) or _is_collection(value_type)):
            return None
    return models


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

    Referência que não se resolve fica como está. Uma união resolvida volta
    montada por `typing.Union[...]`, tenha sido escrita com `typing.Union` ou
    com `X | Y`. Antes do 3.14, isso leva `X | Y` à forma de `typing.Union`,
    igual a ela por `==` e tratada pelos helpers deste módulo junto com
    `types.UnionType`; a partir do 3.14, as duas formas são o mesmo objeto.
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
    if origin in (typing.Union, types.UnionType):
        return typing.Union[resolved]  # noqa: UP007 (`|` daria types.UnionType antes do 3.14)
    if isinstance(annotation, types.GenericAlias):
        return types.GenericAlias(origin, resolved)
    return annotation.copy_with(resolved)


def get_nested_pydantic_models(field_type: object) -> list:
    """Extrai todos os modelos Pydantic de uma anotação de tipo.

    Trata List[Model], Optional[List[Model]], Union[Model, None], etc.
    Usado para detectar campos de busca em modelos aninhados. Cada nível passa
    por `_value_types`, e os metadados de `Annotated` ficam de fora.

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
    for value_type in _value_types(field_type):
        if _is_pydantic_model(value_type):
            models.append(value_type)
        else:
            for arg in get_args(value_type):
                models.extend(get_nested_pydantic_models(arg))
    return models
