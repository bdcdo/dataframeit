"""Função dataframeit: valida a configuração, processa as linhas e devolve o resultado."""

from __future__ import annotations

import collections.abc
import dataclasses
import datetime as dt
import hashlib
import importlib.util
import inspect
import itertools
import json
import logging
import math
import numbers
import re
import sys
import threading
import time
import warnings
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from functools import partial
from pathlib import Path
from types import CodeType, MethodType, UnionType
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NotRequired,
    Self,
    TypedDict,
    Union,
    cast,
    get_args,
    get_origin,
    overload,
)

import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionArray, ExtensionDtype
from pandas.api.types import infer_dtype, is_bool, is_scalar

# O mesmo `pandas.util.hash_pandas_object`, que o `pandas.util` entrega por um
# `__getattr__` que o checker de tipos lê como `object`.
from pandas.core.util.hashing import hash_pandas_object
from pydantic import BaseModel, ValidationError
from tqdm import tqdm
from typing_extensions import override

from .conditional import (
    _CONDITIONAL_KEYS,
    _FIELD_CONFIG_KEYS,
    _collect_configured_fields,
    _model_field_configs,
    _walk_fields,
    evaluate_condition,
    get_field_execution_order,
    get_group_execution_units,
)
from .errors import (
    ProviderAbortError,
    ProviderUsageLimitError,
    dump_response,
    get_friendly_error_message,
    is_rate_limit_error,
    is_recoverable_error,
    validate_provider_dependencies,
    validate_search_dependencies,
)
from .llm import (
    _USAGE_COUNTERS,
    LLMConfig,
    SearchConfig,
    SearchGroupConfig,
    _BuildOnce,
    _take_failed_usage,
    build_structured_llm,
    call_langchain,
    with_shared_chat_model,
)
from .search import get_available_providers, get_provider
from .utils import (
    _FLOAT_NAN_TEXT,
    _USER_COLUMN,
    DEFAULT_TEXT_COLUMN,
    ORIGINAL_TYPE_PANDAS_DF,
    ORIGINAL_TYPE_POLARS_DF,
    TOKEN_COLUMNS,
    ConversionInfo,
    _accepts_only_decimal,
    _admits_any,
    _admits_none,
    _any_from_text,
    _column_rank,
    _dropped_by_model_serializer,
    _excluded_fields,
    _is_structure_class,
    _json_default,
    _keeps_nan,
    _mixes_text_and_number,
    _quoted_nan_text,
    _quoted_scalar_text,
    _reads_as_other,
    _resolved,
    _silent_slice_write,
    _skipping_model,
    _user_stacklevel,
    accepts_only_text,
    from_pandas,
    get_complex_fields,
    get_nested_pydantic_models,
    normalize_complex_columns,
    normalize_value,
    read_df,
    to_pandas,
)

if TYPE_CHECKING:
    import os
    from collections.abc import (
        Callable,
        Collection,
        Container,
        Generator,
        Hashable,
        Iterable,
        Sequence,
    )

    from polars import DataFrame as PolarsDataFrame
    from polars import Series as PolarsSeries
    from pydantic.fields import FieldInfo

    from .llm import Usage

    # (posição na fila, índice da linha, linha) que o modo paralelo processa
    _RowData = tuple[int, Hashable, pd.Series]


class _FallbackHandler(logging.Handler):
    """Escreve em stderr só quando nenhum outro handler recebe o registro.

    Com a aplicação configurando o logging (basicConfig, pytest, um handler no
    logger `dataframeit`), o registro sobe por propagação e sai por ela, e
    escrever aqui também o duplicaria. O stderr é lido na hora da escrita,
    porque o notebook e o pytest o trocam depois do import.

    O handler se reconhece pelo atributo, e não pela classe, porque recarregar
    o módulo (importlib.reload, %autoreload) cria outra classe enquanto o
    logger guarda a instância da anterior.
    """

    is_dataframeit_fallback = True

    @override
    def emit(self, record: logging.LogRecord) -> None:
        logger: logging.Logger | None = logging.getLogger(record.name)
        while logger is not None:
            if any(not _is_fallback(handler) for handler in logger.handlers):
                return
            if not logger.propagate:
                break
            logger = logger.parent
        try:
            sys.stderr.write(self.format(record) + "\n")
        except Exception:  # noqa: BLE001 (falha de escrita segue o tratamento do logging)
            self.handleError(record)


# O resumo de uso sai em INFO por um logger próprio, e não pelo pai
# `dataframeit`: subir o pai para INFO mostraria também o registro por campo
# pulado de conditional.py. O handler de reserva mantém o resumo visível sem
# configuração, e o nível WARNING nesse logger o silencia.
def _is_fallback(handler: logging.Handler) -> bool:
    return hasattr(handler, "is_dataframeit_fallback")


def _install_fallback_handler(logger: logging.Logger) -> None:
    """Põe o handler de reserva no logger, uma vez só mesmo com o módulo recarregado.

    Dois deles no mesmo logger se tomariam por handler da aplicação, e o
    resumo sumiria.
    """
    if not any(_is_fallback(handler) for handler in logger.handlers):
        logger.addHandler(_FallbackHandler())


_stats_logger = logging.getLogger("dataframeit.stats")
_stats_logger.setLevel(logging.INFO)
_install_fallback_handler(_stats_logger)


# Nomes candidatos consultados quando o usuário não passa text_column explicitamente.
# Ordem: convenção da lib ('texto'), inglês ('text'), juscraper cjpg/cjsg ('decisao'),
# e convenções comuns de ETL ('content', 'content_text').
TEXT_COLUMN_CANDIDATES = ("texto", "text", "decisao", "content", "content_text")

# Modelo usado quando o usuário escolhe o provider sem escolher o modelo. Mandar
# o modelo de um provider para outro falha em toda linha, por isso o default
# acompanha o provider. 'codex' e 'claude_code' ficam de fora: com model=None,
# o runtime de cada um escolhe o modelo.
DEFAULT_MODELS = {
    "openai": "gpt-6-luna",
    "google_genai": "gemini-3.8-flash",
    "anthropic": "claude-sonnet-5",
    "groq": "openai/gpt-oss-120b",
}
_RUNTIME_DEFAULT_PROVIDERS = frozenset({"codex", "claude_code"})


# Limite de queries concorrentes acima do qual vale avisar o usuário.
_RECOMMENDED_MAX_CONCURRENT_SEARCH_QUERIES = 10

# Faixa aceita de max_results, global ou por grupo e campo.
_MAX_RESULTS_LIMIT = 20

# Prazo de cada espera do laço paralelo pelas linhas em andamento; ver o wait
# de _process_rows_parallel.
_WAIT_SLICE_SECONDS = 0.1

# Numera os executores do laço paralelo, um por execução, para que o nome das
# threads de cada um não se confunda com o das de outro; ver _closing_in_slices.
_executor_ids = itertools.count()

# Segundos que o aviso de rate limit supõe para cada execução do agente com
# busca; a linha leva uma execução por campo no modo por campo, mais o
# rate_limit_delay. A estimativa do aviso é um piso calculado só da
# configuração: supõe uma busca por execução do agente. Um agente que faz mais
# buscas, até max_search_calls, pode passar do limite sem aviso.
_MIN_SEARCH_RUN_SECONDS = 1.0


class TokenStats(TypedDict):
    """Uso somado de uma execução, que vai ao resumo do fim.

    Os contadores começam zerados. As demais chaves só existem depois que o
    laço as grava: `abort` quando a execução é interrompida, e as de tempo e
    de workers ao fim do modo paralelo.
    """

    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    total_tokens: int
    reasoning_tokens: int
    search_credits: int
    search_count: int
    cost_usd: float
    abort: NotRequired[str]
    requests_completed: NotRequired[int]
    elapsed_seconds: NotRequired[float]
    initial_workers: NotRequired[int]
    final_workers: NotRequired[int]
    workers_reduced: NotRequired[bool]


def _empty_token_stats() -> TokenStats:
    """Contadores zerados de tokens, busca e custo de uma execução."""
    return {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "reasoning_tokens": 0,
        "search_credits": 0,
        "search_count": 0,
        "cost_usd": 0.0,
    }


@dataclass(frozen=True)
class ProviderBackend:
    """Nome e função de chamada vinculados a uma única configuração.

    `invoke_partial(text, only_fields, known)` existe nos backends que extraem
    campo a campo, e deixa reprocess_columns pedir só as colunas escolhidas.
    """

    label: str
    invoke: Callable[[str], dict[str, Any]]
    invoke_partial: Callable[[str, set[str], dict[str, Any]], dict[str, Any]] | None = None


@dataclass(frozen=True)
class RowResult:
    """O que uma linha processada deixou gravada, entregue a `on_row_done`.

    `index` é o rótulo da linha no DataFrame, e `position` a ordem (de 0) entre
    as linhas processadas nesta chamada: a linha pulada por `resume` não conta.
    `values` traz os campos do modelo que esta chamada gravou na linha, que numa
    linha reprocessada são só os de `reprocess_columns`; em erro, fica vazio.
    `error` é o texto gravado em `_error_details`, que numa linha bem-sucedida
    pode registrar as tentativas repetidas; `success` é o que diz se a linha
    deu certo. `elapsed_seconds` conta do início do processamento da linha,
    inclusive as tentativas repetidas e as pausas de rate limit.
    """

    index: Any
    position: int
    success: bool
    error: str | None
    values: dict[str, Any]
    elapsed_seconds: float


_row_warning_site = threading.local()


def _user_warning_site() -> tuple[str, int, str, dict[str, Any]] | None:
    """Local, módulo e registro de avisos da chamada pública, para as threads."""
    frame = inspect.currentframe()
    while frame is not None:
        if (
            frame.f_code.co_name in {"dataframeit", "x_dataframeit__mutmut_orig"}
            and frame.f_globals.get("__name__") == __name__
        ):
            # Uma chamada Python da função pública sempre tem frame chamador.
            caller = cast("Any", frame.f_back)
            if frame.f_code.co_name == "x_dataframeit__mutmut_orig":
                while caller.f_code.co_name in {"trampoline", "_trampoline_wrapper"}:
                    caller = caller.f_back
            return (
                caller.f_code.co_filename,
                caller.f_lineno,
                cast("str", caller.f_globals.get("__name__", "__main__")),
                cast("dict[str, Any]", caller.f_globals.setdefault("__warningregistry__", {})),
            )
        frame = frame.f_back
    return None


def _warn_at_user(message: str) -> None:
    """Emite aviso na chamada pública, também quando a linha roda em outra thread."""
    if _user_warning_site() is not None:
        warnings.warn(message, stacklevel=_user_stacklevel())
        return
    site = getattr(_row_warning_site, "value", None)
    if site is not None:
        warnings.warn_explicit(
            message, UserWarning, site[0], site[1], module=site[2], registry=site[3]
        )
    else:
        warnings.warn(message, stacklevel=2)


def _recorded_error(df: pd.DataFrame, idx: Hashable) -> str | None:
    """Texto gravado em `_error_details` da linha, ou None com a célula vazia."""
    value = _get_cell(df, idx, "_error_details")
    return None if value is None or pd.isna(value) else str(value)


def _notify_row_done(on_row_done: Callable[[RowResult], None] | None, result: RowResult) -> None:
    """Entrega a linha ao callback; a falha dele vira aviso, e a execução segue."""
    if on_row_done is None:
        return
    try:
        on_row_done(result)
    except Exception as e:  # noqa: BLE001 (o callback é do usuário; a falha dele não para a execução)
        msg = f"on_row_done falhou na linha {result.index}: {_error_text(e)}"
        _warn_at_user(msg)


# Detalhe gravado na linha cujo texto está vazio, e que por isso não vai ao LLM.
_MISSING_TEXT_DETAIL = "Texto ausente"


# Detalhe da linha já processada que uma interrupção deixou sem reprocessar.
_INTERRUPTED_REPROCESS_DETAIL = "Reprocessamento interrompido; a linha mantém os valores anteriores"


# Sufixo do erro de uma linha já processada cujo reprocessamento falhou.
_KEPT_VALUES_NOTE = " (reprocess_columns falhou; a linha mantém os valores anteriores)"


def _success_details(retry_info: dict[str, Any]) -> str | None:
    """Detalhe gravado numa linha bem-sucedida: os retries, se houve."""
    retries = retry_info.get("retries", 0)
    return f"Sucesso após {retries} retry(s)" if retries > 0 else None


def _missing_text_detail(
    *, row_already_processed: bool, reprocess_columns: Sequence[str] | None
) -> str:
    """Detalhe da linha sem texto; sob reprocess_columns, a linha mantém os valores."""
    if row_already_processed and bool(reprocess_columns):
        return _MISSING_TEXT_DETAIL + _KEPT_VALUES_NOTE
    return _MISSING_TEXT_DETAIL


def _as_object_columns(
    df: pd.DataFrame, columns: list[str], nan_columns: Collection[str] = ()
) -> None:
    """Converte para object as colunas que vão receber texto, lista ou None.

    Uma coluna lida de CSV ou XLSX toda vazia volta como float, e gravar texto
    nela levanta TypeError no pandas 3. Vale para as colunas do modelo e para
    as de controle (status e detalhe de erro).

    Nas colunas de `nan_columns`, cujo NaN é valor (`_keeps_nan`), o NaN de uma
    coluna que não é object vira None. Numa coluna float64, como a do
    checkpoint CSV ou parquet relido, o NaN é também a linha pendente e a
    resposta None, e a validação da retomada já o conta como ausente
    (`_validate_processed_rows`). Sem a troca, ele chegaria à coluna object
    como NaN, que dali em diante é valor: a saída polars, que guarda o NaN
    dessas colunas, mostraria NaN na linha pendente e na de erro.
    """
    # No pandas 3, o contexto não lê o DataFrame, e trocá-lo por None, como faz o
    # teste de mutação, não muda nada; no pandas 2, a troca levanta.
    with _silent_slice_write(df):  # pragma: no mutate
        for col in columns:
            if col in df.columns and df[col].dtype != object:
                values = df[col].astype(object)
                if col in nan_columns:
                    values = values.where(df[col].notna(), None)
                df[col] = values


def _is_missing_text(value: object) -> bool:
    """None, NaN ou texto só com espaços: não há o que mandar ao LLM."""
    if isinstance(value, str):
        return not value.strip()
    return is_scalar(value) and bool(pd.isna(value))


def _warn_missing_texts(df: pd.DataFrame, text_column: str, rows: list[Hashable]) -> None:
    """Avisa uma vez quantas linhas a processar não têm texto."""
    missing = sum(1 for idx in rows if _is_missing_text(_get_cell(df, idx, text_column)))
    if missing:
        warnings.warn(
            f"{missing} linha(s) sem texto não vão ao LLM e ficam com status 'error' "
            f"('{_MISSING_TEXT_DETAIL}').",
            stacklevel=_user_stacklevel(),
        )


def _invoke_row(  # noqa: PLR0913 (a linha e o que a validação da retomada sabe dela)
    backend: ProviderBackend,
    text: str,
    row: pd.Series,
    reprocess_on_row: Sequence[str] | None,
    expected_columns: list[str],
    *,
    nan_columns: Collection[str] = (),
) -> dict[str, Any]:
    """Chama o backend para uma linha, pedindo só as colunas a reprocessar quando dá.

    `reprocess_on_row` é reprocess_columns numa linha já processada, e None
    nas demais. Numa linha já processada, só essas colunas são gravadas.
    Com um backend campo a campo, os demais campos nem são pedidos: voltam com
    o valor já gravado, que também alimenta as condições. A célula ausente
    vai como None, pela mesma regra da validação da retomada (`_is_absent`),
    e o NaN de `nan_columns` vai como NaN, inclusive o do campo `Any` que ela
    contou como valor (`_any_nan_columns`). A validação já trocou por None o
    NaN que a coluna não distingue do nulo, e o que resta nela é valor.

    O campo `Iterable` ou `Generator` volta do model_dump como iterador, e a
    célula o guardaria no lugar dos itens. A falha vem do modelo e se repete
    a cada resposta com esse campo preenchido, e por isso interrompe a
    execução, com o uso da resposta já paga na exceção.
    """
    if bool(reprocess_on_row) and backend.invoke_partial is not None:
        known = {
            col: None if _is_absent(row[col], nan_is_value=col in nan_columns) else row[col]
            for col in expected_columns
            if col not in reprocess_on_row and col in row.index
        }
        result = backend.invoke_partial(text, set(reprocess_on_row), known)
    else:
        result = backend.invoke(text)
    data = result["data"]
    lazy = [col for col in expected_columns if col in data and _holds_iterator(data[col])]
    if lazy:
        msg = (
            f"A resposta trouxe um iterador no lugar dos itens em {lazy}. O pydantic valida "
            "e serializa a anotação Iterable ou Generator como iterador, e a célula guardaria "
            "o iterador. Repetir a linha não resolve: declare list[...] nesse campo, ou no "
            "campo de dentro dele que tem a anotação"
        )
        error = ProviderAbortError(msg)
        error.row_usage = result.get("usage")  # ty: ignore[unresolved-attribute]
        raise error
    return result


def _holds_iterator(value: object) -> bool:
    """Se o valor é um iterador ou o guarda numa coleção de `_is_structure_class`."""
    if isinstance(value, collections.abc.Iterator):
        return True
    if not _is_structure_class(type(value)):
        return False
    items = value.values() if isinstance(value, collections.abc.Mapping) else value
    return any(_holds_iterator(item) for item in items)  # ty: ignore[not-iterable]


def _is_absent(value: object, *, nan_is_value: bool) -> bool:
    """Se a célula relida conta como ausente: None, NaN, NaT ou pd.NA.

    O pandas conta o Decimal NaN como ausente, mas ele é um valor que o campo
    Decimal com allow_inf_nan aceita, e a linha que o recebeu já foi paga; sem
    a exceção, a retomada o acusaria como incompatível. O Decimal sai antes do
    pd.isna, que levanta InvalidOperation com o NaN sinalizador (sNaN). O float
    NaN é valor quando `nan_is_value`, na coluna que o distingue do nulo
    (`_keeps_nan`).
    """
    if isinstance(value, Decimal) or not (is_scalar(value) and bool(pd.isna(value))):
        return False
    return not (nan_is_value and isinstance(value, float))


def _executed_form(pydantic_model: type[BaseModel]) -> Callable[[str, object], object]:
    """Converte o valor relido de um campo à forma em que a execução o passa às condições.

    A execução passa às condições o model_dump do valor com os validadores do
    modelo, que é o que ela grava, e o checkpoint grava a data, o Decimal e a
    chave de dict como texto. Validado de novo sem strict pelo modelo do
    campo, sem os validadores, que já rodaram, o valor volta ao tipo do campo,
    e só substitui o relido onde diz o mesmo
    (`_validated_where_same`): o item que um validador muda a cada passada
    fica como foi relido. O campo com exclude=True, que o dump deixa de fora,
    a execução viu como None.
    """
    from .agent import _field_model  # noqa: PLC0415 (ver _provider_backend)

    excluded_fields = _excluded_fields(pydantic_model)
    field_models: dict[str, type[BaseModel]] = {}

    def convert(field_name: str, value: object) -> object:
        if value is None or field_name in excluded_fields:
            return None
        if field_name not in field_models:
            field_models[field_name] = _field_model(pydantic_model, field_name)
        try:
            validated = dump_response(
                field_models[field_name].model_validate(
                    {field_name: value}, strict=False, by_alias=False, by_name=True
                )
            ).get(field_name)
        except ValidationError:
            return value
        return _validated_where_same(validated, value)

    return convert


def _condition_holds(condition: object, row_values: dict[str, Any], field_name: str) -> bool:
    """Avalia a condição com os valores da linha; erro conta como verdadeira.

    Verdadeira é o lado conservador: o campo ausente continua acusado, e a
    retomada pede para reprocessá-lo.
    """
    try:
        return bool(evaluate_condition(condition, row_values, field_name))
    except Exception:  # noqa: BLE001 (ver a docstring)
        return True


def _validate_processed_rows(  # noqa: C901, PLR0912, PLR0913, PLR0915 (validação por linha e por campo)
    df: pd.DataFrame,
    status_col: str,
    pydantic_model: type[BaseModel],
    complex_fields: set[str],
    *,
    empty_text_fields: Collection[str] = frozenset(),
    any_nan_columns: Collection[str] = frozenset(),
) -> tuple[list[str], dict[tuple[int, str], Any]]:
    """Valida linhas concluídas sem alterar o checkpoint recebido.

    Em `any_nan_columns`, os campos `Any` cujo NaN é valor (`_any_nan_columns`).

    Em `empty_text_fields`, os campos cuja célula ausente é o texto vazio que o
    CSV ou o XLSX do checkpoint gravou (ver `_resume_from_checkpoint`), salvo
    na linha em que a condição do campo é falsa, que o deixa None. A
    execução grava o model_dump do modelo, e o texto vazio
    restaurado que a validação da linha não devolve como "" não é o que ela
    gravou: a célula é o None que um validador deu ao texto vazio, que o tipo
    do campo recusa, e o campo é acusado. O None que um validador dá a outra
    resposta não se separa do texto vazio na célula: o "" restaurado passa
    pelo validador como é, e o campo volta "", sem acusação.
    """
    incompatible_fields: set[str] = set()
    values_to_fill: dict[tuple[int, str], Any] = {}
    expected_columns = list(pydantic_model.model_fields)

    def rejected_fields(error: ValidationError) -> set[str]:
        """Campos acusados pelo erro; o erro do modelo inteiro acusa todos."""
        rejected = set()
        for detail in error.errors():
            location = detail["loc"]
            # A linha se valida pelo nome (_validate_row), que o erro traz.
            if location and location[0] in pydantic_model.model_fields:
                rejected.add(str(location[0]))
            else:
                rejected.update(expected_columns)
        return rejected

    if status_col not in df.columns:
        return [], values_to_fill

    # Campo com `condition` fica None quando a condição é falsa na linha, mesmo
    # que o tipo declarado seja obrigatório. Só nesse caso o None é aceito, e só
    # o campo pulado sai da validação: um valor presente passa pelas restrições
    # do modelo, e a condição verdadeira com valor ausente continua acusada.
    conditions = {
        field_name: field.json_schema_extra["condition"]
        for field_name, field in pydantic_model.model_fields.items()
        if isinstance(field.json_schema_extra, dict) and "condition" in field.json_schema_extra
    }
    # Na ordem das dependências, como na execução: a condição lê o None do
    # campo que outra condição pulou antes dela. A configuração que não tem
    # ordem é recusada antes de processar, e aqui fica a ordem do modelo.
    if conditions:
        with suppress(ValueError):
            execution_order, _ = get_field_execution_order(
                pydantic_model, _model_field_configs(pydantic_model)
            )
            conditions = {
                field_name: conditions[field_name]
                for field_name in execution_order
                if field_name in conditions
            }
    as_executed = _executed_form(pydantic_model)
    # O campo com exclude=True não sai no model_dump, e a execução não grava a
    # coluna dele: a célula guarda o None com que a coluna foi criada ou o que a
    # entrada trazia. Não há resposta paga a conferir, e a célula fica fora da
    # validação. O campo com default o recebe, como na execução; o obrigatório
    # sai da validação como o campo pulado pela condição.
    excluded_fields = _excluded_fields(pydantic_model)
    required_excluded = frozenset(
        field_name
        for field_name in excluded_fields
        if pydantic_model.model_fields[field_name].is_required()
    )
    # O campo que o dump pode deixar de fora conforme a resposta, pelo
    # exclude_if ou pelo @model_serializer, tem a célula vazia quando a
    # execução não o gravou, e ela fica fora da validação como a do campo com
    # exclude=True: o com default a recebe, o obrigatório sai, e a célula
    # volta None. A célula com valor é conferida como a de qualquer campo, e
    # a coluna ausente da entrada e do checkpoint, que não é célula gravada,
    # também.
    serializer_dropped = _dropped_by_model_serializer(pydantic_model, excluded_fields)
    omittable_fields = _omittable_fields(pydantic_model, excluded_fields, serializer_dropped)
    # Na coluna que veio da entrada ou do checkpoint, a célula vazia do campo
    # que admite None e tem outro default, dado direto ou pela fábrica, é o
    # None que a execução gravou, e ele vai à validação: sem ele, ela
    # preencheria o default, que a execução não gravou. No CSV e no XLSX, a
    # célula pode ter sido também o texto vazio, e volta None, como no campo
    # Optional[str] sem default. Com o default None, a célula fica fora da
    # validação e recebe o próprio None, sem passar pelos validadores do campo.
    # Só sem a coluna, a linha não tem resposta do campo, e ele recebe o default.
    written_none_fields = frozenset(
        field_name
        for field_name, field in pydantic_model.model_fields.items()
        if field_name in df.columns
        and _admits_none(field.annotation)
        and (field.default_factory is not None or field.default is not None)
    )

    # O NaN só é valor na coluna de objetos, que guarda o None à parte. Na
    # coluna float64, como a do CSV relido, o NaN é também a resposta None e a
    # linha pendente, e segue contando como ausente.
    nan_columns = frozenset(any_nan_columns) | {
        field_name
        for field_name, field in pydantic_model.model_fields.items()
        if field_name in df.columns
        and df[field_name].dtype == object
        and _keeps_nan(field.annotation)
    }
    # A coluna de objetos de um campo `Any` sem nenhum None: o NaN de uma linha
    # processada conta como ausente e volta None, e o aviso diz em quais.
    ambiguous_any = _any_object_columns(df, pydantic_model) - nan_columns
    dropped_nan: set[str] = set()
    # O isin lê o pd.NA, status ausente da coluna `string` de convert_dtypes.
    processed_positions = np.flatnonzero(df[status_col].isin(["processed"])).tolist()
    for position in processed_positions:
        row = df.iloc[position]
        row_values = {}
        for field_name in pydantic_model.model_fields:
            if field_name not in df.columns:
                row_values[field_name] = None
                continue
            value = row[field_name]
            if _is_absent(value, nan_is_value=field_name in nan_columns):
                row_values[field_name] = None
                if field_name in ambiguous_any and isinstance(value, float):
                    dropped_nan.add(field_name)
            elif field_name in complex_fields:
                row_values[field_name] = normalize_value(value)
            elif isinstance(value, pd.Timestamp):
                # O datetime relido do XLSX ou do parquet; o pydantic devolveria
                # o próprio Timestamp, e a coluna misturaria os dois tipos.
                row_values[field_name] = value.to_pydatetime()
            else:
                row_values[field_name] = value

        empty_cells = {
            field_name for field_name in empty_text_fields if row_values[field_name] is None
        }
        skipped_fields = set()
        if any(row_values[field_name] is None for field_name in conditions):
            # A condição vê os valores, e o texto vazio de outro campo, como a
            # execução os viu.
            condition_values = {
                **{name: as_executed(name, value) for name, value in row_values.items()},
                **dict.fromkeys(empty_cells, ""),
            }
            for field_name, condition in conditions.items():
                if row_values[field_name] is None and not _condition_holds(
                    condition, condition_values, field_name
                ):
                    skipped_fields.add(field_name)
                    condition_values[field_name] = None
        skipped = frozenset(skipped_fields)

        projected = {}
        # Na ordem do modelo, que o preenchimento dos defaults da linha recusada segue.
        missing_values = []
        restored = set()
        omitted_required = set()
        for field_name, field in pydantic_model.model_fields.items():
            if field_name in excluded_fields:
                continue
            value = row_values[field_name]
            if value is None:
                if field_name in omittable_fields and field_name in df.columns:
                    if field.is_required():
                        omitted_required.add(field_name)
                    values_to_fill[(position, field_name)] = None
                    continue
                if field_name in empty_cells and field_name not in skipped:
                    # Gravado também quando a linha é recusada por outro campo:
                    # com esse campo em reprocess_columns, a linha fica, e este
                    # mantém o valor que tinha.
                    projected[field_name] = ""
                    values_to_fill[(position, field_name)] = ""
                    restored.add(field_name)
                    continue
                missing_values.append(field_name)
                if field_name in skipped or not (
                    field.is_required() or field_name in written_none_fields
                ):
                    continue
            projected[field_name] = value

        validated_data, rejected = _validate_row(
            _skipping_model(
                pydantic_model, skipped, required_excluded | frozenset(omitted_required)
            ),
            projected,
            rejected_fields,
        )
        if validated_data is None:
            incompatible_fields.update(rejected)
            for field_name in missing_values:
                field = pydantic_model.model_fields[field_name]
                if field.is_required():
                    continue
                if field_name in written_none_fields or field_name in skipped:
                    # O None gravado, que a linha mantida mantém.
                    values_to_fill[(position, field_name)] = None
                    continue
                try:
                    default = field.get_default(call_default_factory=True)
                except ValueError:
                    # This factory needs the fields that are being reprocessed.
                    incompatible_fields.add(field_name)
                else:
                    values_to_fill[(position, field_name)] = default
            continue

        incompatible_fields.update(
            field_name for field_name in restored if validated_data[field_name] != ""
        )

        for field_name in validated_data.keys() & missing_values:
            values_to_fill[(position, field_name)] = validated_data[field_name]

        # O arquivo não guarda o tipo de todo valor (ver _validate_row). O
        # model_dump da linha validada traz o tipo declarado de volta, com o
        # contexto da linha inteira para os validadores. Ele só substitui o
        # valor relido onde diz o mesmo (_validated_where_same): a volta muda
        # o tipo, nunca o conteúdo. O item que um validador muda de novo fica
        # como foi relido.
        for field_name, value in projected.items():
            if field_name not in validated_data:
                continue
            validated = validated_data[field_name]
            # O texto e o número que já voltam com o tipo e o valor validados
            # são a maioria das células; gravá-los de novo, célula a célula,
            # multiplicaria o tempo da retomada.
            if (
                type(validated) is type(value)
                and isinstance(value, (str, int, float))
                and validated == value
            ):
                continue
            # O validado que diz o mesmo é gravado mesmo quando é o próprio
            # valor relido: a célula guarda o Timestamp, e o valor relido, o
            # datetime que a validação devolve.
            if _same_value(validated, value):
                values_to_fill[(position, field_name)] = validated
            elif (converted := _validated_where_same(validated, value)) is not value:
                values_to_fill[(position, field_name)] = converted

    if dropped_nan:
        warnings.warn(
            f"As colunas {[field for field in expected_columns if field in dropped_nan]}, de "
            "campo `Any`, têm NaN em linhas já processadas e nenhum None: o NaN conta como a "
            "célula vazia de um arquivo relido e volta None. Ele só é mantido como valor "
            "quando a coluna guarda algum None, como o da linha pendente de uma execução "
            "interrompida.",
            UserWarning,
            stacklevel=_user_stacklevel(),
        )
    ordered_incompatible = [field for field in expected_columns if field in incompatible_fields]
    return ordered_incompatible, values_to_fill


def _any_object_columns(df: pd.DataFrame, pydantic_model: type[BaseModel]) -> frozenset[str]:
    """Os campos `Any` do modelo com coluna de objetos na entrada."""
    return frozenset(
        field_name
        for field_name, field in pydantic_model.model_fields.items()
        if field_name in df.columns
        and df[field_name].dtype == object
        and _admits_any(field.annotation)
    )


def _any_nan_columns(df: pd.DataFrame, pydantic_model: type[BaseModel]) -> frozenset[str]:
    """Os campos `Any` cujo NaN é valor: os de coluna de objetos com algum None.

    O campo `Any` recebe o NaN que o modelo devolveu, mas a coluna de objetos
    dele chega também de um arquivo relido sem o modelo, com NaN em toda
    célula vazia. Nele o NaN só é valor quando a coluna guarda algum None,
    como o da linha pendente da saída interrompida, que o leitor de arquivo
    não produz. A validação da retomada e o laço das linhas leem o mesmo
    conjunto, tirado da entrada antes de a validação gravar None nela.
    """
    return frozenset(
        field_name
        for field_name in _any_object_columns(df, pydantic_model)
        if any(value is None for value in df[field_name])
    )


def _validate_row(
    model: type[BaseModel],
    projected: dict[str, Any],
    rejected_fields: Callable[[ValidationError], set[str]],
) -> tuple[dict[str, Any] | None, set[str]]:
    """Valida a linha relida e devolve o model_dump dela, ou None e os campos recusados.

    O arquivo muda a forma dos valores: o JSON grava data como texto ISO,
    tupla como lista e Enum pelo valor, a coluna com linha pendente devolve o
    inteiro como float, e o XLSX devolve o bool como 1.0 e a data como
    datetime à meia-noite. O modelo com strict=True recusa essas formas.
    Quando a validação configurada recusa a linha, ela é refeita sem strict,
    e cada campo recusado só é aceito se o valor validado diz o mesmo que o
    relido no caminho de cada erro (_same_at): o "3" num campo int strict
    continua recusado, porque o conteúdo do checkpoint não é o que o modelo
    aceita. O resto do campo não é comparado: o validador de um submodelo que
    muda o valor a cada passada não recusa a linha pela data ao lado.

    A execução grava a resposta pelo nome dos campos, também dentro do
    submodelo, e a linha se valida só pelo nome: o alias igual ao nome de
    outro campo não lê o valor dele.
    """
    try:
        return dump_response(model.model_validate(projected, by_alias=False, by_name=True)), set()
    except ValidationError as error:
        rejected = rejected_fields(error)
        locations = [detail["loc"] for detail in error.errors()]
    try:
        validated_data = dump_response(
            model.model_validate(projected, strict=False, by_alias=False, by_name=True)
        )
    except ValidationError as error:
        return None, rejected_fields(error)
    # O campo que o dump não traz, como o de exclude_if, fica sem o valor para
    # comparar e continua recusado. O campo acusado pelo erro do modelo inteiro,
    # sem caminho próprio, se compara inteiro.
    changed = {
        field_name
        for field_name in rejected & projected.keys()
        if field_name not in validated_data
        or not all(
            _same_at(validated_data[field_name], projected[field_name], path)
            for path in _error_paths(locations, field_name)
        )
    }
    if changed:
        return None, changed
    return validated_data, set()


def _error_paths(
    locations: list[tuple[int | str, ...]], field_name: str
) -> list[tuple[int | str, ...]]:
    """O caminho dentro do campo de cada erro dele; sem erro próprio, o campo inteiro."""
    return [loc[1:] for loc in locations if loc[:1] == (field_name,)] or [()]


def _same_at(validated: object, value: object, path: tuple[int | str, ...]) -> bool:
    """Se o valor validado diz o mesmo que o relido no caminho de um erro de validação.

    Desce pelo caminho enquanto os dois lados têm o passo: a chave do dict, ou
    o índice da lista ou da tupla. O passo que um lado não tem, como o nome do
    membro que o erro de uma união traz, encerra a descida, e o valor inteiro
    daquele nível é comparado. O campo se compara por `_same_value`, e o que
    está abaixo dele, que o checkpoint grava em JSON, pelo texto JSON
    (`_same_json`), como em `_validated_where_same`.
    """
    same = _same_value
    for step in path:
        if isinstance(validated, dict) and isinstance(value, dict):
            if step not in validated or step not in value:
                break
            validated, value = validated[step], value[step]
        elif (
            isinstance(validated, (list, tuple))
            and isinstance(value, (list, tuple))
            and isinstance(step, int)
            and step < min(len(validated), len(value))
        ):
            validated, value = validated[step], value[step]
        else:
            break
        same = _same_json
    return same(validated, value)


# A partir dele, o float não distingue dois inteiros vizinhos.
_EXACT_INT_IN_FLOAT = 2**53


def _same_value(validated: object, value: object) -> bool:  # noqa: PLR0911 (uma saída por forma de valor)
    """Se o valor validado diz o mesmo que o relido, em JSON ou como escalar.

    O CSV, o XLSX e o parquet mudam o tipo de um escalar sem mudar o
    conteúdo, e a comparação em JSON, que distingue 1 de 1.0, não os
    reconhece: o inteiro volta como float, o Decimal do CSV e do XLSX como
    float, o bool do XLSX como 1.0 e a data do XLSX como datetime à
    meia-noite. O Enum é comparado pelo valor, que é
    o que o checkpoint grava.

    A igualdade do Python é mais larga do que essas mudanças, e fica restrita
    a elas: True == 1, mas o bool só diz o mesmo que outro bool ou que o
    float do XLSX; e o float só diz o mesmo que um int abaixo de 2**53, a
    partir do qual a coluna em float pode ter perdido dígitos do inteiro. O
    Decimal diz o mesmo que o float que o representa: Decimal("0.1") é
    diferente do float 0.1 em Python, e é o float que o arquivo relê.
    """
    if isinstance(validated, Enum) and not isinstance(value, Enum):
        return _same_value(validated.value, value)
    if _same_json(validated, value):
        return True
    if not (is_scalar(validated) and is_scalar(value)):
        return False
    if is_bool(validated) != is_bool(value) and not (
        is_bool(validated) and isinstance(value, float)
    ):
        return False
    if (
        isinstance(validated, int)
        and isinstance(value, float)
        and abs(value) >= _EXACT_INT_IN_FLOAT
    ):
        return False
    if type(validated) is dt.date and isinstance(value, dt.datetime):
        return value == dt.datetime.combine(validated, dt.time())
    if isinstance(validated, Decimal):
        return _same_decimal(validated, value)
    return bool(validated == value)


def _same_decimal(validated: Decimal, value: object) -> bool:
    """Se o Decimal validado diz o mesmo que o escalar relido, que não deu o mesmo JSON.

    O NaN só diz o mesmo pelo texto, que o JSON já comparou; a igualdade e o
    float() do Decimal levantam com o NaN sinalizador, de um lado ou do outro.
    """
    if validated.is_nan():
        return False
    if isinstance(value, float):
        return float(validated) == value
    try:
        return bool(validated == value)
    except InvalidOperation:
        return False


def _same_json(left: object, right: object) -> bool:
    """Se os dois valores dão o mesmo texto JSON, como o checkpoint os gravaria.

    Compara o texto, e não os valores relidos, porque 1, 1.0 e True são iguais
    em Python e diferentes no JSON. As chaves ficam na ordem em que vieram,
    porque ordenar poria a chave int 10 antes de 2 de um lado e depois do
    outro. A validação preserva a ordem de um dict; um submodelo sai na ordem
    de declaração, que é a do checkpoint que o dataframeit gravou, e o que
    chega em outra ordem, montado fora dele, fica como veio. Estrutura funda
    demais para o JSON também conta como diferente.
    """
    try:
        return json.dumps(left, default=_json_default) == json.dumps(right, default=_json_default)
    except (TypeError, ValueError, RecursionError):
        return False


def _validated_where_same(
    validated: object, value: object, same: Callable[[object, object], bool] = _same_value
) -> object:
    """O valor validado onde diz o mesmo que o relido, e o relido no resto.

    Desce pelo dict, pela lista e pela tupla de mesmo tamanho dos dois lados,
    item a item e chave a chave na ordem em que vieram: um validador que muda
    um item a cada passada deixa só esse item como relido, e a data ao lado
    dele volta ao tipo do campo. O valor do campo se compara por `same`
    (`_same_value`), e os itens, que o checkpoint grava em JSON, pelo texto
    JSON (`_same_json`), que distingue 1 de 1.0. A chave precisa dizer o
    mesmo como chave JSON, como a int que o checkpoint grava como texto. Sem
    a mesma forma dos dois lados, devolve o próprio `value`.
    """
    if same(validated, value):
        return validated
    if isinstance(validated, dict) and isinstance(value, dict) and len(validated) == len(value):
        if not _same_json(dict.fromkeys(validated), dict.fromkeys(value)):
            return value
        # Os dois dicts têm o mesmo tamanho, e o strict do zip nunca levanta; o
        # teste de mutação o troca por None ou pelo default, que dão o mesmo.
        pairs = zip(validated.items(), value.values(), strict=True)  # pragma: no mutate
        return {key: _validated_where_same(item, read, _same_json) for (key, item), read in pairs}
    if (
        isinstance(validated, (list, tuple))
        and isinstance(value, (list, tuple))
        and len(validated) == len(value)
    ):
        # As duas sequências têm o mesmo tamanho; ver o zip do dict acima.
        items = [
            _validated_where_same(item, read, _same_json)
            for item, read in zip(validated, value, strict=True)  # pragma: no mutate
        ]
        return tuple(items) if isinstance(validated, tuple) else items
    return value


def _apply_processed_values(
    df: pd.DataFrame,
    values: dict[tuple[int, str], Any],
) -> None:
    """Grava por posição os valores completados na validação do checkpoint.

    Cada coluna é regravada de uma vez: com .iat célula a célula, a retomada
    de um checkpoint grande gasta mais tempo gravando do que validando. O
    array de objetos guarda lista ou dict como valor único da célula, como
    _set_cell, e a Series com dtype object impede o pandas de converter uma
    coluna só de datetime em datetime64, que devolveria Timestamp.
    """
    by_column: dict[str, dict[int, Any]] = {}
    for (position, field_name), value in values.items():
        by_column.setdefault(field_name, {})[position] = value
    # No pandas 3, o contexto não lê o DataFrame, e trocá-lo por None, como faz o
    # teste de mutação, não muda nada; no pandas 2, a troca levanta.
    with _silent_slice_write(df):  # pragma: no mutate
        for field_name, cells in by_column.items():
            column = df[field_name].to_numpy(dtype=object, copy=True)
            for position, value in cells.items():
                column[position] = value
            df[field_name] = pd.Series(column, index=df.index, dtype=object)


@contextmanager
def _provider_backend(
    config: LLMConfig,
    pydantic_model: type[BaseModel],
    user_prompt: str,
    trace_mode: str | None,
) -> Generator[ProviderBackend, None, None]:
    """Seleciona e vincula uma única implementação para toda a execução.

    Os módulos de backend (agent, codex, claude_code) são importados na
    chamada: o import do pacote não carrega o que a execução não usa, e as
    funções ficam substituíveis no próprio módulo.
    """
    search_config = config.search_config
    if search_config and search_config.enabled:
        config = with_shared_chat_model(config)
        from .agent import (  # noqa: PLC0415 (ver _provider_backend)
            build_search_agent,
            call_agent,
            field_extractor,
        )

        if not search_config.per_field:
            # Schema fixo: o agente é montado uma vez. Nos modos por campo e por
            # grupo, o modelo de cada chamada é criado na hora, e o agente também.
            search_agent = _BuildOnce(lambda: build_search_agent(pydantic_model, config))
            yield ProviderBackend(
                label="langchain",
                invoke=lambda text: call_agent(
                    text,
                    pydantic_model,
                    user_prompt,
                    config,
                    trace_mode,
                    search_agent=search_agent,
                ),
            )
            return

        extract = field_extractor(pydantic_model, user_prompt, config, trace_mode)
        # Os valores já gravados chegam às condições como a execução os deixa;
        # a linha que a validação da retomada recusou os mantém como relidos.
        as_executed = _executed_form(pydantic_model)
        yield ProviderBackend(
            label="langchain",
            invoke=extract,
            invoke_partial=lambda text, only_fields, known: extract(
                text, only_fields, {name: as_executed(name, value) for name, value in known.items()}
            ),
        )
        return

    if config.provider == "codex":
        from .codex import open_codex_backend  # noqa: PLC0415 (ver _provider_backend)

        with open_codex_backend(config, pydantic_model, user_prompt) as backend:
            yield ProviderBackend(label="codex", invoke=backend.invoke)
        return

    if config.provider == "claude_code":
        from .claude_code import call_claude_code  # noqa: PLC0415 (ver _provider_backend)

        yield ProviderBackend(
            label="claude_code",
            invoke=lambda text: call_claude_code(text, pydantic_model, user_prompt, config),
        )
        return

    langchain_call = call_langchain
    structured_llm = _BuildOnce(lambda: build_structured_llm(pydantic_model, config))
    yield ProviderBackend(
        label="langchain",
        invoke=lambda text: langchain_call(
            text, pydantic_model, user_prompt, config, structured_llm=structured_llm
        ),
    )


def _warn_search_rate_limit(  # noqa: PLR0913 (configuração de busca da execução)
    *,
    num_rows: int,
    num_fields: int,
    parallel_requests: int,
    search_per_field: bool,
    rate_limit_delay: float,
    search_provider: str = "tavily",
) -> None:
    """Avisa quando a configuração pode exceder rate limits do provedor de busca.

    Dispara quando (a) o número de queries concorrentes é alto ou (b) a taxa
    estimada ultrapassa ~80% do limite do provedor. Inclui recomendações de
    ``parallel_requests`` e ``rate_limit_delay`` seguros.
    """
    queries_per_row = num_fields if search_per_field else 1
    total_queries = num_rows * queries_per_row

    provider = get_provider(search_provider)
    provider_limit = provider.requests_per_minute
    provider_name = provider.friendly_name

    issues: list[str] = []

    # _FieldExtractor roda os agentes de uma linha um depois do outro: cada
    # worker tem um agente por vez, em qualquer modo.
    if parallel_requests > _RECOMMENDED_MAX_CONCURRENT_SEARCH_QUERIES:
        issues.append(
            f"queries concorrentes estimadas: {parallel_requests} "
            f"(limite recomendado: {_RECOMMENDED_MAX_CONCURRENT_SEARCH_QUERIES})"
        )

    # Cada worker conclui uma linha a cada queries_per_row execuções do agente,
    # de _MIN_SEARCH_RUN_SECONDS cada, mais a pausa, que vem uma vez por linha.
    # A conta supõe uma busca por execução. É um piso: o agente pode fazer até
    # max_search_calls buscas por execução, e então a taxa real passa da
    # estimada sem aviso. Com search_groups a linha tem menos execuções que
    # campos, e contar uma por campo só sobe a estimativa, que cresce com elas.
    run_seconds = queries_per_row * _MIN_SEARCH_RUN_SECONDS
    estimated_rpm = 60 * parallel_requests * queries_per_row / (run_seconds + rate_limit_delay)

    if estimated_rpm > provider_limit * 0.8:
        issues.append(
            f"taxa estimada ~{estimated_rpm:.0f} req/min "
            f"(limite {provider_name}: ~{provider_limit}/min)"
        )

    if not issues:
        return

    # A dica nunca sobe o paralelismo. A pausa sai da conta de estimated_rpm,
    # invertida para levar a taxa a 70% do limite, abaixo dos 80% do gatilho; o
    # arredondamento para cima não deixa a pausa ficar curta.
    recommended_parallel = min(parallel_requests, _RECOMMENDED_MAX_CONCURRENT_SEARCH_QUERIES)
    recommended_delay = (
        60 * recommended_parallel * queries_per_row / (provider_limit * 0.7) - run_seconds
    )
    recommended_delay = math.ceil(recommended_delay * 10) / 10

    recs = [f"parallel_requests={recommended_parallel}"]
    if rate_limit_delay < recommended_delay:
        recs.append(f"rate_limit_delay={recommended_delay}")

    msg = (
        f"Configuração pode exceder rate limits de busca ({provider_name}). "
        f"parallel_requests={parallel_requests}, search_per_field={search_per_field}, "
        f"rate_limit_delay={rate_limit_delay}s, total de queries estimadas={total_queries}. "
        "Problemas: "
        + "; ".join(issues)
        + ". "
        + "Para evitar HTTP 429, use: "
        + ", ".join(recs)
        + "."
    )

    warnings.warn(msg, stacklevel=_user_stacklevel())


def _validate_search_overrides(
    label: str,
    search_depth: object = None,
    max_results: object = None,
    max_search_calls: object = None,
) -> None:
    """Valida os overrides de busca de um grupo ou campo; None é ausência."""
    if search_depth is not None and search_depth not in ("basic", "advanced"):
        msg = f"{label}: search_depth deve ser 'basic' ou 'advanced'"
        raise ValueError(msg)
    if max_results is not None and (
        isinstance(max_results, bool)
        or not isinstance(max_results, numbers.Real)
        or not 1 <= float(max_results) <= _MAX_RESULTS_LIMIT
    ):
        msg = f"{label}: max_results deve estar entre 1 e 20"
        raise ValueError(msg)
    if max_search_calls is not None and not _is_positive_int(max_search_calls):
        msg = f"{label}: max_search_calls deve ser int >= 1"
        raise ValueError(msg)


def _validate_prompts(label: str, config: dict[str, Any], keys: tuple[str, ...]) -> None:
    """Recusa prompts inválidos ou duas substituições para o mesmo campo."""
    for key in keys:
        value = config.get(key)
        if value is not None and not isinstance(value, str):
            msg = f"{label}: {key} deve ser texto, não {type(value).__name__}"
            raise ValueError(msg)
        if isinstance(value, str) and value and not value.strip():
            msg = f"{label}: {key} não pode conter só espaços"
            raise ValueError(msg)
    if "prompt_replace" in keys and config.get("prompt") and config.get("prompt_replace"):
        msg = f"{label}: use apenas prompt ou prompt_replace, não os dois"
        raise ValueError(msg)


def _is_positive_int(value: object) -> bool:
    """Inteiro >= 1; bool é subclasse de int, mas True não é uma contagem."""
    return isinstance(value, numbers.Integral) and not isinstance(value, bool) and int(value) >= 1


def _validate_field_configs(
    questions: type[BaseModel],
    search_config: SearchConfig | None,
    *,
    use_search: bool,
    search_per_field: bool,
) -> None:
    """Valida json_schema_extra de todos os campos antes de processar linhas.

    Erro de configuração aparece uma vez aqui, e não linha a linha com o
    prefixo de tentativas, que sugere uma falha do provider.
    """
    per_field = bool(search_config and search_config.per_field)

    configured = _collect_configured_fields(questions)
    if configured and not per_field:
        missing = [
            flag
            for flag, on in (
                ("use_search=True", use_search),
                ("search_per_field=True", search_per_field),
            )
            if not on
        ]
        msg = (
            "Campos com configuração em json_schema_extra (prompt, prompt_append, "
            f"search_depth, max_results, max_search_calls) requerem {' e '.join(missing)}"
        )
        raise ValueError(msg)

    for path, field_info, list_depth in _walk_fields(questions):
        extra = field_info.json_schema_extra
        if not isinstance(extra, dict):
            continue

        # Condições só são avaliadas entre campos de primeiro nível
        if "." in path and any(k in extra for k in _CONDITIONAL_KEYS):
            msg = (
                f"Campo '{path}' usa 'condition' ou 'depends_on' em json_schema_extra, "
                "que só são aplicados a campos de primeiro nível do modelo"
            )
            raise ValueError(msg)

        if any(k in extra for k in _FIELD_CONFIG_KEYS):
            # Um item de lista é enriquecido no próprio dicionário; uma segunda
            # lista no caminho não tem item único onde gravar o valor.
            if list_depth > 1:
                msg = (
                    f"Campo '{path}' tem configuração de busca dentro de uma lista que "
                    "está dentro de outra lista, o que não é suportado"
                )
                raise ValueError(msg)
            _validate_prompts(
                f"Campo '{path}'", extra, ("prompt", "prompt_replace", "prompt_append")
            )
            _validate_search_overrides(
                f"Campo '{path}'",
                extra.get("search_depth"),
                extra.get("max_results"),
                extra.get("max_search_calls"),
            )

    # Sem busca por campo, todos os campos saem de uma única chamada, e não há
    # momento para avaliar a condição antes de extrair o campo.
    if not per_field:
        conditional_fields = [
            field_name
            for field_name, field_info in questions.model_fields.items()
            if isinstance(field_info.json_schema_extra, dict)
            and any(k in field_info.json_schema_extra for k in _CONDITIONAL_KEYS)
        ]
        if conditional_fields:
            msg = (
                f"Campos {conditional_fields} usam 'condition' ou 'depends_on' em "
                "json_schema_extra, que só são aplicados com use_search=True e "
                "search_per_field=True"
            )
            raise ValueError(msg)
        return

    _, dependencies = get_field_execution_order(questions, _model_field_configs(questions))
    if bool(search_config.groups):
        get_group_execution_units(questions, search_config.groups, dependencies)


def _validate_search_groups(  # noqa: C901 (uma checagem por regra de search_groups)
    search_groups: dict[str, dict[str, Any]],
    pydantic_model: type[BaseModel],
    *,
    use_search: bool,
    search_per_field: bool,
) -> dict[str, SearchGroupConfig]:
    """Valida e converte search_groups para SearchGroupConfig.

    Args:
        search_groups: Dicionário de grupos de busca do usuário.
        pydantic_model: Modelo Pydantic para validar campos.
        use_search: Se busca está habilitada.
        search_per_field: Se modo per_field está habilitado.

    Returns:
        Dicionário de SearchGroupConfig validados.

    Raises:
        ValueError: Se validação falhar.
    """
    # Validar pré-requisitos
    if not use_search:
        msg = "search_groups requer use_search=True"
        raise ValueError(msg)
    if not search_per_field:
        msg = "search_groups requer search_per_field=True"
        raise ValueError(msg)

    expected_fields = set(pydantic_model.model_fields.keys())
    all_grouped_fields = set()
    validated_groups = {}

    for group_name, group_config in search_groups.items():
        # Validar estrutura
        if not isinstance(group_config, dict):
            msg = f"Grupo '{group_name}' deve ser um dicionário"
            raise ValueError(msg)  # noqa: TRY004 (ValueError, como todo erro de search_groups)
        if "fields" not in group_config:
            msg = f"Grupo '{group_name}' deve ter chave 'fields'"
            raise ValueError(msg)

        fields = group_config["fields"]
        if not isinstance(fields, list) or not fields:
            msg = f"Grupo '{group_name}': 'fields' deve ser uma lista não-vazia"
            raise ValueError(msg)

        # Validar que campos existem no modelo
        unknown_fields = set(fields) - expected_fields
        if unknown_fields:
            msg = (
                f"Grupo '{group_name}': campos {unknown_fields} não existem no modelo Pydantic. "
                f"Campos disponíveis: {expected_fields}"
            )
            raise ValueError(msg)

        # Validar que campos não pertencem a múltiplos grupos
        duplicate_fields = all_grouped_fields & set(fields)
        if duplicate_fields:
            msg = (
                f"Campos {duplicate_fields} pertencem a múltiplos grupos. "
                f"Cada campo pode pertencer a apenas um grupo."
            )
            raise ValueError(msg)
        all_grouped_fields.update(fields)

        # Validar que campos do grupo não têm json_schema_extra de busca
        for field_name in fields:
            field_info = pydantic_model.model_fields[field_name]
            extra = field_info.json_schema_extra
            if isinstance(extra, dict):
                conflicting_keys = set(extra.keys()) & set(_FIELD_CONFIG_KEYS)
                if conflicting_keys:
                    msg = (
                        f"Campo '{field_name}' no grupo '{group_name}' tem json_schema_extra "
                        f"com chaves de busca {conflicting_keys}. Escolha entre configuração "
                        f"per-field (json_schema_extra) ou grupo (search_groups), não ambos."
                    )
                    raise ValueError(msg)

        _validate_prompts(f"Grupo '{group_name}'", group_config, ("prompt",))
        _validate_search_overrides(
            f"Grupo '{group_name}'",
            group_config.get("search_depth"),
            group_config.get("max_results"),
            group_config.get("max_search_calls"),
        )

        # Criar SearchGroupConfig
        validated_groups[group_name] = SearchGroupConfig(
            fields=fields,
            prompt=group_config.get("prompt"),
            max_results=group_config.get("max_results"),
            search_depth=group_config.get("search_depth"),
            max_search_calls=group_config.get("max_search_calls"),
        )

    return validated_groups


def _has_field_config(pydantic_model: type[BaseModel]) -> bool:
    """Verifica se algum campo, inclusive aninhado, tem configuração per-field."""
    return bool(_collect_configured_fields(pydantic_model))


# O que a execução grava na coluna de status; a célula vazia marca a linha pendente.
_STATUS_VALUES = ("processed", "error")


def _validate_status_column(
    df: pd.DataFrame, status_col: str, *, text_column: str, expected_columns: list[str]
) -> None:
    """Recusa a coluna de status que ocuparia uma coluna de entrada, de resposta ou de controle.

    A execução grava o status de cada linha por cima do que houver na coluna e,
    na saída sem erro, a remove. Uma coluna existente só serve de status quando
    traz apenas o que a própria execução grava nela, e aí é a de uma execução
    anterior, que se retoma.
    """
    if status_col == text_column:
        msg = (
            f"status_column='{status_col}' é a coluna de texto, e a coluna de status "
            "ocuparia o texto de entrada. Escolha outro nome."
        )
        raise ValueError(msg)
    if status_col in expected_columns:
        msg = (
            f"status_column='{status_col}' é um campo do modelo, e a coluna de status "
            "ocuparia a resposta dele. Escolha outro nome."
        )
        raise ValueError(msg)
    # _column_rank diz quais colunas a execução acrescenta; o status padrão é a
    # única delas que pode ser a coluna de status, e com o nome de outra cada
    # linha teria dois valores numa célula. O nome fica recusado mesmo com a
    # opção que cria a coluna desligada, porque uma retomada com ela ligada
    # (track_tokens é o padrão) gravaria por cima do status.
    if (
        status_col != "_dataframeit_status"
        and _column_rank(
            status_col,
            # O segundo argumento só decide o rank do próprio status padrão, que a
            # linha de cima já excluiu: o teste de mutação o troca à toa.
            "_dataframeit_status",  # pragma: no mutate
        )
        != _USER_COLUMN
    ):
        msg = (
            f"status_column='{status_col}' é o nome de uma coluna de controle do dataframeit, "
            "gravada na mesma linha que o status, e um dos dois valores se perderia. "
            "Escolha outro nome."
        )
        raise ValueError(msg)
    if status_col in df.columns:
        status = df[status_col]
        foreign = status[status.notna() & ~status.isin(_STATUS_VALUES)]
        if not foreign.empty:
            # tolist() devolve escalares Python, cujo repr não traz o tipo do numpy.
            value, label = foreign.tolist()[0], foreign.index.tolist()[0]
            allowed = ", ".join(repr(status_value) for status_value in _STATUS_VALUES)
            msg = (
                f"A coluna de status '{status_col}' já existe no DataFrame com valores que não "
                f"são de status do dataframeit ({value!r} na linha {label!r}), e a execução os "
                f"leria como status e sobrescreveria a coluna. A coluna de status só guarda "
                f"{allowed} ou vazio: passe em status_column outro nome, ou renomeie a coluna "
                "existente."
            )
            raise ValueError(msg)


# Os overloads ligam o tipo da entrada ao da saída. O de pandas vem primeiro:
# sem polars instalado, PolarsDataFrame não resolve e casaria com qualquer
# entrada se fosse o primeiro.
@overload
def dataframeit(
    data: pd.DataFrame | pd.Series | list[Any] | dict[Any, Any],
    questions: type[BaseModel] | None = None,
    prompt: str | None = None,
    perguntas: type[BaseModel] | None = None,  # Deprecated: use 'questions'
    resume: bool = True,  # noqa: FBT001, FBT002 (posicional na API pública)
    reprocess_columns: str | list[str] | tuple[str, ...] | None = None,
    model: str | None = None,
    provider: str = "openai",
    status_column: str | None = None,
    text_column: str | None = None,
    api_key: str | None = None,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    rate_limit_delay: float = 0.0,
    track_tokens: bool = True,  # noqa: FBT001, FBT002 (posicional na API pública)
    model_kwargs: dict[str, Any] | None = None,
    parallel_requests: int = 1,
    # Parâmetros de busca web
    use_search: bool = False,  # noqa: FBT001, FBT002 (posicional na API pública)
    search_provider: str = "tavily",
    search_per_field: bool = False,  # noqa: FBT001, FBT002 (posicional na API pública)
    max_results: int = 5,
    search_depth: str = "basic",
    max_search_calls: int = 10,
    search_groups: dict[str, dict[str, Any]] | None = None,
    save_trace: bool | Literal["full", "minimal"] | None = None,  # noqa: FBT001 (posicional na API pública)
    batch_size: int | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
    on_row_done: Callable[[RowResult], None] | None = None,
) -> pd.DataFrame: ...


@overload
def dataframeit(
    data: PolarsDataFrame | PolarsSeries,
    questions: type[BaseModel] | None = None,
    prompt: str | None = None,
    perguntas: type[BaseModel] | None = None,  # Deprecated: use 'questions'
    resume: bool = True,  # noqa: FBT001, FBT002 (posicional na API pública)
    reprocess_columns: str | list[str] | tuple[str, ...] | None = None,
    model: str | None = None,
    provider: str = "openai",
    status_column: str | None = None,
    text_column: str | None = None,
    api_key: str | None = None,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    rate_limit_delay: float = 0.0,
    track_tokens: bool = True,  # noqa: FBT001, FBT002 (posicional na API pública)
    model_kwargs: dict[str, Any] | None = None,
    parallel_requests: int = 1,
    # Parâmetros de busca web
    use_search: bool = False,  # noqa: FBT001, FBT002 (posicional na API pública)
    search_provider: str = "tavily",
    search_per_field: bool = False,  # noqa: FBT001, FBT002 (posicional na API pública)
    max_results: int = 5,
    search_depth: str = "basic",
    max_search_calls: int = 10,
    search_groups: dict[str, dict[str, Any]] | None = None,
    save_trace: bool | Literal["full", "minimal"] | None = None,  # noqa: FBT001 (posicional na API pública)
    batch_size: int | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
    on_row_done: Callable[[RowResult], None] | None = None,
) -> PolarsDataFrame: ...


# Fora do teste de mutação: o corpo ainda excede o tempo do job; os avisos já
# reconhecem a função renomeada e o trampolim do mutmut.
def dataframeit(  # noqa: C901, PLR0912, PLR0913, PLR0915, PLR0917 (API pública: cada parâmetro é uma opção documentada)
    data: pd.DataFrame | pd.Series | PolarsDataFrame | PolarsSeries | list[Any] | dict[Any, Any],
    questions: type[BaseModel] | None = None,
    prompt: str | None = None,
    perguntas: type[BaseModel] | None = None,  # Deprecated: use 'questions'
    resume: bool = True,  # noqa: FBT001, FBT002 (posicional na API pública)
    reprocess_columns: str | list[str] | tuple[str, ...] | None = None,
    model: str | None = None,
    provider: str = "openai",
    status_column: str | None = None,
    text_column: str | None = None,
    api_key: str | None = None,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    rate_limit_delay: float = 0.0,
    track_tokens: bool = True,  # noqa: FBT001, FBT002 (posicional na API pública)
    model_kwargs: dict[str, Any] | None = None,
    parallel_requests: int = 1,
    # Parâmetros de busca web
    use_search: bool = False,  # noqa: FBT001, FBT002 (posicional na API pública)
    search_provider: str = "tavily",
    search_per_field: bool = False,  # noqa: FBT001, FBT002 (posicional na API pública)
    max_results: int = 5,
    search_depth: str = "basic",
    max_search_calls: int = 10,
    search_groups: dict[str, dict[str, Any]] | None = None,
    save_trace: bool | Literal["full", "minimal"] | None = None,  # noqa: FBT001 (posicional na API pública)
    batch_size: int | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
    on_row_done: Callable[[RowResult], None] | None = None,
) -> pd.DataFrame | PolarsDataFrame:  # pragma: no mutate block
    """Processa textos usando LLMs para extrair informações estruturadas.

    Suporta múltiplos tipos de entrada:
    - pandas.DataFrame: Retorna DataFrame com colunas extraídas
    - polars.DataFrame: Retorna DataFrame polars com colunas extraídas
    - pandas.Series: Retorna DataFrame com resultados indexados
    - polars.Series: Retorna DataFrame polars com resultados
    - list: Retorna DataFrame pandas com índice numérico
    - dict: Retorna DataFrame pandas com as chaves como índice

    Args:
        data: Dados contendo textos (DataFrame, Series, list ou dict).
        questions: Modelo Pydantic definindo estrutura a extrair.
        prompt: Template do prompt (use {texto} para indicar onde inserir o texto).
        perguntas: (Deprecated) Use 'questions'.
        resume: Se True (padrão), continua de onde parou: só as linhas sem status
            são processadas, e as marcadas 'processed' ou 'error' ficam como estão.
            Se False, processa toda linha que não esteja 'processed'. Com False e a
            coluna de status presente, se alguma coluna do modelo já existe no
            DataFrame e reprocess_columns não foi passado, emite um aviso e devolve
            a entrada sem processar; fora desse caso, a linha 'processed' é
            validada como com True: o campo obrigatório que ela não tem levanta
            ValueError, e o que tem padrão ou admite None recebe o padrão, ou
            None, com aviso.
            Quando a entrada chega sem a coluna de status e não há checkpoint a
            reler, com True ou False, toda linha conta como nova, e uma coluna com
            algum valor e com o nome de um campo do modelo levanta ValueError
            antes de qualquer chamada, a menos que esteja em reprocess_columns; o
            mesmo vale para o valor nas linhas sem status, com a coluna de status
            na entrada, e nas que o checkpoint deixou pendentes, na retomada de um
            checkpoint_path. A coluna de um campo com exclude=True, que a execução
            não grava, não conta.
        reprocess_columns: Lista de colunas para forçar reprocessamento. Útil para
            atualizar colunas específicas com novas instruções sem perder outras.
        model: Nome do modelo LLM. Se None, usa o default do provider em
            DEFAULT_MODELS; com 'codex' e 'claude_code', o runtime escolhe.
        provider: Provider do LangChain ('openai', 'google_genai', 'anthropic', etc),
            'claude_code' ou 'codex'. Codex usa o SDK Python oficial.
        status_column: Coluna para rastrear progresso. Se None, usa
            "_dataframeit_status". Não pode ser a coluna de texto, um campo do
            modelo nem outra coluna de controle do dataframeit, e a coluna
            existente com esse nome só serve se trouxer apenas 'processed',
            'error' ou vazio; fora disso, levanta ValueError.
        text_column: Nome da coluna com textos. Se None em um DataFrame, a lib
                    infere dentre TEXT_COLUMN_CANDIDATES ('texto', 'text',
                    'decisao', 'content', 'content_text'); DataFrames com uma
                    única coluna usam-na direto. Se nenhum candidato bater e o
                    DataFrame tiver múltiplas colunas, levanta ValueError.
                    Automático para Series/list/dict.
        api_key: Chave API específica.
        max_retries: Número total de tentativas por linha, contando a primeira (int >= 1).
        base_delay: Delay base para retry.
        max_delay: Delay máximo para retry.
        rate_limit_delay: Pausa em segundos que cada worker faz depois de cada linha
            processada com sucesso (padrão: 0.0). Linhas com erro não pausam. O teto de
            vazão fica em parallel_requests * 60 / rate_limit_delay linhas por minuto,
            sem contar o tempo das próprias chamadas.
        track_tokens: Se True, grava o uso de tokens por linha e, no fim, registra o
            resumo de uso em INFO no logger "dataframeit.stats" (padrão: True). Sem
            logging configurado, o resumo sai em stderr; setLevel(logging.WARNING)
            nesse logger o silencia e mantém as colunas.
        model_kwargs: Parâmetros extras do modelo (ex: temperature, reasoning_effort).
            Com provider='codex', aceita somente effort.
        parallel_requests: Número de requisições paralelas (padrão: 1 = sequencial).
            Se > 1, processa múltiplas linhas simultaneamente.
            Quando uma linha falha depois de todas as tentativas com erro de rate limit,
            o número de workers cai pela metade. No codex, o rate limit e a sobrecarga do
            app-server são reconhecidos pelo código ou pelo status, sem ler a mensagem;
            no claude_code, o rate limit (429) e a sobrecarga (529), pelo status que o
            SDK informa a partir do claude-agent-sdk 0.1.76 (do 0.1.48 ao 0.1.75, sem o
            status, são repetidos como falha transitória e não reduzem os workers); nos
            providers do LangChain, o rate limit é reconhecido pelo status 429 ou pela
            classe do erro e, sem nenhum dos dois, pelo texto da mensagem.
            Dica: use track_tokens=True para ver métricas de throughput (RPM, TPM) e calibrar.
        use_search: Se True, habilita busca web antes de processar. Padrão: False.
        search_provider: Provedor de busca web a usar. Opções:
            - "tavily": Motor de busca otimizado para IA (padrão). Requer TAVILY_API_KEY.
              Melhor para volume baixo-médio (<2667 buscas/mês).
            - "exa": Motor de busca semântico. Requer EXA_API_KEY.
              Mais econômico para alto volume (>2667 buscas/mês).
        search_per_field: Se True, executa um agente separado para cada campo do modelo Pydantic.
            Útil quando o modelo tem muitos campos e um único contexto ficaria sobrecarregado.
            Padrão: False (um agente responde todos os campos).
        max_results: Número máximo de resultados por busca (1-20). Padrão: 5.
        search_depth: Profundidade da busca - "basic" (1 crédito) ou "advanced" (2 créditos).
            Apenas para Tavily. Padrão: "basic".
        max_search_calls: Máximo de buscas por execução do agente (int >= 1). Ao atingi-lo,
            as buscas seguintes são bloqueadas e o agente responde com o que encontrou.
            Aceita override por grupo (search_groups) e por campo (json_schema_extra).
            Padrão: 10.
        search_groups: Grupos de campos que compartilham contexto de busca. Formato:
            {"nome_grupo": {"fields": ["campo1", "campo2"], "prompt": "...", ...}}
            Chaves de cada grupo: "fields" (obrigatória, lista não vazia de campos
            do modelo, cada campo em um só grupo), "prompt" (substitui o prompt
            do grupo; {query} vale como {texto}), "max_results", "search_depth" e
            "max_search_calls" (substituem os valores globais para o grupo).
            Permite reduzir chamadas de API quando múltiplos campos precisam do mesmo contexto.
            Requer use_search=True e search_per_field=True.
        save_trace: Salva o trace completo do raciocínio do agente. Requer use_search=True.
            - None/False: Desabilitado (padrão)
            - True/"full": Trace completo com conteúdo das mensagens
            - "minimal": Apenas queries e contagens, sem conteúdo de tool results
            Colunas geradas: "_trace" (agente único) ou "_trace_{campo}" (per_field);
            com search_groups, "_trace_{grupo}" para cada grupo e "_trace_{campo}"
            para os campos fora de grupos.
        batch_size: Se definido (int >= 1), salva o DataFrame em `checkpoint_path` a cada
            N linhas processadas nesta execução. Combinado com `resume=True`, permite
            retomar execuções longas após kill/crash sem perder progresso.
            Requer `checkpoint_path`.
        checkpoint_path: Caminho do arquivo de checkpoint, em texto ou `os.PathLike`.
            Formato inferido pela extensão (.csv, .xlsx, .parquet). Escrita atômica
            via .tmp + rename. Requer `batch_size`.
            Ao lado dele, `<checkpoint>.dataframeit.json` guarda a assinatura da
            execução; com `resume=True`, a mesma chamada retoma do checkpoint
            quando a assinatura bate, e senão avisa e o sobrescreve.
            Exemplo: ``dataframeit(df, Model, PROMPT, batch_size=100, checkpoint_path="ckpt.xlsx")``.
        on_row_done: Função chamada uma vez por linha processada nesta chamada, inclusive
            em erro, depois de a linha estar gravada no DataFrame, com um `RowResult`
            (rótulo do índice, posição entre as linhas processadas, sucesso, texto de
            `_error_details`, campos gravados e segundos decorridos). A linha pulada por
            `resume` e a que uma interrupção deixou sem status não são entregues. Com
            `parallel_requests` > 1, a chamada vem da thread do worker que processou a
            linha, enquanto as outras seguem gravando: a função sincroniza o que
            compartilhar e lê o que precisa do `RowResult`, e não do DataFrame. Uma
            exceção dentro dela vira aviso, e a execução segue.

    Returns:
        DataFrame (polars para entrada polars, pandas nos demais casos) com uma
        coluna por campo do modelo, além das colunas de controle. Uma linha sem
        texto não vai ao LLM e fica com status 'error' e "Texto ausente" em
        "_error_details". As colunas de status e "_error_details" são removidas
        quando nenhuma linha tem status 'error' nem detalhe gravado em
        "_error_details", o que inclui os avisos de retry de linhas bem-sucedidas.

    Raises:
        ValueError: Se parâmetros obrigatórios faltarem.
        TypeError: Se tipo de dados não for suportado.
    """
    # Compatibilidade com API antiga
    if questions is None and perguntas is not None:
        warnings.warn(
            "O parâmetro 'perguntas' está depreciado; use 'questions'.",
            DeprecationWarning,
            stacklevel=_user_stacklevel(),
        )
        questions = perguntas
    elif questions is None:
        msg = "Parâmetro 'questions' é obrigatório"
        raise ValueError(msg)

    if prompt is None:
        msg = "Parâmetro 'prompt' é obrigatório"
        raise ValueError(msg)

    # Se {texto} não estiver no template, adiciona automaticamente ao final
    if "{texto}" not in prompt:
        prompt = prompt.rstrip() + "\n\nTexto a analisar:\n{texto}"

    # bool é subclasse de int, mas True não é uma contagem de tentativas.
    if (
        not isinstance(max_retries, numbers.Integral)
        or isinstance(max_retries, bool)
        or max_retries < 1
    ):
        msg = (
            f"max_retries deve ser int >= 1 (número total de tentativas por linha); "
            f"recebido {max_retries!r}"
        )
        raise ValueError(msg)

    # Validar parâmetros de checkpoint
    if (batch_size is None) != (checkpoint_path is None):
        msg = "batch_size e checkpoint_path devem ser usados juntos"
        raise ValueError(msg)
    checkpoint = None
    if batch_size is not None and checkpoint_path is not None:
        if (
            not isinstance(batch_size, numbers.Integral)
            or isinstance(batch_size, bool)
            or batch_size < 1
        ):
            msg = f"batch_size deve ser int >= 1; recebido {batch_size!r}"
            raise ValueError(msg)
        path = Path(checkpoint_path)
        _validate_checkpoint_extension(path)
        checkpoint = _Checkpoint(path, batch_size)

    # Providers de SDK usam structured output direto, sem o agente LangChain de busca.
    if use_search and provider in {"claude_code", "codex"}:
        msg = (
            f"Busca web (use_search=True) não é suportada com provider='{provider}'. "
            "Use um provider LangChain como 'google_genai' ou 'openai' para busca web."
        )
        raise ValueError(msg)

    # Validar parâmetros de busca
    if use_search:
        available_providers = get_available_providers()
        if search_provider not in available_providers:
            msg = f"search_provider deve ser um de {available_providers}"
            raise ValueError(msg)
        if search_provider == "tavily" and search_depth not in ("basic", "advanced"):
            msg = "search_depth deve ser 'basic' ou 'advanced'"
            raise ValueError(msg)
        if not 1 <= max_results <= _MAX_RESULTS_LIMIT:
            msg = "max_results deve estar entre 1 e 20"
            raise ValueError(msg)
        if not _is_positive_int(max_search_calls):
            msg = f"max_search_calls deve ser int >= 1; recebido {max_search_calls!r}"
            raise ValueError(msg)

    # Validar e normalizar save_trace
    trace_mode = None
    if bool(save_trace):
        if not use_search:
            msg = "save_trace requer use_search=True"
            raise ValueError(msg)
        if save_trace is True:
            trace_mode = "full"
        elif save_trace in ("full", "minimal"):
            trace_mode = save_trace
        else:
            msg = "save_trace deve ser True, 'full' ou 'minimal'"
            raise ValueError(msg)

    # Criar SearchConfig se busca habilitada
    search_config = None
    if use_search:
        search_config = SearchConfig(
            enabled=True,
            provider=search_provider,
            per_field=search_per_field,
            max_results=max_results,
            search_depth=search_depth,
            max_search_calls=max_search_calls,
        )

    # Converter para pandas se necessário
    annotations = {name: field.annotation for name, field in questions.model_fields.items()}
    df_pandas, conversion_info = to_pandas(data, annotations)

    # Determinar coluna de texto
    is_dataframe_type = conversion_info.original_type in (
        ORIGINAL_TYPE_PANDAS_DF,
        ORIGINAL_TYPE_POLARS_DF,
    )

    if is_dataframe_type:
        if text_column is None:
            matches = [c for c in TEXT_COLUMN_CANDIDATES if c in df_pandas.columns]
            if matches:
                text_column = matches[0]
                if len(matches) > 1:
                    warnings.warn(
                        f"Múltiplas colunas candidatas encontradas: {matches}. "
                        f"Usando '{text_column}'. Passe text_column= para suprimir este aviso.",
                        UserWarning,
                        stacklevel=_user_stacklevel(),
                    )
            elif len(df_pandas.columns) == 1:
                text_column = df_pandas.columns[0]
            else:
                msg = (
                    f"Nenhuma coluna de texto identificada entre {TEXT_COLUMN_CANDIDATES}. "
                    f"Colunas disponíveis: {list(df_pandas.columns)}. "
                    f"Passe text_column= explicitamente."
                )
                raise ValueError(msg)
        if text_column not in df_pandas.columns:
            msg = (
                f"Coluna '{text_column}' não encontrada no DataFrame. "
                f"Colunas disponíveis: {list(df_pandas.columns)}."
            )
            raise ValueError(msg)
    else:
        # Para Series/list/dict, usa coluna interna
        text_column = DEFAULT_TEXT_COLUMN

    # Extrair campos do modelo Pydantic
    expected_columns = list(questions.model_fields.keys())
    if not expected_columns:
        msg = "Modelo Pydantic não pode estar vazio"
        raise ValueError(msg)

    # Cada campo extraído é gravado numa coluna de mesmo nome, que não pode ser
    # a do texto de entrada.
    if text_column in expected_columns:
        msg = (
            f"O campo '{text_column}' do modelo tem o nome da coluna de texto, e a "
            "resposta sobrescreveria o texto de entrada. Renomeie o campo ou a coluna."
        )
        raise ValueError(msg)

    # Validar e processar search_groups
    if bool(search_groups):
        validated_groups = _validate_search_groups(
            search_groups,
            questions,
            use_search=use_search,
            search_per_field=search_per_field,
        )
        # _validate_search_groups exige use_search=True, com que a SearchConfig
        # foi montada acima; o cast só registra isso para o ty.
        cast("SearchConfig", search_config).groups = validated_groups

    # Validar reprocess_columns
    if reprocess_columns is not None:
        if not isinstance(reprocess_columns, (list, tuple)):
            reprocess_columns = [reprocess_columns]
        # Verificar que todas as colunas a reprocessar estão no modelo
        invalid_cols = [col for col in reprocess_columns if col not in expected_columns]
        if invalid_cols:
            msg = (
                f"Colunas {invalid_cols} não estão no modelo Pydantic. "
                f"Colunas disponíveis: {expected_columns}"
            )
            raise ValueError(msg)

    status_col = status_column or "_dataframeit_status"
    _validate_status_column(
        df_pandas, status_col, text_column=text_column, expected_columns=expected_columns
    )
    complex_fields = get_complex_fields(questions)
    # Campos cujo NaN é valor, separado do nulo (`_keeps_nan`).
    nan_columns = frozenset(
        field_name
        for field_name, field in questions.model_fields.items()
        if _keeps_nan(field.annotation)
    )

    # Entrada sem coluna de status é entrada nova; se o checkpoint já existe e
    # é desta execução, ela continua dele. Quem carrega o checkpoint à mão e o
    # passa como entrada já traz a coluna, e o arquivo não é lido de novo.
    signature = None
    empty_text_fields: frozenset[str] = frozenset()
    reprocessed_columns = frozenset(reprocess_columns or [])
    input_has_status = status_col in df_pandas.columns
    if checkpoint is not None:
        signature = _run_signature(
            questions,
            prompt,
            provider=provider,
            model=model if model is not None else DEFAULT_MODELS.get(provider),
            model_kwargs=model_kwargs,
            search_config=search_config,
            text_column=text_column,
            status_col=status_col,
        )
        # Tirado da entrada antes da releitura, que a altera.
        checkpoint = dataclasses.replace(
            checkpoint,
            meta={"signature": signature, "texts": _text_hashes(df_pandas[text_column])},
            structures=frozenset(complex_fields),
            any_fields=frozenset(
                field_name
                for field_name, field in questions.model_fields.items()
                if _admits_any(field.annotation)
            ),
            nan_fields=nan_columns,
            decimal_fields=frozenset(
                field_name
                for field_name, field in questions.model_fields.items()
                if _accepts_only_decimal(field.annotation)
            ),
            scalar_fields=frozenset(
                field_name
                for field_name, field in questions.model_fields.items()
                if _mixes_text_and_number(field.annotation)
            ),
        )
        if resume and status_col not in df_pandas.columns and checkpoint.path.exists():
            with _silent_slice_write(df_pandas):
                empty_text_fields = _resume_from_checkpoint(
                    df_pandas,
                    checkpoint,
                    questions,
                    text_column=text_column,
                    status_col=status_col,
                    reprocessed_columns=reprocessed_columns,
                    nan_columns=nan_columns,
                )

    # Entradas vazias têm um resultado bem definido e não dependem de provider.
    if df_pandas.empty:
        _setup_columns(
            df_pandas,
            expected_columns,
            status_column,
            track_tokens=track_tokens,
            search_config=search_config,
            trace_mode=trace_mode,
            pydantic_model=questions,
        )
        return from_pandas(df_pandas, conversion_info, status_col)

    # Sem coluna de status, nem a da execução retomada do checkpoint, toda
    # linha conta como nova; com ela na entrada, a linha sem status. A linha
    # pendente do checkpoint retomado já foi checada antes de a entrada
    # recebê-lo. Vem depois da entrada vazia, que não tem linha a sobrescrever.
    if status_col not in df_pandas.columns or input_has_status:
        _raise_if_overwritten(
            df_pandas,
            questions,
            reprocessed_columns,
            nan_columns,
            text_column=text_column,
            status_col=status_col,
            pending=df_pandas[status_col].isna().to_numpy() if input_has_status else None,
        )
    # A coluna do campo com exclude=True, que a execução não grava, não conta.
    excluded_fields = _excluded_fields(questions)
    value_columns = [col for col in expected_columns if col not in excluded_fields]
    existing_cols = [col for col in value_columns if col in df_pandas.columns]
    # Sem a coluna de status, toda linha é nova, como com resume=True, e o
    # valor que a execução sobrescreveria já foi recusado em _raise_if_overwritten.
    if existing_cols and not resume and not bool(reprocess_columns) and input_has_status:
        _warn_at_user(
            f"Colunas {existing_cols} já existem. Use resume=True para continuar ou renomeie-as."
        )
        return from_pandas(df_pandas, conversion_info, status_col)

    # O NaN de `Any` que a validação abaixo conta como valor segue valor no
    # laço das linhas, para o reprocessamento campo a campo e para as marcas da
    # interrupção. Tirado antes dela, que grava None nas células ausentes.
    any_nan_columns = _any_nan_columns(df_pandas, questions)
    if checkpoint is not None and status_col in df_pandas.columns and not input_has_status:
        # A coluna que veio do checkpoint desta execução (`_resume_from_checkpoint`)
        # guarda o NaN à parte da célula vazia, também sem nenhum None.
        any_nan_columns |= checkpoint.any_fields.intersection(df_pandas.columns)
    # Com resume True ou False, a linha 'processed' não vai ao modelo; o campo
    # obrigatório que ela não tem é acusado aqui, em vez de sair vazio.
    incompatible_columns, processed_values = _validate_processed_rows(
        df_pandas,
        status_col,
        questions,
        complex_fields,
        empty_text_fields=empty_text_fields,
        any_nan_columns=any_nan_columns,
    )
    row_nan_columns = nan_columns | any_nan_columns
    # O campo sem coluna nunca foi perguntado às linhas já processadas, que só
    # o recebem pelo reprocessamento; sem ele, ficam com o padrão ou None.
    has_processed_rows = (
        status_col in df_pandas.columns and df_pandas[status_col].isin(["processed"]).any()
    )
    new_columns = (
        [col for col in value_columns if col not in df_pandas.columns] if has_processed_rows else []
    )
    to_reprocess = {*reprocessed_columns, *incompatible_columns, *new_columns}
    suggested_columns = [col for col in expected_columns if col in to_reprocess]
    uncovered_columns = [
        column for column in incompatible_columns if column not in reprocessed_columns
    ]
    if uncovered_columns:
        msg = (
            "O DataFrame contém linhas processadas incompatíveis com o modelo atual: "
            f"campos incompatíveis {uncovered_columns}. "
            f"Inclua-os em reprocess_columns={suggested_columns!r}."
        )
        raise ValueError(msg)
    unfilled_columns = [column for column in new_columns if column not in reprocessed_columns]
    if unfilled_columns:
        warnings.warn(
            f"As linhas já processadas não têm as colunas {unfilled_columns}, e esta "
            "execução não as preenche nelas: ficam com o padrão de cada campo, ou None. "
            f"Para preenchê-las, passe reprocess_columns={suggested_columns!r}.",
            UserWarning,
            stacklevel=_user_stacklevel(),
        )

    # Um checkpoint sem posição pendente não depende do provider nem de autenticação.
    if (
        resume
        and not bool(reprocess_columns)
        and status_col in df_pandas.columns
        and df_pandas[status_col].notna().all()
    ):
        _setup_columns(
            df_pandas,
            expected_columns,
            status_column,
            track_tokens=track_tokens,
            search_config=search_config,
            trace_mode=trace_mode,
            pydantic_model=questions,
        )
        _as_object_columns(df_pandas, expected_columns, nan_columns)
        _apply_processed_values(df_pandas, processed_values)
        if complex_fields:
            normalize_complex_columns(df_pandas, complex_fields)
        return from_pandas(df_pandas, conversion_info, status_col, annotations)

    # Os resultados são gravados por rótulo (df.at); com rótulo repetido, o
    # resultado de uma linha cairia em outra. Um checkpoint já concluído, que
    # retornou acima, só é completado por posição e não passa por aqui.
    if not df_pandas.index.is_unique:
        duplicated = df_pandas.index[df_pandas.index.duplicated()].unique().tolist()
        msg = (
            f"O índice tem rótulos repetidos ({duplicated[:5]}). "
            "Use df.reset_index(drop=True) antes de chamar dataframeit."
        )
        raise ValueError(msg)

    if model is None and provider not in _RUNTIME_DEFAULT_PROVIDERS:
        if provider not in DEFAULT_MODELS:
            msg = (
                f"provider='{provider}' não tem modelo padrão em DEFAULT_MODELS. "
                f"Confira o nome do provider ou informe 'model'. "
                f"Providers com modelo padrão: {', '.join(DEFAULT_MODELS)}."
            )
            raise ValueError(msg)
        model = DEFAULT_MODELS[provider]

    # Criar config do LLM
    config = LLMConfig(
        model=model,
        provider=provider,
        api_key=api_key,
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        rate_limit_delay=rate_limit_delay,
        model_kwargs=model_kwargs or {},
        search_config=search_config,
    )

    _validate_field_configs(
        questions,
        config.search_config,
        use_search=use_search,
        search_per_field=search_per_field,
    )

    # Só execuções com trabalho pendente validam dependências e rate limits.
    if use_search:
        validate_search_dependencies(search_provider)
        # Com um worker só, a taxa estimada fica em até 60 buscas por minuto,
        # abaixo do gatilho de todo provedor, e o aviso não teria o que dizer.
        if parallel_requests > 1:
            _warn_search_rate_limit(
                num_rows=len(df_pandas),
                num_fields=len(expected_columns),
                parallel_requests=parallel_requests,
                search_per_field=search_per_field,
                rate_limit_delay=rate_limit_delay,
                search_provider=search_provider,
            )
    validate_provider_dependencies(provider)

    # Entrar no backend conclui o preflight antes de qualquer mutação do DataFrame.
    with _provider_backend(config, questions, prompt, trace_mode) as backend:
        _setup_columns(
            df_pandas,
            expected_columns,
            status_column,
            track_tokens=track_tokens,
            search_config=search_config,
            trace_mode=trace_mode,
            pydantic_model=questions,
        )
        control_columns = [status_col, "_error_details"]
        _as_object_columns(df_pandas, expected_columns + control_columns, nan_columns)
        # O ausente que chega como pd.NA nas colunas de controle, como o da
        # coluna `string` de convert_dtypes, vira None: a marca das linhas não
        # alcançadas compara o status e o detalhe com texto, e a comparação com
        # o pd.NA não é booleana.
        with _silent_slice_write(df_pandas):
            for col in control_columns:
                df_pandas[col] = df_pandas[col].where(df_pandas[col].notna(), None)
        _apply_processed_values(df_pandas, processed_values)

        # Normalizar colunas complexas (listas, dicts, tuples) que podem ter sido
        # serializadas como strings JSON ao salvar/carregar de arquivos.
        if complex_fields and resume:
            normalize_complex_columns(df_pandas, complex_fields)

        # De novo depois da normalização, cujo apply volta a inferir float; sem
        # isso, gravar lista ou texto falharia depois da chamada paga. A coluna
        # de estrutura não guarda o NaN como valor, e `nan_columns` não muda nada.
        _as_object_columns(df_pandas, expected_columns)

        to_process, processed_count = _get_processing_indices(
            df_pandas, status_col, resume=resume, reprocess=bool(reprocess_columns)
        )
        _warn_missing_texts(
            df_pandas,
            text_column,
            [idx for idx, chosen in zip(df_pandas.index, to_process, strict=True) if chosen],
        )

        if parallel_requests > 1:
            token_stats = _process_rows_parallel(
                df_pandas,
                text_column=text_column,
                status_col=status_col,
                expected_columns=expected_columns,
                nan_columns=row_nan_columns,
                value_columns=value_columns,
                config=config,
                backend=backend,
                to_process=to_process,
                processed_count=processed_count,
                conversion_info=conversion_info,
                track_tokens=track_tokens,
                reprocess_columns=reprocess_columns,
                parallel_requests=parallel_requests,
                trace_mode=trace_mode,
                checkpoint=checkpoint,
                on_row_done=on_row_done,
            )
        else:
            token_stats = _process_rows(
                df_pandas,
                text_column=text_column,
                status_col=status_col,
                expected_columns=expected_columns,
                nan_columns=row_nan_columns,
                value_columns=value_columns,
                config=config,
                backend=backend,
                to_process=to_process,
                processed_count=processed_count,
                conversion_info=conversion_info,
                track_tokens=track_tokens,
                reprocess_columns=reprocess_columns,
                trace_mode=trace_mode,
                checkpoint=checkpoint,
                on_row_done=on_row_done,
            )

    if bool(token_stats.get("abort")):
        pending = int(df_pandas[status_col].isna().sum())
        kept = int(df_pandas["_error_details"].eq(_INTERRUPTED_REPROCESS_DETAIL).sum())
        summary = f"{pending} linha(s) ficaram sem status"
        if kept:
            summary += (
                f", e {kept} não foram reprocessadas e mantêm os valores anteriores "
                "(marcadas em _error_details); repetir a chamada com reprocess_columns "
                "refaz todas as linhas, inclusive as já reprocessadas"
            )
        # resume=True sobre a saída só retoma quando a saída traz a coluna de
        # texto, e só a saída de entrada DataFrame a traz; para list, dict e
        # Series a via de retomada é o checkpoint.
        output_has_text = conversion_info.original_type in (
            ORIGINAL_TYPE_PANDAS_DF,
            ORIGINAL_TYPE_POLARS_DF,
        )
        if checkpoint is not None:
            how = "rode de novo com o mesmo checkpoint_path"
            if output_has_text:
                how += ", ou com resume=True sobre esta saída"
        elif output_has_text:
            how = "rode de novo com resume=True sobre esta saída"
        else:
            how = (
                "a entrada não é um DataFrame e a saída não traz o texto, então só um "
                "checkpoint_path permite retomar"
            )
        warnings.warn(
            f"Execução interrompida: {token_stats['abort'].rstrip('.')}. {summary}. "
            f"Para retomar, {how}.",
            UserWarning,
            stacklevel=_user_stacklevel(),
        )

    if track_tokens and any(token_stats.values()):
        summary_text = _format_token_stats(
            token_stats,
            model,
            parallel_requests,
            search_provider=search_provider if use_search else None,
        )
        if summary_text:
            _stats_logger.info("%s", summary_text)

    # Independe de track_tokens: o número final de workers é a dica de configuração.
    if bool(token_stats.get("workers_reduced")):
        warnings.warn(
            f"Workers reduzidos por rate limit: de {token_stats['initial_workers']} para "
            f"{token_stats['final_workers']}. Considere usar "
            f"parallel_requests={token_stats['final_workers']} para evitar rate limits.",
            UserWarning,
            stacklevel=_user_stacklevel(),
        )

    # Retornar no formato original (remove colunas de status/erro se não houver erros)
    return from_pandas(df_pandas, conversion_info, status_col, annotations)


def _setup_columns(  # noqa: PLR0913 (uma família de colunas por recurso ligado)
    df: pd.DataFrame,
    expected_columns: list[str],
    status_column: str | None,
    *,
    track_tokens: bool,
    search_config: SearchConfig | None = None,
    trace_mode: str | None = None,
    pydantic_model: type[BaseModel] | None = None,
) -> None:
    """Configura colunas necessárias no DataFrame (in-place)."""
    status_col = status_column or "_dataframeit_status"
    error_col = "_error_details"
    token_cols = TOKEN_COLUMNS if track_tokens else ()
    search_cols = ["_search_credits"] if (search_config and search_config.enabled) else []

    # Colunas de trace
    trace_cols = []
    if bool(trace_mode):
        if search_config is not None and search_config.per_field and pydantic_model is not None:
            if bool(search_config.groups):
                # Com grupos: trace por grupo + trace por campo isolado
                grouped_fields = set()
                for group_config in search_config.groups.values():
                    grouped_fields.update(group_config.fields)

                # Adicionar colunas de trace para grupos
                trace_cols.extend(f"_trace_{group_name}" for group_name in search_config.groups)

                # Adicionar colunas de trace para campos isolados (não em grupos)
                trace_cols.extend(
                    f"_trace_{field}"
                    for field in pydantic_model.model_fields
                    if field not in grouped_fields
                )
            else:
                # Sem grupos: uma coluna por campo
                trace_cols = [f"_trace_{field}" for field in pydantic_model.model_fields]
        else:
            # Coluna única
            trace_cols = ["_trace"]

    columns = (*expected_columns, status_col, error_col, *token_cols, *search_cols, *trace_cols)
    # No pandas 3, o contexto não lê o DataFrame, e trocá-lo por None, como faz o
    # teste de mutação, não muda nada; no pandas 2, a troca levanta.
    with _silent_slice_write(df):  # pragma: no mutate
        for col in columns:
            if col not in df.columns:
                df[col] = None


def _get_processing_indices(
    df: pd.DataFrame, status_col: str, *, resume: bool, reprocess: bool = False
) -> tuple[list[bool], int]:
    """Retorna (linhas a processar por posição, contagem de linhas com status).

    A seleção depende só do status de cada linha, e nunca da ordem dos rótulos do
    índice, que pode vir fora de ordem (sort_values, sample, filtro), ser textual ou
    vir das chaves de um dict.

    Com resume=True, só a linha sem status fica pendente: linhas 'processed' e
    'error' são preservadas, e o usuário limpa o status de um erro para reprocessá-lo.
    Com resume=False, fica pendente toda linha que não esteja 'processed'.
    Com reprocess=True (reprocess_columns definido), toda linha é processada,
    e a contagem é a mesma.
    """
    status = df[status_col]
    if resume:
        is_pending = status.isna()
        processed_count = int((~is_pending).sum())
    else:
        is_pending = status.ne("processed")
        processed_count = 0
    if reprocess:
        return [True] * len(df), processed_count
    return is_pending.tolist(), processed_count


def _format_token_stats(
    token_stats: TokenStats,
    model: str | None,
    parallel_requests: int = 1,
    search_provider: str | None = None,
) -> str:
    """Monta o resumo de uso de tokens e throughput, vazio quando não há uso.

    Args:
        token_stats: Dict com contadores de tokens e métricas de tempo.
        model: Nome do modelo usado; None quando o runtime do provider escolhe.
        parallel_requests: Número de workers paralelos usados.
        search_provider: Provedor de busca usado, que dá nome à seção de busca.
    """
    if token_stats["total_tokens"] == 0 and not token_stats["cost_usd"]:
        return ""

    lines = [
        "=" * 60,
        "ESTATISTICAS DE USO",
        "=" * 60,
        f"Modelo: {model or 'escolhido pelo runtime do provider'}",
        f"Total de tokens: {token_stats['total_tokens']:,}",
        f"  - Input:  {token_stats['input_tokens']:,} tokens",
    ]
    if token_stats["cached_input_tokens"] > 0:
        lines.append(f"    └─ Cache: {token_stats['cached_input_tokens']:,} (incluído no Input)")
    lines.append(f"  - Output: {token_stats['output_tokens']:,} tokens")
    if token_stats["reasoning_tokens"] > 0:
        lines.append(f"    └─ Reasoning: {token_stats['reasoning_tokens']:,} (incluído no Output)")
    # Só providers que informam o custo, como o claude_code, preenchem este total,
    # que inclui as tentativas re-tentadas e as linhas que falharam.
    if token_stats["cost_usd"] > 0:
        lines.append(f"Custo informado pelo provider: US$ {token_stats['cost_usd']:.4f}")

    # Métricas de throughput (se disponíveis)
    if "elapsed_seconds" in token_stats and token_stats["elapsed_seconds"] > 0:
        elapsed = token_stats["elapsed_seconds"]
        requests = token_stats.get("requests_completed", 0)

        lines += [
            "-" * 60,
            "METRICAS DE THROUGHPUT",
            "-" * 60,
            f"Tempo total: {elapsed:.1f}s",
            f"Workers paralelos: {parallel_requests}",
        ]
        if requests > 0:
            rpm = (requests / elapsed) * 60
            lines += [f"Requisicoes: {requests}", f"  - RPM (req/min): {rpm:.1f}"]

        tpm = (token_stats["total_tokens"] / elapsed) * 60
        lines.append(f"  - TPM (tokens/min): {tpm:,.0f}")

    # Métricas de busca (se houver)
    if token_stats["search_count"] > 0:
        lines += [
            "-" * 60,
            f"METRICAS DE BUSCA ({search_provider.upper() if bool(search_provider) else 'TAVILY'})",
            "-" * 60,
            f"Total de buscas: {token_stats['search_count']}",
            f"Creditos usados: {token_stats['search_credits']}",
        ]

    lines.append("=" * 60)
    return "\n".join(lines)


_SUPPORTED_CHECKPOINT_EXTS = (".csv", ".xlsx", ".parquet")

# Formatos que gravam o texto vazio e o None na mesma célula vazia. O parquet
# guarda os dois à parte, e a célula nula dele é o None que a execução gravou.
_EMPTY_TEXT_AS_ABSENT_EXTS = frozenset({".csv", ".xlsx"})

# Extensões que exigem dependência opcional para pandas serializar.
# Validamos antes do loop para falhar rápido — um ModuleNotFoundError no primeiro
# save (após N linhas de LLM) desperdiça horas de trabalho.
_CHECKPOINT_EXT_REQUIRES = {
    ".xlsx": ("openpyxl", "openpyxl"),
    ".parquet": ("pyarrow", "pyarrow"),
}


def _signature_path(path: Path) -> Path:
    return Path(f"{path}.dataframeit.json")


# Chaves de model_kwargs que não mudam a resposta da linha. O prazo do turno
# fica de fora porque o caso típico de mudá-lo é retomar uma execução que
# travou, e com ele na assinatura o checkpoint seria recusado.
_UNSIGNED_MODEL_KWARGS = frozenset({"timeout"})


def _run_signature(  # noqa: PLR0913 (cada parâmetro muda o que a execução responde)
    questions: type[BaseModel],
    prompt: str,
    *,
    provider: str,
    model: str | None,
    model_kwargs: dict[str, Any] | None,
    search_config: SearchConfig | None,
    text_column: str,
    status_col: str,
) -> str:
    """Hash do que decide a resposta de cada linha: modelo, prompt, schema e busca.

    Um checkpoint gravado com outra assinatura não é retomado, porque juntar as
    linhas dele às desta execução misturaria respostas a perguntas diferentes.
    """
    schema = _schema_signature(questions)
    payload = json.dumps(
        {
            "prompt": prompt,
            "provider": provider,
            "model": model,
            "model_kwargs": _stable(
                {
                    key: value
                    for key, value in (model_kwargs or {}).items()
                    if key not in _UNSIGNED_MODEL_KWARGS
                }
            ),
            "schema": schema,
            "search": repr(search_config),
            "text_column": text_column,
            "status_col": status_col,
        },
        sort_keys=True,
        default=_stable_repr,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _schema_signature(questions: type[BaseModel]) -> object:
    """JSON Schema do modelo ou, quando ele não serve de assinatura, a forma canônica.

    Além do que `_has_unstable_extra` acha, a forma canônica entra quando o JSON
    Schema não serializa: a chave de dict que o JSON não aceita, como a tupla, o
    Pydantic deixa crua no schema do model_config (no do campo, ele a converte
    em texto).
    """
    if not _has_unstable_extra(questions):
        try:
            schema = questions.model_json_schema()
        except Exception:  # noqa: BLE001 (sem JSON Schema, a forma canônica, que desce nos aninhados, serve de assinatura)
            return _schema_form(questions)
        try:
            json.dumps(schema, sort_keys=True, default=_stable_repr)
        except TypeError:
            pass
        else:
            return schema
    return _schema_form(questions)


def _has_unstable_extra(questions: type[BaseModel]) -> bool:
    """Se o json_schema_extra de um campo ou de um model_config tem o que não é JSON.

    Contam os campos e os modelos, inclusive aninhados. Com função, conjunto ou
    contêiner que contém a si mesmo, o JSON Schema não serve de assinatura: a
    função faz model_json_schema() falhar; o conjunto vira lista na ordem de
    iteração, que muda com a semente de hash do processo, também quando o
    Pydantic o deixa cru no schema, como faz com o do model_config; e o ciclo
    derruba o json.dumps. A tabela e o array do pandas e do numpy entram no JSON
    Schema pelo repr, que mostra só as primeiras e as últimas posições. A busca
    desce em toda coleção de `_is_structure_class`, como o deque e a UserList.
    Os outros modelos ficam com o JSON Schema, e a assinatura deles é a mesma do
    checkpoint já gravado com eles.

    Cada modelo é reconstruído antes da leitura, como o Pydantic faz na
    validação: o campo guarda a referência adiantada até lá, e nem esta busca
    nem a forma canônica chegariam ao modelo que ela nomeia. O modelo completo
    não muda.
    """
    models = [questions]
    for model in models:  # a lista cresce com os modelos aninhados achados
        model.model_rebuild(raise_errors=False)
        for field_info in model.model_fields.values():
            for nested in get_nested_pydantic_models(field_info.annotation):
                if nested not in models:
                    models.append(nested)
    extras = itertools.chain(
        (model.model_config.get("json_schema_extra") for model in models),
        (field.json_schema_extra for model in models for field in model.model_fields.values()),
    )
    return any(isinstance(extra, dict) and _outside_json(extra) for extra in extras)


def _outside_json(value: object, path: frozenset[int] = frozenset()) -> bool:
    # O chamável vem antes do contêiner: o NamedTuple ou a subclasse de dict com
    # `__call__` é lido pelo código, e ler só os itens deixaria o modelo no JSON
    # Schema, que não leva o código; a iteração ainda passaria pelo `__iter__`
    # da classe do usuário.
    if callable(value) or isinstance(value, _TABLES):
        return True
    # O Pydantic escreve a dataclass e o modelo pelos campos, com o conjunto de
    # um campo na ordem de iteração.
    if _is_record(value):
        return _holds_unordered_set(value)
    if not _is_structure_class(type(value)):
        return False
    if isinstance(value, collections.abc.Set):
        # O hash do inteiro é o próprio valor: o conjunto só de inteiros itera na
        # mesma ordem em todo processo, e o JSON Schema dele é estável.
        return not all(isinstance(item, int) for item in value)
    # `path` traz os contêineres abertos acima deste: o que reaparece nele
    # contém a si mesmo, o que o JSON não representa.
    if id(value) in path:
        return True
    if isinstance(value, collections.abc.Mapping):
        # A chave vai ao JSON Schema do campo pelo `str`, que escreve o conjunto
        # dentro dela na ordem de iteração.
        items: Iterable[object] = [*value.keys(), *value.values()]
    else:  # a Sequence, o que sobra de `_is_structure_class`; o ty não estreita por ele
        # O cast não age em tempo de execução, e o teste de mutação troca o tipo à toa.
        items = cast("Sequence[object]", value)  # pragma: no mutate
    inner = path | {id(value)}
    return any(_outside_json(item, inner) for item in items)


# Atributos do campo na forma canônica: os que o JSON Schema levaria ao LLM e a
# configuração da biblioteca em json_schema_extra.
_SIGNED_FIELD_ATTRIBUTES = (
    "annotation",
    "default",
    "default_factory",
    "alias",
    "validation_alias",
    "title",
    "description",
    "examples",
    "deprecated",
    "discriminator",
    "json_schema_extra",
    "metadata",
)


def _schema_form(questions: type[BaseModel]) -> object:
    """Forma canônica do modelo, para o lugar do JSON Schema na assinatura.

    Cada campo, inclusive aninhado, entra pelo caminho; o modelo aninhado, o Enum
    e o chamável entram pelo que `_schema_leaf` tira deles. O caminho que se
    repete, como o campo de mesmo nome em dois ramos de `A | B`, entra com a
    lista dos campos, na ordem da união.
    """
    by_path: dict[str, list[dict[str, object]]] = {}
    for path, field_info, _ in _walk_fields(questions):
        by_path.setdefault(path, []).append(
            {attribute: getattr(field_info, attribute) for attribute in _SIGNED_FIELD_ATTRIBUTES}
        )
    fields = {path: found[0] if len(found) == 1 else found for path, found in by_path.items()}
    return _stable({"model": questions, "fields": fields}, _Descent())


class _Descent:
    """O que a descida no estado dos objetos chamáveis guarda durante uma assinatura.

    `depth` dá, pelo `id`, a profundidade de cada objeto do caminho atual, de
    cujo estado o valor faz parte. `lowest` traz, para cada objeto aberto, a
    menor profundidade a que um marcador de ciclo da subárvore dele apontou.
    `forms` guarda a forma do objeto cuja subárvore não apontou para ele nem para
    cima dele, ao lado do próprio objeto, para que o `id` não passe a outro
    objeto enquanto a assinatura dura. `containers` traz o `id` dos contêineres
    abertos desde o objeto aberto mais fundo (ver `_stable`). `aliases` traz os
    apelidos `type` abertos no caminho atual (ver `_named_type_leaf`).

    `leaves` diz se a folha passa por `_schema_leaf`. Na forma canônica, sim;
    no `_stable` chamado sem descida, como o dos `model_kwargs`, a folha fica
    como veio, e a descida serve só à guarda de ciclo dos contêineres.
    """

    # Declarados na classe, como em `_SnapshotWriter`: sem isso, o typecheck do
    # teste de mutação infere o tipo de `leaves` também pelo `self.leaves = None`
    # do `__init__` mutado, e acusa o `if descent.leaves` de `_stable`, que não
    # foi mutado.
    leaves: bool
    depth: dict[int, int]
    lowest: list[int]
    forms: dict[int, tuple[object, object]]
    containers: set[int]
    aliases: list[object]

    def __init__(self, *, leaves: bool = True) -> None:
        self.leaves = leaves
        self.depth = {}
        self.lowest = []
        self.forms = {}
        self.containers = set()
        self.aliases = []

    def reach(self, depth: int) -> None:
        """Registra, no objeto aberto mais fundo, um marcador que aponta para `depth`."""
        if self.lowest:
            self.lowest[-1] = min(self.lowest[-1], depth)


def _schema_leaf(value: object, descent: _Descent) -> object:
    """Folha da forma canônica: tipo pelo nome e conteúdo, chamável pelo código.

    O NewType, o apelido `type` e o TypeVar entram pelo tipo a que se referem
    (`_resolved`), como a anotação escrita direto com ele. O apelido recursivo
    volta a si mesmo, e ali entra por um marcador com o nome
    (`_named_type_leaf`). O que não é tipo vai a `_callable_leaf`.
    """
    resolved = _resolved(value)
    if resolved is not value:
        return _named_type_leaf(value, resolved, descent)
    origin = get_origin(value)
    if origin is Union:
        # `X | Y` e `Union[X, Y]` são a mesma união, e antes do 3.14 têm origens
        # diferentes: as duas grafias assinam pela de `X | Y`, que é uma classe e
        # entra pelo nome. A de `Union` entraria pelo código de uma função
        # interna do `typing`, que muda com a versão do Python.
        origin = UnionType
    if origin is not None:
        return {"origin": _schema_leaf(origin, descent), "args": _stable(get_args(value), descent)}
    if isinstance(value, type) and issubclass(value, BaseModel):
        # Os campos dele já entram pelo caminho; aqui, o que o JSON Schema tiraria
        # da classe: o título, a descrição e o model_config.
        return {
            "model": value.__name__,
            "doc": value.__doc__,
            "config": _stable(value.model_config, descent),
        }
    if isinstance(value, type) and issubclass(value, Enum):
        return {
            "enum": value.__name__,
            "members": _stable({member.name: member.value for member in value}, descent),
        }
    if isinstance(value, type):
        return value.__name__
    return _callable_leaf(value, descent)


def _named_type_leaf(named: object, resolved: object, descent: _Descent) -> object:
    """A forma de `resolved`, o tipo a que `named` se refere, ou o marcador de `named` já aberto."""
    if named in descent.aliases:
        return {"cycle": _stable_repr(named)}
    descent.aliases.append(named)
    form = _stable(resolved, descent)
    descent.aliases.pop()
    return form


def _callable_leaf(value: object, descent: _Descent) -> object:
    """Chamável pelo código e pelo que ele fixa; o que não é chamável, como veio.

    A função entra pelo nome qualificado e pelo código; o `partial`, pela função
    que embrulha e pelos argumentos fixados; o objeto chamável, pelo `__call__`
    da classe, como a função, e pelos atributos da instância. O método ligado
    entra pela função e pelo dono: a classe, pelo nome; a instância, pelos
    atributos. O objeto `staticmethod` e o `classmethod`, pela função que
    embrulham. O resto, inclusive o chamável cujo `__call__` não é código Python,
    como a função embutida, fica para o `default` do json.dumps em
    `_run_signature`, que tira o endereço do repr.
    """
    # Antes do ramo da função: o método ligado repassa o `__code__` da função, e
    # por ele entraria sem o dono, que guarda o que a condição compara.
    if isinstance(value, MethodType):
        owner = value.__self__
        return {
            "method": _callable_leaf(value.__func__, descent),
            "self": owner.__name__ if isinstance(owner, type) else _instance_state(owner, descent),
        }
    if isinstance(value, (staticmethod, classmethod)):
        return _callable_leaf(value.__func__, descent)
    if isinstance(value, partial):
        return {
            "partial": _schema_leaf(value.func, descent),
            "args": _stable(value.args, descent),
            "keywords": _stable(value.keywords, descent),
        }
    code = _attribute_or_none(value, "__code__")
    if isinstance(code, CodeType):
        return {"function": _attribute_or_none(value, "__qualname__"), "code": _code_form(code)}
    call = _python_call(value)
    if call is not None:
        return {
            "callable": _callable_leaf(call, descent),
            "state": _instance_state(value, descent),
        }
    return _record_form(value, descent)


def _is_record(value: object) -> bool:
    """Se o valor é instância de dataclass ou de modelo Pydantic, que o Pydantic escreve pelos campos."""
    return isinstance(value, BaseModel) or (
        dataclasses.is_dataclass(value) and not isinstance(value, type)
    )


def _record_fields(value: object) -> list[object]:
    """Os valores dos campos da dataclass ou do modelo Pydantic."""
    names = (
        type(value).model_fields
        if isinstance(value, BaseModel)
        else [field.name for field in dataclasses.fields(cast("Any", value))]
    )
    return [_attribute_or_none(value, name) for name in names]


def _record_form(value: object, descent: _Descent) -> object:
    """A dataclass ou o modelo cujo repr muda de um processo a outro, pelos atributos; o resto, como veio.

    O repr escreve o conjunto que um campo guarda na ordem de iteração, que muda
    com a semente de hash do processo (`_holds_unordered_set`). O registro sem
    esse conjunto fica pelo repr, como o checkpoint já gravado com ele.
    """
    if _is_record(value) and _holds_unordered_set(value):
        return {"record": type(value).__qualname__, "state": _instance_state(value, descent)}
    return value


def _holds_unordered_set(value: object, path: frozenset[int] = frozenset()) -> bool:
    """Se o valor tem um conjunto que não é só de inteiros, em qualquer nível.

    Desce pelas coleções de `_is_structure_class`, como a tupla, a lista e o
    deque, pelas chaves e pelos valores do Mapping e pelos campos do registro
    (`_is_record`). O hash do inteiro é o próprio valor, e o conjunto só de
    inteiros itera na mesma ordem em todo processo. `path` traz o `id` do que
    está aberto acima do valor: o que volta a ele não é percorrido de novo.
    """
    structure = _is_structure_class(type(value))
    if structure and isinstance(value, collections.abc.Set):
        return not all(isinstance(item, int) for item in value)
    if id(value) in path:
        return False
    if _is_record(value):
        items: Iterable[object] = _record_fields(value)
    elif structure and isinstance(value, collections.abc.Mapping):
        items = [*value.keys(), *value.values()]
    elif structure:
        # O cast não age em tempo de execução, e o teste de mutação troca o tipo à toa.
        items = cast("Iterable[object]", value)  # pragma: no mutate
    else:
        return False
    inner = path | {id(value)}
    return any(_holds_unordered_set(item, inner) for item in items)


def _python_call(value: object) -> object | None:
    """O `__call__` da classe do valor, quando escrito em Python; None no resto.

    O `__call__` da classe embutida é outro chamável sem código, e descer nele
    não terminaria.
    """
    call = type(value).__call__ if callable(value) else None
    return call if isinstance(getattr(call, "__code__", None), CodeType) else None


def _instance_state(value: object, descent: _Descent) -> object:
    """Atributos da instância, do `__dict__` e dos `__slots__`, pela folha da forma canônica.

    O atributo pode voltar ao próprio objeto, direto ou por outro objeto. O
    objeto que já está no caminho da descida entra por um marcador com o nome da
    classe, sem o endereço, que muda a cada processo. O marcador depende só do
    caminho, e não de tudo o que já foi visto: o mesmo objeto em dois atributos
    entra pelos dois, como entrariam dois objetos iguais.

    Por isso a forma do objeto que está num ciclo depende de por onde se chegou a
    ele, e a dele é refeita a cada caminho. A do objeto fora de ciclo não depende:
    nenhum objeto abaixo dele volta a ele, e nenhum objeto acima dele no caminho
    é alcançado de novo. Ela é calculada uma vez e reaproveitada, e o objeto
    alcançado por vários ramos não refaz a descida em cada um, o que dobraria o
    tempo a cada nível. O que separa um do outro é o marcador mais alto da
    subárvore: no objeto do ciclo, ele aponta para o próprio objeto ou para
    cima dele.
    """
    key = id(value)
    if key in descent.depth:
        descent.reach(descent.depth[key])
        return {"cycle": type(value).__name__}
    if key in descent.forms:
        return descent.forms[key][1]
    depth = len(descent.depth)
    descent.depth[key] = depth
    # O valor inicial só é comparado com profundidades, pelo `min` e pelo
    # `lowest > depth`, e qualquer valor acima de `depth` dá o mesmo resultado.
    descent.lowest.append(depth + 1)  # pragma: no mutate
    # O estado do objeto abre um trecho novo de contêineres (ver `_stable`).
    outer, descent.containers = descent.containers, set()
    # No objeto só com `__slots__`, o nome `__dict__` cai no `__getattr__` da
    # classe, que pode devolver qualquer coisa; só o dict é o `__dict__`.
    own = _attribute_or_none(value, "__dict__")
    state = {**(own if isinstance(own, dict) else {}), **_slot_values(value)}
    if (
        dataclasses.is_dataclass(value)
        and not isinstance(value, type)
        and _holds_unordered_set(value)
    ):
        # Um campo init=False pode guardar o default só na classe. Na forma
        # canônica, ele também precisa entrar pelo valor que a instância lê.
        for field in dataclasses.fields(cast("Any", value)):
            if field.name not in state:
                state[field.name] = _attribute_or_none(value, field.name)
    if isinstance(value, _PANDAS_OBJECTS):
        internals = _pandas_internals(type(value))
        state = {name: item for name, item in state.items() if name not in internals}
    form = _stable(state, descent)
    held = _held_value(value, descent)
    if held is not None:
        form = {"value": held, "state": form}
    descent.containers = outer
    del descent.depth[key]
    lowest = descent.lowest.pop()
    descent.reach(lowest)
    if lowest > depth:
        descent.forms[key] = (value, form)
    return form


# Tipos embutidos cujo valor a subclasse guarda fora do `__dict__` e dos
# `__slots__`, como o 5 de `Limite(5)`, a tupla de um NamedTuple ou os itens
# de uma subclasse de deque. Os primeiros saem pelo construtor da classe; os de
# data, cujo construtor não recebe a própria instância, pelo `__repr__` da
# classe, que leva também o fuso e o `fold`.
_BUILTIN_VALUE_TYPES = (
    int,
    float,
    complex,
    str,
    bytes,
    bytearray,
    tuple,
    list,
    dict,
    set,
    frozenset,
    collections.deque,
    Decimal,
)
_DATE_VALUE_TYPES = (dt.datetime, dt.date, dt.time, dt.timedelta)
_VALUE_TYPES = _BUILTIN_VALUE_TYPES + _DATE_VALUE_TYPES


def _held_value(value: object, descent: _Descent) -> object:
    """O que a instância guarda fora dos atributos, na forma canônica; None no resto.

    A tabela e o array do pandas e do numpy, inclusive o Index e o array do
    pandas, entram pelos dados inteiros (`_data_form`), que o pandas guarda no
    `_mgr` ou no array Arrow, cujo repr só os resume e mostra os blocos, e o
    numpy, no buffer; a subclasse de tipo embutido, pelo valor dele.
    """
    if isinstance(value, _TABLES):
        return _data_form(value, descent)
    # O `_stable` devolve o None como veio.
    return _stable(_builtin_value(value), descent)


def _builtin_value(value: object) -> object:
    """O valor de tipo embutido da instância de subclasse dele; None no resto.

    Sai pela classe embutida mais próxima na MRO: pelo construtor dela
    (`int(self)`, `tuple(self)`) ou, nos tipos de data, pelo `__repr__` dela,
    sem o endereço que o repr do fuso possa levar. O construtor passa pelo
    `__int__` ou pelo `__iter__` que a subclasse redefina, e a exceção dele
    deixa o valor de fora, como o atributo que não se lê (ver
    `_attribute_or_none`).
    """
    base = next((cls for cls in type(value).__mro__ if cls in _VALUE_TYPES), None)
    if base is None:
        return None
    try:
        if base in _DATE_VALUE_TYPES:
            return _without_address(vars(base)["__repr__"](value))
        return base(value)
    except Exception:  # noqa: BLE001 (ver a docstring)
        return None


def _attribute_or_none(value: object, name: str) -> object:
    """O atributo, ou None quando a busca dele levanta qualquer exceção.

    O nome que falta vai ao `__getattr__` da classe do usuário, que pode levantar
    outra coisa que não AttributeError, como o KeyError do objeto que guarda os
    atributos num dict. A assinatura não pode levantar: ela é calculada antes da
    primeira linha, e a exceção dela derrubaria a execução inteira.
    """
    try:
        return getattr(value, name)
    except Exception:  # noqa: BLE001 (ver a docstring)
        return None


def _slot_values(value: object) -> dict[str, object]:
    """Atributos atribuídos nos `__slots__` de cada classe da MRO.

    O slot privado (`__peso` na classe `Regra`) fica guardado com o nome da
    classe na frente (`_Regra__peso`), e é por esse nome que se lê. O slot
    `__dict__` já entra por `_instance_state`, e o `__weakref__` é a referência
    fraca de quem observa o objeto, não configuração dele.
    """
    values: dict[str, object] = {}
    for cls in type(value).__mro__:
        namespace = vars(cls)
        slots = namespace.get("__slots__", ())
        for name in (slots,) if isinstance(slots, str) else slots:
            if name in ("__dict__", "__weakref__"):
                continue
            private = name.startswith("__") and not name.endswith("__")
            attribute = f"_{cls.__name__.lstrip('_')}{name}" if private else name
            # Pelo descritor que a própria classe criou para o slot, e não pela
            # busca de atributo, que passaria pela property de mesmo nome de uma
            # subclasse e, no slot não atribuído, pelo `__getattr__`.
            try:
                values[attribute] = namespace[attribute].__get__(value, cls)
            except Exception:  # noqa: BLE001, S112
                # AttributeError é o slot ainda não atribuído, que não tem valor.
                # Outra exceção vem do descritor posto no lugar do slot, ou tirado
                # dele, depois de a classe ser criada; o slot também fica de fora,
                # porque a assinatura não pode levantar (ver `_attribute_or_none`).
                continue
    return values


def _code_form(code: CodeType) -> list[object]:
    """Bytecode, constantes e nomes lidos pela função, sem o arquivo nem a linha.

    Duas funções com o mesmo nome qualificado, como a lambda editada entre uma
    execução e outra, só assinam igual com o mesmo código. O que a função lê de
    fora dele (variável global ou do escopo de fora, valor padrão de argumento, o
    corpo de outra função que ela chama) não entra. O bytecode muda de uma versão
    menor do Python para outra (3.11, 3.12), e o checkpoint de um modelo com função
    não é retomado depois dessa troca. A constante frozenset, que o compilador gera
    para `x in {"a", "b"}`, é ordenada pelo `_stable`.
    """
    return [
        code.co_code.hex(),
        [
            _code_form(constant) if isinstance(constant, CodeType) else _stable(constant)
            for constant in code.co_consts
        ],
        list(code.co_names),
    ]


# O que o repr mostra só pelas primeiras e últimas posições: entra por todas.
_ARRAYS = (np.ndarray, pd.Index, ExtensionArray)
_TABLES = (pd.DataFrame, pd.Series, *_ARRAYS)

_PANDAS_OBJECTS = (pd.DataFrame, pd.Series, pd.Index, ExtensionArray)

# Atributos que o próprio pandas guarda na instância do DataFrame e da Series:
# os dados, que entram por `_data_form`, e o que muda com o uso ou com o arranjo
# dos blocos, como o `_item_cache` do pandas 2 e o repr do `_mgr`.
_TABLE_INTERNALS = frozenset(
    name for plain in (pd.DataFrame(), pd.Series(dtype=object)) for name in vars(plain)
)
_INDEX_INTERNALS = frozenset(vars(pd.Index([])))


def _pandas_internals(cls: type) -> frozenset[str]:
    """Atributos que o pandas guarda na instância da classe dele ou de uma subclasse.

    No DataFrame e na Series, os do DataFrame e da Series vazios. No Index e no
    array do pandas, os do Index vazio, entre eles o `_cache` que o
    `cache_readonly` preenche com o uso, e os que as classes do pandas na MRO
    declaram, como o `_pa_array` do array Arrow, cujo repr mostra o arranjo
    dos blocos.
    """
    if issubclass(cls, (pd.DataFrame, pd.Series)):
        return _TABLE_INTERNALS
    declared = (
        name
        for base in cls.__mro__
        if base.__module__.partition(".")[0] == "pandas"
        for name in inspect.get_annotations(base)
    )
    return _INDEX_INTERNALS | set(declared)


def _stable(value: object, descent: _Descent | None = None) -> object:
    """Forma que serializa igual em qualquer processo.

    As coleções entram pela ABC (`_is_structure_class`): a Sequence como lista,
    o Mapping como dict e o Set como lista ordenada, porque a ordem de iteração
    dele muda com a semente de hash do processo. Chaves de Mapping viram texto
    (ver `_key_form`), para ordenar dict com chaves int e str juntas. O DataFrame
    e a Series entram por `_table_form`; o ndarray, o Index e o array do pandas,
    por `_array_form`.
    Com `descent`, o que não é contêiner passa por `_schema_leaf`; sem ele, fica
    como está.

    O contêiner que já está aberto no caminho entra por um marcador com o nome
    da classe, sem o endereço. O caminho guardado é só o trecho desde o objeto
    aberto mais fundo, que `_instance_state` recomeça. Assim, o ciclo que passa
    por um objeto é marcado no objeto, pela regra de `_instance_state`, e o
    marcador do contêiner só aparece no ciclo feito só de contêineres (ou de
    contêineres e partials). Ele aponta sempre para dentro do estado do objeto
    mais fundo, e por isso não decide se a forma desse objeto é reaproveitada.
    """
    if descent is None:
        # `leaves` só é lido como condição, e o None no lugar do False daria o mesmo.
        descent = _Descent(leaves=False)  # pragma: no mutate
    # O chamável vem antes do contêiner: o NamedTuple com `__call__` entraria
    # pela lista dos valores, sem o código, e a subclasse de dict, só pelos itens.
    # `_callable_leaf` o lê pelo `__call__` e pelo valor embutido ao lado do estado.
    is_container = isinstance(value, _TABLES) or _is_structure_class(type(value))
    if descent.leaves and (not is_container or _python_call(value) is not None):
        return _schema_leaf(value, descent)
    if not is_container:
        return _record_form(value, descent)
    key = id(value)
    if key in descent.containers:
        return {"cycle": type(value).__name__}
    descent.containers.add(key)
    if isinstance(value, _TABLES):
        form: object = _data_form(value, descent)
    elif isinstance(value, collections.abc.Mapping):
        form = {_key_form(name, descent): _stable(item, descent) for name, item in value.items()}
    elif isinstance(value, collections.abc.Set):
        form = sorted((_stable(item, descent) for item in value), key=_stable_repr)
    else:  # a Sequence, o que sobra de `_is_structure_class`; o ty não estreita por ele
        # O cast não age em tempo de execução, e o teste de mutação troca o tipo à toa.
        items = cast("Sequence[object]", value)  # pragma: no mutate
        form = [_stable(item, descent) for item in items]
    descent.containers.remove(key)
    return form


def _key_form(name: object, descent: _Descent) -> str:
    """Chave de Mapping pelo `str`, sem o endereço de memória que ele leve.

    O conjunto, sozinho ou dentro de uma tupla, de uma dataclass ou de um
    modelo, entra pela lista ordenada, e não pela ordem de iteração, que muda
    com a semente de hash do processo (ver `_record_form`). A tupla sem conjunto
    entra pelo `str` dela.
    """
    if _key_needs_form(name):
        name = _stable(name, descent)
    return _without_address(str(name))


def _key_needs_form(name: object) -> bool:
    """Se a chave passa pela forma canônica: o conjunto, o registro com conjunto de texto, e a tupla que contém um deles.

    O registro sem esse conjunto (`_holds_unordered_set`) fica pelo `str`,
    também dentro da tupla, como no checkpoint já gravado com ele.
    """
    if isinstance(name, (set, frozenset)) or (_is_record(name) and _holds_unordered_set(name)):
        return True
    return isinstance(name, tuple) and any(_key_needs_form(item) for item in name)


def _data_form(
    value: pd.DataFrame | pd.Series | np.ndarray | pd.Index | ExtensionArray, descent: _Descent
) -> object:
    """DataFrame e Series por `_table_form`; ndarray, Index e array do pandas, por `_array_form`."""
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return _table_form(value, descent)
    return _array_form(value, descent)


def _table_form(value: pd.DataFrame | pd.Series, descent: _Descent) -> object:
    """DataFrame ou Series pelo conteúdo inteiro, pelos índices, pelos nomes e tipos e pelos `attrs`.

    O repr do pandas mostra só as primeiras e as últimas linhas, e a mudança
    numa célula do meio não apareceria nele.
    """
    frame = isinstance(value, pd.DataFrame)
    columns = list(value.items()) if frame else [(value.name, value)]
    return {
        "table": type(value).__name__,
        "index": _array_form(value.index, descent),
        "labels": _array_form(value.columns, descent) if frame else None,
        "attrs": _stable(value.attrs, descent),
        "columns": [
            [
                _stable(label, descent),
                _dtype_form(column.dtype, descent),
                _column_cells(column, descent),
            ]
            for label, column in columns
        ],
    }


def _array_form(value: np.ndarray | pd.Index | ExtensionArray, descent: _Descent) -> object:
    """ndarray, Index ou array do pandas pelo formato, pelo tipo, pelos nomes e por todas as células.

    O ndarray entra pelos bytes, a custo linear; o de objetos, que não estão nos
    bytes, e o Index e o array do pandas entram como coluna (ver
    `_column_cells`); e o de tipo composto ou de float e complex estendidos, que
    podem ter bytes de preenchimento sem valor definido, pelas células na forma
    canônica.
    """
    names = _stable(list(value.names), descent) if isinstance(value, pd.Index) else None
    if isinstance(value, pd.Index):
        cells = _column_cells(value.to_series(), descent)
    elif not isinstance(value, np.ndarray):
        cells = _column_cells(pd.Series(value), descent)
    elif bool(value.dtype.names) or _extended(value.dtype):
        cells = _stable(value.tolist(), descent)
    elif value.dtype.hasobject:
        cells = _column_cells(pd.Series(np.asarray(value).ravel()), descent)
    else:
        cells = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
    return {
        "shape": list(value.shape),
        "names": names,
        "dtype": _dtype_form(value.dtype, descent),
        "cells": cells,
    }


def _dtype_form(dtype: object, descent: _Descent) -> object:
    """Tipo pelo nome; o categórico, também pelas categorias, na ordem, e por `ordered`."""
    if isinstance(dtype, pd.CategoricalDtype):
        return {"category": _array_form(dtype.categories, descent), "ordered": dtype.ordered}
    return str(dtype)


# Os tipos que o `hash_pandas_object` lê pelos valores: bool, int, uint, timedelta,
# datetime, float e complex (o float e o complex estendidos ficam de fora, ver
# `_extended`).
_HASHED_KINDS = frozenset("biumMfc")
# O maior float e o maior complex que o `hash_pandas_object` lê, em bytes.
_FLOAT64_SIZE = np.dtype(np.float64).itemsize
_COMPLEX128_SIZE = np.dtype(np.complex128).itemsize


def _column_cells(column: pd.Series, descent: _Descent) -> object:
    """Células da coluna, na ordem, por um valor que não muda de um processo a outro.

    A coluna de número, data ou texto entra pelo `hash_pandas_object`, que lê
    toda célula, a custo linear, com chave fixa: o resultado é o mesmo em todo
    processo. A coluna com outros objetos entra pelas células na forma canônica,
    a mesma do resto da assinatura. O pandas não as lê: com lista ou conjunto na
    célula ele levanta, e com outro objeto ele o converte por `str()`, que
    escreve o conjunto na ordem de iteração e leva o endereço do objeto. O
    `to_json` também escreve o conjunto na ordem de iteração, e o repr é truncado.
    O `hash_pandas_object` levanta com o float e o complex estendidos, como o
    `longdouble`, e dá o mesmo valor a todo nulo, como `None` e `NaN` numa
    coluna de objetos; essas colunas também entram pelas células. A coluna
    categórica entra pelo código de cada célula, e as categorias, pelo tipo (ver
    `_dtype_form`).
    """
    if isinstance(column.dtype, pd.CategoricalDtype):
        column = column.cat.codes
    hashable = (column.dtype.kind in _HASHED_KINDS and not _extended(column.dtype)) or infer_dtype(
        column,
        skipna=False,
    ) == "string"
    if not hashable:
        return _stable(column.tolist(), descent)
    cells = hash_pandas_object(
        column,
        index=False,
    ).to_numpy()
    return hashlib.sha256(cells.tobytes()).hexdigest()


def _extended(dtype: np.dtype | ExtensionDtype) -> bool:
    """Se o tipo é float ou complex estendido, como o `longdouble`, maior que o de 64 bits.

    O tipo do pandas sem `itemsize`, como o esparso, não é estendido.
    """
    itemsize = getattr(dtype, "itemsize", None)
    if itemsize is None:
        return False
    return (dtype.kind == "f" and itemsize > _FLOAT64_SIZE) or (
        dtype.kind == "c" and itemsize > _COMPLEX128_SIZE
    )


def _stable_repr(value: object) -> str:
    """Repr sem endereço de memória, que muda a cada processo."""
    return _without_address(repr(value))


_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")


def _without_address(text: str) -> str:
    return _ADDRESS.sub("", text)


def _text_hashes(texts: pd.Series) -> list[str | None]:
    """Hash do texto de cada linha, tirado da memória e não do arquivo.

    CSV e XLSX não devolvem o texto como foi escrito ("001" volta 1, "NA" volta
    vazio, "=x" vira fórmula); comparar o arquivo com a entrada recusaria a
    própria entrada.
    """
    return [
        None if _is_missing_text(text) else hashlib.sha256(str(text).encode()).hexdigest()[:16]
        for text in texts
    ]


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_checkpoint_signature(path: Path, meta: dict[str, Any]) -> None:
    """Grava, ao lado do checkpoint, a assinatura, o hash dos textos e o do arquivo."""
    target = _signature_path(path)
    tmp = target.with_name(target.name + ".tmp")
    try:
        tmp.write_bytes(json.dumps({**meta, "checkpoint": _file_hash(path)}).encode())
        tmp.replace(target)
    except OSError as error:
        warnings.warn(
            f"Falha ao gravar a assinatura do checkpoint em {target}: {error}. "
            "O checkpoint é gravado, mas não será retomado automaticamente.",
            stacklevel=_user_stacklevel(),
        )


def _read_checkpoint_signature(path: Path) -> dict[str, Any] | None:
    try:
        meta = json.loads(_signature_path(path).read_bytes())
    except (OSError, ValueError):
        return None
    return meta if isinstance(meta, dict) else None


def _checkpoint_mismatch(
    df: pd.DataFrame,
    checkpoint: _Checkpoint,
    questions: type[BaseModel],
    *,
    status_col: str,
) -> tuple[str | None, pd.DataFrame | None]:
    """(motivo para não retomar, ou None; o checkpoint lido)."""
    path, run = checkpoint.path, checkpoint.meta or {}
    meta = _read_checkpoint_signature(path)
    if meta is None:
        return "não tem a assinatura da execução que o gravou", None
    if meta.get("checkpoint") != _file_hash(path):
        return "não é o arquivo que a assinatura ao lado descreve", None
    if meta.get("signature") != run.get("signature"):
        return (
            (
                "foi gravado com outra configuração (prompt, provider, modelo, model_kwargs, "
                "schema, busca ou colunas de texto e de status)"
            ),
            None,
        )
    if meta.get("texts") != run.get("texts"):
        return "foi gravado para outra entrada", None
    saved = read_df(path, questions)
    # O CSV nem sempre relê o que gravou (texto com retorno de carro isolado
    # volta partido em duas linhas em algumas versões do pandas).
    if status_col not in saved.columns or len(saved) != len(df):
        return "não relê com a coluna de status e as linhas da entrada", None
    return None, saved


def _holds_value(
    df: pd.DataFrame, column: str, values: Iterable[object], nan_columns: Collection[str]
) -> bool:
    """Se alguma das células `values` da coluna conta como valor, e não como ausente."""
    nan_is_value = _nan_is_value(df, column, nan_columns)
    return any(not _is_absent(value, nan_is_value=nan_is_value) for value in values)


def _nan_is_value(df: pd.DataFrame, column: str, nan_columns: Collection[str]) -> bool:
    """Se o float NaN da coluna conta como valor (`_is_absent`)."""
    return column in nan_columns and df[column].dtype == object


def _raise_if_overwritten(  # noqa: PLR0913 (as configurações da execução que a checagem lê)
    df: pd.DataFrame,
    questions: type[BaseModel],
    reprocessed_columns: frozenset[str],
    nan_columns: frozenset[str],
    *,
    text_column: str,
    status_col: str,
    pending: np.ndarray | None = None,
) -> None:
    """Recusa a entrada cujo valor num campo do modelo a resposta sobrescreveria.

    A linha nova, que é toda linha da entrada sem coluna de status ou a linha
    `pending`, sem status na entrada ou deixada pelo checkpoint retomado,
    recebe a resposta em todos os campos, e a coluna de entrada com o nome de
    um deles perderia o valor que tem nela; só a de reprocess_columns foi
    pedida, a coluna sem valor nas linhas novas não tem o que perder, e a do
    campo com exclude=True a execução não grava. A linha sem texto não vai ao
    LLM e fica com o valor que tem.
    """
    expected_columns = list(questions.model_fields)
    excluded_fields = _excluded_fields(questions)
    candidates = [
        col
        for col in expected_columns
        if col in df.columns and col not in reprocessed_columns and col not in excluded_fields
    ]
    if not candidates:
        return
    has_text = ~df[text_column].map(_is_missing_text).astype(bool).to_numpy()
    new_rows = has_text if pending is None else has_text & pending
    overwritten = [
        col for col in candidates if _holds_value(df, col, df[col][new_rows], nan_columns)
    ]
    if not overwritten:
        return
    # A sugestão soma as colunas já pedidas, para não tirá-las do reprocessamento.
    suggested = [
        col for col in expected_columns if col in reprocessed_columns or col in overwritten
    ]
    if pending is None:
        reason = f"e sem a coluna de status '{status_col}' a resposta do LLM as sobrescreveria"
        resume_hint = (
            " Se o DataFrame é uma saída do dataframeit, devolva a ele a coluna de status "
            f"com df['{status_col}'] = 'processed' no pandas ou "
            f"df = df.with_columns(pl.lit('processed').alias('{status_col}')) no polars: "
            "a execução o retoma, e reprocess_columns refaz só as colunas indicadas."
        )
    else:
        reason = (
            "e a resposta do LLM sobrescreveria o valor que elas têm nas linhas que a "
            "execução ainda não processou"
        )
        resume_hint = (
            " Com reprocess_columns, a execução refaz essas colunas também nas linhas já "
            "processadas."
        )
    msg = (
        f"As colunas {overwritten} já existem no DataFrame com o nome de campos do "
        f"modelo, {reason}. Renomeie ou remova essas colunas, ou passe "
        f"reprocess_columns={suggested!r} para sobrescrevê-las de propósito.{resume_hint}"
    )
    raise ValueError(msg)


def _resume_from_checkpoint(  # noqa: PLR0913 (a entrada, o checkpoint e a configuração)
    df: pd.DataFrame,
    checkpoint: _Checkpoint,
    questions: type[BaseModel],
    *,
    text_column: str,
    status_col: str,
    reprocessed_columns: frozenset[str],
    nan_columns: frozenset[str],
) -> frozenset[str]:
    """Traz para a entrada, in-place, o que um checkpoint desta execução já processou.

    O checkpoint é retomado só quando a assinatura gravada ao lado dele é a
    desta execução, o arquivo é o que a assinatura descreve e o texto de cada
    linha, casada por posição, é o da entrada. Fora disso, a execução avisa e
    começa do zero, e o arquivo é sobrescrito na primeira gravação. Antes de
    alterar a entrada, recusa o valor dela num campo do modelo nas linhas que
    o checkpoint deixou pendentes (`_raise_if_overwritten`).

    Das linhas já concluídas, com sucesso ou com erro, vêm as colunas que a
    execução acrescenta e os campos do modelo, sobre o que a entrada trazia
    neles, salvo o campo com exclude=True, que a execução não grava. Na linha
    com sucesso, é a resposta paga que o checkpoint gravou; na linha com erro,
    são as respostas anteriores que um reprocessamento guardou, ou o que a
    entrada daquela execução trazia.
    A entrada não tem
    como dizer se um valor ali é correção à mão ou resposta de outra execução,
    e quem quer manter a saída como está a passa com a coluna de status, que o
    checkpoint não relê. As demais colunas de entrada ficam como o usuário as
    passou, porque o CSV e o XLSX não preservam o tipo delas. As colunas se
    casam pelo nome em texto, que é como o CSV as devolve. Na linha ainda
    pendente, o campo do modelo fica com o que a entrada trazia, que fora de
    reprocess_columns é ausente, ou com o None com que a execução cria a
    coluna, que é de objetos.

    Devolve os campos cuja célula vazia, numa linha processada, é o texto
    vazio, para a validação da retomada (`_validate_processed_rows`); sem
    retomar, nenhum. O CSV e o XLSX gravam o texto vazio e o None na mesma
    célula vazia, que o leitor devolve ausente. O parquet guarda os dois à
    parte, e dele também não vem nenhum: a célula nula é o None gravado. A
    assinatura garante que o checkpoint foi gravado com este modelo, e nele o
    campo que só aceita texto, não admite None e cujo default não pode ser None
    (`_default_may_be_none`) só fica None quando a condição dele é falsa ou
    quando um validador troca uma resposta por None, que o tipo do campo
    recusa. O campo cuja fábrica lê os outros campos não é restaurado: a célula
    vazia dele conta como ausente, e a validação a preenche com o valor da
    fábrica, mesmo onde a execução gravou "". A validação da retomada lê a
    condição na linha, e do validador só pega o que troca o próprio texto
    vazio: o "" restaurado que ela não devolve como "" é acusado. O None que o
    validador dá a outra resposta, como "N/A", fica na mesma célula vazia, e a
    retomada do CSV e do XLSX o devolve como ""; o parquet guarda o None, e a
    retomada o acusa. Quem
    quer guardar esse None declara o campo Optional[str]. O campo com
    exclude=True fica de fora, porque o model_dump não o traz e a célula dele
    guarda o None que a execução pôs ao criar a coluna, e também o campo que o
    dump pode deixar de fora conforme a resposta (`_omittable_fields`), cuja
    célula vazia pode ser esse None. Sem essa garantia,
    como no checkpoint carregado à mão, a célula vazia continua ausente,
    porque com outro modelo ela pode ser o None de um campo que o admitia. No
    campo que admite None, ela volta None, e o texto vazio só sobrevive no
    parquet.
    """
    reason, saved = _checkpoint_mismatch(df, checkpoint, questions, status_col=status_col)
    # _checkpoint_mismatch só devolve o checkpoint lido quando não há motivo
    # para recusá-lo.
    if saved is None:
        warnings.warn(
            f"O checkpoint em {checkpoint.path} {reason}; esta execução começa do zero e o "
            "sobrescreve. Para retomá-lo mesmo assim, carregue-o com read_df e "
            "passe-o como entrada.",
            stacklevel=_user_stacklevel(),
        )
        return frozenset()

    concluded = saved[status_col].notna().to_numpy()
    _raise_if_overwritten(
        df,
        questions,
        reprocessed_columns,
        nan_columns,
        text_column=text_column,
        # Com `pending`, a checagem não lê o status_col, que só entra na mensagem
        # da entrada sem coluna de status; o teste de mutação o troca à toa.
        status_col=status_col,  # pragma: no mutate
        pending=~concluded,
    )
    by_name = {str(column): column for column in df.columns}
    model_fields = set(questions.model_fields)
    # A execução não grava o campo com exclude=True, e o checkpoint não tem o
    # que trazer dele: a coluna fica como na execução, com o None da criação
    # ou o que a entrada trazia. Relida do CSV ou do XLSX, ela viria NaN.
    excluded_fields = _excluded_fields(questions)
    for column in saved.columns:
        target = by_name.get(str(column))
        if target == text_column or str(column) in excluded_fields:
            continue
        if str(column) not in model_fields:
            if target is None:
                df[column] = saved[column].to_numpy()
            continue
        # A linha pendente fica com o que a entrada trazia no campo ou com o
        # None com que a execução cria a coluna, e que ela mantém na linha com
        # erro e quando o dump deixa o campo de fora; relida do CSV ou do XLSX,
        # ela viria NaN. A coluna é de objetos, como a criada pela execução,
        # porque o pandas 3 infere a coluna de texto como `str`, que troca o
        # None por NaN. Na coluna relida com outro dtype, como o float64 do
        # parquet, a célula nula é o None que a execução gravou, e nos objetos
        # o NaN contaria como valor. Na coluna de objetos, `read_df` já devolve
        # None na célula ausente, e o NaN que resta é o que o modelo devolveu.
        # O np.empty de objetos começa com None em toda célula.
        if target is None:
            values = np.empty(len(df), dtype=object)
            target = column
        else:
            values = df[target].to_numpy(dtype=object).copy()
        # Cópia: o array da coluna de objetos é o da própria coluna, só de leitura.
        saved_values = saved[column].to_numpy(dtype=object, copy=True)
        if saved[column].dtype != object:
            saved_values[saved[column].isna().to_numpy()] = None
        values[concluded] = saved_values[concluded]
        df[target] = pd.Series(
            values,
            index=df.index,
            dtype=object,
            # copy só decide se o pandas copia `values`, que é um array novo.
            copy=False,  # pragma: no mutate
        )
    if checkpoint.path.suffix.lower() not in _EMPTY_TEXT_AS_ABSENT_EXTS:
        return frozenset()
    omittable_fields = _omittable_fields(
        questions, excluded_fields, _dropped_by_model_serializer(questions, excluded_fields)
    )
    return frozenset(
        field_name
        for field_name, field in questions.model_fields.items()
        if accepts_only_text(field.annotation)
        and not _admits_none(field.annotation)
        and not _default_may_be_none(field)
        and field_name not in excluded_fields
        and field_name not in omittable_fields
    )


def _omittable_fields(
    model: type[BaseModel], excluded_fields: frozenset[str], serializer_dropped: frozenset[str]
) -> frozenset[str]:
    """Os campos que o model_dump pode deixar de fora conforme a resposta.

    O campo com exclude_if sai do dump quando a condição vale para o valor, e
    o @model_serializer do modelo tira os de `serializer_dropped`
    (`_dropped_by_model_serializer`). A execução não grava o campo que ficou
    de fora, e a célula dele guarda o None com que a coluna foi criada. O
    campo com exclude=True, que nunca sai no dump, fica em `excluded_fields`.
    """
    return serializer_dropped | frozenset(
        field_name
        for field_name, field in model.model_fields.items()
        if field.exclude_if is not None and field_name not in excluded_fields
    )


def _default_may_be_none(field: FieldInfo) -> bool:
    """Se o default do campo, dado direto ou pela fábrica, pode ser None.

    O pydantic não valida o default, e o None dele fica no campo que o tipo
    recusa. A fábrica que lê os outros campos só é chamada com eles, e pode
    dar None em alguma linha. Por isso o campo dela não é restaurado: na
    retomada do CSV e do XLSX, a célula vazia conta como ausente e a fábrica a
    preenche, mesmo onde a execução gravou "".

    A flag do pydantic vem antes da chamada porque essa fábrica, chamada sem
    os dados, levanta ValueError, e a resposta dela não pode depender de como
    a chamada falha. Qualquer outra fábrica que levanta aqui, como a que lê
    os.environ["VARIAVEL"] e levanta KeyError quando a variável falta, também
    levantaria na execução, e a execução só não a chamou porque o LLM
    respondeu o campo. Então a linha gravada não veio dela, a célula vazia é
    o "" da resposta, e a resposta é False. Com True, a célula vazia contaria
    como ausente, e a validação da linha chamaria a fábrica, que levantaria
    na retomada.
    """
    if field.default_factory_takes_validated_data is True:
        return True
    try:
        return field.get_default(call_default_factory=True) is None
    except Exception:  # noqa: BLE001 (a fábrica do usuário pode levantar qualquer erro; ver a docstring)
        return False


def _validate_checkpoint_extension(path: Path) -> None:
    """Valida extensão suportada e dependência opcional necessária."""
    ext = path.suffix.lower()
    if ext not in _SUPPORTED_CHECKPOINT_EXTS:
        msg = (
            f"Extensão {ext or '(nenhuma)'} não suportada para checkpoint. "
            f"Use uma de: {', '.join(_SUPPORTED_CHECKPOINT_EXTS)}"
        )
        raise ValueError(msg)
    requires = _CHECKPOINT_EXT_REQUIRES.get(ext)
    if requires is not None:
        module, pip_name = requires
        if importlib.util.find_spec(module) is None:
            msg = (
                f"Checkpoint {ext} requer o pacote '{module}', que não está instalado. "
                f"Execute: pip install {pip_name}"
            )
            raise ImportError(msg)


def _array_as_list(value: object) -> object:
    """Array do numpy como lista; qualquer outro valor como está."""
    tolist = getattr(value, "tolist", None)
    if callable(tolist) and not is_scalar(value):
        return tolist()
    return value


def _enum_as_value(value: object) -> object:
    return value.value if isinstance(value, Enum) else value


def _structure_as_json(value: object) -> object:
    value = _array_as_list(value)
    if _is_structure_class(type(value)):
        return json.dumps(
            value,
            ensure_ascii=False,
            default=_json_default,
        )
    return _enum_as_value(value)


def _cell_for_checkpoint(  # noqa: PLR0913 (uma opção por forma de gravar a célula)
    value: object,
    *,
    structures: bool,
    any_text: bool,
    text_file: bool,
    decimal_text: bool,
    nan_text: bool,
    scalar_text: bool,
) -> object:
    """A célula de uma coluna de objeto como o checkpoint a grava (`_structures_as_json`)."""
    if decimal_text and isinstance(value, Decimal):
        return str(value)
    if nan_text and isinstance(value, float) and math.isnan(value):
        return _FLOAT_NAN_TEXT
    if scalar_text and isinstance(value, str):
        return _quoted_scalar_text(value)
    # No campo que também é `Any`, o escape de `any_text` abaixo já põe o texto
    # "NaN" entre aspas, e o texto que se leria como estrutura precisa dele.
    if nan_text and not any_text and isinstance(value, str):
        return _quoted_nan_text(value)
    if (
        any_text
        and isinstance(value, str)
        and (_any_from_text(value) is not value or (text_file and _reads_as_other(value)))
    ):
        return json.dumps(value, ensure_ascii=False)
    return _structure_as_json(value) if structures else _enum_as_value(value)


def _structures_as_json(  # noqa: PLR0913, PLR0917 (uma família de colunas por forma de gravar)
    df: pd.DataFrame,
    columns: Collection[str] | None = None,
    any_columns: Collection[str] = (),
    nan_columns: Collection[str] = (),
    decimal_columns: Collection[str] = (),
    scalar_columns: Collection[str] = (),
    *,
    text_file: bool = False,
) -> pd.DataFrame:
    """Cópia com as coleções de `_is_structure_class` e os arrays do numpy serializados como JSON.

    CSV e XLSX gravariam o repr Python ("['a', 'b']"), que json.loads não lê
    de volta, e o de um array ("['a' 'b']") o literal_eval lê como ['ab']. O
    parquet gravaria uma coluna de dicts como struct, com as chaves de todas as
    linhas em cada uma. JSON é o que read_df e a retomada normalizam.

    O array chega quando a entrada é uma saída polars ou um parquet relido, que
    entregam o campo lista assim, inclusive dentro do dict de um modelo
    aninhado, e divide a coluna com as listas das linhas novas; serializar um
    sem o outro deixaria a coluna mista, que o parquet recusa.

    Com `columns`, só essas colunas são serializadas; sem, todas as de objeto.

    As de `any_columns`, as dos campos `Any` do modelo, são sempre
    serializadas, e nelas o texto que `read_df` leria como outro valor
    (`_any_from_text`), como "[1, 2]", vai como texto JSON entre aspas: o campo
    aceita o texto e a estrutura, e a célula precisa dizer qual dos dois o
    modelo devolveu. Com `text_file`, para o CSV e o XLSX, vai entre aspas
    também o texto que o leitor desses formatos devolveria como número, bool
    ou ausência (`_reads_as_other`), como "123", "NaN" e "".

    Com `text_file`, o Decimal das `decimal_columns`, as dos campos só de
    `Decimal`, vai como texto, que o XLSX gravaria como double, e o NaN das
    `nan_columns`, as dos campos que admitem `float` e dos campos `Any`, vai
    como o texto "NaN": o CSV e o XLSX o gravariam como a célula vazia da
    linha pendente. O texto "NaN" que o modelo devolveu num campo desses que
    também admite texto vai com um par de aspas (`_quoted_nan_text`). Nas
    `scalar_columns`, as da união de texto e número (`_mixes_text_and_number`),
    vai com um par de aspas todo texto que o leitor devolveria como número,
    bool ou ausência (`_quoted_scalar_text`). `read_df` com o modelo relê
    todos (`_raw_restorer`).

    Em qualquer formato, na coluna com um Decimal NaN todo Decimal vai como
    texto. O pandas conta o Decimal NaN como ausente; o
    decimal128 do parquet não tem NaN, e a coluna que juntasse Decimal e o
    texto "NaN" seria recusada. A validação da retomada devolve o texto ao
    tipo do campo.

    O Enum vai pelo valor em toda coluna de objeto, como dentro do JSON: o CSV
    e o XLSX gravariam o texto dele ("Cor.AZUL"), que a validação recusa, e o
    parquet recusa a coluna inteira.
    """
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            convert = partial(
                _cell_for_checkpoint,
                structures=columns is None or col in columns or col in any_columns,
                any_text=col in any_columns,
                text_file=text_file,
                decimal_text=(text_file and col in decimal_columns)
                or any(isinstance(value, Decimal) and value.is_nan() for value in out[col]),
                nan_text=text_file and col in nan_columns,
                scalar_text=text_file and col in scalar_columns,
            )
            converted = out[col].map(convert)
            # Series.map infere o tipo do resultado, e uma coluna com int ao
            # lado de None ou de float vira float64, que arredonda o int acima
            # de 2**53 antes de chegar ao arquivo. Sempre que o tipo inferido é
            # float, a coluna é refeita como object, que o CSV grava com o int
            # exato. O parquet recusa o int acima de 2**53 ao lado de float, e
            # a gravação falha com aviso em vez de arredondar. Nos demais casos
            # fica o tipo inferido.
            if converted.dtype.kind == "f":
                converted = pd.Series(
                    [convert(value) for value in out[col]], index=out.index, dtype=object
                )
            out[col] = converted
    return out


def _to_parquet_keeping_nan(df: pd.DataFrame, path: Path, nan_columns: Collection[str]) -> None:
    """Grava o parquet com o NaN das `nan_columns` à parte do nulo.

    O to_parquet do pandas grava como nulo o NaN de uma coluna de objetos, ao
    lado do None da linha pendente e da resposta None. A coluna de um campo
    que admite `float`, ou de um campo `Any`, só com número e None e com algum
    NaN (`_holds_only_number_with_nan`), é refeita na tabela do Arrow como
    float64, com o NaN como valor; `read_df` com o modelo o relê
    (`_reread_parquet`). Sem coluna assim, a gravação é a do pandas.
    """
    keeping = {
        col: values
        for col in df.columns
        if col in nan_columns
        and df[col].dtype == object
        # O isna, vetorizado, poupa a lista da coluna sem NaN nem None.
        and df[col].isna().any()
        and _holds_only_number_with_nan(values := df[col].tolist())
    }
    if not keeping:
        df.to_parquet(path, index=False)
        return
    import pyarrow as pa  # noqa: PLC0415 (dependência opcional, exigida pelo checkpoint parquet)
    import pyarrow.parquet as pq  # noqa: PLC0415

    table = pa.Table.from_pandas(df, preserve_index=False)
    for col, values in keeping.items():
        table = table.set_column(
            table.schema.get_field_index(col),
            col,
            # De uma lista de float, int e None, o Arrow infere o float64 e mantém o NaN.
            pa.array(values),
        )
    pq.write_table(table, path)


# O maior inteiro até o qual todo inteiro tem um float64 igual a ele.
_MAX_EXACT_FLOAT_INT = 2**53


def _holds_only_number_with_nan(values: list[object]) -> bool:
    """Se as células são só float, int que o float64 guarda exato e None, com algum NaN.

    O int ao lado de um float já sai float64 da gravação do pandas, e da
    coluna com nulo o leitor o devolve float. O bool e o int acima de 2**53
    não têm float64 igual a eles, e a coluna com um deles fica com a gravação
    do pandas, que grava o NaN como nulo.
    """
    return all(
        value is None
        or type(value) is float
        or (type(value) is int and abs(value) <= _MAX_EXACT_FLOAT_INT)
        for value in values
    ) and any(isinstance(value, float) and math.isnan(value) for value in values)


def _save_checkpoint(  # noqa: PLR0913, PLR0917 (o arquivo e o que a gravação sabe dos campos do modelo)
    df: pd.DataFrame,
    path: Path,
    meta: dict[str, Any] | None = None,
    structures: Collection[str] | None = None,
    any_fields: Collection[str] = (),
    nan_fields: Collection[str] = (),
    decimal_fields: Collection[str] = (),
    scalar_fields: Collection[str] = (),
) -> None:
    """Salva DataFrame em disco com escrita atômica. Formato inferido pela extensão.

    Com `meta`, a assinatura é gravada depois do checkpoint, com o hash do
    arquivo que acabou de ser escrito: se o processo morre entre as duas
    gravações, o hash não bate e a retomada recomeça em vez de aceitar o par.

    `structures` são os campos de estrutura do modelo, e `any_fields`, os
    campos `Any`, que recebem estrutura e texto. O parquet serializa só
    eles, que a retomada devolve aos tipos declarados, e grava as demais
    colunas como vieram, salvo o Enum, pelo valor; CSV e XLSX serializam toda
    coluna de objeto, porque gravariam o repr de qualquer estrutura.
    `nan_fields` são os campos cujo NaN é valor (`_keeps_nan`), gravado à
    parte da célula vazia, como o dos `any_fields`; `decimal_fields`, os
    campos só de `Decimal`, que o CSV e o XLSX gravam como texto; e
    `scalar_fields`, as uniões de texto e número, cujo texto o CSV e o XLSX
    gravam entre aspas quando o leitor o trocaria.
    """
    ext = path.suffix.lower()
    tmp = path.with_name(path.name + ".tmp")
    nan_fields = {*nan_fields, *any_fields}
    if ext in (".csv", ".xlsx"):
        frame = _structures_as_json(
            df, None, any_fields, nan_fields, decimal_fields, scalar_fields, text_file=True
        )
        if ext == ".csv":
            # O to_csv trata None como False em index, e o teste de mutação faz a troca à toa.
            frame.to_csv(
                tmp,
                index=False,  # pragma: no mutate
            )
        else:
            _to_excel_as_text(frame, tmp)
    elif ext == ".parquet":
        _to_parquet_keeping_nan(_structures_as_json(df, structures, any_fields), tmp, nan_fields)
    else:
        msg = (
            f"Extensão {ext} não suportada para checkpoint. "
            f"Use uma de: {', '.join(_SUPPORTED_CHECKPOINT_EXTS)}"
        )
        raise ValueError(msg)
    tmp.replace(path)
    if meta is not None:
        _write_checkpoint_signature(path, meta)


def _to_excel_as_text(df: pd.DataFrame, path: Path) -> None:
    """Grava o XLSX com o texto que começa com "=" como texto, e não como fórmula.

    O openpyxl toma por fórmula todo texto que começa com "=", e o leitor do
    pandas devolve vazia a célula de fórmula, que nunca foi calculada. O
    checkpoint não grava fórmula: toda célula que o openpyxl marcou assim é
    texto da execução ou da entrada, e volta a ser texto. O engine é fixado
    porque a marca é do openpyxl, que o checkpoint .xlsx já exige.
    """
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(
            writer,
            index=False,
        )
        for row in next(iter(writer.sheets.values())).iter_rows():
            for cell in row:
                if cell.data_type == "f":
                    cell.data_type = "s"


def _try_save_checkpoint(df: pd.DataFrame, checkpoint: _Checkpoint) -> bool:
    """Grava o checkpoint e devolve se deu certo; falha vira aviso.

    Uma falha de gravação (disco cheio, arquivo aberto no Excel, coluna que o
    parquet não serializa) não diz nada sobre a linha que acabou de ser
    processada, e por isso nunca muda o status dela nem interrompe a execução.
    A próxima gravação tenta de novo, com o estado completo.
    """
    try:
        _save_checkpoint(
            df,
            checkpoint.path,
            checkpoint.meta,
            checkpoint.structures,
            checkpoint.any_fields,
            checkpoint.nan_fields,
            checkpoint.decimal_fields,
            checkpoint.scalar_fields,
        )
    except Exception as error:  # noqa: BLE001 (ver a docstring)
        warnings.warn(
            f"Falha ao gravar o checkpoint em {checkpoint.path}: {type(error).__name__}: {error}. "
            "O processamento continua, e a próxima gravação tenta de novo.",
            stacklevel=_user_stacklevel(),
        )
        return False
    return True


@dataclass(frozen=True)
class _Checkpoint:
    """Arquivo do checkpoint e a cada quantas linhas processadas gravá-lo.

    batch_size e checkpoint_path só são aceitos juntos, e um único objeto
    opcional deixa o par inteiro presente ou ausente.
    """

    path: Path
    batch_size: int
    # Assinatura da execução e hash dos textos, gravados ao lado do checkpoint
    # a cada gravação (ver _write_checkpoint_signature).
    meta: dict[str, Any] | None = dataclasses.field(default=None, compare=False)
    # Campos de estrutura do modelo, que o parquet grava como JSON
    # (ver _save_checkpoint); sem eles, toda coluna de objeto.
    structures: frozenset[str] | None = dataclasses.field(default=None, compare=False)
    # Campos `Any` do modelo, cujo texto e cuja estrutura a gravação separa
    # (ver _structures_as_json).
    any_fields: frozenset[str] = dataclasses.field(default=frozenset(), compare=False)
    # Campos cujo NaN é valor (`_keeps_nan`), gravado à parte da célula vazia.
    nan_fields: frozenset[str] = dataclasses.field(default=frozenset(), compare=False)
    # Campos só de `Decimal`, que o CSV e o XLSX gravam como texto.
    decimal_fields: frozenset[str] = dataclasses.field(default=frozenset(), compare=False)
    # Uniões de texto e número, cujo texto o CSV e o XLSX gravam entre aspas
    # quando o leitor o trocaria (`_quoted_scalar_text`).
    scalar_fields: frozenset[str] = dataclasses.field(default=frozenset(), compare=False)


class _SnapshotWriter:
    """Grava os snapshots do modo paralelo sem regredir o checkpoint.

    As threads gravam fora da trava das linhas, e um snapshot mais antigo
    pode chegar depois de um mais novo. O rótulo cresce com as linhas
    contadas, e o snapshot de rótulo maior contém tudo o que os de rótulo
    menor contêm; o atrasado é descartado sem perda. A trava própria impede
    duas gravações simultâneas no mesmo arquivo temporário.
    """

    # Declarados na classe: sem isso, o typecheck do teste de mutação infere o
    # tipo de cada atributo também pelas atribuições dos `__init__` mutados,
    # como `self._lock = None`, e acusa o erro no código que não foi mutado.
    checkpoint: _Checkpoint
    last_saved: int
    _lock: threading.Lock

    def __init__(self, checkpoint: _Checkpoint) -> None:
        self.checkpoint = checkpoint
        self.last_saved = 0
        self._lock = threading.Lock()

    def save(self, frame: pd.DataFrame, label: int, *, force: bool = False) -> None:
        """Grava o snapshot, a menos que um de rótulo maior já tenha sido gravado.

        Com `force`, grava também com o rótulo da última gravação, para levar
        ao arquivo uma mudança que não conta linha, como as marcas da saída por
        exceção.
        """
        with self._lock:
            if label < self.last_saved or (label == self.last_saved and not force):
                return
            if _try_save_checkpoint(frame, self.checkpoint):
                self.last_saved = label


def _mark_unreached(  # noqa: PLR0913 (as linhas e o que a execução sabe das colunas delas)
    df: pd.DataFrame,
    registered: Container[Hashable],
    *,
    to_process: Sequence[bool],
    status_col: str,
    reprocess_columns: Sequence[str] | None,
    value_columns: Collection[str],
    nan_columns: Collection[str],
) -> bool:
    """Marca as linhas que uma interrupção impediu de processar e diz se alguma mudou.

    A linha não alcançada é a que a execução ia processar (`to_process`) e
    não registrou (`registered`). O abort e a exceção que sai do laço, como o
    Ctrl-C, marcam por esta mesma definição, no sequencial e no paralelo.

    A linha já processada que ia ser reprocessada guarda os valores anteriores,
    e o detalhe registra que eles não foram atualizados. Sem reprocess_columns,
    a linha 'processed' só pode ter sido gravada por esta execução, como a que
    o Ctrl-C alcança antes de ela entrar em `registered`, e fica como está. A
    linha 'error' com valor numa das `value_columns`, como a que guarda os
    valores de um reprocessamento que falhou, fica como estava: sem status, a
    retomada seguinte recusaria esses valores (`_raise_if_overwritten`). As
    demais ficam sem status, inclusive a que tinha 'error' e ia ser refeita,
    para que a retomada as encontre.
    """
    unreached = np.fromiter((idx not in registered for idx in df.index), bool)
    unreached &= np.asarray(to_process)
    status = df[status_col].to_numpy()
    details = df["_error_details"].to_numpy()
    processed = unreached & (status == "processed")
    flag = processed & (details != _INTERRUPTED_REPROCESS_DETAIL) & bool(reprocess_columns)
    clear = unreached & ~processed & ~(pd.isna(status) & pd.isna(details))
    # A linha 'error' sai da verificação na primeira coluna em que tem valor.
    errors = np.flatnonzero(clear & (status == "error"))
    for column in value_columns:
        nan_is_value = _nan_is_value(df, column, nan_columns)
        cells = df[column].to_numpy()[errors]
        holds = np.fromiter(
            (not _is_absent(cell, nan_is_value=nan_is_value) for cell in cells), bool
        )
        # As linhas de `errors` estão marcadas em `clear`; fica só a que não tem valor.
        clear[errors] = ~holds
        errors = errors[~holds]
    if flag.any():
        df.loc[flag, "_error_details"] = _INTERRUPTED_REPROCESS_DETAIL
    if clear.any():
        df.loc[clear, [status_col, "_error_details"]] = None
    return bool(flag.any() or clear.any())


def _get_cell(df: pd.DataFrame, idx: Hashable, column: str) -> object:
    """Lê uma célula por rótulo.

    `.at` lê também o rótulo-tupla da entrada dict, que `to_pandas` guarda num
    Index simples; `.loc` toma a tupla por chave de vários níveis e levanta
    AssertionError.
    """
    return df.at[idx, column]  # noqa: PD008 (ver a docstring)


def _set_cell(df: pd.DataFrame, idx: Hashable, column: str, value: object) -> None:
    """Grava uma célula por rótulo.

    `.at` grava lista ou dict como valor único da célula; `.loc` tentaria
    alinhar a lista com a seleção e levantaria.
    """
    df.at[idx, column] = value  # noqa: PD008 (ver a docstring)


def _record_missing_text(
    df: pd.DataFrame,
    idx: Hashable,
    *,
    status_col: str,
    row_already_processed: bool,
    reprocess_columns: Sequence[str] | None,
) -> None:
    """Marca como erro a linha sem texto, que não vai ao LLM."""
    _set_cell(df, idx, status_col, "error")
    _set_cell(
        df,
        idx,
        "_error_details",
        _missing_text_detail(
            row_already_processed=row_already_processed, reprocess_columns=reprocess_columns
        ),
    )


def _set_field_traces(df: pd.DataFrame, idx: Hashable, traces: dict[str, object]) -> None:
    """Grava o trace de cada campo ou grupo na coluna `_trace_{chave}`.

    O trace de campo aninhado e o de item de lista não têm coluna criada de
    antemão: a coluna nasce na primeira linha que o traz, porque o .at avisa
    ao criar coluna.
    """
    for key, trace in traces.items():
        column = f"_trace_{key}"
        if column not in df.columns:
            # No pandas 3, o contexto não lê o DataFrame, e trocá-lo por None, como faz o
            # teste de mutação, não muda nada; no pandas 2, a troca levanta.
            with _silent_slice_write(df):  # pragma: no mutate
                df[column] = None
        _set_cell(df, idx, column, _trace_text(trace))


def _trace_text(trace: object) -> str:
    """O rastro em JSON, com o texto não ASCII como está."""
    return json.dumps(
        trace,
        ensure_ascii=False,
    )


def _record_success(  # noqa: PLR0913 (grava cada família de colunas da linha)
    df: pd.DataFrame,
    idx: Hashable,
    result: dict[str, Any],
    *,
    status_col: str,
    expected_columns: list[str],
    config: LLMConfig,
    track_tokens: bool,
    row_already_processed: bool,
    reprocess_columns: Sequence[str] | None,
    trace_mode: str | None,
    token_stats: TokenStats,
) -> dict[str, Any]:
    """Grava na linha o resultado de uma chamada bem-sucedida e soma o uso.

    Numa linha já processada sob reprocess_columns, só essas colunas são
    gravadas; nas demais linhas, todas as colunas do modelo. Devolve os campos
    gravados, que o RowResult entrega ao callback.
    """
    extracted = result["data"]
    usage = result.get("usage")
    retry_info = result.get("_retry_info", {})
    only_reprocessed = reprocess_columns if row_already_processed else None

    written = {}
    for col in expected_columns:
        if col in extracted and (not bool(only_reprocessed) or col in only_reprocessed):
            _set_cell(df, idx, col, extracted[col])
            written[col] = extracted[col]

    # Armazenar tokens no DataFrame (se habilitado); o total vai só ao resumo.
    if track_tokens and usage:
        for column in TOKEN_COLUMNS:
            _set_cell(df, idx, column, usage.get(column.removeprefix("_"), 0))

    # Armazenar métricas de busca (search_count mantido só para o resumo)
    search_config = config.search_config
    if search_config and search_config.enabled and usage:
        _set_cell(df, idx, "_search_credits", usage.get("search_credits", 0))
    _add_usage_stats(token_stats, usage or {})

    # Armazenar traces: um por campo ou grupo nos modos por campo, ou um só
    if bool(trace_mode):
        if search_config and search_config.per_field:
            # Com trace_mode, a linha passou pelo agente campo a campo, que sempre devolve os traces.
            _set_field_traces(df, idx, result["traces"])
        else:
            # Com trace_mode, a linha passou por call_agent, que sempre devolve o trace.
            _set_cell(df, idx, "_trace", _trace_text(result["trace"]))

    _set_cell(df, idx, status_col, "processed")
    # Registra retries mesmo no sucesso; sem retry, some o erro de uma execução anterior
    _set_cell(df, idx, "_error_details", _success_details(retry_info))
    return written


def _add_usage_stats(token_stats: TokenStats, usage: Usage) -> None:
    """Soma ao resumo da execução o uso de uma linha."""
    for key in _USAGE_COUNTERS:
        token_stats[key] += usage.get(key, 0)
    token_stats["cost_usd"] += usage.get("cost_usd") or 0


def _add_failed_usage(token_stats: TokenStats, error: BaseException) -> None:
    """Soma ao resumo o que a linha que falhou já gastou.

    O custo vem em `cost_usd` (claude_code), e o uso das chamadas já feitas,
    em `row_usage` (LangChain sem busca e modo por campo), atributos que o
    backend põe na exceção.
    """
    failed = _take_failed_usage(error)
    if failed is not None:
        _add_usage_stats(token_stats, failed.usage)
        token_stats["cost_usd"] += failed.cost_usd
    else:
        _add_usage_stats(token_stats, getattr(error, "row_usage", None) or {})
        token_stats["cost_usd"] += getattr(error, "cost_usd", None) or 0


def _record_error(  # noqa: PLR0913 (o que a linha grava vem da execução inteira)
    df: pd.DataFrame,
    idx: Hashable,
    error: Exception,
    *,
    status_col: str,
    config: LLMConfig,
    row_already_processed: bool,
    reprocess_columns: Sequence[str] | None,
    token_stats: TokenStats,
) -> str:
    """Grava a falha da linha e devolve o aviso dela, com a mensagem amigável.

    Quem chama emite o aviso depois de contar a linha: tratado como erro, ele
    sai com a linha já registrada. O custo informado pelo provider e o uso das
    chamadas que a linha já fez (`row_usage`, do LangChain sem busca e do modo
    por campo) entram no resumo mesmo com a linha falhando.
    """
    # Antes da mensagem: se o texto da exceção levantar, o registro de reserva
    # assume a linha, e o uso já está no resumo.
    _add_failed_usage(token_stats, error)
    error_msg = f"{type(error).__name__}: {error}"

    # Erro recuperável chegou aqui depois de esgotar as tentativas; o não
    # recuperável, sem retry.
    if is_recoverable_error(error):
        error_details = f"[Falhou após {config.max_retries} tentativa(s)] {error_msg}"
    else:
        error_details = f"[Erro não-recuperável] {error_msg}"
    if row_already_processed and bool(reprocess_columns):
        error_details += _KEPT_VALUES_NOTE

    friendly_msg = get_friendly_error_message(error, config.provider)
    _set_cell(df, idx, status_col, "error")
    _set_cell(df, idx, "_error_details", error_details)
    return f"Falha ao processar linha {idx}.\n{friendly_msg}"


def _error_text(error: BaseException) -> str:
    """Tipo e texto da exceção, mesmo quando o `__str__` dela levanta.

    O repr padrão de uma exceção vem de `args` e não passa pelo `__str__`; se
    também ele levantar, fica só o tipo.
    """
    name = type(error).__name__
    try:
        text = str(error)
    except Exception:  # noqa: BLE001 (o texto de exceção alheia pode levantar qualquer coisa)
        try:
            text = repr(error)
        except Exception:  # noqa: BLE001 (idem)
            return name
    return f"{name}: {text}"


def _abort_reason(error: ProviderAbortError) -> str:
    """Motivo da interrupção para o aviso; na cota esgotada, com o que a resolve.

    O que vale só para uma origem da cota, como a dica da busca web, vem na
    mensagem do próprio erro.
    """
    reason = _error_text(error)
    if isinstance(error, ProviderUsageLimitError):
        reason = (
            f"{reason.rstrip('.')}. Repetir a linha não resolve: retome quando a cota "
            "voltar, com crédito, mudança de plano ou renovação do período na conta do serviço"
        )
    return reason


def _record_error_safely(  # noqa: PLR0913 (repassa os parâmetros de _record_error)
    df: pd.DataFrame,
    idx: Hashable,
    error: Exception,
    *,
    status_col: str,
    config: LLMConfig,
    row_already_processed: bool,
    reprocess_columns: Sequence[str] | None,
    token_stats: TokenStats,
) -> str:
    """Grava a falha da linha por _record_error e, se o registro falhar, pelo mínimo.

    Sem isto, uma falha ao montar a mensagem (exceção com `__str__` quebrado,
    por exemplo) deixaria a linha sem status, o que a apaga da saída, ou
    derrubaria a execução inteira no modo sequencial. O mínimo usa só o tipo
    e o texto da exceção original. Devolve o aviso da falha, que quem chama
    emite depois de contar a linha.
    """
    try:
        return _record_error(
            df,
            idx,
            error,
            status_col=status_col,
            config=config,
            row_already_processed=row_already_processed,
            reprocess_columns=reprocess_columns,
            token_stats=token_stats,
        )
    except Exception:  # noqa: BLE001 (ver a docstring)
        kept = _KEPT_VALUES_NOTE if row_already_processed and bool(reprocess_columns) else ""
        return _record_unrecorded_error(df, idx, error, status_col=status_col, note=kept)


def _record_unrecorded_error(
    df: pd.DataFrame, idx: Hashable, error: BaseException, *, status_col: str, note: str = ""
) -> str:
    """Marca como erro a linha cuja falha não pôde ser registrada normalmente.

    Devolve o aviso da falha, que quem chama emite depois de contar a linha.
    """
    error_msg = _error_text(error)
    _set_cell(df, idx, status_col, "error")
    _set_cell(df, idx, "_error_details", f"[Falha ao registrar o erro] {error_msg}{note}")
    return f"Falha ao processar linha {idx}."


def _progress_description(
    config: LLMConfig,
    backend: ProviderBackend,
    conversion_info: ConversionInfo,
    workers_label: str,
) -> str:
    """Rótulo da barra de progresso: entrada, backend, busca e workers."""
    type_labels = {
        ORIGINAL_TYPE_POLARS_DF: "polars→pandas",
        ORIGINAL_TYPE_PANDAS_DF: "pandas",
    }
    engine = type_labels.get(conversion_info.original_type, conversion_info.original_type)
    search_mode = "+search" if (config.search_config and config.search_config.enabled) else ""
    return f"Processando [{engine}+{backend.label}{search_mode}]{workers_label}"


def _progress_suffix(
    df: pd.DataFrame, reprocess_columns: Sequence[str] | None, processed_count: int
) -> str:
    """Final do rótulo: colunas reprocessadas ou ponto de retomada."""
    if bool(reprocess_columns):
        return f" (reprocessando: {', '.join(reprocess_columns)})"
    if processed_count > 0:
        return f" (resumindo de {processed_count}/{len(df)})"
    return ""


def _process_rows(  # noqa: C901, PLR0913, PLR0915 (estado da execução repassado por dataframeit, e um ramo por desfecho da linha)
    df: pd.DataFrame,
    *,
    text_column: str,
    status_col: str,
    expected_columns: list[str],
    nan_columns: Collection[str] = (),
    value_columns: Collection[str] = (),
    config: LLMConfig,
    backend: ProviderBackend,
    to_process: list[bool],
    processed_count: int,
    conversion_info: ConversionInfo,
    track_tokens: bool,
    reprocess_columns: Sequence[str] | None = None,
    trace_mode: str | None = None,
    checkpoint: _Checkpoint | None = None,
    on_row_done: Callable[[RowResult], None] | None = None,
) -> TokenStats:
    """Processa cada linha do DataFrame, em sequência.

    Args:
        df: DataFrame de trabalho, alterado in-place.
        text_column: Coluna com o texto de cada linha.
        status_col: Coluna de status ('processed' ou 'error').
        expected_columns: Campos do modelo, gravados em colunas de mesmo nome.
        nan_columns: Campos cujo NaN é valor, repassados a `_invoke_row`.
        value_columns: Campos que a execução grava, sem os de exclude=True.
        config: Configuração do LLM.
        backend: Implementação do provider vinculada à execução.
        to_process: Por posição, se a execução processa a linha.
        processed_count: Linhas com status antes desta execução.
        conversion_info: Metadados da conversão da entrada, usados no rótulo.
        track_tokens: Se True, grava tokens por linha e soma o uso.
        reprocess_columns: Colunas a reprocessar nas linhas já processadas.
        trace_mode: Modo de trace ("full", "minimal") ou None para desabilitar.
        checkpoint: Arquivo do checkpoint e intervalo de gravação, ou None.
        on_row_done: Chamado com o RowResult de cada linha depois de gravá-la.

    Returns:
        Dict com estatísticas de tokens: {'input_tokens', 'output_tokens', 'total_tokens'}
    """
    desc = _progress_description(config, backend, conversion_info, "")

    # Adicionar info de rate limiting (se ativo)
    if config.rate_limit_delay > 0:
        req_per_min = int(60 / config.rate_limit_delay)
        desc += f" [~{req_per_min} req/min]"
    desc += _progress_suffix(df, reprocess_columns, processed_count)

    token_stats = _empty_token_stats()
    rows_processed_this_run = 0
    rows_saved = 0
    # Linhas registradas nesta execução; o abort e a exceção que sai do laço
    # marcam as demais.
    registered: set[Hashable] = set()
    # Se as marcas da saída por exceção mudaram o DataFrame, e a gravação deve
    # levá-las. Só essa gravação o lê, depois de a marcação o regravar logo de
    # início, e o teste de mutação troca à toa o valor inicial.
    marked = False  # pragma: no mutate
    mark_unreached = partial(
        _mark_unreached,
        df,
        to_process=to_process,
        status_col=status_col,
        reprocess_columns=reprocess_columns,
        value_columns=value_columns,
        nan_columns=nan_columns,
    )

    def process_row(idx: Hashable, row: pd.Series) -> tuple[bool, dict[str, Any], str | None]:
        """Registra o desfecho da linha.

        Devolve se a chamada deu certo, os campos gravados e o aviso da falha.
        """
        # Verificar se linha já foi processada
        row_already_processed = row[status_col] == "processed"

        success = False
        values: dict[str, Any] = {}
        failure: str | None = None
        if _is_missing_text(row[text_column]):
            _record_missing_text(
                df,
                idx,
                status_col=status_col,
                row_already_processed=row_already_processed,
                reprocess_columns=reprocess_columns,
            )
        else:
            try:
                result = _invoke_row(
                    backend,
                    str(row[text_column]),
                    row,
                    reprocess_columns if row_already_processed else None,
                    expected_columns,
                    nan_columns=nan_columns,
                )
                values = _record_success(
                    df,
                    idx,
                    result,
                    status_col=status_col,
                    expected_columns=expected_columns,
                    config=config,
                    track_tokens=track_tokens,
                    row_already_processed=row_already_processed,
                    reprocess_columns=reprocess_columns,
                    trace_mode=trace_mode,
                    token_stats=token_stats,
                )
                success = True
            except ProviderAbortError as e:
                token_stats["abort"] = _abort_reason(e)
                _add_failed_usage(token_stats, e)
                mark_unreached(registered)
            except Exception as e:  # noqa: BLE001 (qualquer falha da linha vira status 'error')
                failure = _record_error_safely(
                    df,
                    idx,
                    e,
                    status_col=status_col,
                    config=config,
                    row_already_processed=row_already_processed,
                    reprocess_columns=reprocess_columns,
                    token_stats=token_stats,
                )
        return success, values, failure

    def mark_unreached_on_exception() -> None:
        """Marca a linha cuja chamada a exceção cortou e as seguintes."""
        nonlocal marked
        # Um Ctrl-C que corte as marcas ainda grava as que entraram.
        marked = True
        marked = mark_unreached(registered)

    def save_on_exception() -> None:
        """Grava as linhas registradas e as marcas que a última gravação não levou."""
        if checkpoint and (
            rows_processed_this_run > rows_saved or bool(token_stats.get("abort")) or marked
        ):
            _try_save_checkpoint(df, checkpoint)

    # Exceção que sai do laço, como o Ctrl-C ou um aviso tratado como erro:
    # grava as linhas registradas desde a última gravação, que a retomada
    # pagaria de novo, e relança a exceção. A linha cuja chamada foi cortada não
    # chegou a ser registrada; ela e as seguintes são marcadas como na
    # interrupção por ProviderAbortError, e as marcas também vão ao arquivo. O
    # Ctrl-C que corta a gravação final a refaz. A falha dessa gravação não
    # troca a exceção que sai; ver _on_exception.
    # A barra fecha na saída do bloco, depois das marcas e de toda gravação;
    # ver _ProgressBar. Sem o fechamento, ela ficaria aberta até o coletor de
    # lixo fechá-la, no meio de outra execução.
    with (
        _ProgressBar(total=len(df), desc=desc) as pbar,
        _on_exception(mark_unreached_on_exception, save_on_exception),
    ):
        for idx, row in itertools.compress(pbar.track(df.iterrows()), to_process):
            started = time.monotonic()
            success, values, failure = process_row(idx, row)
            if "abort" in token_stats:
                break
            registered.add(idx)

            # A linha conta antes do callback e do aviso da falha: a exceção
            # que sai deles, como um Ctrl-C ou o aviso tratado como erro, chega
            # com a linha já gravada no DataFrame, e a gravação da saída a leva.
            position = rows_processed_this_run
            rows_processed_this_run += 1
            _notify_row_done(
                on_row_done,
                RowResult(
                    index=idx,
                    position=position,
                    success=success,
                    error=_recorded_error(df, idx),
                    values=values,
                    elapsed_seconds=time.monotonic() - started,
                ),
            )
            if failure is not None:
                _warn_at_user(failure)
            if (
                checkpoint
                and rows_processed_this_run % checkpoint.batch_size == 0
                and _try_save_checkpoint(df, checkpoint)
            ):
                rows_saved = rows_processed_this_run

            if success and config.rate_limit_delay > 0:
                time.sleep(config.rate_limit_delay)

        # Save final: a cauda (< batch_size), o que uma gravação que falhou deixou de
        # fora e as marcas de uma interrupção, que não contam como linha processada.
        if checkpoint and (rows_processed_this_run > rows_saved or bool(token_stats.get("abort"))):
            _try_save_checkpoint(df, checkpoint)

    return token_stats


class _ProgressBar:
    """Barra de progresso do laço das linhas, que não interrompe a execução.

    A barra é só apresentação: a falha ao criá-la, avançá-la ou fechá-la, como
    o BrokenPipeError de um stderr que o leitor fechou, não sai daqui. As
    linhas seguem sendo processadas e gravadas, e o resultado ou a interrupção
    chegam a quem chamou. A primeira falha vira um aviso na saída do bloco,
    que os dois laços põem por fora das marcas e das gravações: nem o aviso
    tratado como erro as impede. Na saída por exceção, o aviso tratado como
    erro entra nela como nota.
    """

    def __init__(self, *, total: int, desc: str) -> None:
        self._failure: str | None = None
        self._bar: tqdm | None = None
        with self._tolerating_failure():
            self._bar = tqdm(total=total, desc=desc)

    @contextmanager
    def _tolerating_failure(self) -> Generator[None, None, None]:
        try:
            yield
        except Exception as e:  # noqa: BLE001 (nenhuma falha da barra interrompe a execução)
            if self._failure is None:
                self._failure = _error_text(e)

    def update(self) -> None:
        """Avança a barra em uma linha."""
        if self._bar is not None:
            with self._tolerating_failure():
                self._bar.update(1)

    def track(
        self, rows: Iterable[tuple[Hashable, pd.Series]]
    ) -> Generator[tuple[Hashable, pd.Series], None, None]:
        """Entrega as linhas e avança a barra quando o laço pede a seguinte."""
        for row in rows:
            yield row
            self.update()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, _exc_type: object, leaving: BaseException | None, _traceback: object
    ) -> None:
        if self._bar is not None:
            with self._tolerating_failure():
                self._bar.close()
        if self._failure is None:
            return
        # O aviso tratado como erro não troca a exceção que sai; ver _on_exception.
        try:
            _warn_at_user(
                f"Falha na barra de progresso, que não interrompeu a execução: {self._failure}"
            )
        except Exception as failure:
            if leaving is None:
                raise
            _note_on_exit(leaving, failure)


def _note_on_exit(leaving: BaseException, failure: Exception) -> None:
    """Põe na exceção que sai, como nota, a falha que ocorreu na saída por ela."""
    leaving.add_note(f"Na saída por esta exceção: {_error_text(failure)}")


@contextmanager
def _on_exception(
    mark: Callable[[], object], save: Callable[[], object]
) -> Generator[None, None, None]:
    """Chama `mark` e `save` quando uma exceção atravessa o bloco, e a relança.

    Vale para toda exceção que sai do laço das linhas: o Ctrl-C, o aviso
    tratado como erro (`-W error`) e o `SystemExit` de um callback. As linhas
    já processadas chegam ao checkpoint antes de a exceção sair de
    `dataframeit()`. `save` roda também quando um Ctrl-C corta `mark`.

    A falha de `save` não troca a exceção que sai. A de gravação já vira aviso
    em _try_save_checkpoint; com o aviso tratado como erro, ele sairia no lugar
    da exceção original, e por isso entra nela como nota. Um Ctrl-C durante
    `save` sai como chegou.
    """
    try:
        yield
    except BaseException as leaving:
        try:
            mark()
        finally:
            try:
                save()
            except Exception as failure:  # noqa: BLE001 (ver a docstring)
                _note_on_exit(leaving, failure)
        raise


@contextmanager
def _closing_in_slices(max_workers: int) -> Generator[ThreadPoolExecutor, None, None]:
    """Abre o executor da execução e, na saída, o encerra e espera as threads dele em fatias.

    Faz as vezes do `with ThreadPoolExecutor`, cuja saída espera as threads sem
    prazo: com prazo em cada espera, um segundo Ctrl-C a corta; ver o `wait` de
    _process_rows_parallel. O encerramento cancela as linhas que ainda estão na
    fila do executor, e nenhuma thread começa outra depois dele: a linha
    cancelada não é paga e fica sem status, para a retomada. Isso cobre o
    Ctrl-C que sai de dentro do `Thread.start()`, com a thread já lançada e
    ainda sem marcar o início; sem o cancelamento, ela começaria depois da
    gravação do checkpoint e pagaria fora dele a linha da fila. Sem exceção,
    a saída chega com a fila vazia, porque o laço espera todos os futuros em
    voo.

    A espera é pelas threads, e não pelos futuros em voo, porque o Ctrl-C pode
    sair de dentro do `submit` depois de uma thread tirar a linha da fila. A
    linha roda e é paga sem que o futuro dela chegue ao laço, e a gravação do
    checkpoint precisa encontrá-la registrada. A espera cobre também a linha
    cujo `on_row_done` ainda roda: o callback vem da thread do worker, que não
    recebe o KeyboardInterrupt, e a linha já está contada quando ele começa.

    As threads são achadas pelo nome, com o prefixo deste executor, em
    `threading.enumerate()`. O conjunto `_threads` do executor é privado e só
    recebe a thread depois de `Thread.start()` voltar; `enumerate` a lista
    desde antes. `is_alive` deixa de fora a thread que ainda não marcou o
    início, que `join` recusaria: ela nunca começa ou, ao começar, encontra a
    fila vazia e sai sem rodar linha. A lista se refaz a cada fatia e pega
    também a thread que começa durante a espera. O número no prefixo separa as
    threads de duas chamadas concorrentes, e o `_` depois dele impede que o
    executor 1 tome por suas as threads do 10.
    """
    prefix = f"dataframeit-paralelo{next(_executor_ids)}"
    executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=prefix)
    try:
        yield executor
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        while alive := [
            thread
            for thread in threading.enumerate()
            if thread.name.startswith(f"{prefix}_") and thread.is_alive()
        ]:
            alive[0].join(_WAIT_SLICE_SECONDS)


def _process_rows_parallel(  # noqa: C901, PLR0913, PLR0915 (estado compartilhado entre threads)
    df: pd.DataFrame,
    *,
    text_column: str,
    status_col: str,
    expected_columns: list[str],
    nan_columns: Collection[str] = (),
    value_columns: Collection[str] = (),
    config: LLMConfig,
    backend: ProviderBackend,
    to_process: list[bool],
    processed_count: int,
    conversion_info: ConversionInfo,
    track_tokens: bool,
    reprocess_columns: Sequence[str] | None,
    parallel_requests: int,
    trace_mode: str | None = None,
    checkpoint: _Checkpoint | None = None,
    on_row_done: Callable[[RowResult], None] | None = None,
) -> TokenStats:
    """Processa linhas do DataFrame em paralelo, com fila contínua e auto-redução.

    Até `parallel_requests` linhas ficam em voo ao mesmo tempo, e uma linha
    nova é despachada assim que outra termina.

    Args:
        df: DataFrame de trabalho, alterado in-place sob lock.
        text_column: Coluna com o texto de cada linha.
        status_col: Coluna de status ('processed' ou 'error').
        expected_columns: Campos do modelo, gravados em colunas de mesmo nome.
        nan_columns: Campos cujo NaN é valor, repassados a `_invoke_row`.
        value_columns: Campos que a execução grava, sem os de exclude=True.
        config: Configuração do LLM.
        backend: Implementação do provider vinculada à execução.
        to_process: Por posição, se a execução processa a linha.
        processed_count: Linhas com status antes desta execução.
        conversion_info: Metadados da conversão da entrada, usados no rótulo.
        track_tokens: Se True, grava tokens por linha e soma o uso.
        reprocess_columns: Colunas a reprocessar nas linhas já processadas.
        parallel_requests: Teto inicial de linhas em voo, que também dimensiona
            o executor. Reduzido pela metade a cada erro de rate limit (429).
        trace_mode: Modo de trace ("full", "minimal") ou None para desabilitar.
        checkpoint: Arquivo do checkpoint e intervalo de gravação, ou None.
        on_row_done: Chamado, na thread do worker, com o RowResult de cada
            linha depois de gravá-la.

    Returns:
        Dict com estatísticas de tokens e métricas de throughput.
    """
    start_time = time.time()

    # Estado compartilhado (thread-safe)
    lock = threading.Lock()
    current_workers = parallel_requests
    initial_workers = parallel_requests
    rate_limit_event = threading.Event()
    abort_event = threading.Event()
    # Linhas gravadas e contadas pelas threads; a coleta só marca como erro a
    # que escapou antes disso, e não a que falhou depois, ao gravar o snapshot.
    # O abort e a exceção que sai do laço marcam as demais.
    counted: set[Hashable] = set()
    # Avisos das linhas que falharam, na ordem em que as threads os guardam
    # sob `lock`, com a linha já registrada. A thread principal os emite quando
    # um futuro termina: tratado como erro, o aviso sai de `dataframeit()` como
    # no modo sequencial, e a saída espera o callback das linhas em andamento.
    row_warnings: list[str] = []
    # Se as marcas da saída por exceção mudaram o DataFrame, e a gravação deve
    # levá-las. Só essa gravação o lê, depois de a marcação o regravar logo de
    # início, e o teste de mutação troca à toa o valor inicial.
    marked = False  # pragma: no mutate
    # Se a parada guardada em token_stats["abort"] é a da cota esgotada. A
    # primeira parada o define antes de qualquer leitura, e o teste de mutação
    # troca à toa o valor inicial.
    quota_abort = False  # pragma: no mutate
    checkpoint_counter = 0
    # A gravação do checkpoint não segura `lock`, para que as threads que
    # processam linhas sigam durante a I/O. Cada snapshot é copiado sob `lock`
    # na mesma seção que incrementa checkpoint_counter e leva esse valor como
    # rótulo, que o _SnapshotWriter usa para descartar o snapshot atrasado.
    writer = _SnapshotWriter(checkpoint) if checkpoint else None
    mark_unreached = partial(
        _mark_unreached,
        df,
        to_process=to_process,
        status_col=status_col,
        reprocess_columns=reprocess_columns,
        value_columns=value_columns,
        nan_columns=nan_columns,
    )

    def _count_row(idx: Hashable) -> Callable[[], None] | None:
        """Conta a linha; na hora do checkpoint, copia o DataFrame. Chamar sob `lock`."""
        nonlocal checkpoint_counter
        counted.add(idx)
        checkpoint_counter += 1
        if writer and checkpoint_counter % writer.checkpoint.batch_size == 0:
            # Copia sob lock, serializa fora, para não bloquear threads na I/O.
            return partial(writer.save, df.copy(), checkpoint_counter)
        return None

    token_stats = _empty_token_stats()
    token_stats["requests_completed"] = 0

    desc = _progress_description(
        config, backend, conversion_info, f" [{parallel_requests} workers]"
    )
    desc += _progress_suffix(df, reprocess_columns, processed_count)

    # Linhas a processar, com a posição de cada uma entre as processadas nesta
    # chamada, que o RowResult entrega ao callback.
    rows_to_process = [
        (position, idx, row)
        for position, (idx, row) in enumerate(itertools.compress(df.iterrows(), to_process))
    ]

    if not rows_to_process:
        return token_stats

    warning_site = _user_warning_site()

    def _finish(  # noqa: PLR0913 (o que a linha gravou, repassado ao RowResult)
        row_data: _RowData,
        started: float,
        snapshot: Callable[[], None] | None,
        *,
        success: bool,
        error: str | None,
        values: dict[str, Any],
    ) -> None:
        """Entrega a linha gravada ao callback e grava o snapshot, fora de `lock`.

        O callback vem antes do snapshot, cuja I/O pode demorar, e um snapshot
        que falha não repete a entrega: a linha já está contada.
        """
        position, idx, _row = row_data
        _notify_row_done(
            on_row_done,
            RowResult(
                index=idx,
                position=position,
                success=success,
                error=error,
                values=values,
                elapsed_seconds=time.monotonic() - started,
            ),
        )
        if snapshot is not None:
            snapshot()

    def record_abort(error: ProviderAbortError) -> None:
        """Guarda o motivo da parada, soma o uso da linha e para o despacho.

        A cota esgotada passa à frente de outra parada que chegou antes dela,
        porque a orientação dela (`_abort_reason`) é o que resolve a retomada,
        e a outra parada vem depois dela no motivo.
        """
        nonlocal quota_abort
        with lock:
            quota = isinstance(error, ProviderUsageLimitError)
            reason = _abort_reason(error)
            if "abort" not in token_stats:
                token_stats["abort"] = reason
                quota_abort = quota
            elif quota and not quota_abort:
                token_stats["abort"] = f"{reason}. Antes, outra linha parou: {token_stats['abort']}"
                quota_abort = True
            _add_failed_usage(token_stats, error)
        abort_event.set()

    def process_single_row(row_data: _RowData) -> None:
        """Processa uma única linha (executada em thread separada)."""
        nonlocal current_workers

        started = time.monotonic()
        _position, idx, row = row_data
        row_already_processed = row[status_col] == "processed"
        if _is_missing_text(row[text_column]):
            with lock:
                _record_missing_text(
                    df,
                    idx,
                    status_col=status_col,
                    row_already_processed=row_already_processed,
                    reprocess_columns=reprocess_columns,
                )
                error = _recorded_error(df, idx)
                snapshot = _count_row(idx)
            _finish(row_data, started, snapshot, success=False, error=error, values={})
            return

        # Linha que ainda não foi ao provider quando a execução foi interrompida
        # é marcada no fim, com as que nem foram despachadas.
        if abort_event.is_set():
            return

        # Verificar se devemos pausar devido a rate limit
        if rate_limit_event.is_set():
            time.sleep(2.0)  # Pausa breve quando rate limit detectado

        try:
            result = _invoke_row(
                backend,
                str(row[text_column]),
                row,
                reprocess_columns if row_already_processed else None,
                expected_columns,
                nan_columns=nan_columns,
            )

            # Atualizar DataFrame (com lock para thread-safety)
            with lock:
                values = _record_success(
                    df,
                    idx,
                    result,
                    status_col=status_col,
                    expected_columns=expected_columns,
                    config=config,
                    track_tokens=track_tokens,
                    row_already_processed=row_already_processed,
                    reprocess_columns=reprocess_columns,
                    trace_mode=trace_mode,
                    token_stats=token_stats,
                )
                token_stats["requests_completed"] += 1
                error = _recorded_error(df, idx)
                snapshot = _count_row(idx)

        except ProviderAbortError as e:
            record_abort(e)
            return
        except Exception as e:  # noqa: BLE001 (qualquer falha da linha vira status 'error')
            # A classificação lê o texto da exceção; se ele não puder ser montado,
            # a linha segue para o registro, que tem o próprio recurso.
            try:
                rate_limited = is_rate_limit_error(e)
            except Exception:  # noqa: BLE001 (ver o comentário acima)
                # Só o `if` abaixo o lê, e o None que o teste de mutação põe no
                # lugar é também falso.
                rate_limited = False  # pragma: no mutate
            with lock:
                if rate_limited and current_workers > 1:
                    old_workers = current_workers
                    current_workers = max(1, current_workers // 2)
                    row_warnings.append(
                        f"Rate limit detectado! Reduzindo workers de {old_workers} para {current_workers}."
                    )
                    rate_limit_event.set()
                    # Limpar evento após um tempo
                    threading.Timer(5.0, rate_limit_event.clear).start()
                row_warnings.append(
                    _record_error_safely(
                        df,
                        idx,
                        e,
                        status_col=status_col,
                        config=config,
                        row_already_processed=row_already_processed,
                        reprocess_columns=reprocess_columns,
                        token_stats=token_stats,
                    )
                )
                error = _recorded_error(df, idx)
                snapshot = _count_row(idx)
            _finish(row_data, started, snapshot, success=False, error=error, values={})
            return

        _finish(row_data, started, snapshot, success=True, error=error, values=values)
        if config.rate_limit_delay > 0:
            time.sleep(config.rate_limit_delay)

    def process_with_warning_site(row_data: _RowData) -> None:
        """Leva o local da chamada pública à thread que processa a linha."""
        _row_warning_site.value = warning_site
        try:
            process_single_row(row_data)
        finally:
            del _row_warning_site.value

    def mark_uncounted_rows() -> None:
        """Marca, quando uma exceção sai do laço, as linhas que nenhuma thread registrou.

        Como na interrupção por ProviderAbortError, a linha já processada que
        ia ser reprocessada mantém os valores anteriores, com a marca em
        `_error_details`. A linha que uma thread deixada por um segundo Ctrl-C
        registrar depois troca a marca pelo valor novo.
        """
        nonlocal marked
        with lock:
            # Um Ctrl-C que corte as marcas ainda grava as que entraram.
            marked = True
            marked = mark_unreached(counted)

    def save_registered_rows() -> None:
        """Grava o checkpoint quando uma exceção sai do laço.

        O executor espera as linhas em andamento antes de chegar aqui, e elas
        chegam registradas. Um Ctrl-C durante essa espera deixa threads
        rodando: a cópia sob `lock` leva só as linhas que terminaram, e as
        demais ficam pendentes para a retomada. Pelo rótulo, o _SnapshotWriter
        não regrava o arquivo sem linha contada desde o último snapshot gravado,
        salvo quando mark_uncounted_rows mudou linhas, o que não conta linha.
        """
        if writer:
            with lock:
                frame, label = df.copy(), checkpoint_counter
            writer.save(frame, label, force=marked)

    # Um executor para a execução inteira. O teto de linhas em voo é
    # current_workers, lido a cada reposição: a redução por rate limit só deixa
    # de despachar, sem cancelar o que já está em voo, e vale até o fim. Na
    # saída por exceção, o executor fecha e espera as threads antes de
    # save_registered_rows gravar: os `with` saem na ordem inversa. A barra
    # fecha na saída do bloco de fora, depois das marcas e de toda gravação;
    # ver _ProgressBar.
    pending_rows = iter(rows_to_process)
    in_flight: dict[Future[None], tuple[_RowData, float]] = {}

    def collect(future: Future[None]) -> None:
        """Fecha a linha cujo futuro terminou e emite os avisos guardados até aqui."""
        row, started = in_flight.pop(future)
        _position, idx, _row = row
        try:
            future.result()
        except Exception as e:  # noqa: BLE001 (falha fora da linha não para a execução)
            # Escapou do tratamento da própria linha; sem isto ela ficaria sem
            # status e sumiria da saída.
            if idx in counted:
                with lock:
                    row_warnings.append(
                        f"Erro inesperado no executor depois de registrar a linha "
                        f"{idx}, que mantém o status gravado: {_error_text(e)}",
                    )
            else:
                with lock:
                    row_warnings.append(_record_unrecorded_error(df, idx, e, status_col=status_col))
                    error = _recorded_error(df, idx)
                    snapshot = _count_row(idx)
                _finish(row, started, snapshot, success=False, error=error, values={})
        with lock:
            messages = row_warnings.copy()
            row_warnings.clear()
        for message in messages:
            _warn_at_user(message)

    with _ProgressBar(total=len(rows_to_process), desc=desc) as pbar:
        with (
            _on_exception(mark_uncounted_rows, save_registered_rows),
            _closing_in_slices(parallel_requests) as executor,
        ):
            while True:
                with lock:
                    limit = current_workers
                while len(in_flight) < limit and not abort_event.is_set():
                    row = next(pending_rows, None)
                    if row is None:
                        break
                    # A linha que não foi despachada nunca rodou: falhar ao
                    # despachar interrompe a execução e a deixa sem status, para
                    # a retomada. Como 'error', o resume=True a preservaria.
                    try:
                        in_flight[executor.submit(process_with_warning_site, row)] = (
                            row,
                            time.monotonic(),
                        )
                    except Exception as e:  # noqa: BLE001 (ver o comentário acima)
                        with lock:
                            token_stats.setdefault("abort", _error_text(e))
                        abort_event.set()
                        break

                if not in_flight:
                    break
                # A espera tem prazo por causa do polars, que, ao ser importado,
                # instala um handler de SIGINT com SA_RESTART. Com ele, o kernel
                # reinicia a espera sem prazo de um lock em vez de interrompê-la, e
                # o Ctrl-C só viraria KeyboardInterrupt quando uma linha terminasse.
                # O que corta a espera é haver prazo, e não a duração da fatia: no
                # Linux, a espera com prazo de um semáforo não é reiniciada nem com
                # SA_RESTART (ver signal(7)) e volta com EINTR assim que o sinal
                # chega, e o interpretador levanta o KeyboardInterrupt na hora. A
                # fatia curta só limita a demora numa plataforma que reinicie também
                # a espera com prazo. Sem linha terminada no prazo, `done` volta
                # vazio e o laço só dá a volta: relê o teto e espera de novo.
                done, _ = wait(in_flight, timeout=_WAIT_SLICE_SECONDS, return_when=FIRST_COMPLETED)
                for future in done:
                    collect(future)
                    pbar.update()

        # A gravação do `with` acima não cobre o que vem depois dele. Uma exceção
        # durante as marcas de uma interrupção ou durante a gravação final, como
        # o Ctrl-C, grava o checkpoint (ou o regrava) antes de ser relançada, e
        # a cauda chega ao arquivo, como no modo sequencial. O Ctrl-C que corta
        # as marcas as completa antes de gravar.
        def save_tail() -> None:
            if writer:
                _try_save_checkpoint(df, writer.checkpoint)

        with _on_exception(mark_uncounted_rows, save_tail):
            if abort_event.is_set():
                mark_unreached(counted)

            # Save final: a cauda (< batch_size), o que uma gravação que falhou deixou de
            # fora e as marcas de uma interrupção, que não contam como linha processada.
            if writer and (checkpoint_counter > writer.last_saved or abort_event.is_set()):
                _try_save_checkpoint(df, writer.checkpoint)

    elapsed = time.time() - start_time
    token_stats["elapsed_seconds"] = elapsed
    token_stats["initial_workers"] = initial_workers
    token_stats["final_workers"] = current_workers
    token_stats["workers_reduced"] = current_workers < initial_workers

    return token_stats
