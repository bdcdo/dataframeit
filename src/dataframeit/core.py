"""Função dataframeit: valida a configuração, processa as linhas e devolve o resultado."""

from __future__ import annotations

import importlib.util
import json
import numbers
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
from pandas.api.types import is_scalar
from pydantic import BaseModel, ConfigDict, ValidationError, create_model
from tqdm import tqdm

from .conditional import (
    _CONDITIONAL_KEYS,
    _FIELD_CONFIG_KEYS,
    _collect_configured_fields,
    _walk_fields,
    evaluate_condition,
    get_field_execution_order,
    get_group_execution_units,
)
from .errors import (
    get_friendly_error_message,
    is_rate_limit_error,
    is_recoverable_error,
    validate_provider_dependencies,
    validate_search_dependencies,
)
from .llm import (
    LLMConfig,
    SearchConfig,
    SearchGroupConfig,
    _BuildOnce,
    build_structured_llm,
    call_langchain,
    with_shared_chat_model,
)
from .search import get_available_providers, get_provider
from .utils import (
    DEFAULT_TEXT_COLUMN,
    ORIGINAL_TYPE_PANDAS_DF,
    ORIGINAL_TYPE_POLARS_DF,
    TOKEN_COLUMNS,
    ConversionInfo,
    from_pandas,
    get_complex_fields,
    normalize_complex_columns,
    normalize_value,
    to_pandas,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterator, Sequence

    from polars import DataFrame as PolarsDataFrame
    from polars import Series as PolarsSeries

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

# Chamadas de busca por campo acima das quais vale avisar sobre rate limit.
_PER_FIELD_SEARCH_WARNING_CALLS = 100

# Contadores de uso somados de cada linha para o resumo.
_TOKEN_STAT_KEYS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "total_tokens",
    "reasoning_tokens",
)


def _empty_token_stats() -> dict:
    """Contadores zerados de tokens, busca e custo de uma execução."""
    return {
        **dict.fromkeys(_TOKEN_STAT_KEYS, 0),
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
    invoke: Callable[[str], dict]
    invoke_partial: Callable[[str, set, dict], dict] | None = None


# Detalhe gravado na linha cujo texto está vazio, e que por isso não vai ao LLM.
_MISSING_TEXT_DETAIL = "Texto ausente"


# Sufixo do erro de uma linha já processada cujo reprocessamento falhou.
_KEPT_VALUES_NOTE = " (reprocess_columns falhou; a linha mantém os valores anteriores)"


def _success_details(retry_info: dict) -> str | None:
    """Detalhe gravado numa linha bem-sucedida: os retries, se houve."""
    retries = retry_info.get("retries", 0)
    return f"Sucesso após {retries} retry(s)" if retries > 0 else None


def _missing_text_detail(
    *, row_already_processed: bool, reprocess_columns: Sequence[str] | None
) -> str:
    """Detalhe da linha sem texto; sob reprocess_columns, a linha mantém os valores."""
    if row_already_processed and reprocess_columns:
        return _MISSING_TEXT_DETAIL + _KEPT_VALUES_NOTE
    return _MISSING_TEXT_DETAIL


def _as_object_columns(df: pd.DataFrame, columns: list) -> None:
    """Converte para object as colunas que vão receber texto, lista ou None.

    Uma coluna lida de CSV ou XLSX toda vazia volta como float, e gravar texto
    nela levanta TypeError no pandas 3. Vale para as colunas do modelo e para
    as de controle (status e detalhe de erro).
    """
    for col in columns:
        if col in df.columns and df[col].dtype != object:
            df[col] = df[col].astype(object)


def _is_missing_text(value: object) -> bool:
    """None, NaN ou texto só com espaços: não há o que mandar ao LLM."""
    if isinstance(value, str):
        return not value.strip()
    return is_scalar(value) and bool(pd.isna(value))


def _warn_missing_texts(df: pd.DataFrame, text_column: str, rows: list) -> None:
    """Avisa uma vez quantas linhas a processar não têm texto."""
    missing = sum(1 for idx in rows if _is_missing_text(df.loc[idx, text_column]))
    if missing:
        warnings.warn(
            f"{missing} linha(s) sem texto não vão ao LLM e ficam com status 'error' "
            f"('{_MISSING_TEXT_DETAIL}').",
            UserWarning,
            stacklevel=3,
        )


def _invoke_row(
    backend: ProviderBackend,
    text: str,
    row: pd.Series,
    reprocess_on_row: Sequence[str] | None,
    expected_columns: list,
) -> dict:
    """Chama o backend para uma linha, pedindo só as colunas a reprocessar quando dá.

    `reprocess_on_row` é reprocess_columns numa linha já processada, e None
    nas demais. Numa linha já processada, só essas colunas são gravadas.
    Com um backend campo a campo, os demais campos nem são pedidos: voltam com
    o valor já gravado, que também alimenta as condições.
    """
    if reprocess_on_row and backend.invoke_partial:
        known = {
            col: None if is_scalar(row[col]) and pd.isna(row[col]) else row[col]
            for col in expected_columns
            if col not in reprocess_on_row and col in row.index
        }
        return backend.invoke_partial(text, set(reprocess_on_row), known)
    return backend.invoke(text)


def _condition_holds(condition: object, row_values: dict, field_name: str) -> bool:
    """Avalia a condição com os valores da linha; erro conta como verdadeira.

    Verdadeira é o lado conservador: o campo ausente continua acusado, e a
    retomada pede para reprocessá-lo.
    """
    try:
        return bool(evaluate_condition(condition, row_values, field_name))
    except Exception:  # noqa: BLE001 (ver a docstring)
        return True


def _validate_processed_rows(  # noqa: C901, PLR0912, PLR0915 (validação por linha e por campo)
    df: pd.DataFrame,
    status_col: str,
    pydantic_model: type[BaseModel],
    complex_fields: set[str],
) -> tuple[list[str], dict[tuple[int, str], Any]]:
    """Valida linhas concluídas sem alterar o checkpoint recebido."""
    incompatible_fields: set[str] = set()
    values_to_fill: dict[tuple[int, str], Any] = {}
    expected_columns = list(pydantic_model.model_fields)
    field_by_alias = {field_name: field_name for field_name in expected_columns}
    for field_name, field in pydantic_model.model_fields.items():
        if isinstance(field.alias, str):
            field_by_alias[field.alias] = field_name
        if isinstance(field.validation_alias, str):
            field_by_alias[field.validation_alias] = field_name

    if status_col not in df.columns:
        return [], values_to_fill

    validation_model = pydantic_model
    if any(
        field.alias is not None or field.validation_alias is not None
        for field in pydantic_model.model_fields.values()
    ):
        validation_model = type(
            f"{pydantic_model.__name__}CheckpointValidation",
            (pydantic_model,),
            {
                "model_config": ConfigDict(
                    **{
                        **pydantic_model.model_config,
                        "populate_by_name": True,
                        "validate_by_name": True,
                    }
                ),
                "__module__": pydantic_model.__module__,
            },
        )

    # Campo com `condition` fica None quando a condição é falsa na linha, mesmo
    # que o tipo declarado seja obrigatório. Só nesse caso o None é aceito, e só
    # o campo pulado sai da validação: um valor presente passa pelas restrições
    # do modelo, e a condição verdadeira com valor ausente continua acusada.
    conditions = {
        field_name: field.json_schema_extra["condition"]
        for field_name, field in pydantic_model.model_fields.items()
        if isinstance(field.json_schema_extra, dict) and "condition" in field.json_schema_extra
    }
    skipping_models: dict[frozenset, Any] = {}

    def model_skipping(skipped: frozenset) -> type[BaseModel]:
        if not skipped:
            return validation_model
        if skipped not in skipping_models:
            # Campos por **kwargs, que o ty não casa com as sobrecargas de create_model.
            skipping_models[skipped] = create_model(  # ty: ignore[no-matching-overload]
                f"{pydantic_model.__name__}CheckpointSkipped",
                __base__=validation_model,
                **dict.fromkeys(skipped, (Any, None)),
            )
        return skipping_models[skipped]

    processed_positions = [
        position for position, status in enumerate(df[status_col]) if status == "processed"
    ]
    for position in processed_positions:
        row = df.iloc[position]
        row_values = {}
        for field_name in pydantic_model.model_fields:
            if field_name not in df.columns:
                row_values[field_name] = None
                continue
            value = row[field_name]
            if is_scalar(value) and bool(pd.isna(value)):
                row_values[field_name] = None
            elif field_name in complex_fields:
                row_values[field_name] = normalize_value(value)
            else:
                row_values[field_name] = value

        skipped = frozenset(
            field_name
            for field_name, condition in conditions.items()
            if row_values[field_name] is None
            and not _condition_holds(condition, row_values, field_name)
        )

        projected = {}
        missing_values = set()
        for field_name, field in pydantic_model.model_fields.items():
            value = row_values[field_name]
            if value is None and (field_name not in df.columns or pd.isna(row[field_name])):
                missing_values.add(field_name)
                if not field.is_required() or field_name in skipped:
                    continue
            projected[field_name] = value

        try:
            validated = model_skipping(skipped).model_validate(projected)
        except ValidationError as error:
            for detail in error.errors():
                location = detail.get("loc", ())
                field_name = field_by_alias.get(location[0]) if location else None
                if field_name is not None:
                    incompatible_fields.add(field_name)
                else:
                    incompatible_fields.update(expected_columns)
            for field_name in missing_values:
                field = pydantic_model.model_fields[field_name]
                if field.is_required():
                    continue
                try:
                    default = field.get_default(call_default_factory=True)
                except ValueError:
                    # This factory needs the fields that are being reprocessed.
                    incompatible_fields.add(field_name)
                else:
                    values_to_fill[(position, field_name)] = default
            continue

        validated_data = validated.model_dump()
        for field_name in missing_values:
            values_to_fill[(position, field_name)] = validated_data[field_name]

    ordered_incompatible = [field for field in expected_columns if field in incompatible_fields]
    return ordered_incompatible, values_to_fill


def _apply_processed_values(
    df: pd.DataFrame,
    values: dict[tuple[int, str], Any],
) -> None:
    """Grava por posição os valores completados na validação do checkpoint."""
    for (position, field_name), value in values.items():
        column_position = df.columns.get_loc(field_name)
        # .iat grava lista ou dict como valor único da célula, como _set_cell.
        df.iat[position, column_position] = value  # noqa: PD009


@contextmanager
def _provider_backend(
    config: LLMConfig,
    pydantic_model: type[BaseModel],
    user_prompt: str,
    trace_mode: str | None,
) -> Iterator[ProviderBackend]:
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
            call_agent_per_field,
            call_agent_per_group,
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

        search_call = call_agent_per_group if search_config.groups else call_agent_per_field

        def invoke_partial(text: str, only_fields: set, known: dict) -> dict:
            return search_call(
                text,
                pydantic_model,
                user_prompt,
                config,
                trace_mode,
                only_fields=only_fields,
                known=known,
            )

        yield ProviderBackend(
            label="langchain",
            invoke=lambda text: search_call(text, pydantic_model, user_prompt, config, trace_mode),
            invoke_partial=invoke_partial,
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
    concurrent_queries = parallel_requests * queries_per_row

    provider = get_provider(search_provider)
    provider_limit = provider.requests_per_minute
    provider_name = provider.friendly_name

    issues: list[str] = []

    if concurrent_queries > _RECOMMENDED_MAX_CONCURRENT_SEARCH_QUERIES:
        issues.append(
            f"queries concorrentes estimadas: {concurrent_queries} "
            f"(limite recomendado: {_RECOMMENDED_MAX_CONCURRENT_SEARCH_QUERIES})"
        )

    if rate_limit_delay > 0:
        estimated_rpm = (60 / rate_limit_delay) * parallel_requests
    else:
        estimated_rpm = parallel_requests * 60
    if search_per_field:
        estimated_rpm *= num_fields

    if estimated_rpm > provider_limit * 0.8:
        issues.append(
            f"taxa estimada ~{estimated_rpm:.0f} req/min "
            f"(limite {provider_name}: ~{provider_limit}/min)"
        )

    if not issues:
        return

    # Recomendações: reduzir paralelismo e sugerir delay adequado ao provedor.
    if search_per_field:
        recommended_parallel = max(1, _RECOMMENDED_MAX_CONCURRENT_SEARCH_QUERIES // num_fields)
        recommended_delay = (60 * recommended_parallel * num_fields) / (provider_limit * 0.7)
    else:
        recommended_parallel = min(parallel_requests, _RECOMMENDED_MAX_CONCURRENT_SEARCH_QUERIES)
        recommended_delay = (60 * recommended_parallel) / (provider_limit * 0.7)
    recommended_parallel = max(1, recommended_parallel)
    recommended_delay = max(0.5, round(recommended_delay, 1))

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

    warnings.warn(msg, UserWarning, stacklevel=3)


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

    from .agent import _get_field_config  # noqa: PLC0415 (ver _provider_backend)

    field_configs = {
        field_name: _get_field_config(field_info.json_schema_extra)
        if isinstance(field_info.json_schema_extra, dict)
        else {}
        for field_name, field_info in questions.model_fields.items()
    }
    _, dependencies = get_field_execution_order(questions, field_configs)
    if search_config.groups:
        get_group_execution_units(questions, search_config.groups, dependencies)


def _validate_search_groups(  # noqa: C901 (uma checagem por regra de search_groups)
    search_groups: dict[str, dict],
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


def dataframeit(  # noqa: C901, PLR0912, PLR0913, PLR0915, PLR0917 (API pública: cada parâmetro é uma opção documentada)
    data: pd.DataFrame | pd.Series | PolarsDataFrame | PolarsSeries | list | dict,
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
    search_groups: dict[str, dict] | None = None,
    save_trace: bool | Literal["full", "minimal"] | None = None,  # noqa: FBT001 (posicional na API pública)
    batch_size: int | None = None,
    checkpoint_path: str | Path | None = None,
) -> pd.DataFrame | PolarsDataFrame:
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
            Se False, processa toda linha que não esteja 'processed'. Com False, se
            alguma coluna do modelo já existe no DataFrame e reprocess_columns não
            foi passado, emite um aviso e devolve a entrada sem processar.
        reprocess_columns: Lista de colunas para forçar reprocessamento. Útil para
            atualizar colunas específicas com novas instruções sem perder outras.
        model: Nome do modelo LLM. Se None, usa o default do provider em
            DEFAULT_MODELS; com 'codex' e 'claude_code', o runtime escolhe.
        provider: Provider do LangChain ('openai', 'google_genai', 'anthropic', etc),
            'claude_code' ou 'codex'. Codex usa o SDK Python oficial.
        status_column: Coluna para rastrear progresso. Se None, usa
            "_dataframeit_status".
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
        track_tokens: Se True, rastreia uso de tokens e exibe estatísticas (padrão: True).
        model_kwargs: Parâmetros extras do modelo (ex: temperature, reasoning_effort).
            Com provider='codex', aceita somente effort.
        parallel_requests: Número de requisições paralelas (padrão: 1 = sequencial).
            Se > 1, processa múltiplas linhas simultaneamente.
            Ao detectar erro de rate limit (429), o número de workers é reduzido automaticamente.
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
        checkpoint_path: Caminho do arquivo de checkpoint. Formato inferido pela extensão
            (.csv, .xlsx, .parquet). Escrita atômica via .tmp + rename. Requer `batch_size`.
            Exemplo: ``dataframeit(df, Model, PROMPT, batch_size=100, checkpoint_path="ckpt.xlsx")``.

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
            stacklevel=2,
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
    if batch_size is not None and checkpoint_path is not None:
        if (
            not isinstance(batch_size, numbers.Integral)
            or isinstance(batch_size, bool)
            or batch_size < 1
        ):
            msg = f"batch_size deve ser int >= 1; recebido {batch_size!r}"
            raise ValueError(msg)
        _validate_checkpoint_extension(checkpoint_path)

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
    if save_trace:
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
    df_pandas, conversion_info = to_pandas(data)

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
                        stacklevel=2,
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
    if search_groups:
        validated_groups = _validate_search_groups(
            search_groups,
            questions,
            use_search=use_search,
            search_per_field=search_per_field,
        )
        # search_config existe: _validate_search_groups exige use_search=True.
        if search_config is not None:
            search_config.groups = validated_groups

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
    complex_fields = get_complex_fields(questions)

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

    # Verificar conflitos de colunas
    existing_cols = [col for col in expected_columns if col in df_pandas.columns]
    if existing_cols and not resume and not reprocess_columns:
        warnings.warn(
            f"Colunas {existing_cols} já existem. Use resume=True para continuar ou renomeie-as.",
            stacklevel=1,
        )
        return from_pandas(df_pandas, conversion_info, status_col)

    # Sem coluna de status, resume=True trata toda linha como pendente. Numa
    # saída anterior sem erros a coluna foi removida, e rodar de novo refaria
    # todas as chamadas.
    if (
        resume
        and not reprocess_columns
        and status_col not in df_pandas.columns
        and existing_cols == expected_columns
        and df_pandas[expected_columns].notna().to_numpy().any()
    ):
        warnings.warn(
            f"As colunas {expected_columns} já estão preenchidas e não há coluna "
            f"'{status_col}': todas as linhas serão processadas de novo. Se o "
            "DataFrame já é uma saída do dataframeit, não é preciso rodar outra vez; "
            "para refazer só algumas colunas, use reprocess_columns.",
            UserWarning,
            stacklevel=2,
        )

    reprocessed_columns = set(reprocess_columns or [])
    incompatible_columns = []
    processed_values = {}
    if resume or reprocess_columns:
        incompatible_columns, processed_values = _validate_processed_rows(
            df_pandas,
            status_col,
            questions,
            complex_fields,
        )
    uncovered_columns = [
        column for column in incompatible_columns if column not in reprocessed_columns
    ]
    if uncovered_columns:
        msg = (
            "O DataFrame contém linhas processadas incompatíveis com o modelo atual: "
            f"campos incompatíveis {uncovered_columns}. "
            f"Inclua-os em reprocess_columns={incompatible_columns!r}."
        )
        raise ValueError(msg)

    # Um checkpoint sem posição pendente não depende do provider nem de autenticação.
    if (
        resume
        and not reprocess_columns
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
        _as_object_columns(df_pandas, expected_columns)
        _apply_processed_values(df_pandas, processed_values)
        if complex_fields:
            normalize_complex_columns(df_pandas, complex_fields)
        return from_pandas(df_pandas, conversion_info, status_col)

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
        is_risky = parallel_requests > 1 or (
            search_per_field
            and len(expected_columns) * len(df_pandas) > _PER_FIELD_SEARCH_WARNING_CALLS
        )
        if is_risky:
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
        _as_object_columns(df_pandas, expected_columns + control_columns)
        _apply_processed_values(df_pandas, processed_values)

        # Normalizar colunas complexas (listas, dicts, tuples) que podem ter sido
        # serializadas como strings JSON ao salvar/carregar de arquivos.
        if complex_fields and resume:
            normalize_complex_columns(df_pandas, complex_fields)

        # De novo depois da normalização, cujo apply volta a inferir float; sem
        # isso, gravar lista ou texto falharia depois da chamada paga.
        _as_object_columns(df_pandas, expected_columns)

        is_pending, processed_count = _get_processing_indices(df_pandas, status_col, resume=resume)
        _warn_missing_texts(
            df_pandas,
            text_column,
            [
                idx
                for idx, pending in zip(df_pandas.index, is_pending, strict=True)
                if pending or reprocess_columns
            ],
        )

        if parallel_requests > 1:
            token_stats = _process_rows_parallel(
                df_pandas,
                text_column=text_column,
                status_col=status_col,
                expected_columns=expected_columns,
                config=config,
                backend=backend,
                is_pending=is_pending,
                processed_count=processed_count,
                conversion_info=conversion_info,
                track_tokens=track_tokens,
                reprocess_columns=reprocess_columns,
                parallel_requests=parallel_requests,
                trace_mode=trace_mode,
                batch_size=batch_size,
                checkpoint_path=checkpoint_path,
            )
        else:
            token_stats = _process_rows(
                df_pandas,
                text_column=text_column,
                status_col=status_col,
                expected_columns=expected_columns,
                config=config,
                backend=backend,
                is_pending=is_pending,
                processed_count=processed_count,
                conversion_info=conversion_info,
                track_tokens=track_tokens,
                reprocess_columns=reprocess_columns,
                trace_mode=trace_mode,
                batch_size=batch_size,
                checkpoint_path=checkpoint_path,
            )

    # Exibir estatísticas de tokens e throughput
    if track_tokens and token_stats and any(token_stats.values()):
        _print_token_stats(
            token_stats,
            model,
            parallel_requests,
            search_provider=search_provider if use_search else None,
        )

    # Aviso de workers reduzidos (aparece SEMPRE, independente de track_tokens)
    if token_stats.get("workers_reduced"):
        print("\n" + "=" * 60)
        print("AVISO: WORKERS REDUZIDOS POR RATE LIMIT")
        print("=" * 60)
        print(f"Workers iniciais: {token_stats['initial_workers']}")
        print(f"Workers finais:   {token_stats['final_workers']}")
        print(
            f"\nDica: Considere usar parallel_requests={token_stats['final_workers']} "
            f"para evitar rate limits."
        )
        print("=" * 60 + "\n")

    # Retornar no formato original (remove colunas de status/erro se não houver erros)
    return from_pandas(df_pandas, conversion_info, status_col)


def _setup_columns(  # noqa: C901, PLR0912, PLR0913 (uma família de colunas por recurso ligado)
    df: pd.DataFrame,
    expected_columns: list,
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
    if trace_mode:
        if search_config and search_config.per_field and pydantic_model:
            if search_config.groups:
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

    # Identificar colunas que precisam ser criadas
    new_cols = [col for col in expected_columns if col not in df.columns]
    needs_status = status_col not in df.columns
    needs_error = error_col not in df.columns
    needs_tokens = [col for col in token_cols if col not in df.columns] if track_tokens else []
    needs_search = [col for col in search_cols if col not in df.columns]
    needs_trace = [col for col in trace_cols if col not in df.columns]

    if (
        not new_cols
        and not needs_status
        and not needs_error
        and not needs_tokens
        and not needs_search
        and not needs_trace
    ):
        return

    # Criar colunas
    with pd.option_context("mode.chained_assignment", None):
        for col in new_cols:
            df[col] = None
        if needs_status:
            df[status_col] = None
        if needs_error:
            df[error_col] = None
        if track_tokens:
            for col in needs_tokens:
                df[col] = None
        for col in needs_search:
            df[col] = None
        for col in needs_trace:
            df[col] = None


def _get_processing_indices(
    df: pd.DataFrame, status_col: str, *, resume: bool
) -> tuple[list[bool], int]:
    """Retorna (linhas pendentes por posição, contagem de linhas com status).

    A seleção depende só do status de cada linha, e nunca da ordem dos rótulos do
    índice, que pode vir fora de ordem (sort_values, sample, filtro), ser textual ou
    vir das chaves de um dict.

    Com resume=True, só a linha sem status fica pendente: linhas 'processed' e
    'error' são preservadas, e o usuário limpa o status de um erro para reprocessá-lo.
    Com resume=False, fica pendente toda linha que não esteja 'processed'.
    Quando reprocess_columns está definido, _process_rows e _process_rows_parallel
    ignoram a seleção e percorrem todas as linhas.
    """
    status = df[status_col]
    if not resume:
        return status.ne("processed").tolist(), 0

    is_pending = status.isna()
    processed_count = int((~is_pending).sum())
    return is_pending.tolist(), processed_count


def _print_token_stats(
    token_stats: dict,
    model: str | None,
    parallel_requests: int = 1,
    search_provider: str | None = None,
) -> None:
    """Exibe estatísticas de uso de tokens e throughput.

    Args:
        token_stats: Dict com contadores de tokens e métricas de tempo.
        model: Nome do modelo usado; None quando o runtime do provider escolhe.
        parallel_requests: Número de workers paralelos usados.
        search_provider: Provedor de busca usado, que dá nome à seção de busca.
    """
    if not token_stats or (
        token_stats.get("total_tokens", 0) == 0 and not token_stats.get("cost_usd")
    ):
        return

    print("\n" + "=" * 60)
    print("ESTATISTICAS DE USO")
    print("=" * 60)
    print(f"Modelo: {model or 'escolhido pelo runtime do provider'}")
    print(f"Total de tokens: {token_stats['total_tokens']:,}")
    print(f"  - Input:  {token_stats['input_tokens']:,} tokens")
    if token_stats.get("cached_input_tokens", 0) > 0:
        print(f"    └─ Cache: {token_stats['cached_input_tokens']:,} (incluído no Input)")
    print(f"  - Output: {token_stats['output_tokens']:,} tokens")
    if token_stats.get("reasoning_tokens", 0) > 0:
        print(f"    └─ Reasoning: {token_stats['reasoning_tokens']:,} (incluído no Output)")
    # Só providers que informam o custo, como o claude_code, preenchem este total,
    # que inclui as tentativas re-tentadas e as linhas que falharam.
    if token_stats.get("cost_usd", 0) > 0:
        print(f"Custo informado pelo provider: US$ {token_stats['cost_usd']:.4f}")

    # Métricas de throughput (se disponíveis)
    if "elapsed_seconds" in token_stats and token_stats["elapsed_seconds"] > 0:
        elapsed = token_stats["elapsed_seconds"]
        requests = token_stats.get("requests_completed", 0)

        print("-" * 60)
        print("METRICAS DE THROUGHPUT")
        print("-" * 60)
        print(f"Tempo total: {elapsed:.1f}s")
        print(f"Workers paralelos: {parallel_requests}")

        if requests > 0:
            rpm = (requests / elapsed) * 60
            print(f"Requisicoes: {requests}")
            print(f"  - RPM (req/min): {rpm:.1f}")

        tpm = (token_stats["total_tokens"] / elapsed) * 60
        print(f"  - TPM (tokens/min): {tpm:,.0f}")

    # Métricas de busca (se houver)
    if token_stats.get("search_count", 0) > 0:
        print("-" * 60)
        print(f"METRICAS DE BUSCA ({(search_provider or 'tavily').upper()})")
        print("-" * 60)
        print(f"Total de buscas: {token_stats['search_count']}")
        print(f"Creditos usados: {token_stats['search_credits']}")

    print("=" * 60 + "\n")


_SUPPORTED_CHECKPOINT_EXTS = (".csv", ".xlsx", ".parquet")

# Extensões que exigem dependência opcional para pandas serializar.
# Validamos antes do loop para falhar rápido — um ModuleNotFoundError no primeiro
# save (após N linhas de LLM) desperdiça horas de trabalho.
_CHECKPOINT_EXT_REQUIRES = {
    ".xlsx": ("openpyxl", "openpyxl"),
    ".parquet": ("pyarrow", "pyarrow"),
}


def _validate_checkpoint_extension(path: str | Path) -> None:
    """Valida extensão suportada e dependência opcional necessária."""
    ext = Path(path).suffix.lower()
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


def _structures_as_json(df: pd.DataFrame) -> pd.DataFrame:
    """Cópia com listas, dicts e tuplas serializados como JSON.

    CSV e XLSX gravariam o repr Python ("['a', 'b']"), que json.loads não lê
    de volta. JSON é o que read_df e a retomada normalizam.
    """

    def to_json(value: object) -> object:
        if isinstance(value, (list, dict, tuple)):
            return json.dumps(value, ensure_ascii=False, default=str)
        return value

    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].map(to_json)
    return out


def _save_checkpoint(df: pd.DataFrame, path: str | Path) -> None:
    """Salva DataFrame em disco com escrita atômica. Formato inferido pela extensão."""
    path = Path(path)
    ext = path.suffix.lower()
    tmp = path.with_name(path.name + ".tmp")
    if ext == ".csv":
        _structures_as_json(df).to_csv(tmp, index=False)
    elif ext == ".xlsx":
        _structures_as_json(df).to_excel(tmp, index=False)
    elif ext == ".parquet":
        df.to_parquet(tmp, index=False)
    else:
        msg = (
            f"Extensão {ext} não suportada para checkpoint. "
            f"Use uma de: {', '.join(_SUPPORTED_CHECKPOINT_EXTS)}"
        )
        raise ValueError(msg)
    tmp.replace(path)


def _try_save_checkpoint(df: pd.DataFrame, path: str | Path | None) -> bool:
    """Grava o checkpoint e devolve se deu certo; falha vira aviso.

    Uma falha de gravação (disco cheio, arquivo aberto no Excel, coluna que o
    parquet não serializa) não diz nada sobre a linha que acabou de ser
    processada, e por isso nunca muda o status dela nem interrompe a execução.
    A próxima gravação tenta de novo, com o estado completo. Sem caminho,
    não há checkpoint a gravar.
    """
    if path is None:
        return False
    try:
        _save_checkpoint(df, path)
    except Exception as error:  # noqa: BLE001 (ver a docstring)
        warnings.warn(
            f"Falha ao gravar o checkpoint em {path}: {type(error).__name__}: {error}. "
            "O processamento continua, e a próxima gravação tenta de novo.",
            # usuário -> dataframeit -> _process_rows -> aqui; no modo paralelo
            # o aviso sai de uma thread do executor e não tem quadro do usuário.
            stacklevel=4,
        )
        return False
    return True


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


def _record_success(  # noqa: C901, PLR0913 (grava cada família de colunas da linha)
    df: pd.DataFrame,
    idx: Hashable,
    result: dict,
    *,
    status_col: str,
    expected_columns: list,
    config: LLMConfig,
    track_tokens: bool,
    row_already_processed: bool,
    reprocess_columns: Sequence[str] | None,
    trace_mode: str | None,
    token_stats: dict,
) -> None:
    """Grava na linha o resultado de uma chamada bem-sucedida e soma o uso.

    Numa linha já processada sob reprocess_columns, só essas colunas são
    gravadas; nas demais linhas, todas as colunas do modelo.
    """
    extracted = result["data"]
    usage = result.get("usage")
    retry_info = result.get("_retry_info", {})
    only_reprocessed = reprocess_columns if row_already_processed else None

    for col in expected_columns:
        if col in extracted and (not only_reprocessed or col in only_reprocessed):
            _set_cell(df, idx, col, extracted[col])

    # Armazenar tokens no DataFrame (se habilitado); o total vai só ao resumo.
    if track_tokens and usage:
        for column in TOKEN_COLUMNS:
            _set_cell(df, idx, column, usage.get(column.removeprefix("_"), 0))
        for key in _TOKEN_STAT_KEYS:
            token_stats[key] += usage.get(key, 0)
        token_stats["cost_usd"] += usage.get("cost_usd") or 0

    # Armazenar métricas de busca (search_count mantido só para o resumo)
    search_config = config.search_config
    if search_config and search_config.enabled and usage:
        _set_cell(df, idx, "_search_credits", usage.get("search_credits", 0))
        token_stats["search_credits"] += usage.get("search_credits", 0)
        token_stats["search_count"] += usage.get("search_count", 0)

    # Armazenar traces: um por campo ou grupo nos modos por campo, ou um só
    if trace_mode:
        if search_config and search_config.per_field:
            for field_name, trace in result.get("traces", {}).items():
                _set_cell(df, idx, f"_trace_{field_name}", json.dumps(trace, ensure_ascii=False))
        else:
            trace = result.get("trace")
            if trace:
                _set_cell(df, idx, "_trace", json.dumps(trace, ensure_ascii=False))

    _set_cell(df, idx, status_col, "processed")
    # Registra retries mesmo no sucesso; sem retry, some o erro de uma execução anterior
    _set_cell(df, idx, "_error_details", _success_details(retry_info))


def _record_error(  # noqa: PLR0913 (o que a linha grava vem da execução inteira)
    df: pd.DataFrame,
    idx: Hashable,
    error: Exception,
    *,
    status_col: str,
    config: LLMConfig,
    row_already_processed: bool,
    reprocess_columns: Sequence[str] | None,
    token_stats: dict,
) -> str:
    """Grava a falha da linha, mostra a mensagem amigável e devolve o erro em texto.

    O custo informado pelo provider entra no resumo mesmo com a linha falhando.
    """
    error_msg = f"{type(error).__name__}: {error}"
    token_stats["cost_usd"] += getattr(error, "cost_usd", 0) or 0

    # Erro recuperável chegou aqui depois de esgotar as tentativas; o não
    # recuperável, sem retry.
    if is_recoverable_error(error):
        error_details = f"[Falhou após {config.max_retries} tentativa(s)] {error_msg}"
    else:
        error_details = f"[Erro não-recuperável] {error_msg}"
    if row_already_processed and reprocess_columns:
        error_details += _KEPT_VALUES_NOTE

    friendly_msg = get_friendly_error_message(error, config.provider)
    print(f"\n{friendly_msg}\n")

    warnings.warn(f"Falha ao processar linha {idx}.", stacklevel=1)
    _set_cell(df, idx, status_col, "error")
    _set_cell(df, idx, "_error_details", error_details)
    return error_msg


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
    if reprocess_columns:
        return f" (reprocessando: {', '.join(reprocess_columns)})"
    if processed_count > 0:
        return f" (resumindo de {processed_count}/{len(df)})"
    return ""


def _process_rows(  # noqa: PLR0913 (estado da execução repassado por dataframeit)
    df: pd.DataFrame,
    *,
    text_column: str,
    status_col: str,
    expected_columns: list,
    config: LLMConfig,
    backend: ProviderBackend,
    is_pending: list[bool],
    processed_count: int,
    conversion_info: ConversionInfo,
    track_tokens: bool,
    reprocess_columns: Sequence[str] | None = None,
    trace_mode: str | None = None,
    batch_size: int | None = None,
    checkpoint_path: str | Path | None = None,
) -> dict:
    """Processa cada linha do DataFrame, em sequência.

    Args:
        df: DataFrame de trabalho, alterado in-place.
        text_column: Coluna com o texto de cada linha.
        status_col: Coluna de status ('processed' ou 'error').
        expected_columns: Campos do modelo, gravados em colunas de mesmo nome.
        config: Configuração do LLM.
        backend: Implementação do provider vinculada à execução.
        is_pending: Por posição, se a linha está pendente.
        processed_count: Linhas com status antes desta execução.
        conversion_info: Metadados da conversão da entrada, usados no rótulo.
        track_tokens: Se True, grava tokens por linha e soma o uso.
        reprocess_columns: Lista de colunas para forçar reprocessamento.
            Se especificado, não pula linhas já processadas.
        trace_mode: Modo de trace ("full", "minimal") ou None para desabilitar.
        batch_size: A cada quantas linhas processadas o checkpoint é gravado.
        checkpoint_path: Arquivo do checkpoint.

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

    # Processar cada linha
    for i, (idx, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc=desc)):
        # Verificar se linha já foi processada
        row_already_processed = pd.notna(row[status_col]) and row[status_col] == "processed"

        # Com reprocess_columns, todas as linhas são processadas.
        if not reprocess_columns and not is_pending[i]:
            continue

        success = False
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
                )
                _record_success(
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
            except Exception as e:  # noqa: BLE001 (qualquer falha da linha vira status 'error')
                _record_error(
                    df,
                    idx,
                    e,
                    status_col=status_col,
                    config=config,
                    row_already_processed=row_already_processed,
                    reprocess_columns=reprocess_columns,
                    token_stats=token_stats,
                )

        rows_processed_this_run += 1
        if (
            batch_size
            and rows_processed_this_run % batch_size == 0
            and _try_save_checkpoint(df, checkpoint_path)
        ):
            rows_saved = rows_processed_this_run

        if success and config.rate_limit_delay > 0:
            time.sleep(config.rate_limit_delay)

    # Save final: a cauda (< batch_size) e o que uma gravação que falhou deixou de fora.
    if batch_size and rows_processed_this_run > rows_saved:
        _try_save_checkpoint(df, checkpoint_path)

    return token_stats


def _process_rows_parallel(  # noqa: C901, PLR0913, PLR0915 (estado compartilhado entre threads)
    df: pd.DataFrame,
    *,
    text_column: str,
    status_col: str,
    expected_columns: list,
    config: LLMConfig,
    backend: ProviderBackend,
    is_pending: list[bool],
    processed_count: int,
    conversion_info: ConversionInfo,
    track_tokens: bool,
    reprocess_columns: Sequence[str] | None,
    parallel_requests: int,
    trace_mode: str | None = None,
    batch_size: int | None = None,
    checkpoint_path: str | Path | None = None,
) -> dict:
    """Processa linhas do DataFrame em paralelo com auto-redução de workers.

    Args:
        df: DataFrame de trabalho, alterado in-place sob lock.
        text_column: Coluna com o texto de cada linha.
        status_col: Coluna de status ('processed' ou 'error').
        expected_columns: Campos do modelo, gravados em colunas de mesmo nome.
        config: Configuração do LLM.
        backend: Implementação do provider vinculada à execução.
        is_pending: Por posição, se a linha está pendente.
        processed_count: Linhas com status antes desta execução.
        conversion_info: Metadados da conversão da entrada, usados no rótulo.
        track_tokens: Se True, grava tokens por linha e soma o uso.
        reprocess_columns: Lista de colunas para forçar reprocessamento.
            Se especificado, não pula linhas já processadas.
        parallel_requests: Número inicial de workers paralelos.
            Será reduzido automaticamente se detectar erros de rate limit (429).
        trace_mode: Modo de trace ("full", "minimal") ou None para desabilitar.
        batch_size: A cada quantas linhas processadas o checkpoint é gravado.
        checkpoint_path: Arquivo do checkpoint.

    Returns:
        Dict com estatísticas de tokens e métricas de throughput.
    """
    start_time = time.time()

    # Estado compartilhado (thread-safe)
    lock = threading.Lock()
    current_workers = parallel_requests
    initial_workers = parallel_requests
    workers_reduced = False
    rate_limit_event = threading.Event()
    checkpoint_counter = 0
    # A gravação do checkpoint não segura `lock`, para que as threads que
    # processam linhas sigam durante a I/O, mas tem trava própria: duas
    # gravações simultâneas disputam o mesmo arquivo temporário. Cada snapshot
    # é copiado sob `lock` na mesma seção que incrementa checkpoint_counter e
    # leva esse valor como rótulo; por isso o snapshot de rótulo maior contém
    # tudo o que os de rótulo menor contêm, e um mais antigo que chegue depois
    # pode ser descartado sem perda.
    checkpoint_write_lock = threading.Lock()
    last_saved_checkpoint = 0

    def _save_snapshot(snapshot: tuple[pd.DataFrame, int] | None) -> None:
        nonlocal last_saved_checkpoint
        if snapshot is None:
            return
        frame, label = snapshot
        with checkpoint_write_lock:
            if label <= last_saved_checkpoint:
                return
            if _try_save_checkpoint(frame, checkpoint_path):
                last_saved_checkpoint = label

    def _count_row() -> tuple[pd.DataFrame, int] | None:
        """Conta a linha; na hora do checkpoint, copia o DataFrame. Chamar sob `lock`."""
        nonlocal checkpoint_counter
        checkpoint_counter += 1
        if batch_size and checkpoint_counter % batch_size == 0:
            # Copia sob lock, serializa fora, para não bloquear threads na I/O.
            return (df.copy(), checkpoint_counter)
        return None

    token_stats = _empty_token_stats()
    token_stats["requests_completed"] = 0

    desc = _progress_description(
        config, backend, conversion_info, f" [{parallel_requests} workers]"
    )
    desc += _progress_suffix(df, reprocess_columns, processed_count)

    # Identificar linhas a processar
    rows_to_process = [
        (i, idx, row)
        for i, (idx, row) in enumerate(df.iterrows())
        if reprocess_columns or is_pending[i]
    ]

    if not rows_to_process:
        return token_stats

    def process_single_row(row_data: tuple) -> dict:
        """Processa uma única linha (executada em thread separada)."""
        nonlocal current_workers, workers_reduced

        _i, idx, row = row_data
        row_already_processed = pd.notna(row[status_col]) and row[status_col] == "processed"
        if _is_missing_text(row[text_column]):
            with lock:
                _record_missing_text(
                    df,
                    idx,
                    status_col=status_col,
                    row_already_processed=row_already_processed,
                    reprocess_columns=reprocess_columns,
                )
                snapshot = _count_row()
            _save_snapshot(snapshot)
            return {"success": False, "idx": idx, "error": _MISSING_TEXT_DETAIL}

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
            )

            # Atualizar DataFrame (com lock para thread-safety)
            with lock:
                _record_success(
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
                snapshot = _count_row()

        except Exception as e:  # noqa: BLE001 (qualquer falha da linha vira status 'error')
            # Verificar se é erro de rate limit
            if is_rate_limit_error(e):
                with lock:
                    if current_workers > 1:
                        old_workers = current_workers
                        current_workers = max(1, current_workers // 2)
                        workers_reduced = True
                        warnings.warn(
                            f"Rate limit detectado! Reduzindo workers de {old_workers} para {current_workers}.",
                            stacklevel=2,
                        )
                        rate_limit_event.set()
                        # Limpar evento após um tempo
                        threading.Timer(5.0, rate_limit_event.clear).start()

            with lock:
                error_msg = _record_error(
                    df,
                    idx,
                    e,
                    status_col=status_col,
                    config=config,
                    row_already_processed=row_already_processed,
                    reprocess_columns=reprocess_columns,
                    token_stats=token_stats,
                )
                snapshot = _count_row()
            _save_snapshot(snapshot)
            return {"success": False, "idx": idx, "error": error_msg}

        _save_snapshot(snapshot)
        if config.rate_limit_delay > 0:
            time.sleep(config.rate_limit_delay)
        return {"success": True, "idx": idx}

    # Processar com ThreadPoolExecutor
    with tqdm(total=len(rows_to_process), desc=desc) as pbar:
        # Usar abordagem iterativa para permitir ajuste dinâmico de workers
        pending_rows = list(rows_to_process)

        while pending_rows:
            # Pegar batch com número atual de workers
            with lock:
                worker_batch = min(current_workers, len(pending_rows))
            batch = pending_rows[:worker_batch]
            pending_rows = pending_rows[worker_batch:]

            with ThreadPoolExecutor(max_workers=worker_batch) as executor:
                futures = {executor.submit(process_single_row, row): row for row in batch}

                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:  # noqa: BLE001 (falha fora da linha não para a execução)
                        warnings.warn(f"Erro inesperado no executor: {e}", stacklevel=1)
                    pbar.update(1)

    # Save final: a cauda (< batch_size) e o que uma gravação que falhou deixou de fora.
    if batch_size and checkpoint_counter > last_saved_checkpoint:
        _try_save_checkpoint(df, checkpoint_path)

    elapsed = time.time() - start_time
    token_stats["elapsed_seconds"] = elapsed
    token_stats["initial_workers"] = initial_workers
    token_stats["final_workers"] = current_workers
    token_stats["workers_reduced"] = workers_reduced

    return token_stats
