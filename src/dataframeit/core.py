import json
import logging
import numbers
import os
import threading
import time
import warnings
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pandas.api.types import is_scalar
from pydantic import ConfigDict, ValidationError, create_model
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
from .llm import LLMConfig, SearchConfig, SearchGroupConfig, call_langchain
from .search import get_available_providers, get_provider
from .utils import (
    DEFAULT_TEXT_COLUMN,
    ORIGINAL_TYPE_PANDAS_DF,
    ORIGINAL_TYPE_POLARS_DF,
    TOKEN_COLUMNS,
    from_pandas,
    get_complex_fields,
    normalize_complex_columns,
    normalize_value,
    to_pandas,
)

# Suprimir mensagens de retry do LangChain (elas são redundantes com nossos warnings)
logging.getLogger('langchain_google_genai').setLevel(logging.ERROR)
logging.getLogger('langchain_core').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)

# Nomes candidatos consultados quando o usuário não passa text_column explicitamente.
# Ordem: convenção da lib ('texto'), inglês ('text'), juscraper cjpg/cjsg ('decisao'),
# e convenções comuns de ETL ('content', 'content_text').
TEXT_COLUMN_CANDIDATES = ('texto', 'text', 'decisao', 'content', 'content_text')

# Modelo usado quando o usuário escolhe o provider sem escolher o modelo. Mandar
# o modelo de um provider para outro falha em toda linha, por isso o default
# acompanha o provider. 'codex' e 'claude_code' ficam de fora: com model=None,
# o runtime de cada um escolhe o modelo.
DEFAULT_MODELS = {
    'openai': 'gpt-6-luna',
    'google_genai': 'gemini-3.8-flash',
    'anthropic': 'claude-sonnet-5',
    'groq': 'openai/gpt-oss-120b',
}
_RUNTIME_DEFAULT_PROVIDERS = frozenset({'codex', 'claude_code'})


# Limite de queries concorrentes acima do qual vale avisar o usuário.
_RECOMMENDED_MAX_CONCURRENT_SEARCH_QUERIES = 10

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
_MISSING_TEXT_DETAIL = 'Texto ausente'


# Sufixo do erro de uma linha já processada cujo reprocessamento falhou.
_KEPT_VALUES_NOTE = ' (reprocess_columns falhou; a linha mantém os valores anteriores)'


def _success_details(retry_info: dict) -> str | None:
    """Detalhe gravado numa linha bem-sucedida: os retries, se houve."""
    retries = retry_info.get('retries', 0)
    return f"Sucesso após {retries} retry(s)" if retries > 0 else None


def _missing_text_detail(row_already_processed: bool, reprocess_columns) -> str:
    """Detalhe da linha sem texto; sob reprocess_columns, a linha mantém os valores."""
    if row_already_processed and reprocess_columns:
        return _MISSING_TEXT_DETAIL + _KEPT_VALUES_NOTE
    return _MISSING_TEXT_DETAIL


def _as_object_columns(df: pd.DataFrame, columns) -> None:
    """Converte para object as colunas que vão receber texto, lista ou None.

    Uma coluna lida de CSV ou XLSX toda vazia volta como float, e gravar texto
    nela levanta TypeError no pandas 3. Vale para as colunas do modelo e para
    as de controle (status e detalhe de erro).
    """
    for col in columns:
        if col in df.columns and df[col].dtype != object:
            df[col] = df[col].astype(object)


def _is_missing_text(value) -> bool:
    """None, NaN ou texto só com espaços: não há o que mandar ao LLM."""
    if isinstance(value, str):
        return not value.strip()
    return is_scalar(value) and bool(pd.isna(value))


def _warn_missing_texts(df: pd.DataFrame, text_column: str, rows) -> None:
    """Avisa uma vez quantas linhas a processar não têm texto."""
    missing = sum(1 for idx in rows if _is_missing_text(df.at[idx, text_column]))
    if missing:
        warnings.warn(
            f"{missing} linha(s) sem texto não vão ao LLM e ficam com status 'error' "
            f"('{_MISSING_TEXT_DETAIL}').",
            UserWarning,
            stacklevel=3,
        )


def _invoke_row(backend, text, row, row_already_processed, reprocess_columns, expected_columns):
    """Chama o backend para uma linha, pedindo só as colunas a reprocessar quando dá.

    Numa linha já processada, só as colunas de reprocess_columns são gravadas.
    Com um backend campo a campo, os demais campos nem são pedidos: voltam com
    o valor já gravado, que também alimenta as condições.
    """
    if row_already_processed and reprocess_columns and backend.invoke_partial:
        known = {
            col: None if is_scalar(row[col]) and pd.isna(row[col]) else row[col]
            for col in expected_columns
            if col not in reprocess_columns and col in row.index
        }
        return backend.invoke_partial(text, set(reprocess_columns), known)
    return backend.invoke(text)


def _condition_holds(condition, row_values: dict, field_name: str) -> bool:
    """Avalia a condição com os valores da linha; erro conta como verdadeira.

    Verdadeira é o lado conservador: o campo ausente continua acusado, e a
    retomada pede para reprocessá-lo.
    """
    try:
        return bool(evaluate_condition(condition, row_values, field_name))
    except Exception:
        return True


def _validate_processed_rows(
    df: pd.DataFrame,
    status_col: str,
    pydantic_model,
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
        field_name: field.json_schema_extra['condition']
        for field_name, field in pydantic_model.model_fields.items()
        if isinstance(field.json_schema_extra, dict) and 'condition' in field.json_schema_extra
    }
    skipping_models: dict[frozenset, Any] = {}

    def model_skipping(skipped: frozenset):
        if not skipped:
            return validation_model
        if skipped not in skipping_models:
            skipping_models[skipped] = create_model(
                f"{pydantic_model.__name__}CheckpointSkipped",
                __base__=validation_model,
                **dict.fromkeys(skipped, (Any, None)),
            )
        return skipping_models[skipped]

    processed_positions = [
        position
        for position, status in enumerate(df[status_col])
        if status == 'processed'
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
                location = detail.get('loc', ())
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

    ordered_incompatible = [
        field for field in expected_columns if field in incompatible_fields
    ]
    return ordered_incompatible, values_to_fill


def _apply_processed_values(
    df: pd.DataFrame,
    values: dict[tuple[int, str], Any],
) -> None:
    for (position, field_name), value in values.items():
        column_position = df.columns.get_loc(field_name)
        df.iat[position, column_position] = value


@contextmanager
def _provider_backend(
    config: LLMConfig,
    pydantic_model,
    user_prompt: str,
    trace_mode: str | None,
) -> Iterator[ProviderBackend]:
    """Seleciona e vincula uma única implementação para toda a execução."""
    if config.search_config and config.search_config.enabled:
        from .agent import call_agent, call_agent_per_field, call_agent_per_group

        if not config.search_config.per_field:
            search_call = call_agent
        elif config.search_config.groups:
            search_call = call_agent_per_group
        else:
            search_call = call_agent_per_field

        invoke_partial = None
        if search_call is not call_agent:
            def invoke_partial(text, only_fields, known):
                return search_call(
                    text, pydantic_model, user_prompt, config, trace_mode,
                    only_fields=only_fields, known=known,
                )

        yield ProviderBackend(
            label="langchain",
            invoke=lambda text: search_call(
                text, pydantic_model, user_prompt, config, trace_mode
            ),
            invoke_partial=invoke_partial,
        )
        return

    if config.provider == "codex":
        from .codex import open_codex_backend

        with open_codex_backend(config, pydantic_model, user_prompt) as backend:
            yield ProviderBackend(label="codex", invoke=backend.invoke)
        return

    if config.provider == "claude_code":
        from .claude_code import call_claude_code

        yield ProviderBackend(
            label="claude_code",
            invoke=lambda text: call_claude_code(
                text, pydantic_model, user_prompt, config
            ),
        )
        return

    langchain_call = call_langchain
    yield ProviderBackend(
        label="langchain",
        invoke=lambda text: langchain_call(
            text, pydantic_model, user_prompt, config
        ),
    )


def _warn_search_rate_limit(
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
        + "Problemas: " + "; ".join(issues) + ". "
        + "Para evitar HTTP 429, use: " + ", ".join(recs) + "."
    )

    warnings.warn(msg, UserWarning, stacklevel=3)


def _validate_search_overrides(
    label: str, search_depth=None, max_results=None, max_search_calls=None
) -> None:
    """Valida os overrides de busca de um grupo ou campo; None é ausência."""
    if search_depth is not None and search_depth not in ('basic', 'advanced'):
        raise ValueError(f"{label}: search_depth deve ser 'basic' ou 'advanced'")
    if max_results is not None and (
        isinstance(max_results, bool)
        or not isinstance(max_results, numbers.Real)
        or not 1 <= max_results <= 20
    ):
        raise ValueError(f"{label}: max_results deve estar entre 1 e 20")
    if max_search_calls is not None and not _is_positive_int(max_search_calls):
        raise ValueError(f"{label}: max_search_calls deve ser int >= 1")


def _is_positive_int(value) -> bool:
    """Inteiro >= 1; bool é subclasse de int, mas True não é uma contagem."""
    return isinstance(value, numbers.Integral) and not isinstance(value, bool) and value >= 1


def _validate_field_configs(questions, search_config, use_search: bool, search_per_field: bool) -> None:
    """Valida json_schema_extra de todos os campos antes de processar linhas.

    Erro de configuração aparece uma vez aqui, e não linha a linha com o
    prefixo de tentativas, que sugere uma falha do provider.
    """
    per_field = bool(search_config and search_config.per_field)

    configured = _collect_configured_fields(questions)
    if configured and not per_field:
        missing = [
            flag for flag, on in (('use_search=True', use_search), ('search_per_field=True', search_per_field))
            if not on
        ]
        raise ValueError(
            "Campos com configuração em json_schema_extra (prompt, prompt_append, "
            f"search_depth, max_results, max_search_calls) requerem {' e '.join(missing)}"
        )

    for path, field_info, list_depth in _walk_fields(questions):
        extra = field_info.json_schema_extra
        if not isinstance(extra, dict):
            continue

        # Condições só são avaliadas entre campos de primeiro nível
        if '.' in path and any(k in extra for k in _CONDITIONAL_KEYS):
            raise ValueError(
                f"Campo '{path}' usa 'condition' ou 'depends_on' em json_schema_extra, "
                "que só são aplicados a campos de primeiro nível do modelo"
            )

        if any(k in extra for k in _FIELD_CONFIG_KEYS):
            # Um item de lista é enriquecido no próprio dicionário; uma segunda
            # lista no caminho não tem item único onde gravar o valor.
            if list_depth > 1:
                raise ValueError(
                    f"Campo '{path}' tem configuração de busca dentro de uma lista que "
                    "está dentro de outra lista, o que não é suportado"
                )
            _validate_search_overrides(
                f"Campo '{path}'",
                extra.get('search_depth'),
                extra.get('max_results'),
                extra.get('max_search_calls'),
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
            raise ValueError(
                f"Campos {conditional_fields} usam 'condition' ou 'depends_on' em "
                "json_schema_extra, que só são aplicados com use_search=True e "
                "search_per_field=True"
            )
        return

    from .agent import _get_field_config

    field_configs = {
        field_name: _get_field_config(field_info.json_schema_extra)
        if isinstance(field_info.json_schema_extra, dict) else {}
        for field_name, field_info in questions.model_fields.items()
    }
    _, dependencies = get_field_execution_order(questions, field_configs)
    if search_config.groups:
        get_group_execution_units(questions, search_config.groups, dependencies)


def _validate_search_groups(
    search_groups: dict[str, dict],
    pydantic_model,
    use_search: bool,
    search_per_field: bool
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
        raise ValueError("search_groups requer use_search=True")
    if not search_per_field:
        raise ValueError("search_groups requer search_per_field=True")

    expected_fields = set(pydantic_model.model_fields.keys())
    all_grouped_fields = set()
    validated_groups = {}

    for group_name, group_config in search_groups.items():
        # Validar estrutura
        if not isinstance(group_config, dict):
            raise ValueError(f"Grupo '{group_name}' deve ser um dicionário")
        if 'fields' not in group_config:
            raise ValueError(f"Grupo '{group_name}' deve ter chave 'fields'")

        fields = group_config['fields']
        if not isinstance(fields, list) or not fields:
            raise ValueError(f"Grupo '{group_name}': 'fields' deve ser uma lista não-vazia")

        # Validar que campos existem no modelo
        unknown_fields = set(fields) - expected_fields
        if unknown_fields:
            raise ValueError(
                f"Grupo '{group_name}': campos {unknown_fields} não existem no modelo Pydantic. "
                f"Campos disponíveis: {expected_fields}"
            )

        # Validar que campos não pertencem a múltiplos grupos
        duplicate_fields = all_grouped_fields & set(fields)
        if duplicate_fields:
            raise ValueError(
                f"Campos {duplicate_fields} pertencem a múltiplos grupos. "
                f"Cada campo pode pertencer a apenas um grupo."
            )
        all_grouped_fields.update(fields)

        # Validar que campos do grupo não têm json_schema_extra de busca
        for field_name in fields:
            field_info = pydantic_model.model_fields[field_name]
            extra = field_info.json_schema_extra
            if isinstance(extra, dict):
                conflicting_keys = set(extra.keys()) & set(_FIELD_CONFIG_KEYS)
                if conflicting_keys:
                    raise ValueError(
                        f"Campo '{field_name}' no grupo '{group_name}' tem json_schema_extra "
                        f"com chaves de busca {conflicting_keys}. Escolha entre configuração "
                        f"per-field (json_schema_extra) ou grupo (search_groups), não ambos."
                    )

        _validate_search_overrides(
            f"Grupo '{group_name}'",
            group_config.get('search_depth'),
            group_config.get('max_results'),
            group_config.get('max_search_calls'),
        )

        # Criar SearchGroupConfig
        validated_groups[group_name] = SearchGroupConfig(
            fields=fields,
            prompt=group_config.get('prompt'),
            max_results=group_config.get('max_results'),
            search_depth=group_config.get('search_depth'),
            max_search_calls=group_config.get('max_search_calls'),
        )

    return validated_groups


def _has_field_config(pydantic_model) -> bool:
    """Verifica se algum campo, inclusive aninhado, tem configuração per-field."""
    return bool(_collect_configured_fields(pydantic_model))


def dataframeit(
    data,
    questions=None,
    prompt=None,
    perguntas=None,  # Deprecated: use 'questions'
    resume=True,
    reprocess_columns=None,
    model=None,
    provider='openai',
    status_column=None,
    text_column: str | None = None,
    api_key=None,
    max_retries=3,
    base_delay=1.0,
    max_delay=30.0,
    rate_limit_delay=0.0,
    track_tokens=True,
    model_kwargs=None,
    parallel_requests=1,
    # Parâmetros de busca web
    use_search=False,
    search_provider="tavily",
    search_per_field=False,
    max_results=5,
    search_depth="basic",
    max_search_calls=10,
    search_groups: dict[str, dict] | None = None,
    save_trace: bool | Literal["full", "minimal"] | None = None,
    batch_size: int | None = None,
    checkpoint_path: str | Path | None = None,
) -> Any:
    """Processa textos usando LLMs para extrair informações estruturadas.

    Suporta múltiplos tipos de entrada:
    - pandas.DataFrame: Retorna DataFrame com colunas extraídas
    - polars.DataFrame: Retorna DataFrame polars com colunas extraídas
    - pandas.Series: Retorna DataFrame com resultados indexados
    - polars.Series: Retorna DataFrame polars com resultados
    - list: Retorna lista de dicionários com os resultados
    - dict: Retorna dicionário {chave: {campos extraídos}}

    Args:
        data: Dados contendo textos (DataFrame, Series, list ou dict).
        questions: Modelo Pydantic definindo estrutura a extrair.
        prompt: Template do prompt (use {texto} para indicar onde inserir o texto).
        perguntas: (Deprecated) Use 'questions'.
        resume: Se True, continua de onde parou.
        reprocess_columns: Lista de colunas para forçar reprocessamento. Útil para
            atualizar colunas específicas com novas instruções sem perder outras.
        model: Nome do modelo LLM. Se None, usa o default do provider em
            DEFAULT_MODELS; com 'codex' e 'claude_code', o runtime escolhe.
        provider: Provider do LangChain ('openai', 'google_genai', 'anthropic', etc),
            'claude_code' ou 'codex'. Codex usa o SDK Python oficial.
        status_column: Coluna para rastrear progresso.
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
        rate_limit_delay: Delay em segundos entre requisições para evitar rate limits (padrão: 0.0).
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
            Permite reduzir chamadas de API quando múltiplos campos precisam do mesmo contexto.
            Requer use_search=True e search_per_field=True.
        save_trace: Salva o trace completo do raciocínio do agente. Requer use_search=True.
            - None/False: Desabilitado (padrão)
            - True/"full": Trace completo com conteúdo das mensagens
            - "minimal": Apenas queries e contagens, sem conteúdo de tool results
            Colunas geradas: "_trace" (agente único) ou "_trace_{campo}" (per_field).
        batch_size: Se definido (int >= 1), salva o DataFrame em `checkpoint_path` a cada
            N linhas processadas nesta execução. Combinado com `resume=True`, permite
            retomar execuções longas após kill/crash sem perder progresso.
            Requer `checkpoint_path`.
        checkpoint_path: Caminho do arquivo de checkpoint. Formato inferido pela extensão
            (.csv, .xlsx, .parquet). Escrita atômica via .tmp + rename. Requer `batch_size`.
            Exemplo: ``dataframeit(df, Model, PROMPT, batch_size=100, checkpoint_path="ckpt.xlsx")``.

    Returns:
        Dados com informações extraídas no mesmo formato da entrada.

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
        raise ValueError("Parâmetro 'questions' é obrigatório")

    if prompt is None:
        raise ValueError("Parâmetro 'prompt' é obrigatório")

    # Se {texto} não estiver no template, adiciona automaticamente ao final
    if '{texto}' not in prompt:
        prompt = prompt.rstrip() + "\n\nTexto a analisar:\n{texto}"

    # bool é subclasse de int, mas True não é uma contagem de tentativas.
    if not isinstance(max_retries, numbers.Integral) or isinstance(max_retries, bool) or max_retries < 1:
        raise ValueError(
            f"max_retries deve ser int >= 1 (número total de tentativas por linha); "
            f"recebido {max_retries!r}"
        )

    # Validar parâmetros de checkpoint
    if (batch_size is None) != (checkpoint_path is None):
        raise ValueError("batch_size e checkpoint_path devem ser usados juntos")
    if batch_size is not None:
        if not isinstance(batch_size, numbers.Integral) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError(f"batch_size deve ser int >= 1; recebido {batch_size!r}")
        _validate_checkpoint_extension(checkpoint_path)

    # Providers de SDK usam structured output direto, sem o agente LangChain de busca.
    if use_search and provider in {'claude_code', 'codex'}:
        raise ValueError(
            f"Busca web (use_search=True) não é suportada com provider='{provider}'. "
            "Use um provider LangChain como 'google_genai' ou 'openai' para busca web."
        )

    # Validar parâmetros de busca
    if use_search:
        available_providers = get_available_providers()
        if search_provider not in available_providers:
            raise ValueError(f"search_provider deve ser um de {available_providers}")
        if search_provider == "tavily" and search_depth not in ("basic", "advanced"):
            raise ValueError("search_depth deve ser 'basic' ou 'advanced'")
        if not 1 <= max_results <= 20:
            raise ValueError("max_results deve estar entre 1 e 20")
        if not _is_positive_int(max_search_calls):
            raise ValueError(f"max_search_calls deve ser int >= 1; recebido {max_search_calls!r}")

    # Validar e normalizar save_trace
    trace_mode = None
    if save_trace:
        if not use_search:
            raise ValueError("save_trace requer use_search=True")
        if save_trace is True:
            trace_mode = "full"
        elif save_trace in ("full", "minimal"):
            trace_mode = save_trace
        else:
            raise ValueError("save_trace deve ser True, 'full' ou 'minimal'")

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
                raise ValueError(
                    f"Nenhuma coluna de texto identificada entre {TEXT_COLUMN_CANDIDATES}. "
                    f"Colunas disponíveis: {list(df_pandas.columns)}. "
                    f"Passe text_column= explicitamente."
                )
        if text_column not in df_pandas.columns:
            raise ValueError(
                f"Coluna '{text_column}' não encontrada no DataFrame. "
                f"Colunas disponíveis: {list(df_pandas.columns)}."
            )
    else:
        # Para Series/list/dict, usa coluna interna
        text_column = DEFAULT_TEXT_COLUMN

    # Extrair campos do modelo Pydantic
    expected_columns = list(questions.model_fields.keys())
    if not expected_columns:
        raise ValueError("Modelo Pydantic não pode estar vazio")

    # Cada campo extraído é gravado numa coluna de mesmo nome, que não pode ser
    # a do texto de entrada.
    if text_column in expected_columns:
        raise ValueError(
            f"O campo '{text_column}' do modelo tem o nome da coluna de texto, e a "
            "resposta sobrescreveria o texto de entrada. Renomeie o campo ou a coluna."
        )

    # Validar e processar search_groups
    if search_groups:
        validated_groups = _validate_search_groups(
            search_groups, questions, use_search, search_per_field
        )
        # Adicionar grupos ao search_config
        search_config.groups = validated_groups

    # Validar reprocess_columns
    if reprocess_columns is not None:
        if not isinstance(reprocess_columns, (list, tuple)):
            reprocess_columns = [reprocess_columns]
        # Verificar que todas as colunas a reprocessar estão no modelo
        invalid_cols = [col for col in reprocess_columns if col not in expected_columns]
        if invalid_cols:
            raise ValueError(
                f"Colunas {invalid_cols} não estão no modelo Pydantic. "
                f"Colunas disponíveis: {expected_columns}"
            )

    status_col = status_column or '_dataframeit_status'
    complex_fields = get_complex_fields(questions)

    # Entradas vazias têm um resultado bem definido e não dependem de provider.
    if df_pandas.empty:
        _setup_columns(
            df_pandas,
            expected_columns,
            status_column,
            track_tokens,
            search_config,
            trace_mode,
            questions,
        )
        return from_pandas(df_pandas, conversion_info, status_col)

    # Verificar conflitos de colunas
    existing_cols = [col for col in expected_columns if col in df_pandas.columns]
    if existing_cols and not resume and not reprocess_columns:
        warnings.warn(
            f"Colunas {existing_cols} já existem. Use resume=True para continuar ou renomeie-as."
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
        column
        for column in incompatible_columns
        if column not in reprocessed_columns
    ]
    if uncovered_columns:
        raise ValueError(
            "O DataFrame contém linhas processadas incompatíveis com o modelo atual: "
            f"campos incompatíveis {uncovered_columns}. "
            f"Inclua-os em reprocess_columns={incompatible_columns!r}."
        )

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
            track_tokens,
            search_config,
            trace_mode,
            questions,
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
        raise ValueError(
            f"O índice tem rótulos repetidos ({duplicated[:5]}). "
            "Use df.reset_index(drop=True) antes de chamar dataframeit."
        )

    if model is None and provider not in _RUNTIME_DEFAULT_PROVIDERS:
        if provider not in DEFAULT_MODELS:
            raise ValueError(
                f"provider='{provider}' não tem modelo padrão em DEFAULT_MODELS. "
                f"Confira o nome do provider ou informe 'model'. "
                f"Providers com modelo padrão: {', '.join(DEFAULT_MODELS)}."
            )
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

    _validate_field_configs(questions, config.search_config, use_search, search_per_field)

    # Só execuções com trabalho pendente validam dependências e rate limits.
    if use_search:
        validate_search_dependencies(search_provider)
        is_risky = parallel_requests > 1 or (
            search_per_field and len(expected_columns) * len(df_pandas) > 100
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
            track_tokens,
            search_config,
            trace_mode,
            questions,
        )
        control_columns = [status_col, '_error_details']
        _as_object_columns(df_pandas, expected_columns + control_columns)
        _apply_processed_values(df_pandas, processed_values)

        # Normalizar colunas complexas (listas, dicts, tuples) que podem ter sido
        # serializadas como strings JSON ao salvar/carregar de arquivos.
        if complex_fields and resume:
            normalize_complex_columns(df_pandas, complex_fields)

        # De novo depois da normalização, cujo apply volta a inferir float; sem
        # isso, gravar lista ou texto falharia depois da chamada paga.
        _as_object_columns(df_pandas, expected_columns)

        is_pending, processed_count = _get_processing_indices(
            df_pandas, status_col, resume
        )
        _warn_missing_texts(
            df_pandas,
            text_column,
            [idx for idx, pending in zip(df_pandas.index, is_pending) if pending or reprocess_columns],
        )

        if parallel_requests > 1:
            token_stats = _process_rows_parallel(
                df_pandas,
                text_column,
                status_col,
                expected_columns,
                config,
                backend,
                is_pending,
                processed_count,
                conversion_info,
                track_tokens,
                reprocess_columns,
                parallel_requests,
                trace_mode,
                batch_size,
                checkpoint_path,
            )
        else:
            token_stats = _process_rows(
                df_pandas,
                text_column,
                status_col,
                expected_columns,
                config,
                backend,
                is_pending,
                processed_count,
                conversion_info,
                track_tokens,
                reprocess_columns,
                trace_mode,
                batch_size,
                checkpoint_path,
            )

    # Exibir estatísticas de tokens e throughput
    if track_tokens and token_stats and any(token_stats.values()):
        _print_token_stats(
            token_stats, model, parallel_requests,
            search_provider=search_provider if use_search else None,
        )

    # Aviso de workers reduzidos (aparece SEMPRE, independente de track_tokens)
    if token_stats.get('workers_reduced'):
        print("\n" + "=" * 60)
        print("AVISO: WORKERS REDUZIDOS POR RATE LIMIT")
        print("=" * 60)
        print(f"Workers iniciais: {token_stats['initial_workers']}")
        print(f"Workers finais:   {token_stats['final_workers']}")
        print(f"\nDica: Considere usar parallel_requests={token_stats['final_workers']} "
              f"para evitar rate limits.")
        print("=" * 60 + "\n")

    # Retornar no formato original (remove colunas de status/erro se não houver erros)
    return from_pandas(df_pandas, conversion_info, status_col)


def _setup_columns(
    df: pd.DataFrame,
    expected_columns: list,
    status_column: str | None,
    track_tokens: bool,
    search_config: SearchConfig | None = None,
    trace_mode: str | None = None,
    pydantic_model=None,
):
    """Configura colunas necessárias no DataFrame (in-place)."""
    status_col = status_column or '_dataframeit_status'
    error_col = '_error_details'
    token_cols = TOKEN_COLUMNS if track_tokens else ()
    search_cols = ['_search_credits'] if (search_config and search_config.enabled) else []

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
                for group_name in search_config.groups.keys():
                    trace_cols.append(f'_trace_{group_name}')

                # Adicionar colunas de trace para campos isolados (não em grupos)
                for field in pydantic_model.model_fields.keys():
                    if field not in grouped_fields:
                        trace_cols.append(f'_trace_{field}')
            else:
                # Sem grupos: uma coluna por campo
                trace_cols = [f'_trace_{field}' for field in pydantic_model.model_fields.keys()]
        else:
            # Coluna única
            trace_cols = ['_trace']

    # Identificar colunas que precisam ser criadas
    new_cols = [col for col in expected_columns if col not in df.columns]
    needs_status = status_col not in df.columns
    needs_error = error_col not in df.columns
    needs_tokens = [col for col in token_cols if col not in df.columns] if track_tokens else []
    needs_search = [col for col in search_cols if col not in df.columns]
    needs_trace = [col for col in trace_cols if col not in df.columns]

    if not new_cols and not needs_status and not needs_error and not needs_tokens and not needs_search and not needs_trace:
        return

    # Criar colunas
    with pd.option_context('mode.chained_assignment', None):
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


def _get_processing_indices(df: pd.DataFrame, status_col: str, resume: bool) -> tuple[list[bool], int]:
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
        return status.ne('processed').tolist(), 0

    is_pending = status.isnull()
    processed_count = int((~is_pending).sum())
    return is_pending.tolist(), processed_count


def _print_token_stats(
    token_stats: dict,
    model: str | None,
    parallel_requests: int = 1,
    search_provider: str | None = None,
):
    """Exibe estatísticas de uso de tokens e throughput.

    Args:
        token_stats: Dict com contadores de tokens e métricas de tempo.
        model: Nome do modelo usado; None quando o runtime do provider escolhe.
        parallel_requests: Número de workers paralelos usados.
        search_provider: Provedor de busca usado, que dá nome à seção de busca.
    """
    if not token_stats or (
        token_stats.get('total_tokens', 0) == 0 and not token_stats.get('cost_usd')
    ):
        return

    print("\n" + "=" * 60)
    print("ESTATISTICAS DE USO")
    print("=" * 60)
    print(f"Modelo: {model or 'escolhido pelo runtime do provider'}")
    print(f"Total de tokens: {token_stats['total_tokens']:,}")
    print(f"  - Input:  {token_stats['input_tokens']:,} tokens")
    if token_stats.get('cached_input_tokens', 0) > 0:
        print(f"    └─ Cache: {token_stats['cached_input_tokens']:,} (incluído no Input)")
    print(f"  - Output: {token_stats['output_tokens']:,} tokens")
    if token_stats.get('reasoning_tokens', 0) > 0:
        print(f"    └─ Reasoning: {token_stats['reasoning_tokens']:,} (incluído no Output)")
    # Só providers que informam o custo, como o claude_code, preenchem este total,
    # que inclui as tentativas re-tentadas e as linhas que falharam.
    if token_stats.get('cost_usd', 0) > 0:
        print(f"Custo informado pelo provider: US$ {token_stats['cost_usd']:.4f}")

    # Métricas de throughput (se disponíveis)
    if 'elapsed_seconds' in token_stats and token_stats['elapsed_seconds'] > 0:
        elapsed = token_stats['elapsed_seconds']
        requests = token_stats.get('requests_completed', 0)

        print("-" * 60)
        print("METRICAS DE THROUGHPUT")
        print("-" * 60)
        print(f"Tempo total: {elapsed:.1f}s")
        print(f"Workers paralelos: {parallel_requests}")

        if requests > 0:
            rpm = (requests / elapsed) * 60
            print(f"Requisicoes: {requests}")
            print(f"  - RPM (req/min): {rpm:.1f}")

        tpm = (token_stats['total_tokens'] / elapsed) * 60
        print(f"  - TPM (tokens/min): {tpm:,.0f}")

    # Métricas de busca (se houver)
    if token_stats.get('search_count', 0) > 0:
        print("-" * 60)
        print(f"METRICAS DE BUSCA ({(search_provider or 'tavily').upper()})")
        print("-" * 60)
        print(f"Total de buscas: {token_stats['search_count']}")
        print(f"Creditos usados: {token_stats['search_credits']}")

    print("=" * 60 + "\n")


_SUPPORTED_CHECKPOINT_EXTS = ('.csv', '.xlsx', '.parquet')

# Extensões que exigem dependência opcional para pandas serializar.
# Validamos antes do loop para falhar rápido — um ModuleNotFoundError no primeiro
# save (após N linhas de LLM) desperdiça horas de trabalho.
_CHECKPOINT_EXT_REQUIRES = {
    '.xlsx': ('openpyxl', 'openpyxl'),
    '.parquet': ('pyarrow', 'pyarrow'),
}


def _validate_checkpoint_extension(path: str | Path) -> None:
    """Valida extensão suportada e dependência opcional necessária."""
    import importlib.util

    ext = Path(path).suffix.lower()
    if ext not in _SUPPORTED_CHECKPOINT_EXTS:
        raise ValueError(
            f"Extensão {ext or '(nenhuma)'} não suportada para checkpoint. "
            f"Use uma de: {', '.join(_SUPPORTED_CHECKPOINT_EXTS)}"
        )
    requires = _CHECKPOINT_EXT_REQUIRES.get(ext)
    if requires is not None:
        module, pip_name = requires
        if importlib.util.find_spec(module) is None:
            raise ImportError(
                f"Checkpoint {ext} requer o pacote '{module}', que não está instalado. "
                f"Execute: pip install {pip_name}"
            )


def _structures_as_json(df: pd.DataFrame) -> pd.DataFrame:
    """Cópia com listas, dicts e tuplas serializados como JSON.

    CSV e XLSX gravariam o repr Python ("['a', 'b']"), que json.loads não lê
    de volta. JSON é o que read_df e a retomada normalizam.
    """
    def to_json(value):
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
    tmp = path.with_name(path.name + '.tmp')
    if ext == '.csv':
        _structures_as_json(df).to_csv(tmp, index=False)
    elif ext == '.xlsx':
        _structures_as_json(df).to_excel(tmp, index=False)
    elif ext == '.parquet':
        df.to_parquet(tmp, index=False)
    else:
        raise ValueError(
            f"Extensão {ext} não suportada para checkpoint. "
            f"Use uma de: {', '.join(_SUPPORTED_CHECKPOINT_EXTS)}"
        )
    os.replace(tmp, path)


def _try_save_checkpoint(df: pd.DataFrame, path: str | Path) -> bool:
    """Grava o checkpoint e devolve se deu certo; falha vira aviso.

    Uma falha de gravação (disco cheio, arquivo aberto no Excel, coluna que o
    parquet não serializa) não diz nada sobre a linha que acabou de ser
    processada, e por isso nunca muda o status dela nem interrompe a execução.
    A próxima gravação tenta de novo, com o estado completo.
    """
    try:
        _save_checkpoint(df, path)
    except Exception as error:
        warnings.warn(
            f"Falha ao gravar o checkpoint em {path}: {type(error).__name__}: {error}. "
            "O processamento continua, e a próxima gravação tenta de novo.",
            # usuário -> dataframeit -> _process_rows -> aqui; no modo paralelo
            # o aviso sai de uma thread do executor e não tem quadro do usuário.
            stacklevel=4,
        )
        return False
    return True


def _process_rows(
    df: pd.DataFrame,
    text_column: str,
    status_col: str,
    expected_columns: list,
    config: LLMConfig,
    backend: ProviderBackend,
    is_pending: list[bool],
    processed_count: int,
    conversion_info,
    track_tokens: bool,
    reprocess_columns=None,
    trace_mode: str | None = None,
    batch_size: int | None = None,
    checkpoint_path: str | Path | None = None,
) -> dict:
    """Processa cada linha do DataFrame.

    Args:
        reprocess_columns: Lista de colunas para forçar reprocessamento.
            Se especificado, não pula linhas já processadas.
        trace_mode: Modo de trace ("full", "minimal") ou None para desabilitar.

    Returns:
        Dict com estatísticas de tokens: {'input_tokens', 'output_tokens', 'total_tokens'}
    """
    # Criar descrição para progresso
    type_labels = {
        ORIGINAL_TYPE_POLARS_DF: 'polars→pandas',
        ORIGINAL_TYPE_PANDAS_DF: 'pandas',
    }
    engine = type_labels.get(conversion_info.original_type, conversion_info.original_type)
    search_mode = '+search' if (config.search_config and config.search_config.enabled) else ''
    desc = f"Processando [{engine}+{backend.label}{search_mode}]"

    # Adicionar info de rate limiting (se ativo)
    if config.rate_limit_delay > 0:
        req_per_min = int(60 / config.rate_limit_delay)
        desc += f" [~{req_per_min} req/min]"

    if reprocess_columns:
        desc += f" (reprocessando: {', '.join(reprocess_columns)})"
    elif processed_count > 0:
        desc += f" (resumindo de {processed_count}/{len(df)})"

    # Inicializar contadores de tokens e busca
    token_stats = {
        'input_tokens': 0,
        'cached_input_tokens': 0,
        'output_tokens': 0,
        'total_tokens': 0,
        'reasoning_tokens': 0,
        'search_credits': 0,
        'search_count': 0,
        'cost_usd': 0.0,
    }

    rows_processed_this_run = 0
    rows_saved = 0

    # Processar cada linha
    for i, (idx, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc=desc)):
        # Verificar se linha já foi processada
        row_already_processed = pd.notna(row[status_col]) and row[status_col] == 'processed'

        # Com reprocess_columns, todas as linhas são processadas.
        if not reprocess_columns and not is_pending[i]:
            continue

        if _is_missing_text(row[text_column]):
            df.at[idx, status_col] = 'error'
            df.at[idx, '_error_details'] = _missing_text_detail(
                row_already_processed, reprocess_columns
            )
            rows_processed_this_run += 1
            if batch_size and rows_processed_this_run % batch_size == 0:
                if _try_save_checkpoint(df, checkpoint_path):
                    rows_saved = rows_processed_this_run
            continue

        text = str(row[text_column])

        try:
            result = _invoke_row(
                backend, text, row, row_already_processed, reprocess_columns, expected_columns
            )

            # Extrair dados e usage metadata
            extracted = result['data']
            usage = result.get('usage')
            retry_info = result.get('_retry_info', {})

            # Atualizar DataFrame com dados extraídos
            # Se linha já processada e reprocess_columns definido: só atualiza colunas especificadas
            # Caso contrário: atualiza todas as colunas do modelo
            for col in expected_columns:
                if col in extracted:
                    if row_already_processed and reprocess_columns:
                        # Linha já processada: só atualiza se col está em reprocess_columns
                        if col in reprocess_columns:
                            df.at[idx, col] = extracted[col]
                    else:
                        # Linha nova: atualiza tudo
                        df.at[idx, col] = extracted[col]

            # Armazenar tokens no DataFrame (se habilitado)
            if track_tokens and usage:
                df.at[idx, '_input_tokens'] = usage.get('input_tokens', 0)
                df.at[idx, '_cached_input_tokens'] = usage.get('cached_input_tokens', 0)
                df.at[idx, '_output_tokens'] = usage.get('output_tokens', 0)
                df.at[idx, '_reasoning_tokens'] = usage.get('reasoning_tokens', 0)

                # Acumular estatísticas (total exibido apenas no summary do console)
                token_stats['input_tokens'] += usage.get('input_tokens', 0)
                token_stats['cached_input_tokens'] += usage.get('cached_input_tokens', 0)
                token_stats['output_tokens'] += usage.get('output_tokens', 0)
                token_stats['total_tokens'] += usage.get('total_tokens', 0)
                token_stats['reasoning_tokens'] += usage.get('reasoning_tokens', 0)
                token_stats['cost_usd'] += usage.get('cost_usd') or 0

            # Armazenar métricas de busca (se habilitado)
            if config.search_config and config.search_config.enabled and usage:
                df.at[idx, '_search_credits'] = usage.get('search_credits', 0)

                # Acumular estatísticas de busca (search_count mantido só para o summary)
                token_stats['search_credits'] += usage.get('search_credits', 0)
                token_stats['search_count'] += usage.get('search_count', 0)

            # Armazenar traces (se habilitado)
            if trace_mode:
                if config.search_config and config.search_config.per_field:
                    # Traces por campo
                    traces = result.get('traces', {})
                    for field_name, trace in traces.items():
                        df.at[idx, f'_trace_{field_name}'] = json.dumps(trace, ensure_ascii=False)
                else:
                    # Trace único
                    trace = result.get('trace')
                    if trace:
                        df.at[idx, '_trace'] = json.dumps(trace, ensure_ascii=False)

            df.at[idx, status_col] = 'processed'
            # Registra retries mesmo no sucesso; sem retry, some o erro de uma execução anterior
            df.at[idx, '_error_details'] = _success_details(retry_info)

            rows_processed_this_run += 1
            if batch_size and rows_processed_this_run % batch_size == 0:
                if _try_save_checkpoint(df, checkpoint_path):
                    rows_saved = rows_processed_this_run

            if config.rate_limit_delay > 0:
                time.sleep(config.rate_limit_delay)

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            token_stats['cost_usd'] += getattr(e, 'cost_usd', 0) or 0

            # Determinar se foi erro recuperável ou não para mensagem correta
            if is_recoverable_error(e):
                # Erro recuperável que esgotou tentativas
                error_details = f"[Falhou após {config.max_retries} tentativa(s)] {error_msg}"
            else:
                # Erro não-recuperável (não fez retry)
                error_details = f"[Erro não-recuperável] {error_msg}"
            if row_already_processed and reprocess_columns:
                error_details += _KEPT_VALUES_NOTE

            # Exibir mensagem amigável para o usuário
            friendly_msg = get_friendly_error_message(e, config.provider)
            print(f"\n{friendly_msg}\n")

            warnings.warn(f"Falha ao processar linha {idx}.")
            df.at[idx, status_col] = 'error'
            df.at[idx, '_error_details'] = error_details

            rows_processed_this_run += 1
            if batch_size and rows_processed_this_run % batch_size == 0:
                if _try_save_checkpoint(df, checkpoint_path):
                    rows_saved = rows_processed_this_run

    # Save final: a cauda (< batch_size) e o que uma gravação que falhou deixou de fora.
    if batch_size and rows_processed_this_run > rows_saved:
        _try_save_checkpoint(df, checkpoint_path)

    return token_stats


def _process_rows_parallel(
    df: pd.DataFrame,
    text_column: str,
    status_col: str,
    expected_columns: list,
    config: LLMConfig,
    backend: ProviderBackend,
    is_pending: list[bool],
    processed_count: int,
    conversion_info,
    track_tokens: bool,
    reprocess_columns,
    parallel_requests: int,
    trace_mode: str | None = None,
    batch_size: int | None = None,
    checkpoint_path: str | Path | None = None,
) -> dict:
    """Processa linhas do DataFrame em paralelo com auto-redução de workers.

    Args:
        parallel_requests: Número inicial de workers paralelos.
            Será reduzido automaticamente se detectar erros de rate limit (429).
        trace_mode: Modo de trace ("full", "minimal") ou None para desabilitar.

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

    def _save_snapshot(snapshot: pd.DataFrame, label: int) -> None:
        nonlocal last_saved_checkpoint
        with checkpoint_write_lock:
            if label <= last_saved_checkpoint:
                return
            if _try_save_checkpoint(snapshot, checkpoint_path):
                last_saved_checkpoint = label

    # Contadores
    token_stats = {
        'input_tokens': 0,
        'cached_input_tokens': 0,
        'output_tokens': 0,
        'total_tokens': 0,
        'reasoning_tokens': 0,
        'requests_completed': 0,
        'search_credits': 0,
        'search_count': 0,
        'cost_usd': 0.0,
    }

    # Criar descrição para progresso
    type_labels = {
        ORIGINAL_TYPE_POLARS_DF: 'polars→pandas',
        ORIGINAL_TYPE_PANDAS_DF: 'pandas',
    }
    engine = type_labels.get(conversion_info.original_type, conversion_info.original_type)
    search_mode = '+search' if (config.search_config and config.search_config.enabled) else ''
    desc = (
        f"Processando [{engine}+{backend.label}{search_mode}] "
        f"[{parallel_requests} workers]"
    )

    if reprocess_columns:
        desc += f" (reprocessando: {', '.join(reprocess_columns)})"
    elif processed_count > 0:
        desc += f" (resumindo de {processed_count}/{len(df)})"

    # Identificar linhas a processar
    rows_to_process = []
    for i, (idx, row) in enumerate(df.iterrows()):
        if reprocess_columns or is_pending[i]:
            rows_to_process.append((i, idx, row))

    if not rows_to_process:
        return token_stats

    def process_single_row(row_data):
        """Processa uma única linha (executada em thread separada)."""
        nonlocal current_workers, workers_reduced, checkpoint_counter

        i, idx, row = row_data
        row_already_processed = pd.notna(row[status_col]) and row[status_col] == 'processed'
        if _is_missing_text(row[text_column]):
            snapshot = None
            with lock:
                df.at[idx, status_col] = 'error'
                df.at[idx, '_error_details'] = _missing_text_detail(
                    row_already_processed, reprocess_columns
                )
                checkpoint_counter += 1
                if batch_size and checkpoint_counter % batch_size == 0:
                    snapshot = (df.copy(), checkpoint_counter)
            if snapshot is not None:
                _save_snapshot(*snapshot)
            return {'success': False, 'idx': idx, 'error': _MISSING_TEXT_DETAIL}

        text = str(row[text_column])

        # Verificar se devemos pausar devido a rate limit
        if rate_limit_event.is_set():
            time.sleep(2.0)  # Pausa breve quando rate limit detectado

        try:
            result = _invoke_row(
                backend, text, row, row_already_processed, reprocess_columns, expected_columns
            )

            # Extrair dados
            extracted = result['data']
            usage = result.get('usage')
            retry_info = result.get('_retry_info', {})

            # Atualizar DataFrame (com lock para thread-safety)
            snapshot = None
            with lock:
                for col in expected_columns:
                    if col in extracted:
                        if row_already_processed and reprocess_columns:
                            if col in reprocess_columns:
                                df.at[idx, col] = extracted[col]
                        else:
                            df.at[idx, col] = extracted[col]

                if track_tokens and usage:
                    df.at[idx, '_input_tokens'] = usage.get('input_tokens', 0)
                    df.at[idx, '_cached_input_tokens'] = usage.get('cached_input_tokens', 0)
                    df.at[idx, '_output_tokens'] = usage.get('output_tokens', 0)
                    df.at[idx, '_reasoning_tokens'] = usage.get('reasoning_tokens', 0)

                    token_stats['input_tokens'] += usage.get('input_tokens', 0)
                    token_stats['cached_input_tokens'] += usage.get('cached_input_tokens', 0)
                    token_stats['output_tokens'] += usage.get('output_tokens', 0)
                    token_stats['total_tokens'] += usage.get('total_tokens', 0)
                    token_stats['reasoning_tokens'] += usage.get('reasoning_tokens', 0)
                    token_stats['cost_usd'] += usage.get('cost_usd') or 0

                if config.search_config and config.search_config.enabled and usage:
                    df.at[idx, '_search_credits'] = usage.get('search_credits', 0)

                    token_stats['search_credits'] += usage.get('search_credits', 0)
                    token_stats['search_count'] += usage.get('search_count', 0)

                if trace_mode:
                    if config.search_config and config.search_config.per_field:
                        traces = result.get('traces', {})
                        for field_name, trace in traces.items():
                            df.at[idx, f'_trace_{field_name}'] = json.dumps(trace, ensure_ascii=False)
                    else:
                        trace = result.get('trace')
                        if trace:
                            df.at[idx, '_trace'] = json.dumps(trace, ensure_ascii=False)

                token_stats['requests_completed'] += 1
                df.at[idx, status_col] = 'processed'
                df.at[idx, '_error_details'] = _success_details(retry_info)

                checkpoint_counter += 1
                if batch_size and checkpoint_counter % batch_size == 0:
                    # Copia sob lock, serializa fora — evita bloquear threads na I/O.
                    snapshot = (df.copy(), checkpoint_counter)

            if snapshot is not None:
                _save_snapshot(*snapshot)

            if config.rate_limit_delay > 0:
                time.sleep(config.rate_limit_delay)

            return {'success': True, 'idx': idx}

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"

            # Verificar se é erro de rate limit
            if is_rate_limit_error(e):
                with lock:
                    if current_workers > 1:
                        old_workers = current_workers
                        current_workers = max(1, current_workers // 2)
                        workers_reduced = True
                        warnings.warn(
                            f"Rate limit detectado! Reduzindo workers de {old_workers} para {current_workers}.",
                            stacklevel=2
                        )
                        rate_limit_event.set()
                        # Limpar evento após um tempo
                        threading.Timer(5.0, rate_limit_event.clear).start()

            snapshot = None
            with lock:
                token_stats['cost_usd'] += getattr(e, 'cost_usd', 0) or 0
                if is_recoverable_error(e):
                    error_details = f"[Falhou após {config.max_retries} tentativa(s)] {error_msg}"
                else:
                    error_details = f"[Erro não-recuperável] {error_msg}"
                if row_already_processed and reprocess_columns:
                    error_details += _KEPT_VALUES_NOTE

                friendly_msg = get_friendly_error_message(e, config.provider)
                print(f"\n{friendly_msg}\n")

                warnings.warn(f"Falha ao processar linha {idx}.")
                df.at[idx, status_col] = 'error'
                df.at[idx, '_error_details'] = error_details

                checkpoint_counter += 1
                if batch_size and checkpoint_counter % batch_size == 0:
                    snapshot = (df.copy(), checkpoint_counter)

            if snapshot is not None:
                _save_snapshot(*snapshot)

            return {'success': False, 'idx': idx, 'error': error_msg}

    # Processar com ThreadPoolExecutor
    with tqdm(total=len(rows_to_process), desc=desc) as pbar:
        # Usar abordagem iterativa para permitir ajuste dinâmico de workers
        pending_rows = list(rows_to_process)
        completed = 0

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
                        pbar.update(1)
                        completed += 1
                    except Exception as e:
                        pbar.update(1)
                        completed += 1
                        warnings.warn(f"Erro inesperado no executor: {e}")

    # Save final: a cauda (< batch_size) e o que uma gravação que falhou deixou de fora.
    if batch_size and checkpoint_counter > last_saved_checkpoint:
        _try_save_checkpoint(df, checkpoint_path)

    elapsed = time.time() - start_time
    token_stats['elapsed_seconds'] = elapsed
    token_stats['initial_workers'] = initial_workers
    token_stats['final_workers'] = current_workers
    token_stats['workers_reduced'] = workers_reduced

    return token_stats
