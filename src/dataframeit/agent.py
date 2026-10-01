"""Processamento baseado em agente com busca web.

Suporta múltiplos provedores de busca:
- Tavily: Motor de busca otimizado para IA
- Exa: Motor de busca semântico
"""

from __future__ import annotations

import logging
import time
from copy import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, create_model

from .conditional import (
    _CONDITIONAL_KEYS,
    _FIELD_CONFIG_KEYS,
    _collect_configured_fields,
    get_field_execution_order,
    get_group_execution_units,
    should_skip_field,
    topological_sort,
)
from .errors import retry_with_backoff
from .llm import (
    LLMConfig,
    SearchConfig,
    SearchGroupConfig,
    _BuildOnce,
    _parse_usage_metadata,
    build_prompt,
    chat_model,
)
from .search import get_provider
from .utils import is_list_of_pydantic_model, resolve_forward_refs

if TYPE_CHECKING:
    from langchain_core.runnables import Runnable
    from pydantic.fields import FieldInfo

    from .search import SearchProvider

logger = logging.getLogger(__name__)

_USAGE_COUNTERS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "total_tokens",
    "reasoning_tokens",
    "search_credits",
    "search_count",
)


def _empty_usage(**metadata: object) -> dict:
    return {**dict.fromkeys(_USAGE_COUNTERS, 0), **metadata}


_LIBRARY_EXTRA_KEYS = frozenset(_FIELD_CONFIG_KEYS + _CONDITIONAL_KEYS)

# Campos simples do item que entram no contexto da busca por item.
_MAX_CONTEXT_FIELDS = 3


def _search_config(config: LLMConfig) -> SearchConfig:
    """SearchConfig da execução com busca.

    O core só chama os modos de busca com use_search=True, e é nesse caso que
    ele monta a SearchConfig; o cast apenas registra essa garantia para o ty.
    """
    return cast("SearchConfig", config.search_config)


def _query_model(name: str, fields: dict[str, tuple]) -> type[BaseModel]:
    """Modelo Pydantic montado na hora com os campos pedidos numa chamada."""
    # Os campos chegam por **kwargs, e o ty não casa esse dict com as
    # sobrecargas de create_model.
    return create_model(name, **fields)  # ty: ignore[no-matching-overload]


def _llm_field(field_info: FieldInfo) -> FieldInfo:
    """Cópia do FieldInfo sem as chaves de configuração da biblioteca.

    `condition`, `depends_on` e as chaves de busca por campo servem ao
    dataframeit e não descrevem o campo para o LLM. Elas iriam no JSON Schema
    do modelo montado para a chamada, e uma `condition` callable nem
    serializa. O FieldInfo original fica intacto, porque é do modelo do
    usuário e é relido a cada linha. Um json_schema_extra callable é do
    usuário e não carrega essas chaves, então passa como está.

    Só o campo extraído é limpo. Num modelo aninhado, as chaves de busca dos
    campos internos seguem no $defs do schema como metadado; `condition` e
    `depends_on` ali são recusados antes do processamento.
    """
    extra = field_info.json_schema_extra
    if not isinstance(extra, dict) or not (_LIBRARY_EXTRA_KEYS & extra.keys()):
        return field_info

    cleaned = copy(field_info)
    cleaned.json_schema_extra = {
        key: value for key, value in extra.items() if key not in _LIBRARY_EXTRA_KEYS
    } or None
    # copy() compartilha _attributes_set, de onde o Pydantic remonta o campo
    # ao mesclar FieldInfo; sem ajustar, a condition voltaria por ali.
    attributes = dict(getattr(field_info, "_attributes_set", {}))
    if cleaned.json_schema_extra is None:
        attributes.pop("json_schema_extra", None)
    else:
        attributes["json_schema_extra"] = cleaned.json_schema_extra
    cleaned._attributes_set = attributes  # noqa: SLF001 (ver o comentário acima)
    return cleaned


def _llm_field_spec(field_info: FieldInfo, owner: type[BaseModel]) -> tuple:
    """Par (anotação, FieldInfo) para create_model de um campo do modelo `owner`.

    A anotação vai com as referências adiantadas resolvidas: fora do modelo
    dono, `list['Item']` não se resolve, e o Pydantic recusa o modelo novo.
    """
    return resolve_forward_refs(field_info.annotation, owner), _llm_field(field_info)


def _get_field_config(extra: dict) -> dict:
    """Extrai configurações relevantes do json_schema_extra.

    Args:
        extra: Dicionário json_schema_extra do campo Pydantic.

    Returns:
        Dicionário com configurações extraídas (prompt, prompt_append,
        search_depth, max_results, depends_on, condition). `depends_on`
        é normalmente derivado automaticamente de `condition` (quando
        dict) — só precisa ser declarado para `condition` callable.
    """
    return {
        "prompt": extra.get("prompt") or extra.get("prompt_replace"),
        "prompt_append": extra.get("prompt_append"),
        "search_depth": extra.get("search_depth"),
        "max_results": extra.get("max_results"),
        "max_search_calls": extra.get("max_search_calls"),
        "depends_on": extra.get("depends_on", []),
        "condition": extra.get("condition"),
    }


def _with_text_placeholder(prompt: str) -> str:
    """Garante que o prompt leve o texto da linha, como o prompt principal."""
    if "{texto}" in prompt:
        return prompt
    return f"{prompt.rstrip()}\n\nTexto a analisar:\n{{texto}}"


def _build_field_prompt(
    user_prompt: str, field_name: str, field_description: str | None, field_config: dict
) -> str:
    """Constrói o prompt para um campo específico.

    Args:
        user_prompt: Template do prompt base do usuário.
        field_name: Nome do campo sendo processado.
        field_description: Descrição do campo (opcional).
        field_config: Configurações extraídas do json_schema_extra.

    Returns:
        Prompt construído para o campo.
    """
    prompt_replace = field_config.get("prompt")
    prompt_append = field_config.get("prompt_append")

    if prompt_replace:
        # Substitui o prompt do usuário, que é quem levava o {texto}
        return _with_text_placeholder(prompt_replace)

    # Base: prompt original + instrução do campo
    base_prompt = f"{user_prompt}\n\nResponda APENAS o campo: {field_name}"
    if field_description:
        base_prompt += f" ({field_description})"

    if prompt_append:
        # Adiciona texto customizado
        base_prompt += f"\n\n{prompt_append}"

    return base_prompt


def _with_search_overrides(
    config: LLMConfig,
    search_depth: str | None = None,
    max_results: int | None = None,
    max_search_calls: int | None = None,
) -> LLMConfig:
    """Cria novo LLMConfig com os overrides de busca de um campo ou grupo.

    Args:
        config: Configuração base do LLM.
        search_depth: Profundidade que substitui a global, ou None.
        max_results: Número de resultados que substitui o global, ou None.
        max_search_calls: Teto de buscas que substitui o global, ou None.

    Returns:
        LLMConfig original se não há overrides, ou cópia com a SearchConfig
        sobrescrita. A config original nunca é mutada, porque é compartilhada
        entre linhas e threads.
    """
    if search_depth is None and max_results is None and max_search_calls is None:
        return config

    new_config = copy(config)
    new_search_config = copy(_search_config(config))

    if search_depth is not None:
        new_search_config.search_depth = search_depth
    if max_results is not None:
        new_search_config.max_results = max_results
    if max_search_calls is not None:
        new_search_config.max_search_calls = max_search_calls

    new_config.search_config = new_search_config
    return new_config


def _with_field_overrides(config: LLMConfig, field_config: dict) -> LLMConfig:
    """Aplica os overrides de busca de um campo, lidos de _get_field_config."""
    return _with_search_overrides(
        config,
        field_config.get("search_depth"),
        field_config.get("max_results"),
        field_config.get("max_search_calls"),
    )


def _get_list_fields_with_nested_search(pydantic_model: type[BaseModel]) -> dict:
    """Identifica campos List[Model] que têm configuração de busca em modelos internos.

    Args:
        pydantic_model: Modelo Pydantic a analisar.

    Returns:
        Dicionário: {field_name: {'inner_model': Model, 'search_fields': [(relative_path, field_name, field_info)]}}
    """
    list_fields_with_search = {}

    for field_name, field_info in pydantic_model.model_fields.items():
        is_list, inner_model = is_list_of_pydantic_model(
            resolve_forward_refs(field_info.annotation, pydantic_model)
        )

        if is_list and inner_model:
            # Coletar campos de busca dentro do modelo interno
            inner_search_fields = _collect_configured_fields(inner_model)

            if inner_search_fields:
                list_fields_with_search[field_name] = {
                    "inner_model": inner_model,
                    "search_fields": inner_search_fields,
                }

    return list_fields_with_search


def _enrich_list_items_with_search(  # noqa: PLR0913, PLR0917 (laço por item e por campo)
    list_items: list,
    inner_model: type[BaseModel],
    search_fields: list,
    text: str,
    config: LLMConfig,
    save_trace: str | None = None,
) -> tuple:
    """Enriquece cada item de uma lista com buscas específicas.

    Args:
        list_items: Lista de dicionários (items extraídos pelo LLM).
        inner_model: Modelo Pydantic do item interno.
        search_fields: Lista de tuplas (path, field_name, field_info, parent_model, has_config).
        text: Texto original sendo processado.
        config: Configuração do LLM.
        save_trace: Modo de trace.

    Returns:
        Tupla (enriched_items, usage, traces).
    """
    enriched_items = []
    total_usage = _empty_usage()
    traces = [] if save_trace else None

    for item_idx, item in enumerate(list_items or []):
        # A resposta do agente passa por model_dump, e cada item de List[Model]
        # chega como dicionário. Outro valor não tem campos onde gravar a busca.
        if not isinstance(item, dict):
            enriched_items.append(item)
            continue
        item_dict = item.copy()

        item_traces = {} if save_trace else None

        # Construir contexto do item para a busca
        item_context = _build_item_context(item_dict, inner_model)

        # Para cada campo de busca no modelo interno
        for path, field_name, field_info, parent_model, _ in search_fields:
            extra = field_info.json_schema_extra
            field_config = _get_field_config(extra) if isinstance(extra, dict) else {}

            # Criar modelo temporário para a busca
            single_field_model = _query_model(
                f"ItemSearch_{item_idx}_{path.replace('.', '_')}",
                {field_name: _llm_field_spec(field_info, parent_model)},
            )

            # Construir prompt para busca do campo com contexto do item
            field_prompt = _build_field_prompt(
                f"Pesquise informações para: {item_context}",
                field_name,
                field_info.description,
                field_config,
            )

            # Criar config com overrides do campo
            effective_config = _with_field_overrides(config, field_config)

            # Chamar agente para buscar informações
            result = call_agent(
                text, single_field_model, field_prompt, effective_config, save_trace
            )

            # Atualizar o item com o resultado da busca
            search_result = result["data"].get(field_name)
            _set_nested_value(item_dict, path, search_result)

            # Somar usage
            if result.get("usage"):
                for key in total_usage:
                    total_usage[key] += result["usage"].get(key, 0)

            # Coletar trace
            if item_traces is not None and result.get("trace"):
                item_traces[path] = result["trace"]

        enriched_items.append(item_dict)

        if traces is not None and item_traces:
            traces.append(item_traces)

    return enriched_items, total_usage, traces


def _build_item_context(item_dict: dict, inner_model: type[BaseModel]) -> str:
    """Constrói uma string de contexto para um item de lista.

    Usa os primeiros campos não-nulos do item para criar contexto.
    """
    context_parts = []

    for field_name in inner_model.model_fields:
        value = item_dict.get(field_name)
        if value is not None and not isinstance(value, (dict, list)):
            # Usar apenas valores simples (strings, números)
            context_parts.append(f"{field_name}: {value}")
            if len(context_parts) >= _MAX_CONTEXT_FIELDS:
                break

    return ", ".join(context_parts) if context_parts else "item"


def _set_nested_value(obj: dict, path: str, value: object) -> None:
    """Define um valor em um caminho aninhado de um dicionário.

    Args:
        obj: Dicionário a modificar.
        path: Caminho no formato "a.b.c".
        value: Valor a definir.
    """
    parts = path.split(".")
    current = obj

    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]

    current[parts[-1]] = value


def _recursion_limit(max_search_calls: int) -> int:
    """Passos do grafo que uma execução do agente pode dar.

    Cada busca custa 3 passos (modelo, middleware do teto, ferramenta), e a
    resposta final, outros 3. Um modelo que insiste em buscar depois do teto
    gasta 2 passos por insistência, e sem este limite chegaria ao do LangGraph,
    com milhares de chamadas ao modelo. A folga de 20 cobre umas oito
    insistências ou novas tentativas do structured output.
    """
    return 3 * max_search_calls + 20


@dataclass(frozen=True)
class SearchAgent:
    """Agente de busca montado para um modelo Pydantic e uma SearchConfig."""

    agent: Runnable
    provider: SearchProvider
    tool_name: str


def build_search_agent(pydantic_model: type[BaseModel], config: LLMConfig) -> SearchAgent:
    """Monta o agente com a ferramenta de busca e o teto de buscas.

    O agente compilado pode ser compartilhado entre linhas e threads: o
    ToolCallLimitMiddleware conta as chamadas no estado de cada invocação, e
    não na instância.
    """
    # Resolvidos na chamada: o import do pacote fica leve e create_agent pode
    # ser trocado em langchain.agents.
    from langchain.agents import create_agent  # noqa: PLC0415
    from langchain.agents.middleware import ToolCallLimitMiddleware  # noqa: PLC0415
    from langchain.agents.structured_output import ToolStrategy  # noqa: PLC0415

    search_config = _search_config(config)

    # Criar ferramenta de busca via factory (suporta Tavily, Exa, etc.)
    provider = get_provider(search_config.provider)
    search_tool = provider.create_tool(
        max_results=search_config.max_results,
        search_depth=search_config.search_depth,
    )

    # Sem teto, um modelo que insiste em buscar gasta créditos até o limite de
    # recursão do grafo. "continue" bloqueia as buscas excedentes e deixa o
    # modelo responder com o que já tem.
    agent = create_agent(
        model=chat_model(config),
        tools=[search_tool],
        response_format=ToolStrategy(pydantic_model),
        middleware=[
            ToolCallLimitMiddleware(
                tool_name=search_tool.name,
                run_limit=search_config.max_search_calls,
                exit_behavior="continue",
            )
        ],
    )
    return SearchAgent(agent=agent, provider=provider, tool_name=search_tool.name)


def call_agent(  # noqa: PLR0913, PLR0917 (assinatura comum aos modos de busca, chamada pelo core)
    text: str,
    pydantic_model: type[BaseModel],
    user_prompt: str,
    config: LLMConfig,
    save_trace: str | None = None,
    search_agent: _BuildOnce | None = None,
) -> dict:
    """Processa texto usando agente LangChain com busca web.

    Args:
        text: Texto a ser processado (ex: nome do medicamento, país, etc.).
        pydantic_model: Modelo Pydantic para estruturar resposta.
        user_prompt: Template do prompt do usuário.
        config: Configuração do LLM incluindo SearchConfig.
        save_trace: Modo de trace ("full", "minimal") ou None para desabilitar.
        search_agent: Agente compartilhado pela execução. Se None, é montado
            nesta chamada, como nos modos por campo e por grupo, em que o
            modelo Pydantic de cada chamada é criado na hora.

    Returns:
        Dicionário com 'data' (dados extraídos), 'usage' (metadata incluindo
        search_credits e search_count) e 'trace' (se save_trace habilitado).
    """
    search_config = _search_config(config)
    if search_agent is not None:
        built = search_agent()
    else:
        built = build_search_agent(pydantic_model, config)
    agent, provider, tool_name = built.agent, built.provider, built.tool_name

    def _call() -> dict:
        prompt = build_prompt(user_prompt, text)

        # Medir tempo de execução
        start_time = time.perf_counter()
        result = agent.invoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config={"recursion_limit": _recursion_limit(search_config.max_search_calls)},
        )
        duration = time.perf_counter() - start_time

        # Extrair resposta estruturada
        structured = result.get("structured_response")
        if structured is None:
            msg = "Agente não retornou resposta estruturada"
            raise ValueError(msg)

        data = structured.model_dump() if hasattr(structured, "model_dump") else structured

        # Calcular usage (tokens + search credits via provider)
        usage = _extract_usage(result, provider, search_config, tool_name)

        response = {"data": data, "usage": usage}

        # Extrair trace se habilitado
        if save_trace:
            response["trace"] = _extract_trace(
                result, config.model, duration, save_trace, provider, tool_name
            )

        return response

    return retry_with_backoff(_call, config.max_retries, config.base_delay, config.max_delay)


def _run_nested_searches(
    text: str, nested_fields: list, config: LLMConfig, save_trace: str | None = None
) -> tuple:
    """Executa buscas para campos configurados em modelos aninhados.

    Args:
        text: Texto a ser processado.
        nested_fields: Lista de tuplas (path, field_name, field_info, parent_model, has_config).
        config: Configuração do LLM.
        save_trace: Modo de trace.

    Returns:
        Tupla (search_context: dict, usage: dict, traces: dict).
        search_context mapeia path -> resultado da busca.
    """
    search_context = {}
    total_usage = _empty_usage()
    traces = {} if save_trace else None

    for path, field_name, field_info, parent_model, _ in nested_fields:
        # Extrair configurações do campo
        extra = field_info.json_schema_extra
        field_config = _get_field_config(extra) if isinstance(extra, dict) else {}

        # Criar modelo temporário para a busca
        single_field_model = _query_model(
            f"NestedSearch_{path.replace('.', '_')}",
            {field_name: _llm_field_spec(field_info, parent_model)},
        )

        # Construir prompt para busca do campo aninhado
        field_prompt = _build_field_prompt(
            f"Pesquise informações para preencher o campo '{path}'",
            field_name,
            field_info.description,
            field_config,
        )

        # Criar config com overrides do campo (se houver)
        effective_config = _with_field_overrides(config, field_config)

        # Chamar agente para buscar informações
        result = call_agent(text, single_field_model, field_prompt, effective_config, save_trace)

        # Armazenar contexto de busca
        search_context[path] = result["data"].get(field_name)

        # Somar usage
        if result.get("usage"):
            for key in total_usage:
                total_usage[key] += result["usage"].get(key, 0)

        # Coletar trace
        if traces is not None and result.get("trace"):
            traces[path] = result["trace"]

    return search_context, total_usage, traces


@dataclass(frozen=True)
class _Unit:
    """Uma chamada ao agente na extração: um grupo de search_groups ou um campo fora de grupo.

    `key` é o nome do grupo ou do campo, e vira a chave do trace da chamada.
    """

    key: str
    fields: tuple[str, ...]
    group: SearchGroupConfig | None


@dataclass
class _Row:
    """O que a extração de uma linha acumula entre as chamadas ao agente."""

    text: str
    only_fields: set | None
    known: dict
    traces: dict | None
    data: dict = field(default_factory=dict)
    usage: dict = field(default_factory=_empty_usage)
    nested_context: dict = field(default_factory=dict)

    def wants(self, field_name: str) -> bool:
        return self.only_fields is None or field_name in self.only_fields

    def add_usage(self, usage: dict) -> None:
        for key in _USAGE_COUNTERS:
            self.usage[key] += usage.get(key, 0)

    def add_trace(self, key: str, trace: object) -> None:
        # traces é None quando save_trace está desligado, e aí nenhuma
        # chamada devolve trace.
        if self.traces is not None and trace:
            self.traces[key] = trace

    def add_result(self, result: dict, key: str) -> None:
        self.add_usage(result.get("usage") or {})
        self.add_trace(key, result.get("trace"))


def _nested_context_note(nested_context: dict, field_names: list[str]) -> str:
    """Resultado das buscas aninhadas dos campos pedidos, para o fim do prompt."""
    lines = [
        f"- {path}: {value}"
        for path, value in nested_context.items()
        if path.split(".")[0] in field_names
    ]
    if not lines:
        return ""
    return "\n\nContexto de buscas realizadas para campos aninhados:\n" + "\n".join(lines)


@dataclass(frozen=True)
class _FieldExtractor:
    """Extração com busca por campo ou por grupo, montada uma vez por execução.

    O que depende só do modelo e da config (ordem de execução, unidades,
    campos aninhados com busca) é calculado na montagem. O estado de cada
    linha fica em _Row, e a instância pode ser compartilhada entre threads.
    """

    pydantic_model: type[BaseModel]
    user_prompt: str
    config: LLMConfig
    save_trace: str | None
    field_configs: dict[str, dict]
    execution_order: list[str]
    dependencies: dict[str, list[str]]
    units: tuple[_Unit, ...]
    list_fields: dict
    nested_fields: tuple

    def __call__(
        self, text: str, only_fields: set | None = None, known: dict | None = None
    ) -> dict:
        row = _Row(text, only_fields, known or {}, traces={} if self.save_trace else None)
        self._search_nested_fields(row)
        for unit in self.units:
            self._run_unit(row, unit)

        response = {
            "data": {name: row.data.get(name) for name in self.pydantic_model.model_fields},
            "usage": {**row.usage, "search_provider": _search_config(self.config).provider},
        }
        if self.save_trace:
            response["traces"] = row.traces
        return response

    def _search_nested_fields(self, row: _Row) -> None:
        """Busca os campos aninhados com configuração própria antes das unidades.

        Campo dentro de List[Model] fica de fora: ele só existe depois que a
        lista é extraída, e é buscado item a item em _enrich_items.
        """
        nested_fields = [path for path in self.nested_fields if row.wants(path[0].split(".")[0])]
        if not nested_fields:
            return
        context, usage, traces = _run_nested_searches(
            row.text, nested_fields, self.config, self.save_trace
        )
        row.nested_context = context
        row.add_usage(usage)
        for path, trace in (traces or {}).items():
            row.add_trace(path, trace)

    def _run_unit(self, row: _Row, unit: _Unit) -> None:
        requested = [name for name in unit.fields if row.wants(name)]
        for name in unit.fields:
            if name not in requested:
                row.data[name] = row.known.get(name)

        active, post_call = self._active_fields(row, requested)
        if not active:
            return

        if unit.group is None:
            model, prompt, config = self._field_request(unit.key)
        else:
            model, prompt, config = self._group_request(unit.key, unit.group, active)
        prompt += _nested_context_note(row.nested_context, active)

        result = call_agent(row.text, model, prompt, config, self.save_trace)
        for name in active:
            row.data[name] = result["data"].get(name)
        # Em ordem de dependência, para que um campo anulado aqui também anule
        # quem depende dele na mesma chamada.
        for name in post_call:
            if should_skip_field(name, self.field_configs[name], row.data):
                row.data[name] = None
        for name in active:
            self._enrich_items(row, name)
        row.add_result(result, unit.key)

    def _active_fields(self, row: _Row, requested: list[str]) -> tuple[list[str], list[str]]:
        """Campos pedidos ao agente e, entre eles, os de condição avaliada depois da resposta.

        A condição que depende só de campos de outras unidades já pode ser
        avaliada, e o campo pulado nem vai ao agente. A que depende de outro
        campo da mesma chamada só pode ser avaliada com a resposta.
        """
        active, post_call = [], set()
        for name in requested:
            if any(dep.split(".")[0] in requested for dep in self.dependencies[name]):
                active.append(name)
                post_call.add(name)
            elif should_skip_field(name, self.field_configs[name], row.data):
                row.data[name] = None
            else:
                active.append(name)
        return active, [name for name in self.execution_order if name in post_call]

    def _field_request(self, field_name: str) -> tuple[type[BaseModel], str, LLMConfig]:
        field_info = self.pydantic_model.model_fields[field_name]
        field_config = self.field_configs[field_name]
        model = _query_model(
            f"{self.pydantic_model.__name__}_{field_name}",
            {field_name: _llm_field_spec(field_info, self.pydantic_model)},
        )
        prompt = _build_field_prompt(
            self.user_prompt, field_name, field_info.description, field_config
        )
        return model, prompt, _with_field_overrides(self.config, field_config)

    def _group_request(
        self, group_name: str, group: SearchGroupConfig, active: list[str]
    ) -> tuple[type[BaseModel], str, LLMConfig]:
        # O modelo e o prompt do grupo seguem a ordem de search_groups; a ordem
        # de dependência só importa para as condições avaliadas depois da resposta.
        model = _query_model(
            f"{self.pydantic_model.__name__}_group_{group_name}",
            {
                name: _llm_field_spec(self.pydantic_model.model_fields[name], self.pydantic_model)
                for name in active
            },
        )
        if group.prompt:
            # {query} é sinônimo de {texto} no prompt de grupo
            prompt = _with_text_placeholder(group.prompt.replace("{query}", "{texto}"))
        else:
            prompt = f"{self.user_prompt}\n\nResponda os campos: {', '.join(active)}"
        config = _with_search_overrides(
            self.config, group.search_depth, group.max_results, group.max_search_calls
        )
        return model, prompt, config

    def _enrich_items(self, row: _Row, field_name: str) -> None:
        """Busca, item a item, os campos configurados de um List[Model] já extraído."""
        spec = self.list_fields.get(field_name)
        items = row.data[field_name]
        if spec is None or not items:
            return
        enriched, usage, traces = _enrich_list_items_with_search(
            items,
            spec["inner_model"],
            spec["search_fields"],
            row.text,
            self.config,
            self.save_trace,
        )
        row.data[field_name] = enriched
        row.add_usage(usage)
        row.add_trace(f"{field_name}_items", traces)


def field_extractor(
    pydantic_model: type[BaseModel],
    user_prompt: str,
    config: LLMConfig,
    save_trace: str | None = None,
) -> _FieldExtractor:
    """Monta a extração com busca dos modos por campo e por grupo.

    Cada grupo de `search_groups` vira uma chamada ao agente com os campos do
    grupo, e cada campo fora de grupo, uma chamada só dele. Sem grupos, é o
    modo por campo. As chamadas seguem a ordem exigida pelas dependências de
    `condition`, e entre unidades independentes, a ordem do modelo. Campo com
    condição falsa fica None e não é pedido ao agente; se a condição depende
    de outro campo da mesma chamada, ele é anulado depois da resposta.

    Campos de modelos aninhados com configuração de busca própria são buscados
    antes, e o resultado entra no prompt da chamada que extrai o campo de
    primeiro nível. Os de um List[Model] são buscados item a item depois que
    a lista é extraída.

    A extração devolvida recebe `(text, only_fields=None, known=None)`.
    `only_fields` são os campos de primeiro nível a extrair (reprocess_columns);
    os demais voltam com o valor de `known`, que também alimenta as condições.
    Ela devolve 'data' (todos os campos, na ordem do modelo), 'usage' (soma
    das chamadas, com 'search_provider') e, com save_trace, 'traces', um por
    grupo, campo, caminho aninhado e lista enriquecida ('{campo}_items').

    Raises:
        ValueError: Se há dependências inválidas ou circulares, inclusive entre
            um grupo e campos de fora dele.
    """
    field_configs = {
        name: _get_field_config(info.json_schema_extra)
        if isinstance(info.json_schema_extra, dict)
        else {}
        for name, info in pydantic_model.model_fields.items()
    }
    execution_order, dependencies = get_field_execution_order(pydantic_model, field_configs)
    groups = _search_config(config).groups or {}
    units, _, unit_dependencies = get_group_execution_units(pydantic_model, groups, dependencies)

    ordered_units = []
    for key in topological_sort(unit_dependencies):
        _, name, group = units[int(key)]
        fields = tuple(group.fields) if group is not None else (name,)
        ordered_units.append(_Unit(key=name, fields=fields, group=group))

    list_fields = _get_list_fields_with_nested_search(pydantic_model)
    nested_fields = tuple(
        configured
        for configured in _collect_configured_fields(pydantic_model)
        if "." in configured[0] and configured[0].split(".")[0] not in list_fields
    )

    return _FieldExtractor(
        pydantic_model=pydantic_model,
        user_prompt=user_prompt,
        config=config,
        save_trace=save_trace,
        field_configs=field_configs,
        execution_order=execution_order,
        dependencies=dependencies,
        units=tuple(ordered_units),
        list_fields=list_fields,
        nested_fields=nested_fields,
    )


# Início do texto que o ToolCallLimitMiddleware grava na ToolMessage de uma
# chamada bloqueada ("Tool call limit exceeded. Do not call 'x' again.").
_TOOL_LIMIT_MESSAGE_PREFIX = "Tool call limit exceeded"


def _blocked_tool_call_ids(messages: list) -> set:
    """Ids das chamadas de ferramenta que o teto de buscas bloqueou."""
    return {
        msg.tool_call_id
        for msg in messages
        if getattr(msg, "type", None) == "tool"
        and getattr(msg, "status", None) == "error"
        and str(getattr(msg, "content", "")).startswith(_TOOL_LIMIT_MESSAGE_PREFIX)
    }


def _extract_usage(  # noqa: C901 (o diagnóstico em DEBUG descreve cada mensagem)
    agent_result: dict,
    provider: SearchProvider,
    search_config: SearchConfig,
    search_tool_name: str,
) -> dict[str, Any]:
    """Extrai métricas de uso do resultado do agente.

    Args:
        agent_result: Resultado retornado pelo agent.invoke().
        provider: Instância do SearchProvider usado.
        search_config: Configuração de busca (SearchConfig).
        search_tool_name: Nome da ferramenta de busca passada ao agente.

    Returns:
        Dicionário com tokens e créditos de busca.
    """
    usage = _empty_usage(search_provider=provider.name)

    # Extrair token usage das mensagens
    messages = agent_result.get("messages", [])

    # Diagnóstico: logar detalhes de cada mensagem (ativado com logging.DEBUG)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("[token_tracking] Total messages in agent result: %s", len(messages))
        for i, msg in enumerate(messages):
            msg_type = getattr(msg, "type", type(msg).__name__)
            has_metadata = hasattr(msg, "usage_metadata") and msg.usage_metadata is not None
            has_tool_calls = hasattr(msg, "tool_calls") and msg.tool_calls

            metadata_info = ""
            if has_metadata:
                parsed = _parse_usage_metadata(msg.usage_metadata)
                metadata_info = f"in={parsed['input_tokens']}, out={parsed['output_tokens']}"

            tool_info = ""
            if has_tool_calls:
                tool_names = []
                for tc in msg.tool_calls:
                    name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
                    tool_names.append(name)
                tool_info = f", tools={tool_names}"

            logger.debug(
                "[token_tracking]   [%s] %s: has_usage_metadata=%s%s%s",
                i,
                msg_type,
                has_metadata,
                f", {metadata_info}" if metadata_info else "",
                tool_info,
            )

    # Reasoning tokens (GPT-5, o-series, Claude thinking) já estão
    # contabilizados em output_tokens — output_token_details.reasoning é
    # apenas um breakdown.
    for msg in messages:
        if hasattr(msg, "usage_metadata") and msg.usage_metadata:
            parsed = _parse_usage_metadata(msg.usage_metadata)
            usage["input_tokens"] += parsed["input_tokens"]
            usage["cached_input_tokens"] += parsed["cached_input_tokens"]
            usage["output_tokens"] += parsed["output_tokens"]
            usage["total_tokens"] += parsed["total_tokens"]
            usage["reasoning_tokens"] += parsed["reasoning_tokens"]

    # Só a ferramenta de busca conta. O agente também recebe a ferramenta de
    # structured output do ToolStrategy, cujo nome é o do modelo Pydantic
    # (ex.: `NestedSearch_*`, `ResearchResult`), e por isso casar substring
    # como "search" inflaria search_count e search_credits.
    # Chamada bloqueada pelo teto de buscas não executa nem é cobrada.
    blocked = _blocked_tool_call_ids(messages)
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
                tool_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                if tool_name == search_tool_name and tool_id not in blocked:
                    usage["search_count"] += 1

    # Calcular créditos usando método do provider
    usage["search_credits"] = provider.calculate_credits(
        search_count=usage["search_count"],
        search_depth=search_config.search_depth,
        max_results=search_config.max_results,
    )

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("[token_tracking] Final usage: %s", usage)

    return usage


def _extract_trace(  # noqa: PLR0913, PLR0917 (os dados do trace vêm de fontes distintas)
    agent_result: dict,
    model: str | None,
    duration: float,
    mode: str,
    provider: SearchProvider | None = None,
    search_tool_name: str | None = None,
) -> dict:
    """Extrai trace do resultado do agente LangChain.

    Args:
        agent_result: Resultado de agent.invoke().
        model: Nome do modelo usado.
        duration: Tempo de execução em segundos.
        mode: "full" ou "minimal".
        provider: Instância do SearchProvider usado (opcional).
        search_tool_name: Nome da ferramenta de busca; só as chamadas dela
            executadas entram em search_queries.

    Returns:
        Dicionário com trace estruturado. total_tool_calls conta todas as
        chamadas de ferramenta, inclusive a de structured output.
    """
    messages = agent_result.get("messages", [])
    blocked = _blocked_tool_call_ids(messages)
    trace = {
        "messages": [],
        "search_queries": [],
        "total_tool_calls": 0,
        "duration_seconds": round(duration, 3),
        "model": model,
        "search_provider": provider.name if provider else None,
    }

    for msg in messages:
        msg_data = {"type": msg.type}

        # Content - omite para tool messages em modo minimal
        if msg.type == "tool" and mode == "minimal":
            msg_data["content"] = "[omitted]"
        else:
            msg_data["content"] = msg.content

        # Tool calls (AIMessage)
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            msg_data["tool_calls"] = []
            for tc in msg.tool_calls:
                msg_data["tool_calls"].append(
                    {
                        "name": tc.get("name", ""),
                        "args": tc.get("args", {}),
                        "id": tc.get("id", ""),
                        "type": tc.get("type", "tool_call"),
                    }
                )
                trace["total_tool_calls"] += 1
                if tc.get("name") == search_tool_name and tc.get("id") not in blocked:
                    query = tc.get("args", {}).get("query", "")
                    if query:
                        trace["search_queries"].append(query)

        # Tool call reference (ToolMessage)
        if hasattr(msg, "tool_call_id") and msg.tool_call_id:
            msg_data["tool_call_id"] = msg.tool_call_id

        trace["messages"].append(msg_data)

    return trace
