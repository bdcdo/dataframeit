"""Processamento baseado em agente com busca web.

Suporta múltiplos provedores de busca:
- Tavily: Motor de busca otimizado para IA
- Exa: Motor de busca semântico
"""

import logging
import time
from copy import copy
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

from pydantic import create_model

from .llm import LLMConfig, build_prompt, _create_langchain_llm
from .errors import retry_with_backoff
from .search import get_provider
from .utils import get_nested_pydantic_models, is_list_of_pydantic_model

# Chaves de configuração per-field reconhecidas em json_schema_extra
_FIELD_CONFIG_KEYS = ('prompt', 'prompt_replace', 'prompt_append', 'search_depth', 'max_results')


def _get_field_config(extra: dict) -> dict:
    """Extrai configurações relevantes do json_schema_extra.

    Args:
        extra: Dicionário json_schema_extra do campo Pydantic.

    Returns:
        Dicionário com configurações extraídas (prompt, prompt_append,
        search_depth, max_results, depends_on, condition).
    """
    return {
        'prompt': extra.get('prompt') or extra.get('prompt_replace'),
        'prompt_append': extra.get('prompt_append'),
        'search_depth': extra.get('search_depth'),
        'max_results': extra.get('max_results'),
        'depends_on': extra.get('depends_on', []),
        'condition': extra.get('condition'),
    }


def _build_field_prompt(
    user_prompt: str,
    field_name: str,
    field_description: str | None,
    field_config: dict
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
    prompt_replace = field_config.get('prompt')
    prompt_append = field_config.get('prompt_append')

    if prompt_replace:
        # Substitui completamente o prompt
        return prompt_replace

    # Base: prompt original + instrução do campo
    base_prompt = f"{user_prompt}\n\nResponda APENAS o campo: {field_name}"
    if field_description:
        base_prompt += f" ({field_description})"

    if prompt_append:
        # Adiciona texto customizado
        base_prompt += f"\n\n{prompt_append}"

    return base_prompt


def _apply_field_overrides(config: LLMConfig, field_config: dict) -> LLMConfig:
    """Cria novo LLMConfig com overrides do campo (se houver).

    Args:
        config: Configuração base do LLM.
        field_config: Configurações extraídas do json_schema_extra.

    Returns:
        LLMConfig original se não há overrides, ou novo LLMConfig com
        parâmetros de busca sobrescritos.
    """
    search_depth = field_config.get('search_depth')
    max_results = field_config.get('max_results')

    # Se não há overrides, retorna config original
    if not search_depth and not max_results:
        return config

    # Criar nova SearchConfig com overrides
    new_config = copy(config)
    new_search_config = copy(config.search_config)

    if search_depth:
        new_search_config.search_depth = search_depth
    if max_results:
        new_search_config.max_results = max_results

    new_config.search_config = new_search_config
    return new_config


def _collect_configured_fields(pydantic_model, prefix: str = "", _visited: set = None) -> list:
    """Coleta todos os campos com json_schema_extra de busca, incluindo aninhados.

    Args:
        pydantic_model: Modelo Pydantic a analisar.
        prefix: Prefixo do caminho (usado internamente para recursão).
        _visited: Conjunto de modelos já visitados (previne loops infinitos).

    Returns:
        Lista de tuplas: (path, field_name, field_info, parent_model, has_config)
        Ex: ("pedidos.info_medicamento", "status_anvisa_atual", <FieldInfo>, InformacoesMedicamento, True)
    """
    if _visited is None:
        _visited = set()

    # Evitar loops em modelos auto-referenciais
    model_id = id(pydantic_model)
    if model_id in _visited:
        return []
    _visited.add(model_id)

    results = []

    for field_name, field_info in pydantic_model.model_fields.items():
        # Construir caminho completo
        path = f"{prefix}.{field_name}" if prefix else field_name

        # Verificar se este campo tem configuração de busca
        extra = field_info.json_schema_extra
        has_config = False
        if isinstance(extra, dict):
            has_config = any(k in extra for k in _FIELD_CONFIG_KEYS)

        # Adicionar este campo se tiver configuração
        if has_config:
            results.append((path, field_name, field_info, pydantic_model, True))

        # Buscar modelos Pydantic aninhados no tipo do campo
        nested_models = get_nested_pydantic_models(field_info.annotation)
        for nested_model in nested_models:
            # Recursivamente coletar campos do modelo aninhado
            nested_results = _collect_configured_fields(nested_model, path, _visited)
            results.extend(nested_results)

    return results


def _get_list_fields_with_nested_search(pydantic_model) -> dict:
    """Identifica campos List[Model] que têm configuração de busca em modelos internos.

    Args:
        pydantic_model: Modelo Pydantic a analisar.

    Returns:
        Dicionário: {field_name: {'inner_model': Model, 'search_fields': [(relative_path, field_name, field_info)]}}
    """
    list_fields_with_search = {}

    for field_name, field_info in pydantic_model.model_fields.items():
        is_list, inner_model = is_list_of_pydantic_model(field_info.annotation)

        if is_list and inner_model:
            # Coletar campos de busca dentro do modelo interno
            inner_search_fields = _collect_configured_fields(inner_model)

            if inner_search_fields:
                list_fields_with_search[field_name] = {
                    'inner_model': inner_model,
                    'search_fields': inner_search_fields,
                }

    return list_fields_with_search


def _enrich_list_items_with_search(
    list_items: list,
    inner_model,
    search_fields: list,
    text: str,
    config: LLMConfig,
    save_trace: Optional[str] = None
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
    total_usage = {
        'input_tokens': 0,
        'output_tokens': 0,
        'total_tokens': 0,
        'search_credits': 0,
        'search_count': 0,
    }
    traces = [] if save_trace else None

    for item_idx, item in enumerate(list_items or []):
        if item is None:
            enriched_items.append(item)
            continue

        # Converter item para dicionário se necessário
        if hasattr(item, 'model_dump'):
            item_dict = item.model_dump()
        elif isinstance(item, dict):
            item_dict = item.copy()
        else:
            enriched_items.append(item)
            continue

        item_traces = {} if save_trace else None

        # Construir contexto do item para a busca
        item_context = _build_item_context(item_dict, inner_model)

        # Para cada campo de busca no modelo interno
        for path, field_name, field_info, parent_model, has_config in search_fields:
            if not has_config:
                continue

            extra = field_info.json_schema_extra
            field_config = _get_field_config(extra) if isinstance(extra, dict) else {}

            # Criar modelo temporário para a busca
            SingleFieldModel = create_model(
                f'ItemSearch_{item_idx}_{path.replace(".", "_")}',
                **{field_name: (field_info.annotation, field_info)}
            )

            # Construir prompt para busca do campo com contexto do item
            field_prompt = _build_field_prompt(
                f"Pesquise informações para: {item_context}",
                field_name,
                field_info.description,
                field_config
            )

            # Criar config com overrides do campo
            effective_config = _apply_field_overrides(config, field_config)

            # Chamar agente para buscar informações
            result = call_agent(text, SingleFieldModel, field_prompt, effective_config, save_trace)

            # Atualizar o item com o resultado da busca
            search_result = result['data'].get(field_name)
            _set_nested_value(item_dict, path, search_result)

            # Somar usage
            if result.get('usage'):
                for key in total_usage:
                    total_usage[key] += result['usage'].get(key, 0)

            # Coletar trace
            if save_trace and result.get('trace'):
                item_traces[path] = result['trace']

        enriched_items.append(item_dict)

        if save_trace and item_traces:
            traces.append(item_traces)

    return enriched_items, total_usage, traces


def _build_item_context(item_dict: dict, inner_model) -> str:
    """Constrói uma string de contexto para um item de lista.

    Usa os primeiros campos não-nulos do item para criar contexto.
    """
    context_parts = []

    for field_name, field_info in inner_model.model_fields.items():
        value = item_dict.get(field_name)
        if value is not None and not isinstance(value, (dict, list)):
            # Usar apenas valores simples (strings, números)
            context_parts.append(f"{field_name}: {value}")
            if len(context_parts) >= 3:  # Limitar a 3 campos para não sobrecarregar
                break

    return ", ".join(context_parts) if context_parts else "item"


def _set_nested_value(obj: dict, path: str, value):
    """Define um valor em um caminho aninhado de um dicionário.

    Args:
        obj: Dicionário a modificar.
        path: Caminho no formato "a.b.c".
        value: Valor a definir.
    """
    parts = path.split(".")
    current = obj

    for i, part in enumerate(parts[:-1]):
        if part not in current:
            current[part] = {}
        elif not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]

    current[parts[-1]] = value


def call_agent(
    text: str,
    pydantic_model,
    user_prompt: str,
    config: LLMConfig,
    save_trace: Optional[str] = None
) -> dict:
    """Processa texto usando agente LangChain com busca web.

    Args:
        text: Texto a ser processado (ex: nome do medicamento, país, etc.).
        pydantic_model: Modelo Pydantic para estruturar resposta.
        user_prompt: Template do prompt do usuário.
        config: Configuração do LLM incluindo SearchConfig.
        save_trace: Modo de trace ("full", "minimal") ou None para desabilitar.

    Returns:
        Dicionário com 'data' (dados extraídos), 'usage' (metadata incluindo
        search_credits e search_count) e 'trace' (se save_trace habilitado).
    """
    from langchain.agents import create_agent
    from langchain.agents.structured_output import ToolStrategy

    search_config = config.search_config

    # Criar ferramenta de busca via factory (suporta Tavily, Exa, etc.)
    provider = get_provider(search_config.provider)
    search_tool = provider.create_tool(
        max_results=search_config.max_results,
        search_depth=search_config.search_depth,
    )

    # Criar modelo LLM inicializado
    llm = _create_langchain_llm(config.model, config.provider, config.api_key, config.model_kwargs)

    # Criar agente com structured output
    agent = create_agent(
        model=llm,
        tools=[search_tool],
        response_format=ToolStrategy(pydantic_model),
    )

    def _call():
        prompt = build_prompt(user_prompt, text)

        # Medir tempo de execução
        start_time = time.perf_counter()
        result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
        duration = time.perf_counter() - start_time

        # Extrair resposta estruturada
        structured = result.get("structured_response")
        if structured is None:
            raise ValueError("Agente não retornou resposta estruturada")

        data = structured.model_dump() if hasattr(structured, 'model_dump') else structured

        # Calcular usage (tokens + search credits via provider)
        usage = _extract_usage(result, provider, search_config)

        response = {'data': data, 'usage': usage}

        # Extrair trace se habilitado
        if save_trace:
            response['trace'] = _extract_trace(result, config.model, duration, save_trace, provider)

        return response

    return retry_with_backoff(_call, config.max_retries, config.base_delay, config.max_delay)


def _run_nested_searches(
    text: str,
    nested_fields: list,
    config: LLMConfig,
    save_trace: Optional[str] = None
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
    total_usage = {
        'input_tokens': 0,
        'output_tokens': 0,
        'total_tokens': 0,
        'search_credits': 0,
        'search_count': 0,
    }
    traces = {} if save_trace else None

    for path, field_name, field_info, parent_model, has_config in nested_fields:
        if not has_config:
            continue

        # Extrair configurações do campo
        extra = field_info.json_schema_extra
        field_config = _get_field_config(extra) if isinstance(extra, dict) else {}

        # Criar modelo temporário para a busca
        SingleFieldModel = create_model(
            f'NestedSearch_{path.replace(".", "_")}',
            **{field_name: (field_info.annotation, field_info)}
        )

        # Construir prompt para busca do campo aninhado
        field_prompt = _build_field_prompt(
            f"Pesquise informações para preencher o campo '{path}'",
            field_name,
            field_info.description,
            field_config
        )

        # Criar config com overrides do campo (se houver)
        effective_config = _apply_field_overrides(config, field_config)

        # Chamar agente para buscar informações
        result = call_agent(text, SingleFieldModel, field_prompt, effective_config, save_trace)

        # Armazenar contexto de busca
        search_context[path] = result['data'].get(field_name)

        # Somar usage
        if result.get('usage'):
            for key in total_usage:
                total_usage[key] += result['usage'].get(key, 0)

        # Coletar trace
        if save_trace and result.get('trace'):
            traces[path] = result['trace']

    return search_context, total_usage, traces


def call_agent_per_field(
    text: str,
    pydantic_model,
    user_prompt: str,
    config: LLMConfig,
    save_trace: Optional[str] = None
) -> dict:
    """Processa cada campo do modelo Pydantic com agente separado.

    Útil quando o modelo tem muitos campos e um único contexto ficaria
    sobrecarregado com informações de múltiplas buscas.

    Suporta execução condicional de campos baseada em:
    - depends_on: lista de campos que devem ser processados primeiro
    - condition: condição para executar o campo (baseado em valores de outros campos)

    Suporta campos aninhados em List[Model], Optional[Model], etc.
    Para campos List[Model] com configuração de busca interna:
    1. Primeiro extrai a lista básica (estrutura)
    2. Depois enriquece cada item com buscas específicas por item

    Args:
        text: Texto a ser processado.
        pydantic_model: Modelo Pydantic completo.
        user_prompt: Template do prompt do usuário.
        config: Configuração do LLM.
        save_trace: Modo de trace ("full", "minimal") ou None para desabilitar.

    Returns:
        Dicionário com 'data' (todos os campos combinados), 'usage' (soma
        de todos os tokens e créditos) e 'traces' (dict por campo, se habilitado).

    Raises:
        ValueError: Se há dependências circulares ou inválidas.
    """
    from .conditional import get_field_execution_order, should_skip_field

    combined_data = {}
    search_provider = config.search_config.provider if config.search_config else None
    total_usage = {
        'input_tokens': 0,
        'output_tokens': 0,
        'total_tokens': 0,
        'search_credits': 0,
        'search_count': 0,
    }
    traces = {} if save_trace else None

    # Identificar campos List[Model] com configuração de busca interna
    list_fields_with_search = _get_list_fields_with_nested_search(pydantic_model)
    list_field_names = set(list_fields_with_search.keys())

    # Coletar campos configurados aninhados que NÃO estão em List[Model]
    # (campos em List[Model] serão processados por item após a extração da lista)
    all_configured_fields = _collect_configured_fields(pydantic_model)
    nested_configured_fields = [
        f for f in all_configured_fields
        if '.' in f[0] and not any(f[0].startswith(lf + '.') for lf in list_field_names)
    ]

    # Executar buscas para campos aninhados configurados (não em listas)
    nested_context = {}
    if nested_configured_fields:
        nested_context, nested_usage, nested_traces = _run_nested_searches(
            text, nested_configured_fields, config, save_trace
        )

        # Somar usage das buscas aninhadas
        for key in total_usage:
            total_usage[key] += nested_usage.get(key, 0)

        # Coletar traces aninhados
        if save_trace and nested_traces:
            for path, trace in nested_traces.items():
                traces[path] = trace

    # Extrair configurações de todos os campos
    field_configs = {}
    for field_name, field_info in pydantic_model.model_fields.items():
        extra = field_info.json_schema_extra
        field_configs[field_name] = _get_field_config(extra) if isinstance(extra, dict) else {}

    # Determinar ordem de execução baseada em dependências
    try:
        execution_order, dependencies = get_field_execution_order(pydantic_model, field_configs)
    except ValueError as e:
        logger.error(f"Erro ao determinar ordem de execução: {e}")
        raise

    logger.debug(f"Ordem de execução de campos: {execution_order}")
    if any(dependencies.values()):
        logger.debug(f"Mapa de dependências: {dependencies}")

    # Processar campos na ordem determinada
    for field_name in execution_order:
        field_info = pydantic_model.model_fields[field_name]
        field_config = field_configs[field_name]

        # Verificar se o campo deve ser pulado (condição não satisfeita)
        if should_skip_field(field_name, field_config, combined_data):
            logger.info(f"Campo '{field_name}' pulado (condição não satisfeita)")
            combined_data[field_name] = None
            continue

        # Criar modelo temporário com apenas este campo
        SingleFieldModel = create_model(
            f'{pydantic_model.__name__}_{field_name}',
            **{field_name: (field_info.annotation, field_info)}
        )

        # Construir prompt para este campo
        field_prompt = _build_field_prompt(
            user_prompt, field_name, field_info.description, field_config
        )

        # Adicionar contexto de buscas aninhadas ao prompt se houver
        relevant_context = {
            path: value for path, value in nested_context.items()
            if path.startswith(f"{field_name}.")
        }
        if relevant_context:
            context_str = "\n".join(
                f"- {path}: {value}" for path, value in relevant_context.items()
            )
            field_prompt += f"\n\nContexto de buscas realizadas para campos aninhados:\n{context_str}"

        # Criar config com overrides do campo (se houver)
        effective_config = _apply_field_overrides(config, field_config)

        # Chamar agente para este campo
        result = call_agent(text, SingleFieldModel, field_prompt, effective_config, save_trace)

        # Obter resultado do campo
        field_value = result['data'].get(field_name)

        # FASE 2: Se é um campo List[Model] com busca interna, enriquecer cada item
        if field_name in list_fields_with_search and field_value:
            list_config = list_fields_with_search[field_name]
            inner_model = list_config['inner_model']
            search_fields = list_config['search_fields']

            logger.debug(f"Enriquecendo {len(field_value) if isinstance(field_value, list) else 0} itens de '{field_name}' com buscas")

            enriched_items, enrich_usage, enrich_traces = _enrich_list_items_with_search(
                field_value,
                inner_model,
                search_fields,
                text,
                config,
                save_trace
            )

            # Atualizar o valor do campo com itens enriquecidos
            field_value = enriched_items

            # Somar usage das buscas de enriquecimento
            for key in enrich_usage:
                total_usage[key] += enrich_usage.get(key, 0)

            # Coletar traces de enriquecimento
            if save_trace and enrich_traces:
                traces[f'{field_name}_items'] = enrich_traces

        # Combinar resultado
        combined_data[field_name] = field_value

        # Somar usage de todas as chamadas (exceto campos não numéricos)
        if result.get('usage'):
            for key in total_usage:
                total_usage[key] += result['usage'].get(key, 0)

        # Coletar trace por campo
        if save_trace and result.get('trace'):
            traces[field_name] = result['trace']

    # Adicionar search_provider ao usage
    total_usage['search_provider'] = search_provider

    response = {'data': combined_data, 'usage': total_usage}
    if save_trace:
        response['traces'] = traces

    return response


def call_agent_per_group(
    text: str,
    pydantic_model,
    user_prompt: str,
    config: LLMConfig,
    save_trace: Optional[str] = None
) -> dict:
    """Processa campos agrupados com agente compartilhado e campos isolados individualmente.

    Campos em grupos compartilham a mesma busca, reduzindo chamadas de API.
    Campos fora de grupos são processados individualmente como em call_agent_per_field.

    Args:
        text: Texto a ser processado.
        pydantic_model: Modelo Pydantic completo.
        user_prompt: Template do prompt do usuário.
        config: Configuração do LLM incluindo search_config.groups.
        save_trace: Modo de trace ("full", "minimal") ou None para desabilitar.

    Returns:
        Dicionário com 'data' (todos os campos combinados), 'usage' (soma de
        todos os tokens e créditos), e 'traces' (dict por grupo/campo, se habilitado).
    """
    combined_data = {}
    total_usage = {
        'input_tokens': 0,
        'output_tokens': 0,
        'total_tokens': 0,
        'search_credits': 0,
        'search_count': 0,
    }
    traces = {} if save_trace else None

    groups = config.search_config.groups

    # Identificar campos em grupos vs campos isolados
    grouped_fields = set()
    for group_config in groups.values():
        grouped_fields.update(group_config.fields)

    isolated_fields = [f for f in pydantic_model.model_fields.keys() if f not in grouped_fields]

    # 1. Processar cada grupo (busca compartilhada)
    for group_name, group_config in groups.items():
        # Criar modelo com campos do grupo
        group_field_infos = {
            field_name: (pydantic_model.model_fields[field_name].annotation,
                        pydantic_model.model_fields[field_name])
            for field_name in group_config.fields
        }
        GroupModel = create_model(
            f'{pydantic_model.__name__}_group_{group_name}',
            **group_field_infos
        )

        # Construir prompt do grupo
        if group_config.prompt:
            # Substituir {query} pelo texto se presente
            group_prompt = group_config.prompt.replace('{query}', text)
            if '{texto}' not in group_prompt:
                group_prompt = f"{group_prompt}\n\nTexto: {{texto}}"
        else:
            # Prompt padrão com instruções sobre os campos do grupo
            field_list = ', '.join(group_config.fields)
            group_prompt = f"{user_prompt}\n\nResponda os campos: {field_list}"

        # Criar config com overrides do grupo (se houver)
        effective_config = _apply_group_overrides(config, group_config)

        # Chamar agente para o grupo
        result = call_agent(text, GroupModel, group_prompt, effective_config, save_trace)

        # Combinar resultados
        for field_name in group_config.fields:
            combined_data[field_name] = result['data'].get(field_name)

        # Somar usage
        if result.get('usage'):
            for key in total_usage:
                total_usage[key] += result['usage'].get(key, 0)

        # Coletar trace por grupo
        if save_trace and result.get('trace'):
            traces[group_name] = result['trace']

    # 2. Processar campos isolados (um agente por campo)
    for field_name in isolated_fields:
        field_info = pydantic_model.model_fields[field_name]

        # Criar modelo temporário com apenas este campo
        SingleFieldModel = create_model(
            f'{pydantic_model.__name__}_{field_name}',
            **{field_name: (field_info.annotation, field_info)}
        )

        # Extrair configurações do campo (opcional)
        extra = field_info.json_schema_extra
        field_config = _get_field_config(extra) if isinstance(extra, dict) else {}

        # Construir prompt para este campo
        field_prompt = _build_field_prompt(
            user_prompt, field_name, field_info.description, field_config
        )

        # Criar config com overrides do campo (se houver)
        effective_config = _apply_field_overrides(config, field_config)

        # Chamar agente para este campo
        result = call_agent(text, SingleFieldModel, field_prompt, effective_config, save_trace)

        # Combinar resultado
        combined_data[field_name] = result['data'].get(field_name)

        # Somar usage
        if result.get('usage'):
            for key in total_usage:
                total_usage[key] += result['usage'].get(key, 0)

        # Coletar trace por campo isolado
        if save_trace and result.get('trace'):
            traces[field_name] = result['trace']

    response = {'data': combined_data, 'usage': total_usage}
    if save_trace:
        response['traces'] = traces

    return response


def _apply_group_overrides(config: LLMConfig, group_config) -> LLMConfig:
    """Cria novo LLMConfig com overrides do grupo (se houver).

    Args:
        config: Configuração base do LLM.
        group_config: SearchGroupConfig com possíveis overrides.

    Returns:
        LLMConfig original se não há overrides, ou novo LLMConfig com
        parâmetros de busca sobrescritos.
    """
    search_depth = group_config.search_depth
    max_results = group_config.max_results

    # Se não há overrides, retorna config original
    if not search_depth and not max_results:
        return config

    # Criar nova SearchConfig com overrides
    new_config = copy(config)
    new_search_config = copy(config.search_config)

    if search_depth:
        new_search_config.search_depth = search_depth
    if max_results:
        new_search_config.max_results = max_results

    new_config.search_config = new_search_config
    return new_config


def _extract_usage(agent_result: dict, provider, search_config) -> Dict[str, Any]:
    """Extrai métricas de uso do resultado do agente.

    Args:
        agent_result: Resultado retornado pelo agent.invoke().
        provider: Instância do SearchProvider usado.
        search_config: Configuração de busca (SearchConfig).

    Returns:
        Dicionário com tokens e créditos de busca.
    """
    from .search import SearchProvider

    usage = {
        'input_tokens': 0,
        'output_tokens': 0,
        'total_tokens': 0,
        'search_credits': 0,
        'search_count': 0,
        'search_provider': provider.name,
    }

    # Extrair token usage das mensagens
    messages = agent_result.get("messages", [])

    # Diagnóstico: logar detalhes de cada mensagem (ativado com logging.DEBUG)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"[token_tracking] Total messages in agent result: {len(messages)}")
        for i, msg in enumerate(messages):
            msg_type = getattr(msg, 'type', type(msg).__name__)
            has_metadata = hasattr(msg, 'usage_metadata') and msg.usage_metadata is not None
            has_tool_calls = hasattr(msg, 'tool_calls') and msg.tool_calls

            metadata_info = ""
            if has_metadata:
                meta = msg.usage_metadata
                # Suportar tanto dict quanto objeto com atributos
                if isinstance(meta, dict):
                    metadata_info = f"in={meta.get('input_tokens', 0)}, out={meta.get('output_tokens', 0)}"
                else:
                    metadata_info = f"in={getattr(meta, 'input_tokens', 0)}, out={getattr(meta, 'output_tokens', 0)}"

            tool_info = ""
            if has_tool_calls:
                tool_names = []
                for tc in msg.tool_calls:
                    name = tc.get('name', '') if isinstance(tc, dict) else getattr(tc, 'name', '')
                    tool_names.append(name)
                tool_info = f", tools={tool_names}"

            logger.debug(
                f"[token_tracking]   [{i}] {msg_type}: "
                f"has_usage_metadata={has_metadata}"
                f"{f', {metadata_info}' if metadata_info else ''}"
                f"{tool_info}"
            )

    for msg in messages:
        if hasattr(msg, 'usage_metadata') and msg.usage_metadata:
            meta = msg.usage_metadata
            # Suportar tanto dict quanto objeto com atributos
            if isinstance(meta, dict):
                usage['input_tokens'] += meta.get('input_tokens', 0)
                usage['output_tokens'] += meta.get('output_tokens', 0)
                usage['total_tokens'] += meta.get('total_tokens', 0)
            else:
                usage['input_tokens'] += getattr(meta, 'input_tokens', 0)
                usage['output_tokens'] += getattr(meta, 'output_tokens', 0)
                usage['total_tokens'] += getattr(meta, 'total_tokens', 0)

    # Contar chamadas de busca (tool calls) usando padrão do provider
    tool_pattern = provider.get_tool_name_pattern()
    for msg in messages:
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_name = tc.get('name', '') if isinstance(tc, dict) else getattr(tc, 'name', '')
                # Identificar tool calls do provider ou genéricos de busca
                if tool_pattern in tool_name.lower() or 'search' in tool_name.lower():
                    usage['search_count'] += 1

    # Calcular créditos usando método do provider
    usage['search_credits'] = provider.calculate_credits(
        search_count=usage['search_count'],
        search_depth=search_config.search_depth,
        max_results=search_config.max_results,
    )

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"[token_tracking] Final usage: {usage}")

    return usage


def _extract_trace(agent_result: dict, model: str, duration: float, mode: str, provider=None) -> dict:
    """Extrai trace do resultado do agente LangChain.

    Args:
        agent_result: Resultado de agent.invoke().
        model: Nome do modelo usado.
        duration: Tempo de execução em segundos.
        mode: "full" ou "minimal".
        provider: Instância do SearchProvider usado (opcional).

    Returns:
        Dicionário com trace estruturado.
    """
    messages = agent_result.get("messages", [])
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
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            msg_data["tool_calls"] = []
            for tc in msg.tool_calls:
                msg_data["tool_calls"].append({
                    "name": tc.get("name", ""),
                    "args": tc.get("args", {}),
                    "id": tc.get("id", ""),
                    "type": tc.get("type", "tool_call"),
                })
                # Track search queries
                if "search" in tc.get("name", "").lower():
                    query = tc.get("args", {}).get("query", "")
                    if query:
                        trace["search_queries"].append(query)
                    trace["total_tool_calls"] += 1

        # Tool call reference (ToolMessage)
        if hasattr(msg, 'tool_call_id') and msg.tool_call_id:
            msg_data["tool_call_id"] = msg.tool_call_id

        trace["messages"].append(msg_data)

    return trace
