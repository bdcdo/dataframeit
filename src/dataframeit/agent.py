"""Processamento baseado em agente com busca web via Tavily."""

import time
from copy import copy
from typing import Any, Dict, Optional

from pydantic import create_model

from .llm import LLMConfig, build_prompt, _create_langchain_llm
from .errors import retry_with_backoff


def _get_field_config(extra: dict) -> dict:
    """Extrai configurações relevantes do json_schema_extra.

    Args:
        extra: Dicionário json_schema_extra do campo Pydantic.

    Returns:
        Dicionário com configurações extraídas (prompt, prompt_append,
        search_depth, max_results).
    """
    return {
        'prompt': extra.get('prompt') or extra.get('prompt_replace'),
        'prompt_append': extra.get('prompt_append'),
        'search_depth': extra.get('search_depth'),
        'max_results': extra.get('max_results'),
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


def call_agent(
    text: str,
    pydantic_model,
    user_prompt: str,
    config: LLMConfig,
    save_trace: Optional[str] = None
) -> dict:
    """Processa texto usando agente LangChain com busca Tavily.

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
    from langchain_tavily import TavilySearch

    search_config = config.search_config

    # Criar ferramenta de busca Tavily
    tavily_tool = TavilySearch(
        max_results=search_config.max_results,
        search_depth=search_config.search_depth,
        include_raw_content=False,
        include_answer=False,
    )

    # Criar modelo LLM inicializado
    llm = _create_langchain_llm(config.model, config.provider, config.api_key, config.model_kwargs)

    # Criar agente com structured output
    agent = create_agent(
        model=llm,
        tools=[tavily_tool],
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

        # Calcular usage (tokens + search credits)
        usage = _extract_usage(result, search_config.search_depth)

        response = {'data': data, 'usage': usage}

        # Extrair trace se habilitado
        if save_trace:
            response['trace'] = _extract_trace(result, config.model, duration, save_trace)

        return response

    return retry_with_backoff(_call, config.max_retries, config.base_delay, config.max_delay)


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

    Args:
        text: Texto a ser processado.
        pydantic_model: Modelo Pydantic completo.
        user_prompt: Template do prompt do usuário.
        config: Configuração do LLM.
        save_trace: Modo de trace ("full", "minimal") ou None para desabilitar.

    Returns:
        Dicionário com 'data' (todos os campos combinados), 'usage' (soma
        de todos os tokens e créditos) e 'traces' (dict por campo, se habilitado).
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

    # Iterar por cada campo do modelo Pydantic
    for field_name, field_info in pydantic_model.model_fields.items():
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

        # Somar usage de todas as chamadas
        if result.get('usage'):
            for key in total_usage:
                total_usage[key] += result['usage'].get(key, 0)

        # Coletar trace por campo
        if save_trace and result.get('trace'):
            traces[field_name] = result['trace']

    response = {'data': combined_data, 'usage': total_usage}
    if save_trace:
        response['traces'] = traces

    return response


def _extract_usage(agent_result: dict, search_depth: str) -> Dict[str, Any]:
    """Extrai métricas de uso do resultado do agente.

    Args:
        agent_result: Resultado retornado pelo agent.invoke().
        search_depth: Profundidade da busca ("basic" ou "advanced").

    Returns:
        Dicionário com tokens e créditos de busca.
    """
    usage = {
        'input_tokens': 0,
        'output_tokens': 0,
        'total_tokens': 0,
        'search_credits': 0,
        'search_count': 0,
    }

    # Extrair token usage das mensagens
    messages = agent_result.get("messages", [])
    for msg in messages:
        if hasattr(msg, 'usage_metadata') and msg.usage_metadata:
            usage['input_tokens'] += msg.usage_metadata.get('input_tokens', 0)
            usage['output_tokens'] += msg.usage_metadata.get('output_tokens', 0)
            usage['total_tokens'] += msg.usage_metadata.get('total_tokens', 0)

    # Contar chamadas de busca (tool calls)
    depth_cost = 2 if search_depth == "advanced" else 1
    for msg in messages:
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tc in msg.tool_calls:
                # TavilySearch tool name pode variar
                tool_name = tc.get('name', '') if isinstance(tc, dict) else getattr(tc, 'name', '')
                if 'tavily' in tool_name.lower() or 'search' in tool_name.lower():
                    usage['search_count'] += 1

    usage['search_credits'] = usage['search_count'] * depth_cost

    return usage


def _extract_trace(agent_result: dict, model: str, duration: float, mode: str) -> dict:
    """Extrai trace do resultado do agente LangChain.

    Args:
        agent_result: Resultado de agent.invoke().
        model: Nome do modelo usado.
        duration: Tempo de execução em segundos.
        mode: "full" ou "minimal".

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
