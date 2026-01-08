"""Processamento baseado em agente com busca web via Tavily."""

from typing import Any, Dict
from pydantic import create_model

from .llm import LLMConfig, build_prompt
from .errors import retry_with_backoff


def call_agent(text: str, pydantic_model, user_prompt: str, config: LLMConfig) -> dict:
    """Processa texto usando agente LangChain com busca Tavily.

    Args:
        text: Texto a ser processado (ex: nome do medicamento, país, etc.).
        pydantic_model: Modelo Pydantic para estruturar resposta.
        user_prompt: Template do prompt do usuário.
        config: Configuração do LLM incluindo SearchConfig.

    Returns:
        Dicionário com 'data' (dados extraídos) e 'usage' (metadata incluindo
        search_credits e search_count).
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

    # Criar agente com structured output
    agent = create_agent(
        model=config.model,
        model_provider=config.provider,
        tools=[tavily_tool],
        response_format=ToolStrategy(pydantic_model),
        api_key=config.api_key,
        **config.model_kwargs,
    )

    def _call():
        prompt = build_prompt(user_prompt, text)
        result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})

        # Extrair resposta estruturada
        structured = result.get("structured_response")
        if structured is None:
            raise ValueError("Agente não retornou resposta estruturada")

        data = structured.model_dump() if hasattr(structured, 'model_dump') else structured

        # Calcular usage (tokens + search credits)
        usage = _extract_usage(result, search_config.search_depth)

        return {'data': data, 'usage': usage}

    return retry_with_backoff(_call, config.max_retries, config.base_delay, config.max_delay)


def call_agent_per_field(text: str, pydantic_model, user_prompt: str, config: LLMConfig) -> dict:
    """Processa cada campo do modelo Pydantic com agente separado.

    Útil quando o modelo tem muitos campos e um único contexto ficaria
    sobrecarregado com informações de múltiplas buscas.

    Args:
        text: Texto a ser processado.
        pydantic_model: Modelo Pydantic completo.
        user_prompt: Template do prompt do usuário.
        config: Configuração do LLM.

    Returns:
        Dicionário com 'data' (todos os campos combinados) e 'usage' (soma
        de todos os tokens e créditos).
    """
    combined_data = {}
    total_usage = {
        'input_tokens': 0,
        'output_tokens': 0,
        'total_tokens': 0,
        'search_credits': 0,
        'search_count': 0,
    }

    # Iterar por cada campo do modelo Pydantic
    for field_name, field_info in pydantic_model.model_fields.items():
        # Criar modelo temporário com apenas este campo
        SingleFieldModel = create_model(
            f'{pydantic_model.__name__}_{field_name}',
            **{field_name: (field_info.annotation, field_info)}
        )

        # Prompt focado no campo específico
        field_prompt = f"{user_prompt}\n\nResponda APENAS o campo: {field_name}"
        if field_info.description:
            field_prompt += f" ({field_info.description})"

        # Chamar agente para este campo
        result = call_agent(text, SingleFieldModel, field_prompt, config)

        # Combinar resultado
        combined_data[field_name] = result['data'].get(field_name)

        # Somar usage de todas as chamadas
        if result.get('usage'):
            for key in total_usage:
                total_usage[key] += result['usage'].get(key, 0)

    return {'data': combined_data, 'usage': total_usage}


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
