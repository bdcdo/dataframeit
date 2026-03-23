"""Integração com Claude Code SDK para usar créditos do Claude Code.

Este módulo permite usar o claude-agent-sdk como provider alternativo ao LangChain,
fazendo chamadas LLM via créditos do Claude Code em vez de créditos de API.
"""
import asyncio
import json

from .llm import LLMConfig, build_prompt
from .utils import check_dependency, parse_json
from .errors import retry_with_backoff


def _build_json_system_prompt(json_schema: dict) -> str:
    """Gera system prompt com instruções para output JSON estruturado.

    Args:
        json_schema: JSON Schema gerado por pydantic_model.model_json_schema().

    Returns:
        System prompt com schema e instruções de formatação.
    """
    schema_str = json.dumps(json_schema, indent=2, ensure_ascii=False)
    return (
        "Você é um assistente de extração de dados. "
        "Responda APENAS com JSON válido seguindo o schema abaixo.\n"
        "Não inclua texto antes ou depois do JSON. "
        "Não use blocos de código markdown.\n\n"
        f"JSON Schema:\n{schema_str}\n\n"
        "Responda com um único objeto JSON que corresponda exatamente a este schema."
    )


async def _async_query(prompt: str, options):
    """Executa query assíncrona no claude-agent-sdk.

    Args:
        prompt: Prompt do usuário.
        options: ClaudeAgentOptions configurado.

    Returns:
        Tupla (response_text, cost_usd).
    """
    from claude_agent_sdk import query, AssistantMessage, ResultMessage, TextBlock

    response_text = ""
    cost = None

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    response_text += block.text
        elif isinstance(message, ResultMessage):
            cost = getattr(message, "total_cost_usd", None)

    return response_text, cost


def call_claude_code(text: str, pydantic_model, user_prompt: str, config: LLMConfig) -> dict:
    """Processa texto usando Claude Code SDK com structured output via JSON schema.

    Args:
        text: Texto a ser processado.
        pydantic_model: Modelo Pydantic para estruturar resposta.
        user_prompt: Template do prompt do usuário.
        config: Configuração do LLM.

    Returns:
        Dicionário com 'data' (dados extraídos) e 'usage' (metadata).
    """
    check_dependency("claude_agent_sdk", "claude-agent-sdk")

    from claude_agent_sdk import ClaudeAgentOptions

    # Construir prompt e schema
    prompt = build_prompt(user_prompt, text)
    json_schema = pydantic_model.model_json_schema()
    system_prompt = _build_json_system_prompt(json_schema)

    # Construir options a partir de config
    model_kwargs = config.model_kwargs or {}

    options_kwargs = {
        "system_prompt": system_prompt,
        "allowed_tools": [],
        "permission_mode": "bypassPermissions",
        "max_turns": model_kwargs.get("max_turns", 1),
        "max_budget_usd": model_kwargs.get("max_budget_usd", 0.50),
    }

    if config.model:
        options_kwargs["model"] = config.model

    effort = model_kwargs.get("effort")
    if effort:
        options_kwargs["effort"] = effort

    options = ClaudeAgentOptions(**options_kwargs)

    def _call():
        response_text, cost = asyncio.run(_async_query(prompt, options))

        if not response_text.strip():
            raise ValueError("Claude Code SDK retornou resposta vazia")

        # Parse e validação
        parsed = parse_json(response_text)
        validated = pydantic_model.model_validate(parsed)

        usage = {
            'input_tokens': 0,
            'output_tokens': 0,
            'total_tokens': 0,
            'cost_usd': cost,
        }

        return {'data': validated.model_dump(), 'usage': usage}

    return retry_with_backoff(_call, config.max_retries, config.base_delay, config.max_delay)
