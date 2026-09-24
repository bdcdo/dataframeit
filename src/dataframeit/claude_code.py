"""Integração com Claude Code SDK para usar créditos do Claude Code.

Este módulo permite usar o claude-agent-sdk como provider alternativo ao LangChain,
fazendo chamadas LLM via créditos do Claude Code em vez de créditos de API.
"""
import asyncio
import concurrent.futures
import json

from .errors import retry_with_backoff
from .llm import LLMConfig, _parse_usage_metadata, build_prompt
from .utils import check_dependency, parse_json


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
        Tupla (response_text, usage), em que ``usage`` é o dict bruto de
        ``ResultMessage.usage`` ou None quando o SDK não o informa.
    """
    from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock, query

    response_text = ""
    usage = None

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    response_text += block.text
        elif isinstance(message, ResultMessage):
            usage = getattr(message, "usage", None)

    return response_text, usage


def _run_coroutine(coro):
    """Executa a corrotina até o fim a partir de código síncrono.

    ``asyncio.run`` recusa rodar quando o thread atual já tem um event loop
    ativo, caso do Jupyter. Nesse caso a corrotina roda num thread próprio,
    com loop novo, e este thread espera o resultado.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coro).result()


def _usage_from_sdk(usage) -> dict | None:
    """Converte ``ResultMessage.usage`` para o formato de tokens do core.

    O SDK repassa o usage da API da Anthropic, em que ``input_tokens`` exclui
    os tokens lidos do cache e os gravados nele. O core espera ``input_tokens``
    com o cache incluído e ``cached_input_tokens`` como a parcela lida do
    cache, a mesma convenção do provider LangChain.
    """
    if not usage:
        return None

    cache_read = usage.get("cache_read_input_tokens") or 0
    cache_creation = usage.get("cache_creation_input_tokens") or 0
    input_tokens = (usage.get("input_tokens") or 0) + cache_read + cache_creation
    output_tokens = usage.get("output_tokens") or 0

    return _parse_usage_metadata({
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_token_details": {"cache_read": cache_read},
    })


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

    # O texto das linhas é conteúdo não confiável e pode conter instruções
    # injetadas. `tools=[]` remove todas as ferramentas embutidas do contexto
    # do modelo (`allowed_tools` só pré-aprova ferramentas, não restringe).
    # `permission_mode="default"` explícito sobrepõe um `defaultMode` vindo
    # dos settings do usuário. Como nenhum `can_use_tool` é registrado, o SDK
    # não liga canal de pergunta ao CLI, e pedido de permissão é negado.
    options_kwargs = {
        "system_prompt": system_prompt,
        "tools": [],
        "permission_mode": "default",
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
        response_text, sdk_usage = _run_coroutine(_async_query(prompt, options))

        if not response_text.strip():
            raise ValueError("Claude Code SDK retornou resposta vazia")

        # Parse e validação
        parsed = parse_json(response_text)
        validated = pydantic_model.model_validate(parsed)

        return {'data': validated.model_dump(), 'usage': _usage_from_sdk(sdk_usage)}

    return retry_with_backoff(_call, config.max_retries, config.base_delay, config.max_delay)
