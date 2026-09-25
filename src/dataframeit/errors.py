"""Tratamento de erros e mensagens amigáveis para usuários iniciantes.

Este módulo contém funções para:
- Classificar erros como recuperáveis ou não-recuperáveis
- Gerar mensagens de erro amigáveis com instruções de resolução
- Validar dependências de providers
- Executar funções com retry e backoff exponencial
"""

from __future__ import annotations

import importlib
import os
import random
import re
import time
import warnings
from typing import TYPE_CHECKING, TypeVar

from .search import get_provider

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from .search import SearchProvider

T = TypeVar("T")

CODEX_FILE_AUTH_LOGIN_COMMAND = "codex --config cli_auth_credentials_store='\"file\"' login"


# langchain-core >= 1.6 classifica os erros dos providers em ModelError, com
# is_retryable por classe. Versões anteriores não têm a hierarquia, e a
# classificação cai no status HTTP e nos padrões de mensagem.
try:
    from langchain_core.exceptions import ModelError as _ModelError
    from langchain_core.exceptions import ModelRateLimitError as _ModelRateLimitError
except ImportError:
    _ModelError = None
    _ModelRateLimitError = None


class ProviderError(RuntimeError):
    """Falha de execução reportada por um provider."""


class ProviderTransientError(ProviderError):
    """Falha transitória que pode ser repetida sem reduzir o paralelismo."""


class ProviderOverloadedError(ProviderTransientError):
    """Falha transitória causada por sobrecarga ou limitação do provider."""


class ProviderRejectedOutputError(ProviderTransientError, ValueError):
    """Resposta recusada pela validação do modelo; a tentativa seguinte pede a correção.

    É transitória por classe, e não pela mensagem: um número como '404' no texto
    analisado faria a classificação por texto tratá-la como erro HTTP definitivo.
    Continua ValueError para quem já capturava a falha de parsing.
    """


class ProviderConfigurationError(ValueError):
    """Configuração local incompatível com o contrato de um provider."""


class ProviderOutputError(ValueError):
    """Resposta definitiva incompatível com o contrato de saída."""


# Erros considerados recuperáveis (transientes)
RECOVERABLE_ERRORS = (
    # Timeouts e deadlines
    "DeadlineExceeded",
    "Timeout",
    "TimeoutError",
    "ReadTimeout",
    "ConnectTimeout",
    # Rate limits
    "RateLimitError",
    "ResourceExhausted",
    "TooManyRequests",
    "429",
    # Erros de servidor temporários
    "ServiceUnavailable",
    "InternalServerError",
    "500",
    "502",
    "503",
    "504",
    # Erros de conexão
    "ConnectionError",
    "ConnectionReset",
    "SSLError",
    # Tavily - rate limit/quota
    "UsageLimitExceededError",
)

# Erros não-recuperáveis (não adianta tentar novamente)
NON_RECOVERABLE_ERRORS = (
    "AuthenticationError",
    "InvalidAPIKey",
    "PermissionDenied",
    "InvalidArgument",
    "NotFound",
    "401",
    "403",
    "404",
    # Tavily - erros de autenticação e argumentos
    "MissingAPIKeyError",
    "InvalidAPIKeyError",
    "BadRequestError",
    # Prompt maior que a janela de contexto: repetir não muda o tamanho
    "ContextOverflowError",
)


# Auth via credenciais de SDK (sem env var de API key) compartilhada por
# todos os providers Bedrock. Linhas mantidas curtas para caber em caixas de 80 cols.
_BEDROCK_BASE = {
    "package": "langchain_aws",
    "install": "langchain-aws",
    "env_var": None,
    "auth_hint": (
        "aws configure\n"
        "OU exporte AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY\n"
        "(opcional: AWS_SESSION_TOKEN, AWS_REGION)"
    ),
}

# Providers cuja heurística simples (langchain_{provider} + {PROVIDER}_API_KEY) não bate com a realidade.
# env_var=None indica auth por SDK (ADC, AWS creds), não por API key.
_PROVIDER_OVERRIDES = {
    "claude_code": {
        "package": "claude_agent_sdk",
        "install": "dataframeit[claude-code]",
        "env_var": None,
        "name": "Claude Code",
        "auth_hint": "Autentique o Claude Code conforme a documentação do SDK.",
        "uses_langchain": False,
    },
    "codex": {
        "package": "openai_codex",
        "install": "dataframeit[codex]",
        "env_var": None,
        "name": "OpenAI Codex",
        "auth_hint": CODEX_FILE_AUTH_LOGIN_COMMAND,
        "uses_langchain": False,
    },
    "google_vertexai": {
        "package": "langchain_google_vertexai",
        "install": "langchain-google-vertexai",
        "env_var": None,
        "name": "Google Vertex AI",
        "auth_hint": (
            "gcloud auth application-default login\n"
            "OU exporte GOOGLE_APPLICATION_CREDENTIALS=/caminho/service-account.json"
        ),
    },
    "bedrock": {**_BEDROCK_BASE, "name": "AWS Bedrock"},
    "bedrock_converse": {**_BEDROCK_BASE, "name": "AWS Bedrock (Converse)"},
    "mistralai": {
        "package": "langchain_mistralai",
        "install": "langchain-mistralai",
        "env_var": "MISTRAL_API_KEY",
        "name": "Mistral AI",
    },
    "azure_openai": {
        "package": "langchain_openai",
        "install": "langchain-openai",
        "env_var": "AZURE_OPENAI_API_KEY",
        "name": "Azure OpenAI",
        "auth_hint": (
            'export AZURE_OPENAI_API_KEY="sua-chave-aqui"\n'
            'export AZURE_OPENAI_ENDPOINT="https://<recurso>.openai.azure.com/"\n'
            'export OPENAI_API_VERSION="2025-03-01-preview"'
        ),
    },
}


def _infer_provider_info(provider: str | None) -> dict:
    """Infere informações do provider dinamicamente.

    Args:
        provider: Nome do provider (google_genai, openai, anthropic, etc).

    Returns:
        Dict com package, install, env_var e name inferidos.
    """
    if not provider:
        return {"package": None, "install": None, "env_var": "API_KEY", "name": "LLM"}

    if provider in _PROVIDER_OVERRIDES:
        return dict(_PROVIDER_OVERRIDES[provider])

    # Inferir nome do pacote: provider -> langchain_{provider}
    package = f"langchain_{provider}"
    # Inferir nome para pip: langchain_{provider} -> langchain-{provider}
    install = package.replace("_", "-")

    # Inferir variável de ambiente
    # google_genai -> GOOGLE_API_KEY, openai -> OPENAI_API_KEY
    provider_upper = provider.replace("_genai", "").replace("_ai", "").upper()
    env_var = f"{provider_upper}_API_KEY"

    # Nome amigável
    name_map = {
        "google_genai": "Google Gemini",
        "openai": "OpenAI",
        "anthropic": "Anthropic Claude",
        "cohere": "Cohere",
        "mistralai": "Mistral AI",
        "fireworks": "Fireworks AI",
        "together": "Together AI",
        "groq": "Groq",
    }
    name = name_map.get(provider, provider.replace("_", " ").title())

    return {
        "package": package,
        "install": install,
        "env_var": env_var,
        "name": name,
    }


def _get_missing_package_message(
    package: str,
    install_name: str,
    friendly_name: str,
    alternative_install: str | None = None,
) -> str:
    """Gera mensagem amigável para pacote não instalado."""
    alternative = ""
    if alternative_install:
        alternative = f"""║                                                                              ║
║  Ou, para instalar todas as dependências recomendadas:                       ║
║                                                                              ║
║      pip install {alternative_install:<62} ║
║                                                                              ║
"""
    return f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  BIBLIOTECA NÃO INSTALADA                                                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  A biblioteca '{package}' é necessária para usar {friendly_name}.            ║
║                                                                              ║
║  COMO RESOLVER:                                                              ║
║                                                                              ║
║  Execute o seguinte comando no terminal:                                     ║
║                                                                              ║
║      pip install {install_name:<62} ║
║                                                                              ║
{alternative}║  Após instalar, execute seu código novamente.                                ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""".strip()


def validate_provider_dependencies(provider: str) -> None:
    """Valida se as dependências do provider estão instaladas ANTES de iniciar.

    Args:
        provider: Nome do provider (google_genai, openai, anthropic, claude_code, etc).

    Raises:
        ImportError: Com mensagem amigável se dependência não estiver instalada.
    """
    provider_data = _infer_provider_info(provider)

    # Providers de SDK falam diretamente com seus runtimes, sem LangChain.
    if not provider_data.get("uses_langchain", True):
        try:
            importlib.import_module(provider_data["package"])
        except ImportError as err:
            raise ImportError(
                _get_missing_package_message(
                    provider_data["package"], provider_data["install"], provider_data["name"]
                )
            ) from err
        return

    # Validar LangChain base
    try:
        importlib.import_module("langchain")
    except ImportError as err:
        raise ImportError(
            _get_missing_package_message("langchain", "langchain", "LangChain", "dataframeit[all]")
        ) from err

    try:
        importlib.import_module("langchain_core")
    except ImportError as err:
        raise ImportError(
            _get_missing_package_message(
                "langchain_core",
                "langchain-core",
                "LangChain Core",
                "dataframeit[all]",
            )
        ) from err

    # Validar provider específico (inferir dinamicamente)
    if provider:
        package = provider_data["package"]
        install = provider_data["install"]
        name = provider_data["name"]
        try:
            importlib.import_module(package)
        except ImportError as err:
            raise ImportError(
                _get_missing_package_message(package, install, name, "dataframeit[all]")
            ) from err


def validate_search_dependencies(search_provider: str = "tavily") -> None:
    """Valida se as dependências do provedor de busca estão instaladas e API key configurada.

    Args:
        search_provider: Nome do provedor de busca ("tavily" ou "exa").

    Raises:
        ImportError: Com mensagem amigável se pacote do provedor não estiver instalado.
        ValueError: Com mensagem amigável se API key não estiver configurada.
    """
    provider = get_provider(search_provider)

    try:
        importlib.import_module(provider.package_name)
    except ImportError as err:
        raise ImportError(
            _get_missing_package_message(
                provider.package_name,
                provider.install_name,
                provider.friendly_name,
            )
        ) from err

    if not os.environ.get(provider.env_var):
        raise ValueError(_get_missing_search_api_key_message(provider))


def _get_missing_search_api_key_message(provider: SearchProvider) -> str:
    """Gera mensagem amigável para API key de busca não configurada."""
    return f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  CHAVE DE API DO {provider.friendly_name.upper()} NÃO CONFIGURADA                                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Para usar busca web com {provider.friendly_name}, você precisa de uma API key.           ║
║                                                                              ║
║  COMO OBTER:                                                                 ║
║  1. Acesse {provider.signup_url:<58} ║
║  2. Crie uma conta ({provider.free_tier})                                          ║
║  3. Copie sua API key                                                        ║
║                                                                              ║
║  COMO CONFIGURAR:                                                            ║
║                                                                              ║
║  No Linux/Mac:                                                               ║
║      export {provider.env_var}="sua-chave-aqui"                                   ║
║                                                                              ║
║  No Windows (PowerShell):                                                    ║
║      $env:{provider.env_var}="sua-chave-aqui"                                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""".strip()


# Textos que só aparecem em erro de chave inválida, qualquer que seja o status.
_INVALID_KEY_MARKERS = (
    "api_key_invalid",
    "api key not valid",
    "invalid api key",
    "incorrect api key",
    "invalid x-api-key",
)


def get_friendly_error_message(  # noqa: C901, PLR0911 (uma saída por categoria de erro)
    error: Exception, provider: str | None = None
) -> str:
    """Converte erro técnico em mensagem amigável para usuários iniciantes.

    Args:
        error: Exceção original.
        provider: Nome do provider (google_genai, openai, etc).

    Returns:
        Mensagem de erro amigável com instruções de como resolver.
    """
    if isinstance(error, ProviderRejectedOutputError):
        # Decidida pela classe: a mensagem não traz o texto analisado, mas um
        # número como '401' no caminho de um campo não pode virar erro de chave.
        return (
            "RESPOSTA RECUSADA PELA VALIDAÇÃO DO MODELO PYDANTIC\n"
            f"{error}\n"
            "As tentativas levaram o erro de volta ao modelo, sem sucesso. Veja as regras "
            "do modelo que falharam; se o texto é ambíguo, a instrução do campo pode ajudar."
        )

    error_str = f"{type(error).__name__}: {error}".lower()
    error_name = type(error).__name__
    status = _http_error_status(error)
    # 'exa' como palavra, contando '_' como separador: casa 'EXA_API_KEY' e
    # 'exa-py', mas não 'hexagonal'.
    is_exa = re.search(r"(?<![a-z0-9])exa(?![a-z0-9])", error_str) is not None

    def is_category(patterns: Iterable[str], codes: Iterable[int] = ()) -> bool:
        """Com status HTTP estruturado, só ele decide; sem, valem os padrões."""
        if status is not None:
            return status in codes
        return any(
            _matches_error_pattern(pattern, error_str) for pattern in (*patterns, *map(str, codes))
        )

    # Obter informações do provider dinamicamente
    provider_data = _infer_provider_info(provider)
    provider_name = provider_data["name"]
    env_var = provider_data["env_var"]

    # === ERROS TAVILY ===
    if "tavily" in error_str and any(
        p in error_str for p in ["apikey", "api_key", "missing", "invalid"]
    ):
        return """
╔══════════════════════════════════════════════════════════════════════════════╗
║  ERRO DE AUTENTICAÇÃO TAVILY                                                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Sua chave de API do Tavily está inválida ou não configurada.                ║
║                                                                              ║
║  COMO RESOLVER:                                                              ║
║  1. Acesse https://app.tavily.com e obtenha sua API key                      ║
║  2. Configure a variável de ambiente:                                        ║
║                                                                              ║
║     export TAVILY_API_KEY="sua-chave-aqui"                                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""".strip()

    if "usagelimitexceeded" in error_str or ("tavily" in error_str and "limit" in error_str):
        return """
╔══════════════════════════════════════════════════════════════════════════════╗
║  LIMITE DE USO TAVILY EXCEDIDO                                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Você atingiu o limite de buscas do seu plano Tavily.                        ║
║                                                                              ║
║  COMO RESOLVER:                                                              ║
║  1. Aguarde até o próximo mês (plano gratuito renova mensalmente)            ║
║  2. Ou faça upgrade do plano em https://tavily.com/pricing                   ║
║  3. Ou continue sem busca web (use_search=False)                             ║
║  4. Ou mude para outro provedor (search_provider="exa")                      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""".strip()

    # === ERROS EXA ===
    if is_exa and any(
        p in error_str for p in ["apikey", "api_key", "missing", "invalid", "unauthorized"]
    ):
        return """
╔══════════════════════════════════════════════════════════════════════════════╗
║  ERRO DE AUTENTICAÇÃO EXA                                                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Sua chave de API do Exa está inválida ou não configurada.                   ║
║                                                                              ║
║  COMO RESOLVER:                                                              ║
║  1. Acesse https://exa.ai e obtenha sua API key                              ║
║  2. Configure a variável de ambiente:                                        ║
║                                                                              ║
║     export EXA_API_KEY="sua-chave-aqui"                                      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""".strip()

    if is_exa and any(p in error_str for p in ["limit", "quota", "exceeded"]):
        return """
╔══════════════════════════════════════════════════════════════════════════════╗
║  LIMITE DE USO EXA EXCEDIDO                                                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Você atingiu o limite de buscas do seu plano Exa.                           ║
║                                                                              ║
║  COMO RESOLVER:                                                              ║
║  1. Verifique seu saldo em https://exa.ai                                    ║
║  2. Adicione créditos à sua conta                                            ║
║  3. Ou continue sem busca web (use_search=False)                             ║
║  4. Ou mude para outro provedor (search_provider="tavily")                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""".strip()

    # === ERROS DE AUTENTICAÇÃO ===
    # Chave inválida decide pelo texto mesmo com status: o Google a devolve
    # como 400 INVALID_ARGUMENT, que o status sozinho não distingue.
    invalid_key = any(marker in error_str for marker in _INVALID_KEY_MARKERS)
    if invalid_key or is_category(
        ["authenticationerror", "invalidapikey", "api_key", "api key"], (401,)
    ):
        if env_var is None:
            # Auth via credenciais de SDK (Vertex AI ADC, AWS creds, etc).
            auth_hint = provider_data.get(
                "auth_hint", "Configure as credenciais conforme a documentação do provider."
            )
            hint_lines = "\n".join(f"║     {line:<73}║" for line in auth_hint.split("\n"))
            msg = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ERRO DE AUTENTICAÇÃO - Credenciais não configuradas                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Provider: {provider_name:<66}║
║                                                                              ║
║  Este provider usa credenciais de SDK (não uma API key tradicional).         ║
║                                                                              ║
║  COMO RESOLVER:                                                              ║
║                                                                              ║
{hint_lines}
║                                                                              ║
║  Verifique também se região, projeto/conta e modelo estão corretos.          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
            return msg.strip()
        export_linux = f'export {env_var}="sua-chave-aqui"'
        export_win = f'$env:{env_var}="sua-chave-aqui"'
        msg = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ERRO DE AUTENTICAÇÃO - Chave de API inválida ou não configurada             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Provider: {provider_name:<66}║
║                                                                              ║
║  Sua chave de API não foi aceita.                                            ║
║                                                                              ║
║  COMO RESOLVER:                                                              ║
║                                                                              ║
║  1. Obtenha uma chave de API no site/console do provider.                    ║
║                                                                              ║
║  2. Configure no terminal (antes de executar seu código):                    ║
║                                                                              ║
║     No Linux/Mac:                                                            ║
║     {export_linux:<73}║
║                                                                              ║
║     No Windows (PowerShell):                                                 ║
║     {export_win:<73}║
║                                                                              ║
║  3. OU passe diretamente no código:                                          ║
║     dataframeit(..., api_key="sua-chave-aqui")                               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        return msg.strip()

    # === ERROS DE PERMISSÃO ===
    if is_category(["permissiondenied", "forbidden"], (403,)):
        return f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ERRO DE PERMISSÃO - Sua chave não tem acesso a este recurso                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Sua chave de API do {provider_name} não tem permissão para usar este modelo.║
║                                                                              ║
║  POSSÍVEIS CAUSAS:                                                           ║
║  • A chave é de uma conta gratuita com acesso limitado                       ║
║  • O modelo solicitado requer um plano pago                                  ║
║  • A chave foi revogada ou expirou                                           ║
║                                                                              ║
║  COMO RESOLVER:                                                              ║
║  1. Verifique seu plano no site/console do {provider_name}                   ║
║  2. Tente usar um modelo mais básico                                         ║
║  3. Gere uma nova chave de API                                               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""".strip()

    # === ERROS DE RATE LIMIT ===
    if is_category(["ratelimit", "resourceexhausted", "toomanyrequests"], (429,)):
        return f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  LIMITE DE REQUISIÇÕES ATINGIDO                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Você fez muitas requisições em pouco tempo para o {provider_name}.          ║
║                                                                              ║
║  COMO RESOLVER:                                                              ║
║  1. Aguarde alguns minutos e tente novamente                                 ║
║  2. Use o parâmetro rate_limit_delay para espaçar as requisições:            ║
║                                                                              ║
║     dataframeit(..., rate_limit_delay=1.0)  # 1 segundo entre requisições    ║
║                                                                              ║
║  3. Considere atualizar seu plano para limites maiores                       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""".strip()

    # === ERROS DE TIMEOUT ===
    if is_category(["timeout", "deadlineexceeded"], (408, 504)):
        return f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  TEMPO ESGOTADO (TIMEOUT)                                                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  O {provider_name} demorou muito para responder.                             ║
║                                                                              ║
║  POSSÍVEIS CAUSAS:                                                           ║
║  • Servidor do {provider_name} sobrecarregado                                ║
║  • Conexão de internet instável                                              ║
║  • Texto muito longo para processar                                          ║
║                                                                              ║
║  COMO RESOLVER:                                                              ║
║  1. O sistema já tentou automaticamente várias vezes                         ║
║  2. Use resume=True para continuar de onde parou:                            ║
║                                                                              ║
║     df = dataframeit(df, ..., resume=True)                                   ║
║                                                                              ║
║  3. Tente novamente em alguns minutos                                        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""".strip()

    # === ERROS DE CONEXÃO ===
    if is_category(["connectionerror", "connectionreset", "sslerror", "network"]):
        return f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ERRO DE CONEXÃO                                                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Não foi possível conectar ao {provider_name}.                               ║
║                                                                              ║
║  POSSÍVEIS CAUSAS:                                                           ║
║  • Sem conexão com a internet                                                ║
║  • Firewall ou proxy bloqueando a conexão                                    ║
║  • Servidor do {provider_name} temporariamente indisponível                  ║
║                                                                              ║
║  COMO RESOLVER:                                                              ║
║  1. Verifique sua conexão com a internet                                     ║
║  2. Tente acessar google.com no navegador                                    ║
║  3. Se estiver em rede corporativa, consulte o suporte de TI                 ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""".strip()

    # === ERRO GENÉRICO ===
    return f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ERRO NO PROCESSAMENTO                                                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Tipo: {error_name:<70} ║
║                                                                              ║
║  Detalhes: {str(error)[:66]:<66} ║
║                                                                              ║
║  Se este erro persistir, você pode:                                          ║
║  1. Verificar se suas credenciais estão corretas                             ║
║  2. Tentar novamente com resume=True                                         ║
║  3. Reportar o problema em:                                                  ║
║     https://github.com/bdcdo/dataframeit/issues                              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""".strip()


# Status HTTP 4xx que indicam falha transitória; os demais 4xx são definitivos.
_RECOVERABLE_CLIENT_STATUSES = frozenset({408, 409, 429})
_HTTP_ERROR_MIN = 400
_HTTP_SERVER_ERROR_MIN = 500
_HTTP_ERROR_MAX = 599
_HTTP_TOO_MANY_REQUESTS = 429

# Atributos em que SDKs e clientes HTTP expõem o status da resposta:
# status_code (openai, anthropic, groq, mistral, cohere), code (google.genai,
# google.api_core, urllib) e http_status. O `code` do openai é textual e é
# descartado pela checagem de tipo em _own_http_status.
_HTTP_STATUS_ATTRIBUTES = ("status_code", "code", "http_status")


def _own_http_status(error: BaseException) -> int | None:
    """Status HTTP de erro (400-599) declarado pela própria exceção, se houver."""
    candidates = [getattr(error, name, None) for name in _HTTP_STATUS_ATTRIBUTES]
    candidates.append(getattr(getattr(error, "response", None), "status_code", None))
    for value in candidates:
        # A faixa descarta códigos que não são status HTTP de erro, inclusive bool.
        if isinstance(value, int) and _HTTP_ERROR_MIN <= value <= _HTTP_ERROR_MAX:
            return value
    return None


def _http_error_status(error: BaseException) -> int | None:
    """Status HTTP da exceção ou da primeira causa explícita que o declare.

    Wrappers como o ChatGoogleGenerativeAIError não carregam status próprio e
    encadeiam o erro do SDK com `raise ... from`, por isso a busca segue
    `__cause__`. `__context__` fica de fora: é a exceção que estava sendo tratada
    quando esta surgiu, e não necessariamente a sua causa.
    """
    seen = set()
    current = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        status = _own_http_status(current)
        if status is not None:
            return status
        current = current.__cause__
    return None


def _matches_error_pattern(pattern: str, error_str: str) -> bool:
    """Casa um padrão com a mensagem, já em minúsculas.

    Padrões numéricos casam só como número isolado, para que '401' não case com
    '4015 tokens'.
    """
    if pattern.isdigit():
        return re.search(rf"\b{pattern}\b", error_str) is not None
    return pattern.lower() in error_str


def is_recoverable_error(error: Exception) -> bool:  # noqa: PLR0911 (precedência da docstring)
    """Verifica se um erro é recuperável (vale a pena fazer retry).

    A decisão segue esta precedência: as classes Provider*Error; o
    `is_retryable` do ModelError do langchain-core, quando existe; o status HTTP
    estruturado da exceção ou da sua causa (408, 409, 429 e 5xx são recuperáveis,
    os demais 4xx não); os padrões de NON_RECOVERABLE_ERRORS e RECOVERABLE_ERRORS
    na mensagem; e, se nada casar, o erro é tratado como recuperável.

    Args:
        error: Exceção a ser analisada.

    Returns:
        True se o erro é recuperável, False caso contrário.
    """
    if isinstance(error, ProviderTransientError):
        return True
    if isinstance(
        error,
        (ProviderError, ProviderConfigurationError, ProviderOutputError),
    ):
        return False
    if _ModelError is not None and isinstance(error, _ModelError):
        return bool(error.is_retryable)

    status = _http_error_status(error)
    if status is not None:
        return status >= _HTTP_SERVER_ERROR_MIN or status in _RECOVERABLE_CLIENT_STATUSES

    error_str = f"{type(error).__name__}: {error}".lower()

    # Verificar se é explicitamente não-recuperável
    for pattern in NON_RECOVERABLE_ERRORS:
        if _matches_error_pattern(pattern, error_str):
            return False

    # Verificar se é explicitamente recuperável
    for pattern in RECOVERABLE_ERRORS:
        if _matches_error_pattern(pattern, error_str):
            return True

    return True


def is_rate_limit_error(error: Exception) -> bool:
    """Verifica se um erro é especificamente de rate limit.

    Args:
        error: Exceção a ser analisada.

    Returns:
        True se o erro é de rate limit, False caso contrário.
    """
    if isinstance(error, ProviderOverloadedError):
        return True
    if isinstance(error, ProviderTransientError):
        return False
    if _ModelRateLimitError is not None and isinstance(error, _ModelRateLimitError):
        return True
    if _ModelError is not None and isinstance(error, _ModelError):
        return False

    status = _http_error_status(error)
    if status is not None:
        return status == _HTTP_TOO_MANY_REQUESTS

    error_str = f"{type(error).__name__}: {error}".lower()
    rate_limit_patterns = ("ratelimit", "resourceexhausted", "toomanyrequests", "429")
    return any(_matches_error_pattern(pattern, error_str) for pattern in rate_limit_patterns)


def retry_with_backoff(
    func: Callable[[], T],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
) -> T:
    """Executa função com retry e backoff exponencial.

    Args:
        func: Função a ser executada.
        max_retries: Número máximo de tentativas.
        base_delay: Delay base em segundos.
        max_delay: Delay máximo em segundos.

    Returns:
        Dicionário com 'result' (resultado da função) e 'retry_info' (informações de retry).

    Raises:
        Exception: Última exceção após esgotar tentativas ou erro não-recuperável.
    """
    retry_info = {
        "attempts": 0,
        "retries": 0,
        "errors": [],
    }

    for attempt in range(max_retries):
        retry_info["attempts"] = attempt + 1
        try:
            result = func()
        except Exception as e:
            error_name = type(e).__name__
            error_msg = str(e)
            retry_info["errors"].append(f"{error_name}: {error_msg[:100]}")

            # Verificar se é erro não-recuperável
            if not is_recoverable_error(e):
                warnings.warn(
                    f"Erro não-recuperável detectado ({error_name}). Não será feito retry.",
                    stacklevel=3,
                )
                raise

            # Última tentativa - não fazer mais retry
            if attempt == max_retries - 1:
                raise

            # Calcular delay com backoff exponencial
            delay = min(base_delay * (2**attempt), max_delay)
            jitter = random.uniform(0, 0.1) * delay  # noqa: S311 (jitter de espera, sem uso criptográfico)
            total_delay = delay + jitter

            retry_info["retries"] = attempt + 1

            # Warning informativo sobre o retry
            warnings.warn(
                f"Tentativa {attempt + 1}/{max_retries} falhou ({error_name}). "
                f"Aguardando {total_delay:.1f}s antes de tentar novamente...",
                stacklevel=3,
            )

            time.sleep(total_delay)
        else:
            # Adicionar retry_info ao resultado se for dict
            if isinstance(result, dict):
                result["_retry_info"] = retry_info
            return result

    # range(max_retries) vazio: nenhuma tentativa foi feita.
    msg = f"max_retries deve ser >= 1; recebido {max_retries!r}"
    raise ValueError(msg)
