"""Tratamento de erros e mensagens amigáveis para usuários iniciantes.

Este módulo contém funções para:
- Classificar erros como recuperáveis ou não-recuperáveis
- Gerar mensagens de erro amigáveis com instruções de resolução
- Validar dependências de providers
- Executar funções com retry e backoff exponencial
"""
import importlib
import time
import random
import warnings


# Erros considerados recuperáveis (transientes)
RECOVERABLE_ERRORS = (
    # Timeouts e deadlines
    'DeadlineExceeded',
    'Timeout',
    'TimeoutError',
    'ReadTimeout',
    'ConnectTimeout',
    # Rate limits
    'RateLimitError',
    'ResourceExhausted',
    'TooManyRequests',
    '429',
    # Erros de servidor temporários
    'ServiceUnavailable',
    'InternalServerError',
    '500',
    '502',
    '503',
    '504',
    # Erros de conexão
    'ConnectionError',
    'ConnectionReset',
    'SSLError',
    # Tavily - rate limit/quota
    'UsageLimitExceededError',
)

# Erros não-recuperáveis (não adianta tentar novamente)
NON_RECOVERABLE_ERRORS = (
    'AuthenticationError',
    'InvalidAPIKey',
    'PermissionDenied',
    'InvalidArgument',
    'NotFound',
    '401',
    '403',
    '404',
    # Tavily - erros de autenticação e argumentos
    'MissingAPIKeyError',
    'InvalidAPIKeyError',
    'BadRequestError',
)


def _infer_provider_info(provider: str) -> dict:
    """Infere informações do provider dinamicamente.

    Args:
        provider: Nome do provider (google_genai, openai, anthropic, etc).

    Returns:
        Dict com package, install, env_var e name inferidos.
    """
    if not provider:
        return {'package': None, 'install': None, 'env_var': 'API_KEY', 'name': 'LLM'}

    # Inferir nome do pacote: provider -> langchain_{provider}
    package = f"langchain_{provider}"
    # Inferir nome para pip: langchain_{provider} -> langchain-{provider}
    install = package.replace('_', '-')

    # Inferir variável de ambiente
    # google_genai -> GOOGLE_API_KEY, openai -> OPENAI_API_KEY
    provider_upper = provider.replace('_genai', '').replace('_ai', '').upper()
    env_var = f"{provider_upper}_API_KEY"

    # Nome amigável
    name_map = {
        'google_genai': 'Google Gemini',
        'openai': 'OpenAI',
        'anthropic': 'Anthropic Claude',
        'cohere': 'Cohere',
        'mistralai': 'Mistral AI',
        'fireworks': 'Fireworks AI',
        'together': 'Together AI',
        'groq': 'Groq',
    }
    name = name_map.get(provider, provider.replace('_', ' ').title())

    return {
        'package': package,
        'install': install,
        'env_var': env_var,
        'name': name,
    }


def _get_missing_package_message(package: str, install_name: str, friendly_name: str) -> str:
    """Gera mensagem amigável para pacote não instalado."""
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
║  Ou, para instalar todas as dependências recomendadas:                       ║
║                                                                              ║
║      pip install dataframeit[all]                                            ║
║                                                                              ║
║  Após instalar, execute seu código novamente.                                ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""".strip()


def validate_provider_dependencies(provider: str):
    """Valida se as dependências do provider estão instaladas ANTES de iniciar.

    Args:
        provider: Nome do provider (google_genai, openai, anthropic, etc).

    Raises:
        ImportError: Com mensagem amigável se dependência não estiver instalada.
    """
    # Validar LangChain base
    try:
        importlib.import_module('langchain')
    except ImportError:
        raise ImportError(_get_missing_package_message('langchain', 'langchain', 'LangChain'))

    try:
        importlib.import_module('langchain_core')
    except ImportError:
        raise ImportError(_get_missing_package_message('langchain_core', 'langchain-core', 'LangChain Core'))

    # Validar provider específico (inferir dinamicamente)
    if provider:
        provider_data = _infer_provider_info(provider)
        package = provider_data['package']
        install = provider_data['install']
        name = provider_data['name']
        try:
            importlib.import_module(package)
        except ImportError:
            raise ImportError(_get_missing_package_message(package, install, name))


def validate_search_dependencies():
    """Valida se as dependências do Tavily estão instaladas e API key configurada.

    Raises:
        ImportError: Com mensagem amigável se langchain-tavily não estiver instalado.
        ValueError: Com mensagem amigável se TAVILY_API_KEY não estiver configurada.
    """
    import os

    # Validar langchain-tavily
    try:
        importlib.import_module('langchain_tavily')
    except ImportError:
        raise ImportError(_get_missing_package_message(
            'langchain_tavily',
            'langchain-tavily',
            'Tavily Search'
        ))

    # Validar API key
    if not os.environ.get('TAVILY_API_KEY'):
        raise ValueError("""
╔══════════════════════════════════════════════════════════════════════════════╗
║  CHAVE DE API DO TAVILY NÃO CONFIGURADA                                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Para usar busca web, você precisa de uma chave de API do Tavily.            ║
║                                                                              ║
║  COMO OBTER:                                                                 ║
║  1. Acesse https://app.tavily.com                                            ║
║  2. Crie uma conta (gratuita - 1000 buscas/mês)                              ║
║  3. Copie sua API key                                                        ║
║                                                                              ║
║  COMO CONFIGURAR:                                                            ║
║                                                                              ║
║  No Linux/Mac:                                                               ║
║      export TAVILY_API_KEY="sua-chave-aqui"                                  ║
║                                                                              ║
║  No Windows (PowerShell):                                                    ║
║      $env:TAVILY_API_KEY="sua-chave-aqui"                                    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""".strip())


def get_friendly_error_message(error: Exception, provider: str = None) -> str:
    """Converte erro técnico em mensagem amigável para usuários iniciantes.

    Args:
        error: Exceção original.
        provider: Nome do provider (google_genai, openai, etc).

    Returns:
        Mensagem de erro amigável com instruções de como resolver.
    """
    error_str = f"{type(error).__name__}: {error}".lower()
    error_name = type(error).__name__

    # Obter informações do provider dinamicamente
    provider_data = _infer_provider_info(provider)
    provider_name = provider_data['name']
    env_var = provider_data['env_var']

    # === ERROS DE AUTENTICAÇÃO ===
    if any(p in error_str for p in ['authenticationerror', 'invalidapikey', '401', 'api_key', 'api key']):
        msg = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ERRO DE AUTENTICAÇÃO - Chave de API inválida ou não configurada             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  O {provider_name} não aceitou sua chave de API.                             ║
║                                                                              ║
║  COMO RESOLVER:                                                              ║
║                                                                              ║
║  1. Obtenha uma chave de API no site/console do {provider_name}              ║
║                                                                              ║
║  2. Configure a chave no terminal (antes de executar seu código):            ║
║                                                                              ║
║     No Linux/Mac:                                                            ║
║     export {env_var}="sua-chave-aqui"                                        ║
║                                                                              ║
║     No Windows (PowerShell):                                                 ║
║     $env:{env_var}="sua-chave-aqui"                                          ║
║                                                                              ║
║  3. OU passe diretamente no código:                                          ║
║     dataframeit(..., api_key="sua-chave-aqui")                               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        return msg.strip()

    # === ERROS DE PERMISSÃO ===
    if any(p in error_str for p in ['permissiondenied', '403', 'forbidden']):
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
    if any(p in error_str for p in ['ratelimit', 'resourceexhausted', 'toomanyrequests', '429']):
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
    if any(p in error_str for p in ['timeout', 'deadlineexceeded', '504']):
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
    if any(p in error_str for p in ['connectionerror', 'connectionreset', 'sslerror', 'network']):
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

    # === ERROS TAVILY ===
    if 'tavily' in error_str and any(p in error_str for p in ['apikey', 'api_key', 'missing', 'invalid']):
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

    if 'usagelimitexceeded' in error_str or ('tavily' in error_str and 'limit' in error_str):
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


def is_recoverable_error(error: Exception) -> bool:
    """Verifica se um erro é recuperável (vale a pena fazer retry).

    Args:
        error: Exceção a ser analisada.

    Returns:
        True se o erro é recuperável, False caso contrário.
    """
    error_str = f"{type(error).__name__}: {error}"

    # Verificar se é explicitamente não-recuperável
    for pattern in NON_RECOVERABLE_ERRORS:
        if pattern.lower() in error_str.lower():
            return False

    # Verificar se é explicitamente recuperável
    for pattern in RECOVERABLE_ERRORS:
        if pattern.lower() in error_str.lower():
            return True

    # Por padrão, tentar recuperar (comportamento original)
    return True


def is_rate_limit_error(error: Exception) -> bool:
    """Verifica se um erro é especificamente de rate limit.

    Args:
        error: Exceção a ser analisada.

    Returns:
        True se o erro é de rate limit, False caso contrário.
    """
    error_str = f"{type(error).__name__}: {error}".lower()
    rate_limit_patterns = ('ratelimit', 'resourceexhausted', 'toomanyrequests', '429')
    return any(pattern in error_str for pattern in rate_limit_patterns)


def retry_with_backoff(func, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 30.0) -> dict:
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
        'attempts': 0,
        'retries': 0,
        'errors': [],
    }

    for attempt in range(max_retries):
        retry_info['attempts'] = attempt + 1
        try:
            result = func()
            # Adicionar retry_info ao resultado se for dict
            if isinstance(result, dict):
                result['_retry_info'] = retry_info
            return result
        except Exception as e:
            error_name = type(e).__name__
            error_msg = str(e)
            retry_info['errors'].append(f"{error_name}: {error_msg[:100]}")

            # Verificar se é erro não-recuperável
            if not is_recoverable_error(e):
                warnings.warn(
                    f"Erro não-recuperável detectado ({error_name}). Não será feito retry.",
                    stacklevel=5
                )
                raise

            # Última tentativa - não fazer mais retry
            if attempt == max_retries - 1:
                raise

            # Calcular delay com backoff exponencial
            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = random.uniform(0, 0.1) * delay
            total_delay = delay + jitter

            retry_info['retries'] = attempt + 1

            # Warning informativo sobre o retry
            warnings.warn(
                f"Tentativa {attempt + 1}/{max_retries} falhou ({error_name}). "
                f"Aguardando {total_delay:.1f}s antes de tentar novamente...",
                stacklevel=5
            )

            time.sleep(total_delay)
