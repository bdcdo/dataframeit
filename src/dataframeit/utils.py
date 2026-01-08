import re
import json
import importlib
import time
import random
import warnings
from typing import Tuple, List, Union, Any
import pandas as pd

# Import opcional de Polars
try:
    import polars as pl
except ImportError:
    pl = None


def parse_json(resposta: str) -> dict:
    """Extrai e faz parse de JSON da resposta de um LLM.

    Args:
        resposta: String de resposta do LLM ou objeto com atributo 'content'.

    Returns:
        Dicionário com os dados do JSON.

    Raises:
        ValueError: Se o JSON não puder ser extraído ou decodificado.
    """
    # Extrair conteúdo se for objeto do LangChain
    if hasattr(resposta, 'content'):
        if isinstance(resposta.content, list):
            content = "".join(str(item) for item in resposta.content)
        else:
            content = resposta.content
    else:
        content = str(resposta)

    # Tentar extrair JSON de bloco de código markdown
    match = re.search(r"```json\n(.*?)\n```", content, re.DOTALL)
    if match:
        json_string = match.group(1).strip()
    else:
        # Tentar extrair entre chaves
        start = content.find('{')
        end = content.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_string = content[start:end + 1]
        else:
            json_string = content.strip()

    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError(f"Falha ao decodificar JSON. Erro: {e}. Resposta: '{json_string[:200]}'...")


def check_dependency(package: str, install_name: str = None):
    """Verifica se dependência está instalada.

    Args:
        package: Nome do pacote para importação.
        install_name: Nome do pacote para instalação (padrão: package).

    Raises:
        ImportError: Se a dependência não estiver instalada.
    """
    install_name = install_name or package
    try:
        importlib.import_module(package)
    except ImportError:
        raise ImportError(
            f"'{package}' não instalado. Instale com: pip install {install_name}"
        )


def validate_provider_dependencies(provider: str, use_openai: bool = False):
    """Valida se as dependências do provider estão instaladas ANTES de iniciar.

    Args:
        provider: Nome do provider (google_genai, openai, anthropic).
        use_openai: Se True, valida dependências do OpenAI direto.

    Raises:
        ImportError: Com mensagem amigável se dependência não estiver instalada.
    """
    # Se usar OpenAI diretamente
    if use_openai:
        try:
            importlib.import_module('openai')
        except ImportError:
            raise ImportError(_get_missing_package_message('openai', 'openai', 'OpenAI'))

    # Validar LangChain base
    try:
        importlib.import_module('langchain')
    except ImportError:
        raise ImportError(_get_missing_package_message('langchain', 'langchain', 'LangChain'))

    try:
        importlib.import_module('langchain_core')
    except ImportError:
        raise ImportError(_get_missing_package_message('langchain_core', 'langchain-core', 'LangChain Core'))

    # Validar provider específico
    provider_data = PROVIDER_INFO.get(provider)
    if provider_data:
        package = provider_data['package']
        install = provider_data['install']
        name = provider_data['name']
        try:
            importlib.import_module(package)
        except ImportError:
            raise ImportError(_get_missing_package_message(package, install, name))


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
)

# Mapeamento de provedores para nomes de pacotes e variáveis de ambiente
PROVIDER_INFO = {
    'google_genai': {
        'package': 'langchain_google_genai',
        'install': 'langchain-google-genai',
        'env_var': 'GOOGLE_API_KEY',
        'name': 'Google Gemini',
        'get_key_url': 'https://aistudio.google.com/app/apikey',
    },
    'openai': {
        'package': 'langchain_openai',
        'install': 'langchain-openai',
        'env_var': 'OPENAI_API_KEY',
        'name': 'OpenAI',
        'get_key_url': 'https://platform.openai.com/api-keys',
    },
    'anthropic': {
        'package': 'langchain_anthropic',
        'install': 'langchain-anthropic',
        'env_var': 'ANTHROPIC_API_KEY',
        'name': 'Anthropic Claude',
        'get_key_url': 'https://console.anthropic.com/settings/keys',
    },
}


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

    # Obter informações do provider
    provider_data = PROVIDER_INFO.get(provider, {})
    provider_name = provider_data.get('name', provider or 'LLM')
    env_var = provider_data.get('env_var', 'API_KEY')
    get_key_url = provider_data.get('get_key_url', '')

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
║  1. Obtenha uma chave de API em: {get_key_url:<43} ║
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
║  1. Verifique seu plano em: {get_key_url:<43} ║
║  2. Tente usar um modelo diferente (ex: gemini-1.5-flash)                    ║
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
║  2. Tente acessar {get_key_url} no navegador                                 ║
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
                    stacklevel=3
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
                stacklevel=3
            )

            time.sleep(total_delay)


def to_pandas(df) -> Tuple[pd.DataFrame, bool]:
    """Converte DataFrame para pandas se necessário.

    Args:
        df: DataFrame pandas ou polars.

    Returns:
        Tupla (DataFrame pandas, flag se era polars).

    Raises:
        TypeError: Se não for pandas nem polars DataFrame.
    """
    if pl is not None and isinstance(df, pl.DataFrame):
        return df.to_pandas(), True
    elif isinstance(df, pd.DataFrame):
        return df, False
    else:
        raise TypeError("df deve ser pandas.DataFrame ou polars.DataFrame")


def from_pandas(df: pd.DataFrame, was_polars: bool) -> Union[pd.DataFrame, Any]:
    """Converte de volta para polars se necessário.

    Args:
        df: DataFrame pandas.
        was_polars: Se o DataFrame original era polars.

    Returns:
        DataFrame no formato original.
    """
    if was_polars and pl is not None:
        return pl.from_pandas(df)
    return df
