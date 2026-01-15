import json
import warnings
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Union, Any, Optional, Literal
import pandas as pd
from tqdm import tqdm

from .llm import LLMConfig, SearchConfig, call_langchain
from .utils import (
    to_pandas,
    from_pandas,
    get_complex_fields,
    normalize_complex_columns,
    DEFAULT_TEXT_COLUMN,
    ORIGINAL_TYPE_PANDAS_DF,
    ORIGINAL_TYPE_POLARS_DF,
)
from .errors import validate_provider_dependencies, validate_search_dependencies, get_friendly_error_message, is_recoverable_error, is_rate_limit_error


# Suprimir mensagens de retry do LangChain (elas são redundantes com nossos warnings)
logging.getLogger('langchain_google_genai').setLevel(logging.ERROR)
logging.getLogger('langchain_core').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)

# Chaves de configuração per-field reconhecidas em json_schema_extra
_FIELD_CONFIG_KEYS = ('prompt', 'prompt_replace', 'prompt_append', 'search_depth', 'max_results')


def _has_field_config(pydantic_model) -> bool:
    """Verifica se algum campo tem configuração customizada em json_schema_extra.

    Args:
        pydantic_model: Modelo Pydantic a ser verificado.

    Returns:
        True se algum campo tiver configuração per-field, False caso contrário.
    """
    for field_info in pydantic_model.model_fields.values():
        extra = field_info.json_schema_extra
        if isinstance(extra, dict):
            if any(k in extra for k in _FIELD_CONFIG_KEYS):
                return True
    return False


# Limites de rate limit conhecidos (requests por minuto) para provedores de busca
_SEARCH_PROVIDER_RATE_LIMITS = {
    'tavily': 100,  # ~100 requests/minute no plano gratuito/básico
    'exa': 300,     # ~5 QPS (queries/segundo) = ~300/min no plano padrão
}

# Número máximo de queries simultâneas recomendado para evitar rate limits
_RECOMMENDED_MAX_CONCURRENT_SEARCH_QUERIES = 10


def _warn_search_rate_limit(
    num_rows: int,
    num_fields: int,
    parallel_requests: int,
    search_per_field: bool,
    rate_limit_delay: float,
    search_provider: str = "tavily",
) -> None:
    """Emite warning se a configuração pode exceder rate limits de busca.

    Args:
        num_rows: Número de linhas a processar.
        num_fields: Número de campos no modelo Pydantic.
        parallel_requests: Número de workers paralelos.
        search_per_field: Se True, executa uma busca por campo.
        rate_limit_delay: Delay entre requisições configurado.
        search_provider: Provedor de busca ('tavily' ou 'exa').
    """
    # Calcular queries estimadas
    queries_per_row = num_fields if search_per_field else 1
    total_queries = num_rows * queries_per_row
    concurrent_queries = parallel_requests * queries_per_row

    # Obter limite do provedor (usa tavily como fallback)
    provider_limit = _SEARCH_PROVIDER_RATE_LIMITS.get(
        search_provider,
        _SEARCH_PROVIDER_RATE_LIMITS['tavily']
    )
    provider_name = search_provider.capitalize()

    # Condições para warning:
    # 1. Muitas queries concorrentes (podem sobrecarregar a API instantaneamente)
    # 2. Taxa estimada pode exceder o limite (considerando rate_limit_delay)
    should_warn = False
    warning_reasons = []
    recommendations = []

    # Verificar queries concorrentes
    if concurrent_queries > _RECOMMENDED_MAX_CONCURRENT_SEARCH_QUERIES:
        should_warn = True
        warning_reasons.append(
            f"- Queries concorrentes estimadas: {concurrent_queries} "
            f"(limite recomendado: {_RECOMMENDED_MAX_CONCURRENT_SEARCH_QUERIES})"
        )

    # Calcular taxa estimada de queries por minuto
    if rate_limit_delay > 0:
        # Com delay, a taxa é limitada
        estimated_rpm = (60 / rate_limit_delay) * parallel_requests
    else:
        # Sem delay, assume processamento rápido (~1 req/segundo por worker como estimativa)
        estimated_rpm = parallel_requests * 60

    if search_per_field:
        estimated_rpm *= num_fields

    if estimated_rpm > provider_limit * 0.8:  # 80% do limite como margem de segurança
        should_warn = True
        warning_reasons.append(
            f"- Taxa estimada: ~{estimated_rpm:.0f} queries/min "
            f"(limite {provider_name}: ~{provider_limit}/min)"
        )

    if not should_warn:
        return

    # Calcular recomendações
    if search_per_field:
        # Com search_per_field, recomendar parallel_requests mais baixo
        recommended_parallel = max(1, _RECOMMENDED_MAX_CONCURRENT_SEARCH_QUERIES // num_fields)
        # Calcular delay necessário para ficar abaixo do limite
        # queries/min = (60 / delay) * parallel * fields
        # delay = (60 * parallel * fields) / queries_limit
        recommended_delay = (60 * recommended_parallel * num_fields) / (provider_limit * 0.7)
    else:
        recommended_parallel = min(parallel_requests, _RECOMMENDED_MAX_CONCURRENT_SEARCH_QUERIES)
        recommended_delay = (60 * recommended_parallel) / (provider_limit * 0.7)

    recommended_delay = max(0.5, round(recommended_delay, 1))
    recommended_parallel = max(1, recommended_parallel)

    recommendations.append(f"parallel_requests={recommended_parallel}")
    if rate_limit_delay < recommended_delay:
        recommendations.append(f"rate_limit_delay={recommended_delay}")

    # Emitir warning
    warning_msg = (
        f"\n{'='*60}\n"
        f"AVISO: Configuração pode exceder rate limits de busca ({provider_name})\n"
        f"{'='*60}\n"
        f"Configuração atual:\n"
        f"  - Provedor de busca: {search_provider}\n"
        f"  - Linhas a processar: {num_rows}\n"
        f"  - Campos no modelo: {num_fields}\n"
        f"  - parallel_requests: {parallel_requests}\n"
        f"  - search_per_field: {search_per_field}\n"
        f"  - rate_limit_delay: {rate_limit_delay}s\n"
        f"  - Total de queries estimadas: {total_queries}\n"
        f"\nProblemas detectados:\n"
        + "\n".join(warning_reasons) +
        f"\n\nRecomendações para evitar HTTP 429 (rate limit):\n"
        f"  dataframeit(..., {', '.join(recommendations)})\n"
        f"\nAlternativamente, use search_per_field=False se não precisar\n"
        f"de buscas específicas por campo.\n"
        f"{'='*60}\n"
    )

    warnings.warn(warning_msg, UserWarning, stacklevel=3)


def dataframeit(
    data,
    questions=None,
    prompt=None,
    perguntas=None,  # Deprecated: use 'questions'
    resume=True,
    reprocess_columns=None,
    model='gemini-3.0-flash',
    provider='google_genai',
    status_column=None,
    text_column: Optional[str] = None,
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
    save_trace: Optional[Union[bool, Literal["full", "minimal"]]] = None,
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
        model: Nome do modelo LLM.
        provider: Provider do LangChain ('google_genai', 'openai', 'anthropic', etc).
        status_column: Coluna para rastrear progresso.
        text_column: Nome da coluna com textos (obrigatório para DataFrames,
                    automático para Series/list/dict).
        api_key: Chave API específica.
        max_retries: Número máximo de tentativas.
        base_delay: Delay base para retry.
        max_delay: Delay máximo para retry.
        rate_limit_delay: Delay em segundos entre requisições para evitar rate limits (padrão: 0.0).
        track_tokens: Se True, rastreia uso de tokens e exibe estatísticas (padrão: True).
        model_kwargs: Parâmetros extras para o modelo LangChain (ex: temperature, reasoning_effort).
        parallel_requests: Número de requisições paralelas (padrão: 1 = sequencial).
            Se > 1, processa múltiplas linhas simultaneamente.
            Ao detectar erro de rate limit (429), o número de workers é reduzido automaticamente.
            Dica: use track_tokens=True para ver métricas de throughput (RPM, TPM) e calibrar.
        use_search: Se True, habilita busca web antes de processar. Padrão: False.
        search_provider: Provedor de busca web a usar. Opções:
            - "tavily": Motor de busca otimizado para IA (padrão). Requer TAVILY_API_KEY.
              Melhor para volume baixo-médio (<2667 buscas/mês) ou quando precisa >25 resultados.
            - "exa": Motor de busca semântico. Requer EXA_API_KEY.
              Mais econômico para alto volume (>2667 buscas/mês com 1-25 resultados).
        search_per_field: Se True, executa um agente separado para cada campo do modelo Pydantic.
            Útil quando o modelo tem muitos campos e um único contexto ficaria sobrecarregado.
            Padrão: False (um agente responde todos os campos).
        max_results: Número máximo de resultados por busca (1-20). Padrão: 5.
        search_depth: Profundidade da busca - "basic" (1 crédito) ou "advanced" (2 créditos).
            Apenas para Tavily. Padrão: "basic".
        save_trace: Salva o trace completo do raciocínio do agente. Requer use_search=True.
            - None/False: Desabilitado (padrão)
            - True/"full": Trace completo com conteúdo das mensagens
            - "minimal": Apenas queries e contagens, sem conteúdo de tool results
            Colunas geradas: "_trace" (agente único) ou "_trace_{campo}" (per_field).

    Returns:
        Dados com informações extraídas no mesmo formato da entrada.

    Raises:
        ValueError: Se parâmetros obrigatórios faltarem.
        TypeError: Se tipo de dados não for suportado.
    """
    # Compatibilidade com API antiga
    if questions is None and perguntas is not None:
        questions = perguntas
    elif questions is None:
        raise ValueError("Parâmetro 'questions' é obrigatório")

    if prompt is None:
        raise ValueError("Parâmetro 'prompt' é obrigatório")

    # Se {texto} não estiver no template, adiciona automaticamente ao final
    if '{texto}' not in prompt:
        prompt = prompt.rstrip() + "\n\nTexto a analisar:\n{texto}"

    # Validar dependências ANTES de iniciar (falha rápido com mensagem clara)
    validate_provider_dependencies(provider)

    # Validar parâmetros de busca
    if use_search:
        if search_provider not in ("tavily", "exa"):
            raise ValueError("search_provider deve ser 'tavily' ou 'exa'")
        if search_provider == "tavily" and search_depth not in ("basic", "advanced"):
            raise ValueError("search_depth deve ser 'basic' ou 'advanced'")
        if not 1 <= max_results <= 20:
            raise ValueError("max_results deve estar entre 1 e 20")
        validate_search_dependencies(search_provider)

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
        )

    # Converter para pandas se necessário
    df_pandas, conversion_info = to_pandas(data)

    # Determinar coluna de texto
    is_dataframe_type = conversion_info.original_type in (
        ORIGINAL_TYPE_PANDAS_DF,
        ORIGINAL_TYPE_POLARS_DF,
    )

    if is_dataframe_type:
        # Para DataFrames, usa 'texto' como padrão se não especificado
        if text_column is None:
            text_column = 'texto'
        if text_column not in df_pandas.columns:
            raise ValueError(f"Coluna '{text_column}' não encontrada no DataFrame")
    else:
        # Para Series/list/dict, usa coluna interna
        text_column = DEFAULT_TEXT_COLUMN

    # Extrair campos do modelo Pydantic
    expected_columns = list(questions.model_fields.keys())
    if not expected_columns:
        raise ValueError("Modelo Pydantic não pode estar vazio")

    # Verificar potencial excesso de rate limits de busca
    if use_search and parallel_requests > 1:
        _warn_search_rate_limit(
            num_rows=len(df_pandas),
            num_fields=len(expected_columns),
            parallel_requests=parallel_requests,
            search_per_field=search_per_field,
            rate_limit_delay=rate_limit_delay,
            search_provider=search_provider,
        )

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

    # Verificar conflitos de colunas
    existing_cols = [col for col in expected_columns if col in df_pandas.columns]
    if existing_cols and not resume and not reprocess_columns:
        warnings.warn(
            f"Colunas {existing_cols} já existem. Use resume=True para continuar ou renomeie-as."
        )
        return from_pandas(df_pandas, conversion_info)

    # Configurar colunas
    _setup_columns(df_pandas, expected_columns, status_column, resume, track_tokens, search_config, trace_mode, questions)

    # Normalizar colunas complexas (listas, dicts, tuples) que podem ter sido
    # serializadas como strings JSON ao salvar/carregar de arquivos
    complex_fields = get_complex_fields(questions)
    if complex_fields and resume:
        normalize_complex_columns(df_pandas, complex_fields)

    # Determinar coluna de status
    status_col = status_column or '_dataframeit_status'

    # Determinar onde começar
    start_pos, processed_count = _get_processing_indices(df_pandas, status_col, resume, reprocess_columns)

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

    # Validar: campos com json_schema_extra requerem search_per_field=True
    if _has_field_config(questions):
        if not (config.search_config and config.search_config.per_field):
            raise ValueError(
                "Campos com configuração em json_schema_extra (prompt, prompt_append, "
                "search_depth, max_results) requerem search_per_field=True"
            )

    # Processar linhas (escolher entre sequencial e paralelo)
    if parallel_requests > 1:
        token_stats = _process_rows_parallel(
            df_pandas,
            questions,
            prompt,
            text_column,
            status_col,
            expected_columns,
            config,
            start_pos,
            processed_count,
            conversion_info,
            track_tokens,
            reprocess_columns,
            parallel_requests,
            trace_mode,
        )
    else:
        token_stats = _process_rows(
            df_pandas,
            questions,
            prompt,
            text_column,
            status_col,
            expected_columns,
            config,
            start_pos,
            processed_count,
            conversion_info,
            track_tokens,
            reprocess_columns,
            trace_mode,
        )

    # Exibir estatísticas de tokens e throughput
    if track_tokens and token_stats and any(token_stats.values()):
        _print_token_stats(token_stats, model, parallel_requests)

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
    return from_pandas(df_pandas, conversion_info)


def _setup_columns(df: pd.DataFrame, expected_columns: list, status_column: Optional[str], resume: bool, track_tokens: bool, search_config: Optional[SearchConfig] = None, trace_mode: Optional[str] = None, pydantic_model=None):
    """Configura colunas necessárias no DataFrame (in-place)."""
    status_col = status_column or '_dataframeit_status'
    error_col = '_error_details'
    token_cols = ['_input_tokens', '_output_tokens', '_total_tokens'] if track_tokens else []
    search_cols = ['_search_credits', '_search_count'] if (search_config and search_config.enabled) else []

    # Colunas de trace
    trace_cols = []
    if trace_mode:
        if search_config and search_config.per_field and pydantic_model:
            # Uma coluna por campo
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


def _get_processing_indices(df: pd.DataFrame, status_col: str, resume: bool, reprocess_columns=None) -> tuple[int, int]:
    """Retorna (posição inicial, contagem de processados).

    Nota: quando reprocess_columns está definido, start_pos é ignorado em _process_rows
    pois todas as linhas são processadas (mas só atualiza colunas específicas nas já processadas).
    """
    if not resume:
        return 0, 0

    # Encontrar primeira linha não processada
    null_mask = df[status_col].isnull()
    unprocessed_indices = df.index[null_mask]

    if not unprocessed_indices.empty:
        first_unprocessed = unprocessed_indices.min()
        start_pos = df.index.get_loc(first_unprocessed)
    else:
        start_pos = len(df)

    processed_count = len(df) - len(unprocessed_indices)
    return start_pos, processed_count


def _print_token_stats(token_stats: dict, model: str, parallel_requests: int = 1):
    """Exibe estatísticas de uso de tokens e throughput.

    Args:
        token_stats: Dict com contadores de tokens e métricas de tempo.
        model: Nome do modelo usado.
        parallel_requests: Número de workers paralelos usados.
    """
    if not token_stats or token_stats.get('total_tokens', 0) == 0:
        return

    print("\n" + "=" * 60)
    print("ESTATISTICAS DE USO")
    print("=" * 60)
    print(f"Modelo: {model}")
    print(f"Total de tokens: {token_stats['total_tokens']:,}")
    print(f"  - Input:  {token_stats['input_tokens']:,} tokens")
    print(f"  - Output: {token_stats['output_tokens']:,} tokens")

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
        print("METRICAS DE BUSCA (TAVILY)")
        print("-" * 60)
        print(f"Total de buscas: {token_stats['search_count']}")
        print(f"Creditos usados: {token_stats['search_credits']}")

    print("=" * 60 + "\n")


def _process_rows(
    df: pd.DataFrame,
    pydantic_model,
    user_prompt: str,
    text_column: str,
    status_col: str,
    expected_columns: list,
    config: LLMConfig,
    start_pos: int,
    processed_count: int,
    conversion_info,
    track_tokens: bool,
    reprocess_columns=None,
    trace_mode: Optional[str] = None,
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
    desc = f"Processando [{engine}+langchain{search_mode}]"

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
        'output_tokens': 0,
        'total_tokens': 0,
        'search_credits': 0,
        'search_count': 0,
    }

    # Processar cada linha
    for i, (idx, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc=desc)):
        # Verificar se linha já foi processada
        row_already_processed = pd.notna(row[status_col]) and row[status_col] == 'processed'

        # Decidir se deve processar esta linha
        if reprocess_columns:
            # Com reprocess_columns: processa todas as linhas
            pass
        else:
            # Sem reprocess_columns: pula linhas já processadas (comportamento normal)
            if i < start_pos or row_already_processed:
                continue

        text = str(row[text_column])

        try:
            # Chamar LLM ou agente com busca
            if config.search_config and config.search_config.enabled:
                from .agent import call_agent, call_agent_per_field
                if config.search_config.per_field:
                    result = call_agent_per_field(text, pydantic_model, user_prompt, config, trace_mode)
                else:
                    result = call_agent(text, pydantic_model, user_prompt, config, trace_mode)
            else:
                result = call_langchain(text, pydantic_model, user_prompt, config)

            # Extrair dados e usage metadata
            extracted = result.get('data', result)  # Retrocompatibilidade
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
                df.at[idx, '_output_tokens'] = usage.get('output_tokens', 0)
                df.at[idx, '_total_tokens'] = usage.get('total_tokens', 0)

                # Acumular estatísticas
                token_stats['input_tokens'] += usage.get('input_tokens', 0)
                token_stats['output_tokens'] += usage.get('output_tokens', 0)
                token_stats['total_tokens'] += usage.get('total_tokens', 0)

            # Armazenar métricas de busca (se habilitado)
            if config.search_config and config.search_config.enabled and usage:
                df.at[idx, '_search_credits'] = usage.get('search_credits', 0)
                df.at[idx, '_search_count'] = usage.get('search_count', 0)

                # Acumular estatísticas de busca
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

            # Registrar se houve retries (mesmo em caso de sucesso)
            if retry_info.get('retries', 0) > 0:
                df.at[idx, '_error_details'] = f"Sucesso após {retry_info['retries']} retry(s)"

            # Rate limiting: aguardar antes da próxima requisição
            if config.rate_limit_delay > 0:
                time.sleep(config.rate_limit_delay)

        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"

            # Determinar se foi erro recuperável ou não para mensagem correta
            if is_recoverable_error(e):
                # Erro recuperável que esgotou tentativas
                error_details = f"[Falhou após {config.max_retries} tentativa(s)] {error_msg}"
            else:
                # Erro não-recuperável (não fez retry)
                error_details = f"[Erro não-recuperável] {error_msg}"

            # Exibir mensagem amigável para o usuário
            friendly_msg = get_friendly_error_message(e, config.provider)
            print(f"\n{friendly_msg}\n")

            warnings.warn(f"Falha ao processar linha {idx}.")
            df.at[idx, status_col] = 'error'
            df.at[idx, '_error_details'] = error_details

    return token_stats


def _process_rows_parallel(
    df: pd.DataFrame,
    pydantic_model,
    user_prompt: str,
    text_column: str,
    status_col: str,
    expected_columns: list,
    config: LLMConfig,
    start_pos: int,
    processed_count: int,
    conversion_info,
    track_tokens: bool,
    reprocess_columns,
    parallel_requests: int,
    trace_mode: Optional[str] = None,
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

    # Contadores
    token_stats = {
        'input_tokens': 0,
        'output_tokens': 0,
        'total_tokens': 0,
        'requests_completed': 0,
        'search_credits': 0,
        'search_count': 0,
    }

    # Criar descrição para progresso
    type_labels = {
        ORIGINAL_TYPE_POLARS_DF: 'polars→pandas',
        ORIGINAL_TYPE_PANDAS_DF: 'pandas',
    }
    engine = type_labels.get(conversion_info.original_type, conversion_info.original_type)
    search_mode = '+search' if (config.search_config and config.search_config.enabled) else ''
    desc = f"Processando [{engine}+langchain{search_mode}] [{parallel_requests} workers]"

    if reprocess_columns:
        desc += f" (reprocessando: {', '.join(reprocess_columns)})"
    elif processed_count > 0:
        desc += f" (resumindo de {processed_count}/{len(df)})"

    # Identificar linhas a processar
    rows_to_process = []
    for i, (idx, row) in enumerate(df.iterrows()):
        row_already_processed = pd.notna(row[status_col]) and row[status_col] == 'processed'

        if reprocess_columns:
            rows_to_process.append((i, idx, row))
        else:
            if i >= start_pos and not row_already_processed:
                rows_to_process.append((i, idx, row))

    if not rows_to_process:
        return token_stats

    def process_single_row(row_data):
        """Processa uma única linha (executada em thread separada)."""
        nonlocal current_workers, workers_reduced

        i, idx, row = row_data
        text = str(row[text_column])
        row_already_processed = pd.notna(row[status_col]) and row[status_col] == 'processed'

        # Verificar se devemos pausar devido a rate limit
        if rate_limit_event.is_set():
            time.sleep(2.0)  # Pausa breve quando rate limit detectado

        try:
            # Chamar LLM ou agente com busca
            if config.search_config and config.search_config.enabled:
                from .agent import call_agent, call_agent_per_field
                if config.search_config.per_field:
                    result = call_agent_per_field(text, pydantic_model, user_prompt, config, trace_mode)
                else:
                    result = call_agent(text, pydantic_model, user_prompt, config, trace_mode)
            else:
                result = call_langchain(text, pydantic_model, user_prompt, config)

            # Extrair dados
            extracted = result.get('data', result)
            usage = result.get('usage')
            retry_info = result.get('_retry_info', {})

            # Atualizar DataFrame (com lock para thread-safety)
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
                    df.at[idx, '_output_tokens'] = usage.get('output_tokens', 0)
                    df.at[idx, '_total_tokens'] = usage.get('total_tokens', 0)

                    token_stats['input_tokens'] += usage.get('input_tokens', 0)
                    token_stats['output_tokens'] += usage.get('output_tokens', 0)
                    token_stats['total_tokens'] += usage.get('total_tokens', 0)

                # Métricas de busca
                if config.search_config and config.search_config.enabled and usage:
                    df.at[idx, '_search_credits'] = usage.get('search_credits', 0)
                    df.at[idx, '_search_count'] = usage.get('search_count', 0)

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

                token_stats['requests_completed'] += 1
                df.at[idx, status_col] = 'processed'

                if retry_info.get('retries', 0) > 0:
                    df.at[idx, '_error_details'] = f"Sucesso após {retry_info['retries']} retry(s)"

            # Rate limiting entre requisições (se configurado)
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

            # Registrar erro
            with lock:
                if is_recoverable_error(e):
                    error_details = f"[Falhou após {config.max_retries} tentativa(s)] {error_msg}"
                else:
                    error_details = f"[Erro não-recuperável] {error_msg}"

                friendly_msg = get_friendly_error_message(e, config.provider)
                print(f"\n{friendly_msg}\n")

                warnings.warn(f"Falha ao processar linha {idx}.")
                df.at[idx, status_col] = 'error'
                df.at[idx, '_error_details'] = error_details

            return {'success': False, 'idx': idx, 'error': error_msg}

    # Processar com ThreadPoolExecutor
    with tqdm(total=len(rows_to_process), desc=desc) as pbar:
        # Usar abordagem iterativa para permitir ajuste dinâmico de workers
        pending_rows = list(rows_to_process)
        completed = 0

        while pending_rows:
            # Pegar batch com número atual de workers
            with lock:
                batch_size = min(current_workers, len(pending_rows))
            batch = pending_rows[:batch_size]
            pending_rows = pending_rows[batch_size:]

            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = {executor.submit(process_single_row, row): row for row in batch}

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        pbar.update(1)
                        completed += 1
                    except Exception as e:
                        pbar.update(1)
                        completed += 1
                        warnings.warn(f"Erro inesperado no executor: {e}")

    # Calcular métricas finais
    elapsed = time.time() - start_time
    token_stats['elapsed_seconds'] = elapsed
    token_stats['initial_workers'] = initial_workers
    token_stats['final_workers'] = current_workers
    token_stats['workers_reduced'] = workers_reduced

    return token_stats
