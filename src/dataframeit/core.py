import warnings
import time
import logging
from typing import Union, Any, Optional
import pandas as pd
from tqdm import tqdm

from .llm import LLMConfig, call_openai, call_langchain
from .utils import to_pandas, from_pandas, get_complex_fields, normalize_complex_columns
from .errors import validate_provider_dependencies, get_friendly_error_message, is_recoverable_error


# Suprimir mensagens de retry do LangChain (elas são redundantes com nossos warnings)
logging.getLogger('langchain_google_genai').setLevel(logging.ERROR)
logging.getLogger('langchain_core').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)


def dataframeit(
    df,
    questions=None,
    prompt=None,
    perguntas=None,  # Deprecated: use 'questions'
    resume=True,
    model='gemini-2.5-flash',
    provider='google_genai',
    status_column=None,
    text_column: str = 'texto',
    use_openai=False,
    openai_client=None,
    reasoning_effort='minimal',
    verbosity='low',
    api_key=None,
    max_retries=3,
    base_delay=1.0,
    max_delay=30.0,
    rate_limit_delay=0.0,
    track_tokens=False,
) -> Union[pd.DataFrame, Any]:
    """Processa textos em DataFrame usando LLMs para extrair informações estruturadas.

    Args:
        df: DataFrame pandas ou polars contendo textos.
        questions: Modelo Pydantic definindo estrutura a extrair.
        prompt: Template do prompt (use {texto} para indicar onde inserir o texto).
        perguntas: (Deprecated) Use 'questions'.
        resume: Se True, continua de onde parou.
        model: Nome do modelo LLM.
        provider: Provider do LangChain ('google_genai', etc).
        status_column: Coluna para rastrear progresso.
        text_column: Nome da coluna com textos.
        use_openai: Se True, usa OpenAI em vez de LangChain.
        openai_client: Cliente OpenAI customizado.
        reasoning_effort: Esforço de raciocínio OpenAI.
        verbosity: Verbosidade OpenAI.
        api_key: Chave API específica.
        max_retries: Número máximo de tentativas.
        base_delay: Delay base para retry.
        max_delay: Delay máximo para retry.
        rate_limit_delay: Delay em segundos entre requisições para evitar rate limits (padrão: 0.0).
        track_tokens: Se True, rastreia uso de tokens e exibe estatísticas (padrão: False).

    Returns:
        DataFrame com colunas originais + extraídas.

    Raises:
        ValueError: Se parâmetros obrigatórios faltarem.
        TypeError: Se df não for pandas nem polars.
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
    validate_provider_dependencies(provider, use_openai)

    # Converter para pandas se necessário
    df_pandas, was_polars = to_pandas(df)

    # Validar coluna de texto
    if text_column not in df_pandas.columns:
        raise ValueError(f"Coluna '{text_column}' não encontrada no DataFrame")

    # Extrair campos do modelo Pydantic
    expected_columns = list(questions.model_fields.keys())
    if not expected_columns:
        raise ValueError("Modelo Pydantic não pode estar vazio")

    # Verificar conflitos de colunas
    existing_cols = [col for col in expected_columns if col in df_pandas.columns]
    if existing_cols and not resume:
        warnings.warn(
            f"Colunas {existing_cols} já existem. Use resume=True para continuar ou renomeie-as."
        )
        return from_pandas(df_pandas, was_polars)

    # Configurar colunas
    _setup_columns(df_pandas, expected_columns, status_column, resume, track_tokens)

    # Normalizar colunas complexas (listas, dicts, tuples) que podem ter sido
    # serializadas como strings JSON ao salvar/carregar de arquivos
    complex_fields = get_complex_fields(questions)
    if complex_fields and resume:
        normalize_complex_columns(df_pandas, complex_fields)

    # Determinar coluna de status
    status_col = status_column or '_dataframeit_status'

    # Determinar onde começar
    start_pos, processed_count = _get_processing_indices(df_pandas, status_col, resume)

    # Criar config do LLM
    config = LLMConfig(
        model=model,
        provider=provider,
        use_openai=use_openai,
        api_key=api_key,
        openai_client=openai_client,
        reasoning_effort=reasoning_effort,
        verbosity=verbosity,
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        rate_limit_delay=rate_limit_delay,
    )

    # Processar linhas
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
        was_polars,
        track_tokens,
    )

    # Exibir estatísticas de tokens
    if track_tokens and token_stats and any(token_stats.values()):
        _print_token_stats(token_stats, model)

    # Retornar no formato original (remove colunas de status/erro se não houver erros)
    return from_pandas(df_pandas, was_polars)


def _setup_columns(df: pd.DataFrame, expected_columns: list, status_column: Optional[str], resume: bool, track_tokens: bool):
    """Configura colunas necessárias no DataFrame (in-place)."""
    status_col = status_column or '_dataframeit_status'
    error_col = '_error_details'
    token_cols = ['_input_tokens', '_output_tokens', '_total_tokens'] if track_tokens else []

    # Identificar colunas que precisam ser criadas
    new_cols = [col for col in expected_columns if col not in df.columns]
    needs_status = status_col not in df.columns
    needs_error = error_col not in df.columns
    needs_tokens = [col for col in token_cols if col not in df.columns] if track_tokens else []

    if not new_cols and not needs_status and not needs_error and not needs_tokens:
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


def _get_processing_indices(df: pd.DataFrame, status_col: str, resume: bool) -> tuple[int, int]:
    """Retorna (posição inicial, contagem de processados)."""
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


def _print_token_stats(token_stats: dict, model: str):
    """Exibe estatísticas de uso de tokens.

    Args:
        token_stats: Dict com contadores de tokens.
        model: Nome do modelo usado.
    """
    if not token_stats or token_stats.get('total_tokens', 0) == 0:
        return

    print("\n" + "=" * 60)
    print("📊 ESTATÍSTICAS DE USO DE TOKENS")
    print("=" * 60)
    print(f"Modelo: {model}")
    print(f"Total de tokens: {token_stats['total_tokens']:,}")
    print(f"  • Input:  {token_stats['input_tokens']:,} tokens")
    print(f"  • Output: {token_stats['output_tokens']:,} tokens")
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
    was_polars: bool,
    track_tokens: bool,
) -> dict:
    """Processa cada linha do DataFrame.

    Returns:
        Dict com estatísticas de tokens: {'input_tokens', 'output_tokens', 'total_tokens'}
    """
    # Criar descrição para progresso
    engine = 'polars→pandas' if was_polars else 'pandas'
    llm_engine = 'openai' if config.use_openai else 'langchain'
    desc = f"Processando [{engine}+{llm_engine}]"

    # Adicionar info de rate limiting (se ativo)
    if config.rate_limit_delay > 0:
        req_per_min = int(60 / config.rate_limit_delay)
        desc += f" [~{req_per_min} req/min]"

    if processed_count > 0:
        desc += f" (resumindo de {processed_count}/{len(df)})"

    # Inicializar contadores de tokens
    token_stats = {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}

    # Processar cada linha
    for i, (idx, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc=desc)):
        # Pular linhas já processadas
        if i < start_pos or pd.notna(row[status_col]):
            continue

        text = str(row[text_column])

        try:
            # Chamar LLM apropriado
            if config.use_openai:
                result = call_openai(text, pydantic_model, user_prompt, config)
            else:
                result = call_langchain(text, pydantic_model, user_prompt, config)

            # Extrair dados e usage metadata
            extracted = result.get('data', result)  # Retrocompatibilidade
            usage = result.get('usage')
            retry_info = result.get('_retry_info', {})

            # Atualizar DataFrame com dados extraídos
            for col in expected_columns:
                if col in extracted:
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
