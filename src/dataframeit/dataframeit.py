from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from tqdm import tqdm
import pandas as pd
import warnings
 
# Import opcional de Polars
try:
    import polars as pl  # type: ignore
except Exception:  # Polars não instalado
    pl = None  # type: ignore

# Import opcional de OpenAI
try:
    from openai import OpenAI  # type: ignore
except ImportError:  # OpenAI não instalado
    OpenAI = None  # type: ignore

from .utils import parse_json

def _call_openai_chain(client, prompt_template, perguntas, text_value, model, reasoning_effort, verbosity):
    """
    Processa texto usando API OpenAI diretamente.

    Constrói o prompt com instruções de formato Pydantic e chama o modelo OpenAI
    com configurações de raciocínio e verbosidade específicas.

    Args:
        client: Cliente OpenAI configurado
        prompt_template: Template do prompt com placeholder {sentenca}
        perguntas: Modelo Pydantic para instruções de formato
        text_value: Texto a ser processado
        model: Nome do modelo OpenAI
        reasoning_effort: Esforço de raciocínio ('minimal', 'medium', 'high')
        verbosity: Nível de verbosidade ('low', 'medium', 'high')

    Returns:
        str: Resposta JSON do modelo OpenAI
    """
    # Gerar instruções de formato JSON usando parser Pydantic
    parser = PydanticOutputParser(pydantic_object=perguntas)
    format_instructions = parser.get_format_instructions()
    
    # Combinar template com instruções de formato
    full_prompt = f"""
    {prompt_template}
    
    {format_instructions}
    """.format(sentenca=text_value, format=format_instructions)
    
    # Enviar requisição para API OpenAI com configurações específicas
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": full_prompt}],
        reasoning={
            "effort": reasoning_effort
        },
        completion={
            "verbosity": verbosity
        }
    )
    
    return response.choices[0].message.content

def dataframeit(
    df,
    perguntas,
    prompt,
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
):
    """
    Processa textos em um DataFrame usando LLMs para extrair informações estruturadas.

    Suporta processamento via OpenAI ou LangChain com diferentes modelos e providers.
    Converte automaticamente DataFrames do Polars para pandas quando necessário.

    Args:
        df: DataFrame pandas ou polars contendo os textos para processar
        perguntas: Modelo Pydantic definindo a estrutura das informações a extrair
        prompt: Template do prompt com placeholder {sentenca} para o texto
        resume: Se True, continua processamento de onde parou usando status_column
        model: Nome do modelo LLM (padrão: 'gemini-2.5-flash')
        provider: Provider do LangChain (padrão: 'google_genai')
        status_column: Coluna para rastrear progresso (padrão: primeira coluna do modelo)
        text_column: Nome da coluna contendo os textos (padrão: 'texto')
        use_openai: Se True, usa OpenAI em vez de LangChain
        openai_client: Cliente OpenAI customizado (opcional)
        reasoning_effort: Esforço de raciocínio para OpenAI ('minimal', 'medium', 'high')
        verbosity: Nível de verbosidade para OpenAI ('low', 'medium', 'high')
        api_key: Chave API específica (opcional, senão usa variável de ambiente)

    Returns:
        DataFrame com colunas originais mais as definidas no modelo Pydantic

    Raises:
        ImportError: Se OpenAI não estiver instalado quando use_openai=True
        TypeError: Se df não for pandas.DataFrame nem polars.DataFrame
        ValueError: Se text_column não existir no DataFrame
    """
    # Configurar cliente baseado na escolha OpenAI vs LangChain
    if use_openai:
        if OpenAI is None:
            raise ImportError("OpenAI not installed. Install with: pip install openai")
        if openai_client:
            client = openai_client
        elif api_key:
            client = OpenAI(api_key=api_key)
        else:
            client = OpenAI()
        chain_g = None
    else:
        # Configurar chain LangChain
        parser = PydanticOutputParser(pydantic_object=perguntas)
        prompt_inicial = ChatPromptTemplate.from_template(prompt)
        prompt_intermediario = prompt_inicial.partial(format=parser.get_format_instructions())
        # Incluir chave API se fornecida
        model_kwargs = {"model_provider": provider, "temperature": 0}
        if api_key:
            model_kwargs["api_key"] = api_key

        llm = init_chat_model(model, **model_kwargs)
        chain_g = prompt_intermediario | llm
        client = None

    # Converter Polars para pandas se necessário
    original_was_polars = False
    
    if pl is not None and hasattr(pl, 'DataFrame') and isinstance(df, pl.DataFrame):
        original_was_polars = True
        df = df.to_pandas()
    elif not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas.DataFrame or polars.DataFrame")
    
    # Criar identificador do processamento para barra de progresso
    engine_parts = [
        'polars→pandas' if original_was_polars else 'pandas',
        'openai' if use_openai else 'langchain'
    ]
    engine_label = '+'.join(engine_parts)

    # Extrair colunas esperadas do modelo Pydantic
    expected_columns = list(perguntas.__fields__.keys())
    
    # Verificar se coluna de texto existe
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame.")

    # Identificar colunas existentes e novas
    existing_columns = df.columns
    new_columns = [col for col in expected_columns if col not in existing_columns]
    existing_result_columns = [col for col in expected_columns if col in existing_columns]
    
    # Tratar conflitos de colunas existentes
    if existing_result_columns and not resume:
        warnings.warn(f"Columns {existing_result_columns} already exist. Use resume=True to continue or rename them.")
        return df
    
    # Criar apenas colunas que não existem
    if new_columns:
        for col in new_columns:
            df.loc[:, col] = None
    
    # Definir coluna para controle de progresso
    if status_column is None:
        # Usar primeira coluna do modelo como padrão
        status_column = expected_columns[0]
    
    # Criar coluna de status se não existir
    if status_column not in df.columns:
        df.loc[:, status_column] = None
    
    # Determinar quais linhas precisam ser processadas
    if resume:
        # Identificar linhas já processadas pela coluna de status
        null_mask = df[status_column].isnull()
        unprocessed_indices = df.index[null_mask].tolist()
        start_idx = min(unprocessed_indices) if unprocessed_indices else len(df)
        processed_count = len(df) - len(unprocessed_indices)
    else:
        start_idx = 0
        processed_count = 0
    
    total = len(df)
    
    # Configurar descrição da barra de progresso
    desc = (
        f"Processando [{engine_label}] (resumindo de {processed_count}/{total})"
        if processed_count > 0
        else f"Processando [{engine_label}]"
    )
    
    for i, row in enumerate(tqdm(df.iterrows(), total=total, desc=desc)):
        idx, row_data = row
        
        # Pular linhas já processadas
        if i < start_idx:
            continue
            
        # Verificar se linha específica já foi processada
        if pd.notna(row_data[status_column]):
            continue
            
        if use_openai:
            resposta = _call_openai_chain(client, prompt, perguntas, row_data[text_column], 
                                        model, reasoning_effort, verbosity)
        else:
            resposta = chain_g.invoke({'sentenca': row_data[text_column]})
        
        # Parsear resposta JSON do LLM
        novas_infos = parse_json(resposta)
        
        # Atualizar DataFrame com as informações extraídas
        for col in expected_columns:
            if col in novas_infos:
                df.at[idx, col] = novas_infos[col]
        
        # Marcar linha como processada
        if status_column in novas_infos:
            # Usar valor retornado pelo LLM
            df.at[idx, status_column] = novas_infos[status_column]
        else:
            # Usar marcador padrão se não retornado pelo LLM
            df.at[idx, status_column] = "processed"
    
    # Retornar ao formato original se DataFrame veio do Polars
    if original_was_polars and pl is not None:
        return pl.from_pandas(df)

    return df
