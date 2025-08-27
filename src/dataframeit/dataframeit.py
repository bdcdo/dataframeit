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
    Função helper para processar texto usando OpenAI em vez de LangChain
    """
    # Obter instruções de formato do parser Pydantic (para manter compatibilidade)
    parser = PydanticOutputParser(pydantic_object=perguntas)
    format_instructions = parser.get_format_instructions()
    
    # Construir prompt completo
    full_prompt = f"""
    {prompt_template}
    
    {format_instructions}
    """.format(sentenca=text_value, format=format_instructions)
    
    # Chamar OpenAI
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

# Função principal
# Trabalho maior seria incluir muitas mensages de erro e garantir que funciona com diferentes LLMs
# O usuário precisaria apenas definir um objeto pydantic com as perguntas e definir o template
def dataframeit(
    df,
    perguntas,
    prompt,
    resume=True,
    model='gemini-2.5-flash',
    provider='google_genai',
    status_column=None,
    text_column: str = 'texto',
    # Novos parâmetros para suporte ao OpenAI
    use_openai=False,
    openai_client=None,
    reasoning_effort='minimal',
    verbosity='low',
):
    # Lógica condicional: OpenAI vs LangChain
    if use_openai:
        if OpenAI is None:
            raise ImportError("OpenAI not installed. Install with: pip install openai")
        client = openai_client or OpenAI()
        chain_g = None  # Não usado com OpenAI
    else:
        # Lógica LangChain original
        parser = PydanticOutputParser(pydantic_object=perguntas)
        prompt_inicial = ChatPromptTemplate.from_template(prompt)
        prompt_intermediario = prompt_inicial.partial(format=parser.get_format_instructions())
        llm = init_chat_model(model, model_provider=provider, temperature=0)
        chain_g = prompt_intermediario | llm
        client = None  # Não usado com LangChain

    # Detectar engine e converter se necessário (polars -> pandas)
    original_was_polars = False
    
    if pl is not None and hasattr(pl, 'DataFrame') and isinstance(df, pl.DataFrame):
        original_was_polars = True
        df = df.to_pandas()
    elif not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas.DataFrame or polars.DataFrame")
    
    # Construir engine label
    engine_parts = [
        'polars→pandas' if original_was_polars else 'pandas',
        'openai' if use_openai else 'langchain'
    ]
    engine_label = '+'.join(engine_parts)

    # Get expected columns from Pydantic model
    expected_columns = list(perguntas.__fields__.keys())
    
    # Validar existência de text_column
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame.")

    # Check existing columns
    existing_columns = df.columns
    new_columns = [col for col in expected_columns if col not in existing_columns]
    existing_result_columns = [col for col in expected_columns if col in existing_columns]
    
    # Handle conflicts
    if existing_result_columns and not resume:
        warnings.warn(f"Columns {existing_result_columns} already exist. Use resume=True to continue or rename them.")
        return df
    
    # Add only missing columns
    if new_columns:
        for col in new_columns:
            df.loc[:, col] = None
    
    # Determine which column to use for checking processed status
    if status_column is None:
        # Use the first expected column as default
        status_column = expected_columns[0]
    
    # Add status column if it doesn't exist
    if status_column not in df.columns:
        df.loc[:, status_column] = None
    
    # Find rows that need processing (status column is null)
    if resume:
        # Check which rows have been processed using the status column
        null_mask = df[status_column].isnull()
        unprocessed_indices = df.index[null_mask].tolist()
        start_idx = min(unprocessed_indices) if unprocessed_indices else len(df)
        processed_count = len(df) - len(unprocessed_indices)
    else:
        start_idx = 0
        processed_count = 0
    
    total = len(df)
    
    # Update progress bar description
    desc = (
        f"Processando [{engine_label}] (resumindo de {processed_count}/{total})"
        if processed_count > 0
        else f"Processando [{engine_label}]"
    )
    
    for i, row in enumerate(tqdm(df.iterrows(), total=total, desc=desc)):
        idx, row_data = row
        
        # Skip already processed rows
        if i < start_idx:
            continue
            
        # Check if this specific row is already processed using status column
        if pd.notna(row_data[status_column]):
            continue
            
        if use_openai:
            resposta = _call_openai_chain(client, prompt, perguntas, row_data[text_column], 
                                        model, reasoning_effort, verbosity)
        else:
            resposta = chain_g.invoke({'sentenca': row_data[text_column]})
        
        # Acho que isso eu jogaria para utils. Suponho que diferentes LLMs vão responder de maneira um pouco diferente
        novas_infos = parse_json(resposta)
        
        # Update DataFrame immediately for this row (in-place operation)
        for col in expected_columns:
            if col in novas_infos:
                df.at[idx, col] = novas_infos[col]
        
        # Mark this row as processed by setting the status column to a non-null value
        if status_column in novas_infos:
            # Use the actual value from LLM response
            df.at[idx, status_column] = novas_infos[status_column]
        else:
            # Set a default "processed" marker
            df.at[idx, status_column] = "processed"
    
    # Converter de volta para Polars se a entrada original era Polars
    if original_was_polars and pl is not None:
        return pl.from_pandas(df)

    return df
