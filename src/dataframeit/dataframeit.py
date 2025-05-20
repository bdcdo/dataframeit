from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from tqdm import tqdm
import polars as pl

from .utils import parse_json

# Função principal
# Trabalho maior seria incluir muitas mensages de erro e garantir que funciona com diferentes LLMs
# O usuário precisaria apenas definir um objeto pydantic com as perguntas e definir o template
def dataframeit(df, perguntas, prompt):
    parser = PydanticOutputParser(pydantic_object=perguntas)
    prompt_inicial = ChatPromptTemplate.from_template(prompt)
    prompt_intermediario = prompt_inicial.partial(format=parser.get_format_instructions())
    llm = init_chat_model('gemini-2.5-flash-preview-04-17', model_provider='google_genai', temperature=0)

    chain_g = prompt_intermediario | llm

    infos_completas = []
    total=df.height

    for row in tqdm(df.iter_rows(named=True), total=total, desc='Processando'):
        resposta = chain_g.invoke({'sentenca': row['texto']})

        # Acho que isso eu jogaria para utils. Suponho que diferentes LLMs vão responder de maneira um pouco diferente
        novas_infos = parse_json(resposta)

        infos_completas.append(row | novas_infos)

    return pl.DataFrame(infos_completas, strict=False)
