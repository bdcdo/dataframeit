from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from tqdm import tqdm

from respostaFinal import RespostaFinalWrapper, TEMPLATE

import polars as pl
import json
import re

# Acho que isso eu jogaria para utils. Suponho que diferentes LLMs vão responder de maneira um pouco diferente
def parse_json(resposta):
    if isinstance(resposta.content, list):
        langchain_output_content = "".join(str(item) for item in resposta.content)
    else:
        langchain_output_content = resposta.content

    match = re.search(r"```json\n(.*?)\n```", langchain_output_content, re.DOTALL)
    if match:
        json_string_extraida = match.group(1).strip()
    else:
        start_brace = langchain_output_content.find('{')
        end_brace = langchain_output_content.rfind('}')
        if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
            json_string_extraida = langchain_output_content[start_brace : end_brace+1].strip()
        else:
            json_string_extraida = langchain_output_content.strip() # Assume que pode ser JSON direto

    data_dict = json.loads(json_string_extraida)

    return data_dict

# Função principal
# Trabalho maior seria incluir muitas mensages de erro e garantir que funciona com diferentes LLMs
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
        novas_infos = parse_json(resposta)

        infos_completas.append(row | novas_infos)

    return pl.DataFrame(infos_completas, strict=False)

# O usuário precisaria apenas definir um objeto pydantic com as perguntas e definir o template
if __name__=='__main__':
    df = pl.read_excel('clusters_saude_cluster_amostras_rodada5_5porCluster.xlsx')
    
    df_final=dataframeit(df, RespostaFinalWrapper, TEMPLATE)

    df_final.write_excel('clusters_analisados.xlsx')