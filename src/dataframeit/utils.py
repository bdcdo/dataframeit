import re
import json

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
