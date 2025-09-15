import re
import json
import importlib

def parse_json(resposta: str) -> dict:
    """
    Extrai e faz o parse de uma string JSON contida na resposta de um LLM.

    Tenta extrair o JSON de blocos de código (```json) ou encontrando o
    primeiro e último caracter '{' e '}'. Se a extração falhar ou o JSON
    for inválido, levanta um ValueError.

    Args:
        resposta: A string de resposta do LLM.

    Returns:
        Um dicionário com os dados do JSON.

    Raises:
        ValueError: Se o JSON não puder ser extraído ou decodificado.
    """
    # Se a resposta for um objeto com atributo 'content', extrai o conteúdo
    if hasattr(resposta, 'content'):
        if isinstance(resposta.content, list):
            langchain_output_content = "".join(str(item) for item in resposta.content)
        else:
            langchain_output_content = resposta.content
    else:
        # Assume que a resposta já é uma string
        langchain_output_content = str(resposta)


    json_string_extraida = None
    match = re.search(r"```json\n(.*?)\n```", langchain_output_content, re.DOTALL)
    
    if match:
        json_string_extraida = match.group(1).strip()
    else:
        start_brace = langchain_output_content.find('{')
        end_brace = langchain_output_content.rfind('}')
        if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
            json_string_extraida = langchain_output_content[start_brace : end_brace + 1]

    if not json_string_extraida:
        # Se nenhuma das estratégias acima funcionou, tenta usar a string toda
        json_string_extraida = langchain_output_content.strip()

    try:
        return json.loads(json_string_extraida)
    except json.JSONDecodeError as e:
        raise ValueError(f"Falha ao decodificar JSON. Erro: {e}. Resposta recebida: '{json_string_extraida[:200]}'...")

def check_dependency(dependency: str, pip_name: str = None):
    """Verifica se uma dependência está instalada e lança um erro claro se não estiver."""
    pip_name = pip_name or dependency
    try:
        importlib.import_module(dependency)
    except ImportError:
        raise ImportError(
            f"A dependência '{dependency}' não está instalada. "
            f"Por favor, instale-a com: uv pip install '{pip_name}'"
        )
