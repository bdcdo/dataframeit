# DataFrameIt

Uma biblioteca Python para enriquecer DataFrames com análises de texto usando Modelos de Linguagem (LLMs).

## Descrição

DataFrameIt é uma ferramenta que permite processar textos contidos em um DataFrame e extrair informações estruturadas usando LLMs. A biblioteca utiliza LangChain e pandas para manipulação de dados. Polars é suportado via conversão interna (entrada Polars é convertida para pandas durante o processamento e convertida de volta ao final).

## Funcionalidades

- Processar cada linha de um DataFrame que contenha textos
- Utilizar prompt templates para análise específica de domínio
- Extrair informações estruturadas usando modelos Pydantic
- Suporte para diferentes modelos de linguagem (atualmente implementado para Gemini)

## Uso Básico

```python
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
from dataframeit import dataframeit

# Defina um modelo Pydantic para estruturar as respostas
class SuaClasse(BaseModel):
    campo1: str = Field(..., description="Descrição do campo 1")
    campo2: Literal['opcao1', 'opcao2'] = Field(..., description="Descrição do campo 2")

# Defina seu template de prompt
TEMPLATE = """
Instruções para o modelo de linguagem...
{format}
Texto a ser analisado:
{sentenca}
"""

# Carregue seus dados
df = pd.read_excel('seu_arquivo.xlsx')

# Processe os dados
df_resultado = dataframeit(df, SuaClasse, TEMPLATE)

# Salve o resultado
df_resultado.to_excel('resultado.xlsx', index=False)
```

## Exemplo

O diretório `example/` contém um caso de uso para análise de decisões judiciais relacionadas a tratamentos de saúde para doenças raras.

## Contribuições

Contribuições são bem-vindas! Este é um projeto em desenvolvimento inicial.

## Licença

Veja o arquivo LICENSE para detalhes.