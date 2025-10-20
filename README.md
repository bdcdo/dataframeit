# DataFrameIt

Uma biblioteca Python para enriquecer DataFrames com análises de texto usando Modelos de Linguagem (LLMs).

## Descrição

DataFrameIt é uma ferramenta que permite processar textos contidos em um DataFrame e extrair informações estruturadas usando LLMs. A biblioteca suporta tanto **LangChain** quanto **OpenAI** como provedores de modelos. Pandas é utilizado para manipulação de dados, com suporte para Polars via conversão interna.

## Funcionalidades

- Processar cada linha de um DataFrame que contenha textos
- Utilizar prompt templates para análise específica de domínio
- Extrair informações estruturadas usando modelos Pydantic
- **Suporte híbrido**: LangChain (Gemini, etc.) ou OpenAI (GPT-4, etc.)
- Suporte para Polars e Pandas
- Processamento incremental com resumo automático
- **Retry automático** com backoff exponencial para resiliência
- **Rastreamento de erros** com coluna automática `error_details`

## Instalação

### Dependências Base
```bash
pip install dataframeit
```

### Para usar OpenAI
```bash
pip install dataframeit openai
```

### Para usar LangChain
```bash
# Dependências base do LangChain
pip install dataframeit langchain langchain-core

# Para Google Gemini (provider padrão)
pip install langchain-google-genai

# Para outros providers (exemplos)
pip install langchain-anthropic  # Claude
pip install langchain-openai     # GPT via LangChain
```

### Para usar Polars
```bash
pip install dataframeit[polars]
```

## Uso Básico

### Com LangChain (comportamento padrão)

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
# IMPORTANTE: Use {documento} como placeholder (ou customize com o parâmetro 'placeholder')
TEMPLATE = """
Instruções para o modelo de linguagem...

Texto a ser analisado:
{documento}
"""

# Carregue seus dados
df = pd.read_excel('seu_arquivo.xlsx')

# IMPORTANTE: Sua coluna de texto deve se chamar 'texto' por padrão
# ou use text_column='nome_da_coluna' para especificar outra coluna

# Processe os dados (usa LangChain por padrão)
df_resultado = dataframeit(df, SuaClasse, TEMPLATE)

# Salve o resultado
df_resultado.to_excel('resultado.xlsx', index=False)
```

### Com OpenAI

```python
from openai import OpenAI
from dataframeit import dataframeit

# Configure seu cliente OpenAI (opcional)
client = OpenAI(api_key="sua-chave-aqui")

# Processe usando OpenAI
df_resultado = dataframeit(
    df,
    SuaClasse,
    TEMPLATE,
    use_openai=True,                    # Ativa o provider OpenAI
    model='gpt-4o-mini',                # Modelo OpenAI
    openai_client=client,               # Cliente customizado (opcional)
    reasoning_effort='minimal',         # 'minimal', 'low', 'medium', 'high'
    verbosity='low'                     # 'low', 'medium', 'high'
)
```

## Como Funciona o Template

O sistema de templates do DataFrameIt usa dois tipos de placeholders:

### 1. `{format}` - Instruções de Formatação (Automático)
Este placeholder é **opcional** e será substituído automaticamente pelas instruções de formatação JSON geradas pelo Pydantic. Se você não incluir `{format}` no seu template, as instruções serão adicionadas ao final automaticamente.

### 2. Placeholder do Texto (Configurável)
Por padrão, use `{documento}` para indicar onde o texto da linha será inserido. Você pode customizar este nome com o parâmetro `placeholder`.

### Exemplo de Template Completo

```python
TEMPLATE = """
Você é um analista especializado.
Analise o documento a seguir e extraia as informações solicitadas.

{format}

Documento:
{documento}
"""
```

### Customizando o Placeholder

```python
# Se preferir usar outro nome de placeholder
TEMPLATE = """
Analise o seguinte texto:
{meu_texto}
"""

df_resultado = dataframeit(
    df,
    SuaClasse,
    TEMPLATE,
    placeholder='meu_texto'  # Customiza o placeholder
)
```

## Parâmetros

### Parâmetros Gerais
- **`df`**: DataFrame pandas ou polars contendo os textos
- **`questions`**: Modelo Pydantic definindo a estrutura dos dados a extrair
- **`prompt`**: Template do prompt com placeholder para o texto
- **`text_column='texto'`**: Nome da coluna que contém os textos a serem analisados

### Parâmetros de Processamento
- **`resume=True`**: Continua processamento de onde parou (útil para grandes datasets)
- **`status_column=None`**: Nome customizado para coluna de status (padrão: `_dataframeit_status`)
- **`placeholder='documento'`**: Nome do placeholder no template que será substituído pelo texto

### Parâmetros de Resiliência
- **`max_retries=3`**: Número máximo de tentativas em caso de erro
- **`base_delay=1.0`**: Delay inicial em segundos para retry (cresce exponencialmente)
- **`max_delay=30.0`**: Delay máximo em segundos entre tentativas

### Parâmetros LangChain
- **`model='gemini-2.5-flash'`**: Modelo a ser usado
- **`provider='google_genai'`**: Provider do LangChain ('google_genai', 'anthropic', 'openai', etc.)
- **`api_key=None`**: Chave API específica (opcional, usa variáveis de ambiente se None)

### Parâmetros OpenAI
- **`use_openai=False`**: Ativa o uso da OpenAI em vez de LangChain
- **`openai_client=None`**: Cliente OpenAI customizado (usa padrão se None)
- **`model='gpt-4o-mini'`**: Modelo OpenAI (quando `use_openai=True`)
- **`reasoning_effort='minimal'`**: Nível de raciocínio ('minimal', 'low', 'medium', 'high')
- **`verbosity='low'`**: Verbosidade das respostas ('low', 'medium', 'high')

## Tratamento de Erros

O DataFrameIt possui um sistema robusto de tratamento de erros:

### Colunas de Status
- **`_dataframeit_status`**: Coluna automática com status de cada linha
  - `'processed'`: Linha processada com sucesso
  - `'error'`: Linha falhou após todas as tentativas
  - `None/NaN`: Linha ainda não processada

- **`error_details`**: Coluna automática com detalhes de erros
  - Contém mensagem de erro quando status é `'error'`
  - `None/NaN` quando processamento foi bem-sucedido

### Exemplo: Verificando Erros

```python
df_resultado = dataframeit(df, SuaClasse, TEMPLATE)

# Verificar linhas com erro
linhas_com_erro = df_resultado[df_resultado['_dataframeit_status'] == 'error']
print(f"Total de erros: {len(linhas_com_erro)}")

# Ver detalhes dos erros
for idx, row in linhas_com_erro.iterrows():
    print(f"Linha {idx}: {row['error_details']}")

# Salvar apenas linhas processadas com sucesso
df_sucesso = df_resultado[df_resultado['_dataframeit_status'] == 'processed']
df_sucesso.to_excel('resultado_limpo.xlsx', index=False)
```

### Sistema de Retry

O DataFrameIt tenta automaticamente processar linhas com falha usando backoff exponencial:

```python
# Configurando retry mais agressivo
df_resultado = dataframeit(
    df,
    SuaClasse,
    TEMPLATE,
    max_retries=5,        # Tentar até 5 vezes
    base_delay=2.0,       # Começar com 2 segundos de espera
    max_delay=60.0        # Esperar no máximo 60 segundos entre tentativas
)
```

A espera entre tentativas cresce exponencialmente: 2s → 4s → 8s → 16s → 32s (limitado a 60s).

## Processamento Incremental

Para grandes datasets, use `resume=True` para continuar de onde parou:

```python
# Primeira execução (processa 100 linhas e falha)
df_resultado = dataframeit(df, SuaClasse, TEMPLATE, resume=True)
df_resultado.to_excel('resultado_parcial.xlsx', index=False)

# Segunda execução (continua das linhas não processadas)
df = pd.read_excel('resultado_parcial.xlsx')
df_resultado = dataframeit(df, SuaClasse, TEMPLATE, resume=True)
df_resultado.to_excel('resultado_completo.xlsx', index=False)
```

## Exemplo Completo

Veja o diretório `example/` para um caso de uso completo com análise de decisões judiciais, incluindo:
- Modelo Pydantic complexo com classes aninhadas
- Template detalhado com instruções específicas de domínio
- Uso de campos opcionais e tipos Literal
- Processamento de listas e tuplas

## Configuração de Variáveis de Ambiente

### Para OpenAI
```bash
export OPENAI_API_KEY="sua-chave-openai"
```

### Para Google Gemini (LangChain)
```bash
export GOOGLE_API_KEY="sua-chave-google"
```

### Para Anthropic Claude (LangChain)
```bash
export ANTHROPIC_API_KEY="sua-chave-anthropic"
```

## Contribuições

Contribuições são bem-vindas! Este é um projeto em desenvolvimento inicial.

## Licença

Veja o arquivo LICENSE para detalhes.
