# Referência para LLMs

Esta página contém toda a informação necessária para usar DataFrameIt em uma única página compacta, otimizada para assistentes de código.

---

## O que é

DataFrameIt processa textos em DataFrames usando LLMs e extrai informações estruturadas definidas por modelos Pydantic.

## Instalação

```bash
pip install dataframeit[google]    # Google Gemini (padrão)
pip install dataframeit[openai]    # OpenAI
pip install dataframeit[anthropic] # Anthropic Claude
```

**Variáveis de ambiente:**
```bash
export GOOGLE_API_KEY="..."     # Para Gemini
export OPENAI_API_KEY="..."     # Para OpenAI
export ANTHROPIC_API_KEY="..."  # Para Anthropic
```

---

## Assinatura da Função

```python
from dataframeit import dataframeit

resultado = dataframeit(
    data,                    # DataFrame, Series, list ou dict
    questions,               # Modelo Pydantic
    prompt,                  # Template do prompt
    text_column='texto',     # Coluna com textos (obrigatório para DataFrame)
    model='gemini-3.0-flash',
    provider='google_genai', # 'google_genai', 'openai', 'anthropic'
    resume=True,             # Continua de onde parou
    parallel_requests=1,     # Workers paralelos
    rate_limit_delay=0.0,    # Delay entre requisições (segundos)
    max_retries=3,           # Tentativas em caso de erro
    track_tokens=True,       # Rastreia uso de tokens
    api_key=None,            # API key (usa env var se None)
    model_kwargs=None,       # Parâmetros extras (temperature, etc)
    # Busca web (requer TAVILY_API_KEY)
    use_search=False,        # Habilita busca web
    search_per_field=False,  # Busca separada por campo
    max_results=5,           # Resultados por busca
    search_depth='basic',    # 'basic' ou 'advanced'
)
```

---

## Exemplo Completo

```python
from pydantic import BaseModel, Field
from typing import Literal, List, Optional
import pandas as pd
from dataframeit import dataframeit

# 1. Definir modelo Pydantic
class Analise(BaseModel):
    sentimento: Literal['positivo', 'negativo', 'neutro']
    confianca: Literal['alta', 'media', 'baixa']
    temas: List[str] = Field(description="Temas principais")
    resumo: str = Field(description="Resumo em uma frase")

# 2. Dados
df = pd.DataFrame({
    'texto': [
        'Produto excelente! Entrega rápida.',
        'Péssimo atendimento, demorou muito.',
        'Ok, nada de especial.'
    ]
})

# 3. Processar
resultado = dataframeit(
    df,
    Analise,
    "Analise o texto e extraia as informações solicitadas."
)

# 4. Resultado contém colunas: texto, sentimento, confianca, temas, resumo
print(resultado)
```

---

## Tipos de Entrada Suportados

```python
# DataFrame (precisa text_column)
df = pd.DataFrame({'texto': ['A', 'B']})
resultado = dataframeit(df, Model, PROMPT)

# Lista (não precisa text_column)
textos = ['Texto 1', 'Texto 2']
resultado = dataframeit(textos, Model, PROMPT)

# Dicionário (chaves viram índice)
docs = {'id1': 'Texto 1', 'id2': 'Texto 2'}
resultado = dataframeit(docs, Model, PROMPT)

# Series (preserva índice)
series = pd.Series(['A', 'B'], index=['x', 'y'])
resultado = dataframeit(series, Model, PROMPT)
```

---

## Modelos Pydantic

```python
from pydantic import BaseModel, Field
from typing import Literal, List, Optional

# Campos com valores fixos
class Exemplo(BaseModel):
    categoria: Literal['A', 'B', 'C']

# Campos opcionais
class Exemplo(BaseModel):
    nota: Optional[str] = Field(default=None, description="Observações")

# Listas
class Exemplo(BaseModel):
    tags: List[str] = Field(description="Lista de tags")

# Modelos aninhados
class Endereco(BaseModel):
    cidade: str
    estado: str

class Pessoa(BaseModel):
    nome: str
    endereco: Optional[Endereco] = None
```

---

## Providers

```python
# Google Gemini (padrão)
resultado = dataframeit(df, Model, PROMPT)

# OpenAI
resultado = dataframeit(
    df, Model, PROMPT,
    provider='openai',
    model='gpt-5.2-mini'
)

# Anthropic
resultado = dataframeit(
    df, Model, PROMPT,
    provider='anthropic',
    model='claude-sonnet-4.5'
)

# Com parâmetros extras
resultado = dataframeit(
    df, Model, PROMPT,
    provider='openai',
    model='gpt-5.2-mini',
    model_kwargs={'temperature': 0.2}
)
```

---

## Performance

```python
# Processamento paralelo
resultado = dataframeit(
    df, Model, PROMPT,
    parallel_requests=5  # 5 workers simultâneos
)

# Rate limiting (previne erro 429)
resultado = dataframeit(
    df, Model, PROMPT,
    rate_limit_delay=1.0  # 1 segundo entre requisições
)

# Combinados
resultado = dataframeit(
    df, Model, PROMPT,
    parallel_requests=5,
    rate_limit_delay=0.5
)
```

---

## Tratamento de Erros

```python
resultado = dataframeit(df, Model, PROMPT, max_retries=5)

# Verificar erros
erros = resultado[resultado['_dataframeit_status'] == 'error']
print(erros['_error_details'])

# Filtrar sucesso
sucesso = resultado[resultado['_dataframeit_status'] == 'processed']
```

---

## Colunas Adicionadas Automaticamente

| Coluna | Descrição |
|--------|-----------|
| `_dataframeit_status` | `'processed'`, `'error'`, `None` |
| `_error_details` | Mensagem de erro |
| `_input_tokens` | Tokens de entrada |
| `_output_tokens` | Tokens de saída |
| `_total_tokens` | Total de tokens |

---

## Processamento Incremental

```python
# Processa e salva
resultado = dataframeit(df, Model, PROMPT, resume=True)
resultado.to_excel('parcial.xlsx', index=False)

# Carrega e continua
df = pd.read_excel('parcial.xlsx')
resultado = dataframeit(df, Model, PROMPT, resume=True)
```

---

## Prompt Template

```python
# Simples - texto adicionado ao final
PROMPT = "Classifique o sentimento do texto."

# Com placeholder - controle a posição
PROMPT = """
Analise o documento abaixo:

{texto}

Extraia as informações solicitadas.
"""
```
