# Referência para LLMs

Esta página contém toda a informação necessária para usar DataFrameIt em uma única página compacta, otimizada para assistentes de código.

---

## O que é

DataFrameIt processa textos em DataFrames usando LLMs e extrai informações estruturadas definidas por modelos Pydantic.

## Instalação

```bash
pip install dataframeit[openai]       # OpenAI (padrão)
pip install dataframeit[google]       # Google Gemini
pip install dataframeit[anthropic]    # Anthropic Claude
pip install dataframeit[groq]         # Groq
pip install dataframeit[codex]        # Codex SDK oficial (experimental)
pip install dataframeit[claude-code]  # Claude Code pelo Claude Agent SDK
pip install dataframeit[search]       # Busca web com Tavily
pip install dataframeit[search-exa]   # Busca web com Exa
pip install dataframeit[polars]       # Entrada e saída em polars (inclui pyarrow, para .parquet)
pip install dataframeit[excel]        # Leitura e checkpoint em .xlsx
```

**Variáveis de ambiente:**
```bash
export OPENAI_API_KEY="..."     # Para OpenAI
export GOOGLE_API_KEY="..."     # Para Gemini
export ANTHROPIC_API_KEY="..."  # Para Anthropic
export GROQ_API_KEY="..."       # Para Groq
export TAVILY_API_KEY="..."     # Para busca com Tavily
export EXA_API_KEY="..."        # Para busca com Exa
```

O provider `codex` é opcional, não faz parte do extra `all`, usa o runtime empacotado e requer autenticação local em arquivo, sem `OPENAI_API_KEY`. Consulte [Instalação](../getting-started/installation.md) para configurar o extra e as credenciais.

---

## Assinatura da Função

```python
from dataframeit import dataframeit

resultado = dataframeit(
    data,                    # DataFrame, Series, list ou dict
    questions,               # Modelo Pydantic
    prompt,                  # Template do prompt
    text_column=None,        # Coluna com textos (None = inferência automática)
    model=None,              # None = modelo padrão do provider
    provider='openai',       # 'openai', 'google_genai', 'anthropic', 'groq', 'claude_code', 'codex'
    resume=True,             # Continua de onde parou
    reprocess_columns=None,  # Colunas a refazer mesmo em linhas já processadas
    status_column=None,      # None = '_dataframeit_status'
    parallel_requests=1,     # Workers paralelos
    rate_limit_delay=0.0,    # Pausa de cada worker depois de cada linha bem-sucedida (segundos)
    max_retries=3,           # Total de tentativas por linha, contando a primeira
    base_delay=1.0,          # Espera antes da primeira nova tentativa; dobra a cada uma
    max_delay=30.0,          # Teto da espera entre tentativas
    track_tokens=True,       # Rastreia uso de tokens
    api_key=None,            # API key (usa env var se None)
    model_kwargs=None,       # Parâmetros extras (temperature, etc)
    batch_size=None,         # Salva checkpoint a cada N linhas
    checkpoint_path=None,    # Arquivo do checkpoint (.csv, .xlsx, .parquet)
    # Busca web (requer TAVILY_API_KEY ou EXA_API_KEY)
    use_search=False,        # Habilita busca web
    search_provider='tavily',  # 'tavily' ou 'exa'
    search_per_field=False,  # Busca separada por campo
    max_results=5,           # Resultados por busca
    search_depth='basic',    # 'basic' ou 'advanced'
    max_search_calls=10,     # Máximo de buscas por execução do agente
    search_groups=None,      # Campos que compartilham uma busca
    save_trace=None,         # True/'full' ou 'minimal'
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
# DataFrame (text_column inferida pelo nome; ver referência da API)
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
# OpenAI com gpt-6-luna (padrão)
resultado = dataframeit(df, Model, PROMPT)

# Google Gemini
resultado = dataframeit(
    df, Model, PROMPT,
    provider='google_genai',
    model='gemini-3.8-flash'
)

# Anthropic
resultado = dataframeit(
    df, Model, PROMPT,
    provider='anthropic',
    model='claude-sonnet-5'
)

# Codex SDK oficial (experimental)
resultado = dataframeit(
    df, Model, PROMPT,
    provider='codex',                  # model=None: o runtime escolhe
    model_kwargs={'effort': 'medium'}
)

# Claude Code, pelo Claude Agent SDK
resultado = dataframeit(
    df, Model, PROMPT,
    provider='claude_code',
    model='haiku',
    model_kwargs={'max_budget_usd': 0.25}
)

# Com parâmetros extras
resultado = dataframeit(
    df, Model, PROMPT,
    provider='openai',
    model_kwargs={'temperature': 0.2}
)
```

O provider `codex` aceita somente `effort` em `model_kwargs` e não suporta `use_search=True`. A integração desativa busca web, shell e servidores MCP, nega aprovações e usa sandbox somente leitura para bloquear escrita; o runtime ainda pode apresentar utilitários internos, como `apply_patch`, sem conceder permissão para alterar arquivos. Consulte [Instalação](../getting-started/installation.md) para os requisitos de runtime e autenticação.

O provider `claude_code` usa a autenticação do Claude Code (credenciais de um login do Claude Code na máquina, ou `ANTHROPIC_API_KEY`) e ignora `api_key`. Em `model_kwargs`, lê só `max_turns`, `max_budget_usd` (teto por tentativa) e `effort`. Não suporta `use_search=True`. Roda sem ferramentas e sem os settings e servidores MCP do usuário. Ver [Provedores](../guides/providers.md#claude-code).

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
    rate_limit_delay=1.0  # cada worker pausa 1 segundo depois de cada linha
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

# As colunas de status só existem se alguma linha falhou ou registrou detalhe
if '_dataframeit_status' in resultado.columns:
    erros = resultado[resultado['_dataframeit_status'] == 'error']
    print(erros['_error_details'])
```

Falhas de configuração (extra faltando, parâmetro inválido) levantam exceção antes da primeira chamada. Falhas numa linha não interrompem a execução: a linha fica com status `'error'` e o motivo em `_error_details`. Ver [Exceções](exceptions.md).

---

## Colunas Adicionadas Automaticamente

Com `track_tokens=True`, o DataFrameIt cria `_input_tokens`, `_cached_input_tokens`, `_output_tokens` e `_reasoning_tokens` para todos os providers. Sem telemetria de uso, esses valores podem permanecer nulos; quando o provider informa uso total, mas não informa cache ou raciocínio, a métrica correspondente fica em zero. Tokens de cache são uma parcela do total de entrada, e tokens de raciocínio são uma parcela do total de saída.

| Coluna | Descrição |
|--------|-----------|
| `_dataframeit_status` | `'processed'`, `'error'`, `None`; removida quando nenhuma linha falhou nem registrou detalhe |
| `_error_details` | Mensagem de erro, `"Sucesso após N retry(s)"` ou `"Texto ausente"`; removida junto com a de status |
| `_input_tokens` | Tokens de entrada (com `track_tokens=True`) |
| `_cached_input_tokens` | Parcela da entrada atendida por cache (com `track_tokens=True`) |
| `_output_tokens` | Tokens de saída (com `track_tokens=True`) |
| `_reasoning_tokens` | Parcela da saída usada em raciocínio (com `track_tokens=True`) |
| `_search_credits` | Créditos de busca gastos na linha (com `use_search=True`) |
| `_trace`, `_trace_{campo}`, `_trace_{grupo}` | Trace do agente em JSON (com `save_trace`) |

---

## Processamento Incremental

```python
# Checkpoint automático a cada 100 linhas
resultado = dataframeit(
    df, Model, PROMPT,
    batch_size=100,
    checkpoint_path='parcial.parquet',
)

# Se a execução parar, recarregue o checkpoint e continue
from dataframeit import read_df

df = read_df('parcial.parquet', Model)
resultado = dataframeit(df, Model, PROMPT, resume=True)
```

Com `resume=True` (padrão), linhas com status `'error'` não são refeitas. Para refazê-las, limpe o status delas antes de rodar de novo.

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
