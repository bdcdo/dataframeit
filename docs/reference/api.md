# Referência da API

Documentação completa de todas as funções e classes públicas.

## dataframeit()

Função principal para processar textos com LLMs.

```python
def dataframeit(
    data,
    questions,
    prompt,
    resume=True,
    reprocess_columns=None,
    model='gemini-3.0-flash',
    provider='google_genai',
    status_column=None,
    text_column=None,
    api_key=None,
    max_retries=3,
    base_delay=1.0,
    max_delay=30.0,
    rate_limit_delay=0.0,
    track_tokens=True,
    model_kwargs=None,
    parallel_requests=1,
    # Parâmetros de busca web
    use_search=False,
    search_per_field=False,
    max_results=5,
    search_depth="basic",
) -> Any
```

### Parâmetros

#### Dados

| Parâmetro | Tipo | Obrigatório | Descrição |
|-----------|------|-------------|-----------|
| `data` | DataFrame, Series, list, dict | Sim | Dados contendo os textos a processar |
| `questions` | Pydantic BaseModel | Sim | Modelo Pydantic definindo campos a extrair |
| `prompt` | str | Sim | Template do prompt. Use `{texto}` para posicionar o texto |
| `text_column` | str | DataFrame: Sim | Nome da coluna com textos. Padrão: `'texto'` |

#### Processamento

| Parâmetro | Tipo | Padrão | Descrição |
|-----------|------|--------|-----------|
| `resume` | bool | `True` | Continua de onde parou (pula linhas já processadas) |
| `reprocess_columns` | list | `None` | Lista de colunas para forçar reprocessamento |
| `status_column` | str | `None` | Nome customizado para coluna de status |

#### Modelo

| Parâmetro | Tipo | Padrão | Descrição |
|-----------|------|--------|-----------|
| `model` | str | `'gemini-3.0-flash'` | Nome do modelo LLM |
| `provider` | str | `'google_genai'` | Provider LangChain |
| `api_key` | str | `None` | API key (usa env var se None) |
| `model_kwargs` | dict | `None` | Parâmetros extras (temperature, etc.) |

#### Resiliência

| Parâmetro | Tipo | Padrão | Descrição |
|-----------|------|--------|-----------|
| `max_retries` | int | `3` | Máximo de tentativas por linha |
| `base_delay` | float | `1.0` | Delay inicial para retry (segundos) |
| `max_delay` | float | `30.0` | Delay máximo para retry (segundos) |
| `rate_limit_delay` | float | `0.0` | Delay entre requisições (segundos) |

#### Performance

| Parâmetro | Tipo | Padrão | Descrição |
|-----------|------|--------|-----------|
| `parallel_requests` | int | `1` | Workers paralelos (1 = sequencial) |
| `track_tokens` | bool | `True` | Rastreia uso de tokens |

#### Busca Web

| Parâmetro | Tipo | Padrão | Descrição |
|-----------|------|--------|-----------|
| `use_search` | bool | `False` | Habilita busca web via Tavily |
| `search_per_field` | bool | `False` | Executa busca separada por campo |
| `max_results` | int | `5` | Resultados por busca (1-20) |
| `search_depth` | str | `'basic'` | `'basic'` ou `'advanced'` |

### Retorno

Retorna dados no mesmo formato da entrada com colunas extraídas adicionadas.

| Entrada | Saída |
|---------|-------|
| `pd.DataFrame` | `pd.DataFrame` com colunas do modelo Pydantic |
| `pl.DataFrame` | `pl.DataFrame` com colunas do modelo Pydantic |
| `pd.Series` | `pd.DataFrame` preservando índice |
| `pl.Series` | `pl.DataFrame` |
| `list` | `pd.DataFrame` com índice numérico |
| `dict` | `pd.DataFrame` com chaves como índice |

### Colunas Adicionadas

| Coluna | Descrição |
|--------|-----------|
| `_dataframeit_status` | `'processed'`, `'error'`, ou `None` |
| `_error_details` | Detalhes do erro (quando aplicável) |
| `_input_tokens` | Tokens de entrada (se `track_tokens=True`) |
| `_output_tokens` | Tokens de saída (se `track_tokens=True`) |
| `_total_tokens` | Total de tokens (se `track_tokens=True`) |

### Exemplos

```python
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
from dataframeit import dataframeit

class Sentimento(BaseModel):
    sentimento: Literal['positivo', 'negativo', 'neutro']

df = pd.DataFrame({'texto': ['Ótimo!', 'Péssimo!']})

# Básico
resultado = dataframeit(df, Sentimento, "Analise o sentimento.")

# Com configurações
resultado = dataframeit(
    df,
    Sentimento,
    "Analise o sentimento.",
    provider='openai',
    model='gpt-4o-mini',
    parallel_requests=5,
    rate_limit_delay=0.5,
    max_retries=5
)
```

---

## read_df()

Lê arquivos em diversos formatos para DataFrame.

```python
def read_df(path: str, **kwargs) -> pd.DataFrame
```

### Parâmetros

| Parâmetro | Tipo | Descrição |
|-----------|------|-----------|
| `path` | str | Caminho do arquivo |
| `**kwargs` | | Argumentos passados para pandas |

### Formatos Suportados

- `.xlsx`, `.xls` - Excel
- `.csv` - CSV
- `.json` - JSON
- `.parquet` - Parquet

### Exemplo

```python
from dataframeit import read_df

df = read_df('dados.xlsx')
df = read_df('dados.csv', encoding='utf-8')
```

---

## normalize_value()

Normaliza valores Python para tipos compatíveis com pandas.

```python
def normalize_value(value: Any) -> Any
```

Converte:
- `tuple` → `list`
- Objetos Pydantic → `dict`
- Valores aninhados recursivamente

---

## normalize_complex_columns()

Normaliza colunas com tipos complexos em um DataFrame.

```python
def normalize_complex_columns(df: pd.DataFrame, complex_fields: list) -> pd.DataFrame
```

---

## get_complex_fields()

Identifica campos complexos em um modelo Pydantic.

```python
def get_complex_fields(pydantic_model) -> list[str]
```

Retorna lista de nomes de campos que contêm `List`, `Tuple`, ou modelos aninhados.
