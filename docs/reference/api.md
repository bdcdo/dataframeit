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
    model=None,
    provider='openai',
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
    search_provider="tavily",
    search_per_field=False,
    max_results=5,
    search_depth="basic",
    max_search_calls=10,
    search_groups=None,
    save_trace=None,
    batch_size=None,
    checkpoint_path=None,
) -> Any
```

### Parâmetros

#### Dados

| Parâmetro | Tipo | Obrigatório | Descrição |
|-----------|------|-------------|-----------|
| `data` | DataFrame, Series, list, dict | Sim | Dados contendo os textos a processar |
| `questions` | Pydantic BaseModel | Sim | Modelo Pydantic definindo campos a extrair (`perguntas` é aceito como nome obsoleto do mesmo parâmetro) |
| `prompt` | str | Sim | Template do prompt. Use `{texto}` para posicionar o texto |
| `text_column` | str | Não | Nome da coluna com textos. Se `None`, tenta `texto`, `text`, `decisao`, `content`, `content_text` na ordem (ou a única coluna, se o DataFrame tiver apenas uma). Sem candidato e com várias colunas, levanta `ValueError` |

#### Processamento

| Parâmetro | Tipo | Padrão | Descrição |
|-----------|------|--------|-----------|
| `resume` | bool | `True` | Com `True`, continua de onde parou: processa só as linhas sem status e preserva as que têm `'processed'` ou `'error'`. Com `False`, processa toda linha que não esteja `'processed'`; se as colunas do modelo já existirem e `reprocess_columns` não for passado, emite um aviso e devolve os dados sem processar |
| `reprocess_columns` | list | `None` | Lista de colunas para forçar reprocessamento; ao retomar com modelo alterado, deve cobrir os campos incompatíveis das linhas já processadas |
| `status_column` | str | `None` | Nome da coluna de status; `None` usa `_dataframeit_status` |

#### Modelo

| Parâmetro | Tipo | Padrão | Descrição |
|-----------|------|--------|-----------|
| `model` | str \| None | `None` | Nome do modelo LLM; `None` usa o modelo padrão do provider, listado em [Provedores](../guides/providers.md) |
| `provider` | str | `'openai'` | Identificador do provider; `claude_code` e `codex` usam os SDKs oficiais em vez de LangChain (ver [Provedores](../guides/providers.md)) |
| `api_key` | str | `None` | API key (usa env var se None); não aceito com `provider='codex'` e ignorado com `provider='claude_code'` |
| `model_kwargs` | dict | `None` | Parâmetros extras; com `claude_code`, só `max_turns`, `max_budget_usd` e `effort` são lidos, e o resto é ignorado; com `codex`, apenas `effort` é aceito |

#### Resiliência

| Parâmetro | Tipo | Padrão | Descrição |
|-----------|------|--------|-----------|
| `max_retries` | int | `3` | Total de tentativas por linha, contando a primeira (int >= 1) |
| `base_delay` | float | `1.0` | Espera antes da primeira nova tentativa (segundos); dobra a cada tentativa |
| `max_delay` | float | `30.0` | Teto da espera entre tentativas (segundos) |
| `rate_limit_delay` | float | `0.0` | Pausa de cada worker depois de cada linha bem-sucedida (segundos) |

#### Performance

| Parâmetro | Tipo | Padrão | Descrição |
|-----------|------|--------|-----------|
| `parallel_requests` | int | `1` | Workers paralelos (1 = sequencial) |
| `track_tokens` | bool | `True` | Rastreia uso de tokens |
| `batch_size` | int | `None` | Salva checkpoint a cada N linhas processadas (requer `checkpoint_path`) |
| `checkpoint_path` | str \| Path | `None` | Destino do checkpoint; extensão define formato (`.csv`, `.xlsx`, `.parquet`) |

#### Busca Web

| Parâmetro | Tipo | Padrão | Descrição |
|-----------|------|--------|-----------|
| `use_search` | bool | `False` | Habilita busca web; não suportado com `provider='claude_code'` nem `'codex'` |
| `search_provider` | str | `'tavily'` | `'tavily'` (requer `TAVILY_API_KEY`) ou `'exa'` (requer `EXA_API_KEY`) |
| `search_per_field` | bool | `False` | Executa um agente separado por campo; necessário para `condition` e `search_groups` |
| `max_results` | int | `5` | Resultados por busca (1-20) |
| `search_depth` | str | `'basic'` | `'basic'` (1 crédito) ou `'advanced'` (2 créditos); só Tavily |
| `max_search_calls` | int | `10` | Máximo de buscas por execução do agente; as seguintes são bloqueadas e o agente responde com o que encontrou |
| `search_groups` | dict | `None` | Grupos de campos que compartilham uma busca: `{"grupo": {"fields": [...], "prompt": ..., "max_results": ..., "search_depth": ..., "max_search_calls": ...}}` |
| `save_trace` | bool \| str | `None` | Salva o trace do agente: `True`/`"full"` ou `"minimal"`; requer `use_search=True`; gera `_trace`, `_trace_{campo}` ou `_trace_{grupo}` |

Com `use_search=True` e `search_per_field=True`, os campos do modelo aceitam configuração de busca própria em `json_schema_extra` (`prompt`, `prompt_replace`, `prompt_append`, `search_depth`, `max_results`, `max_search_calls`, `condition`, `depends_on`). Ver [Busca Web](../guides/web-search.md) e [Campos Condicionais](../examples/conditional-fields.md).

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

| Coluna | Quando existe | Conteúdo |
|--------|---------------|----------|
| `_dataframeit_status` (ou o nome em `status_column`) | Ver nota abaixo | `'processed'`, `'error'` ou `None` (linha ainda não processada) |
| `_error_details` | Ver nota abaixo | Mensagem do erro; numa linha bem-sucedida que precisou de novas tentativas, `"Sucesso após N retry(s)"`; numa linha com texto vazio, `"Texto ausente"` |
| `_input_tokens`, `_output_tokens`, `_cached_input_tokens`, `_reasoning_tokens` | `track_tokens=True` | Uso de tokens da linha; ver [Referência para LLMs](llm-reference.md#colunas-adicionadas-automaticamente) |
| `_search_credits` | `use_search=True` | Créditos de busca gastos na linha |
| `_trace`, `_trace_{campo}` ou `_trace_{grupo}` | `save_trace` definido | Trace do agente em JSON |

Quando nenhuma linha termina com erro e nenhuma registra detalhe, `_dataframeit_status` e `_error_details` são removidas da saída. Por isso, antes de filtrar por status, confira se a coluna existe:

```python
if '_dataframeit_status' in resultado.columns:
    erros = resultado[resultado['_dataframeit_status'] == 'error']
```

Linhas com texto vazio ou ausente não vão ao modelo: recebem status `'error'` e o detalhe `"Texto ausente"`.

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
    model='gpt-6-luna',
    parallel_requests=5,
    rate_limit_delay=0.5,
    max_retries=5
)
```

---

## read_df()

Lê arquivos em diversos formatos para DataFrame e devolve listas, dicts e modelos aninhados gravados como JSON (ou como repr Python) às estruturas originais. É a forma de recarregar um checkpoint ou uma saída salva para retomar com `resume=True`.

```python
def read_df(path: str, model=None, normalize: bool = True, **kwargs) -> pd.DataFrame
```

### Parâmetros

| Parâmetro | Tipo | Descrição |
|-----------|------|-----------|
| `path` | str | Caminho do arquivo |
| `model` | type[BaseModel] | Modelo Pydantic. Com ele, só os campos de estrutura são normalizados, e em `.csv`/`.xlsx` os campos de texto são lidos como texto cru: `"2023"` não vira número e `"N/A"` não vira ausência. Passar `dtype`, `converters`, `na_values`, `keep_default_na`, `na_filter` ou `usecols` desliga essa leitura. |
| `normalize` | bool | Se `False`, não converte nenhuma coluna |
| `**kwargs` | | Argumentos passados para pandas |

CSV e XLSX gravam `""` e ausência do mesmo jeito. Um campo obrigatório de texto que era `""` volta ausente, e a retomada o acusa para reprocessar; para guardar essa diferença, use checkpoint `.parquet`.

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

Converte em estrutura Python um valor que foi gravado como texto, como acontece ao salvar listas e dicionários em CSV ou Excel.

```python
def normalize_value(value: Any) -> Any
```

- Uma string que começa com `[` ou `{` é lida como JSON; se não for JSON válido, é lida como literal Python (`"['a', 'b']"`), sem executar código.
- Listas, dicionários e tuplas voltam como estão.
- Qualquer outro valor, inclusive string que não seja estrutura, volta inalterado.

```python
normalize_value('[1, 2, 3]')      # [1, 2, 3]
normalize_value('{"a": 1}')       # {'a': 1}
normalize_value('texto normal')   # 'texto normal'
```

---

## normalize_complex_columns()

Aplica `normalize_value()` às colunas indicadas, alterando o DataFrame no lugar.

```python
def normalize_complex_columns(df: pd.DataFrame, complex_fields: set) -> None
```

Colunas ausentes do DataFrame são ignoradas.

---

## get_complex_fields()

Identifica os campos de um modelo Pydantic que guardam estrutura em vez de valor simples.

```python
def get_complex_fields(pydantic_model) -> set
```

Retorna o conjunto de nomes dos campos cujo tipo é `list`, `dict`, `tuple` ou modelo Pydantic aninhado, inclusive dentro de `Optional` e `Union`.

```python
from dataframeit import get_complex_fields, normalize_complex_columns, read_df

df = read_df('saida.csv')
normalize_complex_columns(df, get_complex_fields(MeuModelo))
```

`read_df(path, model=MeuModelo)` faz essa normalização sozinho.

---

## Exceções

As exceções que o dataframeit levanta estão em [Exceções](exceptions.md).

## Versão

```python
import dataframeit
dataframeit.__version__
```
