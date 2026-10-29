# Referência da API

Documentação completa de todas as funções e classes públicas.

## dataframeit()

Função principal para processar textos com LLMs.

```python
def dataframeit(
    data,
    questions,
    prompt,
    perguntas=None,  # nome obsoleto de questions
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
    on_row_done=None,
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
| `model_kwargs` | dict | `None` | Parâmetros extras; com `claude_code`, só `max_turns`, `max_budget_usd` e `effort` são lidos, e o resto é ignorado; com `codex`, apenas `effort` e `timeout` são aceitos |

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
| `on_row_done` | Callable | `None` | Chamada uma vez por linha processada, inclusive em erro, com um [`RowResult`](#rowresult); na thread do worker quando `parallel_requests` > 1 |

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

Retorna um DataFrame com as colunas extraídas: polars para entrada polars e pandas para as demais.

| Entrada | Saída |
|---------|-------|
| `pd.DataFrame` | `pd.DataFrame` com colunas do modelo Pydantic |
| `pl.DataFrame` | `pl.DataFrame` com colunas do modelo Pydantic |
| `pd.Series` | `pd.DataFrame` preservando índice |
| `pl.Series` | `pl.DataFrame` |
| `list` | `pd.DataFrame` com índice numérico |
| `dict` | `pd.DataFrame` com chaves como índice |

Com entrada polars, as colunas de entrada que não são campo do modelo nem têm nome de uma das [colunas adicionadas](#colunas-adicionadas) voltam como entraram, com o mesmo tipo e os mesmos valores. Na saída polars, um campo `Enum` sai pelo valor do membro. Quando as respostas de um campo não cabem numa coluna tipada do polars, como um `Union[int, str]` que recebeu número numa linha e texto em outra, a coluna sai com o tipo `Object`, com os objetos Python que o modelo devolveu, exceto o membro de `Enum` fora de lista ou dict, que também nela sai pelo valor, e um aviso diz quais colunas são; converta-as antes de gravar em arquivo. O campo cuja anotação junta `date`, `datetime`, `time` ou `timedelta` a outro tipo, `float` a `bool` ou `bytes` a `str`, como `Union[date, int]`, `Union[timedelta, int]`, `Union[float, bool]`, `tuple[date, int]` ou `list[Union[date, datetime]]`, sai `Object` quando a entrada tem linhas, com esse aviso, porque o tipo da coluna inferido pelo primeiro valor converteria os demais. Pela mesma razão, com qualquer anotação, sai `Object` a coluna em que essas combinações aparecem nos valores, no mesmo campo de um modelo aninhado ou nos itens de uma lista, como `Any` com `1.5` numa linha e `True` em outra, a coluna em que os dicts do mesmo campo trazem chaves diferentes, como `Any` com `{"a": 1}` numa linha e `{}` em outra, que como `Struct` teriam a união das chaves, e a coluna com `datetime` com e sem fuso, `time` com fuso ou `Decimal` NaN. O NaN de um campo `float`, ou de uma união com `float`, sai NaN, separado do nulo, também quando a saída volta com `resume=True`. Relido de um [checkpoint](../guides/performance.md), ele conta como ausente, como a linha pendente, porque nenhum dos formatos o guarda à parte do nulo. Em `Any`, nos itens de lista e nos campos de modelo aninhado, o NaN sai nulo; no campo `dict`, que sai em texto JSON, ele fica `NaN` no texto. Por isso, a saída polars de um campo `list[float]`, ou de modelo aninhado com campo `float`, que recebeu NaN tem a linha acusada como incompatível quando volta com `resume=True`, porque o nulo não é `float`, e a retomada pede `reprocess_columns`; a retomada pelo checkpoint, que grava a lista em JSON com o NaN, e a da saída pandas não a acusam. Mudam de tipo sem mudar de valor: o inteiro ao lado de um float na mesma coluna ou lista sai float (`[1.5, 2]` sai `[1.5, 2.0]`), o `Decimal` sai com a escala comum da coluna (`1.5` ao lado de `2.25` sai `1.50`), e o `datetime` com fuso diferente do primeiro sai no fuso dele, no mesmo instante. Na saída, na retomada e em [`read_df()`](#read_df), o `NewType` e o apelido `type` contam pelo tipo a que se referem, inclusive o apelido genérico usado com argumentos, como `Opcional[int]` de `type Opcional[T] = T | None`.

Um campo anotado como `dict`, como `dict[str, int]` ou `Optional[dict[str, Any]]`, sai na saída polars como texto JSON, com nulo onde a linha não tem resposta. O texto é o mesmo que o checkpoint grava, e os valores que o JSON não representa saem nele como texto. O tipo das demais colunas do modelo vem das respostas. A coluna sem nenhum valor, toda nula ou só com listas vazias, sai com o tipo da anotação quando ela é `str`, `int`, `float`, `bool`, `date`, `datetime` ou lista de um deles: `String`, `Int64`, `Float64`, `Boolean`, `Date`, `Datetime` ou `List` de um desses. Com outra anotação, a coluna sem valor fica com o tipo que o polars infere. Na entrada sem linhas, a coluna do modelo que a entrada já traz mantém o tipo dela, e a que ela não traz sai `String`.

### RowResult

Dataclass imutável que `on_row_done` recebe a cada linha processada nesta chamada, depois de a linha estar gravada no DataFrame. A linha pulada por `resume` não é entregue.

| Campo | Tipo | Conteúdo |
|-------|------|----------|
| `index` | Any | Rótulo da linha no índice do DataFrame |
| `position` | int | Ordem, de 0, entre as linhas processadas nesta chamada |
| `success` | bool | `True` quando a linha ficou `'processed'` |
| `error` | str \| None | Texto gravado em `_error_details`; numa linha bem-sucedida pode registrar as tentativas repetidas |
| `values` | dict | Campos do modelo que esta chamada gravou na linha, só os de `reprocess_columns` numa linha reprocessada; vazio em erro |
| `elapsed_seconds` | float | Segundos do início do processamento da linha, com as tentativas repetidas e as pausas de rate limit |

Com `parallel_requests` > 1, a chamada vem da thread do worker que processou a linha, enquanto as outras seguem gravando; o callback sincroniza o que compartilhar e lê o que precisa do `RowResult`, não do DataFrame. Uma exceção dentro dele vira aviso, e a execução segue. Ver [Performance](../guides/performance.md#callback-por-linha).

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

### Entrada modificada no lugar

Com entrada `pd.DataFrame`, o `dataframeit()` grava no próprio DataFrame que recebe as colunas do modelo e as [colunas adicionadas](#colunas-adicionadas), inclusive `_dataframeit_status`, que fica nele mesmo quando a saída a remove. Por isso uma interrupção deixa no DataFrame as linhas já processadas, e `resume=True` continua delas.

Pela mesma razão, uma segunda chamada com o mesmo DataFrame, ou com uma cópia feita depois da primeira, não manda ao modelo as linhas com status `'processed'`. Com um modelo de mesmos campos, ela devolve os resultados da primeira chamada. Com um modelo de outros campos e `resume=True`, o padrão, o resultado depende dos campos que a primeira chamada não preencheu. Se algum deles não admite `None`, como `categoria: str`, a chamada levanta `ValueError` por campos incompatíveis. Se todos admitem `None`, como `Optional[str]`, com ou sem padrão, a chamada não levanta erro nem avisa da diferença de campos, não manda nenhuma linha ao modelo e devolve `None` nesses campos em todas as linhas. O mesmo acontece com o modelo da primeira chamada acrescido de um campo `Optional`. O erro só protege, portanto, quando o campo novo não admite `None`. Para aplicar outro modelo ou outro prompt aos mesmos textos, passe a cada chamada um DataFrame novo ou uma cópia feita antes da primeira:

```python
resultado_a = dataframeit(df.copy(), ModeloA, PROMPT_A)
resultado_b = dataframeit(df.copy(), ModeloB, PROMPT_B)
```

A entrada polars, `pd.Series`, `list` e `dict` não é alterada.

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
| `model` | type[BaseModel] | Modelo Pydantic. Com ele, só os campos de estrutura são normalizados, e em `.csv`, `.xlsx` e `.xls` os campos de texto são lidos como texto cru: `"2023"` não vira número e `"N/A"` não vira ausência. Qualquer argumento de leitura extra, dos `**kwargs`, desliga essa leitura, e o texto fica como o pandas o devolve. |
| `normalize` | bool | Se `False`, não converte nenhuma coluna |
| `**kwargs` | | Argumentos passados para pandas. Com o que faz o leitor devolver outra coisa que um DataFrame, como `chunksize`, `iterator` ou `sheet_name=None` ou lista, a função devolve o objeto do pandas, como `TextFileReader`, `JsonReader`, dict de abas ou a `Series` de `typ="series"`, sem normalizar nem reler nenhuma coluna, mesmo com `model` e `normalize=True` |

CSV e XLSX gravam `""` e ausência do mesmo jeito. Um campo obrigatório de texto que era `""` volta ausente, e a retomada o acusa para reprocessar; para guardar essa diferença, use checkpoint `.parquet`.

Chamada com `model` e sem argumentos de leitura extras, os `**kwargs` que vão ao pandas, a função relê de `.csv` e `.parquet` a coluna de um campo `int`, `Literal` só de inteiros, ou um deles com `None`, que o pandas não devolve em tipo inteiro: ela volta como `object`, com o `int` exato e `None` na célula vazia, em vez de float64 com `NaN`, que arredonda o inteiro acima de 2**53. A célula que o arquivo guarda como float, como `7.5`, fica como lida. A coluna `Int64` anulável do parquet continua `Int64`, e a coluna só com células vazias fica como lida: float no CSV e, na coluna nula que o checkpoint grava no parquet, `object` com `None`. Argumento de leitura extra desliga essa releitura; `normalize` não a desliga. O `.xlsx`, que guarda número como double, fica de fora.

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

## field_condition()

Monta o `json_schema_extra` de um campo com condição callable, para os [campos condicionais](../examples/conditional-fields.md).

```python
def field_condition(
    condition: Callable[[dict[str, Any]], bool],
    depends_on: str | list[str] | tuple[str, ...],
) -> JsonDict
```

- `condition` recebe os campos já preenchidos da linha e devolve bool; o campo só é pedido ao LLM quando ela devolve verdadeiro.
- `depends_on` nomeia os campos lidos pela função, que são extraídos antes dela: o nome de um campo em texto, ou uma lista ou tupla de nomes.
- Retorna `{'condition': condition, 'depends_on': [...]}`, com tipo que o `Field` do Pydantic aceita no checker de tipos.
- Levanta `ValueError` se `condition` não é função, ou se `depends_on` não é texto, lista ou tupla, ou está vazio.

```python
from pydantic import BaseModel, Field
from dataframeit import field_condition

class Pedido(BaseModel):
    tipo_cliente: str
    desconto: float | None = Field(
        default=None,
        json_schema_extra=field_condition(
            lambda dados: dados.get('tipo_cliente') == 'vip',
            depends_on='tipo_cliente',
        ),
    )
```

---

## Exceções

As exceções que o dataframeit levanta estão em [Exceções](exceptions.md).

## Versão

```python
import dataframeit
dataframeit.__version__
```
