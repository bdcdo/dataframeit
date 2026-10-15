# API Reference

Complete documentation of all public functions and classes.

## dataframeit()

Main function to process texts with LLMs.

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
    # Web search parameters
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

### Parameters

#### Data

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `data` | DataFrame, Series, list, dict | Yes | Data containing texts to process |
| `questions` | Pydantic BaseModel | Yes | Pydantic model defining fields to extract (`perguntas` is accepted as a deprecated name for the same parameter) |
| `prompt` | str | Yes | Prompt template. Use `{texto}` to position text |
| `text_column` | str | No | Column name with texts. If `None`, tries `texto`, `text`, `decisao`, `content`, `content_text` in order (or the single column if the DataFrame has only one). With no candidate and several columns, raises `ValueError` |

#### Processing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `resume` | bool | `True` | With `True`, continues from where it stopped: processes only rows without status and keeps those with `'processed'` or `'error'`. With `False`, processes every row that is not `'processed'`; if the model columns already exist and `reprocess_columns` is not passed, it emits a warning and returns the data unprocessed |
| `reprocess_columns` | list | `None` | Fields to force reprocessing; when resuming with a changed model, it must cover fields incompatible with previously processed rows |
| `status_column` | str | `None` | Name of the status column; `None` uses `_dataframeit_status` |

#### Model

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str \| None | `None` | LLM model name; `None` uses the provider's default model, listed in [Providers](../guides/providers.md) |
| `provider` | str | `'openai'` | Provider identifier; `claude_code` and `codex` use the official SDKs instead of LangChain (see [Providers](../guides/providers.md)) |
| `api_key` | str | `None` | API key (uses env var if None); not accepted with `provider='codex'` and ignored with `provider='claude_code'` |
| `model_kwargs` | dict | `None` | Extra parameters; with `claude_code`, only `max_turns`, `max_budget_usd` and `effort` are read and the rest is ignored; with `codex`, only `effort` and `timeout` are accepted |

#### Resilience

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_retries` | int | `3` | Total attempts per row, counting the first one (int >= 1) |
| `base_delay` | float | `1.0` | Wait before the first retry (seconds); doubles on each attempt |
| `max_delay` | float | `30.0` | Ceiling for the wait between attempts (seconds) |
| `rate_limit_delay` | float | `0.0` | Pause each worker takes after each successful row (seconds) |

#### Performance

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `parallel_requests` | int | `1` | Parallel workers (1 = sequential) |
| `track_tokens` | bool | `True` | Track token usage |
| `batch_size` | int | `None` | Save checkpoint every N processed rows (requires `checkpoint_path`) |
| `checkpoint_path` | str \| Path | `None` | Checkpoint file destination; extension sets format (`.csv`, `.xlsx`, `.parquet`) |

#### Web Search

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_search` | bool | `False` | Enable web search; not supported with `provider='claude_code'` or `'codex'` |
| `search_provider` | str | `'tavily'` | `'tavily'` (requires `TAVILY_API_KEY`) or `'exa'` (requires `EXA_API_KEY`) |
| `search_per_field` | bool | `False` | Run a separate agent per field; required for `condition` and `search_groups` |
| `max_results` | int | `5` | Results per search (1-20) |
| `search_depth` | str | `'basic'` | `'basic'` (1 credit) or `'advanced'` (2 credits); Tavily only |
| `max_search_calls` | int | `10` | Maximum searches per agent run; later ones are blocked and the agent answers with what it found |
| `search_groups` | dict | `None` | Groups of fields that share one search: `{"group": {"fields": [...], "prompt": ..., "max_results": ..., "search_depth": ..., "max_search_calls": ...}}` |
| `save_trace` | bool \| str | `None` | Save the agent trace: `True`/`"full"` or `"minimal"`; requires `use_search=True`; creates `_trace`, `_trace_{field}` or `_trace_{group}` |

With `use_search=True` and `search_per_field=True`, model fields accept their own search settings in `json_schema_extra` (`prompt`, `prompt_replace`, `prompt_append`, `search_depth`, `max_results`, `max_search_calls`, `condition`, `depends_on`). See [Web Search](../guides/web-search.md) and [Conditional Fields](../examples/conditional-fields.md).

### Return

Returns data in the same format as input with extracted columns added.

| Input | Output |
|-------|--------|
| `pd.DataFrame` | `pd.DataFrame` with Pydantic model columns |
| `pl.DataFrame` | `pl.DataFrame` with Pydantic model columns |
| `pd.Series` | `pd.DataFrame` preserving index |
| `pl.Series` | `pl.DataFrame` |
| `list` | `pd.DataFrame` with numeric index |
| `dict` | `pd.DataFrame` with keys as index |

With polars input, the input columns that are not model fields and do not have the name of one of the [added columns](#added-columns) come back as they went in, with the same type and the same values. In polars output, an `Enum` field comes out as the member's value. When the answers of a field do not fit a typed polars column, such as a `Union[int, str]` that received a number in one row and text in another, the column comes out with the `Object` type, holding the Python objects the model returned, except an `Enum` member outside a list or dict, which comes out as its value there too, and a warning names the columns; convert them before writing to a file. A field whose annotation combines `date`, `datetime` or `time` with another type, such as `Union[date, int]`, `tuple[date, int]` or `list[Union[date, datetime]]`, comes out as `Object` when the input has rows, with that warning, because a column type inferred from the first value would convert the others.

A field annotated as `dict`, such as `dict[str, int]` or `Optional[dict[str, Any]]`, comes out in polars output as JSON text, with null where the row has no answer. The text is the same the checkpoint writes, and values that JSON does not represent come out in it as text. The type of the other model columns comes from the answers. A column with no value at all, all null or holding only empty lists, comes out with the type of the annotation when it is `str`, `int`, `float`, `bool`, `date`, `datetime`, or a list of one of them: `String`, `Int64`, `Float64`, `Boolean`, `Date`, `Datetime`, or a `List` of one of these. With any other annotation, the column with no value keeps the type polars infers. With input that has no rows, a model column the input already has keeps its type, and one it does not have comes out as `String`.

### Added Columns

| Column | When it exists | Content |
|--------|----------------|---------|
| `_dataframeit_status` (or the name in `status_column`) | See note below | `'processed'`, `'error'` or `None` (row not yet processed) |
| `_error_details` | See note below | Error message; on a successful row that needed retries, `"Sucesso após N retry(s)"`; on a row with empty text, `"Texto ausente"` |
| `_input_tokens`, `_output_tokens`, `_cached_input_tokens`, `_reasoning_tokens` | `track_tokens=True` | Token usage of the row; see [LLM Reference](llm-reference.md#automatically-added-columns) |
| `_search_credits` | `use_search=True` | Search credits spent on the row |
| `_trace`, `_trace_{field}` or `_trace_{group}` | `save_trace` set | Agent trace in JSON |

When no row ends with an error and none records any detail, `_dataframeit_status` and `_error_details` are removed from the output. So, before filtering by status, check that the column exists:

```python
if '_dataframeit_status' in result.columns:
    errors = result[result['_dataframeit_status'] == 'error']
```

Rows with empty or missing text are not sent to the model: they get status `'error'` and the detail `"Texto ausente"`.

### Examples

```python
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
from dataframeit import dataframeit

class Sentiment(BaseModel):
    sentiment: Literal['positive', 'negative', 'neutral']

df = pd.DataFrame({'text': ['Great!', 'Terrible!']})

# Basic
result = dataframeit(df, Sentiment, "Analyze the sentiment.")

# With configurations
result = dataframeit(
    df,
    Sentiment,
    "Analyze the sentiment.",
    provider='openai',
    model='gpt-6-luna',
    parallel_requests=5,
    rate_limit_delay=0.5,
    max_retries=5
)
```

---

## read_df()

Reads files in various formats to a DataFrame and turns lists, dicts and nested models saved as JSON (or as Python repr) back into their original structures. Use it to reload a checkpoint or a saved result before resuming with `resume=True`.

```python
def read_df(path: str, model=None, normalize: bool = True, **kwargs) -> pd.DataFrame
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | str | File path |
| `model` | type[BaseModel] | Pydantic model. With it, only structured fields are normalized, and in `.csv`, `.xlsx` and `.xls` text fields are read as raw text: `"2023"` does not become a number and `"N/A"` does not become missing. Any extra read argument, from `**kwargs`, turns this off, and text is left as pandas returns it. |
| `normalize` | bool | If `False`, no column is converted |
| `**kwargs` | | Arguments passed to pandas. With one that makes the reader return something other than a DataFrame, such as `chunksize`, `iterator` or `sheet_name=None` or a list, the function returns the pandas object (`TextFileReader`, `JsonReader` or a dict of sheets) without normalizing or rereading any column, even with `model` and `normalize=True` |

CSV and XLSX store `""` and missing values the same way. A required text field that was `""` comes back missing, and resuming flags it for reprocessing; use a `.parquet` checkpoint to keep that difference.

Called with `model` and no extra read arguments, the `**kwargs` passed to pandas, the function rereads from `.csv` and `.parquet` the column of an `int` or `Optional[int]` field that pandas does not return as an integer dtype: it comes back as `object`, with the exact `int` and `None` in empty cells, instead of float64 with `NaN`, which rounds integers above 2**53. A cell the file stores as a float, such as `7.5`, stays as read. A nullable `Int64` parquet column stays `Int64`, and a column with only empty cells stays as read: float in CSV and, in the null column the checkpoint writes to parquet, `object` with `None`. An extra read argument turns this reread off; `normalize` does not. `.xlsx`, which stores numbers as doubles, is left out.

### Supported Formats

- `.xlsx`, `.xls` - Excel
- `.csv` - CSV
- `.json` - JSON
- `.parquet` - Parquet

### Example

```python
from dataframeit import read_df

df = read_df('data.xlsx')
df = read_df('data.csv', encoding='utf-8')
```

---

## normalize_value()

Converts into a Python structure a value that was saved as text, as happens when lists and dictionaries are saved to CSV or Excel.

```python
def normalize_value(value: Any) -> Any
```

- A string that starts with `[` or `{` is read as JSON; if it is not valid JSON, it is read as a Python literal (`"['a', 'b']"`), without executing code.
- Lists, dictionaries and tuples come back as they are.
- Any other value, including a string that is not a structure, comes back unchanged.

```python
normalize_value('[1, 2, 3]')      # [1, 2, 3]
normalize_value('{"a": 1}')       # {'a': 1}
normalize_value('plain text')     # 'plain text'
```

---

## normalize_complex_columns()

Applies `normalize_value()` to the given columns, modifying the DataFrame in place.

```python
def normalize_complex_columns(df: pd.DataFrame, complex_fields: set) -> None
```

Columns missing from the DataFrame are ignored.

---

## get_complex_fields()

Identifies the fields of a Pydantic model that hold a structure instead of a simple value.

```python
def get_complex_fields(pydantic_model) -> set
```

Returns the set of names of the fields whose type is `list`, `dict`, `tuple` or a nested Pydantic model, including inside `Optional` and `Union`.

```python
from dataframeit import get_complex_fields, normalize_complex_columns, read_df

df = read_df('output.csv')
normalize_complex_columns(df, get_complex_fields(MyModel))
```

`read_df(path, model=MyModel)` does this normalization on its own.

---

## field_condition()

Builds the `json_schema_extra` of a field with a callable condition, for [conditional fields](../examples/conditional-fields.md).

```python
def field_condition(
    condition: Callable[[dict[str, Any]], bool],
    depends_on: str | list[str] | tuple[str, ...],
) -> JsonDict
```

- `condition` receives the row's fields already filled and returns a bool; the field is only requested from the LLM when it returns true.
- `depends_on` names the fields the function reads, which are extracted before it: a field name as a string, or a list or tuple of names.
- Returns `{'condition': condition, 'depends_on': [...]}`, with a type that Pydantic's `Field` accepts under a type checker.
- Raises `ValueError` if `condition` is not a function, or if `depends_on` is not a string, list or tuple, or is empty.

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

## Exceptions

The exceptions that dataframeit raises are listed in [Exceptions](exceptions.md).

## Version

```python
import dataframeit
dataframeit.__version__
```
