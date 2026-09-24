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
| `questions` | Pydantic BaseModel | Yes | Pydantic model defining fields to extract |
| `prompt` | str | Yes | Prompt template. Use `{texto}` to position text |
| `text_column` | str | No | Column name with texts. If `None`, tries `texto`, `text`, `decisao`, `content`, `content_text` in order (or the single column if the DataFrame has only one). With no candidate and several columns, raises `ValueError` |

#### Processing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `resume` | bool | `True` | Continue from where it stopped: processes only rows without status, without retrying rows marked `'error'` |
| `reprocess_columns` | list | `None` | Fields to force reprocessing; when resuming with a changed model, it must cover fields incompatible with previously processed rows |
| `status_column` | str | `None` | Custom name for status column |

#### Model

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str \| None | `None` | LLM model name; `None` uses the provider's default model, listed in [Providers](../guides/providers.md) |
| `provider` | str | `'openai'` | Provider identifier; `claude_code` and `codex` use the official SDKs instead of LangChain (see [Providers](../guides/providers.md)) |
| `api_key` | str | `None` | API key (uses env var if None); not accepted with `provider='codex'` and ignored with `provider='claude_code'` |
| `model_kwargs` | dict | `None` | Extra parameters; with `claude_code`, only `max_turns`, `max_budget_usd` and `effort` are read and the rest is ignored; with `codex`, only `effort` is accepted |

#### Resilience

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_retries` | int | `3` | Maximum attempts per row |
| `base_delay` | float | `1.0` | Initial retry delay (seconds) |
| `max_delay` | float | `30.0` | Maximum retry delay (seconds) |
| `rate_limit_delay` | float | `0.0` | Delay between requests (seconds) |

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

### Added Columns

The status columns below exist independently of token tracking. When `track_tokens=True`, see the [LLM Reference](llm-reference.md#automatically-added-columns) for the usage columns and their semantics.

| Column | Description |
|--------|-------------|
| `_dataframeit_status` | `'processed'`, `'error'`, or `None` |
| `_error_details` | Error details (when applicable) |

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
result = dataframeit(df, Sentiment, "Analyze the sentiment.", text_column='text')

# With configurations
result = dataframeit(
    df,
    Sentiment,
    "Analyze the sentiment.",
    text_column='text',
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
| `model` | type[BaseModel] | Pydantic model. With it, only structured fields are normalized, and in `.csv`/`.xlsx` text fields are read as raw text: `"2023"` does not become a number and `"N/A"` does not become missing. Passing `dtype`, `converters`, `na_values`, `keep_default_na`, `na_filter` or `usecols` turns this off. |
| `normalize` | bool | If `False`, no column is converted |
| `**kwargs` | | Arguments passed to pandas |

CSV and XLSX store `""` and missing values the same way. A required text field that was `""` comes back missing, and resuming flags it for reprocessing; use a `.parquet` checkpoint to keep that difference.

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

Normalizes Python values to pandas-compatible types.

```python
def normalize_value(value: Any) -> Any
```

Converts:
- `tuple` → `list`
- Pydantic objects → `dict`
- Nested values recursively

---

## normalize_complex_columns()

Normalizes columns with complex types in a DataFrame.

```python
def normalize_complex_columns(df: pd.DataFrame, complex_fields: list) -> pd.DataFrame
```

---

## get_complex_fields()

Identifies complex fields in a Pydantic model.

```python
def get_complex_fields(pydantic_model) -> list[str]
```

Returns list of field names containing `List`, `Tuple`, or nested models.
