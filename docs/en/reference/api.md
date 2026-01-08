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
    # Web search parameters
    use_search=False,
    search_per_field=False,
    max_results=5,
    search_depth="basic",
) -> Any
```

### Parameters

#### Data

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `data` | DataFrame, Series, list, dict | Yes | Data containing texts to process |
| `questions` | Pydantic BaseModel | Yes | Pydantic model defining fields to extract |
| `prompt` | str | Yes | Prompt template. Use `{texto}` to position text |
| `text_column` | str | DataFrame: Yes | Column name with texts. Default: `'texto'` |

#### Processing

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `resume` | bool | `True` | Continue from where it stopped (skips processed rows) |
| `reprocess_columns` | list | `None` | List of columns to force reprocessing |
| `status_column` | str | `None` | Custom name for status column |

#### Model

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | `'gemini-3.0-flash'` | LLM model name |
| `provider` | str | `'google_genai'` | LangChain provider |
| `api_key` | str | `None` | API key (uses env var if None) |
| `model_kwargs` | dict | `None` | Extra parameters (temperature, etc.) |

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

#### Web Search

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_search` | bool | `False` | Enable web search via Tavily |
| `search_per_field` | bool | `False` | Execute separate search per field |
| `max_results` | int | `5` | Results per search (1-20) |
| `search_depth` | str | `'basic'` | `'basic'` or `'advanced'` |

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

| Column | Description |
|--------|-------------|
| `_dataframeit_status` | `'processed'`, `'error'`, or `None` |
| `_error_details` | Error details (when applicable) |
| `_input_tokens` | Input tokens (if `track_tokens=True`) |
| `_output_tokens` | Output tokens (if `track_tokens=True`) |
| `_total_tokens` | Total tokens (if `track_tokens=True`) |

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
    model='gpt-4o-mini',
    parallel_requests=5,
    rate_limit_delay=0.5,
    max_retries=5
)
```

---

## read_df()

Reads files in various formats to DataFrame.

```python
def read_df(path: str, **kwargs) -> pd.DataFrame
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | str | File path |
| `**kwargs` | | Arguments passed to pandas |

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
