# LLM Reference

This page contains all the information needed to use DataFrameIt in a single compact page, optimized for code assistants.

---

## What it is

DataFrameIt processes texts in DataFrames using LLMs and extracts structured information defined by Pydantic models.

## Installation

```bash
pip install dataframeit[openai]       # OpenAI (default)
pip install dataframeit[google]       # Google Gemini
pip install dataframeit[anthropic]    # Anthropic Claude
pip install dataframeit[groq]         # Groq
pip install dataframeit[codex]        # Official Codex SDK (experimental)
pip install dataframeit[claude-code]  # Claude Code via the Claude Agent SDK
pip install dataframeit[search]       # Web search with Tavily
pip install dataframeit[search-exa]   # Web search with Exa
pip install dataframeit[polars]       # Polars input and output (includes pyarrow, for .parquet)
pip install dataframeit[excel]        # Reading and checkpoints in .xlsx
```

**Environment variables:**
```bash
export OPENAI_API_KEY="..."     # For OpenAI
export GOOGLE_API_KEY="..."     # For Gemini
export ANTHROPIC_API_KEY="..."  # For Anthropic
export GROQ_API_KEY="..."       # For Groq
export TAVILY_API_KEY="..."     # For search with Tavily
export EXA_API_KEY="..."        # For search with Exa
```

The `codex` provider is optional, is not included in the `all` extra, uses the bundled runtime, and requires local file-backed authentication without `OPENAI_API_KEY`. See [Installation](../getting-started/installation.md) to configure the extra and credentials.

---

## Function Signature

```python
from dataframeit import dataframeit

result = dataframeit(
    data,                    # DataFrame, Series, list or dict
    questions,               # Pydantic model
    prompt,                  # Prompt template
    text_column=None,        # Column with texts (None = automatic inference)
    model=None,              # None = provider's default model
    provider='openai',       # 'openai', 'google_genai', 'anthropic', 'groq', 'claude_code', 'codex'
    resume=True,             # Continue from where it stopped
    reprocess_columns=None,  # Columns to redo even on rows already processed
    status_column=None,      # None = '_dataframeit_status'
    parallel_requests=1,     # Parallel workers
    rate_limit_delay=0.0,    # Pause of each worker after each successful row (seconds)
    max_retries=3,           # Total attempts per row, counting the first one
    base_delay=1.0,          # Wait before the first retry; doubles on each one
    max_delay=30.0,          # Ceiling for the wait between attempts
    track_tokens=True,       # Track token usage
    api_key=None,            # API key (uses env var if None)
    model_kwargs=None,       # Extra parameters (temperature, etc)
    batch_size=None,         # Save a checkpoint every N rows
    checkpoint_path=None,    # Checkpoint file (.csv, .xlsx, .parquet)
    # Web search (requires TAVILY_API_KEY or EXA_API_KEY)
    use_search=False,        # Enable web search
    search_provider='tavily',  # 'tavily' or 'exa'
    search_per_field=False,  # Separate search per field
    max_results=5,           # Results per search
    search_depth='basic',    # 'basic' or 'advanced'
    max_search_calls=10,     # Maximum searches per agent run
    search_groups=None,      # Fields that share one search
    save_trace=None,         # True/'full' or 'minimal'
)
```

---

## Complete Example

```python
from pydantic import BaseModel, Field
from typing import Literal, List, Optional
import pandas as pd
from dataframeit import dataframeit

# 1. Define Pydantic model
class Analysis(BaseModel):
    sentiment: Literal['positive', 'negative', 'neutral']
    confidence: Literal['high', 'medium', 'low']
    topics: List[str] = Field(description="Main topics")
    summary: str = Field(description="Summary in one sentence")

# 2. Data
df = pd.DataFrame({
    'text': [
        'Excellent product! Fast delivery.',
        'Terrible service, took too long.',
        'Ok, nothing special.'
    ]
})

# 3. Process
result = dataframeit(
    df,
    Analysis,
    "Analyze the text and extract the requested information."
)

# 4. Result contains columns: text, sentiment, confidence, topics, summary
print(result)
```

---

## Supported Input Types

```python
# DataFrame (text_column inferred by name; see the API reference)
df = pd.DataFrame({'text': ['A', 'B']})
result = dataframeit(df, Model, PROMPT)

# List (no text_column needed)
texts = ['Text 1', 'Text 2']
result = dataframeit(texts, Model, PROMPT)

# Dictionary (keys become index)
docs = {'id1': 'Text 1', 'id2': 'Text 2'}
result = dataframeit(docs, Model, PROMPT)

# Series (preserves index)
series = pd.Series(['A', 'B'], index=['x', 'y'])
result = dataframeit(series, Model, PROMPT)
```

---

## Pydantic Models

```python
from pydantic import BaseModel, Field
from typing import Literal, List, Optional

# Fields with fixed values
class Example(BaseModel):
    category: Literal['A', 'B', 'C']

# Optional fields
class Example(BaseModel):
    notes: Optional[str] = Field(default=None, description="Observations")

# Lists
class Example(BaseModel):
    tags: List[str] = Field(description="List of tags")

# Nested models
class Address(BaseModel):
    city: str
    state: str

class Person(BaseModel):
    name: str
    address: Optional[Address] = None
```

---

## Providers

```python
# OpenAI with gpt-6-luna (default)
result = dataframeit(df, Model, PROMPT)

# Google Gemini
result = dataframeit(
    df, Model, PROMPT,
    provider='google_genai',
    model='gemini-3.8-flash'
)

# Anthropic
result = dataframeit(
    df, Model, PROMPT,
    provider='anthropic',
    model='claude-sonnet-5'
)

# Official Codex SDK (experimental)
result = dataframeit(
    df, Model, PROMPT,
    provider='codex',                  # model=None: the runtime chooses
    model_kwargs={'effort': 'medium'}
)

# Claude Code, via the Claude Agent SDK
result = dataframeit(
    df, Model, PROMPT,
    provider='claude_code',
    model='haiku',
    model_kwargs={'max_budget_usd': 0.25}
)

# With extra parameters
result = dataframeit(
    df, Model, PROMPT,
    provider='openai',
    model_kwargs={'temperature': 0.2}
)
```

The `codex` provider accepts only `effort` in `model_kwargs` and does not support `use_search=True`. The integration disables web search, shell access, and MCP servers, denies approvals, and uses a read-only sandbox to block writes; the runtime may still present internal utilities such as `apply_patch` without granting permission to change files. See [Installation](../getting-started/installation.md) for runtime and authentication requirements.

The `claude_code` provider uses Claude Code's authentication (credentials of a Claude Code login on the machine, or `ANTHROPIC_API_KEY`) and ignores `api_key`. In `model_kwargs`, it reads only `max_turns`, `max_budget_usd` (a cap per attempt) and `effort`. It does not support `use_search=True`. It runs with no tools and without the user's settings and MCP servers. See [Providers](../guides/providers.md#claude-code).

---

## Performance

```python
# Parallel processing
result = dataframeit(
    df, Model, PROMPT,
    parallel_requests=5  # 5 simultaneous workers
)

# Rate limiting (prevents 429 error)
result = dataframeit(
    df, Model, PROMPT,
    rate_limit_delay=1.0  # each worker pauses 1 second after each row
)

# Combined
result = dataframeit(
    df, Model, PROMPT,
    parallel_requests=5,
    rate_limit_delay=0.5
)
```

---

## Error Handling

```python
result = dataframeit(df, Model, PROMPT, max_retries=5)

# The status columns only exist if some row failed or recorded a detail
if '_dataframeit_status' in result.columns:
    errors = result[result['_dataframeit_status'] == 'error']
    print(errors['_error_details'])
```

Configuration failures (missing extra, invalid parameter) raise an exception before the first call. Failures on a row do not stop the run: the row gets status `'error'` and the reason in `_error_details`. See [Exceptions](exceptions.md).

---

## Automatically Added Columns

With `track_tokens=True`, DataFrameIt creates `_input_tokens`, `_cached_input_tokens`, `_output_tokens`, and `_reasoning_tokens` for every provider. Without usage telemetry, these values may remain null; when a provider reports total usage but omits cached input or reasoning, the corresponding metric is zero. Cached tokens are a subset of total input, and reasoning tokens are a subset of total output.

| Column | Description |
|--------|-------------|
| `_dataframeit_status` | `'processed'`, `'error'`, `None`; removed when no row failed or recorded any detail |
| `_error_details` | Error message, `"Sucesso após N retry(s)"` or `"Texto ausente"`; removed together with the status column |
| `_input_tokens` | Input tokens (with `track_tokens=True`) |
| `_cached_input_tokens` | Input subset served from cache (with `track_tokens=True`) |
| `_output_tokens` | Output tokens (with `track_tokens=True`) |
| `_reasoning_tokens` | Output subset used for reasoning (with `track_tokens=True`) |
| `_search_credits` | Search credits spent on the row (with `use_search=True`) |
| `_trace`, `_trace_{field}`, `_trace_{group}` | Agent trace in JSON (with `save_trace`) |

---

## Incremental Processing

```python
# Automatic checkpoint every 100 rows
result = dataframeit(
    df, Model, PROMPT,
    batch_size=100,
    checkpoint_path='partial.parquet',
)

# If the run stops, reload the checkpoint and continue
from dataframeit import read_df

df = read_df('partial.parquet', Model)
result = dataframeit(df, Model, PROMPT, resume=True)
```

With `resume=True` (default), rows with status `'error'` are not redone. To redo them, clear their status before running again.

---

## Prompt Template

```python
# Simple - text added at the end
PROMPT = "Classify the sentiment of the text."

# With placeholder - control the position
PROMPT = """
Analyze the document below:

{texto}

Extract the requested information.
"""
```
