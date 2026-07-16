# LLM Reference

This page contains all the information needed to use DataFrameIt in a single compact page, optimized for code assistants.

---

## What it is

DataFrameIt processes texts in DataFrames using LLMs and extracts structured information defined by Pydantic models.

## Installation

```bash
pip install dataframeit[google]    # Google Gemini (default)
pip install dataframeit[openai]    # OpenAI
pip install dataframeit[anthropic] # Anthropic Claude
pip install dataframeit[codex]     # Official Codex SDK (experimental)
```

**Environment variables:**
```bash
export GOOGLE_API_KEY="..."     # For Gemini
export OPENAI_API_KEY="..."     # For OpenAI
export ANTHROPIC_API_KEY="..."  # For Anthropic
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
    model='gemini-3-flash-preview',
    provider='google_genai', # 'google_genai', 'openai', 'anthropic', 'codex'
    resume=True,             # Continue from where it stopped
    parallel_requests=1,     # Parallel workers
    rate_limit_delay=0.0,    # Delay between requests (seconds)
    max_retries=3,           # Retry attempts on error
    track_tokens=True,       # Track token usage
    api_key=None,            # API key (uses env var if None)
    model_kwargs=None,       # Extra parameters (temperature, etc)
    # Web search (requires TAVILY_API_KEY)
    use_search=False,        # Enable web search
    search_per_field=False,  # Separate search per field
    max_results=5,           # Results per search
    search_depth='basic',    # 'basic' or 'advanced'
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
    "Analyze the text and extract the requested information.",
    text_column='text'
)

# 4. Result contains columns: text, sentiment, confidence, topics, summary
print(result)
```

---

## Supported Input Types

```python
# DataFrame (requires text_column)
df = pd.DataFrame({'text': ['A', 'B']})
result = dataframeit(df, Model, PROMPT, text_column='text')

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
# Google Gemini (default)
result = dataframeit(df, Model, PROMPT, text_column='text')

# OpenAI
result = dataframeit(
    df, Model, PROMPT,
    text_column='text',
    provider='openai',
    model='gpt-5.2-mini'
)

# Anthropic
result = dataframeit(
    df, Model, PROMPT,
    text_column='text',
    provider='anthropic',
    model='claude-sonnet-4-5'
)

# Official Codex SDK (experimental)
result = dataframeit(
    df, Model, PROMPT,
    text_column='text',
    provider='codex',
    model='gpt-5.4',
    model_kwargs={'effort': 'medium'}
)

# With extra parameters
result = dataframeit(
    df, Model, PROMPT,
    text_column='text',
    provider='openai',
    model='gpt-5.2-mini',
    model_kwargs={'temperature': 0.2}
)
```

The `codex` provider accepts only `effort` in `model_kwargs` and does not support `use_search=True`. The integration disables web search, shell access, and MCP servers, denies approvals, and uses a read-only sandbox to block writes; the runtime may still present internal utilities such as `apply_patch` without granting permission to change files. See [Installation](../getting-started/installation.md) for runtime and authentication requirements.

---

## Performance

```python
# Parallel processing
result = dataframeit(
    df, Model, PROMPT,
    text_column='text',
    parallel_requests=5  # 5 simultaneous workers
)

# Rate limiting (prevents 429 error)
result = dataframeit(
    df, Model, PROMPT,
    text_column='text',
    rate_limit_delay=1.0  # 1 second between requests
)

# Combined
result = dataframeit(
    df, Model, PROMPT,
    text_column='text',
    parallel_requests=5,
    rate_limit_delay=0.5
)
```

---

## Error Handling

```python
result = dataframeit(df, Model, PROMPT, text_column='text', max_retries=5)

# Check errors
errors = result[result['_dataframeit_status'] == 'error']
print(errors['_error_details'])

# Filter success
success = result[result['_dataframeit_status'] == 'processed']
```

---

## Automatically Added Columns

With `track_tokens=True`, DataFrameIt creates `_input_tokens`, `_cached_input_tokens`, `_output_tokens`, and `_reasoning_tokens` for every provider. Without usage telemetry, these values may remain null; when a provider reports total usage but omits cached input or reasoning, the corresponding metric is zero. Cached tokens are a subset of total input, and reasoning tokens are a subset of total output.

| Column | Description |
|--------|-------------|
| `_dataframeit_status` | `'processed'`, `'error'`, `None` |
| `_error_details` | Error message |
| `_input_tokens` | Input tokens (with `track_tokens=True`) |
| `_cached_input_tokens` | Input subset served from cache (with `track_tokens=True`) |
| `_output_tokens` | Output tokens (with `track_tokens=True`) |
| `_reasoning_tokens` | Output subset used for reasoning (with `track_tokens=True`) |

---

## Incremental Processing

```python
# Process and save
result = dataframeit(df, Model, PROMPT, text_column='text', resume=True)
result.to_excel('partial.xlsx', index=False)

# Load and continue
df = pd.read_excel('partial.xlsx')
result = dataframeit(df, Model, PROMPT, text_column='text', resume=True)
```

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
