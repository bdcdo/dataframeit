# Concepts

Understand the fundamental concepts of DataFrameIt.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        dataframeit()                         │
├─────────────────────────────────────────────────────────────┤
│  Input             │  Processing         │  Output          │
│  ─────             │  ──────────         │  ──────          │
│  • DataFrame       │  • For each row:    │  • DataFrame     │
│  • Series          │    1. Build prompt  │    with extracted│
│  • List            │    2. Call LLM      │    columns       │
│  • Dict            │    3. Validate resp.│                  │
│                    │    4. Retry on error│                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     LangChain + Provider                     │
├─────────────────────────────────────────────────────────────┤
│  Google Gemini │ OpenAI │ Anthropic │ Cohere │ Mistral      │
└─────────────────────────────────────────────────────────────┘
```

## Main Components

### 1. Pydantic Model

The Pydantic model defines **what** you want to extract. Each field becomes a column in the output DataFrame.

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional

class Analysis(BaseModel):
    # Required field with fixed values
    category: Literal['A', 'B', 'C'] = Field(
        description="Item category"
    )

    # Required field with free text
    summary: str = Field(
        description="Summary in one sentence"
    )

    # Optional field
    notes: Optional[str] = Field(
        default=None,
        description="Additional notes, if any"
    )
```

!!! info "Why Pydantic?"
    - **Automatic validation**: The LLM is forced to return data in the correct format
    - **Documentation**: Field descriptions help the LLM understand what to extract
    - **Type safety**: Type errors are caught automatically

### 2. Prompt Template

The prompt defines **how** the LLM should process each text.

```python
# Simple - text is automatically added at the end
PROMPT = "Classify the sentiment of the text."

# With placeholder - control where text appears
PROMPT = """
You are a specialized analyst.

Document:
{texto}

Extract the requested information from the document above.
"""
```

### 3. Providers via LangChain

DataFrameIt uses LangChain to abstract different LLM providers:

| Provider | Popular Models | Environment Variable |
|----------|----------------|---------------------|
| `google_genai` | gemini-2.0-flash, gemini-1.5-pro | `GOOGLE_API_KEY` |
| `openai` | gpt-4o, gpt-4o-mini, o1, o3-mini | `OPENAI_API_KEY` |
| `anthropic` | claude-3-5-sonnet, claude-3-opus | `ANTHROPIC_API_KEY` |

## Processing Flow

```
For each DataFrame row:
│
├─► 1. Build prompt (template + row text)
│
├─► 2. Send to LLM via LangChain
│
├─► 3. Receive structured response
│
├─► 4. Validate with Pydantic
│   │
│   ├─► Success: mark as 'processed'
│   │
│   └─► Error: retry with exponential backoff
│       │
│       ├─► Success after retry: mark as 'processed'
│       │
│       └─► Failure after max_retries: mark as 'error'
│
└─► 5. Add extracted fields to DataFrame
```

## Automatic Columns

DataFrameIt automatically adds control columns:

| Column | Description |
|--------|-------------|
| `_dataframeit_status` | Status: `'processed'`, `'error'`, or `None` |
| `_error_details` | Error details (when status is `'error'`) |
| `_input_tokens` | Input tokens (with `track_tokens=True`) |
| `_output_tokens` | Output tokens (with `track_tokens=True`) |
| `_total_tokens` | Total tokens (with `track_tokens=True`) |

## Next Steps

- [Basic Usage](../guides/basic-usage.md): Practical examples
- [Error Handling](../guides/error-handling.md): Configure retry and fallbacks
- [Performance](../guides/performance.md): Parallelism and rate limiting
