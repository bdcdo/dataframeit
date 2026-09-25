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

### 3. Providers

DataFrameIt uses LangChain to abstract different LLM providers. The `codex` and `claude_code` providers use the official SDKs of those tools, with their local authentication; see [Providers](../guides/providers.md).

| Provider | Popular Models | Environment Variable |
|----------|----------------|---------------------|
| `openai` | gpt-6-luna, gpt-6-sol, gpt-6-astra | `OPENAI_API_KEY` |
| `google_genai` | gemini-3.8-flash, gemini-3.6-flash, gemini-3.5-flash-lite | `GOOGLE_API_KEY` |
| `anthropic` | claude-sonnet-5, claude-opus-5-5, claude-haiku-4-5 | `ANTHROPIC_API_KEY` |
| `groq` | openai/gpt-oss-120b, openai/gpt-oss-20b | `GROQ_API_KEY` |

## Processing Flow

```
For each DataFrame row:
│
├─► 1. Build prompt (template + row text)
│
├─► 2. Send to the provider (with web search, to an agent that searches before answering)
│
├─► 3. Receive structured response
│
├─► 4. Validate with Pydantic
│   │
│   ├─► Success: mark as 'processed'
│   │
│   └─► Transient error or rejected response: new attempt with exponential backoff
│       │
│       ├─► Success on a new attempt: mark as 'processed'
│       │
│       └─► Attempts exhausted, or permanent error: mark as 'error'
│
└─► 5. Add extracted fields to DataFrame
```

## Automatic Columns

DataFrameIt automatically adds the status columns. When `track_tokens=True`, it also adds usage columns; see the [LLM Reference](../reference/llm-reference.md#automatically-added-columns) for the complete table and the semantics of cached input and reasoning.

| Column | Description |
|--------|-------------|
| `_dataframeit_status` | Status: `'processed'`, `'error'`, or `None` |
| `_error_details` | Error details, or details of the retries on a row that succeeded |

Both columns are removed from the output when no row fails or records any detail.

## Next Steps

- [Basic Usage](../guides/basic-usage.md): Practical examples
- [Error Handling](../guides/error-handling.md): Configure retry and monitor failures
- [Performance](../guides/performance.md): Parallelism and rate limiting
