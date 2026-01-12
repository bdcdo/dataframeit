# Web Search

Enrich your data with web search using Tavily.

## Overview

DataFrameIt can search the web for information to complement the analysis of each text. This is useful when you need additional context not present in the original text.

## Setup

### 1. Install Dependency

```bash
pip install dataframeit[search]
# or
pip install langchain-tavily
```

### 2. Configure API Key

```bash
export TAVILY_API_KEY="your-tavily-key"
```

Get your key at: [Tavily](https://tavily.com/)

## Basic Usage

```python
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
from dataframeit import dataframeit

class CompanyInfo(BaseModel):
    sector: Literal['technology', 'health', 'finance', 'retail', 'other']
    description: str = Field(description="Brief company description")
    founded: str = Field(description="Year founded, if found")

# Data with company names
df = pd.DataFrame({
    'text': ['Microsoft', 'Stripe', 'DoorDash']
})

PROMPT = """
Based on available information and web search,
extract information about the mentioned company.
"""

# Enable web search with use_search=True
result = dataframeit(
    df,
    CompanyInfo,
    PROMPT,
    text_column='text',
    use_search=True,      # Enable web search
    max_results=5         # Number of results per search
)
```

## Search Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_search` | bool | `False` | Enable web search via Tavily |
| `search_per_field` | bool | `False` | Execute separate search for each model field |
| `max_results` | int | `5` | Results per search (1-20) |
| `search_depth` | str | `'basic'` | `'basic'` (1 credit) or `'advanced'` (2 credits) |
| `save_trace` | bool/str | `None` | Save agent trace: `True`/`"full"` or `"minimal"` |

## Examples

### Basic Search

```python
result = dataframeit(
    df, Model, PROMPT,
    text_column='text',
    use_search=True
)
```

### Search per Field

When the model has many fields, it can be useful to do separate searches:

```python
result = dataframeit(
    df, Model, PROMPT,
    text_column='text',
    use_search=True,
    search_per_field=True  # One search per model field
)
```

### Deep Search

```python
# More detailed search (slower, more expensive)
result = dataframeit(
    df, Model, PROMPT,
    text_column='text',
    use_search=True,
    search_depth='advanced',
    max_results=10
)
```

## Debug: Save Agent Trace (v0.5.3+)

To debug and audit agent reasoning, use the `save_trace` parameter.

### Parameters

| Value | Description |
|-------|-------------|
| `False` / `None` | Disabled (default) |
| `True` / `"full"` | Complete trace with message content |
| `"minimal"` | Only queries and counts, without search result content |

### Generated Columns

- **Single agent**: `_trace`
- **Per-field**: `_trace_{field_name}` for each field

### Trace Structure

```python
{
    "messages": [
        {"type": "human", "content": "Analyze the medication..."},
        {"type": "ai", "content": "", "tool_calls": [...]},
        {"type": "tool", "content": "[search results]", "tool_call_id": "..."}
    ],
    "search_queries": ["query1", "query2"],
    "total_tool_calls": 2,
    "duration_seconds": 3.45,
    "model": "gpt-4o-mini"
}
```

### Example: Full Trace

```python
import json

result = dataframeit(
    df,
    MedicationInfo,
    PROMPT,
    use_search=True,
    save_trace=True  # or "full"
)

# Access trace from first row
trace = json.loads(result['_trace'].iloc[0])
print(f"Queries performed: {trace['search_queries']}")
print(f"Duration: {trace['duration_seconds']}s")
print(f"Model: {trace['model']}")
```

### Example: Minimal Trace

For audits where only the search queries matter:

```python
result = dataframeit(
    df, Model, PROMPT,
    use_search=True,
    save_trace="minimal"  # Excludes search result content
)
```

### Example: Per-Field Trace

```python
result = dataframeit(
    df,
    MedicationInfo,
    PROMPT,
    use_search=True,
    search_per_field=True,
    save_trace="full"
)

# Each field has its own trace
trace_ingredient = json.loads(result['_trace_active_ingredient'].iloc[0])
trace_indication = json.loads(result['_trace_indication'].iloc[0])
```

## Use Case: Fact Checking

```python
from pydantic import BaseModel, Field
from typing import Literal, List

class FactCheck(BaseModel):
    claim: str = Field(description="The original claim")
    verdict: Literal['true', 'false', 'partially_true', 'inconclusive']
    sources: List[str] = Field(description="Sources supporting the verdict")
    explanation: str = Field(description="Explanation of the verdict")

PROMPT = """
Verify the truthfulness of the claim using web search information.
Cite the sources found.
"""

result = dataframeit(
    df_claims,
    FactCheck,
    PROMPT,
    text_column='text',
    use_search=True,
    max_results=5,
    search_depth='advanced'
)
```

## Use Case: Lead Enrichment

```python
class EnrichedLead(BaseModel):
    company: str
    website: str = Field(description="Official website")
    linkedin: str = Field(description="LinkedIn URL")
    size: Literal['startup', 'sme', 'enterprise']
    technologies: List[str] = Field(description="Technologies used")

result = dataframeit(
    df_leads,
    EnrichedLead,
    "Research information about the company.",
    text_column='text',
    use_search=True,
    max_results=3
)
```

## Costs and Limits

!!! warning "Watch costs"
    Each DataFrame row makes a web search. For large datasets, this can generate significant costs on the Tavily API.

- **Free tier**: 1000 searches/month
- **Basic search**: ~$0.01 per search
- **Advanced search**: ~$0.02 per search

### Tips to Save

1. Use `max_results=3` to `5` (enough for most cases)
2. Prefer `search_depth='basic'`
3. Filter your DataFrame before processing
4. Use `search_per_field=False` when possible
