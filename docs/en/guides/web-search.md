# Web Search

Enrich your data with web search using Tavily or Exa.

## Overview

DataFrameIt can search the web for information to complement the analysis of each text. This is useful when you need additional context not present in the original text.

With `use_search=True`, each row is processed by an agent that decides when to search, with up to `max_search_calls` searches (default 10), and then answers in the format of the Pydantic model. Search works with the providers via LangChain; `claude_code` and `codex` do not support it.

## Setup

### 1. Install Dependency

```bash
pip install dataframeit[search]       # Tavily (default)
pip install dataframeit[search-exa]   # Exa
pip install dataframeit[search-all]   # both
```

### 2. Configure API Key

```bash
export TAVILY_API_KEY="your-tavily-key"   # https://tavily.com/
export EXA_API_KEY="your-exa-key"         # https://exa.ai/
```

### 3. Choose the Search Provider

| Provider | `search_provider` | Depth | Billing |
|----------|-------------------|-------|---------|
| Tavily | `"tavily"` (default) | `search_depth='basic'` or `'advanced'` | 1 credit per basic search, 2 per advanced |
| Exa | `"exa"` | ignores `search_depth` | 1 credit (US$ 0.005) per search with up to 25 results, 5 above that |

```python
result = dataframeit(df, Model, PROMPT, use_search=True, search_provider="exa")
```

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
    use_search=True,      # Enable web search
    max_results=5         # Number of results per search
)
```

## Search Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_search` | bool | `False` | Enable web search |
| `search_provider` | str | `'tavily'` | `'tavily'` or `'exa'` |
| `search_per_field` | bool | `False` | Run a separate agent for each model field |
| `max_results` | int | `5` | Results per search (1-20) |
| `search_depth` | str | `'basic'` | `'basic'` (1 credit) or `'advanced'` (2 credits); Tavily only |
| `max_search_calls` | int | `10` | Maximum searches per agent run; later ones are blocked and the agent answers with what it found |
| `search_groups` | dict | `None` | Fields that share a search agent; see [Search Groups](#search-groups) |
| `save_trace` | bool/str | `None` | Save agent trace: `True`/`"full"` or `"minimal"` |

With search, the output gains the `_search_credits` column, with the credits spent on each row. Fields can also be conditional (`condition` and `depends_on`); see [Conditional Fields](../examples/conditional-fields.md).

## Examples

### Basic Search

```python
result = dataframeit(
    df, Model, PROMPT,
    use_search=True
)
```

### Search per Field

When the model has many fields, it can be useful to give each field its own agent:

```python
result = dataframeit(
    df, Model, PROMPT,
    use_search=True,
    search_per_field=True  # One search agent per model field
)
```

### Deep Search

```python
# More detailed search (slower, more expensive)
result = dataframeit(
    df, Model, PROMPT,
    use_search=True,
    search_depth='advanced',
    max_results=10
)
```

## Per-Field Configuration

You can configure prompts and search parameters specific to each field using Pydantic's `json_schema_extra`.

### Available Options

| Option | Description |
|--------|-------------|
| `prompt` or `prompt_replace` | Replaces the base prompt for this field; without `{texto}`, the row text is appended at the end |
| `prompt_append` | Appends text to the base prompt |
| `search_depth` | Depth override: `"basic"` or `"advanced"` |
| `max_results` | Results override (1-20) |
| `max_search_calls` | Override of this field agent's maximum searches |

!!! note "Requires search_per_field=True"
    Per-field configuration only works when `search_per_field=True`. If you use `json_schema_extra` with prompt or search settings without enabling `search_per_field`, an error is raised.

### Example: Custom Prompt

```python
from pydantic import BaseModel, Field

class MedicationInfo(BaseModel):
    # Field with default behavior
    active_ingredient: str = Field(description="Active ingredient of the medication")

    # Field with the prompt fully replaced
    rare_disease: str = Field(
        description="Rare disease classification",
        json_schema_extra={
            "prompt": "Search Orphanet (orpha.net) and the FDA Orphan Drug Database. Analyze: {texto}"
        }
    )

    # Field with an additional prompt (append)
    conitec_evaluation: str = Field(
        description="CONITEC evaluation",
        json_schema_extra={
            "prompt_append": "Search ONLY the CONITEC website (gov.br/conitec)."
        }
    )

result = dataframeit(
    df,
    MedicationInfo,
    "Analyze the medication: {texto}",
    use_search=True,
    search_per_field=True,  # Required to use json_schema_extra
)
```

### Example: Per-Field Search Parameters

```python
class DetailedResearch(BaseModel):
    quick_summary: str = Field(
        description="Summary in 2 lines",
        json_schema_extra={
            "search_depth": "basic",
            "max_results": 3
        }
    )

    deep_analysis: str = Field(
        description="Detailed analysis with sources",
        json_schema_extra={
            "prompt_append": "Include citations from the sources found.",
            "search_depth": "advanced",
            "max_results": 10
        }
    )
```

### Combining Prompt and Parameters

You can combine prompt settings and search parameters:

```python
clinical_studies: str = Field(
    description="Relevant clinical studies",
    json_schema_extra={
        "prompt_append": "Search for clinical trials published in the last five years.",
        "search_depth": "advanced",
        "max_results": 15
    }
)
```

## Debug: Save Agent Trace

To debug and audit agent reasoning, use the `save_trace` parameter.

### Parameters

| Value | Description |
|-------|-------------|
| `False` / `None` | Disabled (default) |
| `True` / `"full"` | Complete trace with message content |
| `"minimal"` | Only queries and counts, without search result content |

### Generated Columns

- **Single agent**: `_trace`
- **Per field**: `_trace_{field_name}` for each field
- **With groups**: `_trace_{group_name}` for each group, plus the isolated fields

### Trace Structure

```python
{
    "messages": [
        {"type": "human", "content": "Analyze the medication..."},
        {"type": "ai", "content": "", "tool_calls": [...]},
        {"type": "tool", "content": "[search results]", "tool_call_id": "..."}
    ],
    "search_queries": ["query1", "query2"],
    "total_tool_calls": 3,  # the two searches and the structured answer
    "duration_seconds": 3.45,
    "model": "gpt-6-luna"
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
trace_rare_disease = json.loads(result['_trace_rare_disease'].iloc[0])
```

## Search Groups

When multiple fields need the same search context, you can group them to reduce redundant API calls.

### Motivation

Without groups, 6 fields with `search_per_field=True` mean 6 agents per row, each with its own searches. With groups, related fields share one agent.

**Example:**
- Fields `fda_status`, `ema_approval`, `clinical_trials` are all about regulation
- Without groups: 3 agents, which tend to repeat the same searches
- With groups: 1 shared agent

### Group Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `fields` | list | Yes | List of fields belonging to the group |
| `prompt` | str | No | Custom prompt for the group. Use `{texto}` (or the synonym `{query}`) for the text |
| `max_results` | int | No | Results override (1-20) |
| `search_depth` | str | No | Override: `"basic"` or `"advanced"` |
| `max_search_calls` | int | No | Override of the group agent's maximum searches |

### Basic Example

```python
from pydantic import BaseModel, Field

class DrugRegulatory(BaseModel):
    # Group "regulatory" fields (1 shared agent)
    fda_status: str = Field(description="FDA approval status")
    ema_approval: str = Field(description="EMA approval status")
    clinical_trials: str = Field(description="Ongoing clinical trials")

    # Isolated fields (1 agent each)
    name: str = Field(description="Drug name")
    manufacturer: str = Field(description="Manufacturer")

result = dataframeit(
    df,
    DrugRegulatory,
    "Research the drug: {texto}",
    use_search=True,
    search_per_field=True,
    search_groups={
        "regulatory": {
            "fields": ["fda_status", "ema_approval", "clinical_trials"],
            "prompt": "Search regulatory status (FDA, EMA, clinical trials) for: {query}",
            "search_depth": "advanced",
        }
    }
)
```

**Result:** 3 agents per row (1 for the group + 2 isolated) instead of 5.

### Multiple Groups

```python
search_groups={
    "regulatory": {
        "fields": ["fda_status", "ema_approval"],
        "prompt": "Search regulatory status: {query}",
    },
    "clinical": {
        "fields": ["efficacy", "safety"],
        "prompt": "Search clinical studies about: {query}",
        "search_depth": "advanced",
    }
}
```

### Traces with Groups

With `save_trace=True`, traces are organized by group:

```python
result = dataframeit(
    df, Model, PROMPT,
    use_search=True,
    search_per_field=True,
    search_groups={"regulatory": {"fields": ["fda_status", "ema_approval"]}},
    save_trace=True
)

# Group trace
trace_regulatory = json.loads(result['_trace_regulatory'].iloc[0])

# Isolated field traces
trace_name = json.loads(result['_trace_name'].iloc[0])
```

### Validation Rules

1. **Requires `use_search=True` and `search_per_field=True`**
2. **Fields must exist in the Pydantic model**
3. **Fields cannot be in multiple groups**
4. **Fields in groups cannot have `json_schema_extra` for search**: choose between per-field or group configuration, not both

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
    use_search=True,
    max_results=3
)
```

## Costs and Limits

!!! warning "Watch costs"
    Each agent can make up to `max_search_calls` searches (default 10), and there is one agent per row, or one per field and per row with `search_per_field=True`. For large datasets, this can generate significant costs. The credits spent are recorded in `_search_credits` and in the summary at the end of the run.

Prices change; check each provider's page ([Tavily](https://tavily.com/pricing), [Exa](https://exa.ai/pricing)). Tavily has a free plan with 1000 searches per month.

### Tips to Save

1. Use `max_results=3` to `5` (enough for most cases)
2. Prefer `search_depth='basic'`
3. Filter your DataFrame before processing
4. Use `search_per_field=False` when possible, or group fields with `search_groups`
5. Reduce `max_search_calls` when one or two searches are enough

## Rate Limits and Parallel Processing

!!! danger "HTTP 429 errors"
    Using `parallel_requests` with web search can easily exceed the search provider's rate limits. A failed search stops the agent and the row goes back to the retry cycle; once the attempts are exhausted, the row gets status `'error'`.

### Provider Limits

| Provider | Approximate rate limit |
|----------|-----------------------|
| Tavily   | ~100 req/min          |
| Exa      | ~300 req/min          |

If you need higher throughput, consider `search_provider="exa"`.

### How Queries are Counted

| Configuration | Agents per row | Searches per row, at most |
|---------------|----------------|---------------------------|
| `search_per_field=False` | 1 | `max_search_calls` |
| `search_per_field=True` | 1 per field or group | `max_search_calls` per agent |

With `parallel_requests=20` and `search_per_field=True` on a 4-field model, 80 agents run at the same time, well above either provider's limit.

### Recommended Settings

**Tavily (default):**

| Scenario | `parallel_requests` | `rate_limit_delay` |
|----------|---------------------|--------------------|
| `search_per_field=False` | 5–10 | 0.5s |
| `search_per_field=True` (2–3 fields) | 3–5 | 0.5s |
| `search_per_field=True` (4+ fields) | 2–3 | 1.0s |

**Exa:**

| Scenario | `parallel_requests` | `rate_limit_delay` |
|----------|---------------------|--------------------|
| `search_per_field=False` | 10–15 | 0.3s |
| `search_per_field=True` (2–3 fields) | 5–8 | 0.3s |
| `search_per_field=True` (4+ fields) | 3–5 | 0.5s |

```python
# Safe settings for Tavily with multiple fields
result = dataframeit(
    df, Model, PROMPT,
    use_search=True, search_per_field=True,
    parallel_requests=3, rate_limit_delay=0.5,
)

# Higher throughput with Exa
result = dataframeit(
    df, Model, PROMPT,
    use_search=True, search_provider="exa",
    search_per_field=True,
    parallel_requests=5, rate_limit_delay=0.3,
)
```

### Automatic Warning

DataFrameIt emits a `UserWarning` when the configuration looks risky (high concurrent queries or estimated rate close to the provider limit), with recommended `parallel_requests` and `rate_limit_delay` values to avoid HTTP 429. The warning also fires on sequential runs when `search_per_field=True` produces many queries (>100 total).
