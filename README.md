# DataFrameIt

[![PyPI version](https://badge.fury.io/py/dataframeit.svg)](https://badge.fury.io/py/dataframeit)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**English** · [Português](https://github.com/bdcdo/dataframeit/blob/main/README.pt-BR.md) · [Español](https://github.com/bdcdo/dataframeit/blob/main/README.es.md)

**Enrich DataFrames with LLMs, simply and in a structured way.**

DataFrameIt processes text in DataFrames using Large Language Models (LLMs) and extracts structured information defined by Pydantic models.

**[Full Documentation](https://bdcdo.github.io/dataframeit/en/)** | **[LLM Reference](https://bdcdo.github.io/dataframeit/en/reference/llm-reference/)**

## Installation

```bash
pip install dataframeit[openai]  # OpenAI (default provider)
pip install dataframeit[google]  # Google Gemini
pip install dataframeit[anthropic]  # Anthropic Claude
pip install dataframeit[codex]  # Official Codex SDK (experimental)
```

Set up provider authentication:

```bash
export OPENAI_API_KEY="your-key"  # or GOOGLE_API_KEY, ANTHROPIC_API_KEY
```

The experimental `codex` provider is optional, is not part of the `all` extra, uses the bundled runtime and requires local file-based authentication. See the [installation docs](https://bdcdo.github.io/dataframeit/en/getting-started/installation/) to set up the extra and the credentials.

## Quick Example

```python
from pydantic import BaseModel
from typing import Literal
import pandas as pd
from dataframeit import dataframeit

# 1. Define what to extract
class Sentiment(BaseModel):
    sentiment: Literal['positive', 'negative', 'neutral']
    confidence: Literal['high', 'medium', 'low']

# 2. Your data
df = pd.DataFrame({
    'text': [
        'Excellent product! Exceeded my expectations.',
        'Terrible service, never buying again.',
        'Delivery was fine, product is average.'
    ]
})

# 3. Process!
result = dataframeit(df, Sentiment, "Analyze the sentiment of the text.")
print(result)
```

**Output:**

| text | sentiment | confidence |
|------|-----------|------------|
| Excellent product! ... | positive | high |
| Terrible service... | negative | high |
| Delivery was fine... | neutral | medium |

Field and class names are arbitrary — the examples in the [`example/`](https://github.com/bdcdo/dataframeit/tree/main/example) notebooks use Portuguese ones.

## Features

- **Multiple providers**: Google Gemini, OpenAI, Anthropic, Cohere and Mistral via LangChain, plus Claude Code and Codex through their SDKs
- **Multiple input types**: DataFrame, Series, list, dict
- **Structured output**: Automatic validation with Pydantic
- **Resilience**: Automatic retry with exponential backoff
- **Performance**: Parallel processing, configurable rate limiting
- **Web search**: Tavily integration to enrich data
- **Tracking**: Token monitoring and throughput metrics
- **Per-field configuration**: Custom prompts and search parameters per field (v0.5.2+)

## Per-Field Configuration (New in v0.5.2)

Set field-specific prompts and search parameters using `json_schema_extra`:

```python
from pydantic import BaseModel, Field

class DrugInfo(BaseModel):
    # Field with the default prompt
    active_ingredient: str = Field(description="Active ingredient of the drug")

    # Field with a custom prompt (replaces the base prompt)
    rare_disease: str = Field(
        description="Rare disease classification",
        json_schema_extra={
            "prompt": "Search Orphanet (orpha.net). Analyze: {texto}"
        }
    )

    # Field with an additional prompt (appended to the base prompt)
    conitec_assessment: str = Field(
        description="CONITEC assessment",
        json_schema_extra={
            "prompt_append": "Search ONLY the CONITEC website (gov.br/conitec)."
        }
    )

    # Field with custom search parameters
    clinical_trials: str = Field(
        description="Relevant clinical trials",
        json_schema_extra={
            "prompt_append": "Search for recent clinical trials.",
            "search_depth": "advanced",
            "max_results": 10
        }
    )

# Requires search_per_field=True
result = dataframeit(
    df,
    DrugInfo,
    "Analyze the drug: {texto}",
    use_search=True,
    search_per_field=True,
)
```

**Available options in `json_schema_extra`:**

| Option | Description |
|--------|-------------|
| `prompt` or `prompt_replace` | Fully replaces the base prompt |
| `prompt_append` | Appends text to the base prompt |
| `search_depth` | `"basic"` or `"advanced"` (per-field override) |
| `max_results` | Number of search results (1-20) |

## Documentation

- [Quickstart](https://bdcdo.github.io/dataframeit/en/getting-started/quickstart/)
- [Guides](https://bdcdo.github.io/dataframeit/en/guides/basic-usage/)
- [API Reference](https://bdcdo.github.io/dataframeit/en/reference/api/)
- [LLM Reference](https://bdcdo.github.io/dataframeit/en/reference/llm-reference/) - Compact page optimized for coding assistants

## Examples

See the [`example/`](https://github.com/bdcdo/dataframeit/tree/main/example) folder for Jupyter notebooks with complete use cases.

## License

MIT
