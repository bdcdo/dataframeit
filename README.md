# DataFrameIt

[![PyPI version](https://badge.fury.io/py/dataframeit.svg)](https://badge.fury.io/py/dataframeit)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/bdcdo/dataframeit/actions/workflows/tests.yml/badge.svg)](https://github.com/bdcdo/dataframeit/actions/workflows/tests.yml)

**English** · [Português](https://github.com/bdcdo/dataframeit/blob/main/README.pt-BR.md) · [Español](https://github.com/bdcdo/dataframeit/blob/main/README.es.md)

**Enrich DataFrames with LLMs, simply and in a structured way.**

DataFrameIt processes text in DataFrames using Large Language Models (LLMs) and extracts structured information defined by Pydantic models.

**[Full Documentation](https://brunodcdo.com.br/dataframeit/en/)** | **[LLM Reference](https://brunodcdo.com.br/dataframeit/en/reference/llm-reference/)**

## Installation

```bash
pip install dataframeit[openai]       # OpenAI (default provider)
pip install dataframeit[google]       # Google Gemini
pip install dataframeit[anthropic]    # Anthropic Claude
pip install dataframeit[groq]         # Groq
pip install dataframeit[codex]        # Official Codex SDK (experimental)
pip install dataframeit[claude-code]  # Claude Code via the Claude Agent SDK
pip install dataframeit[all]          # the providers above except Codex, plus web search, Polars and Excel
```

Optional extras: `search` (Tavily), `search-exa` (Exa), `search-all`, `polars`, `excel`.

Set up provider authentication:

```bash
export OPENAI_API_KEY="your-key"  # or GOOGLE_API_KEY, ANTHROPIC_API_KEY, GROQ_API_KEY
```

The experimental `codex` provider is optional, is not part of the `all` extra, uses the bundled runtime and requires local file-based authentication. See the [installation docs](https://brunodcdo.com.br/dataframeit/en/getting-started/installation/) to set up the extra and the credentials.

## Quick Example

```python
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from dataframeit import dataframeit

# 1. Define what to extract from each decision
class Decision(BaseModel):
    defendant: str = Field(description="Name of the defendant")
    defendant_type: Literal['public', 'private'] = Field(description="Whether the defendant is a public body or a private party")
    outcome: Literal['granted', 'partially granted', 'denied']

# 2. Your data: one decision per row
df = pd.DataFrame({
    'case_number': ['0001234-56.2024.8.26.0100', '0004567-89.2024.8.26.0053', '0007890-12.2024.8.26.0114'],
    'ruling': [
        'Maria Souza sues Banco Alfa S.A. over an undue charge. '
        'The claim is GRANTED and the defendant must pay R$ 5,000.00 in moral damages.',
        'José Lima asks the State of São Paulo to supply the prescribed drug. '
        'The claim is PARTIALLY GRANTED, only for the generic version.',
        'Ana Costa seeks damages from the Municipality of Campinas after falling into a pothole. '
        'There is no proof of harm. The claim is DENIED.',
    ],
})

# 3. Process!
result = dataframeit(
    df,
    Decision,
    "Extract the information from the ruling.",
    provider='openai',     # the default; or 'google_genai', 'anthropic', 'groq'
    text_column='ruling',  # the column with the text to analyze
)
print(result)
```

**Output:**

| case_number | ruling | defendant | defendant_type | outcome |
|-------------|--------|-----------|----------------|---------|
| 0001234-56.2024.8.26.0100 | Maria Souza sues... | Banco Alfa S.A. | private | granted |
| 0004567-89.2024.8.26.0053 | José Lima asks... | State of São Paulo | public | partially granted |
| 0007890-12.2024.8.26.0114 | Ana Costa seeks... | Municipality of Campinas | public | denied |

Without `text_column`, dataframeit looks for the text column among a few common names and raises `ValueError` when a DataFrame with several columns has none of them ([FAQ](https://brunodcdo.com.br/dataframeit/en/guides/faq/#valueerror-about-the-text-column)). With `track_tokens=True`, the default, the output also has the token usage columns `_input_tokens`, `_cached_input_tokens`, `_output_tokens` and `_reasoning_tokens`, left out of the table above.

Field and class names are arbitrary; the examples in the [`example/`](https://github.com/bdcdo/dataframeit/tree/main/example) notebooks use Portuguese ones.

## Features

- **Multiple providers**: OpenAI, Google Gemini, Anthropic and Groq with their own extras, any other LangChain provider (Cohere, Mistral, Vertex AI, Bedrock, Azure) by installing its package, plus Claude Code and Codex through their official SDKs
- **Multiple input types**: pandas or Polars DataFrame and Series, list, dict
- **Structured output**: Pydantic validation; with the LangChain providers, a rejected answer goes back to the model with the error
- **Resilience**: Automatic retry with exponential backoff, and periodic checkpoints (`batch_size` + `checkpoint_path`) to resume long runs
- **Performance**: Continuous parallel queue that halves the rows in flight on rate limits, configurable rate limiting, and a per-row callback (`on_row_done`) that runs on the worker thread
- **Web search**: Tavily or Exa, per field or per group of fields, with conditional fields (`condition`, `depends_on`)
- **Tracking**: Token usage, search credits and throughput metrics

## Per-Field Configuration

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

The full list of per-field options (including `max_search_calls`, `condition` and `depends_on`) is in the [web search guide](https://brunodcdo.com.br/dataframeit/en/guides/web-search/).

## Documentation

- [Quickstart](https://brunodcdo.com.br/dataframeit/en/getting-started/quickstart/)
- [Guides](https://brunodcdo.com.br/dataframeit/en/guides/basic-usage/)
- [API Reference](https://brunodcdo.com.br/dataframeit/en/reference/api/)
- [LLM Reference](https://brunodcdo.com.br/dataframeit/en/reference/llm-reference/) - Compact page optimized for coding assistants
- [FAQ](https://brunodcdo.com.br/dataframeit/en/guides/faq/)
- [Changelog](https://github.com/bdcdo/dataframeit/blob/main/CHANGELOG.md)

## Examples

See the [`example/`](https://github.com/bdcdo/dataframeit/tree/main/example) folder for Jupyter notebooks with complete use cases.

## Contributing

See [CONTRIBUTING.md](https://github.com/bdcdo/dataframeit/blob/main/CONTRIBUTING.md).

## License

MIT
