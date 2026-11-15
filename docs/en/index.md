<div class="hero" markdown>

# DataFrameIt

<p class="tagline">Enrich DataFrames with LLMs in a simple and structured way</p>

<div class="badges">
  <a href="https://pypi.org/project/dataframeit/"><img src="https://badge.fury.io/py/dataframeit.svg" alt="PyPI version"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</div>

<div class="cta-buttons" markdown>

[:material-rocket-launch: Get Started](getting-started/quickstart.md){ .cta-button .primary }

[:material-robot: LLM Reference](reference/llm-reference.md){ .cta-button .secondary }

</div>

</div>

## What is it?

DataFrameIt processes text in DataFrames using **Large Language Models (LLMs)** and extracts structured information defined by **Pydantic models**. One function, one model and one prompt are all you need.

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

Without `text_column`, dataframeit looks for the text column among a few common names and raises `ValueError` when a DataFrame with several columns has none of them ([FAQ](guides/faq.md#valueerror-about-the-text-column)). With `track_tokens=True`, the default, the output also has the token usage columns `_input_tokens`, `_cached_input_tokens`, `_output_tokens` and `_reasoning_tokens`, left out of the table above.

## Features

<div class="feature-grid" markdown>

<div class="feature-card" markdown>
<div class="icon" markdown>:material-cloud-sync:</div>

### Multiple Providers

OpenAI, Google Gemini, Anthropic, Groq and others via LangChain, plus Codex and Claude Code through the official SDKs.
</div>

<div class="feature-card" markdown>
<div class="icon" markdown>:material-check-decagram:</div>

### Structured Output

Automatic validation with Pydantic. Define fields, types, and descriptions; a response that does not match the model goes back to the LLM with the error for correction.
</div>

<div class="feature-card" markdown>
<div class="icon" markdown>:material-shield-refresh:</div>

### Resilience

Automatic retry with exponential backoff, configurable rate limiting and periodic checkpoints to resume long runs.
</div>

<div class="feature-card" markdown>
<div class="icon" markdown>:material-rocket-launch:</div>

### Performance

Parallel processing that reduces the workers on rate limits. With `parallel_requests > 1`, the summary at the end of the run includes throughput metrics.
</div>

<div class="feature-card" markdown>
<div class="icon" markdown>:material-web:</div>

### Web Search

Tavily or Exa to enrich data with information from the internet, with per-field search and conditional fields.
</div>

<div class="feature-card" markdown>
<div class="icon" markdown>:material-format-list-bulleted-type:</div>

### Multiple Inputs

pandas or Polars DataFrame and Series, list and dictionary. The output is a DataFrame, a Polars one when the input is Polars.
</div>

</div>

## Quick Installation

```bash
pip install dataframeit[openai]     # OpenAI (default provider)
pip install dataframeit[google]     # Google Gemini
pip install dataframeit[anthropic]  # Anthropic
pip install dataframeit[all]        # providers with an extra except Codex, plus web search, Polars and Excel
```

## Next Steps

<div class="nav-grid" markdown>

<div class="nav-card" markdown>
### :material-download: [Installation](getting-started/installation.md)
Set up with your preferred provider
</div>

<div class="nav-card" markdown>
### :material-rocket-launch: [Quickstart](getting-started/quickstart.md)
First project in 5 minutes
</div>

<div class="nav-card" markdown>
### :material-book-open-variant: [Guides](guides/basic-usage.md)
Parallelism, retry, web search
</div>

<div class="nav-card" markdown>
### :material-robot: [LLM Reference](reference/llm-reference.md)
Compact docs for code assistants
</div>

</div>
