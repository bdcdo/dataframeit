<div class="hero" markdown>

# DataFrameIt

<p class="tagline">Enrich DataFrames with LLMs in a simple and structured way</p>

<div class="badges">
  <a href="https://pypi.org/project/dataframeit/"><img src="https://badge.fury.io/py/dataframeit.svg" alt="PyPI version"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
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
from pydantic import BaseModel
from typing import Literal
import pandas as pd
from dataframeit import dataframeit

class Sentiment(BaseModel):
    sentiment: Literal['positive', 'negative', 'neutral']
    confidence: Literal['high', 'medium', 'low']

df = pd.DataFrame({'text': ['Excellent product!', 'Terrible service.']})
result = dataframeit(df, Sentiment, "Analyze the sentiment of the text.")
```

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

Parallel processing with auto-adjustment. Real-time throughput metrics.
</div>

<div class="feature-card" markdown>
<div class="icon" markdown>:material-web:</div>

### Web Search

Tavily or Exa to enrich data with information from the internet, with per-field search and conditional fields.
</div>

<div class="feature-card" markdown>
<div class="icon" markdown>:material-format-list-bulleted-type:</div>

### Multiple Inputs

pandas or Polars DataFrame and Series, list and dictionary. The output comes back in the same format as the input.
</div>

</div>

## Quick Installation

```bash
pip install dataframeit[openai]     # OpenAI (default provider)
pip install dataframeit[google]     # Google Gemini
pip install dataframeit[anthropic]  # Anthropic
pip install dataframeit[all]        # all providers, web search, Polars and Excel
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
