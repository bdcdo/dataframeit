<div class="hero" markdown>

# DataFrameIt

<p class="tagline">Enrich DataFrames with LLMs in a simple and structured way</p>

<div class="badges">
  <a href="https://pypi.org/project/dataframeit/"><img src="https://badge.fury.io/py/dataframeit.svg" alt="PyPI version"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</div>

<div class="cta-buttons" markdown>
  <a href="getting-started/quickstart/" class="cta-button primary" markdown>
    :material-rocket-launch: Get Started
  </a>
  <a href="reference/llm-reference/" class="cta-button secondary" markdown>
    :material-robot: LLM Reference
  </a>
</div>

</div>

## What is it?

DataFrameIt processes text in DataFrames using **Large Language Models (LLMs)** and extracts structured information defined by **Pydantic models**. One function, one model, one prompt — done.

```python
from pydantic import BaseModel
from typing import Literal
import pandas as pd
from dataframeit import dataframeit

class Sentiment(BaseModel):
    sentiment: Literal['positive', 'negative', 'neutral']
    confidence: Literal['high', 'medium', 'low']

df = pd.DataFrame({'text': ['Excellent product!', 'Terrible service.']})
result = dataframeit(df, Sentiment, "Analyze the sentiment of the text.", text_column='text')
```

## Features

<div class="feature-grid" markdown>

<div class="feature-card" markdown>
<div class="icon" markdown>:material-cloud-sync:</div>

### Multiple Providers

Google Gemini, OpenAI GPT-5, Anthropic Claude 4.5, Cohere, Mistral — all via LangChain.
</div>

<div class="feature-card" markdown>
<div class="icon" markdown>:material-check-decagram:</div>

### Structured Output

Automatic validation with Pydantic. Define fields, types, and descriptions — the LLM respects them.
</div>

<div class="feature-card" markdown>
<div class="icon" markdown>:material-shield-refresh:</div>

### Resilience

Automatic retry with exponential backoff. Configurable rate limiting. Never lose progress.
</div>

<div class="feature-card" markdown>
<div class="icon" markdown>:material-rocket-launch:</div>

### Performance

Parallel processing with auto-adjustment. Real-time throughput metrics.
</div>

<div class="feature-card" markdown>
<div class="icon" markdown>:material-web:</div>

### Web Search

Tavily integration to enrich data with information from the internet.
</div>

<div class="feature-card" markdown>
<div class="icon" markdown>:material-format-list-bulleted-type:</div>

### Multiple Inputs

DataFrame, Series, list, dictionary — everything works. Polars included.
</div>

</div>

## Quick Installation

```bash
pip install dataframeit[google]  # Google Gemini 3 (recommended)
pip install dataframeit[openai]  # OpenAI GPT-5
pip install dataframeit[anthropic]  # Anthropic Claude 4.5
```

## Next Steps

<div class="nav-grid" markdown>

<a href="getting-started/installation/" class="nav-card" markdown>
  <div class="nav-icon" markdown>:material-download:</div>
  <div class="nav-content">
    <h4>Installation</h4>
    <p>Set up with your preferred provider</p>
  </div>
</a>

<a href="getting-started/quickstart/" class="nav-card" markdown>
  <div class="nav-icon" markdown>:material-rocket-launch:</div>
  <div class="nav-content">
    <h4>Quickstart</h4>
    <p>First project in 5 minutes</p>
  </div>
</a>

<a href="guides/basic-usage/" class="nav-card" markdown>
  <div class="nav-icon" markdown>:material-book-open-variant:</div>
  <div class="nav-content">
    <h4>Guides</h4>
    <p>Parallelism, retry, web search</p>
  </div>
</a>

<a href="reference/llm-reference/" class="nav-card" markdown>
  <div class="nav-icon" markdown>:material-robot:</div>
  <div class="nav-content">
    <h4>LLM Reference</h4>
    <p>Compact docs for code assistants</p>
  </div>
</a>

</div>
