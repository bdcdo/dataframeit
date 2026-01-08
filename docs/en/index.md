# DataFrameIt

**Enrich DataFrames with LLMs in a simple and structured way.**

DataFrameIt is a Python library that processes text in DataFrames and extracts structured information using Large Language Models (LLMs). Define what you want to extract with Pydantic, and the library handles the rest.

## Why use it?

- **Simple**: One function, one Pydantic model, one prompt. Done.
- **Structured**: Outputs automatically validated with Pydantic
- **Resilient**: Automatic retry, rate limiting, parallel processing
- **Flexible**: Supports Gemini, OpenAI, Anthropic, Cohere, Mistral and more

## Quick Installation

```bash
pip install dataframeit[google]  # For Google Gemini (recommended)
```

## Example in 30 Seconds

```python
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
from dataframeit import dataframeit

# 1. Define what you want to extract
class Sentiment(BaseModel):
    sentiment: Literal['positive', 'negative', 'neutral']
    confidence: Literal['high', 'medium', 'low']

# 2. Your data
df = pd.DataFrame({
    'text': [
        'Excellent product! Exceeded expectations.',
        'Terrible service, never buying again.',
        'Delivery ok, average product.'
    ]
})

# 3. Process!
result = dataframeit(df, Sentiment, "Analyze the sentiment of the text.", text_column='text')
print(result)
```

**Output:**

| text | sentiment | confidence |
|------|-----------|------------|
| Excellent product! Exceeded expectations. | positive | high |
| Terrible service, never buying again. | negative | high |
| Delivery ok, average product. | neutral | medium |

## Next Steps

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } **Installation**

    ---

    Set up the library with your preferred provider

    [:octicons-arrow-right-24: Installation guide](getting-started/installation.md)

-   :material-rocket-launch:{ .lg .middle } **Quickstart**

    ---

    Create your first project in 5 minutes

    [:octicons-arrow-right-24: Quickstart](getting-started/quickstart.md)

-   :material-book-open-variant:{ .lg .middle } **Guides**

    ---

    Learn advanced features like parallelism and retry

    [:octicons-arrow-right-24: View guides](guides/basic-usage.md)

-   :material-robot:{ .lg .middle } **LLM Reference**

    ---

    Compact documentation optimized for code assistants

    [:octicons-arrow-right-24: LLM Reference](reference/llm-reference.md)

</div>
