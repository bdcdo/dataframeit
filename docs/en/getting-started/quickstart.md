# Quickstart

This guide shows you how to use DataFrameIt in 5 minutes.

## Step 1: Define your Pydantic Model

The Pydantic model defines the structure of data you want to extract:

```python
from pydantic import BaseModel, Field
from typing import Literal

class Sentiment(BaseModel):
    """Sentiment analysis of a text."""
    sentiment: Literal['positive', 'negative', 'neutral'] = Field(
        description="Overall sentiment of the text"
    )
    confidence: Literal['high', 'medium', 'low'] = Field(
        description="Confidence level in the classification"
    )
```

!!! tip "Tip"
    Use `Literal` for fields with fixed values. This ensures the LLM only returns valid values.

## Step 2: Prepare your Data

DataFrameIt accepts various input types:

=== "DataFrame"

    ```python
    import pandas as pd

    df = pd.DataFrame({
        'text': [
            'Excellent product! Exceeded expectations.',
            'Terrible service, never buying again.',
            'Delivery ok, average product.'
        ]
    })
    ```

=== "List"

    ```python
    texts = [
        'Excellent product! Exceeded expectations.',
        'Terrible service, never buying again.',
        'Delivery ok, average product.'
    ]
    ```

=== "Dictionary"

    ```python
    texts = {
        'review_001': 'Excellent product! Exceeded expectations.',
        'review_002': 'Terrible service, never buying again.',
        'review_003': 'Delivery ok, average product.'
    }
    ```

## Step 3: Process!

```python
from dataframeit import dataframeit

result = dataframeit(
    df,                                      # Your data
    Sentiment,                               # Pydantic model
    "Analyze the sentiment of the text.",    # Prompt
    text_column='text'                       # Column name
)

print(result)
```

**Output:**

```
                                        text sentiment confidence
0  Excellent product! Exceeded expectations.  positive       high
1      Terrible service, never buying again.  negative       high
2               Delivery ok, average product.   neutral     medium
```

## Complete Example

```python
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
from dataframeit import dataframeit

# 1. Pydantic Model
class Sentiment(BaseModel):
    sentiment: Literal['positive', 'negative', 'neutral']
    confidence: Literal['high', 'medium', 'low']

# 2. Data
df = pd.DataFrame({
    'text': [
        'Excellent product! Exceeded expectations.',
        'Terrible service, never buying again.',
        'Delivery ok, average product.'
    ]
})

# 3. Process
result = dataframeit(df, Sentiment, "Analyze the sentiment of the text.", text_column='text')

# 4. Save
result.to_excel('result.xlsx', index=False)
```

## Next Steps

- [Concepts](concepts.md): Understand how DataFrameIt works
- [Basic Usage](../guides/basic-usage.md): More practical examples
- [Error Handling](../guides/error-handling.md): Dealing with failures
