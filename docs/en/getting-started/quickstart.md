# Quickstart

This guide shows you how to use DataFrameIt in 5 minutes.

## Step 1: Define your Pydantic Model

The Pydantic model defines the structure of the data you want to extract from each text:

```python
from typing import Literal

from pydantic import BaseModel, Field

class Decision(BaseModel):
    """Information extracted from a ruling."""
    defendant: str = Field(description="Name of the defendant")
    defendant_type: Literal['public', 'private'] = Field(
        description="Whether the defendant is a public body or a private party"
    )
    outcome: Literal['granted', 'partially granted', 'denied']
```

!!! tip "Tip"
    Use `Literal` for fields with fixed values. This ensures the LLM only returns valid values.

## Step 2: Prepare your Data

DataFrameIt accepts various input types:

=== "DataFrame"

    ```python
    import pandas as pd

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
    ```

=== "List"

    ```python
    texts = [
        'Maria Souza sues Banco Alfa S.A. over an undue charge. '
        'The claim is GRANTED and the defendant must pay R$ 5,000.00 in moral damages.',
        'José Lima asks the State of São Paulo to supply the prescribed drug. '
        'The claim is PARTIALLY GRANTED, only for the generic version.',
        'Ana Costa seeks damages from the Municipality of Campinas after falling into a pothole. '
        'There is no proof of harm. The claim is DENIED.',
    ]
    ```

=== "Dictionary"

    ```python
    texts = {
        '0001234-56.2024.8.26.0100': (
            'Maria Souza sues Banco Alfa S.A. over an undue charge. '
            'The claim is GRANTED and the defendant must pay R$ 5,000.00 in moral damages.'
        ),
        '0004567-89.2024.8.26.0053': (
            'José Lima asks the State of São Paulo to supply the prescribed drug. '
            'The claim is PARTIALLY GRANTED, only for the generic version.'
        ),
        '0007890-12.2024.8.26.0114': (
            'Ana Costa seeks damages from the Municipality of Campinas after falling into a pothole. '
            'There is no proof of harm. The claim is DENIED.'
        ),
    }
    ```

## Step 3: Process!

```python
from dataframeit import dataframeit

result = dataframeit(
    df,                                          # Your data
    Decision,                                    # Pydantic model
    "Extract the information from the ruling.",  # Prompt
    provider='openai',                           # the default; or 'google_genai', 'anthropic', 'groq'
    text_column='ruling',                        # the column with the text to analyze
)

print(result)
```

**Output:**

| case_number | ruling | defendant | defendant_type | outcome |
|-------------|--------|-----------|----------------|---------|
| 0001234-56.2024.8.26.0100 | Maria Souza sues... | Banco Alfa S.A. | private | granted |
| 0004567-89.2024.8.26.0053 | José Lima asks... | State of São Paulo | public | partially granted |
| 0007890-12.2024.8.26.0114 | Ana Costa seeks... | Municipality of Campinas | public | denied |

Without `text_column`, dataframeit looks for the text column among a few common names and raises `ValueError` when a DataFrame with several columns has none of them ([FAQ](../guides/faq.md#valueerror-about-the-text-column)). With a list or a dictionary, pass `texts` instead of `df` and leave out `text_column`: there is no column to choose.

With `track_tokens=True`, the default, the output also has the token usage columns `_input_tokens`, `_cached_input_tokens`, `_output_tokens` and `_reasoning_tokens`, left out above; see [Added Columns](../reference/api.md#added-columns).

## Complete Example

```python
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from dataframeit import dataframeit

# 1. Pydantic Model
class Decision(BaseModel):
    defendant: str = Field(description="Name of the defendant")
    defendant_type: Literal['public', 'private'] = Field(description="Whether the defendant is a public body or a private party")
    outcome: Literal['granted', 'partially granted', 'denied']

# 2. Data
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

# 3. Process
result = dataframeit(
    df,
    Decision,
    "Extract the information from the ruling.",
    provider='openai',
    text_column='ruling',
)

# 4. Save
result.to_excel('result.xlsx', index=False)
```

## Next Steps

- [Concepts](concepts.md): Understand how DataFrameIt works
- [Basic Usage](../guides/basic-usage.md): More practical examples
- [Error Handling](../guides/error-handling.md): Dealing with failures
