# Basic Usage

Practical examples for common use cases.

## Sentiment Analysis

```python
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
from dataframeit import dataframeit

class Sentiment(BaseModel):
    sentiment: Literal['positive', 'negative', 'neutral']
    confidence: Literal['high', 'medium', 'low']

df = pd.DataFrame({
    'text': [
        'Love this product!',
        'Terrible, do not recommend.',
        'Average, nothing special.'
    ]
})

result = dataframeit(df, Sentiment, "Analyze the sentiment of the text.", text_column='text')
```

## Category Classification

```python
from pydantic import BaseModel, Field
from typing import Literal

class Category(BaseModel):
    category: Literal['technology', 'health', 'finance', 'education', 'other']
    subcategory: str = Field(description="More specific subcategory")

result = dataframeit(
    df,
    Category,
    "Classify the text into the most appropriate category.",
    text_column='text'
)
```

## Entity Extraction

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class Entities(BaseModel):
    people: List[str] = Field(description="Names of people mentioned")
    organizations: List[str] = Field(description="Company/organization names")
    locations: List[str] = Field(description="Locations mentioned")
    dates: List[str] = Field(description="Dates mentioned")

PROMPT = """
Extract all named entities from the text.
If there are no entities of some type, return an empty list.
"""

result = dataframeit(df, Entities, PROMPT, text_column='text')
```

## Text Summarization

```python
from pydantic import BaseModel, Field

class Summary(BaseModel):
    summary: str = Field(description="Summary in up to 50 words")
    key_points: list[str] = Field(description="List of 3-5 main points")
    main_topic: str = Field(description="Central topic in one word")

PROMPT = """
Analyze the text and extract a concise summary.
Identify the main points and central topic.
"""

result = dataframeit(df, Summary, PROMPT, text_column='text')
```

## Using Different Input Types

### With List

```python
texts = ['Text 1', 'Text 2', 'Text 3']
result = dataframeit(texts, Sentiment, PROMPT)
# Returns DataFrame with numeric index
```

### With Dictionary

```python
documents = {
    'doc_001': 'Content of document 1',
    'doc_002': 'Content of document 2',
}
result = dataframeit(documents, Sentiment, PROMPT)
# Returns DataFrame with keys as index
```

### With Series

```python
series = pd.Series(['Text A', 'Text B'], index=['id_1', 'id_2'])
result = dataframeit(series, Sentiment, PROMPT)
# Preserves original index
```

## Using Different Providers

```python
# Google Gemini (default)
result = dataframeit(df, Model, PROMPT, text_column='text')

# OpenAI
result = dataframeit(
    df, Model, PROMPT,
    text_column='text',
    provider='openai',
    model='gpt-5.2-mini'
)

# Anthropic Claude
result = dataframeit(
    df, Model, PROMPT,
    text_column='text',
    provider='anthropic',
    model='claude-sonnet-4.5'
)
```

## Next Steps

- [Structured Output](structured-output.md): Advanced Pydantic models
- [Error Handling](error-handling.md): Retry and fallbacks
- [Performance](performance.md): Parallelism and rate limiting
