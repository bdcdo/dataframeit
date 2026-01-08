# Error Handling

Configure retry, fallbacks and monitor errors in processing.

## Status Columns

DataFrameIt automatically adds control columns:

| Column | Values | Description |
|--------|--------|-------------|
| `_dataframeit_status` | `'processed'`, `'error'`, `None` | Processing status |
| `_error_details` | string or `None` | Error details |

## Checking Errors

```python
from dataframeit import dataframeit

result = dataframeit(df, Model, PROMPT, text_column='text')

# Count errors
total_errors = (result['_dataframeit_status'] == 'error').sum()
print(f"Total errors: {total_errors}")

# Filter rows with errors
errors = result[result['_dataframeit_status'] == 'error']
for idx, row in errors.iterrows():
    print(f"Row {idx}: {row['_error_details']}")

# Filter only successes
success = result[result['_dataframeit_status'] == 'processed']
success.to_excel('clean_result.xlsx', index=False)
```

## Configuring Retry

DataFrameIt uses exponential backoff for automatic retry:

```python
result = dataframeit(
    df,
    Model,
    PROMPT,
    text_column='text',
    max_retries=5,        # Maximum attempts (default: 3)
    base_delay=2.0,       # Initial delay in seconds (default: 1.0)
    max_delay=60.0        # Maximum delay in seconds (default: 30.0)
)
```

**How backoff works:**

```
Attempt 1: fails → wait 2s
Attempt 2: fails → wait 4s
Attempt 3: fails → wait 8s
Attempt 4: fails → wait 16s
Attempt 5: fails → wait 32s (limited to 60s)
Attempt 6: fails → mark as error
```

## Error Types

### Transient Errors (automatic retry)

- **Rate limit (429)**: Too many requests
- **Timeout**: Server took too long
- **Connection error**: Network issues
- **5xx errors**: Server problems

### Permanent Errors (no retry)

- **Validation error**: Response doesn't match Pydantic model
- **Authentication error (401/403)**: Invalid API key
- **Parsing error**: Malformed response

## Incremental Processing

For large datasets, use `resume=True` to continue from where you left off:

```python
# First run
result = dataframeit(df, Model, PROMPT, text_column='text', resume=True)
result.to_excel('partial.xlsx', index=False)

# If interrupted, load and continue
df = pd.read_excel('partial.xlsx')
result = dataframeit(df, Model, PROMPT, text_column='text', resume=True)
result.to_excel('complete.xlsx', index=False)
```

!!! tip "How it works"
    With `resume=True`, DataFrameIt skips rows that already have `_dataframeit_status == 'processed'`.

## Reprocessing Errors

```python
# Load result with errors
df = pd.read_excel('result.xlsx')

# Clear status of error rows to reprocess
df.loc[df['_dataframeit_status'] == 'error', '_dataframeit_status'] = None
df.loc[df['_error_details'].notna(), '_error_details'] = None

# Reprocess only rows without status
result = dataframeit(df, Model, PROMPT, text_column='text', resume=True)
```

## Strategies to Reduce Errors

### 1. Use Rate Limiting

```python
# Prevents rate limit errors
result = dataframeit(
    df, Model, PROMPT,
    text_column='text',
    rate_limit_delay=1.0  # 1 second between requests
)
```

### 2. Simplify the Model

```python
# Very complex model may fail
class ComplexModel(BaseModel):
    field1: str
    field2: List[SubModel]
    field3: Dict[str, AnotherModel]  # Avoid if possible

# Simpler model = fewer errors
class SimpleModel(BaseModel):
    field1: str
    field2: List[str]
```

### 3. Improve the Prompt

```python
# Vague prompt
BAD_PROMPT = "Analyze the text."

# Clear prompt
GOOD_PROMPT = """
Analyze the text and extract:
1. Overall sentiment (positive, negative, or neutral)
2. Classification confidence (high, medium, or low)

If the text is ambiguous, classify as neutral with low confidence.
"""
```

### 4. Use More Capable Models

```python
# If errors persist, try a more capable model
result = dataframeit(
    df, Model, PROMPT,
    text_column='text',
    model='gemini-2.5-pro'  # More capable than flash
)
```
