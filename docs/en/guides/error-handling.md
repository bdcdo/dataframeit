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

### Response rejected by validation (retry with the error)

- **Validation error**: the response fails the Pydantic model, including custom validators (`model_validator`, `field_validator`)
- **Parsing error**: the response is not valid JSON or did not come in the structured format

With LangChain providers, the next attempt sends the model the rejected response and the list of errors, each with the field path and the rejected value, asking it to answer again fixing those points. Repeating the same prompt tends to repeat the same error. When the response is not JSON, the request carries the parser's error text.

With OpenAI, the SDK validates the response inside the call and raises before returning the message; the rejected response and its tokens are read from the HTTP response attached to the error. When no raw response is available, the correction request goes with the prompt, carrying the rejected values. When a later attempt succeeds, tokens from the rejected ones are added to `_input_tokens` and `_output_tokens`, because they are billed too; if every attempt fails, the row gets status `error`, no token count, and `_error_details` names the field and the rule of each error.

### Permanent Errors (no retry)

- **Authentication error (401/403)**: Invalid API key
- **Not found (404)**: model or endpoint unknown to the provider
- **Invalid request** (`BadRequestError`, `InvalidArgument`): a parameter the provider rejects
- **Prompt larger than the context window** (`ContextOverflowError`)
- **Local configuration incompatible with the provider**

## Incremental Processing

For large datasets, use `resume=True` to continue from where you left off:

```python
# First run
result = dataframeit(df, Model, PROMPT, text_column='text', resume=True)
result.to_excel('partial.xlsx', index=False)

# If interrupted, load and continue
from dataframeit import read_df

df = read_df('partial.xlsx', Model)
result = dataframeit(df, Model, PROMPT, text_column='text', resume=True)
result.to_excel('complete.xlsx', index=False)
```

!!! tip "How it works"
    With `resume=True`, DataFrameIt processes only rows without `_dataframeit_status`. Rows with `'processed'` or `'error'` are left as they are; to retry errors, clear their status as shown in the section below.

## Reprocessing Errors

```python
from dataframeit import read_df

# Load result with errors
df = read_df('result.xlsx', Model)

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
    model='gpt-6-sol'  # More capable than gpt-6-luna
)
```
