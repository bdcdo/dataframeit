# Performance

Optimize processing with parallelism, rate limiting, and token tracking.

## Parallel Processing

Use `parallel_requests` to speed up processing:

```python
result = dataframeit(
    df,
    Model,
    PROMPT,
    parallel_requests=5  # 5 simultaneous requests
)
```

### Recommendations by Size

| Dataset | Configuration |
|---------|---------------|
| < 50 rows | `parallel_requests=1` (default) |
| 50-500 rows | `parallel_requests=3` to `5` |
| > 500 rows | `parallel_requests=5` to `10` |

### Auto-reduction on Rate Limits

When a 429 error is detected, DataFrameIt automatically reduces workers:

```
Start: 10 workers
Rate limit detected → 5 workers
Rate limit detected → 2 workers
Rate limit detected → 1 worker
```

!!! info "Safety"
    Workers are only **reduced**, never automatically increased. This prevents unexpected costs.

## Rate Limiting

Use `rate_limit_delay` to prevent rate limit errors:

```python
result = dataframeit(
    df,
    Model,
    PROMPT,
    rate_limit_delay=1.0  # each worker pauses 1 second after each row
)
```

### Calculating Ideal Delay

`rate_limit_delay` is a pause each worker takes after each row completed successfully. With several workers, the pauses run in parallel, and the maximum request rate is `parallel_requests × 60 / rate_limit_delay` per minute. To stay below a limit:

```
delay = 60 × parallel_requests / requests_per_minute

Examples:
- 60 req/min,  1 worker   → delay = 1.0s
- 60 req/min,  5 workers  → delay = 5.0s
- 500 req/min, 5 workers  → delay = 0.6s
```

The actual rate stays below this ceiling, because each call to the model also takes time.

### By Provider

The requests-per-minute limit depends on the model and on the account tier, and changes often. Look up your account's value on the official page and apply the formula above:

- [Google Gemini](https://ai.google.dev/gemini-api/docs/rate-limits)
- [OpenAI](https://developers.openai.com/api/docs/guides/rate-limits)
- [Anthropic](https://platform.claude.com/docs/en/api/rate-limits)
- [Groq](https://console.groq.com/docs/rate-limits)

### Combining with Parallelism

```python
# 5 workers, each pausing 0.5s after each row: up to 600 req/min
result = dataframeit(
    df,
    Model,
    PROMPT,
    parallel_requests=5,
    rate_limit_delay=0.5
)
```

## Checkpoints for Long Runs

On large datasets (thousands of rows, hours of runtime), a kill or crash loses all in-memory progress. Use `batch_size` + `checkpoint_path` to persist the DataFrame every N processed rows:

```python
result = dataframeit(
    df,
    Model,
    PROMPT,
    batch_size=100,
    checkpoint_path="checkpoint.xlsx",
)
```

The format is inferred from the file extension (`.csv`, `.xlsx`, `.parquet`). If execution is interrupted, reload the DataFrame with `read_df`, which returns lists, dicts and text with the model's types, and re-run with `resume=True`:

```python
from dataframeit import read_df

partial = read_df("checkpoint.xlsx", Model)
result = dataframeit(
    partial, Model, PROMPT,
    resume=True, batch_size=100, checkpoint_path="checkpoint.xlsx",
)
```

## Token Tracking

Monitor usage and costs with `track_tokens=True`:

```python
result = dataframeit(
    df,
    Model,
    PROMPT,
    track_tokens=True
)
```

At the end, DataFrameIt prints a summary (always in Portuguese):

```
============================================================
ESTATISTICAS DE USO
============================================================
Modelo: gpt-6-luna
Total de tokens: 15,432
  - Input:  12,345 tokens
  - Output: 3,087 tokens
------------------------------------------------------------
METRICAS DE THROUGHPUT
------------------------------------------------------------
Tempo total: 45.2s
Workers paralelos: 5
Requisicoes: 100
  - RPM (req/min): 132.7
  - TPM (tokens/min): 20,478
============================================================
```

Cache and reasoning lines appear when the provider reports those tokens. With web search, the summary gains a section on searches and credits; with `claude_code`, the cost reported by the SDK.

### Added Columns

The result records usage per row; the [LLM Reference](../reference/llm-reference.md#automatically-added-columns) defines each column and how to interpret null or zero values in `_cached_input_tokens`.

### Calculating Costs

```python
result = dataframeit(df, Model, PROMPT, track_tokens=True)

# Example: gpt-6-luna prices
price_input = 0.10 / 1_000_000    # $0.10 per 1M tokens
price_output = 0.50 / 1_000_000   # $0.50 per 1M tokens

cost_input = result['_input_tokens'].sum() * price_input
cost_output = result['_output_tokens'].sum() * price_output
total_cost = cost_input + cost_output

print(f"Estimated cost: ${total_cost:.4f}")
```

## Throughput Metrics

The throughput section of the summary above shows the effective requests and tokens per minute. Use these numbers to calibrate `parallel_requests` and `rate_limit_delay` to your account's limits.

## Optimized Configurations

### For Maximum Speed

```python
result = dataframeit(
    df,
    Model,
    PROMPT,
    parallel_requests=10,     # Many workers
    rate_limit_delay=0.0,     # No delay
    max_retries=5,            # Aggressive retry
    track_tokens=True
)
```

### For Stability

```python
result = dataframeit(
    df,
    Model,
    PROMPT,
    parallel_requests=3,      # Few workers
    rate_limit_delay=1.0,     # Conservative delay
    max_retries=3,
    base_delay=2.0,
    track_tokens=True
)
```

### For Economy

```python
result = dataframeit(
    df,
    Model,
    PROMPT,
    parallel_requests=1,      # Sequential
    rate_limit_delay=1.5,     # High delay
    model='gpt-6-luna',       # Cheap model
    track_tokens=True
)
```
