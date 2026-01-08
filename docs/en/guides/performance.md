# Performance

Optimize processing with parallelism, rate limiting, and token tracking.

## Parallel Processing

Use `parallel_requests` to speed up processing:

```python
result = dataframeit(
    df,
    Model,
    PROMPT,
    text_column='text',
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
    text_column='text',
    rate_limit_delay=1.0  # 1 second between requests
)
```

### Calculating Ideal Delay

```
delay = 60 / requests_per_minute

Examples:
- 60 req/min  → delay = 1.0s
- 500 req/min → delay = 0.12s
- 50 req/min  → delay = 1.2s
```

### By Provider

| Provider | Free Tier | Recommended Delay |
|----------|-----------|-------------------|
| Google Gemini | 60 req/min | 1.0s |
| OpenAI (Tier 1) | 500 req/min | 0.15s |
| Anthropic (Free) | 50 req/min | 1.2s |

### Combining with Parallelism

```python
# 5 workers + delay between requests
result = dataframeit(
    df,
    Model,
    PROMPT,
    text_column='text',
    parallel_requests=5,
    rate_limit_delay=0.5
)
```

## Token Tracking

Monitor usage and costs with `track_tokens=True`:

```python
result = dataframeit(
    df,
    Model,
    PROMPT,
    text_column='text',
    track_tokens=True
)

# At the end, displays:
# ============================================================
# TOKEN USAGE STATISTICS
# ============================================================
# Model: gemini-2.0-flash
# Total tokens: 15,432
#   • Input:  12,345 tokens
#   • Output: 3,087 tokens
# ============================================================
```

### Added Columns

| Column | Description |
|--------|-------------|
| `_input_tokens` | Input tokens per row |
| `_output_tokens` | Output tokens per row |
| `_total_tokens` | Total per row |

### Calculating Costs

```python
result = dataframeit(df, Model, PROMPT, text_column='text', track_tokens=True)

# Example: Gemini 2.0 Flash prices
price_input = 0.075 / 1_000_000   # $0.075 per 1M tokens
price_output = 0.30 / 1_000_000   # $0.30 per 1M tokens

cost_input = result['_input_tokens'].sum() * price_input
cost_output = result['_output_tokens'].sum() * price_output
total_cost = cost_input + cost_output

print(f"Estimated cost: ${total_cost:.4f}")
```

## Throughput Metrics

DataFrameIt displays metrics automatically:

```
============================================================
THROUGHPUT METRICS
============================================================
Total time: 45.2s
Parallel workers: 5
Requests: 100
  - RPM (req/min): 132.7
  - TPM (tokens/min): 20,478
============================================================
```

Use these metrics to calibrate `parallel_requests` for your account.

## Optimized Configurations

### For Maximum Speed

```python
result = dataframeit(
    df,
    Model,
    PROMPT,
    text_column='text',
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
    text_column='text',
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
    text_column='text',
    parallel_requests=1,      # Sequential
    rate_limit_delay=1.5,     # High delay
    model='gemini-2.0-flash', # Cheap model
    track_tokens=True
)
```
