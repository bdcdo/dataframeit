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

At the end of the run, a `UserWarning` reports the initial and final workers and suggests using the final number in `parallel_requests`.

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

The format is inferred from the file extension (`.csv`, `.xlsx`, `.parquet`). In every format, the model's list, dict and tuple fields are saved as JSON text; `read_df` turns them back into structures, and resuming, into the types declared in the model. If execution is interrupted, run the same call again with the same input: with `resume=True`, the default, and the file present, execution continues from the checkpoint. On every save, `<checkpoint>.dataframeit.json` is written next to it with the run's signature (prompt, provider, model, `model_kwargs` except `timeout`, schema, search, and the text and status columns), a hash of each row's text, and a hash of the checkpoint itself. The checkpoint is resumed only when all three match; with another configuration, another input, or a file that is not the one the signature describes, the run warns, starts over, and overwrites it. In a row already concluded, successfully or with an error, the model fields come from the checkpoint, over whatever the input carried in them; in a successful row, that is what an uninterrupted run would give, because it writes the answer over them. To keep an output corrected by hand as it is, pass it with the status column, and the checkpoint is not read again. A complete output is returned without it, and the column comes back with `output["_dataframeit_status"] = "processed"` (or the name passed in `status_column`). With `resume=False`, the file is ignored and overwritten.

To inspect what has already been processed, `read_df` loads the file with lists, dicts and text in the model's types. That DataFrame also works as input for resuming, and then the file is not read again:

```python
from dataframeit import read_df

partial = read_df("checkpoint.xlsx", Model)
result = dataframeit(
    partial, Model, PROMPT,
    resume=True, batch_size=100, checkpoint_path="checkpoint.xlsx",
)
```

When the provider reports a failure that prevents any further row, such as an exhausted usage quota (`ProviderUsageLimitError`) or a terminated `codex` runtime (`ProviderAbortError`), execution stops dispatching rows, saves the checkpoint, and warns how many rows were left without status. The control columns stay in the output, and running again with `resume=True` processes only the pending rows. With `reprocess_columns`, rows the interruption left without reprocessing keep their previous values and are marked in `_error_details`.

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

At the end, DataFrameIt logs a summary (always in Portuguese) on the `dataframeit.stats` logger, at INFO level. Without logging configured, the summary goes to stderr; with logging configured, it goes through the application's handlers:

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

To silence the summary and keep the usage columns, raise the logger level:

```python
import logging

logging.getLogger("dataframeit.stats").setLevel(logging.WARNING)
```

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
