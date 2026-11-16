# Performance

Optimize processing with parallelism, rate limiting, and token tracking.

## Parallel Processing

Use `parallel_requests` to speed up processing. Up to `parallel_requests` rows are in flight at once, and a new row is dispatched as soon as another finishes: a worker that finishes early does not wait for the slowest one.

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

When a row fails after all attempts with a rate limit error, DataFrameIt halves the cap on rows in flight; the rows already in flight finish:

```
Start: 10 workers
Rate limit detected → 5 workers
Rate limit detected → 2 workers
Rate limit detected → 1 worker
```

With `codex`, the rate limit and the app-server overload are recognized by the error code or status, without reading the message. With `claude_code`, the rate limit (429) and the overload (529) are recognized by the status the SDK reports from `claude-agent-sdk` 0.1.76 on; from 0.1.48 to 0.1.75, the SDK does not report the status, and 401, 400, 429 and 529 are retried as transient failures, without reducing the workers. With the LangChain providers, the rate limit is recognized by the 429 status or the error class and, when the error carries neither, by the message text; Anthropic's 529 overload also reduces the workers, and 503 does not. The request-limit box in the row failure warning appears for the errors recognized as rate limits in this way, also when there are no workers left to reduce, as in sequential mode.

!!! info "Safety"
    Workers are only **reduced**, never automatically increased. This prevents unexpected costs.

### Per-Row Callback

`on_row_done` receives a `RowResult` for each row processed in this call, including errors, after the row is written to the DataFrame. A row skipped by `resume` is not delivered.

```python
from dataframeit import RowResult, dataframeit

def on_done(row: RowResult) -> None:
    print(row.position, row.index, row.success, row.values)

result = dataframeit(
    df,
    Model,
    PROMPT,
    parallel_requests=5,
    on_row_done=on_done,
)
```

The `RowResult` fields are in the [API reference](../reference/api.md#rowresult).

!!! warning "Worker thread"
    With `parallel_requests` > 1, the callback runs on the worker thread that processed the row while the others keep writing. Whatever it shares (a counter, a queue, a screen) needs its own synchronization, and what it needs from the row comes from the `RowResult`, not from the DataFrame. An exception inside it becomes a warning, and the run continues.

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

The format is inferred from the file extension (`.csv`, `.xlsx`, `.parquet`). In every format, the model's structure fields, that is, list, tuple, `deque`, set, dict, `Sequence`, `AbstractSet`, `Mapping`, `TypedDict`, `NamedTuple` and nested model, are saved as JSON text, sets in a fixed order, and `Enum` values by their value; a `set` or `frozenset` field of submodels already comes out of the run as a list, with the dict of each item, in the order of its JSON text, and comes back as the same list; a list, dict or tuple that an `Any` field received is also saved as JSON text, and resuming returns it as a list or dict: the tuple of an `Any` field comes back as a list, because JSON does not tell them apart; the text of an `Any` field that would read as a structure, such as `"[1, 2]"`, is saved as quoted JSON text and comes back as the same text, and in CSV and XLSX so is the text the reader would return as a number, a bool or a missing value, such as `"123"`, `"true"`, `"NaN"` and `""`; this way, in CSV, the number and the bool of an `Any` field come back as number and bool also next to a text in the same column. `read_df` turns the structures back, and resuming returns structures and date, time, `datetime`, `Enum`, `int` and `bool` fields with their declared type, also with `strict=True`: a date saved as text, an integer read back as a float and a tuple read back as a list come back as `date`, `int` and `tuple`. CSV and XLSX write an empty string and `None` as the same empty cell. When resuming, that cell comes back as `""` in a text field that does not accept `None`, such as `str` or `str = "default"`, and as `None` in a field that accepts it, such as `Optional[str]` or `str | None = "x"`, never as its default; to keep the `""` of an optional field, use parquet. In a `str` field whose default is `None`, given directly or by a factory, it also comes back as `None`. The empty string of a field whose factory reads the other fields is not restored: the empty cell counts as missing, and resuming fills it with the factory's value, even where the run saved `""`. A validator that turns an answer into `None` in a `str` field gives a value that the field's type rejects. In parquet, resuming flags that `None` for reprocessing. In CSV and XLSX, it flags the field when the validator turns the empty string itself into `None`, and returns `""` when it turns another answer, such as `"N/A"`, into `None`, because the empty cell does not say which of the two was saved. To keep that `None`, declare the field as `Optional[str]`. Parquet stores the empty string and `None` apart, and resuming returns the empty string as it was saved. In XLSX, text that starts with `=` is saved as text, not as a formula. XLSX stores numbers as doubles, and an integer above 2**53 comes back rounded; for ids that large, use CSV or parquet. The NaN of a `float` field, or of a union with `float`, is saved apart from the empty cell of a pending row and of a `None` answer: as the text `NaN` in CSV and XLSX, and as NaN, not null, in parquet, in a column that holds only floats. Resuming returns it as NaN; in a field that also accepts text, such as `Union[float, str]`, CSV and XLSX save in quotes the text `NaN` that the model returned, and it comes back as text. In parquet, the column of a union that also received an integer, a bool or a text still has the NaN saved as null. CSV and XLSX save as text the `Decimal` of a `Decimal` or `Optional[Decimal]` field, and resuming returns it with every digit and with the trailing zero: `1.10` comes back as `1.10`. The `Decimal` NaN of a field that accepts it, such as `Annotated[Decimal, Field(allow_inf_nan=True)]`, quiet or signaling, is saved as the text `NaN` or `sNaN`, in parquet with the other `Decimal` values of the column also as text, and resuming returns it as a `Decimal` NaN. In parquet, an integer above 2**63 - 1 next to a pending row and an integer above 2**53 next to a float in the same column are not saved, and the checkpoint save fails with a warning. If execution is interrupted, run the same call again with the same input: with `resume=True`, the default, and the file present, execution continues from the checkpoint. On every save, `<checkpoint>.dataframeit.json` is written next to it with the run's signature (prompt, provider, model, `model_kwargs` except `timeout`, schema, search, and the text and status columns), a hash of each row's text, and a hash of the checkpoint itself. The checkpoint is resumed only when all three match; with another configuration, another input, or a file that is not the one the signature describes, the run warns, starts over, and overwrites it. In a row already concluded, successfully or with an error, the model fields come from the checkpoint, over whatever the input carried in them. In all three formats, the `None` the run saved in a field that accepts `None` comes back as `None`, not as the field's default, as in `str | None = "x"` or `Optional[int] = 5`; the default only fills a field whose column is neither in the input nor in the checkpoint. A field with `exclude=True` is left out of `model_dump`, and the run does not write it: the checkpoint does not bring its column back, which comes back as `None` as in the run, and resuming does not check the field, even when it is required. A field that `model_dump` leaves out depending on the answer, through the field's `exclude_if` or a model `@model_serializer` that drops it from the dump, is not written either in the row where it is left out: its empty cell comes back as `None`, without the default or the empty string, and is not checked, even when the field is required; a cell with a value is checked like any other field's. In CSV and XLSX, the empty cell of such a field comes back as `None` also where the run saved `""`. A `@model_serializer` counts for a field it drops when every field is `None`; a field it drops only for some values is checked like the others. To keep an output corrected by hand as it is, pass it with the status column, and the checkpoint is not read again. A complete output is returned without it, and the column comes back with `output["_dataframeit_status"] = "processed"` (or the name passed in `status_column`). With `resume=False`, the file is ignored and overwritten.

In the signature, the schema is the model's JSON Schema. When the `json_schema_extra` of a field or of a model's `model_config`, nested ones included, holds a function, such as the condition from `field_condition`, or a set with a value that is not an integer, such as `{"in": {"pf", "mei"}}` in a `condition`, the attributes of each field take its place, with the set sorted and the function entered by its qualified name and its code: bytecode, constants and the names it reads. A `functools.partial` enters by the function it wraps and its fixed arguments; an object with `__call__`, by the code of `__call__` and the instance attributes, `__slots__` included, and, when it subclasses a built-in type such as `int` or `NamedTuple`, also by its value; a pandas `DataFrame` or `Series`, by the content of every cell, the index and the column names and dtypes; and a bound method, by the method's code and its owner: an object by its attributes, and the class of a `classmethod` by its name only. Editing the function, the `__call__`, the method or an argument of the `partial`, or changing an attribute of the object, changes the signature, and the run starts over. What the function reads from outside its own code does not enter: with another value in a global variable or in an enclosing scope, another default argument value or another body in a function it calls, the signature is the same, and resuming accepts the rows already concluded, in which the field was asked or skipped by the previous condition. To change the condition through one of these paths, pass `resume=False` or another `checkpoint_path`. The bytecode changes from one minor Python version to another, such as from 3.11 to 3.12, and the checkpoint of a model with a function is not resumed after that change. The same goes for a model that is signed by those attributes with no function at all, such as one with only a set of text in `json_schema_extra`, when one of its fields uses `Literal`, `Annotated` inside another type, `Union` or `Optional`: `Literal` and, from Python 3.13 on, `Annotated` are signed by the code of an internal `typing` function, and the union is signed by the name of its class, which is `UnionType` up to 3.13 and `Union` in 3.14.

To inspect what has already been processed, `read_df` loads the file with lists, dicts and text in the model's types. That DataFrame also works as input for resuming, and then the file is not read again:

```python
from dataframeit import read_df

partial = read_df("checkpoint.xlsx", Model)
result = dataframeit(
    partial, Model, PROMPT,
    resume=True, batch_size=100, checkpoint_path="checkpoint.xlsx",
)
```

When the provider reports a failure that prevents any further row, such as an exhausted usage quota (`ProviderUsageLimitError`) or a terminated `codex` runtime (`ProviderAbortError`), execution stops dispatching rows, saves the checkpoint, and warns how many rows were left without status. The control columns stay in the output. When running again with `resume=True` processes only the pending rows, and what happens with `reprocess_columns`, is covered in [Failures that stop the run](../reference/exceptions.md#failures-that-stop-the-run).

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

The throughput section appears only with `parallel_requests > 1`. Cache and reasoning lines appear when the provider reports those tokens. With web search, the summary gains a section on searches and credits; with `claude_code`, the cost reported by the SDK.

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
