# FAQ

## Installation and authentication

### `ImportError` asking you to install a package

The chosen provider needs its package, and the message says what to install. For LangChain providers, it suggests the provider's package or the `all` extra:

```bash
pip install langchain-anthropic
# or
pip install dataframeit[all]
```

The provider's extra, such as `dataframeit[anthropic]`, installs the same package. `all` brings the providers that have an extra, except Codex; LangChain providers without their own extra, such as Cohere and Mistral, need the `langchain-<provider>` package, which `all` does not include. For `claude_code` and `codex`, the message asks for the provider's extra (`dataframeit[claude-code]`, `dataframeit[codex]`), and for web search, the search provider's package. The extras are listed in [Installation](../getting-started/installation.md).

### Authentication error (401 or 403)

The API key was not found or was rejected. Check the provider's environment variable (`OPENAI_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, `GROQ_API_KEY`) in the same process that runs Python; in a notebook, set it before calling `dataframeit()`. An authentication error gets no new attempt: every row ends with status `'error'`.

The `codex` and `claude_code` providers use the local authentication of those tools. `codex` rejects `api_key` with `ProviderConfigurationError`, and `claude_code` ignores it; see [Providers](providers.md).

## Result

### `KeyError: '_dataframeit_status'`

When no row fails and none records any detail, the status column and `_error_details` are removed from the output. Check that the column exists before filtering:

```python
if '_dataframeit_status' in result.columns:
    errors = result[result['_dataframeit_status'] == 'error']
```

### `ValueError` about the text column

Without `text_column`, dataframeit looks for the columns `texto`, `text`, `decisao`, `content` and `content_text`, in that order, or uses the DataFrame's only column. With several columns and none of these names, set the column:

```python
result = dataframeit(df, Model, PROMPT, text_column='comment')
```

### Some rows ended up with `"Texto ausente"`

Rows with empty or null text are not sent to the model. They get status `'error'` and this detail, so that they do not pass as processed.

### I ran it again and the rows with errors were not redone

With `resume=True` (default), only rows without status are processed. To redo the ones that failed, clear their status; see [Reprocessing Errors](error-handling.md#reprocessing-errors).

### The model columns already exist and nothing was processed

With `resume=False`, if the model columns are already in the DataFrame and `reprocess_columns` was not passed, dataframeit emits a warning and returns the data unprocessed, so as not to overwrite results. Use `resume=True` to continue, or `reprocess_columns=[...]` to redo specific columns.

## Limits and costs

### Many 429 errors (rate limit)

The provider rejected requests for exceeding its rate. dataframeit retries with backoff and, when the row fails after all attempts, halves the workers. With `codex`, the rate limit and the app-server overload are recognized by the error code or status, without reading the message; with `claude_code`, the rate limit (429) and the overload (529) are recognized by the status the SDK reports from `claude-agent-sdk` 0.1.76 on (from 0.1.48 to 0.1.75, without the status, they are retried as transient failures and do not reduce the workers); with the LangChain providers, the rate limit is recognized by the 429 status or the error class and, when the error carries neither, by the message text. The stable fix is to lower `parallel_requests` or raise `rate_limit_delay`. The calculation is in [Performance](performance.md#calculating-ideal-delay). With web search, the search provider's limit is usually the tightest; see [Web Search](web-search.md#rate-limits-and-parallel-processing).

### How much will it cost?

Run a sample first (`df.sample(30)`) with `track_tokens=True`. The summary at the end of the run shows the tokens and, with search, the credits spent; multiply by the ratio between the dataset size and the sample size. Prices per model are in [Providers](providers.md).

### A long run was interrupted

Use `batch_size` and `checkpoint_path` to save progress during the run, and resume by running the same call again, which continues from the file; see [Checkpoints for Long Runs](performance.md#checkpoints-for-long-runs).
