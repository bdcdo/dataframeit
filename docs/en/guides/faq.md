# FAQ

## Installation and authentication

### `ImportError` asking you to install a package

The chosen provider needs the corresponding extra. The message says which one to install, for example:

```bash
pip install dataframeit[anthropic]
```

The extras are listed in [Installation](../getting-started/installation.md). LangChain providers without their own extra, such as Cohere and Mistral, need the `langchain-<provider>` package.

### Authentication error (401 or 403)

The API key was not found or was rejected. Check the provider's environment variable (`OPENAI_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, `GROQ_API_KEY`) in the same process that runs Python; in a notebook, set it before calling `dataframeit()`. An authentication error gets no new attempt: every row ends with status `'error'`.

The `codex` and `claude_code` providers use the local authentication of those tools and do not accept `api_key`; see [Providers](providers.md).

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

The provider rejected requests for exceeding its rate. dataframeit already halves the workers on each 429 and retries with backoff, but the stable fix is to lower `parallel_requests` or raise `rate_limit_delay`. The calculation is in [Performance](performance.md#calculating-ideal-delay). With web search, the search provider's limit is usually the tightest; see [Web Search](web-search.md#rate-limits-and-parallel-processing).

### How much will it cost?

Run a sample first (`df.sample(30)`) with `track_tokens=True`. The summary at the end of the run shows the tokens and, with search, the credits spent; multiply by the ratio between the dataset size and the sample size. Prices per model are in [Providers](providers.md).

### A long run was interrupted

Use `batch_size` and `checkpoint_path` to save progress during the run, and resume with `read_df` and `resume=True`; see [Checkpoints for Long Runs](performance.md#checkpoints-for-long-runs).
