# Exceptions

`dataframeit()` handles errors in two ways, depending on when they happen.

## Errors before processing

The configuration is validated before the first call to the model. A problem here stops the run with an exception, and no row is processed.

| Exception | When it occurs |
|-----------|----------------|
| `ImportError` | The package for the provider or the search provider is missing. The message says what to install: for LangChain providers, the provider's package (e.g. `pip install langchain-anthropic`) or the `all` extra; for `claude_code` and `codex`, the provider's extra; for web search, the search provider's package (e.g. `pip install langchain-tavily`) |
| `ValueError` | Invalid parameter: `questions` or `prompt` missing, `max_retries < 1`, `batch_size` without `checkpoint_path`, `search_groups` without `search_per_field=True`, missing search API key, previously processed rows incompatible with the current model |
| `ProviderConfigurationError` | Configuration incompatible with the provider's contract, such as a `model_kwargs` that `codex` does not accept or a Pydantic schema it cannot represent. It is a subclass of `ValueError` |

```python
from dataframeit import dataframeit, ProviderConfigurationError

try:
    result = dataframeit(df, Model, PROMPT, provider='codex', model_kwargs={'max_tokens': 500})
except ProviderConfigurationError as error:
    print(f"Configuration rejected by the provider: {error}")
```

## Errors during processing

A failure on a row does not stop the run, except the failures that prevent any further row, described in [Failures that stop the run](#failures-that-stop-the-run). The row gets status `'error'`, the reason goes to `_error_details` and processing moves on to the next row. The text in `_error_details` has the form `[Falhou após N tentativa(s)] Class: message` (or `[Erro não-recuperável] Class: message`), which lets you filter by type:

```python
if '_dataframeit_status' in result.columns:
    errors = result[result['_dataframeit_status'] == 'error']
    overloaded = errors[errors['_error_details'].str.contains('ProviderOverloadedError', na=False)]
```

The classes below classify the failure and decide whether it gets a new attempt. All of them are available in `dataframeit` and in `dataframeit.errors`.

```
RuntimeError
└── ProviderError                     definitive failure reported by the provider
    ├── ProviderTransientError        transient: gets a new attempt
    │   ├── ProviderOverloadedError   provider overload or rate limit
    │   └── ProviderRejectedOutputError  response rejected by validation (also ValueError)
    └── ProviderAbortError            prevents any further row: the run stops
        └── ProviderUsageLimitError   account usage quota exhausted

ValueError
├── ProviderConfigurationError        local configuration incompatible with the provider
└── ProviderOutputError               definitive response outside the output contract
```

| Exception | New attempt | Typical situation |
|-----------|-------------|-------------------|
| `ProviderError` | No | Definitive provider error, such as rejected authentication or exhausted `claude_code` budget |
| `ProviderTransientError` | Yes | Network or service failure that usually goes away on its own |
| `ProviderOverloadedError` | Yes | Provider overload or HTTP 429 |
| `ProviderRejectedOutputError` | Yes | The response failed validation against the Pydantic model. With LangChain providers, the new attempt sends the model the rejected response and the error; with `claude_code`, it repeats the request |
| `ProviderOutputError` | No | The provider finished without a usable response, such as a `codex` turn with no content or with a response that fails validation against the Pydantic model |
| `ProviderAbortError` | No | A failure after which no row would get through, such as a terminated `codex` runtime; the run stops |
| `ProviderUsageLimitError` | No | Account usage quota exhausted, such as the `codex` one; the run stops |

`ProviderError` and its subclasses carry in `status_code` the HTTP status the provider reported outside the message, or `None` when it reported none: in `claude_code`, the SDK's `api_error_status`; in `codex`, the HTTP status of the failed turn, and 401 for the `unauthorized` code. The box in the row failure warning comes from the class and this status, not from the message, which may carry model text. The class decides first: `ProviderOverloadedError` gives the request-limit box, with or without a status, and `ProviderRejectedOutputError` gives the validation box. For the other classes, the status decides: 401 gives the authentication box, 403 the permission box, and 408 or 504, on a `ProviderTransientError`, the timeout box. When neither the class nor the status decides, the box is the generic one, with the type and the message in up to three lines, ending in an ellipsis when it is longer. The connection box, which only the message text decides, does not appear for a `ProviderError`.

Errors from providers via LangChain do not use these classes. They are classified by the type LangChain itself declares, by the HTTP status and by the message. The details are in [Error Handling](../guides/error-handling.md).

## Failures that stop the run

`ProviderAbortError` and its subclass `ProviderUsageLimitError` do not reach the caller of `dataframeit()`. When a row gets one of them, the run stops dispatching rows: the row that got the error and the rows not yet dispatched are left without status, and, in parallel mode, the rows already in flight finish and get status `'processed'`. The checkpoint, if there is one, is saved, and a warning says how many are pending and how to resume. Running again with `resume=True` processes only the pending rows when the call receives the status of the interrupted run. With `pd.DataFrame` input, that holds for the same input, which is [modified in place](api.md#input-modified-in-place), and for the returned output. With polars input, it holds only for the returned output: the input is not changed, and repeating the call on it processes every row again. With input that is not a DataFrame, such as `list`, the output does not carry the text, and only the same `checkpoint_path` allows resuming. With `checkpoint_path`, repeating the call with the same input resumes from the checkpoint, polars and `list` included; see [Checkpoints for Long Runs](../guides/performance.md#checkpoints-for-long-runs). With `reprocess_columns`, the interruption leaves no row without status: the rows not yet reprocessed, including the one that got the error, keep status `'processed'` and their previous values, marked in `_error_details`, and `resume=True` alone does not redo them. To redo them, repeat the call with `reprocess_columns`, which processes every row again. The `codex` cases are in [Providers](../guides/providers.md#openai-codex-experimental).
