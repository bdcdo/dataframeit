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

Errors from providers via LangChain do not use these classes. They are classified by the type LangChain itself declares, by the HTTP status and by the message. The details are in [Error Handling](../guides/error-handling.md).

## Failures that stop the run

`ProviderAbortError` and its subclass `ProviderUsageLimitError` do not reach the caller of `dataframeit()`. When a row gets one of them, the run stops dispatching rows: that row and the following ones are left without status, the checkpoint, if there is one, is saved, and a warning says how many are pending and how to resume. Running again with `resume=True` processes only the pending rows; see [Checkpoints for Long Runs](../guides/performance.md#checkpoints-for-long-runs). The `codex` cases are in [Providers](../guides/providers.md#openai-codex-experimental).
