# Exceptions

`dataframeit()` handles errors in two ways, depending on when they happen.

## Errors before processing

The configuration is validated before the first call to the model. A problem here stops the run with an exception, and no row is processed.

| Exception | When it occurs |
|-----------|----------------|
| `ImportError` | The extra for the provider or the search provider is missing (e.g. `pip install dataframeit[anthropic]`). The message says which one to install |
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

A failure on a row does not stop the run. The row gets status `'error'`, the reason goes to `_error_details` and processing moves on to the next row. The text in `_error_details` has the form `[Falhou após N tentativa(s)] Class: message` (or `[Erro não-recuperável] Class: message`), which lets you filter by type:

```python
if '_dataframeit_status' in result.columns:
    errors = result[result['_dataframeit_status'] == 'error']
    overloaded = errors[errors['_error_details'].str.contains('ProviderOverloadedError', na=False)]
```

The classes below classify the failure and decide whether it gets a new attempt. All of them are available in `dataframeit` and in `dataframeit.errors`.

```
RuntimeError
└── ProviderError                     definitive failure reported by the provider
    └── ProviderTransientError        transient: gets a new attempt
        ├── ProviderOverloadedError   provider overload or rate limit
        └── ProviderRejectedOutputError  response rejected by validation (also ValueError)

ValueError
├── ProviderConfigurationError        local configuration incompatible with the provider
└── ProviderOutputError               definitive response outside the output contract
```

| Exception | New attempt | Typical situation |
|-----------|-------------|-------------------|
| `ProviderError` | No | Definitive provider error, such as rejected authentication or exhausted `claude_code` budget |
| `ProviderTransientError` | Yes | Network or service failure that usually goes away on its own |
| `ProviderOverloadedError` | Yes | Provider overload or HTTP 429 |
| `ProviderRejectedOutputError` | Yes | The response failed validation against the Pydantic model; the new attempt sends the model the rejected response and the error |
| `ProviderOutputError` | No | The provider finished without a usable response, such as a `codex` turn with no content |

Errors from providers via LangChain do not use these classes. They are classified by the type LangChain itself declares, by the HTTP status and by the message. The details are in [Error Handling](../guides/error-handling.md).
