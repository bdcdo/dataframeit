# Installation

## Basic Installation

DataFrameIt integrates multiple LLM providers through LangChain or official SDKs for local tools. Choose the provider you want to use:

=== "Google Gemini (Recommended)"

    ```bash
    pip install dataframeit[google]
    ```

    Models: `gemini-3-flash-preview`, `gemini-2.5-flash`, `gemini-2.5-pro`

=== "OpenAI"

    ```bash
    pip install dataframeit[openai]
    ```

    Models: `gpt-5.2`, `gpt-5.2-mini`, `gpt-4.1`

=== "Anthropic"

    ```bash
    pip install dataframeit[anthropic]
    ```

    Models: `claude-sonnet-4-5`, `claude-opus-4-6`, `claude-haiku-4-5`

=== "Codex (Experimental)"

    ```bash
    pip install dataframeit[codex]
    # or
    uv add "dataframeit[codex]"
    ```

    This extra pins the official Python SDK and its compatible runtime. DataFrameIt always uses that bundled runtime; an external `codex` command does not participate in execution. The provider remains experimental because the pinned SDK and runtime versions are still prereleases.

=== "All Providers"

    ```bash
    pip install dataframeit[all]
    ```

    While experimental, the Codex provider is not included in `all`; install `dataframeit[codex]` separately.

## With Polars (Optional)

If you use Polars instead of Pandas:

```bash
pip install dataframeit[google,polars]
```

## With Excel (Optional)

For `.xlsx` checkpoints or reading Excel files via `read_df()`:

```bash
pip install dataframeit[excel]
```

## Authentication Configuration

Configure the credentials for your provider:

=== "Google Gemini"

    ```bash
    export GOOGLE_API_KEY="your-google-key"
    ```

    Get your key at: [Google AI Studio](https://aistudio.google.com/apikey)

=== "OpenAI"

    ```bash
    export OPENAI_API_KEY="your-openai-key"
    ```

    Get your key at: [OpenAI Platform](https://platform.openai.com/api-keys)

=== "Anthropic"

    ```bash
    export ANTHROPIC_API_KEY="your-anthropic-key"
    ```

    Get your key at: [Anthropic Console](https://console.anthropic.com/)

=== "Codex"

    If `auth.json` does not exist yet, install the [official Codex CLI](https://learn.chatgpt.com/docs/codex/cli) and authenticate once:

    ```bash
    codex --config cli_auth_credentials_store='"file"' login
    ```

    The external CLI is used only to create `auth.json`; the explicit option prevents the credentials from being stored only in the system keyring. DataFrameIt shares only that file with an ephemeral `CODEX_HOME` and executes the runtime pinned by the extra; do not pass `api_key` to `dataframeit()` for this provider.

## Verifying Installation

```python
from dataframeit import dataframeit
print("DataFrameIt installed successfully!")
```

## Next Step

Now that you've installed DataFrameIt, see the [Quickstart](quickstart.md) to create your first project.
