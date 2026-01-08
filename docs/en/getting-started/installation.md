# Installation

## Basic Installation

DataFrameIt uses [LangChain](https://langchain.com/) to support multiple LLM providers. Choose the provider you want to use:

=== "Google Gemini (Recommended)"

    ```bash
    pip install dataframeit[google]
    ```

    Models: `gemini-3.0-flash`, `gemini-2.5-flash`, `gemini-2.5-pro`

=== "OpenAI"

    ```bash
    pip install dataframeit[openai]
    ```

    Models: `gpt-5.2`, `gpt-5.2-mini`, `gpt-4.1`

=== "Anthropic"

    ```bash
    pip install dataframeit[anthropic]
    ```

    Models: `claude-sonnet-4.5`, `claude-opus-4.5`, `claude-haiku-4.5`

=== "All Providers"

    ```bash
    pip install dataframeit[all]
    ```

## With Polars (Optional)

If you use Polars instead of Pandas:

```bash
pip install dataframeit[google,polars]
```

## API Keys Configuration

Set the environment variable for your provider:

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

## Verifying Installation

```python
from dataframeit import dataframeit
print("DataFrameIt installed successfully!")
```

## Next Step

Now that you've installed DataFrameIt, see the [Quickstart](quickstart.md) to create your first project.
