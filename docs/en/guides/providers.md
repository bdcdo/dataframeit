# Providers

Configure different LLM providers through LangChain or official SDKs for local tools.

## Supported Providers

| Provider | Identifier | Current models | Default model |
|----------|------------|----------------|---------------|
| OpenAI | `openai` | gpt-6-luna, gpt-6-sol, gpt-6-astra | `gpt-6-luna` |
| Google | `google_genai` | gemini-3.8-flash, gemini-3.6-flash, gemini-3.5-flash-lite | `gemini-3.8-flash` |
| Anthropic | `anthropic` | claude-sonnet-5, claude-opus-5-5, claude-haiku-4-5 | `claude-sonnet-5` |
| Groq | `groq` | openai/gpt-oss-120b, openai/gpt-oss-20b | `openai/gpt-oss-120b` |
| OpenAI Codex (experimental) | `codex` | Models supported by the bundled runtime | Chosen by the runtime |
| Claude Code | `claude_code` | Models and aliases accepted by Claude Code (`sonnet`, `haiku`, `opus`) | Chosen by the runtime |
| Cohere | `cohere` | command-r, command-r-plus | Pass `model` |
| Mistral | `mistralai` | mistral-large, mistral-small | Pass `model` |

Without `provider`, dataframeit uses `openai` with `gpt-6-luna`. With `provider` and no `model`, it uses that provider's default model from the table; providers without a default model require `model`.

## OpenAI (Default)

```bash
pip install dataframeit[openai]
export OPENAI_API_KEY="your-key"
```

```python
# Default - no need to specify
result = dataframeit(df, Model, PROMPT, text_column='text')

# With a more advanced model
result = dataframeit(
    df, Model, PROMPT,
    text_column='text',
    provider='openai',
    model='gpt-6-sol',
    model_kwargs={
        'reasoning_effort': 'medium'
    }
)
```

### Recommended Models

| Model | Use | Price (1M tokens, input / output) |
|-------|-----|-----------------------------------|
| `gpt-6-luna` | High volume, focused tasks | $0.10 / $0.50 |
| `gpt-6-sol` | Complex and agentic tasks | $2.00 / $10.00 |
| `gpt-6-astra` | Maximum quality | $10.00 / $50.00 |

## Google Gemini

```bash
pip install dataframeit[google]
export GOOGLE_API_KEY="your-key"
```

```python
result = dataframeit(
    df, Model, PROMPT,
    text_column='text',
    provider='google_genai'  # uses gemini-3.8-flash
)

# With extra parameters
result = dataframeit(
    df, Model, PROMPT,
    text_column='text',
    provider='google_genai',
    model='gemini-3.5-flash-lite',
    model_kwargs={
        'thinking_level': 'low'
    }
)
```

### Recommended Models

| Model | Use | Price (1M tokens, input / output) |
|-------|-----|-----------------------------------|
| `gemini-3.8-flash` | General use, newest | $0.75 / $3.75 through 12/31/2026; $1.50 / $7.50 from 2027 |
| `gemini-3.6-flash` | General use | $0.75 / $3.75 through 12/31/2026; $1.50 / $7.50 from 2027 |
| `gemini-3.5-flash-lite` | High volume, economical | $0.30 / $2.50 |

## OpenAI Codex (Experimental)

The `codex` provider uses the [official Python SDK](https://github.com/openai/codex/tree/main/sdk/python) and remains experimental. For extra installation, runtime selection, and local file-backed authentication, see [Installation](../getting-started/installation.md).

```python
result = dataframeit(
    df,
    Model,
    PROMPT,
    text_column='text',
    provider='codex',
    model='gpt-5.4',
    model_kwargs={'effort': 'medium'},
    parallel_requests=3,
)
```

For this provider, `model_kwargs` accepts only `effort`. `use_search=True` is not supported. The Pydantic model must have fields at the root and use the [JSON Schema subset accepted by Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs#supported-schemas); `RootModel`, `Any`, dynamic-key `dict` fields, fixed tuples, and sets are rejected during preflight. Authentication configured during installation comes from `auth.json`, so do not pass `api_key` to `dataframeit()`.

DataFrameIt keeps one `codex app-server` per DataFrame run and opens one ephemeral thread per row. Every run uses isolated `CODEX_HOME` and workspace directories; `auth.json` is the only file from Codex's persistent state linked into the runtime, which still inherits the process environment variables. While one run uses the credential, another DataFrameIt run with the same `auth.json` fails before starting the runtime; this prevents concurrent refresh without affecting `parallel_requests` within the active run. This lock coordinates DataFrameIt instances only, so do not run the Codex CLI with the same credential until processing finishes. Web search, shell access, and MCP servers are disabled; approvals are denied, and the read-only sandbox blocks writes. The runtime may still present internal utilities such as `apply_patch` without granting permission to change files.

## Claude Code

The `claude_code` provider uses the [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python), which runs the Claude Code CLI. Authentication is Claude Code's own: the credentials of a login done with a Claude Code installation on the machine (the CLI shipped with the extra is not on `PATH`), or `ANTHROPIC_API_KEY`. The `api_key` parameter is ignored.

```bash
pip install dataframeit[claude-code]
```

```python
result = dataframeit(
    df,
    Model,
    PROMPT,
    provider='claude_code',
    model='haiku',
    model_kwargs={'max_budget_usd': 0.25, 'effort': 'low'},
)
```

In `model_kwargs`, the provider reads `max_turns` (default 1), `max_budget_usd` (default 0.50) and `effort`; other keys are ignored. `max_budget_usd` caps the spending of each attempt: since an empty or off-schema answer is retried, a row can spend up to `max_retries` times that value. `use_search=True` is not supported.

Row text is treated as untrusted content. The run has no tools, no user or project settings and uses `--strict-mcp-config`, so MCP servers and `permissions.allow` rules configured in Claude Code do not reach it. A row that exceeds `max_budget_usd` or `max_turns` ends in a final error, with no retry.

With `track_tokens=True`, the summary at the end of the run shows the cost reported by the SDK, including retried attempts and rows that failed.

## Anthropic Claude

```bash
pip install dataframeit[anthropic]
export ANTHROPIC_API_KEY="your-key"
```

```python
result = dataframeit(
    df, Model, PROMPT,
    text_column='text',
    provider='anthropic'  # uses claude-sonnet-5
)

# With max_tokens
result = dataframeit(
    df, Model, PROMPT,
    text_column='text',
    provider='anthropic',
    model='claude-opus-5-5',
    model_kwargs={
        'max_tokens': 4096
    }
)
```

### Recommended Models

| Model | Use | Price (1M tokens, input / output) |
|-------|-----|-----------------------------------|
| `claude-sonnet-5` | General use, speed and quality | $2.00 / $10.00 |
| `claude-opus-5-5` | Maximum quality, agentic | $4.00 / $20.00 |
| `claude-haiku-4-5` | Fast, economical | $1.00 / $5.00 |

## Groq

```bash
pip install dataframeit[groq]
export GROQ_API_KEY="your-key"
```

```python
result = dataframeit(
    df, Model, PROMPT,
    text_column='text',
    provider='groq'  # uses openai/gpt-oss-120b
)

# Faster/cheaper model
result = dataframeit(
    df, Model, PROMPT,
    text_column='text',
    provider='groq',
    model='openai/gpt-oss-20b'
)
```

### Recommended Models

Production:

| Model | Context | Throughput | Use |
|-------|---------|-----------|-----|
| `openai/gpt-oss-120b` | 131K | ~500 t/s | General use, reasoning |
| `openai/gpt-oss-20b` | 131K | ~1000 t/s | Faster than 120b, low cost |

Preview (may change or be discontinued):

| Model | Use |
|-------|-----|
| `qwen/qwen3.8-27b` | Qwen 3.8, thinking and instruct modes |

!!! note "Availability and free tier"
    Groq offers a free tier with per-minute request limits per model. The model catalog changes frequently (especially `preview` models); check [console.groq.com/docs/models](https://console.groq.com/docs/models) for the current list and limits.

## Cohere

```bash
pip install langchain-cohere
export COHERE_API_KEY="your-key"
```

```python
result = dataframeit(
    df, Model, PROMPT,
    text_column='text',
    provider='cohere',
    model='command-r-plus'
)
```

## Mistral

```bash
pip install langchain-mistralai
export MISTRAL_API_KEY="your-key"
```

```python
result = dataframeit(
    df, Model, PROMPT,
    text_column='text',
    provider='mistralai',
    model='mistral-large-latest'
)
```

## Brazilian region (São Paulo)

The providers above use global public endpoints. To serve from Brazil — for latency, data residency or regulatory reasons — use one of the three options below. In all of them, `dataframeit` forwards `model_kwargs` straight to LangChain.

### Vertex AI (Gemini in `southamerica-east1`)

Two variants. The first one needs no extra package install:

```python
# Variant A: uses langchain-google-genai (already a dep of provider 'google_genai')
result = dataframeit(
    df, Model, PROMPT,
    text_column='text',
    provider='google_genai',
    model='gemini-3.8-flash',
    model_kwargs={
        'vertexai': True,
        'project': '<gcp-project-id>',
        'location': 'southamerica-east1',
    },
)
```

```python
# Variant B: uses langchain-google-vertexai (dedicated provider)
# pip install langchain-google-vertexai
result = dataframeit(
    df, Model, PROMPT,
    text_column='text',
    provider='google_vertexai',
    model='gemini-3.8-flash',
    model_kwargs={
        'project': '<gcp-project-id>',
        'location': 'southamerica-east1',
    },
)
```

Authentication (any variant):

```bash
gcloud auth application-default login
# OR
export GOOGLE_APPLICATION_CREDENTIALS=/path/service-account.json
```

### AWS Bedrock (`sa-east-1`)

```bash
pip install langchain-aws
aws configure  # or export AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
```

```python
result = dataframeit(
    df, Model, PROMPT,
    text_column='text',
    provider='bedrock_converse',
    model='global.anthropic.claude-sonnet-5',
    model_kwargs={'region_name': 'sa-east-1'},
)
```

On Bedrock, the Converse API only accepts Claude Sonnet 5 through an inference profile, and in `sa-east-1` the only profile offered is the global one (`global.`), which may process the request outside Brazil. If data residency is a requirement, check in the Bedrock console which models the region offers with a regional profile.

For the legacy Bedrock API (non-converse), switch to `provider='bedrock'` keeping the same `model_kwargs`. The newer API (`bedrock_converse`) is recommended for new projects.

### Azure OpenAI (Brazil South)

```bash
pip install langchain-openai
export AZURE_OPENAI_API_KEY="your-key"
export AZURE_OPENAI_ENDPOINT="https://<your-resource>.openai.azure.com/"
export OPENAI_API_VERSION="2025-03-01-preview"
```

```python
result = dataframeit(
    df, Model, PROMPT,
    text_column='text',
    provider='azure_openai',
    model='gpt-4o',  # or the deployment name
    model_kwargs={'azure_deployment': '<deployment-name>'},
)
```

The region is encoded in `AZURE_OPENAI_ENDPOINT` — provision the resource in "Brazil South" via the Azure portal.

The API version (`OPENAI_API_VERSION`) changes often. Check the latest stable version at [aka.ms/azure-openai-api-versions](https://aka.ms/azure-openai-api-versions).

!!! note "Prices change"
    Prices in the tables above are standard-tier list prices per 1M tokens, with already announced changes shown in the table itself. Check current prices on the providers' official websites.

## Passing API Key Directly

If you prefer not to use environment variables:

```python
result = dataframeit(
    df, Model, PROMPT,
    text_column='text',
    provider='openai',
    api_key='sk-...'  # Your key directly
)
```

!!! warning "Security"
    Avoid putting API keys directly in code. Prefer environment variables.

## Common Parameters (model_kwargs)

| Parameter | Description | Providers |
|-----------|-------------|-----------|
| `temperature` | Creativity. dataframeit sends no default value | Model-dependent: several current models reject it (e.g. Claude Sonnet 5, OpenAI GPT-6 with reasoning and o-series) |
| `top_p` | Nucleus sampling | Model-dependent, like `temperature` |
| `max_tokens` | Output limit | All |
