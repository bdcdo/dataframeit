# Providers

Configure different LLM providers via LangChain.

## Supported Providers

| Provider | Identifier | Current Models (2025) |
|----------|------------|----------------------|
| Google | `google_genai` | gemini-3-flash-preview, gemini-2.5-flash, gemini-2.5-pro |
| OpenAI | `openai` | gpt-5.2, gpt-5.2-mini, gpt-4.1 |
| Anthropic | `anthropic` | claude-sonnet-4-5, claude-opus-4-6, claude-haiku-4-5 |
| Groq | `groq` | llama-3.3-70b-versatile, llama-3.1-8b-instant, openai/gpt-oss-120b, openai/gpt-oss-20b, groq/compound |
| Cohere | `cohere` | command-r, command-r-plus |
| Mistral | `mistral` | mistral-large, mistral-small |

## Google Gemini (Default)

```bash
pip install dataframeit[google]
export GOOGLE_API_KEY="your-key"
```

```python
# Default - no need to specify
result = dataframeit(df, Model, PROMPT, text_column='text')

# Explicit
result = dataframeit(
    df, Model, PROMPT,
    text_column='text',
    provider='google_genai',
    model='gemini-3-flash-preview'
)

# With extra parameters
result = dataframeit(
    df, Model, PROMPT,
    text_column='text',
    provider='google_genai',
    model='gemini-2.5-pro',
    model_kwargs={
        'temperature': 0.2,
        'top_p': 0.9
    }
)
```

### Recommended Models

| Model | Use | Cost |
|-------|-----|------|
| `gemini-3-flash-preview` | General use, newest | Low |
| `gemini-2.5-flash` | General use, fast | Low |
| `gemini-2.5-pro` | Complex tasks, reasoning | Medium |

## OpenAI

```bash
pip install dataframeit[openai]
export OPENAI_API_KEY="your-key"
```

```python
result = dataframeit(
    df, Model, PROMPT,
    text_column='text',
    provider='openai',
    model='gpt-5.2-mini'
)

# With advanced model
result = dataframeit(
    df, Model, PROMPT,
    text_column='text',
    provider='openai',
    model='gpt-5.2',
    model_kwargs={
        'temperature': 0.2
    }
)
```

### Recommended Models

| Model | Use | Cost |
|-------|-----|------|
| `gpt-5.2-mini` | General use, economical | Low |
| `gpt-5.2` | Maximum quality | High |
| `gpt-4.1` | Coding, precise instructions | Medium |

## Anthropic Claude

```bash
pip install dataframeit[anthropic]
export ANTHROPIC_API_KEY="your-key"
```

```python
result = dataframeit(
    df, Model, PROMPT,
    text_column='text',
    provider='anthropic',
    model='claude-sonnet-4-5'
)

# With max_tokens
result = dataframeit(
    df, Model, PROMPT,
    text_column='text',
    provider='anthropic',
    model='claude-opus-4-6',
    model_kwargs={
        'max_tokens': 4096
    }
)
```

### Recommended Models

| Model | Use | Cost |
|-------|-----|------|
| `claude-sonnet-4-5` | General use, excellent quality | Medium |
| `claude-opus-4-6` | Maximum quality, agentic | High |
| `claude-haiku-4-5` | Fast, economical | Low |

## Groq

```bash
pip install dataframeit[groq]
export GROQ_API_KEY="your-key"
```

```python
result = dataframeit(
    df, Model, PROMPT,
    text_column='text',
    provider='groq',
    model='llama-3.3-70b-versatile'
)

# Faster / cheaper model
result = dataframeit(
    df, Model, PROMPT,
    text_column='text',
    provider='groq',
    model='llama-3.1-8b-instant',
    model_kwargs={
        'temperature': 0.2
    }
)

# GPT-OSS for heavier reasoning tasks
result = dataframeit(
    df, Model, PROMPT,
    text_column='text',
    provider='groq',
    model='openai/gpt-oss-120b'
)
```

### Recommended Models

Production:

| Model | Context | Throughput | Use |
|-------|---------|-----------|-----|
| `llama-3.3-70b-versatile` | 131K | ~280 t/s | General use, good quality |
| `llama-3.1-8b-instant` | 131K | ~560 t/s | High speed, economical |
| `openai/gpt-oss-120b` | 131K | ~500 t/s | Reasoning, complex tasks |
| `openai/gpt-oss-20b` | 131K | ~1000 t/s | Faster than 120b, low cost |
| `groq/compound` | - | ~450 t/s | Agentic system with built-in web search and code execution |

Preview (may change or be deprecated):

| Model | Use |
|-------|-----|
| `meta-llama/llama-4-scout-17b-16e-instruct` | Llama 4 Scout, high speed |
| `qwen/qwen3-32b` | Qwen3, good for reasoning |

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
    provider='mistral',
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
    model='gemini-2.5-flash',
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
    model='gemini-2.5-flash',
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
    model='anthropic.claude-3-5-sonnet-20240620-v1:0',
    model_kwargs={'region_name': 'sa-east-1'},
)
```

Available model IDs in `sa-east-1` change — check the Bedrock console first. For cross-region inference, use the `us.` prefix (e.g. `us.anthropic.claude-...`) and set `region_name` to whatever your account has enabled.

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

## Price Comparison (Approximate - 2025)

| Provider | Model | Input (1M tokens) | Output (1M tokens) |
|----------|-------|-------------------|-------------------|
| Google | gemini-3-flash-preview | $0.50 | $3.00 |
| Google | gemini-2.5-pro | $1.25 | $5.00 |
| OpenAI | gpt-5.2-mini | $0.30 | $1.20 |
| OpenAI | gpt-5.2 | $5.00 | $15.00 |
| Anthropic | claude-sonnet-4-5 | $3.00 | $15.00 |
| Anthropic | claude-haiku-4-5 | $1.00 | $5.00 |

!!! note "Prices change"
    Check current prices on provider official websites.

## Passing API Key Directly

If you prefer not to use environment variables:

```python
result = dataframeit(
    df, Model, PROMPT,
    text_column='text',
    provider='openai',
    model='gpt-5.2-mini',
    api_key='sk-...'  # Your key directly
)
```

!!! warning "Security"
    Avoid putting API keys directly in code. Prefer environment variables.

## Common Parameters (model_kwargs)

| Parameter | Description | Providers |
|-----------|-------------|-----------|
| `temperature` | Creativity (0-1) | All |
| `top_p` | Nucleus sampling | All |
| `max_tokens` | Output limit | All |
