# Providers

Configure different LLM providers via LangChain.

## Supported Providers

| Provider | Identifier | Current Models (2025) |
|----------|------------|----------------------|
| Google | `google_genai` | gemini-3.0-flash, gemini-2.5-flash, gemini-2.5-pro |
| OpenAI | `openai` | gpt-5, gpt-5-mini, gpt-4.1, o3, o4-mini |
| Anthropic | `anthropic` | claude-sonnet-4.5, claude-opus-4.5, claude-haiku-4.5 |
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
    model='gemini-3.0-flash'
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
| `gemini-3.0-flash` | General use, newest | Low |
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
    model='gpt-5-mini'
)

# With reasoning (o3, o4 models)
result = dataframeit(
    df, Model, PROMPT,
    text_column='text',
    provider='openai',
    model='o4-mini',
    model_kwargs={
        'reasoning_effort': 'medium'  # 'low', 'medium', 'high'
    }
)
```

### Recommended Models

| Model | Use | Cost |
|-------|-----|------|
| `gpt-5-mini` | General use, economical | Low |
| `gpt-5` | Maximum quality | High |
| `gpt-4.1` | Coding, precise instructions | Medium |
| `o4-mini` | Reasoning, STEM | Medium |
| `o3` | Advanced reasoning | High |

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
    model='claude-sonnet-4.5'
)

# With max_tokens
result = dataframeit(
    df, Model, PROMPT,
    text_column='text',
    provider='anthropic',
    model='claude-opus-4.5',
    model_kwargs={
        'max_tokens': 4096
    }
)
```

### Recommended Models

| Model | Use | Cost |
|-------|-----|------|
| `claude-sonnet-4.5` | General use, excellent quality | Medium |
| `claude-opus-4.5` | Maximum quality, agentic | High |
| `claude-haiku-4.5` | Fast, economical | Low |

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

## Price Comparison (Approximate - 2025)

| Provider | Model | Input (1M tokens) | Output (1M tokens) |
|----------|-------|-------------------|-------------------|
| Google | gemini-3.0-flash | $0.50 | $3.00 |
| Google | gemini-2.5-pro | $1.25 | $5.00 |
| OpenAI | gpt-5-mini | $0.30 | $1.20 |
| OpenAI | gpt-5 | $5.00 | $15.00 |
| Anthropic | claude-sonnet-4.5 | $3.00 | $15.00 |
| Anthropic | claude-haiku-4.5 | $1.00 | $5.00 |

!!! note "Prices change"
    Check current prices on provider official websites.

## Passing API Key Directly

If you prefer not to use environment variables:

```python
result = dataframeit(
    df, Model, PROMPT,
    text_column='text',
    provider='openai',
    model='gpt-5-mini',
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
| `reasoning_effort` | Reasoning effort | OpenAI (o3, o4) |
