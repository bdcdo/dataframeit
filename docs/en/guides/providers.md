# Providers

Configure different LLM providers via LangChain.

## Supported Providers

| Provider | Identifier | Popular Models |
|----------|------------|----------------|
| Google | `google_genai` | gemini-2.0-flash, gemini-1.5-pro |
| OpenAI | `openai` | gpt-4o, gpt-4o-mini, o1, o3-mini |
| Anthropic | `anthropic` | claude-3-5-sonnet, claude-3-opus |
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
    model='gemini-2.0-flash'
)

# With extra parameters
result = dataframeit(
    df, Model, PROMPT,
    text_column='text',
    provider='google_genai',
    model='gemini-1.5-pro',
    model_kwargs={
        'temperature': 0.2,
        'top_p': 0.9
    }
)
```

### Recommended Models

| Model | Use | Cost |
|-------|-----|------|
| `gemini-2.0-flash` | General use, fast | Low |
| `gemini-1.5-flash` | General use, stable | Low |
| `gemini-1.5-pro` | Complex tasks | Medium |

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
    model='gpt-4o-mini'
)

# With reasoning (o1, o3 models)
result = dataframeit(
    df, Model, PROMPT,
    text_column='text',
    provider='openai',
    model='o3-mini',
    model_kwargs={
        'reasoning_effort': 'medium'  # 'low', 'medium', 'high'
    }
)
```

### Recommended Models

| Model | Use | Cost |
|-------|-----|------|
| `gpt-4o-mini` | General use, economical | Low |
| `gpt-4o` | High quality | High |
| `o3-mini` | Reasoning, complex problems | Medium |

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
    model='claude-3-5-sonnet-20241022'
)

# With max_tokens
result = dataframeit(
    df, Model, PROMPT,
    text_column='text',
    provider='anthropic',
    model='claude-3-5-sonnet-20241022',
    model_kwargs={
        'max_tokens': 4096
    }
)
```

### Recommended Models

| Model | Use | Cost |
|-------|-----|------|
| `claude-3-5-sonnet-20241022` | General use, excellent quality | Medium |
| `claude-3-opus-20240229` | Maximum quality | High |
| `claude-3-haiku-20240307` | Fast, economical | Low |

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

## Price Comparison (Approximate)

| Provider | Model | Input (1M tokens) | Output (1M tokens) |
|----------|-------|-------------------|-------------------|
| Google | gemini-2.0-flash | $0.075 | $0.30 |
| Google | gemini-1.5-pro | $1.25 | $5.00 |
| OpenAI | gpt-4o-mini | $0.15 | $0.60 |
| OpenAI | gpt-4o | $2.50 | $10.00 |
| Anthropic | claude-3-5-sonnet | $3.00 | $15.00 |
| Anthropic | claude-3-haiku | $0.25 | $1.25 |

!!! note "Prices change"
    Check current prices on provider official websites.

## Passing API Key Directly

If you prefer not to use environment variables:

```python
result = dataframeit(
    df, Model, PROMPT,
    text_column='text',
    provider='openai',
    model='gpt-4o-mini',
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
| `reasoning_effort` | Reasoning effort | OpenAI (o1, o3) |
