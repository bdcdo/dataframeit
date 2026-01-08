# Provedores

Configure diferentes provedores de LLM via LangChain.

## Providers Suportados

| Provider | Identificador | Modelos Atuais (2025/2026) |
|----------|---------------|----------------------|
| Google | `google_genai` | gemini-3.0-flash, gemini-2.5-flash, gemini-2.5-pro |
| OpenAI | `openai` | gpt-5.2, gpt-5.2-mini, gpt-4.1 |
| Anthropic | `anthropic` | claude-sonnet-4.5, claude-opus-4.5, claude-haiku-4.5 |
| Cohere | `cohere` | command-r, command-r-plus |
| Mistral | `mistral` | mistral-large, mistral-small |

## Google Gemini (Padrão)

```bash
pip install dataframeit[google]
export GOOGLE_API_KEY="sua-chave"
```

```python
# Padrão - não precisa especificar
resultado = dataframeit(df, Model, PROMPT)

# Explícito
resultado = dataframeit(
    df, Model, PROMPT,
    provider='google_genai',
    model='gemini-3.0-flash'
)

# Com parâmetros extras
resultado = dataframeit(
    df, Model, PROMPT,
    provider='google_genai',
    model='gemini-2.5-pro',
    model_kwargs={
        'temperature': 0.2,
        'top_p': 0.9
    }
)
```

### Modelos Recomendados

| Modelo | Uso | Custo |
|--------|-----|-------|
| `gemini-3.0-flash` | Uso geral, mais recente | Baixo |
| `gemini-2.5-flash` | Uso geral, rápido | Baixo |
| `gemini-2.5-pro` | Tarefas complexas, reasoning | Médio |

## OpenAI

```bash
pip install dataframeit[openai]
export OPENAI_API_KEY="sua-chave"
```

```python
resultado = dataframeit(
    df, Model, PROMPT,
    provider='openai',
    model='gpt-5.2-mini'
)

# Com modelo mais avançado
resultado = dataframeit(
    df, Model, PROMPT,
    provider='openai',
    model='gpt-5.2',
    model_kwargs={
        'temperature': 0.2
    }
)
```

### Modelos Recomendados

| Modelo | Uso | Custo |
|--------|-----|-------|
| `gpt-5.2-mini` | Uso geral, econômico | Baixo |
| `gpt-5.2` | Máxima qualidade | Alto |
| `gpt-4.1` | Coding, instruções precisas | Médio |

## Anthropic Claude

```bash
pip install dataframeit[anthropic]
export ANTHROPIC_API_KEY="sua-chave"
```

```python
resultado = dataframeit(
    df, Model, PROMPT,
    provider='anthropic',
    model='claude-sonnet-4.5'
)

# Com max_tokens
resultado = dataframeit(
    df, Model, PROMPT,
    provider='anthropic',
    model='claude-opus-4.5',
    model_kwargs={
        'max_tokens': 4096
    }
)
```

### Modelos Recomendados

| Modelo | Uso | Custo |
|--------|-----|-------|
| `claude-sonnet-4.5` | Uso geral, excelente qualidade | Médio |
| `claude-opus-4.5` | Máxima qualidade, agentic | Alto |
| `claude-haiku-4.5` | Rápido, econômico | Baixo |

## Cohere

```bash
pip install langchain-cohere
export COHERE_API_KEY="sua-chave"
```

```python
resultado = dataframeit(
    df, Model, PROMPT,
    provider='cohere',
    model='command-r-plus'
)
```

## Mistral

```bash
pip install langchain-mistralai
export MISTRAL_API_KEY="sua-chave"
```

```python
resultado = dataframeit(
    df, Model, PROMPT,
    provider='mistral',
    model='mistral-large-latest'
)
```

## Comparação de Preços (Aproximado - 2025)

| Provider | Modelo | Input (1M tokens) | Output (1M tokens) |
|----------|--------|-------------------|-------------------|
| Google | gemini-3.0-flash | $0.50 | $3.00 |
| Google | gemini-2.5-pro | $1.25 | $5.00 |
| OpenAI | gpt-5.2-mini | $0.30 | $1.20 |
| OpenAI | gpt-5.2 | $5.00 | $15.00 |
| Anthropic | claude-sonnet-4.5 | $3.00 | $15.00 |
| Anthropic | claude-haiku-4.5 | $1.00 | $5.00 |

!!! note "Preços mudam"
    Verifique os preços atuais nos sites oficiais dos providers.

## Passando API Key Diretamente

Se preferir não usar variáveis de ambiente:

```python
resultado = dataframeit(
    df, Model, PROMPT,
    provider='openai',
    model='gpt-5.2-mini',
    api_key='sk-...'  # Sua chave diretamente
)
```

!!! warning "Segurança"
    Evite colocar API keys diretamente no código. Prefira variáveis de ambiente.

## Parâmetros Comuns (model_kwargs)

| Parâmetro | Descrição | Providers |
|-----------|-----------|-----------|
| `temperature` | Criatividade (0-1) | Todos |
| `top_p` | Nucleus sampling | Todos |
| `max_tokens` | Limite de saída | Todos |
