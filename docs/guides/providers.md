# Provedores

Configure diferentes provedores de LLM via LangChain.

## Providers Suportados

| Provider | Identificador | Modelos Populares |
|----------|---------------|-------------------|
| Google | `google_genai` | gemini-2.0-flash, gemini-1.5-pro |
| OpenAI | `openai` | gpt-4o, gpt-4o-mini, o1, o3-mini |
| Anthropic | `anthropic` | claude-3-5-sonnet, claude-3-opus |
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
    model='gemini-2.0-flash'
)

# Com parâmetros extras
resultado = dataframeit(
    df, Model, PROMPT,
    provider='google_genai',
    model='gemini-1.5-pro',
    model_kwargs={
        'temperature': 0.2,
        'top_p': 0.9
    }
)
```

### Modelos Recomendados

| Modelo | Uso | Custo |
|--------|-----|-------|
| `gemini-2.0-flash` | Uso geral, rápido | Baixo |
| `gemini-1.5-flash` | Uso geral, estável | Baixo |
| `gemini-1.5-pro` | Tarefas complexas | Médio |

## OpenAI

```bash
pip install dataframeit[openai]
export OPENAI_API_KEY="sua-chave"
```

```python
resultado = dataframeit(
    df, Model, PROMPT,
    provider='openai',
    model='gpt-4o-mini'
)

# Com reasoning (modelos o1, o3)
resultado = dataframeit(
    df, Model, PROMPT,
    provider='openai',
    model='o3-mini',
    model_kwargs={
        'reasoning_effort': 'medium'  # 'low', 'medium', 'high'
    }
)
```

### Modelos Recomendados

| Modelo | Uso | Custo |
|--------|-----|-------|
| `gpt-4o-mini` | Uso geral, econômico | Baixo |
| `gpt-4o` | Alta qualidade | Alto |
| `o3-mini` | Reasoning, problemas complexos | Médio |

## Anthropic Claude

```bash
pip install dataframeit[anthropic]
export ANTHROPIC_API_KEY="sua-chave"
```

```python
resultado = dataframeit(
    df, Model, PROMPT,
    provider='anthropic',
    model='claude-3-5-sonnet-20241022'
)

# Com max_tokens
resultado = dataframeit(
    df, Model, PROMPT,
    provider='anthropic',
    model='claude-3-5-sonnet-20241022',
    model_kwargs={
        'max_tokens': 4096
    }
)
```

### Modelos Recomendados

| Modelo | Uso | Custo |
|--------|-----|-------|
| `claude-3-5-sonnet-20241022` | Uso geral, excelente qualidade | Médio |
| `claude-3-opus-20240229` | Máxima qualidade | Alto |
| `claude-3-haiku-20240307` | Rápido, econômico | Baixo |

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

## Comparação de Preços (Aproximado)

| Provider | Modelo | Input (1M tokens) | Output (1M tokens) |
|----------|--------|-------------------|-------------------|
| Google | gemini-2.0-flash | $0.075 | $0.30 |
| Google | gemini-1.5-pro | $1.25 | $5.00 |
| OpenAI | gpt-4o-mini | $0.15 | $0.60 |
| OpenAI | gpt-4o | $2.50 | $10.00 |
| Anthropic | claude-3-5-sonnet | $3.00 | $15.00 |
| Anthropic | claude-3-haiku | $0.25 | $1.25 |

!!! note "Preços mudam"
    Verifique os preços atuais nos sites oficiais dos providers.

## Passando API Key Diretamente

Se preferir não usar variáveis de ambiente:

```python
resultado = dataframeit(
    df, Model, PROMPT,
    provider='openai',
    model='gpt-4o-mini',
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
| `reasoning_effort` | Esforço de reasoning | OpenAI (o1, o3) |
