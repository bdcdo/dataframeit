# Provedores

Configure diferentes provedores de LLM via LangChain.

## Providers Suportados

| Provider | Identificador | Modelos Atuais (2025/2026) |
|----------|---------------|----------------------|
| **Groq** ⭐ | `groq` | moonshotai/kimi-k2-instruct-0905, llama-3.1-8b-instant, llama-3.3-70b-versatile |
| Google | `google_genai` | gemini-2.0-flash, gemini-2.5-flash, gemini-2.5-pro |
| OpenAI | `openai` | gpt-5.2, gpt-5.2-mini, gpt-4.1 |
| Anthropic | `anthropic` | claude-sonnet-4.5, claude-opus-4.5, claude-haiku-4.5 |
| Cohere | `cohere` | command-r, command-r-plus |
| Mistral | `mistral` | mistral-large, mistral-small |

## Groq (Padrão) ⚡

**Free tier permanente:** 60 RPM, 10.000 TPM - Ultra-rápido e gratuito!

```bash
pip install dataframeit  # langchain-groq já incluído
export GROQ_API_KEY="sua-chave"
```

```python
# Padrão - não precisa especificar
resultado = dataframeit(df, Model, PROMPT)

# Explícito
resultado = dataframeit(
    df, Model, PROMPT,
    provider='groq',
    model='moonshotai/kimi-k2-instruct-0905'
)

# Com parâmetros extras
resultado = dataframeit(
    df, Model, PROMPT,
    provider='groq',
    model='moonshotai/kimi-k2-instruct-0905',
    model_kwargs={
        'temperature': 0.2
    }
)
```

### Modelos Recomendados

| Modelo | Parâmetros | Context | Velocidade | Custo | Uso |
|--------|-----------|---------|-----------|-------|-----|
| `moonshotai/kimi-k2-instruct-0905` ⭐ | 1T (32B ativos) | 256K | 200+ t/s | $1.00/$3.00 | **Default** - Melhor equilíbrio |
| `llama-3.1-8b-instant` | 8B | 128K | 1000+ t/s | $0.05/$0.08 | Mais rápido, mais barato |
| `llama-3.3-70b-versatile` | 70B | 128K | 276 t/s | $0.59/$0.79 | Mais qualidade |

**Por que Groq como default?**
- ✅ Free tier permanente e generoso (60 RPM, 10K TPM)
- ✅ Ultra-rápido (200-1000+ tokens/segundo)
- ✅ Kimi K2: 256K context, maior do Groq
- ✅ Structured outputs + Function calling nativos
- ✅ Prompt caching com 50% desconto
- ✅ Open-source friendly (modelos Apache 2.0)

## Google Gemini

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
    model='gemini-2.5-pro',
    model_kwargs={
        'temperature': 0.2,
        'top_p': 0.9
    }
)
```

### Modelos Recomendados

| Modelo | Context | Free Tier TPM | Custo | Uso |
|--------|---------|---------------|-------|-----|
| `gemini-2.0-flash` | 1M | 1.000.000 TPM 🏆 | $0.10/$0.40 | Datasets grandes |
| `gemini-2.5-flash-lite` | 1M | 250.000 TPM | Muito baixo | Rápido e econômico |
| `gemini-2.5-pro` | 2M | Limitado | $1.25/$5.00 | Tarefas complexas |

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

## Comparação de Preços (Aproximado - 2026)

| Provider | Modelo | Input (1M tokens) | Output (1M tokens) | Free Tier |
|----------|--------|-------------------|-------------------|-----------|
| **Groq** | kimi-k2-instruct-0905 | $1.00 | $3.00 | ✅ 60 RPM, 10K TPM |
| **Groq** | llama-3.1-8b-instant | $0.05 | $0.08 | ✅ 30 RPM, 6K TPM |
| Google | gemini-2.0-flash | $0.10 | $0.40 | ✅ 15 RPM, 1M TPM 🏆 |
| Google | gemini-2.5-pro | $1.25 | $5.00 | ✅ Limitado |
| OpenAI | gpt-5.2-mini | $0.30 | $1.20 | ❌ $5 por 3 meses |
| OpenAI | gpt-5.2 | $5.00 | $15.00 | ❌ |
| Anthropic | claude-sonnet-4.5 | $3.00 | $15.00 | ❌ |
| Anthropic | claude-haiku-4.5 | $1.00 | $5.00 | ❌ |

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
