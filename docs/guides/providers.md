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

**🎉 100% GRATUITO - Free tier permanente sem cartão de crédito!**

- ✅ **60 RPM** (requisições por minuto)
- ✅ **10.000 TPM** (tokens por minuto)
- ✅ **Sem limite de tempo** - use para sempre!
- ✅ **Ultra-rápido** - 200-1000+ tokens/segundo
- ✅ **Sem surpresas** - não precisa de cartão de crédito

**Cadastre-se grátis:** [console.groq.com](https://console.groq.com)

```bash
pip install dataframeit  # langchain-groq já incluído
export GROQ_API_KEY="sua-chave"  # Pegue em console.groq.com (grátis!)
```

### Uso Básico

```python
# Padrão - não precisa especificar nada!
resultado = dataframeit(df, Model, PROMPT)

# Explícito
resultado = dataframeit(
    df, Model, PROMPT,
    provider='groq',
    model='moonshotai/kimi-k2-instruct-0905'
)
```

### Otimizando para o Free Tier (Recomendado!)

Para evitar rate limits (429 errors), adicione delay entre requisições:

```python
# Opção 1: Processamento sequencial (mais simples)
resultado = dataframeit(
    df, Model, PROMPT,
    rate_limit_delay=1.0  # 1 req/segundo = 60 RPM (máximo do free tier)
)

# Opção 2: Processamento paralelo (mais rápido, REQUER ajuste do delay!)
resultado = dataframeit(
    df, Model, PROMPT,
    rate_limit_delay=2.0,      # 2s × 2 workers = 60 RPM total
    parallel_requests=2,       # 2 workers simultâneos
    track_tokens=True          # Monitore RPM em tempo real
)

# Opção 3: Conservador para datasets grandes (evita picos)
resultado = dataframeit(
    df, Model, PROMPT,
    rate_limit_delay=1.5,      # 1.5s = ~40 RPM (margem de segurança)
    track_tokens=True
)
```

**⚠️ IMPORTANTE - Cálculo do delay com paralelismo:**

O rate limit de 60 RPM é **compartilhado entre todos os workers**. Use esta fórmula:

```
rate_limit_delay = parallel_requests × (60s / 60 RPM)
                 = parallel_requests × 1.0
```

**Exemplos:**
- 1 worker: `rate_limit_delay=1.0` → 60 RPM
- 2 workers: `rate_limit_delay=2.0` → 60 RPM total (30 RPM cada)
- 3 workers: `rate_limit_delay=3.0` → 60 RPM total (20 RPM cada)
- 4 workers: `rate_limit_delay=4.0` → 60 RPM total (15 RPM cada)

**❌ Erro comum:** Usar `parallel_requests=3` com `rate_limit_delay=1.0` resulta em ~180 RPM → ERRO 429!

**💡 Dica:** Use `track_tokens=True` para ver estatísticas em tempo real:
- Requests por minuto (RPM) atual
- Tokens por minuto (TPM) atual
- Tempo restante estimado
- Progresso com barra de status

Isso ajuda a validar se você está dentro dos limites!

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

### 🔧 Troubleshooting: Rate Limits

**Erro: "429 Too Many Requests" / "Rate limit reached"**

Isso significa que você excedeu o limite de 60 RPM ou 10.000 TPM. Soluções:

1. **Adicione delay entre requisições:**
   ```python
   resultado = dataframeit(df, Model, PROMPT, rate_limit_delay=1.0)
   ```

2. **Reduza paralelismo:**
   ```python
   resultado = dataframeit(df, Model, PROMPT, parallel_requests=2)  # Ao invés de 5+
   ```

3. **Use modelo menor para economizar tokens:**
   ```python
   resultado = dataframeit(
       df, Model, PROMPT,
       model='llama-3.1-8b-instant'  # Mais rápido, consome menos tokens
   )
   ```

4. **Monitore em tempo real com track_tokens:**
   ```python
   resultado = dataframeit(df, Model, PROMPT, track_tokens=True)
   ```
   Você verá algo como:
   ```
   Processing: 100%|████████| 50/50 [00:52<00:00]
   RPM: 57.3 | TPM: 8,234 | Avg: 143.7 tokens/req
   ```

**Precisa de mais?**
- Upgrade para [Developer Tier](https://console.groq.com/settings/billing): 1.000 RPM, 500K RPD, 260K TPM
- Ou use Google Gemini 2.0 Flash: 1M TPM no free tier (mas apenas 15 RPM)

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

!!! tip "Free Tier Permanente = Groq"
    **Groq é o único com free tier permanente sem cartão de crédito!**

    - ✅ Groq: Free forever (60 RPM, 10K TPM)
    - ✅ Google Gemini: Free tier (mas limitado: 15 RPM)
    - ❌ OpenAI: Apenas $5 de créditos que expiram em 3 meses
    - ❌ Anthropic: Sem free tier

    Para começar sem gastar nada, use Groq! Para datasets muito grandes (>1000 linhas), considere Gemini 2.0 Flash (1M TPM).

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
