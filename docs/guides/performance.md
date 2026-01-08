# Performance

Otimize o processamento com paralelismo, rate limiting e tracking de tokens.

## Processamento Paralelo

Use `parallel_requests` para acelerar o processamento:

```python
resultado = dataframeit(
    df,
    Model,
    PROMPT,
    parallel_requests=5  # 5 requisições simultâneas
)
```

### Recomendações por Tamanho

| Dataset | Configuração |
|---------|--------------|
| < 50 linhas | `parallel_requests=1` (padrão) |
| 50-500 linhas | `parallel_requests=3` a `5` |
| > 500 linhas | `parallel_requests=5` a `10` |

### Auto-redução em Rate Limits

Quando detecta erro 429, o DataFrameIt reduz workers automaticamente:

```
Início: 10 workers
Rate limit detectado → 5 workers
Rate limit detectado → 2 workers
Rate limit detectado → 1 worker
```

!!! info "Segurança"
    Workers são apenas **reduzidos**, nunca aumentados automaticamente. Isso evita custos inesperados.

## Rate Limiting

Use `rate_limit_delay` para prevenir erros de rate limit:

```python
resultado = dataframeit(
    df,
    Model,
    PROMPT,
    rate_limit_delay=1.0  # 1 segundo entre requisições
)
```

### Calculando o Delay Ideal

```
delay = 60 / requisições_por_minuto

Exemplos:
- 60 req/min  → delay = 1.0s
- 500 req/min → delay = 0.12s
- 50 req/min  → delay = 1.2s
```

### Por Provider

| Provider | Free Tier | Delay Recomendado |
|----------|-----------|-------------------|
| Google Gemini | 60 req/min | 1.0s |
| OpenAI (Tier 1) | 500 req/min | 0.15s |
| Anthropic (Free) | 50 req/min | 1.2s |

### Combinando com Paralelismo

```python
# 5 workers + delay entre requisições
resultado = dataframeit(
    df,
    Model,
    PROMPT,
    parallel_requests=5,
    rate_limit_delay=0.5
)
```

## Tracking de Tokens

Monitore uso e custos com `track_tokens=True`:

```python
resultado = dataframeit(
    df,
    Model,
    PROMPT,
    track_tokens=True
)

# Ao final, exibe:
# ============================================================
# ESTATÍSTICAS DE USO DE TOKENS
# ============================================================
# Modelo: gemini-2.0-flash
# Total de tokens: 15,432
#   • Input:  12,345 tokens
#   • Output: 3,087 tokens
# ============================================================
```

### Colunas Adicionadas

| Coluna | Descrição |
|--------|-----------|
| `_input_tokens` | Tokens de entrada por linha |
| `_output_tokens` | Tokens de saída por linha |
| `_total_tokens` | Total por linha |

### Calculando Custos

```python
resultado = dataframeit(df, Model, PROMPT, track_tokens=True)

# Exemplo: preços Gemini 2.0 Flash
preco_input = 0.075 / 1_000_000   # $0.075 por 1M tokens
preco_output = 0.30 / 1_000_000   # $0.30 por 1M tokens

custo_input = resultado['_input_tokens'].sum() * preco_input
custo_output = resultado['_output_tokens'].sum() * preco_output
custo_total = custo_input + custo_output

print(f"Custo estimado: ${custo_total:.4f}")
```

## Métricas de Throughput

O DataFrameIt exibe métricas automaticamente:

```
============================================================
MÉTRICAS DE THROUGHPUT
============================================================
Tempo total: 45.2s
Workers paralelos: 5
Requisições: 100
  - RPM (req/min): 132.7
  - TPM (tokens/min): 20,478
============================================================
```

Use essas métricas para calibrar `parallel_requests` para sua conta.

## Configuração Otimizada

### Para Máxima Velocidade

```python
resultado = dataframeit(
    df,
    Model,
    PROMPT,
    parallel_requests=10,     # Muitos workers
    rate_limit_delay=0.0,     # Sem delay
    max_retries=5,            # Retry agressivo
    track_tokens=True
)
```

### Para Estabilidade

```python
resultado = dataframeit(
    df,
    Model,
    PROMPT,
    parallel_requests=3,      # Poucos workers
    rate_limit_delay=1.0,     # Delay conservador
    max_retries=3,
    base_delay=2.0,
    track_tokens=True
)
```

### Para Economia

```python
resultado = dataframeit(
    df,
    Model,
    PROMPT,
    parallel_requests=1,      # Sequencial
    rate_limit_delay=1.5,     # Delay alto
    model='gemini-2.0-flash', # Modelo barato
    track_tokens=True
)
```
