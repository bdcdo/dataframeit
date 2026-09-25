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
    rate_limit_delay=1.0  # cada worker pausa 1 segundo depois de cada linha
)
```

### Calculando o Delay Ideal

`rate_limit_delay` é uma pausa de cada worker depois de cada linha concluída com sucesso. Com vários workers, as pausas correm em paralelo, e a taxa máxima de requisições é `parallel_requests × 60 / rate_limit_delay` por minuto. Para ficar abaixo de um limite:

```
delay = 60 × parallel_requests / requisições_por_minuto

Exemplos:
- 60 req/min,  1 worker   → delay = 1.0s
- 60 req/min,  5 workers  → delay = 5.0s
- 500 req/min, 5 workers  → delay = 0.6s
```

A taxa real fica abaixo desse teto, porque cada chamada ao modelo também leva tempo.

### Por Provider

O limite de requisições por minuto depende do modelo e do nível da conta, e muda com frequência. Consulte o valor da sua conta na página oficial e aplique a fórmula acima:

- [Google Gemini](https://ai.google.dev/gemini-api/docs/rate-limits)
- [OpenAI](https://developers.openai.com/api/docs/guides/rate-limits)
- [Anthropic](https://platform.claude.com/docs/en/api/rate-limits)
- [Groq](https://console.groq.com/docs/rate-limits)

### Combinando com Paralelismo

```python
# 5 workers, cada um pausando 0,5s depois de cada linha: até 600 req/min
resultado = dataframeit(
    df,
    Model,
    PROMPT,
    parallel_requests=5,
    rate_limit_delay=0.5
)
```

## Checkpoints em Execuções Longas

Em datasets grandes (milhares de linhas, horas de execução), um kill/crash perde
todo o progresso em memória. Use `batch_size` + `checkpoint_path` para persistir
o DataFrame a cada N linhas processadas:

```python
resultado = dataframeit(
    df,
    Model,
    PROMPT,
    batch_size=100,
    checkpoint_path="checkpoint.xlsx",
)
```

Formato inferido pela extensão do arquivo (`.csv`, `.xlsx`, `.parquet`). Em caso
de interrupção, recarregue o DataFrame com `read_df`, que devolve listas, dicts e
textos com os tipos do modelo, e re-execute com `resume=True`:

```python
from dataframeit import read_df

df_parcial = read_df("checkpoint.xlsx", Model)
resultado = dataframeit(
    df_parcial, Model, PROMPT,
    resume=True, batch_size=100, checkpoint_path="checkpoint.xlsx",
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
```

Ao final, o DataFrameIt imprime um resumo (sempre em português):

```
============================================================
ESTATISTICAS DE USO
============================================================
Modelo: gpt-6-luna
Total de tokens: 15,432
  - Input:  12,345 tokens
  - Output: 3,087 tokens
------------------------------------------------------------
METRICAS DE THROUGHPUT
------------------------------------------------------------
Tempo total: 45.2s
Workers paralelos: 5
Requisicoes: 100
  - RPM (req/min): 132.7
  - TPM (tokens/min): 20,478
============================================================
```

Linhas de cache e raciocínio aparecem quando o provider informa esses tokens. Com busca web, o resumo ganha a seção de buscas e créditos; com `claude_code`, o custo informado pelo SDK.

### Colunas Adicionadas

O resultado registra o uso por linha; a [Referência LLM](../reference/llm-reference.md#colunas-adicionadas-automaticamente) define as colunas e como interpretar valores nulos ou zero em `_cached_input_tokens`.

### Calculando Custos

```python
resultado = dataframeit(df, Model, PROMPT, track_tokens=True)

# Exemplo: preços do gpt-6-luna
preco_input = 0.10 / 1_000_000    # $0.10 por 1M tokens
preco_output = 0.50 / 1_000_000   # $0.50 por 1M tokens

custo_input = resultado['_input_tokens'].sum() * preco_input
custo_output = resultado['_output_tokens'].sum() * preco_output
custo_total = custo_input + custo_output

print(f"Custo estimado: ${custo_total:.4f}")
```

## Métricas de Throughput

A seção de throughput do resumo acima mostra as requisições e os tokens por minuto efetivos. Use esses números para calibrar `parallel_requests` e `rate_limit_delay` para os limites da sua conta.

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
    model='gpt-6-luna',       # Modelo barato
    track_tokens=True
)
```
