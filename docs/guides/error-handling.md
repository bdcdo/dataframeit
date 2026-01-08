# Tratamento de Erros

Configure retry, fallbacks e monitore erros no processamento.

## Colunas de Status

O DataFrameIt adiciona automaticamente colunas de controle:

| Coluna | Valores | Descrição |
|--------|---------|-----------|
| `_dataframeit_status` | `'processed'`, `'error'`, `None` | Status do processamento |
| `_error_details` | string ou `None` | Detalhes do erro |

## Verificando Erros

```python
from dataframeit import dataframeit

resultado = dataframeit(df, Model, PROMPT)

# Contar erros
total_erros = (resultado['_dataframeit_status'] == 'error').sum()
print(f"Total de erros: {total_erros}")

# Filtrar linhas com erro
erros = resultado[resultado['_dataframeit_status'] == 'error']
for idx, row in erros.iterrows():
    print(f"Linha {idx}: {row['_error_details']}")

# Filtrar apenas sucesso
sucesso = resultado[resultado['_dataframeit_status'] == 'processed']
sucesso.to_excel('resultado_limpo.xlsx', index=False)
```

## Configurando Retry

O DataFrameIt usa backoff exponencial para retry automático:

```python
resultado = dataframeit(
    df,
    Model,
    PROMPT,
    max_retries=5,        # Máximo de tentativas (padrão: 3)
    base_delay=2.0,       # Delay inicial em segundos (padrão: 1.0)
    max_delay=60.0        # Delay máximo em segundos (padrão: 30.0)
)
```

**Como funciona o backoff:**

```
Tentativa 1: falha → espera 2s
Tentativa 2: falha → espera 4s
Tentativa 3: falha → espera 8s
Tentativa 4: falha → espera 16s
Tentativa 5: falha → espera 32s (limitado a 60s)
Tentativa 6: falha → marca como erro
```

## Tipos de Erros

### Erros Transientes (retry automático)

- **Rate limit (429)**: Muitas requisições
- **Timeout**: Servidor demorou muito
- **Erro de conexão**: Problemas de rede
- **Erro 5xx**: Problemas no servidor

### Erros Permanentes (sem retry)

- **Erro de validação**: Resposta não segue o modelo Pydantic
- **Erro de autenticação (401/403)**: API key inválida
- **Erro de parsing**: Resposta mal formatada

## Processamento Incremental

Para datasets grandes, use `resume=True` para continuar de onde parou:

```python
# Primeira execução
resultado = dataframeit(df, Model, PROMPT, resume=True)
resultado.to_excel('parcial.xlsx', index=False)

# Se houver interrupção, carregue e continue
df = pd.read_excel('parcial.xlsx')
resultado = dataframeit(df, Model, PROMPT, resume=True)
resultado.to_excel('completo.xlsx', index=False)
```

!!! tip "Como funciona"
    Com `resume=True`, o DataFrameIt pula linhas que já têm `_dataframeit_status == 'processed'`.

## Reprocessando Erros

```python
# Carregar resultado com erros
df = pd.read_excel('resultado.xlsx')

# Limpar status das linhas com erro para reprocessar
df.loc[df['_dataframeit_status'] == 'error', '_dataframeit_status'] = None
df.loc[df['_error_details'].notna(), '_error_details'] = None

# Reprocessar apenas as linhas sem status
resultado = dataframeit(df, Model, PROMPT, resume=True)
```

## Estratégias para Reduzir Erros

### 1. Use Rate Limiting

```python
# Previne erros de rate limit
resultado = dataframeit(
    df, Model, PROMPT,
    rate_limit_delay=1.0  # 1 segundo entre requisições
)
```

### 2. Simplifique o Modelo

```python
# Modelo muito complexo pode falhar
class ModeloComplexo(BaseModel):
    campo1: str
    campo2: List[SubModelo]
    campo3: Dict[str, OutroModelo]  # Evite se possível

# Modelo mais simples = menos erros
class ModeloSimples(BaseModel):
    campo1: str
    campo2: List[str]
```

### 3. Melhore o Prompt

```python
# Prompt vago
PROMPT_RUIM = "Analise o texto."

# Prompt claro
PROMPT_BOM = """
Analise o texto e extraia:
1. Sentimento geral (positivo, negativo ou neutro)
2. Confiança na classificação (alta, média ou baixa)

Se o texto for ambíguo, classifique como neutro com confiança baixa.
"""
```

### 4. Use Modelos Mais Capazes

```python
# Se erros persistem, tente um modelo mais capaz
resultado = dataframeit(
    df, Model, PROMPT,
    model='gemini-1.5-pro'  # Mais capaz que flash
)
```
