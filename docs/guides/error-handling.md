# Tratamento de Erros

Configure as novas tentativas e monitore os erros do processamento. Falhas de configuração, como extra faltando ou parâmetro inválido, levantam exceção antes da primeira chamada ao modelo; ver [Exceções](../reference/exceptions.md). Esta página trata das falhas por linha, que não interrompem a execução.

## Colunas de Status

O DataFrameIt adiciona automaticamente colunas de controle:

| Coluna | Valores | Descrição |
|--------|---------|-----------|
| `_dataframeit_status` | `'processed'`, `'error'`, `None` | Status do processamento; `None` é linha ainda não processada |
| `_error_details` | string ou `None` | Motivo do erro; numa linha bem-sucedida que precisou de novas tentativas, `"Sucesso após N retry(s)"`; numa linha com texto vazio, `"Texto ausente"` |

Quando nenhuma linha falha e nenhuma registra detalhe, as duas colunas são removidas da saída. Confira se a coluna existe antes de filtrar por ela.

## Verificando Erros

```python
from dataframeit import dataframeit

resultado = dataframeit(df, Model, PROMPT)

if '_dataframeit_status' not in resultado.columns:
    print("Nenhuma linha falhou.")
else:
    erros = resultado[resultado['_dataframeit_status'] == 'error']
    print(f"Total de erros: {len(erros)}")
    for idx, row in erros.iterrows():
        print(f"Linha {idx}: {row['_error_details']}")

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
    max_retries=5,        # Total de tentativas, contando a primeira (padrão: 3)
    base_delay=2.0,       # Espera antes da primeira nova tentativa (padrão: 1.0)
    max_delay=60.0        # Teto da espera (padrão: 30.0)
)
```

**Como funciona o backoff** com a configuração acima:

```
Tentativa 1: falha → espera 2s
Tentativa 2: falha → espera 4s
Tentativa 3: falha → espera 8s
Tentativa 4: falha → espera 16s
Tentativa 5: falha → marca como erro
```

A espera antes da tentativa `n + 1` é `min(base_delay × 2^(n-1), max_delay)`, acrescida de até 10% de variação aleatória para que workers paralelos não repitam a chamada no mesmo instante. Um erro permanente encerra a linha na hora, sem esgotar as tentativas.

## Tipos de Erros

### Erros Transientes (retry automático)

- **Rate limit (429)**: Muitas requisições
- **Timeout**: Servidor demorou muito
- **Erro de conexão**: Problemas de rede
- **Erro 5xx**: Problemas no servidor

### Resposta recusada pela validação (retry com o erro)

- **Erro de validação**: a resposta não passa no modelo Pydantic, inclusive nos validadores próprios (`model_validator`, `field_validator`)
- **Erro de parsing**: a resposta não é JSON válido ou não veio no formato estruturado

Nos providers do LangChain, a tentativa seguinte leva ao modelo a resposta recusada e a lista de erros, cada um com o caminho do campo e o valor recusado, e pede que ele responda de novo corrigindo esses pontos. Repetir o mesmo prompt tende a repetir o mesmo erro. Quando a resposta não é JSON, o pedido leva o texto do erro do parser.

Na OpenAI, o SDK valida a resposta dentro da chamada e levanta o erro antes de devolver a mensagem; a resposta recusada e os tokens são lidos da resposta HTTP anexada ao erro. Quando nenhuma resposta bruta está disponível, o pedido de correção vai junto do prompt, com os valores recusados. Quando uma tentativa seguinte dá certo, os tokens das recusadas entram em `_input_tokens` e `_output_tokens`, porque também são cobrados; se todas falham, a linha fica com status `error`, sem contagem de tokens, e `_error_details` diz o campo e a regra de cada erro.

### Erros Permanentes (sem retry)

- **Erro de autenticação (401/403)**: API key inválida
- **Recurso inexistente (404)**: modelo ou endpoint que o provider não conhece
- **Requisição inválida** (`BadRequestError`, `InvalidArgument`): parâmetro que o provider recusa
- **Prompt maior que a janela de contexto** (`ContextOverflowError`)
- **Configuração local incompatível com o provider**
- **Orçamento ou limite de turnos do `claude_code` esgotado** (`max_budget_usd`, `max_turns`)

## Processamento Incremental

Para datasets grandes, salve checkpoints durante a execução e continue de onde parou:

```python
# Grava o progresso a cada 100 linhas
resultado = dataframeit(
    df, Model, PROMPT,
    batch_size=100,
    checkpoint_path='parcial.xlsx',
)

# Se houver interrupção, carregue o checkpoint e continue
from dataframeit import read_df

df = read_df('parcial.xlsx', Model)
resultado = dataframeit(df, Model, PROMPT, resume=True)
resultado.to_excel('completo.xlsx', index=False)
```

!!! tip "Como funciona"
    Com `resume=True`, o DataFrameIt processa só as linhas sem `_dataframeit_status`. Linhas com `'processed'` ou `'error'` ficam como estão; para re-tentar erros, limpe o status delas, como na seção abaixo.

## Reprocessando Erros

```python
from dataframeit import read_df

# Carregar resultado com erros
df = read_df('resultado.xlsx', Model)

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
    rate_limit_delay=1.0  # cada worker pausa 1 segundo depois de cada linha
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
    model='gpt-6-sol'  # Mais capaz que o gpt-6-luna
)
```
