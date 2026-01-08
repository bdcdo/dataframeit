# DataFrameIt

Uma biblioteca Python para enriquecer DataFrames com análises de texto usando Modelos de Linguagem (LLMs).

## Descrição

DataFrameIt é uma ferramenta que permite processar textos contidos em um DataFrame e extrair informações estruturadas usando LLMs. A biblioteca suporta tanto **LangChain** quanto **OpenAI** como provedores de modelos. Pandas é utilizado para manipulação de dados, com suporte para Polars via conversão interna.

## Funcionalidades

- Processar cada linha de um DataFrame que contenha textos
- Utilizar prompt templates para análise específica de domínio
- Extrair informações estruturadas usando modelos Pydantic
- **Suporte híbrido**: LangChain (Gemini, etc.) ou OpenAI (GPT-4, etc.)
- **Múltiplos tipos de dados**: DataFrames, Series, listas e dicionários
- Suporte para Polars e Pandas
- Processamento incremental com resumo automático
- **Processamento paralelo** com auto-redução de workers em rate limits
- **Retry automático** com backoff exponencial para resiliência
- **Rate limiting** configurável para respeitar limites de APIs
- **Rastreamento de erros** com coluna automática `_error_details`
- **Tracking de tokens** e métricas de throughput (RPM, TPM)

## Instalação

### Dependências Base
```bash
pip install dataframeit
```

### Para usar OpenAI
```bash
pip install dataframeit openai
```

### Para usar LangChain
```bash
# Dependências base do LangChain
pip install dataframeit langchain langchain-core

# Para Google Gemini (provider padrão)
pip install langchain-google-genai

# Para outros providers (exemplos)
pip install langchain-anthropic  # Claude
pip install langchain-openai     # GPT via LangChain
```

### Para usar Polars
```bash
pip install dataframeit[polars]
```

## Uso Básico

### Com LangChain (comportamento padrão)

```python
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
from dataframeit import dataframeit

# Defina um modelo Pydantic para estruturar as respostas
class SuaClasse(BaseModel):
    campo1: str = Field(..., description="Descrição do campo 1")
    campo2: Literal['opcao1', 'opcao2'] = Field(..., description="Descrição do campo 2")

# Defina seu template de prompt
TEMPLATE = "Classifique o texto conforme as categorias definidas."

# Carregue seus dados (a coluna de texto deve se chamar 'texto' por padrão)
df = pd.read_excel('seu_arquivo.xlsx')

# Processe os dados (usa LangChain por padrão)
df_resultado = dataframeit(df, SuaClasse, TEMPLATE)

# Salve o resultado
df_resultado.to_excel('resultado.xlsx', index=False)
```

### Com OpenAI

```python
from openai import OpenAI
from dataframeit import dataframeit

# Configure seu cliente OpenAI (opcional)
client = OpenAI(api_key="sua-chave-aqui")

# Processe usando OpenAI
df_resultado = dataframeit(
    df,
    SuaClasse,
    TEMPLATE,
    use_openai=True,                    # Ativa o provider OpenAI
    model='gpt-4o-mini',                # Modelo OpenAI
    openai_client=client,               # Cliente customizado (opcional)
    reasoning_effort='minimal',         # 'minimal', 'low', 'medium', 'high'
    verbosity='low'                     # 'low', 'medium', 'high'
)
```

## Tipos de Dados Suportados

Além de DataFrames, o DataFrameIt aceita outros tipos de dados para facilitar o uso em diferentes cenários.

### Com Lista de Textos

```python
from dataframeit import dataframeit

textos = [
    "Ótimo produto! Chegou rápido.",
    "Péssimo atendimento.",
    "Produto ok, nada de especial."
]

# Não precisa especificar text_column para listas
resultado = dataframeit(textos, SuaClasse, TEMPLATE)

# Retorna DataFrame com índice numérico + colunas extraídas
#   | sentimento | confianca
# 0 | positivo   | alta
# 1 | negativo   | alta
# 2 | neutro     | media
```

### Com Dicionário

```python
documentos = {
    'doc_001': 'Texto do primeiro documento...',
    'doc_002': 'Texto do segundo documento...',
    'doc_003': 'Texto do terceiro documento...',
}

resultado = dataframeit(documentos, SuaClasse, TEMPLATE)

# Retorna DataFrame com chaves como índice
#         | sentimento | confianca
# doc_001 | positivo   | alta
# doc_002 | negativo   | media
# doc_003 | neutro     | baixa
```

### Com pandas.Series

```python
import pandas as pd

series = pd.Series(
    ['Texto A', 'Texto B', 'Texto C'],
    index=['review_1', 'review_2', 'review_3'],
    name='avaliacoes'
)

resultado = dataframeit(series, SuaClasse, TEMPLATE)

# Retorna DataFrame preservando o índice original
#          | sentimento | confianca
# review_1 | positivo   | alta
# review_2 | negativo   | media
# review_3 | neutro     | baixa
```

### Resumo dos Tipos

| Tipo de Entrada | `text_column` | Tipo de Retorno |
|-----------------|---------------|-----------------|
| `pd.DataFrame` | Obrigatório (padrão: `'texto'`) | `pd.DataFrame` com colunas originais + extraídas |
| `pl.DataFrame` | Obrigatório (padrão: `'texto'`) | `pl.DataFrame` com colunas originais + extraídas |
| `list` | Automático | `pd.DataFrame` (índice numérico) |
| `dict` | Automático | `pd.DataFrame` (chaves como índice) |
| `pd.Series` | Automático | `pd.DataFrame` (índice preservado) |
| `pl.Series` | Automático | `pl.DataFrame` |

## Como Funciona o Template

O template define as instruções para o LLM analisar cada texto. Basta escrever suas instruções:

```python
TEMPLATE = "Classifique o sentimento do texto."
```

O texto de cada linha do DataFrame será adicionado automaticamente ao final do prompt.

### Controlando a Posição do Texto (Opcional)

Se preferir controlar onde o texto aparece no prompt, use `{texto}`:

```python
TEMPLATE = """
Você é um analista especializado.

Documento:
{texto}

Extraia as informações solicitadas do documento acima.
"""
```

## Parâmetros

### Parâmetros Gerais
- **`data`**: Dados contendo os textos. Aceita:
  - `pandas.DataFrame` ou `polars.DataFrame`
  - `pandas.Series` ou `polars.Series`
  - `list` (lista de strings)
  - `dict` (dicionário onde valores são os textos)
- **`questions`**: Modelo Pydantic definindo a estrutura dos dados a extrair
- **`prompt`**: Template do prompt (use `{texto}` para controlar onde o texto aparece)
- **`text_column`**: Nome da coluna com os textos (obrigatório para DataFrames, padrão: `'texto'`; automático para Series/list/dict)

### Parâmetros de Processamento
- **`resume=True`**: Continua processamento de onde parou (útil para grandes datasets)
- **`status_column=None`**: Nome customizado para coluna de status (padrão: `_dataframeit_status`)

### Parâmetros de Resiliência
- **`max_retries=3`**: Número máximo de tentativas em caso de erro
- **`base_delay=1.0`**: Delay inicial em segundos para retry (cresce exponencialmente)
- **`max_delay=30.0`**: Delay máximo em segundos entre tentativas
- **`rate_limit_delay=0.0`**: Delay em segundos entre requisições para evitar rate limits

### Parâmetros de Paralelismo
- **`parallel_requests=1`**: Número de requisições paralelas (1 = sequencial)
  - Ao detectar erro 429 (rate limit), reduz automaticamente pela metade
  - Use `track_tokens=True` para ver métricas de throughput (RPM, TPM)

### Parâmetros de Monitoramento
- **`track_tokens=True`**: Rastreia uso de tokens e exibe estatísticas ao final (requer LangChain 1.0+)

### Parâmetros LangChain
- **`model='gemini-2.5-flash'`**: Modelo a ser usado
- **`provider='google_genai'`**: Provider do LangChain ('google_genai', 'anthropic', 'openai', etc.)
- **`api_key=None`**: Chave API específica (opcional, usa variáveis de ambiente se None)

### Parâmetros OpenAI
- **`use_openai=False`**: Ativa o uso da OpenAI em vez de LangChain
- **`openai_client=None`**: Cliente OpenAI customizado (usa padrão se None)
- **`model='gpt-4o-mini'`**: Modelo OpenAI (quando `use_openai=True`)
- **`reasoning_effort='minimal'`**: Nível de raciocínio ('minimal', 'low', 'medium', 'high')
- **`verbosity='low'`**: Verbosidade das respostas ('low', 'medium', 'high')

## Tratamento de Erros

O DataFrameIt possui um sistema robusto de tratamento de erros:

### Colunas de Status
- **`_dataframeit_status`**: Coluna automática com status de cada linha
  - `'processed'`: Linha processada com sucesso
  - `'error'`: Linha falhou após todas as tentativas
  - `None/NaN`: Linha ainda não processada

- **`_error_details`**: Coluna automática com detalhes de erros
  - Contém mensagem de erro quando status é `'error'`
  - `None/NaN` quando processamento foi bem-sucedido

### Exemplo: Verificando Erros

```python
df_resultado = dataframeit(df, SuaClasse, TEMPLATE)

# Verificar linhas com erro
linhas_com_erro = df_resultado[df_resultado['_dataframeit_status'] == 'error']
print(f"Total de erros: {len(linhas_com_erro)}")

# Ver detalhes dos erros
for idx, row in linhas_com_erro.iterrows():
    print(f"Linha {idx}: {row['_error_details']}")

# Salvar apenas linhas processadas com sucesso
df_sucesso = df_resultado[df_resultado['_dataframeit_status'] == 'processed']
df_sucesso.to_excel('resultado_limpo.xlsx', index=False)
```

### Sistema de Retry

O DataFrameIt tenta automaticamente processar linhas com falha usando backoff exponencial:

```python
# Configurando retry mais agressivo
df_resultado = dataframeit(
    df,
    SuaClasse,
    TEMPLATE,
    max_retries=5,        # Tentar até 5 vezes
    base_delay=2.0,       # Começar com 2 segundos de espera
    max_delay=60.0        # Esperar no máximo 60 segundos entre tentativas
)
```

A espera entre tentativas cresce exponencialmente: 2s → 4s → 8s → 16s → 32s (limitado a 60s).

## Rate Limiting

O DataFrameIt oferece controle proativo de rate limiting para evitar atingir limites de requisições das APIs.

### Por que usar Rate Limiting?

- **Prevenir erros**: Evita atingir limites de requisições antes de acontecer
- **Eficiência**: Reduz desperdício de retries em datasets grandes
- **Economia**: Algumas APIs cobram por tentativa, mesmo as que falham
- **Complementar ao Retry**: O rate limiting PREVINE erros, o retry TRATA erros

### Quando usar?

Use `rate_limit_delay` quando:
- Processar datasets grandes (> 100 linhas)
- Conhecer os limites da API (ex: 60 req/min)
- Fazer processamento em lote
- Quiser economizar retries para erros reais

### Exemplos Práticos

```python
# Google Gemini: 60 requisições por minuto (free tier)
# Solução: 1 requisição por segundo = 60 req/min
df_resultado = dataframeit(
    df,
    SuaClasse,
    TEMPLATE,
    rate_limit_delay=1.0
)

# OpenAI GPT-4: limite de 500 req/min (tier 1)
# Solução: ~0.15 segundos entre requisições = ~400 req/min (margem de segurança)
df_resultado = dataframeit(
    df,
    SuaClasse,
    TEMPLATE,
    use_openai=True,
    model='gpt-4o-mini',
    rate_limit_delay=0.15
)

# Anthropic Claude: limite de 50 req/min (free tier)
# Solução: 1.2 segundos entre requisições = 50 req/min
df_resultado = dataframeit(
    df,
    SuaClasse,
    TEMPLATE,
    provider='anthropic',
    model='claude-3-5-sonnet-20241022',
    rate_limit_delay=1.2
)

# Dataset pequeno: não precisa de rate limiting
df_resultado = dataframeit(
    df,
    SuaClasse,
    TEMPLATE,
    rate_limit_delay=0.0  # Padrão: sem delay
)
```

### Como calcular o delay ideal?

```
rate_limit_delay = 60 / limite_de_requisições_por_minuto

Exemplos:
- 60 req/min  → delay = 60/60  = 1.0 segundo
- 500 req/min → delay = 60/500 = 0.12 segundos
- 50 req/min  → delay = 60/50  = 1.2 segundos
```

### Rate Limiting + Retry: Dupla Proteção

```python
df_resultado = dataframeit(
    df,
    SuaClasse,
    TEMPLATE,
    # Rate limiting proativo
    rate_limit_delay=1.0,      # Previne rate limits

    # Retry reativo
    max_retries=3,             # Trata erros inesperados
    base_delay=2.0,
    max_delay=30.0
)
```

## Processamento Paralelo

O DataFrameIt suporta processamento paralelo para acelerar o processamento de grandes datasets.

### Parâmetro `parallel_requests`

```python
# Processar com 5 requisições paralelas
df_resultado = dataframeit(
    df,
    SuaClasse,
    TEMPLATE,
    parallel_requests=5,     # Número de workers paralelos
    track_tokens=True        # Recomendado: ver métricas de throughput
)
```

### Métricas de Throughput

Quando `track_tokens=True` é usado com processamento paralelo, o DataFrameIt exibe métricas detalhadas:

```
============================================================
ESTATISTICAS DE USO
============================================================
Modelo: gemini-2.5-flash
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

Use essas métricas para calibrar o número ideal de `parallel_requests` para sua conta/tier.

### Auto-redução de Workers em Rate Limits

Quando um erro de rate limit (429) é detectado, o DataFrameIt **reduz automaticamente** o número de workers pela metade:

```python
# Começa com 10 workers
df_resultado = dataframeit(
    df,
    SuaClasse,
    TEMPLATE,
    parallel_requests=10
)

# Se detectar rate limit:
# - Workers são reduzidos: 10 → 5 → 2 → 1
# - Você verá um aviso: "Rate limit detectado! Reduzindo workers de 10 para 5."
# - Ao final, as estatísticas mostram a redução
```

**Importante**: Os workers são apenas **reduzidos**, nunca aumentados automaticamente. Isso evita custos inesperados para o usuário.

### Combinando com Rate Limit Delay

Para máxima estabilidade, combine `parallel_requests` com `rate_limit_delay`:

```python
# 5 workers paralelos, com 0.5s entre cada requisição por worker
df_resultado = dataframeit(
    df,
    SuaClasse,
    TEMPLATE,
    parallel_requests=5,
    rate_limit_delay=0.5     # Delay adicional por requisição
)
```

### Dicas de Uso

| Cenário | Configuração Recomendada |
|---------|-------------------------|
| Dataset pequeno (< 50 linhas) | `parallel_requests=1` (padrão) |
| Dataset médio (50-500 linhas) | `parallel_requests=3` a `5` |
| Dataset grande (> 500 linhas) | `parallel_requests=5` a `10` + `track_tokens=True` |
| API com limite baixo | `parallel_requests=2` + `rate_limit_delay=1.0` |
| Tier pago com limites altos | `parallel_requests=10` ou mais |

## Processamento Incremental

Para grandes datasets, use `resume=True` para continuar de onde parou:

```python
# Primeira execução (processa 100 linhas e falha)
df_resultado = dataframeit(df, SuaClasse, TEMPLATE, resume=True)
df_resultado.to_excel('resultado_parcial.xlsx', index=False)

# Segunda execução (continua das linhas não processadas)
df = pd.read_excel('resultado_parcial.xlsx')
df_resultado = dataframeit(df, SuaClasse, TEMPLATE, resume=True)
df_resultado.to_excel('resultado_completo.xlsx', index=False)
```

## Tracking de Tokens e Custos

O DataFrameIt pode rastrear automaticamente o uso de tokens para monitoramento de custos (disponível com LangChain 1.0+).

### Como Usar

```python
# Habilitar tracking de tokens
df_resultado = dataframeit(
    df,
    SuaClasse,
    TEMPLATE,
    track_tokens=True  # Habilita tracking de tokens
)

# Ao final do processamento, exibe estatísticas:
# ============================================================
# 📊 ESTATÍSTICAS DE USO DE TOKENS
# ============================================================
# Modelo: gemini-2.5-flash
# Total de tokens: 15,432
#   • Input:  12,345 tokens
#   • Output: 3,087 tokens
# ============================================================
```

### Colunas Adicionadas ao DataFrame

Quando `track_tokens=True`, o DataFrame incluirá automaticamente:

- **`_input_tokens`**: Tokens de entrada (prompt) por linha
- **`_output_tokens`**: Tokens de saída (resposta) por linha
- **`_total_tokens`**: Total de tokens por linha

### Analisando Custos

```python
# Processar com tracking
df_resultado = dataframeit(df, SuaClasse, TEMPLATE, track_tokens=True)

# Analisar uso por linha
print(f"Linha mais cara: {df_resultado['_total_tokens'].max()} tokens")
print(f"Média de tokens: {df_resultado['_total_tokens'].mean():.1f} tokens")

# Calcular custo estimado (exemplo: Gemini 2.5 Flash)
# Input: $0.075 por 1M tokens, Output: $0.30 por 1M tokens
custo_input = df_resultado['_input_tokens'].sum() * 0.075 / 1_000_000
custo_output = df_resultado['_output_tokens'].sum() * 0.30 / 1_000_000
custo_total = custo_input + custo_output
print(f"Custo estimado: ${custo_total:.4f}")
```

### Compatibilidade

- ✅ **LangChain 1.0+** com `usage_metadata` (Gemini, Claude, GPT via LangChain)
- ✅ **OpenAI** com `response.usage` (GPT-4, GPT-4o, etc.)
- ⚠️ Versões anteriores do LangChain não incluem `usage_metadata`

## Exemplo Completo

Veja o diretório `example/` para um caso de uso completo com análise de decisões judiciais, incluindo:
- Modelo Pydantic complexo com classes aninhadas
- Template detalhado com instruções específicas de domínio
- Uso de campos opcionais e tipos Literal
- Processamento de listas e tuplas

## Configuração de Variáveis de Ambiente

### Para OpenAI
```bash
export OPENAI_API_KEY="sua-chave-openai"
```

### Para Google Gemini (LangChain)
```bash
export GOOGLE_API_KEY="sua-chave-google"
```

### Para Anthropic Claude (LangChain)
```bash
export ANTHROPIC_API_KEY="sua-chave-anthropic"
```

## Contribuições

Contribuições são bem-vindas! Este é um projeto em desenvolvimento inicial.

## Licença

Veja o arquivo LICENSE para detalhes.
