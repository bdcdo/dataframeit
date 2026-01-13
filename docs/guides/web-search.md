# Busca Web

Enriqueça seus dados com busca web usando Tavily.

## Visão Geral

O DataFrameIt pode buscar informações na web para complementar a análise de cada texto. Isso é útil quando você precisa de contexto adicional que não está no texto original.

## Configuração

### 1. Instale a Dependência

```bash
pip install dataframeit[search]
# ou
pip install langchain-tavily
```

### 2. Configure a API Key

```bash
export TAVILY_API_KEY="sua-chave-tavily"
```

Obtenha sua chave em: [Tavily](https://tavily.com/)

## Uso Básico

```python
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
from dataframeit import dataframeit

class EmpresaInfo(BaseModel):
    setor: Literal['tecnologia', 'saude', 'financas', 'varejo', 'outro']
    descricao: str = Field(description="Breve descrição da empresa")
    fundacao: str = Field(description="Ano de fundação, se encontrado")

# Dados com nomes de empresas
df = pd.DataFrame({
    'texto': ['Microsoft', 'Nubank', 'iFood']
})

PROMPT = """
Com base nas informações disponíveis e na busca web,
extraia informações sobre a empresa mencionada.
"""

# Habilita busca web com use_search=True
resultado = dataframeit(
    df,
    EmpresaInfo,
    PROMPT,
    use_search=True,      # Habilita busca web
    max_results=5         # Número de resultados por busca
)
```

## Parâmetros de Busca

| Parâmetro | Tipo | Padrão | Descrição |
|-----------|------|--------|-----------|
| `use_search` | bool | `False` | Habilita busca web via Tavily |
| `search_per_field` | bool | `False` | Executa busca separada para cada campo do modelo |
| `max_results` | int | `5` | Resultados por busca (1-20) |
| `search_depth` | str | `'basic'` | `'basic'` (1 crédito) ou `'advanced'` (2 créditos) |
| `save_trace` | bool/str | `None` | Salva trace do agente: `True`/`"full"` ou `"minimal"` |

## Exemplos

### Busca Básica

```python
resultado = dataframeit(
    df, Model, PROMPT,
    use_search=True
)
```

### Busca por Campo

Quando o modelo tem muitos campos, pode ser útil fazer buscas separadas:

```python
resultado = dataframeit(
    df, Model, PROMPT,
    use_search=True,
    search_per_field=True  # Uma busca por campo do modelo
)
```

### Busca Profunda

```python
# Busca mais detalhada (mais lenta, mais cara)
resultado = dataframeit(
    df, Model, PROMPT,
    use_search=True,
    search_depth='advanced',
    max_results=10
)
```

## Configuração Per-Field (v0.5.2+)

A partir da versão 0.5.2, você pode configurar prompts e parâmetros de busca específicos para cada campo usando `json_schema_extra` do Pydantic.

### Opções Disponíveis

| Opção | Descrição |
|-------|-----------|
| `prompt` ou `prompt_replace` | Substitui completamente o prompt base para este campo |
| `prompt_append` | Adiciona texto ao prompt base |
| `search_depth` | Override de profundidade: `"basic"` ou `"advanced"` |
| `max_results` | Override de número de resultados (1-20) |

!!! note "Requer search_per_field=True"
    A configuração per-field só funciona quando `search_per_field=True`. Se você usar `json_schema_extra` com configurações de prompt ou busca sem habilitar `search_per_field`, um erro será levantado.

### Exemplo: Prompt Customizado

```python
from pydantic import BaseModel, Field

class MedicamentoInfo(BaseModel):
    # Campo com comportamento padrão
    principio_ativo: str = Field(description="Princípio ativo do medicamento")

    # Campo com prompt completamente substituído
    doenca_rara: str = Field(
        description="Classificação de doença rara",
        json_schema_extra={
            "prompt": "Busque em Orphanet (orpha.net) e FDA Orphan Drug Database. Analise: {texto}"
        }
    )

    # Campo com prompt adicional (append)
    avaliacao_conitec: str = Field(
        description="Avaliação da CONITEC",
        json_schema_extra={
            "prompt_append": "Busque APENAS no site da CONITEC (gov.br/conitec)."
        }
    )

resultado = dataframeit(
    df,
    MedicamentoInfo,
    "Analise o medicamento: {texto}",
    use_search=True,
    search_per_field=True,  # Obrigatório para usar json_schema_extra
)
```

### Exemplo: Parâmetros de Busca Por Campo

```python
class PesquisaDetalhada(BaseModel):
    resumo_rapido: str = Field(
        description="Resumo em 2 linhas",
        json_schema_extra={
            "search_depth": "basic",
            "max_results": 3
        }
    )

    analise_profunda: str = Field(
        description="Análise detalhada com fontes",
        json_schema_extra={
            "prompt_append": "Inclua citações das fontes encontradas.",
            "search_depth": "advanced",
            "max_results": 10
        }
    )
```

### Combinando Prompt e Parâmetros

Você pode combinar configurações de prompt e parâmetros de busca:

```python
estudos_clinicos: str = Field(
    description="Estudos clínicos relevantes",
    json_schema_extra={
        "prompt_append": "Busque por ensaios clínicos recentes (2020-2024).",
        "search_depth": "advanced",
        "max_results": 15
    }
)
```

## Debug: Salvar Trace do Agente (v0.5.3+)

Para debugar e auditar o raciocínio do agente, use o parâmetro `save_trace`.

### Parâmetros

| Valor | Descrição |
|-------|-----------|
| `False` / `None` | Desabilitado (padrão) |
| `True` / `"full"` | Trace completo com conteúdo das mensagens |
| `"minimal"` | Apenas queries e contagens, sem conteúdo de resultados de busca |

### Colunas Geradas

- **Agente único**: `_trace`
- **Per-field**: `_trace_{nome_do_campo}` para cada campo

### Estrutura do Trace

```python
{
    "messages": [
        {"type": "human", "content": "Analise o medicamento..."},
        {"type": "ai", "content": "", "tool_calls": [...]},
        {"type": "tool", "content": "[resultados da busca]", "tool_call_id": "..."}
    ],
    "search_queries": ["query1", "query2"],
    "total_tool_calls": 2,
    "duration_seconds": 3.45,
    "model": "gpt-4o-mini"
}
```

### Exemplo: Trace Completo

```python
import json

resultado = dataframeit(
    df,
    MedicamentoInfo,
    PROMPT,
    use_search=True,
    save_trace=True  # ou "full"
)

# Acessar trace da primeira linha
trace = json.loads(resultado['_trace'].iloc[0])
print(f"Queries realizadas: {trace['search_queries']}")
print(f"Duração: {trace['duration_seconds']}s")
print(f"Modelo: {trace['model']}")
```

### Exemplo: Trace Minimal

Para auditorias onde só importa saber o que foi buscado:

```python
resultado = dataframeit(
    df, Model, PROMPT,
    use_search=True,
    save_trace="minimal"  # Não inclui conteúdo das buscas
)
```

### Exemplo: Trace Per-Field

```python
resultado = dataframeit(
    df,
    MedicamentoInfo,
    PROMPT,
    use_search=True,
    search_per_field=True,
    save_trace="full"
)

# Cada campo tem seu próprio trace
trace_principio = json.loads(resultado['_trace_principio_ativo'].iloc[0])
trace_indicacao = json.loads(resultado['_trace_indicacao'].iloc[0])
```

## Caso de Uso: Verificação de Fatos

```python
from pydantic import BaseModel, Field
from typing import Literal, List

class VerificacaoFato(BaseModel):
    afirmacao: str = Field(description="A afirmação original")
    veredicto: Literal['verdadeiro', 'falso', 'parcialmente_verdadeiro', 'inconclusivo']
    fontes: List[str] = Field(description="Fontes que suportam o veredicto")
    explicacao: str = Field(description="Explicação do veredicto")

PROMPT = """
Verifique a veracidade da afirmação usando as informações da busca web.
Cite as fontes encontradas.
"""

resultado = dataframeit(
    df_afirmacoes,
    VerificacaoFato,
    PROMPT,
    use_search=True,
    max_results=5,
    search_depth='advanced'
)
```

## Caso de Uso: Enriquecimento de Leads

```python
class LeadEnriquecido(BaseModel):
    empresa: str
    site: str = Field(description="Website oficial")
    linkedin: str = Field(description="URL do LinkedIn")
    tamanho: Literal['startup', 'pme', 'grande_empresa']
    tecnologias: List[str] = Field(description="Tecnologias utilizadas")

resultado = dataframeit(
    df_leads,
    LeadEnriquecido,
    "Pesquise informações sobre a empresa.",
    use_search=True,
    max_results=3
)
```

## Custos e Limites

!!! warning "Atenção aos custos"
    Cada linha do DataFrame faz uma busca web. Para datasets grandes, isso pode gerar custos significativos na API do Tavily.

- **Free tier**: 1000 buscas/mês
- **Busca básica**: ~$0.01 por busca
- **Busca avançada**: ~$0.02 por busca

### Dicas para Economizar

1. Use `max_results=3` a `5` (suficiente para maioria dos casos)
2. Prefira `search_depth='basic'`
3. Filtre seu DataFrame antes de processar
4. Use `search_per_field=False` quando possível

## Rate Limits e Processamento Paralelo

!!! danger "Erros HTTP 429"
    Ao usar `parallel_requests` com busca web, você pode exceder os limites de taxa do Tavily (~100 requisições/minuto). Isso faz com que buscas falhem silenciosamente e retornem dados incompletos.

### Como as Queries são Calculadas

| Configuração | Queries por Linha | Exemplo (100 linhas, 4 campos) |
|--------------|-------------------|--------------------------------|
| `search_per_field=False` | 1 | 100 queries |
| `search_per_field=True` | 1 por campo | 400 queries |

Com `parallel_requests=20` e `search_per_field=True`, você pode enviar até **80 queries concorrentes** (20 workers × 4 campos), o que excede os limites do Tavily.

### Configurações Recomendadas

| Cenário | `parallel_requests` | `rate_limit_delay` |
|---------|--------------------|--------------------|
| `search_per_field=False` | 5-10 | 0.5s |
| `search_per_field=True` (2-3 campos) | 3-5 | 0.5s |
| `search_per_field=True` (4+ campos) | 2-3 | 1.0s |

### Exemplo de Configuração Segura

```python
# Configurações seguras para busca com múltiplos campos
resultado = dataframeit(
    df,
    Model,
    PROMPT,
    text_column='texto',
    use_search=True,
    search_per_field=True,
    parallel_requests=3,      # Baixo paralelismo
    rate_limit_delay=0.5      # Delay entre requisições
)
```

### Aviso Automático

O DataFrameIt avisa automaticamente quando sua configuração pode exceder rate limits:

```
============================================================
AVISO: Configuração pode exceder rate limits de busca
============================================================
Configuração atual:
  - Linhas a processar: 100
  - Campos no modelo: 4
  - parallel_requests: 20
  - search_per_field: True
  - rate_limit_delay: 0.0s
  - Total de queries estimadas: 400

Problemas detectados:
- Queries concorrentes estimadas: 80 (limite recomendado: 10)
- Taxa estimada: ~4800 queries/min (limite Tavily: ~100/min)

Recomendações para evitar HTTP 429 (rate limit):
  dataframeit(..., parallel_requests=2, rate_limit_delay=1.7)
============================================================
```

### Auto-Recuperação

Quando um erro 429 é detectado, o DataFrameIt automaticamente reduz workers:

```
Rate limit detectado! Reduzindo workers de 10 para 5.
Rate limit detectado! Reduzindo workers de 5 para 2.
```

No entanto, é melhor configurar corretamente desde o início para evitar queries falhadas.
