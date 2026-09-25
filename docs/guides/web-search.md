# Busca Web

Enriqueça seus dados com busca web usando Tavily ou Exa.

## Visão Geral

O DataFrameIt pode buscar informações na web para complementar a análise de cada texto. Isso é útil quando você precisa de contexto adicional que não está no texto original.

Com `use_search=True`, cada linha é processada por um agente que decide quando buscar, com até `max_search_calls` buscas (padrão 10), e depois responde no formato do modelo Pydantic. A busca funciona com os providers via LangChain; `claude_code` e `codex` não a suportam.

## Configuração

### 1. Instale a Dependência

```bash
pip install dataframeit[search]       # Tavily (padrão)
pip install dataframeit[search-exa]   # Exa
pip install dataframeit[search-all]   # os dois
```

### 2. Configure a API Key

```bash
export TAVILY_API_KEY="sua-chave-tavily"   # https://tavily.com/
export EXA_API_KEY="sua-chave-exa"         # https://exa.ai/
```

### 3. Escolha o Provedor de Busca

| Provedor | `search_provider` | Profundidade | Cobrança |
|----------|-------------------|--------------|----------|
| Tavily | `"tavily"` (padrão) | `search_depth='basic'` ou `'advanced'` | 1 crédito por busca básica, 2 por avançada |
| Exa | `"exa"` | ignora `search_depth` | 1 crédito (US$ 0,005) por busca com até 25 resultados, 5 acima disso |

```python
resultado = dataframeit(df, Model, PROMPT, use_search=True, search_provider="exa")
```

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
| `use_search` | bool | `False` | Habilita busca web |
| `search_provider` | str | `'tavily'` | `'tavily'` ou `'exa'` |
| `search_per_field` | bool | `False` | Executa um agente separado para cada campo do modelo |
| `max_results` | int | `5` | Resultados por busca (1-20) |
| `search_depth` | str | `'basic'` | `'basic'` (1 crédito) ou `'advanced'` (2 créditos); só Tavily |
| `max_search_calls` | int | `10` | Máximo de buscas por execução do agente; as seguintes são bloqueadas e o agente responde com o que encontrou |
| `search_groups` | dict | `None` | Campos que compartilham um agente de busca; ver [Grupos de Busca](#grupos-de-busca) |
| `save_trace` | bool/str | `None` | Salva trace do agente: `True`/`"full"` ou `"minimal"` |

Com busca, a saída ganha a coluna `_search_credits`, com os créditos gastos em cada linha. Os campos podem ainda ser condicionais (`condition` e `depends_on`); ver [Campos Condicionais](../examples/conditional-fields.md).

## Exemplos

### Busca Básica

```python
resultado = dataframeit(
    df, Model, PROMPT,
    use_search=True
)
```

### Busca por Campo

Quando o modelo tem muitos campos, pode ser útil dar a cada campo um agente próprio:

```python
resultado = dataframeit(
    df, Model, PROMPT,
    use_search=True,
    search_per_field=True  # Um agente de busca por campo do modelo
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

## Configuração por Campo

Você pode configurar prompts e parâmetros de busca específicos para cada campo usando `json_schema_extra` do Pydantic.

### Opções Disponíveis

| Opção | Descrição |
|-------|-----------|
| `prompt` ou `prompt_replace` | Substitui o prompt base para este campo; sem `{texto}`, o texto da linha é anexado ao final |
| `prompt_append` | Adiciona texto ao prompt base |
| `search_depth` | Override de profundidade: `"basic"` ou `"advanced"` |
| `max_results` | Override de número de resultados (1-20) |
| `max_search_calls` | Override do máximo de buscas do agente deste campo |

!!! note "Requer search_per_field=True"
    A configuração por campo só funciona quando `search_per_field=True`. Se você usar `json_schema_extra` com configurações de prompt ou busca sem habilitar `search_per_field`, um erro será levantado.

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
        "prompt_append": "Busque ensaios clínicos publicados nos últimos cinco anos.",
        "search_depth": "advanced",
        "max_results": 15
    }
)
```

## Debug: Salvar Trace do Agente

Para debugar e auditar o raciocínio do agente, use o parâmetro `save_trace`.

### Parâmetros

| Valor | Descrição |
|-------|-----------|
| `False` / `None` | Desabilitado (padrão) |
| `True` / `"full"` | Trace completo com conteúdo das mensagens |
| `"minimal"` | Apenas queries e contagens, sem conteúdo de resultados de busca |

### Colunas Geradas

- **Agente único**: `_trace`
- **Por campo**: `_trace_{nome_do_campo}` para cada campo
- **Com grupos**: `_trace_{nome_do_grupo}` para cada grupo, além dos campos isolados

### Estrutura do Trace

```python
{
    "messages": [
        {"type": "human", "content": "Analise o medicamento..."},
        {"type": "ai", "content": "", "tool_calls": [...]},
        {"type": "tool", "content": "[resultados da busca]", "tool_call_id": "..."}
    ],
    "search_queries": ["query1", "query2"],
    "total_tool_calls": 3,  # as duas buscas e a resposta estruturada
    "duration_seconds": 3.45,
    "model": "gpt-6-luna"
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

### Exemplo: Trace por Campo

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
trace_doenca_rara = json.loads(resultado['_trace_doenca_rara'].iloc[0])
```

## Grupos de Busca

Quando vários campos precisam do mesmo contexto de busca, você pode agrupá-los para reduzir chamadas de API redundantes.

### Motivação

Sem grupos, 6 campos com `search_per_field=True` significam 6 agentes por linha, cada um com suas buscas. Com grupos, campos relacionados compartilham um agente.

**Exemplo:**
- Campos `status_anvisa`, `avaliacao_conitec`, `existe_pcdt` são todos sobre regulação
- Sem grupos: 3 agentes, que tendem a repetir as mesmas buscas
- Com grupos: 1 agente compartilhado

### Parâmetros de Grupo

| Parâmetro | Tipo | Obrigatório | Descrição |
|-----------|------|-------------|-----------|
| `fields` | list | Sim | Lista de campos que pertencem ao grupo |
| `prompt` | str | Não | Prompt customizado para o grupo. Use `{texto}` (ou o sinônimo `{query}`) para o texto |
| `max_results` | int | Não | Override de número de resultados (1-20) |
| `search_depth` | str | Não | Override: `"basic"` ou `"advanced"` |
| `max_search_calls` | int | Não | Override do máximo de buscas do agente do grupo |

### Exemplo Básico

```python
from pydantic import BaseModel, Field

class MedicamentoRegulatorio(BaseModel):
    # Campos do grupo "regulatory" (1 agente compartilhado)
    status_anvisa: str = Field(description="Status de aprovação na ANVISA")
    avaliacao_conitec: str = Field(description="Avaliação da CONITEC")
    existe_pcdt: str = Field(description="Se existe PCDT publicado")

    # Campos isolados (1 agente cada)
    nome: str = Field(description="Nome comercial")
    fabricante: str = Field(description="Laboratório fabricante")

resultado = dataframeit(
    df,
    MedicamentoRegulatorio,
    "Pesquise sobre o medicamento: {texto}",
    use_search=True,
    search_per_field=True,
    search_groups={
        "regulatory": {
            "fields": ["status_anvisa", "avaliacao_conitec", "existe_pcdt"],
            "prompt": "Busque status regulatório no Brasil (ANVISA, CONITEC, PCDT) para: {query}",
            "search_depth": "advanced",
        }
    }
)
```

**Resultado:** 3 agentes por linha (1 do grupo + 2 isolados) em vez de 5.

### Múltiplos Grupos

```python
search_groups={
    "regulatory": {
        "fields": ["status_anvisa", "avaliacao_conitec"],
        "prompt": "Busque status regulatório: {query}",
    },
    "clinical": {
        "fields": ["eficacia", "seguranca"],
        "prompt": "Busque estudos clínicos sobre: {query}",
        "search_depth": "advanced",
    }
}
```

### Traces com Grupos

Com `save_trace=True`, os traces são organizados por grupo:

```python
resultado = dataframeit(
    df, Model, PROMPT,
    use_search=True,
    search_per_field=True,
    search_groups={"regulatory": {"fields": ["status_anvisa", "avaliacao_conitec"]}},
    save_trace=True
)

# Trace do grupo
trace_regulatory = json.loads(resultado['_trace_regulatory'].iloc[0])

# Traces dos campos isolados
trace_nome = json.loads(resultado['_trace_nome'].iloc[0])
```

### Regras de Validação

1. **Requer `use_search=True` e `search_per_field=True`**
2. **Campos devem existir no modelo Pydantic**
3. **Campos não podem estar em múltiplos grupos**
4. **Campos em grupos não podem ter `json_schema_extra` de busca**: escolha entre configuração por campo ou grupo, não ambos

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
    Cada agente pode fazer até `max_search_calls` buscas (padrão 10), e há um agente por linha, ou um por campo e por linha com `search_per_field=True`. Para datasets grandes, isso pode gerar custos significativos. Os créditos gastos ficam em `_search_credits` e no resumo ao fim da execução.

Os preços mudam; consulte a página de cada provedor ([Tavily](https://tavily.com/pricing), [Exa](https://exa.ai/pricing)). O Tavily tem plano gratuito de 1000 buscas por mês.

### Dicas para Economizar

1. Use `max_results=3` a `5` (suficiente para maioria dos casos)
2. Prefira `search_depth='basic'`
3. Filtre seu DataFrame antes de processar
4. Use `search_per_field=False` quando possível, ou agrupe campos com `search_groups`
5. Reduza `max_search_calls` quando uma ou duas buscas bastam

## Rate Limits e Processamento Paralelo

!!! danger "Erros HTTP 429"
    Ao usar `parallel_requests` com busca web, é fácil exceder os limites de taxa do provedor de busca. Uma busca que falha interrompe o agente e a linha volta ao ciclo de novas tentativas; esgotadas as tentativas, a linha fica com status `'error'`.

### Limites por Provedor

| Provedor | Rate limit aproximado |
|----------|----------------------|
| Tavily   | ~100 req/min         |
| Exa      | ~300 req/min         |

Se você precisa de maior throughput, considere `search_provider="exa"`.

### Como as Queries são Contadas

| Configuração | Agentes por linha | Buscas por linha, no máximo |
|--------------|-------------------|-----------------------------|
| `search_per_field=False` | 1 | `max_search_calls` |
| `search_per_field=True` | 1 por campo ou grupo | `max_search_calls` por agente |

Com `parallel_requests=20` e `search_per_field=True` em um modelo de 4 campos, rodam 80 agentes ao mesmo tempo, muito acima dos limites dos provedores.

### Configurações Recomendadas

**Tavily (padrão):**

| Cenário | `parallel_requests` | `rate_limit_delay` |
|---------|---------------------|--------------------|
| `search_per_field=False` | 5–10 | 0.5s |
| `search_per_field=True` (2–3 campos) | 3–5 | 0.5s |
| `search_per_field=True` (4+ campos) | 2–3 | 1.0s |

**Exa:**

| Cenário | `parallel_requests` | `rate_limit_delay` |
|---------|---------------------|--------------------|
| `search_per_field=False` | 10–15 | 0.3s |
| `search_per_field=True` (2–3 campos) | 5–8 | 0.3s |
| `search_per_field=True` (4+ campos) | 3–5 | 0.5s |

```python
# Configuração segura com Tavily e múltiplos campos
resultado = dataframeit(
    df, Model, PROMPT,
    use_search=True, search_per_field=True,
    parallel_requests=3, rate_limit_delay=0.5,
)

# Maior throughput com Exa
resultado = dataframeit(
    df, Model, PROMPT,
    use_search=True, search_provider="exa",
    search_per_field=True,
    parallel_requests=5, rate_limit_delay=0.3,
)
```

### Aviso Automático

O DataFrameIt emite um `UserWarning` quando a configuração parece arriscada (queries concorrentes altas ou taxa estimada perto do limite), incluindo recomendações de `parallel_requests` e `rate_limit_delay` para evitar HTTP 429. O gatilho também dispara em execuções sequenciais quando `search_per_field=True` produz muitas queries (>100 no total).
