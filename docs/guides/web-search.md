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
