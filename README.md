# DataFrameIt

[![PyPI version](https://badge.fury.io/py/dataframeit.svg)](https://badge.fury.io/py/dataframeit)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Enriqueça DataFrames com LLMs de forma simples e estruturada.**

DataFrameIt processa textos em DataFrames usando Modelos de Linguagem (LLMs) e extrai informações estruturadas definidas por modelos Pydantic.

**[Documentação Completa](https://bdcdo.github.io/dataframeit)** | **[Referência para LLMs](https://bdcdo.github.io/dataframeit/reference/llm-reference/)**

## Instalação

```bash
pip install dataframeit  # Groq incluído (default - free tier permanente!)
pip install dataframeit[google]  # Google Gemini
pip install dataframeit[openai]  # OpenAI
pip install dataframeit[anthropic]  # Anthropic Claude
```

Configure sua API key:

```bash
export GROQ_API_KEY="sua-chave"  # Gratuito em console.groq.com
# Ou: GOOGLE_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY
```

## Exemplo Rápido

```python
from pydantic import BaseModel
from typing import Literal
import pandas as pd
from dataframeit import dataframeit

# 1. Defina o que extrair
class Sentimento(BaseModel):
    sentimento: Literal['positivo', 'negativo', 'neutro']
    confianca: Literal['alta', 'media', 'baixa']

# 2. Seus dados
df = pd.DataFrame({
    'texto': [
        'Produto excelente! Superou expectativas.',
        'Péssimo atendimento, nunca mais compro.',
        'Entrega ok, produto mediano.'
    ]
})

# 3. Processe!
resultado = dataframeit(df, Sentimento, "Analise o sentimento do texto.")
print(resultado)
```

**Saída:**

| texto | sentimento | confianca |
|-------|------------|-----------|
| Produto excelente! ... | positivo | alta |
| Péssimo atendimento... | negativo | alta |
| Entrega ok... | neutro | media |

## Funcionalidades

- **Múltiplos providers**: Groq (default, free tier permanente), Google Gemini, OpenAI, Anthropic, Cohere, Mistral via LangChain
- **Múltiplos tipos de entrada**: DataFrame, Series, list, dict
- **Saída estruturada**: Validação automática com Pydantic
- **Resiliência**: Retry automático com backoff exponencial
- **Performance**: Processamento paralelo, rate limiting configurável
- **Busca web**: Integração com Tavily para enriquecer dados
- **Tracking**: Monitoramento de tokens e métricas de throughput
- **Configuração per-field**: Prompts e parâmetros de busca personalizados por campo (v0.5.2+)

## Configuração Per-Field (Novo em v0.5.2)

Configure prompts e parâmetros de busca específicos para cada campo usando `json_schema_extra`:

```python
from pydantic import BaseModel, Field

class MedicamentoInfo(BaseModel):
    # Campo com prompt padrão
    principio_ativo: str = Field(description="Princípio ativo do medicamento")

    # Campo com prompt customizado (substitui o prompt base)
    doenca_rara: str = Field(
        description="Classificação de doença rara",
        json_schema_extra={
            "prompt": "Busque em Orphanet (orpha.net). Analise: {texto}"
        }
    )

    # Campo com prompt adicional (append ao prompt base)
    avaliacao_conitec: str = Field(
        description="Avaliação da CONITEC",
        json_schema_extra={
            "prompt_append": "Busque APENAS no site da CONITEC (gov.br/conitec)."
        }
    )

    # Campo com parâmetros de busca customizados
    estudos_clinicos: str = Field(
        description="Estudos clínicos relevantes",
        json_schema_extra={
            "prompt_append": "Busque estudos clínicos recentes.",
            "search_depth": "advanced",
            "max_results": 10
        }
    )

# Requer search_per_field=True
resultado = dataframeit(
    df,
    MedicamentoInfo,
    "Analise o medicamento: {texto}",
    use_search=True,
    search_per_field=True,
)
```

**Opções disponíveis em `json_schema_extra`:**

| Opção | Descrição |
|-------|-----------|
| `prompt` ou `prompt_replace` | Substitui completamente o prompt base |
| `prompt_append` | Adiciona texto ao prompt base |
| `search_depth` | `"basic"` ou `"advanced"` (override per-field) |
| `max_results` | Número de resultados de busca (1-20) |

## Documentação

- [Início Rápido](https://bdcdo.github.io/dataframeit/getting-started/quickstart/)
- [Guias](https://bdcdo.github.io/dataframeit/guides/basic-usage/)
- [Referência da API](https://bdcdo.github.io/dataframeit/reference/api/)
- [Referência para LLMs](https://bdcdo.github.io/dataframeit/reference/llm-reference/) - Página compacta otimizada para assistentes de código

## Exemplos

Veja a pasta [`example/`](example/) para notebooks Jupyter com casos de uso completos.

## Licença

MIT
