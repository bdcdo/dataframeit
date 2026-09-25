# DataFrameIt

[![PyPI version](https://badge.fury.io/py/dataframeit.svg)](https://badge.fury.io/py/dataframeit)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/bdcdo/dataframeit/actions/workflows/tests.yml/badge.svg)](https://github.com/bdcdo/dataframeit/actions/workflows/tests.yml)

[English](README.md) · **Português** · [Español](README.es.md)

**Enriqueça DataFrames com LLMs de forma simples e estruturada.**

DataFrameIt processa textos em DataFrames usando Modelos de Linguagem (LLMs) e extrai informações estruturadas definidas por modelos Pydantic.

**[Documentação Completa](https://brunodcdo.com.br/dataframeit)** | **[Referência para LLMs](https://brunodcdo.com.br/dataframeit/reference/llm-reference/)**

## Instalação

```bash
pip install dataframeit[openai]       # OpenAI (provider padrão)
pip install dataframeit[google]       # Google Gemini
pip install dataframeit[anthropic]    # Anthropic Claude
pip install dataframeit[groq]         # Groq
pip install dataframeit[codex]        # Codex SDK oficial (experimental)
pip install dataframeit[claude-code]  # Claude Code pelo Claude Agent SDK
pip install dataframeit[all]          # todos os providers exceto Codex, busca web, Polars e Excel
```

Extras opcionais: `search` (Tavily), `search-exa` (Exa), `search-all`, `polars`, `excel`.

Configure a autenticação do provider:

```bash
export OPENAI_API_KEY="sua-chave"  # ou GOOGLE_API_KEY, ANTHROPIC_API_KEY, GROQ_API_KEY
```

O provider experimental `codex` é opcional, não faz parte do extra `all`, usa o runtime empacotado e requer autenticação local em arquivo. Consulte a [documentação de instalação](https://brunodcdo.com.br/dataframeit/getting-started/installation/) para configurar o extra e as credenciais.

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

Os nomes de classes e campos são arbitrários: os notebooks de [`example/`](example/) usam nomes em português.

## Funcionalidades

- **Múltiplos providers**: OpenAI, Google Gemini, Anthropic e Groq com extras próprios, qualquer outro provider do LangChain (Cohere, Mistral, Vertex AI, Bedrock, Azure) instalando o pacote dele, além de Claude Code e Codex pelos SDKs oficiais
- **Múltiplos tipos de entrada**: DataFrame e Series do pandas ou do Polars, list, dict
- **Saída estruturada**: Validação com Pydantic; uma resposta recusada volta ao modelo com o erro
- **Resiliência**: Retry automático com backoff exponencial e checkpoints periódicos (`batch_size` + `checkpoint_path`) para retomar execuções longas
- **Performance**: Processamento paralelo que reduz os workers pela metade em rate limit, rate limiting configurável
- **Busca web**: Tavily ou Exa, por campo ou por grupo de campos, com campos condicionais (`condition`, `depends_on`)
- **Tracking**: Uso de tokens, créditos de busca e métricas de throughput

## Configuração por Campo

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

A lista completa de opções por campo (inclusive `max_search_calls`, `condition` e `depends_on`) está no [guia de busca web](https://brunodcdo.com.br/dataframeit/guides/web-search/).

## Documentação

- [Início Rápido](https://brunodcdo.com.br/dataframeit/getting-started/quickstart/)
- [Guias](https://brunodcdo.com.br/dataframeit/guides/basic-usage/)
- [Referência da API](https://brunodcdo.com.br/dataframeit/reference/api/)
- [Referência para LLMs](https://brunodcdo.com.br/dataframeit/reference/llm-reference/) - Página compacta otimizada para assistentes de código
- [Perguntas Frequentes](https://brunodcdo.com.br/dataframeit/guides/faq/)
- [Histórico de versões](CHANGELOG.md)

## Exemplos

Veja a pasta [`example/`](example/) para notebooks Jupyter com casos de uso completos.

## Contribuindo

Veja [CONTRIBUTING.md](CONTRIBUTING.md).

## Licença

MIT
