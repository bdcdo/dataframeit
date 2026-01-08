# DataFrameIt

[![PyPI version](https://badge.fury.io/py/dataframeit.svg)](https://badge.fury.io/py/dataframeit)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Enriqueça DataFrames com LLMs de forma simples e estruturada.**

DataFrameIt processa textos em DataFrames usando Modelos de Linguagem (LLMs) e extrai informações estruturadas definidas por modelos Pydantic.

**[Documentação Completa](https://bdcdo.github.io/dataframeit)** | **[Referência para LLMs](https://bdcdo.github.io/dataframeit/reference/llm-reference/)**

## Instalação

```bash
pip install dataframeit[google]  # Google Gemini (recomendado)
pip install dataframeit[openai]  # OpenAI
pip install dataframeit[anthropic]  # Anthropic Claude
```

Configure sua API key:

```bash
export GOOGLE_API_KEY="sua-chave"  # ou OPENAI_API_KEY, ANTHROPIC_API_KEY
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

- **Múltiplos providers**: Google Gemini, OpenAI, Anthropic, Cohere, Mistral via LangChain
- **Múltiplos tipos de entrada**: DataFrame, Series, list, dict
- **Saída estruturada**: Validação automática com Pydantic
- **Resiliência**: Retry automático com backoff exponencial
- **Performance**: Processamento paralelo, rate limiting configurável
- **Busca web**: Integração com Tavily para enriquecer dados
- **Tracking**: Monitoramento de tokens e métricas de throughput

## Documentação

- [Início Rápido](https://bdcdo.github.io/dataframeit/getting-started/quickstart/)
- [Guias](https://bdcdo.github.io/dataframeit/guides/basic-usage/)
- [Referência da API](https://bdcdo.github.io/dataframeit/reference/api/)
- [Referência para LLMs](https://bdcdo.github.io/dataframeit/reference/llm-reference/) - Página compacta otimizada para assistentes de código

## Exemplos

Veja a pasta [`example/`](example/) para notebooks Jupyter com casos de uso completos.

## Licença

MIT
