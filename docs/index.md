<div class="hero" markdown>

# DataFrameIt

<p class="tagline">Enriqueça DataFrames com LLMs de forma simples e estruturada</p>

<div class="badges">
  <a href="https://pypi.org/project/dataframeit/"><img src="https://badge.fury.io/py/dataframeit.svg" alt="PyPI version"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</div>

<div class="cta-buttons" markdown>

[:material-rocket-launch: Começar Agora](getting-started/quickstart.md){ .cta-button .primary }

[:material-robot: Referência para LLMs](reference/llm-reference.md){ .cta-button .secondary }

</div>

</div>

## O que é?

DataFrameIt processa textos em DataFrames usando **Modelos de Linguagem (LLMs)** e extrai informações estruturadas definidas por **modelos Pydantic**. Uma função, um modelo e um prompt bastam.

```python
from pydantic import BaseModel
from typing import Literal
import pandas as pd
from dataframeit import dataframeit

class Sentimento(BaseModel):
    sentimento: Literal['positivo', 'negativo', 'neutro']
    confianca: Literal['alta', 'media', 'baixa']

df = pd.DataFrame({'texto': ['Produto excelente!', 'Péssimo serviço.']})
resultado = dataframeit(df, Sentimento, "Analise o sentimento do texto.")
```

## Funcionalidades

<div class="feature-grid" markdown>

<div class="feature-card" markdown>
<div class="icon" markdown>:material-cloud-sync:</div>

### Múltiplos Provedores

OpenAI, Google Gemini, Anthropic, Groq e outros via LangChain, além de Codex e Claude Code pelos SDKs oficiais.
</div>

<div class="feature-card" markdown>
<div class="icon" markdown>:material-check-decagram:</div>

### Saída Estruturada

Validação automática com Pydantic. Defina campos, tipos e descrições; uma resposta fora do modelo volta ao LLM com o erro para correção.
</div>

<div class="feature-card" markdown>
<div class="icon" markdown>:material-shield-refresh:</div>

### Resiliência

Retry automático com backoff exponencial, rate limiting configurável e checkpoints periódicos para retomar execuções longas.
</div>

<div class="feature-card" markdown>
<div class="icon" markdown>:material-rocket-launch:</div>

### Performance

Processamento paralelo com auto-ajuste. Métricas de throughput em tempo real.
</div>

<div class="feature-card" markdown>
<div class="icon" markdown>:material-web:</div>

### Busca Web

Tavily ou Exa para enriquecer dados com informações da internet, com busca por campo e campos condicionais.
</div>

<div class="feature-card" markdown>
<div class="icon" markdown>:material-format-list-bulleted-type:</div>

### Múltiplas Entradas

DataFrame e Series do pandas ou do Polars, lista e dicionário. A saída volta no mesmo formato da entrada.
</div>

</div>

## Instalação Rápida

```bash
pip install dataframeit[openai]     # OpenAI (provider padrão)
pip install dataframeit[google]     # Google Gemini
pip install dataframeit[anthropic]  # Anthropic
pip install dataframeit[all]        # todos os providers, busca web, Polars e Excel
```

## Próximos Passos

<div class="nav-grid" markdown>

<div class="nav-card" markdown>
### :material-download: [Instalação](getting-started/installation.md)
Configure com seu provider preferido
</div>

<div class="nav-card" markdown>
### :material-rocket-launch: [Início Rápido](getting-started/quickstart.md)
Primeiro projeto em 5 minutos
</div>

<div class="nav-card" markdown>
### :material-book-open-variant: [Guias](guides/basic-usage.md)
Paralelismo, retry, busca web
</div>

<div class="nav-card" markdown>
### :material-robot: [Referência para LLMs](reference/llm-reference.md)
Documentação compacta para assistentes de código
</div>

</div>
