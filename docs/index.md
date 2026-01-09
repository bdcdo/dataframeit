<div class="hero" markdown>

# DataFrameIt

<p class="tagline">Enriqueça DataFrames com LLMs de forma simples e estruturada</p>

<div class="badges">
  <a href="https://pypi.org/project/dataframeit/"><img src="https://badge.fury.io/py/dataframeit.svg" alt="PyPI version"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</div>

<div class="cta-buttons" markdown>
  <a href="getting-started/quickstart/" class="cta-button primary" markdown>
    :material-rocket-launch: Começar Agora
  </a>
  <a href="reference/llm-reference/" class="cta-button secondary" markdown>
    :material-robot: Referência para LLMs
  </a>
</div>

</div>

## O que é?

DataFrameIt processa textos em DataFrames usando **Modelos de Linguagem (LLMs)** e extrai informações estruturadas definidas por **modelos Pydantic**. Uma função, um modelo, um prompt — pronto.

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

### Múltiplos Providers

Google Gemini, OpenAI GPT-5, Anthropic Claude 4.5, Cohere, Mistral — todos via LangChain.
</div>

<div class="feature-card" markdown>
<div class="icon" markdown>:material-check-decagram:</div>

### Saída Estruturada

Validação automática com Pydantic. Defina campos, tipos e descrições — o LLM respeita.
</div>

<div class="feature-card" markdown>
<div class="icon" markdown>:material-shield-refresh:</div>

### Resiliência

Retry automático com backoff exponencial. Rate limiting configurável. Nunca perde progresso.
</div>

<div class="feature-card" markdown>
<div class="icon" markdown>:material-rocket-launch:</div>

### Performance

Processamento paralelo com auto-ajuste. Métricas de throughput em tempo real.
</div>

<div class="feature-card" markdown>
<div class="icon" markdown>:material-web:</div>

### Busca Web

Integração com Tavily para enriquecer dados com informações da internet.
</div>

<div class="feature-card" markdown>
<div class="icon" markdown>:material-format-list-bulleted-type:</div>

### Múltiplas Entradas

DataFrame, Series, lista, dicionário — tudo funciona. Polars incluído.
</div>

</div>

## Instalação Rápida

```bash
pip install dataframeit[google]  # Google Gemini 3 (recomendado)
pip install dataframeit[openai]  # OpenAI GPT-5
pip install dataframeit[anthropic]  # Anthropic Claude 4.5
```

## Próximos Passos

<div class="nav-grid" markdown>

<a href="getting-started/installation/" class="nav-card" markdown>
  <div class="nav-icon" markdown>:material-download:</div>
  <div class="nav-content">
    <h4>Instalação</h4>
    <p>Configure com seu provider preferido</p>
  </div>
</a>

<a href="getting-started/quickstart/" class="nav-card" markdown>
  <div class="nav-icon" markdown>:material-rocket-launch:</div>
  <div class="nav-content">
    <h4>Início Rápido</h4>
    <p>Primeiro projeto em 5 minutos</p>
  </div>
</a>

<a href="guides/basic-usage/" class="nav-card" markdown>
  <div class="nav-icon" markdown>:material-book-open-variant:</div>
  <div class="nav-content">
    <h4>Guias</h4>
    <p>Paralelismo, retry, busca web</p>
  </div>
</a>

<a href="reference/llm-reference/" class="nav-card" markdown>
  <div class="nav-icon" markdown>:material-robot:</div>
  <div class="nav-content">
    <h4>Referência para LLMs</h4>
    <p>Documentação compacta para assistentes de código</p>
  </div>
</a>

</div>
