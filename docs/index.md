<div class="hero" markdown>

# DataFrameIt

<p class="tagline">Enriqueça DataFrames com LLMs de forma simples e estruturada</p>

<div class="badges">
  <a href="https://pypi.org/project/dataframeit/"><img src="https://badge.fury.io/py/dataframeit.svg" alt="PyPI version"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+"></a>
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
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from dataframeit import dataframeit

# 1. Defina o que extrair de cada decisão
class Decisao(BaseModel):
    reu: str = Field(description="Nome do réu")
    tipo_reu: Literal['público', 'privado'] = Field(description="Se o réu é ente público ou pessoa privada")
    resultado: Literal['procedente', 'parcialmente procedente', 'improcedente']

# 2. Seus dados: uma decisão por linha
df = pd.DataFrame({
    'processo': ['0001234-56.2024.8.26.0100', '0004567-89.2024.8.26.0053', '0007890-12.2024.8.26.0114'],
    'sentenca': [
        'Maria Souza move ação contra o Banco Alfa S.A. por cobrança indevida. '
        'Julgo PROCEDENTE o pedido e condeno o réu a pagar R$ 5.000,00 por danos morais.',
        'José Lima pede que o Estado de São Paulo forneça o medicamento prescrito. '
        'Julgo PARCIALMENTE PROCEDENTE o pedido, só para o fornecimento do genérico.',
        'Ana Costa pede indenização ao Município de Campinas por queda em buraco na via. '
        'Não há prova do dano. Julgo IMPROCEDENTE o pedido.',
    ],
})

# 3. Processe!
resultado = dataframeit(
    df,
    Decisao,
    "Extraia as informações da sentença.",
    provider='openai',       # o padrão; ou 'google_genai', 'anthropic', 'groq'
    text_column='sentenca',  # a coluna com o texto a analisar
)
print(resultado)
```

**Saída:**

| processo | sentenca | reu | tipo_reu | resultado |
|----------|----------|-----|----------|-----------|
| 0001234-56.2024.8.26.0100 | Maria Souza move ação... | Banco Alfa S.A. | privado | procedente |
| 0004567-89.2024.8.26.0053 | José Lima pede que... | Estado de São Paulo | público | parcialmente procedente |
| 0007890-12.2024.8.26.0114 | Ana Costa pede... | Município de Campinas | público | improcedente |

Sem `text_column`, o dataframeit procura a coluna de texto entre alguns nomes comuns e levanta `ValueError` se o DataFrame tiver várias colunas e nenhuma com esses nomes ([FAQ](guides/faq.md#valueerror-sobre-a-coluna-de-texto)). Com `track_tokens=True`, o padrão, a saída traz também as colunas de uso de tokens `_input_tokens`, `_cached_input_tokens`, `_output_tokens` e `_reasoning_tokens`, omitidas da tabela acima.

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

Processamento paralelo que reduz os workers diante de rate limit. Com `parallel_requests > 1`, o resumo ao fim da execução traz as métricas de throughput.
</div>

<div class="feature-card" markdown>
<div class="icon" markdown>:material-web:</div>

### Busca Web

Tavily ou Exa para enriquecer dados com informações da internet, com busca por campo e campos condicionais.
</div>

<div class="feature-card" markdown>
<div class="icon" markdown>:material-format-list-bulleted-type:</div>

### Múltiplas Entradas

DataFrame e Series do pandas ou do Polars, lista e dicionário. A saída é um DataFrame, do Polars quando a entrada é do Polars.
</div>

</div>

## Instalação Rápida

```bash
pip install dataframeit[openai]     # OpenAI (provider padrão)
pip install dataframeit[google]     # Google Gemini
pip install dataframeit[anthropic]  # Anthropic
pip install dataframeit[groq]       # Groq
pip install dataframeit[all]        # providers com extra, menos o Codex, mais busca web, Polars e Excel
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
