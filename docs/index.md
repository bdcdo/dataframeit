# DataFrameIt

**Enriqueça DataFrames com LLMs de forma simples e estruturada.**

DataFrameIt é uma biblioteca Python que permite processar textos em DataFrames e extrair informações estruturadas usando Modelos de Linguagem (LLMs). Defina o que você quer extrair com Pydantic, e a biblioteca cuida do resto.

## Por que usar?

- **Simples**: Uma função, um modelo Pydantic, um prompt. Pronto.
- **Estruturado**: Saídas validadas automaticamente com Pydantic
- **Resiliente**: Retry automático, rate limiting, processamento paralelo
- **Flexível**: Suporta Gemini, OpenAI, Anthropic, Cohere, Mistral e mais

## Instalação Rápida

```bash
pip install dataframeit[google]  # Para usar Google Gemini (recomendado)
```

## Exemplo em 30 Segundos

```python
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
from dataframeit import dataframeit

# 1. Defina o que você quer extrair
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
| Produto excelente! Superou expectativas. | positivo | alta |
| Péssimo atendimento, nunca mais compro. | negativo | alta |
| Entrega ok, produto mediano. | neutro | media |

## Próximos Passos

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } **Instalação**

    ---

    Configure a biblioteca com seu provider preferido

    [:octicons-arrow-right-24: Guia de instalação](getting-started/installation.md)

-   :material-rocket-launch:{ .lg .middle } **Início Rápido**

    ---

    Crie seu primeiro projeto em 5 minutos

    [:octicons-arrow-right-24: Quickstart](getting-started/quickstart.md)

-   :material-book-open-variant:{ .lg .middle } **Guias**

    ---

    Aprenda recursos avançados como paralelismo e retry

    [:octicons-arrow-right-24: Ver guias](guides/basic-usage.md)

-   :material-robot:{ .lg .middle } **Referência para LLMs**

    ---

    Documentação compacta otimizada para assistentes de código

    [:octicons-arrow-right-24: LLM Reference](reference/llm-reference.md)

</div>
