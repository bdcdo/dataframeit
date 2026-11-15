# Início Rápido

Este guia mostra como usar o DataFrameIt em 5 minutos.

## Passo 1: Defina seu Modelo Pydantic

O modelo Pydantic define a estrutura dos dados que você quer extrair de cada texto:

```python
from typing import Literal

from pydantic import BaseModel, Field

class Decisao(BaseModel):
    """Informações extraídas de uma sentença."""
    reu: str = Field(description="Nome do réu")
    tipo_reu: Literal['público', 'privado'] = Field(
        description="Se o réu é ente público ou pessoa privada"
    )
    resultado: Literal['procedente', 'parcialmente procedente', 'improcedente']
```

!!! tip "Dica"
    Use `Literal` para campos com valores fixos. Isso garante que o LLM retorne apenas valores válidos.

## Passo 2: Prepare seus Dados

O DataFrameIt aceita vários tipos de entrada:

=== "DataFrame"

    ```python
    import pandas as pd

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
    ```

=== "Lista"

    ```python
    textos = [
        'Maria Souza move ação contra o Banco Alfa S.A. por cobrança indevida. '
        'Julgo PROCEDENTE o pedido e condeno o réu a pagar R$ 5.000,00 por danos morais.',
        'José Lima pede que o Estado de São Paulo forneça o medicamento prescrito. '
        'Julgo PARCIALMENTE PROCEDENTE o pedido, só para o fornecimento do genérico.',
        'Ana Costa pede indenização ao Município de Campinas por queda em buraco na via. '
        'Não há prova do dano. Julgo IMPROCEDENTE o pedido.',
    ]
    ```

=== "Dicionário"

    ```python
    textos = {
        '0001234-56.2024.8.26.0100': (
            'Maria Souza move ação contra o Banco Alfa S.A. por cobrança indevida. '
            'Julgo PROCEDENTE o pedido e condeno o réu a pagar R$ 5.000,00 por danos morais.'
        ),
        '0004567-89.2024.8.26.0053': (
            'José Lima pede que o Estado de São Paulo forneça o medicamento prescrito. '
            'Julgo PARCIALMENTE PROCEDENTE o pedido, só para o fornecimento do genérico.'
        ),
        '0007890-12.2024.8.26.0114': (
            'Ana Costa pede indenização ao Município de Campinas por queda em buraco na via. '
            'Não há prova do dano. Julgo IMPROCEDENTE o pedido.'
        ),
    }
    ```

## Passo 3: Processe!

```python
from dataframeit import dataframeit

resultado = dataframeit(
    df,                                     # Seus dados
    Decisao,                                # Modelo Pydantic
    "Extraia as informações da sentença.",  # Prompt
    provider='openai',                      # o padrão; ou 'google_genai', 'anthropic', 'groq'
    text_column='sentenca',                 # a coluna com o texto a analisar
)

print(resultado)
```

**Saída:**

| processo | sentenca | reu | tipo_reu | resultado |
|----------|----------|-----|----------|-----------|
| 0001234-56.2024.8.26.0100 | Maria Souza move ação... | Banco Alfa S.A. | privado | procedente |
| 0004567-89.2024.8.26.0053 | José Lima pede que... | Estado de São Paulo | público | parcialmente procedente |
| 0007890-12.2024.8.26.0114 | Ana Costa pede... | Município de Campinas | público | improcedente |

Sem `text_column`, o dataframeit procura a coluna de texto entre alguns nomes comuns e levanta `ValueError` se o DataFrame tiver várias colunas e nenhuma com esses nomes ([FAQ](../guides/faq.md#valueerror-sobre-a-coluna-de-texto)). Com lista ou dicionário, passe `textos` no lugar de `df` e deixe `text_column` de fora: não há coluna a escolher.

Com `track_tokens=True`, o padrão, a saída traz também as colunas de uso de tokens `_input_tokens`, `_cached_input_tokens`, `_output_tokens` e `_reasoning_tokens`, omitidas acima; ver [Colunas Adicionadas](../reference/api.md#colunas-adicionadas).

## Exemplo Completo

```python
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from dataframeit import dataframeit

# 1. Modelo Pydantic
class Decisao(BaseModel):
    reu: str = Field(description="Nome do réu")
    tipo_reu: Literal['público', 'privado'] = Field(description="Se o réu é ente público ou pessoa privada")
    resultado: Literal['procedente', 'parcialmente procedente', 'improcedente']

# 2. Dados
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

# 3. Processar
resultado = dataframeit(
    df,
    Decisao,
    "Extraia as informações da sentença.",
    provider='openai',
    text_column='sentenca',
)

# 4. Salvar
resultado.to_excel('resultado.xlsx', index=False)
```

## Próximos Passos

- [Conceitos](concepts.md): Entenda como o DataFrameIt funciona
- [Uso Básico](../guides/basic-usage.md): Mais exemplos práticos
- [Tratamento de Erros](../guides/error-handling.md): Lidando com falhas
