# Início Rápido

Este guia mostra como usar o DataFrameIt em 5 minutos.

## Passo 1: Defina seu Modelo Pydantic

O modelo Pydantic define a estrutura dos dados que você quer extrair:

```python
from pydantic import BaseModel, Field
from typing import Literal

class Sentimento(BaseModel):
    """Análise de sentimento de um texto."""
    sentimento: Literal['positivo', 'negativo', 'neutro'] = Field(
        description="Sentimento geral do texto"
    )
    confianca: Literal['alta', 'media', 'baixa'] = Field(
        description="Nível de confiança na classificação"
    )
```

!!! tip "Dica"
    Use `Literal` para campos com valores fixos. Isso garante que o LLM retorne apenas valores válidos.

## Passo 2: Prepare seus Dados

O DataFrameIt aceita vários tipos de entrada:

=== "DataFrame"

    ```python
    import pandas as pd

    df = pd.DataFrame({
        'texto': [
            'Produto excelente! Superou expectativas.',
            'Péssimo atendimento, nunca mais compro.',
            'Entrega ok, produto mediano.'
        ]
    })
    ```

=== "Lista"

    ```python
    textos = [
        'Produto excelente! Superou expectativas.',
        'Péssimo atendimento, nunca mais compro.',
        'Entrega ok, produto mediano.'
    ]
    ```

=== "Dicionário"

    ```python
    textos = {
        'review_001': 'Produto excelente! Superou expectativas.',
        'review_002': 'Péssimo atendimento, nunca mais compro.',
        'review_003': 'Entrega ok, produto mediano.'
    }
    ```

## Passo 3: Processe!

```python
from dataframeit import dataframeit

resultado = dataframeit(
    df,                                      # Seus dados
    Sentimento,                              # Modelo Pydantic
    "Analise o sentimento do texto."         # Prompt
)

print(resultado)
```

**Saída:**

```
                                       texto sentimento confianca
0  Produto excelente! Superou expectativas.   positivo      alta
1  Péssimo atendimento, nunca mais compro.   negativo      alta
2            Entrega ok, produto mediano.     neutro     media
```

## Exemplo Completo

```python
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
from dataframeit import dataframeit

# 1. Modelo Pydantic
class Sentimento(BaseModel):
    sentimento: Literal['positivo', 'negativo', 'neutro']
    confianca: Literal['alta', 'media', 'baixa']

# 2. Dados
df = pd.DataFrame({
    'texto': [
        'Produto excelente! Superou expectativas.',
        'Péssimo atendimento, nunca mais compro.',
        'Entrega ok, produto mediano.'
    ]
})

# 3. Processar
resultado = dataframeit(df, Sentimento, "Analise o sentimento do texto.")

# 4. Salvar
resultado.to_excel('resultado.xlsx', index=False)
```

## Próximos Passos

- [Conceitos](concepts.md): Entenda como o DataFrameIt funciona
- [Uso Básico](../guides/basic-usage.md): Mais exemplos práticos
- [Tratamento de Erros](../guides/error-handling.md): Lidando com falhas
