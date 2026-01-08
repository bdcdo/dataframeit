# Uso Básico

Exemplos práticos para casos de uso comuns.

## Análise de Sentimento

```python
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
from dataframeit import dataframeit

class Sentimento(BaseModel):
    sentimento: Literal['positivo', 'negativo', 'neutro']
    confianca: Literal['alta', 'media', 'baixa']

df = pd.DataFrame({
    'texto': [
        'Adorei o produto!',
        'Péssimo, não recomendo.',
        'Normal, nada demais.'
    ]
})

resultado = dataframeit(df, Sentimento, "Analise o sentimento do texto.")
```

## Classificação de Categorias

```python
from pydantic import BaseModel, Field
from typing import Literal

class Categoria(BaseModel):
    categoria: Literal['tecnologia', 'saude', 'financas', 'educacao', 'outro']
    subcategoria: str = Field(description="Subcategoria mais específica")

resultado = dataframeit(
    df,
    Categoria,
    "Classifique o texto na categoria mais apropriada."
)
```

## Extração de Entidades

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class Entidades(BaseModel):
    pessoas: List[str] = Field(description="Nomes de pessoas mencionadas")
    organizacoes: List[str] = Field(description="Nomes de empresas/organizações")
    locais: List[str] = Field(description="Locais mencionados")
    datas: List[str] = Field(description="Datas mencionadas")

PROMPT = """
Extraia todas as entidades nomeadas do texto.
Se não houver entidades de algum tipo, retorne lista vazia.
"""

resultado = dataframeit(df, Entidades, PROMPT)
```

## Resumo de Texto

```python
from pydantic import BaseModel, Field

class Resumo(BaseModel):
    resumo: str = Field(description="Resumo em até 50 palavras")
    pontos_chave: list[str] = Field(description="Lista de 3-5 pontos principais")
    tema_principal: str = Field(description="Tema central em uma palavra")

PROMPT = """
Analise o texto e extraia um resumo conciso.
Identifique os pontos principais e o tema central.
"""

resultado = dataframeit(df, Resumo, PROMPT)
```

## Usando Diferentes Tipos de Entrada

### Com Lista

```python
textos = ['Texto 1', 'Texto 2', 'Texto 3']
resultado = dataframeit(textos, Sentimento, PROMPT)
# Retorna DataFrame com índice numérico
```

### Com Dicionário

```python
documentos = {
    'doc_001': 'Conteúdo do documento 1',
    'doc_002': 'Conteúdo do documento 2',
}
resultado = dataframeit(documentos, Sentimento, PROMPT)
# Retorna DataFrame com chaves como índice
```

### Com Series

```python
series = pd.Series(['Texto A', 'Texto B'], index=['id_1', 'id_2'])
resultado = dataframeit(series, Sentimento, PROMPT)
# Preserva o índice original
```

## Usando Diferentes Providers

```python
# Google Gemini (padrão)
resultado = dataframeit(df, Model, PROMPT)

# OpenAI
resultado = dataframeit(
    df, Model, PROMPT,
    provider='openai',
    model='gpt-5.2-mini'
)

# Anthropic Claude
resultado = dataframeit(
    df, Model, PROMPT,
    provider='anthropic',
    model='claude-sonnet-4.5'
)
```

## Próximos Passos

- [Saída Estruturada](structured-output.md): Modelos Pydantic avançados
- [Tratamento de Erros](error-handling.md): Retry e fallbacks
- [Performance](performance.md): Paralelismo e rate limiting
