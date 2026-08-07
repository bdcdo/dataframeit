# DataFrameIt

[![PyPI version](https://badge.fury.io/py/dataframeit.svg)](https://badge.fury.io/py/dataframeit)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[English](README.md) · [Português](README.pt-BR.md) · **Español**

**Enriquece DataFrames con LLMs de forma simple y estructurada.**

DataFrameIt procesa textos en DataFrames usando Modelos de Lenguaje (LLMs) y extrae información estructurada definida por modelos Pydantic.

**[Documentación completa](https://bdcdo.github.io/dataframeit)** | **[Referencia para LLMs](https://bdcdo.github.io/dataframeit/reference/llm-reference/)**

## Instalación

```bash
pip install dataframeit[google]  # Google Gemini (recomendado)
pip install dataframeit[openai]  # OpenAI
pip install dataframeit[anthropic]  # Anthropic Claude
```

Configura tu API key:

```bash
export GOOGLE_API_KEY="tu-clave"  # o OPENAI_API_KEY, ANTHROPIC_API_KEY
```

## Ejemplo rápido

```python
from pydantic import BaseModel
from typing import Literal
import pandas as pd
from dataframeit import dataframeit

# 1. Define qué extraer
class Sentimiento(BaseModel):
    sentimiento: Literal['positivo', 'negativo', 'neutro']
    confianza: Literal['alta', 'media', 'baja']

# 2. Tus datos
df = pd.DataFrame({
    'texto': [
        '¡Producto excelente! Superó mis expectativas.',
        'Pésima atención, nunca más compro aquí.',
        'La entrega bien, el producto mediocre.'
    ]
})

# 3. ¡Procesa!
resultado = dataframeit(df, Sentimiento, "Analiza el sentimiento del texto.")
print(resultado)
```

**Salida:**

| texto | sentimiento | confianza |
|-------|-------------|-----------|
| ¡Producto excelente! ... | positivo | alta |
| Pésima atención... | negativo | alta |
| La entrega bien... | neutro | media |

Los nombres de clases y campos son arbitrarios: los notebooks de [`example/`](example/) usan nombres en portugués.

## Funcionalidades

- **Múltiples proveedores**: Google Gemini, OpenAI, Anthropic, Cohere, Mistral vía LangChain
- **Múltiples tipos de entrada**: DataFrame, Series, list, dict
- **Salida estructurada**: Validación automática con Pydantic
- **Resiliencia**: Reintento automático con backoff exponencial
- **Rendimiento**: Procesamiento paralelo, rate limiting configurable
- **Búsqueda web**: Integración con Tavily para enriquecer datos
- **Seguimiento**: Monitoreo de tokens y métricas de throughput
- **Configuración por campo**: Prompts y parámetros de búsqueda personalizados por campo (v0.5.2+)

## Configuración por campo (nuevo en v0.5.2)

Configura prompts y parámetros de búsqueda específicos para cada campo usando `json_schema_extra`:

```python
from pydantic import BaseModel, Field

class MedicamentoInfo(BaseModel):
    # Campo con el prompt por defecto
    principio_activo: str = Field(description="Principio activo del medicamento")

    # Campo con prompt personalizado (reemplaza el prompt base)
    enfermedad_rara: str = Field(
        description="Clasificación de enfermedad rara",
        json_schema_extra={
            "prompt": "Busca en Orphanet (orpha.net). Analiza: {texto}"
        }
    )

    # Campo con prompt adicional (se añade al prompt base)
    evaluacion_conitec: str = Field(
        description="Evaluación de la CONITEC",
        json_schema_extra={
            "prompt_append": "Busca ÚNICAMENTE en el sitio de la CONITEC (gov.br/conitec)."
        }
    )

    # Campo con parámetros de búsqueda personalizados
    estudios_clinicos: str = Field(
        description="Estudios clínicos relevantes",
        json_schema_extra={
            "prompt_append": "Busca estudios clínicos recientes.",
            "search_depth": "advanced",
            "max_results": 10
        }
    )

# Requiere search_per_field=True
resultado = dataframeit(
    df,
    MedicamentoInfo,
    "Analiza el medicamento: {texto}",
    use_search=True,
    search_per_field=True,
)
```

**Opciones disponibles en `json_schema_extra`:**

| Opción | Descripción |
|--------|-------------|
| `prompt` o `prompt_replace` | Reemplaza por completo el prompt base |
| `prompt_append` | Añade texto al prompt base |
| `search_depth` | `"basic"` o `"advanced"` (override por campo) |
| `max_results` | Número de resultados de búsqueda (1-20) |

## Documentación

- [Inicio rápido](https://bdcdo.github.io/dataframeit/getting-started/quickstart/)
- [Guías](https://bdcdo.github.io/dataframeit/guides/basic-usage/)
- [Referencia de la API](https://bdcdo.github.io/dataframeit/reference/api/)
- [Referencia para LLMs](https://bdcdo.github.io/dataframeit/reference/llm-reference/) - Página compacta optimizada para asistentes de código

## Ejemplos

Consulta la carpeta [`example/`](example/) para notebooks de Jupyter con casos de uso completos.

## Licencia

MIT
