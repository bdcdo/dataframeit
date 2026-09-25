# DataFrameIt

[![PyPI version](https://badge.fury.io/py/dataframeit.svg)](https://badge.fury.io/py/dataframeit)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/bdcdo/dataframeit/actions/workflows/tests.yml/badge.svg)](https://github.com/bdcdo/dataframeit/actions/workflows/tests.yml)

[English](README.md) · [Português](README.pt-BR.md) · **Español**

**Enriquece DataFrames con LLMs de forma simple y estructurada.**

DataFrameIt procesa textos en DataFrames usando Modelos de Lenguaje (LLMs) y extrae información estructurada definida por modelos Pydantic.

**[Documentación completa](https://brunodcdo.com.br/dataframeit)** (en portugués e inglés) | **[Referencia para LLMs](https://brunodcdo.com.br/dataframeit/reference/llm-reference/)**

## Instalación

```bash
pip install dataframeit[openai]       # OpenAI (proveedor por defecto)
pip install dataframeit[google]       # Google Gemini
pip install dataframeit[anthropic]    # Anthropic Claude
pip install dataframeit[groq]         # Groq
pip install dataframeit[codex]        # SDK oficial de Codex (experimental)
pip install dataframeit[claude-code]  # Claude Code mediante el Claude Agent SDK
pip install dataframeit[all]          # todos los proveedores excepto Codex, búsqueda web, Polars y Excel
```

Extras opcionales: `search` (Tavily), `search-exa` (Exa), `search-all`, `polars`, `excel`.

Configura la autenticación del proveedor:

```bash
export OPENAI_API_KEY="tu-clave"  # o GOOGLE_API_KEY, ANTHROPIC_API_KEY, GROQ_API_KEY
```

El proveedor experimental `codex` es opcional, no forma parte del extra `all`, usa el runtime empaquetado y requiere autenticación local en archivo. Consulta la [documentación de instalación](https://brunodcdo.com.br/dataframeit/getting-started/installation/) para configurar el extra y las credenciales.

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

- **Múltiples proveedores**: OpenAI, Google Gemini, Anthropic y Groq con extras propios, cualquier otro proveedor de LangChain (Cohere, Mistral, Vertex AI, Bedrock, Azure) instalando su paquete, además de Claude Code y Codex mediante sus SDKs oficiales
- **Múltiples tipos de entrada**: DataFrame y Series de pandas o de Polars, list, dict
- **Salida estructurada**: Validación con Pydantic; una respuesta rechazada vuelve al modelo con el error
- **Resiliencia**: Reintento automático con backoff exponencial y checkpoints periódicos (`batch_size` + `checkpoint_path`) para retomar ejecuciones largas
- **Rendimiento**: Procesamiento paralelo que reduce los workers a la mitad ante rate limits, rate limiting configurable
- **Búsqueda web**: Tavily o Exa, por campo o por grupo de campos, con campos condicionales (`condition`, `depends_on`)
- **Seguimiento**: Uso de tokens, créditos de búsqueda y métricas de throughput

## Configuración por campo

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

La lista completa de opciones por campo (incluidos `max_search_calls`, `condition` y `depends_on`) está en la [guía de búsqueda web](https://brunodcdo.com.br/dataframeit/guides/web-search/) (en portugués).

## Documentación

- [Inicio rápido](https://brunodcdo.com.br/dataframeit/getting-started/quickstart/)
- [Guías](https://brunodcdo.com.br/dataframeit/guides/basic-usage/)
- [Referencia de la API](https://brunodcdo.com.br/dataframeit/reference/api/)
- [Referencia para LLMs](https://brunodcdo.com.br/dataframeit/reference/llm-reference/) - Página compacta optimizada para asistentes de código
- [Preguntas frecuentes](https://brunodcdo.com.br/dataframeit/guides/faq/)
- [Historial de versiones](CHANGELOG.md) (en portugués)

## Ejemplos

Consulta la carpeta [`example/`](example/) para notebooks de Jupyter con casos de uso completos.

## Contribuir

Consulta [CONTRIBUTING.md](CONTRIBUTING.md) (en portugués).

## Licencia

MIT
