# Conceitos

Entenda os conceitos fundamentais do DataFrameIt.

## Arquitetura

```
┌─────────────────────────────────────────────────────────────┐
│                        dataframeit()                         │
├─────────────────────────────────────────────────────────────┤
│  Entrada           │  Processamento      │  Saída           │
│  ─────────         │  ─────────────      │  ─────           │
│  • DataFrame       │  • Para cada linha: │  • DataFrame     │
│  • Series          │    1. Monta prompt  │    com colunas   │
│  • List            │    2. Chama LLM     │    extraídas     │
│  • Dict            │    3. Valida resp.  │                  │
│                    │    4. Retry se erro │                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     LangChain + Provider                     │
├─────────────────────────────────────────────────────────────┤
│  Google Gemini │ OpenAI │ Anthropic │ Cohere │ Mistral      │
└─────────────────────────────────────────────────────────────┘
```

## Componentes Principais

### 1. Modelo Pydantic

O modelo Pydantic define **o que** você quer extrair. Cada campo vira uma coluna no DataFrame de saída.

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional

class Analise(BaseModel):
    # Campo obrigatório com valores fixos
    categoria: Literal['A', 'B', 'C'] = Field(
        description="Categoria do item"
    )

    # Campo obrigatório com texto livre
    resumo: str = Field(
        description="Resumo em uma frase"
    )

    # Campo opcional
    observacao: Optional[str] = Field(
        default=None,
        description="Observações adicionais, se houver"
    )
```

!!! info "Por que Pydantic?"
    - **Validação automática**: O LLM é forçado a retornar dados no formato correto
    - **Documentação**: As descrições dos campos ajudam o LLM a entender o que extrair
    - **Type safety**: Erros de tipo são capturados automaticamente

### 2. Prompt Template

O prompt define **como** o LLM deve processar cada texto.

```python
# Simples - texto é adicionado automaticamente ao final
PROMPT = "Classifique o sentimento do texto."

# Com placeholder - controle onde o texto aparece
PROMPT = """
Você é um analista especializado.

Documento:
{texto}

Extraia as informações solicitadas do documento acima.
"""
```

### 3. Providers via LangChain

O DataFrameIt usa LangChain para abstrair diferentes provedores de LLM:

| Provider | Modelos Populares | Variável de Ambiente |
|----------|-------------------|---------------------|
| `google_genai` | gemini-2.0-flash, gemini-1.5-pro | `GOOGLE_API_KEY` |
| `openai` | gpt-4o, gpt-4o-mini, o1, o3-mini | `OPENAI_API_KEY` |
| `anthropic` | claude-3-5-sonnet, claude-3-opus | `ANTHROPIC_API_KEY` |

## Fluxo de Processamento

```
Para cada linha do DataFrame:
│
├─► 1. Monta o prompt (template + texto da linha)
│
├─► 2. Envia para o LLM via LangChain
│
├─► 3. Recebe resposta estruturada
│
├─► 4. Valida com Pydantic
│   │
│   ├─► Sucesso: marca como 'processed'
│   │
│   └─► Erro: retry com backoff exponencial
│       │
│       ├─► Sucesso após retry: marca como 'processed'
│       │
│       └─► Falha após max_retries: marca como 'error'
│
└─► 5. Adiciona campos extraídos ao DataFrame
```

## Colunas Automáticas

O DataFrameIt adiciona colunas de controle automaticamente:

| Coluna | Descrição |
|--------|-----------|
| `_dataframeit_status` | Status: `'processed'`, `'error'`, ou `None` |
| `_error_details` | Detalhes do erro (quando status é `'error'`) |
| `_input_tokens` | Tokens de entrada (com `track_tokens=True`) |
| `_output_tokens` | Tokens de saída (com `track_tokens=True`) |
| `_total_tokens` | Total de tokens (com `track_tokens=True`) |

## Próximos Passos

- [Uso Básico](../guides/basic-usage.md): Exemplos práticos
- [Tratamento de Erros](../guides/error-handling.md): Configurar retry e fallbacks
- [Performance](../guides/performance.md): Paralelismo e rate limiting
