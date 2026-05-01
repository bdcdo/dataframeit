# Provedores

Configure diferentes provedores de LLM via LangChain.

## Providers Suportados

| Provider | Identificador | Modelos Atuais (2025/2026) |
|----------|---------------|----------------------|
| Google | `google_genai` | gemini-3-flash-preview, gemini-2.5-flash, gemini-2.5-pro |
| OpenAI | `openai` | gpt-5.2, gpt-5.2-mini, gpt-4.1 |
| Anthropic | `anthropic` | claude-sonnet-4-5, claude-opus-4-6, claude-haiku-4-5 |
| Groq | `groq` | llama-3.3-70b-versatile, llama-3.1-8b-instant, openai/gpt-oss-120b, openai/gpt-oss-20b, groq/compound |
| Cohere | `cohere` | command-r, command-r-plus |
| Mistral | `mistral` | mistral-large, mistral-small |

## Google Gemini (Padrão)

```bash
pip install dataframeit[google]
export GOOGLE_API_KEY="sua-chave"
```

```python
# Padrão - não precisa especificar
resultado = dataframeit(df, Model, PROMPT)

# Explícito
resultado = dataframeit(
    df, Model, PROMPT,
    provider='google_genai',
    model='gemini-3-flash-preview'
)

# Com parâmetros extras
resultado = dataframeit(
    df, Model, PROMPT,
    provider='google_genai',
    model='gemini-2.5-pro',
    model_kwargs={
        'temperature': 0.2,
        'top_p': 0.9
    }
)
```

### Modelos Recomendados

| Modelo | Uso | Custo |
|--------|-----|-------|
| `gemini-3-flash-preview` | Uso geral, mais recente | Baixo |
| `gemini-2.5-flash` | Uso geral, rápido | Baixo |
| `gemini-2.5-pro` | Tarefas complexas, reasoning | Médio |

## OpenAI

```bash
pip install dataframeit[openai]
export OPENAI_API_KEY="sua-chave"
```

```python
resultado = dataframeit(
    df, Model, PROMPT,
    provider='openai',
    model='gpt-5.2-mini'
)

# Com modelo mais avançado
resultado = dataframeit(
    df, Model, PROMPT,
    provider='openai',
    model='gpt-5.2',
    model_kwargs={
        'temperature': 0.2
    }
)
```

### Modelos Recomendados

| Modelo | Uso | Custo |
|--------|-----|-------|
| `gpt-5.2-mini` | Uso geral, econômico | Baixo |
| `gpt-5.2` | Máxima qualidade | Alto |
| `gpt-4.1` | Coding, instruções precisas | Médio |

## Anthropic Claude

```bash
pip install dataframeit[anthropic]
export ANTHROPIC_API_KEY="sua-chave"
```

```python
resultado = dataframeit(
    df, Model, PROMPT,
    provider='anthropic',
    model='claude-sonnet-4-5'
)

# Com max_tokens
resultado = dataframeit(
    df, Model, PROMPT,
    provider='anthropic',
    model='claude-opus-4-6',
    model_kwargs={
        'max_tokens': 4096
    }
)
```

### Modelos Recomendados

| Modelo | Uso | Custo |
|--------|-----|-------|
| `claude-sonnet-4-5` | Uso geral, excelente qualidade | Médio |
| `claude-opus-4-6` | Máxima qualidade, agentic | Alto |
| `claude-haiku-4-5` | Rápido, econômico | Baixo |

## Groq

```bash
pip install dataframeit[groq]
export GROQ_API_KEY="sua-chave"
```

```python
resultado = dataframeit(
    df, Model, PROMPT,
    provider='groq',
    model='llama-3.3-70b-versatile'
)

# Modelo mais rápido/econômico
resultado = dataframeit(
    df, Model, PROMPT,
    provider='groq',
    model='llama-3.1-8b-instant',
    model_kwargs={
        'temperature': 0.2
    }
)

# GPT-OSS para tarefas com raciocínio mais pesado
resultado = dataframeit(
    df, Model, PROMPT,
    provider='groq',
    model='openai/gpt-oss-120b'
)
```

### Modelos Recomendados

Produção:

| Modelo | Contexto | Throughput | Uso |
|--------|----------|-----------|-----|
| `llama-3.3-70b-versatile` | 131K | ~280 t/s | Uso geral, boa qualidade |
| `llama-3.1-8b-instant` | 131K | ~560 t/s | Alta velocidade, econômico |
| `openai/gpt-oss-120b` | 131K | ~500 t/s | Raciocínio, tarefas complexas |
| `openai/gpt-oss-20b` | 131K | ~1000 t/s | Mais rápido que o 120b, custo baixo |
| `groq/compound` | - | ~450 t/s | Sistema agêntico com web search e execução de código embutidos |

Preview (podem mudar ou ser descontinuados):

| Modelo | Uso |
|--------|-----|
| `meta-llama/llama-4-scout-17b-16e-instruct` | Llama 4 Scout, alta velocidade |
| `qwen/qwen3-32b` | Qwen3, bom para reasoning |

!!! note "Disponibilidade e free tier"
    O Groq oferece free tier com limites de requisições por minuto por modelo. A lista de modelos muda com frequência (especialmente os `preview`); verifique [console.groq.com/docs/models](https://console.groq.com/docs/models) para o catálogo atual e limites.

## Cohere

```bash
pip install langchain-cohere
export COHERE_API_KEY="sua-chave"
```

```python
resultado = dataframeit(
    df, Model, PROMPT,
    provider='cohere',
    model='command-r-plus'
)
```

## Mistral

```bash
pip install langchain-mistralai
export MISTRAL_API_KEY="sua-chave"
```

```python
resultado = dataframeit(
    df, Model, PROMPT,
    provider='mistral',
    model='mistral-large-latest'
)
```

## Servidor no Brasil (São Paulo)

Os providers acima usam endpoints públicos globais. Para servir do Brasil — útil por latência, residência de dados ou exigência regulatória — use um dos três caminhos abaixo. Em todos eles, o `dataframeit` repassa o que vier em `model_kwargs` direto para o LangChain.

### Vertex AI (Gemini em `southamerica-east1`)

Duas variantes. A primeira não exige instalar pacote novo:

```python
# Variante A: usa langchain-google-genai (já é dep do provider 'google_genai')
resultado = dataframeit(
    df, Model, PROMPT,
    provider='google_genai',
    model='gemini-2.5-flash',
    model_kwargs={
        'vertexai': True,
        'project': '<id-do-projeto-gcp>',
        'location': 'southamerica-east1',
    },
)
```

```python
# Variante B: usa langchain-google-vertexai (provider dedicado)
# pip install langchain-google-vertexai
resultado = dataframeit(
    df, Model, PROMPT,
    provider='google_vertexai',
    model='gemini-2.5-flash',
    model_kwargs={
        'project': '<id-do-projeto-gcp>',
        'location': 'southamerica-east1',
    },
)
```

Autenticação (qualquer variante):

```bash
gcloud auth application-default login
# OU
export GOOGLE_APPLICATION_CREDENTIALS=/caminho/service-account.json
```

### AWS Bedrock (`sa-east-1`)

```bash
pip install langchain-aws
aws configure  # ou exporte AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
```

```python
resultado = dataframeit(
    df, Model, PROMPT,
    provider='bedrock_converse',
    model='anthropic.claude-3-5-sonnet-20240620-v1:0',
    model_kwargs={'region_name': 'sa-east-1'},
)
```

Os IDs de modelo disponíveis em `sa-east-1` mudam — confira o console Bedrock antes. Para inferência cross-region, use o prefixo `us.` (ex.: `us.anthropic.claude-...`) e ajuste `region_name` ao que sua conta tiver habilitado.

Para a API Bedrock legada (não-converse), troque por `provider='bedrock'` mantendo o mesmo `model_kwargs`. A nova API (`bedrock_converse`) é recomendada para novos projetos.

### Azure OpenAI (Brazil South)

```bash
pip install langchain-openai
export AZURE_OPENAI_API_KEY="sua-chave"
export AZURE_OPENAI_ENDPOINT="https://<seu-recurso>.openai.azure.com/"
export OPENAI_API_VERSION="2025-03-01-preview"
```

```python
resultado = dataframeit(
    df, Model, PROMPT,
    provider='azure_openai',
    model='gpt-4o',  # ou o nome do deployment
    model_kwargs={'azure_deployment': '<nome-do-deployment>'},
)
```

A região é codificada no `AZURE_OPENAI_ENDPOINT` — provisione o recurso em "Brazil South" no portal Azure.

A versão da API (`OPENAI_API_VERSION`) muda com frequência. Confira a versão estável mais recente em [aka.ms/azure-openai-api-versions](https://aka.ms/azure-openai-api-versions).

## Comparação de Preços (Aproximado - 2025)

| Provider | Modelo | Input (1M tokens) | Output (1M tokens) |
|----------|--------|-------------------|-------------------|
| Google | gemini-3-flash-preview | $0.50 | $3.00 |
| Google | gemini-2.5-pro | $1.25 | $5.00 |
| OpenAI | gpt-5.2-mini | $0.30 | $1.20 |
| OpenAI | gpt-5.2 | $5.00 | $15.00 |
| Anthropic | claude-sonnet-4-5 | $3.00 | $15.00 |
| Anthropic | claude-haiku-4-5 | $1.00 | $5.00 |

!!! note "Preços mudam"
    Verifique os preços atuais nos sites oficiais dos providers.

## Passando API Key Diretamente

Se preferir não usar variáveis de ambiente:

```python
resultado = dataframeit(
    df, Model, PROMPT,
    provider='openai',
    model='gpt-5.2-mini',
    api_key='sk-...'  # Sua chave diretamente
)
```

!!! warning "Segurança"
    Evite colocar API keys diretamente no código. Prefira variáveis de ambiente.

## Parâmetros Comuns (model_kwargs)

| Parâmetro | Descrição | Providers |
|-----------|-----------|-----------|
| `temperature` | Criatividade (0-1) | Todos |
| `top_p` | Nucleus sampling | Todos |
| `max_tokens` | Limite de saída | Todos |
