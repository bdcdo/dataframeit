# Provedores

Configure diferentes provedores de LLM via LangChain ou pelos SDKs oficiais de ferramentas locais.

## Providers Suportados

| Provider | Identificador | Modelos atuais | Modelo padrão |
|----------|---------------|----------------|---------------|
| OpenAI | `openai` | gpt-6-luna, gpt-6-sol, gpt-6-astra | `gpt-6-luna` |
| Google | `google_genai` | gemini-3.8-flash, gemini-3.6-flash, gemini-3.5-flash-lite | `gemini-3.8-flash` |
| Anthropic | `anthropic` | claude-sonnet-5, claude-opus-5-5, claude-haiku-4-5 | `claude-sonnet-5` |
| Groq | `groq` | openai/gpt-oss-120b, openai/gpt-oss-20b | `openai/gpt-oss-120b` |
| OpenAI Codex (experimental) | `codex` | Modelos suportados pelo runtime empacotado | Escolhido pelo runtime |
| Claude Code | `claude_code` | Modelos e aliases aceitos pelo Claude Code (`sonnet`, `haiku`, `opus`) | Escolhido pelo runtime |
| Cohere | `cohere` | command-r, command-r-plus | Informe `model` |
| Mistral | `mistralai` | mistral-large, mistral-small | Informe `model` |

Sem `provider`, o dataframeit usa `openai` com `gpt-6-luna`. Com `provider` e sem `model`, usa o modelo padrão do provider da tabela; para providers sem modelo padrão, `model` é obrigatório.

## OpenAI (Padrão)

```bash
pip install dataframeit[openai]
export OPENAI_API_KEY="sua-chave"
```

```python
# Padrão - não precisa especificar
resultado = dataframeit(df, Model, PROMPT)

# Com modelo mais avançado
resultado = dataframeit(
    df, Model, PROMPT,
    provider='openai',
    model='gpt-6-sol',
    model_kwargs={
        'reasoning_effort': 'medium'
    }
)
```

### Modelos Recomendados

| Modelo | Uso | Preço (1M tokens, input / output) |
|--------|-----|-----------------------------------|
| `gpt-6-luna` | Alto volume, tarefas focadas | $0.10 / $0.50 |
| `gpt-6-sol` | Tarefas complexas e agênticas | $2.00 / $10.00 |
| `gpt-6-astra` | Máxima qualidade | $10.00 / $50.00 |

## Google Gemini

```bash
pip install dataframeit[google]
export GOOGLE_API_KEY="sua-chave"
```

```python
resultado = dataframeit(
    df, Model, PROMPT,
    provider='google_genai'  # usa gemini-3.8-flash
)

# Com parâmetros extras
resultado = dataframeit(
    df, Model, PROMPT,
    provider='google_genai',
    model='gemini-3.5-flash-lite',
    model_kwargs={
        'thinking_level': 'low'
    }
)
```

### Modelos Recomendados

| Modelo | Uso | Preço (1M tokens, input / output) |
|--------|-----|-----------------------------------|
| `gemini-3.8-flash` | Uso geral, mais recente | $0.75 / $3.75 até 31/12/2026; $1.50 / $7.50 a partir de 2027 |
| `gemini-3.6-flash` | Uso geral | $0.75 / $3.75 até 31/12/2026; $1.50 / $7.50 a partir de 2027 |
| `gemini-3.5-flash-lite` | Alto volume, econômico | $0.30 / $2.50 |

## OpenAI Codex (Experimental)

O provider `codex` usa o [SDK Python oficial](https://github.com/openai/codex/tree/main/sdk/python) e permanece experimental. Para instalar o extra, entender qual runtime é executado e configurar a autenticação local em arquivo, consulte [Instalação](../getting-started/installation.md).

```python
resultado = dataframeit(
    df,
    Model,
    PROMPT,
    provider='codex',
    model='gpt-5.4',
    model_kwargs={'effort': 'medium'},
    parallel_requests=3,
)
```

Para esse provider, `model_kwargs` aceita somente `effort`. `use_search=True` não é suportado. O modelo Pydantic deve ter campos no nível raiz e usar o [subconjunto de JSON Schema aceito por Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs#supported-schemas); `RootModel`, `Any`, campos `dict` com chaves dinâmicas, tuplas fixas e `set` são rejeitados no preflight. A autenticação configurada durante a instalação vem de `auth.json`, portanto não passe `api_key` ao `dataframeit()`.

O DataFrameIt mantém um `codex app-server` por execução do DataFrame e abre uma thread efêmera por linha. Cada execução usa `CODEX_HOME` e workspace isolados; `auth.json` é o único arquivo do estado persistente do Codex vinculado ao runtime, que ainda herda as variáveis de ambiente do processo. Enquanto uma execução usa a credencial, outra execução do DataFrameIt com o mesmo `auth.json` falha antes de iniciar o runtime; isso impede refresh concorrente sem afetar `parallel_requests` dentro da execução ativa. Esse lock coordena somente instâncias do DataFrameIt, portanto não execute o Codex CLI com a mesma credencial até o processamento terminar. Busca web, shell e servidores MCP ficam desativados; aprovações são negadas e o sandbox somente leitura bloqueia escrita. O runtime ainda pode apresentar utilitários internos, como `apply_patch`, sem conceder permissão para alterar arquivos.

## Claude Code

O provider `claude_code` usa o [Claude Agent SDK](https://github.com/anthropics/claude-agent-sdk-python), que executa o Claude Code CLI. A autenticação é a do próprio Claude Code: as credenciais de um login feito numa instalação do Claude Code na máquina (o CLI que o extra traz não fica no `PATH`), ou `ANTHROPIC_API_KEY`. O parâmetro `api_key` é ignorado.

```bash
pip install dataframeit[claude-code]
```

```python
resultado = dataframeit(
    df,
    Model,
    PROMPT,
    provider='claude_code',
    model='haiku',
    model_kwargs={'max_budget_usd': 0.25, 'effort': 'low'},
)
```

Em `model_kwargs`, o provider lê `max_turns` (padrão 1), `max_budget_usd` (padrão 0.50) e `effort`; outras chaves são ignoradas. `max_budget_usd` é o teto de gasto de cada tentativa: como uma resposta vazia ou fora do schema é tentada de novo, uma linha pode gastar até `max_retries` vezes esse valor. `use_search=True` não é suportado.

O texto das linhas é tratado como conteúdo não confiável. A execução roda sem ferramentas, sem os settings de usuário e de projeto e com `--strict-mcp-config`, de modo que servidores MCP e regras `permissions.allow` configurados no Claude Code não chegam a ela. Uma linha que estoura `max_budget_usd` ou `max_turns` termina em erro definitivo, sem nova tentativa.

Com `track_tokens=True`, o resumo ao fim da execução mostra o custo informado pelo SDK, somando as tentativas re-tentadas e as linhas que falharam.

## Anthropic Claude

```bash
pip install dataframeit[anthropic]
export ANTHROPIC_API_KEY="sua-chave"
```

```python
resultado = dataframeit(
    df, Model, PROMPT,
    provider='anthropic'  # usa claude-sonnet-5
)

# Com max_tokens
resultado = dataframeit(
    df, Model, PROMPT,
    provider='anthropic',
    model='claude-opus-5-5',
    model_kwargs={
        'max_tokens': 4096
    }
)
```

### Modelos Recomendados

| Modelo | Uso | Preço (1M tokens, input / output) |
|--------|-----|-----------------------------------|
| `claude-sonnet-5` | Uso geral, velocidade e qualidade | $2.00 / $10.00 |
| `claude-opus-5-5` | Máxima qualidade, agêntico | $4.00 / $20.00 |
| `claude-haiku-4-5` | Rápido, econômico | $1.00 / $5.00 |

## Groq

```bash
pip install dataframeit[groq]
export GROQ_API_KEY="sua-chave"
```

```python
resultado = dataframeit(
    df, Model, PROMPT,
    provider='groq'  # usa openai/gpt-oss-120b
)

# Modelo mais rápido/econômico
resultado = dataframeit(
    df, Model, PROMPT,
    provider='groq',
    model='openai/gpt-oss-20b'
)
```

### Modelos Recomendados

Produção:

| Modelo | Contexto | Throughput | Uso |
|--------|----------|-----------|-----|
| `openai/gpt-oss-120b` | 131K | ~500 t/s | Uso geral, raciocínio |
| `openai/gpt-oss-20b` | 131K | ~1000 t/s | Mais rápido que o 120b, custo baixo |

Preview (podem mudar ou ser descontinuados):

| Modelo | Uso |
|--------|-----|
| `qwen/qwen3.8-27b` | Qwen 3.8, modos thinking e instruct |

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
    provider='mistralai',
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
    model='gemini-3.8-flash',
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
    model='gemini-3.8-flash',
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
    model='global.anthropic.claude-sonnet-5',
    model_kwargs={'region_name': 'sa-east-1'},
)
```

No Bedrock, a Converse API só aceita o Claude Sonnet 5 por perfil de inferência, e em `sa-east-1` o único perfil oferecido é o global (`global.`), que pode processar a requisição fora do Brasil. Se residência de dados for exigência, confira no console Bedrock quais modelos a região oferece com perfil regional.

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

!!! note "Preços mudam"
    Os preços das tabelas acima são os de tabela padrão, por 1M tokens, com as mudanças já anunciadas indicadas na própria tabela. Verifique os valores atuais nos sites oficiais dos providers.

## Passando API Key Diretamente

Se preferir não usar variáveis de ambiente:

```python
resultado = dataframeit(
    df, Model, PROMPT,
    provider='openai',
    api_key='sk-...'  # Sua chave diretamente
)
```

!!! warning "Segurança"
    Evite colocar API keys diretamente no código. Prefira variáveis de ambiente.

## Parâmetros Comuns (model_kwargs)

| Parâmetro | Descrição | Providers |
|-----------|-----------|-----------|
| `temperature` | Criatividade. O dataframeit não envia valor padrão | Depende do modelo: vários atuais rejeitam (ex.: Claude Sonnet 5, OpenAI GPT-6 com raciocínio e série o) |
| `top_p` | Nucleus sampling | Depende do modelo, como `temperature` |
| `max_tokens` | Limite de saída | Todos |
