# Instalação

## Instalação Básica

O DataFrameIt integra múltiplos provedores de LLM por LangChain ou pelos SDKs oficiais de ferramentas locais. Escolha o provider que deseja usar:

=== "OpenAI (Padrão)"

    ```bash
    pip install dataframeit[openai]
    ```

    Modelos: `gpt-6-luna`, `gpt-6-sol`, `gpt-6-astra`

=== "Google Gemini"

    ```bash
    pip install dataframeit[google]
    ```

    Modelos: `gemini-3.8-flash`, `gemini-3.6-flash`, `gemini-3.5-flash-lite`

=== "Anthropic"

    ```bash
    pip install dataframeit[anthropic]
    ```

    Modelos: `claude-sonnet-5`, `claude-opus-5-5`, `claude-haiku-4-5`

=== "Groq"

    ```bash
    pip install dataframeit[groq]
    ```

    Modelos: `openai/gpt-oss-120b`, `openai/gpt-oss-20b` e outros do [catálogo do Groq](https://console.groq.com/docs/models)

=== "Codex (Experimental)"

    ```bash
    pip install dataframeit[codex]
    # ou
    uv add "dataframeit[codex]"
    ```

    O extra fixa o SDK Python oficial e seu runtime compatível. O DataFrameIt sempre usa esse runtime empacotado; uma instalação externa do comando `codex` não participa da execução. O provider permanece experimental porque as versões fixadas do SDK e do runtime ainda são de pré-lançamento.

=== "Claude Code"

    ```bash
    pip install dataframeit[claude-code]
    ```

    O extra instala o Claude Agent SDK, que traz o Claude Code CLI. A autenticação é a do próprio Claude Code: as credenciais de um login feito numa instalação do Claude Code na máquina, ou `ANTHROPIC_API_KEY`. Ver [Provedores](../guides/providers.md#claude-code).

=== "Todos os Providers"

    ```bash
    pip install dataframeit[all]
    ```

    Instala os providers OpenAI, Google, Anthropic, Groq e Claude Code, a busca web (Tavily e Exa), Polars e Excel. Enquanto experimental, o provider Codex não faz parte de `all`; instale `dataframeit[codex]` separadamente.

Outros providers do LangChain, como Cohere e Mistral, não têm extra: instale o pacote `langchain-<provider>` correspondente. Ver [Provedores](../guides/providers.md).

## Com Busca Web (Opcional)

```bash
pip install dataframeit[search]       # Tavily
pip install dataframeit[search-exa]   # Exa
pip install dataframeit[search-all]   # os dois
```

A chave de API de cada provedor de busca está em [Busca Web](../guides/web-search.md).

## Com Polars (Opcional)

Se você usa Polars ao invés de Pandas, ou quer checkpoints em `.parquet` (o extra traz o `pyarrow`):

```bash
pip install dataframeit[openai,polars]
```

## Com Excel (Opcional)

Para checkpoints em `.xlsx` ou ler arquivos Excel via `read_df()`:

```bash
pip install dataframeit[excel]
```

## Configuração de Autenticação

Configure as credenciais correspondentes ao seu provider:

=== "OpenAI"

    ```bash
    export OPENAI_API_KEY="sua-chave-openai"
    ```

    Obtenha sua chave em: [OpenAI Platform](https://platform.openai.com/api-keys)

=== "Google Gemini"

    ```bash
    export GOOGLE_API_KEY="sua-chave-google"
    ```

    Obtenha sua chave em: [Google AI Studio](https://aistudio.google.com/apikey)

=== "Anthropic"

    ```bash
    export ANTHROPIC_API_KEY="sua-chave-anthropic"
    ```

    Obtenha sua chave em: [Anthropic Console](https://console.anthropic.com/)

=== "Groq"

    ```bash
    export GROQ_API_KEY="sua-chave-groq"
    ```

    Obtenha sua chave em: [Groq Console](https://console.groq.com/keys)

=== "Codex"

    Se `auth.json` ainda não existir, instale o [Codex CLI oficial](https://learn.chatgpt.com/docs/codex/cli) e autentique uma vez:

    ```bash
    codex --config cli_auth_credentials_store='"file"' login
    ```

    O CLI externo serve somente para criar `auth.json`; a opção explícita evita armazenar as credenciais apenas no keyring do sistema. Esse é o único arquivo do estado persistente do Codex vinculado ao `CODEX_HOME` efêmero; o app-server ainda herda as variáveis de ambiente do processo. O DataFrameIt executa o runtime pinado pelo extra; não passe `api_key` ao `dataframeit()` para esse provider.

## Verificando a Instalação

```python
import dataframeit
print(dataframeit.__version__)
```

Se o provider escolhido não estiver instalado, o `dataframeit()` avisa antes de começar qual extra falta.

## Próximo Passo

Agora que você instalou o DataFrameIt, veja o [Início Rápido](quickstart.md) para criar seu primeiro projeto.
