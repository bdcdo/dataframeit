# Instalação

## Instalação Básica

O DataFrameIt integra múltiplos provedores de LLM por LangChain ou pelos SDKs oficiais de ferramentas locais. Escolha o provider que deseja usar:

=== "Google Gemini (Recomendado)"

    ```bash
    pip install dataframeit[google]
    ```

    Modelos: `gemini-3-flash-preview`, `gemini-2.5-flash`, `gemini-2.5-pro`

=== "OpenAI"

    ```bash
    pip install dataframeit[openai]
    ```

    Modelos: `gpt-5.2`, `gpt-5.2-mini`, `gpt-4.1`

=== "Anthropic"

    ```bash
    pip install dataframeit[anthropic]
    ```

    Modelos: `claude-sonnet-4-5`, `claude-opus-4-6`, `claude-haiku-4-5`

=== "Codex (Experimental)"

    ```bash
    pip install dataframeit[codex]
    # ou
    uv add "dataframeit[codex]"
    ```

    O extra fixa o SDK Python oficial e seu runtime compatível. O DataFrameIt sempre usa esse runtime empacotado; uma instalação externa do comando `codex` não participa da execução. O provider permanece experimental porque as versões fixadas do SDK e do runtime ainda são de pré-lançamento.

=== "Todos os Providers"

    ```bash
    pip install dataframeit[all]
    ```

    Enquanto experimental, o provider Codex não faz parte de `all`; instale `dataframeit[codex]` separadamente.

## Com Polars (Opcional)

Se você usa Polars ao invés de Pandas:

```bash
pip install dataframeit[google,polars]
```

## Com Excel (Opcional)

Para checkpoints em `.xlsx` ou ler arquivos Excel via `read_df()`:

```bash
pip install dataframeit[excel]
```

## Configuração de Autenticação

Configure as credenciais correspondentes ao seu provider:

=== "Google Gemini"

    ```bash
    export GOOGLE_API_KEY="sua-chave-google"
    ```

    Obtenha sua chave em: [Google AI Studio](https://aistudio.google.com/apikey)

=== "OpenAI"

    ```bash
    export OPENAI_API_KEY="sua-chave-openai"
    ```

    Obtenha sua chave em: [OpenAI Platform](https://platform.openai.com/api-keys)

=== "Anthropic"

    ```bash
    export ANTHROPIC_API_KEY="sua-chave-anthropic"
    ```

    Obtenha sua chave em: [Anthropic Console](https://console.anthropic.com/)

=== "Codex"

    Se `auth.json` ainda não existir, instale o [Codex CLI oficial](https://learn.chatgpt.com/docs/codex/cli) e autentique uma vez:

    ```bash
    codex --config cli_auth_credentials_store='"file"' login
    ```

    O CLI externo serve somente para criar `auth.json`; a opção explícita evita armazenar as credenciais apenas no keyring do sistema. O DataFrameIt compartilha somente esse arquivo com um `CODEX_HOME` efêmero e executa o runtime pinado pelo extra; não passe `api_key` ao `dataframeit()` para esse provider.

## Verificando a Instalação

```python
from dataframeit import dataframeit
print("DataFrameIt instalado com sucesso!")
```

## Próximo Passo

Agora que você instalou o DataFrameIt, veja o [Início Rápido](quickstart.md) para criar seu primeiro projeto.
