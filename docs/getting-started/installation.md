# Instalação

## Instalação Básica

O DataFrameIt usa [LangChain](https://langchain.com/) para suportar múltiplos provedores de LLM. Escolha o provider que deseja usar:

=== "Google Gemini (Recomendado)"

    ```bash
    pip install dataframeit[google]
    ```

    Modelos: `gemini-3.0-flash`, `gemini-2.5-flash`, `gemini-2.5-pro`

=== "OpenAI"

    ```bash
    pip install dataframeit[openai]
    ```

    Modelos: `gpt-5.2`, `gpt-5.2-mini`, `gpt-4.1`

=== "Anthropic"

    ```bash
    pip install dataframeit[anthropic]
    ```

    Modelos: `claude-sonnet-4.5`, `claude-opus-4.5`, `claude-haiku-4.5`

=== "Todos os Providers"

    ```bash
    pip install dataframeit[all]
    ```

## Com Polars (Opcional)

Se você usa Polars ao invés de Pandas:

```bash
pip install dataframeit[google,polars]
```

## Configuração de API Keys

Configure a variável de ambiente correspondente ao seu provider:

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

## Verificando a Instalação

```python
from dataframeit import dataframeit
print("DataFrameIt instalado com sucesso!")
```

## Próximo Passo

Agora que você instalou o DataFrameIt, veja o [Início Rápido](quickstart.md) para criar seu primeiro projeto.
