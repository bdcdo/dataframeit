# Instalação

## Instalação Básica

O DataFrameIt usa [LangChain](https://langchain.com/) para suportar múltiplos provedores de LLM. Escolha o provider que deseja usar:

=== "Google Gemini (Recomendado)"

    ```bash
    pip install dataframeit[google]
    ```

    Modelos: `gemini-2.0-flash`, `gemini-1.5-pro`, `gemini-1.5-flash`

=== "OpenAI"

    ```bash
    pip install dataframeit[openai]
    ```

    Modelos: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `o1`, `o3-mini`

=== "Anthropic"

    ```bash
    pip install dataframeit[anthropic]
    ```

    Modelos: `claude-3-5-sonnet-20241022`, `claude-3-opus-20240229`

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
