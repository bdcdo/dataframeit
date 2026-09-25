# Exceções

O `dataframeit()` trata erros de duas formas, conforme o momento em que acontecem.

## Erros antes do processamento

A configuração é validada antes da primeira chamada ao modelo. Um problema aqui interrompe a execução com exceção, e nenhuma linha é processada.

| Exceção | Quando ocorre |
|---------|---------------|
| `ImportError` | Falta o extra do provider ou do provedor de busca (ex.: `pip install dataframeit[anthropic]`). A mensagem diz qual instalar |
| `ValueError` | Parâmetro inválido: `questions` ou `prompt` ausentes, `max_retries < 1`, `batch_size` sem `checkpoint_path`, `search_groups` sem `search_per_field=True`, chave de API de busca ausente, linhas já processadas incompatíveis com o modelo atual |
| `ProviderConfigurationError` | Configuração incompatível com o contrato do provider, como um `model_kwargs` que o `codex` não aceita ou um schema Pydantic que ele não consegue representar. É subclasse de `ValueError` |

```python
from dataframeit import dataframeit, ProviderConfigurationError

try:
    resultado = dataframeit(df, Modelo, PROMPT, provider='codex', model_kwargs={'max_tokens': 500})
except ProviderConfigurationError as erro:
    print(f"Configuração recusada pelo provider: {erro}")
```

## Erros durante o processamento

Uma falha numa linha não interrompe a execução. A linha recebe status `'error'`, o motivo vai para `_error_details` e o processamento segue para a próxima. O texto em `_error_details` tem a forma `[Falhou após N tentativa(s)] Classe: mensagem` (ou `[Erro não-recuperável] Classe: mensagem`), o que permite filtrar por tipo:

```python
if '_dataframeit_status' in resultado.columns:
    erros = resultado[resultado['_dataframeit_status'] == 'error']
    sobrecarga = erros[erros['_error_details'].str.contains('ProviderOverloadedError', na=False)]
```

As classes abaixo classificam a falha e decidem se ela ganha nova tentativa. Todas estão em `dataframeit` e em `dataframeit.errors`.

```
RuntimeError
└── ProviderError                     falha definitiva reportada pelo provider
    └── ProviderTransientError        transitória: ganha nova tentativa
        ├── ProviderOverloadedError   sobrecarga ou limite de taxa do provider
        └── ProviderRejectedOutputError  resposta recusada pela validação (também ValueError)

ValueError
├── ProviderConfigurationError        configuração local incompatível com o provider
└── ProviderOutputError               resposta definitiva fora do contrato de saída
```

| Exceção | Nova tentativa | Situação típica |
|---------|----------------|-----------------|
| `ProviderError` | Não | Erro definitivo do provider, como autenticação recusada ou orçamento do `claude_code` esgotado |
| `ProviderTransientError` | Sim | Falha de rede ou do serviço que costuma passar sozinha |
| `ProviderOverloadedError` | Sim | Sobrecarga ou HTTP 429 do provider |
| `ProviderRejectedOutputError` | Sim | A resposta não passou na validação do modelo Pydantic; a nova tentativa leva ao modelo a resposta recusada e o erro |
| `ProviderOutputError` | Não | O provider terminou sem resposta utilizável, como um turno do `codex` sem conteúdo |

Erros dos providers via LangChain não usam essas classes. Eles são classificados pelo tipo que a própria LangChain declara, pelo status HTTP e pela mensagem. Os detalhes estão em [Tratamento de Erros](../guides/error-handling.md).
