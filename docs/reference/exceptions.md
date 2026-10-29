# Exceções

O `dataframeit()` trata erros de duas formas, conforme o momento em que acontecem.

## Erros antes do processamento

A configuração é validada antes da primeira chamada ao modelo. Um problema aqui interrompe a execução com exceção, e nenhuma linha é processada.

| Exceção | Quando ocorre |
|---------|---------------|
| `ImportError` | Falta o pacote do provider ou do provedor de busca. A mensagem diz qual instalar: nos providers do LangChain, o pacote do provider (ex.: `pip install langchain-anthropic`) ou o extra `all`; no `claude_code` e no `codex`, o extra do provider; na busca web, o pacote do provedor (ex.: `pip install langchain-tavily`) |
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

Uma falha numa linha não interrompe a execução, exceto as que impedem qualquer linha seguinte, descritas em [Falhas que interrompem a execução](#falhas-que-interrompem-a-execucao). A linha recebe status `'error'`, o motivo vai para `_error_details` e o processamento segue para a próxima. O texto em `_error_details` tem a forma `[Falhou após N tentativa(s)] Classe: mensagem` (ou `[Erro não-recuperável] Classe: mensagem`), o que permite filtrar por tipo:

```python
if '_dataframeit_status' in resultado.columns:
    erros = resultado[resultado['_dataframeit_status'] == 'error']
    sobrecarga = erros[erros['_error_details'].str.contains('ProviderOverloadedError', na=False)]
```

As classes abaixo classificam a falha e decidem se ela ganha nova tentativa. Todas estão em `dataframeit` e em `dataframeit.errors`.

```
RuntimeError
└── ProviderError                     falha definitiva reportada pelo provider
    ├── ProviderTransientError        transitória: ganha nova tentativa
    │   ├── ProviderOverloadedError   sobrecarga ou limite de taxa do provider
    │   └── ProviderRejectedOutputError  resposta recusada pela validação (também ValueError)
    └── ProviderAbortError            impede qualquer linha seguinte: a execução para
        └── ProviderUsageLimitError   cota de uso da conta esgotada

ValueError
├── ProviderConfigurationError        configuração local incompatível com o provider
└── ProviderOutputError               resposta definitiva fora do contrato de saída
```

| Exceção | Nova tentativa | Situação típica |
|---------|----------------|-----------------|
| `ProviderError` | Não | Erro definitivo do provider, como autenticação recusada ou orçamento do `claude_code` esgotado |
| `ProviderTransientError` | Sim | Falha de rede ou do serviço que costuma passar sozinha |
| `ProviderOverloadedError` | Sim | Sobrecarga ou HTTP 429 do provider |
| `ProviderRejectedOutputError` | Sim | A resposta não passou na validação do modelo Pydantic. Nos providers do LangChain, a nova tentativa leva ao modelo a resposta recusada e o erro; no `claude_code`, repete o pedido |
| `ProviderOutputError` | Não | O provider terminou sem resposta utilizável, como um turno do `codex` sem conteúdo ou com resposta que não passa na validação do modelo Pydantic |
| `ProviderAbortError` | Não | Falha depois da qual nenhuma linha passaria, como o runtime do `codex` encerrado; a execução para |
| `ProviderUsageLimitError` | Não | Cota de uso da conta esgotada, como a do `codex`; a execução para |

O `ProviderError` e as subclasses trazem em `status_code` o status HTTP que o provider informou fora da mensagem, ou `None` quando ele não informou: no `claude_code`, o `api_error_status` do SDK; no `codex`, o status HTTP do turno que falhou, e 401 no código `unauthorized`. A caixa do aviso de falha da linha sai da classe e desse status, e não da mensagem, que pode trazer texto do modelo. A classe decide primeiro: o `ProviderOverloadedError` dá a caixa de limite de requisições, com ou sem status, e o `ProviderRejectedOutputError` dá a caixa da validação. Nas demais classes, o status decide: 401 dá a caixa de autenticação, 403 a de permissão, e 408 ou 504, num `ProviderTransientError`, a de timeout. Sem classe nem status que decidam, a caixa é a genérica, com o tipo e o começo da mensagem. A caixa de conexão, que só o texto decide, não sai para o `ProviderError`.

Erros dos providers via LangChain não usam essas classes. Eles são classificados pelo tipo que a própria LangChain declara, pelo status HTTP e pela mensagem. Os detalhes estão em [Tratamento de Erros](../guides/error-handling.md).

## Falhas que interrompem a execução

`ProviderAbortError` e a subclasse `ProviderUsageLimitError` não chegam a quem chama o `dataframeit()`. Quando uma linha recebe uma delas, a execução para de despachar linhas: a linha que recebeu o erro e as que ainda não tinham sido despachadas ficam sem status, e, no modo paralelo, as que já estavam em andamento terminam e ficam com status `'processed'`. O checkpoint, quando há um, é gravado, e um aviso diz quantas ficaram pendentes e como retomar. Rodar de novo com `resume=True` processa só as pendentes quando a chamada recebe o status da execução interrompida. Com entrada `pd.DataFrame`, isso vale para a mesma entrada, que é [modificada no lugar](api.md#entrada-modificada-no-lugar), e para a saída devolvida. Com entrada polars, vale só para a saída devolvida: a entrada não é alterada, e repetir a chamada sobre ela processa todas as linhas de novo. Com entrada que não é DataFrame, como `list`, a saída não traz o texto, e só o mesmo `checkpoint_path` permite retomar. Com `checkpoint_path`, repetir a chamada com a mesma entrada retoma do checkpoint, inclusive com polars e `list`; ver [Checkpoints em Execuções Longas](../guides/performance.md#checkpoints-em-execucoes-longas). Com `reprocess_columns`, a interrupção não deixa linha sem status: as linhas que não chegaram a ser reprocessadas, inclusive a que recebeu o erro, ficam com status `'processed'` e os valores anteriores, marcadas em `_error_details`, e `resume=True` sozinho não as refaz. Para refazê-las, repita a chamada com `reprocess_columns`, que processa de novo todas as linhas. Os casos do `codex` estão em [Provedores](../guides/providers.md#openai-codex-experimental).
