# Perguntas Frequentes

## Instalação e autenticação

### `ImportError` pedindo para instalar um pacote

O provider escolhido precisa do pacote dele, e a mensagem diz o que instalar. Nos providers do LangChain, ela sugere o pacote do provider ou o extra `all`:

```bash
pip install langchain-anthropic
# ou
pip install dataframeit[all]
```

O extra do provider, como `dataframeit[anthropic]`, instala o mesmo pacote. O `all` traz os providers que têm extra, menos o Codex; os providers do LangChain sem extra próprio, como Cohere e Mistral, precisam do pacote `langchain-<provider>`, que o `all` não inclui. No `claude_code` e no `codex`, a mensagem pede o extra do provider (`dataframeit[claude-code]`, `dataframeit[codex]`), e na busca web, o pacote do provedor de busca. Os extras estão em [Instalação](../getting-started/installation.md).

### Erro de autenticação (401 ou 403)

A chave de API não foi encontrada ou foi recusada. Confira a variável de ambiente do provider (`OPENAI_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, `GROQ_API_KEY`) no mesmo processo que roda o Python; num notebook, defina-a antes de chamar `dataframeit()`. Erro de autenticação não ganha nova tentativa: todas as linhas terminam com status `'error'`.

Os providers `codex` e `claude_code` usam a autenticação local das ferramentas. O `codex` recusa `api_key` com `ProviderConfigurationError`, e o `claude_code` a ignora; ver [Provedores](providers.md).

## Resultado

### `KeyError: '_dataframeit_status'`

Quando nenhuma linha falha e nenhuma registra detalhe, a coluna de status e a `_error_details` são removidas da saída. Confira se a coluna existe antes de filtrar:

```python
if '_dataframeit_status' in resultado.columns:
    erros = resultado[resultado['_dataframeit_status'] == 'error']
```

### `ValueError` sobre a coluna de texto

Sem `text_column`, o dataframeit procura as colunas `texto`, `text`, `decisao`, `content` e `content_text`, nessa ordem, ou usa a única coluna do DataFrame. Com várias colunas e nenhum desses nomes, informe a coluna:

```python
resultado = dataframeit(df, Modelo, PROMPT, text_column='comentario')
```

### Algumas linhas ficaram com `"Texto ausente"`

Linhas com texto vazio ou nulo não vão ao modelo. Elas recebem status `'error'` e esse detalhe, para que não passem por processadas.

### Rodei de novo e as linhas com erro não foram refeitas

Com `resume=True` (padrão), só as linhas sem status são processadas. Para refazer as que falharam, limpe o status delas; ver [Reprocessando Erros](error-handling.md#reprocessando-erros).

### As colunas do modelo já existem e nada foi processado

Com `resume=False` e a coluna de status no DataFrame, se as colunas do modelo já estão nele e `reprocess_columns` não foi passado, o dataframeit emite um aviso e devolve os dados sem processar, para não sobrescrever resultados. Use `resume=True` para continuar, ou `reprocess_columns=[...]` para refazer colunas específicas.

### `ValueError` porque uma coluna já existe com o nome de um campo do modelo

Quando a entrada chega sem a coluna de status e não há checkpoint a reler, toda linha conta como nova, e a resposta do modelo sobrescreveria a coluna da entrada que tem o nome de um campo, como uma classificação feita à mão. O dataframeit recusa a execução antes de qualquer chamada, com `resume=True` ou `resume=False`; a coluna sem nenhum valor e a que está em `reprocess_columns` não contam. Na retomada de um `checkpoint_path`, a coluna acrescentada à entrada depois da interrupção não é checada. Renomeie ou remova a coluna, ou passe-a em `reprocess_columns` para sobrescrevê-la de propósito. Se o DataFrame é uma saída do dataframeit, que sai sem a coluna de status quando nenhuma linha teve erro, devolva a coluna com `df["_dataframeit_status"] = "processed"` no pandas ou `df = df.with_columns(pl.lit("processed").alias("_dataframeit_status"))` no polars (ou o nome passado em `status_column`): a execução retoma dela, e `reprocess_columns` refaz só as colunas indicadas.

## Limites e custos

### Muitos erros 429 (rate limit)

O provider recusou requisições por excesso de taxa. O dataframeit tenta de novo com backoff e, quando a linha falha depois de todas as tentativas, reduz os workers pela metade. No `codex`, o rate limit e a sobrecarga do app-server são reconhecidos pelo código ou pelo status do erro, sem ler a mensagem; no `claude_code`, o rate limit (429) e a sobrecarga (529) são reconhecidos pelo status que o SDK informa a partir do `claude-agent-sdk` 0.1.76 (do 0.1.48 ao 0.1.75, sem o status, são repetidos como falha transitória e não reduzem os workers); nos providers do LangChain, o rate limit é reconhecido pelo status 429 ou pela classe do erro e, quando o erro não traz nenhum dos dois, pelo texto da mensagem. A solução estável é baixar `parallel_requests` ou aumentar `rate_limit_delay`. A conta está em [Performance](performance.md#calculando-o-delay-ideal). Com busca web, o limite do provedor de busca costuma ser o mais apertado; ver [Busca Web](web-search.md#rate-limits-e-processamento-paralelo).

### Quanto vai custar?

Rode primeiro uma amostra (`df.sample(30)`) com `track_tokens=True`. O resumo ao fim da execução mostra os tokens e, com busca, os créditos gastos; multiplique pela razão entre o tamanho do dataset e o da amostra. Os preços por modelo estão em [Provedores](providers.md).

### Uma execução longa foi interrompida

Use `batch_size` e `checkpoint_path` para gravar o progresso durante a execução, e retome rodando a mesma chamada de novo, que continua do arquivo; ver [Checkpoints em Execuções Longas](performance.md#checkpoints-em-execucoes-longas).

Um Ctrl-C grava no checkpoint as linhas concluídas desde a última gravação antes de interromper, e a retomada não as chama de novo. A linha cuja chamada o Ctrl-C cortou fica pendente. Com `parallel_requests`, a interrupção espera as chamadas em andamento terminarem, e elas entram na gravação; um segundo Ctrl-C corta essa espera, e as chamadas que não terminaram ficam pendentes.
