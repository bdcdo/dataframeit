# Perguntas Frequentes

## Instalação e autenticação

### `ImportError` pedindo para instalar um pacote

O provider escolhido precisa do extra correspondente. A mensagem diz qual instalar, por exemplo:

```bash
pip install dataframeit[anthropic]
```

Os extras estão em [Instalação](../getting-started/installation.md). Providers do LangChain sem extra próprio, como Cohere e Mistral, pedem o pacote `langchain-<provider>`.

### Erro de autenticação (401 ou 403)

A chave de API não foi encontrada ou foi recusada. Confira a variável de ambiente do provider (`OPENAI_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, `GROQ_API_KEY`) no mesmo processo que roda o Python; num notebook, defina-a antes de chamar `dataframeit()`. Erro de autenticação não ganha nova tentativa: todas as linhas terminam com status `'error'`.

Os providers `codex` e `claude_code` usam a autenticação local das ferramentas e não aceitam `api_key`; ver [Provedores](providers.md).

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

Com `resume=False`, se as colunas do modelo já estão no DataFrame e `reprocess_columns` não foi passado, o dataframeit emite um aviso e devolve os dados sem processar, para não sobrescrever resultados. Use `resume=True` para continuar, ou `reprocess_columns=[...]` para refazer colunas específicas.

## Limites e custos

### Muitos erros 429 (rate limit)

O provider recusou requisições por excesso de taxa. O dataframeit já reduz os workers pela metade a cada 429 e tenta de novo com backoff, mas a solução estável é baixar `parallel_requests` ou aumentar `rate_limit_delay`. A conta está em [Performance](performance.md#calculando-o-delay-ideal). Com busca web, o limite do provedor de busca costuma ser o mais apertado; ver [Busca Web](web-search.md#rate-limits-e-processamento-paralelo).

### Quanto vai custar?

Rode primeiro uma amostra (`df.sample(30)`) com `track_tokens=True`. O resumo ao fim da execução mostra os tokens e, com busca, os créditos gastos; multiplique pela razão entre o tamanho do dataset e o da amostra. Os preços por modelo estão em [Provedores](providers.md).

### Uma execução longa foi interrompida

Use `batch_size` e `checkpoint_path` para gravar o progresso durante a execução, e retome com `read_df` e `resume=True`; ver [Checkpoints em Execuções Longas](performance.md#checkpoints-em-execucoes-longas).
