# Performance

Otimize o processamento com paralelismo, rate limiting e tracking de tokens.

## Processamento Paralelo

Use `parallel_requests` para acelerar o processamento. Até `parallel_requests` linhas ficam em voo ao mesmo tempo, e uma linha nova é despachada assim que outra termina: o worker que acabou cedo não espera o mais lento.

```python
resultado = dataframeit(
    df,
    Model,
    PROMPT,
    parallel_requests=5  # 5 requisições simultâneas
)
```

### Recomendações por Tamanho

| Dataset | Configuração |
|---------|--------------|
| < 50 linhas | `parallel_requests=1` (padrão) |
| 50-500 linhas | `parallel_requests=3` a `5` |
| > 500 linhas | `parallel_requests=5` a `10` |

### Auto-redução em Rate Limits

Quando uma linha falha depois de todas as tentativas com erro de rate limit, o DataFrameIt reduz pela metade o teto de linhas em voo; as que já estão em voo terminam:

```
Início: 10 workers
Rate limit detectado → 5 workers
Rate limit detectado → 2 workers
Rate limit detectado → 1 worker
```

No `codex`, o rate limit e a sobrecarga do app-server são reconhecidos pelo código ou pelo status do erro, sem ler a mensagem. No `claude_code`, o rate limit (429) e a sobrecarga (529) são reconhecidos pelo status que o SDK informa a partir do `claude-agent-sdk` 0.1.76; do 0.1.48 ao 0.1.75, o SDK não informa o status, e o 401, o 400, o 429 e o 529 são repetidos como falha transitória, sem reduzir os workers. Nos providers do LangChain, o rate limit é reconhecido pelo status 429 ou pela classe do erro e, quando o erro não traz nenhum dos dois, pelo texto da mensagem; a sobrecarga (529, 503) não reduz os workers. A caixa de limite de requisições, no aviso de falha da linha, sai para os erros reconhecidos como rate limit desse modo, também quando não há workers a reduzir, como no modo sequencial.

!!! info "Segurança"
    Workers são apenas **reduzidos**, nunca aumentados automaticamente. Isso evita custos inesperados.

### Callback por Linha

`on_row_done` recebe um `RowResult` a cada linha processada nesta chamada, inclusive em erro, depois de a linha estar gravada no DataFrame. A linha pulada por `resume` não é entregue.

```python
from dataframeit import RowResult, dataframeit

def ao_concluir(linha: RowResult) -> None:
    print(linha.position, linha.index, linha.success, linha.values)

resultado = dataframeit(
    df,
    Model,
    PROMPT,
    parallel_requests=5,
    on_row_done=ao_concluir,
)
```

Os campos do `RowResult` estão na [referência da API](../reference/api.md#rowresult).

!!! warning "Thread do worker"
    Com `parallel_requests` > 1, o callback roda na thread do worker que processou a linha, enquanto as outras seguem gravando. O que ele compartilha (contador, fila, tela) precisa de sincronização própria, e o que ele precisa da linha vem do `RowResult`, não do DataFrame. Uma exceção dentro dele vira aviso, e a execução segue.

No fim da execução, um `UserWarning` informa os workers do início e do fim e sugere usar o número final em `parallel_requests`.

## Rate Limiting

Use `rate_limit_delay` para prevenir erros de rate limit:

```python
resultado = dataframeit(
    df,
    Model,
    PROMPT,
    rate_limit_delay=1.0  # cada worker pausa 1 segundo depois de cada linha
)
```

### Calculando o Delay Ideal

`rate_limit_delay` é uma pausa de cada worker depois de cada linha concluída com sucesso. Com vários workers, as pausas correm em paralelo, e a taxa máxima de requisições é `parallel_requests × 60 / rate_limit_delay` por minuto. Para ficar abaixo de um limite:

```
delay = 60 × parallel_requests / requisições_por_minuto

Exemplos:
- 60 req/min,  1 worker   → delay = 1.0s
- 60 req/min,  5 workers  → delay = 5.0s
- 500 req/min, 5 workers  → delay = 0.6s
```

A taxa real fica abaixo desse teto, porque cada chamada ao modelo também leva tempo.

### Por Provider

O limite de requisições por minuto depende do modelo e do nível da conta, e muda com frequência. Consulte o valor da sua conta na página oficial e aplique a fórmula acima:

- [Google Gemini](https://ai.google.dev/gemini-api/docs/rate-limits)
- [OpenAI](https://developers.openai.com/api/docs/guides/rate-limits)
- [Anthropic](https://platform.claude.com/docs/en/api/rate-limits)
- [Groq](https://console.groq.com/docs/rate-limits)

### Combinando com Paralelismo

```python
# 5 workers, cada um pausando 0,5s depois de cada linha: até 600 req/min
resultado = dataframeit(
    df,
    Model,
    PROMPT,
    parallel_requests=5,
    rate_limit_delay=0.5
)
```

## Checkpoints em Execuções Longas

Em datasets grandes (milhares de linhas, horas de execução), um kill/crash perde
todo o progresso em memória. Use `batch_size` + `checkpoint_path` para persistir
o DataFrame a cada N linhas processadas:

```python
resultado = dataframeit(
    df,
    Model,
    PROMPT,
    batch_size=100,
    checkpoint_path="checkpoint.xlsx",
)
```

Formato inferido pela extensão do arquivo (`.csv`, `.xlsx`, `.parquet`). Em qualquer formato, os campos de estrutura do modelo, isto é, lista, tupla, conjunto, dict, `Sequence`, `AbstractSet`, `Mapping`, `TypedDict`, `NamedTuple` e modelo aninhado, são gravados como texto JSON, o conjunto em ordem fixa, e o `Enum`, pelo valor; `read_df` devolve as estruturas, e a retomada devolve ao tipo declarado as estruturas e os campos de data, hora, `datetime`, `Enum`, `int` e `bool`, inclusive com `strict=True`: a data gravada como texto, o inteiro relido como float e a tupla relida como lista voltam a `date`, `int` e `tuple`. O CSV e o XLSX gravam o texto vazio e o `None` na mesma célula vazia. Na retomada, ela volta `""` no campo de texto que não admite `None`, como `str` ou `str = "padrão"`, e `None` no campo que o admite, como `Optional[str]`; para guardar o `""` de um campo opcional, use parquet. No campo `str` cujo default é `None`, dado direto ou pela fábrica, ela também volta `None`. O texto vazio do campo cuja fábrica lê os outros campos não é restaurado: a célula vazia conta como ausente, e a retomada a preenche com o valor da fábrica, mesmo onde a execução gravou `""`. O validador que troca uma resposta por `None` num campo `str` dá um valor que o tipo do campo recusa. No parquet, a retomada acusa esse `None` para reprocessar. No CSV e no XLSX, ela acusa o campo quando o validador troca o próprio texto vazio por `None`, e devolve `""` quando ele troca outra resposta, como `"N/A"`, porque a célula vazia não diz qual das duas foi gravada. Para guardar esse `None`, declare o campo como `Optional[str]`. Isso vale para a execução que aplica os validadores do modelo; no modo por campo (`search_per_field`), eles não rodam na execução, que grava o texto vazio como veio, e a retomada do CSV e do XLSX acusa o campo cujo validador troca o texto vazio por `None`, enquanto a do parquet devolve o `""` gravado. O parquet grava o texto vazio e o `None` à parte, e a retomada devolve o texto vazio como foi gravado. No XLSX, o texto que começa com `=` é gravado como texto, e não como fórmula. O XLSX guarda número como double, e um inteiro acima de 2**53 volta arredondado dele; para ids desse tamanho, use CSV ou parquet. Nenhum dos três formatos guarda o NaN de um campo `float` à parte do nulo: o CSV e o XLSX gravam os dois como célula vazia, e o parquet, como nulo. Na retomada, o NaN volta ausente, e o campo obrigatório que o recebeu é acusado para reprocessar. No parquet, o inteiro acima de 2**63 - 1 ao lado de uma linha pendente e o inteiro acima de 2**53 ao lado de um float na mesma coluna não são gravados, e a gravação do checkpoint falha com aviso. Em caso de interrupção, rode a mesma chamada de novo, com a mesma entrada: com `resume=True`, o padrão, e o arquivo presente, a execução continua do checkpoint. A cada gravação, `<checkpoint>.dataframeit.json` é gravado ao lado dele com a assinatura da execução (prompt, provider, modelo, `model_kwargs` menos `timeout`, schema, busca e as colunas de texto e de status), o hash do texto de cada linha e o hash do próprio checkpoint. O checkpoint só é retomado quando os três batem; com outra configuração, outra entrada ou um arquivo que não é o que a assinatura descreve, a execução avisa, começa do zero e o sobrescreve. Numa linha já concluída, com sucesso ou com erro, os campos do modelo vêm do checkpoint, sobre o que a entrada trazia neles. Para manter uma saída corrigida à mão como está, passe-a com a coluna de status, e o checkpoint não é relido. A saída completa sai sem ela, e a coluna volta com `saida["_dataframeit_status"] = "processed"` (ou o nome passado em `status_column`). Com `resume=False`, o arquivo é ignorado e sobrescrito.

Na assinatura, o schema é o JSON Schema do modelo. Quando o `json_schema_extra` de um campo, inclusive aninhado, tem uma função, como a condição de `field_condition`, ou um conjunto com valor que não é inteiro, como `{"in": {"pf", "mei"}}` numa `condition`, entram no lugar dele os atributos de cada campo, com o conjunto em ordem e a função pelo nome qualificado e pelo código: bytecode, constantes e nomes que ela lê. O `functools.partial` entra pela função que embrulha e pelos argumentos fixados; o objeto com `__call__`, pelo código do `__call__` e pelos atributos da instância, inclusive os de `__slots__`; e o método ligado, pelo código do método e pelo dono: o objeto entra pelos atributos, e a classe do `classmethod`, só pelo nome. Editar a função, o `__call__`, o método ou um argumento do `partial`, ou mudar um atributo do objeto, muda a assinatura, e a execução recomeça. O que ela lê de fora do próprio código não entra: com outro valor numa variável global ou do escopo de fora, outro valor padrão de argumento ou outro corpo numa função que ela chama, a assinatura é a mesma, e a retomada aceita as linhas já concluídas, em que o campo foi pedido ou pulado pela condição anterior. Para mudar a condição por um desses caminhos, passe `resume=False` ou outro `checkpoint_path`. O bytecode muda de uma versão menor do Python para outra, como da 3.11 para a 3.12, e o checkpoint de um modelo com função não é retomado depois dessa troca.

Para inspecionar o que já foi processado, `read_df` carrega o arquivo com listas, dicts e textos nos tipos do modelo. Esse DataFrame também serve de entrada para a retomada, e aí o arquivo não é lido de novo:

```python
from dataframeit import read_df

df_parcial = read_df("checkpoint.xlsx", Model)
resultado = dataframeit(
    df_parcial, Model, PROMPT,
    resume=True, batch_size=100, checkpoint_path="checkpoint.xlsx",
)
```

Quando o provider informa uma falha que impede qualquer linha seguinte, como a cota de uso esgotada (`ProviderUsageLimitError`) ou o runtime do `codex` encerrado (`ProviderAbortError`), a execução para de despachar linhas, grava o checkpoint e avisa quantas ficaram sem status. As colunas de controle ficam na saída. Em que casos rodar de novo com `resume=True` processa só as pendentes, e o que acontece com `reprocess_columns`, está em [Falhas que interrompem a execução](../reference/exceptions.md#falhas-que-interrompem-a-execucao).

## Tracking de Tokens

Monitore uso e custos com `track_tokens=True`:

```python
resultado = dataframeit(
    df,
    Model,
    PROMPT,
    track_tokens=True
)
```

Ao final, o DataFrameIt registra um resumo (sempre em português) no logger `dataframeit.stats`, em nível INFO. Sem logging configurado, o resumo sai em stderr; com logging configurado, sai pelos handlers da aplicação:

```
============================================================
ESTATISTICAS DE USO
============================================================
Modelo: gpt-6-luna
Total de tokens: 15,432
  - Input:  12,345 tokens
  - Output: 3,087 tokens
------------------------------------------------------------
METRICAS DE THROUGHPUT
------------------------------------------------------------
Tempo total: 45.2s
Workers paralelos: 5
Requisicoes: 100
  - RPM (req/min): 132.7
  - TPM (tokens/min): 20,478
============================================================
```

A seção de throughput só aparece com `parallel_requests > 1`. Linhas de cache e raciocínio aparecem quando o provider informa esses tokens. Com busca web, o resumo ganha a seção de buscas e créditos; com `claude_code`, o custo informado pelo SDK.

Para silenciar o resumo e manter as colunas de uso, suba o nível do logger:

```python
import logging

logging.getLogger("dataframeit.stats").setLevel(logging.WARNING)
```

### Colunas Adicionadas

O resultado registra o uso por linha; a [Referência LLM](../reference/llm-reference.md#colunas-adicionadas-automaticamente) define as colunas e como interpretar valores nulos ou zero em `_cached_input_tokens`.

### Calculando Custos

```python
resultado = dataframeit(df, Model, PROMPT, track_tokens=True)

# Exemplo: preços do gpt-6-luna
preco_input = 0.10 / 1_000_000    # $0.10 por 1M tokens
preco_output = 0.50 / 1_000_000   # $0.50 por 1M tokens

custo_input = resultado['_input_tokens'].sum() * preco_input
custo_output = resultado['_output_tokens'].sum() * preco_output
custo_total = custo_input + custo_output

print(f"Custo estimado: ${custo_total:.4f}")
```

## Métricas de Throughput

A seção de throughput do resumo acima mostra as requisições e os tokens por minuto efetivos. Use esses números para calibrar `parallel_requests` e `rate_limit_delay` para os limites da sua conta.

## Configuração Otimizada

### Para Máxima Velocidade

```python
resultado = dataframeit(
    df,
    Model,
    PROMPT,
    parallel_requests=10,     # Muitos workers
    rate_limit_delay=0.0,     # Sem delay
    max_retries=5,            # Retry agressivo
    track_tokens=True
)
```

### Para Estabilidade

```python
resultado = dataframeit(
    df,
    Model,
    PROMPT,
    parallel_requests=3,      # Poucos workers
    rate_limit_delay=1.0,     # Delay conservador
    max_retries=3,
    base_delay=2.0,
    track_tokens=True
)
```

### Para Economia

```python
resultado = dataframeit(
    df,
    Model,
    PROMPT,
    parallel_requests=1,      # Sequencial
    rate_limit_delay=1.5,     # Delay alto
    model='gpt-6-luna',       # Modelo barato
    track_tokens=True
)
```
