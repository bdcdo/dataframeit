# Changelog

Todas as mudanças notáveis deste projeto serão documentadas neste arquivo.

O formato é baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/),
e este projeto adere ao [Versionamento Semântico](https://semver.org/lang/pt-BR/).

## [Unreleased]

### Alterado

- O `ruff check` do CI e do pre-commit confere também `example/`, com as regras de correção listadas em `example/ruff.toml` e sem as de estilo de biblioteca. Na célula que configura a chave, os notebooks de exemplo capturam só o `ImportError` de fora do Colab e as exceções que `userdata.get` levanta, e mostram a mensagem; antes, um `except:` com `pass` engolia também o secret ausente e o Ctrl+C (#211).

### Corrigido



## [0.11.1] - 2026-10-29

### Corrigido

- A retomada pelo checkpoint em CSV e em XLSX não cai quando a `default_factory` de um campo `str` que não admite `None` levanta uma exceção, como a fábrica que lê `os.environ` e dá `KeyError` sem a variável. A retomada chama a fábrica para saber se o default pode ser `None`, e só tratava o `ValueError`: as outras exceções saíam cruas de `dataframeit()`, mesmo com o campo respondido pelo LLM em todas as linhas, e com o `ValueError` a célula vazia contava como ausente, e a validação da linha chamava a fábrica, que levantava quando o LLM tinha respondido `""`. Agora a célula vazia do campo cuja fábrica levanta volta `""`, como a execução a gravou, e a validação da linha não a chama. O campo cuja fábrica lê os outros campos continua não restaurado, e a retomada preenche a célula vazia dele com o valor da fábrica (#288, #293).
- Com `checkpoint_path`, um Ctrl-C (`KeyboardInterrupt`) grava no checkpoint as linhas concluídas desde a última gravação e relança a interrupção, nos modos sequencial e paralelo. Antes, o arquivo ficava na última gravação de `batch_size`, e a retomada pagava de novo as linhas concluídas depois dela. No paralelo, entram as chamadas já em andamento, que terminam enquanto a interrupção espera as threads do executor, também quando o Ctrl-C sai de dentro do despacho; a linha que ainda está na fila, sem thread que a tenha começado, é cancelada e fica para a retomada, e as que um segundo Ctrl-C deixa sem terminar ficam pendentes. A espera pelas linhas corre em fatias curtas, porque o handler de SIGINT que o polars instala ao ser importado faz o kernel reiniciar a espera sem prazo: com o polars instalado, o Ctrl-C do paralelo só chegava quando uma linha terminava, e o segundo não cortava a espera. A falha ao gravar vira aviso e não troca a interrupção (#213).
- A retomada pelo checkpoint deixa de recusar ou trocar valores já gravados (#196). No CSV e no XLSX, que gravam `""` e `None` na mesma célula vazia, o texto vazio de um campo de texto que não admite `None` volta `""`, em vez de a linha ser acusada como incompatível e a retomada pedir `reprocess_columns` para a coluna inteira; no campo que admite `None` e no campo `str` cujo default é `None`, dado direto ou pela fábrica, a célula vazia continua voltando `None`; o texto vazio do campo cuja fábrica lê os outros campos não é restaurado, e a retomada preenche a célula vazia com o valor da fábrica; no campo cujo validador troca o texto vazio por `None` ela continua acusada, e o `None` que o validador dá a outra resposta, como `"N/A"`, num campo `str` volta `""`, porque a célula vazia não o separa do texto vazio; para guardar esse `None`, o campo é `Optional[str]`. No parquet, que guarda os dois à parte, o texto vazio volta como foi gravado, e o `None` num campo `str` é acusado. No XLSX, o texto que começa com `=` é gravado como texto, e não como fórmula, que voltava vazia. Os campos `Sequence`, `AbstractSet`, `set`, `frozenset`, `Mapping` e as versões mutáveis, `TypedDict` e `NamedTuple` passam a contar como estrutura em `get_complex_fields`: o checkpoint os grava em JSON, o conjunto em ordem fixa, e a retomada os devolve ao tipo do campo, em vez de acusá-los como incompatíveis. O `"NA"` de um `Literal["NA", "deferido", None]` é relido como texto, e não como ausente. O aviso do reprocessamento interrompido diz que repetir a chamada com `reprocess_columns` refaz todas as linhas, inclusive as já reprocessadas.
- Com `search_per_field=True`, o campo com configuração de busca no item de uma lista de modelo é buscado item a item também quando a lista ou o item vêm com `Optional` ou `Annotated`, como em `Optional[Annotated[list[Item], Field(min_length=1)]]` ou `list[Item | None]`. Antes, a validação aceitava a configuração, e a extração fazia uma busca só para a linha e deixava os itens sem o campo buscado. A lista cujo item pode ser mais de um modelo, como `list[Livro | Filme]` ou `list[Livro] | list[Filme]`, tem os campos de busca dos dois modelos buscados uma vez para a linha; antes, `list[Livro] | list[Filme]` aplicava a busca de `Livro` a todos os itens, inclusive aos filmes, e não buscava os campos de `Filme` (#198).
- O caminho aninhado de `condition` e de `depends_on`, como `endereco.cidade`, é conferido inteiro contra o modelo antes de processar, enquanto passa por submodelos. Com um erro de digitação depois da raiz, como `endereco.cidadee`, a condição passava na validação e ficava sempre falsa; agora levanta `ValueError`, como a raiz inexistente, e também o caminho que segue depois de um campo escalar, como `endereco.cidade.nome` com `cidade: str`, e o que atravessa lista, tupla ou outra coleção, como `itens.preco` ou `itens.0.preco` com `itens: list[Item]`, que a avaliação não segue. Na união de modelos, o caminho segue cada modelo que tem a parte (#198).
- A condição em dict sobre um campo `Enum` compara o valor do membro, no campo e no operando, e dá o mesmo resultado na execução e na retomada. Antes, `{"field": "tipo", "equals": "pf"}` era falsa na execução, que recebe o membro, e verdadeira na retomada do checkpoint, que relê o valor em texto (#198).
- A assinatura do checkpoint de um modelo com função ou conjunto no `json_schema_extra` de um campo, como a condição de `field_condition` ou `{"in": {"pf", "mei"}}` numa `condition`, deixa de mudar a cada processo, e a execução interrompida retoma em outro processo sem refazer as linhas concluídas. A função entrava pelo repr, com o endereço de memória, e o conjunto, na ordem de hash do processo. Nesses modelos, a assinatura sai de uma forma canônica dos campos: o conjunto em ordem, a função pelo nome qualificado e pelo bytecode, constantes e nomes que lê, sem o que ela lê de fora do próprio código, como uma variável global, o `functools.partial` pela função e pelos argumentos fixados, o objeto com `__call__` pelo código do `__call__` e pelos atributos da instância, inclusive os de `__slots__`, e o método ligado pelo código do método e pelo dono, que entra pelos atributos quando é um objeto e só pelo nome quando é a classe do `classmethod`. O bytecode muda de uma versão menor do Python para outra, e o checkpoint de modelo com função não é retomado depois dessa troca. Os outros modelos, inclusive com conjunto só de inteiros, mantêm a assinatura; o checkpoint gravado antes com conjunto que não é só de inteiros no `json_schema_extra` de um campo, na `condition` ou em outra chave, não é retomado e a execução recomeça (#191).

## [0.11.0] - 2026-10-29

### Adicionado

- `on_row_done`, função chamada uma vez por linha processada nesta chamada, inclusive em erro, depois de a linha estar gravada no DataFrame, com um `RowResult`, exportado por `dataframeit`: rótulo do índice, posição entre as linhas processadas, sucesso, texto de `_error_details`, campos gravados e segundos decorridos. A linha pulada por `resume` não é entregue. Com `parallel_requests` > 1, a chamada vem da thread do worker que processou a linha; uma exceção dentro dela vira aviso, e a execução segue.
- `field_condition(condition, depends_on=...)`, exportado por `dataframeit`, monta o `json_schema_extra` de um campo com condição callable. O dict escrito à mão era recusado pelo checker de tipos, porque o Pydantic tipa `json_schema_extra` como JSON; o helper devolve o mesmo dict com tipo que o `Field` aceita, e recusa `depends_on` vazio (#171).
- As exceções `ProviderError`, `ProviderTransientError`, `ProviderOverloadedError`, `ProviderRejectedOutputError`, `ProviderConfigurationError` e `ProviderOutputError` e o `__version__` passam a ser exportados por `dataframeit`.
- `ProviderAbortError` e a subclasse `ProviderUsageLimitError`, também exportadas, marcam a falha que impede qualquer linha seguinte. A execução para de despachar linhas, grava o checkpoint e avisa quantas ficaram sem status; a linha que recebeu o erro e as que faltam ficam pendentes para `resume=True`, e as colunas de controle ficam na saída enquanto houver linha pendente. Com `reprocess_columns`, a linha que ficou sem reprocessar mantém os valores anteriores e é marcada em `_error_details`, e as marcas vão ao checkpoint.
- No provider `codex`, uma thread sem turno confere na abertura que o runtime resolveu o `model` pedido, e a divergência levanta `ProviderConfigurationError` antes da primeira linha; sem `model`, um aviso diz qual será usado. Um turno reroteado para outro modelo é interrompido e a linha falha.
- O pacote declara suporte a Python 3.14, e o CI roda os testes também nessa versão (#231).

### Alterado

- Com `parallel_requests` > 1, as linhas são despachadas em fila contínua: até `parallel_requests` linhas ficam em voo, e uma nova entra assim que outra termina. Antes, cada onda de linhas esperava a mais lenta terminar para a seguinte começar, e o worker que acabava cedo ficava parado. A redução pela metade a cada rate limit passa a baixar o teto de linhas em voo, sem cancelar as que já estão em voo, e segue valendo até o fim da execução, como antes.
- Em `is_complex_type` e `resolve_forward_refs`, `typing.Union` e `X | Y` passam por um ramo só, escolhido pela origem da anotação, porque no 3.14 os dois são o mesmo objeto e o ramo de cada forma deixava código sem rodar. `resolve_forward_refs` remonta as duas formas com `typing.Union[...]`: antes do 3.14, um `X | Y` com referência resolvida volta na forma de `typing.Union`, igual a ele por `==` (#231).
- O extra `codex` fixa `openai-codex==0.163.0`; as flags `api_key_cyber_access_programs`, `cli_daybreak` e `code_mode_interrupt` ficam desligadas no provider (#270).

- O extra `dev` passa a trazer o `pytest-repeat`, e o CI roda a suíte duas vezes no mesmo processo (`pytest --count=2 --repeat-scope=session`), o que reprova o teste que só passa na primeira execução (#201).
- `read_df(path, modelo)`, chamada sem argumentos de leitura extras, relê como `int` a célula que o CSV ou o parquet guarda como inteiro num campo `int` ou `Optional[int]`, e a coluna sai como `object`, com `None` na célula vazia, em vez de float64 com `NaN`. É o que faz a retomada devolver exato o inteiro acima de 2**53. A coluna que o leitor já devolve em tipo inteiro, como a `Int64` anulável do parquet, e a coluna só com células vazias ficam como lidas. Com argumento de leitura extra, que vai ao pandas, a função devolve o que ele lê, e o XLSX fica de fora, porque guarda número como double (#223).
- O resumo de uso do fim da execução deixa de sair por `print` no stdout e passa ao logger `dataframeit.stats`, em nível INFO. Sem logging configurado, ele sai em stderr; `logging.getLogger("dataframeit.stats").setLevel(logging.WARNING)` o silencia e mantém as colunas de tokens. O aviso de workers reduzidos por rate limit vira `UserWarning`, e a mensagem amigável do provider passa a fazer parte do aviso de falha da linha, ambos filtráveis por `warnings` e `pytest -W` (#181).
- `dataframeit()` declara o tipo de retorno pelo tipo da entrada: DataFrame ou Series do polars devolve DataFrame do polars, e DataFrame ou Series do pandas, `list` e `dict` devolvem DataFrame do pandas. Antes, o retorno era a união dos dois, e o checker de tipos recusava `resultado["coluna"].tolist()` (#155).
- O piso do pydantic sobe para 2.12. Na 2.11, uma `default_factory` que lê outro campo é chamada com os dados validados vazios quando esse campo falhou, e a retomada levantava o `KeyError` dela em vez do erro de campo incompatível (#154).
- Os extras `polars` e `all` passam a exigir `polars>=1.10`. No 0.20, há versões que convertem a coluna de texto com um valor de outro tipo e põem nulo no lugar dele, sem erro, e versões em que o `to_pandas()` de uma coluna `Object`, pelo qual ela volta na retomada, devolve bytes no lugar dos objetos ou falha. Do 1.6 ao 1.9, a saída polars de um campo `Any` que recebeu só dicts vazios cai com `ShapeError` depois de todas as chamadas (#218, #241).
- O extra `codex` passa a exigir `openai-codex==0.162.1`, que traz o runtime 0.162.1 (#226). Com o runtime 0.137, que o SDK anterior fixava, a conta do ChatGPT recebia HTTP 400 nos modelos lançados depois dele, como o `gpt-6-luna`, embora o catálogo da conta os anunciasse. O `effort` é conferido contra os valores que o SDK declara, agora com `max` e `ultra`, porque o enum do SDK novo aceita qualquer texto. O turno que falha com um código de erro que o SDK não declara falha só a linha, sem nova tentativa. Com `CODEX_API_KEY` no ambiente, o catálogo de modelos passa a ser lido com a credencial do `auth.json`, a mesma dos turnos, e não com essa chave, que o app-server ignora.
- No provider `codex`, os códigos de erro `rateLimitExceeded` e `flexUnavailable` do runtime levantam `ProviderOverloadedError`, e a linha é repetida como no HTTP 429.
- No provider `codex`, as flags `sleep_tool` e `view_image` do runtime ficam desligadas, e as ferramentas de pausa e de leitura de imagem local deixam de ser oferecidas ao modelo.
- No provider `codex`, a flag `unbounded_connection_retries` do runtime fica desligada. Ligada, uma falha de conexão deixa o turno esperando para sempre; desligada, a linha falha com `ProviderTransientError` e é repetida conforme `max_retries`.
- No provider `codex`, o runtime roda com `agents.enabled=false`, e as ferramentas de sub-agentes do namespace `collaboration`, como `spawn_agent` e `send_message`, que o catálogo liga para modelos como o `gpt-6-luna`, deixam de ser oferecidas ao modelo.
- No provider `codex`, o request chega ao modelo sem nenhuma ferramenta, o que corta cerca de 1.850 tokens de entrada por linha no `gpt-6-luna`. `request_user_input` sai por config; `exec`, `wait`, `request_user_input_async`, `clock` e `apply_patch` saem do catálogo de modelos da conta, que o provider lê na abertura e passa ao runtime com `model_catalog_json` (#162).
- Nos providers do LangChain, a tentativa seguinte a uma resposta recusada pela validação do modelo Pydantic, inclusive por validadores próprios, leva ao modelo a resposta e os erros por campo, com pedido de correção; antes, repetia o mesmo prompt. A recusa levanta `ProviderRejectedOutputError`, transitória por classe, e deixa de ser lida como erro HTTP quando o texto analisado tem números como 404. Os tokens das tentativas recusadas são somados ao uso da linha quando a resposta os traz, o que inclui a OpenAI, pela resposta HTTP anexada ao erro do SDK (#144).
- A docstring de `dataframeit()` e a mensagem de rate limit descrevem `rate_limit_delay` como a pausa de cada worker depois de cada linha processada com sucesso, com teto de `parallel_requests * 60 / rate_limit_delay` linhas por minuto.
- No provider `codex`, chaves do schema que não restringem o valor, como anotações próprias em `json_schema_extra` e os metadados `examples` e `deprecated`, são descartadas com aviso em vez de recusadas; keywords de validação fora do subconjunto do Structured Outputs continuam recusadas.
- No provider `codex`, o preflight recusa `format` fora da lista do Structured Outputs, como os de `bytes`, `Path`, `AnyUrl` e `UUID4`, e `pattern` com lookahead, lookbehind ou referência a grupo, como o que o pydantic anterior à 2.14 gera para `Decimal`. Antes, o schema passava, e cada linha falhava no turno: a API recusa esse `format` e o lookaround com HTTP 400 `invalid_json_schema`, e com referência a grupo o turno gera texto até `max_output_tokens` (#158).
- Com `resume=True` e `checkpoint_path` existente, rodar a mesma chamada de novo continua do checkpoint, sem carregá-lo à mão; antes, a execução recomeçava e o sobrescrevia. `<checkpoint>.dataframeit.json`, gravado a cada gravação do checkpoint, guarda a assinatura da execução, o hash do texto de cada linha e o do próprio arquivo, e o checkpoint só é retomado quando os três batem; com outra configuração ou entrada, a execução avisa e recomeça como antes. Das linhas concluídas, com sucesso ou com erro, vêm também os campos do modelo, sobre o que a entrada trazia neles.
- No provider `codex`, o app-server roda com `HOME` no diretório isolado da execução, e as skills do usuário, como as de `~/.agents/skills`, deixam de ser carregadas.
- Com `search_per_field=True`, o modo por grupo também devolve `search_provider` no uso, e o modo por campo devolve os campos na ordem do modelo, como o modo por grupo. O campo pulado por condição é registrado uma vez no log, e não duas.
- O Python mínimo sobe para 3.11. O 3.10 saiu de suporte em outubro de 2026, e as versões atuais do numpy e do pandas já não o aceitam (#229).
- O piso do pyarrow nos extras `polars` e `all` sobe para 10.0.1, a primeira versão com wheel para Python 3.11 (#229).
- Na saída polars, um campo `dict` sai como texto JSON, como no checkpoint. Antes, saía `Struct` com as chaves de todas as linhas: a chave ausente ficava igual à chave com valor nulo, e o dict vazio deixava de ser vazio. A coluna de um campo sem nenhum valor, toda nula ou só com listas vazias, sai com o tipo da anotação quando ela é `str`, `int`, `float`, `bool`, `date`, `datetime` ou lista de um deles. Antes, o tipo vinha só das respostas: um `Optional[int]` todo nulo saía `String`, e um `list[str]` só com listas vazias, `List(Null)` (#219).

### Corrigido

- Quando a entrada chega sem a coluna de status e não há checkpoint a reler, uma coluna da entrada com algum valor e com o nome de um campo do modelo levanta `ValueError` antes de qualquer chamada, com `resume=True` e com `resume=False`, a menos que esteja em `reprocess_columns`; na retomada de um `checkpoint_path`, a coluna acrescentada à entrada depois da interrupção não é checada. O `Decimal` NaN e o NaN de campo float em coluna object contam como valores, como na retomada. Antes, com `resume=True`, a resposta do LLM a sobrescrevia sem aviso quando só parte das colunas do modelo existia, e só avisava quando existiam todas; com `resume=False`, avisava e devolvia a entrada. `status_column` igual à coluna de texto, a um campo do modelo, ao nome de outra coluna de controle do dataframeit ou a uma coluna existente com valores que não são de status também levanta `ValueError`. Antes, a coluna sumia da saída com a resposta que ocupava, ou a execução terminava sem chamar o LLM e a coluna de texto ou a do usuário sumia (#197).
- As anotações dos geradores usados como context managers são verificadas pelo ty 0.0.88 (#270).

- A criação das colunas de saída usa a opção `mode.chained_assignment` só no pandas 2, preservando entradas fatiadas sem emitir aviso de depreciação no pandas 3.1 (#269).

- A documentação, os READMEs e os notebooks de exemplo deixam de descrever o que o código não faz. A referência da API traz `perguntas` na 4ª posição da assinatura, diz que a entrada `pd.DataFrame` é modificada no lugar e que a saída é sempre DataFrame; `basic-usage.md` e os notebooks 04 e 08 deixam de passar a uma segunda chamada o DataFrame que a primeira já processou, e o trace completo da busca web usa um modelo sem configuração por campo. A página de exceções lista `ProviderAbortError` e `ProviderUsageLimitError`, que param a execução; a FAQ mostra a mensagem de dependência que o código emite e diz que só o `codex` recusa `api_key`; o extra `all`, as colunas de tokens da saída e o resumo de throughput, que só sai com `parallel_requests > 1`, aparecem como são (#200).
- O aviso de rate limit da busca estima a taxa com 1 s por linha mais o `rate_limit_delay` e uma busca por execução do agente. É um piso: o agente que faz mais buscas por linha, até `max_search_calls`, pode passar do limite sem aviso. Aumentar a pausa não faz mais o aviso aparecer, e a recomendação nunca passa do `parallel_requests` atual, também com `search_per_field=True`. Antes, com pausa, a conta supunha tempo de processamento zero: com Exa, três campos e um worker, `rate_limit_delay=0.5` avisava ~360 req/min e recomendava `parallel_requests=3` (#199).
- A entrada `dict` com chaves-tupla deixa de levantar `AssertionError` no `dataframeit()`, antes da primeira chamada, e na interrupção de um reprocessamento (#199).
- No provider `claude_code`, com um event loop ativo no thread principal, como no Jupyter, o Ctrl-C chega ao chamador na hora, sem esperar a query em curso terminar (#199).
- Com `max_retries` a partir de 1026 e `base_delay` float, o backoff deixa de levantar `OverflowError` na tentativa 1025, e a linha falha com o erro original (#199).
- O aviso de falha da linha escolhe a caixa de um `ProviderError` pela classe e pelo status HTTP que o provider informa, guardado em `status_code`, e não pelo texto. Antes, um "API key", um "art. 403", um "network" ou um "timeout" no `result` do `claude_code` ou na mensagem de turno do `codex` dava a caixa de autenticação, de permissão, de conexão ou de timeout, que mandam trocar a chave, rever o plano, conferir a internet ou aumentar `max_retries`. A classe decide primeiro: a sobrecarga (`ProviderOverloadedError`) dá a caixa de limite de requisições, com ou sem status, e a resposta recusada (`ProviderRejectedOutputError`), a da validação. Nas demais classes, 401 dá a de autenticação, 403 a de permissão e 408 ou 504, num erro transitório, a de timeout; no `codex`, o código `unauthorized` conta como 401. Sem classe nem status que decidam, sai a caixa genérica, com o tipo e o começo da mensagem, e a de conexão, que só o texto decide, não sai mais para o `ProviderError`. Os demais erros, como os do LangChain, seguem lidos pelo texto como antes: sem status, em todas as caixas; com status, na de chave inválida e nas do Tavily e do Exa (#254).
- No provider `codex`, o recolhimento do app-server que não saiu espera o processo morto sem prazo, e o Ctrl+C que chega nessa espera é levantado depois que o processo foi recolhido, os pipes de stdout e stderr fecharam e os pedidos que esperavam resposta foram acordados. Antes, o quarto Ctrl+C no fechamento de um app-server que não reage ao SIGTERM deixava o processo zumbi e os pipes abertos (#253).
- No provider `codex`, a thread daemon que pede o `thread/unsubscribe` da linha abortada pela morte do app-server deixa de ficar presa para sempre. O processo morto ainda aceita a escrita enquanto termina, e o SDK registra a espera da resposta depois de o leitor do stdout ter acordado os pedidos em voo; agora o fechamento do cliente acorda essa espera (#250).
- `read_df` com argumento de leitura que faz o pandas devolver outra coisa que um DataFrame, como `chunksize`, `iterator`, `sheet_name=None` ou lista e o `chunksize` do JSON em linhas, devolve o `TextFileReader`, o `JsonReader` ou o dict de abas como o pandas o devolve, sem normalizar, com ou sem modelo. Antes, sem modelo, ou com modelo que tem campo de lista, dict ou modelo aninhado, e com `normalize=True`, o padrão, a função levantava `AttributeError` ao procurar as colunas no objeto do leitor (#249).
- Com entrada polars, a saída deixa de trocar sem aviso o valor que o modelo devolveu quando o pyarrow tipa a coluna pelo primeiro valor e converte os demais. Sai `Object`, com o aviso das colunas que o polars não reúne, o campo cuja anotação junta `timedelta` a outro tipo, `float` a `bool` ou `bytes` a `str`, e a coluna cujos valores trazem essas combinações ou as de data e hora, também em `Any`, onde o escalar do numpy, como o `np.int64`, conta como o número do Python, nos itens de lista e no campo de modelo aninhado, além da coluna cujos dicts do mesmo campo têm chaves diferentes e da que traz `datetime` com e sem fuso, `time` com fuso ou `Decimal` NaN. O NaN de campo `float` sai NaN. Antes, em `Union[timedelta, int]` o 5 virava cinco microssegundos, em `Union[float, bool]` o `True` virava `1.0`, o `datetime` sem fuso ganhava o fuso do primeiro, o `time` perdia o fuso, o `Decimal` NaN e o NaN de campo `float` viravam nulo, `{}` ao lado de `{"a": 1}` virava `{"a": None}`, e em `Union[NewType sobre date, int]` o 5 virava `1970-01-06`. Com `resume=True`, a coluna do campo cuja anotação é `int`, um `Literal` com inteiro ou uma união com um deles é lida com os inteiros do polars, e o inteiro acima de 2**53 deixa de voltar arredondado em `Float64`. O `Decimal` NaN e o NaN de campo `float` de uma linha processada deixam de ser acusados como incompatíveis na retomada, também com entrada pandas, e o reprocessamento campo a campo os repassa às condições como valor; na coluna float64, como a do checkpoint relido, o NaN não se distingue do nulo e segue contando como ausente. O `NewType` e o apelido `type`, inclusive o apelido genérico usado com argumentos, como `Opcional[int]` de `type Opcional[T] = T | None`, passam a contar pelo tipo a que se referem na saída polars, na retomada e em `read_df`: o `NewType` sobre `dict` sai em texto JSON, como o campo `dict`, e é retomado, e o campo `NewType` sobre `str` ou sobre `int` é relido do CSV como texto cru ou como inteiro exato (#241).
- No provider `codex`, o fechamento recolhe o app-server que não saiu: o que não sai em 2 s depois do SIGTERM e o que segue vivo depois de um segundo Ctrl+C durante o fechamento são mortos e esperados, e stdout e stderr fecham. Antes, o processo ficava zumbi até o coletor de lixo, com `ResourceWarning`, ou seguia vivo com os pipes abertos depois do `KeyboardInterrupt`. A falha no `start` ou no `initialize` da abertura, e o Ctrl+C durante ela, também recolhem o app-server e fecham os pipes, que antes ficavam abertos e, no Ctrl+C, com o processo vivo (#244).
- O aviso de falha da linha mostra a caixa de limite de requisições para os mesmos erros que reduzem os workers. Antes, a caixa era escolhida pelo texto da exceção: um `ProviderError` com "429" na mensagem, como o turno do `codex` que falha sem código de erro e com "429 Too Many Requests" só no texto, ou o erro do `claude_code` cujo texto do modelo cita o art. 429 do CPC, recebia a caixa sem que os workers fossem reduzidos, e o `ProviderOverloadedError` sem "429" no texto, ou com uma chave de API ou um 403 nele, reduzia os workers e recebia outra caixa. O rate limit do Tavily e do Exa, que chega com o status só no texto, recebe a mesma caixa, com o nome do serviço de busca no lugar do provider de LLM; antes, recebia a caixa de limite de uso do plano, que manda esperar o mês seguinte. A caixa de timeout passa a sair só para o erro que `is_recoverable_error` classifica como recuperável, que o retry repete enquanto houver tentativas em `max_retries`, e manda aumentar `max_retries` em vez de afirmar que o sistema já tentou várias vezes, o que não vale com `max_retries=1` (#243).
- `read_df(path, modelo, **kwargs)` deixa de reler como texto cru os campos de texto quando recebe qualquer argumento de leitura extra, e o texto fica como o pandas o devolve. Antes, a releitura repassava os argumentos com só as colunas de texto: com `on_bad_lines="skip"` ou `"warn"`, a linha malformada deixava de sê-lo, voltava na releitura e, sem aviso, cada texto a partir dela ia para a linha de baixo e o último sumia; com `on_bad_lines` callable, `index_col`, `parse_dates`, `names` com índice implícito, `chunksize`, `iterator` ou `sheet_name=None`, a leitura levantava (#242).
- Com `checkpoint_path`, o campo `int` acima de 2**53 numa coluna que também tem célula vazia, como a das linhas pendentes ou a de um `Optional[int]` com `None`, é gravado exato no CSV e no parquet e volta exato da retomada, também com `strict=True`. Antes, a coluna passava por float64 na gravação e de novo na releitura, e `9007199254740993` voltava `9007199254740992.0`, ou era recusado com `strict=True`. No parquet, o inteiro acima de 2**63 - 1 ao lado de célula vazia e o inteiro acima de 2**53 ao lado de um float na mesma coluna não são gravados, e a gravação do checkpoint falha com aviso; antes, eram gravados arredondados, sem aviso, e no primeiro caso a retomada recusava a linha. O XLSX guarda o número como double, e o inteiro acima de 2**53 continua voltando arredondado dele; com `strict=True`, é recusado enquanto o arredondado cabe no `Int64` e aceito com o valor errado quando passa dele (#223).
- No provider `claude_code`, a chamada recusada pela API é classificada pelo status que o SDK informa no `ResultMessage` de erro, e não pelo texto da exceção que o SDK levanta logo depois. O 401 e o 400 deixam de ser repetidos até `max_retries`; o 429 e o 529 levantam `ProviderOverloadedError`, são repetidos e, quando a linha falha em todas as tentativas, reduzem os workers; o estouro de `max_turns` ou de `max_budget_usd` que chega pelo mesmo caminho falha a linha na primeira tentativa. O custo de cada tentativa recusada entra no custo da linha e no resumo. O status só chega no `ResultMessage` a partir do `claude-agent-sdk` 0.1.76; do 0.1.48 ao 0.1.75, o 401, o 400, o 429 e o 529 continuam transitórios e não reduzem os workers (#227).
- Uma resposta recusada pela validação do modelo Pydantic deixa de ser lida como erro de credencial quando o texto tem números como 401 ou 403, comuns em texto jurídico ("art. 401 da CLT", "Súmula 403"). No `claude_code`, a falha de parsing ou de validação levanta `ProviderRejectedOutputError` e ganha nova tentativa; antes, a linha falhava na primeira tentativa com a instrução de reconfigurar a autenticação. No `codex`, a mensagem amigável do `ProviderOutputError` é decidida pela classe, e não manda mais rodar `codex login`. Nos dois, a mensagem gravada em `_error_details` leva o caminho e a regra de cada erro, sem o valor recusado. `is_rate_limit_error` decide pela classe para todo `ProviderError` e para o `ProviderOutputError`, e entre eles só o `ProviderOverloadedError` conta como rate limit: um '429' no texto, como no resumo de um erro de validação do `codex` ou no campo `result` que o SDK do Claude Code devolve num erro, deixa de reduzir os workers. No `codex`, o erro JSON-RPC de código -32001, com que o app-server recusa o pedido quando a fila de entrada está cheia, levanta `ProviderOverloadedError`: a linha é repetida e, se falhar em todas as tentativas, reduz os workers, como no HTTP 429. Antes, esse erro falhava a linha na primeira tentativa, sem reduzir os workers. O erro do `codex`, de turno ou JSON-RPC, que traz '429 Too Many Requests' só no texto deixa de reduzir os workers, porque o rate limit é reconhecido pelo código do erro (#192).
- No provider `codex`, a saída do backend e o encerramento pelo prazo fecham também os pipes de stdout e stderr do app-server, que o `close` do SDK deixa abertos; antes, os dois ficavam para o coletor de lixo, com um `ResourceWarning: unclosed file` cada. Quando o app-server morre no meio de uma linha, `dataframeit()` deixa de levantar `BrokenPipeError` ao encerrá-lo e devolve as linhas já processadas, com a linha abortada e as seguintes sem status (#225).
- Com entrada polars e `resume=True`, a coluna de um campo `int` do modelo é lida com os inteiros e os nulos do polars. Antes, o nulo de uma linha pendente levava a coluna a float: o inteiro acima de 2**53 voltava arredondado, a coluna saía `Float64` e as respostas novas saíam em float. Uma coluna `UInt64` com valor acima do `Int64` deixa de ser acusada como incompatível e sai `Object`, como na execução sem interrupção. A coluna `Int64` sem nulo do campo `int` também chega em inteiros Python, e um modelo com `strict=True` deixa de acusar como incompatível a linha processada, que o `int64` do numpy não validava (#222).
- Com entrada polars, o campo cuja anotação junta `date`, `datetime` ou `time` a outro tipo, na união, na tupla ou na lista ou tupla de união, sai `Object`, com os valores que a entrada pandas devolveria e o aviso das colunas que o polars recusa. Antes, a coluna era tipada pelo primeiro valor, sem erro nem aviso: em `Union[datetime, date]` com a data primeiro, a hora sumia; em `Union[date, int]` e `tuple[date, int]`, o inteiro 5 virava `1970-01-06`; em `Union[time, int]`, o inteiro fora do intervalo de um dia virava nulo ou meia-noite, conforme a versão do polars; em `list[Union[date, datetime]]`, a hora sumia (#228).
- A retomada com `resume=True` de uma saída polars deixa de recusar a linha processada quando o campo do modelo guarda datas numa lista, como `list[date]`, `list[datetime]` com ou sem fuso, `tuple[date, date]` e o submodelo com `list[date]`, ou inteiros ao lado de nulo, como `list[Optional[int]]`. A conversão da entrada polars para pandas entregava essa lista como array do numpy, com a data em `datetime64`, o datetime com fuso passado a UTC sem o fuso e o inteiro ao lado de nulo em float, arredondado acima de `2**53`. As colunas `List`, `Array` e `Struct` passam a ser convertidas pelo `to_list` do polars, e a linha retomada sai com os valores e o tipo de coluna da execução sem interrupção, inclusive o fuso. Com isso, a coluna de texto do prompt que é `List` ou `Array`, ou `Struct` com lista dentro, chega ao modelo com a lista do Python, como `['a', 'b']` no lugar de `['a' 'b']`, e um checkpoint gravado antes com essa coluna de texto é descartado com aviso e a execução recomeça do zero (#224).
- Com entrada polars, um campo `Decimal` com valor acima de 38 dígitos deixa de derrubar a saída depois de todas as chamadas com o `PanicException` do polars, que não é `Exception`. A coluna sai `Object`, com aviso, como as demais que o polars recusa (#230).
- Com entrada polars, a saída deixa de falhar depois de todas as chamadas quando um campo do modelo guarda valores que o polars não reúne numa coluna tipada, como `Union[int, str]` com número numa linha e texto em outra, `tuple[int, str]`, `Any` com tipos diferentes, lista de `Enum`, inteiro acima de 2**53 ao lado de float, inteiro acima do `Int64` e `datetime` com fuso de deslocamento, como `+05:30`. A coluna sai com o tipo `Object`, com os objetos Python que a entrada pandas devolveria, e um aviso a nomeia. O campo `Enum` sai pelo valor do membro, inclusive na coluna `Object` (#218).
- A coluna `_trace` do agente único fica com as demais colunas de trace na ordem da saída, depois dos campos do modelo. Antes, numa entrada que já a trazia, os campos novos saíam depois dela.
- Com entrada polars, as colunas que a execução não escreve saem do DataFrame de entrada, com o tipo e os valores de lá. Antes, a saída inteira era convertida de volta do pandas: um `Int64` com nulo voltava `Float64`, o que arredondava ids acima de 2**53, um `Date` voltava `Datetime`, e `Enum`, `Decimal`, `Array`, `Null` e o inteiro dentro de `Struct` e `List` também trocavam de tipo; uma coluna `Object` também trocava de tipo ou, com objetos que o Arrow não infere, levantava `ArrowInvalid` depois das chamadas (#195).
- O operando de `in` e `not_in` em `condition` é conferido antes de processar, com `ValueError` quando não é lista, tupla, set ou frozenset. Antes, um operando que não era coleção, como `{"in": 5}`, levantava `TypeError` em cada linha depois das chamadas dos campos anteriores, e um texto, como `{"in": "pfj"}`, fazia busca de substring e aceitava `"pf"` (#182).
- A retomada de um modelo com `strict=True` deixa de acusar como incompatíveis os campos que o checkpoint grava em outra forma, como a data em texto, a tupla como lista, o `Enum` pelo valor e o inteiro que a coluna com linha pendente relê como float. A linha recusada pela validação configurada é validada de novo sem strict, e cada campo recusado só é aceito quando o valor validado diz o mesmo que o relido; `"3"` num campo `int` strict continua acusado. A volta ao tipo declarado passa a valer também para os campos escalares: data, `datetime`, `Enum`, `int` e `bool` relidos do CSV, do XLSX ou do parquet deixam de misturar o tipo da execução com o do arquivo. Os campos de `Enum` de valores em texto são relidos como texto do CSV e do XLSX, como os de `Literal`, e um valor como `"1"` deixa de voltar como número. O checkpoint grava o `Enum` escalar pelo valor; antes, o CSV e o XLSX gravavam `"Cor.AZUL"`, que a retomada recusava, e o parquet falhava ao gravar (#178).
- A forma de `condition` em `json_schema_extra` é conferida antes de processar, com `ValueError` que diz o que é aceito: dict com `field` em texto e um operador (`equals`, `not_equals`, `in`, `not_in` ou `exists`), ou callable. Antes, `condition={}` e os demais valores falsos, como `False` ou `[]`, rodavam o campo sem aviso, embora `evaluate_condition` trate o dict vazio como condição falsa; `field` que não era texto levantava `AttributeError`; e as demais formas eram avisadas a cada linha, com o campo pulado (#163).
- No provider `codex`, a falha de um hook de schema do modelo, como um `json_schema_extra` chamável que levanta `TypeError`, é relatada como falha na geração do JSON Schema, com a mensagem da exceção original; antes, a mensagem dizia que `questions` não era um modelo Pydantic v2 (#152).
- Com `search_groups`, a busca configurada em campo de modelo aninhado e em item de `List[Model]` era ignorada sem aviso. Ela passa a rodar como no modo por campo, também para o campo dentro de um grupo: o resultado da busca aninhada entra no prompt do grupo, e os itens da lista são buscados depois da resposta (#169).
- Fora de uma interrupção, toda linha que chega a rodar termina com status, no modo paralelo e no sequencial. Quando registrar a falha da linha também falha, como num erro cujo `__str__` levanta, a linha sai com `'error'` e o tipo e o texto da exceção em `_error_details`; antes, no paralelo ela ficava sem status e sumia da saída, e no sequencial a chamada inteira levantava. No paralelo, a falha ao despachar uma linha ao executor interrompe a execução como o `ProviderAbortError`: as linhas não despachadas ficam sem status para `resume=True`, o checkpoint é gravado e o aviso de interrupção é emitido (#153).
- O checkpoint `.parquet` grava os campos de lista, dict e tupla do modelo como JSON, como o CSV e o XLSX; as demais colunas continuam gravadas como vieram. Antes, um campo `dict` relido dele, pelo `read_df` ou pela retomada automática, voltava com as chaves de todas as linhas e os inteiros em float. Um checkpoint parquet gravado antes desta correção continua relendo os campos `dict` assim (#161).
- Um campo de lista ou dict anotado dentro de `Optional`, como `Annotated[list[str], Field(min_length=1)] | None`, é reconhecido como estrutura, e um campo de texto na mesma forma, como texto. Antes, `read_df` devolvia o JSON do checkpoint CSV ou XLSX como texto, e a retomada acusava o campo como incompatível.
- A retomada devolve os campos de estrutura como a execução os gravou, em qualquer formato: data, tupla e chave de dict que não é texto, que o JSON guarda como texto, lista e chave em texto, voltam pelo `model_dump` da linha validada, com os validadores, a configuração e os serializadores do modelo, e a coluna deixa de misturar as duas formas. A volta só muda o tipo: quando o valor validado diz outra coisa em JSON, como depois de um validador que muda o valor de novo ou num submodelo que grava pelo alias, fica o valor relido.
- A retomada com um campo `Field(exclude=True)` ausente da entrada levantava `KeyError`.
- Um `Enum` dentro de um campo de estrutura é gravado no checkpoint pelo valor; o texto (`"Cor.AZUL"`) não validava, e a retomada acusava o campo como incompatível.
- Um campo lista que chega como array do numpy, como numa saída polars passada de volta à execução, inclusive dentro de um modelo aninhado, é gravado no checkpoint como JSON. No CSV, o array era gravado pelo repr e relido com as strings concatenadas (`['t1' 'u']` voltava `['t1u']`); no parquet, ao lado das listas das linhas novas, impedia a gravação. Escalares do numpy dentro de estruturas, como `np.int64` e `np.bool_`, vão como número e bool, e não como texto.
- No provider `codex`, um modelo recursivo com metadado ao lado da referência (como `filho: "No" = Field(description=...)`) levanta o erro de schema recursivo com metadados; antes, levantava o de tipo `Any` não suportado, que apontava para outro problema.
- No provider `codex`, o turno que falha é classificado pelo erro tipado que chega no stream. A leitura do histórico com `includeTurns`, que thread efêmera recusa, fazia sobrecarga, HTTP 429 e 5xx virarem erro definitivo, sem nova tentativa nem redução de workers. A cota de uso esgotada levanta `ProviderUsageLimitError`, e o app-server encerrado levanta `ProviderAbortError`; os dois interrompem a execução em vez de falhar cada linha restante.
- No provider `codex`, quando uma tentativa seguinte dá certo, os tokens das que falharam são somados ao uso da linha, como nos providers do LangChain.
- No provider `codex`, cada tentativa tem prazo, da abertura da thread ao fim do turno, configurável em `model_kwargs={"timeout": segundos}`, com padrão de 600 e `None` para não ter prazo. O turno que não termina no prazo é interrompido e a tentativa falha com `ProviderTransientError`, em vez de prender a execução para sempre. Se o app-server não responde a `thread/start` ou `turn/start` no prazo, ele é encerrado e a execução para com `ProviderAbortError`, em vez de a linha esperar para sempre a resposta (#165). O `timeout` não entra na assinatura do checkpoint. O pedido de interrupção, inclusive o do turno reroteado, sai da thread da linha, e um app-server que não responde a ele não prende a linha. Os tokens já informados de um turno que estourou o prazo ou cujo stream terminou sem conclusão também são somados ao uso da linha (#157).
- No provider `codex`, a thread efêmera de cada tentativa e a da sonda do modelo recebem `thread/unsubscribe` ao terminar, e o app-server as descarrega depois de cerca de 60 s sem assinante. Antes, todas ficavam carregadas até o fim da execução, e a memória do app-server crescia com o número de linhas: 100 threads sem turno ocupavam cerca de 330 MB. O pedido sai da thread da linha, e um app-server que não responde a ele não prende a linha (#156).
- `depends_on` em tupla vale como a lista. Antes, era descartado em silêncio, e o aviso dizia que `depends_on` faltava. Forma diferente de texto, lista ou tupla de nomes levanta `ValueError` antes de processar (#150).
- A configuração por campo declarada num modelo dentro de `Annotated` aninhado, como `list[Annotated[list["X"], ...]]`, é encontrada pela varredura e pelas validações. A referência adiantada dentro do `Annotated` é resolvida, e os metadados ficam como estavam (#151).

### Segurança

- Os workflows do CI fixam cada action pelo SHA do commit, com a versão em comentário; o `actions/checkout` deixa de gravar o token do GitHub no `.git/config` do runner (`persist-credentials: false`); e, no `docs.yml`, só o job que publica tem `pages: write` e `id-token: write`. O job `quality` roda o `zizmor`, que reprova a action referida por tag ou branch no lugar do SHA, o `actions/checkout` sem a chave `persist-credentials` e a permissão de escrita no nível do workflow (#209).

## [0.10.0] - 2026-09-24

### Adicionado

- `max_search_calls` (padrão 10) limita as buscas por execução do agente, também por grupo (`search_groups`) e por campo (`json_schema_extra`). Ao atingi-lo, as buscas seguintes são bloqueadas e o agente responde com o que encontrou; antes, um modelo insistente buscava até o limite de recursão do grafo. As buscas bloqueadas não entram em `search_count` nem em `search_credits`. O limite de passos do agente acompanha o teto, para que um modelo que ignore o bloqueio pare em poucas chamadas.
- O provider `claude_code` repassa o custo informado pelo SDK (`total_cost_usd`), somado no resumo de estatísticas ao fim da execução. A soma inclui as tentativas re-tentadas e as linhas que falharam, e aparece mesmo sem contagem de tokens (#129).

### Alterado

- O provider `claude_code` roda sem settings de usuário e de projeto (`setting_sources=[]`) e com `--strict-mcp-config`: servidores MCP configurados pelo usuário e regras `permissions.allow` deixam de chegar à execução, que processa texto não confiável (#129).
- `condition` ou `depends_on` em campo de modelo aninhado ou de item de lista levanta `ValueError` antes de processar, em qualquer modo. Antes era ignorado em silêncio, e a versão callable quebrava o schema em todo provider (#129).
- Linhas com texto ausente (`None`, `NaN` ou só espaços) não vão mais ao LLM: ficam com status `'error'` e detalhe `'Texto ausente'`, e um aviso diz quantas são. Antes, a string `'nan'` era enviada e a resposta gravada como resultado.
- Índice com rótulos repetidos levanta `ValueError` antes do processamento; com ele, o resultado de uma linha era gravado em outra.
- Campo do modelo com o mesmo nome da coluna de texto levanta `ValueError`; a resposta sobrescrevia o texto de entrada.
- `perguntas` emite `DeprecationWarning` (#129).
- `batch_size` aceita inteiros numpy e rejeita `bool`, como `max_retries`.
- Rodar `dataframeit` sobre uma saída sem erros, que não tem coluna de status, emite aviso: todas as linhas seriam processadas de novo.
- O nome exibido, o plano de entrada e o limite aproximado por minuto de cada provedor de busca passam a ser propriedades abstratas de `SearchProvider` (`friendly_name`, `free_tier`, `requests_per_minute`), que uma subclasse registrada com `register_provider` precisa implementar. A validação de `search_provider`, a mensagem de API key ausente e o aviso de rate limit leem do registro de provedores em vez de listas próprias (#130).
- Nos overrides de busca por campo e por grupo, só a ausência (`None`) cai no valor global; `max_results=0` ou `search_depth=''` deixam de ser ignorados em silêncio (#130).
- Os pisos das dependências passam a ser versões que resolvem e passam na suíte: `langchain>=1.2.11`, `langchain-core>=1.2.10`, `pydantic>=2.11.0`, `pandas>=2.1.2`, `tqdm>=4.1.0`, `langchain-openai>=0.3.34`, `langchain-anthropic>=0.3.21`, `langchain-google-genai>=2.1.11`, `langchain-groq>=1.0.0`, `langchain-tavily>=0.2.12` e `langchain-exa>=1.0.0`. Os anteriores aceitavam versões que não instalam juntas ou que quebram no import. Um job de CI roda a suíte com cada dependência direta no piso, com e sem extras (#130).
- O modelo LangChain é criado uma vez por execução, e não por linha, em todos os modos; sem busca, o `with_structured_output` também, e com busca num agente só, o agente. Os modos por campo e por grupo montam o agente por chamada sobre o mesmo modelo. Um erro ao criar o modelo, como chave ausente, continua marcando a linha como erro (#130).

### Corrigido

- Um campo com `condition` callable falhava em toda linha no modo `search_per_field`: o `json_schema_extra` ia inteiro para o modelo montado por campo ou por grupo, e o Pydantic não serializa a função no JSON Schema. As chaves da biblioteca (`condition`, `depends_on`, `prompt`, `prompt_replace`, `prompt_append`, `search_depth`, `max_results`) saem do campo extraído no schema enviado ao LLM; num modelo aninhado, as chaves de busca dos campos internos seguem no `$defs` como metadado (#134).
- `search_depth` e `max_results` em `json_schema_extra` passam pela mesma validação dos globais e de `search_groups`, inclusive em campos aninhados (#129).
- Dependência circular entre campos, ou entre grupo e campo, levanta `ValueError` antes da primeira linha, em vez de marcar cada linha como erro com o prefixo de tentativas (#129).
- A mensagem de configuração por campo sem busca por campo nomeia o que falta: `use_search=True`, `search_per_field=True` ou os dois (#129).
- Um campo cujas dependências tinham a mesma raiz (`depends_on=['endereco.cidade', 'endereco.uf']`, ou o mesmo campo repetido) ficava fora da ordem de execução e voltava `None` com status `processed`.
- `prompt`/`prompt_replace` por campo sem `{texto}` descartava o texto da linha; agora o texto é anexado, como no prompt principal. O prompt de grupo com `{query}` deixa de receber o texto duas vezes.
- Configuração de busca num campo de modelo que está numa lista dentro de outra lista trocava a lista interna por um dicionário. Agora levanta `ValueError`, também para `list[list[Modelo]]`, que caía numa busca única em vez de uma por item.
- Um modelo que referencia outro (ou a si mesmo) com `list['Modelo']` tinha a configuração aninhada ignorada, porque o Pydantic deixa a string sem resolver nessa forma, e no modo por campo cada linha falhava com `PydanticUserError` ao montar o modelo da chamada.
- O mesmo modelo aninhado em dois campos (`residencial` e `comercial`) só tinha a configuração de busca aplicada no primeiro.
- `reprocess_columns` no modo por campo ou por grupo pedia ao agente todos os campos, com as buscas correspondentes, e descartava os que não foram pedidos. Agora só as colunas escolhidas são pedidas, e as condições usam os valores já gravados na linha.
- Uma falha ao gravar o checkpoint (disco cheio, arquivo aberto no Excel, coluna que o parquet não serializa) marcava como `'error'` a linha que tinha acabado de ser processada, com a mensagem de tentativas esgotadas, e podia interromper a execução. Agora vira aviso, o status da linha fica intacto, e a gravação seguinte, ou a final, grava o estado completo (#129).
- Retomar de checkpoint `.csv` ou `.xlsx` acusava como incompatíveis campos de lista, dict ou modelo aninhado, porque essas células eram gravadas como repr Python. Agora vão como JSON, e `read_df` também lê o repr dos arquivos antigos. Com o modelo, `read_df(caminho, Modelo)` lê como texto cru os campos de texto: `"2023"` deixa de voltar como número, e `"N/A"` ou `"NA"` deixam de virar ausência.
- Na retomada, um campo com `condition` que ficou `None` porque a condição, avaliada com os valores da linha, era falsa deixa de ser acusado como incompatível quando o tipo declarado é obrigatório. Com a condição verdadeira, o valor ausente continua acusado, e um valor presente continua validado com as restrições do campo.
- Com `status_column` personalizado, a coluna de status e `_error_details` ficavam na saída de uma execução sem erros e fora do fim da tabela (#129).
- Um DataFrame com coluna de nome não textual (`pd.DataFrame(textos)`, cuja coluna é `0`) falhava depois de todas as chamadas, ao reordenar as colunas.
- `_error_details` de uma execução anterior ficava na linha depois que ela passava a `'processed'`. Quando uma linha já processada falha em `reprocess_columns`, inclusive por texto ausente, o detalhe diz que os valores anteriores foram mantidos.
- Uma coluna do modelo, de status ou de `_error_details` que já existia como `float` (toda vazia, lida de CSV) ou `int` fazia falhar a gravação de lista, texto ou valor padrão do modelo na retomada, às vezes depois da chamada paga.
- As estatísticas de busca saíam rotuladas "TAVILY" também com `search_provider='exa'` (#129).
- A mensagem de erro de API key do Mistral pedia `MISTRALAI_API_KEY`; o `langchain-mistralai` lê `MISTRAL_API_KEY` (#129).
- A caixa de erro amigável passa a ser escolhida pelo status HTTP estruturado, quando existe; sem ele, códigos só contam como número isolado. "4015 tokens" deixava de ser rate limit mas ainda mostrava a caixa de autenticação, e um erro de busca com "api_key" caía na caixa genérica em vez da do Tavily ou do Exa. "exa" só conta como palavra, com `_` como separador, para que `EXA_API_KEY` escolha a caixa do Exa. Chave inválida escolhe a caixa de autenticação pelo texto também com status: o Google a devolve como 400 `INVALID_ARGUMENT` (#129).
- Com `langchain-core` >= 1.6, o `is_retryable` do `ModelError` decide o retry, e `ModelRateLimitError` reduz os workers. Um 400 sem status estruturado deixa de ser re-tentado quando o provider levanta `ModelInvalidRequestError`, e `ContextOverflowError` deixa de ser re-tentado em qualquer versão (#129).
- No provider `claude_code`, um `ResultMessage` com `is_error` virava "resposta vazia" e entrava em retry. Estouro de `max_budget_usd` ou de `max_turns` agora é erro definitivo; o status da API decide entre sobrecarga, falha transitória e falha definitiva (#129).
- Com `search_provider='exa'`, `max_results` e o corte de 1000 caracteres por resultado eram ignorados: o `ExaSearchResults` aceita esses argumentos no construtor sem usá-los, e o modelo escolhia quantos resultados pedir. A ferramenta passa a expor só a consulta e fixa os dois limites na chamada.
- Erros do Tavily e do Exa (quota, chave inválida, falha de rede) voltavam como texto para o modelo, que respondia sem evidência; a linha saía `'processed'` e a busca era contada como crédito. Agora o erro sobe até o retry e a classificação de erros. "Sem resultados" e erro de argumento do Tavily (400, 422, como consulta longa demais) continuam sendo mensagem ao modelo, que pode tentar outra consulta.
- O trace contava `search_queries` por substring "search" no nome da ferramenta, o que incluía o structured output de modelos como `ResearchResult`; agora compara com o nome da ferramenta de busca, e `total_tool_calls` conta todas as chamadas (#129).
- `get_nested_pydantic_models` devolvia o mesmo modelo duas vezes para anotações `Model | None` (#130).
- `search_groups` com `search_depth=''` passava pela validação e chegava ao provedor de busca (#130).

### Removido

- O import de `dataframeit` deixa de mudar o nível dos loggers `langchain_google_genai`, `langchain_core` e `httpx`, que sobrescrevia a configuração de logging do usuário (#130).

## [0.9.0] - 2026-09-23

### Alterado

- O provider padrão passa a ser `openai`, e `model` passa a ter default `None`: sem `model`, cada provider usa o próprio modelo padrão (`DEFAULT_MODELS`: `gpt-6-luna` na OpenAI, `gemini-3.8-flash` no Google, `claude-sonnet-5` na Anthropic, `openai/gpt-oss-120b` na Groq). Providers fora dessa tabela exigem `model`, e `codex` e `claude_code` deixam a escolha ao runtime (#121).
- Documentação, READMEs e notebooks de exemplo atualizados para os modelos atuais; saem modelos desligados, restritos ou deprecados (`gemini-3-flash-preview`, Gemini 2.5, Llama, `qwen3-32b` e `groq/compound` na Groq, `claude-3-5-sonnet`). O identificador do Mistral na documentação passa a ser `mistralai`, o que o LangChain aceita (#121).
- `condition` ou `depends_on` sem `use_search=True` e `search_per_field=True` levanta `ValueError`, em vez de ser ignorado em silêncio (#125).

### Removido

- `SearchProvider.get_tool_name_pattern()`, que só servia à contagem de buscas (#125).

### Corrigido

- A escolha das linhas a processar passa a depender só do status de cada linha. Com índice fora de ordem (depois de `sort_values`, `sample` ou filtro), índice textual ou entrada dict, linhas pendentes eram puladas em silêncio e voltavam com resultado vazio. Com `resume=True`, linhas com status `'error'` ficam como estão em todos os casos, como a documentação descreve (#122).
- `dataframeit()` rejeita `max_retries` que não seja inteiro >= 1 com `ValueError`; com `max_retries=0` nenhuma chamada era feita e cada linha terminava com um `TypeError` registrado como erro (#122).
- A classificação de erros para retry usa primeiro o status HTTP estruturado da exceção ou da sua causa (`status_code`, `code`, `http_status`, `response.status_code`): 408, 409, 429 e 5xx são recuperáveis, os demais 4xx não, inclusive o 499 (requisição cancelada). Um 400 ou 422 fora de `BadRequestError` deixa de ser re-tentado até `max_retries`, e um 429 declarado só no status passa a reduzir os workers no modo paralelo (#123).
- Códigos numéricos na mensagem de erro (`401`, `429`, `503` etc.) só contam como número isolado, e uma mensagem como "4015 tokens" deixa de ser lida como erro de autenticação ou rate limit (#123).
- O provider `claude_code` funciona com event loop já ativo, como no Jupyter, em vez de falhar com `RuntimeError` a cada tentativa (#124).
- O provider `claude_code` registra os tokens informados pelo SDK, com leitura de cache, em vez de zeros fixos, e grava nulo quando o SDK não informa uso (#124).
- `search_count` e `search_credits` contam só as chamadas da ferramenta de busca passada ao agente; antes, qualquer tool call com "search" no nome contava, inclusive a de structured output dos modelos `NestedSearch_*`, `ItemSearch_*` ou de modelos do usuário como `ResearchResult` (#125).
- `search_groups` aplica `condition`: campo com condição falsa fica `None` e não é pedido ao agente, e grupos e campos isolados rodam na ordem das dependências (#125).
- A ordem de execução entre campos independentes segue a do modelo, e os campos de um grupo seguem a ordem de `search_groups`; antes, ambas variavam de um processo para outro (#125).
- Com `parallel_requests > 1` e `batch_size`, duas gravações de checkpoint simultâneas disputavam o mesmo arquivo temporário, e o `FileNotFoundError` resultante regravava como `'error'` uma linha já processada. As gravações passam a ser serializadas, e um snapshot mais antigo que chegue depois de um mais novo é descartado (#128).

### Segurança

- O provider `claude_code` deixa de expor ferramentas ao modelo: as opções usam `tools=[]` e `permission_mode="default"` em vez de `bypassPermissions`, porque o texto das linhas é conteúdo não confiável (#124).

## [0.8.1] - 2026-09-23

### Alterado

- Os providers LangChain deixam de receber `temperature=0` por padrão; o cliente só recebe o que vier em `model_kwargs`. Vários modelos atuais (Claude Sonnet 5 e Opus 4.7+, OpenAI GPT-6 com raciocínio e série o, Gemini 3.6+) rejeitavam a chamada com erro 400 (#116).

### Removido

- Fallback de import de `init_chat_model` a partir de `langchain_core.chat_models`, módulo que não existe no `langchain-core` 1.x; o import vem de `langchain.chat_models`, dependência obrigatória (#117).

### Corrigido

- O `description` dos metadados do pacote passa a ser em inglês, e o link `Documentation` e os links de documentação do `README.md` apontam para a versão em inglês do site, como o `README.md` exibido no PyPI (#119).

## [0.8.0] - 2026-09-23

Primeira versão publicada no PyPI depois da 0.6.0; a 0.7.0 e a 0.7.1 não foram publicadas.

### Adicionado

- Provider experimental `codex` via SDK Python oficial, disponível exclusivamente no extra `dataframeit[codex]`, com runtime pinado, autenticação em arquivo, isolamento por execução e saída estruturada validada (#111).

### Corrigido

- O provider `codex` agora rejeita schemas incompatíveis com Structured Outputs durante o preflight, orienta o login file-backed com o comando correto, compartilha `auth.json` sem depender de symlink privilegiado no Windows e impede que duas execuções do DataFrameIt atualizem a mesma credencial concorrentemente (#111).
- Checkpoints validam as linhas processadas contra o modelo Pydantic atual e exigem `reprocess_columns` somente para campos incompatíveis, evitando resultados marcados como concluídos com valores ausentes sem rejeitar campos opcionais ou com default (#111).
- A telemetria preserva tokens de leitura de cache informados por providers LangChain nos caminhos normal e com busca (#111).
- Falhas transitórias tipadas do Codex recebem retry sem serem confundidas com rate limit, e falhas de geração do JSON Schema são apresentadas como erro de configuração do provider (#111).
- A normalização automática de JSON reconhece tanto colunas `object` do pandas 2 quanto o dtype `str` do pandas 3 (#111).
- `dataframeit.__version__` é lido dos metadados do pacote e deixa de divergir do `pyproject.toml`, onde estava fixo em 0.6.0.
- O link `Changelog` dos metadados do pacote aponta para o `CHANGELOG.md`, e não para a página de releases.
- `call_langchain` em `llm.py` agora aceita `usage_metadata` tanto como dict quanto como objeto, alinhando com o tratamento já feito em `agent._extract_usage`. Antes, providers que devolvessem `usage_metadata` como objeto causavam `AttributeError` (#107).

### Alterado

- O README passa a ser em inglês, com versões em português (`README.pt-BR.md`) e espanhol (`README.es.md`) (#112, #113).
- O CI valida Python 3.10 e 3.13 nos ambientes base e Codex, inicia o runtime empacotado e exercita o lifecycle e a exclusão multiprocesso da autenticação no Windows, além de fazer build da documentação em pull requests; o extra declara o runtime pré-release como limite inferior para permitir resolução limpa pelo `uv`, enquanto o SDK conserva o pin exato (#111).
- A telemetria usa as mesmas quatro colunas de tokens em todos os providers, incluindo `_cached_input_tokens`, mesmo quando a métrica permanece nula ou zero (#111).
- Leitura de `usage_metadata` extraída para helper `_parse_usage_metadata` em `llm.py` e reaproveitada por `agent._extract_usage`, eliminando divergência futura entre os dois caminhos (#107).

## [0.7.1] - 2026-05-01

### Adicionado

- Receitas para usar LLMs hospedados em São Paulo via Vertex AI (`southamerica-east1`), AWS Bedrock (`sa-east-1`) e Azure OpenAI (Brazil South) em `docs/guides/providers.md` (#102).

### Corrigido

- Inferência de pacote/env var em `errors.py` para `google_vertexai`, `bedrock`, `bedrock_converse` e `azure_openai`: mensagens de erro agora indicam o pacote correto (`langchain-aws`, `langchain-google-vertexai`, `langchain-openai`) e, para providers com auth via SDK, orientam configuração de credenciais em vez de uma API key inexistente (#102).

## [0.7.0] - 2026-04-30

### Alterado

- `depends_on` é derivado automaticamente de `condition` quando esta é um dict; só precisa ser declarado explicitamente para `condition` callable (#103).

### Removido

- `depends_on` sem `condition` deixa de afetar a ordem de execução (apenas emite warning) (#103).

## [0.6.0] - 2026-04-20

### Removido

- Colunas `_total_tokens` e `_search_count` do DataFrame de saída (#69). Totais continuam no summary de console; `_search_count` segue interno para o cálculo de `_search_credits`.

### Adicionado

- Inferência automática de `text_column` em DataFrames quando `None` (#93): tenta `texto`, `text`, `decisao`, `content`, `content_text` em ordem; DataFrames de 1 coluna usam-na direto.
- Coluna `_reasoning_tokens` para modelos com reasoning (GPT-5, o-series, Claude thinking) (#65). Extraída de `usage_metadata.output_token_details["reasoning"]`; aparece no summary como sub-bullet do Output.
- **Suporte opcional a Groq** (#94): novo provider disponível via `pip install dataframeit[groq]`. Use com `provider='groq'` e modelos como `llama-3.3-70b-versatile` ou `llama-3.1-8b-instant`. Requer `GROQ_API_KEY`.
- **Aviso de rate limit para busca web** (#67): `dataframeit(...)` agora emite um `UserWarning` quando a combinação de `use_search=True`, `parallel_requests` e `search_per_field` pode exceder o rate limit do provedor de busca (Tavily ou Exa). A mensagem inclui recomendações específicas de `parallel_requests` e `rate_limit_delay`. O aviso também dispara em execuções sequenciais quando o total de queries estimadas (`linhas × campos`) ultrapassa 100.
- **Checkpoint periódico em execuções longas** (#92): novos parâmetros `batch_size` e `checkpoint_path` em `dataframeit()`. Salva o DataFrame a cada N linhas processadas (escrita atômica via `.tmp` + rename) e um save final cobre a cauda quando o total não é múltiplo de `batch_size`. Formatos: `.csv`, `.xlsx`, `.parquet` — dependências (`openpyxl`, `pyarrow`) são validadas antes do processamento iniciar. Combinado com `resume=True`, permite retomar execuções longas após kill/crash sem perder progresso.
- Novo extra `excel` com `openpyxl` (`pip install dataframeit[excel]`), também incluído em `all`. Necessário para `checkpoint_path="*.xlsx"` e `read_df()` sobre arquivos Excel.
- Documentação de rate limits e processamento paralelo em `docs/guides/web-search.md` e `docs/en/guides/web-search.md`, com tabelas de configurações recomendadas por provedor.

### Corrigido

- Filtrar `UserWarning: Field name X shadows ...` do `langchain_tavily` no import do provider (#74). Filtro específico ao módulo upstream.
- `pyarrow` adicionado como dependência dos extras `polars` e `all`. Versões recentes de polars requerem pyarrow para `polars.DataFrame.to_pandas()`; sem isso, passar um polars DataFrame para `dataframeit()` levantava `ModuleNotFoundError`.

## [0.5.4] - 2026-04-13

### Adicionado

- **`provider='claude_code'` - Suporte ao Claude Code SDK**: Novo provider que usa `claude-agent-sdk` para chamadas LLM via créditos do Claude Code ao invés de créditos de API.
  - Instale: `pip install dataframeit[claude-code]`
  - Modelos: `model='haiku'`, `model='sonnet'`, `model='opus'`
  - Parâmetros via `model_kwargs`: `effort`, `max_turns`, `max_budget_usd`
  - Não requer API key (usa credenciais do Claude Code)
  - Busca web (`use_search=True`) não suportada inicialmente

### Corrigido

- **Default `model='gemini-3.0-flash'` inválido**: o modelo não existe na API Google e retornava `404 models/gemini-3.0-flash is not found`. Default trocado para `gemini-3-flash-preview` (Gemini 3 Flash real, lançado em dezembro/2025).
- **IDs de modelos Claude na documentação usavam formato com ponto** (ex: `claude-sonnet-4.5`), que não é aceito pela API Anthropic. Corrigidos para o formato com hífen (`claude-sonnet-4-5`, `claude-opus-4-6`, `claude-haiku-4-5`).
- **Documentação de modelos Gemini desatualizada**: tabelas em `docs/` atualizadas para refletir modelos realmente disponíveis (`gemini-3-flash-preview` preview, `gemini-2.5-flash`/`gemini-2.5-pro` estáveis).

## [0.5.3] - 2026-01-19

### Adicionado

- **`search_groups` - Agrupamento de campos para busca compartilhada** (#77): Novo parâmetro que permite agrupar campos que compartilham contexto de busca, reduzindo chamadas de API redundantes.

  ```python
  result = dataframeit(
      df, MyModel, PROMPT,
      use_search=True,
      search_per_field=True,
      search_groups={
          "regulatory": {
              "fields": ["status_anvisa", "avaliacao_conitec", "existe_pcdt"],
              "prompt": "Search regulatory status: ANVISA, CONITEC, PCDT for {query}",
              "max_results": 5,
              "search_depth": "advanced",  # opcional
          }
      }
  )
  ```

  - **Redução de chamadas**: Campos em um grupo compartilham a mesma busca (1 busca para múltiplos campos)
  - **Prompts customizados**: Cada grupo pode ter seu próprio prompt com `{query}` placeholder
  - **Parâmetros por grupo**: `max_results` e `search_depth` configuráveis por grupo
  - **Traces por grupo**: Com `save_trace=True`, gera `_trace_{nome_grupo}` para grupos
  - **Validações**: Campos não podem estar em múltiplos grupos; campos em grupos não podem ter `json_schema_extra` de busca

## [0.5.2] - 2026-01-12

### Adicionado

- **`save_trace` - Salvar trace do raciocínio do agente** (#64): Novo parâmetro que salva o trace completo do agente em colunas do DataFrame, permitindo debug e auditoria.

  ```python
  result = dataframeit(
      df, Model, PROMPT,
      use_search=True,
      save_trace=True  # ou "full" ou "minimal"
  )

  # Acessar trace
  import json
  trace = json.loads(result['_trace'].iloc[0])
  print(trace['search_queries'])  # Queries realizadas
  print(trace['duration_seconds'])  # Tempo de execução
  ```

  - **Modos**: `True`/`"full"` (trace completo) ou `"minimal"` (apenas queries, sem conteúdo de busca)
  - **Colunas**: `_trace` (agente único) ou `_trace_{campo}` (per-field)
  - **Estrutura**: messages, search_queries, total_tool_calls, duration_seconds, model

- **Configuração per-field via `json_schema_extra`**: Permite configurar prompts e parâmetros de busca específicos para cada campo do modelo Pydantic quando usando `search_per_field=True`.

  ```python
  class MedicamentoInfo(BaseModel):
      # Campo com prompt customizado (substitui o prompt base)
      doenca_rara: str = Field(
          description="Classificação de doença rara",
          json_schema_extra={
              "prompt": "Busque em Orphanet (orpha.net). Analise: {texto}"
          }
      )

      # Campo com prompt adicional (append ao prompt base)
      avaliacao_conitec: str = Field(
          description="Avaliação da CONITEC",
          json_schema_extra={
              "prompt_append": "Busque APENAS no site da CONITEC."
          }
      )

      # Campo com parâmetros de busca customizados
      estudos_clinicos: str = Field(
          description="Estudos clínicos relevantes",
          json_schema_extra={
              "search_depth": "advanced",
              "max_results": 10
          }
      )
  ```

- **Opções de configuração per-field**:
  - `prompt` ou `prompt_replace`: Substitui completamente o prompt base
  - `prompt_append`: Adiciona texto ao prompt base
  - `search_depth`: Override de profundidade (`"basic"` ou `"advanced"`)
  - `max_results`: Override de número de resultados (1-20)

- **Validação**: Erro informativo quando `json_schema_extra` com configurações de prompt/busca é usado sem `search_per_field=True`

- **29 novos testes**: 19 para configuração per-field + 10 para save_trace

### Alterado

- Refatoração de `call_agent_per_field` para usar funções auxiliares modulares:
  - `_get_field_config()`: Extrai configurações do `json_schema_extra`
  - `_build_field_prompt()`: Constrói o prompt para cada campo
  - `_apply_field_overrides()`: Aplica overrides de configuração de busca

### Documentação

- Adicionada seção "Configuração Per-Field (Novo em v0.5.2)" no README
- Adicionada seção completa no guia de Busca Web (`docs/guides/web-search.md`)
- Criado CHANGELOG.md

## [0.5.1] - 2026-01-10

### Corrigido

- Corrigido bug onde `create_agent` recebia `model_provider` em vez do LLM inicializado
- Agente de busca agora usa corretamente o modelo LLM inicializado

## [0.5.0] - 2026-01-08

### Adicionado

- **Busca web via Tavily**: Integração com Tavily para enriquecer dados com informações da web
- Parâmetros `use_search`, `search_per_field`, `max_results`, `search_depth`
- Tracking de créditos de busca (`_search_credits`, `_search_count`)
- Documentação completa para busca web

---

[Unreleased]: https://github.com/bdcdo/dataframeit/compare/v0.11.1...HEAD
[0.11.1]: https://github.com/bdcdo/dataframeit/compare/v0.11.0...v0.11.1
[0.11.0]: https://github.com/bdcdo/dataframeit/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/bdcdo/dataframeit/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/bdcdo/dataframeit/compare/v0.8.1...v0.9.0
[0.8.1]: https://github.com/bdcdo/dataframeit/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/bdcdo/dataframeit/compare/e1fc5adc81d101f9446cc1f9d684ea86d9155fc5...v0.8.0
[0.7.1]: https://github.com/bdcdo/dataframeit/compare/63edb7dbcc84f7659f40c88057db734b633f2855...e1fc5adc81d101f9446cc1f9d684ea86d9155fc5
[0.7.0]: https://github.com/bdcdo/dataframeit/compare/v0.6.0...63edb7dbcc84f7659f40c88057db734b633f2855
[0.6.0]: https://github.com/bdcdo/dataframeit/compare/v0.5.4...v0.6.0
[0.5.4]: https://github.com/bdcdo/dataframeit/compare/51105282203d0e1cdbbc70a4649519ffa0d91bfe...v0.5.4
[0.5.3]: https://github.com/bdcdo/dataframeit/compare/9d6eeed1cfd1c817a1ad926eb1b4fd95a37f6d63...51105282203d0e1cdbbc70a4649519ffa0d91bfe
[0.5.2]: https://github.com/bdcdo/dataframeit/compare/v0.5.1...9d6eeed1cfd1c817a1ad926eb1b4fd95a37f6d63
[0.5.1]: https://github.com/bdcdo/dataframeit/compare/a4acf89cf80cfde2e93a6ca9874c9d32157647bd...v0.5.1
[0.5.0]: https://github.com/bdcdo/dataframeit/commit/a4acf89cf80cfde2e93a6ca9874c9d32157647bd
