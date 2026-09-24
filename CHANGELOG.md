# Changelog

Todas as mudanças notáveis deste projeto serão documentadas neste arquivo.

O formato é baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/),
e este projeto adere ao [Versionamento Semântico](https://semver.org/lang/pt-BR/).

## [Unreleased]

### Adicionado

- `max_search_calls` (padrão 10) limita as buscas por execução do agente, também por grupo (`search_groups`) e por campo (`json_schema_extra`). Ao atingi-lo, as buscas seguintes são bloqueadas e o agente responde com o que encontrou; antes, um modelo insistente buscava até o limite de recursão do grafo. As buscas bloqueadas não entram em `search_count` nem em `search_credits`. O limite de passos do agente acompanha o teto, para que um modelo que ignore o bloqueio pare em poucas chamadas. Exige `langchain>=1.0.4`.
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

## [0.5.3] - 2025-01-19

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

## [0.5.2] - 2025-01-12

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

## [0.5.1] - 2025-01-10

### Corrigido

- Corrigido bug onde `create_agent` recebia `model_provider` em vez do LLM inicializado
- Agente de busca agora usa corretamente o modelo LLM inicializado

## [0.5.0] - 2025-01-08

### Adicionado

- **Busca web via Tavily**: Integração com Tavily para enriquecer dados com informações da web
- Parâmetros `use_search`, `search_per_field`, `max_results`, `search_depth`
- Tracking de créditos de busca (`_search_credits`, `_search_count`)
- Documentação completa para busca web

---

[Unreleased]: https://github.com/bdcdo/dataframeit/compare/v0.9.0...HEAD
[0.9.0]: https://github.com/bdcdo/dataframeit/compare/v0.8.1...v0.9.0
[0.8.1]: https://github.com/bdcdo/dataframeit/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/bdcdo/dataframeit/compare/v0.6.0...v0.8.0
[0.6.0]: https://github.com/bdcdo/dataframeit/compare/v0.5.4...v0.6.0
[0.5.4]: https://github.com/bdcdo/dataframeit/compare/v0.5.3...v0.5.4
[0.5.3]: https://github.com/bdcdo/dataframeit/compare/v0.5.2...v0.5.3
[0.5.2]: https://github.com/bdcdo/dataframeit/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/bdcdo/dataframeit/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/bdcdo/dataframeit/releases/tag/v0.5.0
