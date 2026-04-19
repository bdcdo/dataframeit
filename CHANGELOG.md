# Changelog

Todas as mudanças notáveis deste projeto serão documentadas neste arquivo.

O formato é baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/),
e este projeto adere ao [Versionamento Semântico](https://semver.org/lang/pt-BR/).

## [0.6.0] - Unreleased

### Removido

- Colunas `_total_tokens` e `_search_count` do DataFrame de saída (#69). Totais continuam no summary de console; `_search_count` segue interno para o cálculo de `_search_credits`.

### Adicionado

- Coluna `_reasoning_tokens` para modelos com reasoning (GPT-5, o-series, Claude thinking) (#65). Extraída de `usage_metadata.output_token_details["reasoning"]`; aparece no summary quando > 0.
- **Suporte opcional a Groq** (#94): novo provider disponível via `pip install dataframeit[groq]`. Use com `provider='groq'` e modelos como `llama-3.3-70b-versatile` ou `llama-3.1-8b-instant`. Requer `GROQ_API_KEY`.
- **Aviso de rate limit para busca web** (#67): `dataframeit(...)` agora emite um `UserWarning` quando a combinação de `use_search=True`, `parallel_requests` e `search_per_field` pode exceder o rate limit do provedor de busca (Tavily ou Exa). A mensagem inclui recomendações específicas de `parallel_requests` e `rate_limit_delay`. O aviso também dispara em execuções sequenciais quando o total de queries estimadas (`linhas × campos`) ultrapassa 100.
- Documentação de rate limits e processamento paralelo em `docs/guides/web-search.md` e `docs/en/guides/web-search.md`, com tabelas de configurações recomendadas por provedor.

### Corrigido

- Filtrar `UserWarning: Field name X shadows ...` do `langchain_tavily` no import do provider (#74). Filtro específico ao módulo upstream.

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

[0.5.4]: https://github.com/bdcdo/dataframeit/compare/v0.5.3...v0.5.4
[0.5.3]: https://github.com/bdcdo/dataframeit/compare/v0.5.2...v0.5.3
[0.5.2]: https://github.com/bdcdo/dataframeit/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/bdcdo/dataframeit/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/bdcdo/dataframeit/releases/tag/v0.5.0
