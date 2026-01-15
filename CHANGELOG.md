# Changelog

Todas as mudanças notáveis deste projeto serão documentadas neste arquivo.

O formato é baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/),
e este projeto adere ao [Versionamento Semântico](https://semver.org/lang/pt-BR/).

## [Unreleased]

### Adicionado

- **Aviso de rate limit para busca web com suporte a múltiplos provedores** (#67): O DataFrameIt agora emite um warning quando a configuração de `parallel_requests` combinada com `use_search=True` pode exceder os limites de taxa do provedor de busca selecionado.

  **Rate limits por provedor:**
  - Tavily: ~100 req/min
  - Exa: ~300 req/min (3x maior que Tavily)

  ```
  ============================================================
  AVISO: Configuração pode exceder rate limits de busca (Tavily)
  ============================================================
  Configuração atual:
    - Provedor de busca: tavily
    - parallel_requests: 20
    - search_per_field: True
    - Total de queries estimadas: 400

  Recomendações para evitar HTTP 429:
    dataframeit(..., parallel_requests=2, rate_limit_delay=1.7)
  ============================================================
  ```

  - O warning inclui recomendações específicas de `parallel_requests` e `rate_limit_delay`
  - Calcula queries concorrentes considerando `search_per_field`
  - Usa limites específicos para cada provedor (Tavily vs Exa)
  - Documentação atualizada com tabela de configurações recomendadas por provedor

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

[0.5.2]: https://github.com/bdcdo/dataframeit/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/bdcdo/dataframeit/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/bdcdo/dataframeit/releases/tag/v0.5.0
