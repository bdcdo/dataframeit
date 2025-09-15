Título: Bug: KeyError 'format' e SettingWithCopyWarning após refatoração

Resumo
- Após refatoração, ao processar com LangChain, ocorre KeyError: 'format' ao montar o prompt.
- Além disso, SettingWithCopyWarning é emitido ao inicializar colunas em DataFrames que são slices (views).

Sintomas observados
- SettingWithCopyWarning em dataframeit/core/managers.py:45 ao executar df.loc[:, col] = None em um DF potencialmente view.
- UserWarning por falha ao processar linhas devido a KeyError: 'format' (quando o LLM devolve instruções com chaves adicionais).

Causa raiz
- PromptBuilder usava template.format(**{placeholder: texto}). As format_instructions do LangChain podem conter chaves como {format}, que o Python tenta resolver, gerando KeyError.
- SettingWithCopyWarning ocorre porque colunas são criadas diretamente sobre um DF que pode ser view (slice), tornando a mutação não determinística (view vs copy).

Correções implementadas
- src/dataframeit/core/base.py (PromptBuilder.format_prompt):
  - Troca de str.format() por substituição direta do placeholder: template.replace("{documento}", texto)
  - Evita que chaves alheias nas format_instructions (ex.: {format}) sejam interpretadas.
- src/dataframeit/dataframeit.py (DataFramePreparationPipeline.prepare):
  - Após converter para pandas, garantir cópia: df_pandas = df_pandas.copy()
  - Elimina SettingWithCopyWarning ao criar colunas auxiliares (status/erro/resultados) e garante comportamento determinístico.
- Robustez adicional:
  - Tipagem Optional em PreparedData.row_processor e construtor do Orchestrator.
  - Checagem defensiva antes de usar row_processor no loop.
- Versão: bump para 0.1.1
- Testes: tests/test_regressions.py cobrindo ambos cenários.

Como reproduzir
1) KeyError:
   - Usar LangChain com PromptBuilder e um modelo Pydantic, onde format_instructions incluam {format}.
   - Antes do fix: template.format(...) lança KeyError 'format'.
   - Depois do fix: montagem do prompt ocorre sem erros (apenas o placeholder {documento} é substituído).
2) SettingWithCopyWarning:
   - Criar df_slice = df[df.x > 0] e processar; antes do fix, warnings na criação de colunas via df.loc[:, col] = None.
   - Após o fix (cópia antecipada), sem warnings.

Validação
- Testes automatizados adicionados:
  - Garantem ausência de SettingWithCopyWarning ao configurar colunas numa cópia de um slice.
  - Garantem que {format} em format_instructions não dispara KeyError (substitui somente {documento}).

Impacto e compatibilidade
- Compatível com retomada de processamento: a cópia garante comportamento determinístico; os resultados permanecem no DataFrame retornado pela API.
- Usuários que dependiam de mutação in-place do mesmo objeto devem utilizar o DataFrame retornado (fluxo recomendado). Para tolerância a interrupções de processo, considerar checkpoint periódico em disco (futuro enhancement).

Referências
- pandas view vs copy: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy

Checklist
- [x] Corrigir KeyError 'format' na montagem do prompt
- [x] Remover SettingWithCopyWarning com cópia antecipada
- [x] Adicionar testes de regressão
- [x] Bump de versão para 0.1.1
- [ ] Criar branch, commit e push
- [ ] Abrir PR vinculando a esta issue (opcional)
