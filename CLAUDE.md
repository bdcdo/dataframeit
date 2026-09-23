# Instruções para Claude

## Gerenciador de Pacotes

Use `uv` para gerenciar dependências e executar comandos Python. O workflow `tests.yml` roda testes e build da documentação com `uv run --extra ...`, então o mesmo comando local reproduz o ambiente do CI.

```bash
# Instalar dependências
uv sync

# Executar testes
uv run pytest

# Adicionar dependência
uv add <pacote>

# Executar scripts
uv run python script.py
```

## Fluxo de Trabalho Git

Crie uma branch antes de começar a trabalhar no código e integre por PR. É no PR que o CI de testes roda antes da integração, e todo push na `main` que toque `docs/` ou `mkdocs.yml` republica o site no GitHub Pages (`docs.yml`).

```bash
# Criar e mudar para nova branch
git checkout -b <tipo>/<descricao>

# Tipos comuns:
# - feature/  -> nova funcionalidade
# - fix/      -> correção de bug
# - docs/     -> documentação
# - refactor/ -> refatoração
```

## Versionamento

### CHANGELOG

Atualize o `CHANGELOG.md` ao alterar código. Ele é o registro das mudanças por versão, e a seção `[Unreleased]` guarda o que ainda não tem número:

- Siga o formato [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/)
- Categorias: `Adicionado`, `Alterado`, `Corrigido`, `Removido`, `Depreciado`, `Segurança`
- Inclua referência à issue/PR quando aplicável (ex: `(#123)`)
- **Factual e curto.** Sem subseções de "Migração", "Como migrar", "Impacto para usuários". A lib é de uso interno e esse tipo de nota é ruído.

### Documentação e comentários — sem notas de migração

Ao mudar comportamento, inclusive breaking:

- **Não** adicione notas do tipo `> Desde v0.X.Y, Z foi removido — migre para W.` em `docs/**/*.md`, `README.md` ou arquivos de exemplo. Atualize a tabela/exemplo para o estado **atual** e pronto.
- **Não** deixe comentários no código registrando o que era antes (`# antes era X`, `# invisível até v0.6.0`, etc). Comentários documentam invariantes do código atual, não o histórico.
- **Não** inclua seção "Migration" / "Impacto" nos bodies de PR. Basta descrever o quê e o porquê.
- Bump de versão está OK — serve de âncora no repo. Só não o referencie dentro de docs/comentários.

### Versão

Pergunte ao usuário antes de alterar a versão. O número é o que vai ao PyPI na publicação, e o PyPI não aceita reenviar uma versão já publicada:

> "Deseja manter a versão atual (X.Y.Z) ou fazer bump de versão? (patch/minor/major)"

Arquivos que contêm a versão:
- `pyproject.toml` (campo `version`)

Siga o [Versionamento Semântico](https://semver.org/lang/pt-BR/):
- **patch** (0.0.X): correções de bugs
- **minor** (0.X.0): novas funcionalidades retrocompatíveis
- **major** (X.0.0): mudanças que quebram compatibilidade

## Estrutura do Projeto

```
src/dataframeit/    # Código fonte
tests/              # Testes (pytest)
docs/               # Documentação fonte (markdown)
site/               # Build da documentação (gerado, não commitado)
```

## Testes

Execute os testes antes de finalizar alterações. O CI roda a mesma suíte no PR:

```bash
uv run pytest
```
