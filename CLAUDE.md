# Instruções para Claude

## Gerenciador de Pacotes

**Sempre utilize `uv`** para gerenciamento de dependências e execução de comandos Python.

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

**Sempre crie uma branch antes de começar a trabalhar no código.**

```bash
# Criar e mudar para nova branch
git checkout -b <tipo>/<descricao>

# Tipos comuns:
# - feature/  -> nova funcionalidade
# - fix/      -> correção de bug
# - docs/     -> documentação
# - refactor/ -> refatoração
```

Nunca faça commits diretamente na `main`.

## Versionamento

### CHANGELOG

**Sempre atualize o `CHANGELOG.md`** ao fazer alterações no código:

- Siga o formato [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/)
- Categorias: `Adicionado`, `Alterado`, `Corrigido`, `Removido`, `Depreciado`, `Segurança`
- Inclua referência à issue/PR quando aplicável (ex: `(#123)`)

### Versão

**Sempre pergunte ao usuário** antes de fazer alterações de versão:

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

Sempre execute os testes antes de finalizar alterações:

```bash
uv run pytest
```
