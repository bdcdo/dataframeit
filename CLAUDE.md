# Instruções para Claude

## Gerenciador de Pacotes

Use `uv` para gerenciar dependências e executar comandos Python. O CI (`tests.yml`) roda os testes com `uv run --extra dev` e o build da documentação com `uv run --extra docs`. Rode localmente com os mesmos extras, porque o `pytest` só existe no extra `dev`.

```bash
# Instalar dependências
uv sync --extra dev

# Instalar os hooks do pre-commit (uma vez por clone)
uv run --extra dev pre-commit install

# Executar testes
uv run --extra dev pytest

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

### Revisão antes do merge

PR de risco médio ou alto passa por revisão de um subagente novo antes de ser mesclado, inclusive quando o Bruno já aprovou. Risco baixo é documentação, texto, teste novo e código aditivo sem consumidor: nele o CI verde basta. Na revisão, o agente que implementou não revisa o próprio diff, porque carrega as mesmas suposições que produziram o código. O briefing traz o branch, o SHA, a issue e o diff contra a `main`, mas não o raciocínio da implementação, e pede evidência reproduzível de cada achado (arquivo:linha e script com a saída observada). O agente principal confere a evidência antes de aceitar um achado; o achado confirmado é corrigido no branch, e o recusado ganha o motivo no PR. Só depois disso, e com o CI verde, o PR é mesclado.

## Versionamento

### CHANGELOG

Atualize o `CHANGELOG.md` ao alterar código. Ele é o registro das mudanças por versão, e a seção `[Unreleased]` guarda o que ainda não tem número:

- Siga o formato [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/)
- Categorias: `Adicionado`, `Alterado`, `Corrigido`, `Removido`, `Depreciado`, `Segurança`
- Inclua referência à issue/PR quando aplicável (ex: `(#123)`)
- **Factual e curto.** Sem subseções de "Migração", "Como migrar", "Impacto para usuários".

### Documentação e comentários — sem notas de migração

A documentação e os comentários descrevem o estado atual, e o histórico fica no `CHANGELOG.md` e no git. Ao mudar comportamento, inclusive breaking:

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
src/dataframeit/    # Código fonte: core.py (função dataframeit), llm.py, agent.py,
                    # claude_code.py e codex.py (backends), search/ (Tavily, Exa)
tests/              # Testes (pytest)
example/            # Notebooks e script de exemplo; ruff.toml próprio
docs/               # Documentação fonte (markdown) em português; docs/en/ em inglês
site/               # Build da documentação (gerado, não commitado)
.github/workflows/  # tests.yml (CI do PR) e docs.yml (GitHub Pages)
```

## Testes

Execute os testes antes de finalizar alterações. O CI roda a mesma suíte no PR:

```bash
uv run --extra dev pytest
```

Os testes não chamam provider de verdade e não precisam de chave de API. Os que dependem de um extra, como `--extra codex` ou `--extra search-all`, são pulados sem ele. Para não pular nenhum, rode com os extras do job `quality`:

```bash
uv run --extra dev --extra all --extra search-all --extra polars --extra excel --extra codex pytest
```

## Lint e typecheck

O job `quality` do `tests.yml` roda estes comandos. O pre-commit roda os mesmos, menos o `zizmor`, quando o commit toca `src/`, `tests/` ou `example/`, e roda também a suíte sem os testes lentos (hook `pytest-rapido`, menos de um minuto):

```bash
EXTRAS="--extra dev --extra all --extra search-all --extra polars --extra excel --extra codex"
uv run $EXTRAS ruff format --check src tests
uv run $EXTRAS ruff check src tests example
uv run $EXTRAS ty check
uv run $EXTRAS vulture
uv run $EXTRAS deptry src
uvx zizmor@1.30.1 --offline .github/workflows   # auditoria dos workflows, fora do pre-commit

# Os hooks do pre-commit de uma vez, em todos os arquivos
uv run --extra dev pre-commit run --all-files
```

## Convenções observadas no histórico

- Comentários, docstrings, mensagens de erro e aviso, nomes de teste, títulos de PR e commits em português; os identificadores de `src/` ficam em inglês. O título do PR diz o comportamento novo numa frase afirmativa ("Retomada do CSV e do XLSX não cai com default_factory que levanta"), e o PR entra por squash com o número no fim (`(#292)`).
- Os commits do Dependabot seguem o padrão `chore(deps): ...`.
- Cobertura de ramos de `src/dataframeit` em 100%, somada entre os jobs do CI. Código novo vem com teste que cobre cada ramo.
- A suíte precisa passar duas vezes no mesmo processo (`pytest --count=2 --repeat-scope=session`, job `repeat`): teste não deixa estado global para trás.
- Actions dos workflows fixadas pelo SHA do commit, com a versão em comentário, e `persist-credentials: false` no `actions/checkout`; detalhes em `CONTRIBUTING.md`.
