# Contribuindo com o DataFrameIt

Obrigado pelo interesse. Este guia cobre o ambiente, o fluxo de PR e as convenções do projeto.

## Ambiente

O projeto usa [uv](https://docs.astral.sh/uv/). Os testes e o build da documentação rodam com os mesmos extras que o CI usa:

```bash
git clone https://github.com/bdcdo/dataframeit.git
cd dataframeit

uv run --extra dev pytest                     # suíte de testes
uv run --extra dev --extra all --extra search-all --extra polars --extra excel --extra codex pytest --count=2 --repeat-scope=session  # suíte duas vezes no mesmo processo, sem pular teste
uv run --extra dev ruff check src tests example  # lint
uv run --extra docs mkdocs serve              # documentação em http://127.0.0.1:8000
```

Os testes não chamam nenhum provider de verdade e não precisam de chave de API. Alguns testes só rodam com o extra correspondente instalado, como `--extra codex` ou `--extra search-all`; sem ele, são pulados.

A cobertura de ramos exigida, 100%, é a soma de duas execuções, como no CI: o ramo de `codex.py` que trata o SDK ausente só roda sem o extra `codex`, e o código que usa o SDK só roda com ele. O `--exact` tira do ambiente os pacotes dos extras não pedidos, que o `uv run` deixaria instalados. O relatório confronta as linhas gravadas com o fonte no disco quando é gerado, e um arquivo de `src/` editado durante a execução sai com linhas executadas dadas como não cobertas.

```bash
COVERAGE_FILE=.coverage.unit uv run --exact --extra dev --extra all --extra search-all pytest --cov --cov-report= --cov-fail-under=0
COVERAGE_FILE=.coverage.codex uv run --exact --extra dev --extra codex pytest --cov --cov-report= --cov-fail-under=0
uv run --extra dev coverage combine .coverage.unit .coverage.codex
uv run --extra dev coverage report  # exige 100%
```

### Pre-commit

Os hooks do [pre-commit](https://pre-commit.com/), definidos em `.pre-commit-config.yaml`, rodam a cada commit que toca `src/`, `tests/` ou `example/`. Eles fazem as conferências do job `quality` do CI, menos o `zizmor`. O `ruff format` reformata os arquivos e interrompe o commit; basta adicionar os arquivos reformatados e repetir. O hook `pytest-rapido` roda a suíte sem os testes lentos, listados no `.pre-commit-config.yaml`; a suíte completa fica com o CI e com os comandos `pytest` acima. A instalação é uma vez por clone:

```bash
uv run --extra dev pre-commit install
uv run --extra dev pre-commit run --all-files  # todos os hooks em src/, tests/ e example/
```

## Fluxo

1. Abra uma issue antes de uma mudança grande, para combinar o desenho.
2. Crie uma branch a partir da `main` com o prefixo do tipo de mudança: `feature/`, `fix/`, `docs/` ou `refactor/`.
3. Escreva o teste junto com a mudança. Uma correção de bug vem com o teste que falhava antes dela.
4. Abra o PR contra a `main`. O CI roda lint, a auditoria dos workflows pelo `zizmor`, testes em Python 3.11, 3.13 e 3.14, a suíte duas vezes no mesmo processo, que reprova o teste que só passa na primeira execução, os testes com as versões mínimas das dependências e o build da documentação.

## Convenções

As regras de versionamento, de `CHANGELOG.md` e de escrita de documentação estão em [`CLAUDE.md`](CLAUDE.md), que vale para pessoas e para assistentes de código. Em resumo:

- Toda mudança de código entra no `CHANGELOG.md`, na seção `[Unreleased]`, no formato Keep a Changelog.
- A documentação descreve o estado atual. O histórico fica no `CHANGELOG.md` e no git, sem notas do tipo "desde a versão X".
- O número de versão só muda na preparação de um release.

Os workflows de `.github/workflows/` fixam cada action pelo SHA do commit, com a versão em comentário (`uses: dono/action@<sha> # v1.2.3`), e o `actions/checkout` leva `persist-credentials: false`. O passo `zizmor` do job `quality`, em `tests.yml`, reprova a action referida por tag ou branch no lugar do SHA, o `actions/checkout` sem a chave `persist-credentials` e a permissão de escrita no nível do workflow. O `zizmor --offline` não confere o comentário de versão e aceita `persist-credentials: true`, e essas duas partes da convenção ficam a cargo da revisão do PR. O mesmo comando do passo roda na máquina local.

A documentação tem duas árvores com o mesmo conteúdo: `docs/` em português, que é o idioma padrão do site, e `docs/en/` em inglês. Uma mudança numa página vale para as duas.

## Exemplos

Os notebooks de `example/` são versionados sem saídas. O teste `tests/test_exemplos.py` confere que eles usam só parâmetros que existem em `dataframeit()`. O `ruff check` confere os exemplos só com as regras de correção, listadas em `example/ruff.toml`, sem as de estilo de biblioteca, como docstring e anotação de tipo.
