# Contribuindo com o DataFrameIt

Obrigado pelo interesse. Este guia cobre o ambiente, o fluxo de PR e as convenções do projeto.

## Ambiente

O projeto usa [uv](https://docs.astral.sh/uv/). Os testes e o build da documentação rodam com os mesmos extras que o CI usa:

```bash
git clone https://github.com/bdcdo/dataframeit.git
cd dataframeit

uv run --extra dev pytest                     # suíte de testes
uv run --extra dev --extra all --extra search-all --extra codex pytest --cov  # cobertura de ramos, exige 100%
uv run --extra dev ruff check src tests       # lint
uv run --extra docs mkdocs serve              # documentação em http://127.0.0.1:8000
```

Os testes não chamam nenhum provider de verdade e não precisam de chave de API. Alguns testes só rodam com o extra correspondente instalado, como `--extra codex` ou `--extra search-all`; sem ele, são pulados.

## Fluxo

1. Abra uma issue antes de uma mudança grande, para combinar o desenho.
2. Crie uma branch a partir da `main` com o prefixo do tipo de mudança: `feature/`, `fix/`, `docs/` ou `refactor/`.
3. Escreva o teste junto com a mudança. Uma correção de bug vem com o teste que falhava antes dela.
4. Abra o PR contra a `main`. O CI roda lint, a auditoria dos workflows pelo `zizmor`, testes em Python 3.11, 3.13 e 3.14, os testes com as versões mínimas das dependências e o build da documentação.

## Convenções

As regras de versionamento, de `CHANGELOG.md` e de escrita de documentação estão em [`CLAUDE.md`](CLAUDE.md), que vale para pessoas e para assistentes de código. Em resumo:

- Toda mudança de código entra no `CHANGELOG.md`, na seção `[Unreleased]`, no formato Keep a Changelog.
- A documentação descreve o estado atual. O histórico fica no `CHANGELOG.md` e no git, sem notas do tipo "desde a versão X".
- O número de versão só muda na preparação de um release.

Os workflows de `.github/workflows/` fixam cada action pelo SHA do commit, com a versão em comentário (`uses: dono/action@<sha> # v1.2.3`), e o `actions/checkout` leva `persist-credentials: false`. O passo `zizmor` do job `quality`, em `tests.yml`, reprova a action referida por tag ou branch no lugar do SHA, o `actions/checkout` sem a chave `persist-credentials` e a permissão de escrita no nível do workflow. O `zizmor --offline` não confere o comentário de versão e aceita `persist-credentials: true`, e essas duas partes da convenção ficam a cargo da revisão do PR. O mesmo comando do passo roda na máquina local.

A documentação tem duas árvores com o mesmo conteúdo: `docs/` em português, que é o idioma padrão do site, e `docs/en/` em inglês. Uma mudança numa página vale para as duas.

## Exemplos

Os notebooks de `example/` são versionados sem saídas. O teste `tests/test_exemplos.py` confere que eles usam só parâmetros que existem em `dataframeit()`.
