"""O exemplo rápido de cada README roda e devolve a tabela que o README mostra.

O exemplo é a primeira chamada que quem chega à biblioteca adapta, e por isso
mostra `provider` e `text_column`, os argumentos que quase toda primeira
chamada fora do padrão precisa. O LLM é simulado: cada linha recebe os valores
que a tabela de saída do README mostra para ela.
"""

import ast
import re
from pathlib import Path
from unittest.mock import patch

import pytest

import dataframeit as pacote

RAIZ = Path(__file__).resolve().parent.parent
SECOES = {
    "README.md": "## Quick Example",
    "README.pt-BR.md": "## Exemplo Rápido",
    "README.es.md": "## Ejemplo rápido",
}


def _secao(nome):
    texto = (RAIZ / nome).read_text(encoding="utf-8")
    inicio = texto.index(SECOES[nome] + "\n")
    fim = texto.find("\n## ", inicio + 1)
    return texto[inicio:fim]


def _codigo(secao):
    bloco = re.search(r"```python\n(.*?)```", secao, re.DOTALL)
    assert bloco is not None
    return bloco.group(1)


def _tabela(secao):
    linhas = [linha for linha in secao.splitlines() if linha.startswith("|")]
    cabecalho, _, *corpo = (
        [celula.strip() for celula in linha.strip("|").split("|")] for linha in linhas
    )
    return cabecalho, [dict(zip(cabecalho, linha, strict=True)) for linha in corpo]


def _argumentos_da_chamada(codigo):
    chamadas = [
        no
        for no in ast.walk(ast.parse(codigo))
        if isinstance(no, ast.Call)
        and isinstance(no.func, ast.Name)
        and no.func.id == "dataframeit"
    ]
    assert len(chamadas) == 1
    return {argumento.arg for argumento in chamadas[0].keywords}


def _executa(codigo, linhas_da_tabela):
    """Roda o exemplo com o LLM simulado e devolve (saída, configs enviadas ao LLM)."""
    respostas = iter(linhas_da_tabela)
    configs = []
    saidas = []

    def _resposta(texto, modelo, prompt, config, *args, **kwargs):
        configs.append(config)
        linha = next(respostas)
        dados = modelo.model_validate({campo: linha[campo] for campo in modelo.model_fields})
        return {"data": dados.model_dump(), "usage": None}

    funcao_original = pacote.dataframeit

    def _registra_saida(*args, **kwargs):
        saidas.append(funcao_original(*args, **kwargs))
        return saidas[-1]

    with (
        patch("dataframeit.core.call_langchain", side_effect=_resposta),
        patch("dataframeit.core.validate_provider_dependencies"),
        patch.object(pacote, "dataframeit", _registra_saida),
        patch("builtins.print"),
    ):
        exec(compile(codigo, "<readme>", "exec"), {})  # noqa: S102 (código do próprio README)
    assert len(saidas) == 1
    return saidas[0], configs


@pytest.mark.parametrize("nome", sorted(SECOES))
def test_exemplo_rapido_mostra_provider_e_text_column(nome):
    assert {"provider", "text_column"} <= _argumentos_da_chamada(_codigo(_secao(nome)))


@pytest.mark.parametrize("nome", sorted(SECOES))
def test_exemplo_rapido_devolve_a_tabela_do_readme(nome):
    secao = _secao(nome)
    cabecalho, linhas = _tabela(secao)

    saida, configs = _executa(_codigo(secao), linhas)

    tokens = [coluna for coluna in saida.columns if coluna.endswith("_tokens")]
    assert tokens
    assert all(f"`{coluna}`" in secao for coluna in tokens)
    assert list(saida.columns) == cabecalho + tokens
    assert {config.provider for config in configs} == {"openai"}
    for (_, registro), linha in zip(saida.iterrows(), linhas, strict=True):
        for coluna, celula in linha.items():
            assert str(registro[coluna]).startswith(celula.removesuffix("...").rstrip())
