"""O exemplo rápido de cada README e do site roda e dá a tabela que a página mostra.

O exemplo é a primeira chamada que quem chega à biblioteca adapta, e por isso
mostra `provider` e `text_column`, os argumentos que quase toda primeira
chamada fora do padrão precisa. O LLM é simulado e responde a cada texto com a
linha da tabela de saída da página que começa por ele.
"""

import re
import textwrap
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

import dataframeit as pacote
from dataframeit.utils import TOKEN_COLUMNS

RAIZ = Path(__file__).resolve().parent.parent
SECOES = {
    "README.md": "## Quick Example",
    "README.pt-BR.md": "## Exemplo Rápido",
    "README.es.md": "## Ejemplo rápido",
    "docs/index.md": "## O que é?",
    "docs/en/index.md": "## What is it?",
}
# Títulos do quickstart: os passos em ordem (o último traz a tabela de saída) e
# a seção do exemplo completo.
QUICKSTARTS = {
    "docs/getting-started/quickstart.md": (
        [
            "## Passo 1: Defina seu Modelo Pydantic",
            "## Passo 2: Prepare seus Dados",
            "## Passo 3: Processe!",
        ],
        "## Exemplo Completo",
    ),
    "docs/en/getting-started/quickstart.md": (
        [
            "## Step 1: Define your Pydantic Model",
            "## Step 2: Prepare your Data",
            "## Step 3: Process!",
        ],
        "## Complete Example",
    ),
}


def _secao(nome, titulo):
    texto = (RAIZ / nome).read_text(encoding="utf-8")
    inicio = texto.index(titulo + "\n")
    return texto[inicio : texto.index("\n## ", inicio + 1)]


def _blocos(secao):
    """Blocos Python da seção, sem o recuo de uma aba do MkDocs."""
    blocos = re.findall(r"^( *)```python\n(.*?)^\1```", secao, re.DOTALL | re.MULTILINE)
    assert blocos
    return [textwrap.dedent(codigo) for _, codigo in blocos]


def _codigo(secao):
    return _blocos(secao)[0]


def _tabela(secao):
    linhas = [linha for linha in secao.splitlines() if linha.startswith("|")]
    cabecalho, _, *corpo = (
        [celula.strip() for celula in linha.strip("|").split("|")] for linha in linhas
    )
    return cabecalho, [dict(zip(cabecalho, linha, strict=True)) for linha in corpo]


def _comeco(celula):
    """Texto conhecido da célula: sem o '...' final da célula truncada."""
    return celula.removesuffix("...").rstrip()


def _executa(codigo, linhas_da_tabela):
    """Roda o código e devolve (o DataFrame que ele imprime ou salva, a chamada a dataframeit).

    O texto que o LLM simulado recebe precisa começar por uma célula truncada da
    tabela; se `text_column` apontar para outra coluna, nenhuma linha bate.
    """

    def _resposta(texto, modelo, *args, **kwargs):
        linhas = [
            linha
            for linha in linhas_da_tabela
            if any(
                celula.endswith("...") and texto.startswith(_comeco(celula))
                for celula in linha.values()
            )
        ]
        assert len(linhas) == 1, texto
        dados = modelo.model_validate({campo: linhas[0][campo] for campo in modelo.model_fields})
        return {"data": dados.model_dump(), "usage": None}

    saidas = []
    with (
        patch("dataframeit.core.call_langchain", side_effect=_resposta),
        patch("dataframeit.core.validate_provider_dependencies"),
        patch.object(pacote, "dataframeit", wraps=pacote.dataframeit) as chamada,
        patch.object(
            pd.DataFrame,
            "to_excel",
            autospec=True,
            side_effect=lambda df, *_, **__: saidas.append(df),
        ),
    ):
        exec(compile(codigo, "<exemplo>", "exec"), {"print": saidas.append})  # noqa: S102 (código da própria documentação)
    assert len(saidas) == 1
    return saidas[0], chamada


def _confere(codigo, secao_da_tabela):
    cabecalho, linhas = _tabela(secao_da_tabela)

    saida, chamada = _executa(codigo, linhas)

    chamada.assert_called_once()
    assert {"provider", "text_column"} <= chamada.call_args.kwargs.keys()
    assert list(saida.columns) == cabecalho + list(TOKEN_COLUMNS)
    assert all(f"`{coluna}`" in secao_da_tabela for coluna in TOKEN_COLUMNS)
    for (_, registro), linha in zip(saida.iterrows(), linhas, strict=True):
        for coluna, celula in linha.items():
            if celula.endswith("..."):
                assert str(registro[coluna]).startswith(_comeco(celula))
            else:
                assert str(registro[coluna]) == celula


@pytest.mark.parametrize("nome", sorted(SECOES))
def test_exemplo_rapido_imprime_a_tabela_da_pagina(nome):
    secao = _secao(nome, SECOES[nome])
    _confere(_codigo(secao), secao)


@pytest.mark.parametrize("nome", sorted(QUICKSTARTS))
def test_passos_do_quickstart_imprimem_a_tabela_do_passo_3(nome):
    """Os blocos dos passos, rodados em sequência (no passo 2, a aba DataFrame)."""
    passos, _ = QUICKSTARTS[nome]
    codigo = "\n".join(_codigo(_secao(nome, titulo)) for titulo in passos)
    _confere(codigo, _secao(nome, passos[-1]))


@pytest.mark.parametrize("nome", sorted(QUICKSTARTS))
def test_abas_lista_e_dicionario_do_quickstart_tem_os_textos_do_dataframe(nome):
    aba_df, aba_lista, aba_dict = _blocos(_secao(nome, QUICKSTARTS[nome][0][1]))
    escopos = [{}, {}, {}]
    for codigo, escopo in zip([aba_df, aba_lista, aba_dict], escopos, strict=True):
        exec(codigo, escopo)  # noqa: S102 (código da própria documentação)
    df = escopos[0]["df"]
    lista, dicionario = (
        escopo["texts" if "texts" in escopo else "textos"] for escopo in escopos[1:]
    )

    assert lista == df.iloc[:, 1].tolist()
    assert dicionario == dict(zip(df.iloc[:, 0], df.iloc[:, 1], strict=True))


@pytest.mark.parametrize("nome", sorted(QUICKSTARTS))
def test_exemplo_completo_do_quickstart_salva_a_tabela_do_passo_3(nome):
    passos, completo = QUICKSTARTS[nome]
    _confere(_codigo(_secao(nome, completo)), _secao(nome, passos[-1]))
