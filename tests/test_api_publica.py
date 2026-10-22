import ast
import inspect
import re
from pathlib import Path

import pytest

import dataframeit
from dataframeit import errors

PASTA_DOCS = Path(__file__).resolve().parent.parent / "docs"
IDIOMAS = [PASTA_DOCS, PASTA_DOCS / "en"]


def test_all_so_lista_nomes_que_existem():
    for nome in dataframeit.__all__:
        assert hasattr(dataframeit, nome), nome


def test_excecoes_do_topo_sao_as_de_errors():
    # Quem captura pelo nome do topo precisa pegar a mesma classe que o código levanta.
    for nome in (
        "ProviderError",
        "ProviderTransientError",
        "ProviderOverloadedError",
        "ProviderRejectedOutputError",
        "ProviderConfigurationError",
        "ProviderOutputError",
        "ProviderAbortError",
        "ProviderUsageLimitError",
    ):
        assert nome in dataframeit.__all__
        assert getattr(dataframeit, nome) is getattr(errors, nome)


def test_hierarquia_documentada_das_excecoes():
    assert issubclass(dataframeit.ProviderOverloadedError, dataframeit.ProviderTransientError)
    assert issubclass(dataframeit.ProviderRejectedOutputError, dataframeit.ProviderTransientError)
    assert issubclass(dataframeit.ProviderRejectedOutputError, ValueError)
    assert issubclass(dataframeit.ProviderTransientError, dataframeit.ProviderError)
    assert issubclass(dataframeit.ProviderError, RuntimeError)
    assert issubclass(dataframeit.ProviderConfigurationError, ValueError)
    assert issubclass(dataframeit.ProviderOutputError, ValueError)
    assert not issubclass(dataframeit.ProviderOutputError, dataframeit.ProviderError)
    assert issubclass(dataframeit.ProviderUsageLimitError, dataframeit.ProviderAbortError)
    assert issubclass(dataframeit.ProviderAbortError, dataframeit.ProviderError)
    assert not issubclass(dataframeit.ProviderAbortError, dataframeit.ProviderTransientError)


@pytest.mark.parametrize("pasta", IDIOMAS, ids=["pt", "en"])
def test_assinatura_da_referencia_segue_a_ordem_do_codigo(pasta):
    # Quem passa um parâmetro por posição conta com a ordem que a referência mostra.
    texto = (pasta / "reference" / "api.md").read_text(encoding="utf-8")
    bloco = re.search(
        r"^```python\n(def dataframeit\(.*?\) -> [^\n]+)$", texto, re.DOTALL | re.MULTILINE
    )
    assert bloco, "a referência da API não traz o def de dataframeit()"
    funcao = ast.parse(bloco.group(1) + ": ...").body[0]
    assert isinstance(funcao, ast.FunctionDef)
    documentados = [argumento.arg for argumento in funcao.args.args + funcao.args.kwonlyargs]
    assert documentados == list(inspect.signature(dataframeit.dataframeit).parameters)


@pytest.mark.parametrize("pasta", IDIOMAS, ids=["pt", "en"])
def test_pagina_de_excecoes_lista_todas_as_exportadas(pasta):
    texto = (pasta / "reference" / "exceptions.md").read_text(encoding="utf-8")
    exportadas = [
        nome
        for nome in dataframeit.__all__
        if isinstance(getattr(dataframeit, nome), type)
        and issubclass(getattr(dataframeit, nome), BaseException)
    ]
    assert exportadas
    assert [nome for nome in exportadas if f"`{nome}`" not in texto] == []
