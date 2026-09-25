"""Testes de seleção de linhas quando o índice não está em ordem crescente.

A escolha das linhas a processar depende só do status de cada linha, e não da
ordem dos rótulos do índice. DataFrames depois de sort_values, sample ou filtro,
índices textuais e entrada dict chegam com rótulos fora de ordem.
"""

from unittest.mock import patch

import pandas as pd
import pytest
from pydantic import BaseModel

from dataframeit.core import dataframeit


class ModeloSimples(BaseModel):
    campo1: str


def _llm_que_ecoa_o_texto(textos_enviados):
    def llm_falso(*args, **kwargs):
        texto = args[0]
        textos_enviados.append(texto)
        return {"data": {"campo1": f"resposta {texto}"}, "usage": {}}

    return llm_falso


def _executar(dados, parallel_requests=1, **kwargs):
    textos_enviados = []
    with patch(
        "dataframeit.core.call_langchain",
        side_effect=_llm_que_ecoa_o_texto(textos_enviados),
    ):
        with patch("dataframeit.core.validate_provider_dependencies"):
            resultado = dataframeit(
                dados,
                questions=ModeloSimples,
                prompt="Teste {texto}",
                parallel_requests=parallel_requests,
                **kwargs,
            )
    return resultado, textos_enviados


@pytest.mark.parametrize("parallel_requests", [1, 2])
def test_indice_numerico_fora_de_ordem_processa_todas_as_linhas(parallel_requests):
    df = pd.DataFrame({"texto": ["cinco", "tres", "um"]}, index=[5, 3, 1])

    resultado, textos_enviados = _executar(df, parallel_requests)

    assert sorted(textos_enviados) == ["cinco", "tres", "um"]
    assert resultado["campo1"].tolist() == [
        "resposta cinco",
        "resposta tres",
        "resposta um",
    ]
    assert resultado.index.tolist() == [5, 3, 1]


@pytest.mark.parametrize("parallel_requests", [1, 2])
def test_indice_textual_processa_todas_as_linhas(parallel_requests):
    df = pd.DataFrame(
        {"texto": ["primeiro", "segundo", "terceiro"]},
        index=["processo-c", "processo-a", "processo-b"],
    )

    resultado, textos_enviados = _executar(df, parallel_requests)

    assert sorted(textos_enviados) == ["primeiro", "segundo", "terceiro"]
    assert resultado["campo1"].tolist() == [
        "resposta primeiro",
        "resposta segundo",
        "resposta terceiro",
    ]


def test_dict_com_chaves_fora_de_ordem_processa_todas_as_chaves():
    dados = {"zeta": "primeiro", "alfa": "segundo"}

    resultado, textos_enviados = _executar(dados)

    assert sorted(textos_enviados) == ["primeiro", "segundo"]
    assert resultado.loc["zeta", "campo1"] == "resposta primeiro"
    assert resultado.loc["alfa", "campo1"] == "resposta segundo"


@pytest.mark.parametrize("parallel_requests", [1, 2])
def test_retomada_com_indice_fora_de_ordem_processa_so_as_pendentes(parallel_requests):
    df = pd.DataFrame(
        {
            "texto": ["cinco", "tres", "um"],
            "campo1": [None, "valor anterior", None],
            "_dataframeit_status": [None, "processed", None],
            "_error_details": [None, None, None],
        },
        index=[5, 3, 1],
    )

    resultado, textos_enviados = _executar(df, parallel_requests, resume=True)

    assert sorted(textos_enviados) == ["cinco", "um"]
    assert resultado.loc[5, "campo1"] == "resposta cinco"
    assert resultado.loc[3, "campo1"] == "valor anterior"
    assert resultado.loc[1, "campo1"] == "resposta um"


@pytest.mark.parametrize("parallel_requests", [1, 2])
def test_retomada_nao_reprocessa_linha_com_erro_registrado(parallel_requests):
    df = pd.DataFrame(
        {
            "texto": ["cinco", "tres", "um"],
            "campo1": [None, None, "valor anterior"],
            "_dataframeit_status": [None, "error", "processed"],
            "_error_details": [None, "falha anterior", None],
        },
        index=[5, 3, 1],
    )

    resultado, textos_enviados = _executar(df, parallel_requests, resume=True)

    assert textos_enviados == ["cinco"]
    assert resultado.loc[3, "_dataframeit_status"] == "error"
    assert resultado.loc[1, "campo1"] == "valor anterior"


@pytest.mark.parametrize("parallel_requests", [1, 2])
def test_reprocess_columns_com_indice_fora_de_ordem_processa_todas_as_linhas(
    parallel_requests,
):
    df = pd.DataFrame(
        {
            "texto": ["cinco", "tres", "um"],
            "campo1": ["antigo", "antigo", None],
            "_dataframeit_status": ["processed", "processed", None],
            "_error_details": [None, None, None],
        },
        index=[5, 3, 1],
    )

    resultado, textos_enviados = _executar(df, parallel_requests, reprocess_columns=["campo1"])

    assert sorted(textos_enviados) == ["cinco", "tres", "um"]
    assert resultado["campo1"].tolist() == [
        "resposta cinco",
        "resposta tres",
        "resposta um",
    ]
