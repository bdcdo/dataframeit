"""on_row_done: uma chamada por linha processada, com o que ficou gravado na linha."""

import dataclasses
import threading
from unittest.mock import patch

import pandas as pd
import pytest
from pydantic import BaseModel

import dataframeit as pacote
from dataframeit import RowResult, dataframeit


class Modelo(BaseModel):
    campo1: str
    campo2: str


def _llm(text, *args, **kwargs):
    if text == "b":
        msg = "falhou"
        raise ValueError(msg)
    return {"data": {"campo1": text, "campo2": text.upper()}, "usage": None}


def _roda(df, llm=_llm, **kwargs):
    with (
        patch("dataframeit.core.call_langchain", side_effect=llm),
        patch("dataframeit.core.validate_provider_dependencies"),
    ):
        return dataframeit(df, questions=Modelo, prompt="{texto}", track_tokens=False, **kwargs)


class _Coletor:
    """Guarda cada RowResult e a thread que o entregou, sob trava própria."""

    def __init__(self):
        self.resultados = []
        self.threads = []
        self._trava = threading.Lock()

    def __call__(self, resultado):
        with self._trava:
            self.resultados.append(resultado)
            self.threads.append(threading.current_thread())

    def por_posicao(self):
        return sorted(self.resultados, key=lambda r: r.position)


def test_row_result_e_exportado_e_imutavel():
    assert "RowResult" in pacote.__all__
    assert pacote.RowResult is RowResult
    assert [campo.name for campo in dataclasses.fields(RowResult)] == [
        "index",
        "position",
        "success",
        "error",
        "values",
        "elapsed_seconds",
    ]
    assert RowResult.__dataclass_params__.frozen


@pytest.mark.parametrize("paralelo", [1, 2])
def test_uma_chamada_por_linha_em_sucesso_em_erro_e_em_texto_ausente(paralelo):
    df = pd.DataFrame({"texto": ["a", "b", None, "d"]}, index=["x", "y", "z", "w"])
    coletor = _Coletor()

    with pytest.warns(UserWarning, match="sem texto"):
        saida = _roda(df, parallel_requests=paralelo, on_row_done=coletor)

    linhas = coletor.por_posicao()
    assert [linha.position for linha in linhas] == [0, 1, 2, 3]
    assert [linha.index for linha in linhas] == ["x", "y", "z", "w"]
    assert [linha.success for linha in linhas] == [True, False, False, True]
    assert [linha.values for linha in linhas] == [
        {"campo1": "a", "campo2": "A"},
        {},
        {},
        {"campo1": "d", "campo2": "D"},
    ]
    # `error` é o que ficou na coluna, inclusive None na linha sem detalhe.
    assert [linha.error for linha in linhas] == [
        None if pd.isna(detalhe) else detalhe for detalhe in saida["_error_details"]
    ]
    assert linhas[1].error == "[Falhou após 3 tentativa(s)] ValueError: falhou"
    assert linhas[2].error == "Texto ausente"
    assert all(linha.elapsed_seconds >= 0 for linha in linhas)
    principal = threading.main_thread()
    if paralelo == 1:
        assert all(thread is principal for thread in coletor.threads)
    else:
        assert all(thread is not principal for thread in coletor.threads)


@pytest.mark.parametrize("paralelo", [1, 2])
def test_linha_pulada_por_resume_nao_e_entregue(paralelo):
    df = pd.DataFrame(
        {
            "texto": ["a", "c", "d"],
            "campo1": ["antigo", None, None],
            "campo2": ["ANTIGO", None, None],
            "_dataframeit_status": ["processed", None, None],
            "_error_details": [None, None, None],
        }
    )
    coletor = _Coletor()

    _roda(df, parallel_requests=paralelo, on_row_done=coletor)

    linhas = coletor.por_posicao()
    assert [(linha.position, linha.index) for linha in linhas] == [(0, 1), (1, 2)]


@pytest.mark.parametrize("paralelo", [1, 2])
def test_reprocessamento_entrega_so_as_colunas_reprocessadas(paralelo):
    df = pd.DataFrame(
        {
            "texto": ["a", "c"],
            "campo1": ["antigo", "antigo"],
            "campo2": ["ANTIGO", "ANTIGO"],
            "_dataframeit_status": ["processed", "processed"],
            "_error_details": [None, None],
        }
    )
    coletor = _Coletor()

    _roda(df, parallel_requests=paralelo, on_row_done=coletor, reprocess_columns=["campo1"])

    assert [linha.values for linha in coletor.por_posicao()] == [{"campo1": "a"}, {"campo1": "c"}]


@pytest.mark.parametrize("paralelo", [1, 2])
def test_sucesso_com_retry_entrega_o_detalhe_gravado(paralelo):
    def llm_com_retry(text, *args, **kwargs):
        return {
            "data": {"campo1": text, "campo2": text},
            "usage": None,
            "_retry_info": {"retries": 1},
        }

    coletor = _Coletor()

    saida = _roda(
        pd.DataFrame({"texto": ["a"]}),
        llm_com_retry,
        parallel_requests=paralelo,
        on_row_done=coletor,
    )

    (linha,) = coletor.resultados
    assert linha.success is True
    assert linha.error == "Sucesso após 1 retry(s)" == saida["_error_details"].iloc[0]


@pytest.mark.parametrize("paralelo", [1, 2])
def test_excecao_no_callback_vira_aviso_e_a_execucao_segue(paralelo):
    entregues = []

    def callback_que_falha(resultado):
        entregues.append(resultado.index)
        if resultado.index == 0:
            msg = "boom"
            raise RuntimeError(msg)

    with pytest.warns(UserWarning, match=r"on_row_done falhou na linha 0: RuntimeError: boom"):
        saida = _roda(
            pd.DataFrame({"texto": ["a", "c", "d"]}),
            parallel_requests=paralelo,
            on_row_done=callback_que_falha,
        )

    assert sorted(entregues) == [0, 1, 2]
    assert saida["campo1"].tolist() == ["a", "c", "d"]
    assert "_dataframeit_status" not in saida.columns


@pytest.mark.parametrize("paralelo", [1, 2])
def test_caminho_polars_entrega_o_mesmo_row_result(paralelo):
    pl = pytest.importorskip("polars")
    coletor = _Coletor()

    saida = _roda(
        pl.DataFrame({"texto": ["a", "b", "d"]}), parallel_requests=paralelo, on_row_done=coletor
    )

    assert isinstance(saida, pl.DataFrame)
    linhas = coletor.por_posicao()
    assert [(linha.index, linha.position, linha.success) for linha in linhas] == [
        (0, 0, True),
        (1, 1, False),
        (2, 2, True),
    ]
    assert linhas[0].values == {"campo1": "a", "campo2": "A"}


def test_linha_cuja_gravacao_falhou_fora_do_worker_tambem_e_entregue():
    """A linha registrada pela coleta, e não pelo worker, chega ao callback uma vez."""

    def gravacao_que_falha(*args, **kwargs):
        msg = "gravação"
        raise RuntimeError(msg)

    coletor = _Coletor()
    with (
        patch("dataframeit.core._record_missing_text", side_effect=gravacao_que_falha),
        pytest.warns(UserWarning, match="Falha ao processar linha 1"),
    ):
        saida = _roda(
            pd.DataFrame({"texto": ["a", None, "d"]}), parallel_requests=2, on_row_done=coletor
        )

    linhas = coletor.por_posicao()
    assert [(linha.index, linha.success) for linha in linhas] == [(0, True), (1, False), (2, True)]
    assert linhas[1].error == "[Falha ao registrar o erro] RuntimeError: gravação"
    assert linhas[1].error == saida["_error_details"].iloc[1]
    assert linhas[1].values == {}
