"""Falha que impede as linhas seguintes interrompe a execução sem gastá-las."""

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pandas as pd
import pytest
from pydantic import BaseModel

from dataframeit import ProviderAbortError, ProviderUsageLimitError, dataframeit


class Modelo(BaseModel):
    campo1: str


def _llm_que_esgota(limite: int, erro: Exception):
    chamadas = []

    def llm(*args, **kwargs):
        chamadas.append(1)
        if len(chamadas) > limite:
            raise erro
        return {
            "data": {"campo1": f"v{len(chamadas)}"},
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }

    return chamadas, llm


def _roda(df, llm, **kwargs):
    with (
        patch("dataframeit.core.call_langchain", side_effect=llm),
        patch("dataframeit.core.validate_provider_dependencies"),
    ):
        return dataframeit(df, questions=Modelo, prompt="{texto}", **kwargs)


@pytest.mark.parametrize("paralelo", [1, 2])
def test_limite_de_uso_interrompe_e_deixa_as_linhas_sem_status(paralelo):
    df = pd.DataFrame({"texto": [f"linha{i}" for i in range(8)]})
    chamadas, llm = _llm_que_esgota(2, ProviderUsageLimitError("usage limit"))

    with pytest.warns(UserWarning, match=r"interrompida.*usage limit.*resume=True"):
        saida = _roda(df, llm, parallel_requests=paralelo)

    status = saida["_dataframeit_status"]
    assert (status == "processed").sum() == 2
    assert status.isna().sum() == 6
    assert "error" not in status.tolist()
    # Depois da interrupção nenhuma linha nova é despachada: no paralelo, só as
    # que já estavam em voo quando o limite chegou.
    assert len(chamadas) <= 2 + paralelo


def test_sem_interrupcao_as_colunas_de_controle_somem_como_antes():
    df = pd.DataFrame({"texto": ["a", "b"]})
    _, llm = _llm_que_esgota(10, ProviderUsageLimitError("nunca"))

    saida = _roda(df, llm)

    assert "_dataframeit_status" not in saida.columns


def test_retomada_depois_da_interrupcao_processa_so_o_que_faltou():
    df = pd.DataFrame({"texto": [f"linha{i}" for i in range(5)]})
    _, llm = _llm_que_esgota(3, ProviderAbortError("app-server encerrou"))
    with pytest.warns(UserWarning, match="interrompida"):
        parcial = _roda(df, llm)

    chamadas, llm_ok = _llm_que_esgota(100, ProviderAbortError("nunca"))
    final = _roda(parcial, llm_ok, resume=True)

    assert len(chamadas) == 2
    assert final["campo1"].tolist()[:3] == ["v1", "v2", "v3"]
    assert final["campo1"].notna().all()


@pytest.mark.parametrize("paralelo", [1, 2])
def test_interrupcao_grava_o_checkpoint(tmp_path, paralelo):
    df = pd.DataFrame({"texto": [f"linha{i}" for i in range(6)]})
    ckpt = tmp_path / "ckpt.csv"
    _, llm = _llm_que_esgota(1, ProviderUsageLimitError("usage limit"))

    with pytest.warns(UserWarning, match="interrompida"):
        _roda(df, llm, parallel_requests=paralelo, batch_size=100, checkpoint_path=ckpt)

    gravado = pd.read_csv(ckpt)
    assert (gravado["_dataframeit_status"] == "processed").sum() >= 1
    assert gravado["_dataframeit_status"].isna().sum() >= 6 - paralelo


@pytest.mark.parametrize("paralelo", [1, 2])
def test_interrupcao_no_reprocessamento_marca_as_linhas_que_ficaram_com_valor_antigo(paralelo):
    df = pd.DataFrame(
        {
            "texto": [f"t{i}" for i in range(6)],
            "campo1": ["antigo"] * 6,
            "_dataframeit_status": ["processed"] * 6,
        }
    )
    _, llm = _llm_que_esgota(2, ProviderUsageLimitError("usage limit"))

    with pytest.warns(
        UserWarning, match=r"0 linha\(s\) ficaram sem status, e \d+ não foram reprocessadas"
    ):
        saida = _roda(df, llm, reprocess_columns=["campo1"], parallel_requests=paralelo)

    detalhes = saida["_error_details"]
    antigas = saida["campo1"].eq("antigo")
    assert (saida["_dataframeit_status"] == "processed").all()
    assert antigas.sum() >= 1
    assert detalhes[antigas].str.startswith("Reprocessamento interrompido").all()
    assert detalhes[~antigas].isna().all()


def test_linha_com_erro_que_ia_ser_refeita_volta_a_ficar_pendente():
    df = pd.DataFrame(
        {
            "texto": ["a", "b", "c"],
            "_dataframeit_status": ["error", "error", "error"],
            "_error_details": ["x", "x", "x"],
        }
    )
    _, llm = _llm_que_esgota(1, ProviderUsageLimitError("usage limit"))

    with pytest.warns(UserWarning, match=r"2 linha\(s\) ficaram sem status"):
        saida = _roda(df, llm, resume=False)

    assert saida["_dataframeit_status"].tolist()[0] == "processed"
    assert saida["_dataframeit_status"].isna().tolist()[1:] == [True, True]
    assert saida["_error_details"].isna().tolist()[1:] == [True, True]


@pytest.mark.parametrize(
    ("entrada", "trecho"),
    [
        (lambda: ["a", "b", "c"], "só um checkpoint_path permite retomar"),
        (lambda: pd.DataFrame({"texto": ["a", "b", "c"]}), "resume=True sobre esta saída"),
    ],
)
def test_aviso_diz_como_retomar_conforme_a_entrada(entrada, trecho):
    _, llm = _llm_que_esgota(1, ProviderUsageLimitError("usage limit"))

    with pytest.warns(UserWarning, match=trecho):
        _roda(entrada(), llm)


def test_aviso_com_checkpoint_manda_usar_o_mesmo_arquivo(tmp_path):
    _, llm = _llm_que_esgota(1, ProviderUsageLimitError("usage limit"))

    with pytest.warns(UserWarning, match="mesmo checkpoint_path"):
        _roda(
            pd.DataFrame({"texto": ["a", "b"]}),
            llm,
            batch_size=5,
            checkpoint_path=tmp_path / "c.csv",
        )


def test_falha_ao_despachar_a_linha_interrompe_e_grava_o_checkpoint(tmp_path):
    df = pd.DataFrame({"texto": [f"linha{i}" for i in range(6)]})
    ckpt = tmp_path / "ckpt.csv"
    _, llm = _llm_que_esgota(100, ProviderAbortError("nunca"))
    submit = ThreadPoolExecutor.submit
    despachos = []

    def submit_que_esgota(self, *args, **kwargs):
        despachos.append(1)
        if len(despachos) > 2:
            msg = "can't start new thread"
            raise RuntimeError(msg)
        return submit(self, *args, **kwargs)

    with (
        patch.object(ThreadPoolExecutor, "submit", submit_que_esgota),
        pytest.warns(
            UserWarning,
            match=r"interrompida: RuntimeError: can't start new thread\. 4 linha\(s\) ficaram sem status",
        ),
    ):
        saida = _roda(df, llm, parallel_requests=2, batch_size=100, checkpoint_path=ckpt)

    assert saida["_dataframeit_status"].tolist()[:2] == ["processed", "processed"]
    assert saida["_dataframeit_status"].isna().tolist()[2:] == [True] * 4
    gravado = pd.read_csv(ckpt)
    assert gravado["_dataframeit_status"].tolist()[:2] == ["processed", "processed"]
    assert gravado["_dataframeit_status"].isna().sum() == 4


class _InterrupcaoIlegivel(ProviderAbortError):
    def __str__(self):
        msg = "sem texto"
        raise RuntimeError(msg)


@pytest.mark.parametrize("paralelo", [1, 2])
def test_interrupcao_com_texto_ilegivel_ainda_interrompe(paralelo):
    df = pd.DataFrame({"texto": [f"linha{i}" for i in range(4)]})
    _, llm = _llm_que_esgota(0, _InterrupcaoIlegivel())

    with pytest.warns(
        UserWarning, match=r"interrompida: _InterrupcaoIlegivel: _InterrupcaoIlegivel\(\)\. 4 linha"
    ):
        saida = _roda(df, llm, parallel_requests=paralelo)

    assert saida["_dataframeit_status"].isna().all()
