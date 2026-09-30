"""Falha que impede as linhas seguintes interrompe a execução sem gastá-las."""

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
