"""Retomada automática pelo checkpoint: só o arquivo desta execução é retomado."""

import contextlib
import warnings
from unittest.mock import patch

import pandas as pd
import pytest
from pydantic import BaseModel

from dataframeit import ProviderUsageLimitError, dataframeit, read_df


class Modelo(BaseModel):
    campo1: str


class ModeloOpcional(BaseModel):
    campo1: str | None = None
    campo2: str | None = None


def _llm(limite, erro, prefixo="v"):
    chamadas = []

    def llm(*args, **kwargs):
        chamadas.append(1)
        if len(chamadas) > limite:
            raise erro
        return {
            "data": {"campo1": f"{prefixo}{len(chamadas)}"},
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }

    return chamadas, llm


def _roda(df, llm, prompt="{texto}", modelo=Modelo, **kwargs):
    with (
        patch("dataframeit.core.call_langchain", side_effect=llm),
        patch("dataframeit.core.validate_provider_dependencies"),
    ):
        return dataframeit(df.copy(), questions=modelo, prompt=prompt, **kwargs)


@pytest.mark.parametrize("interrupcao", [KeyboardInterrupt(), ProviderUsageLimitError("limite")])
def test_execucao_recusada_e_interrompida_nao_adota_o_checkpoint_antigo(tmp_path, interrupcao):
    """A assinatura só é gravada junto do checkpoint que ela descreve."""
    ckpt = tmp_path / "ckpt.csv"
    df = pd.DataFrame({"texto": ["a", "b", "c"]})
    _roda(df, _llm(100, SystemExit())[1], prompt="P1 {texto}", batch_size=10, checkpoint_path=ckpt)

    with (
        pytest.warns(UserWarning, match="outra configuração"),
        contextlib.suppress(KeyboardInterrupt),
    ):
        _roda(df, _llm(0, interrupcao)[1], prompt="P2 {texto}", batch_size=10, checkpoint_path=ckpt)

    # Com o limite de uso, B grava o próprio checkpoint, com as linhas pendentes,
    # e C retoma dele; com Ctrl-C, o arquivo ainda é o de A, e C recomeça.
    chamadas, llm = _llm(100, SystemExit(), prefixo="P2-")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        saida = _roda(df, llm, prompt="P2 {texto}", batch_size=10, checkpoint_path=ckpt)

    assert len(chamadas) == 3
    assert saida["campo1"].tolist() == ["P2-1", "P2-2", "P2-3"]


@pytest.mark.parametrize("outra", [["x", "y", "z"], ["x", "y"]])
def test_entrada_trocada_e_interrompida_nao_recebe_respostas_da_anterior(tmp_path, outra):
    ckpt = tmp_path / "ckpt.csv"
    _roda(
        pd.DataFrame({"texto": ["a", "b", "c"]}),
        _llm(100, SystemExit())[1],
        batch_size=10,
        checkpoint_path=ckpt,
    )
    df2 = pd.DataFrame({"texto": outra})

    with pytest.warns(UserWarning, match="outra entrada"), pytest.raises(KeyboardInterrupt):
        _roda(df2, _llm(0, KeyboardInterrupt())[1], batch_size=10, checkpoint_path=ckpt)

    chamadas, llm = _llm(100, SystemExit(), prefixo="novo")
    with pytest.warns(UserWarning, match="outra entrada"):
        saida = _roda(df2, llm, batch_size=10, checkpoint_path=ckpt)

    assert len(chamadas) == len(outra)
    assert saida["campo1"].str.startswith("novo").all()


def test_arquivo_que_nao_e_o_da_assinatura_nao_e_retomado(tmp_path):
    """Processo morto entre a gravação do checkpoint e a da assinatura."""
    ckpt = tmp_path / "ckpt.csv"
    df = pd.DataFrame({"texto": ["a", "b"]})
    _roda(df, _llm(100, SystemExit())[1], batch_size=10, checkpoint_path=ckpt)
    ckpt.write_text(ckpt.read_text(encoding="utf-8").replace("v1", "adulterado"), encoding="utf-8")

    chamadas, llm = _llm(100, SystemExit())
    with pytest.warns(UserWarning, match="não é o arquivo que a assinatura"):
        _roda(df, llm, batch_size=10, checkpoint_path=ckpt)

    assert len(chamadas) == 2


def test_status_column_trocado_avisa_e_recomeca(tmp_path):
    ckpt = tmp_path / "ckpt.csv"
    df = pd.DataFrame({"texto": ["a", "b", "c"]})
    with pytest.raises(SystemExit):
        _roda(df, _llm(1, SystemExit())[1], batch_size=1, checkpoint_path=ckpt, status_column="st")

    chamadas, llm = _llm(100, SystemExit())
    with pytest.warns(UserWarning, match="outra configuração"):
        _roda(df, llm, batch_size=1, checkpoint_path=ckpt)

    assert len(chamadas) == 3


def test_model_kwargs_com_chaves_int_e_str_nao_derruba_a_chamada(tmp_path):
    ckpt = tmp_path / "ckpt.csv"
    df = pd.DataFrame({"texto": ["a"]})
    kwargs = {"logit_bias": {50256: -100, "7": 1}, "stop": ["fim", "."]}
    _roda(df, _llm(100, SystemExit())[1], batch_size=1, checkpoint_path=ckpt, model_kwargs=kwargs)

    chamadas, llm = _llm(100, SystemExit())
    _roda(df, llm, batch_size=1, checkpoint_path=ckpt, model_kwargs=kwargs)

    assert len(chamadas) == 0


@pytest.mark.parametrize("paralelo", [1, 2])
def test_marca_de_reprocessamento_interrompido_chega_ao_checkpoint(tmp_path, paralelo):
    ckpt = tmp_path / "ckpt.csv"
    df = pd.DataFrame({"texto": [f"t{i}" for i in range(4)]})
    saida = _roda(
        df, _llm(100, SystemExit(), prefixo="velho")[1], batch_size=1, checkpoint_path=ckpt
    )

    with pytest.warns(UserWarning, match="não foram reprocessadas"):
        _roda(
            saida,
            _llm(paralelo, ProviderUsageLimitError("limite"), prefixo="novo")[1],
            batch_size=1,
            checkpoint_path=ckpt,
            reprocess_columns=["campo1"],
            parallel_requests=paralelo,
        )

    gravado = read_df(str(ckpt), Modelo)
    velhas = gravado["campo1"].str.startswith("velho")
    assert velhas.any()
    assert (
        gravado.loc[velhas, "_error_details"].str.startswith("Reprocessamento interrompido").all()
    )

    chamadas, llm = _llm(100, SystemExit())
    final = _roda(df, llm, batch_size=1, checkpoint_path=ckpt)
    assert len(chamadas) == 0
    assert final.loc[final["campo1"].str.startswith("velho"), "_error_details"].notna().all()


def test_valor_que_a_entrada_ja_traz_prevalece_sobre_o_checkpoint(tmp_path):
    """Correção à mão numa saída concluída não é desfeita pela retomada."""
    ckpt = tmp_path / "ckpt.csv"
    df = pd.DataFrame({"texto": ["a", "b"], "campo1": [None, None], "campo2": [None, None]})
    saida = _roda(
        df, _llm(100, SystemExit())[1], modelo=ModeloOpcional, batch_size=1, checkpoint_path=ckpt
    )
    corrigida = saida.drop(columns=[c for c in saida.columns if c.startswith("_")], errors="ignore")
    corrigida.loc[0, "campo1"] = "corrigido à mão"

    chamadas, llm = _llm(100, SystemExit())
    final = _roda(corrigida, llm, modelo=ModeloOpcional, batch_size=1, checkpoint_path=ckpt)

    assert len(chamadas) == 0
    assert final["campo1"].tolist() == ["corrigido à mão", "v2"]
