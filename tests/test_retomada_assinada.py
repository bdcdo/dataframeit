"""Retomada automática pelo checkpoint: só o arquivo desta execução é retomado."""

import contextlib
import warnings
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from pydantic import BaseModel

from dataframeit import ProviderUsageLimitError, dataframeit, read_df
from dataframeit.core import _field_hashes


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


class Dois(BaseModel):
    campo1: str | None = None
    campo2: str | None = None


def _llm_dois(limite, erro):
    chamadas = []

    def llm(*args, **kwargs):
        chamadas.append(1)
        if len(chamadas) > limite:
            raise erro
        n = len(chamadas)
        return {"data": {"campo1": f"llm{n}", "campo2": f"llm{n}"}, "usage": None}

    return chamadas, llm


@pytest.mark.parametrize("reprocessar", [None, ["campo1", "campo2"]])
def test_retomada_da_o_mesmo_resultado_que_a_execucao_sem_interrupcao(tmp_path, reprocessar):
    """Valor que a entrada já trazia antes da execução não vence a resposta paga."""
    ckpt = tmp_path / "ckpt.csv"
    df = pd.DataFrame({"texto": ["a", "b"], "campo1": ["humano", None]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _roda(
            df,
            _llm_dois(1, ProviderUsageLimitError("limite"))[1],
            modelo=Dois,
            batch_size=10,
            checkpoint_path=ckpt,
            reprocess_columns=reprocessar,
        )
        final = _roda(
            df,
            _llm_dois(100, SystemExit())[1],
            modelo=Dois,
            batch_size=10,
            checkpoint_path=ckpt,
            reprocess_columns=reprocessar,
        )

    assert final.loc[0, "campo1"] == "llm1"
    assert final.loc[0, "campo2"] == "llm1"


def test_correcao_que_apaga_valor_prevalece(tmp_path):
    ckpt = tmp_path / "ckpt.csv"
    df = pd.DataFrame({"texto": ["a", "b"]})
    saida = _roda(
        df, _llm_dois(100, SystemExit())[1], modelo=Dois, batch_size=10, checkpoint_path=ckpt
    )
    corrigida = saida.drop(columns=[c for c in saida.columns if c.startswith("_")], errors="ignore")
    corrigida.loc[0, "campo2"] = None

    chamadas, llm = _llm_dois(100, SystemExit())
    final = _roda(corrigida, llm, modelo=Dois, batch_size=10, checkpoint_path=ckpt)

    assert len(chamadas) == 0
    assert final.loc[0, "campo2"] is None or pd.isna(final.loc[0, "campo2"])
    assert final.loc[1, "campo2"] == "llm2"


def test_coluna_de_rotulo_nao_textual_nao_ganha_duplicata(tmp_path):
    ckpt = tmp_path / "ckpt.csv"
    df = pd.DataFrame({"texto": ["a", "b"], 0: [10, 20]})
    with pytest.warns(UserWarning, match="interrompida"):
        _roda(
            df, _llm(1, ProviderUsageLimitError("limite"))[1], batch_size=10, checkpoint_path=ckpt
        )

    final = _roda(df, _llm(100, SystemExit())[1], batch_size=10, checkpoint_path=ckpt)

    assert 0 in final.columns
    assert "0" not in final.columns


def test_objeto_com_endereco_de_memoria_em_model_kwargs_ainda_retoma(tmp_path):
    class Callback:
        pass

    ckpt = tmp_path / "ckpt.csv"
    df = pd.DataFrame({"texto": ["a", "b"]})
    with pytest.warns(UserWarning, match="interrompida"):
        _roda(
            df,
            _llm(1, ProviderUsageLimitError("limite"))[1],
            batch_size=10,
            checkpoint_path=ckpt,
            model_kwargs={"callbacks": [Callback()]},
        )

    chamadas, llm = _llm(100, SystemExit())
    _roda(df, llm, batch_size=10, checkpoint_path=ckpt, model_kwargs={"callbacks": [Callback()]})

    assert len(chamadas) == 1


def test_checkpoint_que_nao_rele_com_as_linhas_da_entrada_recomeca(tmp_path, monkeypatch):
    """O CSV nem sempre relê o que gravou; a retomada não casa linhas deslocadas."""
    ckpt = tmp_path / "ckpt.csv"
    df = pd.DataFrame({"texto": ["a", "b", "c"]})
    with pytest.warns(UserWarning, match="interrompida"):
        _roda(
            df, _llm(1, ProviderUsageLimitError("limite"))[1], batch_size=10, checkpoint_path=ckpt
        )
    original = read_df
    monkeypatch.setattr(
        "dataframeit.core.read_df",
        lambda *a, **k: pd.concat([original(*a, **k)] * 2, ignore_index=True),
    )

    chamadas, llm = _llm(100, SystemExit())
    with pytest.warns(UserWarning, match="não relê com a coluna de status e as linhas"):
        _roda(df, llm, batch_size=10, checkpoint_path=ckpt)

    assert len(chamadas) == 3


def test_aviso_de_falha_da_assinatura_aponta_para_quem_chamou(tmp_path):
    ckpt = tmp_path / "ckpt.csv"
    original = Path.write_text

    def falha(self, *args, **kwargs):
        if self.name.endswith(".dataframeit.json.tmp"):
            msg = "disco cheio"
            raise OSError(msg)
        return original(self, *args, **kwargs)

    with patch.object(Path, "write_text", falha), warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        _roda(
            pd.DataFrame({"texto": ["a"]}),
            _llm(100, SystemExit())[1],
            batch_size=10,
            checkpoint_path=ckpt,
        )

    assinatura = [a for a in avisos if "assinatura do checkpoint" in str(a.message)]
    assert assinatura
    assert all(not a.filename.endswith("core.py") for a in assinatura)


def test_hash_dos_campos_da_entrada_distingue_lista_de_ausencia():
    class ComLista(BaseModel):
        tags: list[str] | None = None

    com_lista = pd.DataFrame({"texto": ["a", "b"], "tags": [["x"], None]})
    vazia = pd.DataFrame({"texto": ["a", "b"], "tags": [[], None]})

    hashes = _field_hashes(com_lista, ComLista)

    assert hashes[0] != hashes[1]
    assert _field_hashes(vazia, ComLista)[0] != hashes[1]
    assert hashes == _field_hashes(com_lista.copy(), ComLista)
