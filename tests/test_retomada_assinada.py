"""Retomada automática pelo checkpoint: só o arquivo desta execução é retomado."""

import contextlib
import json
import subprocess
import sys
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
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
        return dataframeit(
            df.clone() if hasattr(df, "clone") else df.copy(),
            questions=modelo,
            prompt=prompt,
            **kwargs,
        )


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


def test_celula_esvaziada_na_saida_volta_com_o_valor_do_checkpoint(tmp_path):
    """A célula vazia não se distingue da entrada original, que também não a trazia."""
    ckpt = tmp_path / "ckpt.csv"
    df = pd.DataFrame({"texto": ["a", "b"]})
    saida = _roda(
        df, _llm_dois(100, SystemExit())[1], modelo=Dois, batch_size=10, checkpoint_path=ckpt
    )
    saida.loc[0, "campo2"] = None

    chamadas, llm = _llm_dois(100, SystemExit())
    final = _roda(saida, llm, modelo=Dois, batch_size=10, checkpoint_path=ckpt)

    assert len(chamadas) == 0
    assert final["campo2"].tolist() == ["llm1", "llm2"]


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


class ComLista(BaseModel):
    tags: list[str] | None = None


def _llm_lista(limite, erro):
    chamadas = []

    def llm(*args, **kwargs):
        chamadas.append(1)
        if len(chamadas) > limite:
            raise erro
        return {"data": {"tags": [f"t{len(chamadas)}", "u"]}, "usage": None}

    return chamadas, llm


def test_campo_lista_em_array_do_numpy_nao_derruba_a_execucao(tmp_path):
    """O parquet relido e o polars entregam campo lista como array do numpy."""
    ckpt = tmp_path / "ckpt.csv"
    df = pd.DataFrame(
        {
            "texto": ["a", "b"],
            "tags": [np.array(["x", "y"], dtype=object), np.array([np.int64(1)], dtype=object)],
        }
    )

    chamadas, llm = _llm_lista(100, SystemExit())
    final = _roda(
        df, llm, modelo=ComLista, batch_size=10, checkpoint_path=ckpt, reprocess_columns=["tags"]
    )

    assert len(chamadas) == 2
    assert list(final.loc[0, "tags"]) == ["t1", "u"]
    como_lista = pd.DataFrame({"texto": ["a", "b"], "tags": [["x", "y"], [1]]})
    assert _field_hashes(df, ComLista) == _field_hashes(como_lista, ComLista)


def test_saida_polars_com_campo_lista_retoma_pelo_mesmo_checkpoint(tmp_path):
    pl = pytest.importorskip("polars")
    ckpt = tmp_path / "ckpt.parquet"
    pytest.importorskip("pyarrow")
    df = pl.DataFrame({"texto": ["a", "b", "c"]})
    with pytest.warns(UserWarning, match="interrompida"):
        saida = _roda(
            df,
            _llm_lista(1, ProviderUsageLimitError("limite"))[1],
            modelo=ComLista,
            batch_size=1,
            checkpoint_path=ckpt,
        )

    chamadas, llm = _llm_lista(100, SystemExit())
    final = _roda(saida, llm, modelo=ComLista, batch_size=1, checkpoint_path=ckpt)

    assert len(chamadas) == 2
    assert final["tags"].to_list() == [["t1", "u"], ["t1", "u"], ["t2", "u"]]


def _retomada_manual_e_depois_a_entrada_original(tmp_path, modelo):
    """Execução interrompida, retomada à mão pela saída e morta, e a entrada original de novo."""
    ckpt = tmp_path / "ckpt.csv"
    df = pd.DataFrame({"texto": ["a", "b", "c"], "campo1": [None, None, None]})
    with pytest.warns(UserWarning, match="interrompida"):
        saida = _roda(
            df,
            _llm(1, ProviderUsageLimitError("limite"), prefixo="llm")[1],
            modelo=modelo,
            batch_size=1,
            checkpoint_path=ckpt,
        )
    _, segunda = _llm(1, SystemExit(), prefixo="llm")
    with pytest.raises(SystemExit):
        # A segunda execução começa a contar do zero: a linha 1 recebe "llm1".
        _roda(saida, segunda, modelo=modelo, batch_size=1, checkpoint_path=ckpt)

    chamadas, llm = _llm(100, SystemExit(), prefixo="nova")
    return chamadas, _roda(df, llm, modelo=modelo, batch_size=1, checkpoint_path=ckpt)


@pytest.mark.parametrize("modelo", [Modelo, ModeloOpcional])
def test_entrada_original_depois_de_retomada_manual_guarda_as_respostas_pagas(tmp_path, modelo):
    chamadas, final = _retomada_manual_e_depois_a_entrada_original(tmp_path, modelo)

    assert len(chamadas) == 1
    assert final["campo1"].tolist() == ["llm1", "llm1", "nova1"]


def test_retomada_manual_com_outra_configuracao_grava_a_propria_entrada(tmp_path):
    ckpt = tmp_path / "ckpt.csv"
    df = pd.DataFrame({"texto": ["a", "b"], "campo1": [None, None]})
    with pytest.warns(UserWarning, match="interrompida"):
        saida = _roda(
            df,
            _llm(1, ProviderUsageLimitError("limite"))[1],
            modelo=ModeloOpcional,
            batch_size=1,
            checkpoint_path=ckpt,
        )

    _roda(
        saida,
        _llm(100, SystemExit())[1],
        prompt="outro {texto}",
        modelo=ModeloOpcional,
        batch_size=1,
        checkpoint_path=ckpt,
    )

    gravado = json.loads(Path(f"{ckpt}.dataframeit.json").read_text(encoding="utf-8"))
    assert gravado["inputs"] == _field_hashes(saida, ModeloOpcional)


def test_correcao_de_um_campo_guarda_a_resposta_paga_do_outro(tmp_path):
    ckpt = tmp_path / "ckpt.csv"
    df = pd.DataFrame({"texto": ["a", "b"], "campo1": ["humano", None], "campo2": [None, None]})
    with pytest.warns(UserWarning, match="interrompida"):
        _roda(
            df,
            _llm_dois(1, ProviderUsageLimitError("limite"))[1],
            modelo=Dois,
            batch_size=10,
            checkpoint_path=ckpt,
        )
    corrigida = df.copy()
    corrigida.loc[0, "campo1"] = "humano revisado"

    chamadas, llm = _llm_dois(100, SystemExit())
    final = _roda(corrigida, llm, modelo=Dois, batch_size=10, checkpoint_path=ckpt)

    assert len(chamadas) == 1
    assert final.loc[0, "campo1"] == "humano revisado"
    assert final.loc[0, "campo2"] == "llm1"


_HASH_DE_CONJUNTO = """
import json
import pandas as pd
from pydantic import BaseModel
from dataframeit.core import _field_hashes

class M(BaseModel):
    c: object = None

valores = [{"alfa", "beta", "gama", "delta"}, frozenset({"alfa", "beta", "gama", "delta"})]
df = pd.DataFrame({"texto": ["a", "b"], "c": pd.Series(valores, dtype=object)})
print(json.dumps(_field_hashes(df, M)))
"""


def test_hash_de_conjunto_nao_depende_da_semente_do_processo():
    saidas = {
        subprocess.run(  # noqa: S603 (roda o próprio interpretador com código fixo do teste)
            [sys.executable, "-c", _HASH_DE_CONJUNTO],
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": semente, "PATH": ""},
        ).stdout
        for semente in ("1", "2", "3", "4")
    }

    assert len(saidas) == 1
    conjunto = pd.DataFrame({"texto": ["a"], "tags": pd.Series([{"b", "a"}], dtype=object)})
    lista = pd.DataFrame({"texto": ["a"], "tags": [["a", "b"]]})
    assert _field_hashes(conjunto, ComLista) == _field_hashes(lista, ComLista)


def test_dict_com_chaves_int_e_str_num_campo_nao_derruba_a_execucao(tmp_path):
    class ComDict(BaseModel):
        mapa: dict | None = None

    df = pd.DataFrame({"texto": ["a"], "mapa": pd.Series([{1: "x", "b": "y"}], dtype=object)})

    def llm(*args, **kwargs):
        return {"data": {"mapa": {"k": "v"}}, "usage": None}

    final = _roda(
        df,
        llm,
        modelo=ComDict,
        batch_size=10,
        checkpoint_path=tmp_path / "c.csv",
        reprocess_columns=["mapa"],
    )

    assert final.loc[0, "mapa"] == {"k": "v"}
