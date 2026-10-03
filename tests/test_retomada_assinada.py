"""Retomada automática pelo checkpoint: só o arquivo desta execução é retomado."""

import contextlib
import datetime
import warnings
from enum import Enum
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from dataframeit import ProviderUsageLimitError, dataframeit, read_df
from dataframeit.core import (
    _run_signature,
    _same_json,
    _save_checkpoint,
    _validate_processed_rows,
)


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


def test_linha_concluida_vem_do_checkpoint_mesmo_com_valor_na_entrada(tmp_path):
    """Como na execução sem interrupção, que grava a resposta por cima da entrada."""
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
    assert final["campo1"].tolist() == ["v1", "v2"]


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
    """Sem coluna de status, a saída é entrada nova, e a linha concluída vem do checkpoint."""
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


def test_conjunto_em_model_kwargs_assina_em_ordem_fixa():
    """A ordem de iteração de um set muda com a semente de hash do processo."""

    def assinatura(stop):
        return _run_signature(
            Modelo,
            "{texto}",
            provider="openai",
            model="m",
            model_kwargs={"stop": stop},
            search_config=None,
            text_column="texto",
            status_col="_dataframeit_status",
        )

    assert assinatura({"fim", "alfa", "zeta"}) == assinatura(["alfa", "fim", "zeta"])
    assert assinatura({1, "1", "a"}) == assinatura({"a", "1", 1})


def test_prazo_em_model_kwargs_nao_entra_na_assinatura():
    """Trocar o prazo do turno não muda a resposta, e não pode impedir a retomada."""

    def assinatura(model_kwargs):
        return _run_signature(
            Modelo,
            "{texto}",
            provider="codex",
            model="m",
            model_kwargs=model_kwargs,
            search_config=None,
            text_column="texto",
            status_col="_dataframeit_status",
        )

    base = assinatura({"effort": "low"})
    assert assinatura({"effort": "low", "timeout": 30}) == base
    assert assinatura({"effort": "low", "timeout": None}) == base
    assert assinatura({"effort": "high", "timeout": 30}) != base


def test_linha_com_erro_no_checkpoint_tambem_vem_dele(tmp_path):
    ckpt = tmp_path / "ckpt.csv"
    df = pd.DataFrame({"texto": ["a", "b", "c"], "campo1": ["m1", "m2", "m3"]})

    def falha_em_b(texto, *args, **kwargs):
        if texto.endswith("b"):
            msg = "resposta ruim"
            raise ValueError(msg)
        return {"data": {"campo1": f"llm-{texto[-1]}"}, "usage": None}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _roda(df, falha_em_b, batch_size=1, checkpoint_path=ckpt, max_retries=1, base_delay=0)
    corrigida = df.copy()
    corrigida.loc[1, "campo1"] = "à mão"

    chamadas, llm = _llm(100, SystemExit())
    final = _roda(corrigida, llm, batch_size=1, checkpoint_path=ckpt)

    assert len(chamadas) == 0
    assert final["campo1"].tolist() == ["llm-a", "m2", "llm-c"]


@pytest.mark.parametrize("status_column", [None, "meu_status"])
def test_saida_com_a_coluna_de_status_devolvida_mantem_a_correcao(tmp_path, status_column):
    ckpt = tmp_path / "ckpt.csv"
    df = pd.DataFrame({"texto": ["a", "b", "c"]})
    saida = _roda(
        df,
        _llm(100, SystemExit())[1],
        batch_size=1,
        checkpoint_path=ckpt,
        status_column=status_column,
    )
    saida.loc[1, "campo1"] = "corrigido"
    saida[status_column or "_dataframeit_status"] = "processed"

    chamadas, llm = _llm(100, SystemExit())
    final = _roda(saida, llm, batch_size=1, checkpoint_path=ckpt, status_column=status_column)

    assert len(chamadas) == 0
    assert final["campo1"].tolist() == ["v1", "corrigido", "v3"]


class ComDict(BaseModel):
    extra: dict


class ComListaDeDicts(BaseModel):
    extra: list[dict]


_ESTRUTURAS = {
    ComDict: {"a": {"x": 1}, "b": {"y": 2}},
    ComListaDeDicts: {"a": [{"x": 1}], "b": [{"y": 2}, {"x": 3, "z": "t"}]},
}


@pytest.mark.parametrize("modelo", list(_ESTRUTURAS))
def test_estrutura_relida_do_checkpoint_parquet_e_a_gravada(tmp_path, modelo):
    """O parquet gravaria dicts como struct, com as chaves de todas as linhas."""
    pytest.importorskip("pyarrow")
    respostas = _ESTRUTURAS[modelo]
    ckpt = tmp_path / "ckpt.parquet"
    chamadas = []

    def llm(text, *args, **kwargs):
        chamadas.append(text)
        return {"data": {"extra": respostas[text]}, "usage": None}

    df = pd.DataFrame({"texto": ["a", "b"]})
    direta = _roda(df, llm, modelo=modelo, batch_size=1, checkpoint_path=ckpt)
    relido = read_df(str(ckpt), modelo)
    relido_sem_modelo = read_df(str(ckpt))
    retomada = _roda(df, llm, modelo=modelo, batch_size=1, checkpoint_path=ckpt)

    esperado = [respostas["a"], respostas["b"]]
    assert list(direta["extra"]) == esperado
    assert list(relido["extra"]) == esperado
    assert list(relido_sem_modelo["extra"]) == esperado
    assert list(retomada["extra"]) == esperado
    assert chamadas == ["a", "b"]


@pytest.mark.parametrize("formato", ["csv", "parquet"])
def test_campo_lista_da_saida_polars_volta_igual_do_checkpoint(tmp_path, formato):
    """A saída polars traz o campo lista como array do numpy, ao lado das listas novas."""
    pl = pytest.importorskip("polars")
    pytest.importorskip("pyarrow")
    ckpt = tmp_path / f"ckpt.{formato}"
    df = pl.DataFrame({"texto": ["a", "b", "c"]})
    with pytest.warns(UserWarning, match="interrompida"):
        saida = _roda(
            df,
            _llm_lista(1, ProviderUsageLimitError("limite"))[1],
            modelo=ComLista,
            batch_size=1,
            checkpoint_path=ckpt,
        )

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Falha ao gravar o checkpoint")
        _roda(
            saida,
            _llm_lista(100, SystemExit())[1],
            modelo=ComLista,
            batch_size=1,
            checkpoint_path=ckpt,
        )

    assert list(read_df(str(ckpt), ComLista)["tags"]) == [["t1", "u"], ["t1", "u"], ["t2", "u"]]


class Itens(BaseModel):
    itens: list[str]


class ComAninhado(BaseModel):
    interno: Itens


@pytest.mark.parametrize("formato", ["csv", "parquet"])
def test_array_dentro_de_modelo_aninhado_volta_igual_do_checkpoint(tmp_path, formato):
    """A saída polars traz a lista do modelo aninhado como array dentro do dict."""
    pl = pytest.importorskip("polars")
    pytest.importorskip("pyarrow")
    ckpt = tmp_path / f"ckpt.{formato}"
    chamadas = []
    limite = ProviderUsageLimitError("limite")

    def llm(text, *args, **kwargs):
        chamadas.append(text)
        if chamadas == ["a", "b"]:
            raise limite
        return {"data": {"interno": {"itens": [text, "z"]}}, "usage": None}

    df = pl.DataFrame({"texto": ["a", "b"]})
    with pytest.warns(UserWarning, match="interrompida"):
        saida = _roda(df, llm, modelo=ComAninhado, batch_size=1, checkpoint_path=ckpt)
    _roda(saida, llm, modelo=ComAninhado, batch_size=1, checkpoint_path=ckpt)

    relido = read_df(str(ckpt), ComAninhado)
    assert list(relido["interno"]) == [{"itens": ["a", "z"]}, {"itens": ["b", "z"]}]


class ComData(BaseModel):
    quando: dict[str, datetime.date]


class ComTupla(BaseModel):
    par: tuple[int, str]


class ComChaveInt(BaseModel):
    mapa: dict[int, str]


_TIPADOS = {
    ComData: ("quando", lambda texto: {texto: datetime.date(2024, 1, 2)}),
    ComTupla: ("par", lambda texto: (1, texto)),
    ComChaveInt: ("mapa", lambda texto: {1: texto, 10: texto, 2: texto}),
}


@pytest.mark.parametrize("formato", ["csv", "parquet"])
@pytest.mark.parametrize("modelo", list(_TIPADOS))
def test_valor_retomado_do_checkpoint_tem_o_tipo_do_campo(tmp_path, modelo, formato):
    """Data, tupla e chave int, que o JSON não guarda, voltam da retomada com o tipo declarado."""
    pytest.importorskip("pyarrow")
    campo, resposta = _TIPADOS[modelo]
    ckpt = tmp_path / f"ckpt.{formato}"

    def llm(text, *args, **kwargs):
        return {"data": modelo(**{campo: resposta(text)}).model_dump(), "usage": None}

    df = pd.DataFrame({"texto": ["a", "b"]})
    direta = _roda(df, llm, modelo=modelo, batch_size=1, checkpoint_path=ckpt)
    _, sem_llm = _llm(0, SystemExit())
    retomada = _roda(df, sem_llm, modelo=modelo, batch_size=1, checkpoint_path=ckpt)

    assert list(retomada[campo]) == list(direta[campo])
    assert [type(v) for v in retomada[campo]] == [type(v) for v in direta[campo]]


def test_coluna_do_usuario_volta_nativa_do_checkpoint_parquet(tmp_path):
    """Só os campos de estrutura do modelo viram JSON no parquet."""
    pytest.importorskip("pyarrow")
    ckpt = tmp_path / "ckpt.parquet"
    df = pd.DataFrame({"texto": ["a", "b"], "meta": [["m1", "m2"], ["m3"]]})
    _roda(df, _llm_lista(100, SystemExit())[1], modelo=ComLista, batch_size=1, checkpoint_path=ckpt)

    relido = read_df(str(ckpt), ComLista)
    assert [list(v) for v in relido["meta"]] == [["m1", "m2"], ["m3"]]
    assert list(relido["tags"]) == [["t1", "u"], ["t2", "u"]]


class Cor(Enum):
    AZUL = "azul"


class CoresComoTexto(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    cores: list[Cor]


class CoresComoEnum(BaseModel):
    cores: list[Cor]


class Exclamado(BaseModel):
    n: list[str]

    @field_validator("n")
    @classmethod
    def exclama(cls, valor):
        return [f"{item}!" for item in valor]


class ComValidador(BaseModel):
    interno: Exclamado


class ExclamaAteDuasVezes(BaseModel):
    """A segunda passada do validador sobre o valor já validado falha."""

    n: list[str]

    @field_validator("n")
    @classmethod
    def exclama(cls, valor):
        if any(item.count("!") > 1 for item in valor):
            msg = "exclamação demais"
            raise ValueError(msg)
        return [f"{item}!" for item in valor]


class FiltraPorOutroCampo(BaseModel):
    """O validador de `itens` lê `tem_itens`, que só a linha inteira traz."""

    tem_itens: bool
    itens: list[str]

    @field_validator("itens")
    @classmethod
    def so_com_itens(cls, valor, info):
        return valor if info.data.get("tem_itens") else []


class DatasDepoisDoInicio(BaseModel):
    inicio: datetime.date
    datas: list[datetime.date]

    @model_validator(mode="after")
    def depois_do_inicio(self):
        if any(data < self.inicio for data in self.datas):
            msg = "data antes do início"
            raise ValueError(msg)
        return self


class DatasEmTexto(BaseModel):
    datas: list[datetime.date]

    @field_serializer("datas")
    def em_texto(self, valor):
        return [str(item) for item in valor]


class NomePorAlias(BaseModel):
    model_config = ConfigDict(serialize_by_alias=True, validate_by_name=True)
    nome: str = Field(alias="Nome")


class SubmodeloPorAlias(BaseModel):
    """A execução grava o submodelo pelo alias, que o model_dump de fora não reescreve."""

    subs: list[NomePorAlias]


class Convergente(BaseModel):
    """O validador muda "b" para "c" e deixa "c" como está: estável, mas não idempotente."""

    n: list[str]

    @field_validator("n")
    @classmethod
    def avanca(cls, valor):
        proximo = {"a": "b", "b": "c"}
        return [proximo.get(item, item) for item in valor]


_POR_EXECUCAO = {
    SubmodeloPorAlias: lambda texto: {"subs": [{"Nome": texto}]},
    Convergente: lambda texto: {"n": ["a"]},
    CoresComoTexto: lambda texto: {"cores": [Cor.AZUL]},
    CoresComoEnum: lambda texto: {"cores": [Cor.AZUL]},
    ComValidador: lambda texto: {"interno": {"n": [texto]}},
    ExclamaAteDuasVezes: lambda texto: {"n": [texto]},
    FiltraPorOutroCampo: lambda texto: {"tem_itens": True, "itens": [texto]},
    DatasDepoisDoInicio: lambda texto: {
        "inicio": datetime.date(2024, 1, 1),
        "datas": [datetime.date(2024, 1, 2)],
    },
    DatasEmTexto: lambda texto: {"datas": [datetime.date(2024, 1, 2)]},
}


@pytest.mark.parametrize("formato", ["csv", "parquet"])
@pytest.mark.parametrize("modelo", list(_POR_EXECUCAO))
def test_retomadas_seguidas_devolvem_o_que_a_execucao_gravou(tmp_path, modelo, formato):
    """Cada execução processa uma linha e é interrompida; a próxima retoma do checkpoint."""
    pytest.importorskip("pyarrow")
    ckpt = tmp_path / f"ckpt.{formato}"
    df = pd.DataFrame({"texto": ["a", "b", "c"]})

    def resposta(text):
        return modelo(**_POR_EXECUCAO[modelo](text)).model_dump()

    direta = _roda(df, lambda text, *a, **k: {"data": resposta(text), "usage": None}, modelo=modelo)
    for _ in range(2):
        with pytest.warns(UserWarning, match="interrompida"):
            _roda(
                df,
                _uma_linha_por_execucao(resposta),
                modelo=modelo,
                batch_size=1,
                checkpoint_path=ckpt,
            )
    final = _roda(
        df, _uma_linha_por_execucao(resposta), modelo=modelo, batch_size=1, checkpoint_path=ckpt
    )

    for campo in modelo.model_fields:
        assert list(final[campo]) == list(direta[campo])


def test_campo_excluido_do_dump_nao_derruba_a_retomada(tmp_path):
    """Campo com exclude=True não sai no model_dump da linha validada."""

    class ComOculto(BaseModel):
        tags: list[str]
        oculto: list[str] = Field(default_factory=list, exclude=True)

    df = pd.DataFrame(
        {
            "texto": ["a"],
            "tags": [["x"]],
            "oculto": [["y"]],
            "_dataframeit_status": ["processed"],
        }
    )
    saida = _roda(df, _llm(0, SystemExit())[1], modelo=ComOculto)

    assert list(saida["tags"]) == [["x"]]
    assert list(saida["oculto"]) == [["y"]]


def test_valor_que_o_json_nao_compara_fica_como_relido():
    """Chave Enum não vira JSON, e a volta ao tipo não tem com o que comparar."""

    class PorCor(BaseModel):
        contagem: dict[Cor, int]

    df = pd.DataFrame(
        {"texto": ["a"], "contagem": [{"azul": 1}], "_dataframeit_status": ["processed"]}
    )
    saida = _roda(df, _llm(0, SystemExit())[1], modelo=PorCor)

    assert list(saida["contagem"]) == [{"azul": 1}]


def test_campo_excluido_e_ausente_nao_derruba_a_retomada():
    """Sem a coluna, o campo excluído do dump não tem valor a completar."""

    class ComOculto(BaseModel):
        tags: list[str]
        oculto: list[str] = Field(default_factory=list, exclude=True)

    df = pd.DataFrame({"texto": ["a"], "tags": [["x"]], "_dataframeit_status": ["processed"]})
    saida = _roda(df, _llm(0, SystemExit())[1], modelo=ComOculto)

    assert list(saida["tags"]) == [["x"]]


class AlternaNumero(BaseModel):
    """O validador troca int por float e float por int a cada passada."""

    n: list[int | float]

    @field_validator("n")
    @classmethod
    def alterna(cls, valor):
        return [float(item) if isinstance(item, int) else int(item) for item in valor]


@pytest.mark.parametrize("formato", ["csv", "parquet"])
def test_volta_ao_tipo_distingue_int_de_float(tmp_path, formato):
    """1 e 1.0 são o mesmo valor em Python, mas não o mesmo JSON."""
    pytest.importorskip("pyarrow")
    ckpt = tmp_path / f"ckpt.{formato}"
    df = pd.DataFrame({"texto": ["a"]})

    def llm(text, *args, **kwargs):
        return {"data": AlternaNumero(n=[1]).model_dump(), "usage": None}

    direta = _roda(df, llm, modelo=AlternaNumero, batch_size=1, checkpoint_path=ckpt)
    retomada = _roda(
        df, _llm(0, SystemExit())[1], modelo=AlternaNumero, batch_size=1, checkpoint_path=ckpt
    )

    assert [type(item) for item in retomada["n"][0]] == [type(item) for item in direta["n"][0]]


def test_estrutura_funda_demais_para_o_json_fica_como_relida():
    profunda: list = []
    for _ in range(100_000):
        profunda = [profunda]

    assert _same_json(profunda, profunda) is False


def _llm_fixo(valor, limite):
    chamadas = []
    limite_de_uso = ProviderUsageLimitError("limite")

    def llm(text, *args, **kwargs):
        chamadas.append(text)
        if len(chamadas) > limite:
            raise limite_de_uso
        return {"data": {"campo1": valor}, "usage": None}

    return chamadas, llm


def test_texto_com_cara_de_numero_volta_como_texto_da_retomada_csv(tmp_path):
    """O checkpoint é relido com o modelo, que lê os campos de texto como texto."""
    ckpt = tmp_path / "ckpt.csv"
    df = pd.DataFrame({"texto": ["a", "b"]})
    with pytest.warns(UserWarning, match="interrompida"):
        _roda(df, _llm_fixo("001", 1)[1], batch_size=1, checkpoint_path=ckpt)

    chamadas, llm = _llm_fixo("002", 100)
    final = _roda(df, llm, batch_size=1, checkpoint_path=ckpt)

    assert list(final["campo1"]) == ["001", "002"]
    assert chamadas == ["b"]


def test_campo_do_modelo_vazio_em_float_recebe_o_texto_do_checkpoint(tmp_path):
    """Coluna do modelo só com NaN chega como float, e o texto relido cabe nela."""
    ckpt = tmp_path / "ckpt.csv"
    df = pd.DataFrame({"texto": ["a", "b"], "campo1": [np.nan, np.nan]})
    with pytest.warns(UserWarning, match="interrompida"):
        _roda(df, _llm_fixo("x", 1)[1], batch_size=1, checkpoint_path=ckpt)

    final = _roda(df, _llm_fixo("y", 100)[1], batch_size=1, checkpoint_path=ckpt)

    assert list(final["campo1"]) == ["x", "y"]


def test_assinatura_que_nao_e_objeto_json_recomeca(tmp_path):
    ckpt = tmp_path / "ckpt.csv"
    df = pd.DataFrame({"texto": ["a"]})
    _roda(df, _llm_fixo("x", 100)[1], batch_size=1, checkpoint_path=ckpt)
    Path(f"{ckpt}.dataframeit.json").write_text("[1, 2]", encoding="utf-8")

    chamadas, llm = _llm_fixo("y", 100)
    with pytest.warns(UserWarning, match="não tem a assinatura"):
        final = _roda(df, llm, batch_size=1, checkpoint_path=ckpt)

    assert chamadas == ["a"]
    assert list(final["campo1"]) == ["y"]


def _uma_linha_por_execucao(resposta):
    """LLM que responde uma linha e interrompe a execução na seguinte."""
    chamadas = []
    limite = ProviderUsageLimitError("limite")

    def llm(text, *args, **kwargs):
        chamadas.append(text)
        if len(chamadas) > 1:
            raise limite
        return {"data": resposta(text), "usage": None}

    return llm


class Estrito(BaseModel):
    model_config = ConfigDict(strict=True)
    datas: list[datetime.date]
    par: tuple[int, str]
    cores: list[Cor]
    dia: datetime.date
    quando: datetime.datetime
    cor: Cor
    n: int
    ok: bool


class Escalares(BaseModel):
    dia: datetime.date
    quando: datetime.datetime
    cor: Cor
    n: int
    ok: bool


_RESPOSTA_TIPADA = {
    "datas": [datetime.date(2024, 1, 2)],
    "par": (1, "a"),
    "cores": [Cor.AZUL],
    "dia": datetime.date(2024, 1, 2),
    "quando": datetime.datetime(2024, 1, 2, 3, 4, 5),  # noqa: DTZ001 (o XLSX não guarda fuso)
    "cor": Cor.AZUL,
    "n": 3,
    "ok": True,
}


@pytest.mark.parametrize("formato", ["csv", "xlsx", "parquet"])
@pytest.mark.parametrize("modelo", [Estrito, Escalares])
def test_retomada_devolve_o_tipo_declarado_em_todo_campo(tmp_path, modelo, formato):
    """Strict ou não, todo campo volta do checkpoint parcial como a execução direta o dá (#178).

    A coluna com linha pendente devolve o int como float, o XLSX devolve a
    data como datetime e o bool como float, e o CSV devolve data e Enum como
    texto.
    """
    pytest.importorskip("pyarrow")
    if formato == "xlsx":
        pytest.importorskip("openpyxl")
    ckpt = tmp_path / f"ckpt.{formato}"
    df = pd.DataFrame({"texto": ["a", "b", "c"]})

    def resposta(text):
        return modelo(
            **{campo: _RESPOSTA_TIPADA[campo] for campo in modelo.model_fields}
        ).model_dump()

    direta = _roda(df, lambda text, *a, **k: {"data": resposta(text), "usage": None}, modelo=modelo)
    for _ in range(2):
        with pytest.warns(UserWarning, match="interrompida"):
            _roda(
                df,
                _uma_linha_por_execucao(resposta),
                modelo=modelo,
                batch_size=1,
                checkpoint_path=ckpt,
            )
    final = _roda(
        df, _uma_linha_por_execucao(resposta), modelo=modelo, batch_size=1, checkpoint_path=ckpt
    )

    for campo in modelo.model_fields:
        assert list(final[campo]) == list(direta[campo])
        assert [type(v) for v in final[campo]] == [type(v) for v in direta[campo]]


class EstritoComNumero(BaseModel):
    model_config = ConfigDict(strict=True)
    datas: list[datetime.date]
    n: int


def test_valor_que_o_strict_recusa_pelo_conteudo_continua_acusado():
    """Sem strict, "3" vira 3; o campo só é aceito quando o valor validado diz o mesmo."""
    df = pd.DataFrame(
        {"datas": ['["2024-01-02"]'], "n": ["3"], "_dataframeit_status": ["processed"]}
    )

    incompativeis, _ = _validate_processed_rows(
        df, "_dataframeit_status", EstritoComNumero, {"datas"}
    )

    assert incompativeis == ["n"]


def test_campo_recusado_tambem_sem_strict_e_o_unico_acusado():
    """A data em texto, que só o strict recusa, não entra no reprocess_columns pedido."""
    df = pd.DataFrame(
        {"datas": ['["2024-01-02"]'], "n": ["três"], "_dataframeit_status": ["processed"]}
    )

    incompativeis, _ = _validate_processed_rows(
        df, "_dataframeit_status", EstritoComNumero, {"datas"}
    )

    assert incompativeis == ["n"]


@pytest.mark.parametrize("formato", ["csv", "xlsx", "parquet"])
def test_checkpoint_grava_enum_escalar_pelo_valor(tmp_path, formato):
    """O CSV e o XLSX gravariam "Cor.AZUL", e o parquet recusaria a coluna."""
    pytest.importorskip("pyarrow")
    if formato == "xlsx":
        pytest.importorskip("openpyxl")
    ckpt = tmp_path / f"ckpt.{formato}"

    _save_checkpoint(pd.DataFrame({"cor": [Cor.AZUL, None]}), ckpt, structures=frozenset())

    assert read_df(str(ckpt))["cor"][0] == "azul"
