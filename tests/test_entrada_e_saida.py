"""Entrada e saída de dataframeit(): colunas de controle, texto ausente, índice e avisos."""

import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel

from dataframeit import core, dataframeit
from dataframeit.core import _print_token_stats


class Modelo(BaseModel):
    x: str


class ModeloLista(BaseModel):
    tags: list[str]


def _responde(**campos):
    def call_langchain(text, *args, **kwargs):
        return {
            "data": {k: (v(text) if callable(v) else v) for k, v in campos.items()},
            "usage": None,
        }

    return call_langchain


def _rodar(df, questions=Modelo, llm=None, **opcoes):
    llm = llm or _responde(x=lambda t: f"x-{t}")
    with (
        patch("dataframeit.core.call_langchain", side_effect=llm) as call_langchain,
        patch("dataframeit.core.validate_provider_dependencies"),
    ):
        opcoes.setdefault("track_tokens", False)
        resultado = dataframeit(df, questions=questions, prompt="Analise {texto}", **opcoes)
    return resultado, call_langchain


# =============================================================================
# Colunas de controle
# =============================================================================


def test_status_column_personalizado_sem_erros_some_da_saida():
    resultado, _ = _rodar(pd.DataFrame({"texto": ["a", "b"]}), status_column="st")
    assert list(resultado.columns) == ["texto", "x"]


def test_status_column_personalizado_com_erro_fica_no_fim():
    def llm(text, *args, **kwargs):
        if text.endswith("b"):
            msg = "falhou"
            raise ValueError(msg)
        return {"data": {"x": "ok"}, "usage": None}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        resultado, _ = _rodar(
            pd.DataFrame({"texto": ["a", "b"]}),
            llm=llm,
            status_column="st",
            max_retries=1,
        )
    assert list(resultado.columns) == ["texto", "x", "st", "_error_details"]


def test_coluna_de_nome_nao_textual():
    resultado, _ = _rodar(pd.DataFrame(["a", "b"]))
    assert resultado["x"].tolist() == ["x-a", "x-b"]


def test_detalhe_de_erro_antigo_some_quando_a_linha_passa():
    df = pd.DataFrame(
        {
            "texto": ["a"],
            "x": [None],
            "_dataframeit_status": [None],
            "_error_details": ["[Falhou após 1 tentativa(s)] ValueError: boom"],
        }
    )
    resultado, _ = _rodar(df, resume=True)
    assert resultado["x"].tolist() == ["x-a"]
    assert "_error_details" not in resultado.columns


@pytest.mark.parametrize("parallel_requests", [1, 2])
def test_detalhe_de_erro_antigo_some_tambem_no_paralelo(parallel_requests):
    df = pd.DataFrame(
        {
            "texto": ["a", "b"],
            "x": [None, None],
            "_dataframeit_status": [None, None],
            "_error_details": ["erro antigo", None],
        }
    )
    resultado, _ = _rodar(df, resume=True, parallel_requests=parallel_requests)
    assert "_error_details" not in resultado.columns


# =============================================================================
# Texto ausente
# =============================================================================


@pytest.mark.parametrize("parallel_requests", [1, 2])
def test_texto_ausente_nao_chama_o_llm(parallel_requests):
    df = pd.DataFrame({"texto": ["a", None, np.nan, "   ", "b"]})
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        resultado, call_langchain = _rodar(df, parallel_requests=parallel_requests)

    assert sorted(c.args[0] for c in call_langchain.call_args_list) == ["a", "b"]
    assert resultado["_dataframeit_status"].tolist() == [
        "processed",
        "error",
        "error",
        "error",
        "processed",
    ]
    assert resultado["_error_details"].iloc[1] == "Texto ausente"
    assert resultado["x"].iloc[1] is None
    assert any("3 linha(s) sem texto" in str(a.message) for a in avisos)


# =============================================================================
# Índice e nomes
# =============================================================================


def test_indice_duplicado_levanta_erro():
    df = pd.DataFrame({"texto": ["a", "b", "c"]}, index=[0, 0, 1])
    with pytest.raises(ValueError, match="reset_index"):
        _rodar(df)


def test_serie_com_indice_duplicado_levanta_erro():
    with pytest.raises(ValueError, match="reset_index"):
        _rodar(pd.Series(["a", "b"], index=["k", "k"]))


def test_campo_com_o_nome_da_coluna_de_texto_levanta_erro():
    class ModeloDecisao(BaseModel):
        decisao: str

    df = pd.DataFrame({"decisao": ["Julgo procedente o pedido."]})
    with pytest.raises(ValueError, match="decisao"):
        _rodar(df, questions=ModeloDecisao, llm=_responde(decisao="procedente"))
    assert df["decisao"].tolist() == ["Julgo procedente o pedido."]


# =============================================================================
# Tipos de coluna já existente
# =============================================================================


def test_lista_em_coluna_existente_float():
    df = pd.DataFrame(
        {
            "texto": ["a", "b"],
            "tags": [np.nan, np.nan],
            "_dataframeit_status": [None, None],
        }
    )
    resultado, _ = _rodar(df, questions=ModeloLista, llm=_responde(tags=["p", "q"]), resume=True)
    assert resultado["tags"].tolist() == [["p", "q"], ["p", "q"]]
    assert "_dataframeit_status" not in resultado.columns


# =============================================================================
# Reexecução sobre a própria saída
# =============================================================================


def test_reexecucao_sobre_a_propria_saida_avisa():
    primeira, _ = _rodar(pd.DataFrame({"texto": ["a", "b"]}))
    assert "_dataframeit_status" not in primeira.columns

    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        _rodar(primeira, resume=True)

    assert any("já estão preenchidas" in str(a.message) for a in avisos)


# =============================================================================
# Parâmetros
# =============================================================================


def test_perguntas_emite_deprecation_warning():
    with (
        pytest.warns(DeprecationWarning, match="questions"),
        patch("dataframeit.core.call_langchain", side_effect=_responde(x="1")),
        patch("dataframeit.core.validate_provider_dependencies"),
    ):
        dataframeit(pd.DataFrame({"texto": ["a"]}), perguntas=Modelo, prompt="{texto}")


@pytest.mark.parametrize("valor", [np.int64(2), 3])
def test_batch_size_aceita_inteiros(valor, tmp_path):
    _rodar(pd.DataFrame({"texto": ["a"]}), batch_size=valor, checkpoint_path=tmp_path / "c.csv")


@pytest.mark.parametrize("valor", [True, 2.0, 0])
def test_batch_size_rejeita_nao_inteiros(valor, tmp_path):
    with pytest.raises(ValueError, match="batch_size"):
        _rodar(pd.DataFrame({"texto": ["a"]}), batch_size=valor, checkpoint_path=tmp_path / "c.csv")


def test_estatisticas_de_busca_usam_o_provider_escolhido(capsys):

    _print_token_stats(
        {
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
            "search_count": 3,
            "search_credits": 3,
        },
        model="m",
        search_provider="exa",
    )
    saida = capsys.readouterr().out
    assert "EXA" in saida
    assert "TAVILY" not in saida


# =============================================================================
# Correções da revisão
# =============================================================================


class ModeloOpcional(BaseModel):
    x: str | None = None


class ModeloComPadroes(BaseModel):
    x: str
    obs: str = ""
    tags: list[str] = []


@pytest.mark.parametrize("parallel_requests", [1, 2])
def test_retomada_de_csv_com_detalhe_vazio_e_texto_ausente(tmp_path, parallel_requests):
    """Detalhe de erro todo vazio volta do CSV como float; gravar texto nele não quebra."""
    caminho = tmp_path / "c.csv"
    pd.DataFrame(
        {
            "texto": ["a", "b", None],
            "x": ["x-a", None, None],
            "_dataframeit_status": ["processed", None, None],
            "_error_details": [None, None, None],
        }
    ).to_csv(caminho, index=False)
    df = pd.read_csv(caminho)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        resultado, llm = _rodar(df, resume=True, parallel_requests=parallel_requests)

    assert llm.call_count == 1
    assert resultado["_dataframeit_status"].tolist() == ["processed", "processed", "error"]
    assert resultado["_error_details"].tolist()[2] == "Texto ausente"


def test_padroes_do_modelo_em_coluna_float_na_retomada():
    df = pd.DataFrame(
        {
            "texto": ["a", "b"],
            "x": ["x-a", np.nan],
            "obs": [np.nan, np.nan],
            "tags": [np.nan, np.nan],
            "_dataframeit_status": ["processed", np.nan],
        }
    )
    resultado, _ = _rodar(
        df,
        questions=ModeloComPadroes,
        llm=_responde(x="x-b", obs="o", tags=["t"]),
        resume=True,
    )
    assert resultado["obs"].tolist() == ["", "o"]
    assert resultado["tags"].tolist() == [[], ["t"]]


def test_reexecucao_avisa_mesmo_com_linha_toda_nula():
    primeira, _ = _rodar(
        pd.DataFrame({"texto": ["a", "b"]}),
        questions=ModeloOpcional,
        llm=_responde(x=lambda t: None if t.endswith("b") else "x"),
    )
    assert primeira["x"].tolist() == ["x", None]

    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        _rodar(primeira, questions=ModeloOpcional, resume=True)

    assert any("já estão preenchidas" in str(a.message) for a in avisos)


@pytest.mark.parametrize(
    "opcoes",
    [
        {"resume": True, "status": ["processed", "processed"]},
        {"resume": True, "status": None, "reprocess_columns": ["x"]},
    ],
)
def test_reexecucao_nao_avisa_com_status_ou_reprocess_columns(opcoes):
    opcoes = dict(opcoes)
    df = pd.DataFrame({"texto": ["a", "b"], "x": ["1", "2"]})
    status = opcoes.pop("status")
    if status is not None:
        df["_dataframeit_status"] = status

    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        _rodar(df, **opcoes)

    assert not any("já estão preenchidas" in str(a.message) for a in avisos)


@pytest.mark.parametrize("parallel_requests", [1, 2])
def test_texto_ausente_em_reprocess_columns_diz_que_os_valores_ficaram(parallel_requests):
    df = pd.DataFrame(
        {
            "texto": ["a", None],
            "x": ["x-a", "antigo"],
            "_dataframeit_status": ["processed", "processed"],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        resultado, _ = _rodar(df, reprocess_columns=["x"], parallel_requests=parallel_requests)

    assert resultado["x"].tolist() == ["x-a", "antigo"]
    detalhe = resultado["_error_details"].tolist()[1]
    assert detalhe.startswith("Texto ausente")
    assert "mantém os valores anteriores" in detalhe


def test_aviso_de_texto_ausente_conta_linhas_de_reprocess_columns():
    df = pd.DataFrame(
        {
            "texto": ["a", None],
            "x": ["x-a", "antigo"],
            "_dataframeit_status": ["processed", "processed"],
        }
    )
    with pytest.warns(UserWarning, match="1 linha") as avisos:
        _rodar(df, reprocess_columns=["x"])
    aviso = next(a for a in avisos if "sem texto" in str(a.message))
    assert aviso.filename == __file__


def test_aviso_de_texto_ausente_aponta_para_o_chamador():
    with pytest.warns(UserWarning, match="sem texto") as avisos:
        _rodar(pd.DataFrame({"texto": ["a", ""]}))
    aviso = next(a for a in avisos if "sem texto" in str(a.message))
    assert aviso.filename == __file__


@pytest.mark.parametrize("parallel_requests", [1, 2])
def test_texto_ausente_conta_para_o_checkpoint(tmp_path, parallel_requests):

    gravacoes = []
    original = core._try_save_checkpoint

    def registra(df, path):
        gravacoes.append(len(df))
        return original(df, path)

    with (
        patch("dataframeit.core._try_save_checkpoint", side_effect=registra),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore")
        _rodar(
            pd.DataFrame({"texto": [None, None, None, None]}),
            batch_size=2,
            checkpoint_path=tmp_path / "c.csv",
            parallel_requests=parallel_requests,
        )

    # Uma gravação a cada duas linhas; a final não se repete quando nada mudou.
    assert len(gravacoes) == 2


def test_status_column_personalizado_em_dataframe_vazio():
    resultado, _ = _rodar(pd.DataFrame({"texto": pd.Series([], dtype=object)}), status_column="st")
    assert list(resultado.columns) == ["texto", "x"]


def test_status_column_personalizado_em_checkpoint_concluido():
    df = pd.DataFrame({"texto": ["a"], "x": ["1"], "st": ["processed"]})
    resultado, llm = _rodar(df, status_column="st", resume=True)
    llm.assert_not_called()
    assert list(resultado.columns) == ["texto", "x"]


def test_status_column_personalizado_com_colunas_existentes():
    df = pd.DataFrame({"texto": ["a"], "x": ["1"], "st": ["processed"]})
    with pytest.warns(UserWarning, match="já existem"):
        resultado, _ = _rodar(df, status_column="st", resume=False)
    assert list(resultado.columns) == ["texto", "x"]


def _entrada(tipo, textos):
    if tipo == "pandas":
        return pd.DataFrame({"texto": textos})
    if tipo == "series":
        return pd.Series(textos)
    pl = pytest.importorskip("polars")
    if tipo == "polars":
        return pl.DataFrame({"texto": textos})
    return pl.Series(textos)


@pytest.mark.parametrize("tipo", ["pandas", "series", "polars", "polars_series"])
def test_status_column_personalizado_fica_depois_dos_tokens(tipo):
    def llm(text, *args, **kwargs):
        if text.endswith("b"):
            msg = "falhou"
            raise ValueError(msg)
        return {
            "data": {"x": "ok"},
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        resultado, _ = _rodar(
            _entrada(tipo, ["a", "b"]),
            llm=llm,
            status_column="st",
            max_retries=1,
            track_tokens=True,
        )
    colunas = list(resultado.columns)
    assert colunas[-2:] == ["st", "_error_details"]
    assert colunas.index("_input_tokens") < colunas.index("st")


def test_estatisticas_de_busca_de_ponta_a_ponta_usam_o_provider(capsys):
    resposta = {
        "data": {"x": "ok"},
        "usage": {
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
            "search_count": 1,
            "search_credits": 1,
        },
    }
    with (
        patch("dataframeit.agent.call_agent", return_value=resposta),
        patch("dataframeit.core.validate_provider_dependencies"),
        patch("dataframeit.core.validate_search_dependencies"),
    ):
        dataframeit(
            pd.DataFrame({"texto": ["a"]}),
            questions=Modelo,
            prompt="{texto}",
            use_search=True,
            search_provider="exa",
        )
    saida = capsys.readouterr().out
    assert "EXA" in saida
    assert "TAVILY" not in saida
