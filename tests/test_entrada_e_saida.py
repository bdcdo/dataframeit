"""Entrada e saída de dataframeit(): colunas de controle, texto ausente, índice e avisos."""

import json
import linecache
import logging
import math
import typing
import warnings
from datetime import UTC, date, datetime, time, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Annotated, Any, Literal, NewType, Optional, Union
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel, ConfigDict, Field, Strict, create_model
from pydantic_core import core_schema
from typing_extensions import TypeAliasType

from dataframeit import ProviderAbortError, ProviderError, core, dataframeit
from dataframeit.core import _format_token_stats


class Modelo(BaseModel):
    x: str


class ModeloLista(BaseModel):
    tags: list[str]


class ModeloNotaInteira(BaseModel):
    nota: int


class ModeloNotaDecimal(BaseModel):
    nota: float


class ModeloNotaMista(BaseModel):
    nota: Union[int, float]


class ModeloContagem(BaseModel):
    x: str
    n: Optional[int] = None


class ModeloFlag(BaseModel):
    relevante: bool


class ModeloFlagEstrita(BaseModel):
    model_config = ConfigDict(strict=True)

    relevante: bool


class Cor(Enum):
    AZUL = "azul"
    VERDE = "verde"


class Tinta(Enum):
    """O valor de cada membro é membro de outro Enum."""

    CLARA = Cor.AZUL
    ESCURA = Cor.VERDE


class ModeloCor(BaseModel):
    cor: Cor


class ModeloMisto(BaseModel):
    x: str
    campo: Union[int, str]


# NewType e apelido `type`, que a saída e a retomada polars resolvem antes de
# olhar a anotação. O TypeAliasType do typing_extensions roda em toda versão
# de Python que o pacote aceita; o `type X = ...` só existe a partir da 3.12.
_DataJudicial = NewType("_DataJudicial", date)
_Id = NewType("_Id", int)
_DataOuNumero = TypeAliasType("_DataOuNumero", Union[date, int])
_MeuInt = TypeAliasType("_MeuInt", int)
_ApelidoDeData = TypeAliasType("_ApelidoDeData", date)
_Metadados = NewType("_Metadados", dict[str, int])
_ApelidoDeDict = TypeAliasType("_ApelidoDeDict", dict[str, int])
_Codigo = NewType("_Codigo", str)
_ApelidoDeTexto = TypeAliasType("_ApelidoDeTexto", str)
_Etiquetas = NewType("_Etiquetas", list[str])
_ApelidoDeLista = TypeAliasType("_ApelidoDeLista", list[str])
_Medida = NewType("_Medida", float)
# Apelido genérico, que só vale pelo tipo depois de trocar os parâmetros pelos
# argumentos: `_ApelidoOpcional[int]` é o `type _ApelidoOpcional[T] = T | None`
# usado com int.
_T = typing.TypeVar("_T")
_K = typing.TypeVar("_K")
_ApelidoOpcional = TypeAliasType("_ApelidoOpcional", Optional[_T], type_params=(_T,))
_ApelidoProprio = TypeAliasType("_ApelidoProprio", _T, type_params=(_T,))
_ApelidoFixo = TypeAliasType("_ApelidoFixo", int, type_params=(_T,))
_ApelidoDoSegundo = TypeAliasType("_ApelidoDoSegundo", Optional[_T], type_params=(_K, _T))
_ApelidoDeDictGenerico = TypeAliasType("_ApelidoDeDictGenerico", dict[str, _T], type_params=(_T,))
_ApelidoDeListaGenerica = TypeAliasType("_ApelidoDeListaGenerica", list[_T], type_params=(_T,))


class _ValorOuFlag(BaseModel):
    valor: Union[float, bool]


class _DataEQuantidade(BaseModel):
    dia: date
    quantidade: int


class _ComDict(BaseModel):
    extra: dict[str, int]


def _responde(**campos):
    def call_langchain(text, *args, **kwargs):
        return {
            "data": {k: (v(text) if callable(v) else v) for k, v in campos.items()},
            "usage": None,
        }

    return call_langchain


def _responde_em_sequencia(*respostas):
    """Uma resposta por chamada; a que for exceção é levantada."""
    fila = iter(respostas)

    def call_langchain(text, *args, **kwargs):
        resposta = next(fila)
        if isinstance(resposta, Exception):
            raise resposta
        return {"data": resposta, "usage": None}

    return call_langchain


def _responde_em_sequencia_com_uso(uso, *respostas):
    sem_uso = _responde_em_sequencia(*respostas)
    return lambda text, *args, **kwargs: {**sem_uso(text), "usage": uso}


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
# Coluna do usuário com o nome de um campo do modelo ou da coluna de status
# =============================================================================


class ModeloCategoria(BaseModel):
    categoria: str
    nota: int


def _rotulos_humanos():
    return pd.DataFrame(
        {"id": [10, 20], "texto": ["a", "b"], "categoria": ["rótulo humano 1", "rótulo humano 2"]}
    )


def _recusa(entrada, match, questions=ModeloCategoria, **opcoes):
    """Roda esperando o ValueError de `match`, sem chamar o LLM nem alterar a entrada."""
    antes = entrada.copy()
    opcoes.setdefault("track_tokens", False)
    with (
        patch(
            "dataframeit.core.call_langchain", side_effect=_responde(categoria="llm", nota=1)
        ) as llm,
        patch("dataframeit.core.validate_provider_dependencies"),
        pytest.raises(ValueError, match=match),
    ):
        dataframeit(entrada, questions=questions, prompt="{texto}", **opcoes)
    llm.assert_not_called()
    assert entrada.equals(antes)


@pytest.mark.parametrize("resume", [True, False])
def test_coluna_com_nome_de_campo_sem_status_levanta(resume):
    _recusa(
        _rotulos_humanos(),
        r"As colunas \['categoria'\] já existem .* Renomeie ou remova",
        resume=resume,
    )


def test_coluna_com_nome_de_campo_e_checkpoint_levanta_sem_gravar(tmp_path):
    checkpoint = tmp_path / "c.csv"
    _recusa(
        _rotulos_humanos(),
        r"As colunas \['categoria'\] já existem",
        batch_size=1,
        checkpoint_path=checkpoint,
    )
    assert not checkpoint.exists()


def test_coluna_de_campo_com_valor_em_uma_linha_levanta():
    df = pd.DataFrame({"texto": ["a", "b"], "categoria": [None, "rótulo humano"]})
    _recusa(df, r"As colunas \['categoria'\] já existem")


def test_coluna_de_campo_sem_valor_nenhum_recebe_a_resposta():
    df = pd.DataFrame({"texto": ["a", "b"], "categoria": [None, np.nan]})

    resultado, llm = _rodar(
        df, questions=ModeloCategoria, llm=_responde(categoria=lambda t: f"llm-{t}", nota=1)
    )

    assert llm.call_count == 2
    assert resultado["categoria"].tolist() == ["llm-a", "llm-b"]


@pytest.mark.parametrize("status_column", [None, "st"])
def test_saida_completa_sem_status_levanta_e_diz_como_retomar(status_column):
    primeira, _ = _rodar(pd.DataFrame({"texto": ["a", "b"]}), status_column=status_column)
    nome = status_column or "_dataframeit_status"
    assert nome not in primeira.columns

    _recusa(
        primeira,
        rf"\['x'\] já existem .* df\['{nome}'\] = 'processed'",
        questions=Modelo,
        status_column=status_column,
    )


def test_reprocess_columns_sem_status_refaz_a_coluna_pedida():
    df = pd.DataFrame({"texto": ["a", "b"], "categoria": ["humano a", "humano b"]})

    resultado, llm = _rodar(
        df,
        questions=ModeloCategoria,
        llm=_responde(categoria=lambda t: f"llm-{t}", nota=1),
        reprocess_columns=["categoria"],
    )

    assert llm.call_count == 2
    assert resultado["categoria"].tolist() == ["llm-a", "llm-b"]


def test_reprocess_columns_sem_status_nao_cobre_outra_coluna_existente():
    df = _rotulos_humanos().assign(nota=[7, 8])
    _recusa(df, r"As colunas \['categoria'\] já existem", reprocess_columns=["nota"])


def test_dataframe_vazio_com_coluna_de_campo_sem_status_nao_levanta():
    vazio = pd.Series([], dtype=object)
    resultado, llm = _rodar(pd.DataFrame({"texto": vazio, "x": vazio}))

    llm.assert_not_called()
    assert list(resultado.columns) == ["texto", "x"]


@pytest.mark.parametrize(
    ("status_column", "match"),
    [
        ("categoria", "status_column='categoria' é um campo do modelo"),
        ("texto", "status_column='texto' é a coluna de texto"),
        ("id", r"A coluna de status 'id' já existe .*\(10 na linha 0\)"),
        ("_error_details", "status_column='_error_details' é o nome de uma coluna de controle"),
        ("_input_tokens", "status_column='_input_tokens' é o nome de uma coluna de controle"),
        ("_search_credits", "status_column='_search_credits' é o nome de uma coluna de controle"),
        ("_trace", "status_column='_trace' é o nome de uma coluna de controle"),
        ("_trace_categoria", "status_column='_trace_categoria' é o nome de uma coluna de controle"),
    ],
)
def test_status_column_que_ocuparia_outra_coluna_levanta(status_column, match):
    _recusa(_rotulos_humanos().drop(columns="categoria"), match, status_column=status_column)


def test_status_column_com_o_nome_da_coluna_interna_de_texto_levanta():
    _recusa(
        pd.Series(["a", "b"]),
        "status_column='_texto' é a coluna de texto",
        status_column="_texto",
    )


def test_coluna_de_status_padrao_com_outros_valores_levanta():
    df = pd.DataFrame({"texto": ["a"], "_dataframeit_status": ["concluído"]})
    _recusa(df, r"A coluna de status '_dataframeit_status' já existe .*\('concluído' na linha 0\)")


def test_status_column_existente_com_valores_de_status_retoma():
    df = pd.DataFrame(
        {
            "texto": ["a", "b", "c"],
            "x": ["feito", "anterior", None],
            "st": ["processed", "error", np.nan],
        }
    )

    resultado, llm = _rodar(df, status_column="st")

    assert llm.call_count == 1
    assert resultado["x"].tolist() == ["feito", "anterior", "x-c"]
    assert resultado["st"].tolist() == ["processed", "error", "processed"]


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


def test_estatisticas_de_busca_usam_o_provider_escolhido():
    saida = _format_token_stats(
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

    def registra(df, checkpoint):
        gravacoes.append(len(df))
        return original(df, checkpoint)

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


def test_estatisticas_de_busca_de_ponta_a_ponta_usam_o_provider(caplog):
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
        caplog.at_level(logging.INFO, logger="dataframeit.stats"),
    ):
        dataframeit(
            pd.DataFrame({"texto": ["a"]}),
            questions=Modelo,
            prompt="{texto}",
            use_search=True,
            search_provider="exa",
        )
    saida = caplog.text
    assert "EXA" in saida
    assert "TAVILY" not in saida


# =============================================================================
# Validação da entrada
# =============================================================================


def test_text_column_inexistente_levanta_com_as_colunas_disponiveis():
    with pytest.raises(ValueError, match=r"Coluna 'corpo' não encontrada.*\['texto'\]"):
        _rodar(pd.DataFrame({"texto": ["a"]}), text_column="corpo")


def test_modelo_sem_campos_levanta():
    class Vazio(BaseModel):
        pass

    with pytest.raises(ValueError, match="Modelo Pydantic não pode estar vazio"):
        _rodar(pd.DataFrame({"texto": ["a"]}), questions=Vazio)


# =============================================================================
# Retomada desligada, com resume=False
# =============================================================================


def test_resume_false_processa_o_dataframe_novo():
    resultado, call_langchain = _rodar(pd.DataFrame({"texto": ["a", "b"]}), resume=False)

    assert resultado["x"].tolist() == ["x-a", "x-b"]
    assert call_langchain.call_count == 2


@pytest.mark.parametrize("parallel_requests", [1, 2])
def test_resume_false_nao_refaz_linha_ja_processada(parallel_requests):
    """Com resume=False, só a linha que não está 'processed' vai ao LLM."""
    df = pd.DataFrame({"texto": ["a", "b"], "_dataframeit_status": ["processed", "processed"]})

    resultado, call_langchain = _rodar(df, resume=False, parallel_requests=parallel_requests)

    call_langchain.assert_not_called()
    assert resultado["x"].isna().all()


# =============================================================================
# Falha de reprocess_columns numa linha já processada
# =============================================================================


@pytest.mark.parametrize("parallel_requests", [1, 2])
def test_falha_ao_reprocessar_diz_que_os_valores_ficaram(parallel_requests):
    def falha(text, *args, **kwargs):
        msg = "chave recusada"
        raise ProviderError(msg)

    df = pd.DataFrame({"texto": ["a"], "x": ["antigo"], "_dataframeit_status": ["processed"]})
    with pytest.warns(UserWarning, match="Falha ao processar linha 0"):
        resultado, _ = _rodar(
            df, llm=falha, reprocess_columns=["x"], parallel_requests=parallel_requests
        )

    assert resultado["x"].tolist() == ["antigo"]
    detalhe = resultado["_error_details"].iloc[0]
    assert detalhe.startswith("[Erro não-recuperável] ProviderError: chave recusada")
    assert detalhe.endswith("(reprocess_columns falhou; a linha mantém os valores anteriores)")


# =============================================================================
# rate_limit_delay
# =============================================================================


@pytest.mark.parametrize(
    ("parallel_requests", "rotulo"), [(1, "[~120 req/min]"), (2, "[2 workers]")]
)
def test_rate_limit_delay_espera_depois_de_cada_linha(monkeypatch, parallel_requests, rotulo):
    esperas = []
    monkeypatch.setattr(core.time, "sleep", esperas.append)
    # O rótulo é lido na chamada ao tqdm, e não no stderr: o tqdm do piso
    # (4.1.0) fixa sys.stderr na importação, fora do alcance do capsys.
    rotulos = []
    tqdm_original = core.tqdm

    def tqdm_que_registra(*args, **kwargs):
        rotulos.append(kwargs.get("desc", ""))
        return tqdm_original(*args, **kwargs)

    monkeypatch.setattr(core, "tqdm", tqdm_que_registra)

    resultado, _ = _rodar(
        pd.DataFrame({"texto": ["a", "b"]}),
        rate_limit_delay=0.5,
        parallel_requests=parallel_requests,
    )

    assert resultado["x"].tolist() == ["x-a", "x-b"]
    assert esperas == [0.5, 0.5]
    assert len(rotulos) == 1
    assert rotulo in rotulos[0]


# =============================================================================
# Resumo de uso
# =============================================================================


def test_resumo_detalha_cache_e_raciocinio():
    saida = _format_token_stats(
        {
            "input_tokens": 100,
            "cached_input_tokens": 40,
            "output_tokens": 30,
            "reasoning_tokens": 10,
            "total_tokens": 130,
        },
        model="m",
    )

    assert "└─ Cache: 40 (incluído no Input)" in saida
    assert "└─ Reasoning: 10 (incluído no Output)" in saida


def test_resumo_sem_requisicao_concluida_omite_a_taxa():
    """Linhas que falharam com custo informado entram no resumo sem uma taxa de requisições."""
    saida = _format_token_stats(
        {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.02,
            "elapsed_seconds": 2.0,
            "requests_completed": 0,
        },
        model="m",
        parallel_requests=2,
    )

    assert "METRICAS DE THROUGHPUT" in saida
    assert "TPM (tokens/min): 0" in saida
    assert "Requisicoes" not in saida


# =============================================================================
# Entrada polars: colunas que a execução não escreve
# =============================================================================

# 2**53 + 1, o primeiro inteiro que o float64 arredonda.
_ID_GRANDE = 9007199254740993

# Um tipo por coluna que a ida e volta pelo pandas trocaria ou arredondaria.
_COLUNAS_POLARS = {
    "int64_com_nulo": lambda pl: pl.Series([_ID_GRANDE, None], dtype=pl.Int64),
    "uint64_com_nulo": lambda pl: pl.Series([2**64 - 1, None], dtype=pl.UInt64),
    "float_com_nan_e_nulo": lambda pl: pl.Series([float("nan"), None], dtype=pl.Float64),
    "date": lambda pl: pl.Series([date(2024, 1, 2), None], dtype=pl.Date),
    "enum": lambda pl: pl.Series(["a", None], dtype=pl.Enum(["a", "b"])),
    "decimal": lambda pl: pl.Series([Decimal("1.10"), None], dtype=pl.Decimal(10, 2)),
    "array": lambda pl: pl.Series([[1, 2], [3, 4]], dtype=pl.Array(pl.Int64, 2)),
    "struct_com_inteiro_grande": lambda pl: pl.Series(
        [{"id": _ID_GRANDE}, {"id": None}], dtype=pl.Struct({"id": pl.Int64})
    ),
    "lista_com_inteiro_grande": lambda pl: pl.Series(
        [[_ID_GRANDE, None], [1]], dtype=pl.List(pl.Int64)
    ),
    "null": lambda pl: pl.Series([None, None], dtype=pl.Null),
    "object": lambda pl: pl.Series([{1}, None], dtype=pl.Object),
}


@pytest.mark.parametrize("tipo", _COLUNAS_POLARS)
def test_coluna_polars_que_a_execucao_nao_escreve_volta_como_entrou(tipo):
    pl = pytest.importorskip("polars")
    entrada = pl.DataFrame({"texto": ["a", "b"], "c": _COLUNAS_POLARS[tipo](pl)})

    resultado, _ = _rodar(entrada)

    assert resultado.columns == ["texto", "c", "x"]
    assert resultado["c"].dtype == entrada["c"].dtype
    # Series.equals não compara Object.
    if entrada["c"].dtype == pl.Object:
        assert resultado["c"].to_list() == entrada["c"].to_list()
    else:
        assert resultado["c"].equals(entrada["c"])


def test_saida_polars_mantem_cada_linha_com_o_seu_resultado():
    pl = pytest.importorskip("polars")
    entrada = pl.DataFrame(
        {"texto": ["a", "b", "c"], "id": pl.Series([_ID_GRANDE, None, 1], dtype=pl.Int64)}
    )

    def llm(text, *args, **kwargs):
        if text.endswith("b"):
            msg = "falhou"
            raise ValueError(msg)
        return {"data": {"x": text[-1]}, "usage": None}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        resultado, _ = _rodar(entrada, llm=llm, max_retries=1)

    assert resultado["id"].to_list() == [_ID_GRANDE, None, 1]
    assert resultado["x"].to_list() == ["a", None, "c"]
    assert resultado["_dataframeit_status"].to_list() == ["processed", "error", "processed"]


def test_campo_reprocessado_em_saida_polars_sai_com_o_tipo_da_resposta_nova():
    pl = pytest.importorskip("polars")
    primeira, _ = _rodar(
        pl.DataFrame({"texto": ["a", "b"]}), questions=ModeloNotaInteira, llm=_responde(nota=7)
    )
    assert primeira["nota"].dtype == pl.Int64

    segunda, _ = _rodar(
        primeira, questions=ModeloNotaDecimal, llm=_responde(nota=7.5), reprocess_columns=["nota"]
    )

    assert segunda["nota"].to_list() == [7.5, 7.5]


def test_retomada_de_saida_polars_nao_trunca_a_resposta_nova():
    pl = pytest.importorskip("polars")
    entrada = pl.DataFrame({"texto": ["a", "b", "c"]})
    with pytest.warns(UserWarning, match="interrompida"):
        interrompida, _ = _rodar(
            entrada,
            questions=ModeloNotaMista,
            llm=_responde_em_sequencia({"nota": 7}, ProviderAbortError("limite")),
        )
    assert interrompida["nota"].dtype == pl.Int64

    retomada, _ = _rodar(
        interrompida,
        questions=ModeloNotaMista,
        llm=_responde_em_sequencia({"nota": 8.5}, {"nota": 9.9}),
        resume=True,
    )

    assert retomada["nota"].to_list() == [7.0, 8.5, 9.9]


def test_retomada_de_saida_polars_sem_resposta_tipa_o_campo_pelas_respostas():
    pl = pytest.importorskip("polars")
    entrada = pl.DataFrame({"texto": ["a", "b", "c"]})
    with pytest.warns(UserWarning, match="interrompida"):
        interrompida, _ = _rodar(
            entrada,
            questions=ModeloContagem,
            llm=_responde_em_sequencia(ProviderAbortError("limite")),
        )
    assert interrompida["n"].null_count() == 3

    retomada, _ = _rodar(
        interrompida,
        questions=ModeloContagem,
        llm=_responde_em_sequencia({"x": "p", "n": 1}, {"x": "q"}, {"x": "r", "n": 3}),
        resume=True,
    )

    assert retomada["n"].dtype == pl.Int64
    assert retomada["n"].to_list() == [1, None, 3]


def test_saida_polars_ja_concluida_devolve_o_campo_de_lista_normalizado():
    pl = pytest.importorskip("polars")
    entrada = pl.DataFrame(
        {"texto": ["a"], "tags": ['["p", "q"]'], "_dataframeit_status": ["processed"]}
    )

    resultado, llm = _rodar(entrada, questions=ModeloLista, resume=True)

    llm.assert_not_called()
    assert resultado["tags"].to_list() == [["p", "q"]]


def test_entrada_polars_com_colunas_existentes_volta_igual():
    pl = pytest.importorskip("polars")
    entrada = pl.DataFrame(
        {
            "texto": ["a"],
            "x": [date(2024, 1, 2)],
            "id": pl.Series([None], dtype=pl.Int64),
            "_dataframeit_status": ["error"],
        }
    )

    with pytest.warns(UserWarning, match="já existem"):
        resultado, llm = _rodar(entrada, resume=False)

    llm.assert_not_called()
    assert resultado.schema == entrada.schema
    assert resultado.equals(entrada)


def test_entrada_polars_vazia_mantem_o_tipo_das_colunas():
    pl = pytest.importorskip("polars")
    entrada = pl.DataFrame(
        {"texto": pl.Series([], dtype=pl.Utf8), "dia": pl.Series([], dtype=pl.Date)}
    )

    resultado, _ = _rodar(entrada)

    assert resultado.columns == ["texto", "dia", "x"]
    assert resultado["dia"].dtype == pl.Date


def test_entrada_polars_vazia_mantem_o_tipo_da_coluna_do_modelo_que_ja_traz():
    # Sem linhas não há resposta a converter, e a coluna do modelo sai da
    # entrada; convertida de volta do pandas, ela perderia o Struct e o fuso.
    pl = pytest.importorskip("polars")
    modelo = create_model(
        "ModeloDoCaso", sub=(Optional[Modelo], None), quando=(Optional[datetime], None)
    )
    entrada = pl.DataFrame(
        {
            "texto": pl.Series([], dtype=pl.Utf8),
            "sub": pl.Series([], dtype=pl.Struct({"x": pl.Utf8})),
            "quando": pl.Series([], dtype=pl.Datetime("us", "UTC")),
        }
    )

    resultado, llm = _rodar(entrada, questions=modelo)

    llm.assert_not_called()
    assert dict(resultado.schema) == dict(entrada.schema)


def test_retomada_de_saida_polars_traz_os_tokens_das_linhas_novas():
    pl = pytest.importorskip("polars")
    uso = {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}
    with pytest.warns(UserWarning, match="interrompida"):
        interrompida, _ = _rodar(
            pl.DataFrame({"texto": ["a", "b"]}),
            llm=_responde_em_sequencia_com_uso(uso, {"x": "p"}, ProviderAbortError("limite")),
            track_tokens=True,
        )
    assert interrompida["_input_tokens"].to_list() == [3, None]

    retomada, _ = _rodar(
        interrompida,
        llm=_responde_em_sequencia_com_uso(uso, {"x": "q"}),
        track_tokens=True,
        resume=True,
    )

    assert retomada["_input_tokens"].to_list() == [3, 3]
    assert retomada["_output_tokens"].to_list() == [2, 2]


@pytest.mark.parametrize("modelo", [ModeloFlag, ModeloFlagEstrita])
def test_retomada_polars_de_campo_bool_gravado_como_inteiro(modelo):
    """Um bool lido de volta de SQLite pelo polars chega como Int64 0/1."""
    pl = pytest.importorskip("polars")
    entrada = pl.DataFrame(
        {
            "texto": ["a", "b", "c"],
            "relevante": pl.Series([1, 0, None], dtype=pl.Int64),
            "_dataframeit_status": ["processed", "processed", None],
        }
    )

    resultado, llm = _rodar(entrada, questions=modelo, llm=_responde(relevante=True), resume=True)

    assert llm.call_count == 1
    assert resultado["relevante"].dtype == pl.Boolean
    assert resultado["relevante"].to_list() == [True, False, True]


class PeriodoComDatas(BaseModel):
    datas: list[date]


# Tipo do campo e as respostas das duas linhas. O to_pandas do polars entregaria
# cada uma dessas colunas ao pandas como array do numpy, com a data em
# datetime64, o datetime com fuso passado a UTC e sem fuso, e a lista de inteiros
# com nulo em float, que arredonda o inteiro acima de 2**53.
_LISTAS_DA_SAIDA_POLARS = {
    "list_date": (list[date], [date(2026, 1, 2)], [date(2026, 3, 4)]),
    "list_datetime": (
        list[datetime],
        [datetime(2026, 1, 2, 10, 30)],  # noqa: DTZ001
        [datetime(2026, 3, 4, 8, 15, 1, 5)],  # noqa: DTZ001
    ),
    "list_datetime_utc": (
        list[datetime],
        [datetime(2026, 1, 2, 10, 30, tzinfo=UTC)],
        [datetime(2026, 3, 4, 8, 15, tzinfo=UTC)],
    ),
    "list_datetime_menos_3": (
        list[datetime],
        [datetime(2026, 1, 2, 10, 30, tzinfo=timezone(timedelta(hours=-3)))],
        [datetime(2026, 3, 4, 8, 15, tzinfo=timezone(timedelta(hours=-3)))],
    ),
    "optional_list_date": (Optional[list[date]], [date(2026, 1, 2), date(2026, 3, 4)], None),
    "list_optional_date": (list[Optional[date]], [date(2026, 1, 2), None], [date(2026, 3, 4)]),
    "list_date_strict": (Annotated[list[date], Strict()], [date(2026, 1, 2)], [date(2026, 3, 4)]),
    "tuple_date_date": (
        tuple[date, date],
        (date(2026, 1, 2), date(2026, 3, 4)),
        (date(2026, 3, 4), date(2026, 1, 2)),
    ),
    "submodelo_com_lista_de_datas": (
        PeriodoComDatas,
        {"datas": [date(2026, 1, 2)]},
        {"datas": [date(2026, 3, 4)]},
    ),
    "list_optional_int_acima_de_2_53": (
        list[Optional[int]],
        [_ID_GRANDE, None],
        [_ID_GRANDE + 2],
    ),
}


def _interrompe_e_retoma(modelo, primeira, segunda, ajusta=lambda df: df):
    """Saídas da execução sem interrupção e da retomada da interrompida na segunda linha.

    `ajusta` muda a saída interrompida antes da retomada, como o usuário faria.
    """
    pl = pytest.importorskip("polars")
    entrada = pl.DataFrame({"texto": ["a", "b"]})
    sem_interrupcao, _ = _rodar(
        entrada,
        questions=modelo,
        llm=_responde_em_sequencia({"campo": primeira}, {"campo": segunda}),
    )
    with pytest.warns(UserWarning, match="interrompida"):
        interrompida, _ = _rodar(
            entrada,
            questions=modelo,
            llm=_responde_em_sequencia({"campo": primeira}, ProviderAbortError("limite")),
        )
    assert interrompida["campo"].dtype == sem_interrupcao["campo"].dtype

    retomada, llm = _rodar(
        ajusta(interrompida),
        questions=modelo,
        llm=_responde_em_sequencia({"campo": segunda}),
        resume=True,
    )

    assert llm.call_count == 1
    return sem_interrupcao, retomada


@pytest.mark.parametrize("caso", _LISTAS_DA_SAIDA_POLARS)
def test_retomada_de_lista_da_saida_polars(caso):
    """A lista da saída polars volta com os valores e o tipo de antes."""
    tipo, primeira, segunda = _LISTAS_DA_SAIDA_POLARS[caso]
    modelo = create_model("ModeloDoCaso", campo=(tipo, ...))

    sem_interrupcao, retomada = _interrompe_e_retoma(modelo, primeira, segunda)

    assert retomada["campo"].dtype == sem_interrupcao["campo"].dtype
    assert retomada["campo"].to_list() == sem_interrupcao["campo"].to_list()


def test_retomada_de_lista_de_datas_guardada_como_array():
    """A coluna do campo que o usuário passou a Array volta como a List da execução."""
    pl = pytest.importorskip("polars")
    modelo = create_model("ModeloDoCaso", campo=(list[date], ...))
    sem_interrupcao, retomada = _interrompe_e_retoma(
        modelo,
        [date(2026, 1, 2), date(2026, 1, 3)],
        [date(2026, 3, 4), date(2026, 3, 5)],
        ajusta=lambda df: df.with_columns(pl.col("campo").cast(pl.Array(pl.Date, 2))),
    )

    assert retomada["campo"].dtype == sem_interrupcao["campo"].dtype
    assert retomada["campo"].to_list() == sem_interrupcao["campo"].to_list()


# =============================================================================
# Entrada polars: retomada do campo do modelo com valor inteiro
# =============================================================================

# Anotações que admitem int: sem o Optional, o Annotated, o NewType e o apelido,
# são int ou uma união com int entre os membros.
_ANOTACOES_DE_INTEIRO = {
    "int": int,
    "optional": Optional[int],
    "uniao_com_none": int | None,
    "annotated": Annotated[int, Field(ge=0)],
    "optional_de_annotated": Optional[Annotated[int, Field(ge=0)]],
    "uniao_de_int_e_float": Union[int, float],
    "uniao_opcional_de_int_e_str": Optional[Union[int, str]],
    "newtype_de_int": _Id,
    "optional_de_newtype_de_int": Optional[_Id],
    "optional_de_annotated_de_newtype_de_int": Optional[Annotated[_Id, Field(ge=0)]],
    "uniao_de_newtype_de_int_e_str": Union[_Id, str],
    "apelido_de_int": _MeuInt,
    "literal_de_int": Literal[9007199254740993, 7, 8],
    "apelido_generico_opcional": _ApelidoOpcional[int],
    "apelido_generico_do_proprio_parametro": _ApelidoProprio[int],
    "apelido_generico_sem_parametro_no_valor": _ApelidoFixo[str],
    "apelido_generico_na_ordem_do_valor": _ApelidoDoSegundo[str, int],
    "optional_de_apelido_generico": Optional[_ApelidoProprio[int]],
}


def _rodar_sem_avisos(df, questions, llm, **opcoes):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        resultado, _ = _rodar(df, questions=questions, llm=llm, **opcoes)
    return resultado


@pytest.mark.parametrize("caso", _ANOTACOES_DE_INTEIRO)
def test_retomada_de_saida_polars_devolve_o_inteiro_grande_exato(caso):
    pl = pytest.importorskip("polars")
    modelo = create_model("ModeloDoCaso", nota=(_ANOTACOES_DE_INTEIRO[caso], ...))
    entrada = pl.DataFrame({"texto": ["a", "b", "c"]})
    interrompida = _rodar_sem_avisos(
        entrada,
        modelo,
        _responde_em_sequencia({"nota": _ID_GRANDE}, ProviderAbortError("limite")),
    )
    assert interrompida["nota"].to_list() == [_ID_GRANDE, None, None]

    retomada = _rodar_sem_avisos(
        interrompida, modelo, _responde_em_sequencia({"nota": 7}, {"nota": 8}), resume=True
    )
    sem_interrupcao = _rodar_sem_avisos(
        entrada, modelo, _responde_em_sequencia({"nota": _ID_GRANDE}, {"nota": 7}, {"nota": 8})
    )

    assert retomada["nota"].dtype == pl.Int64
    assert retomada["nota"].to_list() == [_ID_GRANDE, 7, 8]
    assert retomada.equals(sem_interrupcao)


def test_retomada_polars_de_campo_float_gravado_como_inteiro_grande():
    pl = pytest.importorskip("polars")
    entrada = pl.DataFrame(
        {
            "texto": ["a", "b"],
            "nota": pl.Series([_ID_GRANDE, None], dtype=pl.Int64),
            "_dataframeit_status": ["processed", None],
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        resultado, llm = _rodar(
            entrada, questions=ModeloNotaDecimal, llm=_responde(nota=2.5), resume=True
        )

    assert llm.call_count == 1
    assert resultado["nota"].dtype == pl.Float64
    assert resultado["nota"].to_list() == [float(_ID_GRANDE), 2.5]


# O 2**64 - 1 não cabe no Int64, e a execução sem interrupção o devolve em
# coluna Object (ver _CAMPOS_QUE_O_ARROW_NAO_TIPA).
@pytest.mark.parametrize("valor", [5, 2**64 - 1])
def test_retomada_polars_de_campo_int_gravado_em_uint64(valor):
    pl = pytest.importorskip("polars")
    entrada = pl.DataFrame(
        {
            "texto": ["a", "b"],
            "nota": pl.Series([valor, None], dtype=pl.UInt64),
            "_dataframeit_status": ["processed", None],
        }
    )

    retomada = _rodar_sem_avisos(
        entrada, ModeloNotaInteira, _responde_por_linha("nota", [7]), resume=True
    )
    sem_interrupcao = _rodar_sem_avisos(
        entrada.select("texto"), ModeloNotaInteira, _responde_por_linha("nota", [valor, 7])
    )

    assert retomada["nota"].to_list() == [valor, 7]
    assert retomada["nota"].dtype == sem_interrupcao["nota"].dtype


class ModeloNotaEstrita(BaseModel):
    model_config = ConfigDict(strict=True)

    nota: int
    rotulo: str


def test_retomada_polars_de_campo_int_estrito_em_coluna_sem_nulo():
    # A coluna Int64 sem nulo também chega em inteiros Python. Pelo to_pandas()
    # do polars ela viria em int64 do numpy, que o modelo strict recusa, e a
    # linha processada seria acusada como incompatível.
    pl = pytest.importorskip("polars")
    entrada = pl.DataFrame(
        {
            "texto": ["a", "b"],
            "nota": [_ID_GRANDE, 3],
            "rotulo": ["x", "y"],
            "_dataframeit_status": ["processed", "processed"],
        }
    )

    resultado, llm = _rodar(
        entrada,
        questions=ModeloNotaEstrita,
        llm=_responde_por_linha("rotulo", ["z", "w"]),
        resume=True,
        reprocess_columns=["rotulo"],
    )

    assert llm.call_count == 2
    assert resultado["nota"].dtype == pl.Int64
    assert resultado["nota"].to_list() == [_ID_GRANDE, 3]
    assert resultado["rotulo"].to_list() == ["z", "w"]


class ContagemOpcional(BaseModel):
    n: Optional[int] = None


class ModeloInteirosAninhados(BaseModel):
    nota: Optional[int]
    lista: list[Optional[int]]
    contagem: ContagemOpcional


def test_retomada_polars_de_inteiro_grande_em_campo_lista_e_submodelo():
    # A coluna do campo int passa por _python_ints, e as colunas List e Struct
    # pelo laço de to_pandas; cada uma arredondaria o inteiro grande sem a sua.
    pl = pytest.importorskip("polars")
    respostas = [
        {"nota": _ID_GRANDE, "lista": [_ID_GRANDE, None], "contagem": {"n": _ID_GRANDE}},
        {"nota": None, "lista": [None, _ID_GRANDE + 2], "contagem": {"n": None}},
        {"nota": 7, "lista": [], "contagem": {"n": _ID_GRANDE + 4}},
    ]
    entrada = pl.DataFrame({"texto": ["a", "b", "c"]})
    interrompida = _rodar_sem_avisos(
        entrada,
        ModeloInteirosAninhados,
        _responde_em_sequencia(respostas[0], ProviderAbortError("limite")),
    )
    assert interrompida["nota"].to_list() == [_ID_GRANDE, None, None]

    retomada = _rodar_sem_avisos(
        interrompida, ModeloInteirosAninhados, _responde_em_sequencia(*respostas[1:]), resume=True
    )
    sem_interrupcao = _rodar_sem_avisos(
        entrada, ModeloInteirosAninhados, _responde_em_sequencia(*respostas)
    )

    for campo in ("nota", "lista", "contagem"):
        assert retomada[campo].to_list() == [resposta[campo] for resposta in respostas]
    assert retomada.equals(sem_interrupcao)
    assert retomada.schema == sem_interrupcao.schema


def test_reprocessar_campo_int_de_saida_polars_com_resposta_float_e_texto():
    # A resposta do dublê não passa pela validação do modelo, e a coluna do
    # campo int recebe o float e o texto como vieram.
    pl = pytest.importorskip("polars")
    entrada = pl.DataFrame(
        {
            "texto": ["a", "b"],
            "nota": pl.Series([_ID_GRANDE, None], dtype=pl.Int64),
            "_dataframeit_status": ["processed", None],
        }
    )

    with pytest.warns(UserWarning, match="Object"):
        resultado, llm = _rodar(
            entrada,
            questions=ModeloNotaInteira,
            llm=_responde_por_linha("nota", [7.5, "8"]),
            resume=True,
            reprocess_columns=["nota"],
        )

    assert llm.call_count == 2
    assert resultado["nota"].to_list() == [7.5, "8"]


@pytest.mark.parametrize(
    ("tipo", "respostas"),
    [(Union[int, float], [7, 8.5]), (Optional[int], [_ID_GRANDE, None])],
)
def test_retomada_polars_interrompida_antes_da_primeira_resposta(tipo, respostas):
    pl = pytest.importorskip("polars")
    modelo = create_model("ModeloDoCaso", nota=(tipo, ...))
    entrada = pl.DataFrame({"texto": ["a", "b"]})
    interrompida = _rodar_sem_avisos(
        entrada, modelo, _responde_em_sequencia(ProviderAbortError("limite"))
    )
    assert interrompida["nota"].null_count() == 2

    retomada = _rodar_sem_avisos(
        interrompida, modelo, _responde_por_linha("nota", respostas), resume=True
    )
    sem_interrupcao = _rodar_sem_avisos(entrada, modelo, _responde_por_linha("nota", respostas))

    assert retomada.equals(sem_interrupcao)
    assert retomada["nota"].dtype == sem_interrupcao["nota"].dtype


# =============================================================================
# Entrada polars: campo do modelo que o Arrow não tipa
# =============================================================================

_FUSO_DE_DESLOCAMENTO = timezone(timedelta(hours=5, minutes=30))

# Tipo do campo e as respostas, uma por linha, que o Arrow não põe na mesma coluna.
# O datetime com fuso de deslocamento é recusado pelo polars, com ComputeError,
# que não é ValueError nem TypeError.
_CAMPOS_QUE_O_ARROW_NAO_TIPA = {
    "tupla_heterogenea": (tuple[int, str], [(1, "a"), (2, "b")]),
    "uniao_com_numero_primeiro": (Union[int, str], [1, "dois"]),
    "uniao_com_texto_primeiro": (Union[int, str], ["dois", 1]),
    "any_com_tipos_diferentes": (Any, [1, "x"]),
    "dict_com_chave_que_o_json_nao_aceita": (dict[Cor, int], [{Cor.AZUL: 1}, {Cor.VERDE: 2}]),
    "lista_de_uniao": (list[Union[int, str]], [[1, "a"], [2]]),
    "lista_de_enum": (list[Cor], [[Cor.AZUL], [Cor.VERDE]]),
    "inteiro_que_o_float_arredonda_com_float": (Union[int, float], [_ID_GRANDE, 2.5]),
    "inteiro_acima_do_int64": (int, [2**64 - 1, 1]),
    "datetime_com_fuso_de_deslocamento": (
        datetime,
        [
            datetime(2024, 1, 2, tzinfo=_FUSO_DE_DESLOCAMENTO),
            datetime(2024, 1, 3, tzinfo=_FUSO_DE_DESLOCAMENTO),
        ],
    ),
    # O pyarrow põe o Decimal acima de 38 dígitos numa coluna de 256 bits, que o
    # polars não lê: a conversão entra em pânico, e o pânico não é Exception.
    "decimal_acima_de_38_digitos": (Decimal, [Decimal("1E+40"), Decimal(2)]),
    # Data ou hora com outro tipo na anotação: o Arrow tiparia a coluna pelo
    # primeiro valor e converteria os outros, e a anotação a manda para Object.
    "uniao_de_datetime_e_date": (
        Union[datetime, date],
        [date(2026, 1, 2), datetime.fromisoformat("2026-01-02T15:30")],
    ),
    "uniao_de_time_e_int_fora_de_um_dia": (Union[time, int], [time(1, 2), 86400000000]),
    "uniao_de_date_e_int": (Union[date, int], [date(2026, 1, 2), 5]),
    "tupla_de_date_e_int": (tuple[date, int], [(date(2026, 1, 2), 5), (date(2026, 1, 3), 6)]),
    "lista_de_uniao_de_date_e_datetime": (
        list[Union[date, datetime]],
        [[date(2026, 1, 2), datetime.fromisoformat("2026-01-02T10:00")], [date(2026, 1, 3)]],
    ),
    "tupla_de_date_e_int_opcional": (Optional[tuple[date, int]], [(date(2026, 1, 2), 5), None]),
    "uniao_de_date_anotado_e_int": (
        Optional[Union[Annotated[date, Field(description="dia")], int]],
        [date(2026, 1, 2), 5],
    ),
    "tupla_variavel_de_uniao_de_date_e_int": (
        tuple[Union[date, int], ...],
        [(date(2026, 1, 2), 5), (date(2026, 1, 2), 6), (date(2026, 1, 2), 7)],
    ),
    "uniao_com_barra_de_date_e_int": (date | int, [date(2026, 1, 2), 5, date(2026, 1, 2)]),
    "uniao_de_datetime_e_int": (
        Union[datetime, int],
        [
            datetime.fromisoformat("2026-01-02T15:30"),
            5,
            datetime.fromisoformat("2026-01-02T15:30"),
        ],
    ),
    # O NewType e o apelido `type` são resolvidos antes da regra de data ou hora.
    "uniao_de_newtype_de_date_e_int": (Union[_DataJudicial, int], [date(2026, 1, 2), 5]),
    "apelido_de_uniao_de_date_e_int": (_DataOuNumero, [date(2026, 1, 2), 5]),
    # Duração com número: o 5 viraria cinco microssegundos.
    "uniao_de_timedelta_e_int": (Union[timedelta, int], [timedelta(days=1), 5]),
    "uniao_de_timedelta_e_decimal": (Union[timedelta, Decimal], [timedelta(days=1), Decimal(2)]),
    "lista_de_uniao_de_timedelta_e_int": (
        list[Union[timedelta, int]],
        [[timedelta(days=1), 5], [timedelta(hours=1)]],
    ),
    # A anotação manda para Object mesmo quando as respostas são todas do tipo
    # temporal, porque o tipo da coluna dependeria das respostas.
    "uniao_de_timedelta_e_int_so_com_timedelta": (
        Union[timedelta, int],
        [timedelta(days=1), timedelta(hours=2)],
    ),
    "uniao_de_date_e_int_so_com_date": (Union[date, int], [date(2026, 1, 2), date(2026, 1, 3)]),
    # Sem a anotação para avisar, são os valores que juntam data ou duração a número.
    "any_com_timedelta_e_int": (Any, [timedelta(days=1), 5]),
    "any_com_date_e_int": (Any, [date(2026, 1, 2), 5]),
    # Dicts com chaves diferentes: o Struct teria a união das chaves, e a chave
    # ausente sairia nula.
    "any_com_dicts_de_chaves_diferentes": (Any, [{"a": 1}, {"b": 2}]),
    "any_com_dict_vazio_e_dict_com_chave": (Any, [{"a": 1}, {}]),
    "lista_de_dicts_de_chaves_diferentes": (
        list[dict[str, int]],
        [[{"a": 1}, {"b": 2}], [{"a": 3}]],
    ),
    "dict_de_submodelo_com_chaves_diferentes": (_ComDict, [{"extra": {"a": 1}}, {"extra": {}}]),
    # O True depois de um float viraria 1.0, e o texto depois de um bytes, bytes.
    "uniao_de_float_e_bool": (Union[float, bool], [1.5, True]),
    "uniao_de_bytes_e_str": (Union[bytes, str], [b"x", "a"]),
    # Pela anotação, mesmo quando as respostas são todas de um dos dois tipos.
    "uniao_de_float_e_bool_so_com_float": (Union[float, bool], [1.5, 2.5]),
    "uniao_de_bytes_e_str_so_com_str": (Union[bytes, str], ["a", "b"]),
    # Sem a anotação para avisar, são os valores que juntam float e bool.
    "any_com_float_e_bool": (Any, [1.5, True]),
    # O escalar do numpy conta pelo número do Python: o np.int64 depois de um
    # date viraria 1970-01-06, e o True depois de um np.float32, 1.0.
    "any_com_date_e_int64_do_numpy": (Any, [date(2026, 1, 2), np.int64(5)]),
    "any_com_float32_do_numpy_e_bool": (Any, [np.float32(1.5), True]),
    "submodelo_com_float_e_bool": (_ValorOuFlag, [{"valor": 1.5}, {"valor": True}]),
    # O datetime sem fuso ganharia o fuso do primeiro, e as 10h locais virariam 10h UTC.
    "datetime_com_e_sem_fuso": (
        Optional[datetime],
        [datetime(2026, 1, 2, 10, tzinfo=UTC), datetime.fromisoformat("2026-01-02T10:00")],
    ),
    "lista_de_datetime_com_e_sem_fuso": (
        list[datetime],
        [[datetime(2026, 1, 2, 10, tzinfo=UTC)], [datetime.fromisoformat("2026-01-02T10:00")]],
    ),
    # O tipo Time do polars não tem fuso, e o perderia.
    "time_com_fuso": (time, [time(10, tzinfo=UTC), time(11, tzinfo=UTC)]),
    # O pandas conta o Decimal NaN como ausente, e o pyarrow o gravaria nulo. A
    # lista compara o NaN pela identidade, e a coluna guarda o mesmo objeto.
    "decimal_nan": (
        Annotated[Decimal, Field(allow_inf_nan=True)],
        [Decimal("1.5"), Decimal("NaN")],
    ),
}


def _responde_por_linha(campo, valores):
    fila = iter(valores)
    return lambda text, *args, **kwargs: {"data": {campo: next(fila)}, "usage": None}


@pytest.mark.parametrize("caso", _CAMPOS_QUE_O_ARROW_NAO_TIPA)
def test_campo_que_o_arrow_nao_tipa_sai_como_object_com_os_valores_do_pandas(caso):
    pl = pytest.importorskip("polars")
    tipo, valores = _CAMPOS_QUE_O_ARROW_NAO_TIPA[caso]
    textos = [f"texto {i}" for i in range(len(valores))]
    modelo = create_model("ModeloDoCaso", campo=(tipo, ...))
    pandas, _ = _rodar(
        pd.DataFrame({"texto": textos}),
        questions=modelo,
        llm=_responde_por_linha("campo", valores),
    )

    with pytest.warns(UserWarning, match=r"\['campo'\].*Object"):
        resultado, _ = _rodar(
            pl.DataFrame({"texto": textos}),
            questions=modelo,
            llm=_responde_por_linha("campo", valores),
        )

    assert resultado["campo"].dtype == pl.Object
    assert resultado["campo"].to_list() == pandas["campo"].tolist() == valores


# Anotações que não juntam data ou hora a outro tipo, as respostas, o tipo
# polars inferido e os valores que saem dele.
_CAMPOS_TIPADOS_PELOS_VALORES = {
    "uniao_de_int_e_float": (Union[int, float], [1, 2.5], lambda pl: pl.Float64, [1.0, 2.5]),
    "uniao_de_int_e_str_so_com_int": (Union[int, str], [1, 2], lambda pl: pl.Int64, [1, 2]),
    "date_opcional": (
        Optional[date],
        [date(2026, 1, 2), None],
        lambda pl: pl.Date,
        [date(2026, 1, 2), None],
    ),
    "lista_de_date": (
        list[date],
        [[date(2026, 1, 2)], [date(2026, 1, 3)]],
        lambda pl: pl.List(pl.Date),
        [[date(2026, 1, 2)], [date(2026, 1, 3)]],
    ),
    "tupla_so_de_date": (
        tuple[date, ...],
        [(date(2026, 1, 2), date(2026, 1, 3)), (date(2026, 1, 4),)],
        lambda pl: pl.List(pl.Date),
        [[date(2026, 1, 2), date(2026, 1, 3)], [date(2026, 1, 4)]],
    ),
    "datetime_com_fuso": (
        datetime,
        [datetime(2026, 1, 2, 3, tzinfo=UTC), datetime(2026, 1, 3, tzinfo=UTC)],
        lambda pl: pl.Datetime("us", "UTC"),
        [datetime(2026, 1, 2, 3, tzinfo=UTC), datetime(2026, 1, 3, tzinfo=UTC)],
    ),
    # O nulo não conta como datetime sem fuso.
    "datetime_com_fuso_e_nulo": (
        Optional[datetime],
        [datetime(2026, 1, 2, 3, tzinfo=UTC), None],
        lambda pl: pl.Datetime("us", "UTC"),
        [datetime(2026, 1, 2, 3, tzinfo=UTC), None],
    ),
    "timedelta": (
        timedelta,
        [timedelta(days=1), timedelta(seconds=5)],
        lambda pl: pl.Duration("us"),
        [timedelta(days=1), timedelta(seconds=5)],
    ),
    # Dicts com as mesmas chaves cabem num Struct sem trocar valor.
    "any_com_dicts_de_mesmas_chaves": (
        Any,
        [{"a": 1, "b": "x"}, {"b": "y", "a": 2}],
        lambda pl: pl.Struct({"a": pl.Int64, "b": pl.Utf8}),
        [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}],
    ),
    # Cada campo do submodelo vira um campo do Struct, com tipo próprio, e a
    # data de um não se junta ao número do outro.
    "submodelo_com_date_e_int": (
        _DataEQuantidade,
        [{"dia": date(2026, 1, 2), "quantidade": 5}, {"dia": date(2026, 1, 3), "quantidade": 6}],
        lambda pl: pl.Struct({"dia": pl.Date, "quantidade": pl.Int64}),
        [{"dia": date(2026, 1, 2), "quantidade": 5}, {"dia": date(2026, 1, 3), "quantidade": 6}],
    ),
}


@pytest.mark.parametrize("caso", _CAMPOS_TIPADOS_PELOS_VALORES)
def test_campo_sem_data_ou_hora_com_outro_tipo_fica_com_o_tipo_inferido(caso):
    pl = pytest.importorskip("polars")
    tipo, respostas, esperado, valores = _CAMPOS_TIPADOS_PELOS_VALORES[caso]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        resultado, _ = _rodar(
            pl.DataFrame({"texto": ["a", "b"]}),
            questions=create_model("ModeloDoCaso", campo=(tipo, ...)),
            llm=_responde_por_linha("campo", respostas),
        )

    assert resultado["campo"].dtype == esperado(pl)
    assert resultado["campo"].to_list() == valores


def test_aviso_de_object_nomeia_so_a_coluna_que_o_arrow_nao_tipa():
    pl = pytest.importorskip("polars")
    respostas = iter([{"x": "p", "campo": 1}, {"x": "q", "campo": "dois"}])

    with pytest.warns(UserWarning, match="Object") as avisos:
        resultado, _ = _rodar(
            pl.DataFrame({"texto": ["a", "b"]}),
            questions=ModeloMisto,
            llm=lambda text, *args, **kwargs: {"data": next(respostas), "usage": None},
        )

    assert resultado["x"].to_list() == ["p", "q"]
    assert resultado["x"].dtype != pl.Object
    (aviso,) = [a for a in avisos if "Object" in str(a.message)]
    assert "['campo']" in str(aviso.message)
    # O aviso aponta para quem chamou dataframeit(), que aqui é _rodar.
    assert aviso.filename == __file__


def test_aviso_de_object_aponta_para_a_linha_que_chamou_dataframeit():
    pl = pytest.importorskip("polars")
    entrada = pl.DataFrame({"texto": ["a", "b"]})
    modelo = create_model("ModeloDoCaso", campo=(Union[int, str], ...))

    # Sem _rodar no meio: com ele, o quadro seguinte da pilha também fica neste
    # arquivo, e o stacklevel um nível acima do certo passaria.
    with (
        patch(
            "dataframeit.core.call_langchain", side_effect=_responde_por_linha("campo", [1, "x"])
        ),
        patch("dataframeit.core.validate_provider_dependencies"),
        warnings.catch_warnings(record=True) as avisos,
    ):
        warnings.simplefilter("always")
        dataframeit(entrada, questions=modelo, prompt="Analise {texto}", track_tokens=False)

    (aviso,) = [a for a in avisos if "Object" in str(a.message)]
    assert aviso.filename == __file__
    assert "dataframeit(entrada" in linecache.getline(aviso.filename, aviso.lineno)


def test_um_so_aviso_nomeia_todas_as_colunas_que_saem_object():
    pl = pytest.importorskip("polars")
    modelo = create_model("ModeloDoCaso", a=(Union[int, str], ...), b=(Union[int, str], ...))

    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        resultado, _ = _rodar(
            pl.DataFrame({"texto": ["a", "b"]}),
            questions=modelo,
            llm=_responde_em_sequencia({"a": 1, "b": "x"}, {"a": "dois", "b": 2}),
        )

    assert resultado["a"].dtype == resultado["b"].dtype == pl.Object
    (aviso,) = [a for a in avisos if "Object" in str(a.message)]
    assert "['a', 'b']" in str(aviso.message)


def test_enum_sai_pelo_valor_na_saida_polars():
    pl = pytest.importorskip("polars")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        resultado, _ = _rodar(
            pl.DataFrame({"texto": ["a", "b"]}),
            questions=ModeloCor,
            llm=_responde_por_linha("cor", [Cor.AZUL, Cor.VERDE]),
        )

    assert resultado["cor"].to_list() == ["azul", "verde"]
    assert resultado["cor"].dtype != pl.Object


def test_enum_sai_pelo_valor_tambem_na_coluna_object():
    pl = pytest.importorskip("polars")

    with pytest.warns(UserWarning, match=r"\['campo'\].*Object"):
        resultado, _ = _rodar(
            pl.DataFrame({"texto": ["a", "b", "c"]}),
            questions=create_model("ModeloDoCaso", campo=(Optional[Union[Cor, int]], ...)),
            llm=_responde_por_linha("campo", [Cor.AZUL, None, 2]),
        )

    assert resultado["campo"].dtype == pl.Object
    assert resultado["campo"].to_list() == ["azul", None, 2]


@pytest.mark.parametrize(
    ("respostas", "valores"),
    [
        ([Tinta.CLARA, Tinta.ESCURA], [Cor.AZUL, Cor.VERDE]),
        ([Tinta.CLARA, None], [Cor.AZUL, None]),
    ],
    ids=["so_membros", "membro_e_nulo"],
)
def test_valor_de_enum_que_e_membro_de_outro_enum_fica_inteiro_na_coluna_object(respostas, valores):
    """Sobra um membro de Enum na coluna Object, e o polars não o troca pelo valor dele."""
    pl = pytest.importorskip("polars")

    with pytest.warns(UserWarning, match=r"\['campo'\].*Object"):
        resultado, _ = _rodar(
            pl.DataFrame({"texto": ["a", "b"]}),
            questions=create_model("ModeloDoCaso", campo=(Optional[Tinta], ...)),
            llm=_responde_por_linha("campo", respostas),
        )

    assert resultado["campo"].dtype == pl.Object
    assert resultado["campo"].to_list() == valores


def test_campo_que_o_arrow_nao_tipa_com_entrada_polars_series():
    pl = pytest.importorskip("polars")

    with pytest.warns(UserWarning, match=r"\['campo'\].*Object"):
        resultado, _ = _rodar(
            pl.Series(["a", "b"]),
            questions=create_model("ModeloDoCaso", campo=(Union[int, str], ...)),
            llm=_responde_por_linha("campo", [1, "dois"]),
        )

    assert resultado["campo"].to_list() == [1, "dois"]


def test_interrupcao_na_conversao_da_saida_polars_sobe_em_vez_de_virar_object():
    """A saída captura a recusa da conversão e o pânico do polars, e só eles."""
    pl = pytest.importorskip("polars")

    with (
        patch("dataframeit.utils._to_polars", side_effect=KeyboardInterrupt),
        pytest.raises(KeyboardInterrupt),
    ):
        _rodar(pl.DataFrame({"texto": ["a", "b"]}))


def test_campo_any_so_com_dicts_vazios_sai_na_saida_polars():
    pl = pytest.importorskip("polars")

    resultado, llm = _rodar(
        pl.DataFrame({"texto": ["a", "b", "c"]}),
        questions=create_model("ModeloDoCaso", campo=(Any, ...)),
        llm=_responde_por_linha("campo", [{}, {}, {}]),
    )

    assert llm.call_count == 3
    assert resultado["campo"].to_list() == [{}, {}, {}]


@pytest.mark.parametrize("entrada", ["pandas", "polars"])
def test_retomada_de_saida_com_decimal_nan(entrada):
    pl = pytest.importorskip("polars")
    modelo = create_model(
        "ModeloDoCaso", campo=(Annotated[Decimal, Field(allow_inf_nan=True)], ...)
    )
    textos = {"texto": ["a", "b", "c"]}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        interrompida, _ = _rodar(
            pl.DataFrame(textos) if entrada == "polars" else pd.DataFrame(textos),
            questions=modelo,
            llm=_responde_em_sequencia(
                {"campo": Decimal("1.5")}, {"campo": Decimal("NaN")}, ProviderAbortError("limite")
            ),
        )
        retomada, llm = _rodar(
            interrompida,
            questions=modelo,
            llm=_responde_em_sequencia({"campo": Decimal(2)}),
            resume=True,
        )

    assert llm.call_count == 1
    primeiro, nan, terceiro = list(retomada["campo"])
    assert primeiro == Decimal("1.5")
    assert isinstance(nan, Decimal)
    assert nan.is_nan()
    assert terceiro == Decimal(2)


def test_segunda_interrupcao_mantem_a_coluna_datetime_com_fuso():
    # A linha que segue pendente depois da retomada traz o NaT da conversão da
    # entrada polars, que é datetime sem fuso e não pode contar como valor.
    pl = pytest.importorskip("polars")
    modelo = create_model("ModeloDoCaso", campo=(Optional[datetime], None))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        interrompida, _ = _rodar(
            pl.DataFrame({"texto": ["a", "b", "c"]}),
            questions=modelo,
            llm=_responde_em_sequencia(
                {"campo": datetime(2026, 1, 2, tzinfo=UTC)}, ProviderAbortError("limite")
            ),
        )
        retomada, _ = _rodar(
            interrompida,
            questions=modelo,
            llm=_responde_em_sequencia(
                {"campo": datetime(2026, 1, 3, tzinfo=UTC)}, ProviderAbortError("limite")
            ),
            resume=True,
        )

    assert retomada["campo"].dtype == pl.Datetime("us", "UTC")
    assert retomada["campo"].to_list() == [
        datetime(2026, 1, 2, tzinfo=UTC),
        datetime(2026, 1, 3, tzinfo=UTC),
        None,
    ]


def test_saida_polars_com_coluna_object_e_retomada():
    pl = pytest.importorskip("polars")
    modelo = create_model("ModeloDoCaso", campo=(Union[int, str], ...))
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        interrompida, _ = _rodar(
            pl.DataFrame({"texto": ["a", "b", "c"]}),
            questions=modelo,
            llm=_responde_em_sequencia(
                {"campo": 1}, {"campo": "dois"}, ProviderAbortError("limite")
            ),
        )
    mensagens = [str(a.message) for a in avisos]
    assert any("interrompida" in mensagem for mensagem in mensagens)
    assert any("Object" in mensagem for mensagem in mensagens)
    assert interrompida["campo"].dtype == pl.Object

    with pytest.warns(UserWarning, match="Object"):
        retomada, llm = _rodar(
            interrompida, questions=modelo, llm=_responde_em_sequencia({"campo": 3}), resume=True
        )

    assert llm.call_count == 1
    assert retomada["campo"].to_list() == [1, "dois", 3]


# =============================================================================
# Saída polars: campo dict em texto JSON
# =============================================================================


def test_campo_dict_sai_como_texto_json_na_saida_polars():
    pl = pytest.importorskip("polars")
    modelo = create_model("ModeloDoCaso", campo=(dict[str, int], ...))
    respostas = [{"a": 1}, {"b": 2}, {}]
    pandas, _ = _rodar(
        pd.DataFrame({"texto": ["a", "b", "c"]}),
        questions=modelo,
        llm=_responde_por_linha("campo", respostas),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        resultado, _ = _rodar(
            pl.DataFrame({"texto": ["a", "b", "c"]}),
            questions=modelo,
            llm=_responde_por_linha("campo", respostas),
        )

    assert resultado["campo"].dtype == pl.Utf8
    assert resultado["campo"].to_list() == ['{"a": 1}', '{"b": 2}', "{}"]
    assert [json.loads(texto) for texto in resultado["campo"]] == pandas["campo"].tolist()


# Anotação de dict, as respostas de duas linhas e o texto de cada uma na saída.
_CAMPOS_DICT = {
    "dict_sem_parametro": (dict, [{"a": 1}, {}], ['{"a": 1}', "{}"]),
    "optional_com_linha_nula": (Optional[dict[str, int]], [{"a": 1}, None], ['{"a": 1}', None]),
    "uniao_com_none": (dict[str, int] | None, [{"a": 1}, None], ['{"a": 1}', None]),
    "annotated_dentro_de_optional": (
        Optional[Annotated[dict[str, int], Field(min_length=1)]],
        [{"a": 1}, None],
        ['{"a": 1}', None],
    ),
    "valores_de_tipos_diferentes": (
        dict[str, Any],
        [{"a": 1}, {"a": [1]}],
        ['{"a": 1}', '{"a": [1]}'],
    ),
    "texto_acentuado": (dict[str, str], [{"órgão": "saúde"}, {}], ['{"órgão": "saúde"}', "{}"]),
    "valor_que_o_json_nao_serializa": (
        dict[str, date],
        [{"a": date(2024, 1, 2)}, {}],
        ['{"a": "2024-01-02"}', "{}"],
    ),
    "enum_como_valor": (dict[str, Cor], [{"a": Cor.AZUL}, {}], ['{"a": "azul"}', "{}"]),
}


@pytest.mark.parametrize("caso", _CAMPOS_DICT)
def test_campo_dict_sai_como_texto_json_em_cada_forma_de_anotacao(caso):
    pl = pytest.importorskip("polars")
    tipo, respostas, textos = _CAMPOS_DICT[caso]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        resultado, _ = _rodar(
            pl.DataFrame({"texto": ["a", "b"]}),
            questions=create_model("ModeloDoCaso", campo=(tipo, ...)),
            llm=_responde_por_linha("campo", respostas),
        )

    assert resultado["campo"].dtype == pl.Utf8
    assert resultado["campo"].to_list() == textos


def test_retomada_de_saida_polars_com_campo_dict_em_texto_json():
    pl = pytest.importorskip("polars")
    modelo = create_model("ModeloDoCaso", campo=(dict[str, int], ...))
    with pytest.warns(UserWarning, match="interrompida"):
        interrompida, _ = _rodar(
            pl.DataFrame({"texto": ["a", "b", "c"]}),
            questions=modelo,
            llm=_responde_em_sequencia({"campo": {"a": 1}}, ProviderAbortError("limite")),
        )
    assert interrompida["campo"].to_list() == ['{"a": 1}', None, None]

    retomada, llm = _rodar(
        interrompida,
        questions=modelo,
        llm=_responde_em_sequencia({"campo": {"b": 2}}, {"campo": {}}),
        resume=True,
    )

    assert llm.call_count == 2
    assert retomada["campo"].to_list() == ['{"a": 1}', '{"b": 2}', "{}"]


def test_modelo_aninhado_continua_struct_na_saida_polars():
    pl = pytest.importorskip("polars")

    resultado, _ = _rodar(
        pl.DataFrame({"texto": ["a", "b"]}),
        questions=create_model("ModeloDoCaso", campo=(Modelo, ...)),
        llm=_responde_por_linha("campo", [{"x": "p"}, {"x": "q"}]),
    )

    assert resultado["campo"].dtype == pl.Struct({"x": pl.Utf8})
    assert resultado["campo"].to_list() == [{"x": "p"}, {"x": "q"}]


def test_saida_polars_ja_concluida_mantem_o_campo_dict_em_texto_json():
    pl = pytest.importorskip("polars")
    entrada = pl.DataFrame(
        {"texto": ["a"], "campo": ['{"a": 1}'], "_dataframeit_status": ["processed"]}
    )

    resultado, llm = _rodar(
        entrada,
        questions=create_model("ModeloDoCaso", campo=(dict[str, int], ...)),
        resume=True,
    )

    llm.assert_not_called()
    assert resultado["campo"].to_list() == ['{"a": 1}']


# =============================================================================
# Saída polars: tipo da coluna sem valor
# =============================================================================

# Anotação, um valor dela e o tipo polars da coluna, com ou sem valor.
_TIPOS_PELA_ANOTACAO = {
    "str": (str, "a", lambda pl: pl.Utf8),
    "int": (int, 1, lambda pl: pl.Int64),
    "float": (float, 1.5, lambda pl: pl.Float64),
    "bool": (bool, True, lambda pl: pl.Boolean),
    "date": (date, date(2024, 1, 2), lambda pl: pl.Date),
    "datetime": (
        datetime,
        datetime.fromisoformat("2024-01-02T03:04"),
        lambda pl: pl.Datetime("us"),
    ),
    "lista_de_str": (list[str], ["a"], lambda pl: pl.List(pl.Utf8)),
    "lista_de_date": (list[date], [date(2024, 1, 2)], lambda pl: pl.List(pl.Date)),
    "int_com_restricao": (Annotated[int, Field(ge=0)], 1, lambda pl: pl.Int64),
    "newtype_de_int": (_Id, 1, lambda pl: pl.Int64),
    "apelido_de_date": (_ApelidoDeData, date(2024, 1, 2), lambda pl: pl.Date),
}


@pytest.mark.parametrize("caso", _TIPOS_PELA_ANOTACAO)
def test_coluna_toda_nula_sai_com_o_tipo_da_anotacao(caso):
    pl = pytest.importorskip("polars")
    tipo, valor, esperado = _TIPOS_PELA_ANOTACAO[caso]
    modelo = create_model("ModeloDoCaso", campo=(Optional[tipo], None))
    entrada = pl.DataFrame({"texto": ["a", "b"]})

    com_valor, _ = _rodar(
        entrada, questions=modelo, llm=_responde_por_linha("campo", [valor, None])
    )
    sem_valor, _ = _rodar(entrada, questions=modelo, llm=_responde(campo=None))

    assert com_valor["campo"].dtype == esperado(pl)
    assert sem_valor["campo"].dtype == esperado(pl)
    assert sem_valor["campo"].to_list() == [None, None]


@pytest.mark.parametrize("respostas", [[[], []], [[], None]])
def test_lista_so_com_listas_vazias_sai_com_o_tipo_da_anotacao(respostas):
    pl = pytest.importorskip("polars")

    resultado, _ = _rodar(
        pl.DataFrame({"texto": ["a", "b"]}),
        questions=create_model("ModeloDoCaso", campo=(Optional[list[str]], None)),
        llm=_responde_por_linha("campo", respostas),
    )

    assert resultado["campo"].dtype == pl.List(pl.Utf8)
    assert resultado["campo"].to_list() == respostas


# A resposta do dublê não passa pela validação do modelo, e um inteiro no campo
# float mostra que a coluna com algum valor fica com o tipo dos valores.
@pytest.mark.parametrize(
    ("tipo", "respostas", "esperado"),
    [
        (float, [7, None], lambda pl: pl.Int64),
        (list[float], [[7], []], lambda pl: pl.List(pl.Int64)),
    ],
)
def test_coluna_com_algum_valor_fica_com_o_tipo_inferido(tipo, respostas, esperado):
    pl = pytest.importorskip("polars")

    resultado, _ = _rodar(
        pl.DataFrame({"texto": ["a", "b"]}),
        questions=create_model("ModeloDoCaso", campo=(Optional[tipo], None)),
        llm=_responde_por_linha("campo", respostas),
    )

    assert resultado["campo"].dtype == esperado(pl)
    assert resultado["campo"].to_list() == respostas


# Anotações que não dão tipo à coluna sem valor, com respostas que não trazem valor.
_ANOTACOES_SEM_TIPO_PARA_A_COLUNA = {
    "decimal": (Optional[Decimal], [None, None]),
    "literal": (Optional[Literal[1, 2]], [None, None]),
    "enum": (Optional[Cor], [None, None]),
    "uniao_de_dois_tipos": (Union[int, str, None], [None, None]),
    "modelo_aninhado": (Optional[Modelo], [None, None]),
    "tupla": (Optional[tuple[int, str]], [None, None]),
    "lista_de_decimal": (list[Decimal], [[], []]),
    "lista_sem_item": (typing.List, [[], []]),  # noqa: UP006 (tem origem list e nenhum argumento)
    "lista_de_item_sem_hash": (list[Annotated[int, {"unidade": "dias"}]], [[], []]),
}


@pytest.mark.parametrize("caso", _ANOTACOES_SEM_TIPO_PARA_A_COLUNA)
def test_coluna_sem_valor_de_outra_anotacao_fica_com_o_tipo_inferido(caso):
    pl = pytest.importorskip("polars")
    tipo, respostas = _ANOTACOES_SEM_TIPO_PARA_A_COLUNA[caso]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        resultado, _ = _rodar(
            pl.DataFrame({"texto": ["a", "b"]}),
            questions=create_model("ModeloDoCaso", campo=(tipo, ...)),
            llm=_responde_por_linha("campo", respostas),
        )

    assert resultado["campo"].dtype == pl.from_pandas(pd.Series(respostas, dtype=object)).dtype
    assert resultado["campo"].to_list() == respostas


# =============================================================================
# NewType e apelido `type` contam pelo tipo a que se referem
# =============================================================================

# Anotação, respostas, tipo polars da saída e valores dela. O campo dict sai
# em texto JSON, como o anotado `dict` (ver test_campo_dict_sai_como_texto_json_na_saida_polars).
_NEWTYPE_E_APELIDO = {
    "newtype_de_dict": (
        _Metadados,
        [{"a": 1}, {"b": 2}, {}],
        lambda pl: pl.Utf8,
        ['{"a": 1}', '{"b": 2}', "{}"],
    ),
    "optional_de_newtype_de_dict": (
        Optional[_Metadados],
        [{"a": 1}, None, {}],
        lambda pl: pl.Utf8,
        ['{"a": 1}', None, "{}"],
    ),
    "apelido_de_dict": (
        _ApelidoDeDict,
        [{"a": 1}, {"b": 2}, {}],
        lambda pl: pl.Utf8,
        ['{"a": 1}', '{"b": 2}', "{}"],
    ),
    "newtype_de_str": (_Codigo, ["007", "N/A", "x"], lambda pl: pl.Utf8, ["007", "N/A", "x"]),
    "apelido_de_str": (
        _ApelidoDeTexto,
        ["007", "N/A", "x"],
        lambda pl: pl.Utf8,
        ["007", "N/A", "x"],
    ),
    "newtype_de_int": (_Id, [_ID_GRANDE, 7, 8], lambda pl: pl.Int64, [_ID_GRANDE, 7, 8]),
    "newtype_de_list": (
        _Etiquetas,
        [["a"], ["b", "c"], []],
        lambda pl: pl.List(pl.Utf8),
        [["a"], ["b", "c"], []],
    ),
    "apelido_de_list": (
        _ApelidoDeLista,
        [["a"], ["b", "c"], []],
        lambda pl: pl.List(pl.Utf8),
        [["a"], ["b", "c"], []],
    ),
    "apelido_generico_de_dict": (
        _ApelidoDeDictGenerico[int],
        [{"a": 1}, {"b": 2}, {}],
        lambda pl: pl.Utf8,
        ['{"a": 1}', '{"b": 2}', "{}"],
    ),
    "apelido_generico_de_list": (
        _ApelidoDeListaGenerica[str],
        [["a"], ["b", "c"], []],
        lambda pl: pl.List(pl.Utf8),
        [["a"], ["b", "c"], []],
    ),
}


class _Placa(str):
    """Texto com um atributo `__value__` próprio, que não faz dele um apelido."""

    __slots__ = ()
    __value__ = Union[date, int]

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        return core_schema.no_info_after_validator_function(cls, core_schema.str_schema())


def test_classe_com_atributo_value_nao_e_apelido_na_saida_polars():
    # Como apelido, a anotação juntaria date a int e mandaria a coluna para Object.
    pl = pytest.importorskip("polars")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        resultado, _ = _rodar(
            pl.DataFrame({"texto": ["a", "b"]}),
            questions=create_model("ModeloDoCaso", campo=(_Placa, ...)),
            llm=_responde_por_linha("campo", ["ABC1234", "XYZ9876"]),
        )

    assert resultado["campo"].dtype == pl.Utf8
    assert resultado["campo"].to_list() == ["ABC1234", "XYZ9876"]


@pytest.mark.parametrize("caso", _NEWTYPE_E_APELIDO)
def test_newtype_e_apelido_saem_e_retomam_na_saida_polars(caso):
    pl = pytest.importorskip("polars")
    tipo, respostas, esperado, valores = _NEWTYPE_E_APELIDO[caso]
    modelo = create_model("ModeloDoCaso", campo=(tipo, ...))
    entrada = pl.DataFrame({"texto": ["a", "b", "c"]})
    interrompida = _rodar_sem_avisos(
        entrada,
        modelo,
        _responde_em_sequencia({"campo": respostas[0]}, ProviderAbortError("limite")),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        retomada, llm = _rodar(
            interrompida,
            questions=modelo,
            llm=_responde_por_linha("campo", respostas[1:]),
            resume=True,
        )
        sem_interrupcao, _ = _rodar(
            entrada, questions=modelo, llm=_responde_por_linha("campo", respostas)
        )

    assert llm.call_count == 2
    assert retomada["campo"].dtype == esperado(pl)
    assert retomada["campo"].to_list() == valores
    assert retomada.equals(sem_interrupcao)


# =============================================================================
# NaN de campo float
# =============================================================================

# Anotações cuja coluna guarda o NaN como valor: admitem float e não são estrutura.
_ANOTACOES_COM_FLOAT = {
    "float": float,
    "optional": Optional[float],
    "annotated": Annotated[float, Field(allow_inf_nan=True)],
    "uniao_de_float_e_str": Union[float, str],
    "newtype_de_float": _Medida,
    "apelido_generico_de_float": _ApelidoOpcional[float],
}


def _nan_na_posicao(valores, posicao):
    valor = valores[posicao]
    return isinstance(valor, float) and math.isnan(valor)


@pytest.mark.parametrize("entrada", ["pandas", "polars"])
@pytest.mark.parametrize("caso", _ANOTACOES_COM_FLOAT)
def test_nan_de_campo_float_sai_nan_e_retoma(caso, entrada):
    pl = pytest.importorskip("polars")
    modelo = create_model("ModeloDoCaso", campo=(_ANOTACOES_COM_FLOAT[caso], ...))
    textos = {"texto": ["a", "b", "c"]}
    interrompida = _rodar_sem_avisos(
        pl.DataFrame(textos) if entrada == "polars" else pd.DataFrame(textos),
        modelo,
        _responde_em_sequencia(
            {"campo": 1.5}, {"campo": float("nan")}, ProviderAbortError("limite")
        ),
    )
    valores = list(interrompida["campo"])
    assert valores[0] == 1.5
    assert _nan_na_posicao(valores, 1)
    assert valores[2] is None

    retomada, llm = _rodar(
        interrompida, questions=modelo, llm=_responde_em_sequencia({"campo": 2.0}), resume=True
    )

    assert llm.call_count == 1
    valores = list(retomada["campo"])
    assert valores[0] == 1.5
    assert _nan_na_posicao(valores, 1)
    assert valores[2] == 2.0
    if entrada == "polars":
        assert retomada["campo"].dtype == pl.Float64


def test_nan_e_none_de_campo_float_opcional_seguem_distintos_na_retomada_polars():
    # A linha da resposta None e a pendente voltam None da entrada polars, e o
    # NaN volta NaN; com o to_pandas, as três seriam NaN.
    pl = pytest.importorskip("polars")
    modelo = create_model("ModeloDoCaso", campo=(Optional[float], ...))
    interrompida = _rodar_sem_avisos(
        pl.DataFrame({"texto": ["a", "b", "c", "d"]}),
        modelo,
        _responde_em_sequencia(
            {"campo": float("nan")}, {"campo": None}, ProviderAbortError("limite")
        ),
    )

    with pytest.warns(UserWarning, match="interrompida"):
        retomada, llm = _rodar(
            interrompida,
            questions=modelo,
            llm=_responde_em_sequencia({"campo": 2.0}, ProviderAbortError("limite")),
            resume=True,
        )

    assert llm.call_count == 2
    valores = retomada["campo"].to_list()
    assert _nan_na_posicao(valores, 0)
    assert valores[1:] == [None, 2.0, None]
    assert retomada["campo"].null_count() == 2


def test_nulo_de_estrutura_com_float_segue_nulo_na_retomada_polars():
    # A retomada normaliza a coluna de estrutura com Series.apply, que leva o
    # None de uma coluna só de números a NaN; nela o NaN não é valor.
    pl = pytest.importorskip("polars")
    modelo = create_model("ModeloDoCaso", campo=(Optional[Union[float, list[float]]], ...))
    interrompida = _rodar_sem_avisos(
        pl.DataFrame({"texto": ["a", "b", "c"]}),
        modelo,
        _responde_em_sequencia({"campo": 1.5}, {"campo": None}, ProviderAbortError("limite")),
    )

    retomada, llm = _rodar(
        interrompida, questions=modelo, llm=_responde_em_sequencia({"campo": 2.0}), resume=True
    )

    assert llm.call_count == 1
    assert retomada["campo"].to_list() == [1.5, None, 2.0]


class _ComMedida(BaseModel):
    medida: float


@pytest.mark.parametrize(
    ("tipo", "resposta", "saida"),
    [
        (list[float], [1.5, float("nan")], [1.5, None]),
        (_ComMedida, {"medida": float("nan")}, {"medida": None}),
    ],
    ids=["lista_de_float", "submodelo_com_float"],
)
def test_nan_dentro_de_estrutura_sai_nulo_e_a_retomada_polars_pede_reprocessamento(
    tmp_path, tipo, resposta, saida
):
    # Dentro de estrutura, a saída polars põe o NaN no nulo, que o campo float
    # recusa; a retomada dessa saída pede reprocess_columns. O checkpoint grava
    # a estrutura em JSON com o NaN e retoma sem nova chamada.
    pl = pytest.importorskip("polars")
    modelo = create_model("ModeloDoCaso", campo=(tipo, ...))
    entrada = pl.DataFrame({"texto": ["a", "b"]})
    checkpoint = {"batch_size": 1, "checkpoint_path": tmp_path / "ckpt.csv"}
    interrompida = _rodar_sem_avisos(
        entrada,
        modelo,
        _responde_em_sequencia({"campo": resposta}, ProviderAbortError("limite")),
        **checkpoint,
    )
    assert interrompida["campo"].to_list()[0] == saida

    with pytest.raises(ValueError, match="reprocess_columns"):
        _rodar(interrompida, questions=modelo, llm=_responde(campo=resposta), resume=True)

    retomada, llm = _rodar(
        entrada, questions=modelo, llm=_responde(campo=resposta), resume=True, **checkpoint
    )
    assert llm.call_count == 1
    assert retomada["campo"].to_list() == [saida, saida]


def test_nan_de_coluna_float64_conta_como_ausente_na_retomada():
    # Na coluna float64, como a do CSV relido, o NaN é também a resposta None e
    # não se distingue dela; como valor, ele seria recusado pelo gt=0.
    modelo = create_model("ModeloDoCaso", campo=(Optional[Annotated[float, Field(gt=0)]], None))
    entrada = pd.DataFrame(
        {
            "texto": ["a", "b"],
            "campo": [1.5, np.nan],
            "_dataframeit_status": ["processed", "processed"],
        }
    )

    retomada, llm = _rodar(entrada, questions=modelo, llm=_responde(campo=2.0), resume=True)

    assert llm.call_count == 0
    assert retomada["campo"].iloc[0] == 1.5
    assert pd.isna(retomada["campo"].iloc[1])


@pytest.mark.parametrize("formato", ["parquet", "csv"])
@pytest.mark.parametrize("entrada", ["pandas", "polars"])
@pytest.mark.parametrize(
    ("segunda_rodada", "esperado"),
    [
        ([RuntimeError("falha"), {"campo": 2.5}], [1.5, None, 2.5, None]),
        ([{"campo": 2.5}, {"campo": 3.0}], [1.5, 2.5, 3.0, None]),
    ],
    ids=["com_linha_de_erro", "sem_linha_de_erro"],
)
def test_linha_pendente_relida_de_checkpoint_float64_sai_nula(
    tmp_path, formato, entrada, segunda_rodada, esperado
):
    # O checkpoint relido traz o campo float numa coluna float64, em que a
    # linha pendente é NaN. Ela sai nula, como sem o checkpoint, e não NaN,
    # que na coluna desse campo é valor.
    pl = pytest.importorskip("polars")
    if formato == "parquet":
        pytest.importorskip("pyarrow")
    modelo = create_model("ModeloDoCaso", campo=(Optional[float], None))
    textos = {"texto": ["a", "b", "c", "d"]}
    df = pl.DataFrame(textos) if entrada == "polars" else pd.DataFrame(textos)
    opcoes = {"batch_size": 1, "checkpoint_path": tmp_path / f"ckpt.{formato}", "max_retries": 1}
    _rodar_sem_avisos(
        df, modelo, _responde_em_sequencia({"campo": 1.5}, ProviderAbortError("limite")), **opcoes
    )

    retomada = _rodar_sem_avisos(
        df,
        modelo,
        _responde_em_sequencia(*segunda_rodada, ProviderAbortError("limite")),
        resume=True,
        **opcoes,
    )

    assert list(retomada["campo"]) == esperado
    if entrada == "polars":
        assert retomada["campo"].dtype == pl.Float64
        assert retomada["campo"].null_count() == esperado.count(None)


@pytest.mark.parametrize("formato", ["parquet", "csv"])
@pytest.mark.parametrize("entrada", ["pandas", "polars"])
def test_linha_de_erro_relida_de_checkpoint_concluido_sai_nula(tmp_path, formato, entrada):
    # O checkpoint sem linha pendente sai sem nova chamada, e a linha de erro,
    # que a validação não percorre, sai nula, e não com o NaN da coluna float64.
    pl = pytest.importorskip("polars")
    if formato == "parquet":
        pytest.importorskip("pyarrow")
    modelo = create_model("ModeloDoCaso", campo=(Optional[float], None))
    textos = {"texto": ["a", "b", "c"]}
    df = pl.DataFrame(textos) if entrada == "polars" else pd.DataFrame(textos)
    opcoes = {"batch_size": 1, "checkpoint_path": tmp_path / f"ckpt.{formato}", "max_retries": 1}
    _rodar_sem_avisos(
        df,
        modelo,
        _responde_em_sequencia({"campo": 1.5}, RuntimeError("falha"), {"campo": None}),
        **opcoes,
    )

    retomada, llm = _rodar(df, questions=modelo, resume=True, **opcoes)

    assert llm.call_count == 0
    assert list(retomada["_dataframeit_status"]) == ["processed", "error", "processed"]
    assert list(retomada["campo"]) == [1.5, None, None]


@pytest.mark.parametrize("formato", ["parquet", "csv"])
def test_nan_de_execucao_sem_interrupcao_com_checkpoint_sai_nan(tmp_path, formato):
    # Sem releitura, a coluna do campo float é object, e o NaN que o modelo
    # devolveu segue valor, separado do None.
    pl = pytest.importorskip("polars")
    if formato == "parquet":
        pytest.importorskip("pyarrow")
    modelo = create_model("ModeloDoCaso", campo=(Optional[float], None))

    resultado = _rodar_sem_avisos(
        pl.DataFrame({"texto": ["a", "b", "c"]}),
        modelo,
        _responde_em_sequencia({"campo": 1.5}, {"campo": float("nan")}, {"campo": None}),
        batch_size=1,
        checkpoint_path=tmp_path / f"ckpt.{formato}",
    )

    valores = resultado["campo"].to_list()
    assert valores[0] == 1.5
    assert _nan_na_posicao(valores, 1)
    assert valores[2] is None


def test_nan_de_campo_any_vindo_da_entrada_polars_conta_como_nulo():
    # A coluna Float64 toda nula de um campo Any chega ao pandas com NaN, e a
    # linha sem texto o mantém. Ao lado do True, o NaN contado como float
    # juntaria os dois tipos e mandaria a coluna para Object.
    pl = pytest.importorskip("polars")
    entrada = pl.DataFrame(
        {"texto": ["a", None], "campo": pl.Series([None, None], dtype=pl.Float64)}
    )

    resultado = _rodar_sem_avisos(
        entrada,
        create_model("ModeloDoCaso", campo=(Any, ...)),
        _responde(campo=True),
        resume=True,
    )

    assert resultado["campo"].dtype == pl.Boolean
    assert resultado["campo"].to_list() == [True, None]


@pytest.mark.filterwarnings("ignore:Configuração pode exceder rate limits")
@pytest.mark.parametrize("parallel_requests", [1, 2])
def test_reprocessamento_por_campo_recebe_o_nan_gravado_como_valor(parallel_requests):
    # O backend campo a campo recebe os campos que não reprocessa com o valor
    # gravado, que alimenta as condições; o NaN que a validação aceitou como
    # valor vai como NaN, e não como None. Vale no caminho serial e no paralelo.
    conhecidos = []

    def fabrica(modelo, prompt, config, trace_mode):
        def por_campo(text, only_fields=None, known=None):
            conhecidos.append(known)
            return {"data": {"rotulo": "novo"}, "usage": None}

        return por_campo

    modelo = create_model(
        "ModeloDoCaso",
        valor=(Annotated[Decimal, Field(allow_inf_nan=True)], ...),
        medida=(float, ...),
        rotulo=(str, ...),
    )
    entrada = pd.DataFrame(
        {
            "texto": ["a"],
            "valor": pd.Series([Decimal("NaN")], dtype=object),
            "medida": pd.Series([float("nan")], dtype=object),
            "rotulo": ["antigo"],
            "_dataframeit_status": ["processed"],
        }
    )
    with (
        patch("dataframeit.agent.field_extractor", side_effect=fabrica),
        patch("dataframeit.core.validate_provider_dependencies"),
        patch("dataframeit.core.validate_search_dependencies"),
    ):
        saida = dataframeit(
            entrada,
            questions=modelo,
            prompt="{texto}",
            use_search=True,
            search_per_field=True,
            reprocess_columns=["rotulo"],
            parallel_requests=parallel_requests,
            track_tokens=False,
        )

    [conhecido] = conhecidos
    assert isinstance(conhecido["valor"], Decimal)
    assert conhecido["valor"].is_nan()
    assert _nan_na_posicao([conhecido["medida"]], 0)
    assert saida["rotulo"].tolist() == ["novo"]
