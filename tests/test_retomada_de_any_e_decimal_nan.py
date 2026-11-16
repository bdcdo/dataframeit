"""Retomada do campo `Any` e do `Decimal` NaN: o valor que o modelo devolveu volta como foi."""

import math
import warnings
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any, Optional, Union
from unittest.mock import patch

import pandas as pd
import pytest
from pydantic import BaseModel, ConfigDict, Field, create_model

from dataframeit import ProviderAbortError, dataframeit, read_df
from dataframeit.core import _save_checkpoint, _structures_as_json
from dataframeit.utils import _admits_any, _admits_decimal

_TEXTOS = {"texto": ["a", "b", "c"]}
_DECIMAL_COM_NAN = Annotated[Decimal, Field(allow_inf_nan=True)]


def _entrada(tipo: str):
    if tipo == "polars":
        return pytest.importorskip("polars").DataFrame(_TEXTOS)
    return pd.DataFrame(_TEXTOS)


def _exige_formato(ext: str) -> None:
    if ext != ".csv":
        pytest.importorskip("openpyxl" if ext == ".xlsx" else "pyarrow")


def _rodar(df, anotacao, respostas, **opcoes):
    """Roda com uma resposta de `campo` por chamada; a que for exceção é levantada."""
    modelo = (
        anotacao
        if isinstance(anotacao, type) and issubclass(anotacao, BaseModel)
        else create_model("ModeloDoCaso", campo=(anotacao, ...))
    )
    fila = iter(respostas)

    def llm(*args, **kwargs):
        resposta = next(fila)
        if isinstance(resposta, Exception):
            raise resposta
        return {"data": {"campo": resposta}, "usage": None}

    with (
        warnings.catch_warnings(),
        patch("dataframeit.core.call_langchain", side_effect=llm) as chamadas,
        patch("dataframeit.core.validate_provider_dependencies"),
    ):
        warnings.simplefilter("ignore")
        saida = dataframeit(
            df, questions=modelo, prompt="Analise {texto}", track_tokens=False, **opcoes
        )
    return saida, chamadas.call_count


def _retomar_do_checkpoint(caminho, entrada, anotacao, concluidas, nova):
    """Interrompe depois de `concluidas` e retoma pelo checkpoint com a resposta `nova`."""
    opcoes = {"checkpoint_path": str(caminho), "batch_size": 1}
    _rodar(_entrada(entrada), anotacao, [*concluidas, ProviderAbortError("limite")], **opcoes)
    saida, chamadas = _rodar(_entrada(entrada), anotacao, [nova], resume=True, **opcoes)
    assert chamadas == 1
    return list(saida["campo"])


# =============================================================================
# Anotações
# =============================================================================


@pytest.mark.parametrize(
    ("anotacao", "esperado"),
    [
        (Any, True),
        (Optional[Any], True),
        (object, True),
        (Union[Any, int], True),
        (float, False),
        (dict[str, Any], False),
    ],
)
def test_admits_any(anotacao, esperado):
    assert _admits_any(anotacao) is esperado


@pytest.mark.parametrize(
    ("anotacao", "esperado"),
    [
        (Decimal, True),
        (Optional[_DECIMAL_COM_NAN], True),
        (Union[Decimal, int], True),
        (float, False),
        (list[Decimal], False),
    ],
)
def test_admits_decimal(anotacao, esperado):
    assert _admits_decimal(anotacao) is esperado


# =============================================================================
# NaN de `Any` na retomada da saída pandas
# =============================================================================


@pytest.mark.parametrize("anotacao", [Any, Optional[Any], object])
def test_retomada_da_saida_pandas_mantem_o_nan_de_any(anotacao):
    interrompida, _ = _rodar(
        pd.DataFrame(_TEXTOS), anotacao, [1.5, float("nan"), ProviderAbortError("limite")]
    )

    retomada, chamadas = _rodar(interrompida, anotacao, [2.0], resume=True)

    assert chamadas == 1
    primeiro, nan, terceiro = retomada["campo"].tolist()
    assert (primeiro, terceiro) == (1.5, 2.0)
    assert isinstance(nan, float)
    assert math.isnan(nan)


def test_retomada_da_saida_pandas_mantem_o_none_ao_lado_do_nan_de_any():
    interrompida, _ = _rodar(
        pd.DataFrame(_TEXTOS), Any, [None, float("nan"), ProviderAbortError("limite")]
    )

    retomada, _ = _rodar(interrompida, Any, ["x"], resume=True)

    primeiro, nan, terceiro = retomada["campo"].tolist()
    assert primeiro is None
    assert math.isnan(nan)
    assert terceiro == "x"


# =============================================================================
# Estrutura de `Any` relida de checkpoint
# =============================================================================

_ESTRUTURAS = {
    "mesmas_chaves": [{"a": 1}, {"a": 2}],
    "chaves_diferentes": [{"a": 1}, {"b": 2}],
    "lista_de_dicts": [[{"a": 1}, {"b": 2}], [{"a": 3}]],
    "dict_aninhado": [{"a": {"b": [1, 2]}}, {"a": None}],
}


@pytest.mark.parametrize("caso", _ESTRUTURAS)
@pytest.mark.parametrize("ext", [".parquet", ".csv", ".xlsx"])
@pytest.mark.parametrize("entrada", ["pandas", "polars"])
def test_retomada_do_checkpoint_devolve_a_estrutura_de_any(tmp_path, entrada, ext, caso):
    _exige_formato(ext)
    respostas = _ESTRUTURAS[caso]

    valores = _retomar_do_checkpoint(tmp_path / f"ck{ext}", entrada, Any, respostas, respostas[0])

    assert valores == [*respostas, respostas[0]]
    # O 1 que voltasse 1.0 seria igual em Python.
    assert repr(valores) == repr([*respostas, respostas[0]])


@pytest.mark.parametrize("ext", [".parquet", ".csv", ".xlsx"])
def test_retomada_do_checkpoint_devolve_none_ao_lado_da_estrutura_de_any(tmp_path, ext):
    _exige_formato(ext)

    valores = _retomar_do_checkpoint(
        tmp_path / f"ck{ext}", "pandas", Optional[Any], [{"a": 1}, None], [2]
    )

    assert valores == [{"a": 1}, None, [2]]


@pytest.mark.parametrize("ext", [".parquet", ".csv", ".xlsx"])
def test_retomada_do_checkpoint_devolve_none_ao_lado_do_texto_de_any(tmp_path, ext):
    _exige_formato(ext)

    valores = _retomar_do_checkpoint(tmp_path / f"ck{ext}", "pandas", Any, ["x", None], "y")

    assert valores == ["x", None, "y"]


@pytest.mark.parametrize("ext", [".parquet", ".csv"])
def test_retomada_do_checkpoint_mantem_o_numero_de_any(tmp_path, ext):
    _exige_formato(ext)

    valores = _retomar_do_checkpoint(tmp_path / f"ck{ext}", "pandas", Any, [1.5, 2.5], 3.5)

    assert valores == [1.5, 2.5, 3.5]


def test_parquet_grava_a_estrutura_de_any_como_json(tmp_path):
    pytest.importorskip("pyarrow")
    caminho = tmp_path / "ck.parquet"

    _rodar(
        pd.DataFrame(_TEXTOS),
        Any,
        [{"a": 1}, {"b": 2}, ProviderAbortError("limite")],
        checkpoint_path=str(caminho),
        batch_size=1,
    )

    gravado = pd.read_parquet(caminho)["campo"].tolist()
    assert gravado[:2] == ['{"a": 1}', '{"b": 2}']
    assert pd.isna(gravado[2])


def test_read_df_com_modelo_devolve_a_estrutura_de_any(tmp_path):
    modelo = create_model("ModeloDoCaso", campo=(Any, ...), outro=(Any, ...), numero=(Any, ...))
    caminho = tmp_path / "saida.csv"
    pd.DataFrame(
        {
            "campo": ['{"a": 1}', None, "[1, 2]"],
            "outro": ["x", None, "y"],
            "numero": [1.5, None, 2.5],
        }
    ).to_csv(caminho, index=False)

    lido = read_df(caminho, modelo)

    assert lido["campo"].dtype == object
    assert lido["campo"].tolist() == [{"a": 1}, None, [1, 2]]
    # O número fica como o leitor o deu; o texto sem JSON, com a célula vazia ausente.
    pd.testing.assert_series_equal(lido["numero"], pd.read_csv(caminho)["numero"])
    assert lido["outro"].dropna().tolist() == ["x", "y"]
    assert lido["outro"].isna().tolist() == [False, True, False]


def test_read_df_com_modelo_devolve_none_na_celula_vazia_da_coluna_de_objetos_de_any(tmp_path):
    pytest.importorskip("openpyxl")
    modelo = create_model("ModeloDoCaso", campo=(Any, ...))
    caminho = tmp_path / "saida.xlsx"
    pd.DataFrame({"campo": pd.Series([1.5, None, "x"], dtype=object)}).to_excel(
        caminho, index=False
    )

    lido = read_df(caminho, modelo)

    assert lido["campo"].dtype == object
    assert lido["campo"].tolist() == [1.5, None, "x"]


def test_retomada_do_arquivo_relido_devolve_none_na_celula_vazia_de_any(tmp_path):
    pytest.importorskip("openpyxl")
    caminho = tmp_path / "saida.xlsx"
    interrompida, _ = _rodar(pd.DataFrame(_TEXTOS), Optional[Any], [1.5, None, "x"])
    interrompida.assign(_dataframeit_status="processed").to_excel(caminho, index=False)
    modelo = create_model("ModeloDoCaso", campo=(Optional[Any], ...))

    retomada, chamadas = _rodar(read_df(caminho, modelo), modelo, [], resume=True)

    assert chamadas == 0
    assert retomada["campo"].tolist() == [1.5, None, "x"]


def test_read_df_sem_normalizar_deixa_o_json_de_any_em_texto(tmp_path):
    modelo = create_model("ModeloDoCaso", campo=(Any, ...))
    caminho = tmp_path / "saida.csv"
    pd.DataFrame({"campo": ['{"a": 1}']}).to_csv(caminho, index=False)

    assert read_df(caminho, modelo, normalize=False)["campo"].tolist() == ['{"a": 1}']


# =============================================================================
# Decimal NaN relido de checkpoint
# =============================================================================


def _confere_decimais(valores, esperados):
    assert len(valores) == len(esperados)
    for valor, esperado in zip(valores, esperados, strict=True):
        if esperado is None:
            assert valor is None
            continue
        assert isinstance(valor, Decimal)
        assert str(valor) == str(esperado)


@pytest.mark.parametrize("ext", [".csv", ".xlsx", ".parquet"])
@pytest.mark.parametrize("entrada", ["pandas", "polars"])
def test_retomada_do_checkpoint_devolve_o_decimal_nan(tmp_path, entrada, ext):
    _exige_formato(ext)

    valores = _retomar_do_checkpoint(
        tmp_path / f"ck{ext}",
        entrada,
        _DECIMAL_COM_NAN,
        [Decimal("1.5"), Decimal("NaN")],
        Decimal("2.25"),
    )

    _confere_decimais(valores, [Decimal("1.5"), Decimal("NaN"), Decimal("2.25")])


@pytest.mark.parametrize("ext", [".csv", ".xlsx", ".parquet"])
def test_retomada_do_checkpoint_devolve_o_decimal_nan_com_modelo_estrito(tmp_path, ext):
    _exige_formato(ext)

    class Estrito(BaseModel):
        model_config = ConfigDict(strict=True)

        campo: _DECIMAL_COM_NAN

    valores = _retomar_do_checkpoint(
        tmp_path / f"ck{ext}", "pandas", Estrito, [Decimal("1.5"), Decimal("NaN")], Decimal(2)
    )

    _confere_decimais(valores, [Decimal("1.5"), Decimal("NaN"), Decimal(2)])


@pytest.mark.parametrize("ext", [".csv", ".xlsx", ".parquet"])
def test_retomada_do_checkpoint_devolve_none_ao_lado_do_decimal_nan(tmp_path, ext):
    _exige_formato(ext)

    valores = _retomar_do_checkpoint(
        tmp_path / f"ck{ext}",
        "pandas",
        Optional[_DECIMAL_COM_NAN],
        [None, Decimal("-NaN12")],
        Decimal(2),
    )

    _confere_decimais(valores, [None, Decimal("-NaN12"), Decimal(2)])


def test_checkpoint_grava_como_texto_os_decimais_da_coluna_com_nan():
    df = pd.DataFrame(
        {
            "com_nan": pd.Series([Decimal("1.50"), Decimal("NaN"), None], dtype=object),
            "sem_nan": pd.Series([Decimal("1.50"), Decimal(2), None], dtype=object),
        }
    )

    gravado = _structures_as_json(df, columns=())

    assert gravado["com_nan"].tolist()[:2] == ["1.50", "NaN"]
    assert pd.isna(gravado["com_nan"].tolist()[2])
    assert gravado["sem_nan"].tolist() == [Decimal("1.50"), Decimal(2), None]


def test_csv_grava_o_decimal_nan_como_texto_e_a_linha_pendente_vazia(tmp_path):
    caminho = tmp_path / "ck.csv"
    df = pd.DataFrame({"campo": pd.Series([Decimal("1.5"), Decimal("NaN"), None], dtype=object)})

    _save_checkpoint(df, caminho)

    assert caminho.read_text(encoding="utf-8") == 'campo\n1.5\nNaN\n""\n'


class _ModeloDecimal(BaseModel):
    campo: Decimal


def _csv(tmp_path: Path, celulas: list[str]) -> Path:
    caminho = tmp_path / "saida.csv"
    caminho.write_text("\n".join(["outra,campo", *celulas]) + "\n", encoding="utf-8")
    return caminho


def test_read_df_devolve_o_decimal_nan_e_none_na_celula_vazia(tmp_path):
    lido = read_df(_csv(tmp_path, ["a,1.5", "b,NaN", "c,", "d,-NaN"]), _ModeloDecimal)

    assert lido["campo"].dtype == object
    primeiro, nan, vazio, negativo = lido["campo"].tolist()
    assert primeiro == 1.5
    assert isinstance(nan, Decimal)
    assert str(nan) == "NaN"
    assert vazio is None
    assert str(negativo) == "-NaN"


@pytest.mark.parametrize("celula", ["nan", "NA", "NaN ", "xNaN", "NaNx", "sNaN"])
def test_read_df_so_toma_por_decimal_nan_o_texto_que_o_checkpoint_grava(tmp_path, celula):
    caminho = _csv(tmp_path, ["a,1.5", f"b,{celula}"])

    lido = read_df(caminho, _ModeloDecimal)

    pd.testing.assert_frame_equal(lido, pd.read_csv(caminho))


def test_read_df_sem_decimal_nan_deixa_a_coluna_como_lida(tmp_path):
    lido = read_df(_csv(tmp_path, ["a,1.5", "b,"]), _ModeloDecimal)

    assert lido["campo"].dtype == "float64"


def test_read_df_com_argumento_de_leitura_nao_rele_o_decimal_nan(tmp_path):
    lido = read_df(_csv(tmp_path, ["a,1.5", "b,NaN"]), _ModeloDecimal, sep=",")

    assert lido["campo"].dtype == "float64"


def test_read_df_de_json_nao_rele_o_decimal_nan(tmp_path):
    caminho = tmp_path / "saida.json"
    pd.DataFrame({"campo": ["1.5", "NaN"]}).to_json(caminho)

    pd.testing.assert_frame_equal(read_df(caminho, _ModeloDecimal), pd.read_json(caminho))


def test_read_df_so_rele_o_decimal_nan_no_campo_que_admite_decimal(tmp_path):
    modelo = create_model("ModeloDoCaso", campo=(float, ...))
    caminho = _csv(tmp_path, ["a,1.5", "b,NaN"])

    pd.testing.assert_frame_equal(read_df(caminho, modelo), pd.read_csv(caminho))
