"""Retomada do campo `Any`, do `Decimal` e do NaN: o valor que o modelo devolveu volta como foi."""

import math
import warnings
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any, Optional, Union
from unittest.mock import patch

import pandas as pd
import pytest
from pydantic import BaseModel, ConfigDict, Field, create_model

from dataframeit import ProviderAbortError, core, dataframeit, read_df
from dataframeit.core import _same_value, _save_checkpoint, _structures_as_json
from dataframeit.utils import (
    _READER_NA_TEXTS,
    _admits_any,
    _admits_decimal,
    _any_from_text,
    _quoted_nan_text,
    _reads_as_other,
    _scalar_or_read,
)

_TEXTOS = {"texto": ["a", "b", "c"]}
_DECIMAL_COM_NAN = Annotated[Decimal, Field(allow_inf_nan=True)]


def _entrada(tipo: str, linhas: int = 3):
    textos = {"texto": [f"t{i}" for i in range(linhas)]}
    if tipo == "polars":
        return pytest.importorskip("polars").DataFrame(textos)
    return pd.DataFrame(textos)


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
    linhas = len(concluidas) + 1
    _rodar(
        _entrada(entrada, linhas), anotacao, [*concluidas, ProviderAbortError("limite")], **opcoes
    )
    saida, chamadas = _rodar(_entrada(entrada, linhas), anotacao, [nova], resume=True, **opcoes)
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


def test_retomada_de_coluna_de_any_sem_none_conta_o_nan_como_ausente():
    # Como a do arquivo relido sem o modelo, que traz NaN em toda célula vazia.
    modelo = create_model("ModeloDoCaso", campo=(Optional[Any], ...))
    relido = pd.DataFrame(
        {
            **_TEXTOS,
            "campo": pd.Series([1.5, float("nan"), "x"], dtype=object),
            "_dataframeit_status": "processed",
        }
    )

    retomada, chamadas = _rodar(relido, modelo, [], resume=True)

    assert chamadas == 0
    assert retomada["campo"].tolist() == [1.5, None, "x"]


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


_TEXTOS_AMBIGUOS = ["[1, 2]", " [1] ", "{'a': 1}", '"sim"', '"a" e "b"', "{}", '"', "[x"]


@pytest.mark.parametrize("ext", [".parquet", ".csv", ".xlsx"])
@pytest.mark.parametrize("entrada", ["pandas", "polars"])
def test_retomada_do_checkpoint_devolve_o_texto_de_any_que_parece_json(tmp_path, entrada, ext):
    _exige_formato(ext)

    valores = _retomar_do_checkpoint(tmp_path / f"ck{ext}", entrada, Any, _TEXTOS_AMBIGUOS, "fim")

    assert valores == [*_TEXTOS_AMBIGUOS, "fim"]


@pytest.mark.parametrize(
    ("texto", "valor"),
    [
        ('{"a": 1}', {"a": 1}),
        ("[1, 2]", [1, 2]),
        ('"[1, 2]"', "[1, 2]"),
        ('"\\"sim\\""', '"sim"'),
        ("texto", "texto"),
        ('"a" e "b"', '"a" e "b"'),
        ('"', '"'),
        ('"\\u00e9"', "é"),
    ],
)
def test_any_from_text(texto, valor):
    assert _any_from_text(texto) == valor


@pytest.mark.parametrize("texto", ["texto", '"a" e "b"', '"', "[x"])
def test_any_from_text_devolve_o_proprio_texto_que_nao_e_json(texto):
    assert _any_from_text(texto) is texto


def test_checkpoint_grava_entre_aspas_o_texto_de_any_que_se_leria_como_outro_valor():
    df = pd.DataFrame(
        {
            "campo": pd.Series(["[1]", '"sim"', "ação", [1], None, 2], dtype=object),
            "outra": pd.Series(["[1]", '"sim"', "ação", [1], None, 2], dtype=object),
        }
    )

    gravado = _structures_as_json(df, any_columns=["campo"])

    assert gravado["campo"].tolist() == ['"[1]"', '"\\"sim\\""', "ação", "[1]", None, 2]
    assert gravado["outra"].tolist() == ["[1]", '"sim"', "ação", "[1]", None, 2]


def test_parquet_grava_em_json_a_coluna_de_any_fora_das_estruturas():
    df = pd.DataFrame({"campo": pd.Series([{"a": 1}], dtype=object)})

    assert _structures_as_json(df, [], ["campo"])["campo"].tolist() == ['{"a": 1}']
    assert _structures_as_json(df, [])["campo"].tolist() == [{"a": 1}]


def test_read_df_com_modelo_devolve_a_estrutura_de_any(tmp_path):
    modelo = create_model("ModeloDoCaso", campo=(Any, ...), outro=(Any, ...), numero=(Any, ...))
    caminho = tmp_path / "saida.csv"
    pd.DataFrame(
        {
            "campo": ['{"a": 1}', None, '"[1, 2]"'],
            "outro": ["x", None, "y z"],
            "numero": [1.5, None, 2.5],
        }
    ).to_csv(caminho, index=False)

    lido = read_df(caminho, modelo)

    assert lido["campo"].dtype == object
    assert lido["campo"].tolist() == [{"a": 1}, None, "[1, 2]"]
    # Na coluna de texto, a célula ausente volta None, e não o NaN do leitor.
    assert lido["outro"].dtype == object
    assert lido["outro"].tolist() == ["x", None, "y z"]
    # A coluna de números fica como o leitor a deu.
    pd.testing.assert_series_equal(lido["numero"], pd.read_csv(caminho)["numero"])


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


@pytest.mark.parametrize("ext", [".csv", ".xlsx", ".parquet"])
def test_retomada_do_checkpoint_devolve_o_texto_nan_do_campo_decimal_ou_texto(tmp_path, ext):
    _exige_formato(ext)

    valores = _retomar_do_checkpoint(
        tmp_path / f"ck{ext}", "pandas", Union[Decimal, str], ["abc", "NaN"], "z"
    )

    assert valores == ["abc", "NaN", "z"]


def test_checkpoint_grava_como_texto_os_decimais_da_coluna_com_nan():
    df = pd.DataFrame(
        {
            "com_nan": pd.Series([Decimal("1.50"), Decimal("NaN"), 2], dtype=object),
            "sem_nan": pd.Series([Decimal("1.50"), Decimal(2), None], dtype=object),
            "sinalizador": pd.Series([Decimal("sNaN"), Decimal("1.50"), None], dtype=object),
        }
    )

    gravado = _structures_as_json(df, columns=())

    assert gravado["com_nan"].tolist() == ["1.50", "NaN", 2]
    assert gravado["sem_nan"].tolist() == [Decimal("1.50"), Decimal(2), None]
    assert gravado["sinalizador"].tolist()[:2] == ["sNaN", "1.50"]
    assert gravado["sinalizador"].isna().tolist() == [False, False, True]


def test_arquivo_de_texto_grava_como_texto_o_decimal_do_campo_so_de_decimal():
    coluna = pd.Series([Decimal("1.50"), Decimal(2), None], dtype=object)
    df = pd.DataFrame({"campo": coluna, "outra": coluna})

    # O Decimal de outra coluna, como a de um campo `Any`, fica para o gravador do formato.
    de_texto = _structures_as_json(
        df, any_columns=["outra"], decimal_columns=["campo"], text_file=True
    )
    parquet = _structures_as_json(df, decimal_columns=["campo"])

    assert de_texto["campo"].tolist()[:2] == ["1.50", "2"]
    assert de_texto["campo"].isna().tolist() == [False, False, True]
    assert de_texto["outra"].tolist() == [Decimal("1.50"), Decimal(2), None]
    assert parquet["campo"].tolist() == [Decimal("1.50"), Decimal(2), None]


def test_csv_grava_o_decimal_nan_como_texto_e_a_linha_pendente_vazia(tmp_path):
    caminho = tmp_path / "ck.csv"
    df = pd.DataFrame({"campo": pd.Series([Decimal("1.5"), Decimal("NaN"), None], dtype=object)})

    _save_checkpoint(df, caminho)

    assert caminho.read_text(encoding="utf-8") == 'campo\n1.5\nNaN\n""\n'


class _ModeloDecimal(BaseModel):
    campo: Decimal


class _ModeloDecimalOuInt(BaseModel):
    """Na união, só o texto do NaN volta do texto cru; o campo só de Decimal volta inteiro."""

    campo: Union[Decimal, int]


def _csv(tmp_path: Path, celulas: list[str]) -> Path:
    caminho = tmp_path / "saida.csv"
    caminho.write_text("\n".join(["outra,campo", *celulas]) + "\n", encoding="utf-8")
    return caminho


def test_read_df_devolve_o_texto_do_decimal_nan_e_none_na_celula_vazia(tmp_path):
    lido = read_df(_csv(tmp_path, ["a,1.5", "b,NaN", "c,", "d,-NaN"]), _ModeloDecimalOuInt)

    assert lido["campo"].dtype == object
    assert lido["campo"].tolist() == [1.5, "NaN", None, "-NaN"]
    # Na união, o número fica como o leitor o deu, e não vira Decimal.
    assert type(lido["campo"][0]) is float


def test_read_df_devolve_none_ao_lado_do_decimal_nan_na_coluna_sem_numero(tmp_path):
    lido = read_df(_csv(tmp_path, ["a,NaN", "b,", "c,sNaN"]), _ModeloDecimal)

    # Só com textos, a coluna sem o dtype seria a `str` do pandas 3, com NaN na célula vazia.
    assert lido["campo"].dtype == object
    assert lido["campo"].tolist() == ["NaN", None, "sNaN"]


@pytest.mark.parametrize("celula", ["nan", "NA", "NaN ", "xNaN", "NaNx", "ssNaN", "sNaNx"])
def test_read_df_so_rele_como_decimal_nan_o_texto_que_o_checkpoint_grava(tmp_path, celula):
    caminho = _csv(tmp_path, ["a,1.5", f"b,{celula}"])

    lido = read_df(caminho, _ModeloDecimalOuInt)

    pd.testing.assert_frame_equal(lido, pd.read_csv(caminho))


@pytest.mark.parametrize("celula", ["sNaN", "-sNaN7"])
def test_read_df_devolve_o_texto_do_decimal_nan_sinalizador(tmp_path, celula):
    lido = read_df(_csv(tmp_path, ["a,1.5", f"b,{celula}"]), _ModeloDecimalOuInt)

    # A coluna com o texto do NaN sinalizador, que o leitor não toma por ausente, já vem em texto.
    assert lido["campo"].tolist() == ["1.5", celula]


def test_read_df_sem_decimal_nan_deixa_a_coluna_da_uniao_como_lida(tmp_path):
    lido = read_df(_csv(tmp_path, ["a,1.5", "b,"]), _ModeloDecimalOuInt)

    assert lido["campo"].dtype == "float64"


def test_read_df_com_argumento_de_leitura_nao_rele_o_decimal_nan(tmp_path):
    lido = read_df(_csv(tmp_path, ["a,1.5", "b,NaN"]), _ModeloDecimal, sep=",")

    assert lido["campo"].dtype == "float64"


def test_read_df_de_json_nao_rele_o_decimal_nan(tmp_path):
    caminho = tmp_path / "saida.json"
    pd.DataFrame({"campo": ["1.5", "NaN"]}).to_json(caminho)

    pd.testing.assert_frame_equal(read_df(caminho, _ModeloDecimal), pd.read_json(caminho))


def test_read_df_so_rele_o_decimal_nan_no_campo_que_admite_decimal(tmp_path):
    modelo = create_model("ModeloDoCaso", campo=(bool, ...))
    caminho = _csv(tmp_path, ["a,1.5", "b,NaN"])

    pd.testing.assert_frame_equal(read_df(caminho, modelo), pd.read_csv(caminho))


# =============================================================================
# Decimal relido de CSV e XLSX
# =============================================================================

_DECIMAIS = [Decimal("1.10"), Decimal("12345678901234567890.123456789"), Decimal("1E+2")]


@pytest.mark.parametrize("ext", [".csv", ".xlsx"])
def test_retomada_do_checkpoint_devolve_o_decimal_com_todos_os_digitos(tmp_path, ext):
    _exige_formato(ext)

    valores = _retomar_do_checkpoint(
        tmp_path / f"ck{ext}", "pandas", Decimal, _DECIMAIS, Decimal("3.00")
    )

    _confere_decimais(valores, [*_DECIMAIS, Decimal("3.00")])


@pytest.mark.parametrize("ext", [".csv", ".xlsx"])
def test_retomada_do_checkpoint_com_entrada_polars_devolve_o_valor_do_decimal(tmp_path, ext):
    _exige_formato(ext)
    respostas = [Decimal("0.1"), Decimal("12345678901234567890.123456789")]

    valores = _retomar_do_checkpoint(
        tmp_path / f"ck{ext}", "polars", Decimal, respostas, Decimal(3)
    )

    # A saída polars põe a coluna na escala comum; o valor é o mesmo.
    assert valores == [*respostas, Decimal(3)]


@pytest.mark.parametrize("ext", [".csv", ".xlsx"])
def test_retomada_do_checkpoint_devolve_o_decimal_com_modelo_estrito(tmp_path, ext):
    _exige_formato(ext)

    class Estrito(BaseModel):
        model_config = ConfigDict(strict=True)

        campo: Optional[Decimal]

    valores = _retomar_do_checkpoint(
        tmp_path / f"ck{ext}", "pandas", Estrito, [Decimal("1.10"), None], Decimal("2.0")
    )

    _confere_decimais(valores, [Decimal("1.10"), None, Decimal("2.0")])


def test_xlsx_grava_como_texto_o_decimal_do_campo_so_de_decimal(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    caminho = tmp_path / "ck.xlsx"
    df = pd.DataFrame({"campo": pd.Series([Decimal("1.10"), None], dtype=object)})

    _save_checkpoint(df, caminho, decimal_fields={"campo"})

    # Pelo arquivo aberto aqui: o openpyxl antigo não fecha o que ele mesmo abre.
    with caminho.open("rb") as arquivo:
        celula = openpyxl.load_workbook(arquivo).active["A2"]
        gravado = (celula.data_type, celula.value)
    assert gravado == ("s", "1.10")


def test_parquet_grava_o_decimal_sem_nan_como_decimal(tmp_path):
    pytest.importorskip("pyarrow")
    caminho = tmp_path / "ck.parquet"
    df = pd.DataFrame({"campo": pd.Series([Decimal("1.10"), None], dtype=object)})

    _save_checkpoint(df, caminho)

    assert pd.read_parquet(caminho)["campo"].tolist() == [Decimal("1.10"), None]


def test_read_df_devolve_decimal_na_coluna_do_campo_so_de_decimal(tmp_path):
    celulas = ["a,1.10", "b,", "c,NaN", "d,1E+2", "e,abc", "f,sNaN"]

    lido = read_df(_csv(tmp_path, celulas), _ModeloDecimal)

    assert lido["campo"].dtype == object
    valores = lido["campo"].tolist()
    # O texto do NaN e o que não é número ficam em texto, para a validação.
    assert [str(valor) for valor in valores] == ["1.10", "None", "NaN", "1E+2", "abc", "sNaN"]
    assert [type(valor) for valor in valores] == [Decimal, type(None), str, Decimal, str, str]


def test_read_df_deixa_como_lida_a_coluna_vazia_do_campo_so_de_decimal(tmp_path):
    caminho = _csv(tmp_path, ["a,", "b,"])

    pd.testing.assert_frame_equal(read_df(caminho, _ModeloDecimal), pd.read_csv(caminho))


def test_read_df_devolve_decimal_do_numero_que_o_xlsx_guarda(tmp_path):
    pytest.importorskip("openpyxl")
    caminho = tmp_path / "saida.xlsx"
    pd.DataFrame({"campo": [1.1, None, 2.0]}).to_excel(caminho, index=False)

    lido = read_df(caminho, _ModeloDecimal)

    assert lido["campo"].tolist() == [Decimal("1.1"), None, Decimal(2)]


@pytest.mark.parametrize(
    ("validado", "relido", "esperado"),
    [
        (Decimal("1.5"), "1.5", True),
        (Decimal("1.50"), "1.5", False),
        (Decimal("1.50"), Decimal("1.5"), True),
        (Decimal("1.5"), Decimal("1.6"), False),
        (Decimal("1.5"), "NaN", False),
        (Decimal("1.5"), Decimal("sNaN"), False),
        (Decimal("NaN"), "NaN", True),
        (Decimal("NaN"), "-NaN", False),
        (Decimal("sNaN"), "sNaN", True),
        (Decimal("sNaN"), "1.5", False),
        (Decimal("sNaN"), Decimal("sNaN"), True),
        (Decimal("sNaN"), 1.5, False),
        (Decimal("1.5"), 1.5, True),
        (Decimal("1.5"), 1, False),
    ],
)
def test_same_value_do_decimal(validado, relido, esperado):
    assert _same_value(validado, relido) is esperado


# =============================================================================
# Decimal NaN sinalizador
# =============================================================================


def test_retomada_em_memoria_mantem_o_decimal_nan_sinalizador():
    interrompida, _ = _rodar(
        pd.DataFrame(_TEXTOS),
        _DECIMAL_COM_NAN,
        [Decimal("1.5"), Decimal("sNaN"), ProviderAbortError("limite")],
    )

    retomada, chamadas = _rodar(interrompida, _DECIMAL_COM_NAN, [Decimal(2)], resume=True)

    assert chamadas == 1
    _confere_decimais(retomada["campo"].tolist(), [Decimal("1.5"), Decimal("sNaN"), Decimal(2)])


@pytest.mark.parametrize("ext", [".csv", ".xlsx", ".parquet"])
@pytest.mark.parametrize("entrada", ["pandas", "polars"])
def test_retomada_do_checkpoint_devolve_o_decimal_nan_sinalizador(tmp_path, entrada, ext):
    _exige_formato(ext)

    valores = _retomar_do_checkpoint(
        tmp_path / f"ck{ext}",
        entrada,
        _DECIMAL_COM_NAN,
        [Decimal("1.5"), Decimal("sNaN")],
        Decimal("2.25"),
    )

    _confere_decimais(valores, [Decimal("1.5"), Decimal("sNaN"), Decimal("2.25")])


def test_is_absent_nao_levanta_com_o_decimal_nan_sinalizador():
    assert core._is_absent(Decimal("sNaN"), nan_is_value=False) is False


# =============================================================================
# NaN de campo float relido de checkpoint
# =============================================================================


def _confere_floats(valores, esperados):
    assert len(valores) == len(esperados)
    for valor, esperado in zip(valores, esperados, strict=True):
        if esperado is None:
            assert valor is None
        elif math.isnan(esperado):
            assert isinstance(valor, float)
            assert math.isnan(valor)
        else:
            assert valor == esperado


_NAN = float("nan")


@pytest.mark.parametrize("anotacao", [float, Optional[float]])
@pytest.mark.parametrize("ext", [".csv", ".xlsx", ".parquet"])
@pytest.mark.parametrize("entrada", ["pandas", "polars"])
def test_retomada_do_checkpoint_devolve_o_nan_de_float(tmp_path, entrada, ext, anotacao):
    _exige_formato(ext)

    valores = _retomar_do_checkpoint(tmp_path / f"ck{ext}", entrada, anotacao, [1.5, _NAN], 2.0)

    _confere_floats(valores, [1.5, _NAN, 2.0])


@pytest.mark.parametrize("ext", [".csv", ".xlsx", ".parquet"])
@pytest.mark.parametrize("entrada", ["pandas", "polars"])
def test_retomada_do_checkpoint_devolve_none_ao_lado_do_nan_de_float(tmp_path, entrada, ext):
    _exige_formato(ext)

    valores = _retomar_do_checkpoint(
        tmp_path / f"ck{ext}", entrada, Optional[float], [None, _NAN, 1.5], 2.0
    )

    _confere_floats(valores, [None, _NAN, 1.5, 2.0])


@pytest.mark.parametrize("ext", [".csv", ".xlsx", ".parquet"])
def test_retomada_do_checkpoint_devolve_o_nan_de_float_com_modelo_estrito(tmp_path, ext):
    _exige_formato(ext)

    class Estrito(BaseModel):
        model_config = ConfigDict(strict=True)

        campo: float

    valores = _retomar_do_checkpoint(tmp_path / f"ck{ext}", "pandas", Estrito, [1.5, _NAN], 2.0)

    _confere_floats(valores, [1.5, _NAN, 2.0])


@pytest.mark.parametrize("ext", [".csv", ".xlsx", ".parquet"])
def test_retomada_do_checkpoint_sem_nan_devolve_none_no_float_opcional(tmp_path, ext):
    _exige_formato(ext)

    valores = _retomar_do_checkpoint(
        tmp_path / f"ck{ext}", "pandas", Optional[float], [None, 1.5], 2.0
    )

    _confere_floats(valores, [None, 1.5, 2.0])


@pytest.mark.parametrize("ext", [".csv", ".xlsx"])
@pytest.mark.parametrize(
    "respostas",
    [["abc", "NaN", _NAN, '"NaN"', '""NaN""'], ["NaN", "x"], ['"NaN"', "x"], [_NAN, "x"]],
    ids=["tudo", "so-o-texto", "so-o-texto-com-aspas", "so-o-nan"],
)
def test_retomada_do_checkpoint_separa_o_nan_do_texto_nan_no_campo_float_ou_texto(
    tmp_path, ext, respostas
):
    _exige_formato(ext)

    valores = _retomar_do_checkpoint(
        tmp_path / f"ck{ext}", "pandas", Union[float, str], respostas, "z"
    )

    esperados = [*respostas, "z"]
    assert [type(valor) for valor in valores] == [type(valor) for valor in esperados]
    assert [str(valor) for valor in valores] == [str(valor) for valor in esperados]


@pytest.mark.parametrize(
    ("texto", "gravado"),
    [
        ("NaN", '"NaN"'),
        ('"NaN"', '""NaN""'),
        ("nan", "nan"),
        ('"NaN', '"NaN'),
        ("NaN ", "NaN "),
        ('"x"', '"x"'),
    ],
)
def test_quoted_nan_text(texto, gravado):
    assert _quoted_nan_text(texto) == gravado


def test_csv_grava_o_nan_de_float_como_texto_e_a_linha_pendente_vazia(tmp_path):
    caminho = tmp_path / "ck.csv"
    coluna = pd.Series([1.5, _NAN, None], dtype=object)
    df = pd.DataFrame({"campo": coluna, "outra": coluna})

    _save_checkpoint(df, caminho, nan_fields={"campo"})

    assert caminho.read_text(encoding="utf-8") == "campo,outra\n1.5,1.5\nNaN,\n,\n"


def test_parquet_grava_o_nan_de_float_a_parte_do_nulo(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    caminho = tmp_path / "ck.parquet"
    coluna = pd.Series([1.5, _NAN, None], dtype=object)
    df = pd.DataFrame(
        {
            "campo": coluna,
            "outra": coluna,
            "sem_nan": pd.Series([1.5, None, None], dtype=object),
            "com_inteiro": pd.Series([1, _NAN, None], dtype=object),
            "texto": ["a", "b", "c"],
        }
    )

    _save_checkpoint(df, caminho, nan_fields={"campo", "sem_nan", "com_inteiro", "ausente"})

    tabela = pq.read_table(caminho)
    nulos = {nome: tabela.column(nome).null_count for nome in tabela.column_names}
    assert nulos == {"campo": 1, "outra": 2, "sem_nan": 2, "com_inteiro": 2, "texto": 0}
    # Quem lê o arquivo sem o modelo recebe a coluna de float de sempre.
    lido = pd.read_parquet(caminho)
    assert lido["campo"].dtype == "float64"
    assert lido["texto"].tolist() == ["a", "b", "c"]
    assert len(lido.columns) == 5


def test_parquet_sem_nan_de_float_e_gravado_pelo_pandas(tmp_path):
    pytest.importorskip("pyarrow")
    caminho = tmp_path / "ck.parquet"
    df = pd.DataFrame({"campo": pd.Series([1.5, None], dtype=object)})

    with patch.object(pd.DataFrame, "to_parquet", autospec=True) as gravacao:
        gravacao.side_effect = lambda frame, destino, **_: Path(destino).touch()
        _save_checkpoint(df, caminho, nan_fields={"campo"})

    assert gravacao.call_args.kwargs == {"index": False}


def test_parquet_nao_refaz_a_coluna_de_float_que_nao_e_de_objetos(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    caminho = tmp_path / "ck.parquet"

    _save_checkpoint(pd.DataFrame({"campo": [1.5, _NAN]}), caminho, nan_fields={"campo"})

    # O to_parquet do pandas, que grava como nulo o NaN da coluna float64.
    assert pq.read_table(caminho).column("campo").null_count == 1


class _ModeloFloat(BaseModel):
    campo: Optional[float]


def test_read_df_devolve_o_nan_de_float_e_none_na_celula_vazia(tmp_path):
    lido = read_df(_csv(tmp_path, ["a,1.5", "b,NaN", "c,"]), _ModeloFloat)

    assert lido["campo"].dtype == object
    _confere_floats(lido["campo"].tolist(), [1.5, _NAN, None])


@pytest.mark.parametrize("celula", ["nan", "NA", "-NaN", "NaN "])
def test_read_df_so_rele_como_nan_de_float_o_texto_que_o_checkpoint_grava(tmp_path, celula):
    caminho = _csv(tmp_path, ["a,1.5", f"b,{celula}"])

    pd.testing.assert_frame_equal(read_df(caminho, _ModeloFloat), pd.read_csv(caminho))


def test_read_df_com_argumento_de_leitura_nao_rele_o_nan_de_float(tmp_path):
    lido = read_df(_csv(tmp_path, ["a,1.5", "b,NaN"]), _ModeloFloat, sep=",")

    assert lido["campo"].dtype == "float64"


def test_read_df_separa_o_nan_do_texto_nan_no_campo_float_ou_texto(tmp_path):
    modelo = create_model("ModeloDoCaso", campo=(Union[float, str], ...))

    lido = read_df(_csv(tmp_path, ["a,1.5", "b,NaN", "c,", 'd,"""NaN"""']), modelo)

    primeiro, nan, vazio, texto = lido["campo"].tolist()
    assert (primeiro, vazio, texto) == ("1.5", None, "NaN")
    assert isinstance(nan, float)
    assert math.isnan(nan)


def test_read_df_deixa_como_lida_a_coluna_de_float_ou_texto_sem_nan(tmp_path):
    modelo = create_model("ModeloDoCaso", campo=(Union[float, str], ...))
    caminho = _csv(tmp_path, ["a,1.5", "b,x"])

    pd.testing.assert_frame_equal(read_df(caminho, modelo), pd.read_csv(caminho))


def test_read_df_rele_do_parquet_o_nan_de_float_e_o_int_exato(tmp_path):
    pytest.importorskip("pyarrow")
    modelo = create_model(
        "ModeloDoCaso",
        campo=(Optional[float], ...),
        sem_nan=(Optional[float], ...),
        inteiro=(Optional[int], ...),
        texto=(Union[float, str], ...),
    )
    caminho = tmp_path / "ck.parquet"
    df = pd.DataFrame(
        {
            "campo": pd.Series([1.5, _NAN, None], dtype=object),
            "sem_nan": pd.Series([1.5, None, None], dtype=object),
            "inteiro": pd.Series([2**60 + 1, None, 3], dtype=object),
            "texto": ["a", "b", "c"],
        }
    )
    _save_checkpoint(df, caminho, nan_fields={"campo", "sem_nan"})

    lido = read_df(caminho, modelo)

    assert lido["campo"].dtype == object
    _confere_floats(lido["campo"].tolist(), [1.5, _NAN, None])
    assert lido["sem_nan"].dtype == "float64"
    assert lido["inteiro"].tolist() == [2**60 + 1, None, 3]
    assert lido["texto"].tolist() == ["a", "b", "c"]


def test_read_df_deixa_como_lida_a_coluna_de_int_do_parquet_so_com_nulos(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    modelo = create_model("ModeloDoCaso", inteiro=(Optional[int], ...))
    caminho = tmp_path / "ck.parquet"
    # Sem os metadados do pandas, a coluna de int só com nulos volta float64.
    pq.write_table(pa.table({"inteiro": pa.array([None, None], pa.int64())}), caminho)

    lido = read_df(caminho, modelo)

    assert lido["inteiro"].dtype == "float64"
    assert lido["inteiro"].isna().all()


def test_read_df_do_parquet_sem_campo_a_reler_nao_abre_o_arquivo_de_novo(tmp_path):
    pytest.importorskip("pyarrow")
    modelo = create_model("ModeloDoCaso", texto=(str, ...))
    caminho = tmp_path / "ck.parquet"
    pd.DataFrame({"texto": ["a"]}).to_parquet(caminho)

    # A primeira leitura é a do leitor guardado em `_READERS`, fora do patch.
    with patch("dataframeit.utils.pd.read_parquet", wraps=pd.read_parquet) as releitura:
        read_df(caminho, modelo)

    assert releitura.call_count == 0


def test_read_df_rele_o_csv_uma_vez_so_para_todos_os_campos(tmp_path):
    modelo = create_model(
        "ModeloDoCaso",
        texto=(str, ...),
        inteiro=(Optional[int], ...),
        decimal=(Decimal, ...),
        real=(float, ...),
        livre=(Any, ...),
        sem_releitura=(bool, ...),
    )
    caminho = tmp_path / "saida.csv"
    caminho.write_text(
        "texto,inteiro,decimal,real,livre,sem_releitura\n"
        "2023,9007199254740993,1.10,NaN,5,True\nNA,,2,1.5,x,False\n",
        encoding="utf-8",
    )

    releituras = []
    with patch.dict(
        "dataframeit.utils._TEXT_READERS", {".csv": _conta_chamadas(pd.read_csv, releituras)}
    ):
        lido = read_df(caminho, modelo)

    assert releituras == [["texto", "inteiro", "decimal", "real", "livre"]]
    assert lido["livre"].tolist() == [5, "x"]
    assert lido["texto"].tolist() == ["2023", "NA"]
    assert lido["inteiro"].tolist() == [9007199254740993, None]
    assert [str(valor) for valor in lido["decimal"]] == ["1.10", "2"]
    _confere_floats(lido["real"].tolist(), [_NAN, 1.5])
    assert lido["sem_releitura"].tolist() == [True, False]


def _conta_chamadas(leitor, chamadas):
    """O leitor que guarda em `chamadas` o usecols de cada leitura."""

    def contado(caminho, **opcoes):
        chamadas.append(opcoes["usecols"])
        return leitor(caminho, **opcoes)

    return contado


def test_read_df_sem_campo_a_reler_nao_abre_o_csv_de_novo(tmp_path):
    modelo = create_model("ModeloDoCaso", sem_releitura=(bool, ...), ausente=(str, ...))
    caminho = tmp_path / "saida.csv"
    caminho.write_text("sem_releitura\nTrue\n", encoding="utf-8")

    releituras = []
    with patch.dict(
        "dataframeit.utils._TEXT_READERS", {".csv": _conta_chamadas(pd.read_csv, releituras)}
    ):
        lido = read_df(caminho, modelo)

    assert releituras == []
    assert lido["sem_releitura"].tolist() == [True]


# =============================================================================
# Texto de `Any` que o CSV e o XLSX leriam como número, bool ou ausência
# =============================================================================

_TEXTOS_QUE_O_LEITOR_TROCA = [
    "123",
    "NaN",
    "",
    "1.5",
    "true",
    "N/A",
    "null",
    "None",
    " 7",
    "1e3",
    "inf",
    "abc",
]


@pytest.mark.parametrize("ext", [".csv", ".xlsx", ".parquet"])
@pytest.mark.parametrize("entrada", ["pandas", "polars"])
def test_retomada_do_checkpoint_devolve_o_texto_de_any_que_o_leitor_trocaria(
    tmp_path, entrada, ext
):
    _exige_formato(ext)

    valores = _retomar_do_checkpoint(
        tmp_path / f"ck{ext}", entrada, Any, _TEXTOS_QUE_O_LEITOR_TROCA, "fim"
    )

    assert valores == [*_TEXTOS_QUE_O_LEITOR_TROCA, "fim"]


@pytest.mark.parametrize("ext", [".csv", ".xlsx", ".parquet"])
@pytest.mark.parametrize(
    "respostas", [["123", "456"], ["NaN", "NA"], ["true", "False"], ["", None]]
)
def test_retomada_do_checkpoint_devolve_o_texto_de_any_numa_coluna_que_o_leitor_converteria(
    tmp_path, ext, respostas
):
    _exige_formato(ext)

    valores = _retomar_do_checkpoint(tmp_path / f"ck{ext}", "pandas", Optional[Any], respostas, "x")

    assert valores == [*respostas, "x"]
    assert [type(valor) for valor in valores] == [type(valor) for valor in [*respostas, "x"]]


@pytest.mark.parametrize(
    "texto",
    ["", " ", "123", " 7", "7 ", "-1.5", "1e3", "inf", "nan", "NaN", "NA", "N/A", "<NA>", "None"],
)
def test_reads_as_other_acusa_o_numero_e_a_ausencia_do_leitor(texto):
    assert _reads_as_other(texto) is True


@pytest.mark.parametrize("texto", ["null", "True", "true", "tRue", "FALSE", " false "])
def test_reads_as_other_acusa_o_texto_que_o_leitor_troca(texto):
    assert _reads_as_other(texto) is True


@pytest.mark.parametrize("texto", ["abc", "1 2", "12a", "sim", "verdadeiro", "N/D", "-", "1,5"])
def test_reads_as_other_deixa_o_texto_que_o_leitor_devolve(texto):
    assert _reads_as_other(texto) is False


def test_checkpoint_de_texto_grava_entre_aspas_o_texto_de_any_que_o_leitor_trocaria():
    celulas = ["123", "NaN", "", "abc", "[1]", 123, None]
    df = pd.DataFrame(
        {"campo": pd.Series(celulas, dtype=object), "outra": pd.Series(celulas, dtype=object)}
    )

    de_texto = _structures_as_json(df, any_columns=["campo"], text_file=True)
    parquet = _structures_as_json(df, any_columns=["campo"])

    assert de_texto["campo"].tolist() == ['"123"', '"NaN"', '""', "abc", '"[1]"', 123, None]
    assert de_texto["outra"].tolist() == celulas
    assert parquet["campo"].tolist() == ["123", "NaN", "", "abc", '"[1]"', 123, None]


# =============================================================================
# Tupla de `Any`
# =============================================================================


@pytest.mark.parametrize("ext", [".csv", ".xlsx", ".parquet"])
def test_retomada_do_checkpoint_devolve_a_tupla_de_any_como_lista(tmp_path, ext):
    """O JSON do checkpoint não tem tupla, e `Any` não diz que a lista relida era uma."""
    _exige_formato(ext)

    valores = _retomar_do_checkpoint(tmp_path / f"ck{ext}", "pandas", Any, [(1, 2), [3]], (4,))

    assert valores == [[1, 2], [3], (4,)]
    assert [type(valor) for valor in valores] == [list, list, tuple]


def test_retomada_em_memoria_mantem_a_tupla_de_any():
    interrompida, _ = _rodar(
        pd.DataFrame(_TEXTOS), Any, [(1, 2), [3], ProviderAbortError("limite")]
    )

    retomada, _ = _rodar(interrompida, Any, ["x"], resume=True)

    assert retomada["campo"].tolist() == [(1, 2), [3], "x"]
    assert type(retomada["campo"].tolist()[0]) is tuple


# =============================================================================
# NaN de `Any` na coluna sem None e no reprocessamento campo a campo
# =============================================================================

_AVISO_DO_NAN_DE_ANY = r"As colunas \['campo'\], de campo `Any`, têm NaN em linhas já processadas"


def _retomar(df, modelo, **opcoes):
    """Retoma sem chamada ao LLM e sem esconder os avisos."""
    with (
        patch("dataframeit.core.call_langchain", side_effect=AssertionError("sem chamada")),
        patch("dataframeit.core.validate_provider_dependencies"),
    ):
        return dataframeit(
            df, questions=modelo, prompt="Analise {texto}", track_tokens=False, **opcoes
        )


def test_retomada_da_saida_completa_avisa_que_o_nan_de_any_volta_none():
    modelo = create_model("ModeloDoCaso", campo=(Any, ...))
    completa, _ = _rodar(pd.DataFrame(_TEXTOS), modelo, [1.5, _NAN, "x"])
    assert "_dataframeit_status" not in completa.columns
    completa["_dataframeit_status"] = "processed"

    with pytest.warns(UserWarning, match=_AVISO_DO_NAN_DE_ANY + r".*volta None"):
        retomada = _retomar(completa, modelo)

    assert retomada["campo"].tolist() == [1.5, None, "x"]


def test_retomada_nao_avisa_do_nan_de_any_na_coluna_com_none():
    modelo = create_model("ModeloDoCaso", campo=(Any, ...))
    interrompida, _ = _rodar(
        pd.DataFrame(_TEXTOS), modelo, [1.5, _NAN, ProviderAbortError("limite")]
    )
    interrompida["_dataframeit_status"] = "processed"
    interrompida["campo"] = pd.Series([1.5, _NAN, None], dtype=object)
    opcional = create_model("ModeloDoCaso", campo=(Optional[Any], ...))

    retomada = _retomar(interrompida, opcional)

    _confere_floats(retomada["campo"].tolist(), [1.5, _NAN, None])


def test_retomada_nao_avisa_do_nan_de_any_fora_das_linhas_processadas():
    modelo = create_model("ModeloDoCaso", campo=(Any, ...))
    df = pd.DataFrame(
        {
            **_TEXTOS,
            "campo": pd.Series(["x", "y", _NAN], dtype=object),
            "_dataframeit_status": ["processed", "processed", "error"],
        }
    )

    retomada = _retomar(df, modelo, resume=True)

    assert retomada["campo"].tolist()[:2] == ["x", "y"]


def test_retomada_nao_avisa_do_ausente_que_nao_e_nan_na_coluna_de_any():
    modelo = create_model("ModeloDoCaso", campo=(Optional[Any], ...))
    df = pd.DataFrame(
        {
            **_TEXTOS,
            "campo": pd.Series(["x", pd.NA, pd.NaT], dtype=object),
            "_dataframeit_status": "processed",
        }
    )

    retomada = _retomar(df, modelo)

    assert retomada["campo"].tolist() == ["x", None, None]


def test_retomada_nao_avisa_do_nan_da_coluna_de_any_que_nao_e_de_objetos():
    modelo = create_model("ModeloDoCaso", campo=(Optional[Any], ...))
    df = pd.DataFrame({**_TEXTOS, "campo": [1.5, _NAN, 2.5], "_dataframeit_status": "processed"})

    retomada = _retomar(df, modelo)

    assert retomada["campo"].tolist() == [1.5, None, 2.5]


def test_retomada_avisa_do_nan_de_any_e_mantem_pulado_o_campo_com_condition():
    """O NaN da coluna relida de arquivo segue ausente, e a condição falsa o aceita."""

    class ComCondicao(BaseModel):
        tipo: str
        campo: Any = Field(json_schema_extra={"condition": {"field": "tipo", "equals": "a"}})

    relido = pd.DataFrame(
        {
            "texto": ["a", "b"],
            "tipo": ["a", "b"],
            "campo": pd.Series(["x", _NAN], dtype=object),
            "_dataframeit_status": "processed",
        }
    )

    with pytest.warns(UserWarning, match=_AVISO_DO_NAN_DE_ANY):
        retomada = _retomar(relido, ComCondicao)

    assert retomada["campo"].tolist() == ["x", None]


class _DoisCampos(BaseModel):
    campo: Any
    outro: str


@contextmanager
def _backend_campo_a_campo(conhecidos):
    """Backend que guarda em `conhecidos` o que cada chamada parcial recebe dos outros campos."""

    def invoke_partial(texto, campos, valores):
        conhecidos.append(valores)
        return {"data": {"outro": "novo"}, "usage": None}

    with patch("dataframeit.core.validate_provider_dependencies"):
        yield lambda *args, **kwargs: _contexto(
            core.ProviderBackend(
                label="falso", invoke=lambda texto: {"data": {}}, invoke_partial=invoke_partial
            )
        )


@contextmanager
def _contexto(valor):
    yield valor


@pytest.mark.parametrize("paralelo", [1, 2])
def test_reprocessamento_campo_a_campo_passa_o_nan_de_any_como_valor(paralelo):
    df = pd.DataFrame(
        {
            "texto": ["a", "b", "c"],
            "campo": pd.Series([_NAN, None, "x"], dtype=object),
            "outro": ["antigo"] * 3,
            "_dataframeit_status": ["processed", "error", "processed"],
        }
    )
    conhecidos = []

    with (
        _backend_campo_a_campo(conhecidos) as fabrica,
        patch.object(core, "_provider_backend", fabrica),
    ):
        saida = dataframeit(
            df,
            questions=_DoisCampos,
            prompt="Analise {texto}",
            track_tokens=False,
            reprocess_columns=["outro"],
            parallel_requests=paralelo,
        )

    da_primeira = next(valores for valores in conhecidos if valores["campo"] != "x")
    assert isinstance(da_primeira["campo"], float)
    assert math.isnan(da_primeira["campo"])
    assert saida["outro"].tolist()[0] == "novo"


def test_reprocessamento_campo_a_campo_passa_none_no_nan_de_any_da_coluna_sem_none():
    df = pd.DataFrame(
        {
            "texto": ["a", "b"],
            "campo": pd.Series([_NAN, "x"], dtype=object),
            "outro": ["antigo"] * 2,
            "_dataframeit_status": "processed",
        }
    )
    conhecidos = []
    opcional = create_model("ModeloDoCaso", campo=(Optional[Any], ...), outro=(str, ...))

    with (
        _backend_campo_a_campo(conhecidos) as fabrica,
        patch.object(core, "_provider_backend", fabrica),
        pytest.warns(UserWarning, match=_AVISO_DO_NAN_DE_ANY),
    ):
        dataframeit(
            df,
            questions=opcional,
            prompt="Analise {texto}",
            track_tokens=False,
            reprocess_columns=["outro"],
        )

    assert [valores["campo"] for valores in conhecidos] == [None, "x"]


def test_any_nan_columns_so_conta_a_coluna_de_objetos_de_any_com_none():
    modelo = create_model(
        "ModeloDoCaso",
        com_none=(Any, ...),
        sem_none=(Any, ...),
        de_float=(Any, ...),
        nao_any=(Optional[float], ...),
        ausente=(Any, ...),
    )
    df = pd.DataFrame(
        {
            "com_none": pd.Series([_NAN, None], dtype=object),
            "sem_none": pd.Series([_NAN, "x"], dtype=object),
            "de_float": [_NAN, 1.5],
            "nao_any": pd.Series([_NAN, None], dtype=object),
        }
    )

    assert core._any_nan_columns(df, modelo) == frozenset({"com_none"})


# =============================================================================
# União de Decimal e float, número de `Any` no CSV e ausência de `Any`
# =============================================================================


@pytest.mark.parametrize("anotacao", [Union[Decimal, float], Optional[Union[float, Decimal]]])
@pytest.mark.parametrize("ext", [".csv", ".xlsx"])
def test_retomada_do_checkpoint_devolve_float_nan_na_uniao_de_decimal_e_float(
    tmp_path, ext, anotacao
):
    _exige_formato(ext)

    valores = _retomar_do_checkpoint(tmp_path / f"ck{ext}", "pandas", anotacao, [_NAN, 1.5], 2.5)

    assert isinstance(valores[0], float)
    assert math.isnan(valores[0])
    assert valores[1:] == [1.5, 2.5]


_ESCALARES_DE_ANY = [5, 7.5, True, "12", None, "x", False, -3, 1e-300]


@pytest.mark.parametrize("ext", [".csv", ".xlsx"])
@pytest.mark.parametrize("entrada", ["pandas", "polars"])
def test_retomada_do_checkpoint_devolve_o_numero_de_any_ao_lado_de_um_texto(tmp_path, entrada, ext):
    _exige_formato(ext)

    valores = _retomar_do_checkpoint(
        tmp_path / f"ck{ext}", entrada, Optional[Any], _ESCALARES_DE_ANY, "fim"
    )

    esperados = [*_ESCALARES_DE_ANY, "fim"]
    assert valores == esperados
    assert [type(valor) for valor in valores] == [type(valor) for valor in esperados]


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [
        ("5", 5),
        ("-12", -12),
        ("7.5", 7.5),
        ("1e+300", 1e300),
        ("inf", math.inf),
        ("True", True),
        ("False", False),
    ],
)
def test_scalar_or_read_devolve_o_numero_e_o_bool_que_o_csv_guarda(texto, esperado):
    valor = _scalar_or_read(texto, texto)

    assert valor == esperado
    assert type(valor) is type(esperado)


@pytest.mark.parametrize("texto", ["x", "true", " 7", "7 ", "NaN", "nan", "NA", "12a", '"5"'])
def test_scalar_or_read_deixa_como_lido_o_que_o_checkpoint_nao_grava_como_numero(texto):
    lido = object()

    assert _scalar_or_read(texto, lido) is lido


def test_read_df_deixa_como_lida_a_coluna_de_any_do_csv_sem_celula_a_trocar(tmp_path):
    modelo = create_model("ModeloDoCaso", campo=(Any, ...))
    caminho = _csv(tmp_path, ["a,x", "b,y"])

    pd.testing.assert_frame_equal(read_df(caminho, modelo), pd.read_csv(caminho))


def test_read_df_sem_normalizar_devolve_o_numero_de_any_e_deixa_o_json(tmp_path):
    modelo = create_model("ModeloDoCaso", campo=(Any, ...))
    caminho = tmp_path / "saida.csv"
    pd.DataFrame({"campo": pd.Series([5, '{"a": 1}', None], dtype=object)}).to_csv(
        caminho, index=False
    )

    assert read_df(caminho, modelo, normalize=False)["campo"].tolist() == [5, '{"a": 1}', None]


def test_textos_ausentes_do_leitor_sao_os_do_pandas():
    from pandas._libs.parsers import STR_NA_VALUES  # noqa: PLC0415 (só este teste lê o nome privado)

    assert _READER_NA_TEXTS == STR_NA_VALUES


@pytest.mark.parametrize("ext", [".csv", ".xlsx", ".parquet"])
def test_saida_relida_com_read_df_retoma_sem_o_aviso_do_nan_de_any(tmp_path, ext):
    """A resposta None relida de arquivo volta None, e não o NaN que o aviso acusa."""
    _exige_formato(ext)
    modelo = create_model("ModeloDoCaso", campo=(Optional[Any], ...))
    caminho = tmp_path / f"saida{ext}"
    saida = pd.DataFrame({**_TEXTOS, "campo": pd.Series(["abc", None, "x"], dtype=object)})
    _save_checkpoint(saida, caminho, any_fields={"campo"})

    relida = read_df(caminho, modelo)
    relida["_dataframeit_status"] = "processed"
    retomada = _retomar(relida, modelo)

    assert relida["campo"].tolist() == ["abc", None, "x"]
    assert retomada["campo"].tolist() == ["abc", None, "x"]


@pytest.mark.parametrize("ext", [".csv", ".xlsx"])
def test_retomada_do_checkpoint_devolve_o_texto_do_campo_float_ou_any(tmp_path, ext):
    _exige_formato(ext)
    # O NaN desse campo segue a regra do `Any`: relido de checkpoint, conta como ausente.
    respostas = ["[1, 2]", "NaN", '"NaN"', "x", [1, 2]]

    valores = _retomar_do_checkpoint(
        tmp_path / f"ck{ext}", "pandas", Union[float, Any], respostas, "z"
    )

    esperados = [*respostas, "z"]
    assert [type(valor) for valor in valores] == [type(valor) for valor in esperados]
    assert [str(valor) for valor in valores] == [str(valor) for valor in esperados]


# =============================================================================
# Forma do que a gravação e a releitura devolvem
# =============================================================================


def test_checkpoint_grava_com_acento_o_texto_de_any_entre_aspas():
    df = pd.DataFrame({"campo": pd.Series(['["ação"]', "é"], dtype=object)})

    gravado = _structures_as_json(df, any_columns=["campo"])

    assert gravado["campo"].tolist() == ['"[\\"ação\\"]"', "é"]


def test_parquet_com_nan_de_float_nao_grava_o_indice_da_entrada(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    caminho = tmp_path / "ck.parquet"
    df = pd.DataFrame({"campo": pd.Series([1.5, _NAN], dtype=object)}, index=[10, 20])

    _save_checkpoint(df, caminho, nan_fields={"campo"})

    assert pq.read_table(caminho).column_names == ["campo"]
    assert pd.read_parquet(caminho).index.tolist() == [0, 1]


def test_read_df_rele_o_nan_de_float_do_parquet_com_indice_proprio(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    caminho = tmp_path / "saida.parquet"
    tabela = pa.Table.from_pandas(pd.DataFrame({"campo": [0.0, 0.0, 0.0]}, index=[10, 20, 30]))
    coluna = pa.array([1.5, _NAN, None])
    pq.write_table(tabela.set_column(0, "campo", coluna), caminho)

    lido = read_df(caminho, _ModeloFloat)

    assert lido.index.tolist() == [10, 20, 30]
    _confere_floats(lido["campo"].tolist(), [1.5, _NAN, None])


def test_read_df_rele_como_objeto_o_nan_de_float_do_parquet_sem_nulo(tmp_path):
    pytest.importorskip("pyarrow")
    caminho = tmp_path / "ck.parquet"
    _save_checkpoint(
        pd.DataFrame({"campo": pd.Series([1.5, _NAN], dtype=object)}), caminho, nan_fields={"campo"}
    )

    lido = read_df(caminho, _ModeloFloat)

    assert lido["campo"].dtype == object
    _confere_floats(lido["campo"].tolist(), [1.5, _NAN])


def test_read_df_devolve_o_texto_nan_entre_aspas_da_coluna_de_float_ou_texto_sem_ausente(tmp_path):
    modelo = create_model("ModeloDoCaso", campo=(Union[float, str], ...))

    lido = read_df(_csv(tmp_path, ["a,x", 'b,"""NaN"""']), modelo)

    assert lido["campo"].tolist() == ["x", "NaN"]


def test_read_df_sem_normalizar_devolve_none_na_celula_vazia_do_texto_de_any(tmp_path):
    modelo = create_model("ModeloDoCaso", campo=(Any, ...))

    lido = read_df(_csv(tmp_path, ["a,x", "b,", "c,y"]), modelo, normalize=False)

    assert lido["campo"].dtype == object
    assert lido["campo"].tolist() == ["x", None, "y"]


@pytest.mark.parametrize("ext", [".xlsx", ".parquet"])
def test_read_df_devolve_none_na_celula_vazia_do_texto_de_any(tmp_path, ext):
    _exige_formato(ext)
    modelo = create_model("ModeloDoCaso", campo=(Any, ...))
    caminho = tmp_path / f"saida{ext}"
    saida = pd.DataFrame({"campo": pd.Series(["abc", None, "x"], dtype=object)})
    _save_checkpoint(saida, caminho, any_fields={"campo"})

    lido = read_df(caminho, modelo)

    assert lido["campo"].dtype == object
    assert lido["campo"].tolist() == ["abc", None, "x"]


def test_read_df_devolve_a_estrutura_de_any_depois_de_um_campo_que_nao_e_any(tmp_path):
    modelo = create_model("ModeloDoCaso", outra=(str, ...), campo=(Any, ...))

    lido = read_df(_csv(tmp_path, ['a,"[1, 2]"', "b,x"]), modelo)

    assert lido["campo"].tolist() == [[1, 2], "x"]


def test_read_df_devolve_a_estrutura_de_any_na_linha_do_indice_lido(tmp_path):
    modelo = create_model("ModeloDoCaso", campo=(Any, ...))

    lido = read_df(_csv(tmp_path, ['a,"[1, 2]"', "b,x"]), modelo, index_col="outra")

    assert lido.index.tolist() == ["a", "b"]
    assert lido["campo"].tolist() == [[1, 2], "x"]


def test_read_df_rele_as_colunas_do_csv_com_indice_tirado_da_primeira_coluna(tmp_path):
    """Com um campo a mais que o cabeçalho em cada linha, o leitor faz da primeira coluna o índice."""
    modelo = create_model(
        "ModeloDoCaso",
        inteiro=(Optional[int], ...),
        decimal=(Decimal, ...),
        real=(Optional[float], ...),
        livre=(Any, ...),
    )
    caminho = tmp_path / "saida.csv"
    caminho.write_text(
        "inteiro,decimal,real,livre,outra\n"
        "2,9007199254740993,1.10,NaN,5,a\n0,,2.20,1.5,x,b\n1,7,3.30,,y,c\n",
        encoding="utf-8",
    )

    lido = read_df(caminho, modelo)

    assert lido.index.tolist() == [2, 0, 1]
    assert lido["inteiro"].tolist() == [9007199254740993, None, 7]
    assert [str(valor) for valor in lido["decimal"]] == ["1.10", "2.20", "3.30"]
    _confere_floats(lido["real"].tolist(), [_NAN, 1.5, None])
    assert lido["livre"].tolist() == [5, "x", "y"]
