"""Conversões de entrada e saída, classes de valores e releituras de utils.py."""

import datetime as dt
import re
from decimal import Decimal
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import pytest
from typing_extensions import override

from dataframeit.utils import (
    _AWARE_DATETIME,
    DEFAULT_TEXT_COLUMN,
    ORIGINAL_TYPE_DICT,
    ORIGINAL_TYPE_PANDAS_DF,
    ORIGINAL_TYPE_POLARS_DF,
    ConversionInfo,
    _accepts_only_int,
    _column_rank,
    _group_kinds,
    _mixes_converted_types,
    _python_values,
    _to_polars,
    from_pandas,
    read_df,
    to_pandas,
)


def _exato(msg: str) -> str:
    """O padrão do `match` que casa só com a mensagem inteira."""
    return f"^{re.escape(msg)}$"


# =============================================================================
# to_pandas e from_pandas
# =============================================================================


def test_tipo_nao_suportado_diz_o_tipo_e_os_aceitos():
    msg = (
        "Tipo não suportado: int. "
        "Use pandas.DataFrame, polars.DataFrame, pandas.Series, polars.Series, list ou dict."
    )

    with pytest.raises(TypeError, match=_exato(msg)):
        to_pandas(42)


def test_dict_com_chaves_tupla_volta_com_indice_de_tuplas():
    chaves = [("a", 1), ("b", 2)]
    df, info = to_pandas(dict.fromkeys(chaves, "texto"))
    df["campo"] = ["x", "y"]

    resultado = from_pandas(df, info)

    assert isinstance(resultado, pd.DataFrame)
    assert type(resultado.index) is pd.Index
    assert list(resultado.index) == chaves
    assert list(resultado.columns) == ["campo"]


def test_indice_original_so_e_restaurado_na_entrada_pandas_series():
    df = pd.DataFrame({DEFAULT_TEXT_COLUMN: ["t1", "t2"]}, index=["d1", "d2"])
    info = ConversionInfo(original_type=ORIGINAL_TYPE_DICT, original_index=["x", "y"])

    resultado = from_pandas(df, info)

    assert isinstance(resultado, pd.DataFrame)
    assert list(resultado.index) == ["d1", "d2"]


def test_coluna_texto_do_usuario_fica_na_entrada_pandas_dataframe():
    df = pd.DataFrame({DEFAULT_TEXT_COLUMN: ["t1"], "campo": ["x"]})

    resultado = from_pandas(df, ConversionInfo(original_type=ORIGINAL_TYPE_PANDAS_DF))

    assert list(resultado.columns) == [DEFAULT_TEXT_COLUMN, "campo"]


def test_coluna_que_o_polars_nao_tipa_sai_object_com_aviso_completo():
    pl = pytest.importorskip("polars")
    valores = [dt.date(2024, 1, 2), 3]
    df = pd.DataFrame({"campo": pd.Series(valores, dtype=object)})
    msg = (
        "As colunas ['campo'] têm valores de tipos que o polars não reúne numa coluna tipada "
        "e saem com o tipo Object, com os objetos Python que o modelo devolveu, exceto o Enum "
        "fora de lista ou dict, que sai pelo valor. Converta-as antes de gravar em arquivo."
    )

    with pytest.warns(UserWarning, match=_exato(msg)):
        resultado = from_pandas(df, ConversionInfo(original_type=ORIGINAL_TYPE_POLARS_DF))

    assert resultado.schema == {"campo": pl.Object}
    assert resultado["campo"].to_list() == valores


def test_entrada_polars_guarda_o_int_exato_depois_de_campo_ausente():
    pl = pytest.importorskip("polars")
    grande = 2**53 + 1
    dados = pl.DataFrame({"n": [grande, None]})

    df = _python_values(dados, {"ausente": int, "n": int})

    assert df["n"].dtype == object
    assert df["n"].tolist() == [grande, None]


# =============================================================================
# _to_polars
# =============================================================================


def test_anotacao_que_junta_tipos_convertidos_levanta_com_a_anotacao():
    pytest.importorskip("polars")
    anotacao = Union[dt.date, int]
    msg = f"A anotação {anotacao} junta tipos que o pyarrow converte uns nos outros."

    with pytest.raises(TypeError, match=_exato(msg)):
        _to_polars(pd.Series([dt.date(2024, 1, 2)], dtype=object), anotacao)


def test_valores_que_o_pyarrow_mudaria_levantam_com_mensagem_completa():
    pytest.importorskip("polars")
    valores = pd.Series([dt.date(2024, 1, 2), 3], dtype=object)

    with pytest.raises(TypeError, match=_exato("O pyarrow mudaria valores da coluna ao tipá-la.")):
        _to_polars(valores, None)


# =============================================================================
# _mixes_converted_types
# =============================================================================


@pytest.mark.parametrize(
    "anotacao",
    [
        list[Union[dt.date, int]],
        list[Union[float, bool]],
        tuple[Union[dt.date, int], ...],
        list[list[Union[bytes, str]]],
    ],
)
def test_lista_ou_tupla_com_uniao_convertida_mistura_tipos(anotacao):
    assert _mixes_converted_types(anotacao) is True


@pytest.mark.parametrize("anotacao", [list[int], tuple[str, ...], list[list[dt.date]]])
def test_lista_ou_tupla_de_um_tipo_so_nao_mistura(anotacao):
    assert _mixes_converted_types(anotacao) is False


# =============================================================================
# _group_kinds
# =============================================================================


class _ClassesEmOrdem(set[type]):
    """Conjunto de classes que itera na ordem dada, para fixar a ordem do laço."""

    def __init__(self, ordem):
        super().__init__(ordem)
        self._ordem = list(ordem)

    @override
    def __iter__(self):
        return iter(self._ordem)


@pytest.mark.parametrize("ignorado", [type(None), type(pd.NaT), dict, list, tuple])
def test_nulo_e_estrutura_nao_encerram_a_conferencia_das_outras_classes(ignorado):
    grupo = [1.5, dt.date(2024, 1, 2)]

    assert _group_kinds(grupo, _ClassesEmOrdem([ignorado, float, dt.date])) == {float, dt.date}


def test_datetime_conta_pelo_fuso():
    sem_fuso = dt.datetime(2024, 1, 2, 3)  # noqa: DTZ001 (o caso é o datetime sem fuso)
    com_fuso = dt.datetime(2024, 1, 2, 3, tzinfo=dt.UTC)

    assert _group_kinds([sem_fuso], {dt.datetime}) == {dt.datetime}
    assert _group_kinds([com_fuso], {dt.datetime}) == {_AWARE_DATETIME}


def test_decimal_e_time_sem_valor_especial_contam_pela_classe():
    grupo = [Decimal("1.5"), dt.time(3, 4)]

    assert _group_kinds(grupo, {Decimal, dt.time}) == {Decimal, dt.time}


@pytest.mark.parametrize(
    "valor", [Decimal("NaN"), dt.time(3, 4, tzinfo=dt.UTC)], ids=["decimal", "time"]
)
def test_decimal_nan_e_time_com_fuso_nao_cabem_em_tipo_do_polars(valor):
    assert _group_kinds([valor], {type(valor)}) is None


class _Outra:
    """Classe fora de `_KINDS` e das ABCs de número."""


def test_numero_do_numpy_conta_pela_abc_e_outra_classe_por_ela_mesma():
    grupo = [np.int64(1), np.float32(1.5), _Outra()]

    assert _group_kinds(grupo, {np.int64, np.float32, _Outra}) == {int, float, _Outra}


# =============================================================================
# _column_rank
# =============================================================================


def test_familia_e_posicao_de_cada_coluna():
    colunas = [
        "_trace",
        "_trace_x",
        "_search_credits",
        "_input_tokens",
        "_reasoning_tokens",
        "status",
        "_error_details",
        "campo",
    ]

    assert [_column_rank(col, "status") for col in colunas] == [
        (1, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (3, 3),
        (4, 0),
        (4, 0),
        (0, 0),
    ]


# =============================================================================
# _accepts_only_int e read_df
# =============================================================================


@pytest.mark.parametrize(
    ("anotacao", "esperado"),
    [
        (Literal[1, 2], True),
        (Optional[Literal[1, 2]], True),
        (Literal["a"], False),
        (Literal[1, "a"], False),
        (Literal[True], False),
    ],
)
def test_literal_so_de_ints_aceita_so_int(anotacao, esperado):
    assert _accepts_only_int(anotacao) is esperado


def test_arquivo_ausente_levanta_com_o_caminho(tmp_path):
    arquivo = tmp_path / "nao_existe.csv"

    with pytest.raises(FileNotFoundError, match=_exato(f"Arquivo não encontrado: {arquivo}")):
        read_df(arquivo)
