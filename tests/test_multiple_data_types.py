"""Testes para suporte a múltiplos tipos de dados.

Testa conversão de/para:
- pandas.Series
- polars.Series
- list
- dict
"""

import datetime
import importlib.util
import sys

import pandas as pd
import pytest

# Import opcional de polars
try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

from dataframeit.utils import (
    DEFAULT_TEXT_COLUMN,
    ORIGINAL_TYPE_DICT,
    ORIGINAL_TYPE_LIST,
    ORIGINAL_TYPE_PANDAS_DF,
    ORIGINAL_TYPE_PANDAS_SERIES,
    ORIGINAL_TYPE_POLARS_DF,
    ORIGINAL_TYPE_POLARS_SERIES,
    ConversionInfo,
    _enums_by_value,
    from_pandas,
    to_pandas,
)


class TestToPandas:
    """Testes para a função to_pandas."""

    def test_pandas_dataframe(self):
        """Testa conversão de pandas DataFrame."""
        df = pd.DataFrame({"texto": ["a", "b"], "id": [1, 2]})
        result, info = to_pandas(df)

        assert isinstance(result, pd.DataFrame)
        assert info.original_type == ORIGINAL_TYPE_PANDAS_DF
        assert len(result) == 2

    @pytest.mark.skipif(not HAS_POLARS, reason="polars não instalado")
    def test_polars_dataframe(self):
        """Testa conversão de polars DataFrame."""
        df = pl.DataFrame({"texto": ["a", "b"], "id": [1, 2]})
        result, info = to_pandas(df)

        assert isinstance(result, pd.DataFrame)
        assert info.original_type == ORIGINAL_TYPE_POLARS_DF
        assert len(result) == 2

    def test_pandas_series(self):
        """Testa conversão de pandas Series."""
        series = pd.Series(["texto 1", "texto 2", "texto 3"], name="minha_serie")
        result, info = to_pandas(series)

        assert isinstance(result, pd.DataFrame)
        assert info.original_type == ORIGINAL_TYPE_PANDAS_SERIES
        assert DEFAULT_TEXT_COLUMN in result.columns
        assert len(result) == 3
        assert info.series_name == "minha_serie"

    def test_pandas_series_preserves_index(self):
        """Testa que o índice da Series é preservado."""
        series = pd.Series(["a", "b"], index=["x", "y"])
        _result, info = to_pandas(series)

        assert info.original_index is not None
        assert list(info.original_index) == ["x", "y"]

    @pytest.mark.skipif(not HAS_POLARS, reason="polars não instalado")
    def test_polars_series(self):
        """Testa conversão de polars Series."""
        series = pl.Series("minha_serie", ["texto 1", "texto 2"])
        result, info = to_pandas(series)

        assert isinstance(result, pd.DataFrame)
        assert info.original_type == ORIGINAL_TYPE_POLARS_SERIES
        assert DEFAULT_TEXT_COLUMN in result.columns
        assert len(result) == 2
        assert info.series_name == "minha_serie"

    def test_list(self):
        """Testa conversão de list."""
        data = ["texto 1", "texto 2", "texto 3"]
        result, info = to_pandas(data)

        assert isinstance(result, pd.DataFrame)
        assert info.original_type == ORIGINAL_TYPE_LIST
        assert DEFAULT_TEXT_COLUMN in result.columns
        assert len(result) == 3
        assert list(result[DEFAULT_TEXT_COLUMN]) == data

    def test_dict(self):
        """Testa conversão de dict."""
        data = {
            "doc1": "texto do documento 1",
            "doc2": "texto do documento 2",
            "doc3": "texto do documento 3",
        }
        result, info = to_pandas(data)

        assert isinstance(result, pd.DataFrame)
        assert info.original_type == ORIGINAL_TYPE_DICT
        assert DEFAULT_TEXT_COLUMN in result.columns
        assert len(result) == 3
        assert list(result.index) == ["doc1", "doc2", "doc3"]

    def test_unsupported_type_raises_error(self):
        """Testa que tipos não suportados geram erro."""
        with pytest.raises(TypeError, match="Tipo não suportado"):
            to_pandas(42)

        with pytest.raises(TypeError, match="Tipo não suportado"):
            to_pandas("string simples")


class TestFromPandas:
    """Testes para a função from_pandas."""

    def test_pandas_dataframe(self):
        """Testa reconversão para pandas DataFrame."""
        df = pd.DataFrame({"texto": ["a", "b"], "resultado": [1, 2]})
        info = ConversionInfo(original_type=ORIGINAL_TYPE_PANDAS_DF)

        result = from_pandas(df, info)

        assert isinstance(result, pd.DataFrame)

    @pytest.mark.skipif(not HAS_POLARS, reason="polars não instalado")
    def test_polars_dataframe(self):
        """Testa reconversão para polars DataFrame."""
        df = pd.DataFrame({"texto": ["a", "b"], "resultado": [1, 2]})
        info = ConversionInfo(original_type=ORIGINAL_TYPE_POLARS_DF)

        result = from_pandas(df, info)

        assert isinstance(result, pl.DataFrame)

    def test_pandas_series_returns_dataframe(self):
        """Testa que pandas Series retorna DataFrame com resultados."""
        df = pd.DataFrame(
            {
                DEFAULT_TEXT_COLUMN: ["texto 1", "texto 2"],
                "sentimento": ["positivo", "negativo"],
                "score": [0.9, 0.8],
            }
        )
        original_index = pd.Index(["a", "b"])
        info = ConversionInfo(
            original_type=ORIGINAL_TYPE_PANDAS_SERIES,
            original_index=original_index,
            series_name="textos",
        )

        result = from_pandas(df, info)

        assert isinstance(result, pd.DataFrame)
        assert DEFAULT_TEXT_COLUMN not in result.columns  # Coluna de texto removida
        assert "sentimento" in result.columns
        assert "score" in result.columns
        assert list(result.index) == ["a", "b"]  # Índice restaurado

    def test_list_returns_dataframe(self):
        """Testa que list retorna DataFrame."""
        df = pd.DataFrame(
            {
                DEFAULT_TEXT_COLUMN: ["texto 1", "texto 2"],
                "sentimento": ["positivo", "negativo"],
            }
        )
        info = ConversionInfo(original_type=ORIGINAL_TYPE_LIST)

        result = from_pandas(df, info)

        assert isinstance(result, pd.DataFrame)
        assert DEFAULT_TEXT_COLUMN not in result.columns
        assert "sentimento" in result.columns

    def test_dict_returns_dataframe_with_keys_as_index(self):
        """Testa que dict retorna DataFrame com chaves como índice."""
        df = pd.DataFrame(
            {
                DEFAULT_TEXT_COLUMN: ["texto 1", "texto 2"],
                "categoria": ["A", "B"],
            },
            index=["doc1", "doc2"],
        )
        info = ConversionInfo(
            original_type=ORIGINAL_TYPE_DICT,
            original_index=["doc1", "doc2"],
        )

        result = from_pandas(df, info)

        assert isinstance(result, pd.DataFrame)
        assert DEFAULT_TEXT_COLUMN not in result.columns
        assert list(result.index) == ["doc1", "doc2"]

    def test_retrocompat_bool_false(self):
        """Testa retrocompatibilidade com was_polars=False."""
        df = pd.DataFrame({"a": [1, 2]})

        result = from_pandas(df, False)

        assert isinstance(result, pd.DataFrame)

    @pytest.mark.skipif(not HAS_POLARS, reason="polars não instalado")
    def test_retrocompat_bool_true(self):
        """Testa retrocompatibilidade com was_polars=True."""
        df = pd.DataFrame({"a": [1, 2]})

        result = from_pandas(df, True)

        assert isinstance(result, pl.DataFrame)


class TestRoundTrip:
    """Testes de ida e volta (conversão e reconversão)."""

    def test_pandas_series_roundtrip(self):
        """Testa ida e volta com pandas Series."""
        original = pd.Series(["texto 1", "texto 2"], index=["a", "b"], name="textos")

        df, info = to_pandas(original)
        # Simula processamento adicionando colunas
        df["resultado"] = ["positivo", "negativo"]

        result = from_pandas(df, info)

        assert isinstance(result, pd.DataFrame)
        assert list(result.index) == ["a", "b"]
        assert "resultado" in result.columns

    def test_list_roundtrip(self):
        """Testa ida e volta com list."""
        original = ["texto A", "texto B", "texto C"]

        df, info = to_pandas(original)
        # Simula processamento
        df["categoria"] = ["X", "Y", "Z"]

        result = from_pandas(df, info)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert "categoria" in result.columns

    def test_dict_roundtrip(self):
        """Testa ida e volta com dict."""
        original = {
            "documento_1": "conteúdo do primeiro documento",
            "documento_2": "conteúdo do segundo documento",
        }

        df, info = to_pandas(original)
        # Simula processamento
        df["resumo"] = ["resumo 1", "resumo 2"]

        result = from_pandas(df, info)

        assert isinstance(result, pd.DataFrame)
        assert list(result.index) == ["documento_1", "documento_2"]
        assert "resumo" in result.columns


@pytest.mark.skipif(not HAS_POLARS, reason="polars não instalado")
class TestEntradaPolars:
    """A entrada polars guarda o frame original e não arredonda inteiro com nulo."""

    def test_inteiro_com_nulo_mantem_precisao_e_tipo(self):
        ids = [9007199254740993, None, 1234567890123456789]
        data = pl.DataFrame({"texto": ["a", "b", "c"], "id": pl.Series(ids, dtype=pl.Int64)})

        df, info = to_pandas(data)
        df["resultado"] = ["x", "y", "z"]
        result = from_pandas(df, info)

        assert result["id"].dtype == pl.Int64
        assert result["id"].to_list() == ids
        assert result["resultado"].to_list() == ["x", "y", "z"]

    def test_inteiro_sem_nulo_fica_como_estava(self):
        data = pl.DataFrame({"id": pl.Series([1, 2], dtype=pl.UInt8)})

        df, info = to_pandas(data)

        assert df["id"].dtype == "uint8"
        assert from_pandas(df, info)["id"].dtype == pl.UInt8

    def test_coluna_de_data_continua_data(self):
        data = pl.DataFrame({"texto": ["a"], "dia": [datetime.date(2024, 1, 2)]})

        df, info = to_pandas(data)
        result = from_pandas(df, info)

        assert result["dia"].dtype == pl.Date
        assert result["dia"].to_list() == [datetime.date(2024, 1, 2)]

    def test_campo_do_modelo_que_estava_na_entrada_sai_do_pandas(self):
        data = pl.DataFrame({"texto": ["a"], "nota": [7]})
        df, info = to_pandas(data)
        df["nota"] = [7.5]

        assert from_pandas(df, info)["nota"].to_list() == [7]
        assert from_pandas(df, info, model_columns=["nota"])["nota"].to_list() == [7.5]

    @pytest.mark.parametrize(
        "coluna",
        [
            "_trace",
            "_trace_campo",
            "_search_credits",
            "_input_tokens",
            "_dataframeit_status",
            "_error_details",
        ],
    )
    def test_coluna_de_controle_que_estava_na_entrada_sai_do_pandas(self, coluna):
        data = pl.DataFrame({"texto": ["a", "b"], coluna: ["antes", None]})
        df, info = to_pandas(data)
        df[coluna] = ["depois", "error"]

        result = from_pandas(df, info)

        assert result[coluna].to_list() == ["depois", "error"]

    def test_coluna_que_o_arrow_nao_tipa_sai_como_object_com_aviso(self):
        df = pd.DataFrame({"a": [1, "dois"], "b": ["x", "y"]})

        with pytest.warns(UserWarning, match=r"\['a'\].*Object"):
            result = from_pandas(df, True)

        assert result["a"].dtype == pl.Object
        assert result["a"].to_list() == [1, "dois"]
        assert result["b"].to_list() == ["x", "y"]

    def test_troca_de_enum_devolve_a_mesma_coluna_quando_ela_nao_e_de_objetos(self):
        valores = pd.Series([1.5, 2.5])

        assert _enums_by_value(valores) is valores

    def test_coluna_de_status_personalizada_que_estava_na_entrada_sai_do_pandas(self):
        data = pl.DataFrame({"texto": ["a", "b"], "st": ["error", None]})
        df, info = to_pandas(data)
        df["st"] = ["processed", "error"]

        assert from_pandas(df, info)["st"].to_list() == ["error", None]
        assert from_pandas(df, info, "st")["st"].to_list() == ["processed", "error"]


def test_rotulo_repetido_nao_duplica_coluna_na_reordenacao():
    df = pd.DataFrame([[1, 2, 3, 4]], columns=["_input_tokens", "a", "_input_tokens", "a"])

    resultado = from_pandas(df, ConversionInfo(original_type=ORIGINAL_TYPE_PANDAS_DF))

    assert list(resultado.columns) == ["a", "a", "_input_tokens", "_input_tokens"]
    assert resultado.to_numpy().tolist() == [[2, 4, 1, 3]]


def test_coluna_trace_sem_sufixo_fica_com_as_de_trace():
    df = pd.DataFrame([[1, 2, 3]], columns=["_trace", "texto", "_search_credits"])

    resultado = from_pandas(df, ConversionInfo(original_type=ORIGINAL_TYPE_PANDAS_DF))

    assert list(resultado.columns) == ["texto", "_trace", "_search_credits"]


def test_dict_com_chaves_tupla_mantem_indice_simples():
    """Chaves-tupla viram rótulos de um Index simples, sem MultiIndex."""
    original = {("sp", 2020): "texto a", ("rj", 2021): "texto b"}

    df, info = to_pandas(original)
    df["resumo"] = ["resumo a", "resumo b"]
    resultado = from_pandas(df, info)

    assert isinstance(resultado, pd.DataFrame)
    assert not isinstance(resultado.index, pd.MultiIndex)
    assert list(resultado.index) == [("sp", 2020), ("rj", 2021)]


def test_sem_polars_a_entrada_pandas_continua_funcionando(monkeypatch):
    """Sem o extra polars, o import opcional cai no fallback e o pandas segue igual.

    O módulo é carregado numa cópia à parte, com o import de polars bloqueado,
    para não trocar o dataframeit.utils de que os outros módulos dependem.
    """
    monkeypatch.setitem(sys.modules, "polars", None)
    spec = importlib.util.find_spec("dataframeit.utils")
    assert spec is not None
    assert spec.loader is not None
    copia = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(copia)

    df = pd.DataFrame({"texto": ["a", "b"]})
    convertido, info = copia.to_pandas(df)

    assert copia.pl is None
    assert info.original_type == ORIGINAL_TYPE_PANDAS_DF
    assert copia.from_pandas(convertido, info).equals(df)
