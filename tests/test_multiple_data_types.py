"""Testes para suporte a múltiplos tipos de dados.

Testa conversão de/para:
- pandas.Series
- polars.Series
- list
- dict
"""
import pandas as pd
import pytest

# Import opcional de polars
try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

from dataframeit.utils import (
    to_pandas,
    from_pandas,
    ConversionInfo,
    DEFAULT_TEXT_COLUMN,
    ORIGINAL_TYPE_PANDAS_DF,
    ORIGINAL_TYPE_POLARS_DF,
    ORIGINAL_TYPE_PANDAS_SERIES,
    ORIGINAL_TYPE_POLARS_SERIES,
    ORIGINAL_TYPE_LIST,
    ORIGINAL_TYPE_DICT,
)


class TestToPandas:
    """Testes para a função to_pandas."""

    def test_pandas_dataframe(self):
        """Testa conversão de pandas DataFrame."""
        df = pd.DataFrame({'texto': ['a', 'b'], 'id': [1, 2]})
        result, info = to_pandas(df)

        assert isinstance(result, pd.DataFrame)
        assert info.original_type == ORIGINAL_TYPE_PANDAS_DF
        assert len(result) == 2
        print("✅ to_pandas com pandas.DataFrame")

    @pytest.mark.skipif(not HAS_POLARS, reason="polars não instalado")
    def test_polars_dataframe(self):
        """Testa conversão de polars DataFrame."""
        df = pl.DataFrame({'texto': ['a', 'b'], 'id': [1, 2]})
        result, info = to_pandas(df)

        assert isinstance(result, pd.DataFrame)
        assert info.original_type == ORIGINAL_TYPE_POLARS_DF
        assert len(result) == 2
        print("✅ to_pandas com polars.DataFrame")

    def test_pandas_series(self):
        """Testa conversão de pandas Series."""
        series = pd.Series(['texto 1', 'texto 2', 'texto 3'], name='minha_serie')
        result, info = to_pandas(series)

        assert isinstance(result, pd.DataFrame)
        assert info.original_type == ORIGINAL_TYPE_PANDAS_SERIES
        assert DEFAULT_TEXT_COLUMN in result.columns
        assert len(result) == 3
        assert info.series_name == 'minha_serie'
        print("✅ to_pandas com pandas.Series")

    def test_pandas_series_preserves_index(self):
        """Testa que o índice da Series é preservado."""
        series = pd.Series(['a', 'b'], index=['x', 'y'])
        result, info = to_pandas(series)

        assert info.original_index is not None
        assert list(info.original_index) == ['x', 'y']
        print("✅ to_pandas preserva índice de pandas.Series")

    @pytest.mark.skipif(not HAS_POLARS, reason="polars não instalado")
    def test_polars_series(self):
        """Testa conversão de polars Series."""
        series = pl.Series('minha_serie', ['texto 1', 'texto 2'])
        result, info = to_pandas(series)

        assert isinstance(result, pd.DataFrame)
        assert info.original_type == ORIGINAL_TYPE_POLARS_SERIES
        assert DEFAULT_TEXT_COLUMN in result.columns
        assert len(result) == 2
        assert info.series_name == 'minha_serie'
        print("✅ to_pandas com polars.Series")

    def test_list(self):
        """Testa conversão de list."""
        data = ['texto 1', 'texto 2', 'texto 3']
        result, info = to_pandas(data)

        assert isinstance(result, pd.DataFrame)
        assert info.original_type == ORIGINAL_TYPE_LIST
        assert DEFAULT_TEXT_COLUMN in result.columns
        assert len(result) == 3
        assert list(result[DEFAULT_TEXT_COLUMN]) == data
        print("✅ to_pandas com list")

    def test_dict(self):
        """Testa conversão de dict."""
        data = {
            'doc1': 'texto do documento 1',
            'doc2': 'texto do documento 2',
            'doc3': 'texto do documento 3',
        }
        result, info = to_pandas(data)

        assert isinstance(result, pd.DataFrame)
        assert info.original_type == ORIGINAL_TYPE_DICT
        assert DEFAULT_TEXT_COLUMN in result.columns
        assert len(result) == 3
        assert list(result.index) == ['doc1', 'doc2', 'doc3']
        print("✅ to_pandas com dict")

    def test_unsupported_type_raises_error(self):
        """Testa que tipos não suportados geram erro."""
        with pytest.raises(TypeError, match="Tipo não suportado"):
            to_pandas(42)

        with pytest.raises(TypeError, match="Tipo não suportado"):
            to_pandas("string simples")
        print("✅ to_pandas rejeita tipos não suportados")


class TestFromPandas:
    """Testes para a função from_pandas."""

    def test_pandas_dataframe(self):
        """Testa reconversão para pandas DataFrame."""
        df = pd.DataFrame({'texto': ['a', 'b'], 'resultado': [1, 2]})
        info = ConversionInfo(original_type=ORIGINAL_TYPE_PANDAS_DF)

        result = from_pandas(df, info)

        assert isinstance(result, pd.DataFrame)
        print("✅ from_pandas com pandas.DataFrame")

    @pytest.mark.skipif(not HAS_POLARS, reason="polars não instalado")
    def test_polars_dataframe(self):
        """Testa reconversão para polars DataFrame."""
        df = pd.DataFrame({'texto': ['a', 'b'], 'resultado': [1, 2]})
        info = ConversionInfo(original_type=ORIGINAL_TYPE_POLARS_DF)

        result = from_pandas(df, info)

        assert isinstance(result, pl.DataFrame)
        print("✅ from_pandas com polars.DataFrame")

    def test_pandas_series_returns_dataframe(self):
        """Testa que pandas Series retorna DataFrame com resultados."""
        df = pd.DataFrame({
            DEFAULT_TEXT_COLUMN: ['texto 1', 'texto 2'],
            'sentimento': ['positivo', 'negativo'],
            'score': [0.9, 0.8],
        })
        original_index = pd.Index(['a', 'b'])
        info = ConversionInfo(
            original_type=ORIGINAL_TYPE_PANDAS_SERIES,
            original_index=original_index,
            series_name='textos',
        )

        result = from_pandas(df, info)

        assert isinstance(result, pd.DataFrame)
        assert DEFAULT_TEXT_COLUMN not in result.columns  # Coluna de texto removida
        assert 'sentimento' in result.columns
        assert 'score' in result.columns
        assert list(result.index) == ['a', 'b']  # Índice restaurado
        print("✅ from_pandas com pandas.Series retorna DataFrame")

    def test_list_returns_dataframe(self):
        """Testa que list retorna DataFrame."""
        df = pd.DataFrame({
            DEFAULT_TEXT_COLUMN: ['texto 1', 'texto 2'],
            'sentimento': ['positivo', 'negativo'],
        })
        info = ConversionInfo(original_type=ORIGINAL_TYPE_LIST)

        result = from_pandas(df, info)

        assert isinstance(result, pd.DataFrame)
        assert DEFAULT_TEXT_COLUMN not in result.columns
        assert 'sentimento' in result.columns
        print("✅ from_pandas com list retorna DataFrame")

    def test_dict_returns_dataframe_with_keys_as_index(self):
        """Testa que dict retorna DataFrame com chaves como índice."""
        df = pd.DataFrame(
            {
                DEFAULT_TEXT_COLUMN: ['texto 1', 'texto 2'],
                'categoria': ['A', 'B'],
            },
            index=['doc1', 'doc2'],
        )
        info = ConversionInfo(
            original_type=ORIGINAL_TYPE_DICT,
            original_index=['doc1', 'doc2'],
        )

        result = from_pandas(df, info)

        assert isinstance(result, pd.DataFrame)
        assert DEFAULT_TEXT_COLUMN not in result.columns
        assert list(result.index) == ['doc1', 'doc2']
        print("✅ from_pandas com dict retorna DataFrame com chaves como índice")

    def test_retrocompat_bool_false(self):
        """Testa retrocompatibilidade com was_polars=False."""
        df = pd.DataFrame({'a': [1, 2]})

        result = from_pandas(df, False)

        assert isinstance(result, pd.DataFrame)
        print("✅ from_pandas retrocompatível com bool False")

    @pytest.mark.skipif(not HAS_POLARS, reason="polars não instalado")
    def test_retrocompat_bool_true(self):
        """Testa retrocompatibilidade com was_polars=True."""
        df = pd.DataFrame({'a': [1, 2]})

        result = from_pandas(df, True)

        assert isinstance(result, pl.DataFrame)
        print("✅ from_pandas retrocompatível com bool True")


class TestRoundTrip:
    """Testes de ida e volta (conversão e reconversão)."""

    def test_pandas_series_roundtrip(self):
        """Testa ida e volta com pandas Series."""
        original = pd.Series(['texto 1', 'texto 2'], index=['a', 'b'], name='textos')

        df, info = to_pandas(original)
        # Simula processamento adicionando colunas
        df['resultado'] = ['positivo', 'negativo']

        result = from_pandas(df, info)

        assert isinstance(result, pd.DataFrame)
        assert list(result.index) == ['a', 'b']
        assert 'resultado' in result.columns
        print("✅ Roundtrip pandas.Series")

    def test_list_roundtrip(self):
        """Testa ida e volta com list."""
        original = ['texto A', 'texto B', 'texto C']

        df, info = to_pandas(original)
        # Simula processamento
        df['categoria'] = ['X', 'Y', 'Z']

        result = from_pandas(df, info)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert 'categoria' in result.columns
        print("✅ Roundtrip list")

    def test_dict_roundtrip(self):
        """Testa ida e volta com dict."""
        original = {
            'documento_1': 'conteúdo do primeiro documento',
            'documento_2': 'conteúdo do segundo documento',
        }

        df, info = to_pandas(original)
        # Simula processamento
        df['resumo'] = ['resumo 1', 'resumo 2']

        result = from_pandas(df, info)

        assert isinstance(result, pd.DataFrame)
        assert list(result.index) == ['documento_1', 'documento_2']
        assert 'resumo' in result.columns
        print("✅ Roundtrip dict")


if __name__ == '__main__':
    print("=" * 60)
    print("TESTES DE MÚLTIPLOS TIPOS DE DADOS")
    print("=" * 60)
    print()

    # Testes to_pandas
    test_to = TestToPandas()
    test_to.test_pandas_dataframe()
    test_to.test_pandas_series()
    test_to.test_pandas_series_preserves_index()
    test_to.test_list()
    test_to.test_dict()
    test_to.test_unsupported_type_raises_error()

    if HAS_POLARS:
        test_to.test_polars_dataframe()
        test_to.test_polars_series()

    print()

    # Testes from_pandas
    test_from = TestFromPandas()
    test_from.test_pandas_dataframe()
    test_from.test_pandas_series_returns_dataframe()
    test_from.test_list_returns_dataframe()
    test_from.test_dict_returns_dataframe_with_keys_as_index()
    test_from.test_retrocompat_bool_false()

    if HAS_POLARS:
        test_from.test_polars_dataframe()
        test_from.test_retrocompat_bool_true()

    print()

    # Testes roundtrip
    test_rt = TestRoundTrip()
    test_rt.test_pandas_series_roundtrip()
    test_rt.test_list_roundtrip()
    test_rt.test_dict_roundtrip()

    print()
    print("=" * 60)
    print("TODOS OS TESTES PASSARAM!")
    print("=" * 60)
