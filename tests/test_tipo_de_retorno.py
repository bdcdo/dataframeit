"""O tipo de retorno de dataframeit() acompanha o tipo da entrada, e o de read_df(), o argumento.

Quem confere é o ty, que roda sobre tests/ no CI: um assert_type que não bate é
erro de tipo. O corpo fica sob TYPE_CHECKING porque executá-lo chamaria o LLM
ou leria um arquivo.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Literal, assert_type

    import pandas as pd
    import polars as pl
    from pandas.api.typing import JsonReader
    from pandas.io.parsers import TextFileReader
    from pydantic import BaseModel

    from dataframeit import dataframeit, read_df

    # Tudo o que read_df() pode devolver, que é o que o checador vê quando o
    # argumento não diz qual deles.
    _Tudo = (
        pd.DataFrame | pd.Series | dict[int | str, pd.DataFrame] | TextFileReader | JsonReader[Any]
    )

    def _pandas_devolve_pandas(questions: type[BaseModel], entrada: pd.DataFrame) -> None:
        assert_type(dataframeit(entrada, questions), pd.DataFrame)

    def _serie_pandas_devolve_pandas(questions: type[BaseModel], entrada: pd.Series) -> None:
        assert_type(dataframeit(entrada, questions), pd.DataFrame)

    def _lista_devolve_pandas(questions: type[BaseModel], entrada: list[str]) -> None:
        assert_type(dataframeit(entrada, questions), pd.DataFrame)

    def _dict_devolve_pandas(questions: type[BaseModel], entrada: dict[str, str]) -> None:
        assert_type(dataframeit(entrada, questions), pd.DataFrame)

    def _polars_devolve_polars(questions: type[BaseModel], entrada: pl.DataFrame) -> None:
        assert_type(dataframeit(entrada, questions), pl.DataFrame)

    def _serie_polars_devolve_polars(questions: type[BaseModel], entrada: pl.Series) -> None:
        assert_type(dataframeit(entrada, questions), pl.DataFrame)

    def _leitura_sem_argumento_que_mude_o_leitor_devolve_dataframe(modelo: type[BaseModel]) -> None:
        assert_type(read_df("dados.csv"), pd.DataFrame)
        assert_type(read_df("dados.csv", modelo, False), pd.DataFrame)
        assert_type(read_df("dados.csv", sep=";"), pd.DataFrame)
        assert_type(read_df("dados.csv", chunksize=None), pd.DataFrame)
        assert_type(read_df("dados.csv", iterator=False), pd.DataFrame)
        assert_type(read_df("dados.xlsx", sheet_name="Aba"), pd.DataFrame)
        assert_type(read_df("dados.json", typ="frame"), pd.DataFrame)

    def _chunksize_e_iterator_devolvem_leitor_de_pedacos() -> None:
        assert_type(read_df("dados.csv", chunksize=2), TextFileReader | JsonReader[Any])
        assert_type(
            read_df("dados.json", typ="series", lines=True, chunksize=2),
            TextFileReader | JsonReader[Any],
        )
        assert_type(read_df("dados.csv", iterator=True), TextFileReader)

    def _sheet_name_none_ou_lista_devolve_dict_de_abas(abas: list[str]) -> None:
        assert_type(read_df("dados.xlsx", sheet_name=None), dict[int | str, pd.DataFrame])
        assert_type(read_df("dados.xlsx", sheet_name=[0, "Aba"]), dict[int | str, pd.DataFrame])
        assert_type(read_df("dados.xlsx", sheet_name=abas), dict[int | str, pd.DataFrame])

    def _typ_series_devolve_series() -> None:
        assert_type(read_df("serie.json", typ="series"), pd.Series)
        assert_type(read_df("serie.json", typ="series", chunksize=None), pd.Series)

    def _argumento_de_tipo_mais_largo_devolve_a_uniao(
        tamanho: int | None, tipo: Literal["frame", "series"]
    ) -> None:
        assert_type(read_df("dados.csv", chunksize=tamanho), _Tudo)
        assert_type(read_df("serie.json", typ="series", chunksize=tamanho), _Tudo)
        assert_type(read_df("serie.json", typ=tipo), _Tudo)
