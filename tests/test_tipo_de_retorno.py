"""O tipo de retorno de dataframeit() acompanha o tipo da entrada.

Quem confere é o ty, que roda sobre tests/ no CI: um assert_type que não bate é
erro de tipo. O corpo fica sob TYPE_CHECKING porque executá-lo chamaria o LLM.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    from pydantic import BaseModel
    from typing_extensions import assert_type

    from dataframeit import dataframeit

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
