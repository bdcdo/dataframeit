"""Testes para leitura a partir de fonte SQL (#87)."""

import sqlite3
import pandas as pd
import pytest
from pydantic import BaseModel, Field
from unittest.mock import patch


class _Model(BaseModel):
    resumo: str = Field(description="resumo curto")


def _mock_call_langchain(*args, **kwargs):
    return {"data": {"resumo": "ok"}, "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}


@pytest.fixture
def sqlite_conn():
    """Conexão SQLite em memória com tabela de fixture."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE decisoes (id INTEGER PRIMARY KEY, texto TEXT)")
    conn.executemany(
        "INSERT INTO decisoes (id, texto) VALUES (?, ?)",
        [(1, "texto um"), (2, "texto dois"), (3, "texto três")],
    )
    conn.commit()
    try:
        yield conn
    finally:
        conn.close()


def test_to_pandas_reads_sql_tuple(sqlite_conn):
    """to_pandas aceita tupla (query, conexao)."""
    from dataframeit.utils import to_pandas, ORIGINAL_TYPE_SQL

    df, info = to_pandas(("SELECT id, texto FROM decisoes", sqlite_conn))
    assert info.original_type == ORIGINAL_TYPE_SQL
    assert len(df) == 3
    assert set(df.columns) == {"id", "texto"}


def test_to_pandas_reads_sql_with_con_kwarg(sqlite_conn):
    """to_pandas aceita query string + con=."""
    from dataframeit.utils import to_pandas, ORIGINAL_TYPE_SQL

    df, info = to_pandas("SELECT texto FROM decisoes", con=sqlite_conn)
    assert info.original_type == ORIGINAL_TYPE_SQL
    assert df["texto"].tolist() == ["texto um", "texto dois", "texto três"]


def test_dataframeit_with_sql_tuple(sqlite_conn):
    """Pipeline end-to-end: tupla (query, conn) → DataFrame processado."""
    from dataframeit.core import dataframeit

    with patch("dataframeit.core.call_langchain", side_effect=_mock_call_langchain):
        with patch("dataframeit.core.validate_provider_dependencies"):
            result = dataframeit(
                ("SELECT id, texto FROM decisoes", sqlite_conn),
                _Model,
                "resuma",
            )
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert result["resumo"].tolist() == ["ok", "ok", "ok"]


def test_dataframeit_with_sql_con_kwarg(sqlite_conn):
    """Pipeline end-to-end: query string + con= → DataFrame processado."""
    from dataframeit.core import dataframeit

    with patch("dataframeit.core.call_langchain", side_effect=_mock_call_langchain):
        with patch("dataframeit.core.validate_provider_dependencies"):
            result = dataframeit(
                "SELECT texto FROM decisoes",
                _Model,
                "resuma",
                con=sqlite_conn,
            )
    assert isinstance(result, pd.DataFrame)
    assert result["resumo"].tolist() == ["ok", "ok", "ok"]


def test_plain_string_without_con_still_not_treated_as_sql():
    """String sem con= não é tratada como SQL — continua caindo no TypeError existente."""
    from dataframeit.utils import to_pandas

    with pytest.raises(TypeError):
        to_pandas("SELECT 1")
