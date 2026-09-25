"""Teste de compatibilidade entre código antigo e novo."""

from typing import Literal

import pandas as pd
import pytest
from pydantic import BaseModel, Field

from dataframeit.core import _get_processing_indices, _setup_columns
from dataframeit.core import dataframeit as dataframeit_new
from dataframeit.utils import (
    ORIGINAL_TYPE_PANDAS_DF,
    ConversionInfo,
    from_pandas,
    parse_json,
    to_pandas,
)


class TestModel(BaseModel):
    campo1: str = Field(..., description="Primeiro campo")
    campo2: Literal["A", "B"] = Field(..., description="Segundo campo")


def test_api_compatibility():
    """Testa que a API pública é 100% compatível."""

    # Importar versão nova

    # Criar DataFrame de teste
    df = pd.DataFrame({"texto": ["texto 1", "texto 2"], "id": [1, 2]})

    template = "Analise: {documento}\n{format}"

    # Verificar que ValueError é lançado sem questions/perguntas
    with pytest.raises(ValueError, match=r"(?i)questions"):
        dataframeit_new(df, prompt=template)

    # Verificar que ValueError é lançado sem prompt
    with pytest.raises(ValueError, match=r"(?i)prompt"):
        dataframeit_new(df, questions=TestModel)


def test_column_management():
    """Testa gerenciamento de colunas."""

    df = pd.DataFrame({"texto": ["a", "b"], "id": [1, 2]})
    expected_cols = ["campo1", "campo2"]

    # Testar setup básico
    _setup_columns(df, expected_cols, None, track_tokens=False)
    assert "campo1" in df.columns
    assert "campo2" in df.columns
    assert "_dataframeit_status" in df.columns
    assert "_error_details" in df.columns

    # Testar que não cria duplicatas
    df2 = df.copy()
    _setup_columns(df2, expected_cols, None, track_tokens=False)
    assert list(df.columns) == list(df2.columns)

    # Testar status_column customizada
    df3 = pd.DataFrame({"texto": ["a", "b"], "id": [1, 2]})
    _setup_columns(df3, expected_cols, "meu_status", track_tokens=False)
    assert "meu_status" in df3.columns


def test_resume_functionality():
    """Testa funcionalidade de resume."""

    df = pd.DataFrame(
        {"texto": ["a", "b", "c", "d"], "_dataframeit_status": [None, None, None, None]}
    )

    # Sem resume
    pending, count = _get_processing_indices(df, "_dataframeit_status", resume=False)
    assert pending == [True, True, True, True]
    assert count == 0

    # Com resume e nada processado
    pending, count = _get_processing_indices(df, "_dataframeit_status", resume=True)
    assert pending == [True, True, True, True]
    assert count == 0

    # Com resume e algumas linhas processadas
    df.loc[0, "_dataframeit_status"] = "processed"
    df.loc[1, "_dataframeit_status"] = "processed"
    pending, count = _get_processing_indices(df, "_dataframeit_status", resume=True)
    assert pending == [False, False, True, True]
    assert count == 2

    # Com todas linhas processadas
    df["_dataframeit_status"] = "processed"
    pending, count = _get_processing_indices(df, "_dataframeit_status", resume=True)
    assert pending == [False, False, False, False]
    assert count == 4


def test_utils_functions():
    """Testa funções de utilidade."""

    # Parse JSON básico
    result = parse_json('{"a": 1, "b": "test"}')
    assert result == {"a": 1, "b": "test"}

    # Parse JSON com markdown
    result = parse_json('```json\n{"a": 2}\n```')
    assert result == {"a": 2}

    # Parse JSON com texto extra
    result = parse_json('Aqui está: {"a": 3} fim')
    assert result == {"a": 3}

    # Conversão pandas

    df = pd.DataFrame({"a": [1, 2, 3]})
    df_result, conversion_info = to_pandas(df)
    assert isinstance(df_result, pd.DataFrame)
    assert isinstance(conversion_info, ConversionInfo)
    assert conversion_info.original_type == ORIGINAL_TYPE_PANDAS_DF

    # Conversão de volta (com ConversionInfo)
    df_back = from_pandas(df_result, conversion_info)
    assert isinstance(df_back, pd.DataFrame)

    # Retrocompatibilidade: from_pandas ainda aceita bool
    df_back2 = from_pandas(df_result, False)
    assert isinstance(df_back2, pd.DataFrame)
