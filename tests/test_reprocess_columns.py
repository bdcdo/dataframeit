"""Testes para a funcionalidade reprocess_columns."""

import warnings
import pandas as pd
import pytest
from pydantic import BaseModel
from unittest.mock import patch, MagicMock

from dataframeit.core import dataframeit, _get_processing_indices


class SimpleModel(BaseModel):
    campo1: str
    campo2: str


class ExtendedModel(BaseModel):
    campo1: str
    campo2: str
    campo3: str


def test_reprocess_columns_validation_invalid_column():
    """Testa que reprocess_columns inválido levanta ValueError."""
    df = pd.DataFrame({"texto": ["a", "b"]})

    with patch("dataframeit.core.validate_provider_dependencies"):
        with pytest.raises(ValueError) as exc_info:
            dataframeit(
                df,
                questions=SimpleModel,
                prompt="Teste {texto}",
                reprocess_columns=["campo_inexistente"],
            )

        assert "campo_inexistente" in str(exc_info.value)
        assert "não estão no modelo" in str(exc_info.value)


def test_reprocess_columns_accepts_single_string():
    """Testa que reprocess_columns aceita uma string simples (não apenas lista)."""
    df = pd.DataFrame({"texto": ["a"]})

    # Mock da chamada LLM para não fazer requisições reais
    mock_result = {
        "data": {"campo1": "valor1", "campo2": "valor2"},
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }

    with patch("dataframeit.core.call_langchain", return_value=mock_result):
        with patch("dataframeit.core.validate_provider_dependencies"):
            # Deve aceitar string simples sem erro
            result = dataframeit(
                df,
                questions=SimpleModel,
                prompt="Teste {texto}",
                reprocess_columns="campo1",  # String, não lista
            )
            assert "campo1" in result.columns


def test_reprocess_columns_processes_all_rows():
    """Testa que reprocess_columns processa todas as linhas, mesmo as já processadas."""
    df = pd.DataFrame({
        "texto": ["a", "b", "c"],
        "campo1": ["old1", "old2", "old3"],
        "campo2": ["old_a", "old_b", "old_c"],
        "_dataframeit_status": ["processed", "processed", "processed"],
        "_error_details": [None, None, None],
    })

    # Mock da chamada LLM
    call_count = 0

    def mock_llm(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return {
            "data": {"campo1": f"new{call_count}", "campo2": f"new_{call_count}"},
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        }

    with patch("dataframeit.core.call_langchain", side_effect=mock_llm):
        with patch("dataframeit.core.validate_provider_dependencies"):
            result = dataframeit(
                df,
                questions=SimpleModel,
                prompt="Teste {texto}",
                reprocess_columns=["campo1", "campo2"],
            )

            # Todas as 3 linhas devem ter sido processadas
            assert call_count == 3


def test_reprocess_columns_only_updates_specified_columns():
    """Testa que reprocess_columns só atualiza as colunas especificadas em linhas já processadas."""
    df = pd.DataFrame({
        "texto": ["a", "b"],
        "campo1": ["original1", "original2"],
        "campo2": ["original_a", "original_b"],
        "_dataframeit_status": ["processed", "processed"],
        "_error_details": [None, None],
    })

    def mock_llm(*args, **kwargs):
        return {
            "data": {"campo1": "novo_valor", "campo2": "novo_valor_2"},
            "usage": {},
        }

    with patch("dataframeit.core.call_langchain", side_effect=mock_llm):
        with patch("dataframeit.core.validate_provider_dependencies"):
            result = dataframeit(
                df,
                questions=SimpleModel,
                prompt="Teste {texto}",
                reprocess_columns=["campo1"],  # Só reprocessar campo1
            )

            # campo1 deve ter sido atualizado
            assert result["campo1"].tolist() == ["novo_valor", "novo_valor"]
            # campo2 deve manter os valores originais (não estava em reprocess_columns)
            assert result["campo2"].tolist() == ["original_a", "original_b"]


def test_reprocess_columns_updates_all_columns_for_new_rows():
    """Testa que linhas não processadas recebem todas as colunas, não só as de reprocess."""
    df = pd.DataFrame({
        "texto": ["a", "b", "c"],
        "campo1": ["original1", "original2", None],  # Linha 3 não foi processada
        "campo2": ["original_a", "original_b", None],
        "_dataframeit_status": ["processed", "processed", None],  # Linha 3 pendente
        "_error_details": [None, None, None],
    })

    call_count = 0

    def mock_llm(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return {
            "data": {"campo1": f"novo{call_count}", "campo2": f"novo_{call_count}"},
            "usage": {},
        }

    with patch("dataframeit.core.call_langchain", side_effect=mock_llm):
        with patch("dataframeit.core.validate_provider_dependencies"):
            result = dataframeit(
                df,
                questions=SimpleModel,
                prompt="Teste {texto}",
                reprocess_columns=["campo1"],  # Só reprocessar campo1
            )

            # Linhas 1-2 (já processadas): só campo1 atualizado, campo2 mantém original
            assert result["campo1"].iloc[0] == "novo1"
            assert result["campo1"].iloc[1] == "novo2"
            assert result["campo2"].iloc[0] == "original_a"
            assert result["campo2"].iloc[1] == "original_b"

            # Linha 3 (nova): ambas as colunas atualizadas
            assert result["campo1"].iloc[2] == "novo3"
            assert result["campo2"].iloc[2] == "novo_3"


def test_reprocess_columns_does_not_skip_processed_rows():
    """Testa que linhas com status 'processed' são reprocessadas quando reprocess_columns é usado."""
    df = pd.DataFrame({
        "texto": ["texto1", "texto2"],
        "_dataframeit_status": ["processed", "processed"],
        "_error_details": [None, None],
    })

    call_count = 0

    def mock_llm(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return {
            "data": {"campo1": f"valor{call_count}", "campo2": f"outro{call_count}"},
            "usage": {},
        }

    with patch("dataframeit.core.call_langchain", side_effect=mock_llm):
        with patch("dataframeit.core.validate_provider_dependencies"):
            result = dataframeit(
                df,
                questions=SimpleModel,
                prompt="Teste {texto}",
                reprocess_columns=["campo1"],
            )

            # Ambas as linhas devem ter sido processadas
            assert call_count == 2


def test_resume_true_skips_processed_without_reprocess():
    """Testa que resume=True ainda pula linhas processadas quando reprocess_columns não é usado."""
    df = pd.DataFrame({
        "texto": ["a", "b", "c"],
        "campo1": ["old1", "old2", None],
        "campo2": ["old_a", "old_b", None],
        "_dataframeit_status": ["processed", "processed", None],
        "_error_details": [None, None, None],
    })

    call_count = 0

    def mock_llm(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return {
            "data": {"campo1": f"new{call_count}", "campo2": f"new_{call_count}"},
            "usage": {},
        }

    with patch("dataframeit.core.call_langchain", side_effect=mock_llm):
        with patch("dataframeit.core.validate_provider_dependencies"):
            result = dataframeit(
                df,
                questions=SimpleModel,
                prompt="Teste {texto}",
                resume=True,
                # sem reprocess_columns
            )

            # Apenas a linha não processada deve ser processada
            assert call_count == 1
            # Linhas já processadas mantêm valores antigos
            assert result["campo1"].tolist()[0] == "old1"
            assert result["campo1"].tolist()[1] == "old2"


def test_reprocess_columns_with_existing_columns_no_warning():
    """Testa que não há warning de conflito quando reprocess_columns é usado em colunas existentes."""
    df = pd.DataFrame({
        "texto": ["a"],
        "campo1": ["valor_existente"],
        "campo2": ["outro_existente"],
    })

    mock_result = {
        "data": {"campo1": "novo", "campo2": "outro_novo"},
        "usage": {},
    }

    with patch("dataframeit.core.call_langchain", return_value=mock_result):
        with patch("dataframeit.core.validate_provider_dependencies"):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                result = dataframeit(
                    df,
                    questions=SimpleModel,
                    prompt="Teste {texto}",
                    reprocess_columns=["campo1"],
                    resume=False,  # Normalmente geraria warning
                )

                # Não deve ter warning sobre conflito de colunas
                conflict_warnings = [warning for warning in w if "já existem" in str(warning.message)]
                assert len(conflict_warnings) == 0


def test_reprocess_columns_resume_after_interrupt():
    """Testa que após interrupção durante reprocess, resume normal continua apenas linhas pendentes."""
    df = pd.DataFrame({
        "texto": ["a", "b", "c"],
        "campo1": ["old1", "old2", "old3"],
        "campo2": ["old_a", "old_b", "old_c"],
        "_dataframeit_status": ["processed", "processed", "processed"],
        "_error_details": [None, None, None],
    })

    call_count = 0

    def mock_llm_with_interrupt(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            # Simula interrupção após processar 1 linha
            raise KeyboardInterrupt("Simulando interrupção")
        return {
            "data": {"campo1": f"new{call_count}", "campo2": f"new_{call_count}"},
            "usage": {},
        }

    # Primeira execução: vai processar 1 linha e travar na segunda
    with patch("dataframeit.core.call_langchain", side_effect=mock_llm_with_interrupt):
        with patch("dataframeit.core.validate_provider_dependencies"):
            try:
                dataframeit(
                    df,
                    questions=SimpleModel,
                    prompt="Teste {texto}",
                    reprocess_columns=["campo1"],
                )
            except KeyboardInterrupt:
                pass

    # Após interrupção:
    # - Linha 1: campo1 atualizado para "new1", campo2 manteve "old_a" (reprocess só campo1)
    # - Linhas 2 e 3: não foram processadas ainda
    assert df["campo1"].iloc[0] == "new1"
    assert df["campo2"].iloc[0] == "old_a"  # Manteve original
    assert df["campo1"].iloc[1] == "old2"  # Não foi processada
    assert df["campo1"].iloc[2] == "old3"  # Não foi processada

    # Segunda execução: continuar de onde parou com reprocess_columns
    call_count = 0

    def mock_llm_normal(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return {
            "data": {"campo1": f"resumed{call_count}", "campo2": f"resumed_{call_count}"},
            "usage": {},
        }

    with patch("dataframeit.core.call_langchain", side_effect=mock_llm_normal):
        with patch("dataframeit.core.validate_provider_dependencies"):
            result = dataframeit(
                df,
                questions=SimpleModel,
                prompt="Teste {texto}",
                reprocess_columns=["campo1"],  # Continua o reprocessamento
            )

    # Deve ter processado as 3 linhas (todas ainda estavam como "processed")
    assert call_count == 3
