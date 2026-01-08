"""Testes para a funcionalidade de requisições paralelas."""

import warnings
import time
import pandas as pd
import pytest
from pydantic import BaseModel
from unittest.mock import patch, MagicMock

from dataframeit.core import dataframeit, _process_rows_parallel
from dataframeit.errors import is_rate_limit_error


class SimpleModel(BaseModel):
    campo1: str
    campo2: str


def test_parallel_requests_parameter_exists():
    """Testa que o parâmetro parallel_requests existe e tem default=1."""
    import inspect
    sig = inspect.signature(dataframeit)
    param = sig.parameters.get('parallel_requests')
    assert param is not None
    assert param.default == 1


def test_parallel_requests_1_uses_sequential():
    """Testa que parallel_requests=1 usa processamento sequencial."""
    df = pd.DataFrame({"texto": ["a"]})

    mock_result = {
        "data": {"campo1": "v1", "campo2": "v2"},
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }

    with patch("dataframeit.core._process_rows") as mock_seq:
        with patch("dataframeit.core._process_rows_parallel") as mock_par:
            mock_seq.return_value = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            with patch("dataframeit.core.validate_provider_dependencies"):
                dataframeit(
                    df,
                    questions=SimpleModel,
                    prompt="Teste {texto}",
                    parallel_requests=1,  # Default
                )

                # Deve usar processamento sequencial
                mock_seq.assert_called_once()
                mock_par.assert_not_called()


def test_parallel_requests_gt1_uses_parallel():
    """Testa que parallel_requests > 1 usa processamento paralelo."""
    df = pd.DataFrame({"texto": ["a"]})

    with patch("dataframeit.core._process_rows") as mock_seq:
        with patch("dataframeit.core._process_rows_parallel") as mock_par:
            mock_par.return_value = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            with patch("dataframeit.core.validate_provider_dependencies"):
                dataframeit(
                    df,
                    questions=SimpleModel,
                    prompt="Teste {texto}",
                    parallel_requests=5,
                )

                # Deve usar processamento paralelo
                mock_par.assert_called_once()
                mock_seq.assert_not_called()


def test_parallel_processes_all_rows():
    """Testa que processamento paralelo processa todas as linhas."""
    df = pd.DataFrame({"texto": ["a", "b", "c", "d", "e"]})

    call_count = 0

    def mock_llm(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return {
            "data": {"campo1": f"v{call_count}", "campo2": f"x{call_count}"},
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        }

    with patch("dataframeit.core.call_langchain", side_effect=mock_llm):
        with patch("dataframeit.core.validate_provider_dependencies"):
            result = dataframeit(
                df,
                questions=SimpleModel,
                prompt="Teste {texto}",
                parallel_requests=3,
            )

            # Todas as 5 linhas devem ter sido processadas
            assert call_count == 5
            # Verificar que todas as linhas têm valores (não None)
            assert result["campo1"].notna().all()
            assert result["campo2"].notna().all()


def test_parallel_tracks_tokens():
    """Testa que processamento paralelo rastreia tokens corretamente."""
    df = pd.DataFrame({"texto": ["a", "b", "c"]})

    def mock_llm(*args, **kwargs):
        return {
            "data": {"campo1": "v", "campo2": "x"},
            "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        }

    with patch("dataframeit.core.call_langchain", side_effect=mock_llm):
        with patch("dataframeit.core.validate_provider_dependencies"):
            result = dataframeit(
                df,
                questions=SimpleModel,
                prompt="Teste {texto}",
                parallel_requests=2,
                track_tokens=True,
            )

            # Verificar colunas de tokens
            assert "_input_tokens" in result.columns
            assert "_output_tokens" in result.columns
            assert "_total_tokens" in result.columns

            # Cada linha deve ter os tokens registrados
            assert result["_input_tokens"].tolist() == [100, 100, 100]
            assert result["_output_tokens"].tolist() == [50, 50, 50]


def test_is_rate_limit_error_detects_429():
    """Testa que is_rate_limit_error detecta erros de rate limit."""
    # Erro 429
    error_429 = Exception("Error 429: Too many requests")
    assert is_rate_limit_error(error_429) is True

    # Rate limit explícito
    class RateLimitError(Exception):
        pass
    rate_error = RateLimitError("Rate limit exceeded")
    assert is_rate_limit_error(rate_error) is True

    # Resource exhausted (Google)
    resource_error = Exception("ResourceExhausted: Quota exceeded")
    assert is_rate_limit_error(resource_error) is True

    # Erro normal (não é rate limit)
    normal_error = ValueError("Invalid argument")
    assert is_rate_limit_error(normal_error) is False


def test_parallel_handles_errors_gracefully():
    """Testa que processamento paralelo lida com erros sem quebrar."""
    df = pd.DataFrame({"texto": ["a", "b", "c"]})

    call_count = 0

    def mock_llm_with_error(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise ValueError("Erro no processamento")
        return {
            "data": {"campo1": f"v{call_count}", "campo2": f"x{call_count}"},
            "usage": {},
        }

    with patch("dataframeit.core.call_langchain", side_effect=mock_llm_with_error):
        with patch("dataframeit.core.validate_provider_dependencies"):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                result = dataframeit(
                    df,
                    questions=SimpleModel,
                    prompt="Teste {texto}",
                    parallel_requests=2,
                )

                # Linhas 1 e 3 processadas, linha 2 com erro
                statuses = result["_dataframeit_status"].tolist()
                assert statuses.count("processed") == 2
                assert statuses.count("error") == 1


def test_parallel_respects_resume():
    """Testa que processamento paralelo respeita resume=True."""
    df = pd.DataFrame({
        "texto": ["a", "b", "c"],
        "campo1": ["old1", None, None],
        "campo2": ["old_a", None, None],
        "_dataframeit_status": ["processed", None, None],
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
                parallel_requests=2,
                resume=True,
            )

            # Apenas 2 linhas não processadas devem ser chamadas
            assert call_count == 2
            # Primeira linha mantém valores antigos
            assert result["campo1"].iloc[0] == "old1"


def test_parallel_with_reprocess_columns():
    """Testa que processamento paralelo funciona com reprocess_columns."""
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
                parallel_requests=2,
                reprocess_columns=["campo1"],
            )

            # campo1 deve ter sido atualizado
            assert result["campo1"].tolist() == ["novo_valor", "novo_valor"]
            # campo2 deve manter os valores originais
            assert result["campo2"].tolist() == ["original_a", "original_b"]
