"""Testes para batch_size + checkpoint_path (issue #92)."""

import pandas as pd
import pytest
from pydantic import BaseModel
from unittest.mock import patch

from dataframeit.core import dataframeit, _save_checkpoint


class SimpleModel(BaseModel):
    campo1: str


def _mock_llm_factory():
    """Retorna (call_count_list, mock_fn). Cada chamada retorna payload válido."""
    counter = [0]

    def mock_llm(*args, **kwargs):
        counter[0] += 1
        return {
            "data": {"campo1": f"v{counter[0]}"},
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }

    return counter, mock_llm


def test_checkpoint_fires_on_multiples_sequential(tmp_path):
    """Sequencial: callback disparado em 2 e 4 (não em 5 — sem chamada final)."""
    df = pd.DataFrame({"texto": ["a", "b", "c", "d", "e"]})
    ckpt = tmp_path / "ckpt.csv"

    observed_counts = []

    def spy(df_arg, path_arg):
        observed_counts.append((int((df_arg["_dataframeit_status"] == "processed").sum()), str(path_arg)))

    _, mock_llm = _mock_llm_factory()

    with patch("dataframeit.core._save_checkpoint", side_effect=spy), \
         patch("dataframeit.core.call_langchain", side_effect=mock_llm), \
         patch("dataframeit.core.validate_provider_dependencies"):
        dataframeit(
            df,
            questions=SimpleModel,
            prompt="Teste {texto}",
            batch_size=2,
            checkpoint_path=ckpt,
        )

    processed_counts = [c for c, _ in observed_counts]
    assert processed_counts == [2, 4]
    assert all(p == str(ckpt) for _, p in observed_counts)


def test_checkpoint_fires_on_multiples_parallel(tmp_path):
    """Paralelo: contador monotônico mesmo com ordem fora — 3 chamadas em 3,6,9."""
    df = pd.DataFrame({"texto": [f"linha{i}" for i in range(10)]})
    ckpt = tmp_path / "ckpt.csv"

    observed_counts = []

    def spy(df_arg, path_arg):
        observed_counts.append(int((df_arg["_dataframeit_status"] == "processed").sum()))

    _, mock_llm = _mock_llm_factory()

    with patch("dataframeit.core._save_checkpoint", side_effect=spy), \
         patch("dataframeit.core.call_langchain", side_effect=mock_llm), \
         patch("dataframeit.core.validate_provider_dependencies"):
        dataframeit(
            df,
            questions=SimpleModel,
            prompt="Teste {texto}",
            parallel_requests=4,
            batch_size=3,
            checkpoint_path=ckpt,
        )

    assert len(observed_counts) == 3
    assert observed_counts == sorted(observed_counts)
    assert observed_counts[-1] >= 9


def test_no_checkpoint_when_params_none(tmp_path):
    """Sem batch_size/checkpoint_path: nenhum save é disparado."""
    df = pd.DataFrame({"texto": ["a", "b", "c"]})
    _, mock_llm = _mock_llm_factory()

    with patch("dataframeit.core._save_checkpoint") as mock_save, \
         patch("dataframeit.core.call_langchain", side_effect=mock_llm), \
         patch("dataframeit.core.validate_provider_dependencies"):
        dataframeit(df, questions=SimpleModel, prompt="Teste {texto}")

    mock_save.assert_not_called()


def test_validation_batch_size_without_path():
    df = pd.DataFrame({"texto": ["a"]})
    with patch("dataframeit.core.validate_provider_dependencies"):
        with pytest.raises(ValueError, match="devem ser usados juntos"):
            dataframeit(df, questions=SimpleModel, prompt="Teste {texto}", batch_size=10)


def test_validation_path_without_batch_size(tmp_path):
    df = pd.DataFrame({"texto": ["a"]})
    with patch("dataframeit.core.validate_provider_dependencies"):
        with pytest.raises(ValueError, match="devem ser usados juntos"):
            dataframeit(
                df, questions=SimpleModel, prompt="Teste {texto}",
                checkpoint_path=tmp_path / "x.csv",
            )


@pytest.mark.parametrize("bad_value", [0, -1, 1.5, "10"])
def test_validation_invalid_batch_size(bad_value, tmp_path):
    df = pd.DataFrame({"texto": ["a"]})
    with patch("dataframeit.core.validate_provider_dependencies"):
        with pytest.raises(ValueError, match="batch_size deve ser int"):
            dataframeit(
                df, questions=SimpleModel, prompt="Teste {texto}",
                batch_size=bad_value, checkpoint_path=tmp_path / "x.csv",
            )


def test_unsupported_extension_rejected_early(tmp_path):
    """Extensão inválida falha na validação antes de processar linhas."""
    df = pd.DataFrame({"texto": ["a", "b"]})
    with patch("dataframeit.core.call_langchain") as mock_llm, \
         patch("dataframeit.core.validate_provider_dependencies"):
        with pytest.raises(ValueError, match="Extensão"):
            dataframeit(
                df, questions=SimpleModel, prompt="Teste {texto}",
                batch_size=1, checkpoint_path=tmp_path / "x.txt",
            )
        mock_llm.assert_not_called()


def test_resume_after_simulated_crash(tmp_path):
    """Simula crash após 1º checkpoint; recarrega + resume deve processar só o resto."""
    df = pd.DataFrame({"texto": [f"linha{i}" for i in range(10)]})
    ckpt = tmp_path / "ckpt.csv"

    total_calls = [0]

    def mock_llm(*args, **kwargs):
        total_calls[0] += 1
        # Após 4 chamadas (que garante 2 checkpoints com batch_size=2), crash.
        if total_calls[0] > 4:
            raise SystemExit("simulated kill")
        return {
            "data": {"campo1": f"v{total_calls[0]}"},
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }

    with patch("dataframeit.core.call_langchain", side_effect=mock_llm), \
         patch("dataframeit.core.validate_provider_dependencies"):
        with pytest.raises(SystemExit):
            dataframeit(
                df, questions=SimpleModel, prompt="Teste {texto}",
                batch_size=2, checkpoint_path=ckpt,
            )

    assert ckpt.exists(), "1º checkpoint deve estar persistido após o crash"
    loaded = pd.read_csv(ckpt)
    processed_at_crash = int((loaded["_dataframeit_status"] == "processed").sum())
    assert processed_at_crash >= 2

    calls_before_resume = total_calls[0]

    def mock_llm_ok(*args, **kwargs):
        total_calls[0] += 1
        return {
            "data": {"campo1": f"v{total_calls[0]}"},
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }

    with patch("dataframeit.core.call_langchain", side_effect=mock_llm_ok), \
         patch("dataframeit.core.validate_provider_dependencies"):
        final = dataframeit(
            loaded, questions=SimpleModel, prompt="Teste {texto}",
            resume=True, batch_size=2, checkpoint_path=ckpt,
        )

    resume_calls = total_calls[0] - calls_before_resume
    assert resume_calls == 10 - processed_at_crash
    assert final["campo1"].notna().all()


def test_save_checkpoint_is_atomic(tmp_path):
    """Após save bem-sucedido, arquivo .tmp não existe."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    path = tmp_path / "out.csv"
    _save_checkpoint(df, path)
    assert path.exists()
    assert not path.with_name(path.name + ".tmp").exists()


def test_save_checkpoint_csv_roundtrip(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    path = tmp_path / "out.csv"
    _save_checkpoint(df, path)
    loaded = pd.read_csv(path)
    assert loaded["a"].tolist() == [1, 2]
    assert loaded["b"].tolist() == ["x", "y"]


def test_save_checkpoint_rejects_unsupported_extension(tmp_path):
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match="Extensão"):
        _save_checkpoint(df, tmp_path / "out.json")
