from contextlib import nullcontext

import pandas as pd
import pytest

from dataframeit import llm as llm_module
from dataframeit.core import _setup_columns


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("copy_input", [False, True])
def test_setup_columns_mutates_slice_without_warnings(copy_input):
    df = pd.DataFrame(
        {
            "texto": ["a", "b", "c"],
            "x": [1, 2, 3],
        }
    )
    df_copy = df.iloc[:2]
    if copy_input:
        df_copy = df_copy.copy()

    _setup_columns(
        df_copy,
        expected_columns=["campo1", "campo2"],
        status_column=None,
        track_tokens=False,
    )

    assert list(df.columns) == ["texto", "x"]
    assert list(df_copy.columns) == [
        "texto",
        "x",
        "campo1",
        "campo2",
        "_dataframeit_status",
        "_error_details",
    ]
    generated = df_copy[["campo1", "campo2", "_dataframeit_status", "_error_details"]]
    assert generated.isna().all().all()


def test_build_prompt_replaces_placeholder():
    """Testa que build_prompt substitui corretamente o placeholder {texto}."""
    user_prompt = "Responda às perguntas sobre: {texto}"
    formatted = llm_module.build_prompt(user_prompt, "TEXTO_DE_TESTE")

    # Deve substituir {texto}
    assert "TEXTO_DE_TESTE" in formatted
    assert "{texto}" not in formatted


def test_build_prompt_preserves_other_placeholders():
    """Testa que build_prompt preserva outros placeholders."""
    user_prompt = "Analise: {texto}\nOutro: {outro}"
    formatted = llm_module.build_prompt(user_prompt, "TEXTO")

    # Deve substituir apenas {texto}
    assert "TEXTO" in formatted
    assert "{texto}" not in formatted
    assert "{outro}" in formatted  # Preserva outros placeholders


@pytest.mark.parametrize("legacy_warning", [False, True])
def test_setup_columns_uses_option_only_with_legacy_warning(monkeypatch, legacy_warning):
    calls = []

    def option_context(*args):
        calls.append(args)
        return nullcontext()

    monkeypatch.setattr(pd, "option_context", option_context)
    if legacy_warning:
        monkeypatch.setattr(pd.errors, "SettingWithCopyWarning", UserWarning, raising=False)
    else:
        monkeypatch.delattr(pd.errors, "SettingWithCopyWarning", raising=False)

    df = pd.DataFrame({"texto": ["a"]})
    _setup_columns(df, ["resposta"], None, track_tokens=False)

    assert calls == ([("mode.chained_assignment", None)] if legacy_warning else [])
    assert df["resposta"].isna().all()
