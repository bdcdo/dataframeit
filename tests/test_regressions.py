import warnings
import pandas as pd

from pandas.errors import SettingWithCopyWarning

from dataframeit.core import _setup_columns
from dataframeit import llm as llm_module


def test_setup_columns_no_settingwithcopywarning_on_copy():
    # DataFrame base
    df = pd.DataFrame({
        "texto": ["a", "b", "c"],
        "x": [1, 2, 3],
    })

    # Criar um slice e então garantir cópia (como o pipeline faz)
    df_slice = df.iloc[:2]
    df_copy = df_slice.copy()

    # Não deve haver SettingWithCopyWarning ao configurar colunas em uma cópia
    with warnings.catch_warnings():
        warnings.simplefilter("error", SettingWithCopyWarning)
        _setup_columns(df_copy, expected_columns=["campo1", "campo2"], status_column=None, resume=False, track_tokens=False)


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
