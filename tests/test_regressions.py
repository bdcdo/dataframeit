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
    """Testa que build_prompt substitui corretamente o placeholder."""
    user_prompt = "Responda às perguntas sobre: {documento}"
    formatted = llm_module.build_prompt(user_prompt, "TEXTO_DE_TESTE", "documento")

    # Deve substituir apenas {documento}
    assert "TEXTO_DE_TESTE" in formatted
    assert "{documento}" not in formatted


def test_build_prompt_warns_on_format_placeholder():
    """Testa que build_prompt emite warning quando {format} está presente."""
    user_prompt = "Analise: {documento}\n{format}"

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        formatted = llm_module.build_prompt(user_prompt, "TEXTO", "documento")

        # Deve emitir DeprecationWarning
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "{format}" in str(w[0].message)
