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
        _setup_columns(df_copy, expected_columns=["campo1", "campo2"], status_column=None, resume=False)


def test_promptbuilder_format_prompt_does_not_interpret_other_placeholders(monkeypatch):
    # Monkeypatch para evitar dependência real do LangChain e simular {format} nas instruções
    class FakePydanticOutputParser:
        def __init__(self, pydantic_object):
            self._obj = pydantic_object

        def get_format_instructions(self):
            # Simula instruções que incluem um placeholder {format} (causa do KeyError no passado)
            return "Use {format} ao responder."

    # Substitui a classe utilizada internamente
    monkeypatch.setattr(llm_module, "PydanticOutputParser", FakePydanticOutputParser, raising=True)

    # Testa build_prompt sem depender do LangChain real
    user_prompt = "Responda às perguntas sobre: {documento}"
    formatted = llm_module.build_prompt(object(), user_prompt, "TEXTO_DE_TESTE", "documento")

    # Deve substituir apenas {documento}, preservando {format} nas instruções, sem levantar KeyError
    assert "TEXTO_DE_TESTE" in formatted
    assert "{format}" in formatted
