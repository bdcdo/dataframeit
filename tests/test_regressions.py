import warnings
import pandas as pd

from pandas.errors import SettingWithCopyWarning

from dataframeit.core.managers import ColumnManager
from dataframeit.core import base as base_module


def test_setup_columns_no_settingwithcopywarning_on_copy():
    # DataFrame base
    df = pd.DataFrame({
        "texto": ["a", "b", "c"],
        "x": [1, 2, 3],
    })

    # Criar um slice e então garantir cópia (como o pipeline faz)
    df_slice = df.iloc[:2]
    df_copy = df_slice.copy()

    cm = ColumnManager(expected_columns=["campo1", "campo2"], status_column=None, error_column="_error")

    # Não deve haver SettingWithCopyWarning ao configurar colunas em uma cópia
    with warnings.catch_warnings():
        warnings.simplefilter("error", SettingWithCopyWarning)
        cm.setup_columns(df_copy)


def test_promptbuilder_format_prompt_does_not_interpret_other_placeholders(monkeypatch):
    # Monkeypatch para evitar dependência real do LangChain e simular {format} nas instruções
    class FakePydanticOutputParser:
        def __init__(self, pydantic_object):
            self._obj = pydantic_object

        def get_format_instructions(self):
            # Simula instruções que incluem um placeholder {format} (causa do KeyError no passado)
            return "Use {format} ao responder."

    # Substitui a classe utilizada internamente pelo PromptBuilder
    monkeypatch.setattr(base_module, "PydanticOutputParser", FakePydanticOutputParser, raising=True)

    # Instancia PromptBuilder sem depender do LangChain real
    pb = base_module.PromptBuilder(perguntas=object(), placeholder="documento")

    user_prompt = "Responda às perguntas sobre: {documento}"
    template = pb.build_prompt_template(user_prompt)

    # Deve substituir apenas {documento}, preservando {format} nas instruções, sem levantar KeyError
    formatted = pb.format_prompt(template, "TEXTO_DE_TESTE")

    assert "TEXTO_DE_TESTE" in formatted
    assert "{format}" in formatted
