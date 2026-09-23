"""Testes de validação do parâmetro max_retries de dataframeit()."""

from unittest.mock import patch

import pandas as pd
import pytest
from pydantic import BaseModel

from dataframeit.core import dataframeit


class ModeloSimples(BaseModel):
    campo1: str


@pytest.mark.parametrize("valor_invalido", [0, -1, 2.5, "3", True])
def test_max_retries_invalido_levanta_value_error(valor_invalido):
    df = pd.DataFrame({"texto": ["a", "b"]})

    with patch("dataframeit.core.call_langchain") as llm_falso:
        with patch("dataframeit.core.validate_provider_dependencies"):
            with pytest.raises(ValueError, match="max_retries deve ser int >= 1"):
                dataframeit(
                    df,
                    questions=ModeloSimples,
                    prompt="Teste {texto}",
                    max_retries=valor_invalido,
                )

    llm_falso.assert_not_called()


def test_max_retries_igual_a_um_faz_uma_unica_tentativa():
    df = pd.DataFrame({"texto": ["a"]})
    resposta = {"data": {"campo1": "valor"}, "usage": {}}

    with patch("dataframeit.core.call_langchain", return_value=resposta) as llm_falso:
        with patch("dataframeit.core.validate_provider_dependencies"):
            resultado = dataframeit(
                df,
                questions=ModeloSimples,
                prompt="Teste {texto}",
                max_retries=1,
            )

    assert llm_falso.call_count == 1
    assert resultado["campo1"].tolist() == ["valor"]
