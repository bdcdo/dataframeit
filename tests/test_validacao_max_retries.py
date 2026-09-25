"""Testes de validação do parâmetro max_retries de dataframeit()."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel

from dataframeit.core import dataframeit


class ModeloSimples(BaseModel):
    campo1: str


@pytest.mark.parametrize("valor_invalido", [0, -1, 2.5, "3", True])
def test_max_retries_invalido_levanta_value_error(valor_invalido):
    df = pd.DataFrame({"texto": ["a", "b"]})

    with (
        patch("dataframeit.core.call_langchain") as llm_falso,
        patch("dataframeit.core.validate_provider_dependencies"),
        pytest.raises(ValueError, match="max_retries deve ser int >= 1"),
    ):
        dataframeit(
            df,
            questions=ModeloSimples,
            prompt="Teste {texto}",
            max_retries=valor_invalido,
        )

    llm_falso.assert_not_called()


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("max_retries", [1, 2, np.int64(3)])
def test_max_retries_define_o_total_de_tentativas(max_retries):
    """Com o LLM falhando sempre por erro transitório, cada linha é tentada max_retries vezes."""
    df = pd.DataFrame({"texto": ["a"]})
    llm_falso = MagicMock()
    llm_falso.with_structured_output.return_value.invoke.side_effect = ConnectionError("rede")

    with (
        patch("dataframeit.llm._create_langchain_llm", return_value=llm_falso),
        patch("dataframeit.core.validate_provider_dependencies"),
    ):
        resultado = dataframeit(
            df,
            questions=ModeloSimples,
            prompt="Teste {texto}",
            max_retries=max_retries,
            base_delay=0.0,
            max_delay=0.0,
        )

    assert llm_falso.with_structured_output.return_value.invoke.call_count == int(max_retries)
    assert resultado["_dataframeit_status"].tolist() == ["error"]
