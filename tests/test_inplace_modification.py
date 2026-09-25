"""O DataFrame original é modificado in-place e guarda o progresso parcial."""

import warnings
from unittest.mock import patch

import pandas as pd
import pytest
from pydantic import BaseModel, Field

from dataframeit import dataframeit


class ModeloTeste(BaseModel):
    categoria: str = Field(..., description="Categoria do texto")
    sentimento: str = Field(..., description="Sentimento: positivo, negativo ou neutro")


PROMPT = "Analise o texto e extraia as informações.\n\nTexto:\n{texto}"


def _resposta(text, *args, **kwargs):
    return {"data": {"categoria": f"c-{text}", "sentimento": "neutro"}, "usage": None}


def test_inplace_modification():
    """Uma interrupção deixa no DataFrame original as linhas já processadas."""
    df_teste = pd.DataFrame(
        {
            "texto": [
                "Este é um texto positivo sobre tecnologia",
                "Este é um texto negativo sobre política",
                "Este é um texto neutro sobre esportes",
                "Outro texto positivo sobre ciência",
                "Mais um texto negativo sobre economia",
            ]
        }
    )
    chamadas = []

    def interrompe_na_terceira(text, *args, **kwargs):
        chamadas.append(text)
        if len(chamadas) == 3:
            msg = "Simulando interrupção do usuário"
            raise KeyboardInterrupt(msg)
        return _resposta(text)

    with (
        patch("dataframeit.core.call_langchain", side_effect=interrompe_na_terceira),
        patch("dataframeit.core.validate_provider_dependencies"),
        pytest.raises(KeyboardInterrupt),
    ):
        dataframeit(df_teste, ModeloTeste, PROMPT)

    assert (df_teste["_dataframeit_status"] == "processed").sum() == 2

    with (
        patch("dataframeit.core.call_langchain", side_effect=_resposta) as retomada,
        patch("dataframeit.core.validate_provider_dependencies"),
    ):
        df_final = dataframeit(df_teste, ModeloTeste, PROMPT)

    assert retomada.call_count == 3
    assert df_final["categoria"].notna().all()


def test_no_warnings():
    """Processar um slice não emite SettingWithCopyWarning."""
    df_grande = pd.DataFrame({"id": range(10), "texto": [f"Texto número {i}" for i in range(10)]})
    df_slice = df_grande[df_grande["id"] >= 5]

    with (
        patch("dataframeit.core.call_langchain", side_effect=_resposta),
        patch("dataframeit.core.validate_provider_dependencies"),
        warnings.catch_warnings(record=True) as avisos,
    ):
        warnings.simplefilter("always")
        dataframeit(df_slice, ModeloTeste, PROMPT, max_retries=1)

    assert not [aviso for aviso in avisos if "SettingWithCopyWarning" in str(aviso.category)]
