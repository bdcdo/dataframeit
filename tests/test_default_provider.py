"""Provider e modelo usados quando o usuário não escolhe."""

from unittest.mock import patch

import pandas as pd
import pytest
from pydantic import BaseModel

from dataframeit.core import DEFAULT_MODELS


class _Modelo(BaseModel):
    resumo: str


def _resposta(*args, **kwargs):
    return {"data": {"resumo": "ok"}, "usage": None}


def _configs_enviadas(**kwargs):
    from dataframeit.core import dataframeit

    df = pd.DataFrame({"texto": ["a", "b"]})
    with patch("dataframeit.core.call_langchain", side_effect=_resposta) as chamada, \
            patch("dataframeit.core.validate_provider_dependencies") as validacao:
        dataframeit(df, _Modelo, "resuma: {texto}", track_tokens=False, **kwargs)
    return validacao, [c.args[3] for c in chamada.call_args_list]


def test_sem_provider_nem_modelo_usa_gpt_6_luna_na_openai():
    validacao, configs = _configs_enviadas()

    validacao.assert_called_once_with("openai")
    assert {(c.provider, c.model) for c in configs} == {("openai", "gpt-6-luna")}
    assert all(c.model_kwargs == {} for c in configs)


def test_tabela_de_modelos_padrao():
    assert DEFAULT_MODELS == {
        "openai": "gpt-6-luna",
        "google_genai": "gemini-3.8-flash",
        "anthropic": "claude-sonnet-5",
        "groq": "openai/gpt-oss-120b",
    }


@pytest.mark.parametrize("provider", sorted(DEFAULT_MODELS))
def test_provider_sem_modelo_usa_o_modelo_do_proprio_provider(provider):
    _, configs = _configs_enviadas(provider=provider)

    assert {(c.provider, c.model) for c in configs} == {(provider, DEFAULT_MODELS[provider])}


def test_modelo_explicito_prevalece_sobre_o_default():
    _, configs = _configs_enviadas(provider="google_genai", model="gemini-x")

    assert {c.model for c in configs} == {"gemini-x"}


def test_provider_sem_modelo_padrao_exige_model():
    from dataframeit.core import dataframeit

    df = pd.DataFrame({"texto": ["a"]})
    with pytest.raises(ValueError, match="provider='mistralai' não tem modelo padrão"):
        dataframeit(df, _Modelo, "resuma: {texto}", provider="mistralai")


def test_claude_code_sem_modelo_deixa_o_runtime_escolher():
    from dataframeit.core import dataframeit

    df = pd.DataFrame({"texto": ["a"]})
    with patch("dataframeit.claude_code.call_claude_code", side_effect=_resposta) as chamada, \
            patch("dataframeit.core.validate_provider_dependencies"):
        dataframeit(df, _Modelo, "resuma: {texto}", provider="claude_code", track_tokens=False)

    assert chamada.call_args.args[3].model is None


def test_codex_sem_modelo_deixa_o_runtime_escolher():
    from contextlib import contextmanager

    from dataframeit.core import dataframeit

    configs = []

    @contextmanager
    def backend_falso(config, pydantic_model, user_prompt):
        configs.append(config)
        yield type("Backend", (), {"invoke": staticmethod(_resposta)})()

    df = pd.DataFrame({"texto": ["a"]})
    with patch("dataframeit.codex.open_codex_backend", side_effect=backend_falso), \
            patch("dataframeit.core.validate_provider_dependencies"):
        dataframeit(df, _Modelo, "resuma: {texto}", provider="codex", track_tokens=False)

    assert [c.model for c in configs] == [None]


def test_dataframe_vazio_nao_exige_modelo_padrao():
    from dataframeit.core import dataframeit

    df = pd.DataFrame({"texto": pd.Series([], dtype=str)})
    resultado = dataframeit(df, _Modelo, "resuma: {texto}", provider="mistralai")

    assert len(resultado) == 0
