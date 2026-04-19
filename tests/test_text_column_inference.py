"""Testes para inferência de text_column a partir de nomes comuns (#93)."""

import warnings
import pandas as pd
import pytest
from pydantic import BaseModel, Field
from unittest.mock import patch


class _Model(BaseModel):
    resumo: str = Field(description="resumo curto")


def _mock_call_langchain(*args, **kwargs):
    return {"data": {"resumo": "ok"}, "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}


def test_infers_texto_by_default():
    """DataFrame com coluna 'texto' continua funcionando sem text_column."""
    from dataframeit.core import dataframeit

    df = pd.DataFrame({"id": [1], "texto": ["foo"]})
    with patch("dataframeit.core.call_langchain", side_effect=_mock_call_langchain):
        with patch("dataframeit.core.validate_provider_dependencies"):
            result = dataframeit(df, _Model, "analise")
    assert result["resumo"].tolist() == ["ok"]


def test_infers_decisao_from_juscraper_like_df():
    """DataFrame com coluna 'decisao' (cjpg/cjsg) é detectado automaticamente."""
    from dataframeit.core import dataframeit

    df = pd.DataFrame({"cd_processo": ["001"], "decisao": ["texto longo"]})
    with patch("dataframeit.core.call_langchain", side_effect=_mock_call_langchain):
        with patch("dataframeit.core.validate_provider_dependencies"):
            result = dataframeit(df, _Model, "analise")
    assert result["resumo"].tolist() == ["ok"]


def test_infers_text_english():
    """DataFrame com coluna 'text' (inglês) é aceito."""
    from dataframeit.core import dataframeit

    df = pd.DataFrame({"id": [1], "text": ["hello"]})
    with patch("dataframeit.core.call_langchain", side_effect=_mock_call_langchain):
        with patch("dataframeit.core.validate_provider_dependencies"):
            result = dataframeit(df, _Model, "analise")
    assert result["resumo"].tolist() == ["ok"]


def test_warns_when_multiple_candidates_present():
    """Quando mais de um candidato está presente, emite UserWarning e usa o primeiro da lista."""
    from dataframeit.core import dataframeit

    # 'texto' tem precedência sobre 'content' pela ordem de TEXT_COLUMN_CANDIDATES
    df = pd.DataFrame({"texto": ["foo"], "content": ["bar"]})
    with patch("dataframeit.core.call_langchain", side_effect=_mock_call_langchain):
        with patch("dataframeit.core.validate_provider_dependencies"):
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                dataframeit(df, _Model, "analise")
    messages = [str(w.message) for w in captured if issubclass(w.category, UserWarning)]
    assert any("Múltiplas colunas candidatas" in m for m in messages)


def test_raises_when_no_candidate_matches():
    """Multi-col sem nenhum candidato bater: ValueError informativo."""
    from dataframeit.core import dataframeit

    df = pd.DataFrame({"id": [1], "payload": ["x"]})
    with patch("dataframeit.core.validate_provider_dependencies"):
        with pytest.raises(ValueError) as exc_info:
            dataframeit(df, _Model, "analise")
    msg = str(exc_info.value)
    assert "Nenhuma coluna de texto identificada" in msg
    assert "text_column=" in msg
    assert "payload" in msg


def test_single_column_df_uses_that_column():
    """DataFrame com apenas 1 coluna (sem nome canônico) usa-a como texto."""
    from dataframeit.core import dataframeit

    df = pd.DataFrame({"mensagem": ["hello"]})
    with patch("dataframeit.core.call_langchain", side_effect=_mock_call_langchain):
        with patch("dataframeit.core.validate_provider_dependencies"):
            result = dataframeit(df, _Model, "analise")
    assert result["resumo"].tolist() == ["ok"]


def test_explicit_text_column_overrides_inference():
    """text_column= explícito sempre vence a inferência."""
    from dataframeit.core import dataframeit

    df = pd.DataFrame({"texto": ["ignorado"], "meu_campo": ["usado"]})
    captured_text = []

    def capture(*args, **kwargs):
        captured_text.append(args[0])
        return _mock_call_langchain(*args, **kwargs)

    with patch("dataframeit.core.call_langchain", side_effect=capture):
        with patch("dataframeit.core.validate_provider_dependencies"):
            dataframeit(df, _Model, "analise", text_column="meu_campo")
    assert captured_text == ["usado"]
