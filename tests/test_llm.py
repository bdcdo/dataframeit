"""Testes para dataframeit.llm."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field


class SampleModel(BaseModel):
    campo: str = Field(description="Campo de teste")


class TestBuildPrompt:
    """Substituicao do placeholder {texto} no template."""

    def test_substitui_placeholder(self):
        from dataframeit.llm import build_prompt

        result = build_prompt("Analise: {texto}", "ola mundo")
        assert result == "Analise: ola mundo"

    def test_multiplas_ocorrencias(self):
        from dataframeit.llm import build_prompt

        result = build_prompt("{texto} e {texto}", "X")
        assert result == "X e X"

    def test_template_sem_placeholder(self):
        from dataframeit.llm import build_prompt

        result = build_prompt("template fixo", "ignorado")
        assert result == "template fixo"


class TestSearchGroupConfigDefaults:
    def test_defaults_none(self):
        from dataframeit.llm import SearchGroupConfig

        cfg = SearchGroupConfig(fields=["a", "b"])
        assert cfg.fields == ["a", "b"]
        assert cfg.prompt is None
        assert cfg.max_results is None
        assert cfg.search_depth is None


class TestSearchConfigDefaults:
    def test_defaults(self):
        from dataframeit.llm import SearchConfig

        cfg = SearchConfig()
        assert cfg.enabled is False
        assert cfg.provider == "tavily"
        assert cfg.per_field is False
        assert cfg.max_results == 5
        assert cfg.search_depth == "basic"
        assert cfg.groups is None


class TestLLMConfigDefaults:
    def test_model_kwargs_factory(self):
        """model_kwargs deve ser dict independente entre instancias."""
        from dataframeit.llm import LLMConfig

        cfg1 = LLMConfig(
            model="m", provider="p", api_key=None,
            max_retries=1, base_delay=1.0, max_delay=2.0, rate_limit_delay=0.0,
        )
        cfg2 = LLMConfig(
            model="m", provider="p", api_key=None,
            max_retries=1, base_delay=1.0, max_delay=2.0, rate_limit_delay=0.0,
        )
        cfg1.model_kwargs["x"] = 1
        assert cfg2.model_kwargs == {}

    def test_search_config_default_none(self):
        from dataframeit.llm import LLMConfig

        cfg = LLMConfig(
            model="m", provider="p", api_key=None,
            max_retries=1, base_delay=1.0, max_delay=2.0, rate_limit_delay=0.0,
        )
        assert cfg.search_config is None


def _make_config():
    from dataframeit.llm import LLMConfig

    return LLMConfig(
        model="gemini-test", provider="google_genai", api_key="key",
        max_retries=1, base_delay=0.0, max_delay=0.0, rate_limit_delay=0.0,
    )


def _patch_call_langchain(structured_llm):
    """Patches _create_langchain_llm para retornar um LLM cujo with_structured_output
    devolve structured_llm."""
    base_llm = MagicMock()
    base_llm.with_structured_output.return_value = structured_llm
    return patch("dataframeit.llm._create_langchain_llm", return_value=base_llm)


class TestCallLangchain:
    def test_sucesso_retorna_data_e_usage(self):
        from dataframeit.llm import call_langchain

        parsed = SampleModel(campo="valor")
        raw = SimpleNamespace(usage_metadata={
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "output_token_details": {"reasoning": 2},
        })
        structured_llm = MagicMock()
        structured_llm.invoke.return_value = {"parsed": parsed, "raw": raw, "parsing_error": None}

        with _patch_call_langchain(structured_llm):
            result = call_langchain("texto", SampleModel, "Use: {texto}", _make_config())

        assert result["data"] == {"campo": "valor"}
        assert result["usage"] == {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "reasoning_tokens": 2,
        }
        # retry_with_backoff adiciona _retry_info ao dict de resultado
        assert "_retry_info" in result
        # E o prompt enviado tem o placeholder substituido
        sent_prompt = structured_llm.invoke.call_args[0][0]
        assert sent_prompt == "Use: texto"

    def test_parsing_error_levanta_value_error(self):
        from dataframeit.llm import call_langchain

        structured_llm = MagicMock()
        structured_llm.invoke.return_value = {
            "parsed": None,
            "raw": None,
            "parsing_error": "schema invalido",
        }

        with _patch_call_langchain(structured_llm), pytest.raises(ValueError, match="parsing"):
            call_langchain("t", SampleModel, "{texto}", _make_config())

    def test_parsed_none_sem_parsing_error_levanta_value_error(self):
        from dataframeit.llm import call_langchain

        structured_llm = MagicMock()
        structured_llm.invoke.return_value = {"parsed": None, "raw": None, "parsing_error": None}

        with _patch_call_langchain(structured_llm), pytest.raises(ValueError, match="None"):
            call_langchain("t", SampleModel, "{texto}", _make_config())

    def test_usage_metadata_como_objeto(self):
        """usage_metadata pode vir como objeto (nao dict) em algumas integracoes."""
        from dataframeit.llm import call_langchain

        meta = SimpleNamespace(
            input_tokens=3, output_tokens=4, total_tokens=7,
            output_token_details=SimpleNamespace(reasoning=2),
        )
        raw = SimpleNamespace(usage_metadata=meta)
        structured_llm = MagicMock()
        structured_llm.invoke.return_value = {
            "parsed": SampleModel(campo="x"),
            "raw": raw,
            "parsing_error": None,
        }

        with _patch_call_langchain(structured_llm):
            result = call_langchain("t", SampleModel, "{texto}", _make_config())

        assert result["usage"]["input_tokens"] == 3
        assert result["usage"]["output_tokens"] == 4
        assert result["usage"]["total_tokens"] == 7
        assert result["usage"]["reasoning_tokens"] == 2

    def test_output_token_details_como_objeto_com_meta_dict(self):
        """meta dict + output_token_details como objeto: getattr fallback."""
        from dataframeit.llm import call_langchain

        raw = SimpleNamespace(usage_metadata={
            "input_tokens": 3, "output_tokens": 4, "total_tokens": 7,
            "output_token_details": SimpleNamespace(reasoning=7),
        })
        structured_llm = MagicMock()
        structured_llm.invoke.return_value = {
            "parsed": SampleModel(campo="x"),
            "raw": raw,
            "parsing_error": None,
        }

        with _patch_call_langchain(structured_llm):
            result = call_langchain("t", SampleModel, "{texto}", _make_config())

        assert result["usage"]["reasoning_tokens"] == 7

    def test_sem_output_token_details_reasoning_zero(self):
        from dataframeit.llm import call_langchain

        raw = SimpleNamespace(usage_metadata={
            "input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
        })
        structured_llm = MagicMock()
        structured_llm.invoke.return_value = {
            "parsed": SampleModel(campo="x"),
            "raw": raw,
            "parsing_error": None,
        }

        with _patch_call_langchain(structured_llm):
            result = call_langchain("t", SampleModel, "{texto}", _make_config())

        assert result["usage"]["reasoning_tokens"] == 0

    def test_raw_sem_usage_metadata_retorna_usage_none(self):
        from dataframeit.llm import call_langchain

        raw = SimpleNamespace()  # sem atributo usage_metadata
        structured_llm = MagicMock()
        structured_llm.invoke.return_value = {
            "parsed": SampleModel(campo="x"),
            "raw": raw,
            "parsing_error": None,
        }

        with _patch_call_langchain(structured_llm):
            result = call_langchain("t", SampleModel, "{texto}", _make_config())

        assert result["usage"] is None

    def test_usage_metadata_vazio_retorna_usage_none(self):
        """usage_metadata=None (nao todo provider o expoe) -> usage None."""
        from dataframeit.llm import call_langchain

        raw = SimpleNamespace(usage_metadata=None)
        structured_llm = MagicMock()
        structured_llm.invoke.return_value = {
            "parsed": SampleModel(campo="x"),
            "raw": raw,
            "parsing_error": None,
        }

        with _patch_call_langchain(structured_llm):
            result = call_langchain("t", SampleModel, "{texto}", _make_config())

        assert result["usage"] is None
