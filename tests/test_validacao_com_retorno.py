"""Resposta recusada pela validação: a tentativa seguinte leva a resposta e o erro ao modelo."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, ValidationError, model_validator

from dataframeit.llm import LLMConfig, call_langchain


class ComEvidencia(BaseModel):
    aplicou: bool
    trecho: str | None = None

    @model_validator(mode="after")
    def exigir_trecho(self):
        if self.aplicou and not self.trecho:
            raise ValueError("Aplicar exige trecho.")
        return self


def _config(max_retries=3):
    return LLMConfig(model="m", provider="openai", api_key="k", max_retries=max_retries,
                     base_delay=0.0, max_delay=0.0, rate_limit_delay=0.0)


def _raw(conteudo, tokens=(10, 5), tool_calls=None):
    return SimpleNamespace(
        content=conteudo, tool_calls=tool_calls or [],
        usage_metadata={"input_tokens": tokens[0], "output_tokens": tokens[1], "total_tokens": sum(tokens)},
    )


def _falha(conteudo, erro="falha do parser", **kwargs):
    return {"parsed": None, "raw": _raw(conteudo, **kwargs), "parsing_error": erro}


def _sucesso(modelo, tokens=(20, 7)):
    return {"parsed": modelo, "raw": _raw("{}", tokens=tokens), "parsing_error": None}


def _chamar(respostas, max_retries=3):
    structured = MagicMock()
    structured.invoke.side_effect = respostas
    base = MagicMock()
    base.with_structured_output.return_value = structured
    with patch("dataframeit.llm._create_langchain_llm", return_value=base):
        resultado = call_langchain("TEXTO", ComEvidencia, "Leia: {texto}", _config(max_retries))
    return resultado, structured


def _segunda_chamada(structured):
    return structured.invoke.call_args_list[1].args[0]


class TestNovaTentativaComErro:
    def test_erro_e_resposta_voltam_ao_modelo(self):
        invalido = json.dumps({"aplicou": True, "trecho": None})
        with pytest.warns(UserWarning):
            resultado, structured = _chamar([_falha(invalido), _sucesso(ComEvidencia(aplicou=True, trecho="x"))])
        assert resultado["data"] == {"aplicou": True, "trecho": "x"}
        assert structured.invoke.call_args_list[0].args[0] == "Leia: TEXTO"
        segunda = _segunda_chamada(structured)
        assert [papel for papel, _ in segunda] == ["human", "ai", "human"]
        assert segunda[0][1] == "Leia: TEXTO"
        assert segunda[1][1] == invalido
        # O erro chega por campo, validado de novo sobre a resposta bruta.
        assert "Aplicar exige trecho." in segunda[2][1]

    def test_validation_error_do_parser_e_usado_direto(self):
        try:
            ComEvidencia.model_validate({"aplicou": "talvez"})
        except ValidationError as erro:
            do_parser = erro
        with pytest.warns(UserWarning):
            _, structured = _chamar([_falha("{}", erro=do_parser), _sucesso(ComEvidencia(aplicou=False))])
        assert "- aplicou:" in _segunda_chamada(structured)[2][1]

    def test_uso_soma_as_tentativas_recusadas(self):
        invalido = json.dumps({"aplicou": True})
        with pytest.warns(UserWarning):
            resultado, _ = _chamar([_falha(invalido, tokens=(10, 5)), _falha(invalido, tokens=(11, 6)),
                                    _sucesso(ComEvidencia(aplicou=False), tokens=(20, 7))])
        assert resultado["usage"]["input_tokens"] == 41
        assert resultado["usage"]["output_tokens"] == 18
        assert resultado["usage"]["total_tokens"] == 59

    def test_resposta_por_tool_call(self):
        args = {"aplicou": True}
        with pytest.warns(UserWarning):
            _, structured = _chamar([_falha("", tool_calls=[{"name": "ComEvidencia", "args": args}]),
                                     _sucesso(ComEvidencia(aplicou=False))])
        assert _segunda_chamada(structured)[1] == ("ai", json.dumps(args))
        assert "Aplicar exige trecho." in _segunda_chamada(structured)[2][1]

    def test_resposta_em_blocos(self):
        bruto = json.dumps({"aplicou": True})
        blocos = [{"type": "text", "text": bruto[:5]}, {"type": "text", "text": bruto[5:]}]
        with pytest.warns(UserWarning):
            _, structured = _chamar([_falha(blocos), _sucesso(ComEvidencia(aplicou=False))])
        assert _segunda_chamada(structured)[1] == ("ai", bruto)

    def test_json_malformado_tambem_volta_ao_modelo(self):
        with pytest.warns(UserWarning):
            resultado, structured = _chamar([_falha("{quebrado"), _sucesso(ComEvidencia(aplicou=False))])
        assert resultado["data"]["aplicou"] is False
        segunda = _segunda_chamada(structured)
        assert segunda[1] == ("ai", "{quebrado")
        assert "falha do parser" in segunda[2][1]

    def test_numero_na_resposta_nao_impede_o_retry(self):
        # "404" isolado na mensagem casaria com o padrão de erro HTTP definitivo.
        invalido = json.dumps({"aplicou": True, "processo": "Rcl 404"})
        with pytest.warns(UserWarning):
            resultado, structured = _chamar([_falha(invalido, erro="input 404"), _sucesso(ComEvidencia(aplicou=False))])
        assert structured.invoke.call_count == 2
        assert resultado["data"]["aplicou"] is False

    def test_esgota_as_tentativas(self):
        invalido = json.dumps({"aplicou": True})
        with pytest.warns(UserWarning), pytest.raises(ValueError, match="parsing"):
            _chamar([_falha(invalido)] * 2, max_retries=2)

    def test_sucesso_de_primeira_nao_manda_historico(self):
        resultado, structured = _chamar([_sucesso(ComEvidencia(aplicou=False))])
        assert structured.invoke.call_count == 1
        assert structured.invoke.call_args.args[0] == "Leia: TEXTO"
        assert resultado["usage"]["input_tokens"] == 20
