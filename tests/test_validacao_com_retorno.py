"""Resposta recusada pela validação: a tentativa seguinte leva a resposta e o erro ao modelo."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel, ValidationError, model_validator

from dataframeit.errors import (
    ProviderRejectedOutputError,
    get_friendly_error_message,
    is_rate_limit_error,
    is_recoverable_error,
)
from dataframeit.llm import LLMConfig, _read_sdk_body, _sdk_rejected_response, call_langchain


class ComEvidencia(BaseModel):
    aplicou: bool
    trecho: str | None = None

    @model_validator(mode="after")
    def exigir_trecho(self):
        if self.aplicou and not self.trecho:
            msg = "Aplicar exige trecho."
            raise ValueError(msg)
        return self


def _config(max_retries=3, provider="google_genai"):
    return LLMConfig(
        model="m",
        provider=provider,
        api_key="k",
        max_retries=max_retries,
        base_delay=0.0,
        max_delay=0.0,
        rate_limit_delay=0.0,
    )


def _raw(conteudo, tokens=(10, 5), tool_calls=None, invalid_tool_calls=None):
    return SimpleNamespace(
        content=conteudo,
        tool_calls=tool_calls or [],
        invalid_tool_calls=invalid_tool_calls or [],
        usage_metadata={
            "input_tokens": tokens[0],
            "output_tokens": tokens[1],
            "total_tokens": sum(tokens),
        },
    )


def _erro_do_parser(bruto):
    """Como o PydanticOutputParser entrega: OutputParserException com a validação na causa."""
    try:
        ComEvidencia.model_validate_json(bruto)
    except ValidationError as causa:
        erro = OutputParserException(
            f"Failed to parse ComEvidencia from completion {bruto}. Got: {causa}"
        )
        erro.__cause__ = causa
        return erro
    msg = "o bruto deveria ser inválido"
    raise AssertionError(msg)


def _falha(conteudo, erro=None, **kwargs):
    return {
        "parsed": None,
        "raw": _raw(conteudo, **kwargs),
        "parsing_error": erro or _erro_do_parser(conteudo),
    }


def _sucesso(modelo=None, tokens=(20, 7)):
    return {
        "parsed": modelo or ComEvidencia(aplicou=False),
        "raw": _raw("{}", tokens=tokens),
        "parsing_error": None,
    }


def _chamar(respostas, max_retries=3):
    structured = MagicMock()
    structured.invoke.side_effect = respostas
    base = MagicMock()
    base.with_structured_output.return_value = structured
    with patch("dataframeit.llm._create_langchain_llm", return_value=base):
        resultado = call_langchain("TEXTO", ComEvidencia, "Leia: {texto}", _config(max_retries))
    return resultado, structured


def _chamada(structured, i):
    return structured.invoke.call_args_list[i].args[0]


INVALIDO = json.dumps({"aplicou": True, "trecho": None})


class TestNovaTentativaComErro:
    def test_erro_e_resposta_voltam_ao_modelo(self):
        with pytest.warns(UserWarning, match="Tentativa"):
            resultado, structured = _chamar(
                [_falha(INVALIDO), _sucesso(ComEvidencia(aplicou=True, trecho="x"))]
            )
        assert resultado["data"] == {"aplicou": True, "trecho": "x"}
        assert _chamada(structured, 0) == "Leia: TEXTO"
        segunda = _chamada(structured, 1)
        assert [papel for papel, _ in segunda] == ["human", "ai", "human"]
        assert segunda[0][1] == "Leia: TEXTO"
        assert segunda[1][1] == INVALIDO
        assert "(resposta inteira): Value error, Aplicar exige trecho." in segunda[2][1]

    def test_validation_error_vem_da_causa_do_parser(self):
        # A causa é a validação do bruto; revalidar outro conteúdo daria outro erro.
        bruto = json.dumps({"aplicou": "talvez"})
        falha = {"parsed": None, "raw": _raw("{}"), "parsing_error": _erro_do_parser(bruto)}
        with pytest.warns(UserWarning, match="Tentativa"):
            _, structured = _chamar([falha, _sucesso()])
        correcao = _chamada(structured, 1)[2][1]
        assert "- aplicou: Input should be a valid boolean" in correcao
        assert '"talvez"' in correcao

    def test_cada_recusa_substitui_a_anterior(self):
        outro = json.dumps({"aplicou": True, "trecho": ""})
        with pytest.warns(UserWarning, match="Tentativa"):
            _, structured = _chamar([_falha(INVALIDO), _falha(outro), _sucesso()])
        terceira = _chamada(structured, 2)
        assert len(terceira) == 3
        assert terceira[1][1] == outro

    def test_uso_soma_as_tentativas_recusadas(self):
        with pytest.warns(UserWarning, match="Tentativa"):
            resultado, _ = _chamar(
                [
                    _falha(INVALIDO, tokens=(10, 5)),
                    _falha(INVALIDO, tokens=(11, 6)),
                    _sucesso(tokens=(20, 7)),
                ]
            )
        assert resultado["usage"]["input_tokens"] == 41
        assert resultado["usage"]["output_tokens"] == 18
        assert resultado["usage"]["total_tokens"] == 59

    def test_erro_transitorio_sem_raw_mantem_a_correcao(self):
        with pytest.warns(UserWarning, match="Tentativa"):
            resultado, structured = _chamar([_falha(INVALIDO), TimeoutError("lento"), _sucesso()])
        assert _chamada(structured, 2) == _chamada(structured, 1)
        assert resultado["usage"]["input_tokens"] == 30

    def test_resposta_por_tool_call(self):
        args = {"aplicou": True}
        falha = {
            "parsed": None,
            "raw": _raw("", tool_calls=[{"name": "ComEvidencia", "args": args}]),
            "parsing_error": "erro",
        }
        with pytest.warns(UserWarning, match="Tentativa"):
            _, structured = _chamar([falha, _sucesso()])
        assert _chamada(structured, 1)[1] == ("ai", json.dumps(args))
        assert "Aplicar exige trecho." in _chamada(structured, 1)[2][1]

    def test_tool_call_invalida_volta_como_texto(self):
        falha = {
            "parsed": None,
            "raw": _raw("", invalid_tool_calls=[{"name": "x", "args": '{"aplicou": tru'}]),
            "parsing_error": "json inválido",
        }
        with pytest.warns(UserWarning, match="Tentativa"):
            _, structured = _chamar([falha, _sucesso()])
        assert _chamada(structured, 1)[1] == ("ai", '{"aplicou": tru')

    def test_tool_call_invalida_sem_argumentos_cede_ao_conteudo(self):
        """Sem argumentos na chamada inválida, a resposta bruta é o conteúdo da mensagem."""
        falha = {
            "parsed": None,
            "raw": _raw(INVALIDO, invalid_tool_calls=[{"name": "x", "args": None}]),
            "parsing_error": "json inválido",
        }
        with pytest.warns(UserWarning, match="Tentativa"):
            _, structured = _chamar([falha, _sucesso()])
        assert _chamada(structured, 1)[1] == ("ai", INVALIDO)

    def test_blocos_sem_texto_ficam_de_fora(self):
        blocos = [
            {"type": "thinking", "thinking": "pensando"},
            {"type": "text", "text": INVALIDO[:5]},
            {"type": "text", "text": INVALIDO[5:]},
        ]
        falha = {"parsed": None, "raw": _raw(blocos), "parsing_error": "erro"}
        with pytest.warns(UserWarning, match="Tentativa"):
            _, structured = _chamar([falha, _sucesso()])
        assert _chamada(structured, 1)[1] == ("ai", INVALIDO)

    def test_sem_resposta_bruta_o_pedido_vai_junto_do_prompt(self):
        falha = {"parsed": None, "raw": _raw(""), "parsing_error": "vazio"}
        with pytest.warns(UserWarning, match="Tentativa"):
            _, structured = _chamar([falha, _sucesso()])
        segunda = _chamada(structured, 1)
        assert len(segunda) == 1
        assert segunda[0][0] == "human"
        assert segunda[0][1].startswith("Leia: TEXTO\n\n")
        assert "vazio" in segunda[0][1]

    def test_parsed_none_sem_parsing_error_pede_o_formato(self):
        sem_parse = {"parsed": None, "raw": _raw("texto livre"), "parsing_error": None}
        with pytest.warns(UserWarning, match="Tentativa"):
            _, structured = _chamar([sem_parse, _sucesso()])
        segunda = _chamada(structured, 1)
        assert segunda[1] == ("ai", "texto livre")
        assert "formato estruturado" in segunda[2][1]

    def test_json_malformado(self):
        with pytest.warns(UserWarning, match="Tentativa"):
            _, structured = _chamar([_falha("{quebrado", erro="falha do parser"), _sucesso()])
        segunda = _chamada(structured, 1)
        assert segunda[1] == ("ai", "{quebrado")
        assert "falha do parser" in segunda[2][1]

    def test_tetos_do_pedido(self):
        class Muitos(BaseModel):
            itens: list[int]

        bruto = json.dumps({"itens": ["x" * 400] * 30})
        try:
            Muitos.model_validate_json(bruto)
        except ValidationError as erro:
            falha = {"parsed": None, "raw": _raw(bruto), "parsing_error": erro}
        structured = MagicMock()
        structured.invoke.side_effect = [
            falha,
            {"parsed": Muitos(itens=[1]), "raw": _raw("{}"), "parsing_error": None},
        ]
        base = MagicMock()
        base.with_structured_output.return_value = structured
        with (
            pytest.warns(UserWarning, match="Tentativa"),
            patch("dataframeit.llm._create_langchain_llm", return_value=base),
        ):
            call_langchain("T", Muitos, "{texto}", _config())
        correcao = structured.invoke.call_args_list[1].args[0][2][1]
        linhas = [linha for linha in correcao.splitlines() if linha.startswith("- ")]
        assert len(linhas) == 20
        assert all(len(linha) < 450 for linha in linhas)

    def test_esgota_as_tentativas(self):
        with (
            pytest.warns(UserWarning, match="Tentativa"),
            pytest.raises(ProviderRejectedOutputError, match="parsing"),
        ):
            _chamar([_falha(INVALIDO)] * 2, max_retries=2)


class TestClassificacaoDaRecusa:
    def _recusa(self, bruto):
        with pytest.raises(ProviderRejectedOutputError) as info:
            _chamar([_falha(bruto)], max_retries=1)
        return info.value

    def test_numero_no_texto_nao_vira_erro_http(self):
        erro = self._recusa(json.dumps({"aplicou": True, "processo": "Rcl 401 404 429"}))
        assert "401" not in str(erro)
        assert is_recoverable_error(erro)
        assert not is_rate_limit_error(erro)
        assert "AUTENTICA" not in get_friendly_error_message(erro, "openai").upper()
        assert "RECUSADA" in get_friendly_error_message(erro, "openai")

    def test_recusa_nao_e_sobrecarga(self):
        erro = ProviderRejectedOutputError("429")
        assert not is_rate_limit_error(erro)
        assert isinstance(erro, ValueError)


class TestOpenAIReal:
    """ChatOpenAI de verdade sobre transporte HTTP falso: o SDK valida dentro da chamada."""

    @pytest.fixture
    def openai_falso(self):
        httpx = pytest.importorskip("httpx")
        langchain_openai = pytest.importorskip("langchain_openai")
        requisicoes = []

        def rodar(respostas, responses_api=False):
            def handler(req):
                requisicoes.append(json.loads(req.content))
                conteudo = respostas[min(len(requisicoes) - 1, len(respostas) - 1)]
                if responses_api:
                    return httpx.Response(
                        200,
                        json={
                            "id": "r",
                            "object": "response",
                            "created_at": 0,
                            "model": "gpt-x",
                            "status": "completed",
                            "output": [
                                {
                                    "type": "message",
                                    "id": "m",
                                    "role": "assistant",
                                    "status": "completed",
                                    "content": [
                                        {"type": "output_text", "text": conteudo, "annotations": []}
                                    ],
                                }
                            ],
                            "parallel_tool_calls": False,
                            "tool_choice": "auto",
                            "tools": [],
                            "usage": {
                                "input_tokens": 10,
                                "output_tokens": 5,
                                "total_tokens": 15,
                                "input_tokens_details": {"cached_tokens": 4},
                                "output_tokens_details": {"reasoning_tokens": 2},
                            },
                        },
                    )
                return httpx.Response(
                    200,
                    json={
                        "id": "x",
                        "object": "chat.completion",
                        "created": 0,
                        "model": "gpt-x",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": conteudo},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                            "total_tokens": 15,
                            "prompt_tokens_details": {"cached_tokens": 4},
                            "completion_tokens_details": {"reasoning_tokens": 2},
                        },
                    },
                )

            llm = langchain_openai.ChatOpenAI(
                model="gpt-x",
                api_key="k",
                max_retries=0,
                use_responses_api=responses_api,
                http_client=httpx.Client(transport=httpx.MockTransport(handler)),
            )
            with patch("dataframeit.llm._create_langchain_llm", return_value=llm):
                return call_langchain(
                    "TEXTO", ComEvidencia, "Leia: {texto}", _config(provider="openai")
                )

        return rodar, requisicoes

    @staticmethod
    def _mensagens(corpo):
        return corpo.get("messages") or corpo.get("input")

    @pytest.mark.parametrize("responses_api", [False, True])
    def test_recusa_do_sdk_leva_o_erro_ao_modelo(self, openai_falso, responses_api):
        rodar, requisicoes = openai_falso
        # "404" no texto recusado não pode impedir a nova tentativa.
        invalido = json.dumps({"aplicou": True, "trecho": None, "processo": "Rcl 404"})
        with pytest.warns(UserWarning, match="Tentativa"):
            resultado = rodar(
                [invalido, json.dumps({"aplicou": True, "trecho": "x"})], responses_api
            )
        assert resultado["data"] == {"aplicou": True, "trecho": "x"}
        assert len(requisicoes) == 2
        # A resposta recusada sai da resposta HTTP anexada à exceção do SDK.
        segunda = self._mensagens(requisicoes[1])
        assert [m["role"] for m in segunda] == ["user", "assistant", "user"]
        assert invalido in json.dumps(segunda[1]["content"], ensure_ascii=False).replace('\\"', '"')
        correcao = json.dumps(segunda[2]["content"], ensure_ascii=False)
        assert "(resposta inteira): Value error, Aplicar exige trecho. Valor recusado:" in correcao
        # As duas tentativas foram cobradas e entram no uso.
        assert resultado["usage"] == {
            "input_tokens": 20,
            "cached_input_tokens": 8,
            "output_tokens": 10,
            "total_tokens": 30,
            "reasoning_tokens": 4,
        }

    def test_modelo_com_titulo_proprio(self, openai_falso, monkeypatch):
        rodar, requisicoes = openai_falso
        monkeypatch.setitem(ComEvidencia.model_config, "title", "Evidencia do caso")
        ComEvidencia.model_rebuild(force=True)
        try:
            with pytest.warns(UserWarning, match="Tentativa"):
                rodar([INVALIDO, json.dumps({"aplicou": True, "trecho": "x"})])
        finally:
            monkeypatch.undo()
            ComEvidencia.model_rebuild(force=True)
        assert [m["role"] for m in self._mensagens(requisicoes[1])] == ["user", "assistant", "user"]


class TestCapturaNoInvoke:
    def _chamar_levantando(self, excecoes_e_respostas, max_retries=3):
        return _chamar(excecoes_e_respostas, max_retries=max_retries)

    def test_validation_error_de_outro_modelo_nao_e_resposta_recusada(self):
        class ImageConfig(BaseModel):
            aspect_ratio: str

        try:
            ImageConfig.model_validate({"aspect_ratio": 16})
        except ValidationError as erro:
            de_configuracao = erro
        with pytest.warns(UserWarning, match="Tentativa"), pytest.raises(ValidationError):
            _, _structured = self._chamar_levantando([de_configuracao] * 3)

    def test_recusa_do_sdk_sem_resposta_anexada_vai_junto_do_prompt(self):
        try:
            ComEvidencia.model_validate_json(INVALIDO)
        except ValidationError as erro:
            do_sdk = erro
        with pytest.warns(UserWarning, match="Tentativa"):
            resultado, structured = _chamar([do_sdk, _sucesso()])
        segunda = _chamada(structured, 1)
        assert len(segunda) == 1
        assert (
            "- (resposta inteira): Value error, Aplicar exige trecho. Valor recusado:"
            in segunda[0][1]
        )
        assert resultado["usage"]["input_tokens"] == 20

    def test_output_parser_exception_no_invoke_pede_correcao(self):
        erro = OutputParserException("sem tool call", llm_output="texto solto")
        with pytest.warns(UserWarning, match="Tentativa"):
            _, structured = _chamar([erro, _sucesso()])
        segunda = _chamada(structured, 1)
        assert segunda[1] == ("ai", "texto solto")
        assert "sem tool call" in segunda[2][1]

    def test_recusa_sem_bruto_nao_herda_a_resposta_anterior(self):
        try:
            ComEvidencia.model_validate_json(INVALIDO)
        except ValidationError as erro:
            do_sdk = erro
        with pytest.warns(UserWarning, match="Tentativa"):
            _, structured = _chamar([_falha(INVALIDO), do_sdk, _sucesso()])
        assert len(_chamada(structured, 2)) == 1


class TestCorpoDoSdk:
    def test_corpo_estranho_nao_troca_a_recusa(self):

        for corpo in ({"choices": [None]}, {"choices": []}, [1, 2], {"output": [None]}):
            erro = SimpleNamespace(response=SimpleNamespace(json=lambda corpo=corpo: corpo))
            assert _sdk_rejected_response(erro) == ("", None)
        assert _sdk_rejected_response(SimpleNamespace()) == ("", None)

    def test_total_ausente_soma_entrada_e_saida(self):

        _, uso = _read_sdk_body(
            {
                "choices": [{"message": {"content": "x"}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            }
        )
        assert uso["total_tokens"] == 10

    def test_corpo_sem_contagem_de_tokens_nao_tem_uso(self):
        """Sem usage, a tentativa recusada não soma zeros que pareceriam medidos."""

        assert _read_sdk_body({"choices": [{"message": {"content": "x"}}]}) == ("x", None)

    def test_parser_sem_llm_output_pede_o_formato_generico(self):
        erro = OutputParserException("Consider `method='json_schema'`")
        with pytest.warns(UserWarning, match="Tentativa"):
            _, structured = _chamar([erro, _sucesso()])
        segunda = _chamada(structured, 1)
        assert len(segunda) == 1
        assert "formato estruturado" in segunda[0][1]
        assert "json_schema" not in segunda[0][1]


class TestDiagnostico:
    def test_error_details_diz_a_regra_sem_o_valor(self):
        bruto = json.dumps({"aplicou": True, "processo": "Rcl 401"})
        with pytest.raises(ProviderRejectedOutputError) as info:
            _chamar([_falha(bruto)], max_retries=1)
        assert "(resposta inteira): Value error, Aplicar exige trecho." in str(info.value)
        assert "401" not in str(info.value)

    def test_erro_que_nao_e_de_validacao_diz_o_tipo(self):
        with pytest.raises(
            ProviderRejectedOutputError, match=r"fora do esquema \(OutputParserException\)"
        ):
            _chamar([_falha("{quebrado", erro=OutputParserException("recusa"))], max_retries=1)

    def test_texto_bruto_do_erro_tem_teto(self):
        with pytest.warns(UserWarning, match="Tentativa"):
            _, structured = _chamar([_falha("{x", erro="e" * 5000), _sucesso()])
        assert len(_chamada(structured, 1)[2][1]) < 2300
