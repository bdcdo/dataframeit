"""Testes da classificação de erros em recuperáveis e não recuperáveis."""

from unittest.mock import Mock

import pytest

from dataframeit.errors import (
    ProviderError,
    ProviderTransientError,
    is_rate_limit_error,
    is_recoverable_error,
)


class ErroComStatusCode(Exception):
    """Imita openai.APIStatusError, anthropic.APIStatusError e httpx."""

    def __init__(self, mensagem, status_code):
        super().__init__(mensagem)
        self.status_code = status_code


class ErroComCode(Exception):
    """Imita google.genai.errors.APIError e google.api_core.exceptions."""

    def __init__(self, mensagem, code):
        super().__init__(mensagem)
        self.code = code


class ErroComHttpStatus(Exception):
    def __init__(self, mensagem, http_status):
        super().__init__(mensagem)
        self.http_status = http_status


class ErroComResposta(Exception):
    """Imita exceções que só expõem o status em response.status_code."""

    def __init__(self, mensagem, status_code):
        super().__init__(mensagem)
        self.response = Mock(status_code=status_code)


class ChatGoogleGenerativeAIError(Exception):
    """Imita o wrapper do langchain-google-genai, que não carrega status próprio."""


class ErroGenerico(Exception):
    pass


@pytest.mark.parametrize("status", [400, 401, 403, 404, 413, 422])
def test_status_4xx_estruturado_nao_e_recuperavel(status):
    erro = ErroComStatusCode("requisição rejeitada", status)

    assert is_recoverable_error(erro) is False


@pytest.mark.parametrize("status", [408, 409, 429, 500, 502, 503, 504, 529])
def test_status_transitorio_estruturado_e_recuperavel(status):
    erro = ErroComStatusCode("falha temporária", status)

    assert is_recoverable_error(erro) is True


@pytest.mark.parametrize(
    "erro, esperado",
    [
        (ErroComCode("400 INVALID_ARGUMENT", 400), False),
        (ErroComCode("503 UNAVAILABLE", 503), True),
        (ErroComHttpStatus("rejeitado", 422), False),
        (ErroComHttpStatus("sobrecarga", 503), True),
        (ErroComResposta("rejeitado", 400), False),
        (ErroComResposta("limite", 429), True),
    ],
)
def test_status_estruturado_em_outros_atributos(erro, esperado):
    assert is_recoverable_error(erro) is esperado


def test_status_estruturado_prevalece_sobre_substring_do_nome():
    # O nome sugere erro transitório, mas o servidor respondeu 400.
    erro = ErroComStatusCode("InternalServerError na validação do schema", 400)

    assert is_recoverable_error(erro) is False


def test_wrapper_sem_status_usa_o_status_da_causa():
    causa = ErroComCode("400 INVALID_ARGUMENT. Request contains an invalid argument.", 400)
    erro = ChatGoogleGenerativeAIError("400 INVALID_ARGUMENT: invalid argument")
    erro.__cause__ = causa

    assert is_recoverable_error(erro) is False


def test_wrapper_sem_status_com_causa_transitoria_e_recuperavel():
    causa = ErroComCode("503 UNAVAILABLE", 503)
    erro = ChatGoogleGenerativeAIError("model overloaded")
    erro.__cause__ = causa

    assert is_recoverable_error(erro) is True


def test_code_nao_numerico_nao_conta_como_status():
    # openai.APIError.code é um código textual, como 'invalid_api_key'.
    erro = ErroComCode("falha de rede", "connection_reset")

    assert is_recoverable_error(erro) is True


@pytest.mark.parametrize("valor", [True, 7, 1000, 200, 302])
def test_valor_fora_da_faixa_de_erro_http_e_ignorado(valor):
    erro = ErroComCode("falha qualquer", valor)

    assert is_recoverable_error(erro) is True


def test_numero_de_tokens_contendo_401_nao_marca_como_nao_recuperavel():
    erro = ErroGenerico("context has 4015 tokens, retry later")

    assert is_recoverable_error(erro) is True


@pytest.mark.parametrize(
    "mensagem",
    ["Error code: 401 - invalid key", "HTTP 403 Forbidden", "status 404"],
)
def test_status_numerico_isolado_na_mensagem_continua_nao_recuperavel(mensagem):
    assert is_recoverable_error(ErroGenerico(mensagem)) is False


def test_rate_limit_sem_status_continua_recuperavel():
    class RateLimitError(Exception):
        pass

    erro = RateLimitError("too many requests")

    assert is_recoverable_error(erro) is True
    assert is_rate_limit_error(erro) is True


def test_numero_contendo_429_nao_e_rate_limit():
    erro = ErroGenerico("prompt has 4290 tokens")

    assert is_rate_limit_error(erro) is False


def test_429_isolado_na_mensagem_continua_rate_limit():
    erro = ErroGenerico("HTTP 429 from upstream")

    assert is_rate_limit_error(erro) is True


def test_classes_do_provider_mantem_precedencia_sobre_status():
    transitorio = ProviderTransientError("falha")
    transitorio.status_code = 400
    definitivo = ProviderError("falha")
    definitivo.status_code = 503

    assert is_recoverable_error(transitorio) is True
    assert is_recoverable_error(definitivo) is False


def _resposta_http(status):
    return Mock(status_code=status, headers={}, request=Mock())


def test_excecoes_reais_do_openai():
    openai = pytest.importorskip("openai")

    erro_400 = openai.BadRequestError("inválido", response=_resposta_http(400), body=None)
    erro_503 = openai.InternalServerError("indisponível", response=_resposta_http(503), body=None)

    assert is_recoverable_error(erro_400) is False
    assert is_recoverable_error(erro_503) is True


def test_excecoes_reais_do_anthropic():
    anthropic = pytest.importorskip("anthropic")

    erro_422 = anthropic.UnprocessableEntityError(
        "inválido", response=_resposta_http(422), body=None
    )
    erro_429 = anthropic.RateLimitError("limite", response=_resposta_http(429), body=None)

    assert is_recoverable_error(erro_422) is False
    assert is_recoverable_error(erro_429) is True


def test_cadeia_de_causas_ciclica_termina():
    erro_a = ChatGoogleGenerativeAIError("falha a")
    erro_b = ChatGoogleGenerativeAIError("falha b")
    erro_a.__cause__ = erro_b
    erro_b.__cause__ = erro_a

    assert is_recoverable_error(erro_a) is True


@pytest.mark.parametrize("status, esperado", [(429, True), (503, False), (400, False)])
def test_rate_limit_usa_o_status_estruturado(status, esperado):
    erro = ErroComStatusCode("falha sem pista na mensagem", status)

    assert is_rate_limit_error(erro) is esperado
