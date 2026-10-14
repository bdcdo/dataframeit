"""A mensagem amigável da linha que falha diz o que a execução fez com ela.

A caixa de limite de requisições aparece quando, e só quando, a execução
paralela reduz os workers, e a caixa de timeout, que diz que o sistema já tentou
de novo, só aparece para o erro que ganha nova tentativa. Os testes rodam a
execução de ponta a ponta, com `call_langchain` simulado e dois workers, e
comparam o aviso de cada linha com o aviso de redução e com `_error_details`.

A parametrização percorre toda classe de exceção definida em
`dataframeit.errors`, para que uma classe nova entre no teste sem ser listada.
"""

import inspect
import warnings
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
from pydantic import BaseModel

from dataframeit import core, dataframeit, errors
from dataframeit.claude_code import _raise_for_result_error
from dataframeit.codex import _raise_classified_sdk_error, _raise_turn_error
from dataframeit.errors import (
    ProviderAbortError,
    ProviderError,
    ProviderOutputError,
    ProviderOverloadedError,
    ProviderTransientError,
)

CAIXA_RATE_LIMIT = "LIMITE DE REQUISIÇÕES ATINGIDO"
CAIXA_TIMEOUT = "TEMPO ESGOTADO"
REDUCAO = "Reduzindo workers"


class _Campo(BaseModel):
    campo: str


class _TimerFalso:
    """Não inicia a thread que limparia o evento de rate limit."""

    def __init__(self, intervalo, funcao):
        pass

    def start(self):
        pass


def _executar(fabrica, monkeypatch):
    """Roda duas linhas que falham com o erro da fábrica e lê o que a execução avisou."""
    monkeypatch.setattr(core.time, "sleep", lambda _segundos: None)
    monkeypatch.setattr(core.threading, "Timer", _TimerFalso)

    def llm(*_args, **_kwargs):
        raise fabrica()

    with (
        patch("dataframeit.core.call_langchain", side_effect=llm),
        patch("dataframeit.core.validate_provider_dependencies"),
        warnings.catch_warnings(record=True) as avisos,
    ):
        warnings.simplefilter("always")
        saida = dataframeit(
            pd.DataFrame({"texto": ["a", "b"]}),
            _Campo,
            "{texto}",
            parallel_requests=2,
            max_retries=1,
            track_tokens=False,
        )

    textos = [str(aviso.message) for aviso in avisos]
    falhas = [texto for texto in textos if texto.startswith("Falha ao processar linha")]
    return SimpleNamespace(
        falhas=falhas,
        caixa_rate_limit=any(CAIXA_RATE_LIMIT in falha for falha in falhas),
        caixa_timeout=any(CAIXA_TIMEOUT in falha for falha in falhas),
        reduziu=any(REDUCAO in texto for texto in textos),
        detalhes=saida["_error_details"].dropna().tolist() if "_error_details" in saida else [],
    )


def _conferir_coerencia(execucao):
    """O que a mensagem afirma bate com o que a execução decidiu."""
    assert execucao.caixa_rate_limit == execucao.reduziu
    if execucao.caixa_timeout:
        assert execucao.detalhes
        assert all(detalhe.startswith("[Falhou após") for detalhe in execucao.detalhes)


# =============================================================================
# Toda classe de exceção do pacote, com textos que imitam outras categorias
# =============================================================================

CLASSES_DO_PACOTE = sorted(
    (
        objeto
        for objeto in vars(errors).values()
        if inspect.isclass(objeto)
        and issubclass(objeto, Exception)
        and objeto.__module__ == errors.__name__
    ),
    key=lambda classe: classe.__name__,
)

# Textos que a mensagem de um ProviderError pode trazer, inclusive do modelo,
# e que a classificação pelo texto leria como outra categoria.
TEXTOS = [
    "Turno Codex falhou: 429 Too Many Requests",
    "Não foi possível concluir: o art. 429 do CPC exige...",
    "RateLimitError: slow down",
    "ResourceExhausted: quota",
    "Invalid API key provided",
    "403 Forbidden",
    "request timeout while streaming",
    "server overloaded",
    "tavily api_key missing",
    "exa quota exceeded",
]


def test_a_descoberta_acha_as_classes_do_pacote():
    nomes = {classe.__name__ for classe in CLASSES_DO_PACOTE}
    assert {
        "ProviderError",
        "ProviderTransientError",
        "ProviderOverloadedError",
        "ProviderRejectedOutputError",
        "ProviderAbortError",
        "ProviderUsageLimitError",
        "ProviderConfigurationError",
        "ProviderOutputError",
    } <= nomes


@pytest.mark.parametrize("texto", TEXTOS)
@pytest.mark.parametrize("classe", CLASSES_DO_PACOTE, ids=lambda classe: classe.__name__)
def test_caixa_de_rate_limit_e_reducao_andam_juntas(classe, texto, monkeypatch):
    execucao = _executar(lambda: classe(texto), monkeypatch)

    _conferir_coerencia(execucao)
    if issubclass(classe, ProviderAbortError):
        # A execução para sem gravar a linha nem avisar a falha dela.
        assert execucao.falhas == []
        return
    assert len(execucao.falhas) == 2
    # Nos ProviderError e no ProviderOutputError, que is_rate_limit_error
    # classifica sem ler o texto, rate limit é só a sobrecarga. O
    # ProviderConfigurationError, que os providers levantam antes da primeira
    # linha, é lido pelo texto, e aqui vale só a coerência acima.
    if issubclass(classe, (ProviderError, ProviderOutputError)):
        assert execucao.caixa_rate_limit is issubclass(classe, ProviderOverloadedError)


# =============================================================================
# Erros de fora do pacote e caminhos reais dos providers
# =============================================================================


class _ErroComStatus429(Exception):
    status_code = 429


class _ErroComStatus503(Exception):
    status_code = 503


@pytest.mark.parametrize(
    ("fabrica", "rate_limit"),
    [
        (lambda: Exception("429 Too Many Requests"), True),
        (lambda: _ErroComStatus429("request failed"), True),
        (lambda: _ErroComStatus503("upstream said 429"), False),
        (lambda: Exception("Invalid API key: 429 Too Many Requests"), True),
        (lambda: ValueError("Invalid argument"), False),
        (lambda: TimeoutError("request timed out"), False),
    ],
    ids=["texto-429", "status-429", "status-503-texto-429", "chave-e-429", "comum", "timeout"],
)
def test_erro_de_fora_do_pacote(fabrica, rate_limit, monkeypatch):
    execucao = _executar(fabrica, monkeypatch)

    _conferir_coerencia(execucao)
    assert execucao.caixa_rate_limit is rate_limit


class _ModelErrorFalso(Exception):
    is_retryable = False


class _ModelRateLimitFalso(_ModelErrorFalso):
    is_retryable = True


@pytest.mark.parametrize(
    ("fabrica", "rate_limit"),
    [
        (lambda: _ModelErrorFalso("429 rate limit exceeded"), False),
        (lambda: _ModelRateLimitFalso("slow down"), True),
    ],
    ids=["model-error-com-429", "model-rate-limit"],
)
def test_model_error_do_langchain_decide_pela_classe(fabrica, rate_limit, monkeypatch):
    monkeypatch.setattr(errors, "_ModelError", _ModelErrorFalso)
    monkeypatch.setattr(errors, "_ModelRateLimitError", _ModelRateLimitFalso)

    execucao = _executar(fabrica, monkeypatch)

    _conferir_coerencia(execucao)
    assert execucao.caixa_rate_limit is rate_limit


def _capturar(funcao, *args):
    """A exceção que o caminho de classificação do provider levanta."""
    with pytest.raises(ProviderError) as falha:
        funcao(*args)
    return falha.value


def _resultado_claude(**campos):
    return SimpleNamespace(
        **{
            "is_error": True,
            "subtype": "error_during_execution",
            "api_error_status": None,
            "errors": None,
            "result": None,
            **campos,
        }
    )


@pytest.mark.parametrize(
    ("campos", "classe", "rate_limit"),
    [
        (
            {"result": "Não foi possível concluir: o art. 429 do CPC exige..."},
            ProviderTransientError,
            False,
        ),
        (
            {"subtype": "error_max_turns", "result": "Fundamento: art. 429 do CPC"},
            ProviderError,
            False,
        ),
        ({"api_error_status": 529, "result": "Overloaded"}, ProviderOverloadedError, True),
        (
            {"api_error_status": 529, "result": "Overloaded. Check your API key usage."},
            ProviderOverloadedError,
            True,
        ),
        ({"api_error_status": 429, "result": "Rate limited"}, ProviderOverloadedError, True),
    ],
    ids=["429-no-result", "max-turns-429", "status-529", "status-529-chave", "status-429"],
)
def test_erro_do_claude_code(campos, classe, rate_limit, monkeypatch):
    erro = _capturar(_raise_for_result_error, _resultado_claude(**campos))
    assert type(erro) is classe

    execucao = _executar(
        lambda: _capturar(_raise_for_result_error, _resultado_claude(**campos)), monkeypatch
    )

    _conferir_coerencia(execucao)
    assert execucao.caixa_rate_limit is rate_limit


@pytest.fixture
def codex_gerado():
    pytest.importorskip("openai_codex")
    return pytest.importorskip("openai_codex.generated.v2_all")


@pytest.mark.parametrize(
    ("mensagem", "codigo", "classe", "rate_limit"),
    [
        ("429 Too Many Requests", None, ProviderError, False),
        ("request timeout while streaming", None, ProviderError, False),
        ("server overloaded", "server_overloaded", ProviderOverloadedError, True),
        ("Invalid API key: slow down", "rate_limit_exceeded", ProviderOverloadedError, True),
    ],
    ids=["429-no-texto", "timeout-no-texto", "sobrecarga", "rate-limit-com-chave"],
)
def test_turno_do_codex(codex_gerado, mensagem, codigo, classe, rate_limit, monkeypatch):
    info = (
        codex_gerado.CodexErrorInfo(root=codex_gerado.CodexErrorInfoValue[codigo])
        if codigo
        else None
    )

    def fabrica():
        turno = codex_gerado.TurnError(message=mensagem, codexErrorInfo=info)
        return _capturar(_raise_turn_error, turno)

    assert type(fabrica()) is classe

    execucao = _executar(fabrica, monkeypatch)

    _conferir_coerencia(execucao)
    assert execucao.caixa_rate_limit is rate_limit
    assert not execucao.caixa_timeout


def test_erro_json_rpc_do_codex_com_429_no_texto(codex_gerado, monkeypatch):
    from openai_codex.errors import JsonRpcError  # noqa: PLC0415 (SDK carregado pelo fixture)

    def fabrica():
        return _capturar(
            _raise_classified_sdk_error,
            JsonRpcError(-32603, "upstream returned 429 Too Many Requests"),
        )

    assert type(fabrica()) is ProviderError

    execucao = _executar(fabrica, monkeypatch)

    _conferir_coerencia(execucao)
    assert not execucao.caixa_rate_limit


# =============================================================================
# Caixa de timeout
# =============================================================================


@pytest.mark.parametrize(
    ("fabrica", "timeout"),
    [
        (lambda: ProviderTransientError("request timeout while streaming"), True),
        (lambda: ProviderError("request timeout while streaming"), False),
        (lambda: TimeoutError("request timed out"), True),
        # O 404 torna o erro definitivo, e a caixa não pode dizer que houve repetição.
        (lambda: Exception("timeout: 404 model not found"), False),
    ],
    ids=["transitorio", "definitivo", "timeout-error", "texto-404"],
)
def test_caixa_de_timeout_so_para_o_erro_repetido(fabrica, timeout, monkeypatch):
    execucao = _executar(fabrica, monkeypatch)

    _conferir_coerencia(execucao)
    assert execucao.caixa_timeout is timeout
