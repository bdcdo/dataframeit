"""A mensagem amigável da linha que falha diz o que a execução fez com ela.

A caixa de limite de requisições aparece quando, e só quando, a execução
paralela reduz os workers, e a caixa de timeout, que manda aumentar
`max_retries`, só aparece para o erro que o retry repete. A caixa de um
`ProviderError` sai da classe e do status estruturado, e nunca do texto, que
pode ser do modelo. Os testes rodam a execução de ponta a ponta, com
`call_langchain` simulado e dois workers, e comparam o aviso de cada linha com
o aviso de redução e com `_error_details`.

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
    ProviderRejectedOutputError,
    ProviderTransientError,
    get_friendly_error_message,
)

CAIXA_RATE_LIMIT = "LIMITE DE REQUISIÇÕES ATINGIDO"
CAIXA_TIMEOUT = "TEMPO ESGOTADO"
CAIXA_AUTENTICACAO = "ERRO DE AUTENTICAÇÃO"
CAIXA_PERMISSAO = "ERRO DE PERMISSÃO"
CAIXA_GENERICA = "ERRO NO PROCESSAMENTO"
REDUCAO = "Reduzindo workers"
INTERRUPCAO = "Execução interrompida"


class _Campo(BaseModel):
    campo: str


class _TimerFalso:
    """Não inicia a thread que limparia o evento de rate limit."""

    def __init__(self, intervalo, funcao):
        pass

    def start(self):
        pass


def _titulo(mensagem):
    """Primeira linha de texto da mensagem amigável, sem a moldura da caixa."""
    return next(linha.strip("║ ") for linha in mensagem.splitlines() if linha.strip("╔═╗ "))


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
        # A primeira linha do aviso é "Falha ao processar linha N."; a caixa vem depois.
        titulos=[_titulo(falha.split("\n", 1)[1]) for falha in falhas],
        reduziu=any(REDUCAO in texto for texto in textos),
        interrompida=any(texto.startswith(INTERRUPCAO) for texto in textos),
        detalhes=saida["_error_details"].dropna().tolist() if "_error_details" in saida else [],
    )


def _conferir_coerencia(execucao, *, aborta=False):
    """O que a mensagem afirma bate com o que a execução decidiu.

    Antes, confere que os avisos foram lidos: sem as duas falhas, o teste que
    espera a ausência de uma caixa passaria sem ter visto aviso nenhum. A
    execução interrompida não avisa a falha da linha, e dela se exige o aviso
    de interrupção.
    """
    if aborta:
        assert execucao.interrompida
        assert execucao.falhas == []
    else:
        assert not execucao.interrompida
        assert len(execucao.falhas) == 2
    assert execucao.caixa_rate_limit == execucao.reduziu
    if execucao.caixa_timeout:
        assert execucao.detalhes
        assert all(detalhe.startswith("[Falhou após") for detalhe in execucao.detalhes)
        # Com max_retries=1, nenhuma linha ganhou nova tentativa, e a caixa não
        # pode afirmar que o sistema já tentou de novo.
        assert not any("tentou" in falha for falha in execucao.falhas)


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
    "O recurso diz que a petição transcreve a API key do sistema do tribunal.",
    "403 Forbidden",
    "A decisão aplica o art. 403 do CPC e não há valor a extrair.",
    "request timeout while streaming",
    "O contrato de franquia fala da network de lojas.",
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


# Caixa que a classe decide quando a exceção não traz status; os demais
# ProviderError, sem status, caem na genérica.
CAIXA_DA_CLASSE = {
    ProviderOverloadedError: CAIXA_RATE_LIMIT,
    ProviderRejectedOutputError: "RESPOSTA RECUSADA PELA VALIDAÇÃO DO MODELO PYDANTIC",
    ProviderOutputError: "RESPOSTA FORA DO CONTRATO DE SAÍDA",
}


@pytest.mark.parametrize("texto", TEXTOS)
@pytest.mark.parametrize("classe", CLASSES_DO_PACOTE, ids=lambda classe: classe.__name__)
def test_caixa_segue_a_classe_e_o_que_a_execucao_fez(classe, texto, monkeypatch):
    execucao = _executar(lambda: classe(texto), monkeypatch)

    # O ProviderAbortError para a execução sem gravar a linha nem avisar a falha dela.
    _conferir_coerencia(execucao, aborta=issubclass(classe, ProviderAbortError))
    # Nos ProviderError e no ProviderOutputError a caixa sai da classe, e nenhum
    # texto escolhe a de credencial, permissão, timeout, conexão ou busca; rate
    # limit é só a sobrecarga. O ProviderConfigurationError, que os providers
    # levantam antes da primeira linha, é lido pelo texto, e aqui vale só a
    # coerência acima.
    if issubclass(classe, (ProviderError, ProviderOutputError)):
        assert execucao.caixa_rate_limit is issubclass(classe, ProviderOverloadedError)
        esperada = next(
            (caixa for base, caixa in CAIXA_DA_CLASSE.items() if issubclass(classe, base)),
            CAIXA_GENERICA,
        )
        assert all(titulo == esperada for titulo in execucao.titulos)


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


# O langchain_tavily e o exa_py levantam ValueError com o status só no texto. A
# caixa nomeia o serviço de busca que recusou; o ProviderError vem do provider
# de LLM, o openai padrão do dataframeit, ainda que o texto cite o Tavily.
@pytest.mark.parametrize(
    ("fabrica", "servico"),
    [
        (
            lambda: ValueError("Error 429: Rate limit exceeded. Please contact support@tavily.com"),
            "Tavily",
        ),
        (lambda: ValueError("Request failed with status code 429: exa rate limit"), "Exa"),
        (lambda: ProviderOverloadedError("tavily: 429 Too Many Requests"), "OpenAI"),
    ],
    ids=["tavily", "exa", "provider-error-cita-tavily"],
)
def test_rate_limit_da_busca_nomeia_o_servico(fabrica, servico, monkeypatch):
    execucao = _executar(fabrica, monkeypatch)

    _conferir_coerencia(execucao)
    assert execucao.caixa_rate_limit
    assert all(f"pouco tempo para o {servico}." in falha for falha in execucao.falhas)


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


# O termo que decidiria a caixa pelo texto está no result, que é conteúdo; a
# caixa sai da classe e do api_error_status, que a exceção guarda em status_code.
@pytest.mark.parametrize(
    ("campos", "classe", "caixa"),
    [
        (
            {"result": "O recurso diz que a petição transcreve a API key do sistema do tribunal."},
            ProviderTransientError,
            CAIXA_GENERICA,
        ),
        (
            {"result": "A decisão aplica o art. 403 do CPC e não há valor a extrair."},
            ProviderTransientError,
            CAIXA_GENERICA,
        ),
        (
            {"api_error_status": 400, "result": "A decisão aplica o art. 403 do CPC."},
            ProviderError,
            CAIXA_GENERICA,
        ),
        (
            {"result": "O contrato de franquia fala da network de lojas."},
            ProviderTransientError,
            CAIXA_GENERICA,
        ),
        (
            {"subtype": "error_max_turns", "result": "O campo api_key do JSON ficou vazio."},
            ProviderError,
            CAIXA_GENERICA,
        ),
        (
            {"result": "O edital prevê timeout de sessão de 30 minutos."},
            ProviderTransientError,
            CAIXA_GENERICA,
        ),
        (
            {"api_error_status": 500, "result": "O edital prevê timeout de sessão de 30 minutos."},
            ProviderTransientError,
            CAIXA_GENERICA,
        ),
        (
            {"api_error_status": 401, "result": "Sessão encerrada."},
            ProviderError,
            CAIXA_AUTENTICACAO,
        ),
        ({"api_error_status": 403, "result": "Sessão encerrada."}, ProviderError, CAIXA_PERMISSAO),
        (
            {"api_error_status": 504, "result": "Sessão encerrada."},
            ProviderTransientError,
            CAIXA_TIMEOUT,
        ),
    ],
    ids=[
        "api-key-no-result",
        "art-403-no-result",
        "art-403-com-status-400",
        "network-no-result",
        "api-key-em-max-turns",
        "timeout-no-result",
        "timeout-com-status-500",
        "status-401",
        "status-403",
        "status-504",
    ],
)
def test_caixa_do_claude_code_sai_do_status(campos, classe, caixa, monkeypatch):
    def fabrica():
        return _capturar(_raise_for_result_error, _resultado_claude(**campos))

    erro = fabrica()
    assert type(erro) is classe
    assert _titulo(get_friendly_error_message(erro, "claude_code")).startswith(caixa)

    execucao = _executar(fabrica, monkeypatch)

    _conferir_coerencia(execucao)
    assert all(titulo.startswith(caixa) for titulo in execucao.titulos)


@pytest.fixture
def codex_gerado():
    pytest.importorskip("openai_codex")
    return pytest.importorskip("openai_codex.generated.v2_all")


# Rate limit do pacote é a sobrecarga: a caixa sai para o ProviderOverloadedError.
@pytest.mark.parametrize(
    ("mensagem", "codigo", "classe"),
    [
        ("429 Too Many Requests", None, ProviderError),
        ("request timeout while streaming", None, ProviderError),
        ("server overloaded", "server_overloaded", ProviderOverloadedError),
        ("Invalid API key: slow down", "rate_limit_exceeded", ProviderOverloadedError),
    ],
    ids=["429-no-texto", "timeout-no-texto", "sobrecarga", "rate-limit-com-chave"],
)
def test_turno_do_codex(codex_gerado, mensagem, codigo, classe, monkeypatch):
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
    assert execucao.caixa_rate_limit is (classe is ProviderOverloadedError)
    assert not execucao.caixa_timeout


def _falha_http(gerado, status):
    return gerado.CodexErrorInfo(
        root=gerado.HttpConnectionFailedCodexErrorInfo(
            httpConnectionFailed=gerado.HttpConnectionFailed(httpStatusCode=status)
        )
    )


# A mensagem do turno pode trazer texto da conversa com o modelo; a caixa sai do
# código do erro e do status HTTP que o runtime informa.
@pytest.mark.parametrize(
    ("mensagem", "info", "esperado"),
    [
        ("O documento traz a API key do portal.", lambda _g: None, (ProviderError, CAIXA_GENERICA)),
        ("Aplica-se o art. 403 do CPC.", lambda _g: None, (ProviderError, CAIXA_GENERICA)),
        (
            "A network de distribuidores foi citada.",
            lambda _g: None,
            (ProviderError, CAIXA_GENERICA),
        ),
        (
            "O documento traz a API key do portal.",
            lambda g: _falha_http(g, 400),
            (ProviderError, CAIXA_GENERICA),
        ),
        ("unexpected status", lambda g: _falha_http(g, 401), (ProviderError, CAIXA_AUTENTICACAO)),
        ("unexpected status", lambda g: _falha_http(g, 403), (ProviderError, CAIXA_PERMISSAO)),
        (
            "unexpected status",
            lambda g: _falha_http(g, 504),
            (ProviderTransientError, CAIXA_TIMEOUT),
        ),
        (
            "Sua sessão expirou; faça login de novo.",
            lambda g: g.CodexErrorInfo(root=g.CodexErrorInfoValue.unauthorized),
            (ProviderError, CAIXA_AUTENTICACAO),
        ),
    ],
    ids=[
        "api-key-no-texto",
        "art-403-no-texto",
        "network-no-texto",
        "api-key-com-http-400",
        "http-401",
        "http-403",
        "http-504",
        "unauthorized",
    ],
)
def test_caixa_do_turno_do_codex_sai_do_codigo(codex_gerado, mensagem, info, esperado, monkeypatch):
    classe, caixa = esperado

    def fabrica():
        turno = codex_gerado.TurnError(message=mensagem, codexErrorInfo=info(codex_gerado))
        return _capturar(_raise_turn_error, turno)

    erro = fabrica()
    assert type(erro) is classe
    assert _titulo(get_friendly_error_message(erro, "codex")).startswith(caixa)

    execucao = _executar(fabrica, monkeypatch)

    _conferir_coerencia(execucao)
    assert all(titulo.startswith(caixa) for titulo in execucao.titulos)


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
        # No ProviderError, o texto não decide a caixa, e o status 408 ou 504 decide.
        (lambda: ProviderTransientError("request timeout while streaming"), False),
        (lambda: ProviderTransientError("upstream failed", status_code=504), True),
        (lambda: ProviderTransientError("upstream failed", status_code=408), True),
        (lambda: ProviderTransientError("request timeout", status_code=500), False),
        (lambda: ProviderError("request timeout while streaming"), False),
        # O definitivo não ganha nova tentativa, por mais que se aumente max_retries.
        (lambda: ProviderError("upstream failed", status_code=504), False),
        (lambda: TimeoutError("request timed out"), True),
        # O 404 torna o erro definitivo, e aumentar max_retries não o repete.
        (lambda: Exception("timeout: 404 model not found"), False),
    ],
    ids=[
        "transitorio-texto",
        "transitorio-504",
        "transitorio-408",
        "transitorio-500-texto",
        "definitivo",
        "definitivo-504",
        "timeout-error",
        "texto-404",
    ],
)
def test_caixa_de_timeout_so_para_o_erro_repetido(fabrica, timeout, monkeypatch):
    execucao = _executar(fabrica, monkeypatch)

    _conferir_coerencia(execucao)
    assert execucao.caixa_timeout is timeout
