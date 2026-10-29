"""Classificação e mensagens de erro dos providers."""

import importlib
import importlib.util
from types import SimpleNamespace

import pytest

from dataframeit import errors
from dataframeit.claude_code import _raise_for_result_error
from dataframeit.errors import (
    ProviderError,
    ProviderTransientError,
    get_friendly_error_message,
    is_rate_limit_error,
    is_recoverable_error,
    validate_provider_dependencies,
)

# =============================================================================
# Mensagens amigáveis
# =============================================================================


def test_mistral_usa_mistral_api_key():
    mensagem = get_friendly_error_message(Exception("AuthenticationError: invalid"), "mistralai")
    assert "MISTRAL_API_KEY" in mensagem
    assert "MISTRALAI_API_KEY" not in mensagem


def test_numero_dentro_de_outro_nao_escolhe_a_caixa_de_autenticacao():
    mensagem = get_friendly_error_message(
        ValueError("prompt has 4015 tokens, above the limit"), "openai"
    )
    assert "AUTENTICAÇÃO" not in mensagem


def test_status_estruturado_escolhe_a_caixa():
    class ErroHttp(Exception):
        status_code = 429

    mensagem = get_friendly_error_message(ErroHttp("request failed"), "openai")
    assert "LIMITE DE REQUISIÇÕES" in mensagem


def test_status_estruturado_vence_o_texto():
    class ErroHttp(Exception):
        status_code = 503

    mensagem = get_friendly_error_message(ErroHttp("api key rotation in progress"), "openai")
    assert "AUTENTICAÇÃO" not in mensagem


def test_caixa_do_tavily_nao_e_sombreada_pela_generica():
    mensagem = get_friendly_error_message(
        Exception("MissingAPIKeyError: tavily api_key is missing"), "openai"
    )
    assert "TAVILY" in mensagem.upper()


def test_chave_invalida_do_google_com_status_400_escolhe_a_caixa_de_autenticacao():
    class ErroGoogle(Exception):
        status_code = 400

    erro = ErroGoogle(
        "Error calling model 'gemini-2.5-flash' (INVALID_ARGUMENT): 400 INVALID_ARGUMENT. "
        "{'error': {'message': 'API key not valid. Please pass a valid API key.', "
        "'status': 'INVALID_ARGUMENT', 'details': [{'reason': 'API_KEY_INVALID'}]}}"
    )
    mensagem = get_friendly_error_message(erro, "google_genai")
    assert "AUTENTICAÇÃO" in mensagem
    assert "GOOGLE_API_KEY" in mensagem


def test_401_sem_status_escolhe_a_caixa_de_autenticacao():
    mensagem = get_friendly_error_message(Exception("401 unauthorized"), "openai")
    assert "AUTENTICAÇÃO" in mensagem


def test_mensagem_real_do_exa_escolhe_a_caixa_do_exa():
    erro = ValueError(
        "API key must be provided as an argument or in EXA_API_KEY environment variable"
    )
    mensagem = get_friendly_error_message(erro, "openai")
    assert "AUTENTICAÇÃO EXA" in mensagem


def test_limite_do_exa_escolhe_a_caixa_de_limite():
    mensagem = get_friendly_error_message(Exception("exa: monthly quota exceeded"), "openai")
    assert "LIMITE DE USO EXA" in mensagem


def test_exa_dentro_de_palavra_nao_escolhe_a_caixa_do_exa():
    mensagem = get_friendly_error_message(Exception("hexagonal api_key rotation"), "openai")
    assert "EXA" not in mensagem


def test_limite_do_tavily_escolhe_a_caixa_de_limite():
    mensagem = get_friendly_error_message(
        Exception("UsageLimitExceededError: plan usage exceeded"), "openai"
    )
    assert "LIMITE DE USO TAVILY" in mensagem


# =============================================================================
# ModelError do langchain-core
# =============================================================================


class _ModelErrorFalso(Exception):
    is_retryable = False


class _ModelRateLimitFalso(_ModelErrorFalso):
    is_retryable = True


@pytest.fixture
def model_error_falso(monkeypatch):
    monkeypatch.setattr(errors, "_ModelError", _ModelErrorFalso)
    monkeypatch.setattr(errors, "_ModelRateLimitError", _ModelRateLimitFalso)


def test_model_error_decide_pelo_is_retryable(model_error_falso):
    # Mensagem com "limit 400 requests" e sem status: sem ModelError seria re-tentado
    class InvalidoFalso(_ModelErrorFalso):
        pass

    assert is_recoverable_error(InvalidoFalso("400 INVALID_ARGUMENT: limit 400 requests")) is False
    assert is_recoverable_error(_ModelRateLimitFalso("slow down")) is True


def test_model_rate_limit_error_reduz_workers(model_error_falso):
    assert is_rate_limit_error(_ModelRateLimitFalso("slow down")) is True
    assert is_rate_limit_error(_ModelErrorFalso("bad request")) is False
    # A classe decide, mesmo quando o texto parece de rate limit.
    assert is_rate_limit_error(_ModelErrorFalso("429 rate limit exceeded")) is False


def test_context_overflow_nao_e_re_tentado():
    class ContextOverflowError(Exception):
        pass

    assert is_recoverable_error(ContextOverflowError("prompt too long")) is False


def test_model_error_real_quando_disponivel():
    exceptions = pytest.importorskip("langchain_core.exceptions")
    if not hasattr(exceptions, "ModelError"):
        pytest.skip("langchain-core sem ModelError")
    assert errors._ModelError is exceptions.ModelError
    assert is_recoverable_error(exceptions.ModelInvalidRequestError("x")) is False
    assert is_recoverable_error(exceptions.ModelRateLimitError("x")) is True
    assert is_rate_limit_error(exceptions.ModelRateLimitError("x")) is True


def test_sem_hierarquia_de_erros_do_langchain_core_classifica_pelo_status(monkeypatch):
    """langchain-core anterior a 1.6 não tem ModelError, e o import cai no fallback.

    O módulo é carregado numa cópia à parte: o dataframeit.errors já importado,
    de que os outros módulos guardam referências, fica intacto.
    """
    exceptions = pytest.importorskip("langchain_core.exceptions")
    for nome in ("ModelError", "ModelRateLimitError"):
        monkeypatch.delattr(exceptions, nome, raising=False)
    spec = importlib.util.find_spec("dataframeit.errors")
    assert spec is not None
    assert spec.loader is not None
    copia = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(copia)

    class ErroHttp(Exception):
        status_code = 503

    assert copia._ModelError is None
    assert copia._ModelRateLimitError is None
    assert copia.is_recoverable_error(ErroHttp("service unavailable")) is True
    assert copia.is_rate_limit_error(ErroHttp("service unavailable")) is False


def test_403_escolhe_a_caixa_de_permissao():
    mensagem = get_friendly_error_message(Exception("403 Forbidden: model access"), "openai")
    assert "ERRO DE PERMISSÃO" in mensagem


def test_timeout_escolhe_a_caixa_de_tempo_esgotado():
    mensagem = get_friendly_error_message(TimeoutError("request timed out"), "openai")
    assert "TEMPO ESGOTADO" in mensagem


# =============================================================================
# Caixa genérica
# =============================================================================


def _detalhes(mensagem):
    """Conteúdo das linhas da caixa que vão de 'Detalhes:' até a linha em branco."""
    linhas = [linha.strip("║").strip() for linha in mensagem.splitlines()]
    inicio = next(i for i, linha in enumerate(linhas) if linha.startswith("Detalhes:"))
    fim = linhas.index("", inicio)
    return [linhas[inicio].removeprefix("Detalhes:").strip(), *linhas[inicio + 1 : fim]]


def _caixa_generica(texto):
    mensagem = get_friendly_error_message(RuntimeError(texto), "openai")
    assert "ERRO NO PROCESSAMENTO" in mensagem
    return mensagem


def test_caixa_generica_do_claude_code_mostra_o_texto_do_sdk():
    resultado = SimpleNamespace(
        is_error=True,
        subtype="error_during_execution",
        api_error_status=None,
        errors=None,
        result="Invalid API key · Please run /login",
    )
    with pytest.raises(ProviderTransientError) as erro:
        _raise_for_result_error(resultado)

    detalhes = " ".join(_detalhes(get_friendly_error_message(erro.value, "claude_code")))

    assert detalhes == str(erro.value)
    assert "Invalid API key · Please run /login" in detalhes


def test_caixa_generica_mostra_mensagem_curta_numa_linha():
    assert _detalhes(_caixa_generica("falhou")) == ["falhou"]


def test_caixa_generica_quebra_a_mensagem_em_ate_tres_linhas_com_reticencias():
    texto = " ".join(f"palavra{i}" for i in range(60))

    detalhes = _detalhes(_caixa_generica(texto))

    assert len(detalhes) == 3
    assert detalhes[-1].endswith(" ...")
    assert texto.startswith(" ".join(detalhes).removesuffix(" ..."))


def test_caixa_generica_sem_corte_nao_poe_reticencias():
    texto = " ".join(["palavra"] * 20)

    detalhes = _detalhes(_caixa_generica(texto))

    assert len(detalhes) == 3
    assert " ".join(detalhes) == texto


@pytest.mark.parametrize(
    "texto",
    [
        "",
        "falhou",
        "linha um\nlinha dois\tcom tabulação",
        "x" * 300,
        " ".join(["palavra"] * 100),
        "Claude Code SDK retornou erro (error_during_execution, status None): " + "y" * 200,
    ],
    ids=["vazia", "curta", "quebra-de-linha", "palavra-longa", "longa", "prefixo-do-claude-code"],
)
def test_caixa_generica_mantem_a_moldura_alinhada(texto):
    linhas = _caixa_generica(texto).splitlines()

    assert {len(linha) for linha in linhas} == {len(linhas[0])}
    assert all(linha.startswith(("║", "╔", "╠", "╚")) for linha in linhas)


# =============================================================================
# Status HTTP nos backends sem LangChain
# =============================================================================


class _ErroDeSdk(Exception):
    """Como o erro de SDK que chega pelo LangChain, com o status em status_code."""

    def __init__(self, mensagem, status_code):
        super().__init__(mensagem)
        self.status_code = status_code


def _erro_do_claude_code(status):
    resultado = SimpleNamespace(
        is_error=True,
        subtype="error_during_execution",
        api_error_status=status,
        errors=None,
        result="falhou",
    )
    try:
        _raise_for_result_error(resultado)
    except ProviderError as erro:
        return erro
    pytest.fail("o resultado com erro não levantou")


# Cada status de erro HTTP é recuperável no claude_code se, e só se, o é no erro
# de SDK que chega pelo LangChain.
@pytest.mark.parametrize("status", range(400, 600))
def test_claude_code_repete_os_mesmos_status_que_o_langchain(status):
    erro = _erro_do_claude_code(status)

    assert erro.status_code == status
    esperado = is_recoverable_error(_ErroDeSdk("falhou", status))
    assert is_recoverable_error(erro) is esperado


@pytest.mark.parametrize("status", range(400, 600))
def test_codex_repete_os_mesmos_status_que_o_langchain(status):
    gerado = pytest.importorskip("openai_codex.generated.v2_all")
    from dataframeit.codex import _raise_turn_error  # noqa: PLC0415 (extra codex opcional)

    turno = gerado.TurnError(
        message="falhou",
        codexErrorInfo=gerado.CodexErrorInfo(
            root=gerado.HttpConnectionFailedCodexErrorInfo(
                httpConnectionFailed=gerado.HttpConnectionFailed(httpStatusCode=status)
            )
        ),
    )
    with pytest.raises(ProviderError) as erro:
        _raise_turn_error(turno)

    assert erro.value.status_code == status
    esperado = is_recoverable_error(_ErroDeSdk("falhou", status))
    assert is_recoverable_error(erro.value) is esperado


# =============================================================================
# Dependências do provider
# =============================================================================


@pytest.mark.parametrize(
    ("ausente", "pacote_na_mensagem"),
    [("langchain", "langchain"), ("langchain_core", "langchain-core")],
)
def test_langchain_ausente_pede_a_instalacao(monkeypatch, ausente, pacote_na_mensagem):
    import_module = importlib.import_module

    def importar(nome):
        if nome == ausente:
            raise ImportError(nome)
        return import_module(nome)

    monkeypatch.setattr(importlib, "import_module", importar)

    with pytest.raises(ImportError, match=f"pip install {pacote_na_mensagem} ") as exc_info:
        validate_provider_dependencies("openai")

    assert "dataframeit[all]" in str(exc_info.value)


def test_sem_provider_valida_so_o_langchain(monkeypatch):
    """Com provider=None, o init_chat_model infere o provider pelo nome do modelo."""
    importados = []
    monkeypatch.setattr(importlib, "import_module", importados.append)

    validate_provider_dependencies(None)

    assert importados == ["langchain", "langchain_core"]
