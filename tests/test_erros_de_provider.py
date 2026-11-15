"""Classificação e mensagens de erro dos providers."""

import copy
import importlib
import importlib.metadata
import importlib.util
import pickle
import tomllib
import unicodedata
from pathlib import Path
from types import SimpleNamespace

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from dataframeit import errors
from dataframeit.claude_code import _raise_for_result_error
from dataframeit.errors import (
    ProviderAbortError,
    ProviderError,
    ProviderTransientError,
    ProviderUsageLimitError,
    get_friendly_error_message,
    is_rate_limit_error,
    is_recoverable_error,
    retry_with_backoff,
    validate_provider_dependencies,
    validate_search_dependencies,
)
from dataframeit.search import SearchToolError

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


@pytest.mark.parametrize(
    "texto",
    [
        "exa: monthly quota exceeded",
        "UsageLimitExceededError: plan usage exceeded for tavily",
        "Usage limit exceeded for Tavily",
    ],
)
def test_limite_de_uso_citado_so_no_texto_nao_escolhe_a_caixa_de_cota(texto):
    # A cota da busca vira ProviderUsageLimitError na ferramenta, pelo status; o
    # erro que só a cita no texto não interrompeu a execução.
    mensagem = get_friendly_error_message(Exception(texto), "openai")
    assert "ERRO NO PROCESSAMENTO" in mensagem
    assert "COTA DE USO" not in mensagem


def test_exa_dentro_de_palavra_nao_escolhe_a_caixa_do_exa():
    mensagem = get_friendly_error_message(Exception("hexagonal api_key rotation"), "openai")
    assert "EXA" not in mensagem


def _erro_do_tavily(status, texto):
    # Como _levantar_erro do provedor Tavily o monta: o status só no texto do cliente.
    return SearchToolError("tavily", ValueError(f"Error {status}: {texto}"), status)


def _erro_do_exa(status, texto):
    return SearchToolError(
        "exa", ValueError(f"Request failed with status code {status}: {texto}"), status
    )


def test_chave_invalida_da_busca_aponta_a_chave_do_tavily_e_nao_a_do_llm():
    # O cliente do Tavily não nomeia o serviço no texto.
    mensagem = get_friendly_error_message(_erro_do_tavily(401, "Unauthorized"), "openai")
    assert "AUTENTICAÇÃO TAVILY" in mensagem
    assert "TAVILY_API_KEY" in mensagem
    assert "OPENAI_API_KEY" not in mensagem


def test_falha_da_busca_sem_status_e_sem_caixa_propria_aponta_a_chave_da_busca():
    erro = SearchToolError("tavily", ValueError("401 unauthorized"))
    mensagem = get_friendly_error_message(erro, "openai")
    assert "AUTENTICAÇÃO" in mensagem
    assert "TAVILY_API_KEY" in mensagem
    assert "OPENAI_API_KEY" not in mensagem


def test_chave_invalida_real_do_exa_escolhe_a_caixa_do_exa():
    erro = _erro_do_exa(401, '{"error":"Unauthorized"}')
    mensagem = get_friendly_error_message(erro, "anthropic")
    assert "AUTENTICAÇÃO EXA" in mensagem
    assert "ANTHROPIC_API_KEY" not in mensagem


@pytest.mark.parametrize(
    "erro",
    [
        _erro_do_tavily(500, "Invalid upstream response"),
        _erro_do_tavily(500, "rate limit of upstream"),
        _erro_do_exa(400, "Invalid numResults"),
        _erro_do_exa(400, "numResults limit is 100"),
    ],
)
def test_status_da_busca_decide_a_caixa_e_nao_o_texto(erro):
    mensagem = get_friendly_error_message(erro, "openai")
    assert "AUTENTICAÇÃO" not in mensagem
    assert "COTA DE USO" not in mensagem


def test_rate_limit_da_busca_nomeia_o_servico_da_busca():
    mensagem = get_friendly_error_message(_erro_do_exa(429, "slow down"), "openai")
    assert "LIMITE DE REQUISIÇÕES" in mensagem
    assert "para o Exa" in mensagem


def test_erro_da_busca_sobrevive_a_pickle_e_copy():
    erro = _erro_do_tavily(401, "Unauthorized")
    copias = (pickle.loads(pickle.dumps(erro)), copy.copy(erro))  # noqa: S301 (dado do próprio teste)
    for copia in copias:
        assert (copia.provider, copia.status_code, str(copia)) == (
            "tavily",
            401,
            "ValueError: Error 401: Unauthorized",
        )


# =============================================================================
# Cota de uso esgotada
# =============================================================================


def _falha_sempre(erro):
    chamadas = []

    def func():
        chamadas.append(1)
        raise erro

    return chamadas, func


@pytest.mark.parametrize("classe", [ProviderAbortError, ProviderUsageLimitError])
def test_parada_deliberada_sobe_sem_nova_tentativa_e_sem_aviso(classe, recwarn):
    erro = classe("app-server encerrado")
    chamadas, func = _falha_sempre(erro)

    with pytest.raises(classe) as levantado:
        retry_with_backoff(func, max_retries=3, base_delay=0.0)

    assert levantado.value is erro
    assert len(chamadas) == 1
    assert len(recwarn) == 0


def _cota_esgotada_da_openai():
    httpx = pytest.importorskip("httpx")
    openai = pytest.importorskip("openai")
    requisicao = httpx.Request("POST", "https://api.openai.com/v1/x")
    return openai.RateLimitError(
        "Error code: 429 - {'error': {'message': 'You exceeded your current quota', "
        "'type': 'insufficient_quota', 'code': 'insufficient_quota'}}",
        response=httpx.Response(429, request=requisicao),
        body={"code": "insufficient_quota"},
    )


def test_cota_esgotada_da_openai_interrompe_sem_nova_tentativa():
    erro = _cota_esgotada_da_openai()
    chamadas, func = _falha_sempre(erro)

    with pytest.raises(ProviderUsageLimitError, match="adicione crédito") as levantado:
        retry_with_backoff(func, max_retries=3, base_delay=0.0)

    assert len(chamadas) == 1
    assert levantado.value.__cause__ is erro


def _embrulhada(causa):
    """Como um `raise ErroDaIntegracao(...) from causa`."""
    erro = Exception("chamada ao modelo falhou")
    erro.__cause__ = causa
    return erro


def test_cota_esgotada_embrulhada_por_outra_excecao_tambem_interrompe():
    chamadas, func = _falha_sempre(_embrulhada(_cota_esgotada_da_openai()))

    with pytest.raises(ProviderUsageLimitError):
        retry_with_backoff(func, max_retries=3, base_delay=0.0)

    assert len(chamadas) == 1


@pytest.mark.parametrize(
    ("fabrica", "detalhe"),
    [
        (_cota_esgotada_da_openai, "provider de LLM esgotada"),
        (lambda: _embrulhada(_cota_esgotada_da_openai()), "provider de LLM esgotada"),
        (
            lambda: ProviderUsageLimitError("limite de uso do plano do Tavily esgotado"),
            "plano do Tavily",
        ),
    ],
    ids=["insufficient-quota", "insufficient-quota-embrulhada", "provider-usage-limit-error"],
)
def test_cota_esgotada_escolhe_a_caixa_de_cota_e_nao_a_de_rate_limit(fabrica, detalhe):
    mensagem = get_friendly_error_message(fabrica(), "openai")

    assert "COTA DE USO ESGOTADA" in mensagem
    assert "LIMITE DE REQUISIÇÕES" not in mensagem
    assert detalhe in mensagem


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    ("erro", "tentativas"),
    [
        (_erro_do_tavily(500, "Internal server error."), 2),
        (_erro_do_exa(402, "x"), 1),
    ],
)
def test_outras_falhas_da_busca_nao_viram_limite_de_uso(erro, tentativas):
    chamadas, func = _falha_sempre(erro)

    with pytest.raises(SearchToolError):
        retry_with_backoff(func, max_retries=2, base_delay=0.0)

    assert len(chamadas) == tentativas


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


def test_sobrecarga_529_reduz_workers(model_error_falso):
    class Sobrecarga(Exception):
        status_code = 529

    class SobrecargaDoLangChain(_ModelErrorFalso):
        # O langchain-core classifica o 529 como ModelAPIError, e não como rate limit.
        status_code = 529

    assert is_rate_limit_error(Sobrecarga("overloaded")) is True
    assert is_rate_limit_error(SobrecargaDoLangChain("overloaded")) is True
    assert is_recoverable_error(Sobrecarga("overloaded")) is True


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


def test_caixa_generica_quebra_a_mensagem_em_ate_tres_linhas_com_reticencias():
    texto = " ".join(f"palavra{i}" for i in range(60))

    detalhes = _detalhes(_caixa_generica(texto))

    assert len(detalhes) == 3
    assert detalhes[-1].endswith(" ...")
    assert texto.startswith(" ".join(detalhes).removesuffix(" ..."))


@pytest.mark.parametrize(
    ("texto", "linhas"),
    [("falhou", 1), (" ".join(["palavra"] * 20), 3)],
    ids=["curta", "tres-linhas-exatas"],
)
def test_caixa_generica_sem_corte_nao_poe_reticencias(texto, linhas):
    detalhes = _detalhes(_caixa_generica(texto))

    assert len(detalhes) == linhas
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
        "falhou 🔥 " * 30,
        "接続がタイムアウトしました。" * 15,
        "火" * 300,
        unicodedata.normalize("NFD", "ação não concluída, tente outra vez " * 6),
        "x\u0301y\u0302 sem forma composta " * 8,
        "\x1b[31merro vermelho\x1b[0m " * 12,
        "linha um\u2028linha dois\x85linha três\u2029fim",
        "zero\u200blargura\u200dcom\x07controle",
    ],
    ids=[
        "vazia",
        "curta",
        "quebra-de-linha",
        "palavra-longa",
        "longa",
        "prefixo-do-claude-code",
        "emoji",
        "cjk",
        "palavra-longa-cjk",
        "acento-decomposto",
        "combinante-sem-forma-composta",
        "ansi",
        "separadores-unicode",
        "largura-zero-e-controle",
    ],
)
def test_caixa_generica_mantem_a_moldura_alinhada(texto):
    linhas = _caixa_generica(texto).splitlines()

    assert {_largura_exibida(linha) for linha in linhas} == {80}
    assert all(linha.startswith(("║", "╔", "╠", "╚")) for linha in linhas)


def _largura_exibida(texto):
    """Colunas que o texto ocupa no terminal: largo e cheio contam 2, marca combinante 0."""

    def colunas(char):
        if unicodedata.category(char) in {"Mn", "Me", "Cf", "Cc"}:
            return 0
        if unicodedata.east_asian_width(char) in {"W", "F"}:
            return 2
        return 1

    return sum(map(colunas, texto))


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [
        (unicodedata.normalize("NFD", "ação") + " \x1b[31m接続\x1b[0m\u2028🔥", "ação 接続 🔥"),
        ("\x1b]8;;https://x.io\x07link\x1b]8;;\x1b\\ ok", "link ok"),
        ("\x1b(Bsem\x1b7 escape\x1b=", "sem escape"),
        ("osc \x1b]sem terminador fica", "osc sem terminador fica"),
        ("byte \udcff solto", "byte solto"),
    ],
    ids=[
        "nfc-ansi-separador",
        "osc-hiperlink",
        "escapes-curtos",
        "osc-sem-terminador",
        "surrogate",
    ],
)
def test_caixa_generica_mostra_o_texto_sem_controle_e_em_nfc(texto, esperado):
    mensagem = _caixa_generica(texto)

    assert _detalhes(mensagem) == [esperado]
    mensagem.encode("utf-8")


def test_largura_exibida_de_marca_que_envolve_e_de_caractere_cheio():
    # U+20DD envolve o caractere anterior e não ocupa coluna; U+FF21, o A cheio,
    # ocupa duas.
    assert errors._display_width("a\u20dd") == 1
    assert errors._display_width("\uff21") == 2


def test_texto_que_cabe_exato_na_largura_nao_e_cortado():
    assert errors._cut_display("abc", 3) == "abc"


def test_primeira_palavra_da_linha_nao_ganha_espaco_antes():
    assert errors._wrap_display("um dois", 20, 3) == ["um dois"]


@pytest.mark.parametrize(
    ("texto", "esperado"),
    [
        # A última linha perde palavras inteiras, uma de cada vez, até caber ' ...'.
        ("aaaa bbbb cccc dddd", ["aaaa ..."]),
        ("aa bb cccc dddddddddd", ["aa bb ..."]),
        # A que cabe exato na largura que sobra fica inteira.
        ("aaa bbbb cccccccc", ["aaa bbbb ..."]),
    ],
)
def test_corte_da_ultima_linha_tira_palavras_inteiras(texto, esperado):
    assert errors._wrap_display(texto, 12, 1) == esperado


def test_caixa_generica_de_erro_sem_mensagem_deixa_os_detalhes_em_branco():
    assert _detalhes(_caixa_generica("")) == [""]


def test_caixa_generica_alinha_o_tipo_com_nome_largo():
    class 接続エラー(Exception):  # noqa: N801, PLC2401 (nome largo de propósito)
        pass

    linhas = get_friendly_error_message(接続エラー("falhou"), "openai").splitlines()

    assert "Tipo: 接続エラー" in linhas[4]
    assert {_largura_exibida(linha) for linha in linhas} == {80}


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


def _sem_o_pacote(monkeypatch, pacote):
    import_module = importlib.import_module

    def importar(nome):
        if nome == pacote:
            raise ImportError(nome)
        return import_module(nome)

    monkeypatch.setattr(importlib, "import_module", importar)


def test_sem_metadados_do_pacote_a_dica_do_extra_all_some(monkeypatch):
    """Sem os metadados, não há como saber o que o extra traz, e a dica não sai."""

    def sem_metadados(nome):
        raise importlib.metadata.PackageNotFoundError(nome)

    monkeypatch.setattr(importlib.metadata, "requires", sem_metadados)
    _sem_o_pacote(monkeypatch, "langchain_openai")

    with pytest.raises(ImportError, match="pip install langchain-openai ") as exc_info:
        validate_provider_dependencies("openai")

    assert "dataframeit[all]" not in str(exc_info.value)


def _linha_do_pip(instalar):
    return f"║      pip install {instalar:<62} ║"


def _linha_da_biblioteca(pacote, nome):
    return f"║  A biblioteca '{pacote}' é necessária para usar {nome}.            ║"


@pytest.mark.parametrize(
    ("provider", "ausente", "instalar", "nome", "alternativa"),
    [
        ("claude_code", "claude_agent_sdk", "dataframeit[claude-code]", "Claude Code", None),
        ("openai", "langchain", "langchain", "LangChain", "dataframeit[all]"),
        ("openai", "langchain_core", "langchain-core", "LangChain Core", "dataframeit[all]"),
        ("openai", "langchain_openai", "langchain-openai", "OpenAI", "dataframeit[all]"),
        (
            "azure_openai",
            "langchain_openai",
            "langchain-openai",
            "Azure OpenAI",
            "dataframeit[all]",
        ),
        # O extra all só traz os providers que têm extra próprio.
        ("mistralai", "langchain_mistralai", "langchain-mistralai", "Mistral AI", None),
        ("cohere", "langchain_cohere", "langchain-cohere", "Cohere", None),
        ("bedrock", "langchain_aws", "langchain-aws", "AWS Bedrock", None),
    ],
)
def test_pacote_ausente_nomeia_o_pacote_o_provider_e_o_comando(
    provider, ausente, instalar, nome, alternativa
):
    with pytest.MonkeyPatch.context() as monkeypatch:
        _sem_o_pacote(monkeypatch, ausente)
        with pytest.raises(ImportError) as exc_info:
            validate_provider_dependencies(provider)

    linhas = str(exc_info.value).splitlines()
    assert _linha_da_biblioteca(ausente, nome) in linhas
    pip = linhas.index(_linha_do_pip(instalar))
    if alternativa is None:
        # Sem alternativa, a linha em branco depois do comando leva direto ao fim.
        assert linhas[pip + 2].startswith("║  Após instalar, execute seu código novamente.")
        assert not any("Ou, para instalar" in linha for linha in linhas)
    else:
        assert _linha_do_pip(alternativa) in linhas


@pytest.mark.parametrize(
    ("search_provider", "pacote", "instalar", "nome"),
    [
        ("tavily", "langchain_tavily", "langchain-tavily", "Tavily Search"),
        ("exa", "langchain_exa", "langchain-exa", "Exa Search"),
    ],
)
def test_pacote_de_busca_ausente_nomeia_o_pacote_e_o_servico(
    monkeypatch, search_provider, pacote, instalar, nome
):
    _sem_o_pacote(monkeypatch, pacote)

    with pytest.raises(ImportError) as exc_info:
        validate_search_dependencies(search_provider)

    linhas = str(exc_info.value).splitlines()
    assert _linha_da_biblioteca(pacote, nome) in linhas
    assert _linha_do_pip(instalar) in linhas


def test_pacotes_do_extra_all_lidos_dos_metadados_batem_com_o_pyproject():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    declarados = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"][
        "optional-dependencies"
    ]["all"]

    assert errors._all_extra_packages() == {
        canonicalize_name(Requirement(r).name) for r in declarados
    }


def test_sem_provider_valida_so_o_langchain(monkeypatch):
    """Com provider=None, o init_chat_model infere o provider pelo nome do modelo."""
    importados = []
    monkeypatch.setattr(importlib, "import_module", importados.append)

    validate_provider_dependencies(None)

    assert importados == ["langchain", "langchain_core"]
