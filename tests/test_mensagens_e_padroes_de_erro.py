"""Texto das mensagens de errors.py e cada padrão que decide uma classificação.

Os testes vieram dos mutantes que sobreviviam ao teste de mutação de errors.py:
cada padrão de texto decide a caixa sozinho, e cada mensagem traz o pacote, o
nome e o comando certos, e não só algum deles.
"""

import importlib
import random
import time

import pytest
from pydantic import BaseModel, ValidationError

from dataframeit import errors
from dataframeit.errors import (
    ProviderOutputError,
    ProviderRejectedOutputError,
    _infer_provider_info,
    get_friendly_error_message,
    is_rate_limit_error,
    retry_with_backoff,
    validate_search_dependencies,
    validation_error_summary,
)
from dataframeit.search import SearchToolError

# =============================================================================
# Resumo da validação
# =============================================================================


class _Item(BaseModel):
    n: int


class _Resposta(BaseModel):
    itens: list[_Item]
    nome: str


def test_resumo_da_validacao_junta_o_caminho_com_ponto_e_as_regras_com_ponto_e_virgula():
    with pytest.raises(ValidationError) as exc_info:
        _Resposta.model_validate({"itens": [{"n": "x"}]})
    regra_do_item, regra_do_nome = (item["msg"] for item in exc_info.value.errors())

    assert validation_error_summary(exc_info.value) == (
        f"2 erro(s): itens.0.n: {regra_do_item}; nome: {regra_do_nome}"
    )


# =============================================================================
# Pacote ausente
# =============================================================================


def test_chave_de_busca_ausente_traz_o_servico_em_maiusculas(monkeypatch):
    monkeypatch.setattr(importlib, "import_module", lambda nome: None)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    with pytest.raises(ValueError, match="CHAVE DE API DO TAVILY SEARCH NÃO CONFIGURADA"):
        validate_search_dependencies("tavily")


def test_nome_de_distribuicao_segue_a_normalizacao_da_pep_503():
    assert errors._normalize_distribution_name("LangChain_OpenAI.Beta--x") == (
        "langchain-openai-beta-x"
    )


# =============================================================================
# Provider
# =============================================================================


def test_sem_provider_a_chave_e_a_generica():
    assert _infer_provider_info(None) == {
        "package": None,
        "install": None,
        "env_var": "API_KEY",
        "name": "LLM",
    }


@pytest.mark.parametrize(
    ("provider", "env_var", "nome"),
    [
        ("google_genai", "GOOGLE_API_KEY", "Google Gemini"),
        ("openai", "OPENAI_API_KEY", "OpenAI"),
        ("anthropic", "ANTHROPIC_API_KEY", "Anthropic Claude"),
        ("cohere", "COHERE_API_KEY", "Cohere"),
        ("fireworks", "FIREWORKS_API_KEY", "Fireworks AI"),
        ("together", "TOGETHER_API_KEY", "Together AI"),
        ("groq", "GROQ_API_KEY", "Groq"),
        ("exemplo_ai", "EXEMPLO_API_KEY", "Exemplo Ai"),
    ],
)
def test_caixa_de_chave_invalida_traz_o_nome_e_a_variavel_do_provider(provider, env_var, nome):
    mensagem = get_friendly_error_message(Exception("api_key"), provider)

    assert f"║  Provider: {nome:<66}║" in mensagem
    assert f'export {env_var}="sua-chave-aqui"' in mensagem


def test_todo_provider_sem_variavel_de_chave_traz_a_dica_de_autenticacao():
    sem_variavel = {
        nome: dados
        for nome, dados in errors._PROVIDER_OVERRIDES.items()
        if dados["env_var"] is None
    }

    assert sem_variavel
    assert all(dados["auth_hint"] for dados in sem_variavel.values())


def test_dica_de_autenticacao_de_varias_linhas_sai_uma_por_linha_da_caixa():
    linhas = get_friendly_error_message(Exception("api_key"), "bedrock").splitlines()

    dica = [
        "aws configure",
        "OU exporte AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY",
        "(opcional: AWS_SESSION_TOKEN, AWS_REGION)",
    ]
    inicio = linhas.index(f"║     {dica[0]:<73}║")
    assert linhas[inicio : inicio + 3] == [f"║     {linha:<73}║" for linha in dica]


# =============================================================================
# Padrões de cada caixa
# =============================================================================


@pytest.mark.parametrize("padrao", ["ratelimit", "resourceexhausted", "toomanyrequests", "429"])
def test_cada_padrao_de_rate_limit_basta(padrao):
    assert is_rate_limit_error(Exception(f"falha {padrao}"))


def test_sem_padrao_de_rate_limit_o_erro_nao_e_de_rate_limit():
    assert not is_rate_limit_error(Exception("falha qualquer"))


@pytest.mark.parametrize(
    ("texto", "titulo"),
    [
        *(
            (f"tavily {padrao}", "ERRO DE AUTENTICAÇÃO TAVILY")
            for padrao in ["apikey", "api_key", "missing", "invalid"]
        ),
        *(
            (f"exa {padrao}", "ERRO DE AUTENTICAÇÃO EXA")
            for padrao in ["apikey", "api_key", "missing", "invalid", "unauthorized"]
        ),
        *(
            (padrao, "ERRO DE AUTENTICAÇÃO - Chave de API")
            for padrao in ["authenticationerror", "invalidapikey", "api_key", "api key"]
        ),
        *((padrao, "ERRO DE PERMISSÃO") for padrao in ["permissiondenied", "forbidden"]),
        *((padrao, "TEMPO ESGOTADO") for padrao in ["timeout", "deadlineexceeded"]),
        *(
            (padrao, "ERRO DE CONEXÃO")
            for padrao in ["connectionerror", "connectionreset", "sslerror", "network"]
        ),
    ],
)
def test_cada_padrao_de_texto_decide_a_caixa_sozinho(texto, titulo):
    assert titulo in get_friendly_error_message(Exception(texto), "openai")


def test_permissao_negada_na_busca_nomeia_o_provedor_de_busca():
    erro = SearchToolError("exa", Exception("Error 403"), status_code=403)

    mensagem = get_friendly_error_message(erro, "openai")

    assert "ERRO DE PERMISSÃO" in mensagem
    assert "Sua chave de API do Exa Search não tem permissão" in mensagem
    assert "OpenAI" not in mensagem


def test_caixa_generica_traz_o_tipo_do_erro():
    class ErroQualquer(Exception):
        pass

    mensagem = get_friendly_error_message(ErroQualquer("algo deu errado"), "openai")

    assert f"║  Tipo: {'ErroQualquer':<69} ║" in mensagem.splitlines()


def test_caixa_generica_de_erro_sem_mensagem_deixa_os_detalhes_em_branco():
    mensagem = get_friendly_error_message(Exception(), "openai")

    assert f"║  Detalhes: {'':<65} ║" in mensagem.splitlines()


def test_resposta_recusada_e_fora_do_contrato_tem_o_texto_completo():
    assert get_friendly_error_message(ProviderRejectedOutputError("regra x")) == (
        "RESPOSTA RECUSADA PELA VALIDAÇÃO DO MODELO PYDANTIC\n"
        "regra x\n"
        "Nenhuma tentativa produziu resposta válida. Veja as regras do modelo que "
        "falharam; se o texto é ambíguo, a instrução do campo pode ajudar."
    )
    assert get_friendly_error_message(ProviderOutputError("sem saída")) == (
        "RESPOSTA FORA DO CONTRATO DE SAÍDA\n"
        "sem saída\n"
        "O provider terminou sem uma resposta que sirva ao modelo Pydantic, e a linha "
        "não é tentada de novo."
    )


def test_status_599_conta_como_status_http_de_erro():
    class ErroHttp(Exception):
        status_code = 599

    assert errors._http_error_status(ErroHttp()) == 599


# =============================================================================
# Novas tentativas
# =============================================================================


def _esperas(monkeypatch):
    esperas = []
    monkeypatch.setattr(time, "sleep", esperas.append)
    monkeypatch.setattr(random, "uniform", lambda *_: 0)
    return esperas


def test_padrao_sao_tres_tentativas_com_espera_de_um_e_dois_segundos(monkeypatch):
    esperas = _esperas(monkeypatch)
    chamadas = []

    def falha():
        chamadas.append(1)
        raise TimeoutError

    with pytest.warns(UserWarning, match="Tentativa"), pytest.raises(TimeoutError):
        retry_with_backoff(falha)

    assert len(chamadas) == 3
    assert esperas == [1.0, 2.0]


def test_espera_padrao_para_em_trinta_segundos(monkeypatch):
    esperas = _esperas(monkeypatch)

    def falha():
        raise TimeoutError

    with pytest.warns(UserWarning, match="Tentativa"), pytest.raises(TimeoutError):
        retry_with_backoff(falha, base_delay=100)

    assert esperas == [30.0, 30.0]


def test_sucesso_na_primeira_tentativa_registra_zero_retries():
    assert retry_with_backoff(lambda: {"data": 1})["_retry_info"] == {"retries": 0}
