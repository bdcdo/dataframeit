"""Testes para inferência de provider em dataframeit.errors."""

import pytest
from pydantic import ValidationError, create_model

from dataframeit.errors import (
    MAX_REJECTED_RULES,
    ProviderError,
    ProviderOutputError,
    ProviderRejectedOutputError,
    ProviderTransientError,
    _infer_provider_info,
    get_friendly_error_message,
    is_rate_limit_error,
    retry_with_backoff,
    validation_error_summary,
)


class TestInferProviderInfoOverrides:
    """Overrides para providers cloud (Vertex AI, Bedrock, Azure OpenAI)."""

    def test_google_vertexai(self):
        info = _infer_provider_info("google_vertexai")
        assert info["package"] == "langchain_google_vertexai"
        assert info["install"] == "langchain-google-vertexai"
        assert info["env_var"] is None
        assert info["name"] == "Google Vertex AI"
        assert "gcloud" in info["auth_hint"]

    def test_bedrock(self):
        info = _infer_provider_info("bedrock")
        assert info["package"] == "langchain_aws"
        assert info["install"] == "langchain-aws"
        assert info["env_var"] is None
        assert info["name"] == "AWS Bedrock"
        assert "aws" in info["auth_hint"].lower()

    def test_bedrock_converse(self):
        info = _infer_provider_info("bedrock_converse")
        assert info["package"] == "langchain_aws"
        assert info["install"] == "langchain-aws"
        assert info["env_var"] is None
        assert info["name"] == "AWS Bedrock (Converse)"

    def test_azure_openai(self):
        info = _infer_provider_info("azure_openai")
        assert info["package"] == "langchain_openai"
        assert info["install"] == "langchain-openai"
        assert info["env_var"] == "AZURE_OPENAI_API_KEY"
        assert info["name"] == "Azure OpenAI"
        assert "AZURE_OPENAI_ENDPOINT" in info["auth_hint"]


class TestInferProviderInfoHeuristic:
    """A heurística genérica continua valendo para providers sem override."""

    def test_google_genai(self):
        info = _infer_provider_info("google_genai")
        assert info["package"] == "langchain_google_genai"
        assert info["install"] == "langchain-google-genai"
        assert info["env_var"] == "GOOGLE_API_KEY"
        assert info["name"] == "Google Gemini"

    def test_openai(self):
        info = _infer_provider_info("openai")
        assert info["package"] == "langchain_openai"
        assert info["env_var"] == "OPENAI_API_KEY"

    def test_anthropic(self):
        info = _infer_provider_info("anthropic")
        assert info["package"] == "langchain_anthropic"
        assert info["env_var"] == "ANTHROPIC_API_KEY"

    def test_unknown_provider_falls_back(self):
        info = _infer_provider_info("foo_bar")
        assert info["package"] == "langchain_foo_bar"
        assert info["install"] == "langchain-foo-bar"
        assert info["env_var"] == "FOO_BAR_API_KEY"
        assert info["name"] == "Foo Bar"

    def test_empty_provider(self):
        info = _infer_provider_info("")
        assert info["package"] is None
        assert info["name"] == "LLM"


class TestFriendlyAuthErrorWithoutEnvVar:
    """Mensagem de auth deve adaptar quando provider usa credenciais SDK."""

    def test_vertexai_auth_error_uses_sdk_message(self):

        msg = get_friendly_error_message(
            Exception("AuthenticationError: invalid credentials"),
            provider="google_vertexai",
        )
        assert "Google Vertex AI" in msg
        assert "gcloud auth application-default login" in msg
        # Não deve sugerir export de API key inexistente
        assert "GOOGLE_VERTEXAI_API_KEY" not in msg

    def test_bedrock_auth_error_uses_sdk_message(self):

        msg = get_friendly_error_message(
            Exception("AuthenticationError: 401"),
            provider="bedrock_converse",
        )
        assert "aws configure" in msg.lower() or "AWS_ACCESS_KEY_ID" in msg
        assert "BEDROCK_CONVERSE_API_KEY" not in msg

    def test_openai_auth_error_keeps_api_key_message(self):

        msg = get_friendly_error_message(
            Exception("AuthenticationError: bad key"),
            provider="openai",
        )
        assert "OPENAI_API_KEY" in msg
        assert "export OPENAI_API_KEY" in msg


class _ResultadoQueFalhaAoAnotar(dict):
    """Dict cuja primeira gravação de chave levanta um erro transitório."""

    falhas = 0

    def __setitem__(self, chave, valor):
        if type(self).falhas == 0:
            type(self).falhas += 1
            msg = "falha ao anotar o resultado"
            raise ProviderTransientError(msg)
        super().__setitem__(chave, valor)


def test_falha_ao_anotar_o_resultado_ganha_nova_tentativa():
    chamadas = []

    def funcao():
        chamadas.append(1)
        return _ResultadoQueFalhaAoAnotar()

    with pytest.warns(UserWarning, match="Tentativa 1/2 falhou"):
        resultado = retry_with_backoff(funcao, max_retries=2, base_delay=0, max_delay=0)

    assert len(chamadas) == 2
    assert resultado["_retry_info"]["retries"] == 1


# O resumo de um erro de validação pode trazer texto da resposta, como a
# mensagem de um validador próprio; um número nele não pode decidir a categoria
# do erro.
_RESUMO_COM_NUMERO = (
    "Resposta do Codex não corresponde ao schema: 1 erro(s): "
    "fundamento: Value error, art. {numero} da CLT não consta da lista"
)


def test_resposta_fora_do_contrato_nao_vira_erro_de_credencial():
    erro = ProviderOutputError(_RESUMO_COM_NUMERO.format(numero=401))

    amigavel = get_friendly_error_message(erro, "codex")

    assert amigavel.startswith("RESPOSTA FORA DO CONTRATO DE SAÍDA")
    assert "login" not in amigavel


def test_mensagem_da_resposta_fora_do_contrato_traz_o_detalhe_do_erro():
    erro = ProviderOutputError(_RESUMO_COM_NUMERO.format(numero=401))
    amigavel = get_friendly_error_message(erro, "codex")

    assert str(erro) in amigavel
    assert "não é tentada de novo" in amigavel


def test_resumo_da_validacao_conta_todos_os_erros_e_lista_ate_o_teto():
    campos = {f"campo_{i}": (int, ...) for i in range(MAX_REJECTED_RULES + 5)}
    # Campos por **kwargs, que o ty não casa com as sobrecargas de create_model.
    modelo = create_model("MuitosCampos", **campos)  # ty: ignore[no-matching-overload]
    with pytest.raises(ValidationError) as recusa:
        modelo.model_validate({})

    resumo = validation_error_summary(recusa.value)

    assert resumo.startswith(f"{MAX_REJECTED_RULES + 5} erro(s): ")
    assert resumo.count("Field required") == MAX_REJECTED_RULES


# Só os providers do LangChain levam o erro de volta ao modelo; o claude_code
# repete o pedido, e a mensagem da classe vale para os dois caminhos.
def test_mensagem_da_resposta_recusada_nao_promete_retorno_ao_modelo():
    erro = ProviderRejectedOutputError("Falha na validação da resposta: 1 erro(s): campo: regra")

    amigavel = get_friendly_error_message(erro, "claude_code")

    assert str(erro) in amigavel
    assert "ao modelo" not in amigavel.lower()


def test_numero_no_resumo_da_validacao_nao_conta_como_rate_limit():
    erro = ProviderOutputError(_RESUMO_COM_NUMERO.format(numero=429))

    assert is_rate_limit_error(erro) is False


# O turno do codex que falha sem código classificável só traz o 429 no texto, e
# a redução de workers depende de a mensagem ser lida.
def test_erro_definitivo_com_429_na_mensagem_conta_como_rate_limit():
    erro = ProviderError("Turno Codex falhou: 429 Too Many Requests")

    assert is_rate_limit_error(erro) is True
