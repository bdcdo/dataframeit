"""Testes para o provider claude_code (Claude Code SDK)."""

import asyncio
import json
import logging
import sys
import types
import warnings
from dataclasses import fields, replace
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pydantic import BaseModel

from dataframeit import dataframeit
from dataframeit.claude_code import _build_json_system_prompt, call_claude_code
from dataframeit.core import _format_token_stats
from dataframeit.errors import (
    ProviderError,
    ProviderOverloadedError,
    ProviderRejectedOutputError,
    ProviderTransientError,
    get_friendly_error_message,
    is_rate_limit_error,
    is_recoverable_error,
    validate_provider_dependencies,
)
from dataframeit.llm import LLMConfig
from dataframeit.utils import parse_json


class SampleModel(BaseModel):
    sentimento: str
    confianca: float


class TestValidateProviderDependencies:
    """Testes para validação de dependências do provider claude_code."""

    def test_claude_code_missing_raises_import_error(self):
        """Deve levantar ImportError com mensagem amigável quando claude_agent_sdk não está instalado."""

        with (
            patch("importlib.import_module", side_effect=ImportError("No module")),
            pytest.raises(ImportError, match="claude_agent_sdk"),
        ):
            validate_provider_dependencies("claude_code")

    def test_claude_code_installed_passes(self):
        """Deve passar sem erro quando claude_agent_sdk está instalado."""

        with patch("importlib.import_module", return_value=MagicMock()):
            # Não deve levantar exceção
            validate_provider_dependencies("claude_code")

    def test_claude_code_skips_langchain_validation(self):
        """Deve NÃO validar langchain quando provider é claude_code."""

        call_args = []

        def mock_import(name):
            call_args.append(name)
            return MagicMock()

        with patch("importlib.import_module", side_effect=mock_import):
            validate_provider_dependencies("claude_code")

        # Deve ter importado apenas claude_agent_sdk, não langchain
        assert "claude_agent_sdk" in call_args
        assert "langchain" not in call_args
        assert "langchain_core" not in call_args


class TestUseSearchWithClaudeCode:
    """Testes para validação de use_search com claude_code."""

    def test_use_search_with_claude_code_raises(self):
        """Deve levantar ValueError quando use_search=True com provider='claude_code'."""

        with (
            patch("dataframeit.core.validate_provider_dependencies"),
            pytest.raises(ValueError, match=r"use_search.*claude_code"),
        ):
            dataframeit(
                ["texto teste"],
                questions=SampleModel,
                prompt="Analise: {texto}",
                provider="claude_code",
                model="haiku",
                use_search=True,
            )


class TestBuildJsonSystemPrompt:
    """Testes para geração do system prompt com JSON schema."""

    def test_includes_json_schema(self):
        """System prompt deve incluir o JSON schema do modelo Pydantic."""

        schema = SampleModel.model_json_schema()
        prompt = _build_json_system_prompt(schema)

        assert "sentimento" in prompt
        assert "confianca" in prompt
        assert "JSON Schema" in prompt

    def test_instructs_json_only(self):
        """System prompt deve instruir resposta apenas em JSON."""

        schema = SampleModel.model_json_schema()
        prompt = _build_json_system_prompt(schema)

        assert "APENAS" in prompt or "JSON" in prompt


class TestJsonParsingVariants:
    """Testes para parsing de diferentes formatos de resposta."""

    def test_plain_json(self):
        """Deve parsear JSON puro."""

        result = parse_json('{"sentimento": "positivo", "confianca": 0.95}')
        assert result["sentimento"] == "positivo"
        assert result["confianca"] == 0.95

    def test_json_with_markdown_fences(self):
        """Deve parsear JSON dentro de blocos markdown."""

        response = '```json\n{"sentimento": "negativo", "confianca": 0.8}\n```'
        result = parse_json(response)
        assert result["sentimento"] == "negativo"

    def test_json_with_surrounding_text(self):
        """Deve extrair JSON de resposta com texto ao redor."""

        response = 'Aqui está o resultado: {"sentimento": "neutro", "confianca": 0.5} fim.'
        result = parse_json(response)
        assert result["sentimento"] == "neutro"


# ---------------------------------------------------------------------------
# call_claude_code com SDK falso
# ---------------------------------------------------------------------------
#
# O SDK falso substitui `claude_agent_sdk` em sys.modules, de modo que estes
# testes rodam sem o extra `claude-code` instalado. `ClaudeAgentOptions` só
# guarda os kwargs recebidos, e `query` devolve as mensagens definidas em
# `mensagens_do_sdk`, na mesma forma que o SDK real emite.


class _OpcoesFalsas:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for chave, valor in kwargs.items():
            setattr(self, chave, valor)


class _TextBlockFalso:
    def __init__(self, text):
        self.text = text


class _AssistantMessageFalso:
    def __init__(self, content):
        self.content = content


class _ResultMessageFalso:
    def __init__(self, usage=None, total_cost_usd=None):
        self.usage = usage
        self.total_cost_usd = total_cost_usd


@pytest.fixture
def sdk_falso(monkeypatch):
    """Instala um `claude_agent_sdk` falso e devolve o estado compartilhado."""

    estado: dict[str, Any] = {
        "opcoes": [],
        "mensagens_do_sdk": [
            _AssistantMessageFalso(
                [_TextBlockFalso('{"sentimento": "positivo", "confianca": 0.9}')]
            ),
            _ResultMessageFalso(usage=None),
        ],
    }

    async def query_falsa(prompt, options):
        estado["opcoes"].append(options)
        # 'por_chamada' dá uma lista de mensagens a cada tentativa, em ordem.
        por_chamada = estado.get("por_chamada")
        mensagens = por_chamada.pop(0) if por_chamada else estado["mensagens_do_sdk"]
        for mensagem in mensagens:
            # Uma exceção na lista é levantada naquele ponto do laço, como o SDK
            # faz com a saída do CLI depois do ResultMessage de erro.
            if isinstance(mensagem, BaseException):
                raise mensagem
            yield mensagem

    modulo = types.ModuleType("claude_agent_sdk")
    modulo.__dict__.update(
        ClaudeAgentOptions=_OpcoesFalsas,
        query=query_falsa,
        AssistantMessage=_AssistantMessageFalso,
        ResultMessage=_ResultMessageFalso,
        TextBlock=_TextBlockFalso,
    )
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", modulo)
    return estado


def _config_claude_code(**model_kwargs):

    return LLMConfig(
        model="haiku",
        provider="claude_code",
        api_key=None,
        max_retries=1,
        base_delay=0.0,
        max_delay=0.0,
        rate_limit_delay=0.0,
        model_kwargs=model_kwargs,
    )


def _chamar(config=None):

    return call_claude_code(
        "texto da linha", SampleModel, "Analise: {texto}", config or _config_claude_code()
    )


class TestOpcoesSemFerramentas:
    """O texto das linhas é conteúdo não confiável: nenhuma ferramenta pode ficar disponível."""

    def test_nenhuma_ferramenta_disponivel(self, sdk_falso):
        _chamar()

        opcoes = sdk_falso["opcoes"][0]
        assert opcoes.kwargs.get("tools") == []

    def test_nao_aprova_ferramentas_automaticamente(self, sdk_falso):
        _chamar()

        opcoes = sdk_falso["opcoes"][0]
        assert opcoes.kwargs.get("permission_mode") == "default"
        assert not opcoes.kwargs.get("allowed_tools")

    def test_sdk_real_aceita_tools_vazio(self):
        """A versão instalada do SDK aceita `tools=[]` e o traduz em `--tools ''`."""
        sdk = pytest.importorskip("claude_agent_sdk")
        # Import direto, e não importorskip: se o módulo interno mudar de lugar numa
        # versão nova do SDK, o teste deve quebrar, e não pular.
        from claude_agent_sdk._internal.transport import subprocess_cli  # noqa: PLC0415

        opcoes = sdk.ClaudeAgentOptions(tools=[], permission_mode="default")
        transporte = subprocess_cli.SubprocessCLITransport(prompt="x", options=opcoes)
        transporte._cli_path = "claude"
        comando = transporte._build_command()

        indice = comando.index("--tools")
        assert comando[indice + 1] == ""
        assert comando[comando.index("--permission-mode") + 1] == "default"
        assert "--allowedTools" not in comando


class TestEventLoopAtivo:
    """Em Jupyter já existe um event loop rodando no thread principal."""

    def test_funciona_dentro_de_loop_ativo(self, sdk_falso):

        async def main():
            return _chamar()

        resultado = asyncio.run(main())

        assert resultado["data"] == {"sentimento": "positivo", "confianca": 0.9}

    def test_funciona_sem_loop_ativo(self, sdk_falso):
        resultado = _chamar()

        assert resultado["data"]["sentimento"] == "positivo"


class TestUsageReal:
    """Tokens vêm do `ResultMessage.usage` do SDK, nunca de valores fixos."""

    def test_tokens_extraidos_do_result_message(self, sdk_falso):
        sdk_falso["mensagens_do_sdk"][-1] = _ResultMessageFalso(
            usage={
                "input_tokens": 100,
                "cache_read_input_tokens": 40,
                "cache_creation_input_tokens": 10,
                "output_tokens": 25,
                "service_tier": "standard",
            },
            total_cost_usd=0.0012,
        )

        usage = _chamar()["usage"]

        # input_tokens inclui leitura e criação de cache, como no provider LangChain
        assert usage["input_tokens"] == 150
        assert usage["cached_input_tokens"] == 40
        assert usage["output_tokens"] == 25
        assert usage["total_tokens"] == 175
        assert usage["reasoning_tokens"] == 0

    def test_sem_usage_do_sdk_devolve_none(self, sdk_falso):
        sdk_falso["mensagens_do_sdk"][-1] = _ResultMessageFalso(usage=None)

        resultado = _chamar()

        assert resultado["usage"] is None

    def test_sem_usage_core_deixa_colunas_de_token_vazias(self, sdk_falso):
        """O core aceita usage None: colunas de token ficam nulas e a agregação não quebra."""

        with patch("dataframeit.core.validate_provider_dependencies"):
            df = dataframeit(
                pd.DataFrame({"texto": ["um", "dois"]}),
                questions=SampleModel,
                prompt="Analise: {texto}",
                provider="claude_code",
                model="haiku",
            )

        assert list(df["sentimento"]) == ["positivo", "positivo"]
        assert df["_input_tokens"].isna().all()
        assert df["_output_tokens"].isna().all()


# =============================================================================
# Isolamento, ResultMessage com erro e custo
# =============================================================================


def test_claude_code_nao_carrega_settings_nem_mcp_do_usuario(sdk_falso):
    _chamar()
    opcoes = sdk_falso["opcoes"][0]
    assert opcoes.kwargs.get("setting_sources") == []
    assert "strict-mcp-config" in opcoes.kwargs.get("extra_args", {})


def test_sdk_real_monta_as_flags_de_isolamento():
    sdk = pytest.importorskip("claude_agent_sdk")
    # Import direto pelo mesmo motivo do teste acima.
    from claude_agent_sdk._internal.transport import subprocess_cli  # noqa: PLC0415

    opcoes = sdk.ClaudeAgentOptions(
        tools=[],
        setting_sources=[],
        extra_args={"strict-mcp-config": None},
    )
    transporte = subprocess_cli.SubprocessCLITransport(prompt="x", options=opcoes)
    transporte._cli_path = "claude"
    comando = transporte._build_command()

    # Conforme a versão, o SDK emite '--setting-sources=' ou '--setting-sources', ''
    if "--setting-sources=" in comando:
        pass
    else:
        assert comando[comando.index("--setting-sources") + 1] == ""
    assert "--strict-mcp-config" in comando


def _resultado(**campos):
    mensagem = _ResultMessageFalso(usage=None, total_cost_usd=campos.pop("total_cost_usd", None))
    for chave, valor in campos.items():
        setattr(mensagem, chave, valor)
    return mensagem


@pytest.mark.parametrize(
    ("campos", "classe"),
    [
        ({"subtype": "error_max_budget_usd"}, ProviderError),
        ({"subtype": "error_max_turns"}, ProviderError),
        ({"subtype": "success", "api_error_status": 529}, ProviderOverloadedError),
        ({"subtype": "success", "api_error_status": 429}, ProviderOverloadedError),
        ({"subtype": "success", "api_error_status": 500}, ProviderTransientError),
        ({"subtype": "success", "api_error_status": 400}, ProviderError),
        ({"subtype": "error_during_execution"}, ProviderTransientError),
    ],
)
def test_result_message_com_erro_vira_erro_classificado(sdk_falso, campos, classe):
    sdk_falso["mensagens_do_sdk"] = [_resultado(is_error=True, **campos)]

    with pytest.raises(classe) as erro:
        _chamar()

    esperado_recuperavel = issubclass(classe, ProviderTransientError)
    assert is_recoverable_error(erro.value) is esperado_recuperavel
    assert is_rate_limit_error(erro.value) is (classe is ProviderOverloadedError)


def test_custo_informado_pelo_sdk_entra_no_usage(sdk_falso):
    sdk_falso["mensagens_do_sdk"] = [
        _AssistantMessageFalso([_TextBlockFalso('{"sentimento": "positivo", "confianca": 0.9}')]),
        _resultado(
            is_error=False,
            subtype="success",
            total_cost_usd=0.0125,
            usage={"input_tokens": 10, "output_tokens": 5},
        ),
    ]
    resultado = _chamar()
    assert resultado["usage"]["cost_usd"] == pytest.approx(0.0125)


def test_custo_somado_nas_estatisticas():
    resumo = _format_token_stats(
        {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "cost_usd": 0.25},
        model=None,
    )
    assert "US$ 0.2500" in resumo


def _resposta_valida(custo):
    return [
        _AssistantMessageFalso([_TextBlockFalso('{"sentimento": "positivo", "confianca": 0.9}')]),
        _resultado(
            is_error=False,
            subtype="success",
            total_cost_usd=custo,
            usage={"input_tokens": 10, "output_tokens": 5},
        ),
    ]


def test_custo_soma_as_tentativas_re_tentadas(sdk_falso):
    sdk_falso["por_chamada"] = [
        [
            _AssistantMessageFalso([_TextBlockFalso("não é json")]),
            _resultado(is_error=False, subtype="success", total_cost_usd=0.1),
        ],
        _resposta_valida(0.2),
    ]

    resultado = _chamar(replace(_config_claude_code(), max_retries=2))
    assert resultado["usage"]["cost_usd"] == pytest.approx(0.3)


def test_linha_que_falha_leva_o_custo_na_excecao(sdk_falso):
    sdk_falso["mensagens_do_sdk"] = [
        _resultado(is_error=True, subtype="error_max_budget_usd", total_cost_usd=0.5),
    ]
    with pytest.raises(ProviderError) as erro:
        _chamar()
    # cost_usd é anexado à exceção pelo provider, fora da classe.
    assert erro.value.cost_usd == pytest.approx(0.5)  # ty: ignore[unresolved-attribute]


@pytest.mark.parametrize("parallel_requests", [1, 2])
def test_resumo_soma_o_custo_das_linhas_e_das_falhas(caplog, parallel_requests):

    def call_claude_code(text, *args, **kwargs):
        if text.endswith("falha"):
            erro = ProviderError("orçamento estourado")
            # cost_usd é anexado à exceção pelo provider, fora da classe.
            erro.cost_usd = 0.5  # ty: ignore[unresolved-attribute]
            raise erro
        return {
            "data": {"sentimento": "positivo", "confianca": 0.9},
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "cost_usd": 0.1},
        }

    with (
        patch("dataframeit.claude_code.call_claude_code", side_effect=call_claude_code),
        patch("dataframeit.core.validate_provider_dependencies"),
        warnings.catch_warnings(),
        caplog.at_level(logging.INFO, logger="dataframeit.stats"),
    ):
        warnings.simplefilter("ignore")
        dataframeit(
            pd.DataFrame({"texto": ["a", "b", "falha"]}),
            questions=SampleModel,
            prompt="{texto}",
            provider="claude_code",
            parallel_requests=parallel_requests,
        )
    assert "US$ 0.7000" in caplog.text


def test_custo_sem_tokens_aparece_no_resumo():
    resumo = _format_token_stats(
        {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost_usd": 0.25},
        model=None,
    )
    assert "US$ 0.2500" in resumo


# =============================================================================
# Mensagens do SDK, opções repassadas e resposta vazia
# =============================================================================


class _ThinkingBlockFalso:
    def __init__(self, thinking):
        self.thinking = thinking


class _SystemMessageFalso:
    def __init__(self, subtype):
        self.subtype = subtype


def test_so_os_blocos_de_texto_do_assistente_compoem_a_resposta(sdk_falso):
    """Mensagem de sistema e bloco de raciocínio não entram no JSON lido."""
    sdk_falso["mensagens_do_sdk"] = [
        _SystemMessageFalso("init"),
        _AssistantMessageFalso(
            [
                _ThinkingBlockFalso("vou responder em JSON"),
                _TextBlockFalso('{"sentimento": "neutro", "confianca": 0.5}'),
            ]
        ),
        _ResultMessageFalso(usage=None),
    ]

    resultado = _chamar()

    assert resultado["data"] == {"sentimento": "neutro", "confianca": 0.5}


def test_usage_sem_custo_nao_ganha_chave_de_custo(sdk_falso):
    sdk_falso["mensagens_do_sdk"][-1] = _ResultMessageFalso(
        usage={"input_tokens": 10, "output_tokens": 5}, total_cost_usd=None
    )

    usage = _chamar()["usage"]

    assert usage["total_tokens"] == 15
    assert "cost_usd" not in usage


def test_effort_de_model_kwargs_chega_as_opcoes(sdk_falso):
    _chamar(_config_claude_code(effort="high"))

    assert sdk_falso["opcoes"][0].kwargs["effort"] == "high"


def test_sem_effort_as_opcoes_nao_levam_a_chave(sdk_falso):
    _chamar()

    assert "effort" not in sdk_falso["opcoes"][0].kwargs


def test_sem_modelo_o_runtime_escolhe(sdk_falso):
    """Com model=None, dataframeit não passa modelo e o Claude Code usa o seu padrão."""
    with patch("dataframeit.core.validate_provider_dependencies"):
        resultado = dataframeit(
            pd.DataFrame({"texto": ["um"]}),
            questions=SampleModel,
            prompt="Analise: {texto}",
            provider="claude_code",
        )

    assert resultado["sentimento"].tolist() == ["positivo"]
    assert "model" not in sdk_falso["opcoes"][0].kwargs


def test_resposta_so_com_espacos_e_erro(sdk_falso):
    sdk_falso["mensagens_do_sdk"] = [
        _AssistantMessageFalso([_TextBlockFalso("  \n ")]),
        _ResultMessageFalso(usage=None),
    ]

    with pytest.raises(ValueError, match="Claude Code SDK retornou resposta vazia"):
        _chamar()


# Números como '401' ou '403' aparecem em texto jurídico ("art. 401 da CLT",
# "Súmula 403"); a resposta recusada não pode virar erro de credencial.
@pytest.mark.parametrize(
    "resposta",
    [
        '{"sentimento": "positivo", "confianca": "art. 401 da CLT"}',
        '{"sentimento": "Súmula 403 do STJ", "confianca": 0.9',
    ],
)
def test_resposta_recusada_com_numero_no_texto_e_tentada_de_novo(sdk_falso, resposta):
    sdk_falso["mensagens_do_sdk"] = [
        _AssistantMessageFalso([_TextBlockFalso(resposta)]),
        _ResultMessageFalso(usage=None),
    ]

    with (
        pytest.warns(UserWarning, match=r"Tentativa 1/2 falhou \(ProviderRejectedOutputError\)"),
        pytest.raises(ProviderRejectedOutputError) as erro,
    ):
        _chamar(replace(_config_claude_code(), max_retries=2))

    assert len(sdk_falso["opcoes"]) == 2
    mensagem = str(erro.value)
    assert "401" not in mensagem
    assert "403" not in mensagem
    amigavel = get_friendly_error_message(erro.value, "claude_code")
    assert "AUTENTICAÇÃO" not in amigavel
    assert "PERMISSÃO" not in amigavel


def test_resposta_recusada_leva_caminho_e_regra_sem_o_valor(sdk_falso):
    sdk_falso["mensagens_do_sdk"] = [
        _AssistantMessageFalso(
            [_TextBlockFalso('{"sentimento": "positivo", "confianca": "valor secreto"}')]
        ),
        _ResultMessageFalso(usage=None),
    ]

    with pytest.raises(ProviderRejectedOutputError) as erro:
        _chamar()

    assert str(erro.value) == (
        "Falha na validação da resposta: 1 erro(s): confianca: "
        "Input should be a valid number, unable to parse string as a number"
    )


# =============================================================================
# Erro da API que o SDK levanta depois do ResultMessage de erro
# =============================================================================
#
# Quando a API recusa a chamada, o CLI emite o ResultMessage com is_error e sai
# com código não zero; o SDK entrega a mensagem e depois levanta a saída do
# processo (ResultError nas versões recentes). A exceção falsa imita o texto do
# ResultError, que só traz o número do status quando a mensagem da API o traz.


class _ResultErrorFalso(Exception):
    def __init__(self, texto, status):
        super().__init__(f"Claude Code returned an error result: {texto} (exit code: 1)")
        self.api_error_status = status


def _erro_da_api(status, texto, custo=None):
    return [
        _resultado(
            is_error=True,
            subtype="success",
            api_error_status=status,
            result=texto,
            total_cost_usd=custo,
        ),
        _ResultErrorFalso(texto, status),
    ]


def _config_com_tentativas(max_retries):
    return replace(_config_claude_code(), max_retries=max_retries)


@pytest.mark.parametrize(
    ("status", "texto"),
    [
        (401, "Invalid API key · Please run /login"),
        (400, "API Error: 400 invalid_request_error"),
    ],
)
def test_erro_definitivo_da_api_nao_e_repetido(sdk_falso, status, texto):
    sdk_falso["mensagens_do_sdk"] = _erro_da_api(status, texto)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ProviderError) as erro:
            _chamar(_config_com_tentativas(3))

    assert len(sdk_falso["opcoes"]) == 1
    assert is_recoverable_error(erro.value) is False
    assert is_rate_limit_error(erro.value) is False


def test_sobrecarga_da_api_e_repetida_como_rate_limit(sdk_falso):
    sdk_falso["mensagens_do_sdk"] = _erro_da_api(529, "API Error: Overloaded")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ProviderOverloadedError) as erro:
            _chamar(_config_com_tentativas(3))

    assert len(sdk_falso["opcoes"]) == 3
    assert is_rate_limit_error(erro.value) is True


def test_custo_de_cada_tentativa_recusada_entra_no_total(sdk_falso):
    sdk_falso["por_chamada"] = [
        _erro_da_api(529, "API Error: Overloaded", custo=0.1),
        _erro_da_api(529, "API Error: Overloaded", custo=0.2),
        _resposta_valida(0.3),
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        resultado = _chamar(_config_com_tentativas(3))

    assert resultado["usage"]["cost_usd"] == pytest.approx(0.6)


def test_linha_recusada_pela_api_leva_o_custo_na_excecao(sdk_falso):
    sdk_falso["mensagens_do_sdk"] = _erro_da_api(401, "Invalid API key", custo=0.05)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ProviderError) as erro:
            _chamar(_config_com_tentativas(3))

    # cost_usd é anexado à exceção pelo provider, fora da classe.
    assert erro.value.cost_usd == pytest.approx(0.05)  # ty: ignore[unresolved-attribute]


def test_rate_limit_da_api_reduz_os_workers(sdk_falso, monkeypatch):
    sdk_falso["mensagens_do_sdk"] = _erro_da_api(429, "API Error: Rate limited")
    # O evento de rate limit é limpo na hora, sem o timer de 5 s.
    monkeypatch.setattr(
        "dataframeit.core.threading.Timer",
        lambda _intervalo, funcao: types.SimpleNamespace(start=funcao),
    )

    with (
        patch("dataframeit.core.validate_provider_dependencies"),
        warnings.catch_warnings(record=True) as avisos,
    ):
        warnings.simplefilter("always")
        dataframeit(
            pd.DataFrame({"texto": ["a", "b"]}),
            questions=SampleModel,
            prompt="{texto}",
            provider="claude_code",
            parallel_requests=2,
            max_retries=1,
        )

    assert any("Reduzindo workers de 2 para 1" in str(aviso.message) for aviso in avisos)


@pytest.mark.parametrize(
    "mensagens",
    [
        [],
        [_ResultMessageFalso(usage=None)],
    ],
    ids=["sem_resultado", "depois_de_resultado_sem_erro"],
)
def test_falha_do_sdk_sem_resultado_de_erro_e_repassada(sdk_falso, mensagens):
    sdk_falso["mensagens_do_sdk"] = [*mensagens, RuntimeError("Command failed (exit code: 1)")]

    with pytest.raises(RuntimeError, match="Command failed"):
        _chamar()


@pytest.mark.parametrize("interrupcao", [KeyboardInterrupt, asyncio.CancelledError])
def test_interrupcao_depois_do_resultado_de_erro_sobe_intacta(sdk_falso, interrupcao):
    # Ctrl+C ou cancelamento da tarefa não é falha do SDK: mesmo com um ResultMessage
    # de erro já recebido, a interrupção sobe como veio, sem virar erro da API.
    sdk_falso["mensagens_do_sdk"] = [
        _resultado(is_error=True, subtype="success", api_error_status=529),
        interrupcao(),
    ]

    with pytest.raises(interrupcao):
        _chamar(_config_com_tentativas(3))

    assert len(sdk_falso["opcoes"]) == 1


class _TransporteDoSdkFalso:
    """Transporte no lugar do processo do CLI, para o laço de leitura do SDK instalado.

    Responde ao initialize, entrega o frame `result` de erro da API e falha com
    a saída do processo, a sequência que o SDK converte em exceção.
    """

    def __init__(self, erro_de_saida):
        self.erro_de_saida = erro_de_saida
        self.fila: asyncio.Queue = asyncio.Queue()

    async def connect(self):
        pass

    async def write(self, dados):
        mensagem = json.loads(dados)
        if mensagem.get("type") == "control_request":
            resposta = {"subtype": "success", "request_id": mensagem["request_id"]}
            await self.fila.put({"type": "control_response", "response": resposta})
        elif mensagem.get("type") == "user":
            await self.fila.put(
                {
                    "type": "result",
                    "subtype": "success",
                    "duration_ms": 1,
                    "duration_api_ms": 1,
                    "is_error": True,
                    "num_turns": 1,
                    "session_id": "s",
                    "total_cost_usd": 0.05,
                    "api_error_status": 401,
                    "result": "Invalid API key · Please run /login",
                }
            )
            await self.fila.put(self.erro_de_saida)

    async def read_messages(self):
        while True:
            item = await self.fila.get()
            if isinstance(item, Exception):
                raise item
            yield item

    async def end_input(self):
        pass

    async def close(self):
        pass

    def is_ready(self):
        return True


def _chamar_pelo_sdk_real(monkeypatch):
    """Chama o provider com o SDK instalado e devolve a exceção e o número de chamadas."""
    pytest.importorskip("claude_agent_sdk")
    from claude_agent_sdk import ProcessError  # noqa: PLC0415
    from claude_agent_sdk._internal import client as cliente_do_sdk  # noqa: PLC0415

    transportes = []

    def transporte(**_kwargs):
        transportes.append(_TransporteDoSdkFalso(ProcessError("Command failed", exit_code=1)))
        return transportes[-1]

    monkeypatch.setattr(cliente_do_sdk, "SubprocessCLITransport", transporte)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ProviderError) as erro:
            _chamar(_config_com_tentativas(3))
    return erro.value, len(transportes)


def test_sdk_real_classifica_pelo_resultado_de_erro_e_soma_o_custo(monkeypatch):
    """Laço de leitura do SDK instalado sobre um transporte falso, sem processo nem rede."""
    erro, chamadas = _chamar_pelo_sdk_real(monkeypatch)

    assert erro.cost_usd == pytest.approx(0.05 * chamadas)


def test_sdk_real_nao_repete_o_401(monkeypatch):
    sdk = pytest.importorskip("claude_agent_sdk")
    # As versões do SDK que não leem api_error_status entregam o ResultMessage
    # sem status, e o erro fica transitório.
    if "api_error_status" not in {campo.name for campo in fields(sdk.ResultMessage)}:
        pytest.skip("ResultMessage desta versão do SDK não traz api_error_status")

    erro, chamadas = _chamar_pelo_sdk_real(monkeypatch)

    assert chamadas == 1
    assert "status 401" in str(erro)
