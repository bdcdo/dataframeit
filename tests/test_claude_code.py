"""Testes para o provider claude_code (Claude Code SDK)."""
import json
import pytest
from unittest.mock import patch, MagicMock
from pydantic import BaseModel


class SampleModel(BaseModel):
    sentimento: str
    confianca: float


class TestValidateProviderDependencies:
    """Testes para validação de dependências do provider claude_code."""

    def test_claude_code_missing_raises_import_error(self):
        """Deve levantar ImportError com mensagem amigável quando claude_agent_sdk não está instalado."""
        from dataframeit.errors import validate_provider_dependencies

        with patch('importlib.import_module', side_effect=ImportError("No module")):
            with pytest.raises(ImportError, match="claude_agent_sdk"):
                validate_provider_dependencies('claude_code')

    def test_claude_code_installed_passes(self):
        """Deve passar sem erro quando claude_agent_sdk está instalado."""
        from dataframeit.errors import validate_provider_dependencies

        with patch('importlib.import_module', return_value=MagicMock()):
            # Não deve levantar exceção
            validate_provider_dependencies('claude_code')

    def test_claude_code_skips_langchain_validation(self):
        """Deve NÃO validar langchain quando provider é claude_code."""
        from dataframeit.errors import validate_provider_dependencies

        call_args = []

        def mock_import(name):
            call_args.append(name)
            return MagicMock()

        with patch('importlib.import_module', side_effect=mock_import):
            validate_provider_dependencies('claude_code')

        # Deve ter importado apenas claude_agent_sdk, não langchain
        assert 'claude_agent_sdk' in call_args
        assert 'langchain' not in call_args
        assert 'langchain_core' not in call_args


class TestUseSearchWithClaudeCode:
    """Testes para validação de use_search com claude_code."""

    def test_use_search_with_claude_code_raises(self):
        """Deve levantar ValueError quando use_search=True com provider='claude_code'."""
        from dataframeit import dataframeit

        with patch('dataframeit.core.validate_provider_dependencies'):
            with pytest.raises(ValueError, match=r"use_search.*claude_code"):
                dataframeit(
                    ["texto teste"],
                    questions=SampleModel,
                    prompt="Analise: {texto}",
                    provider='claude_code',
                    model='haiku',
                    use_search=True,
                )


class TestBuildJsonSystemPrompt:
    """Testes para geração do system prompt com JSON schema."""

    def test_includes_json_schema(self):
        """System prompt deve incluir o JSON schema do modelo Pydantic."""
        from dataframeit.claude_code import _build_json_system_prompt

        schema = SampleModel.model_json_schema()
        prompt = _build_json_system_prompt(schema)

        assert 'sentimento' in prompt
        assert 'confianca' in prompt
        assert 'JSON Schema' in prompt

    def test_instructs_json_only(self):
        """System prompt deve instruir resposta apenas em JSON."""
        from dataframeit.claude_code import _build_json_system_prompt

        schema = SampleModel.model_json_schema()
        prompt = _build_json_system_prompt(schema)

        assert 'APENAS' in prompt or 'JSON' in prompt


class TestJsonParsingVariants:
    """Testes para parsing de diferentes formatos de resposta."""

    def test_plain_json(self):
        """Deve parsear JSON puro."""
        from dataframeit.utils import parse_json

        result = parse_json('{"sentimento": "positivo", "confianca": 0.95}')
        assert result['sentimento'] == 'positivo'
        assert result['confianca'] == 0.95

    def test_json_with_markdown_fences(self):
        """Deve parsear JSON dentro de blocos markdown."""
        from dataframeit.utils import parse_json

        response = '```json\n{"sentimento": "negativo", "confianca": 0.8}\n```'
        result = parse_json(response)
        assert result['sentimento'] == 'negativo'

    def test_json_with_surrounding_text(self):
        """Deve extrair JSON de resposta com texto ao redor."""
        from dataframeit.utils import parse_json

        response = 'Aqui está o resultado: {"sentimento": "neutro", "confianca": 0.5} fim.'
        result = parse_json(response)
        assert result['sentimento'] == 'neutro'


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
    import sys
    import types

    estado = {
        'opcoes': [],
        'mensagens_do_sdk': [
            _AssistantMessageFalso([
                _TextBlockFalso('{"sentimento": "positivo", "confianca": 0.9}')
            ]),
            _ResultMessageFalso(usage=None),
        ],
    }

    async def query_falsa(prompt, options):
        estado['opcoes'].append(options)
        for mensagem in estado['mensagens_do_sdk']:
            yield mensagem

    modulo = types.ModuleType('claude_agent_sdk')
    modulo.ClaudeAgentOptions = _OpcoesFalsas
    modulo.query = query_falsa
    modulo.AssistantMessage = _AssistantMessageFalso
    modulo.ResultMessage = _ResultMessageFalso
    modulo.TextBlock = _TextBlockFalso
    monkeypatch.setitem(sys.modules, 'claude_agent_sdk', modulo)
    return estado


def _config_claude_code(**model_kwargs):
    from dataframeit.llm import LLMConfig

    return LLMConfig(
        model='haiku',
        provider='claude_code',
        api_key=None,
        max_retries=1,
        base_delay=0.0,
        max_delay=0.0,
        rate_limit_delay=0.0,
        model_kwargs=model_kwargs,
    )


def _chamar(config=None):
    from dataframeit.claude_code import call_claude_code

    return call_claude_code(
        'texto da linha', SampleModel, 'Analise: {texto}', config or _config_claude_code()
    )


class TestOpcoesSemFerramentas:
    """O texto das linhas é conteúdo não confiável: nenhuma ferramenta pode ficar disponível."""

    def test_nenhuma_ferramenta_disponivel(self, sdk_falso):
        _chamar()

        opcoes = sdk_falso['opcoes'][0]
        assert opcoes.kwargs.get('tools') == []

    def test_nao_aprova_ferramentas_automaticamente(self, sdk_falso):
        _chamar()

        opcoes = sdk_falso['opcoes'][0]
        assert opcoes.kwargs.get('permission_mode') == 'default'
        assert not opcoes.kwargs.get('allowed_tools')

    def test_sdk_real_aceita_tools_vazio(self):
        """A versão instalada do SDK aceita `tools=[]` e o traduz em `--tools ''`."""
        sdk = pytest.importorskip('claude_agent_sdk')
        from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport

        opcoes = sdk.ClaudeAgentOptions(tools=[], permission_mode='default')
        transporte = SubprocessCLITransport(prompt='x', options=opcoes)
        transporte._cli_path = 'claude'
        comando = transporte._build_command()

        indice = comando.index('--tools')
        assert comando[indice + 1] == ''
        assert comando[comando.index('--permission-mode') + 1] == 'default'
        assert '--allowedTools' not in comando


class TestEventLoopAtivo:
    """Em Jupyter já existe um event loop rodando no thread principal."""

    def test_funciona_dentro_de_loop_ativo(self, sdk_falso):
        import asyncio

        async def main():
            return _chamar()

        resultado = asyncio.run(main())

        assert resultado['data'] == {'sentimento': 'positivo', 'confianca': 0.9}

    def test_funciona_sem_loop_ativo(self, sdk_falso):
        resultado = _chamar()

        assert resultado['data']['sentimento'] == 'positivo'


class TestUsageReal:
    """Tokens vêm do `ResultMessage.usage` do SDK, nunca de valores fixos."""

    def test_tokens_extraidos_do_result_message(self, sdk_falso):
        sdk_falso['mensagens_do_sdk'][-1] = _ResultMessageFalso(
            usage={
                'input_tokens': 100,
                'cache_read_input_tokens': 40,
                'cache_creation_input_tokens': 10,
                'output_tokens': 25,
                'service_tier': 'standard',
            },
            total_cost_usd=0.0012,
        )

        usage = _chamar()['usage']

        # input_tokens inclui leitura e criação de cache, como no provider LangChain
        assert usage['input_tokens'] == 150
        assert usage['cached_input_tokens'] == 40
        assert usage['output_tokens'] == 25
        assert usage['total_tokens'] == 175
        assert usage['reasoning_tokens'] == 0

    def test_sem_usage_do_sdk_devolve_none(self, sdk_falso):
        sdk_falso['mensagens_do_sdk'][-1] = _ResultMessageFalso(usage=None)

        resultado = _chamar()

        assert resultado['usage'] is None

    def test_sem_usage_core_deixa_colunas_de_token_vazias(self, sdk_falso):
        """O core aceita usage None: colunas de token ficam nulas e a agregação não quebra."""
        import pandas as pd

        from dataframeit import dataframeit

        with patch('dataframeit.core.validate_provider_dependencies'):
            df = dataframeit(
                pd.DataFrame({'texto': ['um', 'dois']}),
                questions=SampleModel,
                prompt='Analise: {texto}',
                provider='claude_code',
                model='haiku',
            )

        assert list(df['sentimento']) == ['positivo', 'positivo']
        assert df['_input_tokens'].isna().all()
        assert df['_output_tokens'].isna().all()
