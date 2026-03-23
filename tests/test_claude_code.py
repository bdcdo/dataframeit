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
            with pytest.raises(ValueError, match="use_search.*claude_code"):
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
