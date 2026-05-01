"""Testes para inferência de provider em dataframeit.errors."""
from dataframeit.errors import _infer_provider_info


class TestInferProviderInfoOverrides:
    """Overrides para providers cloud (Vertex AI, Bedrock, Azure OpenAI)."""

    def test_google_vertexai(self):
        info = _infer_provider_info('google_vertexai')
        assert info['package'] == 'langchain_google_vertexai'
        assert info['install'] == 'langchain-google-vertexai'
        assert info['env_var'] is None
        assert info['name'] == 'Google Vertex AI'
        assert 'gcloud' in info['auth_hint']

    def test_bedrock(self):
        info = _infer_provider_info('bedrock')
        assert info['package'] == 'langchain_aws'
        assert info['install'] == 'langchain-aws'
        assert info['env_var'] is None
        assert info['name'] == 'AWS Bedrock'
        assert 'aws' in info['auth_hint'].lower()

    def test_bedrock_converse(self):
        info = _infer_provider_info('bedrock_converse')
        assert info['package'] == 'langchain_aws'
        assert info['install'] == 'langchain-aws'
        assert info['env_var'] is None
        assert info['name'] == 'AWS Bedrock (Converse)'

    def test_azure_openai(self):
        info = _infer_provider_info('azure_openai')
        assert info['package'] == 'langchain_openai'
        assert info['install'] == 'langchain-openai'
        assert info['env_var'] == 'AZURE_OPENAI_API_KEY'
        assert info['name'] == 'Azure OpenAI'
        assert 'AZURE_OPENAI_ENDPOINT' in info['auth_hint']


class TestInferProviderInfoHeuristic:
    """A heurística genérica continua valendo para providers sem override."""

    def test_google_genai(self):
        info = _infer_provider_info('google_genai')
        assert info['package'] == 'langchain_google_genai'
        assert info['install'] == 'langchain-google-genai'
        assert info['env_var'] == 'GOOGLE_API_KEY'
        assert info['name'] == 'Google Gemini'

    def test_openai(self):
        info = _infer_provider_info('openai')
        assert info['package'] == 'langchain_openai'
        assert info['env_var'] == 'OPENAI_API_KEY'

    def test_anthropic(self):
        info = _infer_provider_info('anthropic')
        assert info['package'] == 'langchain_anthropic'
        assert info['env_var'] == 'ANTHROPIC_API_KEY'

    def test_unknown_provider_falls_back(self):
        info = _infer_provider_info('foo_bar')
        assert info['package'] == 'langchain_foo_bar'
        assert info['install'] == 'langchain-foo-bar'
        assert info['env_var'] == 'FOO_BAR_API_KEY'
        assert info['name'] == 'Foo Bar'

    def test_empty_provider(self):
        info = _infer_provider_info('')
        assert info['package'] is None
        assert info['name'] == 'LLM'


class TestFriendlyAuthErrorWithoutEnvVar:
    """Mensagem de auth deve adaptar quando provider usa credenciais SDK."""

    def test_vertexai_auth_error_uses_sdk_message(self):
        from dataframeit.errors import get_friendly_error_message

        msg = get_friendly_error_message(
            Exception("AuthenticationError: invalid credentials"),
            provider='google_vertexai',
        )
        assert 'Google Vertex AI' in msg
        assert 'gcloud auth application-default login' in msg
        # Não deve sugerir export de API key inexistente
        assert 'GOOGLE_VERTEXAI_API_KEY' not in msg

    def test_bedrock_auth_error_uses_sdk_message(self):
        from dataframeit.errors import get_friendly_error_message

        msg = get_friendly_error_message(
            Exception("AuthenticationError: 401"),
            provider='bedrock_converse',
        )
        assert 'aws configure' in msg.lower() or 'AWS_ACCESS_KEY_ID' in msg
        assert 'BEDROCK_CONVERSE_API_KEY' not in msg

    def test_openai_auth_error_keeps_api_key_message(self):
        from dataframeit.errors import get_friendly_error_message

        msg = get_friendly_error_message(
            Exception("AuthenticationError: bad key"),
            provider='openai',
        )
        assert 'OPENAI_API_KEY' in msg
        assert 'export OPENAI_API_KEY' in msg
