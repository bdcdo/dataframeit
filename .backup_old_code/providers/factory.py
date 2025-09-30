from ..core.base import LLMStrategy, PromptBuilder
from ..core.services import RetryHandler
from ..core.dependency import DependencyChecker
from ..config import DataFrameConfiguration
from .openai.strategy import OpenAIStrategy
from .openai.client import create_openai_client
from .langchain.strategy import LangChainStrategy
from .langchain.client import create_langchain_llm


class LLMStrategyFactory:
    """Factory para criar estratégias de LLM com dependências específicas."""

    @staticmethod
    def create_strategy(
        config: DataFrameConfiguration, perguntas, user_prompt: str
    ) -> LLMStrategy:
        """Cria estratégia apropriada com injeção de dependências específicas.

        Args:
            config: Configuração que determina qual estratégia criar.
            perguntas: Modelo Pydantic definindo a estrutura esperada.
            user_prompt: Prompt fornecido pelo usuário.

        Returns:
            LLMStrategy: Instância da estratégia apropriada.
        """
        # Criar dependências compartilhadas
        prompt_builder = PromptBuilder(perguntas, config.placeholder)
        retry_handler = RetryHandler(
            config.max_retries, config.base_delay, config.max_delay
        )

        # Verificar dependências específicas e criar estratégias
        dependency_checker = DependencyChecker()

        if config.use_openai:
            dependency_checker.check_dependency("openai", "openai")
            client = create_openai_client(config.api_key, config.openai_client)
            return OpenAIStrategy(
                client=client,
                model=config.model,
                reasoning_effort=config.reasoning_effort,
                verbosity=config.verbosity,
                prompt_builder=prompt_builder,
                retry_handler=retry_handler,
                user_prompt=user_prompt
            )
        else:
            dependency_checker.check_dependency("langchain", "langchain")
            dependency_checker.check_dependency("langchain_core", "langchain-core")
            llm = create_langchain_llm(config.model, config.provider, config.api_key)
            return LangChainStrategy(
                llm=llm,
                prompt_builder=prompt_builder,
                retry_handler=retry_handler,
                user_prompt=user_prompt
            )