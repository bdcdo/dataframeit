from ...core.base import LLMStrategy, PromptBuilder
from ...core.services import RetryHandler


class LangChainStrategy(LLMStrategy):
    """Estratégia para processamento usando LangChain."""

    def __init__(self, llm, prompt_builder: PromptBuilder, retry_handler: RetryHandler, user_prompt: str):
        """Inicializa a estratégia LangChain.

        Args:
            llm: Modelo LLM configurado.
            prompt_builder: Construtor de prompts.
            retry_handler: Handler para retry.
            user_prompt: Prompt fornecido pelo usuário.
        """
        self.llm = llm
        self.prompt_builder = prompt_builder
        self.retry_handler = retry_handler
        self.user_prompt = user_prompt

    def process_text(self, text: str) -> str:
        """Processa um texto usando LangChain.

        Args:
            text: Texto a ser processado pelo modelo configurado no LangChain.

        Returns:
            str: Resposta do modelo processado pelo LangChain.
        """
        def _make_api_call():
            # Usar PromptBuilder como na OpenAIStrategy
            prompt_template = self.prompt_builder.build_prompt_template(self.user_prompt)
            full_prompt = self.prompt_builder.format_prompt(prompt_template, text)

            # Invocar o LLM diretamente
            return self.llm.invoke(full_prompt)

        return self.retry_handler.execute_with_retry(_make_api_call)