from ...core.base import LLMStrategy, PromptBuilder
from ...core.services import RetryHandler


class OpenAIStrategy(LLMStrategy):
    """Estratégia para processamento usando OpenAI."""

    def __init__(self, client, model: str, reasoning_effort: str, verbosity: str,
                 prompt_builder: PromptBuilder, retry_handler: RetryHandler, user_prompt: str):
        """Inicializa a estratégia OpenAI.

        Args:
            client: Cliente OpenAI configurado.
            model: Nome do modelo a ser utilizado.
            reasoning_effort: Esforço de raciocínio ('minimal', 'medium', 'high').
            verbosity: Nível de verbosidade ('low', 'medium', 'high').
            prompt_builder: Construtor de prompts.
            retry_handler: Handler para retry.
            user_prompt: Prompt fornecido pelo usuário.
        """
        self.client = client
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.verbosity = verbosity
        self.prompt_builder = prompt_builder
        self.retry_handler = retry_handler
        self.user_prompt = user_prompt

    def process_text(self, text: str) -> str:
        """Processa um texto usando a API da OpenAI.

        Args:
            text: Texto a ser processado pelo modelo OpenAI.

        Returns:
            str: Resposta do modelo OpenAI.
        """
        def _make_api_call():
            prompt_template = self.prompt_builder.build_prompt_template(self.user_prompt)
            full_prompt = self.prompt_builder.format_prompt(prompt_template, text)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": full_prompt}],
                reasoning={"effort": self.reasoning_effort},
                completion={"verbosity": self.verbosity},
            )
            return response.choices[0].message.content

        return self.retry_handler.execute_with_retry(_make_api_call)