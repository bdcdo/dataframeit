from abc import ABC, abstractmethod

# Imports opcionais do LangChain
try:
    from langchain.output_parsers import PydanticOutputParser
except ImportError:
    PydanticOutputParser = None


class LLMStrategy(ABC):
    """Interface abstrata para estratégias de processamento de LLM.

    Define o contrato para diferentes estratégias de processamento com LLMs,
    permitindo que o código cliente use qualquer estratégia sem conhecer os
    detalhes de implementação específicos.
    """

    @abstractmethod
    def process_text(self, text: str) -> str:
        """Processa um texto usando a estratégia específica do LLM.

        Args:
            text (str): Texto a ser processado pelo LLM.

        Returns:
            str: Resposta do LLM como string.
        """
        pass


class PromptBuilder:
    """Responsável pela construção e formatação de prompts para LLMs.

    Centraliza a lógica de criação de prompts, incluindo formatação de
    instruções e substituição de placeholders.
    """

    def __init__(self, perguntas, placeholder: str = 'documento'):
        """Inicializa o construtor de prompts.

        Args:
            perguntas: Modelo Pydantic definindo a estrutura esperada.
            placeholder: Nome do placeholder para o texto.
        """
        self.perguntas = perguntas
        self.placeholder = placeholder

        # Verificar se PydanticOutputParser está disponível
        if PydanticOutputParser is None:
            raise ImportError("LangChain não está instalado. Instale com: pip install langchain")

        self.parser = PydanticOutputParser(pydantic_object=perguntas)

    def build_prompt_template(self, user_prompt: str) -> str:
        """Constrói template de prompt com instruções de formatação.

        Args:
            user_prompt: Prompt fornecido pelo usuário.

        Returns:
            str: Template de prompt com instruções de formatação.
        """
        format_instructions = self.parser.get_format_instructions()
        return f"{user_prompt}\n\n{format_instructions}"

    def format_prompt(self, template: str, text: str) -> str:
        """Formata prompt substituindo placeholder pelo texto.

        Args:
            template: Template de prompt.
            text: Texto a ser processado.

        Returns:
            str: Prompt formatado pronto para envio ao LLM.
        """
        # Substitui apenas o placeholder do documento para evitar KeyError
        # causado por chaves adicionais em format_instructions (ex.: {format})
        return template.replace(f"{{{self.placeholder}}}", text)
