from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Any, Dict, List, Tuple, Union
import warnings
import time
import random
from tqdm import tqdm
import pandas as pd

# Imports opcionais do LangChain
try:
    from langchain.chat_models import init_chat_model
    from langchain_core.prompts import ChatPromptTemplate
    from langchain.output_parsers import PydanticOutputParser
except ImportError:
    # Será verificado antes do uso
    init_chat_model = None
    ChatPromptTemplate = None
    PydanticOutputParser = None
 
# Import opcional de Polars
try:
    import polars as pl  # type: ignore
except Exception:  # Polars não instalado
    pl = None  # type: ignore

# Import opcional de OpenAI
try:
    from openai import OpenAI  # type: ignore
except ImportError:  # OpenAI não instalado
    OpenAI = None  # type: ignore

from .utils import (
    parse_json,
    check_dependency,
    convert_dataframe_to_pandas,
    convert_dataframe_back,
    validate_text_column,
    validate_columns_conflict
)


# ============================================================================
# TRATAMENTO DE RETRY
# ============================================================================

class RetryHandler:
    """Gerencia lógica centralizada de retry para chamadas de API.

    Encapsula a configuração e execução de retry com backoff exponencial,
    removendo duplicação de lógica entre diferentes estratégias.
    """

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 30.0):
        """Inicializa o handler de retry.

        Args:
            max_retries: Número máximo de tentativas.
            base_delay: Delay base em segundos.
            max_delay: Delay máximo em segundos.
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

    def execute_with_retry(self, func, *args, **kwargs):
        """Executa função com retry usando parâmetros da instância.

        Args:
            func: Função a ser executada com retry.
            *args: Argumentos posicionais para a função.
            **kwargs: Argumentos nomeados para a função.

        Returns:
            Resultado da função executada.

        Raises:
            Exception: A última exceção capturada após esgotar as tentativas.
        """
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e

                if attempt < self.max_retries:
                    # Calcular delay com backoff exponencial
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    # Adicionar jitter para evitar thundering herd
                    jitter = random.uniform(0, 0.1) * delay
                    time.sleep(delay + jitter)
                else:
                    # Última tentativa, propagar exceção
                    raise last_exception


# ============================================================================
# VERIFICAÇÃO DE DEPENDÊNCIAS
# ============================================================================

class DependencyChecker:
    """Verificação de dependências com cache por instância.

    Centraliza a verificação de bibliotecas opcionais evitando
    múltiplas verificações das mesmas dependências.
    """

    def __init__(self):
        """Inicializa o verificador com cache próprio."""
        self._checked_dependencies = set()

    def check_dependency(self, package_name: str, install_name: str) -> None:
        """Verifica se uma dependência está instalada (com cache).

        Args:
            package_name: Nome do pacote para importação.
            install_name: Nome do pacote para instalação.

        Raises:
            ImportError: Se a dependência não estiver instalada.
        """
        if package_name not in self._checked_dependencies:
            check_dependency(package_name, install_name)
            self._checked_dependencies.add(package_name)


# ============================================================================
# VALIDAÇÕES
# ============================================================================

class ValidationService:
    """Centraliza todas as validações do sistema.

    Responsável por validar DataFrames, colunas, dependências e detectar
    conflitos de configuração.
    """

    @staticmethod
    def validate_text_column(df: pd.DataFrame, text_column: str) -> None:
        """Valida se a coluna de texto existe no DataFrame.

        Args:
            df: DataFrame a ser validado.
            text_column: Nome da coluna de texto.

        Raises:
            ValueError: Se a coluna de texto não existir.
        """
        validate_text_column(df, text_column)

    @staticmethod
    def validate_model_fields(perguntas) -> List[str]:
        """Valida e extrai campos do modelo Pydantic.

        Args:
            perguntas: Modelo Pydantic com os campos esperados.

        Returns:
            List[str]: Lista de campos do modelo.

        Raises:
            ValueError: Se o modelo não possuir campos.
        """
        expected_columns = list(perguntas.model_fields.keys())
        if not expected_columns:
            raise ValueError(
                "O modelo Pydantic 'perguntas' não pode estar vazio. Defina pelo menos um campo."
            )
        return expected_columns

    @staticmethod
    def validate_columns_conflict(df: pd.DataFrame, expected_columns: List[str], resume: bool) -> bool:
        """Valida conflitos entre colunas existentes e esperadas.

        Args:
            df: DataFrame a ser validado.
            expected_columns: Colunas que serão criadas.
            resume: Se o modo de retomada está ativado.

        Returns:
            bool: True se pode continuar o processamento, False caso contrário.
        """
        return validate_columns_conflict(df, expected_columns, resume)


# ============================================================================
# CONSTRUÇÃO DE PROMPTS
# ============================================================================

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
        return template.format(**{self.placeholder: text})


# ============================================================================
# TRANSFORMAÇÃO DE DATAFRAMES
# ============================================================================

class DataFrameTransformer:
    """Responsável pela conversão entre formatos de DataFrame (pandas/polars).

    Centraliza toda a lógica de transformação de DataFrames, incluindo
    detecção automática do tipo e conversões bidirecionais.
    """

    @staticmethod
    def to_pandas(df) -> Tuple[pd.DataFrame, bool]:
        """Converte DataFrame para pandas se necessário.

        Args:
            df: DataFrame pandas ou polars.

        Returns:
            Tuple[pd.DataFrame, bool]: DataFrame convertido e flag indicando se era polars.
        """
        return convert_dataframe_to_pandas(df)

    @staticmethod
    def from_pandas(df_pandas: pd.DataFrame, was_polars: bool) -> Union[pd.DataFrame, Any]:
        """Converte DataFrame de volta ao formato original se necessário.

        Args:
            df_pandas: DataFrame em formato pandas.
            was_polars: Flag indicando se o DataFrame original era polars.

        Returns:
            Union[pd.DataFrame, Any]: DataFrame no formato original.
        """
        return convert_dataframe_back(df_pandas, was_polars)


def dataframeit(
    df,
    perguntas,
    prompt,
    resume=True,
    model='gemini-2.5-flash',
    provider='google_genai',
    status_column=None,
    text_column: str = 'texto',
    placeholder: str = 'documento',
    use_openai=False,
    openai_client=None,
    reasoning_effort='minimal',
    verbosity='low',
    api_key=None,
    max_retries=3,
    base_delay=1.0,
    max_delay=30.0,
):
    """
    Processa textos em um DataFrame usando LLMs para extrair informações estruturadas.

    Suporta processamento via OpenAI ou LangChain com diferentes modelos e providers.
    Converte automaticamente DataFrames do Polars para pandas quando necessário.

    Args:
        df: DataFrame pandas ou polars contendo os textos para processar
        perguntas: Modelo Pydantic definindo a estrutura das informações a extrair
        prompt: Template do prompt com placeholder para o texto (padrão: {documento})
        resume: Se True, continua processamento de onde parou usando status_column
        model: Nome do modelo LLM (padrão: 'gemini-2.5-flash')
        provider: Provider do LangChain (padrão: 'google_genai')
        status_column: Coluna para rastrear progresso (padrão: primeira coluna do modelo)
        text_column: Nome da coluna contendo os textos (padrão: 'texto')
        placeholder: Nome do placeholder para o texto no prompt (padrão: 'documento')
        use_openai: Se True, usa OpenAI em vez de LangChain
        openai_client: Cliente OpenAI customizado (opcional)
        reasoning_effort: Esforço de raciocínio para OpenAI ('minimal', 'medium', 'high')
        verbosity: Nível de verbosidade para OpenAI ('low', 'medium', 'high')
        api_key: Chave API específica (opcional, senão usa variável de ambiente)
        max_retries: Número máximo de tentativas para chamadas à API (padrão: 3)
        base_delay: Delay base em segundos para retry (padrão: 1.0)
        max_delay: Delay máximo em segundos para retry (padrão: 30.0)

    Returns:
        DataFrame com colunas originais mais as definidas no modelo Pydantic

    Raises:
        ImportError: Se OpenAI não estiver instalado quando use_openai=True
        TypeError: Se df não for pandas.DataFrame nem polars.DataFrame
        ValueError: Se text_column não existir no DataFrame
    """
    # Criar configuração centralizada
    config = DataFrameConfiguration.create(
        model=model,
        provider=provider,
        use_openai=use_openai,
        api_key=api_key,
        openai_client=openai_client,
        reasoning_effort=reasoning_effort,
        verbosity=verbosity,
        resume=resume,
        status_column=status_column,
        text_column=text_column,
        placeholder=placeholder,
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
    )

    # Processar usando nova arquitetura refatorada
    processor = DataFrameProcessor(config)
    return processor.process(df, perguntas, prompt)

@dataclass
class LLMConfig:
    """Configuração específica para modelos de linguagem."""
    model: str = 'gemini-2.5-flash'
    provider: str = 'google_genai'
    use_openai: bool = False
    api_key: Optional[str] = None
    openai_client: Optional[Any] = None
    reasoning_effort: str = 'minimal'
    verbosity: str = 'low'


@dataclass
class ProcessingConfig:
    """Configuração específica para processamento de DataFrames."""
    resume: bool = True
    status_column: Optional[str] = None
    text_column: str = 'texto'
    placeholder: str = 'documento'
    processed_marker: str = 'processed'
    error_marker: str = 'error'
    error_column: str = 'error_details'


@dataclass
class RetryConfig:
    """Configuração específica para retry de chamadas API."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0


@dataclass
class DataFrameConfiguration:
    """Configuração consolidada para processamento de DataFrame.

    Combina todas as configurações específicas em uma interface unificada
    para manter compatibilidade com a API existente.
    """
    llm_config: LLMConfig
    processing_config: ProcessingConfig
    retry_config: RetryConfig

    @classmethod
    def create(cls, **kwargs):
        """Factory method para criar configuração a partir de kwargs."""
        return cls(
            llm_config=cls._create_llm_config(kwargs),
            processing_config=cls._create_processing_config(kwargs),
            retry_config=cls._create_retry_config(kwargs)
        )

    @staticmethod
    def _create_llm_config(kwargs: Dict[str, Any]) -> LLMConfig:
        """Extrai parâmetros LLM dos kwargs."""
        llm_params = ['model', 'provider', 'use_openai', 'api_key', 'openai_client', 'reasoning_effort', 'verbosity']
        llm_kwargs = {k: v for k, v in kwargs.items() if k in llm_params}
        return LLMConfig(**llm_kwargs)

    @staticmethod
    def _create_processing_config(kwargs: Dict[str, Any]) -> ProcessingConfig:
        """Extrai parâmetros de processamento dos kwargs."""
        processing_params = ['resume', 'status_column', 'text_column', 'placeholder', 'processed_marker', 'error_marker', 'error_column']
        processing_kwargs = {k: v for k, v in kwargs.items() if k in processing_params}
        return ProcessingConfig(**processing_kwargs)

    @staticmethod
    def _create_retry_config(kwargs: Dict[str, Any]) -> RetryConfig:
        """Extrai parâmetros de retry dos kwargs."""
        retry_params = ['max_retries', 'base_delay', 'max_delay']
        retry_kwargs = {k: v for k, v in kwargs.items() if k in retry_params}
        return RetryConfig(**retry_kwargs)

    # Propriedades de acesso para compatibilidade
    @property
    def model(self): return self.llm_config.model
    @property
    def provider(self): return self.llm_config.provider
    @property
    def use_openai(self): return self.llm_config.use_openai
    @property
    def api_key(self): return self.llm_config.api_key
    @property
    def openai_client(self): return self.llm_config.openai_client
    @property
    def reasoning_effort(self): return self.llm_config.reasoning_effort
    @property
    def verbosity(self): return self.llm_config.verbosity
    @property
    def resume(self): return self.processing_config.resume
    @property
    def status_column(self): return self.processing_config.status_column
    @property
    def text_column(self): return self.processing_config.text_column
    @property
    def placeholder(self): return self.processing_config.placeholder
    @property
    def processed_marker(self): return self.processing_config.processed_marker
    @property
    def error_marker(self): return self.processing_config.error_marker
    @property
    def error_column(self): return self.processing_config.error_column
    @property
    def max_retries(self): return self.retry_config.max_retries
    @property
    def base_delay(self): return self.retry_config.base_delay
    @property
    def max_delay(self): return self.retry_config.max_delay


def create_openai_client(config: DataFrameConfiguration) -> Any:
    """Cria cliente OpenAI baseado na configuração.

    Args:
        config: Configuração contendo parâmetros para criar o cliente.

    Returns:
        Any: Instância do cliente OpenAI configurado.

    Raises:
        ImportError: Se a biblioteca OpenAI não estiver instalada.
    """
    if config.openai_client:
        return config.openai_client
    elif config.api_key:
        return OpenAI(api_key=config.api_key)
    else:
        return OpenAI()


def create_langchain_llm(config: DataFrameConfiguration) -> Any:
    """Cria modelo LLM LangChain baseado na configuração.

    Args:
        config: Configuração contendo parâmetros do modelo.

    Returns:
        Any: Modelo LLM configurado.
    """
    # Verificar se os imports estão disponíveis
    if init_chat_model is None:
        raise ImportError("LangChain não está disponível. Instale com: pip install langchain langchain-core")

    model_kwargs = {"model_provider": config.provider, "temperature": 0}
    if config.api_key:
        model_kwargs["api_key"] = config.api_key

    return init_chat_model(config.model, **model_kwargs)

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
            client = create_openai_client(config)
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
            llm = create_langchain_llm(config)
            return LangChainStrategy(
                llm=llm,
                prompt_builder=prompt_builder,
                retry_handler=retry_handler,
                user_prompt=user_prompt
            )

class ColumnManager:
    """Gerencia colunas do DataFrame para processamento.

    Responsável por configurar colunas de resultado, status e erro no DataFrame.
    """

    def __init__(self, expected_columns: List[str], status_column: Optional[str], error_column: str):
        """Inicializa o gerenciador de colunas.

        Args:
            expected_columns: Lista de colunas que serão criadas para resultados.
            status_column: Nome da coluna de status (opcional).
            error_column: Nome da coluna de erro.
        """
        self.expected_columns = expected_columns
        self.status_column = status_column
        self.error_column = error_column

    def get_status_column_name(self) -> str:
        """Obtém o nome da coluna de status.

        Returns:
            str: Nome da coluna de status.
        """
        return self.status_column or self.expected_columns[0]

    def setup_columns(self, df: pd.DataFrame) -> None:
        """Configura colunas necessárias no DataFrame (modifica in-place).

        Args:
            df: DataFrame no qual configurar as colunas.
        """
        # Identificar colunas de resultado novas
        new_result_columns = [col for col in self.expected_columns if col not in df.columns]
        if new_result_columns:
            for col in new_result_columns:
                df.loc[:, col] = None

        # Criar coluna de erro se não existir
        if self.error_column not in df.columns:
            df.loc[:, self.error_column] = None

        # Definir e criar coluna de status se não existir
        status_column = self.get_status_column_name()
        if status_column not in df.columns:
            df.loc[:, status_column] = None


class ProgressManager:
    """Gerencia o progresso de processamento do DataFrame.

    Responsável por determinar onde começar/retomar o processamento
    e criar descrições de progresso.
    """

    def __init__(self, resume: bool):
        """Inicializa o gerenciador de progresso.

        Args:
            resume: Se deve retomar o processamento de onde parou.
        """
        self.resume = resume

    def get_processing_indices(self, df: pd.DataFrame, status_column: str) -> Tuple[int, int]:
        """Retorna a posição inicial de processamento e a contagem de itens já processados.

        Args:
            df: DataFrame para analisar o progresso.
            status_column: Nome da coluna de status.

        Returns:
            Tuple[int, int]: Posição inicial e número de itens já processados.
        """
        if self.resume:
            # Encontra as linhas não processadas (onde o status é nulo)
            null_mask = df[status_column].isnull()
            unprocessed_indices = df.index[null_mask]

            if not unprocessed_indices.empty:
                # Encontra o rótulo do primeiro item não processado
                first_unprocessed_label = unprocessed_indices.min()
                # Converte o rótulo para sua posição numérica (inteiro)
                start_pos = df.index.get_loc(first_unprocessed_label)
            else:
                # Se não há nada para processar, começa no final
                start_pos = len(df)

            processed_count = len(df) - len(unprocessed_indices)
        else:
            start_pos = 0
            processed_count = 0

        return start_pos, processed_count

    @staticmethod
    def create_progress_description(engine_label: str, processed_count: int, total: int) -> str:
        """Cria descrição para barra de progresso.

        Args:
            engine_label: Identificador do motor de processamento.
            processed_count: Número de itens já processados.
            total: Número total de itens a processar.

        Returns:
            str: Descrição formatada para a barra de progresso.
        """
        if processed_count > 0:
            return f"Processando [{engine_label}] (resumindo de {processed_count}/{total})"
        else:
            return f"Processando [{engine_label}]"

class TextProcessor:
    """Processamento de texto usando LLMs com Strategy Pattern.

    Responsável por processar textos individuais usando a estratégia
    de LLM fornecida, seguindo o princípio de injeção de dependências.
    """

    def __init__(self, strategy: LLMStrategy):
        """Inicializa o processador de texto.

        Args:
            strategy: Estratégia LLM específica (OpenAI ou LangChain).
        """
        self.strategy = strategy

    def process_text(self, text: str) -> Dict[str, Any]:
        """Processa texto usando estratégia LLM configurada.

        Args:
            text: Texto a ser processado.

        Returns:
            Dict[str, Any]: Dicionário com os dados extraídos do texto.
        """
        response = self.strategy.process_text(text)
        return parse_json(response)

class DataFrameProcessor:
    """Orquestração principal do processamento de DataFrame.

    Responsável apenas pela coordenação entre componentes especializados,
    seguindo o princípio da responsabilidade única.
    """

    def __init__(self, config: DataFrameConfiguration):
        """Inicializa o processador de DataFrame.

        Args:
            config: Configuração para processamento.
        """
        self.config = config

    def process(self, df, perguntas, prompt: str) -> Union[pd.DataFrame, Any]:
        """Orquestra o processamento do DataFrame usando componentes especializados.

        Args:
            df: DataFrame pandas ou polars contendo os textos para processar.
            perguntas: Modelo Pydantic definindo a estrutura das informações a extrair.
            prompt: Template do prompt com placeholder para o texto.

        Returns:
            Union[pd.DataFrame, Any]: DataFrame processado com colunas adicionais.
        """
        # Transformar para pandas
        df_pandas, was_polars = DataFrameTransformer.to_pandas(df)

        # Validações
        validation_service = ValidationService()
        validation_service.validate_text_column(df_pandas, self.config.text_column)
        expected_columns = validation_service.validate_model_fields(perguntas)

        if not validation_service.validate_columns_conflict(df_pandas, expected_columns, self.config.resume):
            return DataFrameTransformer.from_pandas(df_pandas, was_polars)

        # Configurar colunas
        column_manager = ColumnManager(expected_columns, self.config.status_column, self.config.error_column)
        column_manager.setup_columns(df_pandas)
        status_column = column_manager.get_status_column_name()

        # Configurar progresso
        progress_manager = ProgressManager(self.config.resume)
        start_pos, processed_count = progress_manager.get_processing_indices(df_pandas, status_column)

        # Configurar processador de texto
        strategy = LLMStrategyFactory.create_strategy(self.config, perguntas, prompt)
        text_processor = TextProcessor(strategy)

        # Criar identificador e descrição do processamento
        engine_label = self._create_engine_label(was_polars)
        desc = ProgressManager.create_progress_description(engine_label, processed_count, len(df_pandas))

        # Processar linhas
        self._process_rows(df_pandas, text_processor, expected_columns, status_column, start_pos, desc)

        return DataFrameTransformer.from_pandas(df_pandas, was_polars)

    def _create_engine_label(self, was_polars: bool) -> str:
        """Cria identificador do motor de processamento."""
        engine_parts = [
            'polars→pandas' if was_polars else 'pandas',
            'openai' if self.config.use_openai else 'langchain'
        ]
        return '+'.join(engine_parts)

    def _process_rows(self, df: pd.DataFrame, text_processor: 'TextProcessor',
                     expected_columns: List[str], status_column: str, start_pos: int, desc: str) -> None:
        """Processa as linhas do DataFrame."""
        for i, (idx, row_data) in enumerate(tqdm(df.iterrows(), total=len(df), desc=desc)):
            # Pular linhas já processadas
            if i < start_pos or pd.notna(row_data[status_column]):
                continue

            try:
                extracted_data = text_processor.process_text(row_data[self.config.text_column])
                self._update_row_success(df, idx, extracted_data, expected_columns, status_column)
            except (ValueError, Exception) as e:
                self._update_row_error(df, idx, e, status_column)

    def _update_row_success(self, df: pd.DataFrame, idx: int, extracted_data: Dict[str, Any],
                           expected_columns: List[str], status_column: str) -> None:
        """Atualiza linha com dados extraídos com sucesso."""
        for col in expected_columns:
            if col in extracted_data:
                df.at[idx, col] = extracted_data[col]
        df.at[idx, status_column] = self.config.processed_marker

    def _update_row_error(self, df: pd.DataFrame, idx: int, error: Exception, status_column: str) -> None:
        """Atualiza linha que falhou no processamento."""
        error_msg = f"{type(error).__name__}: {error}"
        warnings.warn(f"Falha ao processar linha {idx}. {error_msg}. Marcando como 'error'.")
        df.at[idx, status_column] = self.config.error_marker
        df.at[idx, self.config.error_column] = error_msg
