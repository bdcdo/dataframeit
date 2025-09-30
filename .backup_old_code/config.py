from dataclasses import dataclass
from typing import Optional, Any, Dict


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