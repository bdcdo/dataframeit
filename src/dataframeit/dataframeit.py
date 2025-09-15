import warnings
from typing import Union, Any, Dict, List
import pandas as pd
from tqdm import tqdm

from .config import DataFrameConfiguration
from .core.services import ValidationService, DataFrameTransformer
from .core.managers import ColumnManager, ProgressManager, TextProcessor
from .providers.factory import LLMStrategyFactory


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