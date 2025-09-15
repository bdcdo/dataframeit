from typing import Union, Any, Tuple, List, Optional
import pandas as pd
from tqdm import tqdm

from .config import DataFrameConfiguration
from .core.services import ValidationService, DataFrameTransformer
from .core.managers import ColumnManager, ProgressManager, TextProcessor, RowProcessor
from .providers.factory import LLMStrategyFactory


class DataFramePreparationPipeline:
    """Pipeline dedicado para preparação de DataFrame antes do processamento.

    Consolida todas as etapas de preparação em um componente coeso:
    - Transformação pandas/polars
    - Validações de entrada
    - Configuração de colunas e progresso
    - Criação de processadores especializados
    """

    def __init__(self, config: DataFrameConfiguration):
        """Inicializa o pipeline de preparação.

        Args:
            config: Configuração para o processamento.
        """
        self.config = config
        self.validation_service = ValidationService()

    def prepare(self, df, questions, prompt: str) -> 'PreparedData':
        """Prepara DataFrame para processamento.

        Args:
            df: DataFrame pandas ou polars contendo os textos.
            questions: Modelo Pydantic definindo a estrutura esperada.
            prompt: Template do prompt com placeholder.

        Returns:
            PreparedData: Dados preparados para processamento.
        """
        # Transformar para pandas
        df_pandas, was_polars = DataFrameTransformer.to_pandas(df)
        # Garantir cópia para evitar SettingWithCopyWarning quando df é um slice/view
        df_pandas = df_pandas.copy()

        # Validações
        self.validation_service.validate_text_column(df_pandas, self.config.text_column)
        expected_columns = self.validation_service.validate_model_fields(questions)

        if not self.validation_service.validate_columns_conflict(df_pandas, expected_columns, self.config.resume):
            return PreparedData(
                df_pandas=df_pandas,
                was_polars=was_polars,
                should_skip_processing=True,
                expected_columns=expected_columns,
                status_column='',
                start_pos=0,
                processed_count=0,
                row_processor=None,
                engine_label='',
                text_column=self.config.text_column
            )

        # Configurar colunas
        column_manager = ColumnManager(expected_columns, self.config.status_column, self.config.error_column)
        column_manager.setup_columns(df_pandas)
        status_column = column_manager.get_status_column_name()

        # Configurar progresso
        progress_manager = ProgressManager(self.config.resume)
        start_pos, processed_count = progress_manager.get_processing_indices(df_pandas, status_column)

        # Configurar processadores
        strategy = LLMStrategyFactory.create_strategy(self.config, questions, prompt)
        text_processor = TextProcessor(strategy)
        row_processor = RowProcessor(
            text_processor=text_processor,
            expected_columns=expected_columns,
            processed_marker=self.config.processed_marker,
            error_marker=self.config.error_marker,
            error_column=self.config.error_column
        )

        # Criar identificador do motor
        engine_label = self._create_engine_label(was_polars)

        return PreparedData(
            df_pandas=df_pandas,
            was_polars=was_polars,
            should_skip_processing=False,
            expected_columns=expected_columns,
            status_column=status_column,
            start_pos=start_pos,
            processed_count=processed_count,
            row_processor=row_processor,
            engine_label=engine_label,
            text_column=self.config.text_column
        )

    def _create_engine_label(self, was_polars: bool) -> str:
        """Cria identificador do motor de processamento."""
        dataframe_engine = 'polars→pandas' if was_polars else 'pandas'
        llm_engine = 'openai' if self.config.use_openai else 'langchain'
        return f'{dataframe_engine}+{llm_engine}'


class DataFrameOrchestrator:
    """Orquestrador principal do processamento de DataFrame.

    Coordena a interação entre o pipeline de preparação e o processador,
    garantindo fluxo limpo e separação de responsabilidades.
    """

    def __init__(self, config: DataFrameConfiguration,
                 preparation_pipeline: Optional[DataFramePreparationPipeline] = None,
                 processor: Optional['DataFrameProcessor'] = None):
        """Inicializa o orquestrador.

        Args:
            config: Configuração para processamento.
            preparation_pipeline: Pipeline de preparação (opcional, criará se None).
            processor: Processador (opcional, criará se None).
        """
        self.config = config
        self.preparation_pipeline = preparation_pipeline or DataFramePreparationPipeline(config)
        self.processor = processor or DataFrameProcessor()

    def execute(self, df, questions, prompt: str) -> Union[pd.DataFrame, Any]:
        """Executa o processamento completo do DataFrame.

        Args:
            df: DataFrame pandas ou polars contendo os textos.
            questions: Modelo Pydantic definindo a estrutura esperada.
            prompt: Template do prompt com placeholder.

        Returns:
            Union[pd.DataFrame, Any]: DataFrame processado.
        """
        # Preparar dados
        prepared_data = self.preparation_pipeline.prepare(df, questions, prompt)

        # Verificar se deve pular processamento
        if prepared_data.should_skip_processing:
            return DataFrameTransformer.from_pandas(prepared_data.df_pandas, prepared_data.was_polars)

        # Processar
        self.processor.process(prepared_data)

        # Retornar no formato original
        return DataFrameTransformer.from_pandas(prepared_data.df_pandas, prepared_data.was_polars)


class PreparedData:
    """Encapsula dados preparados para processamento de DataFrame.

    Contém todos os dados e configurações necessárias para o processamento
    das linhas, eliminando dependências entre componentes.
    """

    def __init__(self, df_pandas: pd.DataFrame, was_polars: bool, should_skip_processing: bool,
                 expected_columns: List[str], status_column: str, start_pos: int,
                 processed_count: int, row_processor: Optional[RowProcessor], engine_label: str, text_column: str):
        self.df_pandas = df_pandas
        self.was_polars = was_polars
        self.should_skip_processing = should_skip_processing
        self.expected_columns = expected_columns
        self.status_column = status_column
        self.start_pos = start_pos
        self.processed_count = processed_count
        self.row_processor = row_processor
        self.engine_label = engine_label
        self.text_column = text_column


def dataframeit(
    df,
    questions=None,
    prompt=None,
    perguntas=None,  # Deprecated: use 'questions'
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
        questions: Modelo Pydantic definindo a estrutura das informações a extrair
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
    # Compatibilidade com API anterior
    if questions is None and perguntas is not None:
        questions = perguntas
    elif questions is None and perguntas is None:
        raise ValueError("Either 'questions' or 'perguntas' parameter must be provided")

    if prompt is None:
        raise ValueError("'prompt' parameter is required")

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

    orchestrator = DataFrameOrchestrator(config)
    return orchestrator.execute(df, questions, prompt)


class DataFrameProcessor:
    """Processador focado exclusivamente no processamento de linhas do DataFrame.

    Responsável apenas pelo loop de processamento das linhas,
    delegando toda a preparação para outros componentes.
    """

    def process(self, prepared_data: PreparedData) -> None:
        """Processa as linhas do DataFrame usando dados preparados.

        Args:
            prepared_data: Dados preparados contendo tudo necessário para processamento.
        """
        desc = ProgressManager.create_progress_description(
            prepared_data.engine_label,
            prepared_data.processed_count,
            len(prepared_data.df_pandas)
        )

        self._process_rows(prepared_data, desc)

    def _process_rows(self, prepared_data: PreparedData, desc: str) -> None:
        """Executa o loop de processamento das linhas."""
        df = prepared_data.df_pandas
        row_processor = prepared_data.row_processor
        if row_processor is None:
            raise RuntimeError("RowProcessor não inicializado. Verifique a preparação do pipeline.")

        for i, (idx, row_data) in enumerate(tqdm(df.iterrows(), total=len(df), desc=desc)):
            if i < prepared_data.start_pos or pd.notna(row_data[prepared_data.status_column]):
                continue

            text = str(row_data[prepared_data.text_column])
            row_processor.process_row(df, idx, text, prepared_data.status_column)
