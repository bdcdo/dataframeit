from dataclasses import dataclass
from typing import Optional, Any, Dict, List, Tuple, Union
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from tqdm import tqdm
import pandas as pd
import warnings
 
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

from .utils import parse_json


def dataframeit(
    df,
    perguntas,
    prompt,
    resume=True,
    model='gemini-2.5-flash',
    provider='google_genai',
    status_column=None,
    text_column: str = 'texto',
    use_openai=False,
    openai_client=None,
    reasoning_effort='minimal',
    verbosity='low',
    api_key=None,
):
    """
    Processa textos em um DataFrame usando LLMs para extrair informações estruturadas.

    Suporta processamento via OpenAI ou LangChain com diferentes modelos e providers.
    Converte automaticamente DataFrames do Polars para pandas quando necessário.

    Args:
        df: DataFrame pandas ou polars contendo os textos para processar
        perguntas: Modelo Pydantic definindo a estrutura das informações a extrair
        prompt: Template do prompt com placeholder {sentenca} para o texto
        resume: Se True, continua processamento de onde parou usando status_column
        model: Nome do modelo LLM (padrão: 'gemini-2.5-flash')
        provider: Provider do LangChain (padrão: 'google_genai')
        status_column: Coluna para rastrear progresso (padrão: primeira coluna do modelo)
        text_column: Nome da coluna contendo os textos (padrão: 'texto')
        use_openai: Se True, usa OpenAI em vez de LangChain
        openai_client: Cliente OpenAI customizado (opcional)
        reasoning_effort: Esforço de raciocínio para OpenAI ('minimal', 'medium', 'high')
        verbosity: Nível de verbosidade para OpenAI ('low', 'medium', 'high')
        api_key: Chave API específica (opcional, senão usa variável de ambiente)

    Returns:
        DataFrame com colunas originais mais as definidas no modelo Pydantic

    Raises:
        ImportError: Se OpenAI não estiver instalado quando use_openai=True
        TypeError: Se df não for pandas.DataFrame nem polars.DataFrame
        ValueError: Se text_column não existir no DataFrame
    """
    # Criar configuração centralizada
    config = DataFrameConfiguration(
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
    )

    # Processar usando nova arquitetura refatorada
    processor = DataFrameProcessor(config)
    return processor.process(df, perguntas, prompt)


# ============================================================================
# NOVA ARQUITETURA REFATORADA
# ============================================================================

@dataclass
class DataFrameConfiguration:
    """Configuração centralizada para processamento de DataFrame."""

    model: str = 'gemini-2.5-flash'
    provider: str = 'google_genai'
    use_openai: bool = False
    api_key: Optional[str] = None
    openai_client: Optional[Any] = None
    reasoning_effort: str = 'minimal'
    verbosity: str = 'low'
    resume: bool = True
    status_column: Optional[str] = None
    text_column: str = 'texto'


class LLMClientFactory:
    """Factory para criação de clientes LLM (OpenAI/LangChain)."""

    @staticmethod
    def create_openai_client(config: DataFrameConfiguration) -> Any:
        """Cria cliente OpenAI baseado na configuração."""
        if OpenAI is None:
            raise ImportError("OpenAI not installed. Install with: pip install openai")

        if config.openai_client:
            return config.openai_client
        elif config.api_key:
            return OpenAI(api_key=config.api_key)
        else:
            return OpenAI()

    @staticmethod
    def create_langchain_chain(config: DataFrameConfiguration, perguntas, prompt: str) -> Any:
        """Cria chain LangChain baseado na configuração."""
        parser = PydanticOutputParser(pydantic_object=perguntas)
        prompt_inicial = ChatPromptTemplate.from_template(prompt)
        prompt_intermediario = prompt_inicial.partial(format=parser.get_format_instructions())

        model_kwargs = {"model_provider": config.provider, "temperature": 0}
        if config.api_key:
            model_kwargs["api_key"] = config.api_key

        llm = init_chat_model(config.model, **model_kwargs)
        return prompt_intermediario | llm


class DataFrameValidator:
    """Validações centralizadas para DataFrame."""

    @staticmethod
    def validate_dataframe(df) -> pd.DataFrame:
        """Valida e converte DataFrame se necessário."""
        if pl is not None and hasattr(pl, 'DataFrame') and isinstance(df, pl.DataFrame):
            return df.to_pandas(), True
        elif isinstance(df, pd.DataFrame):
            return df, False
        else:
            raise TypeError("df must be a pandas.DataFrame or polars.DataFrame")

    @staticmethod
    def validate_text_column(df: pd.DataFrame, text_column: str) -> None:
        """Valida se coluna de texto existe no DataFrame."""
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame.")

    @staticmethod
    def validate_columns_conflict(df: pd.DataFrame, expected_columns: List[str], resume: bool) -> None:
        """Valida conflitos de colunas existentes."""
        existing_result_columns = [col for col in expected_columns if col in df.columns]
        if existing_result_columns and not resume:
            warnings.warn(f"Columns {existing_result_columns} already exist. Use resume=True to continue or rename them.")


class DataFrameTransformer:
    """Transformações de formato de DataFrame."""

    @staticmethod
    def convert_to_pandas(df) -> Tuple[pd.DataFrame, bool]:
        """Converte DataFrame para pandas se necessário."""
        if pl is not None and hasattr(pl, 'DataFrame') and isinstance(df, pl.DataFrame):
            return df.to_pandas(), True
        elif isinstance(df, pd.DataFrame):
            return df, False
        else:
            raise TypeError("df must be a pandas.DataFrame or polars.DataFrame")

    @staticmethod
    def convert_back_if_needed(df: pd.DataFrame, was_polars: bool) -> Union[pd.DataFrame, Any]:
        """Converte de volta para Polars se necessário."""
        if was_polars and pl is not None:
            return pl.from_pandas(df)
        return df


class ProgressManager:
    """Gerenciamento de progresso e colunas de status."""

    def __init__(self, df: pd.DataFrame, expected_columns: List[str], config: DataFrameConfiguration):
        self.df = df
        self.expected_columns = expected_columns
        self.config = config

    def setup_columns(self) -> pd.DataFrame:
        """Configura colunas necessárias no DataFrame."""
        # Identificar colunas existentes e novas
        new_columns = [col for col in self.expected_columns if col not in self.df.columns]

        # Criar apenas colunas que não existem
        if new_columns:
            for col in new_columns:
                self.df.loc[:, col] = None

        # Definir coluna para controle de progresso
        status_column = self.config.status_column or self.expected_columns[0]

        # Criar coluna de status se não existir
        if status_column not in self.df.columns:
            self.df.loc[:, status_column] = None

        return self.df

    def get_processing_indices(self) -> Tuple[int, int, str]:
        """Retorna índices de processamento e coluna de status."""
        status_column = self.config.status_column or self.expected_columns[0]

        if self.config.resume:
            null_mask = self.df[status_column].isnull()
            unprocessed_indices = self.df.index[null_mask].tolist()
            start_idx = min(unprocessed_indices) if unprocessed_indices else len(self.df)
            processed_count = len(self.df) - len(unprocessed_indices)
        else:
            start_idx = 0
            processed_count = 0

        return start_idx, processed_count, status_column

    def create_progress_description(self, engine_label: str, processed_count: int, total: int) -> str:
        """Cria descrição para barra de progresso."""
        if processed_count > 0:
            return f"Processando [{engine_label}] (resumindo de {processed_count}/{total})"
        else:
            return f"Processando [{engine_label}]"


class TextProcessor:
    """Processamento de texto usando LLMs."""

    PROCESSED_MARKER = "processed"

    def __init__(self, config: DataFrameConfiguration, perguntas, prompt: str):
        self.config = config
        self.perguntas = perguntas
        self.prompt = prompt

        # Configurar clientes baseado na escolha
        if config.use_openai:
            self.client = LLMClientFactory.create_openai_client(config)
            self.chain = None
        else:
            self.client = None
            self.chain = LLMClientFactory.create_langchain_chain(config, perguntas, prompt)

    def process_text(self, text: str) -> Dict[str, Any]:
        """Processa texto usando LLM configurado."""
        if self.config.use_openai:
            response = self._process_openai(text)
        else:
            response = self._process_langchain(text)

        return parse_json(response)

    def _process_openai(self, text: str) -> str:
        """Processa texto usando OpenAI."""
        parser = PydanticOutputParser(pydantic_object=self.perguntas)
        format_instructions = parser.get_format_instructions()

        full_prompt = f"""
        {self.prompt}

        {format_instructions}
        """.format(sentenca=text, format=format_instructions)

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": full_prompt}],
            reasoning={"effort": self.config.reasoning_effort},
            completion={"verbosity": self.config.verbosity}
        )

        return response.choices[0].message.content

    def _process_langchain(self, text: str) -> str:
        """Processa texto usando LangChain."""
        return self.chain.invoke({'sentenca': text})

    def update_dataframe_row(self, df: pd.DataFrame, idx: int, extracted_data: Dict[str, Any],
                           expected_columns: List[str], status_column: str) -> None:
        """Atualiza linha do DataFrame com dados extraídos."""
        # Atualizar DataFrame com as informações extraídas
        for col in expected_columns:
            if col in extracted_data:
                df.at[idx, col] = extracted_data[col]

        # Marcar linha como processada
        if status_column in extracted_data:
            df.at[idx, status_column] = extracted_data[status_column]
        else:
            df.at[idx, status_column] = self.PROCESSED_MARKER


class DataFrameProcessor:
    """Orquestração principal do processamento de DataFrame."""

    def __init__(self, config: DataFrameConfiguration):
        self.config = config

    def process(self, df, perguntas, prompt: str) -> Union[pd.DataFrame, Any]:
        """Processa DataFrame usando a nova arquitetura."""
        # Converter para pandas se necessário
        df_pandas, was_polars = DataFrameTransformer.convert_to_pandas(df)

        # Validações
        DataFrameValidator.validate_text_column(df_pandas, self.config.text_column)
        expected_columns = list(perguntas.__fields__.keys())
        DataFrameValidator.validate_columns_conflict(df_pandas, expected_columns, self.config.resume)

        # Configurar progresso e colunas
        progress_manager = ProgressManager(df_pandas, expected_columns, self.config)
        df_pandas = progress_manager.setup_columns()
        start_idx, processed_count, status_column = progress_manager.get_processing_indices()

        # Configurar processador de texto
        text_processor = TextProcessor(self.config, perguntas, prompt)

        # Criar identificador do processamento
        engine_parts = [
            'polars→pandas' if was_polars else 'pandas',
            'openai' if self.config.use_openai else 'langchain'
        ]
        engine_label = '+'.join(engine_parts)

        # Configurar barra de progresso
        total = len(df_pandas)
        desc = progress_manager.create_progress_description(engine_label, processed_count, total)

        # Loop principal de processamento
        for i, (idx, row_data) in enumerate(tqdm(df_pandas.iterrows(), total=total, desc=desc)):
            # Pular linhas já processadas
            if i < start_idx:
                continue

            # Verificar se linha específica já foi processada
            if pd.notna(row_data[status_column]):
                continue

            # Processar texto
            extracted_data = text_processor.process_text(row_data[self.config.text_column])

            # Atualizar DataFrame
            text_processor.update_dataframe_row(df_pandas, idx, extracted_data, expected_columns, status_column)

        # Converter de volta se necessário
        return DataFrameTransformer.convert_back_if_needed(df_pandas, was_polars)


