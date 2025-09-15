from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Any, Dict, List, Tuple, Union
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from tqdm import tqdm
import pandas as pd
 
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
    retry_with_backoff,
    convert_dataframe_to_pandas,
    convert_dataframe_back,
    validate_text_column,
    validate_columns_conflict
)


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
        placeholder=placeholder,
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
    )

    # Processar usando nova arquitetura refatorada
    processor = DataFrameProcessor(config)
    return processor.process(df, perguntas, prompt)


# ============================================================================
# NOVA ARQUITETURA REFATORADA
# ============================================================================

@dataclass
class DataFrameConfiguration:
    """Configuração centralizada para processamento de DataFrame.
    
    Esta classe encapsula todas as configurações necessárias para o processamento
    de DataFrames com LLMs, incluindo parâmetros do modelo, credenciais de API,
    e opções de processamento.
    
    Attributes:
        model (str): Nome do modelo LLM a ser utilizado (padrão: 'gemini-2.5-flash')
        provider (str): Provedor do modelo LangChain (padrão: 'google_genai')
        use_openai (bool): Se True, utiliza OpenAI em vez de LangChain (padrão: False)
        api_key (Optional[str]): Chave API específica, se não usar variável de ambiente
        openai_client (Optional[Any]): Cliente OpenAI customizado (opcional)
        reasoning_effort (str): Esforço de raciocínio para OpenAI ('minimal', 'medium', 'high')
        verbosity (str): Nível de verbosidade para OpenAI ('low', 'medium', 'high')
        resume (bool): Se True, continua processamento de onde parou (padrão: True)
        status_column (Optional[str]): Coluna para rastrear progresso (padrão: primeira coluna do modelo)
        text_column (str): Nome da coluna contendo os textos (padrão: 'texto')
        placeholder (str): Nome do placeholder para o texto no prompt (padrão: 'documento')
        processed_marker (str): Marcador para linhas processadas com sucesso (padrão: 'processed')
        error_marker (str): Marcador para linhas que falharam no processamento (padrão: 'error')
        error_column (str): Coluna para armazenar detalhes do erro (padrão: 'error_details')
        max_retries (int): Número máximo de tentativas para chamadas à API (padrão: 3)
        base_delay (float): Delay base em segundos para retry (padrão: 1)
        max_delay (float): Delay máximo em segundos para retry (padrão: 30)
    """

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
    placeholder: str = 'documento'
    processed_marker: str = 'processed'
    error_marker: str = 'error'
    error_column: str = 'error_details'
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0


def create_openai_client(config: DataFrameConfiguration) -> Any:
    """Cria cliente OpenAI baseado na configuração.
    
    Args:
        config (DataFrameConfiguration): Configuração contendo parâmetros para criar o cliente.
        
    Returns:
        Any: Instância do cliente OpenAI configurado.
        
    Raises:
        ImportError: Se a biblioteca OpenAI não estiver instalada.
    """
    # Verificar dependência antes de usar
    check_dependency("openai", "openai")
    
    if config.openai_client:
        return config.openai_client
    elif config.api_key:
        return OpenAI(api_key=config.api_key)
    else:
        return OpenAI()


def create_langchain_chain(config: DataFrameConfiguration, perguntas, prompt: str) -> Any:
    """Cria chain LangChain baseado na configuração.
    
    Args:
        config (DataFrameConfiguration): Configuração contendo parâmetros do modelo.
        perguntas: Modelo Pydantic definindo a estrutura das informações a extrair.
        prompt (str): Template do prompt com placeholder para o texto.
        
    Returns:
        Any: Chain LangChain configurada pronta para invocação.
    """
    # Verificar dependência antes de usar
    check_dependency("langchain", "langchain")
    check_dependency("langchain_core", "langchain-core")
    check_dependency("langchain.output_parsers", "langchain")
    
    parser = PydanticOutputParser(pydantic_object=perguntas)
    prompt_inicial = ChatPromptTemplate.from_template(prompt)
    prompt_intermediario = prompt_inicial.partial(format=parser.get_format_instructions())

    model_kwargs = {"model_provider": config.provider, "temperature": 0}
    if config.api_key:
        model_kwargs["api_key"] = config.api_key

    llm = init_chat_model(config.model, **model_kwargs)
    return prompt_intermediario | llm


# ============================================================================
# STRATEGY PATTERN PARA LLM PROCESSING
# ============================================================================



# ============================================================================
# STRATEGY PATTERN PARA LLM PROCESSING
# ============================================================================

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
    """Estratégia para processamento usando OpenAI.
    
    Implementa a interface LLMStrategy utilizando a API da OpenAI para
    processar textos com modelos de linguagem.
    """

    def __init__(self, config: DataFrameConfiguration, perguntas, prompt: str, placeholder: str):
        """Inicializa a estratégia OpenAI.
        
        Args:
            config (DataFrameConfiguration): Configuração para o cliente OpenAI.
            perguntas: Modelo Pydantic definindo a estrutura das informações a extrair.
            prompt (str): Template do prompt com placeholder para o texto.
            placeholder (str): Nome do placeholder para o texto no prompt.
        """
        self.client = create_openai_client(config)
        self.config = config
        self.placeholder = placeholder
        parser = PydanticOutputParser(pydantic_object=perguntas)
        format_instructions = parser.get_format_instructions()
        self.prompt_template = f"""
        {prompt}

        {format_instructions}
        """

    @retry_with_backoff
    def process_text(self, text: str) -> str:
        """Processa um texto usando a API da OpenAI.
        
        Args:
            text (str): Texto a ser processado pelo modelo OpenAI.
            
        Returns:
            str: Resposta do modelo OpenAI.
        """
        full_prompt = self.prompt_template.format(**{self.placeholder: text})
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": full_prompt}],
            reasoning={"effort": self.config.reasoning_effort},
            completion={"verbosity": self.config.verbosity},
        )
        return response.choices[0].message.content



class LangChainStrategy(LLMStrategy):
    """Estratégia para processamento usando LangChain.
    
    Implementa a interface LLMStrategy utilizando LangChain para processar
    textos com diversos modelos e provedores suportados.
    """

    def __init__(self, config: DataFrameConfiguration, perguntas, prompt: str, placeholder: str):
        """Inicializa a estratégia LangChain.
        
        Args:
            config (DataFrameConfiguration): Configuração para o modelo LangChain.
            perguntas: Modelo Pydantic definindo a estrutura das informações a extrair.
            prompt (str): Template do prompt com placeholder para o texto.
            placeholder (str): Nome do placeholder para o texto no prompt.
        """
        self.chain = create_langchain_chain(config, perguntas, prompt)
        self.placeholder = placeholder
        self.config = config

    @retry_with_backoff
    def process_text(self, text: str) -> str:
        """Processa um texto usando LangChain.
        
        Args:
            text (str): Texto a ser processado pelo modelo configurado no LangChain.
            
        Returns:
            str: Resposta do modelo processado pelo LangChain.
        """
        return self.chain.invoke({self.placeholder: text})



class LLMStrategyFactory:
    """Factory para criar estratégias de LLM baseado na configuração.
    
    Implementa o padrão Factory para criar instâncias apropriadas de
    estratégias de processamento LLM com base nos parâmetros de configuração.
    """

    @staticmethod
    def create_strategy(
        config: DataFrameConfiguration, perguntas, prompt: str, placeholder: str
    ) -> LLMStrategy:
        """Cria estratégia apropriada baseada na configuração.
        
        Args:
            config (DataFrameConfiguration): Configuração que determina qual estratégia criar.
            perguntas: Modelo Pydantic definindo a estrutura das informações a extrair.
            prompt (str): Template do prompt com placeholder para o texto.
            placeholder (str): Nome do placeholder para o texto no prompt.
            
        Returns:
            LLMStrategy: Instância da estratégia apropriada (OpenAI ou LangChain).
        """
        if config.use_openai:
            return OpenAIStrategy(config, perguntas, prompt, placeholder)
        else:
            return LangChainStrategy(config, perguntas, prompt, placeholder)




class ProgressManager:
    """Gerenciamento de progresso e colunas de status.
    
    Responsável por configurar e gerenciar as colunas de progresso no DataFrame,
    incluindo colunas de resultado, status e erro. Permite a funcionalidade de
    retomada de processamento interrompido.
    """

    def __init__(self, expected_columns: List[str], config: DataFrameConfiguration):
        """Inicializa o gerenciador de progresso.
        
        Args:
            expected_columns (List[str]): Lista de colunas que serão criadas para resultados.
            config (DataFrameConfiguration): Configuração contendo parâmetros de progresso.
        """
        self.expected_columns = expected_columns
        self.config = config

    def _get_status_column_name(self) -> str:
        """Obtém o nome da coluna de status.
        
        Returns:
            str: Nome da coluna de status.
        """
        return self.config.status_column or self.expected_columns[0]

    def setup_columns(self, df: pd.DataFrame) -> None:
        """
        Configura colunas de resultado, status e erro no DataFrame (modifica in-place).
        
        Cria as colunas necessárias no DataFrame para armazenar os resultados do processamento,
        informações de status e detalhes de erro. A modificação in-place é intencional para
        garantir que o progresso parcial seja salvo no DataFrame original, permitindo a
        funcionalidade de resumo (`resume=True`) mesmo se o processo for interrompido.

        Args:
            df (pd.DataFrame): DataFrame no qual configurar as colunas.
        """
        # Identificar colunas de resultado novas
        new_result_columns = [col for col in self.expected_columns if col not in df.columns]
        if new_result_columns:
            for col in new_result_columns:
                df.loc[:, col] = None

        # Criar coluna de erro se não existir
        if self.config.error_column not in df.columns:
            df.loc[:, self.config.error_column] = None

        # Definir e criar coluna de status se não existir
        status_column = self._get_status_column_name()
        if status_column not in df.columns:
            df.loc[:, status_column] = None

    def get_processing_indices(self, df: pd.DataFrame) -> Tuple[int, int, str]:
        """Retorna a posição inicial de processamento e a contagem de itens já processados.
        
        Determina onde começar o processamento com base nas colunas de status e se o modo
        de retomada está ativado. Também conta quantos itens já foram processados.
        
        Args:
            df (pd.DataFrame): DataFrame para analisar o progresso.
            
        Returns:
            Tuple[int, int, str]: Tupla contendo:
                - Posição inicial de processamento
                - Número de itens já processados
                - Nome da coluna de status
        """
        status_column = self._get_status_column_name()

        if self.config.resume:
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

        # Garante que o nome da coluna de status seja retornado também
        return start_pos, processed_count, status_column

    def create_progress_description(self, engine_label: str, processed_count: int, total: int) -> str:
        """Cria descrição para barra de progresso.
        
        Args:
            engine_label (str): Identificador do motor de processamento.
            processed_count (int): Número de itens já processados.
            total (int): Número total de itens a processar.
            
        Returns:
            str: Descrição formatada para a barra de progresso.
        """
        if processed_count > 0:
            return f"Processando [{engine_label}] (resumindo de {processed_count}/{total})"
        else:
            return f"Processando [{engine_label}]"


class TextProcessor:
    """Processamento de texto usando LLMs com Strategy Pattern.
    
    Classe responsável por processar textos individuais usando a estratégia
    de LLM configurada (OpenAI ou LangChain).
    """

    def __init__(self, config: DataFrameConfiguration, perguntas, prompt: str):
        """Inicializa o processador de texto.
        
        Args:
            config (DataFrameConfiguration): Configuração para processamento.
            perguntas: Modelo Pydantic definindo a estrutura das informações a extrair.
            prompt (str): Template do prompt com placeholder para o texto.
        """
        self.config = config
        self.perguntas = perguntas
        self.prompt = prompt

        # Usar factory para criar estratégia (Open/Closed Principle)
        self.strategy = LLMStrategyFactory.create_strategy(
            config, perguntas, prompt, config.placeholder
        )

    def process_text(self, text: str) -> Dict[str, Any]:
        """Processa texto usando estratégia LLM configurada.
        
        Args:
            text (str): Texto a ser processado.
            
        Returns:
            Dict[str, Any]: Dicionário com os dados extraídos do texto.
        """
        response = self.strategy.process_text(text)
        return parse_json(response)


class DataFrameProcessor:
    """Orquestração principal do processamento de DataFrame.
    
    Classe responsável por coordenar todo o processo de transformação de um
    DataFrame, incluindo conversão de tipos, validações, configuração de
    progresso e processamento linha a linha.
    """

    def __init__(self, config: DataFrameConfiguration):
        """Inicializa o processador de DataFrame.
        
        Args:
            config (DataFrameConfiguration): Configuração para processamento.
        """
        self.config = config

    def update_dataframe_row(self, df: pd.DataFrame, idx: int, extracted_data: Dict[str, Any],
                           expected_columns: List[str], status_column: str) -> None:
        """Atualiza linha do DataFrame com dados extraídos.
        
        Args:
            df (pd.DataFrame): DataFrame a ser atualizado.
            idx (int): Índice da linha a ser atualizada.
            extracted_data (Dict[str, Any]): Dados extraídos para inserir no DataFrame.
            expected_columns (List[str]): Lista de colunas esperadas nos dados extraídos.
            status_column (str): Nome da coluna de status.
        """
        # Atualizar DataFrame com as informações extraídas
        for col in expected_columns:
            if col in extracted_data:
                df.at[idx, col] = extracted_data[col]

        # Marcar linha como processada
        # NOTA: A coluna de status é uma coluna de controle do sistema, não algo extraído do texto,
        # então sempre marcaremos como processada
        df.at[idx, status_column] = self.config.processed_marker

    def process(self, df, perguntas, prompt: str) -> Union[pd.DataFrame, Any]:
        """Processa DataFrame usando a nova arquitetura.
        
        Coordena todo o processo de transformação do DataFrame, incluindo:
        1. Conversão para pandas se necessário
        2. Validações iniciais
        3. Configuração de colunas de progresso
        4. Processamento linha a linha
        5. Conversão de volta para o formato original se necessário
        
        Args:
            df: DataFrame pandas ou polars contendo os textos para processar.
            perguntas: Modelo Pydantic definindo a estrutura das informações a extrair.
            prompt (str): Template do prompt com placeholder para o texto.
            
        Returns:
            Union[pd.DataFrame, Any]: DataFrame processado com colunas adicionais.
        """
        # Converter para pandas se necessário
        df_pandas, was_polars = convert_dataframe_to_pandas(df)

        # Validações
        validate_text_column(df_pandas, self.config.text_column)
        expected_columns = list(perguntas.model_fields.keys())

        if not expected_columns:
            raise ValueError(
                "O modelo Pydantic 'perguntas' não pode estar vazio. Defina pelo menos um campo."
            )

        # Validar conflitos e interromper se necessário
        if not validate_columns_conflict(df_pandas, expected_columns, self.config.resume):
            return convert_dataframe_back(df_pandas, was_polars)  # Retornar sem processar

        # Configurar progresso e colunas
        # NOTA: setup_columns modifica o df_pandas in-place para garantir que o progresso
        # parcial seja salvo no DataFrame original, permitindo a funcionalidade de resumo
        progress_manager = ProgressManager(expected_columns, self.config)
        progress_manager.setup_columns(df_pandas)
        start_pos, processed_count, status_column = progress_manager.get_processing_indices(df_pandas)

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
            # Pular linhas já processadas (otimização)
            if i < start_pos:
                continue

            # Verificar se linha específica já foi processada (segurança)
            if pd.notna(row_data[status_column]):
                continue

            try:
                # Processar texto
                extracted_data = text_processor.process_text(
                    row_data[self.config.text_column]
                )

                # Atualizar DataFrame
                self.update_dataframe_row(
                    df_pandas, idx, extracted_data, expected_columns, status_column
                )

            except (ValueError, Exception) as e:
                error_msg = f"{type(e).__name__}: {e}"
                warnings.warn(
                    f"Falha ao processar linha {idx}. {error_msg}. Marcando como 'error'."
                )
                df_pandas.at[idx, status_column] = self.config.error_marker
                df_pandas.at[idx, self.config.error_column] = error_msg

        # Converter de volta se necessário
        return convert_dataframe_back(df_pandas, was_polars)


