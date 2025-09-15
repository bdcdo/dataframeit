import time
import random
import warnings
from typing import List, Tuple, Union, Any
import pandas as pd

from ..utils import validate_text_column, validate_columns_conflict, convert_dataframe_to_pandas, convert_dataframe_back


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