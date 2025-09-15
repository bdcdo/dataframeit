from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
from tqdm import tqdm

from .base import LLMStrategy
from ..utils import parse_json


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