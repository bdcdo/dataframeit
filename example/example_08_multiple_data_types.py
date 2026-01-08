"""
Exemplo 08: Suporte a Múltiplos Tipos de Dados
==============================================

Este exemplo demonstra como usar o DataFrameIt com diferentes tipos de dados
além de DataFrames: listas, dicionários e Series.

Conceitos demonstrados:
- Processamento de listas de textos
- Processamento de dicionários {chave: texto}
- Processamento de pandas.Series
- Retorno de resultados como DataFrame

Para executar este exemplo:
1. Configure sua chave de API: export GOOGLE_API_KEY="sua-chave"
2. Instale dependências: pip install dataframeit langchain langchain-core langchain-google-genai
3. Execute: python3 example_08_multiple_data_types.py
"""

from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
from dataframeit import dataframeit

# ============================================================================
# 1. DEFINIR MODELO PYDANTIC
# ============================================================================
class SentimentAnalysis(BaseModel):
    """Estrutura para análise de sentimento."""

    sentimento: Literal['positivo', 'negativo', 'neutro'] = Field(
        ...,
        description="Classificação do sentimento"
    )

    resumo: str = Field(
        ...,
        description="Resumo em uma frase"
    )


TEMPLATE = """
Analise o sentimento do texto abaixo:

{texto}

Classifique como positivo, negativo ou neutro e forneça um breve resumo.
"""


# ============================================================================
# 2. PROCESSANDO UMA LISTA DE TEXTOS
# ============================================================================
print("=" * 80)
print("EXEMPLO 1: LISTA DE TEXTOS")
print("=" * 80)

textos_lista = [
    "Adorei o produto! Chegou rápido e funciona perfeitamente.",
    "Péssimo atendimento, nunca mais compro aqui.",
    "O produto é ok, nada de especial.",
]

print("\nEntrada (list):")
for i, texto in enumerate(textos_lista):
    print(f"  [{i}] {texto[:50]}...")

# Processa a lista diretamente - retorna DataFrame
resultado_lista = dataframeit(
    textos_lista,        # Lista de textos (não precisa de text_column!)
    SentimentAnalysis,
    TEMPLATE,
    model='gemini-2.0-flash-exp',
)

print("\nSaída (DataFrame):")
print(resultado_lista)
print(f"\nTipo de saída: {type(resultado_lista).__name__}")


# ============================================================================
# 3. PROCESSANDO UM DICIONÁRIO
# ============================================================================
print("\n" + "=" * 80)
print("EXEMPLO 2: DICIONÁRIO {chave: texto}")
print("=" * 80)

textos_dict = {
    'review_001': "Produto excelente! Superou minhas expectativas.",
    'review_002': "Muito ruim, veio quebrado e não consegui trocar.",
    'review_003': "Atende ao esperado, bom custo-benefício.",
}

print("\nEntrada (dict):")
for key, value in textos_dict.items():
    print(f"  [{key}] {value[:40]}...")

# Processa o dicionário - retorna DataFrame com chaves como índice
resultado_dict = dataframeit(
    textos_dict,         # Dicionário (não precisa de text_column!)
    SentimentAnalysis,
    TEMPLATE,
    model='gemini-2.0-flash-exp',
)

print("\nSaída (DataFrame com chaves como índice):")
print(resultado_dict)
print(f"\nTipo de saída: {type(resultado_dict).__name__}")
print(f"Índices: {list(resultado_dict.index)}")


# ============================================================================
# 4. PROCESSANDO UMA PANDAS SERIES
# ============================================================================
print("\n" + "=" * 80)
print("EXEMPLO 3: PANDAS SERIES")
print("=" * 80)

textos_series = pd.Series(
    [
        "Fantástico! A melhor compra que já fiz.",
        "Decepcionado com a qualidade.",
        "Normal, como qualquer outro produto similar.",
    ],
    index=['cliente_A', 'cliente_B', 'cliente_C'],
    name='avaliacoes'
)

print("\nEntrada (pd.Series):")
print(textos_series)

# Processa a Series - retorna DataFrame preservando o índice
resultado_series = dataframeit(
    textos_series,       # Series (não precisa de text_column!)
    SentimentAnalysis,
    TEMPLATE,
    model='gemini-2.0-flash-exp',
)

print("\nSaída (DataFrame preservando índice):")
print(resultado_series)
print(f"\nTipo de saída: {type(resultado_series).__name__}")
print(f"Índices preservados: {list(resultado_series.index)}")


# ============================================================================
# 5. COMPARAÇÃO COM DATAFRAME TRADICIONAL
# ============================================================================
print("\n" + "=" * 80)
print("COMPARAÇÃO: DataFrame vs Novos Tipos")
print("=" * 80)

# Forma tradicional com DataFrame
df_tradicional = pd.DataFrame({
    'id': [1, 2, 3],
    'texto': textos_lista
})

resultado_df = dataframeit(
    df_tradicional,
    SentimentAnalysis,
    TEMPLATE,
    text_column='texto',  # Necessário para DataFrame
    model='gemini-2.0-flash-exp',
)

print("\nDataFrame tradicional:")
print(resultado_df)

print("\n" + "=" * 80)
print("RESUMO")
print("=" * 80)
print("""
Tipo de Entrada     | text_column | Tipo de Saída
--------------------|-------------|-------------------
pandas.DataFrame    | Obrigatório | pandas.DataFrame
polars.DataFrame    | Obrigatório | polars.DataFrame
list                | Automático  | pandas.DataFrame
dict                | Automático  | DataFrame (chaves como índice)
pandas.Series       | Automático  | DataFrame (índice preservado)
polars.Series       | Automático  | polars.DataFrame
""")

print("\n" + "=" * 80)
print("EXEMPLO CONCLUÍDO!")
print("=" * 80)
