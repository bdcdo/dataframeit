"""
Exemplo 01: Uso Básico do DataFrameIt
======================================

Este exemplo demonstra o uso básico do DataFrameIt para análise de sentimento.

Conceitos demonstrados:
- Criação de modelo Pydantic simples
- Template de prompt básico
- Processamento de dados sintéticos
- Uso padrão com LangChain/Gemini

Para executar este exemplo:
1. Configure sua chave de API: export GOOGLE_API_KEY="sua-chave"
2. Instale dependências: pip install dataframeit langchain langchain-core langchain-google-genai
3. Execute: python3 example_01_basic.py
"""

from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
from dataframeit import dataframeit

# ============================================================================
# 1. DEFINIR MODELO PYDANTIC
# ============================================================================
# O modelo Pydantic define a estrutura dos dados que queremos extrair
class SentimentAnalysis(BaseModel):
    """Estrutura para análise de sentimento de textos."""

    sentimento: Literal['positivo', 'negativo', 'neutro'] = Field(
        ...,
        description="Classificação do sentimento geral do texto"
    )

    confianca: Literal['alta', 'media', 'baixa'] = Field(
        ...,
        description="Nível de confiança na classificação"
    )

    tema_principal: str = Field(
        ...,
        description="Tema ou assunto principal mencionado no texto"
    )

# ============================================================================
# 2. DEFINIR TEMPLATE DO PROMPT
# ============================================================================
# O template define como o LLM deve processar cada texto
# {documento} será substituído pelo texto de cada linha
TEMPLATE = """
Você é um analista de sentimentos especializado.

Analise o texto abaixo e classifique o sentimento expresso.

Texto a analisar:
{documento}

Responda de forma estruturada seguindo o formato solicitado.
"""

# ============================================================================
# 3. CRIAR DADOS SINTÉTICOS
# ============================================================================
# Para este exemplo, criamos alguns textos simples
dados = {
    'id': [1, 2, 3, 4, 5],
    'texto': [
        "Adorei o produto! A qualidade é excelente e chegou antes do prazo.",
        "Péssima experiência. O atendimento foi horrível e o produto veio com defeito.",
        "O produto é ok, nada de especial. Atende o básico.",
        "Estou muito feliz com a compra! Recomendo para todos os meus amigos.",
        "Não gostei. O preço é alto demais para a qualidade oferecida."
    ]
}

df = pd.DataFrame(dados)

print("=" * 80)
print("DATAFRAME ORIGINAL")
print("=" * 80)
print(df)
print("\n")

# ============================================================================
# 4. PROCESSAR COM DATAFRAMEIT
# ============================================================================
print("=" * 80)
print("PROCESSANDO COM DATAFRAMEIT...")
print("=" * 80)

# O dataframeit vai:
# 1. Pegar cada texto da coluna 'texto'
# 2. Enviar para o LLM com o template
# 3. Extrair as informações no formato do modelo Pydantic
# 4. Adicionar as novas colunas ao DataFrame
df_resultado = dataframeit(
    df,                      # DataFrame a processar
    SentimentAnalysis,       # Modelo Pydantic com estrutura esperada
    TEMPLATE,                # Template do prompt
    text_column='texto',     # Coluna que contém os textos (padrão: 'texto')
    model='gemini-2.0-flash-exp',  # Modelo do Gemini (padrão: 'gemini-2.5-flash')
    resume=True              # Permite continuar de onde parou se interrompido
)

# ============================================================================
# 5. VISUALIZAR RESULTADOS
# ============================================================================
print("\n")
print("=" * 80)
print("RESULTADOS")
print("=" * 80)

# Mostrar apenas as colunas relevantes
colunas_exibir = ['id', 'sentimento', 'confianca', 'tema_principal']
print(df_resultado[colunas_exibir])

print("\n")
print("=" * 80)
print("ESTATÍSTICAS")
print("=" * 80)

# Contar sentimentos
print("\nDistribuição de sentimentos:")
print(df_resultado['sentimento'].value_counts())

print("\nDistribuição de confiança:")
print(df_resultado['confianca'].value_counts())

# ============================================================================
# 6. SALVAR RESULTADOS (OPCIONAL)
# ============================================================================
# Descomente as linhas abaixo para salvar em arquivo
# df_resultado.to_csv('resultado_sentimentos.csv', index=False)
# df_resultado.to_excel('resultado_sentimentos.xlsx', index=False)
# print("\n✓ Resultados salvos em arquivo!")

print("\n" + "=" * 80)
print("EXEMPLO CONCLUÍDO!")
print("=" * 80)
