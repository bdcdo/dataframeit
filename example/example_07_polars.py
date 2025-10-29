"""
Exemplo 07: Usando Polars DataFrame
====================================

Este exemplo demonstra como usar o DataFrameIt com Polars,
uma alternativa moderna e performática ao Pandas.

Conceitos demonstrados:
- Conversão automática Polars ↔ Pandas
- Todas as funcionalidades funcionam com Polars
- Comparação de performance Pandas vs Polars
- Quando usar Polars

Para executar este exemplo:
1. Configure sua chave de API: export GOOGLE_API_KEY="sua-chave"
2. Instale dependências: pip install dataframeit polars langchain langchain-core langchain-google-genai
3. Execute: python3 example_07_polars.py
"""

from pydantic import BaseModel, Field
from typing import Literal
import polars as pl
import pandas as pd
from dataframeit import dataframeit
import time

# ============================================================================
# 1. DEFINIR MODELO PYDANTIC
# ============================================================================
class TaskCategory(BaseModel):
    """Classificação de tarefas."""

    categoria: Literal['urgente', 'importante', 'rotina', 'delegavel'] = Field(
        ...,
        description="Categoria da tarefa baseada na matriz de Eisenhower"
    )

    estimativa_tempo: Literal['rapido', 'medio', 'demorado'] = Field(
        ...,
        description="Estimativa de tempo necessário"
    )

# ============================================================================
# 2. DEFINIR TEMPLATE
# ============================================================================
TEMPLATE = """
Classifique a tarefa abaixo:

{documento}

Use a matriz de Eisenhower para classificação.
"""

# ============================================================================
# 3. CRIAR DADOS COM POLARS
# ============================================================================
print("=" * 80)
print("USANDO DATAFRAMEIT COM POLARS")
print("=" * 80)

# Criar dados diretamente com Polars
dados = {
    'task_id': [1, 2, 3, 4, 5],
    'texto': [
        "Responder email urgente do cliente sobre bug crítico",
        "Revisar documentação do projeto",
        "Planejar sprint da próxima semana",
        "Organizar arquivos pessoais",
        "Resolver problema de segurança reportado"
    ]
}

# CRIAR POLARS DATAFRAME
df_polars = pl.DataFrame(dados)

print("\n✓ DataFrame Polars criado:")
print(f"  Tipo: {type(df_polars)}")
print(f"  Linhas: {len(df_polars)}")
print(f"  Colunas: {df_polars.columns}")

print("\nPrimeiras linhas:")
print(df_polars)

# ============================================================================
# 4. PROCESSAR COM POLARS
# ============================================================================
print("\n" + "=" * 80)
print("PROCESSANDO COM POLARS...")
print("=" * 80)

start_time = time.time()

# O DataFrameIt aceita Polars diretamente!
# Conversão interna é automática
resultado_polars = dataframeit(
    df_polars,                    # Polars DataFrame
    TaskCategory,
    TEMPLATE,
    text_column='texto',
    model='gemini-2.0-flash-exp'
)

elapsed_time = time.time() - start_time

print(f"\n✓ Processamento concluído em {elapsed_time:.2f}s")
print(f"✓ Tipo do resultado: {type(resultado_polars)}")
print("✓ O DataFrame retornado está no formato ORIGINAL (Polars)")

# ============================================================================
# 5. VISUALIZAR RESULTADOS POLARS
# ============================================================================
print("\n" + "=" * 80)
print("RESULTADOS (POLARS)")
print("=" * 80)

# Selecionar colunas relevantes (sintaxe Polars)
colunas = ['task_id', 'categoria', 'estimativa_tempo']
print(resultado_polars.select(colunas))

# Estatísticas com Polars
print("\n" + "-" * 80)
print("Estatísticas (usando métodos Polars):")

print("\nDistribuição por categoria:")
print(resultado_polars.group_by('categoria').count())

print("\nDistribuição por tempo estimado:")
print(resultado_polars.group_by('estimativa_tempo').count())

# ============================================================================
# 6. COMPARAÇÃO PANDAS VS POLARS
# ============================================================================
print("\n" + "=" * 80)
print("COMPARAÇÃO: PANDAS VS POLARS")
print("=" * 80)

# Criar mesmo dataset com Pandas
df_pandas = pd.DataFrame(dados)

print("\n📊 Processando com Pandas...")
start_pandas = time.time()

resultado_pandas = dataframeit(
    df_pandas,
    TaskCategory,
    TEMPLATE,
    text_column='texto',
    model='gemini-2.0-flash-exp'
)

elapsed_pandas = time.time() - start_pandas

print(f"⏱ Tempo Pandas: {elapsed_pandas:.2f}s")
print(f"⏱ Tempo Polars: {elapsed_time:.2f}s")

print("""
⚠ NOTA: Para datasets pequenos, a diferença de performance é negligível
         pois o gargalo é a chamada à API do LLM, não o processamento do DataFrame.

         A vantagem do Polars aparece em:
         - Pré-processamento de dados grandes
         - Operações complexas de transformação
         - Leitura de arquivos muito grandes
         - Joins e agregações em datasets gigantes
""")

# ============================================================================
# 7. RECURSOS AVANÇADOS COM POLARS
# ============================================================================
print("=" * 80)
print("RECURSOS AVANÇADOS DO POLARS")
print("=" * 80)

print("\n1. Lazy evaluation (avaliação preguiçosa):")
print("""
# Criar LazyFrame (não executado ainda)
lf = pl.scan_csv('dados.csv')

# Aplicar transformações (ainda não executado)
lf_filtered = lf.filter(pl.col('status').is_null())

# Processar com DataFrameIt
# A conversão para DataFrame acontece automaticamente
resultado = dataframeit(lf_filtered.collect(), Model, TEMPLATE)
""")

print("\n2. Expressões poderosas:")
print("""
# Filtrar antes de processar
df_filtrado = df_polars.filter(
    pl.col('texto').str.lengths() > 10
)

resultado = dataframeit(df_filtrado, Model, TEMPLATE)
""")

print("\n3. Performance em larga escala:")
print("""
# Para datasets gigantes (>1M linhas)
# Polars é significativamente mais rápido no pré-processamento

df_large = pl.read_parquet('dados_grandes.parquet')  # Muito rápido!
df_filtered = df_large.filter(pl.col('precisa_processar') == True)
resultado = dataframeit(df_filtered, Model, TEMPLATE, resume=True)
""")

# ============================================================================
# 8. QUANDO USAR POLARS?
# ============================================================================
print("\n" + "=" * 80)
print("QUANDO USAR POLARS?")
print("=" * 80)

print("""
╔════════════════════════════════════════════════════════════════════════╗
║ USE POLARS QUANDO:                                                     ║
╠════════════════════════════════════════════════════════════════════════╣
║ ✓ Trabalha com datasets grandes (>100k linhas)                        ║
║ ✓ Precisa de performance máxima                                       ║
║ ✓ Usa operações complexas de transformação                            ║
║ ✓ Quer sintaxe mais moderna e expressiva                              ║
║ ✓ Trabalha com arquivos Parquet                                       ║
║ ✓ Precisa de lazy evaluation                                          ║
║ ✓ Quer aproveitar múltiplos cores automaticamente                     ║
║                                                                        ║
║ Exemplo:                                                               ║
║   df = pl.read_parquet('dados.parquet')                                ║
║   resultado = dataframeit(df, Model, TEMPLATE)                         ║
╚════════════════════════════════════════════════════════════════════════╝

╔════════════════════════════════════════════════════════════════════════╗
║ USE PANDAS QUANDO:                                                     ║
╠════════════════════════════════════════════════════════════════════════╣
║ ✓ Trabalha com datasets pequenos/médios (<100k linhas)                ║
║ ✓ Já tem código legado em Pandas                                      ║
║ ✓ Precisa de compatibilidade com bibliotecas que só suportam Pandas   ║
║ ✓ Quer documentação e exemplos mais abundantes                        ║
║ ✓ Está aprendendo análise de dados                                    ║
║                                                                        ║
║ Exemplo:                                                               ║
║   df = pd.read_excel('dados.xlsx')                                     ║
║   resultado = dataframeit(df, Model, TEMPLATE)                         ║
╚════════════════════════════════════════════════════════════════════════╝
""")

# ============================================================================
# 9. SALVAR RESULTADOS POLARS
# ============================================================================
print("=" * 80)
print("SALVANDO RESULTADOS")
print("=" * 80)

# Polars suporta múltiplos formatos de forma eficiente
print("\nFormatos suportados pelo Polars:")

# CSV
resultado_polars.write_csv('resultado_polars.csv')
print("✓ CSV: resultado_polars.csv")

# Parquet (formato recomendado para grandes volumes)
resultado_polars.write_parquet('resultado_polars.parquet')
print("✓ Parquet: resultado_polars.parquet")

# JSON
resultado_polars.write_json('resultado_polars.json')
print("✓ JSON: resultado_polars.json")

# Para Excel, converta para Pandas temporariamente
resultado_polars.to_pandas().to_excel('resultado_polars.xlsx', index=False)
print("✓ Excel: resultado_polars.xlsx (via Pandas)")

print("\n💡 Parquet é o formato recomendado para datasets grandes:")
print("   - Compressão eficiente")
print("   - Leitura/escrita muito rápida")
print("   - Preserva tipos de dados")
print("   - Suporte para colunas aninhadas")

# Limpeza
import os
for arquivo in ['resultado_polars.csv', 'resultado_polars.parquet',
                'resultado_polars.json', 'resultado_polars.xlsx']:
    if os.path.exists(arquivo):
        os.remove(arquivo)

# ============================================================================
# 10. RESUMO
# ============================================================================
print("\n" + "=" * 80)
print("RESUMO")
print("=" * 80)

print("""
✓ DataFrameIt suporta Polars nativamente
✓ Conversão Polars ↔ Pandas é automática e transparente
✓ Resultado retorna no formato original (Polars in, Polars out)
✓ Todas as funcionalidades funcionam igualmente (resume, retry, etc.)
✓ Para datasets grandes, Polars oferece melhor performance
✓ A escolha entre Pandas e Polars não afeta o uso do DataFrameIt

💡 DICA: Se está começando, use Pandas. Se precisa de performance
         em datasets grandes, migre para Polars - a API do DataFrameIt
         permanece a mesma!
""")

print("\n" + "=" * 80)
print("EXEMPLO CONCLUÍDO!")
print("=" * 80)
