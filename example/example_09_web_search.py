"""
Exemplo 09: Busca Web com Tavily
================================

Este exemplo demonstra como usar busca web via Tavily para enriquecer
DataFrames com informacoes atualizadas da internet.

Conceitos demonstrados:
- Ativar busca web com use_search=True
- Configurar parametros de busca (max_results, search_depth)
- Usar search_per_field para modelos com muitos campos
- Rastrear creditos e contagem de buscas

Para executar este exemplo:
1. Configure suas chaves de API:
   export TAVILY_API_KEY="sua-chave-tavily"
   export OPENAI_API_KEY="sua-chave-openai"  # ou outro provider
2. Instale dependencias: pip install dataframeit[search,openai]
3. Execute: python3 example_09_web_search.py

Nota: A busca web consome creditos do Tavily.
- search_depth="basic": 1 credito por busca
- search_depth="advanced": 2 creditos por busca
Free tier: 1000 creditos/mes
"""

from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
from dataframeit import dataframeit

# ============================================================================
# 1. DEFINIR MODELO PYDANTIC
# ============================================================================
# Modelo para informacoes de medicamentos que requerem busca na web


class MedicamentoInfo(BaseModel):
    """Estrutura para informacoes de medicamentos."""

    principio_ativo: str = Field(
        ...,
        description="Principio ativo principal do medicamento"
    )

    indicacao: str = Field(
        ...,
        description="Indicacao terapeutica principal"
    )

    contraindicacoes: str = Field(
        ...,
        description="Principais contraindicacoes"
    )

    forma_farmaceutica: Literal[
        'comprimido', 'capsula', 'liquido', 'injetavel', 'pomada', 'outro'
    ] = Field(
        ...,
        description="Forma farmaceutica mais comum"
    )


# ============================================================================
# 2. DEFINIR TEMPLATE DO PROMPT
# ============================================================================
TEMPLATE = """
Voce e um farmaceutico especializado.

Pesquise informacoes sobre o medicamento abaixo e extraia os dados solicitados.

Medicamento: {texto}

Use a ferramenta de busca para encontrar informacoes atualizadas e confiaveis.
Priorize fontes como bulas, Anvisa, e sites medicos reconhecidos.
"""

# ============================================================================
# 3. CRIAR DADOS DE EXEMPLO
# ============================================================================
dados = {
    'id': [1, 2, 3],
    'medicamento': [
        "Paracetamol",
        "Ibuprofeno",
        "Dipirona",
    ]
}

df = pd.DataFrame(dados)

print("=" * 80)
print("DATAFRAME ORIGINAL")
print("=" * 80)
print(df)
print("\n")

# ============================================================================
# 4. PROCESSAR COM BUSCA WEB
# ============================================================================
print("=" * 80)
print("PROCESSANDO COM BUSCA WEB (TAVILY)...")
print("=" * 80)

# O dataframeit vai usar um agente LangChain com acesso a ferramenta de busca
# Tavily para pesquisar informacoes antes de responder
df_resultado = dataframeit(
    df,
    MedicamentoInfo,
    TEMPLATE,
    text_column='medicamento',
    # Configuracao de busca
    use_search=True,              # Ativa busca web via Tavily
    max_results=5,                # Resultados por busca (1-20)
    max_searches=3,               # Max buscas por linha
    search_depth="basic",         # "basic" (1 credito) ou "advanced" (2 creditos)
    # Configuracao do LLM
    model='gpt-4o-mini',          # Modelo recomendado para agentes
    provider='openai',            # Provider (openai, google_genai, anthropic)
    parallel_requests=2,          # Linhas processadas em paralelo
    resume=True,                  # Permite continuar se interrompido
)

# ============================================================================
# 5. VISUALIZAR RESULTADOS
# ============================================================================
print("\n")
print("=" * 80)
print("RESULTADOS")
print("=" * 80)

colunas_exibir = [
    'id', 'medicamento', 'principio_ativo',
    'indicacao', 'forma_farmaceutica'
]
print(df_resultado[colunas_exibir].to_string())

# ============================================================================
# 6. VISUALIZAR METRICAS DE BUSCA
# ============================================================================
print("\n")
print("=" * 80)
print("METRICAS DE BUSCA")
print("=" * 80)

if '_search_count' in df_resultado.columns:
    total_buscas = df_resultado['_search_count'].sum()
    print(f"Total de buscas realizadas: {total_buscas}")

if '_search_credits' in df_resultado.columns:
    total_creditos = df_resultado['_search_credits'].sum()
    print(f"Creditos Tavily consumidos: {total_creditos}")

if '_total_tokens' in df_resultado.columns:
    total_tokens = df_resultado['_total_tokens'].sum()
    print(f"Total de tokens utilizados: {total_tokens}")

# ============================================================================
# 7. EXEMPLO AVANCADO: BUSCA POR CAMPO
# ============================================================================
print("\n")
print("=" * 80)
print("EXEMPLO AVANCADO: BUSCA POR CAMPO (search_per_field)")
print("=" * 80)
print("""
Para modelos com muitos campos, use search_per_field=True.
Isso executa um agente separado para cada campo do modelo Pydantic,
evitando sobrecarga de contexto.

Exemplo:
    df_resultado = dataframeit(
        df,
        ModeloComplexo,
        TEMPLATE,
        use_search=True,
        search_per_field=True,  # Um agente por campo
        ...
    )

Nota: search_per_field aumenta o numero de buscas e tokens,
mas melhora a qualidade para modelos complexos.
""")

print("\n" + "=" * 80)
print("EXEMPLO CONCLUIDO!")
print("=" * 80)
