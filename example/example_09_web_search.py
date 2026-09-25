"""
Exemplo 09: Busca Web
=====================

Este exemplo mostra como usar busca web (Tavily ou Exa) para enriquecer
DataFrames com informações atualizadas da internet.

Conceitos demonstrados:
- Ativar busca web com use_search=True
- Escolher o provedor de busca com search_provider ("tavily" ou "exa")
- Configurar parâmetros de busca (max_results, search_depth, max_search_calls)
- Usar search_per_field para modelos com muitos campos
- Acompanhar créditos de busca e tokens

Para executar este exemplo:
1. Configure suas chaves de API:
   export TAVILY_API_KEY="sua-chave-tavily"   # ou EXA_API_KEY, com search_provider="exa"
   export OPENAI_API_KEY="sua-chave-openai"   # ou a chave de outro provider
2. Instale as dependências: pip install dataframeit[search,openai]
   (para Exa: pip install dataframeit[search-exa,openai])
3. Execute: python3 example_09_web_search.py

Custos: com use_search=True, cada linha é processada por um agente que decide
quando buscar e pode fazer até max_search_calls buscas (padrão 10). Uma linha
pode, portanto, consumir várias buscas. No Tavily:
- search_depth="basic": 1 crédito por busca
- search_depth="advanced": 2 créditos por busca
Os créditos gastos ficam na coluna _search_credits e no resumo ao fim da execução.
"""

from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
from dataframeit import dataframeit

# ============================================================================
# 1. DEFINIR MODELO PYDANTIC
# ============================================================================
# Modelo para informações de medicamentos que exigem busca na web


class MedicamentoInfo(BaseModel):
    """Estrutura para informações de medicamentos."""

    principio_ativo: str = Field(
        ...,
        description="Princípio ativo principal do medicamento"
    )

    indicacao: str = Field(
        ...,
        description="Indicação terapêutica principal"
    )

    contraindicacoes: str = Field(
        ...,
        description="Principais contraindicações"
    )

    forma_farmaceutica: Literal[
        'comprimido', 'capsula', 'liquido', 'injetavel', 'pomada', 'outro'
    ] = Field(
        ...,
        description="Forma farmacêutica mais comum"
    )


# ============================================================================
# 2. DEFINIR TEMPLATE DO PROMPT
# ============================================================================
TEMPLATE = """
Você é um farmacêutico especializado.

Pesquise informações sobre o medicamento abaixo e extraia os dados solicitados.

Medicamento: {texto}

Use a ferramenta de busca para encontrar informações atualizadas e confiáveis.
Priorize fontes como bulas, Anvisa e sites médicos reconhecidos.
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
print("PROCESSANDO COM BUSCA WEB...")
print("=" * 80)

# O dataframeit usa um agente LangChain com acesso à ferramenta de busca, que
# pesquisa antes de responder. Cada linha pode gerar até max_search_calls buscas.
df_resultado = dataframeit(
    df,
    MedicamentoInfo,
    TEMPLATE,
    text_column='medicamento',
    # Configuração de busca
    use_search=True,              # Ativa a busca web
    search_provider="tavily",     # "tavily" (padrão) ou "exa", que usa EXA_API_KEY
    max_results=5,                # Resultados por busca (1-20)
    search_depth="basic",         # Tavily: "basic" (1 crédito) ou "advanced" (2 créditos)
    max_search_calls=3,           # Limite de buscas do agente em cada linha (padrão: 10)
    # Configuração do LLM
    provider='openai',            # Provider (openai, google_genai, anthropic, ...)
    model='gpt-6-luna',           # Modelo padrão do provider openai
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
# 6. VISUALIZAR MÉTRICAS DE BUSCA
# ============================================================================
print("\n")
print("=" * 80)
print("MÉTRICAS DE BUSCA")
print("=" * 80)

if '_search_credits' in df_resultado.columns:
    total_creditos = df_resultado['_search_credits'].sum()
    print(f"Créditos de busca consumidos: {total_creditos}")

if '_input_tokens' in df_resultado.columns:
    total_tokens = (
        df_resultado['_input_tokens'].fillna(0).sum()
        + df_resultado['_output_tokens'].fillna(0).sum()
    )
    print(f"Total de tokens utilizados: {int(total_tokens)}")

# ============================================================================
# 7. EXEMPLO AVANÇADO: BUSCA POR CAMPO
# ============================================================================
print("\n")
print("=" * 80)
print("EXEMPLO AVANÇADO: BUSCA POR CAMPO (search_per_field)")
print("=" * 80)
print("""
Para modelos com muitos campos, use search_per_field=True.
Isso executa um agente separado para cada campo do modelo Pydantic,
evitando sobrecarga de contexto. Cada agente tem o próprio limite de
max_search_calls buscas.

Exemplo:
    df_resultado = dataframeit(
        df,
        ModeloComplexo,
        TEMPLATE,
        use_search=True,
        search_per_field=True,  # Um agente por campo
        ...
    )

Nota: search_per_field aumenta o número de buscas e tokens,
mas melhora a qualidade para modelos complexos.
""")

# ============================================================================
# 8. EXEMPLO AVANÇADO: SALVAR TRACE DO AGENTE
# ============================================================================
print("\n")
print("=" * 80)
print("EXEMPLO AVANÇADO: SALVAR TRACE DO AGENTE (save_trace)")
print("=" * 80)
print("""
Para depurar e auditar o raciocínio do agente, use save_trace=True.
Isso salva o trace completo em uma coluna _trace (JSON), incluindo:
- Todas as mensagens da conversa (human, ai, tool)
- Queries de busca realizadas
- Contagem de tool calls
- Duração e modelo usado

Exemplo:
    df_resultado = dataframeit(
        df,
        MedicamentoInfo,
        TEMPLATE,
        use_search=True,
        save_trace=True,     # Trace completo
        # ou save_trace="minimal"  # Apenas queries, sem conteudo de busca
    )

    # Acessar trace da primeira linha
    import json
    trace = json.loads(df_resultado['_trace'].iloc[0])
    print(f"Buscas realizadas: {trace['search_queries']}")
    print(f"Duração: {trace['duration_seconds']}s")
    print(f"Modelo: {trace['model']}")

Com search_per_field=True, gera colunas _trace_{campo} para cada campo.
""")

print("\n" + "=" * 80)
print("EXEMPLO CONCLUÍDO!")
print("=" * 80)
