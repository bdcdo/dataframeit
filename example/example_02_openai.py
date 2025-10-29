"""
Exemplo 02: Usando OpenAI
==========================

Este exemplo demonstra como usar o DataFrameIt com a API da OpenAI
em vez do LangChain.

Conceitos demonstrados:
- Configuração do cliente OpenAI
- Parâmetro use_openai=True
- Configurações específicas da OpenAI (reasoning_effort, verbosity)
- Comparação com LangChain

Para executar este exemplo:
1. Configure sua chave de API: export OPENAI_API_KEY="sua-chave"
2. Instale dependências: pip install dataframeit openai
3. Execute: python3 example_02_openai.py
"""

from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
from dataframeit import dataframeit
from openai import OpenAI

# ============================================================================
# 1. DEFINIR MODELO PYDANTIC
# ============================================================================
class ProductReview(BaseModel):
    """Estrutura para análise de reviews de produtos."""

    avaliacao: Literal['excelente', 'bom', 'regular', 'ruim', 'pessimo'] = Field(
        ...,
        description="Avaliação geral do produto baseada no review"
    )

    preco_justo: Literal['sim', 'nao', 'nao_mencionado'] = Field(
        ...,
        description="Se o cliente considera o preço justo"
    )

    recomendaria: Literal['sim', 'nao', 'talvez'] = Field(
        ...,
        description="Se o cliente recomendaria o produto"
    )

    aspecto_mais_positivo: str = Field(
        ...,
        description="Principal aspecto positivo mencionado (ou 'nenhum' se não houver)"
    )

# ============================================================================
# 2. DEFINIR TEMPLATE DO PROMPT
# ============================================================================
TEMPLATE = """
Você é um analista de reviews de e-commerce.

Analise o review abaixo e extraia as informações estruturadas solicitadas.
Seja objetivo e baseie-se apenas no texto fornecido.

Review:
{documento}
"""

# ============================================================================
# 3. CRIAR DADOS DE EXEMPLO
# ============================================================================
dados = {
    'produto_id': ['P001', 'P002', 'P003', 'P004'],
    'texto': [
        "Comprei este produto e adorei! A qualidade é excepcional, mesmo sendo um pouco caro vale cada centavo. Super recomendo!",
        "Produto mediano. Funciona mas nada de especial. O preço poderia ser mais baixo.",
        "Péssimo! Chegou com defeito e o suporte não resolveu. Não recomendo de jeito nenhum.",
        "Excelente custo-benefício. Produto simples mas faz exatamente o que promete."
    ]
}

df = pd.DataFrame(dados)

print("=" * 80)
print("REVIEWS ORIGINAIS")
print("=" * 80)
print(df)
print("\n")

# ============================================================================
# 4A. PROCESSAR COM OPENAI (MODO BÁSICO)
# ============================================================================
print("=" * 80)
print("PROCESSANDO COM OPENAI (MODO BÁSICO)...")
print("=" * 80)

# Modo mais simples: apenas ativar use_openai=True
# Usa as configurações padrão da OpenAI
df_resultado_basico = dataframeit(
    df,
    ProductReview,
    TEMPLATE,
    use_openai=True,                # ATIVA O PROVIDER OPENAI
    model='gpt-4o-mini',            # Modelo da OpenAI (mais econômico)
    text_column='texto'
)

print("\n✓ Processamento concluído!\n")

# ============================================================================
# 4B. PROCESSAR COM OPENAI (MODO CUSTOMIZADO)
# ============================================================================
print("=" * 80)
print("PROCESSANDO COM OPENAI (MODO CUSTOMIZADO)...")
print("=" * 80)

# Criar cliente OpenAI customizado (opcional)
# Útil se você quiser configurar timeout, proxy, etc.
client = OpenAI(
    # api_key="sua-chave-aqui",  # Opcional, usa env var se não especificar
    timeout=30.0,                 # Timeout customizado
)

# Modo avançado: cliente customizado + parâmetros específicos
df_resultado_custom = dataframeit(
    df,
    ProductReview,
    TEMPLATE,
    use_openai=True,
    openai_client=client,           # Cliente customizado
    model='gpt-4o-mini',
    reasoning_effort='medium',      # 'minimal' | 'low' | 'medium' | 'high'
    verbosity='medium',             # 'low' | 'medium' | 'high'
    text_column='texto'
)

print("\n✓ Processamento concluído!\n")

# ============================================================================
# 5. VISUALIZAR RESULTADOS
# ============================================================================
print("=" * 80)
print("RESULTADOS")
print("=" * 80)

colunas_exibir = ['produto_id', 'avaliacao', 'preco_justo', 'recomendaria', 'aspecto_mais_positivo']
print(df_resultado_custom[colunas_exibir])

print("\n")
print("=" * 80)
print("ESTATÍSTICAS")
print("=" * 80)

print("\nDistribuição de avaliações:")
print(df_resultado_custom['avaliacao'].value_counts())

print("\nPreço considerado justo:")
print(df_resultado_custom['preco_justo'].value_counts())

print("\nRecomendaria:")
print(df_resultado_custom['recomendaria'].value_counts())

# ============================================================================
# 6. COMPARAÇÃO: LANGCHAIN VS OPENAI
# ============================================================================
print("\n")
print("=" * 80)
print("LANGCHAIN VS OPENAI - QUANDO USAR CADA UM?")
print("=" * 80)

print("""
╔════════════════════════════════════════════════════════════════════════╗
║ LANGCHAIN (padrão)                                                     ║
╠════════════════════════════════════════════════════════════════════════╣
║ ✓ Suporte para múltiplos providers (Gemini, Claude, etc.)             ║
║ ✓ Gemini tem tier gratuito generoso                                   ║
║ ✓ Bom para prototipar e desenvolver                                   ║
║ ✓ Mais flexibilidade para trocar de modelo                            ║
║                                                                        ║
║ Exemplo:                                                               ║
║   dataframeit(df, Model, TEMPLATE)  # Usa Gemini por padrão           ║
╚════════════════════════════════════════════════════════════════════════╝

╔════════════════════════════════════════════════════════════════════════╗
║ OPENAI                                                                 ║
╠════════════════════════════════════════════════════════════════════════╣
║ ✓ GPT-4 é considerado mais poderoso para tarefas complexas            ║
║ ✓ Parâmetros específicos (reasoning_effort, verbosity)                ║
║ ✓ API estável e bem documentada                                       ║
║ ✓ Bom para produção quando performance é crítica                      ║
║                                                                        ║
║ Exemplo:                                                               ║
║   dataframeit(df, Model, TEMPLATE, use_openai=True, model='gpt-4o')   ║
╚════════════════════════════════════════════════════════════════════════╝
""")

print("\n" + "=" * 80)
print("EXEMPLO CONCLUÍDO!")
print("=" * 80)
