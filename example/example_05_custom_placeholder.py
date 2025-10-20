"""
Exemplo 05: Placeholder Customizado
====================================

Este exemplo demonstra como customizar o nome do placeholder no template
em vez de usar o padrão {documento}.

Conceitos demonstrados:
- Parâmetro placeholder para customização
- Uso de placeholders com nomes semânticos
- Múltiplos estilos de template
- Quando e por que customizar placeholders

Para executar este exemplo:
1. Configure sua chave de API: export GOOGLE_API_KEY="sua-chave"
2. Instale dependências: pip install dataframeit langchain langchain-core langchain-google-genai
3. Execute: python3 example_05_custom_placeholder.py
"""

from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
from dataframeit import dataframeit

# ============================================================================
# EXEMPLO 1: PLACEHOLDER PADRÃO {documento}
# ============================================================================
print("=" * 80)
print("EXEMPLO 1: PLACEHOLDER PADRÃO {documento}")
print("=" * 80)

class Sentiment(BaseModel):
    """Estrutura para análise de sentimento."""
    sentimento: Literal['positivo', 'negativo', 'neutro'] = Field(
        ...,
        description="Sentimento expresso"
    )

# Template usando placeholder padrão
TEMPLATE_PADRAO = """
Analise o sentimento do seguinte documento:

{documento}

Classifique o sentimento.
"""

dados1 = {
    'id': [1, 2],
    'texto': [
        "Adorei este produto! Muito bom!",
        "Péssimo, não recomendo."
    ]
}

df1 = pd.DataFrame(dados1)
print("\nProcessando com placeholder padrão {documento}...")

df1_resultado = dataframeit(
    df1,
    Sentiment,
    TEMPLATE_PADRAO,
    text_column='texto',
    # placeholder='documento' é o padrão, não precisa especificar
    model='gemini-2.0-flash-exp'
)

print("\n✓ Resultado:")
print(df1_resultado[['id', 'sentimento']].to_string(index=False))

# ============================================================================
# EXEMPLO 2: PLACEHOLDER CUSTOMIZADO {review}
# ============================================================================
print("\n" + "=" * 80)
print("EXEMPLO 2: PLACEHOLDER CUSTOMIZADO {review}")
print("=" * 80)

class ReviewAnalysis(BaseModel):
    """Análise de review de produto."""
    nota: Literal['1', '2', '3', '4', '5'] = Field(
        ...,
        description="Nota de 1 a 5 estrelas"
    )

# Template usando placeholder customizado {review}
TEMPLATE_REVIEW = """
Você é um analista de reviews de produtos.

Analise o review abaixo e atribua uma nota de 1 a 5 estrelas:

📝 Review do cliente:
{review}

Atribua a nota baseada no sentimento geral.
"""

dados2 = {
    'produto': ['Notebook', 'Mouse'],
    'texto': [
        "Excelente notebook, desempenho incrível!",
        "Mouse comum, funciona mas nada de especial."
    ]
}

df2 = pd.DataFrame(dados2)
print("\nProcessando com placeholder customizado {review}...")

df2_resultado = dataframeit(
    df2,
    ReviewAnalysis,
    TEMPLATE_REVIEW,
    text_column='texto',
    placeholder='review',  # CUSTOMIZA O PLACEHOLDER!
    model='gemini-2.0-flash-exp'
)

print("\n✓ Resultado:")
print(df2_resultado[['produto', 'nota']].to_string(index=False))

# ============================================================================
# EXEMPLO 3: PLACEHOLDER SEMÂNTICO {post_social_media}
# ============================================================================
print("\n" + "=" * 80)
print("EXEMPLO 3: PLACEHOLDER SEMÂNTICO {post_social_media}")
print("=" * 80)

class SocialMediaAnalysis(BaseModel):
    """Análise de post em redes sociais."""
    tipo_conteudo: Literal['promocional', 'informativo', 'pessoal', 'engajamento'] = Field(
        ...,
        description="Tipo de conteúdo do post"
    )
    possui_call_to_action: Literal['sim', 'nao'] = Field(
        ...,
        description="Se o post possui call-to-action"
    )

# Template com placeholder semântico
TEMPLATE_SOCIAL = """
Você é um analista de redes sociais.

Analise o post abaixo:

🔹 POST:
{post_social_media}

Classifique o tipo de conteúdo e identifique se há call-to-action.
"""

dados3 = {
    'plataforma': ['Instagram', 'Twitter'],
    'texto': [
        "Novo produto lançado! Compre agora com 30% de desconto! Link na bio 🎉",
        "Bom dia! Como vocês estão hoje? ☀️"
    ]
}

df3 = pd.DataFrame(dados3)
print("\nProcessando com placeholder semântico {post_social_media}...")

df3_resultado = dataframeit(
    df3,
    SocialMediaAnalysis,
    TEMPLATE_SOCIAL,
    text_column='texto',
    placeholder='post_social_media',  # Nome descritivo e semântico
    model='gemini-2.0-flash-exp'
)

print("\n✓ Resultado:")
print(df3_resultado[['plataforma', 'tipo_conteudo', 'possui_call_to_action']].to_string(index=False))

# ============================================================================
# EXEMPLO 4: PLACEHOLDER {sentenca} (como no exemplo jurídico)
# ============================================================================
print("\n" + "=" * 80)
print("EXEMPLO 4: PLACEHOLDER {sentenca} (contexto jurídico)")
print("=" * 80)

class DecisionAnalysis(BaseModel):
    """Análise simplificada de decisão judicial."""
    resultado: Literal['deferido', 'indeferido', 'parcialmente_deferido'] = Field(
        ...,
        description="Resultado do pedido"
    )

# Template no estilo jurídico
TEMPLATE_JURIDICO = """
Você é um assistente jurídico.

Analise a sentença judicial abaixo:

INÍCIO DA SENTENÇA
{sentenca}
FIM DA SENTENÇA

Identifique o resultado do pedido.
"""

dados4 = {
    'processo': ['0001', '0002'],
    'texto': [
        "Diante do exposto, DEFIRO o pedido do autor.",
        "Ante o exposto, INDEFIRO o pedido por falta de provas."
    ]
}

df4 = pd.DataFrame(dados4)
print("\nProcessando com placeholder {sentenca}...")

df4_resultado = dataframeit(
    df4,
    DecisionAnalysis,
    TEMPLATE_JURIDICO,
    text_column='texto',
    placeholder='sentenca',  # Placeholder específico do domínio jurídico
    model='gemini-2.0-flash-exp'
)

print("\n✓ Resultado:")
print(df4_resultado[['processo', 'resultado']].to_string(index=False))

# ============================================================================
# QUANDO CUSTOMIZAR O PLACEHOLDER?
# ============================================================================
print("\n" + "=" * 80)
print("QUANDO CUSTOMIZAR O PLACEHOLDER?")
print("=" * 80)

print("""
╔════════════════════════════════════════════════════════════════════════╗
║ USE {documento} (padrão) QUANDO:                                       ║
╠════════════════════════════════════════════════════════════════════════╣
║ ✓ Não há contexto específico                                          ║
║ ✓ É um uso geral/genérico                                             ║
║ ✓ Você quer menos configuração                                        ║
║                                                                        ║
║ Exemplo:                                                               ║
║   "Analise o documento: {documento}"                                   ║
╚════════════════════════════════════════════════════════════════════════╝

╔════════════════════════════════════════════════════════════════════════╗
║ CUSTOMIZE O PLACEHOLDER QUANDO:                                       ║
╠════════════════════════════════════════════════════════════════════════╣
║ ✓ Há um termo específico do domínio (sentença, review, post, etc.)   ║
║ ✓ O contexto semântico ajuda o LLM a entender melhor                 ║
║ ✓ Você quer templates mais legíveis e auto-documentados              ║
║ ✓ Está adaptando templates existentes                                 ║
║                                                                        ║
║ Exemplos:                                                              ║
║   "Analise a sentença: {sentenca}"                                     ║
║   "Avalie o review: {review}"                                          ║
║   "Classifique o post: {post_social_media}"                            ║
║   "Processe o email: {email_content}"                                  ║
╚════════════════════════════════════════════════════════════════════════╝
""")

# ============================================================================
# BOAS PRÁTICAS
# ============================================================================
print("=" * 80)
print("BOAS PRÁTICAS PARA PLACEHOLDERS")
print("=" * 80)

print("""
✓ Use nomes descritivos e específicos do domínio
✓ Mantenha consistência nos seus templates
✓ Use snake_case para nomes compostos (post_social_media)
✓ Documente o placeholder usado em comentários
✓ Considere a semântica para ajudar o LLM

✗ NÃO use nomes genéricos quando há termo específico
✗ NÃO use espaços ou caracteres especiais no nome
✗ NÃO mude o placeholder sem atualizar o template
✗ NÃO use nomes muito longos (>20 caracteres)

Exemplos BOM:
  placeholder='sentenca'           ✓
  placeholder='review'             ✓
  placeholder='email_content'      ✓
  placeholder='post_social_media'  ✓

Exemplos RUIM:
  placeholder='x'                  ✗ (muito genérico)
  placeholder='meu texto aqui'     ✗ (tem espaços)
  placeholder='o_super_mega_...'   ✗ (muito longo)
""")

print("\n" + "=" * 80)
print("EXEMPLO CONCLUÍDO!")
print("=" * 80)

print("""
💡 DICA: O placeholder customizado é especialmente útil quando você está
adaptando templates de projetos existentes ou trabalhando em domínios
específicos onde a terminologia técnica importa.
""")
