"""
Exemplo 03: Tratamento de Erros e Retry
========================================

Este exemplo demonstra como o DataFrameIt lida com erros e como você pode
analisar e recuperar de falhas no processamento.

Conceitos demonstrados:
- Sistema automático de retry com backoff exponencial
- Classificação inteligente de erros (recuperáveis vs não-recuperáveis)
- Warnings informativos durante retries
- Coluna _dataframeit_status (processed, error, None)
- Coluna error_details com detalhes de erros e contagem de retries
- Configuração de retry customizado
- Filtragem de linhas com erro
- Estratégias de recuperação

Para executar este exemplo:
1. Configure sua chave de API: export GOOGLE_API_KEY="sua-chave"
2. Instale dependências: pip install dataframeit langchain langchain-core langchain-google-genai
3. Execute: python3 example_03_error_handling.py
"""

from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
from dataframeit import dataframeit

# ============================================================================
# 1. DEFINIR MODELO PYDANTIC
# ============================================================================
class ProductCategory(BaseModel):
    """Estrutura para categorização de produtos."""

    categoria: Literal['eletronicos', 'roupas', 'alimentos', 'livros', 'outros'] = Field(
        ...,
        description="Categoria principal do produto"
    )

    subcategoria: str = Field(
        ...,
        description="Subcategoria mais específica"
    )

    preco_estimado: Literal['barato', 'medio', 'caro', 'premium'] = Field(
        ...,
        description="Faixa de preço estimada baseada na descrição"
    )

# ============================================================================
# 2. DEFINIR TEMPLATE DO PROMPT
# ============================================================================
TEMPLATE = """
Categorize o produto descrito abaixo.

Descrição do produto:
{documento}

Seja preciso na categorização.
"""

# ============================================================================
# 3. CRIAR DADOS COM CASOS PROBLEMÁTICOS
# ============================================================================
# Incluímos alguns casos que podem gerar erros:
# - Textos muito curtos
# - Textos vazios
# - Textos ambíguos
dados = {
    'id': [1, 2, 3, 4, 5, 6, 7, 8],
    'texto': [
        "Smartphone Samsung Galaxy S23 Ultra 256GB com câmera de 200MP",     # OK
        "",                                                                   # VAZIO - pode causar erro
        "Livro 'Clean Code' de Robert C. Martin, programação",              # OK
        "x",                                                                  # MUITO CURTO - pode causar erro
        "Tênis Nike Air Max running profissional",                           # OK
        "Pizza margherita congelada 400g",                                   # OK
        "    ",                                                               # SÓ ESPAÇOS - pode causar erro
        "Fone de ouvido Bluetooth com cancelamento de ruído ativo"          # OK
    ]
}

df = pd.DataFrame(dados)

print("=" * 80)
print("DADOS ORIGINAIS (com casos problemáticos)")
print("=" * 80)
print(df)
print("\n⚠ Observe: Algumas linhas têm textos vazios ou muito curtos\n")

# ============================================================================
# 4. PROCESSAR COM CONFIGURAÇÃO DE RETRY
# ============================================================================
print("=" * 80)
print("PROCESSANDO COM RETRY CONFIGURADO...")
print("=" * 80)

# Configurar retry mais tolerante
df_resultado = dataframeit(
    df,
    ProductCategory,
    TEMPLATE,
    text_column='texto',
    max_retries=3,          # Tentar até 3 vezes (padrão: 3)
    base_delay=1.0,         # Começar com 1 segundo (padrão: 1.0)
    max_delay=10.0,         # Máximo 10 segundos entre tentativas (padrão: 30.0)
    resume=True
)

print("\n✓ Processamento concluído (com possíveis erros)\n")

# ============================================================================
# 5. ANALISAR STATUS DO PROCESSAMENTO
# ============================================================================
print("=" * 80)
print("ANÁLISE DE STATUS")
print("=" * 80)

# Verificar coluna de status automática
print("\nStatus de cada linha:")
print(df_resultado[['id', '_dataframeit_status']].to_string(index=False))

# Contar status
print("\n" + "-" * 80)
print("Resumo:")
status_counts = df_resultado['_dataframeit_status'].value_counts()
total = len(df_resultado)

print(f"  ✓ Processadas com sucesso: {status_counts.get('processed', 0)}/{total}")
print(f"  ✗ Com erro: {status_counts.get('error', 0)}/{total}")
print(f"  ⊗ Não processadas: {df_resultado['_dataframeit_status'].isna().sum()}/{total}")

# ============================================================================
# 6. ANALISAR ERROS EM DETALHES
# ============================================================================
print("\n" + "=" * 80)
print("DETALHES DOS ERROS")
print("=" * 80)

# Filtrar apenas linhas com erro
linhas_com_erro = df_resultado[df_resultado['_dataframeit_status'] == 'error']

if len(linhas_com_erro) > 0:
    print(f"\n⚠ {len(linhas_com_erro)} linha(s) com erro:\n")

    for idx, row in linhas_com_erro.iterrows():
        print(f"Linha {row['id']}:")
        print(f"  Texto: '{row['texto']}'")
        print(f"  Erro: {row['error_details']}")
        print()
else:
    print("\n✓ Nenhum erro encontrado! Todas as linhas foram processadas com sucesso.")

# ============================================================================
# 7. ESTRATÉGIAS DE RECUPERAÇÃO
# ============================================================================
print("=" * 80)
print("ESTRATÉGIAS DE RECUPERAÇÃO")
print("=" * 80)

# Estratégia 1: Salvar apenas linhas bem-sucedidas
linhas_sucesso = df_resultado[df_resultado['_dataframeit_status'] == 'processed']
print(f"\n1. Salvar apenas sucesso: {len(linhas_sucesso)} linhas")
# linhas_sucesso.to_excel('resultado_limpo.xlsx', index=False)

# Estratégia 2: Analisar e corrigir linhas com erro
print(f"\n2. Corrigir linhas com erro: {len(linhas_com_erro)} linhas")
if len(linhas_com_erro) > 0:
    print("   Possíveis soluções:")
    print("   - Remover textos vazios ou muito curtos")
    print("   - Adicionar contexto aos textos ambíguos")
    print("   - Simplificar o modelo Pydantic")
    print("   - Aumentar max_retries")

# Estratégia 3: Pré-processar dados para evitar erros
print("\n3. Pré-processar dados:")
df_limpo = df_resultado.copy()
df_limpo = df_limpo[df_limpo['texto'].str.strip().str.len() > 5]  # Remover textos curtos
print(f"   Após limpeza: {len(df_limpo)} linhas válidas")

# ============================================================================
# 8. VISUALIZAR RESULTADOS VÁLIDOS
# ============================================================================
if len(linhas_sucesso) > 0:
    print("\n" + "=" * 80)
    print("RESULTADOS DAS LINHAS PROCESSADAS COM SUCESSO")
    print("=" * 80)

    colunas_exibir = ['id', 'categoria', 'subcategoria', 'preco_estimado']
    print(linhas_sucesso[colunas_exibir].to_string(index=False))

    print("\n" + "-" * 80)
    print("Estatísticas:")
    print(f"\nDistribuição por categoria:")
    print(linhas_sucesso['categoria'].value_counts())

# ============================================================================
# 9. EXEMPLO DE RETRY MAIS AGRESSIVO
# ============================================================================
print("\n" + "=" * 80)
print("EXEMPLO: CONFIGURAÇÃO DE RETRY MAIS AGRESSIVA")
print("=" * 80)

print("""
Para casos onde você quer máxima tolerância a erros:

df_resultado = dataframeit(
    df,
    ProductCategory,
    TEMPLATE,
    max_retries=5,        # Tentar até 5 vezes
    base_delay=2.0,       # Começar com 2 segundos
    max_delay=60.0,       # Esperar até 60 segundos entre tentativas
    text_column='texto'
)

Progressão do delay: 2s → 4s → 8s → 16s → 32s (limitado a 60s)

⚠ ATENÇÃO: Mais retries = mais tempo e mais custo de API!
Use com moderação.
""")

# ============================================================================
# 10. RESUMO DE BOAS PRÁTICAS
# ============================================================================
print("=" * 80)
print("BOAS PRÁTICAS PARA TRATAMENTO DE ERROS")
print("=" * 80)

print("""
✓ SEMPRE verificar _dataframeit_status após o processamento
✓ Analisar error_details para entender falhas (inclui contagem de retries)
✓ Pré-processar dados (remover textos vazios, muito curtos, etc.)
✓ Usar resume=True para poder continuar após interrupções
✓ Começar com max_retries baixo (3) e aumentar se necessário
✓ Salvar progresso intermediário em arquivos
✓ Considerar processamento em lotes para datasets grandes
✗ NÃO ignorar erros sem análise
✗ NÃO usar retry infinito ou valores muito altos
✗ NÃO processar sem validar dados primeiro

Tipos de erro detectados automaticamente:
- Recuperáveis: Timeout, Rate Limit, 5xx - farão retry automaticamente
- Não-recuperáveis: API key inválida, 401, 403 - falham imediatamente
""")

print("\n" + "=" * 80)
print("EXEMPLO CONCLUÍDO!")
print("=" * 80)
