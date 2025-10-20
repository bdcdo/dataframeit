"""
Exemplo 04: Processamento Incremental com Resume
=================================================

Este exemplo demonstra como usar resume=True para processar datasets grandes
de forma incremental, com capacidade de pausar e continuar.

Conceitos demonstrados:
- Uso de resume=True
- Salvamento de progresso parcial
- Continuação após interrupção
- Monitoramento de progresso
- Estratégias para datasets grandes

Para executar este exemplo:
1. Configure sua chave de API: export GOOGLE_API_KEY="sua-chave"
2. Instale dependências: pip install dataframeit langchain langchain-core langchain-google-genai openpyxl
3. Execute: python3 example_04_resume.py
"""

from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
from dataframeit import dataframeit
import os

# ============================================================================
# 1. DEFINIR MODELO PYDANTIC
# ============================================================================
class EmailClassification(BaseModel):
    """Estrutura para classificação de emails."""

    tipo: Literal['trabalho', 'pessoal', 'spam', 'newsletter', 'urgente'] = Field(
        ...,
        description="Tipo/categoria do email"
    )

    prioridade: Literal['alta', 'media', 'baixa'] = Field(
        ...,
        description="Prioridade do email"
    )

    requer_resposta: Literal['sim', 'nao', 'opcional'] = Field(
        ...,
        description="Se o email requer resposta"
    )

# ============================================================================
# 2. DEFINIR TEMPLATE DO PROMPT
# ============================================================================
TEMPLATE = """
Classifique o email abaixo.

Email:
{documento}

Seja preciso na classificação.
"""

# ============================================================================
# 3. CRIAR DATASET SIMULADO (médio porte)
# ============================================================================
# Simulamos 20 emails para demonstrar processamento incremental
emails = [
    "Reunião de planejamento do Q4 amanhã às 14h. Presença obrigatória.",
    "Oi! Vamos jantar no sábado?",
    "PROMOÇÃO IMPERDÍVEL! 50% de desconto em todos os produtos!",
    "Seu relatório mensal está pronto para revisão.",
    "Parabéns! Você ganhou um iPhone! Clique aqui para resgatar.",
    "Newsletter semanal: As principais notícias de tecnologia",
    "URGENTE: Sistema fora do ar - necessária ação imediata",
    "Lembrete: Aniversário da Maria é amanhã",
    "Proposta comercial anexa para sua análise",
    "Deseja renovar sua assinatura? Desconto especial!",
    "Ata da última reunião para aprovação",
    "Fotos do fim de semana que você pediu",
    "Boleto com vencimento hoje - evite multa",
    "Novo episódio do podcast que você segue",
    "Preciso da sua aprovação urgente no documento",
    "Convite: Churrasco no domingo às 15h",
    "Seu pedido #12345 foi enviado",
    "Comentários na sua publicação do blog",
    "Atualização de política de privacidade",
    "Feliz aniversário! Desejo tudo de bom!"
]

dados = {
    'email_id': [f'E{i:03d}' for i in range(1, len(emails) + 1)],
    'texto': emails
}

df_original = pd.DataFrame(dados)

# ============================================================================
# 4. CENÁRIO 1: PROCESSAMENTO COMPLETO COM RESUME
# ============================================================================
print("=" * 80)
print("CENÁRIO 1: PROCESSAMENTO COMPLETO")
print("=" * 80)

print(f"\n📊 Dataset: {len(df_original)} emails para processar")
print("\nIniciando processamento com resume=True...")

# Processar com resume ativado
df_resultado = dataframeit(
    df_original,
    EmailClassification,
    TEMPLATE,
    text_column='texto',
    resume=True,  # CHAVE: Permite continuar de onde parou
    model='gemini-2.0-flash-exp'
)

print("\n✓ Processamento completo concluído!")

# Salvar resultado
arquivo_parcial = 'emails_processamento_parcial.xlsx'
df_resultado.to_excel(arquivo_parcial, index=False)
print(f"✓ Progresso salvo em: {arquivo_parcial}")

# ============================================================================
# 5. ANÁLISE DO PROGRESSO
# ============================================================================
print("\n" + "=" * 80)
print("ANÁLISE DE PROGRESSO")
print("=" * 80)

status_counts = df_resultado['_dataframeit_status'].value_counts()
total = len(df_resultado)

processados = status_counts.get('processed', 0)
com_erro = status_counts.get('error', 0)
nao_processados = df_resultado['_dataframeit_status'].isna().sum()

print(f"\n  ✓ Processados: {processados}/{total} ({processados/total*100:.1f}%)")
print(f"  ✗ Com erro: {com_erro}/{total}")
print(f"  ⊗ Não processados: {nao_processados}/{total}")

# ============================================================================
# 6. VISUALIZAR RESULTADOS
# ============================================================================
print("\n" + "=" * 80)
print("RESULTADOS DA CLASSIFICAÇÃO")
print("=" * 80)

# Mostrar apenas emails processados
emails_processados = df_resultado[df_resultado['_dataframeit_status'] == 'processed']

if len(emails_processados) > 0:
    print("\nPrimeiros 5 resultados:")
    colunas = ['email_id', 'tipo', 'prioridade', 'requer_resposta']
    print(emails_processados[colunas].head().to_string(index=False))

    print("\n" + "-" * 80)
    print("Estatísticas:")

    print("\nDistribuição por tipo:")
    print(emails_processados['tipo'].value_counts())

    print("\nDistribuição por prioridade:")
    print(emails_processados['prioridade'].value_counts())

# ============================================================================
# 7. CENÁRIO 2: SIMULANDO INTERRUPÇÃO E RETOMADA
# ============================================================================
print("\n" + "=" * 80)
print("CENÁRIO 2: SIMULANDO INTERRUPÇÃO E RETOMADA")
print("=" * 80)

print("""
Imagine que o processamento foi interrompido após 10 emails.
Vamos simular isso e então retomar...
""")

# Criar dataset "parcialmente processado"
df_parcial = df_original.copy()

# Simular que apenas os primeiros 10 foram processados
# (adicionando as colunas necessárias manualmente)
df_parcial['tipo'] = None
df_parcial['prioridade'] = None
df_parcial['requer_resposta'] = None
df_parcial['_dataframeit_status'] = None
df_parcial['error_details'] = None

# Marcar primeiros 10 como processados (simulação)
df_parcial.loc[:9, '_dataframeit_status'] = 'processed'
df_parcial.loc[:9, 'tipo'] = 'trabalho'  # valores dummy
df_parcial.loc[:9, 'prioridade'] = 'media'
df_parcial.loc[:9, 'requer_resposta'] = 'sim'

print(f"✓ Simulação: 10/{len(df_parcial)} emails processados")
print("⊗ Simulação: Processamento interrompido!")

# Salvar estado parcial
arquivo_interrompido = 'emails_interrompido.xlsx'
df_parcial.to_excel(arquivo_interrompido, index=False)
print(f"✓ Estado salvo em: {arquivo_interrompido}")

print("\n📍 Retomando processamento...")

# Carregar e retomar
df_retomado = pd.read_excel(arquivo_interrompido)

# RESUME vai processar apenas as linhas restantes!
df_completo = dataframeit(
    df_retomado,
    EmailClassification,
    TEMPLATE,
    text_column='texto',
    resume=True,  # Continua de onde parou!
    model='gemini-2.0-flash-exp'
)

print("\n✓ Processamento retomado e concluído!")

# Salvar resultado final
arquivo_final = 'emails_completo.xlsx'
df_completo.to_excel(arquivo_final, index=False)
print(f"✓ Resultado final salvo em: {arquivo_final}")

# ============================================================================
# 8. ESTRATÉGIAS PARA DATASETS GRANDES
# ============================================================================
print("\n" + "=" * 80)
print("ESTRATÉGIAS PARA DATASETS GRANDES")
print("=" * 80)

print("""
╔════════════════════════════════════════════════════════════════════════╗
║ 1. PROCESSAMENTO INCREMENTAL COM CHECKPOINTS                          ║
╠════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║ # Processar em blocos de 100 linhas                                   ║
║ for i in range(0, len(df), 100):                                      ║
║     bloco = df[i:i+100]                                                ║
║     resultado = dataframeit(bloco, Model, TEMPLATE, resume=True)      ║
║     resultado.to_excel(f'checkpoint_{i}.xlsx', index=False)           ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝

╔════════════════════════════════════════════════════════════════════════╗
║ 2. MONITORAMENTO DE PROGRESSO                                         ║
╠════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║ # Verificar progresso periodicamente                                  ║
║ df_resultado.to_excel('backup_temp.xlsx', index=False)                ║
║                                                                        ║
║ # Checar quantos faltam                                                ║
║ faltam = df_resultado['_dataframeit_status'].isna().sum()             ║
║ print(f"Faltam {faltam} linhas")                                      ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝

╔════════════════════════════════════════════════════════════════════════╗
║ 3. RECUPERAÇÃO DE FALHAS                                              ║
╠════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║ # Se houver falha, apenas recarregue e continue                       ║
║ df = pd.read_excel('backup_temp.xlsx')                                ║
║ df = dataframeit(df, Model, TEMPLATE, resume=True)                    ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
""")

# ============================================================================
# 9. BOAS PRÁTICAS
# ============================================================================
print("=" * 80)
print("BOAS PRÁTICAS PARA PROCESSAMENTO INCREMENTAL")
print("=" * 80)

print("""
✓ SEMPRE use resume=True para datasets médios/grandes
✓ Salve progresso periodicamente (.xlsx ou .csv)
✓ Monitore _dataframeit_status para ver progresso
✓ Use nomes de arquivo com timestamp para backups
✓ Processe em blocos para datasets muito grandes (>1000 linhas)
✓ Tenha uma estratégia de backup antes de começar
✓ Teste com subset pequeno antes do dataset completo

✗ NÃO processe tudo de uma vez sem checkpoints
✗ NÃO esqueça de salvar progresso intermediário
✗ NÃO use resume=False em datasets grandes
""")

# ============================================================================
# 10. LIMPEZA
# ============================================================================
print("\n" + "=" * 80)
print("LIMPEZA DE ARQUIVOS TEMPORÁRIOS")
print("=" * 80)

arquivos_temp = [arquivo_parcial, arquivo_interrompido, arquivo_final]
for arquivo in arquivos_temp:
    if os.path.exists(arquivo):
        os.remove(arquivo)
        print(f"✓ Removido: {arquivo}")

print("\n" + "=" * 80)
print("EXEMPLO CONCLUÍDO!")
print("=" * 80)
