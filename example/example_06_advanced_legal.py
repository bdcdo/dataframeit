"""
Exemplo 06: Análise Jurídica Avançada
======================================

Este exemplo demonstra um caso de uso real e avançado do DataFrameIt:
análise automatizada de decisões judiciais sobre tratamentos de saúde.

Conceitos demonstrados:
- Modelo Pydantic complexo com classes aninhadas
- Campos opcionais e condicionais
- Uso de List e Tuple
- Tipos Literal com múltiplas opções
- Template detalhado e específico de domínio
- Placeholder customizado {sentenca}
- Tratamento de erros em produção

Este é um exemplo de caso real de uso em pesquisa jurídica.

Para executar este exemplo:
1. Configure sua chave de API: export GOOGLE_API_KEY="sua-chave"
2. Instale dependências: pip install dataframeit langchain langchain-core langchain-google-genai openpyxl
3. Coloque seus dados em um arquivo Excel ou use os dados de exemplo
4. Execute: python3 example_06_advanced_legal.py

NOTA: Este exemplo requer arquivos Excel específicos. Se você não os tiver,
      o exemplo usará dados sintéticos para demonstração.
"""

from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
from dataframeit import dataframeit
import os

# ============================================================================
# 1. DEFINIR MODELOS PYDANTIC COMPLEXOS
# ============================================================================

# Modelo aninhado para tratamento individual
class Tratamento(BaseModel):
    """Informações sobre um tratamento específico pleiteado."""

    nome: str | None = Field(
        ...,
        description="Nome do tratamento pleiteado. 'None' apenas se não foi pleiteado tratamento."
    )

    resultado: Literal['deferido', 'indeferido'] | None = Field(
        None,
        description="Se o tratamento indicado foi deferido ou indeferido. 'None' apenas se não foi pleiteado tratamento."
    )

# Modelo principal com múltiplos campos complexos
class RespostaFinalWrapper(BaseModel):
    """Estrutura completa para análise de decisão judicial sobre saúde."""

    definicao_escopo: Literal["dentro_do_escopo", "escopo_incerto", "fora_do_escopo"] = Field(
        ...,
        description="""Responda 'dentro_do_escopo' se o caso analisado estiver relacionado a casos que cumpram os seguintes requisitos:
1. Trata-se de um pedido de tratamento de saúde.
2. O autor da ação pleiteia tratamento para uma doença rara.

Se o caso não trata de pedido de tratamento de saúde, responda "fora_do_escopo".
Se você estiver em dúvida sobre se o autor da ação tem uma doença rara ou não, responda "escopo_incerto".

ATENÇÃO: É possível que a decisão mencione doenças raras apenas em uma citação de jurisprudência, mas o caso em si do autor não tenha relação com isso.
Considere essa possibilidade ao avaliar se o caso está ou não dentro do escopo.

Se o caso for classificado como "fora_do_escopo", responda todas as perguntas seguintes como "None"."""
    )

    justificativa: str = Field(
        ...,
        description="Justificativa da avaliação anterior sobre o escopo do caso."
    )

    condicoes_de_saude: list[str] | None = Field(
        None,
        description="Lista de nomes das condições de saúde do paciente (sejam elas raras ou não). Mantenha a mesma redação utilizada pelo magistrado ao se referir a cada doença."
    )

    tratamentos: list[Tratamento] | None = None

    genero_paciente: Literal['masculino', 'feminino', 'outro'] | None = Field(
        None,
        description='Gênero do autor da ação. Responda outro se houver mais de um autor ou não for possível identificar. Responda None se o caso é fora do escopo.'
    )

    justica_gratuita: Literal['nao', 'sim mas foi negado', 'sim e foi aceito'] | None = Field(
        None,
        description='Há menção na decisão ao fato de que o autor da ação requereu justiça gratuita?'
    )

    representacao: Literal['sim', 'nao'] | None = Field(
        None,
        description='O autor da ação é representado por seu responsável/genitor ou seu tutor?'
    )

    danos_morais_pedido: tuple[Literal['sim', 'nao'], float] | None = Field(
        ...,
        description='Se o paciente pediu danos morais e, se sim, quanto. Responda "nao" se não há menção a danos morais na decisão. Responda None se o caso é fora do escopo.'
    )

    danos_morais_concedido: tuple[Literal['sim', 'nao'], float] | None = Field(
        None,
        description='Se o magistrado concedeu o pedido danos morais e, se sim, de que valor. Responda None se não há menção a danos morais na decisão.'
    )

# ============================================================================
# 2. DEFINIR TEMPLATE ESPECIALIZADO
# ============================================================================

# IMPORTANTE: Usamos placeholder customizado {sentenca} em vez de {documento}
# porque é mais semântico para o contexto jurídico
TEMPLATE = """
Você é um advogado especialista em direito sanitário.
Irei te apresentar o texto de uma decisão judicial.

Classifique a decisão judicial de acordo com o seguinte formato:

INICIO DAS INSTRUÇÕES DE FORMATAÇÃO
{format}
FIM DAS INSTRUÇÕES DE FORMATAÇÃO

Utilize as informações do texto para preencher os campos.

INICIO DA SENTENÇA
{sentenca}
FIM DA SENTENÇA
"""

# ============================================================================
# 3. CARREGAR OU CRIAR DADOS
# ============================================================================

print("=" * 80)
print("EXEMPLO AVANÇADO: ANÁLISE DE DECISÕES JUDICIAIS")
print("=" * 80)

# Tentar carregar arquivo real se existir
arquivo_entrada = 'clusters_saude_cluster_amostras_rodada5_5porCluster.xlsx'

if os.path.exists(arquivo_entrada):
    print(f"\n✓ Carregando dados reais de: {arquivo_entrada}")
    df = pd.read_excel(arquivo_entrada)
    print(f"  Total de decisões: {len(df)}")
    usar_dados_reais = True
else:
    print(f"\n⚠ Arquivo {arquivo_entrada} não encontrado.")
    print("  Usando dados sintéticos para demonstração...")

    # Criar dados sintéticos para demonstração
    dados_sinteticos = {
        'id': [1, 2, 3],
        'texto': [
            """DECISÃO: Trata-se de ação ordinária na qual o autor, diagnosticado com
            Doença de Gaucher (doença rara), pleiteia o fornecimento de tratamento
            específico consistente em terapia de reposição enzimática. O autor é do
            sexo masculino e está representado por sua genitora. Requereu justiça
            gratuita, o que foi deferido. Pleiteia ainda danos morais no valor de
            R$ 50.000,00. DEFIRO o pedido de fornecimento do tratamento, diante da
            comprovação da necessidade médica. Quanto aos danos morais, DEFIRO
            PARCIALMENTE, fixando indenização em R$ 30.000,00.""",

            """SENTENÇA: Cuida-se de pedido de medicamento para tratamento de diabetes.
            O autor não possui doença rara. Trata-se de condição comum. Não houve
            pedido de justiça gratuita. DEFIRO o fornecimento do medicamento pleiteado
            (insulina NPH).""",

            """DECISÃO: A presente ação versa sobre questão administrativa relacionada
            a concurso público. Não há pedido de tratamento de saúde. O autor questiona
            sua nota na prova discursiva. INDEFIRO o pedido por ausência de requisitos."""
        ]
    }

    df = pd.DataFrame(dados_sinteticos)
    usar_dados_reais = False

# ============================================================================
# 4. PROCESSAR COM DATAFRAMEIT
# ============================================================================

print("\n" + "=" * 80)
print("PROCESSANDO DECISÕES JUDICIAIS...")
print("=" * 80)
print("""
Este processo pode levar alguns minutos dependendo do volume de dados.
O DataFrameIt vai:
  1. Ler cada decisão judicial
  2. Extrair informações estruturadas usando LLM
  3. Preencher automaticamente os campos do modelo Pydantic
  4. Salvar progresso incremental
""")

# Processar com configurações robustas para produção
df_final = dataframeit(
    df,
    RespostaFinalWrapper,
    TEMPLATE,
    text_column='texto',
    placeholder='sentenca',      # IMPORTANTE: Placeholder customizado!
    resume=True,                  # Permite continuar se interrompido
    max_retries=3,                # Retry em caso de erro
    model='gemini-2.0-flash-exp'  # Modelo rápido e eficiente
)

print("\n✓ Processamento concluído!")

# ============================================================================
# 5. SALVAR RESULTADOS
# ============================================================================

arquivo_saida = 'clusters_analisados.xlsx'
df_final.to_excel(arquivo_saida, index=False)
print(f"✓ Resultados salvos em: {arquivo_saida}")

# ============================================================================
# 6. ANÁLISE DOS RESULTADOS
# ============================================================================

print("\n" + "=" * 80)
print("ANÁLISE DOS RESULTADOS")
print("=" * 80)

# Verificar status do processamento
status_counts = df_final['_dataframeit_status'].value_counts()
total = len(df_final)

print(f"\n📊 Status do processamento:")
print(f"  ✓ Processadas: {status_counts.get('processed', 0)}/{total}")
print(f"  ✗ Com erro: {status_counts.get('error', 0)}/{total}")

# Analisar apenas linhas processadas com sucesso
df_sucesso = df_final[df_final['_dataframeit_status'] == 'processed']

if len(df_sucesso) > 0:
    print("\n" + "-" * 80)
    print("📈 Estatísticas das decisões analisadas:")

    # Distribuição por escopo
    print("\nDistribuição por escopo:")
    print(df_sucesso['definicao_escopo'].value_counts())

    # Análise apenas de casos dentro do escopo
    casos_escopo = df_sucesso[df_sucesso['definicao_escopo'] == 'dentro_do_escopo']

    if len(casos_escopo) > 0:
        print(f"\n📋 Análise de {len(casos_escopo)} caso(s) dentro do escopo:")

        # Gênero dos pacientes
        print("\nDistribuição por gênero:")
        print(casos_escopo['genero_paciente'].value_counts())

        # Justiça gratuita
        print("\nJustiça gratuita:")
        print(casos_escopo['justica_gratuita'].value_counts())

        # Representação
        print("\nRepresentação por responsável:")
        print(casos_escopo['representacao'].value_counts())

        # Danos morais
        danos_pedidos = casos_escopo['danos_morais_pedido'].apply(
            lambda x: x[0] if isinstance(x, tuple) else None
        ).value_counts()
        print("\nDanos morais pedidos:")
        print(danos_pedidos)

# ============================================================================
# 7. VERIFICAR ERROS (SE HOUVER)
# ============================================================================

linhas_erro = df_final[df_final['_dataframeit_status'] == 'error']

if len(linhas_erro) > 0:
    print("\n" + "=" * 80)
    print("⚠ ERROS ENCONTRADOS")
    print("=" * 80)

    print(f"\n{len(linhas_erro)} linha(s) com erro:")
    for idx, row in linhas_erro.iterrows():
        print(f"\nLinha {idx}:")
        print(f"  Erro: {row['_error_details']}")

# ============================================================================
# 8. EXEMPLO DE RESULTADO DETALHADO
# ============================================================================

if len(df_sucesso) > 0:
    print("\n" + "=" * 80)
    print("📄 EXEMPLO DE RESULTADO DETALHADO")
    print("=" * 80)

    primeiro = df_sucesso.iloc[0]

    print(f"\nDefinição de escopo: {primeiro['definicao_escopo']}")
    print(f"Justificativa: {primeiro['justificativa'][:100]}...")

    if primeiro['condicoes_de_saude']:
        print(f"Condições de saúde: {', '.join(primeiro['condicoes_de_saude'])}")

    if primeiro['genero_paciente']:
        print(f"Gênero do paciente: {primeiro['genero_paciente']}")

    if primeiro['justica_gratuita']:
        print(f"Justiça gratuita: {primeiro['justica_gratuita']}")

# ============================================================================
# 9. DICAS PARA PRODUÇÃO
# ============================================================================

print("\n" + "=" * 80)
print("💡 DICAS PARA USO EM PRODUÇÃO")
print("=" * 80)

print("""
Este exemplo demonstra várias técnicas importantes para produção:

✓ Modelo Pydantic complexo com validação automática
✓ Placeholder customizado específico do domínio ({sentenca})
✓ Campos opcionais e condicionais
✓ Uso de List e Tuple para dados estruturados
✓ Template detalhado com instruções claras
✓ Configuração de retry para resiliência
✓ Salvamento automático de resultados
✓ Análise de erros e estatísticas

Para adaptar este exemplo ao seu caso:
1. Modifique o modelo Pydantic conforme suas necessidades
2. Ajuste o template com instruções específicas do seu domínio
3. Configure max_retries baseado na criticidade
4. Implemente pré-processamento de dados se necessário
5. Adicione validações customizadas nos resultados
""")

print("\n" + "=" * 80)
print("EXEMPLO CONCLUÍDO!")
print("=" * 80)

if not usar_dados_reais:
    print("""
⚠ Este exemplo usou dados sintéticos para demonstração.
  Para processar seus próprios dados, coloque-os em um arquivo Excel
  com uma coluna 'texto' contendo as decisões judiciais.
""")
