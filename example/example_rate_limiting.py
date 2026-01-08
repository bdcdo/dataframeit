"""
Exemplo: Como usar Rate Limiting no DataFrameIt

Este exemplo demonstra como configurar o rate_limit_delay para evitar
atingir limites de requisições das APIs de LLM.
"""

from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
from dataframeit import dataframeit


# ============================================================================
# 1. Definir modelo Pydantic
# ============================================================================

class AnaliseTexto(BaseModel):
    """Modelo para estruturar análise de textos."""
    sentimento: Literal['positivo', 'negativo', 'neutro'] = Field(
        description="Sentimento geral do texto"
    )
    tema_principal: str = Field(
        description="Tema ou assunto principal do texto"
    )
    resumo: str = Field(
        description="Resumo breve do texto em até 50 palavras"
    )


# ============================================================================
# 2. Definir template de prompt
# ============================================================================

TEMPLATE = """
Você é um analista de texto especializado.

Analise o texto a seguir e extraia:
1. O sentimento geral (positivo, negativo ou neutro)
2. O tema principal abordado
3. Um resumo breve (máximo 50 palavras)

Texto a analisar:
{documento}
"""


# ============================================================================
# 3. Criar dados de exemplo
# ============================================================================

def criar_dataset_exemplo():
    """Cria um dataset de exemplo com vários textos."""
    textos = [
        "A nova política de saúde pública trouxe melhorias significativas no atendimento.",
        "O sistema de transporte continua enfrentando problemas graves de superlotação.",
        "Investimentos em educação foram anunciados pelo governo federal.",
        "A economia apresentou crescimento moderado no último trimestre.",
        "Manifestantes protestam contra aumento de impostos nas principais cidades.",
        "Novo programa social visa reduzir desigualdade no acesso à tecnologia.",
        "Empresas de tecnologia anunciam cortes de pessoal devido à crise.",
        "Cientistas descobrem nova espécie de planta na Amazônia.",
        "Mudanças climáticas afetam produção agrícola em várias regiões.",
        "Campanha de vacinação atinge meta estabelecida pelo ministério.",
    ]

    return pd.DataFrame({'texto': textos})


# ============================================================================
# 4. Exemplos de uso com diferentes configurações de rate limiting
# ============================================================================

def exemplo_sem_rate_limit():
    """Exemplo SEM rate limiting (pode atingir limite da API)."""
    print("=" * 70)
    print("EXEMPLO 1: SEM rate limiting")
    print("=" * 70)
    print("⚠️  Atenção: Em datasets grandes, pode atingir rate limit da API\n")

    df = criar_dataset_exemplo()

    resultado = dataframeit(
        df,
        AnaliseTexto,
        TEMPLATE,
        rate_limit_delay=0.0  # Padrão: sem delay
    )

    print("\n✓ Processamento concluído!")
    print(f"Total de linhas processadas: {len(resultado)}")
    return resultado


def exemplo_google_gemini():
    """Exemplo COM rate limiting para Google Gemini (60 req/min)."""
    print("=" * 70)
    print("EXEMPLO 2: Google Gemini com rate limiting")
    print("=" * 70)
    print("📊 API Limit: 60 requisições/minuto (free tier)")
    print("⏱️  Rate Limit Delay: 1.0 segundo entre requisições\n")

    df = criar_dataset_exemplo()

    resultado = dataframeit(
        df,
        AnaliseTexto,
        TEMPLATE,
        model='gemini-2.0-flash-exp',
        provider='google_genai',
        rate_limit_delay=1.0  # 60 req/min = 1 req/segundo
    )

    print("\n✓ Processamento concluído com rate limiting!")
    print(f"Total de linhas processadas: {len(resultado)}")
    return resultado


def exemplo_openai_gpt():
    """Exemplo COM rate limiting para OpenAI GPT-4 (500 req/min)."""
    print("=" * 70)
    print("EXEMPLO 3: OpenAI GPT-4 com rate limiting otimizado")
    print("=" * 70)
    print("📊 API Limit: 500 requisições/minuto (tier 1)")
    print("⏱️  Rate Limit Delay: 0.15 segundos (~400 req/min com margem)\n")

    df = criar_dataset_exemplo()

    resultado = dataframeit(
        df,
        AnaliseTexto,
        TEMPLATE,
        provider='openai',
        model='gpt-4o-mini',
        rate_limit_delay=0.15  # ~400 req/min (margem de segurança)
    )

    print("\n✓ Processamento concluído com rate limiting otimizado!")
    print(f"Total de linhas processadas: {len(resultado)}")
    return resultado


def exemplo_anthropic_claude():
    """Exemplo COM rate limiting para Anthropic Claude (50 req/min)."""
    print("=" * 70)
    print("EXEMPLO 4: Anthropic Claude com rate limiting")
    print("=" * 70)
    print("📊 API Limit: 50 requisições/minuto (free tier)")
    print("⏱️  Rate Limit Delay: 1.2 segundos entre requisições\n")

    df = criar_dataset_exemplo()

    resultado = dataframeit(
        df,
        AnaliseTexto,
        TEMPLATE,
        provider='anthropic',
        model='claude-3-5-sonnet-20241022',
        rate_limit_delay=1.2  # 50 req/min
    )

    print("\n✓ Processamento concluído com rate limiting!")
    print(f"Total de linhas processadas: {len(resultado)}")
    return resultado


def exemplo_dupla_protecao():
    """Exemplo com rate limiting + retry (dupla proteção)."""
    print("=" * 70)
    print("EXEMPLO 5: Rate Limiting + Retry (Dupla Proteção)")
    print("=" * 70)
    print("🛡️  Rate Limiting: Previne erros de rate limit")
    print("🔄 Retry: Trata erros inesperados com backoff exponencial\n")

    df = criar_dataset_exemplo()

    resultado = dataframeit(
        df,
        AnaliseTexto,
        TEMPLATE,
        # Rate limiting proativo
        rate_limit_delay=1.0,

        # Retry reativo
        max_retries=3,
        base_delay=2.0,
        max_delay=30.0
    )

    print("\n✓ Processamento concluído com dupla proteção!")
    print(f"Total de linhas processadas: {len(resultado)}")

    # Verificar se houve erros
    erros = resultado[resultado['_dataframeit_status'] == 'error']
    sucesso = resultado[resultado['_dataframeit_status'] == 'processed']

    print(f"✓ Sucessos: {len(sucesso)}")
    print(f"✗ Erros: {len(erros)}")

    return resultado


# ============================================================================
# 5. Calcular delay ideal para sua API
# ============================================================================

def calcular_delay_ideal(req_por_minuto: int, margem_seguranca: float = 0.9):
    """
    Calcula o delay ideal baseado no limite de requisições da API.

    Args:
        req_por_minuto: Limite de requisições por minuto da API
        margem_seguranca: Fator de segurança (0.9 = usar 90% do limite)

    Returns:
        Delay em segundos entre requisições
    """
    req_ajustado = req_por_minuto * margem_seguranca
    delay = 60 / req_ajustado

    print(f"\n📊 Cálculo de Rate Limit:")
    print(f"   Limite da API: {req_por_minuto} req/min")
    print(f"   Com margem de segurança ({int(margem_seguranca*100)}%): {req_ajustado:.0f} req/min")
    print(f"   ⏱️  Delay recomendado: {delay:.2f} segundos\n")

    return delay


# ============================================================================
# Executar exemplos
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("DEMONSTRAÇÃO: Rate Limiting no DataFrameIt")
    print("=" * 70 + "\n")

    # Escolha qual exemplo executar:

    # Exemplo 1: Sem rate limiting (não recomendado para datasets grandes)
    # resultado = exemplo_sem_rate_limit()

    # Exemplo 2: Google Gemini (60 req/min)
    resultado = exemplo_google_gemini()

    # Exemplo 3: OpenAI GPT-4 (500 req/min)
    # resultado = exemplo_openai_gpt()

    # Exemplo 4: Anthropic Claude (50 req/min)
    # resultado = exemplo_anthropic_claude()

    # Exemplo 5: Dupla proteção (rate limiting + retry)
    # resultado = exemplo_dupla_protecao()

    # Utilitário: Calcular delay ideal para sua API
    # delay = calcular_delay_ideal(req_por_minuto=60, margem_seguranca=0.9)

    # Visualizar resultados
    print("\n" + "=" * 70)
    print("RESULTADOS")
    print("=" * 70)
    print(resultado[['texto', 'sentimento', 'tema_principal']].head())

    # Salvar resultados
    # resultado.to_excel('resultado_rate_limiting.xlsx', index=False)
    # print("\n✓ Resultados salvos em 'resultado_rate_limiting.xlsx'")
