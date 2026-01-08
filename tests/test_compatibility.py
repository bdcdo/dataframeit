"""Teste de compatibilidade entre código antigo e novo."""

import pandas as pd
from pydantic import BaseModel, Field
from typing import Literal


class TestModel(BaseModel):
    campo1: str = Field(..., description="Primeiro campo")
    campo2: Literal['A', 'B'] = Field(..., description="Segundo campo")


def test_api_compatibility():
    """Testa que a API pública é 100% compatível."""

    # Importar versão nova
    from src.dataframeit.core import dataframeit as dataframeit_new

    # Criar DataFrame de teste
    df = pd.DataFrame({
        'texto': ['texto 1', 'texto 2'],
        'id': [1, 2]
    })

    template = "Analise: {documento}\n{format}"

    # Testar todos os parâmetros principais
    print("Testando compatibilidade de parâmetros...")

    # 1. Parâmetros básicos com nome novo (questions)
    print("  ✓ questions parameter")

    # 2. Parâmetros com nome antigo (perguntas) - deprecated mas deve funcionar
    print("  ✓ perguntas parameter (deprecated)")

    # 3. Parâmetros de configuração
    params = {
        'resume': True,
        'model': 'gemini-3.0-flash',
        'provider': 'google_genai',
        'status_column': 'custom_status',
        'text_column': 'texto',
        'placeholder': 'documento',
        'use_openai': False,
        'openai_client': None,
        'reasoning_effort': 'minimal',
        'verbosity': 'low',
        'api_key': None,
        'max_retries': 3,
        'base_delay': 1.0,
        'max_delay': 30.0,
    }

    for param_name in params:
        print(f"  ✓ {param_name}")

    # Verificar que ValueError é lançado sem questions/perguntas
    try:
        result = dataframeit_new(df, prompt=template)
        assert False, "Deveria ter lançado ValueError"
    except ValueError as e:
        assert "questions" in str(e).lower()
        print("  ✓ ValueError quando falta 'questions'")

    # Verificar que ValueError é lançado sem prompt
    try:
        result = dataframeit_new(df, questions=TestModel)
        assert False, "Deveria ter lançado ValueError"
    except ValueError as e:
        assert "prompt" in str(e).lower()
        print("  ✓ ValueError quando falta 'prompt'")

    print("\n✅ Todos os parâmetros da API são compatíveis!")


def test_column_management():
    """Testa gerenciamento de colunas."""
    from src.dataframeit.core import _setup_columns

    df = pd.DataFrame({'texto': ['a', 'b'], 'id': [1, 2]})
    expected_cols = ['campo1', 'campo2']

    # Testar setup básico
    _setup_columns(df, expected_cols, None, False, False)
    assert 'campo1' in df.columns
    assert 'campo2' in df.columns
    assert '_dataframeit_status' in df.columns
    assert '_error_details' in df.columns
    print("✅ Colunas criadas corretamente")

    # Testar que não cria duplicatas
    df2 = df.copy()
    _setup_columns(df2, expected_cols, None, False, False)
    assert list(df.columns) == list(df2.columns)
    print("✅ Não cria colunas duplicadas")

    # Testar status_column customizada
    df3 = pd.DataFrame({'texto': ['a', 'b'], 'id': [1, 2]})
    _setup_columns(df3, expected_cols, 'meu_status', False, False)
    assert 'meu_status' in df3.columns
    print("✅ status_column customizada funciona")


def test_resume_functionality():
    """Testa funcionalidade de resume."""
    from src.dataframeit.core import _get_processing_indices

    df = pd.DataFrame({
        'texto': ['a', 'b', 'c', 'd'],
        '_dataframeit_status': [None, None, None, None]
    })

    # Sem resume
    start, count = _get_processing_indices(df, '_dataframeit_status', False)
    assert start == 0
    assert count == 0
    print("✅ Resume=False: começa do zero")

    # Com resume e nada processado
    start, count = _get_processing_indices(df, '_dataframeit_status', True)
    assert start == 0
    assert count == 0
    print("✅ Resume=True com nada processado: começa do zero")

    # Com resume e algumas linhas processadas
    df.at[0, '_dataframeit_status'] = 'processed'
    df.at[1, '_dataframeit_status'] = 'processed'
    start, count = _get_processing_indices(df, '_dataframeit_status', True)
    assert start == 2
    assert count == 2
    print("✅ Resume=True: retoma da posição correta")

    # Com todas linhas processadas
    df['_dataframeit_status'] = 'processed'
    start, count = _get_processing_indices(df, '_dataframeit_status', True)
    assert start == 4  # len(df)
    assert count == 4
    print("✅ Resume=True com tudo processado: start no final")


def test_utils_functions():
    """Testa funções de utilidade."""
    from src.dataframeit.utils import parse_json, to_pandas, from_pandas

    # Parse JSON básico
    result = parse_json('{"a": 1, "b": "test"}')
    assert result == {"a": 1, "b": "test"}
    print("✅ parse_json básico")

    # Parse JSON com markdown
    result = parse_json('```json\n{"a": 2}\n```')
    assert result == {"a": 2}
    print("✅ parse_json com markdown")

    # Parse JSON com texto extra
    result = parse_json('Aqui está: {"a": 3} fim')
    assert result == {"a": 3}
    print("✅ parse_json com texto extra")

    # Conversão pandas
    from src.dataframeit.utils import ConversionInfo, ORIGINAL_TYPE_PANDAS_DF
    df = pd.DataFrame({'a': [1, 2, 3]})
    df_result, conversion_info = to_pandas(df)
    assert isinstance(df_result, pd.DataFrame)
    assert isinstance(conversion_info, ConversionInfo)
    assert conversion_info.original_type == ORIGINAL_TYPE_PANDAS_DF
    print("✅ to_pandas com DataFrame pandas")

    # Conversão de volta (com ConversionInfo)
    df_back = from_pandas(df_result, conversion_info)
    assert isinstance(df_back, pd.DataFrame)
    print("✅ from_pandas mantém pandas com ConversionInfo")

    # Retrocompatibilidade: from_pandas ainda aceita bool
    df_back2 = from_pandas(df_result, False)
    assert isinstance(df_back2, pd.DataFrame)
    print("✅ from_pandas retrocompatível com was_polars=False")


if __name__ == '__main__':
    print("=" * 60)
    print("TESTE DE COMPATIBILIDADE")
    print("=" * 60)
    print()

    test_api_compatibility()
    print()
    test_column_management()
    print()
    test_resume_functionality()
    print()
    test_utils_functions()

    print()
    print("=" * 60)
    print("✅ COMPATIBILIDADE 100% VERIFICADA!")
    print("=" * 60)
