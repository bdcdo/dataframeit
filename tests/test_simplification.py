"""Teste funcional para validar a simplificação do código."""

import pandas as pd
from pydantic import BaseModel, Field
from typing import Literal

# Importar a versão nova
from src.dataframeit.core import dataframeit


class TestModel(BaseModel):
    campo1: str = Field(..., description="Primeiro campo")
    campo2: Literal['A', 'B'] = Field(..., description="Segundo campo")


def test_basic_functionality():
    """Testa funcionalidade básica sem fazer chamadas reais ao LLM."""

    # Criar DataFrame de teste
    df = pd.DataFrame({
        'texto': ['texto 1', 'texto 2', 'texto 3'],
        'coluna_existente': [1, 2, 3]
    })

    # Template simples
    template = """
    Analise o texto: {documento}
    {format}
    """

    print("DataFrame original:")
    print(df)
    print("\nModelo Pydantic:", TestModel.model_fields.keys())
    print("Template:", template[:50], "...")

    # Verificar que as colunas são configuradas corretamente
    from src.dataframeit.core import _setup_columns
    df_test = df.copy()
    expected_cols = list(TestModel.model_fields.keys())
    _setup_columns(df_test, expected_cols, None, False, False)

    print("\nColunas após setup:", list(df_test.columns))
    assert 'campo1' in df_test.columns
    assert 'campo2' in df_test.columns
    assert '_dataframeit_status' in df_test.columns
    assert 'error_details' in df_test.columns

    # Verificar índices de processamento
    from src.dataframeit.core import _get_processing_indices
    start, count = _get_processing_indices(df_test, '_dataframeit_status', False)
    print(f"Processamento: start={start}, processed={count}")
    assert start == 0
    assert count == 0

    # Testar com resume
    df_test.at[0, '_dataframeit_status'] = 'processed'
    start, count = _get_processing_indices(df_test, '_dataframeit_status', True)
    print(f"Com resume: start={start}, processed={count}")
    assert start == 1
    assert count == 1

    print("\n✅ Funcionalidade básica OK!")


def test_llm_config():
    """Testa criação de config do LLM."""
    from src.dataframeit.llm import LLMConfig

    config = LLMConfig(
        model='gemini-2.5-flash',
        provider='google_genai',
        use_openai=False,
        api_key=None,
        openai_client=None,
        reasoning_effort='minimal',
        verbosity='low',
        max_retries=3,
        base_delay=1.0,
        max_delay=30.0,
        placeholder='documento',
        rate_limit_delay=0.0
    )

    print("\nConfig criado:")
    print(f"  Model: {config.model}")
    print(f"  Provider: {config.provider}")
    print(f"  Use OpenAI: {config.use_openai}")
    print(f"  Placeholder: {config.placeholder}")

    assert config.model == 'gemini-2.5-flash'
    assert config.provider == 'google_genai'
    print("✅ Config OK!")


def test_utils():
    """Testa funções de utilidade."""
    from src.dataframeit.utils import to_pandas, from_pandas, parse_json

    # Testar conversão pandas
    df_pd = pd.DataFrame({'a': [1, 2, 3]})
    result, was_polars = to_pandas(df_pd)
    assert isinstance(result, pd.DataFrame)
    assert was_polars is False
    print("✅ Conversão pandas OK!")

    # Testar parse_json
    json_str = '{"campo1": "valor", "campo2": "A"}'
    parsed = parse_json(json_str)
    assert parsed['campo1'] == 'valor'
    assert parsed['campo2'] == 'A'
    print("✅ Parse JSON OK!")

    # Testar parse de JSON com markdown
    markdown_json = '''```json
{"campo1": "teste", "campo2": "B"}
```'''
    parsed2 = parse_json(markdown_json)
    assert parsed2['campo1'] == 'teste'
    print("✅ Parse JSON markdown OK!")


def test_prompt_building():
    """Testa construção de prompts."""
    from src.dataframeit.llm import build_prompt

    template = "Analise: {documento}"
    text = "Este é um texto de teste"

    prompt = build_prompt(template, text, 'documento')
    print("\nPrompt construído:")
    print(prompt)
    assert 'Este é um texto de teste' in prompt
    assert '{documento}' not in prompt  # Placeholder deve ter sido substituído
    print("✅ Construção de prompt OK!")


if __name__ == '__main__':
    print("=" * 60)
    print("TESTE DE VALIDAÇÃO DA SIMPLIFICAÇÃO")
    print("=" * 60)

    test_basic_functionality()
    print()
    test_llm_config()
    print()
    test_utils()
    print()
    test_prompt_building()

    print("\n" + "=" * 60)
    print("✅ TODOS OS TESTES PASSARAM!")
    print("=" * 60)
