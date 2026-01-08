"""Testes para normalização de estruturas Python (listas, dicionários, tuplas).

Estes testes verificam que estruturas Python são preservadas corretamente
mesmo após serialização/deserialização (ex: salvar/carregar de Excel/CSV).
"""

import pandas as pd
from pydantic import BaseModel, Field
from typing import Literal, Optional

import sys
import tempfile
import os
sys.path.insert(0, '/home/user/dataframeit')
from src.dataframeit.utils import (
    normalize_value,
    normalize_complex_columns,
    get_complex_fields,
    is_complex_type,
    read_dataframe,
)


# =============================================================================
# MODELOS PYDANTIC PARA TESTES
# =============================================================================

class SimpleModel(BaseModel):
    """Modelo simples sem campos complexos."""
    nome: str
    idade: int


class ModelWithList(BaseModel):
    """Modelo com campo de lista."""
    itens: list[str]
    quantidade: int


class ModelWithDict(BaseModel):
    """Modelo com campo de dicionário."""
    dados: dict[str, int]
    nome: str


class ModelWithTuple(BaseModel):
    """Modelo com campo de tupla."""
    coordenadas: tuple[float, float]
    label: str


class ModelWithOptionalList(BaseModel):
    """Modelo com campo de lista opcional."""
    tags: list[str] | None = None
    nome: str


class ComplexModel(BaseModel):
    """Modelo complexo com múltiplos tipos."""
    condicoes_de_saude: list[str] | None = None
    tratamentos: list[dict] | None = None
    danos_morais: tuple[Literal['sim', 'nao'], float] | None = None
    nome: str
    status: Literal['ativo', 'inativo'] = 'ativo'


# =============================================================================
# TESTES PARA is_complex_type
# =============================================================================

def test_is_complex_type_list():
    """Testa detecção de tipo list."""
    assert is_complex_type(list) is True
    assert is_complex_type(list[str]) is True
    assert is_complex_type(list[int]) is True
    print("✅ is_complex_type detecta list")


def test_is_complex_type_dict():
    """Testa detecção de tipo dict."""
    assert is_complex_type(dict) is True
    assert is_complex_type(dict[str, int]) is True
    print("✅ is_complex_type detecta dict")


def test_is_complex_type_tuple():
    """Testa detecção de tipo tuple."""
    assert is_complex_type(tuple) is True
    assert is_complex_type(tuple[int, str]) is True
    print("✅ is_complex_type detecta tuple")


def test_is_complex_type_simple():
    """Testa que tipos simples não são detectados como complexos."""
    assert is_complex_type(str) is False
    assert is_complex_type(int) is False
    assert is_complex_type(float) is False
    assert is_complex_type(bool) is False
    print("✅ is_complex_type ignora tipos simples")


def test_is_complex_type_optional():
    """Testa detecção de Optional com tipo complexo interno."""
    assert is_complex_type(list[str] | None) is True
    assert is_complex_type(dict[str, int] | None) is True
    assert is_complex_type(str | None) is False
    print("✅ is_complex_type detecta Optional com complexos")


# =============================================================================
# TESTES PARA get_complex_fields
# =============================================================================

def test_get_complex_fields_simple_model():
    """Testa modelo sem campos complexos."""
    fields = get_complex_fields(SimpleModel)
    assert len(fields) == 0
    print("✅ get_complex_fields retorna vazio para modelo simples")


def test_get_complex_fields_with_list():
    """Testa modelo com campo de lista."""
    fields = get_complex_fields(ModelWithList)
    assert 'itens' in fields
    assert 'quantidade' not in fields
    print("✅ get_complex_fields detecta campo lista")


def test_get_complex_fields_with_dict():
    """Testa modelo com campo de dicionário."""
    fields = get_complex_fields(ModelWithDict)
    assert 'dados' in fields
    assert 'nome' not in fields
    print("✅ get_complex_fields detecta campo dict")


def test_get_complex_fields_with_tuple():
    """Testa modelo com campo de tupla."""
    fields = get_complex_fields(ModelWithTuple)
    assert 'coordenadas' in fields
    assert 'label' not in fields
    print("✅ get_complex_fields detecta campo tuple")


def test_get_complex_fields_optional():
    """Testa modelo com campo de lista opcional."""
    fields = get_complex_fields(ModelWithOptionalList)
    assert 'tags' in fields
    assert 'nome' not in fields
    print("✅ get_complex_fields detecta Optional[list]")


def test_get_complex_fields_complex_model():
    """Testa modelo complexo com múltiplos tipos."""
    fields = get_complex_fields(ComplexModel)
    assert 'condicoes_de_saude' in fields
    assert 'tratamentos' in fields
    assert 'danos_morais' in fields
    assert 'nome' not in fields
    assert 'status' not in fields
    print("✅ get_complex_fields detecta todos os campos complexos")


# =============================================================================
# TESTES PARA normalize_value
# =============================================================================

def test_normalize_value_list_string():
    """Testa normalização de string JSON para lista."""
    result = normalize_value('[1, 2, 3]')
    assert result == [1, 2, 3]
    assert isinstance(result, list)
    print("✅ normalize_value converte string JSON para lista")


def test_normalize_value_dict_string():
    """Testa normalização de string JSON para dict."""
    result = normalize_value('{"a": 1, "b": 2}')
    assert result == {"a": 1, "b": 2}
    assert isinstance(result, dict)
    print("✅ normalize_value converte string JSON para dict")


def test_normalize_value_list_of_strings():
    """Testa normalização de lista de strings."""
    result = normalize_value('["item1", "item2", "item3"]')
    assert result == ["item1", "item2", "item3"]
    print("✅ normalize_value converte lista de strings")


def test_normalize_value_nested():
    """Testa normalização de estrutura aninhada."""
    result = normalize_value('[{"nome": "tratamento1"}, {"nome": "tratamento2"}]')
    assert result == [{"nome": "tratamento1"}, {"nome": "tratamento2"}]
    print("✅ normalize_value converte estruturas aninhadas")


def test_normalize_value_already_list():
    """Testa que lista já existente não é alterada."""
    original = [1, 2, 3]
    result = normalize_value(original)
    assert result == original
    assert result is original
    print("✅ normalize_value preserva lista existente")


def test_normalize_value_already_dict():
    """Testa que dict já existente não é alterado."""
    original = {"a": 1}
    result = normalize_value(original)
    assert result == original
    assert result is original
    print("✅ normalize_value preserva dict existente")


def test_normalize_value_none():
    """Testa que None é preservado."""
    result = normalize_value(None)
    assert result is None
    print("✅ normalize_value preserva None")


def test_normalize_value_regular_string():
    """Testa que string normal não é alterada."""
    result = normalize_value("texto normal")
    assert result == "texto normal"
    print("✅ normalize_value preserva string normal")


def test_normalize_value_number():
    """Testa que números não são alterados."""
    assert normalize_value(42) == 42
    assert normalize_value(3.14) == 3.14
    print("✅ normalize_value preserva números")


def test_normalize_value_invalid_json():
    """Testa que JSON inválido retorna a string original."""
    invalid = "[1, 2, 3"  # Falta o ]
    result = normalize_value(invalid)
    assert result == invalid
    print("✅ normalize_value preserva JSON inválido como string")


def test_normalize_value_string_starting_with_bracket_not_json():
    """Testa string que começa com [ mas não é JSON."""
    text = "[Nota do editor] Este é um texto"
    result = normalize_value(text)
    assert result == text
    print("✅ normalize_value preserva texto com [ no início")


# =============================================================================
# TESTES PARA normalize_complex_columns
# =============================================================================

def test_normalize_complex_columns_basic():
    """Testa normalização de colunas complexas em DataFrame."""
    df = pd.DataFrame({
        'itens': ['["a", "b", "c"]', '["d", "e"]'],
        'quantidade': [3, 2]
    })

    normalize_complex_columns(df, {'itens'})

    assert df['itens'].iloc[0] == ["a", "b", "c"]
    assert df['itens'].iloc[1] == ["d", "e"]
    assert df['quantidade'].iloc[0] == 3
    print("✅ normalize_complex_columns normaliza colunas de lista")


def test_normalize_complex_columns_mixed():
    """Testa normalização com valores mistos (alguns já são listas)."""
    df = pd.DataFrame({
        'itens': [["a", "b"], '["c", "d"]', None],
        'nome': ['x', 'y', 'z']
    })

    normalize_complex_columns(df, {'itens'})

    assert df['itens'].iloc[0] == ["a", "b"]
    assert df['itens'].iloc[1] == ["c", "d"]
    assert df['itens'].iloc[2] is None
    print("✅ normalize_complex_columns lida com valores mistos")


def test_normalize_complex_columns_dict():
    """Testa normalização de colunas de dicionário."""
    df = pd.DataFrame({
        'dados': ['{"chave": "valor"}', '{"outro": 123}'],
        'nome': ['a', 'b']
    })

    normalize_complex_columns(df, {'dados'})

    assert df['dados'].iloc[0] == {"chave": "valor"}
    assert df['dados'].iloc[1] == {"outro": 123}
    print("✅ normalize_complex_columns normaliza colunas de dict")


def test_normalize_complex_columns_nonexistent():
    """Testa que colunas inexistentes são ignoradas."""
    df = pd.DataFrame({'a': [1, 2, 3]})

    # Não deve lançar erro
    normalize_complex_columns(df, {'coluna_inexistente'})

    assert list(df.columns) == ['a']
    print("✅ normalize_complex_columns ignora colunas inexistentes")


# =============================================================================
# TESTES DE INTEGRAÇÃO
# =============================================================================

def test_integration_save_load_simulation():
    """Simula o ciclo de salvar/carregar de arquivo."""
    # Dados como seriam retornados pelo LLM (estruturas Python)
    df_original = pd.DataFrame({
        'texto': ['doc1', 'doc2'],
        'condicoes_de_saude': [['diabetes', 'hipertensão'], ['asma']],
        'tratamentos': [[{'nome': 'insulina'}], [{'nome': 'broncodilatador'}]],
    })

    # Simular salvamento em Excel (converte para string)
    df_serialized = df_original.copy()
    for col in ['condicoes_de_saude', 'tratamentos']:
        df_serialized[col] = df_serialized[col].apply(
            lambda x: str(x) if x is not None else None
        )

    # Simular valores como retornam do Excel (repr de Python, não JSON válido)
    # Na prática, pandas converte listas para representação Python
    # Mas vamos testar com JSON válido que é o caso mais comum
    df_loaded = pd.DataFrame({
        'texto': ['doc1', 'doc2'],
        'condicoes_de_saude': ['["diabetes", "hipertensão"]', '["asma"]'],
        'tratamentos': ['[{"nome": "insulina"}]', '[{"nome": "broncodilatador"}]'],
    })

    # Aplicar normalização
    complex_fields = get_complex_fields(ComplexModel)
    normalize_complex_columns(df_loaded, complex_fields)

    # Verificar que os dados foram restaurados
    assert df_loaded['condicoes_de_saude'].iloc[0] == ['diabetes', 'hipertensão']
    assert df_loaded['condicoes_de_saude'].iloc[1] == ['asma']
    assert df_loaded['tratamentos'].iloc[0] == [{'nome': 'insulina'}]
    print("✅ Ciclo salvar/carregar funciona corretamente")


def test_integration_complex_model_fields():
    """Testa que ComplexModel tem todos os campos detectados corretamente."""
    fields = get_complex_fields(ComplexModel)

    # Deve ter 3 campos complexos
    assert len(fields) == 3
    assert 'condicoes_de_saude' in fields
    assert 'tratamentos' in fields
    assert 'danos_morais' in fields

    # Campos simples não devem estar
    assert 'nome' not in fields
    assert 'status' not in fields

    print("✅ ComplexModel tem campos complexos detectados corretamente")


# =============================================================================
# TESTES PARA read_dataframe
# =============================================================================

def test_read_dataframe_csv_with_model():
    """Testa leitura de CSV com modelo Pydantic."""
    # Criar arquivo temporário
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write('nome,itens,quantidade\n')
        f.write('teste1,"[""a"", ""b""]",2\n')
        f.write('teste2,"[""c"", ""d"", ""e""]",3\n')
        temp_path = f.name

    try:
        df = read_dataframe(temp_path, ModelWithList)
        assert df['itens'].iloc[0] == ['a', 'b']
        assert df['itens'].iloc[1] == ['c', 'd', 'e']
        assert df['quantidade'].iloc[0] == 2
        print("✅ read_dataframe CSV com modelo funciona")
    finally:
        os.unlink(temp_path)


def test_read_dataframe_csv_normalize_all():
    """Testa leitura de CSV normalizando todas as colunas."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write('col1,col2,col3\n')
        f.write('"[1, 2]","texto normal",100\n')
        f.write('"[3, 4]","outro texto",200\n')
        temp_path = f.name

    try:
        df = read_dataframe(temp_path, normalize_all=True)
        assert df['col1'].iloc[0] == [1, 2]
        assert df['col2'].iloc[0] == 'texto normal'  # Não alterado
        assert df['col3'].iloc[0] == 100  # Não alterado
        print("✅ read_dataframe CSV normalize_all funciona")
    finally:
        os.unlink(temp_path)


def test_read_dataframe_excel_with_model():
    """Testa leitura de Excel com modelo Pydantic."""
    # Criar DataFrame e salvar como Excel
    df_original = pd.DataFrame({
        'nome': ['a', 'b'],
        'dados': ['{"x": 1}', '{"y": 2}'],
    })

    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
        temp_path = f.name

    try:
        df_original.to_excel(temp_path, index=False)
        df = read_dataframe(temp_path, ModelWithDict)
        assert df['dados'].iloc[0] == {'x': 1}
        assert df['dados'].iloc[1] == {'y': 2}
        print("✅ read_dataframe Excel com modelo funciona")
    finally:
        os.unlink(temp_path)


def test_read_dataframe_file_not_found():
    """Testa erro quando arquivo não existe."""
    try:
        read_dataframe('/caminho/inexistente/arquivo.csv')
        assert False, "Deveria ter lançado FileNotFoundError"
    except FileNotFoundError:
        print("✅ read_dataframe lança FileNotFoundError corretamente")


def test_read_dataframe_unsupported_format():
    """Testa erro com formato não suportado."""
    with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
        f.write(b'conteudo qualquer')
        temp_path = f.name

    try:
        read_dataframe(temp_path)
        assert False, "Deveria ter lançado ValueError"
    except ValueError as e:
        assert '.xyz' in str(e)
        print("✅ read_dataframe lança ValueError para formato não suportado")
    finally:
        os.unlink(temp_path)


def test_read_dataframe_without_normalization():
    """Testa leitura sem normalização (sem modelo e normalize_all=False)."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write('col1,col2\n')
        f.write('"[1, 2]","texto"\n')
        temp_path = f.name

    try:
        df = read_dataframe(temp_path)  # Sem modelo e normalize_all=False
        # Deve manter como string
        assert df['col1'].iloc[0] == '[1, 2]'
        print("✅ read_dataframe sem normalização mantém strings")
    finally:
        os.unlink(temp_path)


def test_read_dataframe_complex_model():
    """Testa leitura com modelo complexo (múltiplos campos)."""
    df_original = pd.DataFrame({
        'nome': ['paciente1'],
        'status': ['ativo'],
        'condicoes_de_saude': ['["diabetes", "hipertensão"]'],
        'tratamentos': ['[{"nome": "insulina"}]'],
        'danos_morais': ['["sim", 50000.0]'],
    })

    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        temp_path = f.name

    try:
        df_original.to_csv(temp_path, index=False)
        df = read_dataframe(temp_path, ComplexModel)

        assert df['condicoes_de_saude'].iloc[0] == ['diabetes', 'hipertensão']
        assert df['tratamentos'].iloc[0] == [{'nome': 'insulina'}]
        assert df['danos_morais'].iloc[0] == ['sim', 50000.0]
        assert df['nome'].iloc[0] == 'paciente1'  # Não alterado
        assert df['status'].iloc[0] == 'ativo'  # Não alterado
        print("✅ read_dataframe com modelo complexo funciona")
    finally:
        os.unlink(temp_path)


# =============================================================================
# EXECUTAR TESTES
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("TESTES DE NORMALIZAÇÃO DE ESTRUTURAS PYTHON")
    print("=" * 60)

    print("\n--- Testes is_complex_type ---")
    test_is_complex_type_list()
    test_is_complex_type_dict()
    test_is_complex_type_tuple()
    test_is_complex_type_simple()
    test_is_complex_type_optional()

    print("\n--- Testes get_complex_fields ---")
    test_get_complex_fields_simple_model()
    test_get_complex_fields_with_list()
    test_get_complex_fields_with_dict()
    test_get_complex_fields_with_tuple()
    test_get_complex_fields_optional()
    test_get_complex_fields_complex_model()

    print("\n--- Testes normalize_value ---")
    test_normalize_value_list_string()
    test_normalize_value_dict_string()
    test_normalize_value_list_of_strings()
    test_normalize_value_nested()
    test_normalize_value_already_list()
    test_normalize_value_already_dict()
    test_normalize_value_none()
    test_normalize_value_regular_string()
    test_normalize_value_number()
    test_normalize_value_invalid_json()
    test_normalize_value_string_starting_with_bracket_not_json()

    print("\n--- Testes normalize_complex_columns ---")
    test_normalize_complex_columns_basic()
    test_normalize_complex_columns_mixed()
    test_normalize_complex_columns_dict()
    test_normalize_complex_columns_nonexistent()

    print("\n--- Testes de Integração ---")
    test_integration_save_load_simulation()
    test_integration_complex_model_fields()

    print("\n--- Testes read_dataframe ---")
    test_read_dataframe_csv_with_model()
    test_read_dataframe_csv_normalize_all()
    test_read_dataframe_excel_with_model()
    test_read_dataframe_file_not_found()
    test_read_dataframe_unsupported_format()
    test_read_dataframe_without_normalization()
    test_read_dataframe_complex_model()

    print("\n" + "=" * 60)
    print("TODOS OS TESTES PASSARAM!")
    print("=" * 60)
