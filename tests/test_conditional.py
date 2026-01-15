"""Testes para funcionalidade de execução condicional de campos."""

import pytest
from pydantic import BaseModel, Field

from dataframeit.conditional import (
    get_nested_value,
    evaluate_condition,
    check_dependencies_exist,
    detect_circular_dependencies,
    topological_sort,
    get_field_execution_order,
    should_skip_field,
)


class TestGetNestedValue:
    """Testes para get_nested_value."""

    def test_simple_field(self):
        """Testa acesso a campo simples."""
        data = {'nome': 'João', 'idade': 30}
        assert get_nested_value(data, 'nome') == 'João'
        assert get_nested_value(data, 'idade') == 30

    def test_nested_field(self):
        """Testa acesso a campo aninhado."""
        data = {
            'nome': 'João',
            'endereco': {
                'cidade': 'São Paulo',
                'rua': 'Av. Paulista',
                'numero': 1000,
            }
        }
        assert get_nested_value(data, 'endereco.cidade') == 'São Paulo'
        assert get_nested_value(data, 'endereco.rua') == 'Av. Paulista'
        assert get_nested_value(data, 'endereco.numero') == 1000

    def test_deeply_nested_field(self):
        """Testa acesso a campo profundamente aninhado."""
        data = {
            'empresa': {
                'endereco': {
                    'localizacao': {
                        'cidade': 'Rio de Janeiro'
                    }
                }
            }
        }
        assert get_nested_value(data, 'empresa.endereco.localizacao.cidade') == 'Rio de Janeiro'

    def test_nonexistent_field(self):
        """Testa acesso a campo inexistente."""
        data = {'nome': 'João'}
        assert get_nested_value(data, 'idade') is None
        assert get_nested_value(data, 'endereco.cidade') is None

    def test_empty_path(self):
        """Testa caminho vazio."""
        data = {'nome': 'João'}
        assert get_nested_value(data, '') is None

    def test_none_value(self):
        """Testa campo com valor None."""
        data = {'nome': None}
        assert get_nested_value(data, 'nome') is None


class TestEvaluateCondition:
    """Testes para evaluate_condition."""

    def test_equals_true(self):
        """Testa condição equals verdadeira."""
        data = {'tipo': 'pf', 'status': 'ativo'}
        condition = {'field': 'tipo', 'equals': 'pf'}
        assert evaluate_condition(condition, data, 'cpf') is True

    def test_equals_false(self):
        """Testa condição equals falsa."""
        data = {'tipo': 'pf', 'status': 'ativo'}
        condition = {'field': 'tipo', 'equals': 'pj'}
        assert evaluate_condition(condition, data, 'cnpj') is False

    def test_not_equals(self):
        """Testa condição not_equals."""
        data = {'tipo': 'pf'}
        assert evaluate_condition({'field': 'tipo', 'not_equals': 'pj'}, data, 'x') is True
        assert evaluate_condition({'field': 'tipo', 'not_equals': 'pf'}, data, 'x') is False

    def test_in_list(self):
        """Testa condição in (valor está na lista)."""
        data = {'status': 'ativo'}
        condition = {'field': 'status', 'in': ['ativo', 'pendente']}
        assert evaluate_condition(condition, data, 'x') is True

        data = {'status': 'inativo'}
        assert evaluate_condition(condition, data, 'x') is False

    def test_not_in_list(self):
        """Testa condição not_in (valor não está na lista)."""
        data = {'status': 'inativo'}
        condition = {'field': 'status', 'not_in': ['ativo', 'pendente']}
        assert evaluate_condition(condition, data, 'x') is True

        data = {'status': 'ativo'}
        assert evaluate_condition(condition, data, 'x') is False

    def test_exists_true(self):
        """Testa condição exists verdadeira."""
        data = {'nome': 'João'}
        condition = {'field': 'nome', 'exists': True}
        assert evaluate_condition(condition, data, 'x') is True

    def test_exists_false(self):
        """Testa condição exists falsa."""
        data = {'nome': None}
        condition = {'field': 'nome', 'exists': True}
        assert evaluate_condition(condition, data, 'x') is False

        data = {}
        condition = {'field': 'nome', 'exists': False}
        assert evaluate_condition(condition, data, 'x') is True

    def test_nested_field_condition(self):
        """Testa condição com campo aninhado."""
        data = {
            'endereco': {
                'cidade': 'São Paulo',
                'estado': 'SP'
            }
        }
        condition = {'field': 'endereco.cidade', 'equals': 'São Paulo'}
        assert evaluate_condition(condition, data, 'x') is True

        condition = {'field': 'endereco.estado', 'in': ['SP', 'RJ']}
        assert evaluate_condition(condition, data, 'x') is True

    def test_callable_condition(self):
        """Testa condição callable."""
        data = {'idade': 25}
        condition = lambda d: d.get('idade', 0) >= 18
        assert evaluate_condition(condition, data, 'pode_votar') is True

        data = {'idade': 15}
        assert evaluate_condition(condition, data, 'pode_votar') is False

    def test_none_condition(self):
        """Testa condição None (sempre True)."""
        data = {'nome': 'João'}
        assert evaluate_condition(None, data, 'x') is True

    def test_invalid_condition_type(self):
        """Testa tipo de condição inválido."""
        data = {'nome': 'João'}
        assert evaluate_condition("invalid", data, 'x') is False
        assert evaluate_condition(123, data, 'x') is False

    def test_condition_without_field(self):
        """Testa condição dict sem campo 'field'."""
        data = {'nome': 'João'}
        condition = {'equals': 'João'}
        assert evaluate_condition(condition, data, 'x') is False

    def test_condition_without_operator(self):
        """Testa condição dict sem operador válido."""
        data = {'nome': 'João'}
        condition = {'field': 'nome'}
        assert evaluate_condition(condition, data, 'x') is False


class TestDependencies:
    """Testes para funções de dependências."""

    def test_check_dependencies_exist_all_valid(self):
        """Testa verificação de dependências válidas."""
        all_fields = {'a', 'b', 'c'}
        missing = check_dependencies_exist('d', ['a', 'b'], all_fields)
        assert missing == []

    def test_check_dependencies_exist_some_invalid(self):
        """Testa verificação com dependências inválidas."""
        all_fields = {'a', 'b', 'c'}
        missing = check_dependencies_exist('d', ['a', 'x', 'y'], all_fields)
        assert set(missing) == {'x', 'y'}

    def test_check_dependencies_nested_field(self):
        """Testa verificação de dependência com campo aninhado."""
        all_fields = {'endereco', 'nome'}
        missing = check_dependencies_exist('x', ['endereco.cidade'], all_fields)
        assert missing == []

    def test_detect_circular_dependencies_no_cycle(self):
        """Testa detecção sem ciclos."""
        deps = {'a': [], 'b': ['a'], 'c': ['a', 'b']}
        assert detect_circular_dependencies(deps) is None

    def test_detect_circular_dependencies_simple_cycle(self):
        """Testa detecção de ciclo simples."""
        deps = {'a': ['b'], 'b': ['a']}
        cycle = detect_circular_dependencies(deps)
        assert cycle is not None
        assert 'a' in cycle and 'b' in cycle

    def test_detect_circular_dependencies_complex_cycle(self):
        """Testa detecção de ciclo complexo."""
        deps = {'a': ['b'], 'b': ['c'], 'c': ['a']}
        cycle = detect_circular_dependencies(deps)
        assert cycle is not None
        assert set(cycle) >= {'a', 'b', 'c'}

    def test_topological_sort_simple(self):
        """Testa ordenação topológica simples."""
        deps = {'a': [], 'b': ['a'], 'c': ['b']}
        result = topological_sort(deps)
        assert result.index('a') < result.index('b')
        assert result.index('b') < result.index('c')

    def test_topological_sort_complex(self):
        """Testa ordenação topológica complexa."""
        deps = {
            'a': [],
            'b': [],
            'c': ['a'],
            'd': ['b'],
            'e': ['c', 'd'],
        }
        result = topological_sort(deps)
        assert result.index('a') < result.index('c')
        assert result.index('b') < result.index('d')
        assert result.index('c') < result.index('e')
        assert result.index('d') < result.index('e')

    def test_topological_sort_with_cycle_raises(self):
        """Testa que ordenação topológica levanta erro com ciclo."""
        deps = {'a': ['b'], 'b': ['c'], 'c': ['a']}
        with pytest.raises(ValueError, match="Dependências circulares"):
            topological_sort(deps)


class TestGetFieldExecutionOrder:
    """Testes para get_field_execution_order."""

    def test_simple_order(self):
        """Testa ordem de execução simples."""
        class SimpleModel(BaseModel):
            a: str
            b: str = Field(json_schema_extra={'depends_on': ['a']})

        field_configs = {
            'a': {},
            'b': {'depends_on': ['a']},
        }

        order, deps = get_field_execution_order(SimpleModel, field_configs)
        assert order.index('a') < order.index('b')

    def test_complex_order(self):
        """Testa ordem de execução complexa."""
        class ComplexModel(BaseModel):
            tipo: str
            cpf: str = Field(json_schema_extra={'depends_on': ['tipo']})
            cnpj: str = Field(json_schema_extra={'depends_on': ['tipo']})
            validacao: str = Field(json_schema_extra={'depends_on': ['cpf', 'cnpj']})

        field_configs = {
            'tipo': {},
            'cpf': {'depends_on': ['tipo']},
            'cnpj': {'depends_on': ['tipo']},
            'validacao': {'depends_on': ['cpf', 'cnpj']},
        }

        order, deps = get_field_execution_order(ComplexModel, field_configs)
        assert order.index('tipo') < order.index('cpf')
        assert order.index('tipo') < order.index('cnpj')
        assert order.index('cpf') < order.index('validacao')
        assert order.index('cnpj') < order.index('validacao')

    def test_missing_dependency_raises(self):
        """Testa que dependência inexistente levanta erro."""
        class ModelWithInvalidDep(BaseModel):
            a: str = Field(json_schema_extra={'depends_on': ['nonexistent']})

        field_configs = {
            'a': {'depends_on': ['nonexistent']},
        }

        with pytest.raises(ValueError, match="depende de campos inexistentes"):
            get_field_execution_order(ModelWithInvalidDep, field_configs)

    def test_circular_dependency_raises(self):
        """Testa que dependência circular levanta erro."""
        class ModelWithCircular(BaseModel):
            a: str = Field(json_schema_extra={'depends_on': ['b']})
            b: str = Field(json_schema_extra={'depends_on': ['a']})

        field_configs = {
            'a': {'depends_on': ['b']},
            'b': {'depends_on': ['a']},
        }

        with pytest.raises(ValueError, match="Dependências circulares"):
            get_field_execution_order(ModelWithCircular, field_configs)


class TestShouldSkipField:
    """Testes para should_skip_field."""

    def test_no_condition(self):
        """Testa campo sem condição (não deve pular)."""
        field_config = {}
        field_data = {'tipo': 'pf'}
        assert should_skip_field('cpf', field_config, field_data) is False

    def test_condition_satisfied(self):
        """Testa condição satisfeita (não deve pular)."""
        field_config = {'condition': {'field': 'tipo', 'equals': 'pf'}}
        field_data = {'tipo': 'pf'}
        assert should_skip_field('cpf', field_config, field_data) is False

    def test_condition_not_satisfied(self):
        """Testa condição não satisfeita (deve pular)."""
        field_config = {'condition': {'field': 'tipo', 'equals': 'pf'}}
        field_data = {'tipo': 'pj'}
        assert should_skip_field('cpf', field_config, field_data) is True

    def test_complex_condition(self):
        """Testa condição complexa."""
        field_config = {
            'condition': {'field': 'status', 'in': ['ativo', 'pendente']}
        }

        field_data = {'status': 'ativo'}
        assert should_skip_field('campo', field_config, field_data) is False

        field_data = {'status': 'inativo'}
        assert should_skip_field('campo', field_config, field_data) is True


class TestIntegrationScenarios:
    """Testes de cenários de integração."""

    def test_pessoa_fisica_juridica_scenario(self):
        """Testa cenário real de pessoa física vs jurídica."""
        class PessoaModel(BaseModel):
            tipo: str = Field(description="Tipo de pessoa: 'pf' ou 'pj'")
            cpf: str = Field(
                description="CPF (pessoa física)",
                json_schema_extra={
                    'depends_on': ['tipo'],
                    'condition': {'field': 'tipo', 'equals': 'pf'}
                }
            )
            cnpj: str = Field(
                description="CNPJ (pessoa jurídica)",
                json_schema_extra={
                    'depends_on': ['tipo'],
                    'condition': {'field': 'tipo', 'equals': 'pj'}
                }
            )

        field_configs = {
            'tipo': {},
            'cpf': {
                'depends_on': ['tipo'],
                'condition': {'field': 'tipo', 'equals': 'pf'}
            },
            'cnpj': {
                'depends_on': ['tipo'],
                'condition': {'field': 'tipo', 'equals': 'pj'}
            },
        }

        order, deps = get_field_execution_order(PessoaModel, field_configs)
        assert order.index('tipo') < order.index('cpf')
        assert order.index('tipo') < order.index('cnpj')

        # Testar skip para pessoa física
        field_data_pf = {'tipo': 'pf'}
        assert should_skip_field('cpf', field_configs['cpf'], field_data_pf) is False
        assert should_skip_field('cnpj', field_configs['cnpj'], field_data_pf) is True

        # Testar skip para pessoa jurídica
        field_data_pj = {'tipo': 'pj'}
        assert should_skip_field('cpf', field_configs['cpf'], field_data_pj) is True
        assert should_skip_field('cnpj', field_configs['cnpj'], field_data_pj) is False

    def test_nested_field_scenario(self):
        """Testa cenário com campos aninhados."""
        class EnderecoModel(BaseModel):
            pais: str
            estado: str = Field(
                json_schema_extra={
                    'depends_on': ['pais'],
                    'condition': {'field': 'pais', 'equals': 'Brasil'}
                }
            )
            cep: str = Field(
                json_schema_extra={
                    'depends_on': ['estado'],
                    'condition': {'field': 'estado', 'exists': True}
                }
            )

        field_configs = {
            'pais': {},
            'estado': {
                'depends_on': ['pais'],
                'condition': {'field': 'pais', 'equals': 'Brasil'}
            },
            'cep': {
                'depends_on': ['estado'],
                'condition': {'field': 'estado', 'exists': True}
            },
        }

        order, deps = get_field_execution_order(EnderecoModel, field_configs)
        assert order.index('pais') < order.index('estado')
        assert order.index('estado') < order.index('cep')

        # Brasil: todos os campos devem ser processados
        field_data = {'pais': 'Brasil', 'estado': 'SP'}
        assert should_skip_field('estado', field_configs['estado'], {'pais': 'Brasil'}) is False
        assert should_skip_field('cep', field_configs['cep'], field_data) is False

        # Outro país: estado e cep devem ser pulados
        field_data = {'pais': 'EUA'}
        assert should_skip_field('estado', field_configs['estado'], field_data) is True
