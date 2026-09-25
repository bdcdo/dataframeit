"""Testes para funcionalidade de execução condicional de campos."""

from unittest.mock import patch

import pytest
from pydantic import BaseModel, Field

from dataframeit.conditional import (
    check_dependencies_exist,
    detect_circular_dependencies,
    evaluate_condition,
    get_field_execution_order,
    get_nested_value,
    should_skip_field,
    topological_sort,
)


class TestGetNestedValue:
    """Testes para get_nested_value."""

    def test_simple_field(self):
        """Testa acesso a campo simples."""
        data = {"nome": "João", "idade": 30}
        assert get_nested_value(data, "nome") == "João"
        assert get_nested_value(data, "idade") == 30

    def test_nested_field(self):
        """Testa acesso a campo aninhado."""
        data = {
            "nome": "João",
            "endereco": {
                "cidade": "São Paulo",
                "rua": "Av. Paulista",
                "numero": 1000,
            },
        }
        assert get_nested_value(data, "endereco.cidade") == "São Paulo"
        assert get_nested_value(data, "endereco.rua") == "Av. Paulista"
        assert get_nested_value(data, "endereco.numero") == 1000

    def test_deeply_nested_field(self):
        """Testa acesso a campo profundamente aninhado."""
        data = {"empresa": {"endereco": {"localizacao": {"cidade": "Rio de Janeiro"}}}}
        assert get_nested_value(data, "empresa.endereco.localizacao.cidade") == "Rio de Janeiro"

    def test_nonexistent_field(self):
        """Testa acesso a campo inexistente."""
        data = {"nome": "João"}
        assert get_nested_value(data, "idade") is None
        assert get_nested_value(data, "endereco.cidade") is None

    def test_empty_path(self):
        """Testa caminho vazio."""
        data = {"nome": "João"}
        assert get_nested_value(data, "") is None

    def test_none_value(self):
        """Testa campo com valor None."""
        data = {"nome": None}
        assert get_nested_value(data, "nome") is None


class TestEvaluateCondition:
    """Testes para evaluate_condition."""

    def test_equals_true(self):
        """Testa condição equals verdadeira."""
        data = {"tipo": "pf", "status": "ativo"}
        condition = {"field": "tipo", "equals": "pf"}
        assert evaluate_condition(condition, data, "cpf") is True

    def test_equals_false(self):
        """Testa condição equals falsa."""
        data = {"tipo": "pf", "status": "ativo"}
        condition = {"field": "tipo", "equals": "pj"}
        assert evaluate_condition(condition, data, "cnpj") is False

    def test_not_equals(self):
        """Testa condição not_equals."""
        data = {"tipo": "pf"}
        assert evaluate_condition({"field": "tipo", "not_equals": "pj"}, data, "x") is True
        assert evaluate_condition({"field": "tipo", "not_equals": "pf"}, data, "x") is False

    def test_in_list(self):
        """Testa condição in (valor está na lista)."""
        data = {"status": "ativo"}
        condition = {"field": "status", "in": ["ativo", "pendente"]}
        assert evaluate_condition(condition, data, "x") is True

        data = {"status": "inativo"}
        assert evaluate_condition(condition, data, "x") is False

    def test_not_in_list(self):
        """Testa condição not_in (valor não está na lista)."""
        data = {"status": "inativo"}
        condition = {"field": "status", "not_in": ["ativo", "pendente"]}
        assert evaluate_condition(condition, data, "x") is True

        data = {"status": "ativo"}
        assert evaluate_condition(condition, data, "x") is False

    def test_exists_true(self):
        """Testa condição exists verdadeira."""
        data = {"nome": "João"}
        condition = {"field": "nome", "exists": True}
        assert evaluate_condition(condition, data, "x") is True

    def test_exists_false(self):
        """Testa condição exists falsa."""
        data = {"nome": None}
        condition = {"field": "nome", "exists": True}
        assert evaluate_condition(condition, data, "x") is False

        data = {}
        condition = {"field": "nome", "exists": False}
        assert evaluate_condition(condition, data, "x") is True

    def test_nested_field_condition(self):
        """Testa condição com campo aninhado."""
        data = {"endereco": {"cidade": "São Paulo", "estado": "SP"}}
        condition = {"field": "endereco.cidade", "equals": "São Paulo"}
        assert evaluate_condition(condition, data, "x") is True

        condition = {"field": "endereco.estado", "in": ["SP", "RJ"]}
        assert evaluate_condition(condition, data, "x") is True

    def test_callable_condition(self):
        """Testa condição callable."""
        data = {"idade": 25}

        def condition(d):
            return d.get("idade", 0) >= 18

        assert evaluate_condition(condition, data, "pode_votar") is True

        data = {"idade": 15}
        assert evaluate_condition(condition, data, "pode_votar") is False

    def test_none_condition(self):
        """Testa condição None (sempre True)."""
        data = {"nome": "João"}
        assert evaluate_condition(None, data, "x") is True

    def test_invalid_condition_type(self):
        """Testa tipo de condição inválido."""
        data = {"nome": "João"}
        assert evaluate_condition("invalid", data, "x") is False
        assert evaluate_condition(123, data, "x") is False

    def test_condition_without_field(self):
        """Testa condição dict sem campo 'field'."""
        data = {"nome": "João"}
        condition = {"equals": "João"}
        assert evaluate_condition(condition, data, "x") is False

    def test_condition_without_operator(self):
        """Testa condição dict sem operador válido."""
        data = {"nome": "João"}
        condition = {"field": "nome"}
        assert evaluate_condition(condition, data, "x") is False


class TestDependencies:
    """Testes para funções de dependências."""

    def test_check_dependencies_exist_all_valid(self):
        """Testa verificação de dependências válidas."""
        all_fields = {"a", "b", "c"}
        missing = check_dependencies_exist("d", ["a", "b"], all_fields)
        assert missing == []

    def test_check_dependencies_exist_some_invalid(self):
        """Testa verificação com dependências inválidas."""
        all_fields = {"a", "b", "c"}
        missing = check_dependencies_exist("d", ["a", "x", "y"], all_fields)
        assert set(missing) == {"x", "y"}

    def test_check_dependencies_nested_field(self):
        """Testa verificação de dependência com campo aninhado."""
        all_fields = {"endereco", "nome"}
        missing = check_dependencies_exist("x", ["endereco.cidade"], all_fields)
        assert missing == []

    def test_detect_circular_dependencies_no_cycle(self):
        """Testa detecção sem ciclos."""
        deps = {"a": [], "b": ["a"], "c": ["a", "b"]}
        assert detect_circular_dependencies(deps) is None

    def test_detect_circular_dependencies_simple_cycle(self):
        """Testa detecção de ciclo simples."""
        deps = {"a": ["b"], "b": ["a"]}
        cycle = detect_circular_dependencies(deps)
        assert cycle is not None
        assert "a" in cycle and "b" in cycle

    def test_detect_circular_dependencies_complex_cycle(self):
        """Testa detecção de ciclo complexo."""
        deps = {"a": ["b"], "b": ["c"], "c": ["a"]}
        cycle = detect_circular_dependencies(deps)
        assert cycle is not None
        assert set(cycle) >= {"a", "b", "c"}

    def test_topological_sort_simple(self):
        """Testa ordenação topológica simples."""
        deps = {"a": [], "b": ["a"], "c": ["b"]}
        result = topological_sort(deps)
        assert result.index("a") < result.index("b")
        assert result.index("b") < result.index("c")

    def test_topological_sort_complex(self):
        """Testa ordenação topológica complexa."""
        deps = {
            "a": [],
            "b": [],
            "c": ["a"],
            "d": ["b"],
            "e": ["c", "d"],
        }
        result = topological_sort(deps)
        assert result.index("a") < result.index("c")
        assert result.index("b") < result.index("d")
        assert result.index("c") < result.index("e")
        assert result.index("d") < result.index("e")

    def test_topological_sort_with_cycle_raises(self):
        """Testa que ordenação topológica levanta erro com ciclo."""
        deps = {"a": ["b"], "b": ["c"], "c": ["a"]}
        with pytest.raises(ValueError, match="Dependências circulares"):
            topological_sort(deps)


class TestGetFieldExecutionOrder:
    """Testes para get_field_execution_order."""

    def test_simple_order(self):
        """Testa ordem de execução simples (depends_on derivado de condition dict)."""

        class SimpleModel(BaseModel):
            a: str
            b: str = Field(json_schema_extra={"condition": {"field": "a", "equals": "x"}})

        field_configs = {
            "a": {},
            "b": {"condition": {"field": "a", "equals": "x"}},
        }

        order, deps = get_field_execution_order(SimpleModel, field_configs)
        assert order.index("a") < order.index("b")
        assert deps["b"] == ["a"]

    def test_complex_order(self):
        """Testa ordem de execução complexa via auto-derivação."""

        class ComplexModel(BaseModel):
            tipo: str
            cpf: str = Field(json_schema_extra={"condition": {"field": "tipo", "equals": "pf"}})
            cnpj: str = Field(json_schema_extra={"condition": {"field": "tipo", "equals": "pj"}})
            validacao: str = Field(
                json_schema_extra={
                    "depends_on": ["cpf", "cnpj"],
                    "condition": lambda data: bool(data.get("cpf") or data.get("cnpj")),
                }
            )

        field_configs = {
            "tipo": {},
            "cpf": {"condition": {"field": "tipo", "equals": "pf"}},
            "cnpj": {"condition": {"field": "tipo", "equals": "pj"}},
            "validacao": {
                "depends_on": ["cpf", "cnpj"],
                "condition": lambda data: bool(data.get("cpf") or data.get("cnpj")),
            },
        }

        order, _deps = get_field_execution_order(ComplexModel, field_configs)
        assert order.index("tipo") < order.index("cpf")
        assert order.index("tipo") < order.index("cnpj")
        assert order.index("cpf") < order.index("validacao")
        assert order.index("cnpj") < order.index("validacao")

    def test_missing_dependency_raises(self):
        """Testa que dependência inexistente levanta erro (via condition dict)."""

        class ModelWithInvalidDep(BaseModel):
            a: str = Field(json_schema_extra={"condition": {"field": "nonexistent", "equals": "x"}})

        field_configs = {
            "a": {"condition": {"field": "nonexistent", "equals": "x"}},
        }

        with pytest.raises(ValueError, match="depende de campos inexistentes"):
            get_field_execution_order(ModelWithInvalidDep, field_configs)

    def test_circular_dependency_raises(self):
        """Testa que dependência circular levanta erro (via condition dict)."""

        class ModelWithCircular(BaseModel):
            a: str = Field(json_schema_extra={"condition": {"field": "b", "equals": "x"}})
            b: str = Field(json_schema_extra={"condition": {"field": "a", "equals": "x"}})

        field_configs = {
            "a": {"condition": {"field": "b", "equals": "x"}},
            "b": {"condition": {"field": "a", "equals": "x"}},
        }

        with pytest.raises(ValueError, match="Dependências circulares"):
            get_field_execution_order(ModelWithCircular, field_configs)

    def test_auto_derive_from_condition_dict(self):
        """Testa que condition dict deriva depends_on automaticamente."""

        class M(BaseModel):
            tipo: str
            cpf: str = Field(json_schema_extra={"condition": {"field": "tipo", "equals": "pf"}})

        configs = {
            "tipo": {},
            "cpf": {"condition": {"field": "tipo", "equals": "pf"}},
        }
        order, deps = get_field_execution_order(M, configs)
        assert deps["cpf"] == ["tipo"]
        assert order.index("tipo") < order.index("cpf")

    def test_explicit_depends_on_unions_with_condition(self):
        """Testa que depends_on explícito é unido com a derivação automática."""

        class M(BaseModel):
            a: str
            b: str
            c: str = Field(
                json_schema_extra={
                    "depends_on": ["a", "b"],
                    "condition": {"field": "a", "equals": "x"},
                }
            )

        configs = {
            "a": {},
            "b": {},
            "c": {
                "depends_on": ["a", "b"],
                "condition": {"field": "a", "equals": "x"},
            },
        }
        order, deps = get_field_execution_order(M, configs)
        assert deps["c"] == ["a", "b"]
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("c")

    def test_explicit_depends_on_unions_with_condition_field(self):
        """Testa união quando o explícito não inclui o campo da condition."""

        class M(BaseModel):
            tipo: str
            x: str
            c: str = Field(
                json_schema_extra={
                    "depends_on": ["x"],
                    "condition": {"field": "tipo", "equals": "pf"},
                }
            )

        configs = {
            "tipo": {},
            "x": {},
            "c": {
                "depends_on": ["x"],
                "condition": {"field": "tipo", "equals": "pf"},
            },
        }
        order, deps = get_field_execution_order(M, configs)
        assert set(deps["c"]) == {"x", "tipo"}
        assert deps["c"][0] == "x"  # explícito preservado primeiro
        assert order.index("x") < order.index("c")
        assert order.index("tipo") < order.index("c")

    def test_nested_condition_field_uses_root(self):
        """Testa que campo aninhado em condition ('endereco.cidade') resolve para raiz ('endereco')."""

        class M(BaseModel):
            endereco: dict
            taxa: float = Field(
                json_schema_extra={"condition": {"field": "endereco.cidade", "equals": "SP"}}
            )

        configs = {
            "endereco": {},
            "taxa": {"condition": {"field": "endereco.cidade", "equals": "SP"}},
        }
        order, deps = get_field_execution_order(M, configs)
        assert deps["taxa"] == ["endereco"]
        assert order.index("endereco") < order.index("taxa")

    def test_depends_on_without_condition_emits_warning(self, caplog):
        """Testa que depends_on sem condition emite warning e é ignorado."""
        import logging

        class M(BaseModel):
            a: str
            b: str = Field(json_schema_extra={"depends_on": ["a"]})

        configs = {
            "a": {},
            "b": {"depends_on": ["a"]},
        }
        with caplog.at_level(logging.WARNING, logger="dataframeit.conditional"):
            _order, deps = get_field_execution_order(M, configs)
        assert deps["b"] == []
        assert any(
            "depends_on" in rec.message and "condition" in rec.message for rec in caplog.records
        )

    def test_callable_condition_without_depends_on_has_no_deps(self):
        """Testa que callable sem depends_on não impõe ordem (não levanta erro)."""

        class M(BaseModel):
            a: str
            b: str = Field(json_schema_extra={"condition": lambda data: bool(data.get("a"))})

        configs = {
            "a": {},
            "b": {"condition": lambda data: bool(data.get("a"))},
        }
        order, deps = get_field_execution_order(M, configs)
        assert deps["b"] == []
        assert set(order) == {"a", "b"}

    def test_callable_condition_without_depends_on_emits_warning(self, caplog):
        """Testa que callable sem depends_on emite warning."""
        import logging

        class M(BaseModel):
            a: str
            b: str = Field(json_schema_extra={"condition": lambda data: bool(data.get("a"))})

        configs = {
            "a": {},
            "b": {"condition": lambda data: bool(data.get("a"))},
        }
        with caplog.at_level(logging.WARNING, logger="dataframeit.conditional"):
            get_field_execution_order(M, configs)
        assert any(
            "callable" in rec.message and "depends_on" in rec.message for rec in caplog.records
        )

    def test_callable_condition_with_depends_on_no_warning(self, caplog):
        """Testa que callable com depends_on não emite warning."""
        import logging

        class M(BaseModel):
            a: str
            b: str = Field(
                json_schema_extra={
                    "depends_on": ["a"],
                    "condition": lambda data: bool(data.get("a")),
                }
            )

        configs = {
            "a": {},
            "b": {
                "depends_on": ["a"],
                "condition": lambda data: bool(data.get("a")),
            },
        }
        with caplog.at_level(logging.WARNING, logger="dataframeit.conditional"):
            _order, deps = get_field_execution_order(M, configs)
        assert deps["b"] == ["a"]
        assert not any(
            "callable" in rec.message and "depends_on" in rec.message for rec in caplog.records
        )


class TestShouldSkipField:
    """Testes para should_skip_field."""

    def test_no_condition(self):
        """Testa campo sem condição (não deve pular)."""
        field_config = {}
        field_data = {"tipo": "pf"}
        assert should_skip_field("cpf", field_config, field_data) is False

    def test_condition_satisfied(self):
        """Testa condição satisfeita (não deve pular)."""
        field_config = {"condition": {"field": "tipo", "equals": "pf"}}
        field_data = {"tipo": "pf"}
        assert should_skip_field("cpf", field_config, field_data) is False

    def test_condition_not_satisfied(self):
        """Testa condição não satisfeita (deve pular)."""
        field_config = {"condition": {"field": "tipo", "equals": "pf"}}
        field_data = {"tipo": "pj"}
        assert should_skip_field("cpf", field_config, field_data) is True

    def test_complex_condition(self):
        """Testa condição complexa."""
        field_config = {"condition": {"field": "status", "in": ["ativo", "pendente"]}}

        field_data = {"status": "ativo"}
        assert should_skip_field("campo", field_config, field_data) is False

        field_data = {"status": "inativo"}
        assert should_skip_field("campo", field_config, field_data) is True


class TestIntegrationScenarios:
    """Testes de cenários de integração."""

    def test_pessoa_fisica_juridica_scenario(self):
        """Testa cenário real de pessoa física vs jurídica."""

        class PessoaModel(BaseModel):
            tipo: str = Field(description="Tipo de pessoa: 'pf' ou 'pj'")
            cpf: str = Field(
                description="CPF (pessoa física)",
                json_schema_extra={
                    "depends_on": ["tipo"],
                    "condition": {"field": "tipo", "equals": "pf"},
                },
            )
            cnpj: str = Field(
                description="CNPJ (pessoa jurídica)",
                json_schema_extra={
                    "depends_on": ["tipo"],
                    "condition": {"field": "tipo", "equals": "pj"},
                },
            )

        field_configs = {
            "tipo": {},
            "cpf": {"depends_on": ["tipo"], "condition": {"field": "tipo", "equals": "pf"}},
            "cnpj": {"depends_on": ["tipo"], "condition": {"field": "tipo", "equals": "pj"}},
        }

        order, _deps = get_field_execution_order(PessoaModel, field_configs)
        assert order.index("tipo") < order.index("cpf")
        assert order.index("tipo") < order.index("cnpj")

        # Testar skip para pessoa física
        field_data_pf = {"tipo": "pf"}
        assert should_skip_field("cpf", field_configs["cpf"], field_data_pf) is False
        assert should_skip_field("cnpj", field_configs["cnpj"], field_data_pf) is True

        # Testar skip para pessoa jurídica
        field_data_pj = {"tipo": "pj"}
        assert should_skip_field("cpf", field_configs["cpf"], field_data_pj) is True
        assert should_skip_field("cnpj", field_configs["cnpj"], field_data_pj) is False

    def test_nested_field_scenario(self):
        """Testa cenário com campos aninhados."""

        class EnderecoModel(BaseModel):
            pais: str
            estado: str = Field(
                json_schema_extra={
                    "depends_on": ["pais"],
                    "condition": {"field": "pais", "equals": "Brasil"},
                }
            )
            cep: str = Field(
                json_schema_extra={
                    "depends_on": ["estado"],
                    "condition": {"field": "estado", "exists": True},
                }
            )

        field_configs = {
            "pais": {},
            "estado": {"depends_on": ["pais"], "condition": {"field": "pais", "equals": "Brasil"}},
            "cep": {"depends_on": ["estado"], "condition": {"field": "estado", "exists": True}},
        }

        order, _deps = get_field_execution_order(EnderecoModel, field_configs)
        assert order.index("pais") < order.index("estado")
        assert order.index("estado") < order.index("cep")

        # Brasil: todos os campos devem ser processados
        field_data = {"pais": "Brasil", "estado": "SP"}
        assert should_skip_field("estado", field_configs["estado"], {"pais": "Brasil"}) is False
        assert should_skip_field("cep", field_configs["cep"], field_data) is False

        # Outro país: estado e cep devem ser pulados
        field_data = {"pais": "EUA"}
        assert should_skip_field("estado", field_configs["estado"], field_data) is True


# =============================================================================
# Condições no modo por grupo e validação de modo
# =============================================================================


_CONDICAO_PF = {"condition": {"field": "tipo", "equals": "pf"}}
_CONDICAO_PJ = {"condition": {"field": "tipo", "equals": "pj"}}


class ModeloPessoaCondicional(BaseModel):
    tipo: str = Field(description="'pf' ou 'pj'")
    cpf: str | None = Field(None, json_schema_extra=_CONDICAO_PF)
    cnpj: str | None = Field(None, json_schema_extra=_CONDICAO_PJ)
    razao_social: str | None = Field(None, json_schema_extra=_CONDICAO_PJ)


def _config_por_grupo(grupos):
    from dataframeit.llm import LLMConfig, SearchConfig, SearchGroupConfig

    return LLMConfig(
        model="teste",
        provider="teste",
        api_key=None,
        max_retries=1,
        base_delay=0.0,
        max_delay=0.0,
        rate_limit_delay=0,
        search_config=SearchConfig(
            enabled=True,
            per_field=True,
            groups={nome: SearchGroupConfig(fields=campos) for nome, campos in grupos.items()},
        ),
    )


def _call_agent_falso(valores, chamadas):
    """Simula call_agent: registra os campos pedidos e devolve `valores`."""

    def call_agent(text, model, prompt, config, save_trace=None):
        campos = list(model.model_fields.keys())
        chamadas.append(campos)
        return {
            "data": {campo: valores[campo] for campo in campos},
            "usage": {"input_tokens": 1, "search_count": 1},
        }

    return call_agent


class TestCondicaoNoModoPorGrupo:
    """call_agent_per_group aplica `condition` como o caminho por campo."""

    def test_grupo_com_condicao_falsa_nao_e_chamado(self):
        from dataframeit.agent import call_agent_per_group

        chamadas = []
        valores = {"tipo": "pf", "cpf": "123", "cnpj": "999", "razao_social": "XYZ"}
        config = _config_por_grupo({"empresa": ["cnpj", "razao_social"]})

        with patch(
            "dataframeit.agent.call_agent", side_effect=_call_agent_falso(valores, chamadas)
        ):
            resultado = call_agent_per_group(
                "texto", ModeloPessoaCondicional, "Analise {texto}", config
            )

        assert resultado["data"] == {
            "tipo": "pf",
            "cpf": "123",
            "cnpj": None,
            "razao_social": None,
        }
        assert sorted(map(tuple, chamadas)) == [("cpf",), ("tipo",)]
        # `tipo` precisa vir antes de `cpf`, que depende dele
        assert chamadas.index(["tipo"]) < chamadas.index(["cpf"])
        assert resultado["usage"]["search_count"] == 2

    def test_grupo_pede_so_os_campos_com_condicao_verdadeira(self):
        from dataframeit.agent import call_agent_per_group

        chamadas = []
        valores = {"tipo": "pj", "cpf": "123", "cnpj": "999", "razao_social": "XYZ"}
        config = _config_por_grupo({"documentos": ["cpf", "cnpj"]})

        with patch(
            "dataframeit.agent.call_agent", side_effect=_call_agent_falso(valores, chamadas)
        ):
            resultado = call_agent_per_group(
                "texto", ModeloPessoaCondicional, "Analise {texto}", config
            )

        assert ["cnpj"] in chamadas
        assert not any("cpf" in campos for campos in chamadas)
        assert resultado["data"]["cpf"] is None
        assert resultado["data"]["cnpj"] == "999"
        assert resultado["data"]["razao_social"] == "XYZ"

    def test_condicao_dentro_do_mesmo_grupo_e_avaliada_na_resposta(self):
        """Dependência no mesmo grupo não tem valor antes da chamada: o campo é anulado depois."""
        from dataframeit.agent import call_agent_per_group

        chamadas = []
        valores = {"tipo": "pj", "cpf": "123", "cnpj": "999", "razao_social": "XYZ"}
        config = _config_por_grupo({"identificacao": ["tipo", "cpf"]})

        with patch(
            "dataframeit.agent.call_agent", side_effect=_call_agent_falso(valores, chamadas)
        ):
            resultado = call_agent_per_group(
                "texto", ModeloPessoaCondicional, "Analise {texto}", config
            )

        assert resultado["data"]["tipo"] == "pj"
        assert resultado["data"]["cpf"] is None
        assert resultado["data"]["cnpj"] == "999"

    def test_ciclo_entre_grupo_e_campo_isolado_levanta_erro(self):
        from dataframeit.agent import call_agent_per_group

        class ModeloCiclico(BaseModel):
            a: str
            b: str | None = Field(
                None, json_schema_extra={"condition": {"field": "a", "exists": True}}
            )
            c: str | None = Field(
                None, json_schema_extra={"condition": {"field": "b", "exists": True}}
            )

        config = _config_por_grupo({"g": ["a", "c"]})

        with (
            patch("dataframeit.agent.call_agent") as call_agent,
            pytest.raises(ValueError, match="grupo 'g'"),
        ):
            call_agent_per_group("texto", ModeloCiclico, "Analise {texto}", config)
        call_agent.assert_not_called()

    def test_grupo_segue_a_ordem_de_search_groups(self):
        from dataframeit.agent import call_agent_per_group

        chamadas = []
        valores = {"tipo": "pj", "cpf": "123", "cnpj": "999", "razao_social": "XYZ"}
        config = _config_por_grupo({"empresa": ["razao_social", "cnpj"]})

        with patch(
            "dataframeit.agent.call_agent", side_effect=_call_agent_falso(valores, chamadas)
        ):
            call_agent_per_group("texto", ModeloPessoaCondicional, "Analise {texto}", config)

        assert ["razao_social", "cnpj"] in chamadas


def test_ordem_de_execucao_nao_depende_do_hash_seed():
    """A ordem entre campos independentes segue o modelo em qualquer processo."""
    import os
    import subprocess
    import sys

    codigo = (
        "from pydantic import BaseModel\n"
        "from dataframeit.conditional import get_field_execution_order\n"
        "class M(BaseModel):\n"
        "    zeta: str\n    alfa: str\n    meio: str\n    beta: str\n"
        "print(get_field_execution_order(M, {})[0])\n"
    )
    saidas = set()
    for semente in ("0", "1", "2", "3"):
        ambiente = {**os.environ, "PYTHONHASHSEED": semente}
        saida = subprocess.run(
            [sys.executable, "-c", codigo],
            env=ambiente,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        saidas.add(saida)

    assert saidas == {"['zeta', 'alfa', 'meio', 'beta']"}


_RESPOSTA_PF = {
    "data": {"tipo": "pf", "cpf": "123", "cnpj": None, "razao_social": None},
    "usage": None,
}


class TestCondicaoForaDoModoPorCampo:
    """`condition` e `depends_on` só são aplicados com search_per_field=True."""

    @pytest.mark.parametrize(
        "opcoes_busca",
        [
            {},
            {"use_search": True, "search_per_field": False},
        ],
    )
    def test_condition_sem_search_per_field_levanta_erro(self, opcoes_busca):
        import pandas as pd

        from dataframeit import dataframeit

        with (
            patch("dataframeit.core.validate_provider_dependencies"),
            patch("dataframeit.core.validate_search_dependencies"),
            patch("dataframeit.core.call_langchain", return_value=_RESPOSTA_PF) as call_langchain,
            patch("dataframeit.agent.call_agent", return_value=_RESPOSTA_PF) as call_agent,
            pytest.raises(ValueError, match="search_per_field=True"),
        ):
            dataframeit(
                pd.DataFrame({"texto": ["x"]}),
                questions=ModeloPessoaCondicional,
                prompt="Analise {texto}",
                **opcoes_busca,
            )
        call_langchain.assert_not_called()
        call_agent.assert_not_called()

    def test_depends_on_sem_search_per_field_levanta_erro(self):
        import pandas as pd

        from dataframeit import dataframeit

        class ModeloDependsOn(BaseModel):
            a: str
            b: str = Field(json_schema_extra={"depends_on": ["a"]})

        with (
            patch("dataframeit.core.validate_provider_dependencies"),
            patch(
                "dataframeit.core.call_langchain",
                return_value={"data": {"a": "1", "b": "2"}, "usage": None},
            ) as call_langchain,
            pytest.raises(ValueError, match="depends_on"),
        ):
            dataframeit(
                pd.DataFrame({"texto": ["x"]}),
                questions=ModeloDependsOn,
                prompt="Analise {texto}",
            )
        call_langchain.assert_not_called()
