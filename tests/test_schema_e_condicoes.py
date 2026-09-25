"""Schema enviado ao LLM e validação de condições antes do processamento.

O fake de call_agent gera o JSON Schema do modelo recebido, como o provider
faria. Os testes que só contam campos deixavam passar modelos cujo schema
não serializa.
"""

import json
from typing import Optional, Union
from unittest.mock import patch

import pandas as pd
import pytest
from pydantic import BaseModel, Field

from dataframeit import dataframeit
from dataframeit.llm import LLMConfig, SearchConfig, SearchGroupConfig, build_prompt

_CHAVES_DA_BIBLIOTECA = (
    "condition",
    "depends_on",
    "prompt",
    "prompt_replace",
    "prompt_append",
    "search_depth",
    "max_results",
)


def _config(groups=None):
    return LLMConfig(
        model="teste",
        provider="teste",
        api_key=None,
        max_retries=1,
        base_delay=0.0,
        max_delay=0.0,
        rate_limit_delay=0,
        search_config=SearchConfig(enabled=True, per_field=True, groups=groups),
    )


def _call_agent_que_gera_schema(valores, schemas, prompts=None):
    def call_agent(text, model, prompt, config, save_trace=None):
        schemas.append(model.model_json_schema())
        if prompts is not None:
            prompts.append(prompt)
        return {
            "data": {campo: valores.get(campo) for campo in model.model_fields},
            "usage": {"search_count": 1},
        }

    return call_agent


def _chaves_da_biblioteca_no_schema(schema) -> set:
    texto = json.dumps(schema)
    return {chave for chave in _CHAVES_DA_BIBLIOTECA if f'"{chave}"' in texto}


class ModeloMulta(BaseModel):
    tem_multa: bool
    valor_multa: Optional[float] = Field(
        default=None,
        json_schema_extra={
            "condition": lambda dados: dados.get("tem_multa") is True,
            "depends_on": ["tem_multa"],
            "prompt_append": "Valor em reais.",
            "search_depth": "advanced",
        },
    )


_VALORES_MULTA = {"tem_multa": True, "valor_multa": 100.0}


def _patches_de_execucao():
    return (
        patch("dataframeit.core.validate_provider_dependencies"),
        patch("dataframeit.core.validate_search_dependencies"),
    )


# =============================================================================
# Schema sem as chaves da biblioteca
# =============================================================================


class TestSchemaSemChavesDaBiblioteca:
    def test_condition_callable_no_modo_por_campo(self):
        from dataframeit.agent import call_agent_per_field

        schemas = []
        falso = _call_agent_que_gera_schema(_VALORES_MULTA, schemas)
        with patch("dataframeit.agent.call_agent", side_effect=falso):
            resultado = call_agent_per_field("texto", ModeloMulta, "Analise {texto}", _config())

        assert resultado["data"] == _VALORES_MULTA
        assert len(schemas) == 2
        assert all(not _chaves_da_biblioteca_no_schema(s) for s in schemas)

    def test_condition_callable_no_modo_por_grupo(self):
        from dataframeit.agent import call_agent_per_group

        class ModeloGrupo(BaseModel):
            tem_multa: bool
            valor_multa: Optional[float] = Field(
                default=None,
                json_schema_extra={
                    "condition": lambda dados: dados.get("tem_multa") is True,
                    "depends_on": ["tem_multa"],
                },
            )
            orgao: Optional[str] = None

        schemas = []
        valores = {**_VALORES_MULTA, "orgao": "Procon"}
        config = _config(groups={"multa": SearchGroupConfig(fields=["valor_multa", "orgao"])})
        falso = _call_agent_que_gera_schema(valores, schemas)
        with patch("dataframeit.agent.call_agent", side_effect=falso):
            resultado = call_agent_per_group("texto", ModeloGrupo, "Analise {texto}", config)

        assert resultado["data"] == valores
        assert all(not _chaves_da_biblioteca_no_schema(s) for s in schemas)

    def test_dataframeit_processa_a_linha(self):
        schemas = []
        falso = _call_agent_que_gera_schema(_VALORES_MULTA, schemas)
        provider, busca = _patches_de_execucao()
        with provider, busca, patch("dataframeit.agent.call_agent", side_effect=falso):
            resultado = dataframeit(
                pd.DataFrame({"texto": ["auto de infração"]}),
                questions=ModeloMulta,
                prompt="Analise {texto}",
                use_search=True,
                search_per_field=True,
            )

        assert resultado["valor_multa"].tolist() == [100.0]
        assert "_dataframeit_status" not in resultado.columns

    def test_llm_field_preserva_o_original_e_o_resto_do_extra(self):
        from dataframeit.agent import _llm_field

        class Modelo(BaseModel):
            a: Optional[str] = Field(
                None,
                json_schema_extra={"examples": ["x"], "condition": {"field": "b", "equals": 1}},
            )
            b: Optional[str] = Field(None, json_schema_extra={"prompt_append": "y"})
            c: Optional[str] = Field(None, json_schema_extra=lambda schema: schema.update(k=1))

        original = Modelo.model_fields["a"]
        limpo = _llm_field(original)
        assert limpo.json_schema_extra == {"examples": ["x"]}
        assert original.json_schema_extra == {
            "examples": ["x"],
            "condition": {"field": "b", "equals": 1},
        }
        assert _llm_field(Modelo.model_fields["b"]).json_schema_extra is None
        # Callable é do usuário e não carrega chaves da biblioteca
        assert _llm_field(Modelo.model_fields["c"]) is Modelo.model_fields["c"]


# =============================================================================
# Validação antes de processar
# =============================================================================


def _executar(questions, **opcoes):
    provider, busca = _patches_de_execucao()
    with (
        provider,
        busca,
        patch("dataframeit.agent.call_agent") as call_agent,
        patch("dataframeit.core.call_langchain") as call_langchain,
    ):
        try:
            return dataframeit(
                pd.DataFrame({"texto": ["x", "y"]}),
                questions=questions,
                prompt="Analise {texto}",
                **opcoes,
            )
        finally:
            call_agent.assert_not_called()
            call_langchain.assert_not_called()


class Endereco(BaseModel):
    uf: str
    cidade: Optional[str] = Field(
        None, json_schema_extra={"condition": {"field": "uf", "equals": "SP"}}
    )


class ModeloComCondicaoAninhada(BaseModel):
    endereco: Endereco


class TestValidacaoAntesDeProcessar:
    @pytest.mark.parametrize(
        "opcoes",
        [
            {},
            {"use_search": True, "search_per_field": True},
        ],
    )
    def test_condition_em_campo_aninhado_levanta_erro(self, opcoes):
        with pytest.raises(ValueError, match="endereco.cidade"):
            _executar(ModeloComCondicaoAninhada, **opcoes)

    def test_condition_em_item_de_lista_levanta_erro(self):
        class Item(BaseModel):
            tipo: str
            valor: Optional[str] = Field(
                None, json_schema_extra={"condition": {"field": "tipo", "equals": "a"}}
            )

        class Modelo(BaseModel):
            itens: list[Item] = []

        with pytest.raises(ValueError, match="itens.valor"):
            _executar(Modelo, use_search=True, search_per_field=True)

    @pytest.mark.parametrize(
        "extra, trecho",
        [
            ({"max_results": 50}, "max_results"),
            ({"max_results": 0}, "max_results"),
            ({"search_depth": "profunda"}, "search_depth"),
        ],
    )
    def test_override_invalido_por_campo_levanta_erro(self, extra, trecho):
        class Modelo(BaseModel):
            campo: Optional[str] = Field(None, json_schema_extra=extra)

        with pytest.raises(ValueError, match=trecho):
            _executar(Modelo, use_search=True, search_per_field=True)

    def test_override_invalido_em_campo_aninhado_levanta_erro(self):
        class Interno(BaseModel):
            campo: Optional[str] = Field(None, json_schema_extra={"max_results": 99})

        class Modelo(BaseModel):
            interno: Optional[Interno] = None

        with pytest.raises(ValueError, match="interno.campo"):
            _executar(Modelo, use_search=True, search_per_field=True)

    def test_dependencia_circular_levanta_erro_antes_das_linhas(self):
        class Modelo(BaseModel):
            a: Optional[str] = Field(
                None, json_schema_extra={"condition": {"field": "b", "equals": "1"}}
            )
            b: Optional[str] = Field(
                None, json_schema_extra={"condition": {"field": "a", "equals": "1"}}
            )

        with pytest.raises(ValueError, match="circular"):
            _executar(Modelo, use_search=True, search_per_field=True)

    def test_dependencia_circular_entre_grupo_e_campo_levanta_erro(self):
        class Modelo(BaseModel):
            a: Optional[str] = Field(
                None, json_schema_extra={"condition": {"field": "c", "equals": "1"}}
            )
            b: Optional[str] = None
            c: Optional[str] = Field(
                None, json_schema_extra={"condition": {"field": "b", "equals": "1"}}
            )

        with pytest.raises(ValueError, match="circular"):
            _executar(
                Modelo,
                use_search=True,
                search_per_field=True,
                search_groups={"g": {"fields": ["a", "b"]}},
            )

    def test_configuracao_em_lista_dentro_de_lista_levanta_erro(self):
        class Sub(BaseModel):
            x: Optional[str] = Field(None, json_schema_extra={"prompt_append": "Busque x."})

        class Item(BaseModel):
            subs: list[Sub] = []

        class Modelo(BaseModel):
            itens: list[Item] = []

        with pytest.raises(ValueError, match="itens.subs.x"):
            _executar(Modelo, use_search=True, search_per_field=True)

    @pytest.mark.parametrize(
        "opcoes, falta",
        [
            ({}, "use_search=True e search_per_field=True"),
            ({"search_per_field": True}, "use_search=True"),
            ({"use_search": True}, "search_per_field=True"),
        ],
    )
    def test_mensagem_de_configuracao_por_campo_nomeia_o_que_falta(self, opcoes, falta):
        class Modelo(BaseModel):
            campo: Optional[str] = Field(None, json_schema_extra={"prompt_append": "x"})

        with pytest.raises(ValueError) as erro:
            _executar(Modelo, **opcoes)
        assert f"requerem {falta}" in str(erro.value)


# =============================================================================
# Ordem de execução
# =============================================================================


class TestOrdemDeExecucao:
    @pytest.mark.parametrize(
        "depends_on",
        [
            ["endereco.cidade", "endereco.uf"],
            ["tipo", "tipo"],
        ],
    )
    def test_dependencias_com_a_mesma_raiz_nao_somem_da_ordem(self, depends_on):
        from dataframeit.conditional import get_field_execution_order

        class Modelo(BaseModel):
            tipo: Optional[str] = None
            endereco: Optional[Endereco] = None
            campo: Optional[str] = None

        configs = {
            "tipo": {},
            "endereco": {},
            "campo": {"condition": lambda dados: True, "depends_on": depends_on},
        }
        ordem, _ = get_field_execution_order(Modelo, configs)
        assert sorted(ordem) == ["campo", "endereco", "tipo"]
        raiz = depends_on[0].split(".")[0]
        assert ordem.index(raiz) < ordem.index("campo")


# =============================================================================
# Prompts por campo e por grupo
# =============================================================================


class TestTextoNoPrompt:
    def test_prompt_por_campo_sem_texto_recebe_o_texto(self):
        from dataframeit.agent import _build_field_prompt

        prompt = _build_field_prompt(
            "Analise {texto}", "campo", None, {"prompt": "Busque o valor da causa."}
        )
        assert build_prompt(prompt, "TEXTO-DA-LINHA").count("TEXTO-DA-LINHA") == 1

    @pytest.mark.parametrize(
        "prompt_do_grupo",
        [
            "Busque: {query}",
            "Busque o contexto",
            "Busque: {texto}",
        ],
    )
    def test_prompt_de_grupo_recebe_o_texto_uma_vez(self, prompt_do_grupo):
        from dataframeit.agent import call_agent_per_group

        class Modelo(BaseModel):
            a: Optional[str] = None
            b: Optional[str] = None

        prompts = []
        config = _config(
            groups={
                "g": SearchGroupConfig(fields=["a", "b"], prompt=prompt_do_grupo),
            }
        )
        falso = _call_agent_que_gera_schema({"a": "1", "b": "2"}, [], prompts)
        with patch("dataframeit.agent.call_agent", side_effect=falso):
            call_agent_per_group("TEXTO-DA-LINHA", Modelo, "Analise {texto}", config)

        assert build_prompt(prompts[0], "TEXTO-DA-LINHA").count("TEXTO-DA-LINHA") == 1


# =============================================================================
# Modelos auto-referenciais com list['Modelo']
# =============================================================================


def test_configuracao_em_modelo_auto_referencial_com_list_builtin_e_detectada():
    from dataframeit.conditional import _collect_configured_fields

    class No(BaseModel):
        nome: Optional[str] = Field(None, json_schema_extra={"prompt_append": "x"})
        filhos: Optional[list["No"]] = None

    class Raiz(BaseModel):
        no: Optional[No] = None

    caminhos = [path for path, *_ in _collect_configured_fields(Raiz)]
    assert caminhos == ["no.nome"]

    class Folha(BaseModel):
        valor: Optional[str] = Field(None, json_schema_extra={"prompt_append": "y"})

    class ComFolhas(BaseModel):
        folhas: list["Folha"] = []

    assert [path for path, *_ in _collect_configured_fields(ComFolhas)] == ["folhas.valor"]


# =============================================================================
# reprocess_columns no modo por campo
# =============================================================================


@pytest.mark.parametrize("parallel_requests", [1, 2])
@pytest.mark.parametrize("search_groups", [None, {"pessoa": {"fields": ["cpf", "nome"]}}])
def test_reprocess_columns_por_campo_chama_so_os_campos_pedidos(parallel_requests, search_groups):
    class Modelo(BaseModel):
        tipo: Optional[str] = None
        cpf: Optional[str] = Field(
            None, json_schema_extra={"condition": {"field": "tipo", "equals": "pf"}}
        )
        nome: Optional[str] = None

    df = pd.DataFrame(
        {
            "texto": ["a", "b"],
            "tipo": ["pf", "pj"],
            "cpf": ["111", None],
            "nome": ["Ana", "Beta"],
            "_dataframeit_status": ["processed", "processed"],
        }
    )
    chamadas = []

    def call_agent(text, model, prompt, config, save_trace=None):
        chamadas.append((text, list(model.model_fields)))
        return {"data": {campo: "novo" for campo in model.model_fields}, "usage": {}}

    provider, busca = _patches_de_execucao()
    with provider, busca, patch("dataframeit.agent.call_agent", side_effect=call_agent):
        resultado = dataframeit(
            df,
            questions=Modelo,
            prompt="Analise {texto}",
            use_search=True,
            search_per_field=True,
            reprocess_columns=["cpf"],
            parallel_requests=parallel_requests,
            search_groups=search_groups,
        )

    # A condição usa o `tipo` já gravado na linha; só `cpf` da linha pf é pedido
    assert sorted(chamadas) == [("a", ["cpf"])]
    # pandas 3 guarda a coluna como string, e o vazio volta como NaN
    assert resultado["cpf"].iloc[0] == "novo"
    assert pd.isna(resultado["cpf"].iloc[1])
    assert resultado["nome"].tolist() == ["Ana", "Beta"]


def test_resolve_forward_refs_preserva_literal_e_a_forma_da_uniao():
    import types
    import typing
    from typing import Literal, Union

    from dataframeit.utils import resolve_forward_refs

    class Folha(BaseModel):
        x: int

    class Dono(BaseModel):
        a: Optional[list["Folha"]] = None

    assert resolve_forward_refs(Literal["Folha", "Dono"], Dono) == Literal["Folha", "Dono"]
    uniao_typing = resolve_forward_refs(Union[list["Folha"], None], Dono)
    assert typing.get_origin(uniao_typing) is typing.Union
    assert typing.get_args(uniao_typing)[0] == list[Folha]
    uniao_pipe = resolve_forward_refs(list["Dono"] | None, Dono)
    assert isinstance(uniao_pipe, types.UnionType)
    assert typing.get_args(uniao_pipe)[0] == list[Dono]
    inexistente = types.GenericAlias(list, ("Inexistente",))
    assert resolve_forward_refs(inexistente, Dono) == inexistente


# =============================================================================
# Caminhos que a primeira versão dos testes não cobria
# =============================================================================


def test_campo_isolado_no_modo_por_grupo_com_condition_callable():
    from dataframeit.agent import call_agent_per_group

    class Modelo(BaseModel):
        tem_multa: bool
        orgao: Optional[str] = None
        valor_multa: Optional[float] = Field(
            default=None,
            json_schema_extra={
                "condition": lambda dados: dados.get("tem_multa") is True,
                "depends_on": ["tem_multa"],
            },
        )

    schemas = []
    valores = {**_VALORES_MULTA, "orgao": "Procon"}
    config = _config(groups={"g": SearchGroupConfig(fields=["tem_multa", "orgao"])})
    falso = _call_agent_que_gera_schema(valores, schemas)
    with patch("dataframeit.agent.call_agent", side_effect=falso):
        resultado = call_agent_per_group("texto", Modelo, "Analise {texto}", config)

    assert resultado["data"] == valores
    assert all(not _chaves_da_biblioteca_no_schema(s) for s in schemas)


def _propriedades_do_campo(schema) -> dict:
    """Propriedades de primeiro nível do schema, sem o $defs dos aninhados."""
    return {nome: prop for nome, prop in schema.get("properties", {}).items()}


def test_item_de_lista_e_busca_aninhada_mandam_o_campo_limpo():
    from dataframeit.agent import call_agent_per_field

    class Item(BaseModel):
        nome: str
        status: Optional[str] = Field(None, json_schema_extra={"prompt_append": "Busque o status."})

    class Interno(BaseModel):
        valor: Optional[str] = Field(None, json_schema_extra={"search_depth": "advanced"})

    class Modelo(BaseModel):
        itens: list[Item] = []
        interno: Optional[Interno] = None

    schemas_por_modelo = {}

    def falso(text, model, prompt, config, save_trace=None):
        schemas_por_modelo[model.__name__] = model.model_json_schema()
        respostas = {"itens": [{"nome": "a"}], "status": "ok", "valor": "v", "interno": None}
        return {"data": {c: respostas.get(c) for c in model.model_fields}, "usage": {}}

    with patch("dataframeit.agent.call_agent", side_effect=falso):
        call_agent_per_field("t", Modelo, "Analise {texto}", _config())

    item = next(s for n, s in schemas_por_modelo.items() if n.startswith("ItemSearch"))
    aninhado = next(s for n, s in schemas_por_modelo.items() if n.startswith("NestedSearch"))
    assert not _chaves_da_biblioteca_no_schema(_propriedades_do_campo(item))
    assert not _chaves_da_biblioteca_no_schema(_propriedades_do_campo(aninhado))


def test_reprocess_columns_nao_repete_busca_aninhada_de_campo_nao_pedido():
    from dataframeit.agent import call_agent_per_field

    class Interno(BaseModel):
        valor: Optional[str] = Field(None, json_schema_extra={"prompt_append": "x"})

    class Modelo(BaseModel):
        interno: Optional[Interno] = None
        nome: Optional[str] = None

    modelos = []

    def falso(text, model, prompt, config, save_trace=None):
        modelos.append(model.__name__)
        return {"data": {c: "v" for c in model.model_fields}, "usage": {}}

    with patch("dataframeit.agent.call_agent", side_effect=falso):
        call_agent_per_field(
            "t",
            Modelo,
            "Analise {texto}",
            _config(),
            only_fields={"nome"},
            known={"interno": {"valor": "antigo"}},
        )

    assert not any(nome.startswith("NestedSearch") for nome in modelos)


def test_reprocess_columns_passa_vazio_como_none_para_a_condicao():
    class Modelo(BaseModel):
        cpf: Optional[str] = None
        confirmado: Optional[str] = Field(
            None, json_schema_extra={"condition": {"field": "cpf", "exists": True}}
        )

    df = pd.DataFrame(
        {
            "texto": ["a"],
            "cpf": [float("nan")],
            "confirmado": ["antigo"],
            "_dataframeit_status": ["processed"],
        }
    )
    provider, busca = _patches_de_execucao()
    with provider, busca, patch("dataframeit.agent.call_agent") as call_agent:
        resultado = dataframeit(
            df,
            questions=Modelo,
            prompt="Analise {texto}",
            use_search=True,
            search_per_field=True,
            reprocess_columns=["confirmado"],
        )

    # cpf vazio: a condição `exists` é falsa e nada é pedido
    call_agent.assert_not_called()
    assert pd.isna(resultado["confirmado"].iloc[0])


def test_lista_com_referencia_adiantada_no_modo_por_campo():
    from dataframeit.agent import call_agent_per_field

    class Item(BaseModel):
        nome: str
        status: Optional[str] = Field(None, json_schema_extra={"prompt_append": "Busque."})

    class Modelo(BaseModel):
        itens: list["Item"] = []

    modelos = []

    def falso(text, model, prompt, config, save_trace=None):
        model.model_json_schema()
        modelos.append(model.__name__)
        respostas = {"itens": [{"nome": "a"}], "status": "ok"}
        return {"data": {c: respostas.get(c) for c in model.model_fields}, "usage": {}}

    with patch("dataframeit.agent.call_agent", side_effect=falso):
        resultado = call_agent_per_field("t", Modelo, "Analise {texto}", _config())

    assert any(nome.startswith("ItemSearch") for nome in modelos)
    assert resultado["data"]["itens"][0]["status"] == "ok"


def test_condition_em_lista_com_referencia_adiantada_levanta_erro():
    class Item(BaseModel):
        tipo: str
        valor: Optional[str] = Field(
            None, json_schema_extra={"condition": {"field": "tipo", "equals": "a"}}
        )

    class Modelo(BaseModel):
        itens: list["Item"] = []

    with pytest.raises(ValueError, match="itens.valor"):
        _executar(Modelo, use_search=True, search_per_field=True)


def test_configuracao_em_lista_de_listas_levanta_erro():
    class Item(BaseModel):
        x: Optional[str] = Field(None, json_schema_extra={"prompt_append": "Busque x."})

    class Modelo(BaseModel):
        matriz: list[list[Item]] = []

    with pytest.raises(ValueError, match="matriz.x"):
        _executar(Modelo, use_search=True, search_per_field=True)


def test_max_results_booleano_levanta_erro():
    class Modelo(BaseModel):
        campo: Optional[str] = Field(None, json_schema_extra={"max_results": True})

    with pytest.raises(ValueError, match="max_results"):
        _executar(Modelo, use_search=True, search_per_field=True)


def test_mesmo_modelo_em_dois_campos_e_coletado_nos_dois():
    from dataframeit.conditional import _collect_configured_fields

    class Endereco2(BaseModel):
        cidade: Optional[str] = Field(None, json_schema_extra={"prompt_append": "x"})

    class Pessoa(BaseModel):
        residencial: Optional[Endereco2] = None
        comercial: Optional[Endereco2] = None

    caminhos = [path for path, *_ in _collect_configured_fields(Pessoa)]
    assert caminhos == ["residencial.cidade", "comercial.cidade"]


def test_lista_de_outro_tipo_na_uniao_nao_conta_como_camada():
    class End(BaseModel):
        cidade: Optional[str] = Field(None, json_schema_extra={"prompt_append": "x"})

    class Parte(BaseModel):
        endereco: Union[list[str], End, None] = None

    class Processo(BaseModel):
        partes: list[Parte] = []

    from dataframeit.conditional import _walk_fields

    profundidades = {path: depth for path, _, depth in _walk_fields(Processo)}
    assert profundidades["partes.endereco.cidade"] == 1


def test_llm_field_limpa_attributes_set_sem_mexer_no_original():
    from dataframeit.agent import _llm_field

    class Modelo(BaseModel):
        a: Optional[str] = Field(None, json_schema_extra={"condition": {"field": "b", "equals": 1}})

    original = Modelo.model_fields["a"]
    limpo = _llm_field(original)
    assert "json_schema_extra" not in limpo._attributes_set
    assert original._attributes_set["json_schema_extra"] == {
        "condition": {"field": "b", "equals": 1}
    }


def test_reprocess_columns_em_grupo_nao_pede_campo_de_condicao_falsa():
    class Modelo(BaseModel):
        tipo: Optional[str] = None
        cpf: Optional[str] = Field(
            None, json_schema_extra={"condition": {"field": "tipo", "equals": "pf"}}
        )
        nome: Optional[str] = None

    df = pd.DataFrame(
        {
            "texto": ["a", "b", "c"],
            "tipo": ["pf", "pj", None],
            "cpf": ["111", None, None],
            "nome": ["Ana", "Beta", None],
            "_dataframeit_status": ["processed", "processed", None],
        }
    )
    chamadas = []

    def call_agent(text, model, prompt, config, save_trace=None):
        chamadas.append((text, list(model.model_fields)))
        valores = {"tipo": "pf", "cpf": "novo", "nome": "Nome"}
        return {"data": {campo: valores[campo] for campo in model.model_fields}, "usage": {}}

    provider, busca = _patches_de_execucao()
    with provider, busca, patch("dataframeit.agent.call_agent", side_effect=call_agent):
        dataframeit(
            df,
            questions=Modelo,
            prompt="Analise {texto}",
            use_search=True,
            search_per_field=True,
            reprocess_columns=["cpf"],
            search_groups={"g": {"fields": ["tipo", "cpf"]}},
        )

    # A linha pj não pede cpf: `tipo` vem da linha gravada e não da mesma chamada
    assert ("b", ["cpf"]) not in chamadas
    assert ("a", ["cpf"]) in chamadas
    assert ("c", ["tipo", "cpf"]) in chamadas
