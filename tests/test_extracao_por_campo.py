"""Contrato da extração por campo e por grupo com busca, com o agente simulado."""

import logging
from typing import Annotated, Optional
from unittest.mock import patch

import pytest
from pydantic import BaseModel, Field, create_model
from typing_extensions import override

from dataframeit import field_condition
from dataframeit.agent import field_extractor
from dataframeit.conditional import _walk_fields
from dataframeit.llm import LLMConfig, SearchConfig, SearchGroupConfig

PROMPT = "Analise {texto}"


def _config(grupos=None, provider="tavily"):
    return LLMConfig(
        model="m",
        provider="p",
        api_key=None,
        max_retries=1,
        base_delay=0.0,
        max_delay=0.0,
        rate_limit_delay=0,
        search_config=SearchConfig(enabled=True, provider=provider, per_field=True, groups=grupos),
    )


def _extrair(modelo, config, texto="t", save_trace=None, **kwargs):
    return field_extractor(modelo, PROMPT, config, save_trace)(texto, **kwargs)


class _Agente:
    """call_agent falso: responde cada campo pedido por `respostas` e registra a chamada."""

    def __init__(self, respostas, usage=None, trace=False):
        self.respostas = respostas
        self.usage = usage or {}
        self.trace = trace
        self.chamadas = []

    def __call__(self, text, model, prompt, config, save_trace=None):
        self.chamadas.append((model.__name__, prompt))
        resultado = {
            "data": {campo: self.respostas.get(campo) for campo in model.model_fields},
            "usage": dict(self.usage),
        }
        if self.trace:
            resultado["trace"] = {"modelo": model.__name__}
        return resultado

    @property
    def nomes(self):
        return [nome for nome, _ in self.chamadas]

    def prompt_de(self, nome):
        return dict(self.chamadas)[nome]


def _rodar(agente, modelo, config, **kwargs):
    with patch("dataframeit.agent.call_agent", side_effect=agente):
        return _extrair(modelo, config, **kwargs)


class Interno(BaseModel):
    valor: Optional[str] = Field(None, json_schema_extra={"prompt": "Busque o valor"})
    outro: Optional[str] = Field(None, json_schema_extra={"prompt": "Busque o outro"})


class Item(BaseModel):
    nome: str
    preco: Optional[str] = Field(None, json_schema_extra={"prompt": "Busque o preço"})


class Completo(BaseModel):
    a: Optional[str] = None
    interno: Optional[Interno] = None
    itens: list[Item] = []


RESPOSTAS = {
    "a": "1",
    "interno": {"valor": None, "outro": None},
    "itens": [{"nome": "x"}],
    "valor": "v",
    "outro": "w",
    "preco": "R$ 10",
}


# =============================================================================
# usage
# =============================================================================


@pytest.mark.parametrize("grupos", [None, {"g": SearchGroupConfig(fields=["a"])}])
def test_usage_traz_o_provedor_de_busca(grupos):
    resultado = _rodar(_Agente({"a": "1", "itens": []}), Completo, _config(grupos, provider="exa"))
    assert resultado["usage"]["search_provider"] == "exa"


def test_usage_soma_contador_ausente_como_zero():
    """Busca aninhada, campos e itens somam só o que cada chamada informa."""
    agente = _Agente(RESPOSTAS, usage={"input_tokens": 2})

    resultado = _rodar(agente, Completo, _config())

    # aninhadas (2), campos (3) e um item enriquecido (1)
    assert len(agente.chamadas) == 6
    assert resultado["usage"]["input_tokens"] == 12
    assert resultado["usage"]["output_tokens"] == 0
    assert resultado["usage"]["search_credits"] == 0


def test_usage_soma_contador_ausente_como_zero_no_modo_por_grupo():
    class Simples(BaseModel):
        a: Optional[str] = None
        b: Optional[str] = None
        c: Optional[str] = None

    agente = _Agente({}, usage={"input_tokens": 2})

    resultado = _rodar(agente, Simples, _config({"g": SearchGroupConfig(fields=["a", "b"])}))

    assert resultado["usage"]["input_tokens"] == 4
    assert resultado["usage"]["output_tokens"] == 0


class _AgenteQueFalha(_Agente):
    """Levanta `erro` na chamada ao modelo `falha`, depois de registrá-la."""

    def __init__(self, falha, erro):
        super().__init__(RESPOSTAS, usage={"input_tokens": 2})
        self.falha = falha
        self.erro = erro

    @override
    def __call__(self, text, model, prompt, config, save_trace=None):
        resultado = super().__call__(text, model, prompt, config, save_trace)
        if model.__name__ == self.falha:
            raise self.erro
        return resultado


@pytest.mark.parametrize(
    "falha",
    ["NestedSearch_interno_outro", "Completo_interno", "ItemSearch_0_preco"],
)
def test_erro_da_linha_leva_o_uso_das_chamadas_ja_feitas(falha):
    agente = _AgenteQueFalha(falha, RuntimeError("caiu"))

    with pytest.raises(RuntimeError, match="caiu") as erro:
        _rodar(agente, Completo, _config())

    assert agente.nomes[-1] == falha
    # Todas as chamadas antes da que falhou, inclusive a do campo da lista enriquecida.
    assert erro.value.row_usage["input_tokens"] == 2 * (len(agente.chamadas) - 1)  # ty: ignore[unresolved-attribute]


class _ErroSemAtributos(Exception):
    @override
    def __setattr__(self, name, value):
        raise AttributeError(name)


def test_erro_que_nao_aceita_atributo_segue_sem_o_uso():
    agente = _AgenteQueFalha("Completo_a", _ErroSemAtributos("caiu"))

    with pytest.raises(_ErroSemAtributos, match="caiu") as erro:
        _rodar(agente, Completo, _config())

    assert not hasattr(erro.value, "row_usage")


# =============================================================================
# prompts
# =============================================================================


class ComDescricao(BaseModel):
    a: Optional[str] = None
    b: Optional[str] = Field(None, description="o segundo")


@pytest.mark.parametrize("grupos", [None, {"g": SearchGroupConfig(fields=["a"])}])
def test_prompt_do_campo_leva_prompt_do_usuario_nome_e_descricao(grupos):
    agente = _Agente({})

    _rodar(agente, ComDescricao, _config(grupos))

    assert agente.prompt_de("ComDescricao_b") == (
        "Analise {texto}\n\nResponda APENAS o campo: b (o segundo)"
    )


def test_contexto_das_buscas_aninhadas_entra_no_prompt_do_campo():
    agente = _Agente(RESPOSTAS)

    _rodar(agente, Completo, _config())

    assert agente.prompt_de("Completo_interno") == (
        "Analise {texto}\n\nResponda APENAS o campo: interno"
        "\n\nContexto de buscas realizadas para campos aninhados:"
        "\n- interno.valor: v\n- interno.outro: w"
    )
    assert agente.prompt_de("Completo_a") == "Analise {texto}\n\nResponda APENAS o campo: a"


def test_prompt_padrao_do_grupo_lista_os_campos():
    agente = _Agente({})

    _rodar(agente, ComDescricao, _config({"g": SearchGroupConfig(fields=["a", "b"])}))

    assert agente.prompt_de("ComDescricao_group_g") == (
        "Analise {texto}\n\nResponda os campos: a, b"
    )


def test_query_no_prompt_do_grupo_vira_texto():
    agente = _Agente({})
    grupo = SearchGroupConfig(fields=["a", "b"], prompt="Busque {query} agora")

    _rodar(agente, ComDescricao, _config({"g": grupo}))

    assert agente.prompt_de("ComDescricao_group_g") == "Busque {texto} agora"


# =============================================================================
# reprocess_columns
# =============================================================================


def test_busca_aninhada_so_roda_para_campo_pedido():
    agente = _Agente(RESPOSTAS)
    _rodar(agente, Completo, _config(), only_fields={"interno"}, known={"a": "antigo"})
    assert agente.nomes == [
        "NestedSearch_interno_valor",
        "NestedSearch_interno_outro",
        "Completo_interno",
    ]

    agente = _Agente(RESPOSTAS)
    resultado = _rodar(agente, Completo, _config(), only_fields={"a"}, known={})
    assert agente.nomes == ["Completo_a"]
    assert resultado["data"]["itens"] is None


# =============================================================================
# condições
# =============================================================================


class ComCondicao(BaseModel):
    tipo: Optional[str] = None
    pulado: Optional[str] = Field(
        None, json_schema_extra={"condition": {"field": "tipo", "equals": "pj"}}
    )
    # Também depende de `tipo`, para rodar depois de `pulado` na ordem de dependências.
    depois: Optional[str] = Field(
        None, json_schema_extra={"condition": {"field": "tipo", "equals": "pf"}}
    )


@pytest.mark.parametrize("grupos", [None, {"g": SearchGroupConfig(fields=["tipo"])}])
def test_campo_pulado_fica_none_e_os_seguintes_rodam(grupos, caplog):
    agente = _Agente({"tipo": "pf", "pulado": "x", "depois": "d"})

    with caplog.at_level(logging.INFO, logger="dataframeit"):
        resultado = _rodar(agente, ComCondicao, _config(grupos))

    assert resultado["data"]["pulado"] is None
    assert resultado["data"]["depois"] == "d"
    assert "ComCondicao_pulado" not in agente.nomes
    assert caplog.messages.count("Campo 'pulado' pulado (condição não satisfeita)") == 1


class DependeDoSeguinte(BaseModel):
    b: Optional[str] = Field(None, json_schema_extra={"condition": {"field": "c", "equals": "sim"}})
    c: Optional[str] = None


@pytest.mark.parametrize("grupos", [None, {"g": SearchGroupConfig(fields=["c"])}])
def test_dados_saem_na_ordem_do_modelo_e_as_chamadas_na_das_dependencias(grupos):
    agente = _Agente({"b": "x", "c": "sim"})

    resultado = _rodar(agente, DependeDoSeguinte, _config(grupos))

    assert list(resultado["data"]) == ["b", "c"]
    assert agente.nomes[-1] == "DependeDoSeguinte_b"


class ListaCondicionada(BaseModel):
    a: Optional[str] = None
    itens: list[Item] = Field([], json_schema_extra={"condition": {"field": "a", "equals": "sim"}})


@pytest.mark.parametrize(("a", "buscas_por_item"), [("sim", 1), ("nao", 0)])
def test_lista_anulada_depois_da_resposta_do_grupo_nao_e_enriquecida(a, buscas_por_item):
    agente = _Agente({"a": a, "itens": [{"nome": "x"}], "preco": "R$ 10"})

    resultado = _rodar(
        agente, ListaCondicionada, _config({"g": SearchGroupConfig(fields=["a", "itens"])})
    )

    assert agente.nomes.count("ItemSearch_0_preco") == buscas_por_item
    assert (resultado["data"]["itens"] is None) is (buscas_por_item == 0)


class Sub(BaseModel):
    tipo: Optional[str] = None


def _a_tipo_x(dados):
    return (dados.get("a") or {}).get("tipo") == "x"


class DependeDeCaminho(BaseModel):
    a: Optional[Sub] = None
    # depends_on explícito guarda o caminho inteiro; a condição dict guardaria só a raiz.
    b: Optional[str] = Field(
        None, json_schema_extra=field_condition(_a_tipo_x, depends_on=["a.tipo"])
    )


@pytest.mark.parametrize(("tipo", "esperado"), [("x", "valor"), ("y", None)])
def test_condicao_sobre_caminho_de_campo_do_mesmo_grupo_roda_depois_da_resposta(
    tipo, esperado, caplog
):
    agente = _Agente({"a": {"tipo": tipo}, "b": "valor"})

    with caplog.at_level(logging.INFO, logger="dataframeit"):
        resultado = _rodar(
            agente, DependeDeCaminho, _config({"g": SearchGroupConfig(fields=["a", "b"])})
        )

    assert agente.nomes == ["DependeDeCaminho_group_g"]
    assert resultado["data"]["b"] == esperado
    pulado = "Campo 'b' pulado (condição não satisfeita)" in caplog.messages
    assert pulado is (esperado is None)


class GrupoComCondicaoDeFora(BaseModel):
    tipo: Optional[str] = None
    a: Optional[str] = Field(
        None, json_schema_extra={"condition": {"field": "tipo", "equals": "pj"}}
    )
    b: Optional[str] = None


def test_campo_do_grupo_pulado_antes_da_chamada_registra_o_nome(caplog):
    agente = _Agente({"tipo": "pf", "a": "x", "b": "y"})

    with caplog.at_level(logging.INFO, logger="dataframeit"):
        resultado = _rodar(
            agente, GrupoComCondicaoDeFora, _config({"g": SearchGroupConfig(fields=["a", "b"])})
        )

    assert resultado["data"]["a"] is None
    assert "Campo 'a' pulado (condição não satisfeita)" in caplog.messages


# =============================================================================
# modelos e traces
# =============================================================================


class ComReferenciaAdiantada(BaseModel):
    a: Optional[str] = None
    # A string fica crua dentro de list[...], e só o modelo dono a resolve.
    itens: list["ItemAdiantado"] = []
    outros: list["ItemAdiantado"] = []


class ItemAdiantado(BaseModel):
    nome: str


ComReferenciaAdiantada.model_rebuild()


def test_referencia_adiantada_no_grupo_e_no_campo_isolado():
    agente = _Agente({"itens": [{"nome": "x"}], "outros": [{"nome": "y"}]})

    resultado = _rodar(
        agente,
        ComReferenciaAdiantada,
        _config({"g": SearchGroupConfig(fields=["a", "itens"])}),
    )

    assert resultado["data"]["itens"] == [{"nome": "x"}]
    assert resultado["data"]["outros"] == [{"nome": "y"}]


def test_trace_do_grupo_e_o_da_chamada():
    agente = _Agente({}, trace=True)

    resultado = _rodar(
        agente,
        ComDescricao,
        _config({"g": SearchGroupConfig(fields=["a", "b"])}),
        save_trace="full",
    )

    assert resultado["traces"] == {"g": {"modelo": "ComDescricao_group_g"}}


class CadeiaNoGrupo(BaseModel):
    a: Optional[str] = None
    b: Optional[str] = Field(None, json_schema_extra={"condition": {"field": "a", "equals": "sim"}})
    c: Optional[str] = Field(None, json_schema_extra={"condition": {"field": "b", "exists": True}})


def test_campo_anulado_depois_da_resposta_anula_quem_depende_dele_no_grupo():
    """As condições pós-resposta seguem a ordem de dependência, e não a do grupo."""
    agente = _Agente({"a": "nao", "b": "B", "c": "C"})

    resultado = _rodar(
        agente, CadeiaNoGrupo, _config({"g": SearchGroupConfig(fields=["c", "b", "a"])})
    )

    assert resultado["data"] == {"a": "nao", "b": None, "c": None}


# =============================================================================
# lista de modelo
# =============================================================================


@pytest.mark.parametrize(
    "anotacao",
    [
        list[Item],
        Optional[list[Item]],
        Optional[Annotated[list[Item], Field(min_length=1)]],
        Annotated[list[Item], "lista"] | None,
        list[Optional[Item]],
        list[Item | None],
        list[Annotated[Item, "item"]],
        list[Optional[Annotated[Item, "item"]]],
    ],
    ids=[
        "lista",
        "lista_opcional",
        "lista_anotada_opcional",
        "lista_anotada_ou_none",
        "item_opcional",
        "item_ou_none",
        "item_anotado",
        "item_anotado_opcional",
    ],
)
def test_lista_de_modelo_e_buscada_item_a_item_e_conta_uma_camada_na_validacao(anotacao):
    """O Optional e o Annotated em volta da lista ou do item não tiram a busca por item.

    A validação conta a mesma camada de lista que a extração percorre: com uma
    camada, a configuração do item é aceita, e cada item tem a sua busca.
    """
    modelo = create_model("Pedido", itens=(anotacao, None))
    agente = _Agente({"itens": [{"nome": "A"}, {"nome": "B"}], "preco": "R$ 1"})

    resultado = _rodar(agente, modelo, _config())

    assert {caminho: camadas for caminho, _, camadas in _walk_fields(modelo)} == {
        "itens": 0,
        "itens.nome": 1,
        "itens.preco": 1,
    }
    assert agente.nomes == ["Pedido_itens", "ItemSearch_0_preco", "ItemSearch_1_preco"]
    assert resultado["data"]["itens"] == [
        {"nome": "A", "preco": "R$ 1"},
        {"nome": "B", "preco": "R$ 1"},
    ]


class _Livro(BaseModel):
    titulo: str
    isbn: Optional[str] = Field(None, json_schema_extra={"prompt": "Busque o ISBN"})


class _Filme(BaseModel):
    nome: str
    diretor: Optional[str] = Field(None, json_schema_extra={"prompt": "Busque o diretor"})


@pytest.mark.parametrize(
    "anotacao",
    [
        list[_Livro | _Filme],
        Optional[list[Annotated[_Livro | _Filme, "obra"]]],
        list[_Livro] | list[_Filme],
    ],
    ids=["item_uniao", "item_uniao_anotado_opcional", "uniao_de_listas"],
)
def test_lista_de_mais_de_um_modelo_busca_os_campos_de_cada_modelo(anotacao):
    """Sem um modelo só para o item, cada campo de busca é buscado uma vez para a linha.

    O ISBN não é aplicado ao filme, e o diretor não fica sem busca.
    """
    modelo = create_model("Obras", obras=(anotacao, None))
    agente = _Agente(
        {
            "obras": [{"titulo": "Dom Casmurro", "isbn": None}],
            "isbn": "978",
            "diretor": "Walter Salles",
        }
    )

    resultado = _rodar(agente, modelo, _config())

    assert {caminho: camadas for caminho, _, camadas in _walk_fields(modelo)} == {
        "obras": 0,
        "obras.titulo": 1,
        "obras.isbn": 1,
        "obras.nome": 1,
        "obras.diretor": 1,
    }
    assert agente.nomes == ["NestedSearch_obras_isbn", "NestedSearch_obras_diretor", "Obras_obras"]
    assert resultado["data"]["obras"] == [{"titulo": "Dom Casmurro", "isbn": None}]


@pytest.mark.parametrize(
    "resposta",
    [[{"titulo": "Dom Casmurro", "isbn": None}], {"nome": "Central do Brasil", "diretor": None}],
    ids=["lista", "modelo"],
)
def test_lista_ou_modelo_busca_os_campos_de_cada_modelo(resposta):
    """Com o modelo também fora da lista, a busca é a da linha, como a do item de mais de um modelo.

    O valor que o agente devolve fica como veio, seja a lista de livros, seja o
    filme, e o diretor do filme é buscado.
    """
    modelo = create_model("Obras", obras=(Optional[list[_Livro] | _Filme], None))
    agente = _Agente({"obras": resposta, "isbn": "978", "diretor": "Walter Salles"})

    resultado = _rodar(agente, modelo, _config())

    assert agente.nomes == ["NestedSearch_obras_isbn", "NestedSearch_obras_diretor", "Obras_obras"]
    assert resultado["data"]["obras"] == resposta


class _Erro(BaseModel):
    motivo: str


@pytest.mark.parametrize(
    ("resposta", "nomes", "esperado"),
    [
        (
            [{"nome": "A"}],
            ["Pedido_itens", "ItemSearch_0_preco"],
            [{"nome": "A", "preco": "R$ 1"}],
        ),
        ({"motivo": "ilegível"}, ["Pedido_itens"], {"motivo": "ilegível"}),
    ],
    ids=["lista", "modelo"],
)
def test_lista_ou_modelo_sem_campo_de_busca_mantem_a_busca_por_item(resposta, nomes, esperado):
    """O modelo de fora da lista sem campo de busca não tira a busca item a item da lista."""
    modelo = create_model("Pedido", itens=(Optional[list[Item] | _Erro], None))
    agente = _Agente({"itens": resposta, "preco": "R$ 1"})

    resultado = _rodar(agente, modelo, _config())

    assert agente.nomes == nomes
    assert resultado["data"]["itens"] == esperado


@pytest.mark.parametrize(
    "anotacao",
    [list[Item] | tuple[Item, ...], list[Item] | dict[str, Item]],
    ids=["tupla", "dict"],
)
def test_lista_ou_colecao_do_mesmo_modelo_busca_uma_vez_para_a_linha(anotacao):
    """O item que pode vir fora da lista também é buscado, uma vez para a linha."""
    modelo = create_model("Pedido", itens=(anotacao, None))
    agente = _Agente({"itens": [{"nome": "A"}], "preco": "R$ 1"})

    _rodar(agente, modelo, _config())

    assert agente.nomes == ["NestedSearch_itens_preco", "Pedido_itens"]


def test_item_none_da_lista_fica_sem_busca():
    """O item None de list[Item | None] não tem campo onde gravar a busca; o outro tem."""
    modelo = create_model("Pedido", itens=(list[Item | None], None))
    agente = _Agente({"itens": [None, {"nome": "B"}], "preco": "R$ 1"})

    resultado = _rodar(agente, modelo, _config())

    assert agente.nomes == ["Pedido_itens", "ItemSearch_1_preco"]
    assert resultado["data"]["itens"] == [None, {"nome": "B", "preco": "R$ 1"}]


def test_lista_de_modelo_ou_texto_guarda_o_texto_sem_buscar_item():
    """O texto devolvido no lugar da lista não é percorrido como se fossem itens."""
    modelo = create_model("Pedido", itens=(Optional[list[Item] | str], None))
    agente = _Agente({"itens": "nenhum", "preco": "R$ 1"})

    resultado = _rodar(agente, modelo, _config())

    assert agente.nomes == ["Pedido_itens"]
    assert resultado["data"]["itens"] == "nenhum"
