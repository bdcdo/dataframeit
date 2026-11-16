"""Submodelo declarado por NewType ou apelido `type` conta como o modelo a que se refere.

Os helpers que percorrem os modelos aninhados (`get_nested_pydantic_models`,
`is_list_of_pydantic_model`, `list_layers`, `models_outside_lists` e
`submodels`) resolvem os dois a cada nível, e a validação da configuração por
campo vê o submodelo como o vê na anotação direta.
"""

import sys
from typing import (  # noqa: UP035 (o `List["X"]` guarda um ForwardRef)
    Annotated,
    Any,
    ForwardRef,
    List,
    Literal,
    NewType,
    Optional,
    TypeVar,
    Union,
)

import pytest
from pydantic import BaseModel, Field, create_model
from typing_extensions import TypeAliasType

from dataframeit.agent import _get_list_fields_with_nested_search
from dataframeit.conditional import _collect_configured_fields, _walk_fields
from dataframeit.core import _has_unstable_extra, _validate_field_configs
from dataframeit.llm import SearchConfig
from dataframeit.utils import (
    _resolved,
    get_nested_pydantic_models,
    is_list_of_pydantic_model,
    list_layers,
    models_outside_lists,
    submodels,
)


class _Item(BaseModel):
    valor: str = Field(json_schema_extra={"prompt": "busque o valor"})


# O conjunto não é JSON, e a anotação de `json_schema_extra` não o aceita.
_EXTRA_COM_CONJUNTO: Any = {"rotulos": {"a", "b"}}


class _ItemComConjunto(BaseModel):
    valor: str = Field(json_schema_extra=_EXTRA_COM_CONJUNTO)


class _ItemDependente(BaseModel):
    valor: str = Field(json_schema_extra={"depends_on": ["texto"]})


_T = TypeVar("_T")

_NovoItem = NewType("_NovoItem", _Item)
_NovaLista = NewType("_NovaLista", list[_Item])
_ApelidoDeItem = TypeAliasType("_ApelidoDeItem", _Item)
_ApelidoDeLista = TypeAliasType("_ApelidoDeLista", list[_Item])
_ListaGenerica = TypeAliasType("_ListaGenerica", list[_T], type_params=(_T,))
_ApelidoAnotado = TypeAliasType("_ApelidoAnotado", Annotated[list[_Item], "itens"])

# A referência adiantada em texto dentro do valor do apelido: no item, no valor
# inteiro, no `ForwardRef` que o `List` de `typing` guarda e ao lado do parâmetro.
_ApelidoComTexto = TypeAliasType("_ApelidoComTexto", list["_Item"])
_ApelidoInteiroEmTexto = TypeAliasType("_ApelidoInteiroEmTexto", "list[_Item]")
_ApelidoComForwardRef = TypeAliasType("_ApelidoComForwardRef", List["_Item"])  # noqa: UP006
_GenericoEmTexto = TypeAliasType("_GenericoEmTexto", "list[_T] | _Item", type_params=(_T,))
_AnotadoComTexto = TypeAliasType("_AnotadoComTexto", Annotated[list["_Item"], "_Item"])

# Por `Any`, o checador de tipos não toma por declaração estática o NewType, o
# apelido e as anotações que os testes montam para cada modelo.
_newtype: Any = NewType
_apelido: Any = TypeAliasType
_typevar: Any = TypeVar
_uniao: Any = Union


def _lista(item: Any) -> Any:
    return list[item]


def _anotado(tipo: Any) -> Any:
    return Annotated[tipo, "x"]


def _opcional(tipo: Any) -> Any:
    return Optional[tipo]


# Cada forma de declarar `list[X]`: a direta, que já valia, e as que passam
# pelo NewType ou pelo apelido, na lista ou no item.
_FORMAS_DE_LISTA = {
    "direta": _lista,
    "newtype-da-lista": lambda item: _newtype("Lista", _lista(item)),
    "apelido-da-lista": lambda item: _apelido("Lista", _lista(item)),
    "apelido-generico": lambda item: _ListaGenerica[item],
    "lista-de-newtype": lambda item: _lista(_newtype("Novo", item)),
    "lista-de-apelido": lambda item: _lista(_apelido("Apelido", item)),
    "annotated-de-apelido": lambda item: _anotado(_apelido("Lista", _lista(item))),
    "optional-de-newtype": lambda item: _opcional(_newtype("Lista", _lista(item))),
}
if sys.version_info >= (3, 12):

    def _apelido_da_linguagem(item):
        escopo = {"item": item}
        exec("type Lista = list[item]", escopo)  # noqa: S102 (sintaxe que o 3.11 não compila)
        return escopo["Lista"]

    _FORMAS_DE_LISTA["type-da-linguagem"] = _apelido_da_linguagem

_forma_de_lista = pytest.mark.parametrize(
    "forma", _FORMAS_DE_LISTA.values(), ids=_FORMAS_DE_LISTA.keys()
)


@pytest.mark.parametrize(
    ("anotacao", "e_lista", "modelos", "camadas"),
    [
        (_NovaLista, (True, _Item), [_Item], 1),
        (_ApelidoDeLista, (True, _Item), [_Item], 1),
        (_ListaGenerica[_Item], (True, _Item), [_Item], 1),
        (_ApelidoAnotado, (True, _Item), [_Item], 1),
        (Annotated[_NovaLista, "x"], (True, _Item), [_Item], 1),
        (Annotated[_ApelidoAnotado, "x"], (True, _Item), [_Item], 1),
        (_NovoItem, (False, None), [_Item], 0),
        (_ApelidoDeItem, (False, None), [_Item], 0),
        (list[_NovoItem], (True, _Item), [_Item], 1),
        (list[_ApelidoDeItem], (True, _Item), [_Item], 1),
        (list[_ApelidoDeLista], (False, None), [_Item], 2),
        (Optional[_ApelidoDeLista], (True, _Item), [_Item], 1),
        (_newtype("_NovoTexto", str), (False, None), [], None),
        (_ApelidoComTexto, (True, _Item), [_Item], 1),
        (_ApelidoInteiroEmTexto, (True, _Item), [_Item], 1),
        (_ApelidoComForwardRef, (True, _Item), [_Item], 1),
        (_AnotadoComTexto, (True, _Item), [_Item], 1),
        (Optional[_ApelidoComTexto], (True, _Item), [_Item], 1),
    ],
)
def test_helpers_resolvem_newtype_e_apelido_em_cada_nivel(anotacao, e_lista, modelos, camadas):
    assert is_list_of_pydantic_model(anotacao) == e_lista
    assert get_nested_pydantic_models(anotacao) == modelos
    assert list_layers(anotacao, _Item) == camadas


def test_modelo_fora_da_lista_e_achado_por_newtype_e_apelido():
    assert models_outside_lists(_NovoItem) == [_Item]
    assert models_outside_lists(_ApelidoDeItem) == [_Item]
    assert models_outside_lists(_ApelidoDeLista) == []


def test_caminho_segue_o_submodelo_por_newtype_e_apelido():
    assert submodels(_NovoItem) == [_Item]
    assert submodels(Optional[_ApelidoDeItem]) == [_Item]
    assert submodels(_ApelidoDeLista) == []


@_forma_de_lista
def test_prompt_no_item_exige_a_busca_por_campo(forma):
    modelo = create_model("Modelo", itens=(forma(_Item), ...))

    assert [path for path, *_ in _collect_configured_fields(modelo)] == ["itens.valor"]
    with pytest.raises(ValueError, match=r"requerem use_search=True e search_per_field=True"):
        _validate_field_configs(modelo, None, use_search=False, search_per_field=False)


@_forma_de_lista
def test_item_da_lista_recebe_a_busca(forma):
    modelo = create_model("Modelo", itens=(forma(_Item), ...))

    campos = _get_list_fields_with_nested_search(modelo)

    assert list(campos) == ["itens"]
    assert campos["itens"]["inner_model"] is _Item
    assert [path for path, *_ in campos["itens"]["search_fields"]] == ["valor"]


@_forma_de_lista
def test_depends_on_no_item_e_recusado(forma):
    modelo = create_model("Modelo", texto=(str, ...), itens=(forma(_ItemDependente), ...))

    assert [path for path, *_ in _walk_fields(modelo)] == ["texto", "itens", "itens.valor"]
    with pytest.raises(ValueError, match=r"'itens\.valor'.*só são aplicados a campos de primeiro"):
        _validate_field_configs(
            modelo,
            SearchConfig(enabled=True, per_field=True),
            use_search=True,
            search_per_field=True,
        )


@_forma_de_lista
def test_busca_em_lista_dentro_de_lista_e_recusada(forma):
    modelo = create_model("Modelo", itens=(_lista(forma(_Item)), ...))

    with pytest.raises(ValueError, match=r"'itens\.valor'.*lista que está dentro de outra lista"):
        _validate_field_configs(
            modelo,
            SearchConfig(enabled=True, per_field=True),
            use_search=True,
            search_per_field=True,
        )


@_forma_de_lista
def test_conjunto_no_extra_do_item_leva_a_assinatura_a_forma_canonica(forma):
    modelo = create_model("Modelo", itens=(forma(_ItemComConjunto), ...))

    assert _has_unstable_extra(modelo)


@pytest.mark.skipif(sys.version_info < (3, 12), reason="o apelido recursivo exige `type X = ...`")
def test_apelido_recursivo_nao_e_percorrido_sem_fim():
    escopo = {"_Item": _Item}
    exec(  # noqa: S102 (sintaxe que o 3.11 não compila)
        "type Json = dict[str, Json] | list[Json] | str | int | None\n"
        "type Arvore = list[Arvore] | _Item\n"
        "type Galho[T] = list[Galho[T]] | T",
        escopo,
    )
    galho = escopo["Galho"][_Item]
    modelo = create_model("Modelo", dados=(escopo["Json"], None), arvore=(escopo["Arvore"], ...))

    assert get_nested_pydantic_models(escopo["Json"]) == []
    assert list_layers(escopo["Json"], _Item) is None
    assert get_nested_pydantic_models(escopo["Arvore"]) == [_Item]
    # A árvore tem lista dentro de lista, e a busca no item dela é recusada.
    assert list_layers(escopo["Arvore"], _Item) == 2
    assert get_nested_pydantic_models(galho) == [_Item]
    assert list_layers(galho, _Item) == 2
    assert [path for path, *_ in _collect_configured_fields(modelo)] == ["arvore.valor"]


def test_prompt_no_item_de_apelido_com_referencia_em_texto_exige_a_busca_por_campo():
    modelo = create_model("Modelo", itens=(_ApelidoComTexto, ...))

    assert [path for path, *_ in _collect_configured_fields(modelo)] == ["itens.valor"]
    with pytest.raises(ValueError, match=r"requerem use_search=True e search_per_field=True"):
        _validate_field_configs(modelo, None, use_search=False, search_per_field=False)
    campos = _get_list_fields_with_nested_search(modelo)
    assert campos["itens"]["inner_model"] is _Item


def test_referencia_em_texto_do_apelido_e_trocada_pelo_tipo():
    assert _resolved(_ApelidoComTexto) == list[_Item]
    assert _resolved(_GenericoEmTexto[int]) == list[int] | _Item
    # O texto nos metadados do `Annotated` não é referência e fica como está.
    assert _resolved(_AnotadoComTexto) == Annotated[list[_Item], "_Item"]


def test_apelido_declarado_em_funcao_acha_o_proprio_nome_e_o_do_modulo():
    arvore = _apelido("Arvore", "list[Arvore] | _Item")

    assert get_nested_pydantic_models(arvore) == [_Item]
    assert list_layers(arvore, _Item) == 2


def test_referencia_em_texto_a_nome_de_dentro_da_funcao_e_recusada():
    class Local(BaseModel):
        valor: str = Field(json_schema_extra={"prompt": "busque o valor"})

    itens = _apelido("Itens", list["Local"])
    modelo = create_model("Modelo", itens=(itens, ...))

    with pytest.raises(
        TypeError,
        match=rf"apelido de tipo 'Itens'.*não foi encontrado no módulo '{__name__}': "
        rf"name '{Local.__name__}' is not defined\. Declare o apelido depois da classe e sem aspas",
    ):
        get_nested_pydantic_models(itens)
    with pytest.raises(TypeError, match=r"apelido de tipo 'Itens'"):
        _validate_field_configs(modelo, None, use_search=False, search_per_field=False)


def test_apelido_de_modulo_que_nao_esta_carregado_resolve_sem_o_modulo():
    escopo = {"__name__": "modulo_que_nao_existe", "apelido": TypeAliasType, "_Item": _Item}
    exec(  # noqa: S102 (o apelido toma o módulo do `__name__` de onde é criado)
        "Direto = apelido('Direto', list[_Item])\nEmTexto = apelido('EmTexto', list['_Item'])",
        escopo,
    )

    assert get_nested_pydantic_models(escopo["Direto"]) == [_Item]
    with pytest.raises(TypeError, match=r"'EmTexto'.*módulo 'modulo_que_nao_existe'"):
        get_nested_pydantic_models(escopo["EmTexto"])


@pytest.mark.skipif(sys.version_info < (3, 12), reason="sintaxe `type X = ...`")
def test_type_da_linguagem_com_referencia_em_texto_e_resolvido_pelo_modulo():
    escopo = {"__name__": __name__}
    exec('type Itens = list["_Item"]', escopo)  # noqa: S102 (sintaxe que o 3.11 não compila)

    assert is_list_of_pydantic_model(escopo["Itens"]) == (True, _Item)


# Referência em texto no limite do TypeVar e no tipo do NewType

_TypeVarComLimiteEmTexto = TypeVar("_TypeVarComLimiteEmTexto", bound="_Item")
_TypeVarComRestricoesEmTexto = _typevar("_TypeVarComRestricoesEmTexto", "_Item", "int")
_NewTypeEmTexto: Any = NewType("_NewTypeEmTexto", "_Item")  # ty: ignore[invalid-newtype]
_NewTypeDeListaEmTexto = NewType("_NewTypeDeListaEmTexto", list["_Item"])


def test_referencia_em_texto_do_typevar_e_do_newtype_e_trocada_pelo_tipo():
    assert _resolved(_TypeVarComLimiteEmTexto) is _Item
    assert _resolved(_TypeVarComRestricoesEmTexto) == _Item | int
    assert _resolved(_NewTypeEmTexto) is _Item
    assert _resolved(_NewTypeDeListaEmTexto) == list[_Item]


@pytest.mark.parametrize(
    "anotacao",
    [_TypeVarComLimiteEmTexto, _NewTypeEmTexto, list[_NewTypeEmTexto], _NewTypeDeListaEmTexto],
    ids=["typevar", "newtype", "lista-de-newtype", "newtype-de-lista"],
)
def test_submodelo_por_referencia_em_texto_no_typevar_ou_no_newtype_entra_na_configuracao(
    anotacao,
):
    modelo = create_model("Modelo", campo=(anotacao, ...))

    assert get_nested_pydantic_models(anotacao) == [_Item]
    assert [path for path, *_ in _collect_configured_fields(modelo)] == ["campo.valor"]
    with pytest.raises(ValueError, match=r"requerem use_search=True e search_per_field=True"):
        _validate_field_configs(modelo, None, use_search=False, search_per_field=False)


@pytest.mark.parametrize(
    "declara",
    [
        lambda: _typevar("Limite", bound="NaoExiste"),
        lambda: _newtype("Novo", "list[NaoExiste]"),
        lambda: _newtype("Novo", "typing.NaoExiste"),
        lambda: _newtype("Novo", "list["),
        lambda: _newtype("Novo", "Novo"),
    ],
    ids=["typevar", "newtype", "atributo", "sintaxe", "o-proprio-nome"],
)
def test_referencia_em_texto_do_typevar_ou_do_newtype_que_nao_se_resolve_fica_como_esta(declara):
    nomeado = declara()
    escopo = {"Novo": nomeado}

    resolvido = _resolved(nomeado, escopo)

    assert isinstance(resolvido, (str, ForwardRef))
    assert get_nested_pydantic_models(nomeado) == []
    modelo = create_model("Modelo", campo=(str, ...))
    _validate_field_configs(modelo, None, use_search=False, search_per_field=False)


def test_typevar_sem_modulo_resolve_sem_ele():
    sem_modulo = _typevar("sem_modulo", bound=int)
    sem_modulo.__module__ = None

    assert _resolved(sem_modulo) is int


# Apelido com referência em texto e classe de mesmo nome dentro da função e no módulo


def _modelo_com_item_local(anotacao: Any):
    """Modelo declarado ao lado de um `_Item` local, com o campo `itens` na anotação dada."""

    class _Item(BaseModel):
        outro: int

    class Modelo(BaseModel):
        itens: anotacao

    return Modelo, _Item


def test_pydantic_resolve_a_referencia_em_texto_do_apelido_pela_classe_local():
    modelo, local = _modelo_com_item_local(_apelido("Itens", list["_Item"]))

    assert type(modelo.model_validate({"itens": [{"outro": 1}]}).itens[0]) is local


_RECUSA_DO_NOME_HOMONIMO = (
    r"do modelo 'Modelo' usa um apelido de tipo, um NewType ou um TypeVar com referência em "
    r"texto a um nome que é uma classe no escopo em que o modelo foi declarado e outra no "
    r"módulo do tipo nomeado: o modelo valida com a primeira, e a configuração por campo leria "
    r"a segunda\. Declare o tipo sem aspas, ou dê nomes diferentes às duas classes\.$"
)

_GenericoComItem: Any = _apelido("Generico", "list[_T] | _Item", type_params=(_T,))
_OpcionalDeGenerico: Any = Optional[_GenericoComItem[int]]
_ApelidoDeApelido = TypeAliasType("_ApelidoDeApelido", Optional[_ApelidoComTexto])
_NewTypeDeApelido = _newtype("_NewTypeDeApelido", _ApelidoComTexto)
_ApelidoDoModeloDono = TypeAliasType("_ApelidoDoModeloDono", list["Modelo"])
_ArvoreEmTexto = TypeAliasType("_ArvoreEmTexto", "list[_ArvoreEmTexto] | _Item")


class Modelo(BaseModel):
    """Homônimo, no módulo, do modelo que os testes declaram dentro da função."""

    valor: str = Field(json_schema_extra={"prompt": "busque o valor"})


@pytest.mark.parametrize(
    "anotacao",
    [
        _apelido("Itens", list["_Item"]),
        _apelido("Itens", "list[_Item]"),
        _OpcionalDeGenerico,
        _ApelidoDeApelido,
        _NewTypeDeApelido,
        _NewTypeDeListaEmTexto,
        _TypeVarComLimiteEmTexto,
        list[_NewTypeEmTexto],
    ],
    ids=[
        "apelido-no-item",
        "apelido-inteiro",
        "apelido-generico",
        "apelido-de-apelido",
        "newtype-de-apelido",
        "newtype",
        "typevar",
        "lista-de-newtype",
    ],
)
def test_referencia_em_texto_a_classe_local_homonima_da_do_modulo_e_recusada(anotacao):
    modelo, _ = _modelo_com_item_local(anotacao)

    with pytest.raises(TypeError, match=_RECUSA_DO_NOME_HOMONIMO):
        _validate_field_configs(modelo, None, use_search=False, search_per_field=False)
    with pytest.raises(TypeError, match=r"^A anotação .* do modelo 'Modelo' usa um apelido"):
        list(_collect_configured_fields(modelo))


_NewTypeSoLocal: Any = _newtype("_NewTypeSoLocal", "SoLocal")
_TuplaComRamoSemResolver = TypeAliasType(
    "_TuplaComRamoSemResolver", tuple["_Item", _NewTypeSoLocal]
)


def test_ramo_que_o_modulo_nao_resolve_nao_esconde_a_classe_homonima_de_outro_ramo():
    class SoLocal(BaseModel):
        outro: int

    modelo, _ = _modelo_com_item_local(_TuplaComRamoSemResolver)

    assert SoLocal.__name__ == "SoLocal"
    with pytest.raises(TypeError, match=_RECUSA_DO_NOME_HOMONIMO):
        list(_collect_configured_fields(modelo))


_UniaoComNomeSoLocal: Any = _newtype("_UniaoComNomeSoLocal", _uniao[ForwardRef("SoLocal"), _Item])


def test_uniao_que_o_escopo_do_modelo_reduz_a_um_membro_nao_e_recusada():
    def declara():
        SoLocal = _Item  # noqa: N806 (o nome que o texto da união procura)

        class Modelo(BaseModel):
            campo: _UniaoComNomeSoLocal

        return Modelo, SoLocal

    modelo, _ = declara()

    assert type(modelo.model_validate({"campo": {"valor": "a"}}).campo) is _Item
    list(_collect_configured_fields(modelo))


def test_referencia_em_texto_ao_nome_do_proprio_modelo_dono_e_recusada():
    def declara():
        class Modelo(BaseModel):
            outro: int = 0
            filhos: _ApelidoDoModeloDono = []

        return Modelo

    with pytest.raises(TypeError, match=_RECUSA_DO_NOME_HOMONIMO):
        list(_collect_configured_fields(declara()))


def test_newtype_com_referencia_em_texto_a_nome_que_so_existe_na_funcao_segue_aceito():
    class SoLocal(BaseModel):
        valor: str = Field(json_schema_extra={"prompt": "busque o valor"})

    lista: Any = _newtype("Lista", "list[SoLocal]")

    class Modelo(BaseModel):
        itens: lista

    # O módulo não acha o nome, e o submodelo fica fora da configuração, como antes.
    assert list(_collect_configured_fields(Modelo)) == []
    assert type(Modelo.model_validate({"itens": [{"valor": "a"}]}).itens[0]) is SoLocal


def test_apelido_com_nome_que_so_existe_na_funcao_segue_recusado_pelo_erro_do_apelido():
    class SoLocal(BaseModel):
        valor: str

    apelido: Any = _apelido("Itens", list["SoLocal"])

    class Modelo(BaseModel):
        itens: apelido

    with pytest.raises(TypeError, match=r"apelido de tipo 'Itens'.*não foi encontrado no módulo"):
        list(_collect_configured_fields(Modelo))


def test_anotacao_com_literal_e_annotated_ao_lado_do_tipo_nomeado_segue_aceita():
    def declara():
        sem_relacao = 1

        class Modelo(BaseModel):
            campo: Annotated[_ApelidoComTexto, "_Item"]
            outro: Literal["_Item"]
            arvore: _ArvoreEmTexto

        return Modelo, sem_relacao

    modelo, _ = declara()

    assert [path for path, *_ in _collect_configured_fields(modelo)] == [
        "campo.valor",
        "arvore.valor",
    ]


def test_apelido_sem_classe_homonima_no_escopo_do_modelo_segue_aceito():
    def declara():
        sem_relacao = 1
        apelido: Any = _apelido("Itens", list["_Item"])

        class Modelo(BaseModel):
            itens: apelido

        return Modelo, sem_relacao

    modelo, _ = declara()

    assert [path for path, *_ in _collect_configured_fields(modelo)] == ["itens.valor"]
    # O modelo do módulo, sem escopo de função, também.
    do_modulo = create_model("Modelo", itens=(_ApelidoComTexto, ...))
    assert [path for path, *_ in _collect_configured_fields(do_modulo)] == ["itens.valor"]
