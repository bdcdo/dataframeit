"""Forma canônica que a assinatura do checkpoint tira de cada folha e de cada contêiner."""

from decimal import Decimal
from enum import Enum
from functools import partial
from types import MethodType
from typing import Any, Literal, NewType, TypeVar

import pandas as pd
import pytest
from pydantic import BaseModel, ConfigDict, Field, create_model
from typing_extensions import TypeAliasType

from dataframeit.core import (
    _callable_leaf,
    _code_form,
    _Descent,
    _has_unstable_extra,
    _held_value,
    _key_form,
    _python_call,
    _run_signature,
    _schema_form,
    _schema_leaf,
    _schema_signature,
    _slot_values,
    _stable,
)
from dataframeit.llm import SearchConfig


def _ecoa(texto, *args, **kwargs):
    return texto


def _forma_da_funcao(funcao):
    return {"function": funcao.__qualname__, "code": _code_form(funcao.__code__)}


def _assinatura(modelo, search_config=None):
    return _run_signature(
        modelo,
        "{texto}",
        provider="openai",
        model="m",
        model_kwargs=None,
        search_config=search_config,
        text_column="texto",
        status_col="_dataframeit_status",
    )


# Por `Any`, o checador de tipos não toma por declaração estática o NewType, o
# apelido e o TypeVar que os testes montam para cada modelo.
_newtype: Any = NewType
_apelido: Any = TypeAliasType
_typevar: Any = TypeVar


def _lista(item: Any) -> Any:
    return list[item]


_T = TypeVar("_T")
_ListaGenerica = TypeAliasType("_ListaGenerica", list[_T], type_params=(_T,))


class Cor(Enum):
    AZUL = "azul"


class Tipo(Enum):
    INTEIRO = int


class ComConfig(BaseModel):
    """Doc."""

    model_config = ConfigDict(json_schema_extra={"tipo": int})  # ty: ignore[invalid-argument-type]

    campo: int


class Regras:
    @classmethod
    def limite(cls, texto):
        return texto


class Dono:
    def __init__(self):
        self.limite = 3

    def responder(self, texto):
        return texto


class ChamaMetodoDeOutro:
    """Classe cujo `__call__` é o método ligado de outra instância."""

    __call__ = Dono().responder


class Par(tuple[object, ...]):
    """Tupla chamável: o valor dela fica fora do `__dict__`."""

    __slots__ = ()

    def __call__(self, texto):
        return texto


class _Xadrez:
    # O slot não atribuído vem antes dos outros, que são lidos mesmo assim.
    __slots__ = ("vazio", "cheio", "__dict__", "__weakref__", "__marca__", "__peca")  # noqa: RUF023

    def __init__(self):
        self.cheio = 2
        self.__marca__ = 1
        self.__peca = "rei"


class ComDescritor:
    __slots__ = ("peso",)


class _DevolveODono:
    def __get__(self, obj, owner):
        return owner.__name__


# O descritor posto no lugar do slot depois de a classe ser criada.
ComDescritor.peso = _DevolveODono()  # ty: ignore[invalid-assignment]


# Tipo genérico, modelo e Enum


def test_tipo_generico_entra_pela_origem_e_pelos_argumentos():
    assert _schema_leaf(list[int], _Descent()) == {"origin": "list", "args": ["int"]}


def test_origem_que_nao_e_tipo_entra_pelo_chamavel_dela():
    origem = _schema_leaf(Literal, _Descent())

    assert _schema_leaf(Literal["a"], _Descent()) == {"origin": origem, "args": ["a"]}
    assert isinstance(origem, dict)
    assert set(origem) == {"callable", "state"}


def test_modelo_entra_pelo_nome_pela_doc_e_pelo_model_config_na_forma_canonica():
    assert _schema_leaf(ComConfig, _Descent()) == {
        "model": "ComConfig",
        "doc": "Doc.",
        "config": {"json_schema_extra": {"tipo": "int"}},
    }


def test_enum_entra_pelo_nome_e_pelos_membros_na_forma_canonica():
    assert _schema_leaf(Cor, _Descent()) == {"enum": "Cor", "members": {"AZUL": "azul"}}
    assert _schema_leaf(Tipo, _Descent()) == {"enum": "Tipo", "members": {"INTEIRO": "int"}}


# Chamáveis


def test_metodo_de_classe_entra_pela_funcao_e_pelo_nome_da_classe():
    assert _callable_leaf(Regras.limite, _Descent()) == {
        "method": _forma_da_funcao(Regras.limite.__func__),
        "self": "Regras",
    }


def test_partial_entra_pela_funcao_e_pelos_argumentos_na_forma_canonica():
    assert _callable_leaf(partial(_ecoa, int, tipo=str), _Descent()) == {
        "partial": _forma_da_funcao(_ecoa),
        "args": ["int"],
        "keywords": {"tipo": "str"},
    }


def test_staticmethod_entra_pelo_partial_que_embrulha():
    assert _callable_leaf(staticmethod(partial(_ecoa, int)), _Descent()) == {
        "partial": _forma_da_funcao(_ecoa),
        "args": ["int"],
        "keywords": {},
    }


def test_metodo_ligado_a_um_partial_entra_pelo_partial_na_forma_canonica():
    metodo = MethodType(partial(_ecoa, int), Regras)

    assert _callable_leaf(metodo, _Descent()) == {
        "method": {"partial": _forma_da_funcao(_ecoa), "args": ["int"], "keywords": {}},
        "self": "Regras",
    }


def test_call_que_e_metodo_de_outra_instancia_entra_pelos_atributos_dela():
    assert _callable_leaf(ChamaMetodoDeOutro(), _Descent()) == {
        "callable": {"method": _forma_da_funcao(Dono.responder), "self": {"limite": 3}},
        "state": {},
    }


def test_membro_de_enum_nao_e_chamavel_pelo_call_da_metaclasse():
    assert _python_call(Cor.AZUL) is None
    assert _callable_leaf(Cor.AZUL, _Descent()) is Cor.AZUL


# Estado da instância


def test_subclasse_chamavel_de_tupla_entra_pelo_valor_e_pelo_estado():
    assert _stable(Par((int,)), _Descent()) == {
        "callable": _forma_da_funcao(Par.__call__),
        "state": {"value": ["int"], "state": {}},
    }


def test_slots_atribuidos_entram_pelo_nome_guardado():
    assert _slot_values(_Xadrez()) == {"cheio": 2, "__marca__": 1, "_Xadrez__peca": "rei"}


def test_descritor_no_lugar_do_slot_e_lido_com_a_classe_como_dono():
    assert _slot_values(ComDescritor()) == {"peso": "ComDescritor"}


# Contêineres


def test_lista_que_contem_a_si_mesma_entra_pelo_marcador_com_o_nome_da_classe():
    lista: list[object] = []
    lista.append(lista)

    assert _stable(lista) == [{"cycle": "list"}]


def test_stable_sem_descida_deixa_a_folha_como_veio():
    assert _stable([int]) == [int]


def test_conjunto_entra_pelas_folhas_na_forma_canonica_ordenadas():
    assert _stable({int, str}, _Descent()) == ["int", "str"]


def test_chave_de_conjunto_entra_pelas_folhas_na_forma_canonica():
    assert _key_form(frozenset({int}), _Descent()) == "['int']"
    assert _stable({frozenset({int}): 1}, _Descent()) == {"['int']": 1}


@pytest.mark.parametrize("forma_de", [_stable, _held_value], ids=["stable", "valor-guardado"])
def test_series_entra_pelas_celulas_na_forma_canonica(forma_de):
    forma = forma_de(pd.Series([int], name="tipo"), _Descent())

    assert isinstance(forma, dict)
    assert forma["columns"] == [["tipo", "object", ["int"]]]


# Schema e assinatura


def test_modelo_com_referencia_adiantada_sem_alvo_nao_derruba_a_busca_de_extra():
    pendente = create_model("Pendente", campo=("NaoDefinido", ...))  # ty: ignore[unresolved-reference]

    assert _has_unstable_extra(pendente) is False


def test_json_schema_com_valor_fora_do_json_continua_a_assinatura():
    class ComDecimal(BaseModel):
        model_config = ConfigDict(json_schema_extra={"limite": Decimal("1.5")})  # ty: ignore[invalid-argument-type]

        campo: int

    assert _schema_signature(ComDecimal) == ComDecimal.model_json_schema()


def test_json_schema_com_chaves_de_tipos_misturados_da_lugar_a_forma_canonica():
    # O Pydantic não ordena o que fica sob "default", e a chave int ao lado da
    # str faz o json.dumps ordenado levantar.
    class ChavesMisturadas(BaseModel):
        model_config = ConfigDict(json_schema_extra={"default": {1: "um", "dois": 2}})  # ty: ignore[invalid-argument-type]

        campo: int

    assert _schema_signature(ChavesMisturadas) == _schema_form(ChavesMisturadas)
    assert len(_assinatura(ChavesMisturadas)) == 64


def test_caminho_repetido_na_uniao_entra_com_os_campos_dos_dois_ramos():
    class A(BaseModel):
        x: str = Field(description="de A")

    class B(BaseModel):
        x: str = Field(description="de B")

    class Uniao(BaseModel):
        campo: A | B

    forma = _schema_form(Uniao)

    assert isinstance(forma, dict)
    assert [campo["description"] for campo in forma["fields"]["campo.x"]] == ["de A", "de B"]


# NewType, apelido `type` e TypeVar


def _com_conjunto(anotacao):
    """Modelo cuja assinatura sai da forma canônica, com `campo` na anotação dada."""
    extra: Any = {"rotulos": {"a", "b"}}
    return create_model(
        "Modelo", campo=(anotacao, ...), outro=(str, Field(json_schema_extra=extra))
    )


@pytest.mark.parametrize(
    "declara",
    [
        lambda tipo: _newtype("Codigo", tipo),
        lambda tipo: _apelido("Codigo", tipo),
        lambda tipo: _ListaGenerica[tipo],
        lambda tipo: _typevar("Codigo", bound=tipo),
    ],
    ids=["newtype", "apelido", "apelido-generico", "typevar"],
)
def test_mudar_o_tipo_atras_do_nome_muda_a_assinatura(declara):
    assert _assinatura(_com_conjunto(declara(str))) != _assinatura(_com_conjunto(declara(int)))


@pytest.mark.parametrize(
    ("por_nome", "direta"),
    [
        (_newtype("Codigo", str), str),
        (_apelido("Itens", list[Cor]), list[Cor]),
        (_ListaGenerica[Cor], list[Cor]),
        (_lista(_newtype("Codigo", int)), list[int]),
    ],
    ids=["newtype", "apelido", "apelido-generico", "newtype-no-item"],
)
def test_anotacao_por_nome_assina_como_a_direta(por_nome, direta):
    assert _schema_form(_com_conjunto(por_nome)) == _schema_form(_com_conjunto(direta))
    assert _assinatura(_com_conjunto(por_nome)) == _assinatura(_com_conjunto(direta))


def test_apelido_recursivo_entra_por_um_marcador_com_o_nome():
    json_ = _apelido("Json", "dict[str, list[Json]]")

    assert _schema_leaf(json_, _Descent()) == {
        "origin": "dict",
        "args": ["str", {"origin": "list", "args": [{"cycle": "Json"}]}],
    }


def test_apelido_repetido_fora_do_proprio_valor_entra_inteiro_nas_duas_vezes():
    codigo: Any = _apelido("Codigo", int)

    assert _schema_leaf(dict[codigo, codigo], _Descent()) == {
        "origin": "dict",
        "args": ["int", "int"],
    }


def test_configuracao_de_busca_muda_a_assinatura():
    class Modelo(BaseModel):
        campo: str

    assert _assinatura(Modelo, SearchConfig(enabled=True, max_results=3)) != _assinatura(
        Modelo, SearchConfig(enabled=True, max_results=5)
    )
