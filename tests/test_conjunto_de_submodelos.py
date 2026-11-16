"""Campo `set` ou `frozenset` de submodelos é recusado antes da primeira chamada.

O `model_dump` monta o conjunto com o dump de cada item, e o dict do
submodelo não é hashable: sem a recusa, toda linha pagaria as tentativas
para cair com `TypeError: unhashable type: 'dict'`.
"""

import dataclasses
from enum import Enum
from typing import Optional
from unittest.mock import patch

import pandas as pd
import pytest
from pydantic import BaseModel, ConfigDict, field_serializer, model_serializer

from dataframeit import dataframeit
from dataframeit.core import _validate_sets_of_submodels


class Tag(Enum):
    PF = "pf"


class Cidade(BaseModel):
    model_config = ConfigDict(frozen=True)
    nome: str


@dataclasses.dataclass(frozen=True)
class Ponto:
    x: int


class CidadeQueViraTexto(BaseModel):
    model_config = ConfigDict(frozen=True)
    nome: str

    @model_serializer
    def _como_texto(self) -> str:
        return self.nome


class ConjuntoDeEnum(BaseModel):
    campo: set[Tag]


class ConjuntoDeModelo(BaseModel):
    campo: set[Cidade]


class ConjuntoCongeladoDeModelo(BaseModel):
    campo: frozenset[Cidade]


class ConjuntoComSerializador(BaseModel):
    campo: set[Cidade]

    @field_serializer("campo")
    def _em_lista(self, valor: set[Cidade]) -> list[Cidade]:
        return sorted(valor, key=lambda cidade: cidade.nome)


class ConjuntoOpcional(BaseModel):
    campo: Optional[set[Cidade]] = None


class ConjuntoDeTupla(BaseModel):
    campo: set[tuple[Cidade, int]]


class ConjuntoDeDataclass(BaseModel):
    campo: frozenset[Ponto]


class ConjuntoDeModeloQueViraTexto(BaseModel):
    campo: set[CidadeQueViraTexto]


class ListaDeConjuntos(BaseModel):
    itens: list[ConjuntoDeModelo]


class ModeloComSerializador(BaseModel):
    campo: set[Cidade]

    @model_serializer
    def _em_lista(self) -> dict[str, list[str]]:
        return {"campo": sorted(cidade.nome for cidade in self.campo)}


class No(BaseModel):
    """Modelo recursivo: o schema dele chega por `definition-ref`, mais de uma vez."""

    model_config = ConfigDict(frozen=True)
    nome: str
    pai: Optional["No"] = None
    padrinho: Optional["No"] = None


class ArvoreSemConjunto(BaseModel):
    raiz: No
    tags: set[Tag] = set()


class ConjuntoDeNos(BaseModel):
    raiz: No
    nos: set[No]


class ChatFalso:
    """Chat model simulado: call_langchain roda inteiro, inclusive o model_dump da resposta."""

    def __init__(self, modelo, resposta):
        self.modelo, self.resposta, self.invocacoes = modelo, resposta, 0

    def with_structured_output(self, modelo, include_raw=True):
        return self

    def invoke(self, mensagens, config=None):
        self.invocacoes += 1
        return {
            "raw": None,
            "parsed": self.modelo.model_validate({"campo": self.resposta}),
            "parsing_error": None,
        }


def _executa(modelo, resposta):
    chat = ChatFalso(modelo, resposta)
    with (
        patch("dataframeit.llm.chat_model", return_value=chat),
        patch("dataframeit.core.validate_provider_dependencies"),
        patch("time.sleep"),
    ):
        saida = dataframeit(
            pd.DataFrame({"texto": ["a"]}),
            questions=modelo,
            prompt="{texto}",
            provider="openai",
            track_tokens=False,
        )
    return chat, saida


@pytest.mark.parametrize(
    ("modelo", "tipo"),
    [(ConjuntoDeModelo, "set"), (ConjuntoCongeladoDeModelo, "frozenset")],
)
def test_conjunto_de_submodelos_e_recusado_sem_pagar_chamada(modelo, tipo):
    chat = ChatFalso(modelo, [{"nome": "SP"}])
    with (
        patch("dataframeit.llm.chat_model", return_value=chat),
        patch("dataframeit.core.validate_provider_dependencies"),
        pytest.raises(ValueError, match=rf"Campo 'campo' é um {tipo} de submodelos") as recusa,
    ):
        dataframeit(
            pd.DataFrame({"texto": ["a"]}),
            questions=modelo,
            prompt="{texto}",
            provider="openai",
            track_tokens=False,
        )
    assert chat.invocacoes == 0
    assert "list ou tuple" in str(recusa.value)
    assert "@field_serializer" in str(recusa.value)


def test_conjunto_de_enum_continua_chegando_a_saida():
    chat, saida = _executa(ConjuntoDeEnum, ["pf"])
    assert saida["campo"].tolist() == [{Tag.PF}]
    assert chat.invocacoes == 1


def test_conjunto_de_submodelos_com_field_serializer_chega_a_saida():
    chat, saida = _executa(ConjuntoComSerializador, [{"nome": "SP"}, {"nome": "RJ"}])
    assert saida["campo"].tolist() == [[{"nome": "RJ"}, {"nome": "SP"}]]
    assert chat.invocacoes == 1


@pytest.mark.parametrize(
    ("modelo", "campo", "tipo"),
    [
        (ConjuntoOpcional, "campo", "set"),
        (ConjuntoDeTupla, "campo", "set"),
        (ConjuntoDeDataclass, "campo", "frozenset"),
        (ListaDeConjuntos, "itens.campo", "set"),
        (ConjuntoDeNos, "nos", "set"),
    ],
)
def test_recusa_aponta_o_campo_do_conjunto(modelo, campo, tipo):
    with pytest.raises(ValueError, match=rf"^Campo '{campo}' é um {tipo} de submodelos"):
        _validate_sets_of_submodels(modelo)


@pytest.mark.parametrize(
    "modelo",
    [
        ConjuntoDeEnum,
        ConjuntoComSerializador,
        ConjuntoDeModeloQueViraTexto,
        ModeloComSerializador,
        ArvoreSemConjunto,
        No,
    ],
)
def test_modelo_cujo_dump_monta_o_conjunto_e_aceito(modelo):
    _validate_sets_of_submodels(modelo)


@pytest.mark.parametrize(
    ("modelo", "valores"),
    [
        (ConjuntoDeModelo, {"campo": [{"nome": "SP"}]}),
        (ConjuntoCongeladoDeModelo, {"campo": [{"nome": "SP"}]}),
        (ConjuntoOpcional, {"campo": [{"nome": "SP"}]}),
        (ConjuntoDeTupla, {"campo": [({"nome": "SP"}, 1)]}),
        (ConjuntoDeDataclass, {"campo": [{"x": 1}]}),
        (ListaDeConjuntos, {"itens": [{"campo": [{"nome": "SP"}]}]}),
        (ConjuntoDeNos, {"raiz": {"nome": "a"}, "nos": [{"nome": "b"}]}),
    ],
)
def test_modelo_recusado_e_o_que_o_model_dump_nao_monta(modelo, valores):
    """A recusa segue o pydantic: cada modelo recusado cai no dump de uma resposta válida."""
    with pytest.raises(TypeError, match="unhashable type: 'dict'"):
        modelo.model_validate(valores).model_dump(by_alias=False)


@pytest.mark.parametrize(
    ("modelo", "valores", "dump"),
    [
        (ConjuntoDeModeloQueViraTexto, {"campo": [{"nome": "SP"}]}, {"campo": {"SP"}}),
        (ModeloComSerializador, {"campo": [{"nome": "SP"}]}, {"campo": ["SP"]}),
    ],
)
def test_modelo_aceito_com_serializador_proprio_passa_pelo_model_dump(modelo, valores, dump):
    assert modelo.model_validate(valores).model_dump(by_alias=False) == dump
