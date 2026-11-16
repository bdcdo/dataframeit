"""Campo `Iterable` ou `Generator` interrompe a execução na primeira resposta que o traz.

O pydantic valida esses dois tipos como iterador, e o `model_dump` devolve um
`SerializationIterator` no lugar dos itens. A célula guardaria o iterador, a
falha vem do modelo e se repete, e por isso não ganha nova tentativa nem
segue para a linha seguinte.
"""

from collections.abc import Generator, Iterable
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pydantic import BaseModel, field_serializer

from dataframeit import dataframeit
from dataframeit.core import ProviderBackend, _invoke_row
from dataframeit.errors import ProviderAbortError


class CampoIterable(BaseModel):
    campo: Iterable[str]


class CampoGenerator(BaseModel):
    campo: Generator[str, None, None]


class IterableOpcional(BaseModel):
    campo: Optional[Iterable[str]] = None


class GeneratorOpcional(BaseModel):
    campo: Optional[Generator[str, None, None]] = None


class IterableComSerializador(BaseModel):
    campo: Optional[Iterable[str]] = None

    @field_serializer("campo")
    def _em_lista(self, valor: Optional[Iterable[str]]) -> Optional[list[str]]:
        return None if valor is None else list(valor)


class Item(BaseModel):
    nomes: Iterable[str]


class ItemEmLista(BaseModel):
    campo: list[Item]


class ItemEmDict(BaseModel):
    campo: dict[str, Item]


class ChatFalso:
    """Chat model simulado: call_langchain roda inteiro, inclusive o model_dump da resposta."""

    def __init__(self, modelo, respostas):
        self.modelo, self.respostas, self.invocacoes = modelo, list(respostas), 0

    def with_structured_output(self, modelo, include_raw=True):
        return self

    def invoke(self, mensagens, config=None):
        resposta = self.respostas[min(self.invocacoes, len(self.respostas) - 1)]
        self.invocacoes += 1
        return {
            "raw": None,
            "parsed": self.modelo.model_validate({"campo": resposta}),
            "parsing_error": None,
        }


def _executa(modelo, respostas, textos=("a",)):
    chat = ChatFalso(modelo, respostas)
    with (
        patch("dataframeit.llm.chat_model", return_value=chat),
        patch("dataframeit.core.validate_provider_dependencies"),
    ):
        saida = dataframeit(
            pd.DataFrame({"texto": list(textos)}),
            questions=modelo,
            prompt="{texto}",
            provider="openai",
            track_tokens=False,
        )
    return chat, saida


@pytest.mark.parametrize("modelo", [CampoIterable, CampoGenerator])
def test_campo_iterable_ou_generator_interrompe_na_primeira_resposta(modelo):
    with pytest.warns(UserWarning, match="Execução interrompida") as avisos:
        chat, saida = _executa(modelo, [["x", "y"]], textos=("a", "b", "c"))

    assert chat.invocacoes == 1
    assert saida["_dataframeit_status"].isna().all()
    assert saida["campo"].isna().all()
    [aviso] = [str(aviso.message) for aviso in avisos]
    assert "A resposta trouxe um iterador no lugar dos itens em ['campo']" in aviso
    assert "declare list[...] nesse campo, ou no campo de dentro dele" in aviso


@pytest.mark.parametrize(("modelo", "resposta"), [
    (ItemEmLista, [{"nomes": ["a"]}]),
    (ItemEmDict, {"k": {"nomes": ["a"]}}),
])  # fmt: skip
def test_iterable_dentro_de_submodelo_em_colecao_interrompe_pelo_campo_de_cima(modelo, resposta):
    with pytest.warns(UserWarning, match=r"Execução interrompida.*itens em \['campo'\]"):
        chat, saida = _executa(modelo, [resposta], textos=("a", "b"))

    assert chat.invocacoes == 1
    assert saida["_dataframeit_status"].isna().all()


@pytest.mark.parametrize("modelo", [IterableOpcional, GeneratorOpcional])
@pytest.mark.parametrize("itens", [["x"], []], ids=["com_itens", "vazia"])
def test_campo_opcional_grava_as_respostas_none_e_interrompe_na_primeira_que_nao_e_none(
    modelo, itens
):
    with pytest.warns(UserWarning, match="Execução interrompida.*trouxe um iterador"):
        chat, saida = _executa(modelo, [None, None, itens], textos=("a", "b", "c", "d"))

    assert chat.invocacoes == 3
    assert saida["_dataframeit_status"].tolist()[:2] == ["processed", "processed"]
    assert saida["_dataframeit_status"].isna().tolist()[2:] == [True, True]
    assert saida["campo"].isna().all()


def test_uso_da_resposta_que_interrompe_vai_na_excecao():
    uso = {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}
    backend = ProviderBackend("falso", lambda texto: {"data": {"campo": iter("x")}, "usage": uso})

    with pytest.raises(ProviderAbortError, match="trouxe um iterador") as erro:
        _invoke_row(backend, "a", pd.Series({"texto": "a"}), None, ["campo", "ausente"])

    assert erro.value.row_usage == uso  # ty: ignore[unresolved-attribute]


def test_iterable_com_field_serializer_chega_a_saida():
    chat, saida = _executa(IterableComSerializador, [["x", "y"]])

    assert saida["campo"].tolist() == [["x", "y"]]
    assert chat.invocacoes == 1


def test_agente_de_busca_interrompe_na_primeira_resposta():
    class DoisCampos(BaseModel):
        campo: Iterable[str]
        outro: str = "o"

    agente = MagicMock()
    agente.invoke.return_value = {
        "structured_response": DoisCampos.model_validate({"campo": ["x"]}),
        "messages": [],
    }
    provider = MagicMock()
    provider.calculate_credits.return_value = 0

    with (
        patch("dataframeit.llm._create_langchain_llm", return_value=object()),
        patch("langchain.agents.create_agent", return_value=agente),
        patch("dataframeit.agent.get_provider", return_value=provider),
        patch("dataframeit.core.validate_provider_dependencies"),
        patch("dataframeit.core.validate_search_dependencies"),
        pytest.warns(UserWarning, match="Execução interrompida.*trouxe um iterador"),
    ):
        saida = dataframeit(
            pd.DataFrame({"texto": ["a", "b"]}),
            DoisCampos,
            "Responda {texto}",
            use_search=True,
            track_tokens=False,
        )

    assert agente.invoke.call_count == 1
    assert saida["_dataframeit_status"].isna().all()


def test_busca_por_campo_com_field_serializer_chega_a_saida():
    """O modelo de cada chamada por campo não leva o serializador; o dump final, sim."""

    class AgenteDoCampo:
        def __init__(self, modelo):
            self.modelo = modelo

        def invoke(self, entrada, config=None):
            resposta = self.modelo.model_validate({"campo": ["x", "y"]})
            return {"structured_response": resposta, "messages": []}

    def monta(modelo, config):
        provider = MagicMock()
        provider.name = "tavily"
        provider.calculate_credits.return_value = 0
        return SimpleNamespace(agent=AgenteDoCampo(modelo), provider=provider, tool_name="busca")

    with (
        patch("dataframeit.agent.build_search_agent", side_effect=monta),
        patch("dataframeit.llm._create_langchain_llm", return_value=object()),
        patch("dataframeit.core.validate_provider_dependencies"),
        patch("dataframeit.core.validate_search_dependencies"),
    ):
        saida = dataframeit(
            pd.DataFrame({"texto": ["a"]}),
            IterableComSerializador,
            "{texto}",
            use_search=True,
            search_per_field=True,
            track_tokens=False,
        )

    assert saida["campo"].tolist() == [["x", "y"]]
