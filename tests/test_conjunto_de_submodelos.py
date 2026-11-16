"""Campo `set` ou `frozenset` de submodelos interrompe a execução na primeira resposta.

O `model_dump` monta o conjunto com o dump de cada item, e o dict do
submodelo não é hashable: a resposta paga cai com
`TypeError: unhashable type: 'dict'`. A falha vem do modelo e se repete em
toda linha, e por isso não ganha nova tentativa nem segue para a linha
seguinte.
"""

from enum import Enum
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pydantic import BaseModel, ConfigDict, field_serializer
from typing_extensions import override

from dataframeit import dataframeit
from dataframeit.errors import ProviderAbortError, dump_response


class Tag(Enum):
    PF = "pf"


class Cidade(BaseModel):
    model_config = ConfigDict(frozen=True)
    nome: str


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


class ChaveDeModelo(BaseModel):
    campo: dict[Cidade, int]


class DumpQueCai(BaseModel):
    campo: str

    @override
    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        msg = f"model_dump do usuário caiu com {self.campo}"
        raise TypeError(msg)


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


def _executa(modelo, resposta, textos=("a",)):
    chat = ChatFalso(modelo, resposta)
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


@pytest.mark.parametrize("modelo", [ConjuntoDeModelo, ConjuntoCongeladoDeModelo])
def test_conjunto_de_submodelos_interrompe_na_primeira_resposta(modelo):
    with pytest.warns(UserWarning, match="Execução interrompida") as avisos:
        chat, saida = _executa(modelo, [{"nome": "SP"}], textos=("a", "b", "c"))

    assert chat.invocacoes == 1
    assert saida["_dataframeit_status"].isna().all()
    [aviso] = [str(aviso.message) for aviso in avisos]
    assert "unhashable type: 'dict'" in aviso
    assert f"O modelo {modelo.__name__} tem um set, um frozenset ou uma chave de dict" in aviso
    assert "declare o conjunto como list ou tuple" in aviso


def test_conjunto_de_enum_chega_a_saida():
    chat, saida = _executa(ConjuntoDeEnum, ["pf"])

    assert saida["campo"].tolist() == [{Tag.PF}]
    assert chat.invocacoes == 1


def test_conjunto_de_submodelos_com_field_serializer_chega_a_saida():
    chat, saida = _executa(ConjuntoComSerializador, [{"nome": "SP"}, {"nome": "RJ"}])

    assert saida["campo"].tolist() == [[{"nome": "RJ"}, {"nome": "SP"}]]
    assert chat.invocacoes == 1


def test_resposta_sem_item_no_conjunto_chega_a_saida():
    chat, saida = _executa(ConjuntoDeModelo, [])

    assert saida["campo"].tolist() == [set()]
    assert chat.invocacoes == 1


def test_dump_response_devolve_o_dump_pelo_nome_dos_campos():
    resposta = ConjuntoDeEnum(campo={Tag.PF})

    assert dump_response(resposta) == {"campo": {Tag.PF}}


def test_dump_response_guarda_o_type_error_como_causa():
    resposta = ConjuntoDeModelo.model_validate({"campo": [{"nome": "SP"}]})

    with pytest.raises(ProviderAbortError, match="Repetir a linha não resolve") as erro:
        dump_response(resposta)

    assert isinstance(erro.value.__cause__, TypeError)


def test_chave_de_dict_que_e_submodelo_interrompe_como_o_conjunto():
    resposta = ChaveDeModelo(campo={Cidade(nome="SP"): 1})

    with pytest.raises(ProviderAbortError, match="unhashable type: 'dict'"):
        dump_response(resposta)


def test_type_error_de_outra_origem_segue_como_veio():
    with pytest.raises(TypeError, match="model_dump do usuário caiu com x"):
        dump_response(DumpQueCai(campo="x"))


def test_agente_de_busca_interrompe_na_primeira_resposta():
    agente = MagicMock()
    agente.invoke.return_value = {
        "structured_response": ConjuntoDeModelo.model_validate({"campo": [{"nome": "SP"}]}),
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
        pytest.warns(UserWarning, match="Execução interrompida.*unhashable type: 'dict'"),
    ):
        saida = dataframeit(
            pd.DataFrame({"texto": ["a", "b"]}),
            ConjuntoDeModelo,
            "Responda {texto}",
            use_search=True,
            track_tokens=False,
        )

    assert agente.invoke.call_count == 1
    assert saida["_dataframeit_status"].isna().all()
