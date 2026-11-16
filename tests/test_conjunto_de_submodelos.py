"""Campo `set` ou `frozenset` de submodelos sai como lista, na resposta e na retomada.

O `model_dump` monta o conjunto com o dump de cada item, e o dict do
submodelo não é hashable: o dump cai com `TypeError: unhashable type: 'dict'`.
O campo que é um conjunto de submodelos sai como lista, com o dump de cada
item. O que sobra, como a chave de dict que é submodelo, vem do modelo e se
repete em toda linha, e por isso interrompe a execução na primeira resposta,
sem nova tentativa.
"""

import datetime as dt
from collections.abc import Set as AbstractSet
from enum import Enum
from typing import Any, Optional, Union
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    create_model,
    field_serializer,
    field_validator,
    model_serializer,
)
from pydantic_core import PydanticSerializationError
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


class ConjuntoAninhado(BaseModel):
    sub: ConjuntoDeModelo
    campo: str


class ConjuntoEChave(BaseModel):
    conjunto: set[Cidade]
    chaves: dict[Cidade, int]


class VariosCampos(BaseModel):
    antes: str
    conjunto: frozenset[Cidade]
    depois: list[int]
    outro: set[Cidade]
    escondido: set[Cidade] = Field(exclude=True)
    misto: set[Union[Cidade, str]]
    opcional: Optional[set[Cidade]]


class Regiao(BaseModel):
    model_config = ConfigDict(frozen=True)
    nome: str
    cidades: frozenset[Cidade]


class ConjuntoDeRegioes(BaseModel):
    campo: set[Regiao]


class ComExtra(BaseModel):
    model_config = ConfigDict(extra="allow")
    campo: set[Cidade]


class DumpQueCaiSemOConjunto(BaseModel):
    campo: set[Cidade]

    @override
    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        if kwargs.get("exclude"):
            msg = "model_dump do usuário caiu sem o conjunto"
            raise TypeError(msg)
        return super().model_dump(**kwargs)


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


class ChatComResposta(ChatFalso):
    """Chat que devolve a resposta já validada, como a que não se monta a partir do JSON."""

    @override
    def invoke(self, mensagens, config=None):
        self.invocacoes += 1
        return {"raw": None, "parsed": self.resposta, "parsing_error": None}


def _executa_resposta(resposta, textos=("a",)):
    return _executa(type(resposta), resposta, textos, chat=ChatComResposta)


def _executa(modelo, resposta, textos=("a",), chat=ChatFalso):
    chat = chat(modelo, resposta)
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
def test_conjunto_de_submodelos_chega_a_saida_como_lista(modelo):
    chat, saida = _executa(modelo, [{"nome": "SP"}, {"nome": "RJ"}], textos=("a", "b"))

    assert chat.invocacoes == 2
    assert saida["campo"].tolist() == [[{"nome": "RJ"}, {"nome": "SP"}]] * 2


def test_chave_de_dict_que_e_submodelo_interrompe_na_primeira_resposta():
    with pytest.warns(UserWarning, match="Execução interrompida") as avisos:
        chat, saida = _executa_resposta(
            ChaveDeModelo(campo={Cidade(nome="SP"): 1}), textos=("a", "b", "c")
        )

    assert chat.invocacoes == 1
    assert saida["_dataframeit_status"].isna().all()
    [aviso] = [str(aviso.message) for aviso in avisos]
    assert "unhashable type: 'dict'" in aviso
    assert "O modelo ChaveDeModelo tem um set, um frozenset ou uma chave de dict" in aviso
    assert (
        "nos campos ['campo'], a chave do dict é um submodelo; declare a chave como texto ou "
        "número, ou o campo como lista de pares" in aviso
    )
    assert "declare como list ou tuple" not in aviso


def test_conjunto_de_enum_chega_a_saida():
    chat, saida = _executa(ConjuntoDeEnum, ["pf"])

    assert saida["campo"].tolist() == [{Tag.PF}]
    assert chat.invocacoes == 1


def test_conjunto_de_submodelos_com_field_serializer_chega_a_saida():
    chat, saida = _executa(ConjuntoComSerializador, [{"nome": "SP"}, {"nome": "RJ"}])

    assert saida["campo"].tolist() == [[{"nome": "RJ"}, {"nome": "SP"}]]
    assert chat.invocacoes == 1


@pytest.mark.parametrize("modelo", [ConjuntoDeModelo, ConjuntoCongeladoDeModelo])
def test_resposta_sem_item_no_conjunto_de_submodelos_sai_lista_vazia(modelo):
    chat, saida = _executa(modelo, [])

    assert saida["campo"].tolist() == [[]]
    assert chat.invocacoes == 1


def test_conjunto_vazio_de_enum_segue_conjunto():
    _, saida = _executa(ConjuntoDeEnum, [])

    assert saida["campo"].tolist() == [set()]


def test_chave_de_dict_que_e_submodelo_guarda_o_type_error_como_causa():
    resposta = ChaveDeModelo(campo={Cidade(nome="SP"): 1})

    with pytest.raises(ProviderAbortError, match="unhashable type: 'dict'") as erro:
        dump_response(resposta)

    assert isinstance(erro.value.__cause__, TypeError)


def test_conjunto_de_submodelos_dentro_de_outro_campo_interrompe_com_a_saida_dele():
    resposta = ConjuntoAninhado(sub=ConjuntoDeModelo(campo={Cidade(nome="SP")}), campo="x")

    with pytest.raises(ProviderAbortError) as erro:
        dump_response(resposta)

    mensagem = str(erro.value)
    assert mensagem.startswith(
        "O model_dump da resposta caiu com TypeError: unhashable type: 'dict'. O modelo "
        "ConjuntoAninhado tem um set, um frozenset ou uma chave de dict cujos itens viram dict "
        "no dump, como os de submodelos, e o dict não é hashable. Repetir a linha não resolve: "
    )
    assert mensagem.endswith(
        "declare como list ou tuple o conjunto que fica dentro de outro campo, e como texto ou "
        "número a chave de dict"
    )


def test_conjunto_de_submodelos_ao_lado_de_chave_que_e_submodelo_interrompe_pela_chave():
    resposta = ConjuntoEChave(conjunto={Cidade(nome="SP")}, chaves={Cidade(nome="RJ"): 1})

    with pytest.raises(ProviderAbortError, match=r"nos campos \['chaves'\], a chave do dict"):
        dump_response(resposta)


def test_dump_do_conjunto_de_submodelos_sai_na_ordem_do_modelo_e_do_texto_dos_itens():
    resposta = VariosCampos(
        antes="a",
        conjunto=frozenset({Cidade(nome="SP"), Cidade(nome="BH"), Cidade(nome="RJ")}),
        depois=[1, 2],
        outro={Cidade(nome="X")},
        escondido={Cidade(nome="Y")},
        misto={Cidade(nome="Z"), "texto"},
        opcional=None,
    )

    dump = dump_response(resposta)

    assert list(dump) == ["antes", "conjunto", "depois", "outro", "misto", "opcional"]
    assert dump["conjunto"] == [{"nome": "BH"}, {"nome": "RJ"}, {"nome": "SP"}]
    assert dump["outro"] == [{"nome": "X"}]
    assert dump["misto"] == ["texto", {"nome": "Z"}]
    assert dump["opcional"] is None
    assert (dump["antes"], dump["depois"]) == ("a", [1, 2])


def test_dump_do_conjunto_de_submodelos_com_conjunto_dentro_do_item():
    resposta = ConjuntoDeRegioes(
        campo={Regiao(nome="Sudeste", cidades=frozenset({Cidade(nome="SP"), Cidade(nome="RJ")}))}
    )

    assert dump_response(resposta) == {
        "campo": [{"nome": "Sudeste", "cidades": [{"nome": "RJ"}, {"nome": "SP"}]}]
    }


def test_dump_mantem_o_campo_extra_do_modelo_que_o_aceita():
    resposta = ComExtra.model_validate({"campo": [{"nome": "SP"}], "a_mais": 1})

    assert dump_response(resposta) == {"campo": [{"nome": "SP"}], "a_mais": 1}


def test_type_error_de_outra_origem_no_dump_sem_o_conjunto_segue_como_veio():
    resposta = DumpQueCaiSemOConjunto(campo={Cidade(nome="SP")})

    with pytest.raises(TypeError, match="model_dump do usuário caiu sem o conjunto"):
        dump_response(resposta)


def test_type_error_de_outra_origem_segue_como_veio():
    with pytest.raises(TypeError, match="model_dump do usuário caiu com x"):
        dump_response(DumpQueCai(campo="x"))


def test_agente_de_busca_devolve_o_conjunto_de_submodelos_como_lista():
    agente = MagicMock()
    agente.invoke.return_value = {
        "structured_response": ConjuntoDeModelo.model_validate(
            {"campo": [{"nome": "SP"}, {"nome": "RJ"}]}
        ),
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
    ):
        saida = dataframeit(
            pd.DataFrame({"texto": ["a", "b"]}),
            ConjuntoDeModelo,
            "Responda {texto}",
            use_search=True,
            track_tokens=False,
        )

    assert agente.invoke.call_count == 2
    assert saida["campo"].tolist() == [[{"nome": "RJ"}, {"nome": "SP"}]] * 2


# =============================================================================
# Retomada
# =============================================================================


class Evento(BaseModel):
    model_config = ConfigDict(frozen=True)
    nome: str
    data: Optional[dt.date] = None


class ConjuntoDeEventos(BaseModel):
    campo: frozenset[Evento]


class ListaDeEventos(BaseModel):
    campo: list[Evento]


class ChaveQueViraModelo(BaseModel):
    campo: dict[Cidade, int]

    @field_validator("campo", mode="before")
    @classmethod
    def _chave_em_modelo(cls, valor):
        return {Cidade(nome=str(nome)): numero for nome, numero in valor.items()}


class ChatEmFila(ChatFalso):
    """Chat que responde uma resposta da fila por chamada; a que for exceção é levantada."""

    @override
    def invoke(self, mensagens, config=None):
        self.invocacoes += 1
        resposta = self.resposta.pop(0)
        if isinstance(resposta, Exception):
            raise resposta
        return {
            "raw": None,
            "parsed": self.modelo.model_validate({"campo": resposta}),
            "parsing_error": None,
        }


def _roda_em_fila(df, modelo, respostas, **opcoes):
    chat = ChatEmFila(modelo, list(respostas))
    with (
        patch("dataframeit.llm.chat_model", return_value=chat),
        patch("dataframeit.core.validate_provider_dependencies"),
    ):
        saida = dataframeit(
            df, questions=modelo, prompt="{texto}", provider="openai", track_tokens=False, **opcoes
        )
    return chat, saida


def _entrada(tipo):
    textos = {"texto": ["a", "b", "c"]}
    if tipo == "polars":
        return pytest.importorskip("polars").DataFrame(textos)
    return pd.DataFrame(textos)


_EVENTOS = [{"nome": "posse", "data": "2024-01-25"}, {"nome": "ato"}]
_EVENTOS_NA_SAIDA = [
    {"nome": "ato", "data": None},
    {"nome": "posse", "data": dt.date(2024, 1, 25)},
]


@pytest.mark.parametrize("ext", [".csv", ".xlsx", ".parquet"])
@pytest.mark.parametrize("entrada", ["pandas", "polars"])
def test_retomada_do_checkpoint_devolve_o_conjunto_de_submodelos_como_lista(tmp_path, entrada, ext):
    if ext != ".csv":
        pytest.importorskip("openpyxl" if ext == ".xlsx" else "pyarrow")
    opcoes = {"checkpoint_path": str(tmp_path / f"ck{ext}"), "batch_size": 1}
    # Fora do pytest.warns: sem o polars, o teste é pulado aqui.
    primeira_entrada = _entrada(entrada)
    with pytest.warns(UserWarning, match="Execução interrompida"):
        _roda_em_fila(
            primeira_entrada,
            ConjuntoDeEventos,
            [_EVENTOS, [], ProviderAbortError("limite")],
            **opcoes,
        )

    chat, saida = _roda_em_fila(_entrada(entrada), ConjuntoDeEventos, [[{"nome": "fim"}]], **opcoes)

    assert chat.invocacoes == 1
    primeira, _, terceira = list(saida["campo"])
    assert list(primeira) == _EVENTOS_NA_SAIDA
    assert list(terceira) == [{"nome": "fim", "data": None}]


def test_retomada_em_memoria_mantem_o_conjunto_de_submodelos_como_lista():
    with pytest.warns(UserWarning, match="Execução interrompida"):
        _, interrompida = _roda_em_fila(
            _entrada("pandas"), ConjuntoDeEventos, [_EVENTOS, [], ProviderAbortError("limite")]
        )

    chat, saida = _roda_em_fila(interrompida, ConjuntoDeEventos, [[{"nome": "fim"}]])

    assert chat.invocacoes == 1
    assert saida["campo"].tolist() == [_EVENTOS_NA_SAIDA, [], [{"nome": "fim", "data": None}]]


def test_retomada_com_a_anotacao_trocada_de_lista_para_conjunto_nao_cai():
    with pytest.warns(UserWarning, match="Execução interrompida"):
        _, interrompida = _roda_em_fila(
            _entrada("pandas"), ListaDeEventos, [_EVENTOS, [], ProviderAbortError("limite")]
        )

    chat, saida = _roda_em_fila(interrompida, ConjuntoDeEventos, [[{"nome": "fim"}]])

    assert chat.invocacoes == 1
    # A lista relida fica na ordem em que foi gravada.
    assert saida["campo"].tolist()[0] == [
        {"nome": "posse", "data": dt.date(2024, 1, 25)},
        {"nome": "ato", "data": None},
    ]


def test_retomada_com_chave_de_dict_que_e_submodelo_falha_com_a_mensagem_do_dump():
    processada = pd.DataFrame(
        {
            "texto": ["a"],
            "campo": pd.Series([{"SP": 1}], dtype=object),
            "_dataframeit_status": ["processed"],
        }
    )

    with (
        patch("dataframeit.core.validate_provider_dependencies"),
        pytest.raises(
            ProviderAbortError,
            match=r"O modelo \w+ tem um set, um frozenset ou uma chave de dict.*"
            r"nos campos \['campo'\], a chave do dict é um submodelo",
        ),
    ):
        dataframeit(processada, questions=ChaveQueViraModelo, prompt="{texto}", track_tokens=False)


def test_dump_do_conjunto_vazio_sai_lista_so_no_campo_de_conjunto_de_submodelos():
    class Vazios(BaseModel):
        de_modelo: set[Cidade]
        opcional: Optional[frozenset[Cidade]]
        ausente: Optional[frozenset[Cidade]] = None
        de_texto: set[str]
        lista: list[Cidade]
        com_itens: set[Tag]

    resposta = Vazios(
        de_modelo=set(), opcional=frozenset(), de_texto=set(), lista=[], com_itens={Tag.PF}
    )

    assert dump_response(resposta) == {
        "de_modelo": [],
        "opcional": [],
        "ausente": None,
        "de_texto": set(),
        "lista": [],
        "com_itens": {Tag.PF},
    }
    assert type(dump_response(resposta)["de_texto"]) is set


def test_conjunto_com_serializador_que_devolve_conjunto_vazio_fica_como_o_serializador_deu():
    class SerializaEmTupla(BaseModel):
        campo: set[Cidade]

        @field_serializer("campo")
        def _em_tupla(self, valor: set[Cidade]) -> tuple[str, ...]:
            return tuple(sorted(cidade.nome for cidade in valor))

    assert dump_response(SerializaEmTupla(campo=set())) == {"campo": ()}


def test_campo_com_serializador_proprio_sai_por_ele_ao_lado_do_conjunto_que_cai():
    class DoisConjuntos(BaseModel):
        com_serializador: set[Cidade]
        fora_do_dump: set[Cidade] = Field(exclude_if=lambda valor: len(valor) == 1)
        sem: set[Cidade]

        @field_serializer("com_serializador")
        def _em_nomes(self, valor: set[Cidade]) -> list[str]:
            return sorted(cidade.nome for cidade in valor)

    cidades = {Cidade(nome="SP")}

    assert dump_response(
        DoisConjuntos(com_serializador=cidades, fora_do_dump=cidades, sem=cidades)
    ) == {"com_serializador": ["SP"], "sem": [{"nome": "SP"}]}


@pytest.mark.parametrize(
    "anotacao",
    [AbstractSet[Cidade], Union[set[Cidade], str], Optional[frozenset[Cidade]]],
    ids=["abstrato", "uniao", "opcional"],
)
def test_conjunto_vazio_sai_lista_em_toda_anotacao_de_conjunto_de_submodelos(anotacao):
    modelo = create_model("Modelo", campo=(anotacao, ...))

    assert dump_response(modelo.model_validate({"campo": frozenset()})) == {"campo": []}
    assert dump_response(modelo.model_validate({"campo": frozenset({Cidade(nome="SP")})})) == {
        "campo": [{"nome": "SP"}]
    }


def test_conjunto_so_com_itens_que_nao_sao_submodelo_sai_lista_no_campo_que_os_admite():
    class Misto(BaseModel):
        campo: set[Union[Cidade, str]]

    assert dump_response(Misto(campo={"b", "a"})) == {"campo": ["a", "b"]}


def test_texto_do_campo_que_tambem_admite_conjunto_de_submodelos_fica_como_veio():
    modelo = create_model("Modelo", campo=(Union[set[Cidade], str], ...))

    assert dump_response(modelo.model_validate({"campo": "texto"})) == {"campo": "texto"}


def test_campo_fora_do_modelo_com_conjunto_fica_como_veio():
    resposta = ComExtra.model_validate({"campo": [], "a_mais": {1, 2}})

    assert dump_response(resposta) == {"campo": [], "a_mais": {1, 2}}


def test_conjunto_cujo_item_o_json_nao_escreve_sai_na_ordem_em_que_veio():
    class ComChaveDeTupla(BaseModel):
        model_config = ConfigDict(frozen=True)
        pares: dict[tuple[int, int], str]

        @override
        def __hash__(self) -> int:
            return 1

    class Modelo(BaseModel):
        campo: set[ComChaveDeTupla]

    resposta = Modelo(campo={ComChaveDeTupla(pares={(1, 2): "a"})})

    assert dump_response(resposta) == {"campo": [{"pares": {(1, 2): "a"}}]}


def test_type_error_de_outra_origem_no_dump_de_um_campo_segue_como_veio():
    class CaiNoCampo(BaseModel):
        campo: set[Cidade]

        @override
        def model_dump(self, **kwargs: Any) -> dict[str, Any]:
            if kwargs.get("include"):
                msg = "model_dump do usuário caiu no campo"
                raise TypeError(msg)
            return super().model_dump(**kwargs)

    with pytest.raises(TypeError, match="model_dump do usuário caiu no campo"):
        dump_response(CaiNoCampo(campo={Cidade(nome="SP")}))


def test_item_cujo_serializador_devolve_um_escalar_entra_na_lista_como_veio():
    class PeloNome(BaseModel):
        model_config = ConfigDict(frozen=True)
        nome: str

        @model_serializer
        def _so_o_nome(self) -> str:
            return self.nome

    class Mistura(BaseModel):
        campo: set[Union[Cidade, PeloNome]]

    resposta = Mistura(campo={Cidade(nome="SP"), PeloNome(nome="RJ")})

    assert dump_response(resposta) == {"campo": ["RJ", {"nome": "SP"}]}


class ComSerializadorEmVolta(BaseModel):
    campo: set[Cidade]

    @model_serializer(mode="wrap")
    def _como_veio(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        return handler(self)


class AninhadoComSerializadorEmVolta(BaseModel):
    sub: ConjuntoDeModelo

    @model_serializer(mode="wrap")
    def _como_veio(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        return handler(self)


def test_conjunto_de_submodelos_sai_lista_com_o_model_serializer_em_volta():
    resposta = ComSerializadorEmVolta(campo={Cidade(nome="SP"), Cidade(nome="RJ")})

    assert dump_response(resposta) == {"campo": [{"nome": "RJ"}, {"nome": "SP"}]}


def test_conjunto_aninhado_com_o_model_serializer_em_volta_interrompe():
    resposta = AninhadoComSerializadorEmVolta(sub=ConjuntoDeModelo(campo={Cidade(nome="SP")}))

    with pytest.raises(ProviderAbortError, match="unhashable type: 'dict'") as erro:
        dump_response(resposta)

    assert isinstance(erro.value.__cause__, PydanticSerializationError)


def test_erro_de_serializacao_de_outra_origem_segue_como_veio():
    class SerializadorQueCai(BaseModel):
        campo: str

        @model_serializer(mode="wrap")
        def _cai(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
            msg = "serializador do usuário caiu"
            raise ValueError(msg)

    with pytest.raises(PydanticSerializationError, match="serializador do usuário caiu"):
        dump_response(SerializadorQueCai(campo="x"))


@pytest.mark.parametrize("campos", ["campo", "*"])
def test_conjunto_que_o_serializador_do_campo_devolve_fica_conjunto(campos):
    class SerializaEmConjunto(BaseModel):
        campo: set[Cidade]

        @field_serializer(campos)
        def _em_nomes(self, valor: set[Cidade]) -> set[str]:
            return {cidade.nome for cidade in valor}

    assert dump_response(SerializaEmConjunto(campo={Cidade(nome="SP")})) == {"campo": {"SP"}}
    assert dump_response(SerializaEmConjunto(campo=set())) == {"campo": set()}


def test_serializador_de_outro_campo_nao_tira_a_lista_do_conjunto_vazio():
    class OutroSerializado(BaseModel):
        campo: set[Cidade]
        outro: str

        @field_serializer("outro")
        def _em_maiusculas(self, valor: str) -> str:
            return valor.upper()

    assert dump_response(OutroSerializado(campo=set(), outro="x")) == {"campo": [], "outro": "X"}
