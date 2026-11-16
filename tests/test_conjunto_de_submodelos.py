"""Campo `set` ou `frozenset` de submodelos sai como lista, na resposta e na retomada.

O `model_dump` monta o conjunto com o dump de cada item, e o dict do
submodelo não é hashable: o dump cai com `TypeError: unhashable type: 'dict'`.
O campo que é um conjunto de submodelos sai como lista, com o dump de cada
item. O que sobra, como a chave de dict que é submodelo, vem do modelo e se
repete em toda linha, e por isso interrompe a execução na primeira resposta,
sem nova tentativa.
"""

import dataclasses
import datetime as dt
from collections import deque
from collections.abc import Set as AbstractSet
from enum import Enum
from typing import Annotated, Any, Optional, Union
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    SerializeAsAny,
    SerializerFunctionWrapHandler,
    WrapSerializer,
    create_model,
    field_serializer,
    field_validator,
    model_serializer,
)
from pydantic.dataclasses import dataclass as pydantic_dataclass
from pydantic_core import PydanticSerializationError
from typing_extensions import override

from dataframeit import dataframeit
from dataframeit.errors import (
    ProviderAbortError,
    _fields_holding_any,
    _models_dumped,
    dump_response,
)


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


class ConjuntoComAlias(BaseModel):
    model_config = ConfigDict(serialize_by_alias=True)
    rotulo: str = Field(alias="apelido")
    campo: set[Cidade]


def test_dump_do_conjunto_de_submodelos_sai_pelo_nome_dos_campos_com_alias():
    resposta = ConjuntoComAlias(apelido="a", campo={Cidade(nome="SP")})

    assert dump_response(resposta) == {"rotulo": "a", "campo": [{"nome": "SP"}]}


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


# =============================================================================
# Conjunto de submodelos dentro de outro campo
# =============================================================================


class Capital(Cidade):
    uf: str


class Pais(BaseModel):
    nome: str
    regioes: list[Regiao]


class ListaDeCidades(BaseModel):
    campo: list[Cidade]


class ConjuntoDaUniao(BaseModel):
    campo: set[Union[Cidade, Capital]]


class CampoQualquer(BaseModel):
    campo: Any


class Colecoes(BaseModel):
    em_lista: list[set[Cidade]]
    em_tupla: tuple[frozenset[Cidade], int]
    em_dict: dict[str, set[Cidade]]
    em_conjunto: frozenset[Regiao]
    em_modelo: ConjuntoDeModelo
    em_lista_de_modelo: list[ConjuntoDeModelo]
    ao_lado: str


def test_conjunto_de_submodelos_dentro_de_outro_campo_sai_como_lista():
    sp, rj = Cidade(nome="SP"), Cidade(nome="RJ")
    resposta = Colecoes(
        em_lista=[{sp, rj}, set()],
        em_tupla=(frozenset({sp}), 7),
        em_dict={"sudeste": {rj, sp}},
        em_conjunto=frozenset({Regiao(nome="Sudeste", cidades=frozenset({sp, rj}))}),
        em_modelo=ConjuntoDeModelo(campo={rj, sp}),
        em_lista_de_modelo=[ConjuntoDeModelo(campo={sp}), ConjuntoDeModelo(campo=set())],
        ao_lado="x",
    )
    em_ordem = [{"nome": "RJ"}, {"nome": "SP"}]

    dump = dump_response(resposta)

    assert dump == {
        "em_lista": [em_ordem, []],
        "em_tupla": ([{"nome": "SP"}], 7),
        "em_dict": {"sudeste": em_ordem},
        "em_conjunto": [{"nome": "Sudeste", "cidades": em_ordem}],
        "em_modelo": {"campo": em_ordem},
        "em_lista_de_modelo": [{"campo": [{"nome": "SP"}]}, {"campo": []}],
        "ao_lado": "x",
    }
    assert list(dump) == list(Colecoes.model_fields)
    assert type(dump["em_tupla"]) is tuple
    assert type(dump["em_lista"]) is list


def test_conjunto_de_submodelos_em_tres_niveis_sai_como_lista():
    regiao = Regiao(nome="Sul", cidades=frozenset({Cidade(nome="POA")}))

    assert dump_response(Pais(nome="BR", regioes=[regiao])) == {
        "nome": "BR",
        "regioes": [{"nome": "Sul", "cidades": [{"nome": "POA"}]}],
    }


def test_conjunto_aninhado_chega_a_saida_como_lista():
    sub = ConjuntoDeModelo(campo={Cidade(nome="SP"), Cidade(nome="BH")})

    chat, saida = _executa_resposta(ConjuntoAninhado(sub=sub, campo="x"))

    assert chat.invocacoes == 1
    assert saida["sub"].tolist() == [{"campo": [{"nome": "BH"}, {"nome": "SP"}]}]


def test_conjunto_aninhado_com_o_model_serializer_em_volta_sai_como_lista():
    resposta = AninhadoComSerializadorEmVolta(sub=ConjuntoDeModelo(campo={Cidade(nome="SP")}))

    assert dump_response(resposta) == {"sub": {"campo": [{"nome": "SP"}]}}


def test_conjunto_vazio_dentro_de_outro_campo_sai_como_lista():
    """O dump que não cai sai igual ao que cai: a lista não depende das outras células da linha."""
    resposta = Colecoes(
        em_lista=[set()],
        em_tupla=(frozenset(), 7),
        em_dict={"vazio": set()},
        em_conjunto=frozenset(),
        em_modelo=ConjuntoDeModelo(campo=set()),
        em_lista_de_modelo=[ConjuntoDeModelo(campo=set())],
        ao_lado="x",
    )

    assert dump_response(resposta) == {
        "em_lista": [[]],
        "em_tupla": ([], 7),
        "em_dict": {"vazio": []},
        "em_conjunto": [],
        "em_modelo": {"campo": []},
        "em_lista_de_modelo": [{"campo": []}],
        "ao_lado": "x",
    }


class ConjuntosDeTexto(BaseModel):
    campo: list[set[str]]
    sub: ConjuntoDeEnum


def test_conjunto_dentro_de_outro_campo_sem_submodelo_segue_conjunto():
    resposta = ConjuntosDeTexto(campo=[{"a"}], sub=ConjuntoDeEnum(campo={Tag.PF}))

    assert dump_response(resposta) == {"campo": [{"a"}], "sub": {"campo": {Tag.PF}}}


class No(BaseModel):
    model_config = ConfigDict(frozen=True)
    nome: str
    filhos: frozenset["No"] = frozenset()


class Arvore(BaseModel):
    raiz: No


def test_conjunto_do_modelo_recursivo_sai_como_lista_em_todo_nivel():
    folha = No(nome="b")

    assert dump_response(Arvore(raiz=No(nome="a", filhos=frozenset({folha})))) == {
        "raiz": {"nome": "a", "filhos": [{"nome": "b", "filhos": []}]}
    }
    assert dump_response(Arvore(raiz=folha)) == {"raiz": {"nome": "b", "filhos": []}}


class SemConjuntoDeModelo(BaseModel):
    par: tuple[Cidade, set[str]]
    filhos: list["SemConjuntoDeModelo"] = []


def test_campo_com_submodelo_e_sem_conjunto_de_submodelos_fica_com_o_model_dump():
    folha = SemConjuntoDeModelo(par=(Cidade(nome="RJ"), {"b"}))
    resposta = SemConjuntoDeModelo(par=(Cidade(nome="SP"), {"a"}), filhos=[folha])

    assert dump_response(resposta) == {
        "par": ({"nome": "SP"}, {"a"}),
        "filhos": [{"par": ({"nome": "RJ"}, {"b"}), "filhos": []}],
    }


class VariosFundos(BaseModel):
    com_serializador: list[set[Cidade]]
    cheio: list[set[Cidade]]
    vazio: list[set[Cidade]]
    outro_vazio: list[set[Cidade]]

    @field_serializer("com_serializador")
    def _so_o_tamanho(self, valor: list[set[Cidade]]) -> int:
        return len(valor)


def test_cada_campo_com_conjunto_fundo_sai_lista_depois_do_que_ja_saiu_e_do_serializado():
    resposta = VariosFundos(
        com_serializador=[set()], cheio=[{Cidade(nome="SP")}], vazio=[set()], outro_vazio=[set()]
    )
    so_vazios = VariosFundos(com_serializador=[set()], cheio=[set()], vazio=[set()], outro_vazio=[])

    assert dump_response(resposta) == {
        "com_serializador": 1,
        "cheio": [[{"nome": "SP"}]],
        "vazio": [[]],
        "outro_vazio": [[]],
    }
    assert dump_response(so_vazios) == {
        "com_serializador": 1,
        "cheio": [[]],
        "vazio": [[]],
        "outro_vazio": [],
    }


class SubComSerializador(BaseModel):
    sub: ConjuntoDeModelo

    @field_serializer("sub")
    def _so_o_tamanho(self, valor: ConjuntoDeModelo) -> int:
        return len(valor.campo)


class ModeloComSerializadorDoDump(BaseModel):
    sub: ConjuntoDeModelo

    @model_serializer(mode="plain")
    def _com_o_conjunto(self) -> dict[str, Any]:
        return {"sub": {"campo": set(), "a_mais": 1}}


def test_campo_com_serializador_proprio_fica_como_o_serializador_deu_ao_lado_do_conjunto_fundo():
    vazio = ConjuntoDeModelo(campo=set())

    assert dump_response(SubComSerializador(sub=vazio)) == {"sub": 0}
    assert dump_response(ModeloComSerializadorDoDump(sub=vazio)) == {
        "sub": {"campo": set(), "a_mais": 1}
    }


def test_conjunto_vazio_dentro_de_colecao_que_o_dump_simples_nao_escreve_fica_como_veio():
    modelo = create_model("ModeloDoCaso", campo=(deque[set[Cidade]], ...))

    dump = dump_response(modelo.model_validate({"campo": [[]]}))

    assert dump == {"campo": deque([set()])}


@pytest.mark.parametrize("modelo", [ConjuntoDeModelo, ConjuntoCongeladoDeModelo])
def test_item_de_subclasse_no_conjunto_sai_com_os_campos_da_classe_declarada(modelo):
    resposta = modelo(campo={Capital(nome="SP", uf="SP"), Cidade(nome="RJ")})
    em_lista = ListaDeCidades(campo=[Capital(nome="SP", uf="SP")])

    assert dump_response(resposta) == {"campo": [{"nome": "RJ"}, {"nome": "SP"}]}
    # É o que o pydantic faz com o item de subclasse numa lista.
    assert dump_response(em_lista) == {"campo": [{"nome": "SP"}]}


def test_item_de_subclasse_declarada_na_uniao_sai_com_os_campos_dela():
    resposta = ConjuntoDaUniao(campo={Capital(nome="SP", uf="SP"), Cidade(nome="RJ")})

    assert dump_response(resposta) == {"campo": [{"nome": "RJ"}, {"nome": "SP", "uf": "SP"}]}


def test_item_de_subclasse_com_conjunto_dentro_sai_pela_classe_declarada():
    class RegiaoComSigla(Regiao):
        sigla: str

    resposta = ConjuntoDeRegioes(
        campo={RegiaoComSigla(nome="Sul", sigla="S", cidades=frozenset({Cidade(nome="POA")}))}
    )

    assert dump_response(resposta) == {"campo": [{"nome": "Sul", "cidades": [{"nome": "POA"}]}]}


def test_item_que_nao_e_de_nenhuma_classe_declarada_sai_pelo_dump_dele():
    resposta = CampoQualquer(campo=[{Capital(nome="SP", uf="SP")}])

    assert dump_response(resposta) == {"campo": [[{"nome": "SP", "uf": "SP"}]]}


class ChaveAninhada(BaseModel):
    campo: list[dict[Cidade, int]]
    conjunto: set[Cidade]


def test_chave_de_dict_que_e_submodelo_dentro_de_uma_lista_interrompe():
    resposta = ChaveAninhada(campo=[{Cidade(nome="SP"): 1}], conjunto={Cidade(nome="RJ")})

    with pytest.raises(ProviderAbortError) as erro:
        dump_response(resposta)

    mensagem = str(erro.value)
    # O texto do TypeError muda com a versão do Python; o "unhashable type" fica.
    assert mensagem.startswith("O model_dump da resposta caiu com TypeError: ")
    assert "unhashable type: 'dict'" in mensagem
    assert (
        ". O modelo ChaveAninhada tem um set, um frozenset ou uma chave de dict cujos itens "
        "viram dict no dump, como os de submodelos, e o dict não é hashable. Repetir a linha "
        "não resolve: " in mensagem
    )
    assert mensagem.endswith(
        "declare como texto ou número a chave de dict que é ou guarda um submodelo, e como list "
        "ou tuple o conjunto que fica num dataclass ou numa coleção que não é lista, tupla, "
        "conjunto nem dict"
    )
    assert isinstance(erro.value.__cause__, TypeError)


class FilaDeConjuntos(BaseModel):
    campo: deque[set[Cidade]]


class ChaveDeConjunto(BaseModel):
    campo: dict[frozenset[Cidade], int]


def test_conjunto_de_submodelos_dentro_de_outra_colecao_interrompe():
    # Com a anotação: no campo `Any`, o pydantic 2.12 não desce no deque, e o dump não cai.
    resposta = FilaDeConjuntos(campo=deque([{Cidade(nome="SP")}]))

    with pytest.raises(ProviderAbortError, match="numa coleção que não é lista"):
        dump_response(resposta)


def test_conjunto_de_submodelos_em_campo_extra_interrompe():
    """O campo extra não tem anotação nem dump próprio, e o resto do dump cai com ele."""
    resposta = ComExtra(campo={Cidade(nome="SP")}, a_mais={Cidade(nome="RJ")})

    with pytest.raises(ProviderAbortError, match="unhashable type: 'dict'"):
        dump_response(resposta)


class SerializadorQueRepassa(BaseModel):
    campo: list[set[Cidade]]

    @field_serializer("campo", mode="wrap")
    def _como_veio(self, valor: Any, handler: SerializerFunctionWrapHandler) -> Any:
        return handler(valor)


def test_conjunto_no_campo_com_serializador_que_repassa_sai_como_lista():
    """O serializador que só repassa o valor cai como o dump sem ele, e o campo sai pela lista."""
    resposta = SerializadorQueRepassa(campo=[{Cidade(nome="SP")}])

    assert dump_response(resposta) == {"campo": [[{"nome": "SP"}]]}


@pydantic_dataclass(frozen=True)
class Ponto:
    x: int


@pydantic_dataclass(frozen=True)
class Grupo:
    cidades: frozenset[Cidade]


@pytest.mark.parametrize(
    "resposta",
    [
        CampoQualquer(campo={Ponto(x=1)}),
        CampoQualquer(campo=[Grupo(cidades=frozenset({Cidade(nome="SP")}))]),
        ChaveDeConjunto(campo={frozenset({Cidade(nome="SP")}): 1}),
        CampoQualquer(campo=[{(Cidade(nome="SP"), 1): {Cidade(nome="RJ")}}]),
        CampoQualquer(campo=[{(Ponto(x=1), 1): {Cidade(nome="RJ")}}]),
    ],
    ids=["dataclass", "conjunto-no-dataclass", "chave-conjunto", "chave-tupla", "chave-dataclass"],
)
def test_valor_que_o_dump_simples_nao_escreve_interrompe(resposta):
    """O dataclass e a chave que guarda um submodelo sairiam como objeto, sem o dict do dump."""
    with pytest.raises(ProviderAbortError, match="unhashable type"):
        dump_response(resposta)


def test_chave_de_tupla_sem_submodelo_segue_chave_ao_lado_do_conjunto():
    resposta = CampoQualquer(campo={(1, "a"): {Cidade(nome="SP")}})

    assert dump_response(resposta) == {"campo": {(1, "a"): [{"nome": "SP"}]}}


class ConjuntoComoQualquer(BaseModel):
    campo: set[SerializeAsAny[Cidade]]
    em_lista: list[set[SerializeAsAny[Cidade]]]
    campo_inteiro: SerializeAsAny[set[Cidade]]


def test_item_de_subclasse_com_serialize_as_any_sai_com_os_campos_dela():
    capital = Capital(nome="SP", uf="SP")
    resposta = ConjuntoComoQualquer(campo={capital}, em_lista=[{capital}], campo_inteiro={capital})

    assert dump_response(resposta) == {
        "campo": [{"nome": "SP", "uf": "SP"}],
        "em_lista": [[{"nome": "SP", "uf": "SP"}]],
        "campo_inteiro": [{"nome": "SP", "uf": "SP"}],
    }


class NoComConjunto(BaseModel):
    conjunto: set[Cidade] = set()
    filhos: list["NoComConjunto"] = []


def test_conjunto_vazio_do_modelo_que_guarda_a_si_mesmo_sai_como_lista():
    vazio = dump_response(NoComConjunto(filhos=[NoComConjunto()]))
    cheio = dump_response(NoComConjunto(filhos=[NoComConjunto(conjunto={Cidade(nome="SP")})]))

    assert vazio == {"conjunto": [], "filhos": [{"conjunto": [], "filhos": []}]}
    assert cheio == {"conjunto": [], "filhos": [{"conjunto": [{"nome": "SP"}], "filhos": []}]}


class SerializadorNoAnnotated(BaseModel):
    simples: Annotated[list[set[Cidade]], PlainSerializer(lambda valor: "serializado")]
    em_volta: Annotated[
        list[set[Cidade]], WrapSerializer(lambda valor, repassa: len(valor), return_type=int)
    ]
    direto: Annotated[set[Cidade], PlainSerializer(lambda valor: set())]


class SerializadorSoDoJson(BaseModel):
    simples: Annotated[list[set[Cidade]], PlainSerializer(lambda valor: "x", when_used="json")]
    em_volta: Annotated[
        list[set[Cidade]],
        WrapSerializer(lambda valor, repassa: "x", when_used="json-unless-none"),
    ]
    sem_none: Annotated[
        list[set[Cidade]], PlainSerializer(lambda valor: "x", when_used="unless-none")
    ]


def test_serializador_so_do_json_nao_tira_a_lista_do_conjunto_vazio():
    resposta = SerializadorSoDoJson(simples=[set()], em_volta=[set()], sem_none=[set()])

    assert dump_response(resposta) == {"simples": [[]], "em_volta": [[]], "sem_none": "x"}


def test_campo_com_serializador_no_annotated_fica_como_o_serializador_deu():
    resposta = SerializadorNoAnnotated(simples=[set()], em_volta=[set()], direto=set())

    assert dump_response(resposta) == {"simples": "serializado", "em_volta": 1, "direto": set()}


class ChaveQueViraModeloPorCampo(BaseModel):
    antes: str
    campo: dict[Cidade, int]

    @field_validator("campo", mode="before")
    @classmethod
    def _chave_em_modelo(cls, valor):
        return {Cidade(nome=str(nome)): numero for nome, numero in valor.items()}


def test_modo_por_campo_interrompe_quando_o_dump_da_linha_cai():
    respostas = {"antes": "x", "campo": {"SP": 1}}

    def responde(text, model, *args, **kwargs):
        return {"data": {campo: respostas[campo] for campo in model.model_fields}, "usage": {}}

    with (
        patch("dataframeit.agent.call_agent", side_effect=responde) as chamadas,
        patch("dataframeit.core.validate_provider_dependencies"),
        patch("dataframeit.core.validate_search_dependencies"),
        pytest.warns(UserWarning, match="Execução interrompida.*a chave do dict é um submodelo"),
    ):
        saida = dataframeit(
            pd.DataFrame({"texto": ["a", "b"]}),
            questions=ChaveQueViraModeloPorCampo,
            prompt="Analise {texto}",
            use_search=True,
            search_per_field=True,
            track_tokens=False,
        )

    # Só a primeira linha foi ao provider, e nenhuma ficou com status de erro.
    assert chamadas.call_count == len(respostas)
    assert saida["_dataframeit_status"].isna().all()


class ChaveQueViraModeloComCondicao(BaseModel):
    campo: dict[Cidade, int]
    tem: bool
    depois: Optional[str] = Field(
        None, json_schema_extra={"condition": {"field": "tem", "equals": True}}
    )

    @field_validator("campo", mode="before")
    @classmethod
    def _chave_em_modelo(cls, valor):
        return {Cidade(nome=str(nome)): numero for nome, numero in valor.items()}


def test_modo_por_campo_interrompe_antes_das_chamadas_que_faltam_na_linha():
    """A condição lê a linha validada depois de cada campo, e o dump que cai ali já interrompe."""
    respostas = {"campo": {"SP": 1}, "tem": True, "depois": "x"}
    pedidos = []

    def responde(text, model, *args, **kwargs):
        pedidos.append(list(model.model_fields))
        return {"data": {campo: respostas[campo] for campo in model.model_fields}, "usage": {}}

    with (
        patch("dataframeit.agent.call_agent", side_effect=responde),
        patch("dataframeit.core.validate_provider_dependencies"),
        patch("dataframeit.core.validate_search_dependencies"),
        pytest.warns(UserWarning, match="Execução interrompida.*a chave do dict é um submodelo"),
    ):
        saida = dataframeit(
            pd.DataFrame({"texto": ["a", "b"]}),
            questions=ChaveQueViraModeloComCondicao,
            prompt="Analise {texto}",
            use_search=True,
            search_per_field=True,
            track_tokens=False,
        )

    assert ["depois"] not in pedidos
    assert len(pedidos) < len(respostas)
    assert saida["_dataframeit_status"].isna().all()


# =============================================================================
# Submodelo que o model_dump deixa como objeto num campo `Any`
# =============================================================================


@pytest.mark.parametrize(
    ("valor", "esperado"),
    [
        (deque([Cidade(nome="SP")]), deque([{"nome": "SP"}])),
        ((deque([Cidade(nome="SP")]), 1), (deque([{"nome": "SP"}]), 1)),
        (
            [{"k": deque([Capital(nome="SP", uf="SP")])}],
            [{"k": deque([{"nome": "SP", "uf": "SP"}])}],
        ),
    ],
)
def test_submodelo_dentro_de_deque_num_campo_any_sai_pelo_dump_dele(valor, esperado):
    """No pydantic 2.12 o item do `deque` saía como objeto; no 2.14, pelo dump dele."""
    assert dump_response(CampoQualquer(campo=valor)) == {"campo": esperado}


def test_deque_de_um_campo_any_mantem_o_tamanho_maximo():
    saida = dump_response(CampoQualquer(campo=deque([Cidade(nome="SP")], maxlen=3)))

    assert saida["campo"].maxlen == 3


@pytest.mark.parametrize(
    "valor",
    [
        {(Cidade(nome="SP"), 2): "v"},
        {Cidade(nome="SP"): "v"},
        deque([{Cidade(nome="SP")}]),
        deque([frozenset({Cidade(nome="SP")})]),
    ],
    ids=["chave_composta", "chave_submodelo", "conjunto_no_deque", "frozenset_no_deque"],
)
def test_chave_e_conjunto_com_submodelo_num_campo_any_interrompem(valor):
    """No pydantic 2.12 o model_dump não cai com eles, e o valor sairia como objeto."""
    with pytest.raises(ProviderAbortError, match="unhashable type: 'dict'"):
        dump_response(CampoQualquer(campo=valor))


@pytest.mark.parametrize(
    ("valor", "esperado"),
    [
        (Cidade(nome="SP"), {"nome": "SP"}),
        ([Cidade(nome="SP"), 1], [{"nome": "SP"}, 1]),
        ((Cidade(nome="SP"),), ({"nome": "SP"},)),
        ({"k": Cidade(nome="SP")}, {"k": {"nome": "SP"}}),
        (deque([Cidade(nome="SP")]), deque([{"nome": "SP"}])),
        ({1, "a"}, {1, "a"}),
        (frozenset({1}), frozenset({1})),
        ("texto", "texto"),
    ],
)
def test_models_dumped_escreve_o_submodelo_que_sobrou_como_objeto(valor, esperado):
    saida = _models_dumped(valor, CampoQualquer(campo=None))

    assert saida == esperado
    assert type(saida) is type(esperado)


class _FilaPropria(deque[object]):
    pass


@dataclasses.dataclass(frozen=True)
class _Registro:
    x: int


def test_models_dumped_devolve_deque_da_subclasse_de_deque():
    saida = _models_dumped(_FilaPropria([Cidade(nome="SP")]), CampoQualquer(campo=None))

    assert type(saida) is deque
    assert saida == deque([{"nome": "SP"}])


@pytest.mark.parametrize("valor", [{_Registro}, {_Registro: 1}, {(_Registro, 1): 2}])
def test_models_dumped_deixa_como_esta_a_classe_de_dataclass_em_conjunto_ou_chave(valor):
    assert _models_dumped(valor, CampoQualquer(campo=None)) == valor
    assert dump_response(CampoQualquer(campo=valor)) == {"campo": valor}


def test_submodelo_com_conjunto_de_submodelos_dentro_de_deque_interrompe():
    """No pydantic 2.14 o model_dump do item cai; no 2.12 o item saía como objeto."""
    resposta = CampoQualquer(campo=deque([ConjuntoDeModelo(campo={Cidade(nome="SP")})]))

    with pytest.raises(ProviderAbortError, match="unhashable type: 'dict'"):
        dump_response(resposta)


def test_models_dumped_interrompe_no_submodelo_cujo_dump_cai():
    with pytest.raises(
        ProviderAbortError, match=r"CampoQualquer.*unhashable|unhashable.*CampoQualquer"
    ):
        _models_dumped(ConjuntoDeModelo(campo={Cidade(nome="SP")}), CampoQualquer(campo=None))


def test_submodelo_com_conjunto_vazio_dentro_de_deque_sai_com_o_conjunto():
    resposta = CampoQualquer(campo=deque([ConjuntoDeModelo(campo=set())]))

    assert dump_response(resposta) == {"campo": deque([{"campo": set()}])}


def test_models_dumped_repassa_o_erro_do_dump_que_nao_e_de_hash():
    class _DumpQuebrado(BaseModel):
        x: int

        @model_serializer
        def _quebra(self) -> dict[str, int]:
            msg = "outro erro"
            raise ValueError(msg)

    with pytest.raises((ValueError, PydanticSerializationError), match="outro erro"):
        _models_dumped(_DumpQuebrado(x=1), CampoQualquer(campo=None))


@pytest.mark.parametrize(
    "valor",
    [
        {Cidade(nome="SP"): 1},
        {(1, Cidade(nome="SP")): 1},
        {(1, frozenset({Cidade(nome="SP")})): 1},
        [{Cidade(nome="SP")}],
        (frozenset({Cidade(nome="SP")}),),
    ],
)
def test_models_dumped_interrompe_na_chave_e_no_conjunto_com_submodelo(valor):
    with pytest.raises(
        ProviderAbortError,
        match=r"CampoQualquer.*unhashable type: 'dict'|unhashable type: 'dict'.*CampoQualquer",
    ):
        _models_dumped(valor, CampoQualquer(campo=None))


class _SemQualquer(BaseModel):
    a: int
    b: Optional[list[Cidade]] = None
    filhos: list["_SemQualquer"] = []


class _QualquerNoSubmodelo(BaseModel):
    sub: list[CampoQualquer]


@pytest.mark.parametrize(
    ("modelo", "esperado"),
    [
        (CampoQualquer, {"campo"}),
        (create_model("ComObject", campo=(object, ...)), {"campo"}),
        (create_model("ComOpcional", campo=(Optional[Any], ...)), {"campo"}),
        (create_model("NaLista", a=(int, ...), campo=(dict[str, list[Any]], ...)), {"campo"}),
        (_QualquerNoSubmodelo, {"sub"}),
        (_SemQualquer, set()),
        (ConjuntoDeModelo, set()),
    ],
)
def test_fields_holding_any(modelo, esperado):
    assert _fields_holding_any(modelo) == esperado


def test_dump_so_refaz_o_campo_com_any():
    """O campo sem `Any` sai como o model_dump o devolveu, sem ser percorrido."""
    modelo = create_model("ListaEQualquer", lista=(list[int], ...), campo=(Any, ...))
    resposta = modelo(lista=[1], campo=[2])

    with patch("dataframeit.errors._models_dumped", side_effect=lambda valor, _: valor) as refeitos:
        saida = dump_response(resposta)

    assert saida == {"lista": [1], "campo": [2]}
    assert [chamada.args[0] for chamada in refeitos.call_args_list] == [[2]]


# =============================================================================
# Limites do dump dos conjuntos de submodelos
# =============================================================================


class _PosicoesComSerializeAsAny(BaseModel):
    conjuntos: tuple[set[SerializeAsAny[Cidade]], set[Cidade]]
    listas: tuple[list[SerializeAsAny[Cidade]], list[Cidade]]


class _PosicoesComBaseESubclasse(BaseModel):
    conjuntos: tuple[set[Cidade], set[Capital]]
    listas: tuple[list[Cidade], list[Capital]]


def test_serialize_as_any_numa_posicao_vale_para_todos_os_conjuntos_do_campo():
    """Na lista, o pydantic só escreve pela subclasse o item da posição marcada."""
    capital = Capital(nome="SP", uf="SP")
    resposta = _PosicoesComSerializeAsAny(
        conjuntos=({capital}, {capital}), listas=([capital], [capital])
    )

    saida = dump_response(resposta)

    assert saida["listas"] == ([{"nome": "SP", "uf": "SP"}], [{"nome": "SP"}])
    assert saida["conjuntos"] == ([{"nome": "SP", "uf": "SP"}], [{"nome": "SP", "uf": "SP"}])


def test_subclasse_declarada_noutra_posicao_sai_com_os_campos_dela_no_conjunto_da_base():
    """Na lista, o item de subclasse na posição da base sai só com os campos da base."""
    capital = Capital(nome="SP", uf="SP")
    resposta = _PosicoesComBaseESubclasse(
        conjuntos=({capital}, {capital}), listas=([capital], [capital])
    )

    saida = dump_response(resposta)

    assert saida["listas"] == ([{"nome": "SP"}], [{"nome": "SP", "uf": "SP"}])
    assert saida["conjuntos"] == ([{"nome": "SP", "uf": "SP"}], [{"nome": "SP", "uf": "SP"}])


_EmCentavos = Annotated[int, PlainSerializer(lambda valor: f"{valor} centavos")]


class _SerializadorDeTipoAoLadoDoConjunto(BaseModel):
    com_conjunto: tuple[set[Cidade], list[_EmCentavos]]
    com_lista: tuple[list[Cidade], list[_EmCentavos]]
    conjunto_serializado: dict[str, Annotated[set[Cidade], PlainSerializer(lambda valor: "x")]]


def test_serializador_de_tipo_dentro_do_campo_com_conjunto_de_submodelos_e_ignorado():
    """O campo cujo dump cai é escrito sem o model_dump, que é quem roda o serializador."""
    cidade = Cidade(nome="SP")
    resposta = _SerializadorDeTipoAoLadoDoConjunto(
        com_conjunto=({cidade}, [5]),
        com_lista=([cidade], [5]),
        conjunto_serializado={"k": {cidade}},
    )

    saida = dump_response(resposta)

    assert saida["com_lista"] == ([{"nome": "SP"}], ["5 centavos"])
    assert saida["com_conjunto"] == ([{"nome": "SP"}], [5])
    assert saida["conjunto_serializado"] == {"k": [{"nome": "SP"}]}
