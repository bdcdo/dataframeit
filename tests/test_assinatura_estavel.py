"""Assinatura do checkpoint igual em qualquer processo, com condição callable e conjunto."""

import functools
import json
import operator
import os
import subprocess
import sys
import textwrap
from enum import Enum
from typing import Any, Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field, create_model

from dataframeit import core, field_condition
from dataframeit.core import _run_signature


def _assinatura(modelo):
    return _run_signature(
        modelo,
        "{texto}",
        provider="openai",
        model="m",
        model_kwargs=None,
        search_config=None,
        text_column="texto",
        status_col="_dataframeit_status",
    )


def _roda_python(codigo, *argumentos, semente, cwd=None):
    """Roda `codigo` num interpretador novo, com a semente de hash dada.

    A falha traz o fim do stderr, para que a exceção do processo filho apareça.
    """
    saida = subprocess.run(  # noqa: S603 (roda o próprio interpretador com código fixo do teste)
        [sys.executable, "-c", codigo, *argumentos],
        env={**os.environ, "PYTHONHASHSEED": semente},
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    assert saida.returncode == 0, saida.stderr[-1000:]
    return json.loads(saida.stdout)


# Um modelo por forma que o JSON Schema não serializa igual em todo processo: a
# função da condição, o conjunto de texto na condição, o conjunto de texto num
# campo aninhado, o conjunto dentro de lista e o conjunto nos argumentos do
# partial e no estado do objeto chamável. O Pedido traz também Literal, Enum e
# modelo aninhado, para que toda a forma canônica dele passe pelas sementes.
_MODELOS = textwrap.dedent(
    '''
    import functools
    from enum import Enum
    from typing import Literal

    from pydantic import BaseModel, Field

    from dataframeit import field_condition


    class Porte(Enum):
        pequeno = "p"
        grande = "g"


    class Endereco(BaseModel):
        """Endereço de entrega."""

        cidade: str


    class Pedido(BaseModel):
        tipo: Literal["vip", "ouro", "comum"]
        porte: Porte | None = None
        endereco: Endereco | None = None
        desconto: float | None = Field(
            default=None,
            json_schema_extra=field_condition(
                lambda d: d.get("tipo") in {"vip", "ouro", "prata", "bronze"},
                depends_on="tipo",
            ),
        )


    class Cadastro(BaseModel):
        tipo: str
        renda: float | None = Field(
            default=None,
            json_schema_extra={"condition": {"field": "tipo", "in": {"pf", "mei", "autonomo", "rural"}}},
        )


    class Bairro(BaseModel):
        nome: str = Field(json_schema_extra={"exemplos": {"centro", "mooca", "liberdade", "lapa"}})


    class Entrega(BaseModel):
        bairro: Bairro


    class Roteiro(BaseModel):
        parada: str = Field(json_schema_extra={"exemplos": [{"centro", "mooca", "liberdade", "lapa"}]})


    def _tipo_em(d, tipos):
        return d.get("tipo") in tipos


    class Oferta(BaseModel):
        tipo: str
        desconto: float | None = Field(
            default=None,
            json_schema_extra=field_condition(
                functools.partial(_tipo_em, tipos={"vip", "ouro", "prata", "bronze"}),
                depends_on="tipo",
            ),
        )


    class TipoEm:
        def __init__(self, tipos):
            self.tipos = tipos

        def __call__(self, d):
            return d.get("tipo") in self.tipos

        # O repr traz o conjunto na ordem de hash do processo, e a assinatura não
        # pode sair dele.
        def __repr__(self):
            return f"TipoEm({self.tipos!r})"


    class Promocao(BaseModel):
        tipo: str
        desconto: float | None = Field(
            default=None,
            json_schema_extra=field_condition(
                TipoEm({"vip", "ouro", "prata", "bronze"}), depends_on="tipo"
            ),
        )
    '''
)

_NOMES = ("Pedido", "Cadastro", "Entrega", "Roteiro", "Oferta", "Promocao")


def test_assinatura_igual_em_processos_com_sementes_diferentes():
    """A assinatura não depende do endereço da função nem da ordem de hash do conjunto."""
    codigo = _MODELOS + textwrap.dedent(
        """
        import json
        from dataframeit.core import _run_signature

        print(json.dumps({
            modelo.__name__: _run_signature(
                modelo, "{texto}", provider="openai", model="m", model_kwargs=None,
                search_config=None, text_column="texto", status_col="_dataframeit_status",
            )
            for modelo in (Pedido, Cadastro, Entrega, Roteiro, Oferta, Promocao)
        }))
        """
    )

    por_semente = [_roda_python(codigo, semente=semente) for semente in ("0", "1", "2")]

    distintas = {nome: len({assinaturas[nome] for assinaturas in por_semente}) for nome in _NOMES}
    assert distintas == dict.fromkeys(_NOMES, 1)


_EXECUCAO = _MODELOS + textwrap.dedent(
    """
    import json
    import sys
    import warnings
    from unittest.mock import patch

    import pandas as pd

    from dataframeit import ProviderUsageLimitError, dataframeit


    class Remessa(BaseModel):
        tipo: str
        desconto: float | None = Field(
            default=None,
            json_schema_extra=field_condition(lambda d: d.get("tipo") == "vip", depends_on="tipo"),
        )
        frete: float | None = Field(
            default=None,
            json_schema_extra={"condition": {"field": "tipo", "in": {"vip", "ouro", "prata", "bronze"}}},
        )


    checkpoint, limite = sys.argv[1], int(sys.argv[2])
    chamadas = []


    def agente(text, model, prompt, config, save_trace=None):
        chamadas.append(text)
        if len(chamadas) > limite:
            raise ProviderUsageLimitError("cota")
        # Tipo comum em toda linha: as duas condições são falsas, e cada linha
        # custa uma chamada, a do tipo.
        return {"data": {campo: "comum" for campo in model.model_fields}, "usage": {}}


    with (
        warnings.catch_warnings(record=True) as avisos,
        patch("dataframeit.core.validate_provider_dependencies"),
        patch("dataframeit.core.validate_search_dependencies"),
        patch("dataframeit.agent.call_agent", side_effect=agente),
    ):
        warnings.simplefilter("always")
        resultado = dataframeit(
            pd.DataFrame({"texto": ["a", "b", "c", "d"]}),
            Remessa,
            "{texto}",
            use_search=True,
            search_per_field=True,
            batch_size=1,
            checkpoint_path=checkpoint,
        )
    print(json.dumps({
        "chamadas": chamadas,
        "avisos": [str(aviso.message) for aviso in avisos],
        "tipo": resultado["tipo"].tolist(),
    }))
    """
)


def test_execucao_interrompida_retoma_em_outro_processo_sem_refazer_linhas(tmp_path):
    """A cota acaba depois de 3 linhas; o outro processo, com outra semente, faz só a 4ª."""
    primeira = _roda_python(_EXECUCAO, "ckpt.csv", "3", semente="1", cwd=tmp_path)
    segunda = _roda_python(_EXECUCAO, "ckpt.csv", "100", semente="3", cwd=tmp_path)

    assert primeira["chamadas"] == ["a", "b", "c", "d"]
    assert any("interrompida" in aviso for aviso in primeira["avisos"])
    assert segunda["chamadas"] == ["d"]
    assert not any("outra configuração" in aviso for aviso in segunda["avisos"])
    assert segunda["tipo"] == ["comum"] * 4


# Modelos em que o JSON Schema serve de assinatura. As assinaturas esperadas
# foram medidas com o código anterior a esta forma canônica: a delas não muda,
# e o checkpoint já gravado com esses modelos continua sendo retomado.


class Modelo(BaseModel):
    campo1: str


class Situacao(Enum):
    ativa = "ativa"
    baixada = "baixada"


class Socio(BaseModel):
    """Sócio da empresa."""

    nome: str


class Empresa(BaseModel):
    tipo: Literal["pf", "pj"]
    situacao: Situacao | None = None
    socios: list[Socio] = []
    cnpj: str | None = Field(
        default=None,
        description="CNPJ com pontuação",
        json_schema_extra={
            "prompt_append": "Use o cadastro da Receita.",
            "condition": {"field": "tipo", "in": ["pj", "mei"]},
        },
    )


# O Pydantic tipa json_schema_extra como JSON, e o conjunto não é JSON.
_CONDICAO_NIVEL: dict[str, Any] = {"condition": {"field": "nivel", "in": {3, 1, 2}}}


class ConjuntoDeInteiros(BaseModel):
    """O hash do inteiro é o próprio valor, e o conjunto itera na mesma ordem em todo processo."""

    nivel: int
    nota: float | None = Field(default=None, json_schema_extra=_CONDICAO_NIVEL)


def _acrescenta_exemplo(schema):
    schema["examples"] = ["x"]


class ExtraEmFuncao(BaseModel):
    """O Pydantic chama a função de json_schema_extra, e o resultado é JSON."""

    campo1: str = Field(json_schema_extra=_acrescenta_exemplo)


class SemSchema(BaseModel):
    campo1: str

    @classmethod
    def model_json_schema(cls, *args, **kwargs):
        msg = "sem schema"
        raise TypeError(msg)


@pytest.mark.parametrize(
    ("modelo", "esperada"),
    [
        (Modelo, "d6faf6f44236625edbf7ba798ce4b39b6f1b4d8cfdfc582b8f9f04c7d2d86647"),
        (Empresa, "199638f0467d90d7c55c6a7bd34bf8e33ca8e91cdfe87e1641c21c4ed27e91c5"),
        (ConjuntoDeInteiros, "c9b0940f4839c9cc73e8a358e15c7c886fc73b7f2f1e0ce47bf067056f90d400"),
        (ExtraEmFuncao, "7f8152ca65baaa0e5dd6f7b23803e8f8e23af2a8dbcb7d6eb871a04f9fd2816d"),
        (SemSchema, "169f3ee5ca5a59b3eb231141f6d5a74c1cd4d1eccf1abd8fa51a8feb008aa070"),
    ],
)
def test_modelo_sem_funcao_nem_conjunto_de_texto_mantem_a_assinatura(modelo, esperada):
    assert _assinatura(modelo) == esperada


# Variações de um modelo com condição callable, que vai pela forma canônica. Cada
# uma muda o que o LLM responde ou o que a biblioteca pergunta, e precisa mudar a
# assinatura.


def _funcao(fonte, nome="condicao"):
    """Função compilada de `fonte`, com o nome qualificado de quem a define.

    Duas fontes com o mesmo nome dão o mesmo `__qualname__`, como a lambda
    editada entre uma execução e outra.
    """
    espaco = {}
    exec(fonte, espaco)  # noqa: S102 (fonte fixa do teste)
    return espaco[nome]


_VIP = "condicao = lambda d: d.get('tipo') == 'vip'"


def _porte(*membros, nome="Porte"):
    """Enum com os membros dados; o mesmo nome, como o Enum editado."""
    return Enum(nome, {membro: membro[0] for membro in membros})


def _pedido(  # noqa: PLR0913 (uma parte do modelo por variação do teste)
    condicao=None,
    *,
    depends_on="tipo",
    tipos=frozenset({"pf", "mei"}),
    enum_porte=None,
    anotacao_etiquetas=list[str],
    nome_endereco="Endereco",
    doc_endereco="Endereço de entrega.",
    titulo_endereco=None,
    descricao_cidade="Cidade da entrega.",
    doc_pedido=None,
    titulo_pedido=None,
):
    condicao = condicao or _funcao(_VIP)
    enum_porte = enum_porte or _porte("pequeno", "grande")
    condicao_frete: dict[str, Any] = {"condition": {"field": "tipo", "in": set(tipos)}}
    # Pelo create_model, o nome da classe muda como na classe renomeada no código.
    # Any porque o ty não aceita a variável de tipo na anotação do campo.
    endereco_modelo: Any = create_model(
        nome_endereco,
        __config__=ConfigDict(title=titulo_endereco),
        __doc__=doc_endereco,
        cidade=(str, Field(description=descricao_cidade)),
    )

    class Pedido(BaseModel):
        model_config = ConfigDict(title=titulo_pedido)

        tipo: str
        porte: enum_porte | None = None
        etiquetas: anotacao_etiquetas = []
        endereco: endereco_modelo | None = None
        desconto: float | None = Field(
            default=None, json_schema_extra=field_condition(condicao, depends_on=depends_on)
        )
        frete: float | None = Field(default=None, json_schema_extra=condicao_frete)

    Pedido.__doc__ = doc_pedido
    return Pedido


def test_mesmo_modelo_criado_duas_vezes_tem_a_mesma_assinatura():
    """A função nova, em outra linha e em outro endereço, com o mesmo código, assina igual."""
    assert _assinatura(_pedido()) == _assinatura(_pedido(_funcao("\n\n\n" + _VIP)))


@pytest.mark.parametrize(
    "variacao",
    [
        pytest.param({"condicao": _funcao(_VIP.replace("vip", "ouro"))}, id="corpo-da-lambda"),
        pytest.param({"condicao": _funcao(_VIP.replace("==", "!="))}, id="operador-da-lambda"),
        pytest.param({"condicao": _funcao(_VIP.replace(".get(", ".pop("))}, id="metodo-da-lambda"),
        pytest.param(
            {
                "condicao": _funcao(
                    _VIP.replace("condicao = lambda d:", "def outra(d): return"), "outra"
                )
            },
            id="nome-da-funcao",
        ),
        pytest.param({"depends_on": ["tipo", "porte"]}, id="depends-on"),
        pytest.param({"tipos": {"pf", "mei", "me"}}, id="conjunto-da-condicao"),
        pytest.param({"enum_porte": _porte("pequeno", "medio", "grande")}, id="membro-do-enum"),
        pytest.param(
            {"enum_porte": _porte("pequeno", "grande", nome="Tamanho")}, id="nome-do-enum"
        ),
        pytest.param({"anotacao_etiquetas": list[int]}, id="argumento-da-anotacao"),
        pytest.param({"doc_endereco": "Endereço de cobrança."}, id="docstring-do-aninhado"),
        pytest.param({"titulo_endereco": "Destino"}, id="model-config-do-aninhado"),
        pytest.param({"descricao_cidade": "Cidade de cobrança."}, id="campo-do-aninhado"),
        pytest.param({"nome_endereco": "Destino"}, id="nome-do-aninhado"),
        pytest.param({"doc_pedido": "Pedido de venda."}, id="docstring-do-raiz"),
        pytest.param({"titulo_pedido": "Venda"}, id="model-config-do-raiz"),
    ],
)
def test_mudanca_no_modelo_com_condicao_callable_muda_a_assinatura(variacao):
    assert _assinatura(_pedido(**variacao)) != _assinatura(_pedido())


_REGRA = "def regra(d, alvo): return d.get('tipo') == alvo"

_REGRA_CHAMAVEL = textwrap.dedent(
    """
    class Regra:
        def __init__(self, alvo):
            self.alvo = alvo

        def __call__(self, d):
            return d.get("tipo") == self.alvo
    """
)


def _parcial(fonte=_REGRA, *args, **keywords):
    """`partial` da `regra` compilada de `fonte`, com os argumentos dados."""
    return functools.partial(_funcao(fonte, "regra"), *args, **keywords)


def _chamavel(alvo="vip", fonte=_REGRA_CHAMAVEL):
    """Instância da `Regra` compilada de `fonte`, a classe editada com o mesmo nome."""
    return _funcao(fonte, "Regra")(alvo)


@pytest.mark.parametrize(
    "condicao",
    [
        pytest.param(lambda: _parcial(alvo="vip"), id="partial"),
        pytest.param(lambda: _chamavel("vip"), id="objeto-chamavel"),
        pytest.param(lambda: operator.methodcaller("get", "tipo"), id="chamavel-embutido"),
    ],
)
def test_condicao_partial_ou_objeto_criada_duas_vezes_tem_a_mesma_assinatura(condicao):
    """A classe e a função compiladas de novo, em outro endereço, assinam igual.

    O chamável embutido, sem código Python no `__call__`, entra pelo repr.
    """
    assert _assinatura(_pedido(condicao())) == _assinatura(_pedido(condicao()))


@pytest.mark.parametrize(
    ("antes", "depois"),
    [
        pytest.param(
            lambda: _parcial(alvo="vip"),
            lambda: _parcial(_REGRA.replace("==", "!="), alvo="vip"),
            id="corpo-da-funcao-do-partial",
        ),
        pytest.param(
            lambda: _parcial(alvo="vip"), lambda: _parcial(alvo="ouro"), id="keyword-do-partial"
        ),
        pytest.param(
            lambda: _parcial(_REGRA, "vip"),
            lambda: _parcial(_REGRA, "ouro"),
            id="argumento-do-partial",
        ),
        pytest.param(
            lambda: _chamavel("vip"),
            lambda: _chamavel("vip", _REGRA_CHAMAVEL.replace("==", "!=")),
            id="corpo-do-call",
        ),
        pytest.param(lambda: _chamavel("vip"), lambda: _chamavel("ouro"), id="estado-do-objeto"),
    ],
)
def test_mudanca_no_partial_ou_no_objeto_chamavel_muda_a_assinatura(antes, depois):
    """O partial e o objeto entram pelo código e pelo que fixam, e não pelo repr."""
    assert _assinatura(_pedido(antes())) != _assinatura(_pedido(depois()))


def test_mudanca_dentro_de_funcao_aninhada_na_condicao_muda_a_assinatura():
    """O gerador da condição tem código próprio, guardado entre as constantes dela."""
    fonte = "condicao = lambda d: any(t == d.get('tipo') for t in ('vip', 'ouro'))"

    assert _assinatura(_pedido(_funcao(fonte))) != _assinatura(
        _pedido(_funcao(fonte.replace("t ==", "t !=")))
    )


@pytest.mark.parametrize(
    "atributo",
    [
        "annotation",
        "default",
        "default_factory",
        "alias",
        "validation_alias",
        "title",
        "description",
        "examples",
        "deprecated",
        "discriminator",
        "json_schema_extra",
        "metadata",
    ],
)
def test_atributo_do_campo_que_o_json_schema_levaria_muda_a_assinatura(atributo):
    """Na forma canônica, o campo entra pelo que o JSON Schema dele diria ao LLM."""
    modelo = _pedido()
    antes = _assinatura(modelo)

    setattr(modelo.model_fields["frete"], atributo, "outro")

    assert _assinatura(modelo) != antes


# Cabeçalho do processo filho que assina condições chamáveis: o Pedido com a
# condição dada e a assinatura dele, pelo mesmo `_run_signature` do teste. O
# modelo já montado, em vez da condição, é assinado como veio.
_ASSINA_CONDICOES = textwrap.dedent(
    """
    import functools
    import json

    from pydantic import BaseModel, Field

    from dataframeit import field_condition
    from dataframeit.core import _run_signature


    def assinatura(condicao):
        class Pedido(BaseModel):
            tipo: str
            desconto: float | None = Field(
                default=None, json_schema_extra=field_condition(condicao, depends_on="tipo")
            )

        modelo = condicao if isinstance(condicao, type) else Pedido
        return _run_signature(
            modelo, "{texto}", provider="openai", model="m", model_kwargs=None,
            search_config=None, text_column="texto", status_col="_dataframeit_status",
        )
    """
)


def _assinaturas_em_tres_sementes(definicoes, condicoes):
    """Assinatura de cada condição em três processos; exige que elas não mudem entre eles.

    `definicoes` é o código que monta as condições, e `condicoes`, a expressão
    do dict de nome para condição, avaliada depois dele.
    """
    codigo = (
        _ASSINA_CONDICOES
        + textwrap.dedent(definicoes)
        + "\nprint(json.dumps({nome: assinatura(condicao)"
        + f" for nome, condicao in ({condicoes}).items()}}))\n"
    )

    por_semente = [_roda_python(codigo, semente=semente) for semente in ("0", "1", "2")]

    assert por_semente[0] == por_semente[1] == por_semente[2]
    return por_semente[0]


# Objeto chamável cujo estado volta a ele: o atributo que aponta para o próprio
# objeto e o ciclo entre dois objetos. A descida no estado não pode recursar sem
# fim, e o que marca o ciclo não pode levar o endereço do objeto.
_CICLOS = """
class Regra:
    def __init__(self, alvo):
        self.alvo = alvo

    def __call__(self, d):
        return d.get("tipo") == self.alvo


def auto_referente(alvo):
    regra = Regra(alvo)
    regra.eu = regra
    return regra


def ciclo(alvo):
    primeira, segunda = Regra(alvo), Regra("ouro")
    primeira.outra, segunda.outra = segunda, primeira
    return primeira
"""


def test_objeto_chamavel_com_ciclo_no_estado_assina_sem_recursao_e_igual_entre_sementes():
    """O ciclo entra por um marcador sem o endereço do objeto."""
    _assinaturas_em_tres_sementes(_CICLOS, "{'auto': auto_referente('vip'), 'ciclo': ciclo('vip')}")


@pytest.mark.parametrize("construtor", ["auto_referente", "ciclo"])
def test_mudanca_no_estado_do_objeto_com_ciclo_muda_a_assinatura(construtor):
    """O marcador fica no lugar do objeto reencontrado, e o resto do estado continua contando."""
    montar = _funcao(_CICLOS, construtor)

    assert _assinatura(_pedido(montar("vip"))) != _assinatura(_pedido(montar("comum")))


def test_mesmo_objeto_em_dois_atributos_assina_como_dois_objetos_iguais():
    """Só o objeto que já está no caminho vira marcador; o repetido em outro ramo entra inteiro."""
    regra = _funcao(_REGRA_CHAMAVEL, "Regra")
    repetido, outro, igual = regra("vip"), regra("vip"), regra("vip")
    repetido.a = repetido.b = outro
    igual.a, igual.b = outro, regra("vip")

    assert _assinatura(_pedido(repetido)) == _assinatura(_pedido(igual))


def test_par_em_ciclo_em_dois_atributos_assina_como_dois_pares_separados():
    """O objeto do ciclo entra pelo caminho por onde se chegou a ele, mesmo já visto por outro ramo.

    Pelo atributo `a`, o par entra a partir de `x`, e o marcador fica em `x`;
    pelo `b`, a partir de `z`, e o marcador fica em `z`. Os dois pares separados
    entram do mesmo jeito.
    """
    regra = _funcao(_REGRA_CHAMAVEL, "Regra")

    def par():
        x, z = regra("x"), regra("z")
        x.outra, z.outra = z, x
        return x, z

    compartilhado, separado = regra("vip"), regra("vip")
    compartilhado.a, compartilhado.b = par()
    separado.a, separado.b = par()[0], par()[1]

    assert _assinatura(_pedido(compartilhado)) == _assinatura(_pedido(separado))


def test_objeto_alcancado_por_varios_ramos_e_lido_uma_vez(monkeypatch):
    """Fora de ciclo, a forma do objeto não depende do caminho, e a descida não a refaz.

    Cada nível aponta duas vezes para o nível de baixo: descer de novo em cada
    ramo leria o estado 2**13 - 1 vezes, e o tempo dobraria a cada nível.
    """
    regra = _funcao(_REGRA_CHAMAVEL, "Regra")
    lidos = []
    ler = core._slot_values
    monkeypatch.setattr(core, "_slot_values", lambda valor: lidos.append(valor) or ler(valor))
    nivel = regra("vip")
    for _ in range(12):
        acima = regra("z")
        acima.a = acima.b = nivel
        nivel = acima

    _assinatura(_pedido(nivel))

    assert len(lidos) == len({id(valor) for valor in lidos}) == 13


# Objeto chamável sem __dict__, com o estado em __slots__: o slot da classe, o
# herdado, o privado (que o Python guarda com o nome da classe na frente, sem os
# sublinhados do começo dele) e o __dict__ declarado como slot, ao lado dos
# outros. O slot com conjunto de texto passa pelas sementes.
_SLOTS = """
class Regra:
    __slots__ = ("alvo",)

    def __init__(self, alvo):
        self.alvo = alvo

    def __call__(self, d):
        return d.get("tipo") in self.alvo


class RegraPesada(Regra):
    __slots__ = "__peso"

    def __init__(self, alvo, peso):
        super().__init__(alvo)
        self.__peso = peso


class _Regra(Regra):
    __slots__ = ("__peso",)

    def __init__(self, alvo, peso):
        super().__init__(alvo)
        self.__peso = peso


class RegraAberta(Regra):
    __slots__ = ("__dict__", "__weakref__", "nota")


def aberta(alvo, **atributos):
    regra = RegraAberta(alvo)
    for nome, valor in atributos.items():
        setattr(regra, nome, valor)
    return regra


TIPOS = {"vip", "ouro", "prata", "bronze"}
VARIACOES = {
    "slot": lambda: Regra(TIPOS),
    "slot-mudado": lambda: Regra(TIPOS | {"lata"}),
    "herdado": lambda: RegraPesada(TIPOS, 1),
    "herdado-mudado": lambda: RegraPesada({"vip"}, 1),
    "privado-mudado": lambda: RegraPesada(TIPOS, 2),
    "privado-de-classe-interna": lambda: _Regra(TIPOS, 1),
    "privado-de-classe-interna-mudado": lambda: _Regra(TIPOS, 2),
    "aberta": lambda: aberta(TIPOS),
    "aberta-com-nota": lambda: aberta(TIPOS, nota=1),
    "aberta-com-dict": lambda: aberta(TIPOS, extra=1),
}
"""

# A referência fraca ao objeto não conta. O Pydantic guarda referência fraca às
# variáveis locais da função que define o modelo, e a condição do Pedido
# montado em `assinatura` tem um `__weakref__` que a deste Pedido, definido no
# módulo, não tem.
_PEDIDO_NO_MODULO = """
class Pedido(BaseModel):
    tipo: str
    desconto: float | None = Field(
        default=None, json_schema_extra=field_condition(aberta(TIPOS), depends_on="tipo")
    )
"""


def test_objeto_chamavel_com_slots_assina_igual_entre_sementes():
    """O slot entra igual em qualquer processo, e a referência fraca ao objeto não conta."""
    assinaturas = _assinaturas_em_tres_sementes(
        _SLOTS + _PEDIDO_NO_MODULO,
        "{nome: montar() for nome, montar in VARIACOES.items()} | {'aberta-no-modulo': Pedido}",
    )

    assert assinaturas["aberta"] == assinaturas["aberta-no-modulo"]


@pytest.mark.parametrize(
    ("antes", "depois"),
    [
        ("slot", "slot-mudado"),
        ("herdado", "herdado-mudado"),
        ("herdado", "privado-mudado"),
        ("privado-de-classe-interna", "privado-de-classe-interna-mudado"),
        ("aberta", "aberta-com-nota"),
        ("aberta", "aberta-com-dict"),
    ],
)
def test_mudanca_no_slot_do_objeto_chamavel_muda_a_assinatura(antes, depois):
    """O atributo em __slots__, sem __dict__, conta como o atributo da instância."""
    variacoes = _funcao(_SLOTS, "VARIACOES")

    assert _assinatura(_pedido(variacoes[antes]())) != _assinatura(_pedido(variacoes[depois]()))


# Objeto chamável com busca de atributo própria: o __getattr__ que devolve None
# para qualquer nome, o que devolve texto vazio, o que levanta KeyError (o objeto que guarda os atributos
# num dict) e o slot sombreado, na subclasse, por property que levanta. A
# assinatura lê o estado sem passar por eles, e o slot não atribuído fica de fora
# em vez de entrar com o valor que o __getattr__ inventa.
_BUSCA_PROPRIA = """
class Nula:
    __slots__ = ("alvo", "extra")

    def __init__(self, alvo):
        self.alvo = alvo

    def __getattr__(self, nome):
        return None

    def __call__(self, d):
        return d.get("tipo") == self.alvo


class Vazia(Nula):
    __slots__ = ()

    def __getattr__(self, nome):
        return ""


class Dicionario:
    __slots__ = ("alvo", "_dados", "extra")

    def __init__(self, alvo):
        self.alvo = alvo
        self._dados = {}

    def __getattr__(self, nome):
        return self._dados[nome]

    def __call__(self, d):
        return d.get("tipo") == self.alvo


class Base:
    __slots__ = ("alvo",)

    def __call__(self, d):
        return d.get("tipo") == "vip"


class Sombreada(Base):
    @property
    def alvo(self):
        raise ValueError("lendo alvo")


def sombreada(alvo):
    regra = Sombreada()
    Base.alvo.__set__(regra, alvo)
    return regra


CONSTRUTORES = {"nula": Nula, "vazia": Vazia, "dicionario": Dicionario, "sombreada": sombreada}
"""


def test_objeto_com_busca_de_atributo_propria_assina_igual_entre_sementes():
    """Nem o __getattr__ nem a property que levantam derrubam a assinatura."""
    _assinaturas_em_tres_sementes(
        _BUSCA_PROPRIA, "{nome: montar('vip') for nome, montar in CONSTRUTORES.items()}"
    )


@pytest.mark.parametrize("construtor", ["nula", "vazia", "dicionario", "sombreada"])
def test_mudanca_no_slot_de_objeto_com_busca_propria_muda_a_assinatura(construtor):
    """O slot entra pelo descritor da classe que o declara, sem a busca de atributo."""
    montar = _funcao(_BUSCA_PROPRIA, "CONSTRUTORES")[construtor]

    assert _assinatura(_pedido(montar("vip"))) != _assinatura(_pedido(montar("ouro")))


# Método como condição: o ligado à instância, o partial dele, o objeto
# staticmethod (o que se tem ao citar o método estático no corpo do próprio
# modelo) e o classmethod ligado à classe. A classe é compilada de novo a cada
# alvo, com o mesmo nome, como a classe editada entre uma execução e outra.
_METODOS = '''
import functools

FONTE = """
class Regra:
    def __init__(self, alvo):
        self.alvo = alvo

    def verifica(self, d):
        return d.get("tipo") == self.alvo

    @staticmethod
    def vip(d):
        return d.get("tipo") == "ALVO"

    @classmethod
    def da_classe(cls, d):
        return d.get("tipo") == "ALVO"
"""


def regra(alvo="vip", modulo="regras"):
    espaco = {"__name__": modulo}
    exec(FONTE.replace("ALVO", alvo), espaco)
    return espaco["Regra"]


def com_atalho(alvo):
    instancia = regra()(alvo)
    instancia.atalho = instancia.verifica
    return instancia.verifica


VARIACOES = {
    "ligado": lambda: regra()("vip").verifica,
    "ligado-mudado": lambda: regra()("comum").verifica,
    "partial-ligado": lambda: functools.partial(regra()("vip").verifica),
    "partial-ligado-mudado": lambda: functools.partial(regra()("comum").verifica),
    "estatico": lambda: vars(regra("vip"))["vip"],
    "estatico-mudado": lambda: vars(regra("comum"))["vip"],
    "da-classe": lambda: regra("vip").da_classe,
    "da-classe-mudado": lambda: regra("comum").da_classe,
    "da-classe-em-outro-modulo": lambda: regra("vip", "outro").da_classe,
    "atalho": lambda: com_atalho("vip"),
    "atalho-mudado": lambda: com_atalho("comum"),
}
'''


def test_metodo_como_condicao_assina_igual_entre_sementes():
    """O método, o partial dele e o dono entram igual em qualquer processo."""
    _assinaturas_em_tres_sementes(
        _METODOS, "{nome: montar() for nome, montar in VARIACOES.items()}"
    )


@pytest.mark.parametrize(
    ("antes", "depois"),
    [
        ("ligado", "ligado-mudado"),
        ("partial-ligado", "partial-ligado-mudado"),
        ("estatico", "estatico-mudado"),
        ("da-classe", "da-classe-mudado"),
        ("atalho", "atalho-mudado"),
    ],
)
def test_mudanca_no_metodo_ou_no_dono_muda_a_assinatura(antes, depois):
    """O método ligado entra pelo código e pelo estado do objeto; o staticmethod, pelo código."""
    variacoes = _funcao(_METODOS, "VARIACOES")

    assert _assinatura(_pedido(variacoes[antes]())) != _assinatura(_pedido(variacoes[depois]()))


def test_classe_dona_do_metodo_entra_pelo_nome_e_nao_pelo_modulo():
    """Como a função, que entra pelo nome qualificado: mover a classe de módulo não muda nada."""
    variacoes = _funcao(_METODOS, "VARIACOES")

    assert _assinatura(_pedido(variacoes["da-classe"]())) == _assinatura(
        _pedido(variacoes["da-classe-em-outro-modulo"]())
    )


def _com_extra(valor):
    """Modelo com `valor` numa chave livre do json_schema_extra, na forma canônica."""
    # O Pydantic tipa json_schema_extra como JSON; o conjunto leva o modelo à
    # forma canônica, e o classmethod não é JSON.
    extra: dict[str, Any] = {"regra": valor, "tipos": {"pf", "mei"}}

    class Pedido(BaseModel):
        tipo: str = Field(json_schema_extra=extra)

    return Pedido


def test_objeto_classmethod_em_chave_livre_assina_pelo_codigo():
    """O field_condition recusa o classmethod, mas a chave livre do extra o aceita."""
    antes = classmethod(_funcao(_REGRA, "regra"))
    depois = classmethod(_funcao(_REGRA.replace("==", "!="), "regra"))

    assert _assinatura(_com_extra(antes)) != _assinatura(_com_extra(depois))
