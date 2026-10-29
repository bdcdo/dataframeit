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

from dataframeit import field_condition
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
    """Roda `codigo` num interpretador novo, com a semente de hash dada."""
    saida = subprocess.run(  # noqa: S603 (roda o próprio interpretador com código fixo do teste)
        [sys.executable, "-c", codigo, *argumentos],
        env={**os.environ, "PYTHONHASHSEED": semente},
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
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
