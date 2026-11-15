"""Extração do JSON da resposta e leitura de anotações em utils.py.

Os testes vieram dos mutantes que sobreviviam ao teste de mutação de utils.py:
cada um fixa um caso em que a troca de um operador, de um limite ou de um
argumento mudaria o resultado.
"""

import sys
from typing import Literal, Optional, Union

import pytest
from pydantic import BaseModel, create_model
from typing_extensions import TypeAliasType, TypeVar

from dataframeit.utils import (
    _admits_int,
    _lookup_forward_ref,
    _resolved,
    list_layers,
    parse_json,
)

# =============================================================================
# parse_json
# =============================================================================


def test_bloco_json_de_varias_linhas_vence_as_chaves_do_texto_em_volta():
    resposta = 'Formato {chave: valor}:\n```json\n{\n  "a": 1\n}\n```\nfim }'

    assert parse_json(resposta) == {"a": 1}


def test_sem_bloco_json_vale_da_primeira_a_ultima_chave():
    assert parse_json('Resposta: {"a": {"b": 1}}.') == {"a": {"b": 1}}


def test_objeto_vazio_no_comeco_do_texto_e_extraido():
    assert parse_json("{} e um texto depois") == {}


@pytest.mark.parametrize(
    "resposta",
    [
        # Só a chave de fechamento, só a de abertura e as duas fora de ordem.
        "sem json }",
        "{ sem fim",
        "} invertido {",
    ],
)
def test_sem_par_de_chaves_a_resposta_inteira_vai_ao_erro(resposta):
    with pytest.raises(ValueError, match="Falha ao decodificar JSON") as exc_info:
        parse_json(resposta)

    assert f"Resposta: '{resposta}'..." in str(exc_info.value)


def test_erro_de_json_corta_a_resposta_em_200_caracteres():
    resposta = "x" * 250

    with pytest.raises(ValueError, match="Falha ao decodificar JSON") as exc_info:
        parse_json(resposta)

    assert str(exc_info.value).endswith(f"Resposta: '{'x' * 200}'...")


# =============================================================================
# Anotações
# =============================================================================


class _Folha(BaseModel):
    x: int


@pytest.mark.parametrize(
    ("anotacao", "admite"),
    [
        (int, True),
        (Literal[1, "a"], True),
        (Optional[Literal[2]], True),
        (Literal["a"], False),
        (Literal[True, False], False),
        (Optional[Literal[1.5]], False),
        (str, False),
    ],
)
def test_so_o_int_e_o_literal_com_algum_int_admitem_int(anotacao, admite):
    assert _admits_int(anotacao) is admite


def test_camadas_de_lista_contam_o_ramo_mais_fundo_da_uniao():
    assert list_layers(Union[_Folha, list[_Folha]], _Folha) == 1


def test_dict_em_volta_do_modelo_nao_conta_como_camada_de_lista():
    assert list_layers(dict[str, _Folha], _Folha) == 0


# O parâmetro com default pode ficar sem argumento, e fica no valor para que o
# TypeVar resolva pelo default.
_T = TypeVar("_T")
_ComDefault = TypeVar("_ComDefault", default=int)
_ApelidoDeDict = TypeAliasType(
    "_ApelidoDeDict", dict[_T, _ComDefault], type_params=(_T, _ComDefault)
)
_ApelidoDoDefault = TypeAliasType("_ApelidoDoDefault", _ComDefault, type_params=(_T, _ComDefault))


def test_parametro_sem_argumento_fica_no_valor_do_apelido():
    assert _resolved(_ApelidoDeDict[str]) == dict[str, _ComDefault]


def test_valor_que_e_parametro_sem_argumento_vale_o_default():
    assert _resolved(_ApelidoDoDefault[str]) is int


def test_referencia_de_modelo_de_modulo_que_nao_foi_importado_nao_resolve():
    modelo = create_model("Modelo", x=(int, ...), __module__="modulo_que_nao_foi_importado")

    assert "modulo_que_nao_foi_importado" not in sys.modules
    assert _lookup_forward_ref("Outro", modelo) is None
