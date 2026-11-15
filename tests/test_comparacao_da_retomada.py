"""Comparação do valor relido na retomada com o que a validação do modelo devolve."""

import datetime as dt
import logging
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from dataframeit.core import (
    _condition_holds,
    _executed_form,
    _same_json,
    _same_value,
    _validate_processed_rows,
    _validate_row,
    _validated_where_same,
)

STATUS = "_dataframeit_status"


def _processadas(**colunas: list[object]) -> pd.DataFrame:
    """DataFrame de colunas de objetos com todas as linhas processadas."""
    linhas = len(next(iter(colunas.values())))
    return pd.DataFrame({**colunas, STATUS: ["processed"] * linhas}, dtype=object)


def _recusados(erro: ValidationError) -> set[str]:
    """Os campos que o erro acusa, pelo primeiro item do loc."""
    return {str(detalhe["loc"][0]) for detalhe in erro.errors()}


# =============================================================================
# _same_json e _same_value
# =============================================================================


def test_mesma_data_dos_dois_lados_da_o_mesmo_json():
    assert _same_json(dt.date(2024, 1, 2), dt.date(2024, 1, 2)) is True


@pytest.mark.parametrize(
    ("validado", "relido"),
    [(np.array([1]), 1), (1, np.array([1]))],
    ids=["estrutura-validada", "estrutura-relida"],
)
def test_estrutura_nao_diz_o_mesmo_que_escalar_que_ela_iguala(validado, relido):
    """O array de um item é == ao escalar, mas o JSON dos dois difere."""
    assert _same_value(validado, relido) is False


def test_datetime_do_mesmo_instante_em_outro_fuso_diz_o_mesmo():
    """Só a data validada se compara ao datetime relido à meia-noite."""
    validado = dt.datetime(2024, 1, 1, tzinfo=dt.UTC)
    relido = dt.datetime(2024, 1, 1, 1, tzinfo=dt.timezone(dt.timedelta(hours=1)))

    assert _same_value(validado, relido) is True


def test_texto_validado_nao_diz_o_mesmo_que_o_datetime_relido():
    assert _same_value("2024-01-01", dt.datetime(2024, 1, 1, tzinfo=dt.UTC)) is False


def test_texto_validado_nao_diz_o_mesmo_que_o_float_que_ele_escreve():
    """Só o Decimal se compara ao float convertido; o texto "1.5" não é o float 1.5."""
    assert _same_value("1.5", 1.5) is False


# =============================================================================
# _validated_where_same
# =============================================================================


def test_item_de_dict_se_compara_pelo_json_e_o_float_relido_fica():
    """No item, 1 e 1.0 diferem como no checkpoint, e o relido fica."""
    resultado = _validated_where_same({"a": 1}, {"a": 1.0})

    assert isinstance(resultado, dict)
    assert resultado == {"a": 1.0}
    assert type(resultado["a"]) is float


def test_dict_com_as_mesmas_chaves_em_json_volta_item_a_item():
    """A chave int diz o mesmo que o texto dela; a data volta, e o float relido fica."""
    resultado = _validated_where_same({1: dt.date(2024, 1, 2), 2: 1}, {"1": "2024-01-02", "2": 1.0})

    assert isinstance(resultado, dict)
    assert resultado == {1: dt.date(2024, 1, 2), 2: 1.0}
    assert type(resultado[2]) is float


def test_dict_com_outras_chaves_devolve_o_relido():
    relido = {"b": 1.0}

    assert _validated_where_same({"a": 1}, relido) is relido


@pytest.mark.parametrize("relido", [["a"], 5], ids=["lista-com-a-chave", "escalar"])
def test_dict_validado_com_relido_que_nao_e_dict_devolve_o_relido(relido):
    assert _validated_where_same({"a": 1}, relido) is relido


# =============================================================================
# _executed_form
# =============================================================================


class Trocado(BaseModel):
    """O alias de cada campo é o nome do outro."""

    a: dt.date = Field(alias="b")
    b: str = Field(alias="a")


class PorAlias(BaseModel):
    model_config = ConfigDict(serialize_by_alias=True)

    x: dt.date = Field(alias="X")


class ComFormas(BaseModel):
    estrita: dt.date = Field(strict=True)
    trocado: Trocado
    por_alias: PorAlias


def test_forma_executada_valida_sem_strict_o_campo_estrito():
    """A data relida como texto volta a data mesmo no campo com strict=True."""
    assert _executed_form(ComFormas)("estrita", "2024-01-02") == dt.date(2024, 1, 2)


def test_forma_executada_le_o_submodelo_pelo_nome_e_nao_pelo_alias():
    relido = {"a": "2024-01-02", "b": "texto"}

    assert _executed_form(ComFormas)("trocado", relido) == {
        "a": dt.date(2024, 1, 2),
        "b": "texto",
    }


def test_forma_executada_devolve_o_submodelo_pelo_nome_mesmo_com_serialize_by_alias():
    relido = {"x": "2024-01-02"}

    assert _executed_form(ComFormas)("por_alias", relido) == {"x": dt.date(2024, 1, 2)}


# =============================================================================
# _validate_row
# =============================================================================


class TrocadoNoTopo(BaseModel):
    a: str = Field(alias="b")
    b: str = Field(alias="a")


class EstritoTrocado(BaseModel):
    model_config = ConfigDict(strict=True)

    a: int = Field(alias="b")
    b: int = Field(alias="a")


class PorAliasNoTopo(BaseModel):
    model_config = ConfigDict(strict=True, serialize_by_alias=True)

    a: int = Field(alias="A")


def test_linha_se_valida_pelo_nome_e_nao_pelo_alias():
    assert _validate_row(TrocadoNoTopo, {"a": "x", "b": "y"}, _recusados) == (
        {"a": "x", "b": "y"},
        set(),
    )


def test_linha_refeita_sem_strict_tambem_se_valida_pelo_nome():
    """O float relido do int estrito é aceito, cada um no próprio campo."""
    assert _validate_row(EstritoTrocado, {"a": 1.0, "b": 2.0}, _recusados) == (
        {"a": 1, "b": 2},
        set(),
    )


@pytest.mark.parametrize("relido", [1, 1.0], ids=["aceito-com-strict", "refeito-sem-strict"])
def test_linha_validada_volta_pelo_nome_mesmo_com_serialize_by_alias(relido):
    assert _validate_row(PorAliasNoTopo, {"a": relido}, _recusados) == ({"a": 1}, set())


class EstritoComValidador(BaseModel):
    model_config = ConfigDict(strict=True)

    n: int
    texto: str

    @field_validator("texto")
    @classmethod
    def _minusculo(cls, valor: str) -> str:
        return valor.lower()


def test_so_os_campos_que_o_strict_recusou_se_comparam_ao_relido():
    """O texto que o validador muda não foi recusado, e a linha passa."""
    assert _validate_row(EstritoComValidador, {"n": 1.0, "texto": "X"}, _recusados) == (
        {"n": 1, "texto": "x"},
        set(),
    )


# =============================================================================
# _condition_holds e a condição na validação das linhas
# =============================================================================


def test_condicao_sem_field_avisa_com_o_nome_do_campo(caplog):
    with caplog.at_level(logging.WARNING, logger="dataframeit.conditional"):
        assert _condition_holds({"equals": 1}, {}, "campo") is False

    assert caplog.messages == ["Condição para campo 'campo' não tem 'field' definido"]


def _condicao_que_falha(dados: dict[str, object]) -> bool:
    mensagem = "falhou"
    raise RuntimeError(mensagem)


_CONDICAO_QUE_FALHA: dict[str, Any] = {"condition": _condicao_que_falha, "depends_on": ["tipo"]}


class ComCondicaoQueFalha(BaseModel):
    tipo: str
    detalhe: str | None = Field(None, json_schema_extra=_CONDICAO_QUE_FALHA)


def test_condicao_que_falha_na_retomada_avisa_com_o_nome_do_campo(caplog):
    df = _processadas(tipo=["a"], detalhe=[None])

    with caplog.at_level(logging.WARNING, logger="dataframeit.conditional"):
        resultado = _validate_processed_rows(df, STATUS, ComCondicaoQueFalha, set())

    assert resultado == ([], {(0, "detalhe"): None})
    assert caplog.messages == ["Erro ao avaliar condição callable para campo 'detalhe': falhou"]


# =============================================================================
# _validate_processed_rows: o que cada campo e cada linha deixam para os seguintes
# =============================================================================


class TextoENumero(BaseModel):
    texto: str
    n: int


def test_so_o_valor_que_muda_de_tipo_e_gravado_de_novo():
    """O texto igual ao validado fica; o float do int volta int, e o campo seguinte também."""
    df = _processadas(texto=["igual"], n=[2.0])

    assert _validate_processed_rows(df, STATUS, TextoENumero, set()) == ([], {(0, "n"): 2})


def test_linha_recusada_nao_impede_a_conferencia_das_seguintes():
    df = _processadas(texto=["a", "b"], n=["abc", 2.0])

    assert _validate_processed_rows(df, STATUS, TextoENumero, set()) == (
        ["n"],
        {(1, "n"): 2},
    )


class OcultoENumero(BaseModel):
    oculto: int | None = Field(default=None, exclude_if=lambda valor: valor == 3)
    n: int


def test_campo_que_o_dump_tira_nao_impede_a_conferencia_dos_seguintes():
    df = _processadas(oculto=[3], n=[2.0])

    assert _validate_processed_rows(df, STATUS, OcultoENumero, set()) == ([], {(0, "n"): 2})


def test_celula_vazia_do_campo_omitivel_nao_impede_a_conferencia_dos_seguintes():
    df = _processadas(oculto=[None], n=[2.0])

    assert _validate_processed_rows(df, STATUS, OcultoENumero, set()) == (
        [],
        {(0, "oculto"): None, (0, "n"): 2},
    )


def test_coluna_ausente_conta_como_valor_ausente_e_os_campos_seguintes_sao_lidos():
    df = _processadas(n=[2.0])

    assert _validate_processed_rows(df, STATUS, TextoENumero, set()) == (["texto"], {})


class PadraoENumero(BaseModel):
    padrao: str = "p"
    n: int


def test_campo_opcional_ausente_nao_impede_a_conferencia_dos_seguintes():
    df = _processadas(padrao=[None], n=[2.0])

    assert _validate_processed_rows(df, STATUS, PadraoENumero, set()) == (
        [],
        {(0, "padrao"): "p", (0, "n"): 2},
    )


def test_texto_vazio_restaurado_nao_impede_a_conferencia_dos_seguintes():
    df = _processadas(texto=[None], n=[2.0])

    assert _validate_processed_rows(
        df, STATUS, TextoENumero, set(), empty_text_fields={"texto"}
    ) == ([], {(0, "texto"): "", (0, "n"): 2})


class ExcluidoENumero(BaseModel):
    interno: str | None = Field(default=None, exclude=True)
    n: int


def test_campo_excluido_nao_impede_a_conferencia_dos_seguintes():
    df = _processadas(interno=[None], n=[2.0])

    assert _validate_processed_rows(df, STATUS, ExcluidoENumero, set()) == ([], {(0, "n"): 2})


class ObrigatorioEPadrao(BaseModel):
    obrigatorio: str
    padrao: str = "p"


def test_linha_recusada_preenche_o_default_depois_do_obrigatorio_ausente():
    df = _processadas(obrigatorio=[None], padrao=[None])

    assert _validate_processed_rows(df, STATUS, ObrigatorioEPadrao, set()) == (
        ["obrigatorio"],
        {(0, "padrao"): "p"},
    )


class GravadoEPadrao(BaseModel):
    anotado: str | None = "x"
    padrao: str = "p"
    n: int


def test_linha_recusada_preenche_o_default_depois_do_none_gravado():
    df = _processadas(anotado=[None], padrao=[None], n=["abc"])

    assert _validate_processed_rows(df, STATUS, GravadoEPadrao, set()) == (
        ["n"],
        {(0, "anotado"): None, (0, "padrao"): "p"},
    )


class FabricaENumero(BaseModel):
    lista: list[str] = Field(default_factory=list)
    n: int


def test_linha_recusada_preenche_o_resultado_da_fabrica():
    df = _processadas(lista=[None], n=["abc"])

    assert _validate_processed_rows(df, STATUS, FabricaENumero, set()) == (
        ["n"],
        {(0, "lista"): []},
    )


def _quantos_campos(dados: dict[str, object]) -> int:
    return len(dados)


class FabricaQueLeOsDados(BaseModel):
    derivado: int = Field(default_factory=_quantos_campos)
    n: int


def test_linha_recusada_acusa_o_campo_da_fabrica_que_le_os_dados():
    df = _processadas(derivado=[None], n=["abc"])

    assert _validate_processed_rows(df, STATUS, FabricaQueLeOsDados, set()) == (
        ["derivado", "n"],
        {},
    )
