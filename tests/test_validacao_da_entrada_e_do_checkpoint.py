"""Mensagens da validação da entrada e efeito da retomada pelo checkpoint."""

import re
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from pydantic import BaseModel, Field

from dataframeit import dataframeit
from dataframeit.core import (
    _raise_if_overwritten,
    _validate_field_configs,
    _validate_status_column,
)


class Modelo(BaseModel):
    campo1: str


def _exata(mensagem: str) -> str:
    return f"^{re.escape(mensagem)}$"


def _llm(prefixo: str, limite: int = 100):
    """LLM que responde `prefixo` numerado e interrompe a execução depois de `limite` linhas."""
    chamadas = []

    def llm(*args, **kwargs):
        chamadas.append(1)
        if len(chamadas) > limite:
            raise SystemExit
        return {
            "data": {"campo1": f"{prefixo}{len(chamadas)}"},
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }

    return chamadas, llm


def _roda(df, llm, **kwargs):
    with (
        patch("dataframeit.core.call_langchain", side_effect=llm),
        patch("dataframeit.core.validate_provider_dependencies"),
    ):
        return dataframeit(df.copy(), questions=Modelo, prompt="{texto}", **kwargs)


def _valida_status(df, status_col):
    _validate_status_column(
        df, status_col, text_column="texto", expected_columns=list(Modelo.model_fields)
    )


@pytest.mark.parametrize(
    ("status_col", "mensagem"),
    [
        (
            "texto",
            (
                "status_column='texto' é a coluna de texto, e a coluna de status ocuparia o texto "
                "de entrada. Escolha outro nome."
            ),
        ),
        (
            "campo1",
            (
                "status_column='campo1' é um campo do modelo, e a coluna de status ocuparia a "
                "resposta dele. Escolha outro nome."
            ),
        ),
        (
            "_error_details",
            (
                "status_column='_error_details' é o nome de uma coluna de controle do dataframeit, "
                "gravada na mesma linha que o status, e um dos dois valores se perderia. "
                "Escolha outro nome."
            ),
        ),
    ],
)
def test_status_column_que_ocupa_outra_coluna_e_recusado_com_a_mensagem_inteira(
    status_col, mensagem
):
    with pytest.raises(ValueError, match=_exata(mensagem)):
        _valida_status(pd.DataFrame({"texto": ["a"]}), status_col)


def test_coluna_de_status_com_valor_alheio_e_recusada_com_os_valores_aceitos():
    df = pd.DataFrame({"texto": ["a", "b"], "st": ["processed", "x"]})
    mensagem = (
        "A coluna de status 'st' já existe no DataFrame com valores que não são de status do "
        "dataframeit ('x' na linha 1), e a execução os leria como status e sobrescreveria a "
        "coluna. A coluna de status só guarda 'processed', 'error' ou vazio: passe em "
        "status_column outro nome, ou renomeie a coluna existente."
    )
    with pytest.raises(ValueError, match=_exata(mensagem)):
        _valida_status(df, "st")


def test_campo_do_modelo_com_valor_e_sem_status_e_recusado_com_a_dica_de_retomada():
    df = pd.DataFrame({"texto": ["a"], "campo1": ["x"]})
    mensagem = (
        "As colunas ['campo1'] já existem no DataFrame com o nome de campos do modelo, e sem a "
        "coluna de status 'st' a resposta do LLM as sobrescreveria. Renomeie ou remova essas "
        "colunas, ou passe reprocess_columns=['campo1'] para sobrescrevê-las de propósito. "
        "Se o DataFrame é uma saída do dataframeit, devolva a ele a coluna de status com "
        "df['st'] = 'processed' no pandas ou "
        "df = df.with_columns(pl.lit('processed').alias('st')) no polars: a execução o "
        "retoma, e reprocess_columns refaz só as colunas indicadas."
    )
    with pytest.raises(ValueError, match=_exata(mensagem)):
        _raise_if_overwritten(
            df, Modelo, frozenset(), frozenset(), text_column="texto", status_col="st"
        )


def _valida_campos(modelo, **kwargs):
    _validate_field_configs(
        modelo, None, **{"use_search": False, "search_per_field": False, **kwargs}
    )


def test_configuracao_por_campo_sem_busca_por_campo_e_recusada_com_a_mensagem_inteira():
    class ComPrompt(BaseModel):
        a: str = Field(json_schema_extra={"prompt": "p"})

    mensagem = (
        "Campos com configuração em json_schema_extra (prompt, prompt_append, search_depth, "
        "max_results, max_search_calls) requerem use_search=True e search_per_field=True"
    )
    with pytest.raises(ValueError, match=_exata(mensagem)):
        _valida_campos(ComPrompt)


def test_condicao_em_campo_aninhado_e_recusada_com_a_mensagem_inteira():
    class Item(BaseModel):
        valor: str = Field(json_schema_extra={"depends_on": ["x"]})

    class Dono(BaseModel):
        itens: list[Item]

    mensagem = (
        "Campo 'itens.valor' usa 'condition' ou 'depends_on' em json_schema_extra, que só são "
        "aplicados a campos de primeiro nível do modelo"
    )
    with pytest.raises(ValueError, match=_exata(mensagem)):
        _valida_campos(Dono)


def test_condicao_sem_busca_por_campo_e_recusada_com_a_mensagem_inteira():
    class Condicional(BaseModel):
        a: str
        b: str = Field(json_schema_extra={"depends_on": ["a"]})

    mensagem = (
        "Campos ['b'] usam 'condition' ou 'depends_on' em json_schema_extra, que só são "
        "aplicados com use_search=True e search_per_field=True"
    )
    with pytest.raises(ValueError, match=_exata(mensagem)):
        _valida_campos(Condicional)


def test_json_schema_extra_sem_chave_do_dataframeit_passa_na_validacao():
    """Só as chaves de configuração e de condição pedem busca ou recusam o aninhamento."""

    class Folha(BaseModel):
        valor: str = Field(json_schema_extra={"examples": ["v"]})

    class Dono(BaseModel):
        a: str = Field(json_schema_extra={"examples": ["v"]})
        grupos: list[list[Folha]]

    assert _valida_campos(Dono) is None


def test_checkpoint_recusado_avisa_na_linha_que_chamou_dataframeit(tmp_path):
    ckpt = tmp_path / "ckpt.csv"
    df = pd.DataFrame({"texto": ["a"]})
    _roda(df, _llm("v")[1], batch_size=1, checkpoint_path=ckpt)
    Path(f"{ckpt}.dataframeit.json").unlink()

    mensagem = (
        f"O checkpoint em {ckpt} não tem a assinatura da execução que o gravou; esta execução "
        "começa do zero e o sobrescreve. Para retomá-lo mesmo assim, carregue-o com read_df e "
        "passe-o como entrada."
    )
    chamadas, llm = _llm("novo")
    with pytest.warns(UserWarning, match=_exata(mensagem)) as avisos:
        saida = _roda(df, llm, batch_size=1, checkpoint_path=ckpt)

    assert [type(aviso.message) for aviso in avisos] == [UserWarning]
    assert avisos[0].filename == __file__
    assert len(chamadas) == 1
    assert saida["campo1"].tolist() == ["novo1"]


def test_retomada_traz_o_campo_que_a_entrada_nao_tem_e_processa_so_as_pendentes(tmp_path):
    ckpt = tmp_path / "ckpt.csv"
    df = pd.DataFrame({"texto": ["a", "b", "c"]})
    opcoes = {
        "batch_size": 1,
        "checkpoint_path": ckpt,
        "status_column": "st",
        "track_tokens": False,
    }
    with pytest.raises(SystemExit):
        _roda(df, _llm("v", limite=1)[1], **opcoes)

    chamadas, llm = _llm("novo")
    saida = _roda(df, llm, **opcoes)

    assert len(chamadas) == 2
    assert saida.columns.tolist() == ["texto", "campo1"]
    assert saida["campo1"].tolist() == ["v1", "novo1", "novo2"]
