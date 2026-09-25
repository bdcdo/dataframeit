"""Checkpoint que falha ao gravar e retomada a partir de CSV/XLSX."""

import datetime
import warnings
from typing import Literal, Optional
from unittest.mock import patch

import pandas as pd
import pytest
from pydantic import BaseModel, Field, model_validator

from dataframeit import read_df
from dataframeit.core import _save_checkpoint, _validate_processed_rows, dataframeit
from dataframeit.utils import accepts_only_text, normalize_value


class ModeloSimples(BaseModel):
    campo: str


def _llm_contador():
    chamadas = []

    def call_langchain(text, *args, **kwargs):
        chamadas.append(text)
        return {"data": {"campo": f"v-{text}"}, "usage": None}

    return chamadas, call_langchain


# =============================================================================
# Falha de gravação não é falha da linha
# =============================================================================


@pytest.mark.parametrize("parallel_requests", [1, 3])
def test_falha_de_gravacao_nao_marca_a_linha_como_erro(tmp_path, parallel_requests):
    """Um OSError ao gravar não reescreve como erro uma linha já processada."""
    gravacoes = []

    def grava_ou_falha(df, path):
        gravacoes.append(int((df["_dataframeit_status"] == "processed").sum()))
        if len(gravacoes) == 1:
            raise OSError(28, "No space left on device")
        _save_checkpoint(df, path)

    _, llm = _llm_contador()
    ckpt = tmp_path / "ckpt.csv"
    with (
        patch("dataframeit.core._save_checkpoint", side_effect=grava_ou_falha),
        patch("dataframeit.core.call_langchain", side_effect=llm),
        patch("dataframeit.core.validate_provider_dependencies"),
        warnings.catch_warnings(record=True) as avisos,
    ):
        warnings.simplefilter("always")
        resultado = dataframeit(
            pd.DataFrame({"texto": list("abcdef")}),
            questions=ModeloSimples,
            prompt="Analise {texto}",
            batch_size=2,
            checkpoint_path=ckpt,
            parallel_requests=parallel_requests,
        )

    assert "_dataframeit_status" not in resultado.columns  # nenhuma linha com erro
    assert resultado["campo"].tolist() == [f"v-{c}" for c in "abcdef"]
    assert any("checkpoint" in str(a.message).lower() for a in avisos)
    # O arquivo final tem todas as linhas, mesmo com a primeira gravação falhando
    assert (pd.read_csv(ckpt)["_dataframeit_status"] == "processed").sum() == 6


@pytest.mark.parametrize("parallel_requests", [1, 3])
def test_gravacao_sempre_falhando_nao_interrompe_a_execucao(tmp_path, parallel_requests):
    _, llm = _llm_contador()
    with (
        patch("dataframeit.core._save_checkpoint", side_effect=OSError("disco cheio")),
        patch("dataframeit.core.call_langchain", side_effect=llm),
        patch("dataframeit.core.validate_provider_dependencies"),
        warnings.catch_warnings(record=True),
    ):
        warnings.simplefilter("always")
        resultado = dataframeit(
            pd.DataFrame({"texto": list("abcde")}),
            questions=ModeloSimples,
            prompt="Analise {texto}",
            batch_size=2,
            checkpoint_path=tmp_path / "ckpt.csv",
            parallel_requests=parallel_requests,
        )

    assert resultado["campo"].tolist() == [f"v-{c}" for c in "abcde"]
    assert "_dataframeit_status" not in resultado.columns


# =============================================================================
# Ida e volta por CSV e XLSX
# =============================================================================


class Parte(BaseModel):
    nome: str
    papel: str


class ModeloRico(BaseModel):
    ano: str
    observacao: Optional[str] = None
    tags: list[str]
    parte: Parte
    extra: Optional[dict[str, int]] = None


_RESPOSTA_RICA = {
    "ano": "2023",
    "observacao": "",
    "tags": ["a", "b"],
    "parte": {"nome": "Ana", "papel": "autora"},
    "extra": {"x": 1},
}


@pytest.mark.parametrize("extensao", [".csv", ".xlsx"])
def test_retomada_de_checkpoint_textual_nao_reprocessa(tmp_path, extensao):
    if extensao == ".xlsx":
        pytest.importorskip("openpyxl")
    ckpt = tmp_path / f"ckpt{extensao}"

    def responde(text, *args, **kwargs):
        observacao = None if text == "x" else "N/A"
        return {"data": {**_RESPOSTA_RICA, "observacao": observacao}, "usage": None}

    with (
        patch("dataframeit.core.call_langchain", side_effect=responde),
        patch("dataframeit.core.validate_provider_dependencies"),
    ):
        dataframeit(
            pd.DataFrame({"texto": ["x", "y"]}),
            questions=ModeloRico,
            prompt="Analise {texto}",
            batch_size=1,
            checkpoint_path=ckpt,
        )

    df_lido = read_df(str(ckpt), ModeloRico)
    with (
        patch("dataframeit.core.call_langchain") as call_langchain,
        patch("dataframeit.core.validate_provider_dependencies"),
    ):
        retomado = dataframeit(
            df_lido,
            questions=ModeloRico,
            prompt="Analise {texto}",
            resume=True,
        )

    call_langchain.assert_not_called()
    primeira = retomado.iloc[0]
    assert primeira["ano"] == "2023"
    assert pd.isna(primeira["observacao"])
    assert primeira["tags"] == ["a", "b"]
    assert primeira["parte"] == {"nome": "Ana", "papel": "autora"}
    assert primeira["extra"] == {"x": 1}
    # "N/A" é resposta, não ausência
    assert retomado.iloc[1]["observacao"] == "N/A"


def test_checkpoint_textual_grava_estruturas_como_json(tmp_path):
    ckpt = tmp_path / "ckpt.csv"
    _save_checkpoint(
        pd.DataFrame({"tags": [["a", "b"]], "parte": [{"nome": "Ana"}], "texto": ["t"]}),
        ckpt,
    )
    bruto = pd.read_csv(ckpt)
    assert bruto["tags"][0] == '["a", "b"]'
    assert bruto["parte"][0] == '{"nome": "Ana"}'


def test_normalize_value_aceita_repr_python_de_checkpoint_antigo():

    assert normalize_value("['a', 'b']") == ["a", "b"]
    assert normalize_value("{'nome': 'Ana', 'ok': True}") == {"nome": "Ana", "ok": True}
    assert normalize_value("[texto solto") == "[texto solto"


# =============================================================================
# Campo condicional pulado na retomada
# =============================================================================


def test_retomada_aceita_campo_condicional_pulado_com_tipo_obrigatorio():
    class Pessoa(BaseModel):
        tipo: str
        cpf: str = Field(json_schema_extra={"condition": {"field": "tipo", "equals": "pf"}})

    df = pd.DataFrame(
        {
            "texto": ["a", "b"],
            "tipo": ["pf", "pj"],
            "cpf": ["111", None],
            "_dataframeit_status": ["processed", "processed"],
        }
    )
    with (
        patch("dataframeit.core.validate_provider_dependencies"),
        patch("dataframeit.core.validate_search_dependencies"),
        patch("dataframeit.agent.call_agent") as call_agent,
    ):
        resultado = dataframeit(
            df,
            questions=Pessoa,
            prompt="Analise {texto}",
            use_search=True,
            search_per_field=True,
            resume=True,
        )

    call_agent.assert_not_called()
    assert resultado["cpf"].iloc[0] == "111"
    assert pd.isna(resultado["cpf"].iloc[1])


def test_texto_obrigatorio_vazio_em_csv_continua_acusado(tmp_path):
    """CSV grava "" e None do mesmo jeito; a retomada não adivinha qual era."""

    class ComObservacao(BaseModel):
        observacao: str

    ckpt = tmp_path / "ckpt.csv"
    with (
        patch(
            "dataframeit.core.call_langchain",
            return_value={"data": {"observacao": ""}, "usage": None},
        ),
        patch("dataframeit.core.validate_provider_dependencies"),
    ):
        dataframeit(
            pd.DataFrame({"texto": ["x"]}),
            questions=ComObservacao,
            prompt="{texto}",
            batch_size=1,
            checkpoint_path=ckpt,
        )

    with (
        patch("dataframeit.core.validate_provider_dependencies"),
        pytest.raises(ValueError, match="observacao"),
    ):
        dataframeit(
            read_df(str(ckpt), ComObservacao),
            questions=ComObservacao,
            prompt="{texto}",
            resume=True,
        )


def test_campo_condicional_ausente_com_condicao_verdadeira_e_acusado():
    class Pessoa(BaseModel):
        tipo: str
        cpf: str = Field(json_schema_extra={"condition": {"field": "tipo", "equals": "pf"}})

    df = pd.DataFrame(
        {
            "texto": ["a"],
            "tipo": ["pf"],
            "cpf": [None],
            "_dataframeit_status": ["processed"],
        }
    )
    with (
        patch("dataframeit.core.validate_provider_dependencies"),
        patch("dataframeit.core.validate_search_dependencies"),
        pytest.raises(ValueError, match="cpf"),
    ):
        dataframeit(
            df,
            questions=Pessoa,
            prompt="{texto}",
            use_search=True,
            search_per_field=True,
            resume=True,
        )


def test_campo_condicional_presente_mantem_as_restricoes():

    class Pessoa(BaseModel):
        tipo: str
        cpf: str = Field(
            min_length=11,
            json_schema_extra={"condition": {"field": "tipo", "equals": "pf"}},
        )

    df = pd.DataFrame(
        {
            "tipo": ["pf", "pj"],
            "cpf": ["123", None],
            "_dataframeit_status": ["processed", "processed"],
        }
    )
    incompativeis, _ = _validate_processed_rows(df, "_dataframeit_status", Pessoa, set())
    assert incompativeis == ["cpf"]


@pytest.mark.parametrize("parallel_requests", [1, 3])
def test_falha_na_ultima_gravacao_intermediaria_e_coberta_pela_final(tmp_path, parallel_requests):
    gravacoes = []

    def grava(df, path):
        processadas = int((df["_dataframeit_status"] == "processed").sum())
        gravacoes.append(processadas)
        if processadas == 6 and len(gravacoes) < 4:
            msg = "disco cheio"
            raise OSError(msg)
        _save_checkpoint(df, path)

    _, llm = _llm_contador()
    ckpt = tmp_path / "ckpt.csv"
    with (
        patch("dataframeit.core._save_checkpoint", side_effect=grava),
        patch("dataframeit.core.call_langchain", side_effect=llm),
        patch("dataframeit.core.validate_provider_dependencies"),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore")
        dataframeit(
            pd.DataFrame({"texto": list("abcdef")}),
            questions=ModeloSimples,
            prompt="{texto}",
            batch_size=2,
            checkpoint_path=ckpt,
            parallel_requests=parallel_requests,
        )

    # A gravação de 6 linhas falhou; a final regrava o estado completo
    assert gravacoes[-1] == 6
    assert (pd.read_csv(ckpt)["_dataframeit_status"] == "processed").sum() == 6


def test_checkpoint_textual_serializa_data_dentro_de_dict(tmp_path):

    ckpt = tmp_path / "ckpt.csv"
    _save_checkpoint(pd.DataFrame({"meta": [{"quando": datetime.date(2026, 9, 24)}]}), ckpt)
    assert pd.read_csv(ckpt)["meta"][0] == '{"quando": "2026-09-24"}'


@pytest.mark.parametrize("texto", ["{[1]: 2} petição", "{[1], [2]}", "[1, 2"])
def test_normalize_value_devolve_o_texto_quando_nao_e_estrutura(texto):

    assert normalize_value(texto) == texto


@pytest.mark.parametrize("texto", ["{1, 2}", "42"])
def test_normalize_value_so_aceita_lista_dict_ou_tupla(texto):

    assert normalize_value(texto) == texto


def test_read_df_respeita_dtype_do_usuario_e_nao_mexe_em_parquet(tmp_path):
    pytest.importorskip("pyarrow")

    class Ano(BaseModel):
        ano: str

    csv = tmp_path / "a.csv"
    pd.DataFrame({"ano": ["2023"]}).to_csv(csv, index=False)
    assert read_df(str(csv), Ano, dtype={"ano": "int64"})["ano"][0] == 2023

    parquet = tmp_path / "a.parquet"
    pd.DataFrame({"ano": ["2023"]}).to_parquet(parquet)
    assert read_df(str(parquet), Ano)["ano"][0] == "2023"


def test_accepts_only_text_cobre_optional_e_literal():

    assert accepts_only_text(Optional[str])
    assert accepts_only_text(Literal["a", "b"])
    assert accepts_only_text(Optional[Literal["a"]])
    assert not accepts_only_text(Literal["a", 1])
    assert not accepts_only_text(Optional[int])
    assert not accepts_only_text(list[str])


# =============================================================================
# Validação das linhas já processadas na retomada
# =============================================================================


def _retomar(df, modelo, **opcoes):
    """Retoma um checkpoint sem provider; devolve o resultado e o dublê do LLM."""
    with (
        patch("dataframeit.core.validate_provider_dependencies"),
        patch("dataframeit.core.validate_search_dependencies"),
        patch("dataframeit.agent.call_agent") as call_agent,
        patch("dataframeit.core.call_langchain") as call_langchain,
    ):
        resultado = dataframeit(df, questions=modelo, prompt="{texto}", resume=True, **opcoes)
    return resultado, call_agent, call_langchain


def test_condicao_que_levanta_na_retomada_conta_como_verdadeira():
    """Sem saber se o campo foi pulado, a retomada acusa o obrigatório ausente.

    'in' com um número levanta TypeError ao comparar; a condição callable não
    serve aqui, porque evaluate_condition já captura o erro dela.
    """

    class Pessoa(BaseModel):
        tipo: str
        cpf: str = Field(json_schema_extra={"condition": {"field": "tipo", "in": 5}})

    df = pd.DataFrame(
        {"texto": ["a"], "tipo": ["pf"], "cpf": [None], "_dataframeit_status": ["processed"]}
    )

    with pytest.raises(ValueError, match=r"campos incompatíveis \['cpf'\]"):
        _retomar(df, Pessoa, use_search=True, search_per_field=True)


def test_erro_em_campo_com_alias_acusa_so_esse_campo():
    """O Pydantic localiza o erro pelo alias; a retomada o traduz para o nome do campo."""

    class Registro(BaseModel):
        nome: str
        cpf: int = Field(alias="CPF")

    df = pd.DataFrame(
        {
            "texto": ["a"],
            "nome": ["Ana"],
            "cpf": ["não é número"],
            "_dataframeit_status": ["processed"],
        }
    )

    with pytest.raises(ValueError, match=r"campos incompatíveis \['cpf'\]"):
        _retomar(df, Registro)


def test_validador_do_modelo_inteiro_acusa_todos_os_campos():
    """Erro sem campo, de um model_validator, não diz qual coluna refazer."""

    class Intervalo(BaseModel):
        inicio: int
        fim: int

        @model_validator(mode="after")
        def fim_depois_do_inicio(self):
            if self.fim < self.inicio:
                msg = "fim antes do início"
                raise ValueError(msg)
            return self

    df = pd.DataFrame(
        {"texto": ["a"], "inicio": [5], "fim": [1], "_dataframeit_status": ["processed"]}
    )

    with pytest.raises(ValueError, match=r"campos incompatíveis \['inicio', 'fim'\]"):
        _retomar(df, Intervalo)


def test_default_factory_que_le_outro_campo_invalido_tambem_e_acusado():
    """Com o campo lido pela factory inválido, o default não pode ser calculado.

    A factory lê com .get: antes da 2.12, o Pydantic a chama mesmo depois de
    'quantidade' falhar, com os dados validados vazios, e um dados['quantidade']
    levantaria KeyError em vez de ValidationError.
    """

    class Pedido(BaseModel):
        quantidade: int
        rotulo: str = Field(default_factory=lambda dados: f"{dados.get('quantidade')} unidades")

    df = pd.DataFrame(
        {
            "texto": ["a"],
            "quantidade": ["muitos"],
            "rotulo": [None],
            "_dataframeit_status": ["processed"],
        }
    )

    with pytest.raises(ValueError, match=r"campos incompatíveis \['quantidade', 'rotulo'\]"):
        _retomar(df, Pedido)


def test_retomada_com_varias_linhas_puladas_pela_mesma_condicao():
    class Pessoa(BaseModel):
        tipo: str
        cpf: str = Field(json_schema_extra={"condition": {"field": "tipo", "equals": "pf"}})

    df = pd.DataFrame(
        {
            "texto": ["a", "b", "c"],
            "tipo": ["pj", "pf", "pj"],
            "cpf": [None, "111", None],
            "_dataframeit_status": ["processed"] * 3,
        }
    )

    resultado, call_agent, _ = _retomar(df, Pessoa, use_search=True, search_per_field=True)

    call_agent.assert_not_called()
    assert resultado["cpf"].iloc[1] == "111"
    assert resultado["cpf"].iloc[[0, 2]].isna().all()
