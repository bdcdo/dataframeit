"""Checkpoint que falha ao gravar e retomada a partir de CSV/XLSX."""

import datetime
import warnings
from enum import Enum
from typing import Annotated, Any, Literal, NewType, Optional, TypeVar
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel, Field, model_validator
from typing_extensions import TypeAliasType

from dataframeit import read_df
from dataframeit.core import _save_checkpoint, _validate_processed_rows, dataframeit
from dataframeit.utils import _admits_none, accepts_only_text, normalize_value


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

    def grava_ou_falha(df, path, *_):
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
    """Passado como entrada, o CSV relido não traz a assinatura que diz que o modelo é o mesmo.

    O CSV grava "" e None do mesmo jeito, e com outro modelo a célula vazia pode
    ser o None de um campo que o admitia; a retomada não adivinha qual era.
    """

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


def test_texto_vazio_do_checkpoint_respeita_a_condicao_do_campo():
    """A célula vazia só vira texto vazio quando a condição do campo é verdadeira na linha."""

    class Pessoa(BaseModel):
        tipo: str
        cpf: str = Field(json_schema_extra={"condition": {"field": "tipo", "equals": "pf"}})

    df = pd.DataFrame(
        {
            "tipo": ["pf", "pj"],
            "cpf": [None, None],
            "_dataframeit_status": ["processed", "processed"],
        }
    )
    incompativeis, valores = _validate_processed_rows(
        df, "_dataframeit_status", Pessoa, set(), empty_text_fields={"cpf"}
    )

    assert incompativeis == []
    assert valores[(0, "cpf")] == ""
    assert valores[(1, "cpf")] is None


@pytest.mark.parametrize(
    "condicao", [{"field": "motivo", "exists": True}, {"field": "motivo", "equals": ""}]
)
def test_condicao_le_o_texto_vazio_restaurado_de_outro_campo(condicao):
    """A condição vê o texto vazio do campo de que depende, como a execução o viu."""

    class Pedido(BaseModel):
        motivo: str
        obs: str = Field(json_schema_extra={"condition": condicao})

    df = pd.DataFrame({"motivo": [None], "obs": [None], "_dataframeit_status": ["processed"]})
    incompativeis, valores = _validate_processed_rows(
        df, "_dataframeit_status", Pedido, set(), empty_text_fields={"motivo", "obs"}
    )

    assert incompativeis == []
    assert valores[(0, "motivo")] == ""
    assert valores[(0, "obs")] == ""


def test_condicao_le_o_none_do_campo_pulado_antes_dela():
    """O campo pulado fica None para as condições que dependem dele, em qualquer ordem do modelo.

    O obs vem antes do motivo, de que depende, e o motivo só existe quando o
    tipo é "pf". Na linha "pj", o motivo pulado é None, e o obs também é pulado.
    """

    class Pedido(BaseModel):
        obs: str = Field(json_schema_extra={"condition": {"field": "motivo", "exists": True}})
        motivo: str = Field(json_schema_extra={"condition": {"field": "tipo", "equals": "pf"}})
        tipo: str

    df = pd.DataFrame(
        {
            "obs": [None, None],
            "motivo": [None, None],
            "tipo": ["pf", "pj"],
            "_dataframeit_status": ["processed", "processed"],
        }
    )
    incompativeis, valores = _validate_processed_rows(
        df, "_dataframeit_status", Pedido, set(), empty_text_fields={"motivo", "obs"}
    )

    assert incompativeis == []
    assert [valores[(0, "motivo")], valores[(0, "obs")]] == ["", ""]
    assert [valores[(1, "motivo")], valores[(1, "obs")]] == [None, None]


@pytest.mark.parametrize("parallel_requests", [1, 3])
def test_falha_na_ultima_gravacao_intermediaria_e_coberta_pela_final(tmp_path, parallel_requests):
    gravacoes = []

    def grava(df, path, *_):
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


def test_checkpoint_textual_serializa_array_e_enum_como_json(tmp_path):
    """Array no topo e dentro de dict vira lista, e Enum vai pelo valor."""

    class Cor(Enum):
        AZUL = "azul"

    ckpt = tmp_path / "ckpt.csv"
    df = pd.DataFrame(
        {
            "tags": [np.array(["x", "y"], dtype=object)],
            "numeros": [[np.int64(5), np.bool_(True)]],
            "interno": [{"itens": np.array(["a"], dtype=object), "cor": Cor.AZUL}],
        }
    )
    _save_checkpoint(df, ckpt)

    relido = pd.read_csv(ckpt)
    assert relido["tags"][0] == '["x", "y"]'
    assert relido["numeros"][0] == "[5, true]"
    assert relido["interno"][0] == '{"itens": ["a"], "cor": "azul"}'


def test_checkpoint_textual_grava_conjunto_em_ordem_fixa(tmp_path):
    """A ordem de um set muda de um processo para outro; o JSON dele sai sempre na mesma.

    O {8, 0} é percorrido como 8, 0 no CPython, porque os dois caem na mesma
    posição da tabela e o 8 entra primeiro.
    """
    ckpt = tmp_path / "ckpt.csv"
    df = pd.DataFrame(
        {"numeros": [{8, 0}], "textos": [frozenset({"b", "a"})], "dentro": [{"s": {8, 0}}]}
    )
    _save_checkpoint(df, ckpt)

    relido = pd.read_csv(ckpt)
    assert relido["numeros"][0] == "[0, 8]"
    assert relido["textos"][0] == '["a", "b"]'
    assert relido["dentro"][0] == '{"s": [0, 8]}'


def test_checkpoint_xlsx_grava_como_texto_o_que_comeca_com_igual(tmp_path):
    """O openpyxl tomaria o texto por fórmula, que o pandas relê vazia."""
    pytest.importorskip("openpyxl")
    ckpt = tmp_path / "ckpt.xlsx"
    df = pd.DataFrame({"resposta": ["=1+1", "=SOMA(A1;A2)"], "_error_details": [None, "=x"]})
    _save_checkpoint(df, ckpt)

    relido = pd.read_excel(ckpt)
    assert relido["resposta"].tolist() == ["=1+1", "=SOMA(A1;A2)"]
    assert relido["_error_details"].tolist()[1] == "=x"


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
    # A forma que o lint recusa por estilo é a que o usuário escreve no modelo.
    assert accepts_only_text(Literal["NA", "deferido", None])  # noqa: PYI061
    assert not accepts_only_text(Literal[None])  # noqa: PYI061
    assert not accepts_only_text(Literal["a", 1])
    assert not accepts_only_text(Optional[int])
    assert not accepts_only_text(list[str])
    assert accepts_only_text(Annotated[str, Field(min_length=1)] | None)
    assert not accepts_only_text(Annotated[int, Field(ge=0)] | None)

    class Texto(Enum):
        A = "1"

    class Numero(Enum):
        A = 1

    assert accepts_only_text(Optional[Texto])
    assert not accepts_only_text(Numero)


# O None dentro do Literal, que o lint recusa por estilo, é a forma que o usuário escreve.
_LITERAL_COM_NONE = Literal["NA", "deferido", None]  # noqa: PYI061
_ANOTADO_COM_NONE = Annotated[Literal["NA", None], Field(description="x")]  # noqa: PYI061
_APELIDO_COM_NONE = TypeAliasType("_APELIDO_COM_NONE", _ANOTADO_COM_NONE)


@pytest.mark.parametrize(
    ("anotacao", "admite"),
    [
        (str, False),
        (Optional[str], True),
        (Any, True),
        (object, True),
        (_LITERAL_COM_NONE, True),
        (Literal["NA"], False),
        (_ANOTADO_COM_NONE | Literal["y"], True),
        (Annotated[str, Field(min_length=1)] | Literal["y"], False),
        (_APELIDO_COM_NONE, True),
    ],
)
def test_admits_none_ve_o_none_na_uniao_no_literal_e_no_annotated(anotacao, admite):
    """O pydantic tira só o Annotated de cima; o de dentro da união e o do apelido ficam."""
    assert _admits_none(anotacao) is admite


class Inteiros(BaseModel):
    nota: Annotated[int, Field(ge=0)] | None = None
    misto: int | str | None = None
    decimal: float | None = None
    medida: Annotated[float, Field(ge=0)] | None = None


@pytest.mark.parametrize("formato", ["csv", "parquet"])
def test_read_df_com_modelo_rele_campo_int_como_int(tmp_path, formato):
    """O leitor devolve float64 na coluna com int e célula vazia; o campo int volta int (#223)."""
    pytest.importorskip("pyarrow")
    arquivo = tmp_path / f"dados.{formato}"
    colunas = {"texto": ["a", "b"]} | {campo: [7, None] for campo in Inteiros.model_fields}
    _save_checkpoint(pd.DataFrame(colunas, dtype=object), arquivo, structures=frozenset())

    lido = read_df(str(arquivo), Inteiros)

    assert list(lido["nota"]) == [7, None]
    assert type(lido["nota"][0]) is int
    assert lido["nota"][1] is None
    assert lido["misto"].dtype == "float64"
    assert lido["decimal"].dtype == "float64"
    assert lido["medida"].dtype == "float64"


def test_read_df_nao_rele_campo_int_do_xlsx(tmp_path):
    """O XLSX guarda número como double; o int dele pode vir arredondado e fica como lido."""
    pytest.importorskip("openpyxl")
    arquivo = tmp_path / "dados.xlsx"
    dados = pd.DataFrame({"texto": ["a", "b"], "nota": [9007199254740993, None]}, dtype=object)
    _save_checkpoint(dados, arquivo, structures=frozenset())

    lido = read_df(str(arquivo), Inteiros)

    assert lido["nota"].dtype == "float64"
    assert lido["nota"][0] == 9007199254740992.0


def test_read_df_sem_releitura_de_int_do_parquet_com_columns(tmp_path):
    """Qualquer argumento de leitura desliga a releitura, também no parquet."""
    pytest.importorskip("pyarrow")
    arquivo = tmp_path / "dados.parquet"
    dados = pd.DataFrame({"texto": ["a", "b"], "nota": [7, None]}, dtype=object)
    _save_checkpoint(dados, arquivo, structures=frozenset())

    lido = read_df(str(arquivo), Inteiros, columns=["nota"])

    pd.testing.assert_frame_equal(lido, pd.read_parquet(arquivo, columns=["nota"]))


@pytest.mark.parametrize("formato", ["csv", "parquet"])
def test_read_df_rele_ausente_e_zero_ao_lado_do_inteiro_grande(tmp_path, formato):
    """Cada célula volta na sua linha: o 0 continua 0, e o ausente antes do int continua antes."""
    pytest.importorskip("pyarrow")
    arquivo = tmp_path / f"dados.{formato}"
    dados = pd.DataFrame(
        {"texto": ["a", "b", "c"], "nota": [None, 0, 9007199254740993]}, dtype=object
    )
    _save_checkpoint(dados, arquivo, structures=frozenset())

    lido = read_df(str(arquivo), Inteiros)

    assert list(lido["nota"]) == [None, 0, 9007199254740993]
    assert [type(v) for v in lido["nota"]] == [type(None), int, int]


def test_read_df_rele_float_e_int_do_csv_sem_celula_vazia(tmp_path):
    """Sem célula vazia, a coluna com 7.5 e 8 também é relida: o 8 volta int, o 7.5 fica float."""
    arquivo = tmp_path / "dados.csv"
    arquivo.write_text("texto,nota\na,7.5\nb,8\n", encoding="utf-8")

    lido = read_df(str(arquivo), Inteiros)

    assert list(lido["nota"]) == [7.5, 8]
    assert [type(v) for v in lido["nota"]] == [float, int]


def test_read_df_com_on_bad_lines_devolve_o_que_o_pandas_le(tmp_path):
    """A linha que o leitor pula não desalinha a releitura: com o argumento, ela não roda."""
    arquivo = tmp_path / "dados.csv"
    arquivo.write_text("nome,nota\na,9007199254740993\nb,,extra,mais\nc,\nd,7\n", encoding="utf-8")

    lido = read_df(str(arquivo), Inteiros, on_bad_lines="skip")

    pd.testing.assert_frame_equal(lido, pd.read_csv(arquivo, on_bad_lines="skip"))


def test_read_df_mantem_o_float_que_o_csv_guarda_no_campo_int(tmp_path):
    """Só a célula gravada como inteiro muda; a de um float ou de "+7" fica como lida.

    A célula que o leitor devolve ausente, vazia ou "NA", volta None, e não NaN.
    """
    arquivo = tmp_path / "dados.csv"
    arquivo.write_text("texto,nota\na,7\nb,7.5\nc,-3\nd,+7\ne,\nf,NA\n", encoding="utf-8")

    lido = read_df(str(arquivo), Inteiros)

    assert list(lido["nota"]) == [7, 7.5, -3, 7.0, None, None]
    assert [type(v) for v in lido["nota"]] == [int, float, int, float, type(None), type(None)]


def test_read_df_rele_do_csv_o_int_acima_do_int64(tmp_path):
    """Com célula vazia, o leitor devolve texto na coluna com int acima do Int64."""
    arquivo = tmp_path / "dados.csv"
    arquivo.write_text("texto,nota\na,9223372036854775813\nb,7\nc,\n", encoding="utf-8")

    lido = read_df(str(arquivo), Inteiros)

    assert list(lido["nota"]) == [9223372036854775813, 7, None]
    assert [type(v) for v in lido["nota"]] == [int, int, type(None)]


def test_read_df_sem_o_campo_int_no_arquivo(tmp_path):
    arquivo = tmp_path / "dados.csv"
    arquivo.write_text("texto,decimal\na,1.5\nb,\n", encoding="utf-8")

    lido = read_df(str(arquivo), Inteiros)

    assert list(lido.columns) == ["texto", "decimal"]


def test_read_df_rele_campo_int_do_parquet_com_indice_proprio(tmp_path):
    """O parquet que guarda o índice o devolve, e a coluna relida fica nas linhas dele."""
    pytest.importorskip("pyarrow")
    arquivo = tmp_path / "dados.parquet"
    pd.DataFrame({"nota": [7, None]}, index=[10, 20], dtype=object).to_parquet(arquivo)

    lido = read_df(str(arquivo), Inteiros)

    assert list(lido.index) == [10, 20]
    assert list(lido["nota"]) == [7, None]


@pytest.mark.parametrize("formato", ["csv", "parquet", "json"])
@pytest.mark.parametrize(("valores", "tipo"), [([7.5, None], "float64"), ([7, 8], "int64")])
def test_read_df_nao_mexe_no_campo_int_sem_celula_a_reler(tmp_path, formato, valores, tipo):
    """Coluna sem célula vazia, float no arquivo e leitor sem releitura ficam como lidos."""
    pytest.importorskip("pyarrow")
    arquivo = tmp_path / f"dados.{formato}"
    getattr(pd.DataFrame({"nota": valores}), f"to_{formato}")(arquivo, index=False)

    assert read_df(str(arquivo), Inteiros)["nota"].dtype == tipo


@pytest.mark.parametrize(
    ("formato", "opcao", "tipo"),
    [
        ("csv", {"dtype": {"nota": "float64"}}, "float64"),
        ("parquet", {"dtype_backend": "pyarrow"}, "double[pyarrow]"),
    ],
)
def test_read_df_respeita_os_tipos_do_usuario_no_campo_int(tmp_path, formato, opcao, tipo):
    pytest.importorskip("pyarrow")
    arquivo = tmp_path / f"dados.{formato}"
    if formato == "csv":
        arquivo.write_text("texto,nota\na,7\nb,\n", encoding="utf-8")
    else:
        pd.DataFrame({"nota": [7.0, None]}).to_parquet(arquivo)

    assert read_df(str(arquivo), Inteiros, **opcao)["nota"].dtype == tipo


@pytest.mark.parametrize(
    "opcao",
    [
        {"converters": {"id": str}},
        {"na_values": ["x"]},
        {"keep_default_na": False},
        {"na_filter": False},
        {"usecols": ["id", "nota"]},
        {"index_col": "id"},
        {"index_col": 0},
        {"parse_dates": ["data"]},
        {"engine": "pyarrow"},
    ],
    ids=lambda opcao: next(iter(opcao)),
)
def test_read_df_sem_releitura_de_int_com_opcao_de_leitura(tmp_path, opcao):
    """Com essas opções, read_df devolve o campo int como o pandas o lê, sem levantar."""
    if opcao.get("engine") == "pyarrow":
        pytest.importorskip("pyarrow")
    arquivo = tmp_path / "dados.csv"
    arquivo.write_text("id,data,nota\n10,2024-01-01,7\n20,2024-01-02,\n", encoding="utf-8")

    lido = read_df(str(arquivo), Inteiros, **opcao)

    pd.testing.assert_frame_equal(lido, pd.read_csv(arquivo, **opcao))


class ComNome(BaseModel):
    nome: str
    n: int


class ComLista(BaseModel):
    nome: str
    itens: list[str]


def _quadros(lido):
    """Os DataFrames do que o leitor devolveu: as abas do dict, ou os pedaços juntados."""
    if isinstance(lido, dict):
        return lido
    with lido:
        return {"pedacos": pd.concat(list(lido))}


# O que o leitor do pandas devolve, por formato, com as opções do teste abaixo.
_TIPO_DO_LEITOR = {"csv": "TextFileReader", "xlsx": "dict", "json": "JsonReader"}


@pytest.mark.parametrize("normalize", [True, False], ids=["normalize", "sem-normalize"])
@pytest.mark.parametrize(
    "modelo", [None, Inteiros, ComNome, ComLista], ids=["sem-modelo", "int", "texto", "lista"]
)
@pytest.mark.parametrize(
    ("formato", "opcao"),
    [
        ("csv", {"chunksize": 1}),
        ("csv", {"iterator": True}),
        ("xlsx", {"sheet_name": None}),
        ("xlsx", {"sheet_name": [0]}),
        ("json", {"lines": True, "chunksize": 1}),
    ],
    ids=["chunksize", "iterator", "sheet_name-None", "sheet_name-lista", "json-chunksize"],
)
def test_read_df_devolve_como_o_pandas_o_que_nao_e_dataframe(
    tmp_path, formato, opcao, modelo, normalize
):
    """O leitor que devolve outra coisa que um DataFrame passa direto, sem releitura nem normalização.

    Com ou sem modelo e com `normalize` ligado ou não, o objeto sai como o
    pandas o devolve: o JSON da coluna `itens` continua texto em cada pedaço e
    em cada aba.
    """
    if formato == "xlsx":
        pytest.importorskip("openpyxl")
    arquivo = tmp_path / f"dados.{formato}"
    dados = pd.DataFrame({"nome": ["a", "b"], "nota": [7, None], "itens": ['["x"]', "[]"]})
    if formato == "csv":
        dados.to_csv(arquivo, index=False)
        leitor = pd.read_csv
    elif formato == "xlsx":
        dados.to_excel(arquivo, index=False)
        leitor = pd.read_excel
    else:
        dados.to_json(arquivo, orient="records", lines=True)
        leitor = pd.read_json

    lido = read_df(str(arquivo), modelo, normalize, **opcao)

    assert type(lido).__name__ == _TIPO_DO_LEITOR[formato]
    quadros = _quadros(lido)
    esperados = _quadros(leitor(arquivo, **opcao))
    assert quadros.keys() == esperados.keys()
    for nome, quadro in quadros.items():
        pd.testing.assert_frame_equal(quadro, esperados[nome])
        assert quadro["itens"].tolist() == ['["x"]', "[]"]


_LINHA_MALFORMADA = "nome,n\na,1\nb,2,extra,mais\nc,3\nd,4\n"


def test_read_df_com_on_bad_lines_mantem_o_texto_na_sua_linha(tmp_path):
    """A linha pulada não leva o texto da seguinte para cima nem traz o próprio de volta."""
    arquivo = tmp_path / "dados.csv"
    arquivo.write_text(_LINHA_MALFORMADA, encoding="utf-8")

    lido = read_df(str(arquivo), ComNome, on_bad_lines="skip")

    assert lido[["nome", "n"]].to_numpy().tolist() == [["a", 1], ["c", 3], ["d", 4]]


@pytest.mark.filterwarnings("ignore::pandas.errors.ParserWarning")
@pytest.mark.parametrize(
    ("formato", "conteudo", "opcao"),
    [
        ("csv", _LINHA_MALFORMADA, {"on_bad_lines": "skip"}),
        ("csv", _LINHA_MALFORMADA, {"on_bad_lines": "warn"}),
        ("csv", _LINHA_MALFORMADA, {"on_bad_lines": lambda linha: linha[:2], "engine": "python"}),
        ("csv", "id,nome,n\n10,a,1\n20,b,2\n", {"index_col": 0}),
        ("csv", "nome,n\na,1\nb,2\n", {"index_col": "n"}),
        ("csv", "nome,n,data\na,1,2024-01-01\nb,2,2024-01-02\n", {"parse_dates": ["data"]}),
        ("csv", "nome,n\nx,a,1\ny,b,2\n", {"names": ["nome", "n"], "skiprows": 1}),
        ("xlsx", "id,nome,n\n10,a,1\n20,b,2\n", {"index_col": 0}),
    ],
    ids=[
        "on_bad_lines-skip",
        "on_bad_lines-warn",
        "on_bad_lines-callable",
        "index_col-posicao",
        "index_col-nome",
        "parse_dates",
        "names-com-indice-implicito",
        "xlsx-index_col",
    ],
)
def test_read_df_sem_releitura_de_texto_com_opcao_de_leitura(tmp_path, formato, conteudo, opcao):
    """Com argumento de leitura, o campo de texto volta como o pandas o lê, sem desalinhar nem levantar.

    Releitas só com as colunas de texto, essas opções devolvem outras linhas
    (a linha malformada deixa de ser malformada) ou levantam (a coluna do
    índice ou da data fica de fora da releitura).
    """
    arquivo = tmp_path / f"dados.{formato}"
    origem = tmp_path / "origem.csv"
    origem.write_text(conteudo, encoding="utf-8")
    if formato == "xlsx":
        pytest.importorskip("openpyxl")
        pd.read_csv(origem).to_excel(arquivo, index=False)
        leitor = pd.read_excel
    else:
        origem.rename(arquivo)
        leitor = pd.read_csv

    lido = read_df(str(arquivo), ComNome, **opcao)

    pd.testing.assert_frame_equal(lido, leitor(arquivo, **opcao))


def test_estrutura_anotada_e_opcional_volta_do_checkpoint_csv(tmp_path):
    """Annotated dentro de Optional: o CSV guarda JSON, e a retomada o relê como lista."""

    class Anotado(BaseModel):
        tags: Annotated[list[str], Field(min_length=1)] | None = None

    ckpt = tmp_path / "ckpt.csv"
    chamadas = []

    def llm(text, *args, **kwargs):
        chamadas.append(text)
        return {"data": {"tags": ["x", "y"]}, "usage": None}

    with (
        patch("dataframeit.core.call_langchain", side_effect=llm),
        patch("dataframeit.core.validate_provider_dependencies"),
    ):
        for _ in range(2):
            saida = dataframeit(
                pd.DataFrame({"texto": ["a"]}),
                questions=Anotado,
                prompt="{texto}",
                batch_size=1,
                checkpoint_path=ckpt,
            )

    assert read_df(str(ckpt), Anotado)["tags"][0] == ["x", "y"]
    assert saida["tags"][0] == ["x", "y"]
    assert chamadas == ["a"]


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

    A factory lê sem proteção: o Pydantic do piso declarado não a chama depois
    de 'quantidade' falhar, e acusa os dois campos em vez de deixar o KeyError
    dela escapar.
    """

    class Pedido(BaseModel):
        quantidade: int
        rotulo: str = Field(default_factory=lambda dados: f"{dados['quantidade']} unidades")

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


# =============================================================================
# NewType e apelido `type` contam pelo tipo a que se referem
# =============================================================================

_Codigo = NewType("_Codigo", str)
_Processo = NewType("_Processo", int)
_ApelidoDeTexto = TypeAliasType("_ApelidoDeTexto", str)
_ApelidoDeInteiro = TypeAliasType("_ApelidoDeInteiro", Optional[int])
_Metadados = NewType("_Metadados", dict[str, int])
_ApelidoDeLista = TypeAliasType("_ApelidoDeLista", list[str])
_T = TypeVar("_T")
_ApelidoDeListaGenerica = TypeAliasType("_ApelidoDeListaGenerica", list[_T], type_params=(_T,))
_ApelidoDeDictGenerico = TypeAliasType("_ApelidoDeDictGenerico", dict[str, _T], type_params=(_T,))


class ComNewTypeEApelido(BaseModel):
    codigo: _Codigo
    rotulo: _ApelidoDeTexto
    processo: Optional[_Processo] = None
    numero: _ApelidoDeInteiro = None
    escolha: Optional[Literal[9007199254740993, 1]] = None
    meta: Optional[_Metadados] = None
    etiquetas: Optional[_ApelidoDeLista] = None
    contagens: Optional[_ApelidoDeListaGenerica[int]] = None
    rotulos: _ApelidoDeListaGenerica[str] = []
    pesos: Optional[_ApelidoDeDictGenerico[int]] = None


def test_read_df_le_newtype_e_apelido_pelo_tipo_a_que_se_referem(tmp_path):
    arquivo = tmp_path / "dados.csv"
    arquivo.write_text(
        "codigo,rotulo,processo,numero,escolha,meta,etiquetas\n"
        '007,N/A,9007199254740993,9007199254740993,9007199254740993,"{""a"": 1}","[""x""]"\n'
        "N/A,007,,,,,\n",
        encoding="utf-8",
    )

    lido = read_df(str(arquivo), ComNewTypeEApelido)

    assert list(lido["codigo"]) == ["007", "N/A"]
    assert list(lido["rotulo"]) == ["N/A", "007"]
    assert list(lido["processo"]) == [9007199254740993, None]
    assert list(lido["numero"]) == [9007199254740993, None]
    assert list(lido["escolha"]) == [9007199254740993, None]
    assert lido["meta"][0] == {"a": 1}
    assert lido["etiquetas"][0] == ["x"]


def test_retomada_de_checkpoint_csv_com_newtype_e_apelido(tmp_path):
    ckpt = tmp_path / "ckpt.csv"
    resposta = {
        "codigo": "007",
        "rotulo": "N/A",
        "processo": 9007199254740993,
        "numero": None,
        "escolha": 9007199254740993,
        "meta": {"a": 1},
        "etiquetas": ["x"],
        "contagens": [1, 2],
        "rotulos": ["007"],
        "pesos": {"a": 1},
    }
    with (
        patch(
            "dataframeit.core.call_langchain",
            return_value={"data": resposta, "usage": None},
        ),
        patch("dataframeit.core.validate_provider_dependencies"),
    ):
        dataframeit(
            pd.DataFrame({"texto": ["x"]}),
            questions=ComNewTypeEApelido,
            prompt="Analise {texto}",
            batch_size=1,
            checkpoint_path=ckpt,
        )

    with (
        patch("dataframeit.core.call_langchain") as call_langchain,
        patch("dataframeit.core.validate_provider_dependencies"),
    ):
        retomado = dataframeit(
            read_df(str(ckpt), ComNewTypeEApelido),
            questions=ComNewTypeEApelido,
            prompt="Analise {texto}",
            resume=True,
        )

    call_langchain.assert_not_called()
    linha = retomado.iloc[0]
    for campo, valor in resposta.items():
        if valor is None:
            assert pd.isna(linha[campo])
        else:
            assert linha[campo] == valor
