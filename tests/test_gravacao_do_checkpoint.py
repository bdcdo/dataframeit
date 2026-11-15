"""Arquivo que a gravação do checkpoint escreve e mensagens dela.

Os testes vieram dos mutantes que sobreviviam ao teste de mutação da gravação
do checkpoint em core.py: o arquivo sai com as colunas exatas, sem índice, com
o texto não ASCII como está, e o temporário da gravação não sobra ao lado.
"""

import re
from pathlib import Path

import pandas as pd
import pytest

from dataframeit import core
from dataframeit.core import (
    _Checkpoint,
    _save_checkpoint,
    _set_field_traces,
    _SnapshotWriter,
    _try_save_checkpoint,
    _validate_checkpoint_extension,
)

_EXTS = ".csv, .xlsx, .parquet"


def _frame() -> pd.DataFrame:
    """Texto não ASCII, uma lista e um índice que não é o padrão."""
    return pd.DataFrame(
        {"texto": ["café", "=1+1"], "lista": [["ação", "pé"], []]},
        index=[10, 20],
    )


# =============================================================================
# Arquivo gravado
# =============================================================================


def test_csv_grava_sem_indice_e_com_texto_nao_ascii(tmp_path: Path) -> None:
    path = tmp_path / "saida.csv"

    _save_checkpoint(_frame(), path)

    assert path.read_text(encoding="utf-8") == ('texto,lista\ncafé,"[""ação"", ""pé""]"\n=1+1,[]\n')


def test_xlsx_grava_sem_indice_e_rele_o_texto_com_igual_como_texto(tmp_path: Path) -> None:
    pytest.importorskip("openpyxl")
    path = tmp_path / "saida.xlsx"

    _save_checkpoint(_frame(), path)

    assert pd.read_excel(path).to_dict("list") == {
        "texto": ["café", "=1+1"],
        "lista": ['["ação", "pé"]', "[]"],
    }


def test_xlsx_grava_toda_celula_como_texto_valido_do_excel(tmp_path: Path) -> None:
    """A célula que o openpyxl tomou por fórmula vira texto do tipo que o Excel conhece."""
    openpyxl = pytest.importorskip("openpyxl")
    path = tmp_path / "saida.xlsx"

    _save_checkpoint(_frame(), path)

    sheet = openpyxl.load_workbook(path).active
    assert [[cell.data_type for cell in row] for row in sheet.iter_rows()] == [["s", "s"]] * 3


def test_xlsx_usa_o_openpyxl_mesmo_com_outro_escritor_padrao_no_pandas(tmp_path: Path) -> None:
    pytest.importorskip("openpyxl")
    path = tmp_path / "saida.xlsx"

    with pd.option_context("io.excel.xlsx.writer", "xlsxwriter"):
        _save_checkpoint(_frame(), path)

    assert pd.read_excel(path)["texto"].tolist() == ["café", "=1+1"]


def test_parquet_grava_sem_o_indice_da_entrada(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    path = tmp_path / "saida.parquet"

    _save_checkpoint(_frame(), path, structures={"lista"})

    relido = pd.read_parquet(path)
    assert relido.columns.tolist() == ["texto", "lista"]
    assert relido.index.equals(pd.RangeIndex(2))
    assert relido["lista"].tolist() == ['["ação", "pé"]', "[]"]


def test_temporario_que_sobrou_de_gravacao_anterior_e_consumido(tmp_path: Path) -> None:
    path = tmp_path / "saida.csv"
    (tmp_path / "saida.csv.tmp").write_text("resto de gravação interrompida\n")

    _save_checkpoint(_frame(), path)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["saida.csv"]
    assert path.read_text(encoding="utf-8").startswith("texto,lista\n")


# =============================================================================
# Rastro por campo
# =============================================================================


def test_rastro_grava_json_com_texto_nao_ascii_e_deixa_as_outras_linhas_vazias() -> None:
    df = pd.DataFrame({"texto": ["a", "b"]}, index=["x", "y"])

    _set_field_traces(df, "x", {"campo": {"motivo": "não há ação"}})

    assert df["_trace_campo"].tolist() == ['{"motivo": "não há ação"}', None]


# =============================================================================
# Mensagens
# =============================================================================


@pytest.mark.parametrize(
    ("nome", "ext"),
    [("saida", "(nenhuma)"), ("saida.txt", ".txt"), ("saida.TXT", ".txt")],
)
def test_extensao_nao_suportada_diz_qual_e_quais_aceita(nome: str, ext: str) -> None:
    msg = f"Extensão {ext} não suportada para checkpoint. Use uma de: {_EXTS}"
    with pytest.raises(ValueError, match=f"^{re.escape(msg)}$"):
        _validate_checkpoint_extension(Path(nome))


def test_gravacao_com_extensao_nao_suportada_diz_quais_aceita(tmp_path: Path) -> None:
    msg = f"Extensão .txt não suportada para checkpoint. Use uma de: {_EXTS}"
    with pytest.raises(ValueError, match=f"^{re.escape(msg)}$"):
        _save_checkpoint(_frame(), tmp_path / "saida.txt")


def test_falha_de_gravacao_vira_aviso_com_a_mensagem_inteira(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def falha(*_args: object) -> None:
        msg = "disco cheio"
        raise OSError(msg)

    monkeypatch.setattr(core, "_save_checkpoint", falha)
    path = tmp_path / "saida.csv"
    msg = (
        f"Falha ao gravar o checkpoint em {path}: OSError: disco cheio. "
        "O processamento continua, e a próxima gravação tenta de novo."
    )

    with pytest.warns(UserWarning, match=f"^{re.escape(msg)}$"):
        assert _try_save_checkpoint(_frame(), _Checkpoint(path=path, batch_size=1)) is False


# =============================================================================
# Snapshots do modo paralelo
# =============================================================================


def test_snapshot_forcado_com_o_rotulo_da_ultima_gravacao_regrava(tmp_path: Path) -> None:
    path = tmp_path / "saida.csv"
    writer = _SnapshotWriter(_Checkpoint(path=path, batch_size=1))

    writer.save(pd.DataFrame({"n": [1]}), 3)
    writer.save(pd.DataFrame({"n": [2]}), 3)
    assert path.read_text() == "n\n1\n"

    writer.save(pd.DataFrame({"n": [3]}), 3, force=True)
    assert path.read_text() == "n\n3\n"
