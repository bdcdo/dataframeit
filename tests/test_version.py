import importlib
import importlib.metadata
import re
from pathlib import Path

import dataframeit


def test_version_acompanha_o_pyproject():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    versao_declarada = re.search(
        r'^version = "([^"]+)"', pyproject.read_text(encoding="utf-8"), re.MULTILINE
    ).group(1)
    assert dataframeit.__version__ == versao_declarada


def test_version_sem_metadados_nao_quebra_o_import(monkeypatch):
    def sem_metadados(nome):
        raise importlib.metadata.PackageNotFoundError(nome)

    monkeypatch.setattr(importlib.metadata, "version", sem_metadados)
    try:
        assert importlib.reload(dataframeit).__version__ == "0+unknown"
    finally:
        monkeypatch.undo()
        importlib.reload(dataframeit)
