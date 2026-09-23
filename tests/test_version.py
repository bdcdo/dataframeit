import re
from pathlib import Path

import dataframeit


def test_version_acompanha_o_pyproject():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    versao_declarada = re.search(
        r'^version = "([^"]+)"', pyproject.read_text(encoding="utf-8"), re.MULTILINE
    ).group(1)
    assert dataframeit.__version__ == versao_declarada
