"""Integração local com o runtime empacotado pelo SDK Codex."""

from pathlib import Path

import pytest

from dataframeit.codex import _CODEX_CONFIG_OVERRIDES

openai_codex = pytest.importorskip("openai_codex")


def test_bundled_runtime_reports_gpt_5_4_without_authentication(tmp_path):
    workspace = tmp_path / "workspace"
    codex_home = tmp_path / "codex-home"
    workspace.mkdir()
    codex_home.mkdir()
    config = openai_codex.CodexConfig(
        cwd=str(workspace),
        config_overrides=_CODEX_CONFIG_OVERRIDES,
        env={
            "CODEX_HOME": str(codex_home),
            "CODEX_SQLITE_HOME": str(codex_home),
        },
    )

    assert config.codex_bin is None
    with openai_codex.Codex(config) as client:
        catalog = client.models(include_hidden=True)

    models = {item.model for item in catalog.data}
    assert "gpt-5.4" in models
    assert Path(config.cwd) == workspace
