"""Integração local com o runtime empacotado pelo SDK Codex."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from dataframeit.codex import _CODEX_CONFIG_OVERRIDES

openai_codex = pytest.importorskip("openai_codex")

# Flags que o runtime fixado deixa ligadas com `_CODEX_CONFIG_OVERRIDES` aplicado.
# Desligar qualquer uma delas não muda o request que o provider envia ao modelo,
# e por isso ficam como o runtime as traz. `unified_exec` continua ligada apesar
# do override, mas nenhuma ferramenta de execução de comandos chega ao request.
# Trocar o pin do SDK muda esta lista, e a flag nova só entra aqui
# depois de conferido se ela expõe ferramenta; se expuser, vai desligada em
# `_CODEX_CONFIG_OVERRIDES`.
_REVIEWED_ENABLED_FEATURES = frozenset(
    {
        "auth_elicitation",
        "browser_use_external",
        "browser_use_full_cdp_access",
        "code_mode_host",
        "collaboration_modes",
        "compaction_image_budget",
        "content_item_kinds",
        "daemon_auto_start",
        "enable_request_compression",
        "fast_mode",
        "guardian_approval",
        "guardian_reuse_parent_compaction",
        "in_app_browser",
        "in_app_chat",
        "in_app_dictation",
        "in_app_local_automation",
        "in_app_updates",
        "item_ids",
        "mentions_v2",
        "plugin_sharing",
        "realtime_conversation",
        "resize_all_images",
        "skill_mcp_dependency_install",
        "skill_search",
        "sqlite",
        "steer",
        "system_proxy_fallback",
        "terminal_resize_reflow",
        "tool_call_mcp_elicitation",
        "tool_search_always_defer_mcp_tools",
        "tool_suggest",
        "tui_app_server",
        "unbounded_connection_retries",
        "unified_exec",
        "unified_exec_tty",
        "unified_exec_zsh_fork",
        "workspace_dependencies",
        "worktrees",
        "write_stdin_approval",
    }
)


def _isolated_config(tmp_path):
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
    return config, workspace


def _feature_states(tmp_path):
    """Estado efetivo de cada flag no runtime fixado, com os overrides do provider."""
    from codex_cli_bin import bundled_codex_path  # noqa: PLC0415 (extra codex opcional)

    home = tmp_path / "home"
    home.mkdir()
    command = [os.fspath(bundled_codex_path())]
    for override in _CODEX_CONFIG_OVERRIDES:
        command += ["-c", override]
    command += ["features", "list"]
    env = {
        **os.environ,
        "CODEX_HOME": os.fspath(home),
        "HOME": os.fspath(home),
        "USERPROFILE": os.fspath(home),
    }
    listing = subprocess.run(  # noqa: S603 (binário empacotado, argumentos fixos)
        command, capture_output=True, text=True, env=env, check=True, timeout=60
    )
    states = {}
    for line in listing.stdout.splitlines():
        fields = line.split()
        if fields:
            states[fields[0]] = fields[-1] == "true"
    return states


def test_bundled_runtime_efforts_are_declared_by_sdk(tmp_path):
    """Todo effort que o catálogo anuncia é membro declarado do enum do SDK.

    A validação de `effort` aceita só os membros declarados, porque o enum do SDK
    cria membro para qualquer texto. Um effort novo no catálogo, fora da lista
    declarada, seria recusado antes da primeira linha.
    """
    from openai_codex.types import ReasoningEffort  # noqa: PLC0415 (extra codex opcional)

    config, workspace = _isolated_config(tmp_path)

    assert config.codex_bin is None
    with openai_codex.Codex(config) as client:
        catalog = client.models(include_hidden=True)

    declared = {member.value for member in ReasoningEffort}
    announced = {
        option.reasoning_effort.value
        for item in catalog.data
        for option in item.supported_reasoning_efforts
    }
    assert catalog.data
    assert announced
    assert announced <= declared
    assert Path(config.cwd) == workspace


def test_disabled_features_exist_in_bundled_runtime(tmp_path):
    """Cada flag que o provider desliga existe no runtime.

    O runtime ignora em silêncio um `features.<nome>` que não conhece, e a flag
    renomeada numa troca de pin voltaria ligada sem erro.
    """
    states = _feature_states(tmp_path)
    disabled = [
        override.removeprefix("features.").removesuffix("=false")
        for override in _CODEX_CONFIG_OVERRIDES
        if override.startswith("features.")
    ]

    assert disabled
    assert set(disabled) <= set(states)


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="a lista revisada vem do runtime Linux; o padrão das flags no Windows não foi revisado",
)
def test_enabled_features_match_reviewed_list(tmp_path):
    enabled = {name for name, on in _feature_states(tmp_path).items() if on}

    assert enabled == _REVIEWED_ENABLED_FEATURES
