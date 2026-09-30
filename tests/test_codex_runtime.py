"""Integração local com o runtime empacotado pelo SDK Codex."""

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from pydantic import BaseModel

from dataframeit import codex as codex_provider
from dataframeit.codex import _CODEX_CONFIG_OVERRIDES
from dataframeit.errors import ProviderError
from dataframeit.llm import LLMConfig

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


def test_enabled_features_match_reviewed_list(tmp_path):
    enabled = {name for name, on in _feature_states(tmp_path).items() if on}
    expected = set(_REVIEWED_ENABLED_FEATURES)
    # `secret_auth_storage` é a única flag cujo padrão o runtime fixa pela
    # plataforma: vem ligada só no Windows.
    if sys.platform == "win32":
        expected.add("secret_auth_storage")

    assert enabled == expected


class _RecordingProvider(BaseHTTPRequestHandler):
    """Provider de modelo local: guarda cada corpo recebido e recusa o turno."""

    bodies: list[tuple[str | None, bytes]]

    def do_POST(self):
        length = int(self.headers.get("content-length", 0))
        self.bodies.append((self.headers.get("content-encoding"), self.rfile.read(length)))
        self.send_response(400)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"error":{"message":"mock","type":"invalid_request_error"}}')

    def log_message(self, *args):
        pass


def _tool_names(node, names):
    """Nomes de toda ferramenta e namespace no request, em qualquer profundidade."""
    if isinstance(node, dict):
        if node.get("type") in {"namespace", "function", "custom"} and "name" in node:
            names.add(node["name"])
        for value in node.values():
            _tool_names(value, names)
    elif isinstance(node, list):
        for value in node:
            _tool_names(value, names)
    return names


class _Answer(BaseModel):
    resposta: str


def test_request_to_model_has_no_subagent_tools(tmp_path, monkeypatch):
    """O request de uma linha não oferece ao modelo as ferramentas de sub-agentes.

    O catálogo do runtime liga os sub-agentes para o `gpt-6-luna`, e só
    `agents.enabled=false` os desliga. O provider local troca o endpoint do
    modelo e mantém os demais overrides e a abertura do provider, e por isso
    o teste não precisa de conta nem de rede.
    """
    handler = type("Handler", (_RecordingProvider,), {"bodies": []})
    server = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]

    source_home = tmp_path / "codex-source"
    source_home.mkdir()
    (source_home / "auth.json").write_text("{}")
    monkeypatch.setenv("CODEX_HOME", os.fspath(source_home))
    monkeypatch.setenv("DATAFRAMEIT_MOCK_KEY", "local")
    provider = (
        f'{{name="mock",base_url="http://127.0.0.1:{port}/v1",wire_api="responses",'
        'env_key="DATAFRAMEIT_MOCK_KEY",request_max_retries=0,stream_max_retries=0}'
    )
    monkeypatch.setattr(
        codex_provider,
        "_CODEX_CONFIG_OVERRIDES",
        (*_CODEX_CONFIG_OVERRIDES, 'model_provider="mock"', f"model_providers.mock={provider}"),
    )
    config = LLMConfig(
        model="gpt-6-luna",
        provider="codex",
        api_key=None,
        max_retries=1,
        base_delay=0,
        max_delay=0,
        rate_limit_delay=0,
        model_kwargs={"effort": "low"},
    )

    try:
        with (
            codex_provider.open_codex_backend(config, _Answer, "Responda: {texto}") as backend,
            pytest.warns(UserWarning, match="não-recuperável"),
            pytest.raises(ProviderError, match="mock"),
        ):
            backend.invoke("oi")
    finally:
        server.shutdown()
        server.server_close()

    assert handler.bodies
    encoding, raw = handler.bodies[0]
    assert encoding is None
    request = json.loads(raw)
    assert request["model"] == "gpt-6-luna"
    names = _tool_names(request, set())
    assert names
    assert "collaboration" not in names
    assert "spawn_agent" not in names
