"""Integração local com o runtime empacotado pelo SDK Codex."""

import json
import os
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from pydantic import BaseModel

from dataframeit import codex as codex_provider
from dataframeit.codex import _CODEX_CONFIG_OVERRIDES
from dataframeit.errors import ProviderError, ProviderTransientError
from dataframeit.llm import LLMConfig

openai_codex = pytest.importorskip("openai_codex")

# Flags que o runtime fixado deixa ligadas com `_CODEX_CONFIG_OVERRIDES` aplicado.
# Desligar qualquer uma delas não muda o request que o provider envia ao modelo,
# e por isso ficam como o runtime as traz. `unified_exec` continua ligada apesar
# do override, mas nenhuma ferramenta de execução de comandos chega ao request.
# Trocar o pin do SDK muda esta lista, e a flag nova só entra aqui
# depois de conferido se ela expõe ferramenta ou prende o turno; se fizer
# um dos dois, vai desligada em `_CODEX_CONFIG_OVERRIDES`.
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
        "unified_exec",
        "unified_exec_tty",
        "unified_exec_zsh_fork",
        "workspace_dependencies",
        "worktrees",
        "write_stdin_approval",
    }
)


# Ferramentas e namespaces que o request de uma linha oferece ao modelo, com
# `_CODEX_CONFIG_OVERRIDES` aplicado e o catálogo de `_write_model_catalog`.
# Trocar o pin do SDK pode mudar este conjunto, e o nome novo só entra aqui
# depois de conferido o que ele alcança; se alcançar arquivo, rede, processo ou
# outro agente, a config ou o campo do catálogo que o liga vai desligado no
# provider.
_REVIEWED_MODEL_TOOLS = frozenset()


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


def test_model_catalog_keeps_every_model_without_tool_fields(tmp_path):
    """O catálogo gravado traz todo modelo do runtime, sem os campos de ferramentas.

    Sem conta, `codex debug models` devolve o catálogo embutido; a comparação
    com `--bundled` mostra que nenhum modelo some no caminho.
    """
    from codex_cli_bin import bundled_codex_path  # noqa: PLC0415 (extra codex opcional)

    workspace = tmp_path / "workspace"
    home = tmp_path / "home"
    workspace.mkdir()
    home.mkdir()
    env = {
        "CODEX_HOME": os.fspath(home),
        "CODEX_SQLITE_HOME": os.fspath(home),
        "HOME": os.fspath(home),
    }

    path = codex_provider._write_model_catalog(workspace, env)
    bundled = subprocess.run(  # noqa: S603 (binário empacotado, argumentos fixos)
        [os.fspath(bundled_codex_path()), "debug", "models", "--bundled"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, **env},
        check=True,
        timeout=60,
    )
    bundled_models = json.loads(bundled.stdout)["models"]
    models = json.loads(path.read_text(encoding="utf-8"))["models"]

    assert path.parent == tmp_path
    assert [model["slug"] for model in models] == [model["slug"] for model in bundled_models]
    # Sem um modelo em code mode no catálogo embutido, o teste não prova a troca.
    assert any(model["tool_mode"] == "code_mode_only" for model in bundled_models)
    for model in models:
        for field, value in codex_provider._CATALOG_TOOL_FIELDS.items():
            assert model[field] == value


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

    def log_message(self, format, *args):  # noqa: A002 (nome do parâmetro da classe base)
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


def _use_local_model_provider(monkeypatch, tmp_path, base_url):
    """Aponta o provider para um endpoint de modelo local, sem conta nem rede.

    O endpoint entra como provider do runtime ao lado dos demais overrides, e a
    abertura do backend é a mesma do provider. O runtime manda o request pelo
    proxy do ambiente mesmo para `127.0.0.1`, e um proxy herdado prenderia o
    turno; por isso as variáveis de proxy saem do ambiente.
    """
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")

    source_home = tmp_path / "codex-source"
    source_home.mkdir()
    (source_home / "auth.json").write_text("{}")
    monkeypatch.setenv("CODEX_HOME", os.fspath(source_home))
    monkeypatch.setenv("DATAFRAMEIT_MOCK_KEY", "local")
    provider = (
        f'{{name="mock",base_url="{base_url}",wire_api="responses",'
        'env_key="DATAFRAMEIT_MOCK_KEY",request_max_retries=0,stream_max_retries=0}'
    )
    monkeypatch.setattr(
        codex_provider,
        "_CODEX_CONFIG_OVERRIDES",
        (*_CODEX_CONFIG_OVERRIDES, 'model_provider="mock"', f"model_providers.mock={provider}"),
    )
    return LLMConfig(
        model="gpt-6-luna",
        provider="codex",
        api_key=None,
        max_retries=1,
        base_delay=0,
        max_delay=0,
        rate_limit_delay=0,
        model_kwargs={"effort": "low"},
    )


def test_request_to_model_offers_only_reviewed_tools(tmp_path, monkeypatch):
    """O request de uma linha oferece ao modelo só as ferramentas revisadas.

    O catálogo do runtime liga o code mode e os sub-agentes para o
    `gpt-6-luna`; o catálogo de `_write_model_catalog` desliga o primeiro, e
    `agents.enabled=false`, os segundos.
    """
    handler = type("Handler", (_RecordingProvider,), {"bodies": []})
    server = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    config = _use_local_model_provider(monkeypatch, tmp_path, f"http://127.0.0.1:{port}/v1")

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
    assert _tool_names(request, set()) == _REVIEWED_MODEL_TOOLS
    # O nome da ferramenta de sub-agente não aparece em parte alguma do corpo.
    # `collaboration` não serve para essa busca, porque aparece no texto das
    # instruções.
    assert b"spawn_agent" not in raw


def test_connection_failure_fails_turn_as_transient(tmp_path, monkeypatch):
    """Sem conexão com o endpoint do modelo, a tentativa falha como transitória.

    Com `unbounded_connection_retries` ligada, o runtime repete a conexão sem
    limite e o turno não termina. Desligada, o turno falha em poucos segundos e
    a nova tentativa fica com o `retry_with_backoff`. O turno roda em outra
    thread, com prazo, para que a regressão falhe o teste em vez de prender a
    suíte; ao sair, o backend encerra o app-server e solta a thread.
    """
    deadline = 30
    outcome = {}

    def invoke_row(backend):
        try:
            backend.invoke("oi")
        except Exception as err:  # noqa: BLE001 (o erro é conferido fora da thread)
            outcome["error"] = err

    # Porta reservada sem `listen`: a conexão é recusada na hora.
    with socket.socket() as closed_port:
        closed_port.bind(("127.0.0.1", 0))
        port = closed_port.getsockname()[1]
        config = _use_local_model_provider(monkeypatch, tmp_path, f"http://127.0.0.1:{port}/v1")
        with codex_provider.open_codex_backend(config, _Answer, "Responda: {texto}") as backend:
            worker = threading.Thread(target=invoke_row, args=(backend,), daemon=True)
            worker.start()
            worker.join(deadline)
            assert not worker.is_alive(), f"o turno não terminou em {deadline}s"

    assert isinstance(outcome.get("error"), ProviderTransientError)
    assert str(outcome["error"]).startswith("Turno Codex falhou:")
