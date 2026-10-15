"""Integração local com o runtime empacotado pelo SDK Codex."""

import contextlib
import dataclasses
import gc
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import warnings
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from pathlib import Path

import pandas as pd
import pytest
from pydantic import BaseModel

from dataframeit import codex as codex_provider
from dataframeit import dataframeit
from dataframeit.codex import _CODEX_CONFIG_OVERRIDES
from dataframeit.errors import ProviderAbortError, ProviderError, ProviderTransientError
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
        "api_key_model_discovery",
        "auth_elicitation",
        "browser_annotation_api",
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
        "in_app_voice",
        # Interrompe a amostragem quando chega entrada do usuário ao turno em
        # curso, como a de `turn/steer`. O provider manda um turno por thread e
        # não manda entrada depois dele, e o `turn/interrupt` do prazo cancela o
        # turno por outro caminho.
        "instant_interrupt",
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
        # Só libera `service_tier="ultrafast"` quando a config o pede, como
        # `fast_mode` faz com `fast`; o provider não configura `service_tier`.
        "ultrafast_mode",
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
    client = openai_codex.Codex(config)
    try:
        catalog = client.models(include_hidden=True)
    finally:
        codex_provider._close_client(client)

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


def _tool_lists(node, lists):
    """Toda lista `tools` do request, em qualquer profundidade."""
    if isinstance(node, dict):
        if isinstance(node.get("tools"), list):
            lists.append(node["tools"])
        for value in node.values():
            _tool_lists(value, lists)
    elif isinstance(node, list):
        for value in node:
            _tool_lists(value, lists)
    return lists


def _offered_tools(lists):
    """Cada entrada das listas `tools`, pelo nome ou, sem nome, pelo tipo.

    Contar pelo tipo faz uma ferramenta de tipo novo, como `tool_search` ou
    `web_search`, também entrar na comparação. Os namespaces entram ao lado das
    ferramentas que agrupam, porque as listas aninhadas também são lidas.
    """
    return {entry.get("name") or entry.get("type") for tools in lists for entry in tools}


class _Answer(BaseModel):
    resposta: str


def _use_local_model_provider(
    monkeypatch, tmp_path, base_url, model="gpt-6-luna", overrides=_CODEX_CONFIG_OVERRIDES
):
    """Aponta o provider para um endpoint de modelo local, sem conta nem rede.

    O endpoint entra como provider do runtime ao lado dos demais overrides, e a
    abertura do backend é a mesma do provider. O runtime manda o request pelo
    proxy do ambiente mesmo para `127.0.0.1`, e um proxy herdado prenderia o
    turno; por isso as variáveis de proxy saem do ambiente. `overrides` troca os
    overrides do provider, para o teste que precisa de uma ferramenta oferecida.
    """
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")

    # Acento e caractere fora do BMP no caminho exercitam a citação do
    # `model_catalog_json` que o provider passa ao app-server.
    source_home = tmp_path / "códex 😀"
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
        (*overrides, 'model_provider="mock"', f"model_providers.mock={provider}"),
    )
    return LLMConfig(
        model=model,
        provider="codex",
        api_key=None,
        max_retries=1,
        base_delay=0,
        max_delay=0,
        rate_limit_delay=0,
        model_kwargs={"effort": "low"},
    )


def _bundled_models(tmp_path):
    from codex_cli_bin import bundled_codex_path  # noqa: PLC0415 (extra codex opcional)

    home = tmp_path / "catalog-home"
    home.mkdir()
    listing = subprocess.run(  # noqa: S603 (binário empacotado, argumentos fixos)
        [os.fspath(bundled_codex_path()), "debug", "models", "--bundled"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "CODEX_HOME": os.fspath(home), "HOME": os.fspath(home)},
        check=True,
        timeout=60,
    )
    return json.loads(listing.stdout)["models"]


def _request_to_model(tmp_path, monkeypatch, slug, overrides=_CODEX_CONFIG_OVERRIDES):
    """Corpo, em bytes, do primeiro request que uma linha manda ao modelo `slug`."""
    run_path = tmp_path / slug
    run_path.mkdir()
    handler = type("Handler", (_RecordingProvider,), {"bodies": []})
    server = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    config = _use_local_model_provider(
        monkeypatch, run_path, f"http://127.0.0.1:{port}/v1", model=slug, overrides=overrides
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
        monkeypatch.undo()

    assert handler.bodies, slug
    encoding, raw = handler.bodies[0]
    assert encoding is None
    assert json.loads(raw)["model"] == slug
    return raw


def test_request_to_model_offers_only_reviewed_tools(tmp_path, monkeypatch):
    """O request de uma linha oferece ao modelo só as ferramentas revisadas.

    Vale para todo modelo do catálogo embutido, porque cada um traz os próprios
    campos de ferramentas. O catálogo liga o code mode e os sub-agentes para
    modelos como o `gpt-6-luna`; o catálogo de `_write_model_catalog` desliga o
    primeiro, e `agents.enabled=false`, os segundos.

    Sem ferramenta, o request dos modelos com `use_responses_lite` não traz
    lista alguma, porque o runtime só põe o item `additional_tools` no `input`
    quando há o que oferecer. Que a busca de `_tool_lists` alcança o lugar onde
    o runtime põe as ferramentas, quando há, é o que prova
    `test_tool_lists_find_tool_offered_by_runtime`.
    """
    slugs = [model["slug"] for model in _bundled_models(tmp_path)]
    assert slugs
    for slug in slugs:
        raw = _request_to_model(tmp_path, monkeypatch, slug)
        assert _offered_tools(_tool_lists(json.loads(raw), [])) == _REVIEWED_MODEL_TOOLS, slug
        # O nome da ferramenta de sub-agente não aparece em parte alguma do corpo.
        # `collaboration` não serve para essa busca, porque aparece no texto das
        # instruções.
        assert b"spawn_agent" not in raw, slug


def test_tool_lists_find_tool_offered_by_runtime(tmp_path, monkeypatch):
    """Com uma ferramenta ligada, `_tool_lists` a acha no request.

    Sem este controle, a igualdade com o conjunto revisado, vazio, passaria
    também se o runtime mudasse o lugar onde põe as ferramentas. O controle
    tira o override de `view_image`, que oferece uma ferramenta, e roda um
    modelo de cada forma de request: com `use_responses_lite`, as ferramentas
    vão num item `additional_tools` do `input`; sem, na chave `tools` do topo.
    """
    override = "features.view_image=false"
    assert override in _CODEX_CONFIG_OVERRIDES
    overrides = tuple(item for item in _CODEX_CONFIG_OVERRIDES if item != override)
    one_per_shape = {}
    for model in _bundled_models(tmp_path):
        one_per_shape.setdefault(bool(model.get("use_responses_lite")), model["slug"])
    assert one_per_shape
    for slug in one_per_shape.values():
        raw = _request_to_model(tmp_path, monkeypatch, slug, overrides=overrides)
        assert "view_image" in _offered_tools(_tool_lists(json.loads(raw), [])), slug


def _pipes_fechados(processo):
    return processo.stdout.closed and processo.stderr.closed


def _espera_pipes_fechados(processo, prazo=10):
    """Espera o fechamento feito pelo relógio, que roda em outra thread.

    A linha acorda quando o leitor do stdout vê o fim do processo, e o relógio
    ainda pode estar esperando as threads de leitura antes de fechar os pipes.
    """
    fim = time.monotonic() + prazo
    while not _pipes_fechados(processo) and time.monotonic() < fim:
        time.sleep(0.05)
    return _pipes_fechados(processo)


@pytest.mark.parametrize(
    "saida",
    [None, ProviderError("falha da linha"), KeyboardInterrupt()],
    ids=["normal", "erro", "interrupcao"],
)
def test_encerramento_do_backend_fecha_os_pipes_do_app_server(tmp_path, monkeypatch, saida):
    """Ao sair do backend, por qualquer caminho, stdout e stderr do app-server fecham.

    O `close` do SDK fecha só o stdin, e os outros dois pipes ficariam para o
    coletor de lixo, cada um com um `ResourceWarning`.
    """
    esperado = pytest.raises(type(saida)) if saida is not None else contextlib.nullcontext()
    with socket.socket() as closed_port:
        closed_port.bind(("127.0.0.1", 0))
        port = closed_port.getsockname()[1]
        config = _use_local_model_provider(monkeypatch, tmp_path, f"http://127.0.0.1:{port}/v1")
        with (
            esperado,
            codex_provider.open_codex_backend(config, _Answer, "Responda: {texto}") as backend,
        ):
            processo = backend._client._client._proc
            assert processo is not None
            assert not _pipes_fechados(processo)
            if saida is not None:
                raise saida

    assert _pipes_fechados(processo)


@contextlib.contextmanager
def _avisos_de_recurso():
    """Registra os `ResourceWarning` do bloco e da coleta de lixo ao fim dele.

    O teste solta as próprias referências ao processo antes do fim do bloco,
    para que o `Popen` ainda sem `returncode` e o pipe aberto, se houver, avisem
    na coleta.
    """
    avisos: list[str] = []
    with warnings.catch_warnings(record=True) as registrados:
        warnings.simplefilter("always", ResourceWarning)
        yield avisos
        gc.collect()
    avisos.extend(str(aviso.message) for aviso in registrados if aviso.category is ResourceWarning)


def _guarda_os_processos(monkeypatch):
    """Guarda cada processo que o `start` do SDK abre, mesmo quando ele falha depois."""
    processos: list[subprocess.Popen[str]] = []
    start = openai_codex.client.CodexClient.start

    def start_que_guarda(self):
        try:
            start(self)
        finally:
            if (processo := self._proc) is not None:
                processos.append(processo)

    monkeypatch.setattr(openai_codex.client.CodexClient, "start", start_que_guarda)
    return processos


def _segundo_ctrl_c_no_wait(processo):
    """O primeiro `wait` do processo levanta `KeyboardInterrupt`, como um segundo Ctrl+C.

    O sinal de verdade, mandado por um `threading.Timer`, chegaria fora do bloco
    se o temporizador atrasasse, e o `KeyboardInterrupt` derrubaria a sessão do
    pytest. Os `wait` seguintes são os do processo.
    """
    espera = processo.wait
    chamadas = []

    def interrompe_a_primeira(timeout=None):
        chamadas.append(timeout)
        if len(chamadas) == 1:
            raise KeyboardInterrupt
        return espera(timeout)

    processo.wait = interrompe_a_primeira


@pytest.mark.skipif(sys.platform == "win32", reason="SIGSTOP só existe em POSIX")
@pytest.mark.parametrize("segundo_ctrl_c", [False, True], ids=["sem_sair_em_2s", "segundo_ctrl_c"])
def test_app_server_que_nao_sai_no_fechamento_e_recolhido(tmp_path, monkeypatch, segundo_ctrl_c):
    """O app-server que não sai depois do SIGTERM é morto, esperado e tem os pipes fechados.

    Com `SIGSTOP`, o processo não sai em 2 s, e o `close` do SDK faz `kill`
    sem `wait`: o processo ficaria zumbi, sem `returncode`, e o `Popen` avisaria
    na coleta de lixo. Com um segundo Ctrl+C durante o `wait` do SDK, o `close`
    sai sem `kill`, e o app-server ficaria vivo, com os pipes abertos, depois
    que o bloco já levantou `KeyboardInterrupt`.
    """
    processos = _guarda_os_processos(monkeypatch)
    esperado = pytest.raises(KeyboardInterrupt) if segundo_ctrl_c else contextlib.nullcontext()
    with socket.socket() as closed_port, _avisos_de_recurso() as avisos:
        closed_port.bind(("127.0.0.1", 0))
        port = closed_port.getsockname()[1]
        config = _use_local_model_provider(monkeypatch, tmp_path, f"http://127.0.0.1:{port}/v1")
        with (
            esperado,
            codex_provider.open_codex_backend(config, _Answer, "Responda: {texto}"),
        ):
            os.kill(processos[0].pid, signal.SIGSTOP)
            if segundo_ctrl_c:
                _segundo_ctrl_c_no_wait(processos[0])
                raise KeyboardInterrupt

        (processo,) = processos
        try:
            assert processo.returncode is not None
            assert _pipes_fechados(processo)
        finally:
            # Sem a correção, o app-server segue parado, e o teste o recolhe.
            processo.kill()
            processo.wait(timeout=10)
        del processo
        processos.clear()

    assert avisos == []


@pytest.mark.parametrize(
    ("modulo", "nome", "erro"),
    [
        ("client", "CodexClient._start_reader_thread", RuntimeError),
        ("api", "validate_initialize_metadata", RuntimeError),
        ("api", "validate_initialize_metadata", KeyboardInterrupt),
    ],
    ids=["start", "initialize", "ctrl_c_no_initialize"],
)
def test_falha_na_abertura_recolhe_o_app_server_e_fecha_os_pipes(
    tmp_path, monkeypatch, modulo, nome, erro
):
    """A abertura que falha depois de abrir o app-server o recolhe e fecha os pipes.

    O construtor do `Codex` chama o `close` do SDK quando o `start` ou o
    `initialize` falham, e o `close` fecha só o stdin; o objeto não chega ao
    backend, e stdout e stderr ficariam para o coletor de lixo. A falha do
    `start` é a da thread de leitura, depois do `Popen`; a do `initialize`, a
    da validação dos metadados que o runtime devolve. Um `KeyboardInterrupt`
    durante a abertura passa sem `close`, e o app-server ficaria vivo.
    """
    processos = _guarda_os_processos(monkeypatch)

    def falha(*args):
        mensagem = f"falha simulada em {nome}"
        raise erro(mensagem)

    monkeypatch.setattr(f"openai_codex.{modulo}.{nome}", falha)
    with socket.socket() as closed_port, _avisos_de_recurso() as avisos:
        closed_port.bind(("127.0.0.1", 0))
        port = closed_port.getsockname()[1]
        config = _use_local_model_provider(monkeypatch, tmp_path, f"http://127.0.0.1:{port}/v1")
        with (
            pytest.raises(erro, match="falha simulada"),
            codex_provider.open_codex_backend(config, _Answer, "Responda: {texto}"),
        ):
            pass

        (processo,) = processos
        try:
            assert processo.returncode is not None
            assert _pipes_fechados(processo)
        finally:
            # Sem a correção, o app-server pode seguir vivo, e o teste o recolhe.
            processo.kill()
            processo.wait(timeout=10)
        del processo
        processos.clear()

    assert avisos == []


def test_falha_ao_criar_o_cliente_de_protocolo_sobe_sem_processo(tmp_path, monkeypatch):
    """O cliente de protocolo que falha ao ser criado não deixa o que fechar.

    A falha vem antes do `start`, e o `Codex` ainda não tem `_client`: a
    abertura levanta a mesma falha, sem processo aberto.
    """
    processos = _guarda_os_processos(monkeypatch)

    def cliente_que_falha(**kwargs):
        mensagem = "cliente de protocolo recusado"
        raise RuntimeError(mensagem)

    monkeypatch.setattr(openai_codex.api, "CodexClient", cliente_que_falha)
    with socket.socket() as closed_port:
        closed_port.bind(("127.0.0.1", 0))
        port = closed_port.getsockname()[1]
        config = _use_local_model_provider(monkeypatch, tmp_path, f"http://127.0.0.1:{port}/v1")
        with (
            pytest.raises(RuntimeError, match="cliente de protocolo recusado"),
            codex_provider.open_codex_backend(config, _Answer, "Responda: {texto}"),
        ):
            pass

    assert processos == []


def _evento_sse(evento):
    return f"event: {evento['type']}\ndata: {json.dumps(evento)}\n\n".encode()


class _ProviderQueMorreNoSegundoTurno(BaseHTTPRequestHandler):
    """Provider de modelo local: responde ao primeiro turno e mata o app-server no segundo.

    O pedido do segundo turno chega enquanto a linha espera o stream, e a
    conexão fica aberta até o teste liberá-la, sem resposta. A thread do turno
    derrubado, que o runtime manda no cabeçalho `thread-id`, é anotada antes do
    `kill`.
    """

    processos: list[subprocess.Popen]
    pedidos: list[int]
    threads_derrubadas: list[str | None]
    liberado: threading.Event

    def do_POST(self):
        self.rfile.read(int(self.headers.get("content-length", 0)))
        self.pedidos.append(1)
        if len(self.pedidos) > 1:
            self.threads_derrubadas.append(self.headers.get("thread-id"))
            self.processos[0].kill()
            self.liberado.wait(30)
            return
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        mensagem = {
            "type": "message",
            "role": "assistant",
            "id": "msg_1",
            "content": [{"type": "output_text", "text": json.dumps({"resposta": "ok"})}],
        }
        uso = {
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        }
        for evento in (
            {"type": "response.created", "response": {"id": "resp_1"}},
            {"type": "response.output_item.done", "item": mensagem},
            {"type": "response.completed", "response": {"id": "resp_1", "usage": uso}},
        ):
            self.wfile.write(_evento_sse(evento))
            self.wfile.flush()

    def log_message(self, format, *args):  # noqa: A002 (nome do parâmetro da classe base)
        pass


def test_app_server_que_morre_no_turno_devolve_as_linhas_processadas(tmp_path, monkeypatch):
    """O app-server que morre no meio de um turno não derruba `dataframeit()`.

    A linha em curso aborta, e o `thread/unsubscribe` dela fica no buffer do
    stdin do processo morto. O `close` do SDK descarrega o buffer ao fechar o
    stdin e levanta `BrokenPipeError`, ou `OSError(EINVAL)` no Windows, antes
    do `terminate`. Os pipes de saída fecham mesmo assim, e a falha do
    fechamento não substitui o resultado: a primeira linha sai processada, e a
    abortada e a seguinte, sem status.

    O pedido de descarga sai numa thread daemon, que pode chegar ao stdin
    depois do fechamento. O teste espera essa escrita e o fim das threads de
    leitura antes do fechamento, para que o caso não dependa da corrida. A
    descarga esperada é reconhecida pela thread do turno que o provider
    derrubou, que ele anota antes do `kill`. A hora em que cada descarga começa
    não a distingue: a da linha abortada pode começar antes de o handler do
    provider voltar a rodar depois do `kill`, e a da primeira linha pode
    atrasar até depois dele.
    """
    handler = type(
        "Handler",
        (_ProviderQueMorreNoSegundoTurno,),
        {
            "processos": [],
            "pedidos": [],
            "threads_derrubadas": [],
            "liberado": threading.Event(),
        },
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    config = _use_local_model_provider(
        monkeypatch, tmp_path, f"http://127.0.0.1:{server.server_address[1]}/v1"
    )

    abre_backend = codex_provider.open_codex_backend

    @contextlib.contextmanager
    def guarda_o_processo(*args, **kwargs):
        with abre_backend(*args, **kwargs) as backend:
            handler.processos.append(backend._client._client._proc)
            yield backend

    descarga_feita = threading.Event()
    descarrega = codex_provider._unsubscribe_quietly

    def descarrega_e_avisa(client, thread_id):
        descarrega(client, thread_id)
        if thread_id in handler.threads_derrubadas:
            descarga_feita.set()

    fecha = codex_provider._close_client

    def fecha_depois_da_descarga(client):
        assert descarga_feita.wait(30), "a linha abortada não descarregou a thread"
        client._client._reader_thread.join(30)
        client._client._stderr_thread.join(30)
        fecha(client)

    monkeypatch.setattr(codex_provider, "open_codex_backend", guarda_o_processo)
    monkeypatch.setattr(codex_provider, "_unsubscribe_quietly", descarrega_e_avisa)
    monkeypatch.setattr(codex_provider, "_close_client", fecha_depois_da_descarga)

    try:
        with (
            pytest.warns(UserWarning, match="não-recuperável"),
            pytest.warns(UserWarning, match="Execução interrompida.*app-server do Codex encerrou"),
        ):
            resultado = dataframeit(
                pd.DataFrame({"texto": ["a", "b", "c"]}),
                _Answer,
                "Responda: {texto}",
                provider="codex",
                model=config.model,
                model_kwargs={"effort": "low"},
                max_retries=1,
            )
    finally:
        handler.liberado.set()
        server.shutdown()
        server.server_close()

    (processo,) = handler.processos
    assert _pipes_fechados(processo)
    assert resultado["_dataframeit_status"].tolist() == ["processed", None, None]
    assert resultado["resposta"].tolist() == ["ok", None, None]
    assert len(handler.pedidos) == 2
    assert processo.wait(timeout=10) is not None


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


@pytest.mark.skipif(sys.platform == "win32", reason="SIGSTOP só existe em POSIX")
def test_app_server_that_stops_answering_is_closed_and_aborts(tmp_path, monkeypatch):
    """O app-server parado sem encerrar é derrubado pelo prazo da tentativa.

    Com `SIGSTOP`, o processo continua vivo e o stdout aberto, e o pedido
    `thread/start` espera na fila do SDK, que não tem timeout. O relógio da
    tentativa encerra o app-server, o leitor do stdout acorda o pedido, e a
    linha aborta em vez de repetir no mesmo app-server. A linha roda em outra
    thread, com prazo, para que a regressão falhe o teste em vez de prender a
    suíte.
    """
    deadline = 30
    outcome = {}

    def invoke_row(backend):
        try:
            backend.invoke("oi")
        except Exception as err:  # noqa: BLE001 (o erro é conferido fora da thread)
            outcome["error"] = err

    with socket.socket() as closed_port:
        closed_port.bind(("127.0.0.1", 0))
        port = closed_port.getsockname()[1]
        config = _use_local_model_provider(monkeypatch, tmp_path, f"http://127.0.0.1:{port}/v1")
        config = dataclasses.replace(config, model_kwargs={"effort": "low", "timeout": 1})
        with codex_provider.open_codex_backend(config, _Answer, "Responda: {texto}") as backend:
            process = backend._client._client._proc
            assert process is not None
            os.kill(process.pid, signal.SIGSTOP)
            try:
                start = time.monotonic()
                worker = threading.Thread(target=invoke_row, args=(backend,), daemon=True)
                worker.start()
                worker.join(deadline)
                assert not worker.is_alive(), f"a linha não terminou em {deadline}s"
                elapsed = time.monotonic() - start
                assert _espera_pipes_fechados(process)
            finally:
                # O relógio pode estar colhendo o processo em outra thread.
                with contextlib.suppress(ProcessLookupError):
                    os.kill(process.pid, signal.SIGCONT)

    assert isinstance(outcome.get("error"), ProviderAbortError)
    assert "não respondeu a thread/start no prazo de 1 s" in str(outcome["error"])
    assert process.wait(timeout=10) is not None
    assert elapsed < deadline


@pytest.mark.skipif(sys.platform == "win32", reason="SIGSTOP só existe em POSIX")
def test_turn_start_blocked_writing_to_stopped_app_server_aborts(tmp_path, monkeypatch):
    """A linha presa escrevendo o `turn/start` num app-server parado é solta.

    O app-server para logo depois de responder ao `thread/start`, e o prompt do
    turno é maior que o buffer do pipe, então a escrita bloqueia segurando a
    trava do buffer. O `close` do SDK fecha o stdin antes de sinalizar o
    processo e esperaria essa trava para sempre; o relógio mata o processo
    antes, o que quebra o pipe e solta a escrita.
    """
    deadline = 30
    outcome = {}

    def invoke_row(backend):
        try:
            backend.invoke("x" * 300_000)
        except Exception as err:  # noqa: BLE001 (o erro é conferido fora da thread)
            outcome["error"] = err

    with socket.socket() as closed_port:
        closed_port.bind(("127.0.0.1", 0))
        port = closed_port.getsockname()[1]
        config = _use_local_model_provider(monkeypatch, tmp_path, f"http://127.0.0.1:{port}/v1")
        config = dataclasses.replace(config, model_kwargs={"effort": "low", "timeout": 1})
        with codex_provider.open_codex_backend(config, _Answer, "Responda: {texto}") as backend:
            process = backend._client._client._proc
            assert process is not None
            thread_start = backend._client.thread_start

            def stop_after_thread_start(**kwargs):
                thread = thread_start(**kwargs)
                os.kill(process.pid, signal.SIGSTOP)
                return thread

            monkeypatch.setattr(backend._client, "thread_start", stop_after_thread_start)
            try:
                worker = threading.Thread(target=invoke_row, args=(backend,), daemon=True)
                worker.start()
                worker.join(deadline)
                assert not worker.is_alive(), f"a linha não terminou em {deadline}s"
                assert _espera_pipes_fechados(process)
            finally:
                # O relógio pode estar colhendo o processo em outra thread.
                with contextlib.suppress(ProcessLookupError):
                    os.kill(process.pid, signal.SIGCONT)

    assert isinstance(outcome.get("error"), ProviderAbortError)
    assert "não respondeu a turn/start no prazo de 1 s" in str(outcome["error"])
    assert process.wait(timeout=10) is not None
