"""O que cada linha grava, a contagem do checkpoint e o resumo, nos dois modos."""

import linecache
import logging
import re
import threading
import time
import warnings
from contextlib import contextmanager, nullcontext
from functools import partial
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pydantic import BaseModel
from typing_extensions import override

from dataframeit import ProviderAbortError, ProviderOverloadedError, core
from dataframeit.core import dataframeit
from dataframeit.errors import get_friendly_error_message


class Modelo(BaseModel):
    campo1: str
    campo2: str


class _ErroIlegivel(Exception):
    @override
    def __str__(self):
        msg = "sem texto"
        raise RuntimeError(msg)


def _llm(erros=None):
    erros = erros or {}

    def llm(text, *args, **kwargs):
        if text in erros:
            raise erros[text]
        return {
            "data": {"campo1": text, "campo2": text},
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }

    return llm


def _roda(textos, llm, **kwargs):
    df = textos if isinstance(textos, pd.DataFrame) else pd.DataFrame({"texto": textos})
    with (
        patch("dataframeit.core.call_langchain", side_effect=llm),
        patch("dataframeit.core.validate_provider_dependencies"),
    ):
        return dataframeit(df, questions=Modelo, prompt="{texto}", **kwargs)


class _ErroComCusto(Exception):
    cost_usd: float = 0


def _config(provider="openai"):
    return SimpleNamespace(max_retries=3, provider=provider)


def _gravacoes(monkeypatch, falhar=False):
    """Registra o rótulo de cada snapshot; com falhar, toda gravação levanta."""
    rotulos = []
    salvar = core._SnapshotWriter.save

    def save(self, frame, label):
        rotulos.append(label)
        if falhar:
            msg = "gravação do snapshot"
            raise RuntimeError(msg)
        salvar(self, frame, label)

    monkeypatch.setattr(core._SnapshotWriter, "save", save)
    return rotulos


# _record_error e o registro de reserva


@pytest.mark.parametrize(("custo", "esperado"), [(None, 0), (0, 0), (0.25, 0.25)])
def test_custo_do_erro_entra_no_resumo_so_quando_informado(custo, esperado):
    erro = _ErroComCusto("falhou") if custo is not None else ValueError("falhou")
    if isinstance(erro, _ErroComCusto):
        erro.cost_usd = custo
    estatisticas = core._empty_token_stats()
    df = pd.DataFrame({"s": [None], "_error_details": [None]}, dtype=object)

    core._record_error(
        df,
        0,
        erro,
        status_col="s",
        config=_config(),
        row_already_processed=False,
        reprocess_columns=None,
        token_stats=estatisticas,
    )

    assert estatisticas["cost_usd"] == esperado


class _ErroIlegivelComCusto(_ErroIlegivel, _ErroComCusto):
    pass


@pytest.mark.parametrize(
    "classe", [_ErroComCusto, _ErroIlegivelComCusto], ids=["normal", "reserva"]
)
def test_uso_das_chamadas_feitas_pela_linha_em_erro_entra_no_resumo(classe):
    erro = classe()
    erro.cost_usd = 0.25
    erro.row_usage = {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "search_credits": 2,
        "search_count": 1,
        "cost_usd": 0.5,
    }
    estatisticas = core._empty_token_stats()
    df = pd.DataFrame({"s": [None], "_error_details": [None]}, dtype=object)

    core._record_error_safely(
        df,
        0,
        erro,
        status_col="s",
        config=_config(),
        row_already_processed=False,
        reprocess_columns=None,
        token_stats=estatisticas,
    )

    assert estatisticas == {
        **core._empty_token_stats(),
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "search_credits": 2,
        "search_count": 1,
        "cost_usd": 0.75,
    }


@pytest.mark.parametrize("registro", ["normal", "reserva"])
@pytest.mark.parametrize(
    ("ja_processada", "reprocessar"), [(True, None), (False, ["campo1"]), (False, None)]
)
def test_nota_de_valores_mantidos_so_na_linha_processada_sob_reprocessamento(
    registro, ja_processada, reprocessar
):
    erro = ValueError("falhou") if registro == "normal" else _ErroIlegivel()
    df = pd.DataFrame({"s": [None], "_error_details": [None]}, dtype=object)

    aviso = core._record_error_safely(
        df,
        0,
        erro,
        status_col="s",
        config=_config(),
        row_already_processed=ja_processada,
        reprocess_columns=reprocessar,
        token_stats=core._empty_token_stats(),
    )

    assert aviso.startswith("Falha ao processar linha 0.")
    assert (aviso == "Falha ao processar linha 0.") == (registro == "reserva")
    assert df.loc[0, "s"] == "error"
    assert not df.loc[0, "_error_details"].endswith(core._KEPT_VALUES_NOTE)


@pytest.mark.parametrize("paralelo", [1, 2])
def test_erro_da_linha_avisa_com_a_mensagem_amigavel_do_provider(capsys, paralelo):
    erro = Exception("401 Unauthorized: invalid api key")
    esperada = get_friendly_error_message(erro, "openai")
    assert esperada != get_friendly_error_message(erro, None)

    with pytest.warns(UserWarning, match="Falha ao processar linha 0") as avisos:
        saida = _roda(["a"], _llm({"a": erro}), provider="openai", parallel_requests=paralelo)

    assert [str(aviso.message) for aviso in avisos] == [f"Falha ao processar linha 0.\n{esperada}"]
    assert capsys.readouterr().out == ""
    assert saida.loc[0, "_error_details"].endswith("Exception: 401 Unauthorized: invalid api key")


# _SnapshotWriter


def test_primeiro_snapshot_e_gravado_com_a_assinatura(tmp_path):
    caminho = tmp_path / "c.csv"
    writer = core._SnapshotWriter(core._Checkpoint(caminho, 1, meta={"versao": "x"}))

    writer.save(pd.DataFrame({"a": [1]}), 1)

    assert writer.last_saved == 1
    assert caminho.exists()
    assinatura = core._read_checkpoint_signature(caminho)
    assert assinatura is not None
    assert assinatura["versao"] == "x"


# Modo paralelo


def test_resumo_do_paralelo_conta_requisicoes_e_tempo(caplog):
    with caplog.at_level(logging.INFO, logger="dataframeit.stats"):
        _roda(["a", "b", "c"], _llm(), parallel_requests=2)

    saida = caplog.text
    assert "METRICAS DE THROUGHPUT" in saida
    assert "Requisicoes: 3\n" in saida
    assert re.search(r"Tempo total: \d{1,3}\.\ds\n", saida)


@contextmanager
def _sem_handler_no_root():
    """Tira os handlers do root, como num script sem logging configurado.

    Devolve-os antes do fim da fase de chamada, em que o pytest remove os dele.
    """
    raiz = logging.getLogger()
    handlers = raiz.handlers[:]
    for handler in handlers:
        raiz.removeHandler(handler)
    try:
        yield
    finally:
        for handler in handlers:
            raiz.addHandler(handler)


def test_resumo_sai_em_stderr_sem_logging_configurado(capsys):
    with _sem_handler_no_root():
        saida = _roda(["a"], _llm())

    capturado = capsys.readouterr()
    assert capturado.out == ""
    assert capturado.err.count("ESTATISTICAS DE USO") == 1
    assert saida["_input_tokens"].tolist() == [1]


def test_resumo_sai_uma_vez_pelo_logging_configurado(capsys, caplog):
    with caplog.at_level(logging.INFO, logger="dataframeit.stats"):
        _roda(["a"], _llm())

    assert [r.name for r in caplog.records if "ESTATISTICAS" in r.getMessage()] == [
        "dataframeit.stats"
    ]
    assert "ESTATISTICAS" not in capsys.readouterr().err


def test_nivel_warning_silencia_o_resumo_e_mantem_as_colunas(capsys):
    stats = logging.getLogger("dataframeit.stats")
    stats.setLevel(logging.WARNING)
    try:
        with _sem_handler_no_root():
            saida = _roda(["a"], _llm())
    finally:
        stats.setLevel(logging.INFO)

    capturado = capsys.readouterr()
    assert capturado.out == ""
    assert "ESTATISTICAS" not in capturado.err
    assert saida["_output_tokens"].tolist() == [1]


class _Coletor(logging.Handler):
    def __init__(self):
        super().__init__()
        self.registros = []

    @override
    def emit(self, record):
        self.registros.append(record)


def test_resumo_sai_uma_vez_por_handler_no_logger_do_pacote(capsys):
    handler = _Coletor()
    pacote = logging.getLogger("dataframeit")
    pacote.addHandler(handler)
    try:
        with _sem_handler_no_root():
            _roda(["a"], _llm())
    finally:
        pacote.removeHandler(handler)

    assert ["ESTATISTICAS" in r.getMessage() for r in handler.registros] == [True]
    assert "ESTATISTICAS" not in capsys.readouterr().err


def _registro_sem_propagacao(monkeypatch, nome):
    logger = logging.getLogger(nome)
    # O pytest pendura os handlers de captura em todo logger que já está sem
    # propagação quando cada fase de teste começa. Se o propagate=False ficasse
    # depois do teste, o logger chegaria à execução seguinte com esses handlers,
    # e a reserva, que os toma por handlers da aplicação, não escreveria.
    monkeypatch.setattr(logger, "propagate", False)
    return logger.makeRecord(nome, logging.INFO, __file__, 0, "resumo", (), None)


def test_logger_sem_propagacao_nao_olha_os_handlers_acima(capsys, monkeypatch):
    """Sem propagação, o handler da raiz não recebe o registro, e a reserva escreve."""
    raiz = _Coletor()
    logging.getLogger().addHandler(raiz)
    try:
        core._FallbackHandler().emit(
            _registro_sem_propagacao(monkeypatch, "dataframeit.teste_sem_propagacao")
        )
    finally:
        logging.getLogger().removeHandler(raiz)

    assert capsys.readouterr().err == "resumo\n"
    assert raiz.registros == []


def test_falha_ao_escrever_segue_o_tratamento_do_logging(monkeypatch):
    class _StderrQueFalha:
        def write(self, _texto):
            msg = "stderr fechado"
            raise OSError(msg)

    handler = core._FallbackHandler()
    registro = _registro_sem_propagacao(monkeypatch, "dataframeit.teste_stderr_fechado")
    monkeypatch.setattr(core.sys, "stderr", _StderrQueFalha())

    with patch.object(handler, "handleError") as tratamento:
        handler.emit(registro)

    tratamento.assert_called_once_with(registro)


def test_handler_de_reserva_e_instalado_uma_vez():
    logger = logging.getLogger("dataframeit.teste_reserva")
    try:
        core._install_fallback_handler(logger)
        core._install_fallback_handler(logger)
        assert len(logger.handlers) == 1
    finally:
        logger.handlers.clear()


class _Barra:
    """Registra o rótulo, o total, os avanços e o fechamento da barra de progresso."""

    criadas: list[Any]

    def __init__(self, total=None, desc=None):
        self.total, self.desc, self.n = total, desc, 0
        self.fechada = False
        _Barra.criadas.append(self)

    def update(self, passos):
        self.n += passos

    def close(self):
        self.fechada = True


def _barra(monkeypatch, classe=_Barra):
    _Barra.criadas = []
    monkeypatch.setattr(core, "tqdm", classe)
    return _Barra.criadas


def test_barra_do_paralelo_mostra_total_e_reprocessamento(monkeypatch):
    barras = _barra(monkeypatch)
    df = pd.DataFrame(
        {
            "texto": ["a", "b", "c"],
            "campo1": ["x", "x", "x"],
            "campo2": ["x", "x", "x"],
            "_dataframeit_status": ["processed"] * 3,
        }
    )
    _roda(df, _llm(), parallel_requests=2, reprocess_columns=["campo1"], track_tokens=False)

    [barra] = barras
    assert barra.desc == "Processando [pandas+langchain] [2 workers] (reprocessando: campo1)"
    assert (barra.total, barra.n) == (3, 3)


@pytest.mark.parametrize(
    ("erro", "esperado"),
    [
        (ProviderAbortError("sem crédito"), pytest.warns(UserWarning, match="sem crédito")),
        (KeyboardInterrupt(), pytest.raises(KeyboardInterrupt)),
    ],
    ids=["erro_que_interrompe", "ctrl_c"],
)
@pytest.mark.parametrize("paralelo", [1, 2])
def test_barra_fecha_quando_a_execucao_e_interrompida(monkeypatch, paralelo, erro, esperado):
    barras = _barra(monkeypatch)

    with esperado:
        _roda(["a", "b", "c"], _llm({"b": erro}), parallel_requests=paralelo, track_tokens=False)

    assert [barra.fechada for barra in barras] == [True]


def _barra_que_falha(monkeypatch, ponto):
    """Barra cuja criação, avanço ou fechamento levanta BrokenPipeError, conforme o ponto.

    Devolve as barras criadas e o llm das linhas "a" e "b", em que "b" levanta
    o erro pedido. No modo paralelo, quando a falha é no avanço, "b" espera o
    avanço de "a": sem isso, as duas podiam terminar juntas, e a interrupção
    sair antes dele.
    """
    avancou = threading.Event()

    def falha(onde):
        if onde in ponto:
            raise BrokenPipeError(32, "Broken pipe")

    class _BarraQueFalha(_Barra):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.avancos = self.fechamentos = 0
            falha("criar")

        @override
        def update(self, passos):
            self.avancos += 1
            avancou.set()
            falha("avancar")
            super().update(passos)

        @override
        def close(self):
            self.fechamentos += 1
            falha("fechar")
            super().close()

    def llm(erro):
        respostas = _llm({"b": erro} if erro else None)

        def responde(text, *args, **kwargs):
            if text == "b" and erro and "avancar" in ponto:
                assert avancou.wait(timeout=10)
            return respostas(text)

        return responde

    return _barra(monkeypatch, _BarraQueFalha), llm


_FINS = pytest.mark.parametrize(
    "fim",
    [(KeyboardInterrupt(), None), (ProviderAbortError("sem crédito"), None), (None, "processed")],
    ids=["ctrl_c", "erro_que_interrompe", "fim_normal"],
)


def _linha_pendente_e_linha_com_erro():
    return pd.DataFrame(
        {
            "texto": ["a", "b"],
            "_dataframeit_status": [None, "error"],
            "_error_details": [None, "x"],
        }
    )


def _confere_gravado(ckpt, segunda):
    gravado = pd.read_csv(ckpt).astype(object).where(pd.notna, None)
    assert gravado["_dataframeit_status"].tolist() == ["processed", segunda]
    assert gravado["_error_details"].tolist() == [None, None]


_PONTOS = pytest.mark.parametrize("ponto", ["criar", "avancar", "fechar", "avancar_e_fechar"])
_AVISO_DA_BARRA = (
    "Falha na barra de progresso, que não interrompeu a execução: "
    "BrokenPipeError: [Errno 32] Broken pipe"
)


@_PONTOS
@_FINS
@pytest.mark.parametrize("paralelo", [1, 2])
def test_falha_da_barra_vira_um_aviso_e_nao_troca_o_resultado_nem_a_interrupcao(
    monkeypatch, tmp_path, paralelo, fim, ponto
):
    """A barra falha, e a execução termina ou é interrompida na segunda linha.

    O resultado ou o Ctrl-C chegam a quem chamou, com um aviso só. A primeira
    linha, já registrada, e a segunda, que tinha erro e ia ser refeita, chegam
    ao arquivo; a interrompida vai sem status. A barra que falhou ao avançar
    ainda é fechada.
    """
    erro, segunda = fim
    barras, llm = _barra_que_falha(monkeypatch, ponto)
    ckpt = tmp_path / "c.csv"
    interrompe = pytest.raises(KeyboardInterrupt) if isinstance(erro, KeyboardInterrupt) else None

    resultado = None
    with warnings.catch_warnings(record=True) as avisos, interrompe or nullcontext() as excinfo:
        warnings.simplefilter("always")
        resultado = _roda(
            _linha_pendente_e_linha_com_erro(),
            llm(erro),
            resume=False,
            parallel_requests=paralelo,
            batch_size=10,
            checkpoint_path=ckpt,
        )

    if excinfo:
        assert excinfo.value.__context__ is None
    else:
        assert resultado is not None
        assert resultado["campo1"].tolist() == ["a", "b" if segunda else None]
    assert [str(aviso.message) for aviso in avisos if "barra" in str(aviso.message)] == [
        _AVISO_DA_BARRA
    ]
    [barra] = barras
    if ponto == "criar":
        assert (barra.avancos, barra.fechamentos) == (0, 0)
    else:
        assert barra.fechamentos == 1
    _confere_gravado(ckpt, segunda)


@_PONTOS
@_FINS
@pytest.mark.parametrize("paralelo", [1, 2])
def test_aviso_da_barra_tratado_como_erro_sai_depois_das_marcas_e_da_gravacao(
    monkeypatch, tmp_path, paralelo, fim, ponto
):
    """O aviso da falha levanta, e só na saída: a barra é o último bloco a fechar.

    O Ctrl-C que já vinha saindo continua sendo o que sai, com o aviso da barra
    como nota.
    """
    erro, segunda = fim
    sai = UserWarning
    if isinstance(erro, KeyboardInterrupt):
        # Um Ctrl-C novo: o dos parâmetros é o mesmo objeto em todos os casos,
        # e a nota ficaria nele.
        sai, erro = KeyboardInterrupt, KeyboardInterrupt()
    _barras, llm = _barra_que_falha(monkeypatch, ponto)
    ckpt = tmp_path / "c.csv"

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Falha na barra de progresso")
        warnings.filterwarnings("ignore", message="Execução interrompida")
        with pytest.raises(sai) as excinfo:
            _roda(
                _linha_pendente_e_linha_com_erro(),
                llm(erro),
                resume=False,
                parallel_requests=paralelo,
                batch_size=10,
                checkpoint_path=ckpt,
            )

    if sai is KeyboardInterrupt:
        assert excinfo.value.__notes__ == [
            f"Na saída por esta exceção: UserWarning: {_AVISO_DA_BARRA}"
        ]
    else:
        assert str(excinfo.value) == _AVISO_DA_BARRA
        assert excinfo.value.__context__ is None
    _confere_gravado(ckpt, segunda)


@pytest.mark.parametrize("paralelo", [1, 2])
def test_tqdm_com_a_saida_quebrada_no_meio_da_execucao_nao_troca_o_resultado(monkeypatch, paralelo):
    """O tqdm de verdade escreve numa saída que passa a levantar BrokenPipeError."""

    class _SaidaQueQuebra:
        quebrada = False

        def write(self, texto):
            if self.quebrada:
                raise BrokenPipeError(32, "Broken pipe")
            return len(texto)

        def flush(self):
            pass

    saida = _SaidaQueQuebra()
    monkeypatch.setattr(core, "tqdm", partial(core.tqdm, file=saida))
    respostas = _llm()

    def llm(text, *args, **kwargs):
        saida.quebrada = True
        return respostas(text)

    with pytest.warns(UserWarning, match=re.escape(_AVISO_DA_BARRA)):
        resultado = _roda(["a"], llm, parallel_requests=paralelo, track_tokens=False)

    assert resultado["campo1"].tolist() == ["a"]


@pytest.mark.parametrize("paralelo", [1, 2])
def test_reprocessamento_grava_tudo_na_linha_que_tinha_erro(paralelo):
    df = pd.DataFrame(
        {
            "texto": ["a", "b"],
            "campo1": ["x", None],
            "campo2": ["x", None],
            "_dataframeit_status": ["processed", "error"],
        }
    )
    saida = _roda(df, _llm(), parallel_requests=paralelo, reprocess_columns=["campo1"])

    assert saida["campo1"].tolist() == ["a", "b"]
    assert saida["campo2"].tolist() == ["x", "b"]


def test_paralelo_pede_so_os_campos_reprocessados_e_grava_o_trace_por_campo():
    chamadas = {}
    modos = []

    def fabrica(modelo, prompt, config, trace_mode):
        modos.append(trace_mode)
        return por_campo

    def por_campo(text, only_fields=None, known=None):
        chamadas[text] = only_fields
        return {
            "data": {"campo1": text, "campo2": text},
            "usage": None,
            "traces": {"campo1": {"t": text}, "campo2": {"t": text}},
        }

    df = pd.DataFrame(
        {
            "texto": ["a", "b"],
            "campo1": ["x", None],
            "campo2": ["x", None],
            "_dataframeit_status": ["processed", None],
        }
    )
    with (
        patch("dataframeit.agent.field_extractor", side_effect=fabrica),
        patch("dataframeit.core.validate_provider_dependencies"),
        patch("dataframeit.core.validate_search_dependencies"),
        pytest.warns(UserWarning, match="pode exceder rate limits de busca"),
    ):
        saida = dataframeit(
            df,
            questions=Modelo,
            prompt="{texto}",
            use_search=True,
            search_per_field=True,
            save_trace="full",
            parallel_requests=2,
            reprocess_columns=["campo1"],
            track_tokens=False,
        )

    assert modos == ["full"]
    assert chamadas == {"a": {"campo1"}, "b": None}
    assert saida["_trace_campo1"].tolist() == ['{"t": "a"}', '{"t": "b"}']


def test_linha_ja_registrada_nao_e_regravada_quando_o_snapshot_falha(monkeypatch, tmp_path):
    _gravacoes(monkeypatch, falhar=True)

    with (
        pytest.warns(UserWarning, match="sem texto não vão ao LLM"),
        pytest.warns(UserWarning, match="Falha ao processar linha 1"),
        pytest.warns(UserWarning, match="Erro inesperado no executor"),
    ):
        saida = _roda(
            [None, "erro", "ok"],
            _llm({"erro": ValueError("falhou")}),
            parallel_requests=3,
            batch_size=1,
            checkpoint_path=tmp_path / "c.csv",
        )

    detalhes = saida["_error_details"].tolist()
    assert saida["_dataframeit_status"].tolist() == ["error", "error", "processed"]
    assert detalhes[0] == core._MISSING_TEXT_DETAIL
    assert detalhes[1].endswith("ValueError: falhou")
    assert not detalhes[1].startswith("[Falha ao registrar o erro]")


@pytest.mark.parametrize("caminho", ["erro", "reserva"])
def test_cada_linha_conta_uma_vez_para_o_checkpoint(monkeypatch, tmp_path, caminho):
    rotulos = _gravacoes(monkeypatch)
    if caminho == "reserva":
        textos = ["a", None, "c"]
        monkeypatch.setattr(core, "_record_missing_text", _levanta)
    else:
        textos = ["a", "erro", "c"]

    # A linha sem texto da reserva avisa antes de cair na falha da gravação.
    aviso_sem_texto = (
        pytest.warns(UserWarning, match="sem texto não vão ao LLM")
        if caminho == "reserva"
        else nullcontext()
    )
    with aviso_sem_texto, pytest.warns(UserWarning, match="Falha ao processar linha 1"):
        _roda(
            textos,
            _llm({"erro": ValueError("falhou")}),
            parallel_requests=2,
            batch_size=1,
            checkpoint_path=tmp_path / "c.csv",
        )

    assert sorted(rotulos) == [1, 2, 3]


def _levanta(*args, **kwargs):
    msg = "gravação"
    raise RuntimeError(msg)


def test_linha_marcada_pela_coleta_mantem_o_erro_depois_da_interrupcao(monkeypatch):
    monkeypatch.setattr(core, "_record_missing_text", _levanta)

    with (
        pytest.warns(UserWarning, match="sem texto não vão ao LLM"),
        pytest.warns(UserWarning, match="Falha ao processar linha 1"),
        pytest.warns(UserWarning, match="interrompida"),
    ):
        saida = _roda(
            ["a", None, "c"],
            _llm({"c": ProviderAbortError("app-server encerrou")}),
            parallel_requests=2,
        )

    status = saida["_dataframeit_status"].tolist()
    assert status[:2] == ["processed", "error"]
    assert pd.isna(status[2])


class _TimerFalso:
    def __init__(self, intervalo, funcao):
        self.funcao = funcao

    def start(self):
        self.funcao()


def test_rate_limit_reduz_os_workers_pela_metade(monkeypatch, capsys):
    monkeypatch.setattr(core.time, "sleep", lambda _: None)
    monkeypatch.setattr(core.threading, "Timer", _TimerFalso)
    textos = [f"t{i}" for i in range(8)]

    with (
        pytest.warns(UserWarning, match="Falha ao processar linha 0"),
        pytest.warns(UserWarning, match="reduzidos por rate limit"),
        pytest.warns(UserWarning, match="Reduzindo workers de 4 para 2") as avisos,
    ):
        saida = _roda(textos, _llm({"t0": ProviderOverloadedError("429")}), parallel_requests=4)

    final = [a for a in avisos if "reduzidos por rate limit" in str(a.message)]
    assert [str(a.message) for a in final] == [
        (
            "Workers reduzidos por rate limit: de 4 para 2. Considere usar "
            "parallel_requests=2 para evitar rate limits."
        )
    ]
    # O aviso aponta a chamada de dataframeit, e não o interior da lib.
    assert final[0].filename == __file__
    reducoes = [a for a in avisos if "Reduzindo workers" in str(a.message)]
    assert len(reducoes) == 1
    assert reducoes[0].filename == __file__
    assert "dataframeit(" in linecache.getline(reducoes[0].filename, reducoes[0].lineno)
    assert capsys.readouterr().out == ""
    assert saida["_error_details"].iloc[0].endswith("ProviderOverloadedError: 429")


@pytest.mark.parametrize("parallel_requests", [1, 2])
def test_erro_sem_texto_legivel_nao_conta_como_rate_limit(parallel_requests):
    with pytest.warns(UserWarning, match="Falha ao processar linha 0") as avisos:
        saida = _roda(["a", "b"], _llm({"a": _ErroIlegivel()}), parallel_requests=parallel_requests)

    assert saida["_dataframeit_status"].tolist() == ["error", "processed"]
    assert not [a for a in avisos if "workers" in str(a.message).lower()]
    assert avisos[0].filename == __file__
    assert "dataframeit(" in linecache.getline(avisos[0].filename, avisos[0].lineno)


def test_aviso_da_thread_respeita_filtro_de_modulo():
    with warnings.catch_warnings(record=True) as filtrados:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", message=r"Falha ao processar linha 0", module=__name__)
        _roda(["a"], _llm({"a": ValueError("falhou")}), parallel_requests=2)

    assert not [aviso for aviso in filtrados if "Falha ao processar linha 0" in str(aviso.message)]


# Modo sequencial


def test_barra_do_sequencial_mostra_ritmo_reprocessamento_e_total(monkeypatch):
    monkeypatch.setattr(core.time, "sleep", lambda _: None)
    barras = _barra(monkeypatch)
    df = pd.DataFrame(
        {
            "texto": ["a", "b", "c"],
            "campo1": ["x", "x", "x"],
            "campo2": ["x", "x", "x"],
            "_dataframeit_status": ["processed"] * 3,
        }
    )
    _roda(df, _llm(), rate_limit_delay=1.0, reprocess_columns=["campo1"], track_tokens=False)

    [barra] = barras
    assert barra.desc == "Processando [pandas+langchain] [~60 req/min] (reprocessando: campo1)"
    assert (barra.total, barra.n) == (3, 3)


def test_barra_do_sequencial_conta_a_linha_que_ja_estava_processada(monkeypatch):
    barras = _barra(monkeypatch)
    df = pd.DataFrame(
        {
            "texto": ["a", "b", "c"],
            "campo1": ["x", None, None],
            "campo2": ["x", None, None],
            "_dataframeit_status": ["processed", None, None],
        }
    )
    chamadas = MagicMock(side_effect=_llm())

    _roda(df, chamadas, track_tokens=False)

    [barra] = barras
    assert chamadas.call_count == 2
    assert (barra.total, barra.n) == (3, 3)


@pytest.mark.parametrize(("pausa", "esperadas"), [(1.0, [1.0]), (0, [])])
def test_sequencial_pausa_so_depois_de_linha_com_sucesso(monkeypatch, pausa, esperadas):
    pausas = []
    monkeypatch.setattr(core.time, "sleep", pausas.append)

    with pytest.warns(UserWarning, match="Falha ao processar linha 1"):
        _roda(["a", "erro"], _llm({"erro": ValueError("falhou")}), rate_limit_delay=pausa)

    assert pausas == esperadas


def test_interrupcao_no_fim_de_um_lote_grava_as_marcas_no_checkpoint(tmp_path):
    df = pd.DataFrame(
        {
            "texto": ["a", "b", "c", "d"],
            "_dataframeit_status": [None, None, "error", "error"],
            "_error_details": [None, None, "x", "x"],
        }
    )
    ckpt = tmp_path / "c.csv"

    with pytest.warns(UserWarning, match="interrompida"):
        _roda(
            df,
            _llm({"c": ProviderAbortError("app-server encerrou")}),
            resume=False,
            batch_size=2,
            checkpoint_path=ckpt,
        )

    gravado = pd.read_csv(ckpt)
    assert gravado["_dataframeit_status"].tolist()[:2] == ["processed", "processed"]
    assert gravado["_dataframeit_status"].isna().tolist()[2:] == [True, True]


def test_paralelo_nao_passa_do_numero_de_workers(monkeypatch):
    em_voo = []
    pico = []
    trava = threading.Lock()

    def llm(text, *args, **kwargs):
        with trava:
            em_voo.append(text)
            pico.append(len(em_voo))
        time.sleep(0.02)
        with trava:
            em_voo.remove(text)
        return {"data": {"campo1": text, "campo2": text}, "usage": None}

    _roda([f"t{i}" for i in range(6)], llm, parallel_requests=2, track_tokens=False)

    assert max(pico) <= 2
