"""O laço das linhas: colunas que ele prepara, o que grava em cada linha e as marcas da interrupção."""

import linecache
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel

from dataframeit import ProviderUsageLimitError, core, dataframeit
from dataframeit.llm import LLMConfig, SearchConfig


class Modelo(BaseModel):
    campo1: str


def _roda(df, llm, **kwargs):
    with (
        patch("dataframeit.core.call_langchain", side_effect=llm),
        patch("dataframeit.core.validate_provider_dependencies"),
    ):
        return dataframeit(df, questions=Modelo, prompt="{texto}", **kwargs)


def _resposta(texto, *args, **kwargs):
    return {"data": {"campo1": texto}, "usage": None}


# =============================================================================
# Colunas preparadas antes do laço
# =============================================================================


@pytest.mark.parametrize("track_tokens", [False, True])
def test_colunas_novas_comecam_vazias_e_as_de_token_so_com_track_tokens(track_tokens):
    df = pd.DataFrame({"texto": ["a", "b"]})

    core._setup_columns(df, ["campo1"], None, track_tokens=track_tokens)

    tokens = {col: [None, None] for col in core.TOKEN_COLUMNS} if track_tokens else {}
    assert df.to_dict("list") == {
        "texto": ["a", "b"],
        "campo1": [None, None],
        "_dataframeit_status": [None, None],
        "_error_details": [None, None],
        **tokens,
    }


def test_coluna_de_trace_comeca_vazia():
    df = pd.DataFrame({"texto": ["a", "b"]})

    core._setup_columns(df, [], None, track_tokens=False, trace_mode="full")

    assert df.to_dict("list") == {
        "texto": ["a", "b"],
        "_dataframeit_status": [None, None],
        "_error_details": [None, None],
        "_trace": [None, None],
    }


def test_colunas_que_ja_existem_ficam_como_estao():
    df = pd.DataFrame(
        {"texto": ["a"], "campo1": ["v"], "s": ["processed"], "_error_details": ["x"]}
    )

    core._setup_columns(df, ["campo1"], "s", track_tokens=False)

    assert df.to_dict("list") == {
        "texto": ["a"],
        "campo1": ["v"],
        "s": ["processed"],
        "_error_details": ["x"],
    }


def test_colunas_viram_object_e_o_nan_so_vira_none_nas_que_guardam_nan():
    df = pd.DataFrame({"a": [1.5, np.nan], "b": [2.5, np.nan]})

    core._as_object_columns(df, ["a", "b", "ausente"], nan_columns={"a"})

    assert list(df.columns) == ["a", "b"]
    assert list(df.dtypes) == [object, object]
    assert df["a"].iloc[0] == 1.5
    assert df["a"].iloc[1] is None
    assert df["b"].iloc[0] == 2.5
    assert math.isnan(df["b"].iloc[1])


def test_valores_completados_entram_numa_coluna_que_nao_e_object():
    df = pd.DataFrame({"campo": [np.nan, np.nan]})

    core._apply_processed_values(df, {(0, "campo"): ["x", "y"], (1, "campo"): "texto"})

    assert df["campo"].dtype == object
    assert df["campo"].tolist() == [["x", "y"], "texto"]


# =============================================================================
# O que cada linha grava
# =============================================================================


def test_reprocessamento_campo_a_campo_passa_ao_backend_so_os_campos_mantidos():
    chamadas = []

    def invoke_partial(texto, campos, conhecidos):
        chamadas.append((texto, campos, conhecidos))
        return {"data": {}}

    backend = core.ProviderBackend(
        label="falso", invoke=lambda texto: {"data": {}}, invoke_partial=invoke_partial
    )
    linha = pd.Series({"texto": "t", "a": "antigo", "b": "mantido"})

    core._invoke_row(backend, "t", linha, ["a"], ["a", "b", "sem_coluna"])

    assert chamadas == [("t", {"a"}, {"b": "mantido"})]


def _grava_sucesso(resultado, *, search_config=None, trace_mode=None):
    df = pd.DataFrame(
        {
            "campo1": [None],
            "s": [None],
            "_error_details": [None],
            "_search_credits": [None],
            "_trace": [None],
        },
        dtype=object,
    )
    core._record_success(
        df,
        0,
        resultado,
        status_col="s",
        expected_columns=["campo1"],
        config=LLMConfig(
            model="m",
            provider="openai",
            api_key=None,
            max_retries=1,
            base_delay=0,
            max_delay=0,
            rate_limit_delay=0,
            search_config=search_config,
        ),
        track_tokens=False,
        row_already_processed=False,
        reprocess_columns=None,
        trace_mode=trace_mode,
        token_stats=core._empty_token_stats(),
    )
    return df.loc[0].to_dict()


def test_creditos_de_busca_ficam_zero_quando_o_uso_nao_os_informa():
    linha = _grava_sucesso(
        {"data": {"campo1": "v"}, "usage": {"input_tokens": 1}},
        search_config=SearchConfig(enabled=True),
    )

    assert linha["_search_credits"] == 0
    assert linha["_search_credits"] is not None


def test_trace_gravado_mantem_os_acentos():
    linha = _grava_sucesso(
        {"data": {"campo1": "v"}, "trace": {"mensagem": "decisão"}}, trace_mode="full"
    )

    assert linha["_trace"] == '{"mensagem": "decisão"}'


class _Relogio:
    """Relógio monotônico falso, que só anda quando o falso LLM o avança."""

    def __init__(self):
        self.agora = 1000.0

    def monotonic(self):
        return self.agora


@pytest.mark.parametrize("paralelo", [1, 2])
def test_tempo_da_linha_entregue_ao_callback_e_o_da_chamada(monkeypatch, paralelo):
    relogio = _Relogio()
    tempo = SimpleNamespace(monotonic=relogio.monotonic, sleep=lambda _s: None, time=lambda: 0.0)
    monkeypatch.setattr(core, "time", tempo)
    entregues = []

    def llm(texto, *args, **kwargs):
        relogio.agora += 2.5
        return _resposta(texto)

    _roda(
        pd.DataFrame({"texto": ["a"]}),
        llm,
        parallel_requests=paralelo,
        on_row_done=entregues.append,
    )

    assert [r.elapsed_seconds for r in entregues] == [2.5]


# =============================================================================
# Marcas das linhas não alcançadas
# =============================================================================


def _marca(df, *, registradas=(), reprocess_columns=None, value_columns=(), nan_columns=()):
    return core._mark_unreached(
        df,
        set(registradas),
        to_process=[True] * len(df),
        status_col="s",
        reprocess_columns=reprocess_columns,
        value_columns=value_columns,
        nan_columns=nan_columns,
    )


def test_marca_limpa_status_e_detalhe_das_linhas_nao_alcancadas_sem_valor():
    df = pd.DataFrame(
        {
            "s": [None, "error", None, None],
            "_error_details": ["falha antiga", None, "falha", None],
            "campo": [None, None, "valor", None],
        },
        dtype=object,
    )

    assert _marca(df, value_columns=["campo"]) is True

    assert df.to_dict("list") == {
        "s": [None, None, None, None],
        "_error_details": [None, None, None, None],
        "campo": [None, None, "valor", None],
    }


def test_linha_com_erro_fica_com_status_se_alguma_coluna_tem_valor():
    """O NaN de uma coluna que o guarda como valor conta, e a coluna com valor pode ser a segunda."""
    df = pd.DataFrame(
        {
            "s": ["error", "error", "error"],
            "_error_details": ["falha"] * 3,
            "a": [float("nan"), None, None],
            "b": [None, "valor", None],
        },
        dtype=object,
    )

    assert _marca(df, value_columns=["a", "b"], nan_columns={"a"}) is True

    assert df["s"].tolist() == ["error", "error", None]
    assert df["_error_details"].tolist() == ["falha", "falha", None]


def test_segunda_marca_do_reprocessamento_nao_muda_nada():
    df = pd.DataFrame(
        {"s": ["processed", "processed"], "_error_details": [None, None]}, dtype=object
    )

    assert _marca(df, registradas=[1], reprocess_columns=["campo"]) is True
    assert df["_error_details"].tolist() == [core._INTERRUPTED_REPROCESS_DETAIL, None]

    assert _marca(df, registradas=[1], reprocess_columns=["campo"]) is False
    assert df["_error_details"].tolist() == [core._INTERRUPTED_REPROCESS_DETAIL, None]


def _llm_que_recebe_ctrl_c(*args, **kwargs):
    raise KeyboardInterrupt


@pytest.mark.parametrize("paralelo", [1, 2])
def test_segundo_ctrl_c_ao_marcar_ainda_grava_as_marcas_no_checkpoint(
    tmp_path, monkeypatch, paralelo
):
    """O Ctrl-C chega na primeira chamada, e outro logo depois das marcas.

    Nenhuma linha foi registrada, e só as marcas mudaram o DataFrame: as linhas
    com erro que iam ser refeitas ficam sem status no arquivo.
    """
    marca = core._mark_unreached

    def marca_e_recebe_ctrl_c(*args, **kwargs):
        marca(*args, **kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr(core, "_mark_unreached", marca_e_recebe_ctrl_c)
    df = pd.DataFrame(
        {"texto": ["t0", "t1"], "_dataframeit_status": ["error"] * 2, "_error_details": ["x"] * 2}
    )
    ckpt = tmp_path / "ckpt.csv"

    with pytest.raises(KeyboardInterrupt):
        _roda(
            df,
            _llm_que_recebe_ctrl_c,
            resume=False,
            parallel_requests=paralelo,
            batch_size=10,
            checkpoint_path=ckpt,
        )

    gravado = pd.read_csv(ckpt)
    assert gravado["texto"].tolist() == ["t0", "t1"]
    assert gravado[["_dataframeit_status", "_error_details"]].isna().all(axis=None)


def test_ctrl_c_na_gravacao_final_de_uma_interrupcao_refaz_a_gravacao(tmp_path, monkeypatch):
    """A cota acaba na primeira linha, e o Ctrl-C corta a gravação final.

    Nenhuma linha foi registrada e as marcas não mudaram nada, mas a gravação
    interrompida é refeita antes de o Ctrl-C ser relançado.
    """
    grava = core._try_save_checkpoint
    gravacoes = []

    def gravacao_cortada_na_primeira_vez(*args, **kwargs):
        gravacoes.append(1)
        if len(gravacoes) == 1:
            raise KeyboardInterrupt
        return grava(*args, **kwargs)

    def llm(*args, **kwargs):
        msg = "usage limit"
        raise ProviderUsageLimitError(msg)

    monkeypatch.setattr(core, "_try_save_checkpoint", gravacao_cortada_na_primeira_vez)
    ckpt = tmp_path / "ckpt.csv"

    with pytest.raises(KeyboardInterrupt):
        _roda(pd.DataFrame({"texto": ["t0", "t1"]}), llm, batch_size=10, checkpoint_path=ckpt)

    assert len(gravacoes) == 2
    assert pd.read_csv(ckpt)["texto"].tolist() == ["t0", "t1"]


# =============================================================================
# Modo paralelo
# =============================================================================


def test_executor_tem_tantas_threads_quanto_parallel_requests(monkeypatch):
    tamanhos = []

    class Executor(ThreadPoolExecutor):
        def __init__(self, max_workers=None, **kwargs):
            tamanhos.append(max_workers)
            super().__init__(max_workers=max_workers, **kwargs)

    monkeypatch.setattr(core, "ThreadPoolExecutor", Executor)

    saida = _roda(pd.DataFrame({"texto": ["a", "b"]}), _resposta, parallel_requests=3)

    assert tamanhos == [3]
    assert saida["campo1"].tolist() == ["a", "b"]


def test_linha_nova_entra_assim_que_uma_termina_sem_esperar_a_fatia(monkeypatch):
    """Com fatia de espera longa, 'c' começa enquanto 'a' ainda roda.

    'a' só termina quando 'c' começa; se o laço esperasse todas as linhas em
    voo, ou a fatia inteira, 'a' desistiria de esperar.
    """
    monkeypatch.setattr(core, "_WAIT_SLICE_SECONDS", 30)
    c_comecou = threading.Event()
    a_viu_c = []

    def llm(texto, *args, **kwargs):
        if texto == "a":
            a_viu_c.append(c_comecou.wait(timeout=2))
        if texto == "c":
            c_comecou.set()
        return _resposta(texto)

    saida = _roda(pd.DataFrame({"texto": ["a", "b", "c"]}), llm, parallel_requests=2)

    assert a_viu_c == [True]
    assert saida["campo1"].tolist() == ["a", "b", "c"]


def test_erro_do_executor_depois_do_registro_aponta_a_chamada(tmp_path, monkeypatch):
    salvar = core._SnapshotWriter.save

    def save_que_falha_uma_vez(self, frame, label, **kwargs):
        if label == 1:
            msg = "gravação do snapshot"
            raise RuntimeError(msg)
        salvar(self, frame, label, **kwargs)

    monkeypatch.setattr(core._SnapshotWriter, "save", save_que_falha_uma_vez)

    with pytest.warns(UserWarning, match="Erro inesperado no executor") as avisos:
        _roda(
            pd.DataFrame({"texto": ["a", "b"]}),
            _resposta,
            parallel_requests=2,
            batch_size=1,
            checkpoint_path=tmp_path / "c.csv",
        )

    (aviso,) = [a for a in avisos if "Erro inesperado no executor" in str(a.message)]
    assert aviso.filename == __file__
    assert "dataframeit(" in linecache.getline(aviso.filename, aviso.lineno)


# =============================================================================
# Exceção que sai do laço
# =============================================================================


class _LimiteDeRequisicoes(Exception):
    status_code = 429


def _caso(sai, mensagem, *, linha=None, callback=None, gravacao=False):
    """O que levanta dentro do laço (na linha t1, no callback dela ou na 1ª gravação) e o que sai."""
    return SimpleNamespace(
        linha=linha, callback=callback, gravacao=gravacao, sai=sai, mensagem=mensagem
    )


@pytest.mark.parametrize("paralelo", [1, 2])
@pytest.mark.parametrize(
    "caso",
    [
        pytest.param(
            _caso(
                UserWarning,
                r"on_row_done falhou|Erro inesperado no executor",
                callback=RuntimeError("callback quebrou"),
            ),
            id="callback",
        ),
        pytest.param(_caso(SystemExit, "3", callback=SystemExit(3)), id="saida_no_callback"),
        pytest.param(
            _caso(UserWarning, "Falha ao processar linha 1", linha=ValueError("resposta ruim")),
            id="falha_de_linha",
        ),
        pytest.param(
            _caso(UserWarning, "Falha ao processar linha 1", linha=_LimiteDeRequisicoes()),
            id="rate_limit",
        ),
        pytest.param(
            _caso(
                UserWarning,
                r"Falha ao gravar o checkpoint|Erro inesperado no executor",
                gravacao=True,
            ),
            id="gravacao",
        ),
    ],
)
def test_excecao_que_sai_do_laco_grava_antes_as_linhas_processadas(
    tmp_path, monkeypatch, paralelo, caso
):
    """A linha t0 termina, e uma exceção sai do laço depois dela.

    A suíte trata aviso como erro, e por isso o aviso do callback que levanta,
    o da falha de linha, o do rate limit e o da gravação periódica que falha
    saem de `dataframeit()` como exceção, assim como o SystemExit do callback.
    A exceção continua saindo, mas a linha já processada chega antes ao
    checkpoint, e a retomada não a chama de novo.
    """
    t0_entregue = threading.Event()

    def llm(texto, *args, **kwargs):
        if texto != "t0":
            assert t0_entregue.wait(5)
        if texto == "t1" and caso.linha is not None:
            raise caso.linha
        return _resposta(texto)

    def on_row_done(linha):
        if linha.index == 0:
            t0_entregue.set()
        elif caso.callback is not None:
            raise caso.callback

    grava = core._save_checkpoint
    gravacoes = []

    def gravacao_que_falha_na_primeira_vez(*args, **kwargs):
        gravacoes.append(1)
        if caso.gravacao and len(gravacoes) == 1:
            msg = "disco cheio"
            raise OSError(msg)
        return grava(*args, **kwargs)

    monkeypatch.setattr(core, "_save_checkpoint", gravacao_que_falha_na_primeira_vez)
    textos = ["t0", "t1", "t2", "t3"]
    ckpt = tmp_path / "ckpt.csv"
    opcoes = {
        "parallel_requests": paralelo,
        "batch_size": 1 if caso.gravacao else 10,
        "checkpoint_path": ckpt,
    }

    with pytest.raises(caso.sai, match=caso.mensagem):
        _roda(pd.DataFrame({"texto": textos}), llm, on_row_done=on_row_done, **opcoes)

    gravado = pd.read_csv(ckpt)
    assert gravado["texto"].tolist() == textos
    assert gravado.loc[0, ["campo1", "_dataframeit_status"]].tolist() == ["t0", "processed"]
    pendentes = gravado.loc[gravado["_dataframeit_status"].isna(), "texto"].tolist()
    assert pendentes

    pagas = []

    def llm_da_retomada(texto, *args, **kwargs):
        pagas.append(texto)
        return _resposta(texto)

    saida = _roda(pd.DataFrame({"texto": textos}), llm_da_retomada, **opcoes)

    assert sorted(pagas) == pendentes
    assert saida["campo1"].tolist() == textos


@pytest.mark.parametrize("paralelo", [1, 2])
def test_excecao_que_sai_do_laco_marca_as_linhas_nao_alcancadas(tmp_path, paralelo):
    """Reprocessamento de 5 linhas processadas e uma com erro; o callback levanta.

    O aviso do callback sai como exceção na primeira linha entregue. Como no
    Ctrl-C, a linha processada que não foi alcançada mantém o valor antigo e
    ganha a marca em `_error_details`, e a linha com erro que ia ser refeita
    fica sem status, no DataFrame do usuário e no checkpoint. No paralelo, as
    duas linhas que já estavam com os workers terminam.
    """
    df = pd.DataFrame(
        {
            "texto": [f"t{i}" for i in range(6)],
            "campo1": ["antigo"] * 5 + [None],
            "_dataframeit_status": ["processed"] * 5 + ["error"],
            "_error_details": [None] * 5 + ["falha"],
        }
    )
    ckpt = tmp_path / "ckpt.csv"
    pagas = []

    def llm(texto, *args, **kwargs):
        pagas.append(texto)
        return {"data": {"campo1": f"r-{texto}"}, "usage": None}

    def on_row_done(linha):
        msg = "callback quebrou"
        raise RuntimeError(msg)

    with pytest.raises(UserWarning, match=r"on_row_done falhou|Erro inesperado no executor"):
        _roda(
            df,
            llm,
            reprocess_columns=["campo1"],
            parallel_requests=paralelo,
            batch_size=10,
            checkpoint_path=ckpt,
            on_row_done=on_row_done,
        )

    assert len(pagas) <= paralelo
    for saida in (df, pd.read_csv(ckpt)):
        alcancadas = saida["texto"].isin(pagas)
        com_erro = saida["texto"].eq("t5")
        antigas = ~alcancadas & ~com_erro
        assert list(saida.loc[alcancadas, "campo1"]) == [
            f"r-{texto}" for texto in saida.loc[alcancadas, "texto"]
        ]
        assert saida.loc[antigas, "campo1"].eq("antigo").all()
        assert saida.loc[antigas, "_error_details"].eq(core._INTERRUPTED_REPROCESS_DETAIL).all()
        assert saida.loc[~com_erro, "_dataframeit_status"].eq("processed").all()
        assert saida.loc[com_erro, ["_dataframeit_status", "_error_details"]].isna().all(axis=None)
