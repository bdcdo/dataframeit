"""Falha que impede as linhas seguintes interrompe a execução sem gastá-las."""

import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pandas as pd
import pytest
from pydantic import BaseModel

from dataframeit import ProviderAbortError, ProviderUsageLimitError, dataframeit


class Modelo(BaseModel):
    campo1: str


def _llm_que_esgota(limite: int, erro: Exception):
    chamadas = []

    def llm(*args, **kwargs):
        chamadas.append(1)
        if len(chamadas) > limite:
            raise erro
        return {
            "data": {"campo1": f"v{len(chamadas)}"},
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }

    return chamadas, llm


def _roda(df, llm, **kwargs):
    with (
        patch("dataframeit.core.call_langchain", side_effect=llm),
        patch("dataframeit.core.validate_provider_dependencies"),
    ):
        return dataframeit(df, questions=Modelo, prompt="{texto}", **kwargs)


@pytest.mark.parametrize("paralelo", [1, 2])
def test_limite_de_uso_interrompe_e_deixa_as_linhas_sem_status(paralelo):
    df = pd.DataFrame({"texto": [f"linha{i}" for i in range(8)]})
    chamadas, llm = _llm_que_esgota(2, ProviderUsageLimitError("usage limit"))

    with pytest.warns(UserWarning, match=r"interrompida.*usage limit.*resume=True"):
        saida = _roda(df, llm, parallel_requests=paralelo)

    status = saida["_dataframeit_status"]
    assert (status == "processed").sum() == 2
    assert status.isna().sum() == 6
    assert "error" not in status.tolist()
    # Depois da interrupção nenhuma linha nova é despachada: no paralelo, só as
    # que já estavam em voo quando o limite chegou.
    assert len(chamadas) <= 2 + paralelo


def test_sem_interrupcao_as_colunas_de_controle_somem_como_antes():
    df = pd.DataFrame({"texto": ["a", "b"]})
    _, llm = _llm_que_esgota(10, ProviderUsageLimitError("nunca"))

    saida = _roda(df, llm)

    assert "_dataframeit_status" not in saida.columns


def test_retomada_depois_da_interrupcao_processa_so_o_que_faltou():
    df = pd.DataFrame({"texto": [f"linha{i}" for i in range(5)]})
    _, llm = _llm_que_esgota(3, ProviderAbortError("app-server encerrou"))
    with pytest.warns(UserWarning, match="interrompida"):
        parcial = _roda(df, llm)

    chamadas, llm_ok = _llm_que_esgota(100, ProviderAbortError("nunca"))
    final = _roda(parcial, llm_ok, resume=True)

    assert len(chamadas) == 2
    assert final["campo1"].tolist()[:3] == ["v1", "v2", "v3"]
    assert final["campo1"].notna().all()


@pytest.mark.parametrize("paralelo", [1, 2])
def test_interrupcao_grava_o_checkpoint(tmp_path, paralelo):
    df = pd.DataFrame({"texto": [f"linha{i}" for i in range(6)]})
    ckpt = tmp_path / "ckpt.csv"
    _, llm = _llm_que_esgota(1, ProviderUsageLimitError("usage limit"))

    with pytest.warns(UserWarning, match="interrompida"):
        _roda(df, llm, parallel_requests=paralelo, batch_size=100, checkpoint_path=ckpt)

    gravado = pd.read_csv(ckpt)
    assert (gravado["_dataframeit_status"] == "processed").sum() >= 1
    assert gravado["_dataframeit_status"].isna().sum() >= 6 - paralelo


@pytest.mark.parametrize("paralelo", [1, 2])
def test_interrupcao_no_reprocessamento_marca_as_linhas_que_ficaram_com_valor_antigo(paralelo):
    df = pd.DataFrame(
        {
            "texto": [f"t{i}" for i in range(6)],
            "campo1": ["antigo"] * 6,
            "_dataframeit_status": ["processed"] * 6,
        }
    )
    _, llm = _llm_que_esgota(2, ProviderUsageLimitError("usage limit"))

    with pytest.warns(
        UserWarning, match=r"0 linha\(s\) ficaram sem status, e \d+ não foram reprocessadas"
    ):
        saida = _roda(df, llm, reprocess_columns=["campo1"], parallel_requests=paralelo)

    detalhes = saida["_error_details"]
    antigas = saida["campo1"].eq("antigo")
    assert (saida["_dataframeit_status"] == "processed").all()
    assert antigas.sum() >= 1
    assert detalhes[antigas].str.startswith("Reprocessamento interrompido").all()
    assert detalhes[~antigas].isna().all()


def test_linha_com_erro_que_ia_ser_refeita_volta_a_ficar_pendente():
    df = pd.DataFrame(
        {
            "texto": ["a", "b", "c"],
            "_dataframeit_status": ["error", "error", "error"],
            "_error_details": ["x", "x", "x"],
        }
    )
    _, llm = _llm_que_esgota(1, ProviderUsageLimitError("usage limit"))

    with pytest.warns(UserWarning, match=r"2 linha\(s\) ficaram sem status"):
        saida = _roda(df, llm, resume=False)

    assert saida["_dataframeit_status"].tolist()[0] == "processed"
    assert saida["_dataframeit_status"].isna().tolist()[1:] == [True, True]
    assert saida["_error_details"].isna().tolist()[1:] == [True, True]


@pytest.mark.parametrize(
    ("entrada", "trecho"),
    [
        (lambda: ["a", "b", "c"], "só um checkpoint_path permite retomar"),
        (lambda: pd.DataFrame({"texto": ["a", "b", "c"]}), "resume=True sobre esta saída"),
    ],
)
def test_aviso_diz_como_retomar_conforme_a_entrada(entrada, trecho):
    _, llm = _llm_que_esgota(1, ProviderUsageLimitError("usage limit"))

    with pytest.warns(UserWarning, match=trecho):
        _roda(entrada(), llm)


def test_aviso_com_checkpoint_manda_usar_o_mesmo_arquivo(tmp_path):
    _, llm = _llm_que_esgota(1, ProviderUsageLimitError("usage limit"))

    with pytest.warns(UserWarning, match="mesmo checkpoint_path"):
        _roda(
            pd.DataFrame({"texto": ["a", "b"]}),
            llm,
            batch_size=5,
            checkpoint_path=tmp_path / "c.csv",
        )


def test_falha_ao_despachar_a_linha_interrompe_e_grava_o_checkpoint(tmp_path):
    df = pd.DataFrame({"texto": [f"linha{i}" for i in range(6)]})
    ckpt = tmp_path / "ckpt.csv"
    _, llm = _llm_que_esgota(100, ProviderAbortError("nunca"))
    submit = ThreadPoolExecutor.submit
    despachos = []

    def submit_que_esgota(self, *args, **kwargs):
        despachos.append(1)
        if len(despachos) > 2:
            msg = "can't start new thread"
            raise RuntimeError(msg)
        return submit(self, *args, **kwargs)

    with (
        patch.object(ThreadPoolExecutor, "submit", submit_que_esgota),
        pytest.warns(
            UserWarning,
            match=r"interrompida: RuntimeError: can't start new thread\. 4 linha\(s\) ficaram sem status",
        ),
    ):
        saida = _roda(df, llm, parallel_requests=2, batch_size=100, checkpoint_path=ckpt)

    assert saida["_dataframeit_status"].tolist()[:2] == ["processed", "processed"]
    assert saida["_dataframeit_status"].isna().tolist()[2:] == [True] * 4
    gravado = pd.read_csv(ckpt)
    assert gravado["_dataframeit_status"].tolist()[:2] == ["processed", "processed"]
    assert gravado["_dataframeit_status"].isna().sum() == 4


class _InterrupcaoIlegivel(ProviderAbortError):
    def __str__(self):
        msg = "sem texto"
        raise RuntimeError(msg)


@pytest.mark.parametrize("paralelo", [1, 2])
def test_interrupcao_com_texto_ilegivel_ainda_interrompe(paralelo):
    df = pd.DataFrame({"texto": [f"linha{i}" for i in range(4)]})
    _, llm = _llm_que_esgota(0, _InterrupcaoIlegivel())

    with pytest.warns(
        UserWarning, match=r"interrompida: _InterrupcaoIlegivel: _InterrupcaoIlegivel\(\)\. 4 linha"
    ):
        saida = _roda(df, llm, parallel_requests=paralelo)

    assert saida["_dataframeit_status"].isna().all()


# =============================================================================
# Ctrl-C
# =============================================================================


def _ctrl_c():
    """Manda SIGINT ao thread principal, como o Ctrl-C do terminal.

    No modo sequencial a chamada roda no thread principal, e o
    `KeyboardInterrupt` sai do próprio `pthread_kill`, antes de a chamada
    terminar. No paralelo, ele chega ao thread principal enquanto este espera as
    linhas, e a chamada do worker termina e é paga.
    """
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
    principal = threading.main_thread().ident
    assert principal is not None
    signal.pthread_kill(principal, signal.SIGINT)


def _llm_com_ctrl_c(na_chamada: int | None = None):
    """Devolve a lista dos textos cujas chamadas terminaram (pagas) e o falso LLM."""
    trava = threading.Lock()
    iniciadas: list[str] = []
    pagas: list[str] = []

    def llm(texto, *args, **kwargs):
        with trava:
            iniciadas.append(texto)
            vez = len(iniciadas)
        if vez == na_chamada:
            _ctrl_c()
        with trava:
            pagas.append(texto)
        return {"data": {"campo1": f"r-{texto}"}, "usage": {}}

    return pagas, llm


def _entrada(biblioteca: str, textos: list[str]):
    if biblioteca == "polars":
        pl = pytest.importorskip("polars")
        return pl.DataFrame({"texto": textos})
    return pd.DataFrame({"texto": textos})


requer_pthread_kill = pytest.mark.skipif(
    not hasattr(signal, "pthread_kill"), reason="Ctrl-C simulado com signal.pthread_kill"
)


@requer_pthread_kill
@pytest.mark.parametrize("paralelo", [1, 3])
@pytest.mark.parametrize("biblioteca", ["pandas", "polars"])
def test_ctrl_c_grava_as_linhas_pagas_e_a_retomada_nao_as_chama_de_novo(
    tmp_path, biblioteca, paralelo
):
    """12 linhas, batch_size=5 e Ctrl-C durante a 9ª chamada.

    No sequencial, a 9ª chamada não termina, e 8 linhas estão pagas. Com 3
    workers, a 9ª é a última do terceiro lote, e as três do lote terminam
    enquanto o executor as espera. A gravação a cada 5 linhas deixaria as
    últimas 3 ou 4 de fora.
    """
    textos = [f"t{i:02d}" for i in range(1, 13)]
    entrada = _entrada(biblioteca, textos)
    ckpt = tmp_path / "ckpt.csv"
    pagas, llm = _llm_com_ctrl_c(na_chamada=9)

    with pytest.raises(KeyboardInterrupt):
        _roda(
            entrada,
            llm,
            parallel_requests=paralelo,
            batch_size=5,
            checkpoint_path=ckpt,
        )

    assert len(pagas) == (8 if paralelo == 1 else 9)
    gravado = pd.read_csv(ckpt)
    processadas = gravado.loc[gravado["_dataframeit_status"].eq("processed"), "texto"]
    assert sorted(processadas) == sorted(pagas)

    retomada, llm_ok = _llm_com_ctrl_c()
    saida = _roda(
        _entrada(biblioteca, textos),
        llm_ok,
        parallel_requests=paralelo,
        batch_size=5,
        checkpoint_path=ckpt,
    )

    assert sorted(retomada) == sorted(set(textos) - set(pagas))
    assert list(saida["campo1"]) == [f"r-{texto}" for texto in textos]


@requer_pthread_kill
@pytest.mark.parametrize("paralelo", [1, 3])
def test_falha_ao_gravar_no_ctrl_c_avisa_e_mantem_o_keyboard_interrupt(tmp_path, paralelo):
    """A falha vira aviso, como nas gravações a cada batch_size."""
    _, llm = _llm_com_ctrl_c(na_chamada=5)

    with (
        patch("dataframeit.core._save_checkpoint", side_effect=OSError("disco cheio")) as grava,
        pytest.warns(UserWarning, match="Falha ao gravar o checkpoint.*OSError: disco cheio"),
        pytest.raises(KeyboardInterrupt),
    ):
        _roda(
            pd.DataFrame({"texto": [f"t{i}" for i in range(9)]}),
            llm,
            parallel_requests=paralelo,
            batch_size=100,
            checkpoint_path=tmp_path / "ckpt.csv",
        )

    assert grava.call_count == 1


@pytest.fixture
def sigint_reinicia_a_espera():
    """Põe SA_RESTART no SIGINT, como o handler que o polars instala ao ser importado.

    Com SA_RESTART, o kernel reinicia a espera sem prazo de um lock quando o
    SIGINT chega, e o thread principal só levanta o `KeyboardInterrupt` quando a
    espera termina. Sem isso, o SIGINT interromperia também a espera sem prazo,
    e o teste não a distinguiria da espera em fatias. Com o polars importado, o
    SA_RESTART já estava lá e fica.
    """
    com_polars = "polars" in sys.modules
    signal.siginterrupt(signal.SIGINT, False)
    yield
    if not com_polars:
        signal.siginterrupt(signal.SIGINT, True)


@requer_pthread_kill
@pytest.mark.usefixtures("sigint_reinicia_a_espera")
def test_segundo_ctrl_c_corta_a_espera_e_a_chamada_lenta_fica_pendente(tmp_path):
    """O segundo Ctrl-C chega enquanto a interrupção espera a chamada lenta.

    A espera acaba sem a chamada terminar, o checkpoint leva só a linha
    concluída, e a retomada chama a lenta de novo.
    """
    textos = ["lenta", "ctrl-c"]
    ckpt = tmp_path / "ckpt.csv"
    encerrando = threading.Event()
    liberada = threading.Event()
    terminou = threading.Event()
    encerra = ThreadPoolExecutor.shutdown

    def encerra_observado(self, *args, **kwargs):
        encerrando.set()
        return encerra(self, *args, **kwargs)

    def llm(texto, *args, **kwargs):
        if texto == "lenta":
            # O primeiro Ctrl-C encerra o executor antes de esperar esta
            # chamada; a folga deixa a espera começar antes do segundo.
            encerrando.wait(timeout=10)
            time.sleep(0.2)
            _ctrl_c()
            liberada.wait(timeout=10)
            terminou.set()
        else:
            _ctrl_c()
        return {"data": {"campo1": f"r-{texto}"}, "usage": {}}

    try:
        with (
            patch.object(ThreadPoolExecutor, "shutdown", encerra_observado),
            pytest.raises(KeyboardInterrupt),
        ):
            _roda(
                pd.DataFrame({"texto": textos}),
                llm,
                parallel_requests=2,
                batch_size=100,
                checkpoint_path=ckpt,
            )
        cortada = not terminou.is_set()
        gravado = pd.read_csv(ckpt)
    finally:
        liberada.set()

    assert cortada
    status = dict(zip(gravado["texto"], gravado["_dataframeit_status"], strict=True))
    assert status["ctrl-c"] == "processed"
    assert pd.isna(status["lenta"])

    retomada, llm_ok = _llm_com_ctrl_c()
    _roda(
        pd.DataFrame({"texto": textos}),
        llm_ok,
        parallel_requests=2,
        batch_size=100,
        checkpoint_path=ckpt,
    )
    assert retomada == ["lenta"]


@requer_pthread_kill
def test_ctrl_c_dentro_do_submit_espera_a_linha_que_ja_entrou_na_fila(tmp_path):
    """O Ctrl-C chega dentro do `submit` da última linha, que já entrou na fila.

    A linha roda e é paga, embora o laço não tenha guardado o futuro dela. A
    interrupção espera as threads do executor, e a linha entra no checkpoint.
    """
    textos = ["a", "b", "lenta"]
    ckpt = tmp_path / "ckpt.csv"
    pagas, llm_rapido = _llm_com_ctrl_c()

    def llm(texto, *args, **kwargs):
        if texto == "lenta":
            # Termina depois da gravação que esperasse só os futuros guardados.
            time.sleep(0.5)
        return llm_rapido(texto, *args, **kwargs)

    ajusta = ThreadPoolExecutor._adjust_thread_count
    despachos: list[int] = []

    def ajusta_e_interrompe(self):
        # O submit põe o item na fila e chama este método, que inicia a thread
        # quando falta uma; o KeyboardInterrupt do pthread_kill sai do submit
        # antes de o laço guardar o futuro.
        ajusta(self)
        despachos.append(1)
        if len(despachos) == len(textos):
            _ctrl_c()

    with (
        patch.object(ThreadPoolExecutor, "_adjust_thread_count", ajusta_e_interrompe),
        pytest.raises(KeyboardInterrupt),
    ):
        _roda(
            pd.DataFrame({"texto": textos}),
            llm,
            parallel_requests=3,
            batch_size=100,
            checkpoint_path=ckpt,
        )

    gravado = pd.read_csv(ckpt)
    status = dict(zip(gravado["texto"], gravado["_dataframeit_status"], strict=True))
    assert status == dict.fromkeys(textos, "processed")
    assert sorted(pagas) == sorted(textos)

    retomada, llm_ok = _llm_com_ctrl_c()
    _roda(
        pd.DataFrame({"texto": textos}),
        llm_ok,
        parallel_requests=3,
        batch_size=100,
        checkpoint_path=ckpt,
    )
    assert retomada == []


@pytest.mark.parametrize("paralelo", [1, 3])
def test_ctrl_c_antes_de_terminar_uma_linha_nao_reescreve_o_checkpoint(tmp_path, paralelo):
    """Sem linha registrada desde a última gravação, o arquivo fica como estava."""
    ckpt = tmp_path / "ckpt.csv"
    df = pd.DataFrame({"texto": ["a", "b", "c"]})
    _roda(
        df.copy(),
        _llm_com_ctrl_c()[1],
        model_kwargs={"temperature": 0.5},
        batch_size=10,
        checkpoint_path=ckpt,
    )
    anterior = ckpt.read_bytes()

    def interrompe(*args, **kwargs):
        raise KeyboardInterrupt

    with pytest.warns(UserWarning, match="outra configuração"), pytest.raises(KeyboardInterrupt):
        _roda(
            df.copy(),
            interrompe,
            parallel_requests=paralelo,
            batch_size=10,
            checkpoint_path=ckpt,
        )

    assert ckpt.read_bytes() == anterior


@requer_pthread_kill
@pytest.mark.parametrize("paralelo", [1, 3])
def test_ctrl_c_sem_checkpoint_deixa_as_linhas_pagas_na_entrada_pandas(paralelo):
    df = pd.DataFrame({"texto": [f"t{i}" for i in range(9)]})
    pagas, llm = _llm_com_ctrl_c(na_chamada=5)

    with pytest.raises(KeyboardInterrupt):
        _roda(df, llm, parallel_requests=paralelo)

    assert sorted(df.loc[df["_dataframeit_status"].eq("processed"), "texto"]) == sorted(pagas)
