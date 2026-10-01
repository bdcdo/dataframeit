"""O que cada linha grava, a contagem do checkpoint e o resumo, nos dois modos."""

import re
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
from pydantic import BaseModel

from dataframeit import ProviderAbortError, ProviderOverloadedError, core
from dataframeit.core import dataframeit
from dataframeit.errors import get_friendly_error_message


class Modelo(BaseModel):
    campo1: str
    campo2: str


class _ErroIlegivel(Exception):
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
    erro = ValueError("falhou")
    if custo is not None:
        erro.cost_usd = custo
    estatisticas = core._empty_token_stats()
    df = pd.DataFrame({"s": [None], "_error_details": [None]}, dtype=object)

    with pytest.warns(UserWarning, match="Falha ao processar linha 0"):
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


@pytest.mark.parametrize("registro", ["normal", "reserva"])
@pytest.mark.parametrize(
    ("ja_processada", "reprocessar"), [(True, None), (False, ["campo1"]), (False, None)]
)
def test_nota_de_valores_mantidos_so_na_linha_processada_sob_reprocessamento(
    registro, ja_processada, reprocessar
):
    erro = ValueError("falhou") if registro == "normal" else _ErroIlegivel()
    df = pd.DataFrame({"s": [None], "_error_details": [None]}, dtype=object)

    with pytest.warns(UserWarning, match="Falha ao processar linha 0"):
        core._record_error_safely(
            df,
            0,
            erro,
            status_col="s",
            config=_config(),
            row_already_processed=ja_processada,
            reprocess_columns=reprocessar,
            token_stats=core._empty_token_stats(),
        )

    assert df.loc[0, "s"] == "error"
    assert not df.loc[0, "_error_details"].endswith(core._KEPT_VALUES_NOTE)


def test_erro_da_linha_mostra_a_mensagem_amigavel_do_provider(capsys):
    erro = Exception("401 Unauthorized: invalid api key")
    esperada = get_friendly_error_message(erro, "openai")
    assert esperada != get_friendly_error_message(erro, None)
    df = pd.DataFrame({"s": [None], "_error_details": [None]}, dtype=object)

    with pytest.warns(UserWarning, match="Falha ao processar linha 0"):
        core._record_error(
            df,
            0,
            erro,
            status_col="s",
            config=_config("openai"),
            row_already_processed=False,
            reprocess_columns=None,
            token_stats=core._empty_token_stats(),
        )

    assert capsys.readouterr().out == f"\n{esperada}\n\n"
    assert df.loc[0, "_error_details"].endswith("Exception: 401 Unauthorized: invalid api key")


# _SnapshotWriter


def test_primeiro_snapshot_e_gravado_com_a_assinatura(tmp_path):
    caminho = tmp_path / "c.csv"
    writer = core._SnapshotWriter(core._Checkpoint(caminho, 1, meta={"versao": "x"}))

    writer.save(pd.DataFrame({"a": [1]}), 1)

    assert writer.last_saved == 1
    assert caminho.exists()
    assert core._read_checkpoint_signature(caminho)["versao"] == "x"


# Modo paralelo


def test_resumo_do_paralelo_conta_requisicoes_e_tempo(capsys):
    _roda(["a", "b", "c"], _llm(), parallel_requests=2)

    saida = capsys.readouterr().out
    assert "METRICAS DE THROUGHPUT" in saida
    assert "Requisicoes: 3\n" in saida
    assert re.search(r"Tempo total: \d\.\ds\n", saida)
    assert "WORKERS REDUZIDOS" not in saida


def test_barra_do_paralelo_mostra_total_e_reprocessamento(capsys):
    df = pd.DataFrame(
        {
            "texto": ["a", "b", "c"],
            "campo1": ["x", "x", "x"],
            "campo2": ["x", "x", "x"],
            "_dataframeit_status": ["processed"] * 3,
        }
    )
    _roda(df, _llm(), parallel_requests=2, reprocess_columns=["campo1"], track_tokens=False)

    barra = capsys.readouterr().err
    assert "(reprocessando: campo1)" in barra
    assert "3/3" in barra
    assert "6/3" not in barra


def test_reprocessamento_no_paralelo_grava_tudo_na_linha_que_tinha_erro():
    df = pd.DataFrame(
        {
            "texto": ["a", "b"],
            "campo1": ["x", None],
            "campo2": ["x", None],
            "_dataframeit_status": ["processed", "error"],
        }
    )
    saida = _roda(df, _llm(), parallel_requests=2, reprocess_columns=["campo1"])

    assert saida["campo1"].tolist() == ["a", "b"]
    assert saida["campo2"].tolist() == ["x", "b"]


def test_paralelo_pede_so_os_campos_reprocessados_e_grava_o_trace_por_campo():
    chamadas = {}

    def por_campo(text, *args, only_fields=None, known=None):
        # args: modelo, prompt, config e trace_mode, nessa ordem.
        chamadas[text] = (args[3], only_fields)
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
        patch("dataframeit.agent.call_agent_per_field", side_effect=por_campo),
        patch("dataframeit.core.validate_provider_dependencies"),
        patch("dataframeit.core.validate_search_dependencies"),
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

    assert chamadas == {"a": ("full", {"campo1"}), "b": ("full", None)}
    assert saida["_trace_campo1"].tolist() == ['{"t": "a"}', '{"t": "b"}']


def test_linha_ja_registrada_nao_e_regravada_quando_o_snapshot_falha(monkeypatch, tmp_path):
    _gravacoes(monkeypatch, falhar=True)

    with pytest.warns(UserWarning, match="Erro inesperado no executor"):
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

    with pytest.warns(UserWarning, match="Falha ao processar linha 1"):
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

    with pytest.warns(UserWarning, match="interrompida"):
        saida = _roda(
            ["a", None, "c", "d"],
            _llm({"c": ProviderAbortError("app-server encerrou")}),
            parallel_requests=2,
        )

    status = saida["_dataframeit_status"].tolist()
    assert status[:2] == ["processed", "error"]
    assert pd.isna(status[2])
    assert pd.isna(status[3])


class _TimerFalso:
    def __init__(self, intervalo, funcao):
        self.funcao = funcao

    def start(self):
        self.funcao()


def test_rate_limit_reduz_os_workers_pela_metade(monkeypatch, capsys):
    monkeypatch.setattr(core.time, "sleep", lambda _: None)
    monkeypatch.setattr(core.threading, "Timer", _TimerFalso)
    textos = [f"t{i}" for i in range(8)]

    with pytest.warns(UserWarning, match="Reduzindo workers de 4 para 2"):
        saida = _roda(textos, _llm({"t0": ProviderOverloadedError("429")}), parallel_requests=4)

    assert "Workers finais:   2\n" in capsys.readouterr().out
    assert saida["_error_details"].iloc[0].endswith("ProviderOverloadedError: 429")


def test_erro_sem_texto_legivel_nao_conta_como_rate_limit(capsys):
    with pytest.warns(UserWarning, match="Falha ao processar linha 0"):
        saida = _roda(["a", "b"], _llm({"a": _ErroIlegivel()}), parallel_requests=2)

    assert saida["_dataframeit_status"].tolist() == ["error", "processed"]
    assert "WORKERS REDUZIDOS" not in capsys.readouterr().out
