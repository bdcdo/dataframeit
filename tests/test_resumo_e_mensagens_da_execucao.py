"""Resumo de uso, rótulo da barra de progresso e mensagens que a execução grava ou avisa."""

import logging
import math
import re

import pandas as pd
import pytest

from dataframeit import ProviderUsageLimitError
from dataframeit.core import (
    ProviderBackend,
    RowResult,
    TokenStats,
    _abort_reason,
    _add_failed_usage,
    _add_usage_stats,
    _empty_token_stats,
    _FallbackHandler,
    _format_token_stats,
    _install_fallback_handler,
    _is_missing_text,
    _missing_text_detail,
    _notify_row_done,
    _progress_description,
    _progress_suffix,
    _recorded_error,
    _warn_missing_texts,
)
from dataframeit.llm import LLMConfig, SearchConfig
from dataframeit.utils import (
    ORIGINAL_TYPE_LIST,
    ORIGINAL_TYPE_PANDAS_DF,
    ORIGINAL_TYPE_POLARS_DF,
    ConversionInfo,
)

_IGUAIS = "=" * 60
_TRACOS = "-" * 60


def _stats(**valores: float) -> TokenStats:
    return {**_empty_token_stats(), **valores}


def test_contadores_comecam_zerados():
    assert _empty_token_stats() == {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "reasoning_tokens": 0,
        "search_credits": 0,
        "search_count": 0,
        "cost_usd": 0.0,
    }


def test_resumo_sem_uso_e_vazio():
    assert _format_token_stats(_empty_token_stats(), "m", 1) == ""


def test_resumo_minimo_omite_cache_raciocinio_custo_tempo_e_busca():
    """Contadores zerados e tempo zero não abrem linha nem seção no resumo."""
    stats = _stats(input_tokens=1, total_tokens=1, elapsed_seconds=0.0, requests_completed=1)

    assert _format_token_stats(stats, "gpt-x", 4).splitlines() == [
        _IGUAIS,
        "ESTATISTICAS DE USO",
        _IGUAIS,
        "Modelo: gpt-x",
        "Total de tokens: 1",
        "  - Input:  1 tokens",
        "  - Output: 0 tokens",
        _IGUAIS,
    ]


def test_resumo_completo_traz_cache_raciocinio_custo_throughput_e_busca():
    stats = _stats(
        input_tokens=1000,
        cached_input_tokens=1,
        output_tokens=200,
        reasoning_tokens=1,
        total_tokens=1200,
        cost_usd=0.5,
        search_count=2,
        search_credits=4,
        elapsed_seconds=10.0,
        requests_completed=1,
    )

    assert _format_token_stats(stats, None, 3, search_provider="exa").splitlines() == [
        _IGUAIS,
        "ESTATISTICAS DE USO",
        _IGUAIS,
        "Modelo: escolhido pelo runtime do provider",
        "Total de tokens: 1,200",
        "  - Input:  1,000 tokens",
        "    └─ Cache: 1 (incluído no Input)",
        "  - Output: 200 tokens",
        "    └─ Reasoning: 1 (incluído no Output)",
        "Custo informado pelo provider: US$ 0.5000",
        _TRACOS,
        "METRICAS DE THROUGHPUT",
        _TRACOS,
        "Tempo total: 10.0s",
        "Workers paralelos: 3",
        "Requisicoes: 1",
        "  - RPM (req/min): 6.0",
        "  - TPM (tokens/min): 7,200",
        _TRACOS,
        "METRICAS DE BUSCA (EXA)",
        _TRACOS,
        "Total de buscas: 2",
        "Creditos usados: 4",
        _IGUAIS,
    ]


def test_resumo_sem_contagem_de_requisicoes_usa_um_worker_e_omite_a_taxa():
    stats = _stats(input_tokens=60, total_tokens=60, elapsed_seconds=60.0)

    assert _format_token_stats(stats, "m").splitlines()[7:] == [
        _TRACOS,
        "METRICAS DE THROUGHPUT",
        _TRACOS,
        "Tempo total: 60.0s",
        "Workers paralelos: 1",
        "  - TPM (tokens/min): 60",
        _IGUAIS,
    ]


def test_secao_de_busca_sem_provider_leva_o_nome_do_tavily():
    stats = _stats(total_tokens=1, search_count=1, search_credits=1)

    assert "METRICAS DE BUSCA (TAVILY)" in _format_token_stats(stats, "m", 1).splitlines()


def test_uso_da_linha_soma_ao_que_ja_havia():
    stats = _stats(input_tokens=10, total_tokens=10, cost_usd=0.25)

    _add_usage_stats(stats, {"input_tokens": 5, "total_tokens": 5, "cost_usd": 0.5})

    assert stats == _stats(input_tokens=15, total_tokens=15, cost_usd=0.75)


def test_falha_sem_custo_nem_uso_nao_muda_o_resumo():
    stats = _stats(total_tokens=3, cost_usd=0.25)

    _add_failed_usage(stats, RuntimeError("falhou"))

    assert stats == _stats(total_tokens=3, cost_usd=0.25)


def test_cota_esgotada_explica_que_repetir_nao_resolve():
    assert _abort_reason(ProviderUsageLimitError("cota do plano MAX.")) == (
        "ProviderUsageLimitError: cota do plano MAX. Repetir a linha não resolve: retome "
        "quando a cota voltar, com crédito, mudança de plano ou renovação do período na "
        "conta do serviço"
    )


@pytest.mark.parametrize(
    ("valor", "ausente"),
    [
        (None, True),
        (math.nan, True),
        ("  ", True),
        ("texto", False),
        (5, False),
        ([1, 2], False),
    ],
)
def test_texto_ausente_e_so_o_vazio_ou_nulo(valor, ausente):
    assert _is_missing_text(valor) is ausente


@pytest.mark.parametrize(
    ("ja_processada", "colunas", "detalhe"),
    [
        (False, ["a"], "Texto ausente"),
        (True, None, "Texto ausente"),
        (
            True,
            ["a"],
            "Texto ausente (reprocess_columns falhou; a linha mantém os valores anteriores)",
        ),
    ],
)
def test_linha_sem_texto_so_mantem_valores_quando_ja_processada_e_reprocessada(
    ja_processada, colunas, detalhe
):
    assert (
        _missing_text_detail(row_already_processed=ja_processada, reprocess_columns=colunas)
        == detalhe
    )


def test_aviso_de_linhas_sem_texto_conta_as_linhas_a_processar():
    df = pd.DataFrame({"texto": ["a", None, " ", "b"]})
    mensagem = "2 linha(s) sem texto não vão ao LLM e ficam com status 'error' ('Texto ausente')."

    with pytest.warns(UserWarning, match=f"^{re.escape(mensagem)}$"):
        _warn_missing_texts(df, "texto", [0, 1, 2, 3])


def test_erro_gravado_como_nan_le_como_none():
    df = pd.DataFrame({"_error_details": [math.nan, "falhou"]})

    assert [_recorded_error(df, 0), _recorded_error(df, 1)] == [None, "falhou"]


def test_falha_do_callback_vira_aviso_com_a_linha_e_o_erro():
    def callback(_resultado: RowResult) -> None:
        msg = "quebrou"
        raise ValueError(msg)

    resultado = RowResult(
        index="x", position=0, success=True, error=None, values={}, elapsed_seconds=0.0
    )
    mensagem = "on_row_done falhou na linha x: ValueError: quebrou"

    with pytest.warns(UserWarning, match=f"^{re.escape(mensagem)}$"):
        _notify_row_done(callback, resultado)


def test_handler_de_reserva_entra_uma_vez_so():
    # Fora do registro global de loggers, o teste não deixa handler para trás.
    logger = logging.Logger("teste-reserva")  # noqa: LOG001 (ver o comentário acima)
    outro = logging.NullHandler()
    logger.addHandler(outro)

    _install_fallback_handler(logger)
    _install_fallback_handler(logger)

    assert logger.handlers[0] is outro
    assert len(logger.handlers) == 2
    assert isinstance(logger.handlers[1], _FallbackHandler)


def _config(busca: SearchConfig | None) -> LLMConfig:
    return LLMConfig(
        model="m",
        provider="openai",
        api_key=None,
        max_retries=1,
        base_delay=0.0,
        max_delay=0.0,
        rate_limit_delay=0.0,
        search_config=busca,
    )


_BACKEND = ProviderBackend(label="langchain", invoke=lambda _texto: {})


@pytest.mark.parametrize(
    ("tipo", "busca", "rotulo"),
    [
        (ORIGINAL_TYPE_POLARS_DF, None, "Processando [polars→pandas+langchain] [2 workers]"),
        (ORIGINAL_TYPE_PANDAS_DF, None, "Processando [pandas+langchain] [2 workers]"),
        (ORIGINAL_TYPE_LIST, None, "Processando [list+langchain] [2 workers]"),
        (
            ORIGINAL_TYPE_PANDAS_DF,
            SearchConfig(enabled=False),
            "Processando [pandas+langchain] [2 workers]",
        ),
        (
            ORIGINAL_TYPE_PANDAS_DF,
            SearchConfig(enabled=True),
            "Processando [pandas+langchain+search] [2 workers]",
        ),
    ],
)
def test_rotulo_da_barra_mostra_entrada_backend_e_busca(tipo, busca, rotulo):
    descricao = _progress_description(
        _config(busca), _BACKEND, ConversionInfo(original_type=tipo), " [2 workers]"
    )

    assert descricao == rotulo


@pytest.mark.parametrize(
    ("colunas", "ja_processadas", "sufixo"),
    [
        (["a", "b"], 1, " (reprocessando: a, b)"),
        (None, 1, " (resumindo de 1/4)"),
        (None, 0, ""),
    ],
)
def test_sufixo_da_barra_mostra_reprocessamento_ou_retomada(colunas, ja_processadas, sufixo):
    df = pd.DataFrame({"texto": ["a", "b", "c", "d"]})

    assert _progress_suffix(df, colunas, ja_processadas) == sufixo
