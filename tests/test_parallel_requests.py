"""Testes para a funcionalidade de requisições paralelas."""

import inspect
import threading
import warnings
from unittest.mock import patch

import pandas as pd
import pytest
from pydantic import BaseModel

from dataframeit import ProviderOverloadedError, core
from dataframeit.core import dataframeit
from dataframeit.errors import is_rate_limit_error


class SimpleModel(BaseModel):
    campo1: str
    campo2: str


def test_parallel_requests_parameter_exists():
    """Testa que o parâmetro parallel_requests existe e tem default=1."""
    sig = inspect.signature(dataframeit)
    param = sig.parameters.get("parallel_requests")
    assert param is not None
    assert param.default == 1


def test_parallel_requests_1_uses_sequential():
    """Testa que parallel_requests=1 usa processamento sequencial."""
    df = pd.DataFrame({"texto": ["a"]})

    with (
        patch("dataframeit.core._process_rows") as mock_seq,
        patch("dataframeit.core._process_rows_parallel") as mock_par,
    ):
        mock_seq.return_value = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        with patch("dataframeit.core.validate_provider_dependencies"):
            dataframeit(
                df,
                questions=SimpleModel,
                prompt="Teste {texto}",
                parallel_requests=1,  # Default
            )

            # Deve usar processamento sequencial
            mock_seq.assert_called_once()
            mock_par.assert_not_called()


def test_parallel_requests_gt1_uses_parallel():
    """Testa que parallel_requests > 1 usa processamento paralelo."""
    df = pd.DataFrame({"texto": ["a"]})

    with (
        patch("dataframeit.core._process_rows") as mock_seq,
        patch("dataframeit.core._process_rows_parallel") as mock_par,
    ):
        mock_par.return_value = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        with patch("dataframeit.core.validate_provider_dependencies"):
            dataframeit(
                df,
                questions=SimpleModel,
                prompt="Teste {texto}",
                parallel_requests=5,
            )

            # Deve usar processamento paralelo
            mock_par.assert_called_once()
            mock_seq.assert_not_called()


def test_parallel_processes_all_rows():
    """Testa que processamento paralelo processa todas as linhas."""
    df = pd.DataFrame({"texto": ["a", "b", "c", "d", "e"]})

    call_count = 0

    def mock_llm(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return {
            "data": {"campo1": f"v{call_count}", "campo2": f"x{call_count}"},
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        }

    with (
        patch("dataframeit.core.call_langchain", side_effect=mock_llm),
        patch("dataframeit.core.validate_provider_dependencies"),
    ):
        result = dataframeit(
            df,
            questions=SimpleModel,
            prompt="Teste {texto}",
            parallel_requests=3,
        )

        # Todas as 5 linhas devem ter sido processadas
        assert call_count == 5
        # Verificar que todas as linhas têm valores (não None)
        assert result["campo1"].notna().all()
        assert result["campo2"].notna().all()


@pytest.mark.parametrize("parallel_requests", [1, 2])
def test_tracks_tokens_with_stable_schema(parallel_requests):
    """Testa o mesmo schema de telemetria nos caminhos sequencial e paralelo."""
    df = pd.DataFrame({"texto": ["a", "b", "c"]})

    def mock_llm(*args, **kwargs):
        return {
            "data": {"campo1": "v", "campo2": "x"},
            "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        }

    with (
        patch("dataframeit.core.call_langchain", side_effect=mock_llm),
        patch("dataframeit.core.validate_provider_dependencies"),
    ):
        result = dataframeit(
            df,
            questions=SimpleModel,
            prompt="Teste {texto}",
            parallel_requests=parallel_requests,
            track_tokens=True,
        )

        # Verificar colunas de tokens
        assert "_input_tokens" in result.columns
        assert "_cached_input_tokens" in result.columns
        assert "_output_tokens" in result.columns
        assert "_reasoning_tokens" in result.columns
        assert "_total_tokens" not in result.columns

        # Cada linha deve ter os tokens registrados
        assert result["_input_tokens"].tolist() == [100, 100, 100]
        assert result["_cached_input_tokens"].tolist() == [0, 0, 0]
        assert result["_output_tokens"].tolist() == [50, 50, 50]
        assert result["_reasoning_tokens"].tolist() == [0, 0, 0]


def test_is_rate_limit_error_detects_429():
    """Testa que is_rate_limit_error detecta erros de rate limit."""
    # Erro 429
    error_429 = Exception("Error 429: Too many requests")
    assert is_rate_limit_error(error_429) is True

    # Rate limit explícito
    class RateLimitError(Exception):
        pass

    rate_error = RateLimitError("Rate limit exceeded")
    assert is_rate_limit_error(rate_error) is True

    # Resource exhausted (Google)
    resource_error = Exception("ResourceExhausted: Quota exceeded")
    assert is_rate_limit_error(resource_error) is True

    # Erro normal (não é rate limit)
    normal_error = ValueError("Invalid argument")
    assert is_rate_limit_error(normal_error) is False


def test_parallel_handles_errors_gracefully():
    """Testa que processamento paralelo lida com erros sem quebrar."""
    df = pd.DataFrame({"texto": ["a", "b", "c"]})

    call_count = 0

    def mock_llm_with_error(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            msg = "Erro no processamento"
            raise ValueError(msg)
        return {
            "data": {"campo1": f"v{call_count}", "campo2": f"x{call_count}"},
            "usage": {},
        }

    with (
        patch("dataframeit.core.call_langchain", side_effect=mock_llm_with_error),
        patch("dataframeit.core.validate_provider_dependencies"),
        warnings.catch_warnings(record=True),
    ):
        warnings.simplefilter("always")
        result = dataframeit(
            df,
            questions=SimpleModel,
            prompt="Teste {texto}",
            parallel_requests=2,
        )

        # Linhas 1 e 3 processadas, linha 2 com erro
        statuses = result["_dataframeit_status"].tolist()
        assert statuses.count("processed") == 2
        assert statuses.count("error") == 1


def test_parallel_respects_resume():
    """Testa que processamento paralelo respeita resume=True."""
    df = pd.DataFrame(
        {
            "texto": ["a", "b", "c"],
            "campo1": ["old1", None, None],
            "campo2": ["old_a", None, None],
            "_dataframeit_status": ["processed", None, None],
            "_error_details": [None, None, None],
        }
    )

    call_count = 0

    def mock_llm(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return {
            "data": {"campo1": f"new{call_count}", "campo2": f"new_{call_count}"},
            "usage": {},
        }

    with (
        patch("dataframeit.core.call_langchain", side_effect=mock_llm),
        patch("dataframeit.core.validate_provider_dependencies"),
    ):
        result = dataframeit(
            df,
            questions=SimpleModel,
            prompt="Teste {texto}",
            parallel_requests=2,
            resume=True,
        )

        # Apenas 2 linhas não processadas devem ser chamadas
        assert call_count == 2
        # Primeira linha mantém valores antigos
        assert result["campo1"].iloc[0] == "old1"


def test_parallel_with_reprocess_columns():
    """Testa que processamento paralelo funciona com reprocess_columns."""
    df = pd.DataFrame(
        {
            "texto": ["a", "b"],
            "campo1": ["original1", "original2"],
            "campo2": ["original_a", "original_b"],
            "_dataframeit_status": ["processed", "processed"],
            "_error_details": [None, None],
        }
    )

    def mock_llm(*args, **kwargs):
        return {
            "data": {"campo1": "novo_valor", "campo2": "novo_valor_2"},
            "usage": {},
        }

    with (
        patch("dataframeit.core.call_langchain", side_effect=mock_llm),
        patch("dataframeit.core.validate_provider_dependencies"),
    ):
        result = dataframeit(
            df,
            questions=SimpleModel,
            prompt="Teste {texto}",
            parallel_requests=2,
            reprocess_columns=["campo1"],
        )

        # campo1 deve ter sido atualizado
        assert result["campo1"].tolist() == ["novo_valor", "novo_valor"]
        # campo2 deve manter os valores originais
        assert result["campo2"].tolist() == ["original_a", "original_b"]


# =============================================================================
# Redução de workers por rate limit e falha fora da linha
# =============================================================================


class _TimerFalso:
    """Registra o intervalo sem iniciar a thread que limparia o evento."""

    intervalos: list

    def __init__(self, intervalo, funcao):
        _TimerFalso.intervalos.append(intervalo)

    def start(self):
        pass


def test_rate_limit_reduz_workers_e_pausa_as_linhas_seguintes(monkeypatch):
    """'a' e 'c' batem no rate limit; só a primeira reduz, porque já resta um worker."""
    esperas = []
    _TimerFalso.intervalos = []
    monkeypatch.setattr(core.time, "sleep", esperas.append)
    monkeypatch.setattr(core.threading, "Timer", _TimerFalso)

    # 'b' roda no mesmo lote que 'a'; a espera garante que ele passe pela
    # checagem do evento antes que a falha de 'a' o acione.
    b_chamado = threading.Event()

    def llm(text, *args, **kwargs):
        if text == "b":
            b_chamado.set()
        if text == "a":
            b_chamado.wait(timeout=5)
        if text in {"a", "c"}:
            msg = "429 too many requests"
            raise ProviderOverloadedError(msg)
        return {"data": {"campo1": text, "campo2": text}, "usage": None}

    with (
        patch("dataframeit.core.call_langchain", side_effect=llm),
        patch("dataframeit.core.validate_provider_dependencies"),
        warnings.catch_warnings(record=True) as avisos,
    ):
        warnings.simplefilter("always")
        resultado = dataframeit(
            pd.DataFrame({"texto": ["a", "b", "c", "d"]}),
            questions=SimpleModel,
            prompt="{texto}",
            parallel_requests=2,
            track_tokens=False,
        )

    reducoes = [str(a.message) for a in avisos if "Rate limit detectado" in str(a.message)]
    assert reducoes == ["Rate limit detectado! Reduzindo workers de 2 para 1."]
    assert _TimerFalso.intervalos == [5.0]
    # 'c' e 'd' rodam depois da redução, com o evento de rate limit ainda ativo.
    assert esperas == [2.0, 2.0]
    assert resultado["_dataframeit_status"].tolist() == ["error", "processed", "error", "processed"]
    finais = [str(a.message) for a in avisos if "reduzidos por rate limit" in str(a.message)]
    assert finais == [
        (
            "Workers reduzidos por rate limit: de 2 para 1. Considere usar "
            "parallel_requests=1 para evitar rate limits."
        )
    ]


class _ErroIlegivel(Exception):
    """Erro cujo texto não pode ser montado, como o de um SDK com __str__ quebrado."""

    def __str__(self):
        msg = "sem texto"
        raise RuntimeError(msg)


@pytest.mark.parametrize("paralelo", [1, 2])
def test_falha_ao_registrar_o_erro_marca_a_linha_e_segue(paralelo):
    def llm(text, *args, **kwargs):
        if text == "b":
            raise _ErroIlegivel
        return {"data": {"campo1": text, "campo2": text}, "usage": None}

    with (
        patch("dataframeit.core.call_langchain", side_effect=llm),
        patch("dataframeit.core.validate_provider_dependencies"),
        pytest.warns(UserWarning, match="Falha ao processar linha 1"),
    ):
        resultado = dataframeit(
            pd.DataFrame({"texto": ["a", "b", "c"]}),
            questions=SimpleModel,
            prompt="{texto}",
            parallel_requests=paralelo,
            track_tokens=False,
        )

    assert resultado["_dataframeit_status"].tolist() == ["processed", "error", "processed"]
    assert resultado["campo1"].tolist()[0::2] == ["a", "c"]
    assert resultado["_error_details"].iloc[1] == (
        "[Falha ao registrar o erro] _ErroIlegivel: _ErroIlegivel()"
    )


def test_falha_fora_do_tratamento_da_linha_marca_a_linha_no_paralelo():
    def gravacao_que_falha(*args, **kwargs):
        msg = "gravação"
        raise RuntimeError(msg)

    with (
        patch(
            "dataframeit.core.call_langchain",
            side_effect=lambda text, *a, **k: {"data": {"campo1": text, "campo2": text}},
        ),
        patch("dataframeit.core.validate_provider_dependencies"),
        patch("dataframeit.core._record_missing_text", side_effect=gravacao_que_falha),
        pytest.warns(UserWarning, match="Falha ao processar linha 1"),
    ):
        resultado = dataframeit(
            pd.DataFrame({"texto": ["a", None, "c"]}),
            questions=SimpleModel,
            prompt="{texto}",
            parallel_requests=2,
            track_tokens=False,
        )

    assert resultado["_dataframeit_status"].tolist() == ["processed", "error", "processed"]
    assert resultado["_error_details"].iloc[1] == (
        "[Falha ao registrar o erro] RuntimeError: gravação"
    )


def test_texto_do_erro_sem_str_nem_repr_fica_so_com_o_tipo():
    class _ErroMudo(_ErroIlegivel):
        def __repr__(self):
            msg = "sem repr"
            raise RuntimeError(msg)

    assert core._error_text(_ErroIlegivel()) == "_ErroIlegivel: _ErroIlegivel()"
    assert core._error_text(_ErroMudo()) == "_ErroMudo"


def test_falha_ao_gravar_o_snapshot_nao_regrava_a_linha_ja_registrada(tmp_path):
    rotulos = []
    salvar = core._SnapshotWriter.save

    def save_que_falha_uma_vez(self, frame, label):
        rotulos.append(label)
        if len(rotulos) == 1:
            msg = "gravação do snapshot"
            raise RuntimeError(msg)
        salvar(self, frame, label)

    with (
        patch(
            "dataframeit.core.call_langchain",
            side_effect=lambda text, *a, **k: {"data": {"campo1": text, "campo2": text}},
        ),
        patch("dataframeit.core.validate_provider_dependencies"),
        patch.object(core._SnapshotWriter, "save", save_que_falha_uma_vez),
        pytest.warns(
            UserWarning,
            match=r"Erro inesperado no executor depois de registrar a linha \d, que mantém o status "
            r"gravado: RuntimeError: gravação",
        ),
    ):
        resultado = dataframeit(
            pd.DataFrame({"texto": ["a", "b", "c"]}),
            questions=SimpleModel,
            prompt="{texto}",
            parallel_requests=2,
            track_tokens=False,
            batch_size=1,
            checkpoint_path=tmp_path / "c.csv",
        )

    assert resultado["campo1"].tolist() == ["a", "b", "c"]
    assert "_dataframeit_status" not in resultado.columns
    assert sorted(rotulos) == [1, 2, 3]


@pytest.mark.parametrize("paralelo", [1, 2])
def test_falha_ao_registrar_o_erro_no_reprocessamento_avisa_que_manteve_os_valores(paralelo):
    def llm(text, *args, **kwargs):
        if text == "b":
            raise _ErroIlegivel
        return {"data": {"campo1": text}, "usage": None}

    df = pd.DataFrame(
        {
            "texto": ["a", "b"],
            "campo1": ["antigo", "antigo"],
            "campo2": ["x", "x"],
            "_dataframeit_status": ["processed", "processed"],
        }
    )
    with (
        patch("dataframeit.core.call_langchain", side_effect=llm),
        patch("dataframeit.core.validate_provider_dependencies"),
        pytest.warns(UserWarning, match="Falha ao processar linha 1"),
    ):
        resultado = dataframeit(
            df,
            questions=SimpleModel,
            prompt="{texto}",
            parallel_requests=paralelo,
            track_tokens=False,
            reprocess_columns=["campo1"],
        )

    assert resultado["campo1"].tolist() == ["a", "antigo"]
    assert resultado["_error_details"].iloc[1] == (
        "[Falha ao registrar o erro] _ErroIlegivel: _ErroIlegivel()" + core._KEPT_VALUES_NOTE
    )
