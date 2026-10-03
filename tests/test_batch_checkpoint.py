"""Testes para batch_size + checkpoint_path (issue #92)."""

import importlib.util
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from pydantic import BaseModel

from dataframeit.core import _Checkpoint, _save_checkpoint, _SnapshotWriter, dataframeit

# Original guardado antes de os testes trocarem importlib.util.find_spec.
_FIND_SPEC = importlib.util.find_spec


class SimpleModel(BaseModel):
    campo1: str


def _mock_llm_factory():
    """Retorna (call_count_list, mock_fn). Cada chamada retorna payload válido."""
    counter = [0]

    def mock_llm(*args, **kwargs):
        counter[0] += 1
        return {
            "data": {"campo1": f"v{counter[0]}"},
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }

    return counter, mock_llm


def test_checkpoint_fires_on_multiples_sequential(tmp_path):
    """Sequencial: saves em 2, 4 e final em 5 (save final cobre a cauda)."""
    df = pd.DataFrame({"texto": ["a", "b", "c", "d", "e"]})
    ckpt = tmp_path / "ckpt.csv"

    observed_counts = []

    def spy(df_arg, path_arg, *_):
        observed_counts.append(
            (int((df_arg["_dataframeit_status"] == "processed").sum()), str(path_arg))
        )

    _, mock_llm = _mock_llm_factory()

    with (
        patch("dataframeit.core._save_checkpoint", side_effect=spy),
        patch("dataframeit.core.call_langchain", side_effect=mock_llm),
        patch("dataframeit.core.validate_provider_dependencies"),
    ):
        dataframeit(
            df,
            questions=SimpleModel,
            prompt="Teste {texto}",
            batch_size=2,
            checkpoint_path=ckpt,
        )

    processed_counts = [c for c, _ in observed_counts]
    assert processed_counts == [2, 4, 5]
    assert all(p == str(ckpt) for _, p in observed_counts)


def test_checkpoint_no_duplicate_final_save_sequential(tmp_path):
    """Sequencial: quando total é múltiplo de batch_size, nenhum save extra."""
    df = pd.DataFrame({"texto": ["a", "b", "c", "d"]})
    ckpt = tmp_path / "ckpt.csv"

    observed_counts = []

    def spy(df_arg, path_arg, *_):
        observed_counts.append(int((df_arg["_dataframeit_status"] == "processed").sum()))

    _, mock_llm = _mock_llm_factory()

    with (
        patch("dataframeit.core._save_checkpoint", side_effect=spy),
        patch("dataframeit.core.call_langchain", side_effect=mock_llm),
        patch("dataframeit.core.validate_provider_dependencies"),
    ):
        dataframeit(
            df,
            questions=SimpleModel,
            prompt="Teste {texto}",
            batch_size=2,
            checkpoint_path=ckpt,
        )

    assert observed_counts == [2, 4]


def test_checkpoint_fires_on_multiples_parallel(tmp_path):
    """Paralelo: contador monotônico, 3 saves incrementais + save final com 10."""
    df = pd.DataFrame({"texto": [f"linha{i}" for i in range(10)]})
    ckpt = tmp_path / "ckpt.csv"

    observed_counts = []

    def spy(df_arg, path_arg, *_):
        observed_counts.append(int((df_arg["_dataframeit_status"] == "processed").sum()))

    _, mock_llm = _mock_llm_factory()

    with (
        patch("dataframeit.core._save_checkpoint", side_effect=spy),
        patch("dataframeit.core.call_langchain", side_effect=mock_llm),
        patch("dataframeit.core.validate_provider_dependencies"),
    ):
        dataframeit(
            df,
            questions=SimpleModel,
            prompt="Teste {texto}",
            parallel_requests=4,
            batch_size=3,
            checkpoint_path=ckpt,
        )

    assert len(observed_counts) == 4
    assert observed_counts == sorted(observed_counts)
    assert observed_counts[-1] == 10


def test_checkpoint_no_duplicate_final_save_parallel(tmp_path):
    """Paralelo: quando total é múltiplo de batch_size, nenhum save extra."""
    df = pd.DataFrame({"texto": [f"linha{i}" for i in range(9)]})
    ckpt = tmp_path / "ckpt.csv"

    observed_counts = []

    def spy(df_arg, path_arg, *_):
        observed_counts.append(int((df_arg["_dataframeit_status"] == "processed").sum()))

    _, mock_llm = _mock_llm_factory()

    with (
        patch("dataframeit.core._save_checkpoint", side_effect=spy),
        patch("dataframeit.core.call_langchain", side_effect=mock_llm),
        patch("dataframeit.core.validate_provider_dependencies"),
    ):
        dataframeit(
            df,
            questions=SimpleModel,
            prompt="Teste {texto}",
            parallel_requests=3,
            batch_size=3,
            checkpoint_path=ckpt,
        )

    assert len(observed_counts) == 3
    assert observed_counts[-1] == 9


def test_missing_openpyxl_rejected_early(tmp_path):
    """Falta de openpyxl para .xlsx falha na validação, não após N linhas."""
    df = pd.DataFrame({"texto": ["a", "b"]})

    def fake_find_spec(name):
        if name == "openpyxl":
            return None
        return _FIND_SPEC(name)

    with (
        patch("dataframeit.core.call_langchain") as mock_llm,
        patch("dataframeit.core.validate_provider_dependencies"),
        patch("importlib.util.find_spec", side_effect=fake_find_spec),
    ):
        with pytest.raises(ImportError, match="openpyxl"):
            dataframeit(
                df,
                questions=SimpleModel,
                prompt="Teste {texto}",
                batch_size=1,
                checkpoint_path=tmp_path / "x.xlsx",
            )
        mock_llm.assert_not_called()


def test_missing_pyarrow_rejected_early(tmp_path):
    """Falta de pyarrow para .parquet falha na validação, não após N linhas."""
    df = pd.DataFrame({"texto": ["a", "b"]})

    def fake_find_spec(name):
        if name == "pyarrow":
            return None
        return _FIND_SPEC(name)

    with (
        patch("dataframeit.core.call_langchain") as mock_llm,
        patch("dataframeit.core.validate_provider_dependencies"),
        patch("importlib.util.find_spec", side_effect=fake_find_spec),
    ):
        with pytest.raises(ImportError, match="pyarrow"):
            dataframeit(
                df,
                questions=SimpleModel,
                prompt="Teste {texto}",
                batch_size=1,
                checkpoint_path=tmp_path / "x.parquet",
            )
        mock_llm.assert_not_called()


def test_no_checkpoint_when_params_none(tmp_path):
    """Sem batch_size/checkpoint_path: nenhum save é disparado."""
    df = pd.DataFrame({"texto": ["a", "b", "c"]})
    _, mock_llm = _mock_llm_factory()

    with (
        patch("dataframeit.core._save_checkpoint") as mock_save,
        patch("dataframeit.core.call_langchain", side_effect=mock_llm),
        patch("dataframeit.core.validate_provider_dependencies"),
    ):
        dataframeit(df, questions=SimpleModel, prompt="Teste {texto}")

    mock_save.assert_not_called()


def test_validation_batch_size_without_path():
    df = pd.DataFrame({"texto": ["a"]})
    with (
        patch("dataframeit.core.validate_provider_dependencies"),
        pytest.raises(ValueError, match="devem ser usados juntos"),
    ):
        dataframeit(df, questions=SimpleModel, prompt="Teste {texto}", batch_size=10)


def test_validation_path_without_batch_size(tmp_path):
    df = pd.DataFrame({"texto": ["a"]})
    with (
        patch("dataframeit.core.validate_provider_dependencies"),
        pytest.raises(ValueError, match="devem ser usados juntos"),
    ):
        dataframeit(
            df,
            questions=SimpleModel,
            prompt="Teste {texto}",
            checkpoint_path=tmp_path / "x.csv",
        )


@pytest.mark.parametrize("bad_value", [0, -1, 1.5, "10"])
def test_validation_invalid_batch_size(bad_value, tmp_path):
    df = pd.DataFrame({"texto": ["a"]})
    with (
        patch("dataframeit.core.validate_provider_dependencies"),
        pytest.raises(ValueError, match="batch_size deve ser int"),
    ):
        dataframeit(
            df,
            questions=SimpleModel,
            prompt="Teste {texto}",
            batch_size=bad_value,
            checkpoint_path=tmp_path / "x.csv",
        )


def test_unsupported_extension_rejected_early(tmp_path):
    """Extensão inválida falha na validação antes de processar linhas."""
    df = pd.DataFrame({"texto": ["a", "b"]})
    with (
        patch("dataframeit.core.call_langchain") as mock_llm,
        patch("dataframeit.core.validate_provider_dependencies"),
    ):
        with pytest.raises(ValueError, match="Extensão"):
            dataframeit(
                df,
                questions=SimpleModel,
                prompt="Teste {texto}",
                batch_size=1,
                checkpoint_path=tmp_path / "x.txt",
            )
        mock_llm.assert_not_called()


def test_resume_after_simulated_crash(tmp_path):
    """Simula crash após 1º checkpoint; recarrega + resume deve processar só o resto."""
    df = pd.DataFrame({"texto": [f"linha{i}" for i in range(10)]})
    ckpt = tmp_path / "ckpt.csv"

    total_calls = [0]

    def mock_llm(*args, **kwargs):
        total_calls[0] += 1
        # Após 4 chamadas (que garante 2 checkpoints com batch_size=2), crash.
        if total_calls[0] > 4:
            msg = "simulated kill"
            raise SystemExit(msg)
        return {
            "data": {"campo1": f"v{total_calls[0]}"},
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }

    with (
        patch("dataframeit.core.call_langchain", side_effect=mock_llm),
        patch("dataframeit.core.validate_provider_dependencies"),
        pytest.raises(SystemExit),
    ):
        dataframeit(
            df,
            questions=SimpleModel,
            prompt="Teste {texto}",
            batch_size=2,
            checkpoint_path=ckpt,
        )

    assert ckpt.exists(), "1º checkpoint deve estar persistido após o crash"
    loaded = pd.read_csv(ckpt)
    processed_at_crash = int((loaded["_dataframeit_status"] == "processed").sum())
    assert processed_at_crash >= 2

    calls_before_resume = total_calls[0]

    def mock_llm_ok(*args, **kwargs):
        total_calls[0] += 1
        return {
            "data": {"campo1": f"v{total_calls[0]}"},
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }

    with (
        patch("dataframeit.core.call_langchain", side_effect=mock_llm_ok),
        patch("dataframeit.core.validate_provider_dependencies"),
    ):
        final = dataframeit(
            loaded,
            questions=SimpleModel,
            prompt="Teste {texto}",
            resume=True,
            batch_size=2,
            checkpoint_path=ckpt,
        )

    resume_calls = total_calls[0] - calls_before_resume
    assert resume_calls == 10 - processed_at_crash
    assert final["campo1"].notna().all()


def test_save_checkpoint_is_atomic(tmp_path):
    """Após save bem-sucedido, arquivo .tmp não existe."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    path = tmp_path / "out.csv"
    _save_checkpoint(df, path)
    assert path.exists()
    assert not path.with_name(path.name + ".tmp").exists()


def test_save_checkpoint_csv_roundtrip(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    path = tmp_path / "out.csv"
    _save_checkpoint(df, path)
    loaded = pd.read_csv(path)
    assert loaded["a"].tolist() == [1, 2]
    assert loaded["b"].tolist() == ["x", "y"]


def test_save_checkpoint_rejects_unsupported_extension(tmp_path):
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match="Extensão"):
        _save_checkpoint(df, tmp_path / "out.json")


def test_checkpoint_paralelo_serializa_gravacoes_em_ordem(tmp_path):
    """Duas gravações simultâneas disputavam o mesmo .tmp, e o FileNotFoundError
    do os.replace regravava como 'error' uma linha já processada."""

    df = pd.DataFrame({"texto": [f"t{i}" for i in range(24)]})
    ckpt = tmp_path / "ckpt.csv"
    ativas = [0]
    max_ativas = [0]
    processadas_por_gravacao = []
    trava = threading.Lock()

    def gravacao_lenta(df_arg, path_arg, *_):
        with trava:
            ativas[0] += 1
            max_ativas[0] = max(max_ativas[0], ativas[0])
        time.sleep(0.02)
        processadas_por_gravacao.append(int((df_arg["_dataframeit_status"] == "processed").sum()))
        with trava:
            ativas[0] -= 1

    def llm_lento(*args, **kwargs):
        time.sleep(0.005)
        return {"data": {"campo1": "ok"}, "usage": None}

    with (
        patch("dataframeit.core.call_langchain", side_effect=llm_lento),
        patch("dataframeit.core.validate_provider_dependencies"),
        patch("dataframeit.core._save_checkpoint", side_effect=gravacao_lenta),
    ):
        resultado = dataframeit(
            df,
            SimpleModel,
            "p {texto}",
            parallel_requests=6,
            batch_size=1,
            checkpoint_path=str(ckpt),
            track_tokens=False,
        )

    assert max_ativas[0] == 1
    assert processadas_por_gravacao == sorted(processadas_por_gravacao)
    assert resultado["campo1"].tolist() == ["ok"] * 24


def test_checkpoint_parquet_grava_o_resultado(tmp_path):
    pytest.importorskip("pyarrow")
    ckpt = tmp_path / "ckpt.parquet"
    _, mock_llm = _mock_llm_factory()

    with (
        patch("dataframeit.core.call_langchain", side_effect=mock_llm),
        patch("dataframeit.core.validate_provider_dependencies"),
    ):
        dataframeit(
            pd.DataFrame({"texto": ["a", "b"]}),
            questions=SimpleModel,
            prompt="Teste {texto}",
            batch_size=1,
            checkpoint_path=ckpt,
        )

    salvo = pd.read_parquet(ckpt)
    assert salvo["campo1"].tolist() == ["v1", "v2"]
    assert salvo["_dataframeit_status"].tolist() == ["processed", "processed"]
    assert not ckpt.with_name(ckpt.name + ".tmp").exists()


def test_snapshot_atrasado_nao_sobrescreve_o_mais_novo(tmp_path):
    """No modo paralelo, o snapshot de rótulo menor que chega depois é descartado.

    A ordem em que as threads disputam a trava não se reproduz pela API; o
    _SnapshotWriter recebe os dois snapshots na ordem invertida diretamente.
    """
    gravados = []
    escritor = _SnapshotWriter(_Checkpoint(tmp_path / "ckpt.csv", batch_size=1))
    novo = pd.DataFrame({"campo1": ["a", "b"]})
    antigo = pd.DataFrame({"campo1": ["a"]})

    with patch("dataframeit.core._save_checkpoint", side_effect=lambda df, *_: gravados.append(df)):
        escritor.save(novo, 2)
        escritor.save(antigo, 1)
        escritor.save(novo, 2)

    assert len(gravados) == 1
    assert gravados[0] is novo
    assert escritor.last_saved == 2


def _crash_depois_de(n: int, total_calls: list):
    def mock_llm(*args, **kwargs):
        total_calls[0] += 1
        if total_calls[0] > n:
            msg = "simulated kill"
            raise SystemExit(msg)
        return {
            "data": {"campo1": f"v{total_calls[0]}"},
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }

    return mock_llm


def _roda(df, llm, ckpt, **kwargs):
    """Roda sobre uma cópia, como um processo novo que relê a mesma entrada.

    A biblioteca escreve as colunas no DataFrame recebido; sem a cópia, a segunda
    chamada receberia o frame já preenchido pela primeira.
    """
    df = df.copy()
    with (
        patch("dataframeit.core.call_langchain", side_effect=llm),
        patch("dataframeit.core.validate_provider_dependencies"),
    ):
        return dataframeit(
            df,
            questions=SimpleModel,
            prompt="Teste {texto}",
            batch_size=2,
            checkpoint_path=ckpt,
            **kwargs,
        )


@pytest.mark.parametrize("ext", [".csv", ".parquet"])
def test_retomada_rele_o_checkpoint_sem_carregar_a_mao(tmp_path, ext):
    """O mesmo DataFrame de entrada, rodado de novo, continua de onde o arquivo parou."""
    if ext == ".parquet":
        pytest.importorskip("pyarrow")
    df = pd.DataFrame({"texto": [f"linha{i}" for i in range(10)], "id": range(10)})
    ckpt = tmp_path / f"ckpt{ext}"
    total_calls = [0]
    with pytest.raises(SystemExit):
        _roda(df, _crash_depois_de(4, total_calls), ckpt)
    salvas = int(
        (pd.read_parquet(ckpt) if ext == ".parquet" else pd.read_csv(ckpt))["_dataframeit_status"]
        .eq("processed")
        .sum()
    )

    antes = total_calls[0]
    final = _roda(df, _crash_depois_de(100, total_calls), ckpt)

    assert total_calls[0] - antes == 10 - salvas
    assert final["campo1"].tolist()[:salvas] == [f"v{i}" for i in range(1, salvas + 1)]
    assert final["campo1"].notna().all()
    assert final["id"].tolist() == list(range(10))


def test_checkpoint_de_outra_entrada_avisa_e_recomeca(tmp_path):
    ckpt = tmp_path / "ckpt.csv"
    total_calls = [0]
    with pytest.raises(SystemExit):
        _roda(pd.DataFrame({"texto": ["a", "b", "c", "d"]}), _crash_depois_de(2, total_calls), ckpt)

    for outra in (["a", "X", "c", "d"], ["a", "b", "c"]):
        antes = total_calls[0]
        with pytest.warns(UserWarning, match="outra entrada; esta execução começa do zero"):
            final = _roda(pd.DataFrame({"texto": outra}), _crash_depois_de(100, total_calls), ckpt)
        assert total_calls[0] - antes == len(outra)
        assert final["campo1"].notna().all()


def test_checkpoint_sem_assinatura_avisa_e_recomeca(tmp_path):
    """Checkpoint gravado antes da assinatura existir, ou por outra ferramenta."""
    ckpt = tmp_path / "ckpt.csv"
    pd.DataFrame(
        {"texto": ["a"], "campo1": ["velho"], "_dataframeit_status": ["processed"]}
    ).to_csv(ckpt, index=False)
    total_calls = [0]

    with pytest.warns(UserWarning, match="não tem a assinatura"):
        final = _roda(pd.DataFrame({"texto": ["a"]}), _crash_depois_de(100, total_calls), ckpt)

    assert total_calls[0] == 1
    assert final["campo1"].tolist() == ["v1"]


def test_checkpoint_de_outro_prompt_ou_modelo_nao_e_reaproveitado(tmp_path):
    ckpt = tmp_path / "ckpt.csv"
    df = pd.DataFrame({"texto": ["a", "b", "c"]})
    _roda(df, _crash_depois_de(100, [0]), ckpt)

    total_calls = [0]
    with (
        patch("dataframeit.core.call_langchain", side_effect=_crash_depois_de(100, total_calls)),
        patch("dataframeit.core.validate_provider_dependencies"),
        pytest.warns(UserWarning, match="outra configuração"),
    ):
        dataframeit(
            df.copy(),
            questions=SimpleModel,
            prompt="Prompt reescrito: {texto}",
            model="outro-modelo",
            batch_size=2,
            checkpoint_path=ckpt,
        )

    assert total_calls[0] == 3


def test_checkpoint_concluido_da_mesma_execucao_e_reaproveitado(tmp_path):
    ckpt = tmp_path / "ckpt.csv"
    df = pd.DataFrame({"texto": ["a", "b"]})
    _roda(df, _crash_depois_de(100, [0]), ckpt)

    total_calls = [0]
    final = _roda(df, _crash_depois_de(100, total_calls), ckpt)

    assert total_calls[0] == 0
    assert final["campo1"].tolist() == ["v1", "v2"]


@pytest.mark.parametrize(
    "textos",
    [
        ["001", "002", "003", "004"],
        ["NA", "b", "c", "d"],
        ["null", "b", "c", "d"],
        ["1.50", "2", "3", "4"],
    ],
)
def test_texto_que_o_csv_altera_nao_impede_a_retomada(tmp_path, textos):
    ckpt = tmp_path / "ckpt.csv"
    df = pd.DataFrame({"texto": textos})
    total_calls = [0]
    with pytest.raises(SystemExit):
        _roda(df, _crash_depois_de(2, total_calls), ckpt)

    antes = total_calls[0]
    final = _roda(df, _crash_depois_de(100, total_calls), ckpt)

    assert total_calls[0] - antes == 2
    assert final["texto"].tolist() == textos
    assert final["campo1"].tolist()[:2] == ["v1", "v2"]


def test_texto_com_igual_no_xlsx_nao_impede_a_retomada(tmp_path):
    pytest.importorskip("openpyxl")
    ckpt = tmp_path / "ckpt.xlsx"
    df = pd.DataFrame({"texto": ["=soma", "b", "c", "d"]})
    total_calls = [0]
    with pytest.raises(SystemExit):
        _roda(df, _crash_depois_de(2, total_calls), ckpt)

    antes = total_calls[0]
    _roda(df, _crash_depois_de(100, total_calls), ckpt)

    assert total_calls[0] - antes == 2


class ModeloOpcional(BaseModel):
    campo1: str | None = None


def test_campo_do_modelo_ja_presente_na_entrada_recebe_o_valor_do_checkpoint(tmp_path):
    """Fluxo de preencher campo faltante: a entrada já traz a coluna, vazia."""
    pytest.importorskip("pyarrow")
    ckpt = tmp_path / "ckpt.parquet"
    df = pd.DataFrame({"texto": ["a", "b", "c", "d"], "campo1": [None] * 4})
    total_calls = [0]
    with (
        patch("dataframeit.core.call_langchain", side_effect=_crash_depois_de(2, total_calls)),
        patch("dataframeit.core.validate_provider_dependencies"),
        pytest.raises(SystemExit),
    ):
        dataframeit(df.copy(), ModeloOpcional, "{texto}", batch_size=1, checkpoint_path=ckpt)

    with (
        patch("dataframeit.core.call_langchain", side_effect=_crash_depois_de(100, total_calls)),
        patch("dataframeit.core.validate_provider_dependencies"),
    ):
        final = dataframeit(
            df.copy(), ModeloOpcional, "{texto}", batch_size=1, checkpoint_path=ckpt
        )

    # As duas primeiras vêm do checkpoint; o contador segue da chamada que caiu.
    assert final["campo1"].tolist() == ["v1", "v2", "v4", "v5"]


def test_resume_false_ignora_o_checkpoint_existente(tmp_path):
    ckpt = tmp_path / "ckpt.csv"
    total_calls = [0]
    with pytest.raises(SystemExit):
        _roda(pd.DataFrame({"texto": ["a", "b", "c", "d"]}), _crash_depois_de(2, total_calls), ckpt)

    antes = total_calls[0]
    _roda(
        pd.DataFrame({"texto": ["a", "b", "c", "d"]}),
        _crash_depois_de(100, total_calls),
        ckpt,
        resume=False,
    )

    assert total_calls[0] - antes == 4


def test_texto_ausente_casa_com_texto_ausente(tmp_path):
    ckpt = tmp_path / "ckpt.csv"
    total_calls = [0]
    entrada = pd.DataFrame({"texto": ["a", None, "c"]})
    with (
        pytest.warns(UserWarning, match="sem texto não vão ao LLM"),
        pytest.raises(SystemExit),
    ):
        _roda(entrada, _crash_depois_de(1, total_calls), ckpt)

    final = _roda(entrada, _crash_depois_de(100, total_calls), ckpt)

    assert final["campo1"].iloc[0] == "v1"


def test_falha_ao_gravar_a_assinatura_vira_aviso(tmp_path):
    ckpt = tmp_path / "ckpt.csv"
    original = Path.write_text

    def falha_na_assinatura(self, *args, **kwargs):
        if self.name.endswith(".dataframeit.json.tmp"):
            msg = "disco cheio"
            raise OSError(msg)
        return original(self, *args, **kwargs)

    with (
        patch.object(Path, "write_text", falha_na_assinatura),
        pytest.warns(UserWarning, match="Falha ao gravar a assinatura.*disco cheio"),
    ):
        final = _roda(pd.DataFrame({"texto": ["a"]}), _crash_depois_de(100, [0]), ckpt)

    assert final["campo1"].tolist() == ["v1"]
    assert ckpt.exists()


def test_modelo_sem_json_schema_ainda_assina_o_checkpoint(tmp_path):
    """Sem JSON Schema, a estrutura dos campos entra na assinatura."""

    class SemSchema(BaseModel):
        campo1: str

        @classmethod
        def model_json_schema(cls, *args, **kwargs):
            msg = "sem schema"
            raise TypeError(msg)

    ckpt = tmp_path / "ckpt.csv"
    df = pd.DataFrame({"texto": ["a", "b"]})
    for _ in range(2):
        total_calls = [0]
        with (
            patch(
                "dataframeit.core.call_langchain", side_effect=_crash_depois_de(100, total_calls)
            ),
            patch("dataframeit.core.validate_provider_dependencies"),
        ):
            dataframeit(df.copy(), SemSchema, "{texto}", batch_size=1, checkpoint_path=ckpt)

    assert total_calls[0] == 0
