"""Forma das tabelas e dos arrays na assinatura, e a assinatura gravada ao lado do checkpoint.

A forma de cada DataFrame, Series, Index e ndarray é comparada inteira, com dados
pequenos e fixos. Um tipo (`int`, `str`) posto numa célula, num rótulo, num nome
ou nos `attrs` mostra que a parte passa pela forma canônica, que o escreve pelo
nome, e não fica como veio.
"""

import hashlib
import json
import re

import numpy as np
import pandas as pd
import pytest
from pandas.core.util.hashing import hash_pandas_object
from pydantic import BaseModel

from dataframeit.core import (
    _array_form,
    _Checkpoint,
    _checkpoint_mismatch,
    _column_cells,
    _Descent,
    _extended,
    _read_checkpoint_signature,
    _signature_path,
    _stable,
    _table_form,
    _text_hashes,
    _write_checkpoint_signature,
)

_LONG_FLOAT = np.dtype(np.longdouble).itemsize > np.dtype(np.float64).itemsize
_LONG_COMPLEX = np.dtype(np.clongdouble).itemsize > np.dtype(np.complex128).itemsize


def _hash(column: pd.Series) -> str:
    """O hash das células da coluna, sem o índice, como a assinatura o calcula."""
    cells = hash_pandas_object(column, index=False).to_numpy()
    return hashlib.sha256(cells.tobytes()).hexdigest()


def _bytes_hash(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes()).hexdigest()


def _sha16(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def test_forma_do_dataframe_traz_indice_rotulos_attrs_e_cada_coluna_na_forma_canonica():
    df = pd.DataFrame({"a": [1, 2], int: [str, float]}, index=pd.Index([10, 20], name=bool))
    df.attrs = {"tipo": complex}

    assert _stable(df, _Descent()) == {
        "table": "DataFrame",
        "index": {
            "shape": [2],
            "names": ["bool"],
            "dtype": "int64",
            "cells": _hash(pd.Series([10, 20])),
        },
        "labels": {"shape": [2], "names": [None], "dtype": "object", "cells": ["a", "int"]},
        "attrs": {"tipo": "complex"},
        "columns": [
            ["a", "int64", _hash(pd.Series([1, 2]))],
            ["int", "object", ["str", "float"]],
        ],
    }


def test_forma_da_series_categorica_traz_as_categorias_na_ordem_e_o_ordered():
    categorias = pd.Categorical([int, str], categories=[str, int], ordered=True)
    serie = pd.Series(categorias, name=float, index=pd.Index([int, str]))

    assert _table_form(serie, _Descent()) == {
        "table": "Series",
        "index": {"shape": [2], "names": [None], "dtype": "object", "cells": ["int", "str"]},
        "labels": None,
        "attrs": {},
        "columns": [
            [
                "float",
                {
                    "category": {
                        "shape": [2],
                        "names": [None],
                        "dtype": "object",
                        "cells": ["str", "int"],
                    },
                    "ordered": True,
                },
                _hash(pd.Series([1, 0], dtype="int8")),
            ]
        ],
    }


def test_categoria_ordenada_ou_nao_da_formas_diferentes():
    ordenada = pd.Series(pd.Categorical(["a", "b"], ordered=True))
    solta = pd.Series(pd.Categorical(["a", "b"], ordered=False))

    assert _stable(ordenada, _Descent()) != _stable(solta, _Descent())


def test_dataframes_de_mesmo_formato_e_conteudo_diferente_dao_formas_diferentes():
    um = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
    outro = pd.DataFrame({"a": [1, 2], "b": [3.0, 5.0]})

    assert _stable(um, _Descent()) != _stable(outro, _Descent())


def test_forma_do_index_traz_os_nomes_e_as_celulas_na_forma_canonica():
    assert _stable(pd.Index([int, str], name=float), _Descent()) == {
        "shape": [2],
        "names": ["float"],
        "dtype": "object",
        "cells": ["int", "str"],
    }


def test_forma_do_index_categorico_traz_as_categorias_na_forma_canonica():
    assert _array_form(pd.CategoricalIndex([int]), _Descent()) == {
        "shape": [1],
        "names": [None],
        "dtype": {
            "category": {"shape": [1], "names": [None], "dtype": "object", "cells": ["int"]},
            "ordered": False,
        },
        "cells": _hash(pd.Series([0], dtype="int8")),
    }


def test_forma_do_array_do_pandas_traz_as_celulas_na_forma_canonica():
    assert _stable(pd.array([int, str], dtype=object), _Descent()) == {
        "shape": [2],
        "names": None,
        "dtype": "object",
        "cells": ["int", "str"],
    }


def test_forma_do_ndarray_de_objetos_traz_as_celulas_na_forma_canonica():
    assert _stable(np.array([[int], [str]], dtype=object), _Descent()) == {
        "shape": [2, 1],
        "names": None,
        "dtype": "object",
        "cells": ["int", "str"],
    }


def test_forma_do_ndarray_de_tipo_composto_traz_as_celulas_na_forma_canonica():
    composto = np.array([(int, 1)], dtype=[("a", object), ("b", np.int64)])

    assert _stable(composto, _Descent()) == {
        "shape": [1],
        "names": None,
        "dtype": "[('a', 'O'), ('b', '<i8')]",
        "cells": [["int", 1]],
    }


def test_forma_do_ndarray_de_tipo_composto_so_de_numeros_vem_pelas_celulas():
    composto = np.array([(1, 2.5)], dtype=[("a", np.int32), ("b", np.float64)])

    assert _array_form(composto, _Descent()) == {
        "shape": [1],
        "names": None,
        "dtype": "[('a', '<i4'), ('b', '<f8')]",
        "cells": [[1, 2.5]],
    }


@pytest.mark.parametrize(
    "array",
    [
        np.array([[1.5, 2.0], [3.0, 4.0]]),
        np.array([1 + 2j, 3 - 1j]),
        np.array([1 + 2j], dtype=np.complex64),
        np.array([True, False]),
        np.array([7, 8], dtype=np.int16),
    ],
    ids=["float64", "complex128", "complex64", "bool", "int16"],
)
def test_forma_do_ndarray_de_numeros_vem_pelos_bytes(array):
    assert _array_form(array, _Descent()) == {
        "shape": list(array.shape),
        "names": None,
        "dtype": str(array.dtype),
        "cells": _bytes_hash(array),
    }


@pytest.mark.skipif(not _LONG_FLOAT, reason="o longdouble desta plataforma é o float64")
def test_forma_do_ndarray_de_float_estendido_vem_pelas_celulas():
    array = np.array([1.5, 2.25], dtype=np.longdouble)

    assert _array_form(array, _Descent()) == {
        "shape": [2],
        "names": None,
        "dtype": str(array.dtype),
        "cells": [1.5, 2.25],
    }


@pytest.mark.skipif(not _LONG_COMPLEX, reason="o clongdouble desta plataforma é o complex128")
def test_forma_do_ndarray_de_complex_estendido_vem_pelas_celulas():
    array = np.array([1 + 2j], dtype=np.clongdouble)

    assert _array_form(array, _Descent()) == {
        "shape": [1],
        "names": None,
        "dtype": str(array.dtype),
        "cells": [1 + 2j],
    }


@pytest.mark.parametrize(
    "dtype",
    [np.float16, np.float32, np.float64, np.complex64, np.complex128, np.int64, np.bool_, "U20"],
)
def test_tipo_que_o_hash_do_pandas_le_nao_e_estendido(dtype):
    assert _extended(np.dtype(dtype)) is False


@pytest.mark.skipif(not _LONG_FLOAT, reason="o longdouble desta plataforma é o float64")
def test_longdouble_maior_que_o_float64_e_estendido():
    assert _extended(np.dtype(np.longdouble)) is True


@pytest.mark.skipif(not _LONG_COMPLEX, reason="o clongdouble desta plataforma é o complex128")
def test_clongdouble_maior_que_o_complex128_e_estendido():
    assert _extended(np.dtype(np.clongdouble)) is True


def test_tipo_do_pandas_sem_itemsize_nao_e_estendido():
    assert _extended(pd.SparseDtype(np.float64)) is False
    assert _extended(pd.StringDtype()) is False


@pytest.mark.parametrize(
    "column",
    [
        pd.Series([True, False]),
        pd.Series([1, -2]),
        pd.Series([1, 2], dtype=np.uint8),
        pd.Series(pd.to_timedelta(["1D", "2h"])),
        pd.Series(pd.to_datetime(["2024-01-01", "2024-06-30"])),
        pd.Series([1.5, np.nan]),
        pd.Series([1 + 2j, 3j]),
        pd.Series(["a", "b"], dtype=object),
        pd.Series(pd.arrays.SparseArray([1.0, np.nan, 2.0])),
    ],
    ids=["bool", "int", "uint", "timedelta", "datetime", "float", "complex", "texto", "esparsa"],
)
def test_coluna_de_numero_data_ou_texto_vem_pelo_hash_do_pandas(column):
    assert _column_cells(column, _Descent()) == _hash(column)


def test_hash_da_coluna_nao_depende_do_indice():
    assert _column_cells(pd.Series([1, 2], index=[5, 6]), _Descent()) == _column_cells(
        pd.Series([1, 2]), _Descent()
    )


def test_coluna_de_outros_objetos_vem_pelas_celulas_na_forma_canonica():
    assert _column_cells(pd.Series([int, None, "a"]), _Descent()) == ["int", None, "a"]


def test_coluna_de_texto_com_none_ou_nan_vem_pelas_celulas_e_os_separa():
    com_none = _column_cells(pd.Series(["a", None], dtype=object), _Descent())
    com_nan = _column_cells(pd.Series(["a", np.nan], dtype=object), _Descent())

    assert com_none == ["a", None]
    assert isinstance(com_nan, list)
    assert com_nan[0] == "a"
    assert np.isnan(com_nan[1])


@pytest.mark.skipif(not _LONG_FLOAT, reason="o longdouble desta plataforma é o float64")
def test_coluna_de_float_estendido_vem_pelas_celulas():
    column = pd.Series(np.array([1.5, 2.25], dtype=np.longdouble))

    assert _column_cells(column, _Descent()) == [1.5, 2.25]


def test_hash_do_texto_tem_16_caracteres_e_o_texto_ausente_fica_none():
    assert _text_hashes(pd.Series(["a", None, "b"])) == [_sha16("a"), None, _sha16("b")]


class _Resposta(BaseModel):
    resposta: str


_STATUS = "_dataframeit_status"
_META = {"signature": "assinatura", "texts": ["h1", "h2"]}


def _df():
    return pd.DataFrame({"texto": ["um", "dois"]})


def _write_checkpoint(path, df=None, meta=_META):
    gravado = _df() if df is None else df
    gravado.assign(resposta=["x", "y"][: len(gravado)], **{_STATUS: "processed"}).to_csv(
        path, index=False
    )
    _write_checkpoint_signature(path, meta)


def test_assinatura_gravada_traz_o_meta_e_o_hash_do_checkpoint(tmp_path):
    path = tmp_path / "ckpt.csv"
    path.write_text("texto\num\n", encoding="utf-8")

    _write_checkpoint_signature(path, {"signature": "ação", "texts": [None, "h"]})

    gravado = _signature_path(path).read_bytes()
    assert json.loads(gravado) == {
        "signature": "ação",
        "texts": [None, "h"],
        "checkpoint": hashlib.sha256(b"texto\num\n").hexdigest(),
    }
    assert _read_checkpoint_signature(path) == json.loads(gravado)
    assert not _signature_path(path).with_name("ckpt.csv.dataframeit.json.tmp").exists()


def test_assinatura_que_nao_grava_avisa_com_a_mensagem_inteira(tmp_path):
    path = tmp_path / "nao_existe.csv"
    target = _signature_path(path)

    msg = (
        f"Falha ao gravar a assinatura do checkpoint em {target}: "
        f"[Errno 2] No such file or directory: {str(path)!r}. "
        "O checkpoint é gravado, mas não será retomado automaticamente."
    )
    with pytest.warns(UserWarning, match=f"^{re.escape(msg)}$"):
        _write_checkpoint_signature(path, _META)


def test_assinatura_em_utf8_e_lida_mesmo_sem_escape(tmp_path):
    path = tmp_path / "ckpt.csv"
    _signature_path(path).write_bytes('{"signature": "ação"}'.encode())

    assert _read_checkpoint_signature(path) == {"signature": "ação"}


def test_assinatura_que_nao_e_dict_ou_nao_e_json_le_como_none(tmp_path):
    path = tmp_path / "ckpt.csv"
    _signature_path(path).write_bytes(b"[1]")
    assert _read_checkpoint_signature(path) is None
    _signature_path(path).write_bytes(b"\xff")
    assert _read_checkpoint_signature(path) is None


def test_checkpoint_com_a_mesma_assinatura_e_retomado(tmp_path):
    path = tmp_path / "ckpt.csv"
    _write_checkpoint(path)

    motivo, salvo = _checkpoint_mismatch(
        _df(), _Checkpoint(path, 1, meta=_META), _Resposta, status_col=_STATUS
    )

    assert motivo is None
    assert salvo is not None
    assert salvo.columns.tolist() == ["texto", "resposta", _STATUS]


def test_checkpoint_sem_assinatura_ao_lado_diz_o_motivo(tmp_path):
    path = tmp_path / "ckpt.csv"
    _df().to_csv(path, index=False)

    assert _checkpoint_mismatch(
        _df(), _Checkpoint(path, 1, meta=_META), _Resposta, status_col=_STATUS
    ) == ("não tem a assinatura da execução que o gravou", None)


def test_checkpoint_mudado_depois_da_assinatura_diz_o_motivo(tmp_path):
    path = tmp_path / "ckpt.csv"
    _write_checkpoint(path)
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    assert _checkpoint_mismatch(
        _df(), _Checkpoint(path, 1, meta=_META), _Resposta, status_col=_STATUS
    ) == ("não é o arquivo que a assinatura ao lado descreve", None)


def test_checkpoint_de_outra_configuracao_diz_o_motivo(tmp_path):
    path = tmp_path / "ckpt.csv"
    _write_checkpoint(path)

    assert _checkpoint_mismatch(
        _df(),
        _Checkpoint(path, 1, meta={**_META, "signature": "outra"}),
        _Resposta,
        status_col=_STATUS,
    ) == (
        (
            "foi gravado com outra configuração (prompt, provider, modelo, model_kwargs, "
            "schema, busca ou colunas de texto e de status)"
        ),
        None,
    )


def test_checkpoint_de_outra_entrada_diz_o_motivo(tmp_path):
    path = tmp_path / "ckpt.csv"
    _write_checkpoint(path)

    assert _checkpoint_mismatch(
        _df(),
        _Checkpoint(path, 1, meta={**_META, "texts": ["h1"]}),
        _Resposta,
        status_col=_STATUS,
    ) == ("foi gravado para outra entrada", None)


def test_checkpoint_com_outro_numero_de_linhas_diz_o_motivo(tmp_path):
    path = tmp_path / "ckpt.csv"
    _write_checkpoint(path, df=_df().head(1))

    assert _checkpoint_mismatch(
        _df(), _Checkpoint(path, 1, meta=_META), _Resposta, status_col=_STATUS
    ) == ("não relê com a coluna de status e as linhas da entrada", None)
