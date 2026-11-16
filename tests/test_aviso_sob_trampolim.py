"""Avisos da API quando o mutmut chama a cópia renomeada de dataframeit."""

import linecache
import types
import warnings
from unittest.mock import patch

import pandas as pd
import pytest
from pydantic import BaseModel

from dataframeit import core


class _Modelo(BaseModel):
    campo: str


@pytest.mark.parametrize("parallel_requests", [1, 2])
def test_aviso_da_linha_aponta_a_chamada_sob_trampolim(parallel_requests):
    # O mutmut mantém o corpo original com este nome e o chama por um
    # _trampoline_wrapper. Recriamos esses quadros sem mutar o código.
    namespace = dict(vars(core))
    original = core.dataframeit
    namespace["x_dataframeit__mutmut_orig"] = types.FunctionType(
        original.__code__.replace(co_name="x_dataframeit__mutmut_orig"),
        namespace,
        "x_dataframeit__mutmut_orig",
        original.__defaults__,
        original.__closure__,
    )
    exec(  # noqa: S102 (reproduz o quadro gerado pelo mutmut no módulo core)
        "def trampoline(*args, **kwargs):\n"
        "    return x_dataframeit__mutmut_orig(*args, **kwargs)\n"
        "def _trampoline_wrapper(*args, **kwargs):\n"
        "    return trampoline(*args, **kwargs)\n",
        namespace,
    )
    namespace["dataframeit"] = namespace["_trampoline_wrapper"]

    with (
        patch("dataframeit.core.call_langchain", side_effect=ValueError("falha observável")),
        patch("dataframeit.core.validate_provider_dependencies"),
        warnings.catch_warnings(record=True) as avisos,
    ):
        warnings.simplefilter("always")
        namespace["dataframeit"](
            pd.DataFrame({"texto": ["a"]}),
            questions=_Modelo,
            prompt="{texto}",
            parallel_requests=parallel_requests,
            track_tokens=False,
        )

    (aviso,) = [a for a in avisos if "Falha ao processar linha 0" in str(a.message)]
    assert aviso.filename == __file__
    assert 'namespace["dataframeit"](' in linecache.getline(aviso.filename, aviso.lineno)
