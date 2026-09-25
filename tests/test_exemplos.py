"""Os exemplos em example/ usam só parâmetros que dataframeit() aceita.

Os notebooks não rodam no CI, porque chamam o LLM. Sem esta checagem, um
parâmetro removido ou renomeado continua nos exemplos e só quebra na mão
de quem os abre no Colab.
"""

import ast
import inspect
import json
import re
from pathlib import Path

import pytest

from dataframeit import dataframeit

PASTA_EXEMPLOS = Path(__file__).resolve().parent.parent / "example"
NOTEBOOKS = sorted(PASTA_EXEMPLOS.glob("*.ipynb"))
SCRIPTS = sorted(PASTA_EXEMPLOS.glob("*.py"))
PARAMETROS_ACEITOS = set(inspect.signature(dataframeit).parameters)


def _codigo_da_celula(celula):
    """Código Python da célula, sem as linhas de comando do Jupyter (! e %)."""
    linhas = "".join(celula["source"]).splitlines()
    return "\n".join(linha for linha in linhas if not linha.lstrip().startswith(("!", "%")))


def _codigos(caminho):
    if caminho.suffix == ".py":
        return [caminho.read_text(encoding="utf-8")]
    notebook = json.loads(caminho.read_text(encoding="utf-8"))
    return [
        _codigo_da_celula(celula) for celula in notebook["cells"] if celula["cell_type"] == "code"
    ]


def argumentos_desconhecidos(codigo):
    """Argumentos nomeados de chamadas a dataframeit() que a função não aceita."""
    desconhecidos = []
    for no in ast.walk(ast.parse(codigo)):
        if (
            isinstance(no, ast.Call)
            and isinstance(no.func, ast.Name)
            and no.func.id == "dataframeit"
        ):
            desconhecidos += [
                argumento.arg
                for argumento in no.keywords
                if argumento.arg is not None and argumento.arg not in PARAMETROS_ACEITOS
            ]
    return desconhecidos


def test_ha_exemplos_para_checar():
    assert NOTEBOOKS and SCRIPTS


@pytest.mark.parametrize("caminho", NOTEBOOKS + SCRIPTS, ids=lambda caminho: caminho.name)
def test_exemplo_so_usa_parametros_existentes(caminho):
    for codigo in _codigos(caminho):
        assert argumentos_desconhecidos(codigo) == []


@pytest.mark.parametrize("caminho", NOTEBOOKS + SCRIPTS, ids=lambda caminho: caminho.name)
def test_exemplo_nao_usa_modelo_gpt_4(caminho):
    assert not re.search(r"""model=["']gpt-4""", caminho.read_text(encoding="utf-8"))


@pytest.mark.parametrize("caminho", NOTEBOOKS, ids=lambda caminho: caminho.name)
def test_notebook_sem_saidas(caminho):
    notebook = json.loads(caminho.read_text(encoding="utf-8"))
    for celula in notebook["cells"]:
        if celula["cell_type"] == "code":
            assert celula["outputs"] == []
            assert celula["execution_count"] is None


def test_checagem_acusa_parametro_inexistente():
    codigo = "dataframeit(df, Modelo, PROMPT, placeholder='sentenca', resume=True)"
    assert argumentos_desconhecidos(codigo) == ["placeholder"]
