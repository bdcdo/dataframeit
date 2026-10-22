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


def _argumento_de_dados(chamada):
    """Nome do DataFrame passado a dataframeit() e se ele vai como cópia (df.copy())."""
    dados = next(
        (argumento.value for argumento in chamada.keywords if argumento.arg == "data"),
        chamada.args[0] if chamada.args else None,
    )
    if isinstance(dados, ast.Name):
        return dados.id, False
    if (
        isinstance(dados, ast.Call)
        and isinstance(dados.func, ast.Attribute)
        and dados.func.attr == "copy"
        and isinstance(dados.func.value, ast.Name)
    ):
        return dados.func.value.id, True
    return None, False


def dataframes_reusados(codigos):
    """Nomes passados a dataframeit() depois de uma chamada que já os processou.

    O dataframeit grava os resultados e o status no próprio DataFrame pandas de
    entrada, e a chamada seguinte sobre ele, ou sobre uma cópia dele, não tem
    linha pendente: com o mesmo modelo não chama o LLM, e com um modelo de
    outros campos levanta ValueError. O nome volta a ser livre quando é
    atribuído de novo; gravar numa coluna dele (`df["x"] = ...`) não o libera,
    porque o status continua lá.

    `codigos` são as células na ordem de execução. A atribuição conta no fim
    do comando, depois da chamada que ela recebe: em `df = dataframeit(df, ...)`,
    o `df` seguinte já é a saída.
    """
    eventos: list[tuple[tuple[int, int | None, int | None], str, str, bool]] = []
    for celula, codigo in enumerate(codigos):
        for no in ast.walk(ast.parse(codigo)):
            if isinstance(no, ast.Assign):
                eventos += [
                    ((celula, no.end_lineno, no.end_col_offset), "atribui", alvo.id, False)
                    for alvo in no.targets
                    if isinstance(alvo, ast.Name)
                ]
            elif (
                isinstance(no, ast.Call)
                and isinstance(no.func, ast.Name)
                and no.func.id == "dataframeit"
            ):
                nome, copia = _argumento_de_dados(no)
                if nome is not None:
                    eventos.append(((celula, no.lineno, no.col_offset), "processa", nome, copia))
    processados = set()
    reusados = []
    for _, tipo, nome, copia in sorted(eventos):
        if tipo == "atribui":
            processados.discard(nome)
        elif nome in processados:
            reusados.append(nome)
        elif not copia:
            processados.add(nome)
    return reusados


def test_ha_exemplos_para_checar():
    assert NOTEBOOKS
    assert SCRIPTS


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


@pytest.mark.parametrize("caminho", NOTEBOOKS + SCRIPTS, ids=lambda caminho: caminho.name)
def test_exemplo_nao_reusa_dataframe_ja_processado(caminho):
    assert dataframes_reusados(_codigos(caminho)) == []


def test_checagem_acusa_parametro_inexistente():
    codigo = "dataframeit(df, Modelo, PROMPT, placeholder='sentenca', resume=True)"
    assert argumentos_desconhecidos(codigo) == ["placeholder"]


@pytest.mark.parametrize(
    ("codigos", "esperado"),
    [
        (["a = dataframeit(df, M, P)\nb = dataframeit(df, M, Q)"], ["df"]),
        (["a = dataframeit(df, M, P)", "b = dataframeit(df.copy(), M, Q)"], ["df"]),
        (["a = dataframeit(df.copy(), M, P)", "b = dataframeit(df.copy(), M, Q)"], []),
        (["a = dataframeit(df, M, P)", "df = carrega()", "b = dataframeit(df, M, Q)"], []),
        (["a = dataframeit(df, M, P)", "df['x'] = 1", "b = dataframeit(df, M, Q)"], ["df"]),
        (["a = dataframeit(data=df, questions=M, prompt=P)", "b = dataframeit(data=df)"], ["df"]),
        (["df = dataframeit(df, M, P)\nb = dataframeit(df, M, Q)"], []),
        (["a = dataframeit(df, M, P)\nb = dataframeit(outro, M, Q)"], []),
        (["a = dataframeit(['texto'], M, P)\nb = dataframeit(df[:2], M, Q)"], []),
    ],
    ids=[
        "mesmo-df",
        "copia-de-df-ja-processado",
        "copia-nas-duas",
        "reatribuido-entre-as-chamadas",
        "coluna-gravada-nao-libera",
        "por-nome",
        "saida-no-mesmo-nome",
        "outro-df",
        "entrada-sem-nome",
    ],
)
def test_checagem_acusa_dataframe_reusado(codigos, esperado):
    assert dataframes_reusados(codigos) == esperado
