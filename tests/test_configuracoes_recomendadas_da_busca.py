"""A tabela de configurações recomendadas da busca web passa no aviso de rate limit.

Quem copia uma linha da tabela não pode receber, na primeira execução, o aviso
de que a configuração excede o limite do provedor. O teste lê as duas versões
da doc e roda as pontas de cada faixa contra o aviso.
"""

import re
import warnings
from pathlib import Path

import pytest

from dataframeit.core import _warn_search_rate_limit

PASTA_DOCS = Path(__file__).resolve().parent.parent / "docs"
SECOES = {
    PASTA_DOCS / "guides" / "web-search.md": "### Configurações Recomendadas",
    PASTA_DOCS / "en" / "guides" / "web-search.md": "### Recommended Settings",
}

# Faixa aberta ("4+ campos") é testada até este número de campos.
CAMPOS_NA_FAIXA_ABERTA = 50

# A doc escreve as faixas com traço médio, como em "5\u201310".
TRACO_DE_FAIXA = "\u2013"


def _faixa(texto: str) -> list[int]:
    """Pontas de uma faixa da tabela: "5 a 10" vira [5, 10], "2" vira [2], "4+" vai até o teto."""
    if texto.endswith("+"):
        return [int(texto[:-1]), CAMPOS_NA_FAIXA_ABERTA]
    return [int(numero) for numero in texto.split(TRACO_DE_FAIXA)]


def _linhas_da_tabela(caminho: Path, titulo: str) -> list[tuple]:
    """(provedor, search_per_field, campos, parallel_requests, rate_limit_delay) de cada linha."""
    secao = caminho.read_text(encoding="utf-8").split(titulo + "\n", 1)[1].split("\n### ", 1)[0]
    linhas = []
    provedor = None
    for linha in secao.splitlines():
        if cabecalho := re.match(r"\*\*(\w+)", linha):
            provedor = cabecalho.group(1).lower()
        elif linha.startswith("| `search_per_field="):
            cenario, workers, pausa = (celula.strip() for celula in linha.strip("|").split("|"))
            por_campo = "=True" in cenario
            # O cenário traz a faixa de campos no começo do parêntese, como em "(4+ campos)".
            campos = cenario.split("(")[1].split()[0] if por_campo else "1"
            linhas.append(
                (
                    provedor,
                    por_campo,
                    _faixa(campos),
                    _faixa(workers),
                    float(pausa.removesuffix("s")),
                )
            )
    return linhas


TABELAS = {caminho: _linhas_da_tabela(caminho, titulo) for caminho, titulo in SECOES.items()}


def test_tabelas_em_portugues_e_ingles_recomendam_o_mesmo():
    portugues, ingles = TABELAS.values()

    assert portugues == ingles
    assert {linha[0] for linha in portugues} == {"tavily", "exa"}
    assert len(portugues) == 6


@pytest.mark.parametrize("linha", TABELAS[next(iter(SECOES))], ids=str)
def test_pontas_das_faixas_recomendadas_nao_disparam_o_aviso(linha):
    provedor, por_campo, faixa_de_campos, faixa_de_workers, pausa = linha
    for campos in faixa_de_campos:
        for workers in faixa_de_workers:
            with warnings.catch_warnings(record=True) as avisos:
                warnings.simplefilter("always")
                _warn_search_rate_limit(
                    num_rows=200,
                    num_fields=campos,
                    parallel_requests=workers,
                    search_per_field=por_campo,
                    rate_limit_delay=pausa,
                    search_provider=provedor,
                )
            assert not avisos, (campos, workers, str(avisos[0].message))
