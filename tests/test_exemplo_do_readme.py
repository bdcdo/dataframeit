"""O exemplo rápido de cada README roda e imprime a tabela que o README mostra.

O exemplo é a primeira chamada que quem chega à biblioteca adapta, e por isso
mostra `provider` e `text_column`, os argumentos que quase toda primeira
chamada fora do padrão precisa. O LLM é simulado e responde a cada texto com a
linha da tabela de saída do README que começa por ele.
"""

import re
from pathlib import Path
from unittest.mock import patch

import pytest

import dataframeit as pacote
from dataframeit.utils import TOKEN_COLUMNS

RAIZ = Path(__file__).resolve().parent.parent
SECOES = {
    "README.md": "## Quick Example",
    "README.pt-BR.md": "## Exemplo Rápido",
    "README.es.md": "## Ejemplo rápido",
}


def _secao(nome):
    texto = (RAIZ / nome).read_text(encoding="utf-8")
    inicio = texto.index(SECOES[nome] + "\n")
    return texto[inicio : texto.index("\n## ", inicio + 1)]


def _tabela(secao):
    linhas = [linha for linha in secao.splitlines() if linha.startswith("|")]
    cabecalho, _, *corpo = (
        [celula.strip() for celula in linha.strip("|").split("|")] for linha in linhas
    )
    return cabecalho, [dict(zip(cabecalho, linha, strict=True)) for linha in corpo]


def _comeco(celula):
    """Texto conhecido da célula: sem o '...' final da célula truncada."""
    return celula.removesuffix("...").rstrip()


def _executa(secao, linhas_da_tabela):
    """Roda o código do exemplo e devolve (o que ele imprime, a chamada a dataframeit).

    O texto que o LLM simulado recebe precisa começar por uma célula truncada da
    tabela; se `text_column` apontar para outra coluna, nenhuma linha bate.
    """
    codigo = re.search(r"```python\n(.*?)```", secao, re.DOTALL)
    assert codigo is not None

    def _resposta(texto, modelo, *args, **kwargs):
        linhas = [
            linha
            for linha in linhas_da_tabela
            if any(
                celula.endswith("...") and texto.startswith(_comeco(celula))
                for celula in linha.values()
            )
        ]
        assert len(linhas) == 1, texto
        dados = modelo.model_validate({campo: linhas[0][campo] for campo in modelo.model_fields})
        return {"data": dados.model_dump(), "usage": None}

    impressos = []
    with (
        patch("dataframeit.core.call_langchain", side_effect=_resposta),
        patch("dataframeit.core.validate_provider_dependencies"),
        patch.object(pacote, "dataframeit", wraps=pacote.dataframeit) as chamada,
    ):
        exec(compile(codigo.group(1), "<readme>", "exec"), {"print": impressos.append})  # noqa: S102 (código do próprio README)
    assert len(impressos) == 1
    return impressos[0], chamada


@pytest.mark.parametrize("nome", sorted(SECOES))
def test_exemplo_rapido_imprime_a_tabela_do_readme(nome):
    secao = _secao(nome)
    cabecalho, linhas = _tabela(secao)

    saida, chamada = _executa(secao, linhas)

    chamada.assert_called_once()
    assert {"provider", "text_column"} <= chamada.call_args.kwargs.keys()
    assert list(saida.columns) == cabecalho + list(TOKEN_COLUMNS)
    assert all(f"`{coluna}`" in secao for coluna in TOKEN_COLUMNS)
    for (_, registro), linha in zip(saida.iterrows(), linhas, strict=True):
        for coluna, celula in linha.items():
            if celula.endswith("..."):
                assert str(registro[coluna]).startswith(_comeco(celula))
            else:
                assert str(registro[coluna]) == celula
