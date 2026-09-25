"""Um modelo LangChain por execução, e não por linha."""

import subprocess
import sys
import threading
import time
import warnings
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pydantic import BaseModel, Field

from dataframeit.core import dataframeit
from dataframeit.llm import _BuildOnce


class Resposta(BaseModel):
    campo: str


class DoisCampos(BaseModel):
    primeiro: str
    segundo: str


TEXTOS = ["a", "b", "c", "d", "e"]


def _llm_falso():
    llm = MagicMock()
    llm.with_structured_output.return_value.invoke.return_value = {
        "parsed": Resposta(campo="ok"),
        "raw": None,
        "parsing_error": None,
    }
    return llm


@pytest.mark.parametrize("parallel_requests", [1, 3])
def test_sem_busca_modelo_e_structured_output_uma_vez(parallel_requests):
    llm = _llm_falso()

    with (
        patch("dataframeit.llm._create_langchain_llm", return_value=llm) as criar,
        patch("dataframeit.core.validate_provider_dependencies"),
    ):
        resultado = dataframeit(
            pd.DataFrame({"texto": TEXTOS}),
            Resposta,
            "Responda {texto}",
            parallel_requests=parallel_requests,
        )

    assert resultado["campo"].tolist() == ["ok"] * len(TEXTOS)
    assert criar.call_count == 1
    assert llm.with_structured_output.call_count == 1
    assert llm.with_structured_output.return_value.invoke.call_count == len(TEXTOS)


def _agente_falso(modelo):
    """create_agent falso que devolve um valor preenchido para cada campo do schema."""
    montagens = []

    def create_agent(**kwargs):
        montagens.append(kwargs)
        schema = kwargs["response_format"].schema
        agente = MagicMock()
        agente.invoke.return_value = {
            "structured_response": schema(**dict.fromkeys(schema.model_fields, "ok")),
            "messages": [],
        }
        return agente

    return montagens, create_agent


def _rodar_com_busca(modelo, parallel_requests, **kwargs):
    montagens, create_agent = _agente_falso(modelo)
    provider = MagicMock()
    provider.create_tool = lambda **kw: MagicMock(name="ferramenta")
    provider.calculate_credits.return_value = 0

    with (
        patch("dataframeit.llm._create_langchain_llm", return_value=object()) as criar,
        patch("langchain.agents.create_agent", create_agent),
        patch("dataframeit.agent.get_provider", return_value=provider),
        patch("dataframeit.core.validate_provider_dependencies"),
        patch("dataframeit.core.validate_search_dependencies"),
    ):
        resultado = dataframeit(
            pd.DataFrame({"texto": TEXTOS}),
            modelo,
            "Responda {texto}",
            use_search=True,
            parallel_requests=parallel_requests,
            **kwargs,
        )
    return resultado, criar, montagens


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("parallel_requests", [1, 3])
def test_agente_unico_montado_uma_vez(parallel_requests):
    resultado, criar, montagens = _rodar_com_busca(Resposta, parallel_requests)

    assert resultado["campo"].tolist() == ["ok"] * len(TEXTOS)
    assert criar.call_count == 1
    assert len(montagens) == 1


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    "kwargs",
    [
        {"search_per_field": True},
        {"search_per_field": True, "search_groups": {"g": {"fields": ["primeiro", "segundo"]}}},
    ],
)
@pytest.mark.parametrize("parallel_requests", [1, 3])
def test_por_campo_e_por_grupo_reusam_o_modelo(parallel_requests, kwargs):
    resultado, criar, montagens = _rodar_com_busca(DoisCampos, parallel_requests, **kwargs)

    assert resultado["primeiro"].tolist() == ["ok"] * len(TEXTOS)
    assert resultado["segundo"].tolist() == ["ok"] * len(TEXTOS)
    assert criar.call_count == 1
    # O agente segue montado por chamada, porque o modelo de cada campo ou
    # grupo é criado na hora; todos recebem o mesmo modelo LangChain.
    assert len(montagens) >= len(TEXTOS)
    assert len({id(m["model"]) for m in montagens}) == 1


class ComOverrides(BaseModel):
    primeiro: str = Field(json_schema_extra={"max_results": 3, "max_search_calls": 2})
    segundo: str
    terceiro: str


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("parallel_requests", [1, 3])
def test_overrides_por_campo_e_por_grupo_mantem_o_modelo_compartilhado(parallel_requests):
    """A cópia da config com override de busca herda o mesmo modelo."""
    resultado, criar, montagens = _rodar_com_busca(
        ComOverrides,
        parallel_requests,
        search_per_field=True,
        search_groups={"g": {"fields": ["segundo", "terceiro"], "search_depth": "advanced"}},
    )

    assert resultado["primeiro"].tolist() == ["ok"] * len(TEXTOS)
    assert criar.call_count == 1
    assert len({id(m["model"]) for m in montagens}) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"use_search": True},
        {"use_search": True, "search_per_field": True},
        {
            "use_search": True,
            "search_per_field": True,
            "search_groups": {"g": {"fields": ["primeiro", "segundo"]}},
        },
    ],
)
@pytest.mark.parametrize("parallel_requests", [1, 3])
def test_erro_de_construcao_e_erro_da_linha_em_todos_os_modos(parallel_requests, kwargs):
    modelo = DoisCampos if kwargs.get("search_per_field") else Resposta
    provider = MagicMock()
    provider.create_tool = lambda **kw: MagicMock(name="ferramenta")
    with (
        patch(
            "dataframeit.llm._create_langchain_llm", side_effect=ValueError("sem chave")
        ) as criar,
        patch("dataframeit.agent.get_provider", return_value=provider),
        patch("dataframeit.core.validate_provider_dependencies"),
        patch("dataframeit.core.validate_search_dependencies"),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore")
        resultado = dataframeit(
            pd.DataFrame({"texto": ["a", "b"]}),
            modelo,
            "Responda {texto}",
            max_retries=1,
            parallel_requests=parallel_requests,
            **kwargs,
        )

    assert resultado["_dataframeit_status"].tolist() == ["error", "error"]
    # A falha não fica em cache: cada linha tenta construir de novo.
    assert criar.call_count >= 2


def test_build_once_nao_guarda_falha():
    tentativas = []

    def construir():
        tentativas.append(1)
        if len(tentativas) == 1:
            msg = "chave ausente"
            raise ValueError(msg)
        return "modelo"

    unico = _BuildOnce(construir)
    with pytest.raises(ValueError, match="chave ausente"):
        unico()
    assert unico() == "modelo"
    assert unico() == "modelo"
    assert len(tentativas) == 2


def test_build_once_constroi_uma_vez_entre_threads():
    chamadas = []
    barreira = threading.Barrier(8)

    def construir():
        chamadas.append(1)
        # Construção lenta, como a de um cliente real: sem o lock, as threads
        # que chegam juntas construiriam cada uma o seu.
        time.sleep(0.05)
        return object()

    unico = _BuildOnce(construir)
    resultados = []

    def trabalhador():
        barreira.wait()
        resultados.append(unico())

    threads = [threading.Thread(target=trabalhador) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(chamadas) == 1
    assert len({id(r) for r in resultados}) == 1


def test_erro_de_construcao_continua_sendo_erro_da_linha():
    """Chave ausente marca as linhas como erro, sem exceção antes do processamento."""
    with (
        patch("dataframeit.llm._create_langchain_llm", side_effect=ValueError("sem chave")),
        patch("dataframeit.core.validate_provider_dependencies"),
        pytest.warns(UserWarning, match="Falha ao processar linha"),
    ):
        resultado = dataframeit(
            pd.DataFrame({"texto": ["a", "b"]}),
            Resposta,
            "Responda {texto}",
            max_retries=1,
        )

    assert resultado["_dataframeit_status"].tolist() == ["error", "error"]


def test_import_nao_altera_loggers_de_outras_bibliotecas():
    codigo = (
        "import logging\n"
        "nomes = ('langchain_google_genai', 'langchain_core', 'httpx')\n"
        "for nome in nomes: logging.getLogger(nome).setLevel(logging.DEBUG)\n"
        "import dataframeit, dataframeit.core, dataframeit.llm, dataframeit.agent\n"
        "print([logging.getLogger(n).level for n in nomes])\n"
    )
    saida = subprocess.run(  # noqa: S603 (roda o próprio interpretador com código fixo do teste)
        [sys.executable, "-c", codigo],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert saida == "[10, 10, 10]"
