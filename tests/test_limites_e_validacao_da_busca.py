"""Texto e fronteiras do aviso de rate limit da busca e da validação dos overrides."""

import re
import warnings
from unittest.mock import patch

import pandas as pd
import pytest
from pydantic import BaseModel

from dataframeit import dataframeit
from dataframeit.core import _validate_search_groups, _warn_search_rate_limit


class Modelo(BaseModel):
    nome: str
    cidade: str


def _aviso(**kwargs) -> str | None:
    """Mensagem do aviso de rate limit, ou None quando ele não sai."""
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        _warn_search_rate_limit(**kwargs)
    return str(avisos[0].message) if avisos else None


def test_aviso_traz_a_configuracao_os_problemas_e_a_dica():
    """Sem busca por campo, cada linha faz uma consulta, qualquer que seja o número de campos."""
    aviso = _aviso(
        num_rows=100,
        num_fields=4,
        parallel_requests=20,
        search_per_field=False,
        rate_limit_delay=0.0,
    )
    assert aviso == (
        "Configuração pode exceder rate limits de busca (Tavily Search). "
        "parallel_requests=20, search_per_field=False, rate_limit_delay=0.0s, "
        "total de queries estimadas=100. Problemas: queries concorrentes estimadas: 20 "
        "(limite recomendado: 10); taxa estimada ~1200 req/min (limite Tavily Search: "
        "~100/min). Para evitar HTTP 429, use: parallel_requests=10, rate_limit_delay=7.6."
    )


def test_taxa_igual_a_80_por_cento_do_limite_nao_avisa():
    """4 workers com pausa de 2 s fazem 80 req/min, exatamente 80% dos 100 do Tavily."""
    assert (
        _aviso(
            num_rows=100,
            num_fields=1,
            parallel_requests=4,
            search_per_field=False,
            rate_limit_delay=2.0,
        )
        is None
    )


def test_pausa_igual_a_recomendada_nao_entra_na_dica():
    """Com 11 workers, a pausa recomendada para 10 é 7,6 s; quem já a usa só reduz workers."""
    aviso = _aviso(
        num_rows=100,
        num_fields=1,
        parallel_requests=11,
        search_per_field=False,
        rate_limit_delay=7.6,
    )
    assert aviso is not None
    assert aviso.endswith("Para evitar HTTP 429, use: parallel_requests=10.")


def test_aviso_de_rate_limit_aponta_a_chamada_do_usuario():
    def call_agent(text, model, *_args, **_kwargs):
        return {"data": dict.fromkeys(model.model_fields, "x"), "usage": {}}

    with (
        patch("dataframeit.core.validate_provider_dependencies"),
        patch("dataframeit.core.validate_search_dependencies"),
        patch("dataframeit.agent.call_agent", side_effect=call_agent),
        pytest.warns(UserWarning, match="pode exceder rate limits de busca") as avisos,
    ):
        dataframeit(
            pd.DataFrame({"texto": ["a", "b"]}),
            questions=Modelo,
            prompt="{texto}",
            use_search=True,
            parallel_requests=20,
        )

    (aviso,) = [a for a in avisos if "rate limits de busca" in str(a.message)]
    assert aviso.filename == __file__


@pytest.mark.parametrize(
    ("use_search", "search_per_field", "mensagem"),
    [
        (False, True, "search_groups requer use_search=True"),
        (True, False, "search_groups requer search_per_field=True"),
    ],
)
def test_search_groups_sem_a_busca_por_campo_diz_o_que_falta(
    use_search, search_per_field, mensagem
):
    with pytest.raises(ValueError, match=f"^{re.escape(mensagem)}$"):
        _validate_search_groups(
            {"g": {"fields": ["nome"]}},
            Modelo,
            use_search=use_search,
            search_per_field=search_per_field,
        )


@pytest.mark.parametrize(
    "override",
    [
        {"search_depth": "basic"},
        {"search_depth": "advanced"},
        {"max_results": 1},
        {"max_results": 20},
        {"max_search_calls": 1},
    ],
    ids=str,
)
def test_override_do_grupo_nas_pontas_da_faixa_e_aceito(override):
    grupos = _validate_search_groups(
        {"g": {"fields": ["nome"], **override}},
        Modelo,
        use_search=True,
        search_per_field=True,
    )
    ((chave, valor),) = override.items()
    assert getattr(grupos["g"], chave) == valor


@pytest.mark.parametrize(
    ("override", "mensagem"),
    [
        ({"search_depth": "profundo"}, "search_depth deve ser 'basic' ou 'advanced'"),
        ({"max_results": 0}, "max_results deve estar entre 1 e 20"),
        ({"max_results": 21}, "max_results deve estar entre 1 e 20"),
        ({"max_search_calls": 0}, "max_search_calls deve ser int >= 1"),
    ],
    ids=str,
)
def test_override_do_grupo_fora_da_faixa_nomeia_o_grupo(override, mensagem):
    with pytest.raises(ValueError, match=f"^Grupo 'g': {re.escape(mensagem)}$"):
        _validate_search_groups(
            {"g": {"fields": ["nome"], **override}},
            Modelo,
            use_search=True,
            search_per_field=True,
        )


def test_estimativa_multiplica_as_execucoes_da_linha_pela_duracao_de_cada_uma(monkeypatch):
    """Com execuções de 2 s, a linha de 4 campos leva 8 s: 11 workers fazem ~330 req/min."""
    monkeypatch.setattr("dataframeit.core._MIN_SEARCH_RUN_SECONDS", 2.0)
    aviso = _aviso(
        num_rows=100,
        num_fields=4,
        parallel_requests=11,
        search_per_field=True,
        rate_limit_delay=0.0,
    )
    assert aviso is not None
    assert aviso.endswith(
        "taxa estimada ~330 req/min (limite Tavily Search: ~100/min). "
        "Para evitar HTTP 429, use: parallel_requests=10, rate_limit_delay=26.3."
    )
