"""Entrada e saída de dataframeit(): colunas de controle, texto ausente, índice e avisos."""

import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel

from dataframeit import dataframeit


class Modelo(BaseModel):
    x: str


class ModeloLista(BaseModel):
    tags: list[str]


def _responde(**campos):
    def call_langchain(text, *args, **kwargs):
        return {'data': {k: (v(text) if callable(v) else v) for k, v in campos.items()}, 'usage': None}
    return call_langchain


def _rodar(df, questions=Modelo, llm=None, **opcoes):
    llm = llm or _responde(x=lambda t: f'x-{t}')
    with patch('dataframeit.core.call_langchain', side_effect=llm) as call_langchain, \
            patch('dataframeit.core.validate_provider_dependencies'):
        opcoes.setdefault('track_tokens', False)
        resultado = dataframeit(df, questions=questions, prompt='Analise {texto}', **opcoes)
    return resultado, call_langchain


# =============================================================================
# Colunas de controle
# =============================================================================

def test_status_column_personalizado_sem_erros_some_da_saida():
    resultado, _ = _rodar(pd.DataFrame({'texto': ['a', 'b']}), status_column='st')
    assert list(resultado.columns) == ['texto', 'x']


def test_status_column_personalizado_com_erro_fica_no_fim():
    def llm(text, *args, **kwargs):
        if text.endswith('b'):
            raise ValueError('falhou')
        return {'data': {'x': 'ok'}, 'usage': None}

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        resultado, _ = _rodar(
            pd.DataFrame({'texto': ['a', 'b']}), llm=llm, status_column='st', max_retries=1,
        )
    assert list(resultado.columns) == ['texto', 'x', 'st', '_error_details']


def test_coluna_de_nome_nao_textual():
    resultado, _ = _rodar(pd.DataFrame(['a', 'b']))
    assert resultado['x'].tolist() == ['x-a', 'x-b']


def test_detalhe_de_erro_antigo_some_quando_a_linha_passa():
    df = pd.DataFrame({
        'texto': ['a'],
        'x': [None],
        '_dataframeit_status': [None],
        '_error_details': ['[Falhou após 1 tentativa(s)] ValueError: boom'],
    })
    resultado, _ = _rodar(df, resume=True)
    assert resultado['x'].tolist() == ['x-a']
    assert '_error_details' not in resultado.columns


@pytest.mark.parametrize('parallel_requests', [1, 2])
def test_detalhe_de_erro_antigo_some_tambem_no_paralelo(parallel_requests):
    df = pd.DataFrame({
        'texto': ['a', 'b'],
        'x': [None, None],
        '_dataframeit_status': [None, None],
        '_error_details': ['erro antigo', None],
    })
    resultado, _ = _rodar(df, resume=True, parallel_requests=parallel_requests)
    assert '_error_details' not in resultado.columns


# =============================================================================
# Texto ausente
# =============================================================================

@pytest.mark.parametrize('parallel_requests', [1, 2])
def test_texto_ausente_nao_chama_o_llm(parallel_requests):
    df = pd.DataFrame({'texto': ['a', None, np.nan, '   ', 'b']})
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter('always')
        resultado, call_langchain = _rodar(df, parallel_requests=parallel_requests)

    assert sorted(c.args[0] for c in call_langchain.call_args_list) == ['a', 'b']
    assert resultado['_dataframeit_status'].tolist() == [
        'processed', 'error', 'error', 'error', 'processed',
    ]
    assert resultado['_error_details'].iloc[1] == 'Texto ausente'
    assert resultado['x'].iloc[1] is None
    assert any('3 linha(s) sem texto' in str(a.message) for a in avisos)


# =============================================================================
# Índice e nomes
# =============================================================================

def test_indice_duplicado_levanta_erro():
    df = pd.DataFrame({'texto': ['a', 'b', 'c']}, index=[0, 0, 1])
    with pytest.raises(ValueError, match='reset_index'):
        _rodar(df)


def test_serie_com_indice_duplicado_levanta_erro():
    with pytest.raises(ValueError, match='reset_index'):
        _rodar(pd.Series(['a', 'b'], index=['k', 'k']))


def test_campo_com_o_nome_da_coluna_de_texto_levanta_erro():
    class ModeloDecisao(BaseModel):
        decisao: str

    df = pd.DataFrame({'decisao': ['Julgo procedente o pedido.']})
    with pytest.raises(ValueError, match="decisao"):
        _rodar(df, questions=ModeloDecisao, llm=_responde(decisao='procedente'))
    assert df['decisao'].tolist() == ['Julgo procedente o pedido.']


# =============================================================================
# Tipos de coluna já existente
# =============================================================================

def test_lista_em_coluna_existente_float():
    df = pd.DataFrame({
        'texto': ['a', 'b'],
        'tags': [np.nan, np.nan],
        '_dataframeit_status': [None, None],
    })
    resultado, _ = _rodar(df, questions=ModeloLista, llm=_responde(tags=['p', 'q']), resume=True)
    assert resultado['tags'].tolist() == [['p', 'q'], ['p', 'q']]
    assert '_dataframeit_status' not in resultado.columns


# =============================================================================
# Reexecução sobre a própria saída
# =============================================================================

def test_reexecucao_sobre_a_propria_saida_avisa():
    primeira, _ = _rodar(pd.DataFrame({'texto': ['a', 'b']}))
    assert '_dataframeit_status' not in primeira.columns

    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter('always')
        _rodar(primeira, resume=True)

    assert any('já estão preenchidas' in str(a.message) for a in avisos)


# =============================================================================
# Parâmetros
# =============================================================================

def test_perguntas_emite_deprecation_warning():
    with pytest.warns(DeprecationWarning, match='questions'):
        with patch('dataframeit.core.call_langchain', side_effect=_responde(x='1')), \
                patch('dataframeit.core.validate_provider_dependencies'):
            dataframeit(pd.DataFrame({'texto': ['a']}), perguntas=Modelo, prompt='{texto}')


@pytest.mark.parametrize('valor', [np.int64(2), 3])
def test_batch_size_aceita_inteiros(valor, tmp_path):
    _rodar(pd.DataFrame({'texto': ['a']}), batch_size=valor, checkpoint_path=tmp_path / 'c.csv')


@pytest.mark.parametrize('valor', [True, 2.0, 0])
def test_batch_size_rejeita_nao_inteiros(valor, tmp_path):
    with pytest.raises(ValueError, match='batch_size'):
        _rodar(pd.DataFrame({'texto': ['a']}), batch_size=valor, checkpoint_path=tmp_path / 'c.csv')


def test_estatisticas_de_busca_usam_o_provider_escolhido(capsys):
    from dataframeit.core import _print_token_stats

    _print_token_stats(
        {'input_tokens': 1, 'output_tokens': 1, 'total_tokens': 2,
         'search_count': 3, 'search_credits': 3},
        model='m', search_provider='exa',
    )
    saida = capsys.readouterr().out
    assert 'EXA' in saida
    assert 'TAVILY' not in saida

