import dataframeit
from dataframeit import errors


def test_all_so_lista_nomes_que_existem():
    for nome in dataframeit.__all__:
        assert hasattr(dataframeit, nome), nome


def test_excecoes_do_topo_sao_as_de_errors():
    # Quem captura pelo nome do topo precisa pegar a mesma classe que o código levanta.
    for nome in (
        'ProviderError',
        'ProviderTransientError',
        'ProviderOverloadedError',
        'ProviderRejectedOutputError',
        'ProviderConfigurationError',
        'ProviderOutputError',
    ):
        assert nome in dataframeit.__all__
        assert getattr(dataframeit, nome) is getattr(errors, nome)


def test_hierarquia_documentada_das_excecoes():
    assert issubclass(dataframeit.ProviderOverloadedError, dataframeit.ProviderTransientError)
    assert issubclass(dataframeit.ProviderRejectedOutputError, dataframeit.ProviderTransientError)
    assert issubclass(dataframeit.ProviderRejectedOutputError, ValueError)
    assert issubclass(dataframeit.ProviderTransientError, dataframeit.ProviderError)
    assert issubclass(dataframeit.ProviderError, RuntimeError)
    assert issubclass(dataframeit.ProviderConfigurationError, ValueError)
    assert issubclass(dataframeit.ProviderOutputError, ValueError)
    assert not issubclass(dataframeit.ProviderOutputError, dataframeit.ProviderError)
