"""Configuração comum da suíte."""

import signal


def pytest_configure() -> None:
    """Devolve ao SIGINT o handler padrão do Python quando o processo o ignora.

    O forkserver do mutmut ignora o SIGINT, e os processos em que ele roda os
    testes herdam isso. Sem o handler padrão, o Ctrl-C simulado dos testes de
    interrupção não vira `KeyboardInterrupt`: o teste sequencial falha antes de
    começar, e o paralelo espera para sempre a interrupção que não chega. O
    mutante sairia morto ou em timeout por causa do ambiente, e não do código.
    """
    if signal.getsignal(signal.SIGINT) is signal.SIG_IGN:
        signal.signal(signal.SIGINT, signal.default_int_handler)
