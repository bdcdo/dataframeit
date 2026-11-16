"""Ctrl-C dos testes num processo que ignora o SIGINT, como o do mutmut."""

import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

TESTE_DE_CTRL_C = (
    Path(__file__).with_name("test_interrupcao.py").as_posix()
    + "::test_ctrl_c_grava_as_linhas_pagas_e_a_retomada_nao_as_chama_de_novo[pandas-1]"
)


@pytest.mark.skipif(
    not hasattr(signal, "pthread_kill"), reason="Ctrl-C simulado com signal.pthread_kill"
)
def test_ctrl_c_simulado_vira_keyboard_interrupt_mesmo_com_o_sigint_ignorado():
    """O forkserver do mutmut ignora o SIGINT, e os testes rodam em processos filhos dele."""
    codigo = textwrap.dedent(
        """
        import signal
        import sys

        import pytest

        signal.signal(signal.SIGINT, signal.SIG_IGN)
        sys.exit(pytest.main([sys.argv[1], "-q", "-p", "no:cacheprovider", "-p", "no:randomly"]))
        """
    )
    saida = subprocess.run(  # noqa: S603 (roda o próprio interpretador com código fixo do teste)
        [sys.executable, "-c", codigo, TESTE_DE_CTRL_C],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert saida.returncode == 0, saida.stdout[-2000:]
