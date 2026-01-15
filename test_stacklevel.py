#!/usr/bin/env python3
"""Script para testar o stacklevel correto dos warnings."""
import warnings

def level3():
    """Nível 3 - emite o warning."""
    warnings.warn("Warning sem stacklevel", UserWarning)
    warnings.warn("Warning com stacklevel=2", UserWarning, stacklevel=2)
    warnings.warn("Warning com stacklevel=3", UserWarning, stacklevel=3)
    warnings.warn("Warning com stacklevel=4", UserWarning, stacklevel=4)

def level2():
    """Nível 2 - chama level3."""
    level3()

def level1():
    """Nível 1 - chama level2."""
    level2()

if __name__ == "__main__":
    print("Testando stacklevel em warnings:\n")
    print("Cadeia de chamadas: __main__ -> level1() -> level2() -> level3()")
    print("=" * 70)

    # Configurar warnings para sempre mostrar
    warnings.simplefilter('always')

    level1()
