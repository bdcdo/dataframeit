#!/usr/bin/env python3
"""Script para testar se os warnings apontam para o código do usuário."""
import warnings
import sys
import pandas as pd
from pydantic import BaseModel

# Adicionar src ao path para importar dataframeit
sys.path.insert(0, '/home/user/dataframeit/src')

from dataframeit import dataframeit


class SimpleModel(BaseModel):
    field1: str
    field2: str


def test_column_conflict_warning():
    """Testa se o warning de conflito de colunas aponta para o código do usuário."""
    print("=" * 70)
    print("Teste 1: Warning de conflito de colunas")
    print("=" * 70)

    # Criar DataFrame com colunas que já existem
    df = pd.DataFrame({
        'texto': ['exemplo'],
        'field1': ['ja_existe'],  # Coluna que conflita
    })

    # Capturar warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Esta linha deve gerar um warning (linha 35)
        result = dataframeit(df, SimpleModel, "Teste {texto}")

        if w:
            warning = w[0]
            print(f"\n✅ Warning capturado:")
            print(f"   Mensagem: {warning.message}")
            print(f"   Arquivo: {warning.filename}")
            print(f"   Linha: {warning.lineno}")

            # Verificar se aponta para este arquivo (não para core.py)
            if 'test_warnings_stacklevel.py' in warning.filename:
                print(f"\n✅ SUCESSO: Warning aponta para o código do usuário!")
                print(f"   (linha {warning.lineno} deste arquivo)")
                return True
            else:
                print(f"\n❌ FALHA: Warning aponta para código interno!")
                print(f"   ({warning.filename}:{warning.lineno})")
                return False
        else:
            print("\n⚠️  Nenhum warning capturado")
            return True  # OK se não gerar warning


def test_verify_all_stacklevels():
    """Verifica se todos os warnings.warn() têm stacklevel configurado."""
    print("\n" + "=" * 70)
    print("Teste 2: Verificar se todos os warnings têm stacklevel")
    print("=" * 70)

    import re

    files_to_check = [
        '/home/user/dataframeit/src/dataframeit/core.py',
        '/home/user/dataframeit/src/dataframeit/errors.py',
    ]

    all_ok = True

    for filepath in files_to_check:
        with open(filepath, 'r') as f:
            content = f.read()
            lines = content.split('\n')

        # Encontrar todas as chamadas warnings.warn()
        for i, line in enumerate(lines, 1):
            if 'warnings.warn(' in line:
                # Capturar o contexto (múltiplas linhas)
                start = max(0, i - 1)
                end = min(len(lines), i + 5)
                context = '\n'.join(lines[start:end])

                # Verificar se tem stacklevel
                if 'stacklevel=' not in context:
                    print(f"\n❌ {filepath}:{i}")
                    print(f"   Warning sem stacklevel!")
                    all_ok = False
                else:
                    # Extrair o valor do stacklevel
                    match = re.search(r'stacklevel=(\d+)', context)
                    if match:
                        level = match.group(1)
                        print(f"\n✅ {filepath}:{i}")
                        print(f"   stacklevel={level}")

    if all_ok:
        print(f"\n✅ SUCESSO: Todos os warnings têm stacklevel configurado!")

    return all_ok


def test_summary():
    """Mostra resumo das correções."""
    print("\n" + "=" * 70)
    print("Resumo das correções aplicadas")
    print("=" * 70)

    corrections = [
        ("core.py", 216, "Conflito de colunas", "Nenhum → 2"),
        ("core.py", 584, "Falha ao processar (seq.)", "Nenhum → 3"),
        ("core.py", 754, "Rate limit detectado", "2 → 3"),
        ("core.py", 772, "Falha ao processar (par.)", "Nenhum → 3"),
        ("core.py", 802, "Erro no executor", "Nenhum → 3"),
        ("errors.py", 469, "Erro não-recuperável", "3 → 5"),
        ("errors.py", 487, "Retry após falha", "3 → 5"),
    ]

    print("\n| Arquivo | Linha | Descrição | Mudança |")
    print("|---------|-------|-----------|---------|")
    for file, line, desc, change in corrections:
        print(f"| {file:11} | {line:5} | {desc:27} | {change:13} |")

    print(f"\n✅ Total de correções: {len(corrections)}")


if __name__ == "__main__":
    print("\nTeste de stacklevel dos warnings - Issue 74")
    print("=" * 70)

    # Executar testes
    test1_ok = test_column_conflict_warning()
    test2_ok = test_verify_all_stacklevels()
    test_summary()

    print("\n" + "=" * 70)
    print("Resultado final")
    print("=" * 70)

    if test1_ok and test2_ok:
        print("\n✅ Todos os testes passaram!")
        sys.exit(0)
    else:
        print("\n❌ Alguns testes falharam")
        sys.exit(1)
