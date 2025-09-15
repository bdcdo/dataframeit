from ..utils import check_dependency


class DependencyChecker:
    """Verificação de dependências com cache por instância.

    Centraliza a verificação de bibliotecas opcionais evitando
    múltiplas verificações das mesmas dependências.
    """

    def __init__(self):
        """Inicializa o verificador com cache próprio."""
        self._checked_dependencies = set()

    def check_dependency(self, package_name: str, install_name: str) -> None:
        """Verifica se uma dependência está instalada (com cache).

        Args:
            package_name: Nome do pacote para importação.
            install_name: Nome do pacote para instalação.

        Raises:
            ImportError: Se a dependência não estiver instalada.
        """
        if package_name not in self._checked_dependencies:
            check_dependency(package_name, install_name)
            self._checked_dependencies.add(package_name)