"""Sistema de condicionais para execução condicional de campos."""

import logging
from typing import Any, Dict, List, Set, Tuple, Callable, Optional

logger = logging.getLogger(__name__)


def get_nested_value(data: Dict[str, Any], field_path: str) -> Any:
    """Obtém valor de um campo potencialmente aninhado.

    Args:
        data: Dicionário com os dados.
        field_path: Caminho do campo (ex: 'endereco.cidade' ou 'nome').

    Returns:
        Valor do campo ou None se não encontrado.

    Examples:
        >>> data = {'endereco': {'cidade': 'SP', 'rua': 'Paulista'}, 'nome': 'João'}
        >>> get_nested_value(data, 'nome')
        'João'
        >>> get_nested_value(data, 'endereco.cidade')
        'SP'
        >>> get_nested_value(data, 'nao_existe')
        None
    """
    if not field_path:
        return None

    parts = field_path.split('.')
    current = data

    for part in parts:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
        if current is None:
            return None

    return current


def evaluate_condition(
    condition: Any,
    field_data: Dict[str, Any],
    field_name: str
) -> bool:
    """Avalia se uma condição é satisfeita.

    Args:
        condition: Condição a avaliar. Pode ser:
            - Dict com chaves 'field' e 'equals'/'not_equals'/'in'/'not_in'
            - Callable que recebe field_data e retorna bool
            - None (sempre True)
        field_data: Dados dos campos já processados.
        field_name: Nome do campo atual (para logging).

    Returns:
        True se a condição é satisfeita, False caso contrário.

    Examples:
        >>> data = {'tipo': 'pf', 'status': 'ativo'}
        >>> evaluate_condition({'field': 'tipo', 'equals': 'pf'}, data, 'cpf')
        True
        >>> evaluate_condition({'field': 'tipo', 'equals': 'pj'}, data, 'cnpj')
        False
        >>> evaluate_condition({'field': 'status', 'in': ['ativo', 'pendente']}, data, 'x')
        True
    """
    if condition is None:
        return True

    # Callable condition
    if callable(condition):
        try:
            return bool(condition(field_data))
        except Exception as e:
            logger.warning(
                f"Erro ao avaliar condição callable para campo '{field_name}': {e}"
            )
            return False

    # Dict-based condition
    if isinstance(condition, dict):
        field_path = condition.get('field')
        if not field_path:
            logger.warning(
                f"Condição para campo '{field_name}' não tem 'field' definido"
            )
            return False

        value = get_nested_value(field_data, field_path)

        # equals
        if 'equals' in condition:
            result = value == condition['equals']
            logger.debug(
                f"Campo '{field_name}': {field_path}={value} == {condition['equals']} -> {result}"
            )
            return result

        # not_equals
        if 'not_equals' in condition:
            result = value != condition['not_equals']
            logger.debug(
                f"Campo '{field_name}': {field_path}={value} != {condition['not_equals']} -> {result}"
            )
            return result

        # in (value in list)
        if 'in' in condition:
            result = value in condition['in']
            logger.debug(
                f"Campo '{field_name}': {field_path}={value} in {condition['in']} -> {result}"
            )
            return result

        # not_in
        if 'not_in' in condition:
            result = value not in condition['not_in']
            logger.debug(
                f"Campo '{field_name}': {field_path}={value} not in {condition['not_in']} -> {result}"
            )
            return result

        # exists (campo existe e não é None)
        if 'exists' in condition:
            expected_exists = condition['exists']
            result = (value is not None) == expected_exists
            logger.debug(
                f"Campo '{field_name}': {field_path} exists={value is not None} == {expected_exists} -> {result}"
            )
            return result

        logger.warning(
            f"Condição para campo '{field_name}' não tem operador válido "
            f"(equals, not_equals, in, not_in, exists)"
        )
        return False

    logger.warning(
        f"Condição para campo '{field_name}' tem tipo inválido: {type(condition)}"
    )
    return False


def check_dependencies_exist(
    field_name: str,
    depends_on: List[str],
    all_fields: Set[str]
) -> List[str]:
    """Verifica se todas as dependências existem no modelo.

    Args:
        field_name: Nome do campo.
        depends_on: Lista de campos dos quais este campo depende.
        all_fields: Set com todos os nomes de campos do modelo.

    Returns:
        Lista de dependências inexistentes (vazia se todas existem).
    """
    missing = []
    for dep in depends_on:
        # Suporta campos aninhados (ex: 'endereco.cidade')
        # Verifica apenas o campo raiz
        root_field = dep.split('.')[0]
        if root_field not in all_fields:
            missing.append(dep)

    return missing


def detect_circular_dependencies(
    dependencies: Dict[str, List[str]]
) -> Optional[List[str]]:
    """Detecta dependências circulares usando DFS.

    Args:
        dependencies: Dict mapeando campo -> lista de dependências.

    Returns:
        Lista com ciclo detectado, ou None se não há ciclos.

    Examples:
        >>> deps = {'a': ['b'], 'b': ['c'], 'c': ['a']}
        >>> detect_circular_dependencies(deps)
        ['a', 'b', 'c', 'a']
        >>> deps = {'a': ['b'], 'b': ['c'], 'c': []}
        >>> detect_circular_dependencies(deps)
        None
    """
    # Estado de cada nó: 0 = não visitado, 1 = em progresso, 2 = concluído
    state = {field: 0 for field in dependencies}
    path = []

    def dfs(field: str) -> Optional[List[str]]:
        if state[field] == 2:  # Já processado
            return None
        if state[field] == 1:  # Ciclo detectado
            # Retorna o ciclo completo
            cycle_start = path.index(field)
            return path[cycle_start:] + [field]

        state[field] = 1
        path.append(field)

        for dep in dependencies.get(field, []):
            # Ignora campos aninhados - verifica apenas campo raiz
            root_dep = dep.split('.')[0]
            if root_dep in dependencies:
                cycle = dfs(root_dep)
                if cycle:
                    return cycle

        path.pop()
        state[field] = 2
        return None

    for field in dependencies:
        if state[field] == 0:
            cycle = dfs(field)
            if cycle:
                return cycle

    return None


def topological_sort(dependencies: Dict[str, List[str]]) -> List[str]:
    """Ordena campos baseado em dependências usando ordenação topológica.

    Args:
        dependencies: Dict mapeando campo -> lista de dependências.

    Returns:
        Lista de campos ordenados (dependências primeiro).

    Raises:
        ValueError: Se há dependências circulares.

    Examples:
        >>> deps = {'a': [], 'b': ['a'], 'c': ['a', 'b']}
        >>> topological_sort(deps)
        ['a', 'b', 'c']
    """
    # Detectar ciclos primeiro
    cycle = detect_circular_dependencies(dependencies)
    if cycle:
        raise ValueError(
            f"Dependências circulares detectadas: {' -> '.join(cycle)}"
        )

    # Grau de entrada (in-degree) para cada nó = número de dependências que tem
    in_degree = {}
    for field, deps in dependencies.items():
        # Contar apenas dependências que existem (ignorar campos aninhados parcialmente)
        count = 0
        for dep in deps:
            root_dep = dep.split('.')[0]
            if root_dep in dependencies:
                count += 1
        in_degree[field] = count

    # Fila com nós sem dependências
    queue = [field for field, degree in in_degree.items() if degree == 0]
    result = []

    while queue:
        # Pega o próximo campo sem dependências
        current = queue.pop(0)
        result.append(current)

        # Reduz in-degree dos campos que dependem deste
        for field, deps in dependencies.items():
            # Verifica se o campo depende do current
            has_dependency = False
            for dep in deps:
                root_dep = dep.split('.')[0]
                if root_dep == current:
                    has_dependency = True
                    break

            if has_dependency:
                in_degree[field] -= 1
                if in_degree[field] == 0:
                    queue.append(field)

    return result


def get_field_execution_order(
    pydantic_model,
    field_configs: Dict[str, dict]
) -> Tuple[List[str], Dict[str, List[str]]]:
    """Determina ordem de execução dos campos baseado em dependências.

    Args:
        pydantic_model: Modelo Pydantic.
        field_configs: Dict mapeando campo -> config (com 'depends_on').

    Returns:
        Tupla (ordem_de_execução, mapa_de_dependências).

    Raises:
        ValueError: Se há dependências inválidas ou circulares.
    """
    all_fields = set(pydantic_model.model_fields.keys())
    dependencies = {}

    # Construir mapa de dependências
    for field_name in all_fields:
        config = field_configs.get(field_name, {})
        depends_on = config.get('depends_on', [])

        # Normalizar para lista
        if isinstance(depends_on, str):
            depends_on = [depends_on]
        elif not isinstance(depends_on, list):
            depends_on = []

        # Verificar se dependências existem
        missing = check_dependencies_exist(field_name, depends_on, all_fields)
        if missing:
            raise ValueError(
                f"Campo '{field_name}' depende de campos inexistentes: {missing}"
            )

        dependencies[field_name] = depends_on

    # Ordenar topologicamente
    ordered = topological_sort(dependencies)

    return ordered, dependencies


def should_skip_field(
    field_name: str,
    field_config: dict,
    field_data: Dict[str, Any]
) -> bool:
    """Verifica se um campo deve ser pulado baseado em suas condições.

    Args:
        field_name: Nome do campo.
        field_config: Configuração do campo (com 'condition').
        field_data: Dados dos campos já processados.

    Returns:
        True se o campo deve ser pulado, False caso contrário.
    """
    condition = field_config.get('condition')
    if not condition:
        return False

    # Se a condição não é satisfeita, pular o campo
    satisfied = evaluate_condition(condition, field_data, field_name)

    if not satisfied:
        logger.info(f"Campo '{field_name}' pulado (condição não satisfeita)")

    return not satisfied
