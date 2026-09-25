"""Sistema de condicionais para execução condicional de campos."""

import logging
from typing import Any, get_args, get_origin

from .utils import get_nested_pydantic_models, resolve_forward_refs

logger = logging.getLogger(__name__)

# Chaves de configuração per-field reconhecidas em json_schema_extra
_FIELD_CONFIG_KEYS = (
    "prompt",
    "prompt_replace",
    "prompt_append",
    "search_depth",
    "max_results",
    "max_search_calls",
)

# Chaves de execução condicional em json_schema_extra. Só call_agent_per_field
# e call_agent_per_group as aplicam, e só nos campos de primeiro nível.
_CONDITIONAL_KEYS = ("condition", "depends_on")


def _collect_configured_fields(pydantic_model, prefix: str = "", _visited: set = None) -> list:
    """Coleta todos os campos com json_schema_extra de busca, incluindo aninhados.

    Args:
        pydantic_model: Modelo Pydantic a analisar.
        prefix: Prefixo do caminho (usado internamente para recursão).
        _visited: Conjunto de modelos já visitados (previne loops infinitos).

    Returns:
        Lista de tuplas: (path, field_name, field_info, parent_model, has_config)
        Ex: ("pedidos.info_medicamento", "status_anvisa_atual", <FieldInfo>, InformacoesMedicamento, True)
    """
    if _visited is None:
        _visited = set()

    # Evitar loops em modelos auto-referenciais. O conjunto guarda só os
    # ancestrais do caminho: o mesmo modelo em dois campos irmãos
    # (residencial e comercial) é percorrido nos dois.
    model_id = id(pydantic_model)
    if model_id in _visited:
        return []
    _visited = _visited | {model_id}

    results = []

    for field_name, field_info in pydantic_model.model_fields.items():
        path = f"{prefix}.{field_name}" if prefix else field_name

        extra = field_info.json_schema_extra
        if isinstance(extra, dict) and any(k in extra for k in _FIELD_CONFIG_KEYS):
            results.append((path, field_name, field_info, pydantic_model, True))

        annotation = resolve_forward_refs(field_info.annotation, pydantic_model)
        for nested_model in get_nested_pydantic_models(annotation):
            results.extend(_collect_configured_fields(nested_model, path, _visited))

    return results


def _list_layers(annotation, target) -> int | None:
    """Quantas listas envolvem `target` na anotação: list[list[X]] dá 2.

    Só conta os ramos que chegam ao modelo; em Union[list[str], X], a lista
    de strings não envolve X. Devolve None quando `target` não aparece.
    """
    if annotation is target:
        return 0
    depths = [
        depth
        for depth in (_list_layers(arg, target) for arg in get_args(annotation))
        if depth is not None
    ]
    if not depths:
        return None
    return max(depths) + (1 if get_origin(annotation) is list else 0)


def _walk_fields(pydantic_model, prefix: str = "", list_depth: int = 0, _visited: set = None):
    """Percorre todos os campos, inclusive aninhados, com a profundidade de lista.

    Yields:
        Tuplas (path, field_info, list_depth), em que list_depth conta quantos
        List[Model] o caminho atravessa até o modelo que contém o campo.
    """
    if _visited is None:
        _visited = set()
    if id(pydantic_model) in _visited:
        return
    _visited = _visited | {id(pydantic_model)}

    for field_name, field_info in pydantic_model.model_fields.items():
        path = f"{prefix}.{field_name}" if prefix else field_name
        yield path, field_info, list_depth

        annotation = resolve_forward_refs(field_info.annotation, pydantic_model)
        for nested_model in get_nested_pydantic_models(annotation):
            yield from _walk_fields(
                nested_model,
                path,
                list_depth + (_list_layers(annotation, nested_model) or 0),
                _visited,
            )


def get_nested_value(data: dict[str, Any], field_path: str) -> Any:
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

    parts = field_path.split(".")
    current = data

    for part in parts:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
        if current is None:
            return None

    return current


def evaluate_condition(condition: Any, field_data: dict[str, Any], field_name: str) -> bool:
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
            logger.warning(f"Erro ao avaliar condição callable para campo '{field_name}': {e}")
            return False

    # Dict-based condition
    if isinstance(condition, dict):
        field_path = condition.get("field")
        if not field_path:
            logger.warning(f"Condição para campo '{field_name}' não tem 'field' definido")
            return False

        value = get_nested_value(field_data, field_path)

        # equals
        if "equals" in condition:
            result = value == condition["equals"]
            logger.debug(
                f"Campo '{field_name}': {field_path}={value} == {condition['equals']} -> {result}"
            )
            return result

        # not_equals
        if "not_equals" in condition:
            result = value != condition["not_equals"]
            logger.debug(
                f"Campo '{field_name}': {field_path}={value} != {condition['not_equals']} -> {result}"
            )
            return result

        # in (value in list)
        if "in" in condition:
            result = value in condition["in"]
            logger.debug(
                f"Campo '{field_name}': {field_path}={value} in {condition['in']} -> {result}"
            )
            return result

        # not_in
        if "not_in" in condition:
            result = value not in condition["not_in"]
            logger.debug(
                f"Campo '{field_name}': {field_path}={value} not in {condition['not_in']} -> {result}"
            )
            return result

        # exists (campo existe e não é None)
        if "exists" in condition:
            expected_exists = condition["exists"]
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

    logger.warning(f"Condição para campo '{field_name}' tem tipo inválido: {type(condition)}")
    return False


def check_dependencies_exist(
    field_name: str, depends_on: list[str], all_fields: set[str]
) -> list[str]:
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
        root_field = dep.split(".")[0]
        if root_field not in all_fields:
            missing.append(dep)

    return missing


def detect_circular_dependencies(dependencies: dict[str, list[str]]) -> list[str] | None:
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

    def dfs(field: str) -> list[str] | None:
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
            root_dep = dep.split(".")[0]
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


def topological_sort(dependencies: dict[str, list[str]]) -> list[str]:
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
        raise ValueError(f"Dependências circulares detectadas: {' -> '.join(cycle)}")

    # Cada campo depende de um conjunto de raízes: 'endereco.cidade' e
    # 'endereco.uf' são a mesma dependência, e contá-las duas vezes deixaria o
    # grau de entrada acima de zero para sempre, com o campo fora da ordem.
    root_deps = {
        field: {dep.split(".")[0] for dep in deps if dep.split(".")[0] in dependencies}
        for field, deps in dependencies.items()
    }
    in_degree = {field: len(roots) for field, roots in root_deps.items()}

    # Fila com nós sem dependências
    queue = [field for field, degree in in_degree.items() if degree == 0]
    result = []

    while queue:
        current = queue.pop(0)
        result.append(current)

        for field, roots in root_deps.items():
            if current in roots:
                in_degree[field] -= 1
                if in_degree[field] == 0:
                    queue.append(field)

    return result


def _resolve_depends_on(field_name: str, config: dict) -> list[str]:
    """Resolve as dependências de um campo a partir de sua configuração.

    Regras:
    1. Sem `condition`, `depends_on` é ignorado (com warning) — sem condition, ordem não afeta o resultado.
    2. Com `condition` dict, a dep do campo raiz de `condition['field']` é unida ao `depends_on` explícito.
    3. Com `condition` callable sem `depends_on`, retorna lista vazia (com warning).
    """
    explicit = config.get("depends_on") or []
    if isinstance(explicit, str):
        explicit = [explicit]
    elif not isinstance(explicit, list):
        explicit = []

    condition = config.get("condition")

    if condition is None:
        if explicit:
            logger.warning(
                f"Campo '{field_name}' tem 'depends_on' mas não tem 'condition' — "
                f"depends_on será ignorado (sem condition, ordem não afeta o resultado)."
            )
        return []

    derived: list[str] = []
    if isinstance(condition, dict):
        field_path = condition.get("field")
        if field_path:
            derived = [field_path.split(".")[0]]
    elif callable(condition) and not explicit:
        logger.warning(
            f"Campo '{field_name}' tem 'condition' callable sem 'depends_on' — "
            f"a ordem de execução não é garantida. Declare 'depends_on' com os campos "
            f"lidos pela função."
        )

    result = list(explicit)
    for dep in derived:
        if dep not in result:
            result.append(dep)
    return result


def get_field_execution_order(
    pydantic_model, field_configs: dict[str, dict]
) -> tuple[list[str], dict[str, list[str]]]:
    """Determina ordem de execução dos campos baseado em dependências.

    Dependências são derivadas automaticamente de `condition` (quando dict)
    ou declaradas explicitamente via `depends_on` (necessário apenas para
    `condition` callable).

    Args:
        pydantic_model: Modelo Pydantic.
        field_configs: Dict mapeando campo -> config (com 'condition' e/ou 'depends_on').

    Returns:
        Tupla (ordem_de_execução, mapa_de_dependências).

    Raises:
        ValueError: Se há dependências inválidas ou circulares.
    """
    all_fields = set(pydantic_model.model_fields.keys())
    dependencies = {}

    # A ordem do modelo, e não a do set, fixa a ordem entre campos
    # independentes: topological_sort segue a ordem de inserção do dict.
    for field_name in pydantic_model.model_fields:
        config = field_configs.get(field_name, {})
        depends_on = _resolve_depends_on(field_name, config)

        missing = check_dependencies_exist(field_name, depends_on, all_fields)
        if missing:
            raise ValueError(f"Campo '{field_name}' depende de campos inexistentes: {missing}")

        dependencies[field_name] = depends_on

    ordered = topological_sort(dependencies)

    return ordered, dependencies


def get_group_execution_units(pydantic_model, groups: dict, dependencies: dict[str, list[str]]):
    """Monta as unidades de execução do modo por grupo e suas dependências.

    Cada grupo e cada campo fora de grupo é uma unidade. As chaves são índices
    em texto porque topological_sort interpreta '.' como caminho aninhado, e
    nome de grupo é livre.

    Returns:
        Tupla (units, unit_of_field, unit_dependencies): a lista de
        (tipo, nome, config do grupo ou None), o índice da unidade de cada
        campo e o grafo de dependências entre unidades.

    Raises:
        ValueError: Se há ciclo entre unidades, inclusive entre um grupo e um
            campo de fora dele.
    """
    grouped_fields = set()
    for group_config in groups.values():
        grouped_fields.update(group_config.fields)

    units = [("group", group_name, group_config) for group_name, group_config in groups.items()]
    units += [
        ("field", field_name, None)
        for field_name in pydantic_model.model_fields
        if field_name not in grouped_fields
    ]

    unit_of_field = {}
    for index, (kind, name, group_config) in enumerate(units):
        for field_name in group_config.fields if kind == "group" else [name]:
            unit_of_field[field_name] = str(index)

    unit_dependencies = {str(index): [] for index in range(len(units))}
    for field_name, deps in dependencies.items():
        unit = unit_of_field[field_name]
        for dep in deps:
            dep_unit = unit_of_field[dep.split(".")[0]]
            if dep_unit != unit and dep_unit not in unit_dependencies[unit]:
                unit_dependencies[unit].append(dep_unit)

    cycle = detect_circular_dependencies(unit_dependencies)
    if cycle:
        labels = [
            f"grupo '{units[int(key)][1]}'"
            if units[int(key)][0] == "group"
            else f"'{units[int(key)][1]}'"
            for key in cycle
        ]
        raise ValueError(f"Dependências circulares entre grupos e campos: {' -> '.join(labels)}")

    return units, unit_of_field, unit_dependencies


def should_skip_field(field_name: str, field_config: dict, field_data: dict[str, Any]) -> bool:
    """Verifica se um campo deve ser pulado baseado em suas condições.

    Args:
        field_name: Nome do campo.
        field_config: Configuração do campo (com 'condition').
        field_data: Dados dos campos já processados.

    Returns:
        True se o campo deve ser pulado, False caso contrário.
    """
    condition = field_config.get("condition")
    if not condition:
        return False

    # Se a condição não é satisfeita, pular o campo
    satisfied = evaluate_condition(condition, field_data, field_name)

    if not satisfied:
        logger.info(f"Campo '{field_name}' pulado (condição não satisfeita)")

    return not satisfied
