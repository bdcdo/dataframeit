# Conditional Fields

With `use_search=True` and `search_per_field=True`, each model field is filled by its own agent call. In that mode, a field can depend on the value of another:

1. **Conditional execution**: the field is only requested from the LLM when the condition holds; otherwise it stays `None` and spends no search.
2. **Nested fields** can be read in the condition with dot notation.
3. **Execution order** is derived from the condition: the field that is read runs before the field that depends on it.

## Basic Setup

Use the `condition` key in the field's `json_schema_extra`:

- `condition` (dict): the dependency, and therefore the execution order, is derived from the field in `condition['field']`.
- `condition` (callable): receives the fields already filled and returns a bool. Declare the fields it reads in `depends_on`, so they run first.

Since a skipped field stays `None`, declare it as optional (`str | None = None`).

## Example 1: Individual vs Company

```python
from pydantic import BaseModel, Field
from dataframeit import dataframeit

class PessoaInfo(BaseModel):
    tipo: str = Field(
        description="Person type: 'pf' for an individual or 'pj' for a company"
    )
    cpf: str | None = Field(
        default=None,
        description="Individual's CPF",
        json_schema_extra={'condition': {'field': 'tipo', 'equals': 'pf'}},
    )
    cnpj: str | None = Field(
        default=None,
        description="Company's CNPJ",
        json_schema_extra={'condition': {'field': 'tipo', 'equals': 'pj'}},
    )
    razao_social: str | None = Field(
        default=None,
        description="Company's legal name",
        json_schema_extra={'condition': {'field': 'tipo', 'equals': 'pj'}},
    )

textos = [
    "João Silva, CPF 123.456.789-00",
    "Empresa XYZ LTDA, CNPJ 12.345.678/0001-90",
]

result = dataframeit(
    textos,
    PessoaInfo,
    "Extract the person's or company's information: {texto}",
    use_search=True,
    search_per_field=True,  # condition requires per-field search
)
# Row 1: tipo='pf', cpf='123.456.789-00', cnpj=None, razao_social=None
# Row 2: tipo='pj', cpf=None, cnpj='12.345.678/0001-90', razao_social='Empresa XYZ LTDA'
```

## Example 2: Chained Fields

```python
class EnderecoInfo(BaseModel):
    pais: str = Field(description="Country")
    estado: str | None = Field(
        default=None,
        description="State (Brazil only)",
        json_schema_extra={'condition': {'field': 'pais', 'equals': 'Brasil'}},
    )
    cep: str | None = Field(
        default=None,
        description="CEP (Brazil only)",
        json_schema_extra={'condition': {'field': 'estado', 'exists': True}},
    )
    zip_code: str | None = Field(
        default=None,
        description="Postal code (other countries only)",
        json_schema_extra={'condition': {'field': 'pais', 'not_equals': 'Brasil'}},
    )
```

The execution order is `pais → estado → cep` and `pais → zip_code`.

## Example 3: Callable Condition with Several Dependencies

When the condition combines several fields, use a callable and declare the fields it reads in `depends_on`:

```python
class PedidoInfo(BaseModel):
    tipo_cliente: str = Field(description="Customer type: 'novo' or 'vip'")
    valor_pedido: float = Field(description="Order total")
    desconto: float | None = Field(
        default=None,
        description="Discount applied",
        json_schema_extra={
            'depends_on': ['tipo_cliente', 'valor_pedido'],
            'condition': lambda dados: (
                dados.get('tipo_cliente') == 'vip'
                and (dados.get('valor_pedido') or 0) > 1000
            ),
        },
    )
```

## Condition Operators

```python
# Equality
{'field': 'tipo', 'equals': 'pf'}

# Inequality
{'field': 'tipo', 'not_equals': 'pj'}

# In the list
{'field': 'status', 'in': ['ativo', 'pendente']}

# Not in the list
{'field': 'status', 'not_in': ['inativo', 'cancelado']}

# Field filled (not None)
{'field': 'campo_opcional', 'exists': True}
```

Without `depends_on`, the order of a callable is not guaranteed.

## Nested Fields in Conditions

A condition can read a field of a nested model with dot notation. The derived dependency is the root field:

```python
class Endereco(BaseModel):
    cidade: str = Field(description="City")
    uf: str = Field(description="State")

class Entrega(BaseModel):
    endereco: Endereco = Field(description="Delivery address")
    taxa_entrega: float | None = Field(
        default=None,
        description="Delivery fee",
        json_schema_extra={
            'condition': {'field': 'endereco.cidade', 'in': ['São Paulo', 'Rio de Janeiro']}
        },
    )
```

The condition belongs to a top-level field. `condition` or `depends_on` on a field of a nested model, or of a list item, raises `ValueError` before processing.

## Circular Dependencies

A cycle is detected before the first row:

```python
class ModeloInvalido(BaseModel):
    a: str | None = Field(default=None, json_schema_extra={'condition': {'field': 'b', 'equals': 'x'}})
    b: str | None = Field(default=None, json_schema_extra={'condition': {'field': 'a', 'equals': 'x'}})

# ValueError: Dependências circulares detectadas: a -> b -> a
```

## Logging and Debugging

The execution order, the dependencies and each condition's evaluation are logged at DEBUG level, and the skipped fields at INFO, on the `dataframeit.conditional` and `dataframeit.agent` loggers:

```python
import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger('dataframeit').setLevel(logging.DEBUG)
```

## Combining with Other Settings

A condition works alongside the other per-field search keys:

```python
class InfoCompleta(BaseModel):
    tipo: str
    detalhes_pf: str | None = Field(
        default=None,
        json_schema_extra={
            'condition': {'field': 'tipo', 'equals': 'pf'},
            'search_depth': 'advanced',
            'max_results': 10,
            'max_search_calls': 3,
            'prompt_append': 'Include detailed history.',
        },
    )
```

## Limitations

1. Conditions only work with `use_search=True` and `search_per_field=True`; otherwise, `condition` or `depends_on` in the model raises `ValueError`.
2. With `search_groups`, a grouped field's condition is evaluated before the group call, and a field whose condition is false is left out of it; if the condition depends on another field of the same group, it is evaluated with the group's answer, and a field whose condition is false stays `None`.
3. A condition can only read fields that exist in the model.
4. Circular dependencies are not allowed, including between a group and fields outside it.
5. Each condition is evaluated once: before the field's call, or after the group's answer when it depends on another field of the same group.
6. Fields with conditions are processed in sequence, in dependency order; a skipped field spends no search.
