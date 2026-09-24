# Campos Condicionais

Com `use_search=True` e `search_per_field=True`, cada campo do modelo é preenchido por uma chamada própria do agente. Nesse modo, um campo pode depender do valor de outro:

1. **Execução condicional**: o campo só é pedido ao LLM quando a condição vale; caso contrário, fica `None` e não gasta busca.
2. **Campos aninhados** podem ser lidos na condição, com notação de ponto.
3. **Ordem de execução** derivada da condição: o campo lido vem antes do campo que depende dele.

## Configuração Básica

Use a chave `condition` no `json_schema_extra` do campo:

- `condition` (dict): a dependência, e portanto a ordem de execução, é derivada do campo em `condition['field']`.
- `condition` (callable): recebe os campos já preenchidos e devolve bool. Declare os campos lidos em `depends_on`, para que venham antes.

Como o campo pulado fica `None`, declare-o como opcional (`str | None = None`).

## Exemplo 1: Pessoa Física vs Jurídica

```python
from pydantic import BaseModel, Field
from dataframeit import dataframeit

class PessoaInfo(BaseModel):
    tipo: str = Field(
        description="Tipo de pessoa: 'pf' para pessoa física ou 'pj' para pessoa jurídica"
    )
    cpf: str | None = Field(
        default=None,
        description="CPF da pessoa física",
        json_schema_extra={'condition': {'field': 'tipo', 'equals': 'pf'}},
    )
    cnpj: str | None = Field(
        default=None,
        description="CNPJ da pessoa jurídica",
        json_schema_extra={'condition': {'field': 'tipo', 'equals': 'pj'}},
    )
    razao_social: str | None = Field(
        default=None,
        description="Razão social da empresa",
        json_schema_extra={'condition': {'field': 'tipo', 'equals': 'pj'}},
    )

textos = [
    "João Silva, CPF 123.456.789-00",
    "Empresa XYZ LTDA, CNPJ 12.345.678/0001-90",
]

resultado = dataframeit(
    textos,
    PessoaInfo,
    "Extraia as informações da pessoa ou empresa: {texto}",
    use_search=True,
    search_per_field=True,  # condition exige busca por campo
)
# Linha 1: tipo='pf', cpf='123.456.789-00', cnpj=None, razao_social=None
# Linha 2: tipo='pj', cpf=None, cnpj='12.345.678/0001-90', razao_social='Empresa XYZ LTDA'
```

## Exemplo 2: Campos Encadeados

```python
class EnderecoInfo(BaseModel):
    pais: str = Field(description="País")
    estado: str | None = Field(
        default=None,
        description="Estado (apenas para Brasil)",
        json_schema_extra={'condition': {'field': 'pais', 'equals': 'Brasil'}},
    )
    cep: str | None = Field(
        default=None,
        description="CEP (apenas para Brasil)",
        json_schema_extra={'condition': {'field': 'estado', 'exists': True}},
    )
    zip_code: str | None = Field(
        default=None,
        description="Código postal (apenas para outros países)",
        json_schema_extra={'condition': {'field': 'pais', 'not_equals': 'Brasil'}},
    )
```

A ordem de execução é `pais → estado → cep` e `pais → zip_code`.

## Exemplo 3: Condição Callable com Múltiplas Dependências

Quando a condição combina vários campos, use um callable e declare os campos lidos em `depends_on`:

```python
class PedidoInfo(BaseModel):
    tipo_cliente: str = Field(description="Tipo do cliente: 'novo' ou 'vip'")
    valor_pedido: float = Field(description="Valor total do pedido")
    desconto: float | None = Field(
        default=None,
        description="Desconto aplicado",
        json_schema_extra={
            'depends_on': ['tipo_cliente', 'valor_pedido'],
            'condition': lambda dados: (
                dados.get('tipo_cliente') == 'vip'
                and (dados.get('valor_pedido') or 0) > 1000
            ),
        },
    )
```

## Operadores de Condição

```python
# Igualdade
{'field': 'tipo', 'equals': 'pf'}

# Diferença
{'field': 'tipo', 'not_equals': 'pj'}

# Está na lista
{'field': 'status', 'in': ['ativo', 'pendente']}

# Não está na lista
{'field': 'status', 'not_in': ['inativo', 'cancelado']}

# Campo preenchido (não é None)
{'field': 'campo_opcional', 'exists': True}
```

Sem `depends_on`, a ordem de um callable não é garantida.

## Campos Aninhados em Condições

A condição pode ler um campo de modelo aninhado com notação de ponto. A dependência derivada é o campo raiz:

```python
class Endereco(BaseModel):
    cidade: str = Field(description="Cidade")
    uf: str = Field(description="UF")

class Entrega(BaseModel):
    endereco: Endereco = Field(description="Endereço de entrega")
    taxa_entrega: float | None = Field(
        default=None,
        description="Taxa de entrega",
        json_schema_extra={
            'condition': {'field': 'endereco.cidade', 'in': ['São Paulo', 'Rio de Janeiro']}
        },
    )
```

A condição fica no campo de nível superior. `condition` ou `depends_on` num campo de modelo aninhado, ou de item de lista, levanta `ValueError` antes de processar.

## Dependências Circulares

Um ciclo é detectado antes da primeira linha:

```python
class ModeloInvalido(BaseModel):
    a: str | None = Field(default=None, json_schema_extra={'condition': {'field': 'b', 'equals': 'x'}})
    b: str | None = Field(default=None, json_schema_extra={'condition': {'field': 'a', 'equals': 'x'}})

# ValueError: Dependências circulares detectadas: a -> b -> a
```

## Logging e Debug

A ordem de execução, as dependências e a avaliação de cada condição saem em nível DEBUG, e os campos pulados, em INFO, nos loggers `dataframeit.conditional` e `dataframeit.agent`:

```python
import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger('dataframeit').setLevel(logging.DEBUG)
```

## Combinando com Outras Configurações

A condição convive com as outras chaves de busca por campo:

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
            'prompt_append': 'Inclua informações detalhadas sobre histórico.',
        },
    )
```

## Limitações

1. Condicionais só funcionam com `use_search=True` e `search_per_field=True`; sem isso, `condition` ou `depends_on` no modelo levanta `ValueError`.
2. Com `search_groups`, a condição de um campo agrupado é avaliada antes da chamada do grupo, e o campo com condição falsa fica fora dela; se a condição depende de outro campo do mesmo grupo, ela é avaliada com a resposta do grupo, e o campo com condição falsa fica `None`.
3. A condição só pode ler campos que existem no modelo.
4. Dependências circulares não são permitidas, inclusive entre um grupo e campos de fora dele.
5. Cada condição é avaliada uma vez: antes da chamada do campo, ou depois da resposta do grupo quando depende de outro campo do mesmo grupo.
6. Os campos com condição são processados em sequência, na ordem das dependências; um campo pulado não consome busca.
