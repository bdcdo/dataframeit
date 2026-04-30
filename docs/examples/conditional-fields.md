# Campos Condicionais

O DataFrameIt suporta execução condicional de campos quando usando `search_per_field=True`. Isso permite que você:

1. **Execute campos condicionalmente** baseado em valores de outros campos
2. **Acesse campos aninhados** nas condições
3. **Controle a ordem de execução** — derivada automaticamente da condição

## Configuração Básica

Use a chave `condition` no `json_schema_extra` de cada campo:

- `condition` (dict): condição para executar o campo. A dependência (e portanto a ordem de execução) é **derivada automaticamente** do campo referenciado em `condition['field']`.
- `condition` (callable): função que recebe os campos já processados e retorna bool. Para que a ordem seja respeitada, declare os campos lidos via `depends_on`.

## Exemplo 1: Pessoa Física vs Jurídica

Este exemplo mostra como processar campos diferentes baseado no tipo de pessoa:

```python
from pydantic import BaseModel, Field
import dataframeit as dfi

class PessoaInfo(BaseModel):
    tipo: str = Field(
        description="Tipo de pessoa: 'pf' para pessoa física ou 'pj' para pessoa jurídica"
    )

    cpf: str = Field(
        description="CPF da pessoa física",
        json_schema_extra={
            'condition': {'field': 'tipo', 'equals': 'pf'}
        }
    )

    cnpj: str = Field(
        description="CNPJ da pessoa jurídica",
        json_schema_extra={
            'condition': {'field': 'tipo', 'equals': 'pj'}
        }
    )

    razao_social: str = Field(
        description="Razão social da empresa",
        json_schema_extra={
            'condition': {'field': 'tipo', 'equals': 'pj'}
        }
    )

# Dados de exemplo
data = [
    "João Silva, CPF 123.456.789-00",
    "Empresa XYZ LTDA, CNPJ 12.345.678/0001-90"
]

# Processar
result = dfi.dataframeit(
    data=data,
    pydantic_model=PessoaInfo,
    user_prompt="Extraia as informações da pessoa ou empresa: {texto}",
    search_per_field=True,  # Necessário para usar condicionais
    search_enabled=True,
    model="gpt-4o-mini",
    provider="openai"
)

print(result)
# Para a primeira linha: tipo='pf', cpf='123.456.789-00', cnpj=None, razao_social=None
# Para a segunda linha: tipo='pj', cpf=None, cnpj='12.345.678/0001-90', razao_social='Empresa XYZ LTDA'
```

## Exemplo 2: Campos Encadeados

Este exemplo mostra como encadear condições em sequência:

```python
class EnderecoInfo(BaseModel):
    pais: str = Field(description="País")

    estado: str = Field(
        description="Estado (apenas para Brasil)",
        json_schema_extra={
            'condition': {'field': 'pais', 'equals': 'Brasil'}
        }
    )

    cep: str = Field(
        description="CEP (apenas para Brasil)",
        json_schema_extra={
            'condition': {'field': 'estado', 'exists': True}
        }
    )

    zip_code: str = Field(
        description="ZIP Code (apenas para outros países)",
        json_schema_extra={
            'condition': {'field': 'pais', 'not_equals': 'Brasil'}
        }
    )
```

## Exemplo 3: Condição Callable com Múltiplas Dependências

Quando a condição combina vários campos, use um callable e declare os campos lidos via `depends_on`:

```python
class PedidoInfo(BaseModel):
    tipo_cliente: str = Field(description="Tipo do cliente: 'novo' ou 'vip'")

    valor_pedido: float = Field(description="Valor total do pedido")

    desconto: float = Field(
        description="Desconto aplicado",
        json_schema_extra={
            'depends_on': ['tipo_cliente', 'valor_pedido'],
            'condition': lambda data: (
                data.get('tipo_cliente') == 'vip' and
                data.get('valor_pedido', 0) > 1000
            )
        }
    )
```

## Operadores de Condição

As condições suportam os seguintes operadores:

### Dicionário com operadores

```python
# Igualdade
{'field': 'tipo', 'equals': 'pf'}

# Diferença
{'field': 'tipo', 'not_equals': 'pj'}

# Está na lista
{'field': 'status', 'in': ['ativo', 'pendente']}

# Não está na lista
{'field': 'status', 'not_in': ['inativo', 'cancelado']}

# Campo existe (não é None)
{'field': 'campo_opcional', 'exists': True}
```

### Função Callable

Para condições mais complexas, use uma função:

```python
{
    'condition': lambda data: (
        data.get('idade', 0) >= 18 and
        data.get('pais') == 'Brasil'
    )
}
```

Lembre-se de declarar `depends_on` com os campos lidos pelo lambda — sem isso, a ordem de execução não é garantida.

## Campos Aninhados em Condições

Você pode acessar campos aninhados usando notação de ponto. A dependência derivada é o campo raiz:

```python
class ProdutoInfo(BaseModel):
    endereco: dict = Field(description="Endereço de entrega")

    taxa_entrega: float = Field(
        description="Taxa de entrega",
        json_schema_extra={
            'condition': {'field': 'endereco.cidade', 'in': ['São Paulo', 'Rio de Janeiro']}
        }
    )
```

## Ordem de Execução

O DataFrameIt deriva a ordem automaticamente do campo referenciado em cada `condition`:

```python
class ModeloEncadeado(BaseModel):
    a: str
    b: str = Field(json_schema_extra={'condition': {'field': 'a', 'equals': 'x'}})
    c: str = Field(json_schema_extra={'condition': {'field': 'b', 'equals': 'y'}})
    d: str = Field(json_schema_extra={'condition': {'field': 'c', 'equals': 'z'}})
```

A ordem de execução será: `a → b → c → d`.

## Detecção de Dependências Circulares

O sistema detecta automaticamente dependências circulares e gera um erro:

```python
class ModeloInvalido(BaseModel):
    a: str = Field(json_schema_extra={'condition': {'field': 'b', 'equals': 'x'}})
    b: str = Field(json_schema_extra={'condition': {'field': 'a', 'equals': 'x'}})

# ValueError: Dependências circulares detectadas: a -> b -> a
```

## Logging e Debug

Para ver detalhes sobre a execução condicional, ative o logging em nível DEBUG:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('dataframeit.agent')
logger.setLevel(logging.DEBUG)
```

Isso mostrará:
- Ordem de execução determinada
- Mapa de dependências
- Campos que foram pulados e por quê
- Avaliação de cada condição

## Considerações de Performance

- Campos condicionais são processados sequencialmente na ordem de dependência
- Campos que são pulados não consomem créditos de busca
- Use condições para otimizar custos pulando campos desnecessários

## Combinando com Outras Configurações

Condicionais funcionam em conjunto com outras configurações per-field:

```python
class InfoCompleta(BaseModel):
    tipo: str

    detalhes_pf: str = Field(
        json_schema_extra={
            'condition': {'field': 'tipo', 'equals': 'pf'},
            'search_depth': 'advanced',  # Busca mais profunda
            'max_results': 10,  # Mais resultados
            'prompt_append': 'Inclua informações detalhadas sobre histórico.'
        }
    )
```

## Limitações

1. Condicionais só funcionam com `search_per_field=True`
2. Não é possível criar dependências de campos que não existem no modelo
3. Dependências circulares não são permitidas
4. Condições são avaliadas uma vez antes de processar cada campo
