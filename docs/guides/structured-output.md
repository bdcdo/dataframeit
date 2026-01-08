# Saída Estruturada

Aprenda a criar modelos Pydantic avançados para extrações complexas.

## Tipos de Campos

### Campos Obrigatórios

```python
from pydantic import BaseModel, Field

class Modelo(BaseModel):
    # Obrigatório sem default
    titulo: str = Field(description="Título do documento")

    # Obrigatório com validação
    nota: int = Field(ge=1, le=10, description="Nota de 1 a 10")
```

### Campos Opcionais

```python
from typing import Optional

class Modelo(BaseModel):
    # Opcional - pode ser None
    observacao: Optional[str] = Field(
        default=None,
        description="Observações adicionais, se houver"
    )
```

### Campos com Valores Fixos (Literal)

```python
from typing import Literal

class Modelo(BaseModel):
    # Só aceita esses valores
    prioridade: Literal['baixa', 'media', 'alta', 'critica']

    # Múltiplas opções
    status: Literal['pendente', 'em_andamento', 'concluido', 'cancelado']
```

!!! tip "Quando usar Literal"
    Use `Literal` sempre que os valores possíveis são conhecidos e finitos. Isso:

    - Força o LLM a escolher entre opções válidas
    - Evita variações indesejadas (ex: "Alto" vs "alta" vs "ALTA")
    - Facilita análises posteriores

### Listas

```python
from typing import List

class Modelo(BaseModel):
    # Lista de strings
    tags: List[str] = Field(description="Tags relevantes")

    # Lista de objetos
    itens: List[Item] = Field(description="Lista de itens extraídos")
```

## Modelos Aninhados

Para estruturas complexas, use modelos aninhados:

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class Endereco(BaseModel):
    rua: str
    cidade: str
    estado: str
    cep: Optional[str] = None

class Contato(BaseModel):
    nome: str
    email: Optional[str] = None
    telefone: Optional[str] = None
    endereco: Optional[Endereco] = None

class Empresa(BaseModel):
    nome: str
    cnpj: Optional[str] = None
    contatos: List[Contato] = Field(description="Lista de contatos da empresa")
    setor: Literal['tecnologia', 'saude', 'financas', 'varejo', 'outro']
```

## Exemplo Real: Análise Jurídica

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Tuple

class Parte(BaseModel):
    """Parte envolvida no processo."""
    nome: str = Field(description="Nome completo da parte")
    tipo: Literal['autor', 'reu', 'terceiro'] = Field(description="Tipo da parte")
    cpf_cnpj: Optional[str] = Field(default=None, description="CPF ou CNPJ")

class Pedido(BaseModel):
    """Pedido feito no processo."""
    descricao: str = Field(description="Descrição do pedido")
    valor: Optional[float] = Field(default=None, description="Valor em reais")
    deferido: Optional[bool] = Field(default=None, description="Se foi deferido")

class DecisaoJudicial(BaseModel):
    """Análise completa de uma decisão judicial."""

    # Identificação
    numero_processo: str = Field(description="Número do processo")
    tribunal: str = Field(description="Tribunal (ex: TJSP, STF)")
    data_decisao: str = Field(description="Data da decisão (YYYY-MM-DD)")

    # Partes
    partes: List[Parte] = Field(description="Partes envolvidas")

    # Mérito
    tipo_decisao: Literal['sentenca', 'acordao', 'despacho', 'decisao_interlocutoria']
    resultado: Literal['procedente', 'improcedente', 'parcialmente_procedente', 'extinto']

    # Pedidos
    pedidos: List[Pedido] = Field(description="Pedidos analisados")

    # Resumo
    resumo: str = Field(description="Resumo da decisão em até 100 palavras")
    fundamentos: List[str] = Field(description="Principais fundamentos jurídicos")

PROMPT = """
Analise a decisão judicial abaixo e extraia todas as informações relevantes.
Seja preciso com datas, valores e nomes.
Se alguma informação não estiver disponível, use null.
"""

resultado = dataframeit(df_decisoes, DecisaoJudicial, PROMPT)
```

## Validações Customizadas

```python
from pydantic import BaseModel, Field, field_validator

class Documento(BaseModel):
    cpf: str = Field(description="CPF no formato XXX.XXX.XXX-XX")
    email: str = Field(description="Email válido")

    @field_validator('cpf')
    @classmethod
    def validar_cpf(cls, v):
        # Remove caracteres não numéricos
        numeros = ''.join(filter(str.isdigit, v))
        if len(numeros) != 11:
            raise ValueError('CPF deve ter 11 dígitos')
        return v
```

!!! warning "Cuidado com validações"
    Validações muito restritivas podem causar erros frequentes. Use com moderação.

## Boas Práticas

1. **Use descrições claras**: O LLM usa as descrições para entender o que extrair

2. **Prefira Literal a str**: Quando os valores são conhecidos, use `Literal`

3. **Use Optional para campos incertos**: Se a informação pode não existir, marque como `Optional`

4. **Quebre modelos complexos**: Use modelos aninhados para melhor organização

5. **Teste com exemplos**: Valide seu modelo com textos reais antes de processar tudo
