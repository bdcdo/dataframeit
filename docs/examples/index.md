# Exemplos

Exemplos práticos em Jupyter Notebooks para aprender DataFrameIt.

## Notebooks Disponíveis

Os exemplos estão organizados do básico ao avançado. Recomendamos seguir na ordem.

### 1. Básico
**[01_basic.ipynb](https://github.com/bdcdo/dataframeit/blob/main/example/01_basic.ipynb)**

Introdução ao DataFrameIt com análise de sentimentos.

- Criar modelo Pydantic simples
- Processar DataFrame básico
- Entender a saída

```python
from pydantic import BaseModel
from typing import Literal

class Sentimento(BaseModel):
    sentimento: Literal['positivo', 'negativo', 'neutro']
```

---

### 2. Tratamento de Erros
**[02_error_handling.ipynb](https://github.com/bdcdo/dataframeit/blob/main/example/02_error_handling.ipynb)**

Como lidar com erros e configurar retry.

- Configurar `max_retries`, `base_delay`, `max_delay`
- Verificar coluna `_dataframeit_status`
- Analisar `_error_details`

---

### 3. Processamento Incremental
**[03_resume.ipynb](https://github.com/bdcdo/dataframeit/blob/main/example/03_resume.ipynb)**

Continuar processamento de onde parou.

- Usar `resume=True`
- Salvar e carregar resultados parciais
- Reprocessar apenas linhas com erro

---

### 4. Placeholder Customizado
**[04_custom_placeholder.ipynb](https://github.com/bdcdo/dataframeit/blob/main/example/04_custom_placeholder.ipynb)**

Controlar onde o texto aparece no prompt.

- Usar `{texto}` no template
- Criar prompts complexos multi-parte

---

### 5. Caso Avançado: Análise Jurídica
**[05_advanced_legal.ipynb](https://github.com/bdcdo/dataframeit/blob/main/example/05_advanced_legal.ipynb)**

Exemplo real com modelo Pydantic complexo.

- Modelos aninhados
- Campos opcionais
- Listas de objetos
- Extração de múltiplas entidades

```python
class Parte(BaseModel):
    nome: str
    tipo: Literal['autor', 'reu']

class Decisao(BaseModel):
    partes: List[Parte]
    resultado: Literal['procedente', 'improcedente']
```

---

### 6. Polars
**[06_polars.ipynb](https://github.com/bdcdo/dataframeit/blob/main/example/06_polars.ipynb)**

Usar DataFrameIt com Polars ao invés de Pandas.

- Entrada com `polars.DataFrame`
- Saída preserva tipo Polars

---

### 7. Múltiplos Tipos de Dados
**[07_multiple_data_types.ipynb](https://github.com/bdcdo/dataframeit/blob/main/example/07_multiple_data_types.ipynb)**

Processar diferentes tipos de entrada.

- Listas
- Dicionários
- Series

---

### 8. Rate Limiting
**[08_rate_limiting.ipynb](https://github.com/bdcdo/dataframeit/blob/main/example/08_rate_limiting.ipynb)**

Controlar taxa de requisições.

- Configurar `rate_limit_delay`
- Usar `parallel_requests`
- Combinar para máxima eficiência

---

## Executando os Exemplos

### 1. Clone o Repositório

```bash
git clone https://github.com/bdcdo/dataframeit.git
cd dataframeit
```

### 2. Instale as Dependências

```bash
pip install dataframeit[google]
pip install jupyter
```

### 3. Configure sua API Key

```bash
export GOOGLE_API_KEY="sua-chave"
```

### 4. Execute o Jupyter

```bash
jupyter notebook example/
```

---

## Contribuindo com Exemplos

Se você criou um exemplo interessante, considere contribuir! Abra uma issue ou pull request no [GitHub](https://github.com/bdcdo/dataframeit).
