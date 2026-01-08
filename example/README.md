# Exemplos do DataFrameIt

Este diretório contém exemplos práticos de uso do DataFrameIt, disponíveis tanto como scripts Python quanto como notebooks Jupyter prontos para rodar no Google Colab.

## Notebooks (Google Colab)

Clique nos badges abaixo para abrir os notebooks diretamente no Google Colab:

| Notebook | Descrição | Colab |
|----------|-----------|-------|
| [01_basic.ipynb](01_basic.ipynb) | Uso básico e análise de sentimento | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bdcdo/dataframeit/blob/main/example/01_basic.ipynb) |
| [02_error_handling.ipynb](02_error_handling.ipynb) | Tratamento de erros e retry | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bdcdo/dataframeit/blob/main/example/02_error_handling.ipynb) |
| [03_resume.ipynb](03_resume.ipynb) | Processamento incremental | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bdcdo/dataframeit/blob/main/example/03_resume.ipynb) |
| [04_custom_placeholder.ipynb](04_custom_placeholder.ipynb) | Placeholder customizado | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bdcdo/dataframeit/blob/main/example/04_custom_placeholder.ipynb) |
| [05_advanced_legal.ipynb](05_advanced_legal.ipynb) | Análise jurídica avançada | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bdcdo/dataframeit/blob/main/example/05_advanced_legal.ipynb) |
| [06_polars.ipynb](06_polars.ipynb) | Usando Polars DataFrame | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bdcdo/dataframeit/blob/main/example/06_polars.ipynb) |
| [07_multiple_data_types.ipynb](07_multiple_data_types.ipynb) | Listas, dicts e Series | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bdcdo/dataframeit/blob/main/example/07_multiple_data_types.ipynb) |
| [08_rate_limiting.ipynb](08_rate_limiting.ipynb) | Rate limiting e proteção | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bdcdo/dataframeit/blob/main/example/08_rate_limiting.ipynb) |

## Scripts Python

Os mesmos exemplos estão disponíveis como scripts Python para execução local:

### Exemplos Básicos

#### 📝 [example_01_basic.py](example_01_basic.py) - Exemplo Básico
**Conceitos**: Fundamentos, análise de sentimento
- Criação de modelo Pydantic simples
- Template de prompt básico
- Processamento de dados sintéticos
- **Ideal para começar!**

#### ⚠️ [example_03_error_handling.py](example_03_error_handling.py) - Tratamento de Erros
**Conceitos**: Resiliência, retry, error tracking
- Verificação de status de processamento
- Análise de erros com `_error_details`
- Configuração de retry customizado
- Filtragem de linhas com erro

#### 🔄 [example_04_resume.py](example_04_resume.py) - Processamento Incremental
**Conceitos**: Resume, datasets grandes, interrupção e retomada
- Uso de `resume=True`
- Salvamento de progresso
- Continuação de processamento interrompido

#### 🔧 [example_05_custom_placeholder.py](example_05_custom_placeholder.py) - Placeholder Customizado
**Conceitos**: Configuração avançada de template
- Uso de placeholder customizado (ex: `{meu_texto}` ao invés de `{texto}`)
- Parâmetro `placeholder`

### Exemplos Avançados

#### ⚖️ [example_06_advanced_legal.py](example_06_advanced_legal.py) - Análise Jurídica Complexa
**Conceitos**: Modelo complexo, classes aninhadas, domínio específico
- Modelo Pydantic com classes aninhadas
- Campos opcionais e condicionais
- Listas, tuplas e tipos Literal
- Template detalhado para domínio jurídico
- **Exemplo de caso real de uso**

#### 🐻 [example_07_polars.py](example_07_polars.py) - Usando Polars
**Conceitos**: Integração com Polars DataFrame
- Conversão automática Polars ↔ Pandas
- Mesmas funcionalidades com Polars

#### 📊 [example_08_multiple_data_types.py](example_08_multiple_data_types.py) - Múltiplos Tipos de Dados
**Conceitos**: Flexibilidade de entrada
- Processamento de listas de textos
- Processamento de dicionários
- Processamento de Series

#### ⏱️ [example_rate_limiting.py](example_rate_limiting.py) - Rate Limiting
**Conceitos**: Controle de taxa de requisições
- Configuração de `rate_limit_delay`
- Proteção contra rate limits
- Combinação com retry

## Como Executar

### No Google Colab (Recomendado)

1. Clique no badge "Open in Colab" do notebook desejado
2. Configure sua API key no Colab Secrets (recomendado) ou diretamente no código
3. Execute as células em ordem

### Localmente

```bash
# Instalar DataFrameIt
pip install dataframeit[google]

# Configurar variável de ambiente
export GOOGLE_API_KEY="sua-chave-google"

# Executar um exemplo
cd example/
python3 example_01_basic.py
```

### Para outros providers

```bash
# OpenAI
pip install dataframeit[openai]
export OPENAI_API_KEY="sua-chave-openai"

# Anthropic
pip install dataframeit[anthropic]
export ANTHROPIC_API_KEY="sua-chave-anthropic"

# Polars (opcional)
pip install dataframeit[google,polars]
```

## Ordem Sugerida de Aprendizado

1. **01_basic** - Entenda os fundamentos
2. **02_error_handling** - Domine o tratamento de erros
3. **03_resume** - Aprenda a trabalhar com datasets grandes
4. **04_custom_placeholder** - Personalize seus templates
5. **05_advanced_legal** - Veja um caso real complexo
6. **06_polars** - Use com Polars se preferir
7. **07_multiple_data_types** - Conheça a flexibilidade de entrada
8. **08_rate_limiting** - Configure proteção contra rate limits

## Dados de Exemplo

- **sample_data.csv** - Dados sintéticos simples para exemplos básicos
- **clusters_saude_*.xlsx** - Dados reais para exemplo avançado de análise jurídica

## Dúvidas?

Consulte a [documentação principal](../README.md) para mais detalhes sobre parâmetros e funcionalidades.
