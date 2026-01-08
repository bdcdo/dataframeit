# Exemplos do DataFrameIt

Este diretório contém exemplos práticos de uso do DataFrameIt em notebooks Jupyter prontos para rodar no Google Colab.

## Notebooks

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

# Abrir Jupyter
jupyter notebook
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
