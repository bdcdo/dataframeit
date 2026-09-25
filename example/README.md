# Exemplos do DataFrameIt

Este diretório contém exemplos práticos de uso do DataFrameIt em notebooks Jupyter prontos para rodar no Google Colab.

## Notebooks

Clique nos badges abaixo para abrir os notebooks diretamente no Google Colab:

| Notebook | Descrição | Colab |
|----------|-----------|-------|
| [01_basic.ipynb](01_basic.ipynb) | Uso básico e análise de sentimento | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bdcdo/dataframeit/blob/main/example/01_basic.ipynb) |
| [02_error_handling.ipynb](02_error_handling.ipynb) | Tratamento de erros e retry | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bdcdo/dataframeit/blob/main/example/02_error_handling.ipynb) |
| [03_resume.ipynb](03_resume.ipynb) | Checkpoint e retomada com `resume=True` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bdcdo/dataframeit/blob/main/example/03_resume.ipynb) |
| [04_custom_placeholder.ipynb](04_custom_placeholder.ipynb) | Prompt com `{texto}` e escolha da coluna de texto | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bdcdo/dataframeit/blob/main/example/04_custom_placeholder.ipynb) |
| [05_advanced_legal.ipynb](05_advanced_legal.ipynb) | Análise jurídica avançada | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bdcdo/dataframeit/blob/main/example/05_advanced_legal.ipynb) |
| [06_polars.ipynb](06_polars.ipynb) | Usando Polars DataFrame | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bdcdo/dataframeit/blob/main/example/06_polars.ipynb) |
| [07_multiple_data_types.ipynb](07_multiple_data_types.ipynb) | Listas, dicts e Series | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bdcdo/dataframeit/blob/main/example/07_multiple_data_types.ipynb) |
| [08_rate_limiting.ipynb](08_rate_limiting.ipynb) | Rate limiting e proteção | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bdcdo/dataframeit/blob/main/example/08_rate_limiting.ipynb) |

## Script

| Arquivo | Descrição |
|---------|-----------|
| [example_09_web_search.py](example_09_web_search.py) | Busca web com Tavily ou Exa (`use_search=True`) |

O script roda localmente: `pip install dataframeit[search,openai]`, configure `TAVILY_API_KEY` e `OPENAI_API_KEY` e execute `python example_09_web_search.py`.

## Como Executar

### No Google Colab (Recomendado)

1. Clique no badge "Open in Colab" do notebook desejado
2. Configure sua API key no Colab Secrets (recomendado) ou diretamente no código
3. Execute as células em ordem

### Localmente

```bash
# Instalar DataFrameIt
pip install dataframeit[openai]

# Configurar variável de ambiente
export OPENAI_API_KEY="sua-chave-openai"

# Abrir Jupyter
jupyter notebook
```

### Para outros providers

```bash
# Google Gemini (usado também na seção 7 do 08_rate_limiting)
pip install dataframeit[google]
export GOOGLE_API_KEY="sua-chave-google"

# Anthropic
pip install dataframeit[anthropic]
export ANTHROPIC_API_KEY="sua-chave-anthropic"

# Polars (opcional)
pip install dataframeit[openai,polars]
```

## Ordem Sugerida de Aprendizado

1. **01_basic** - Entenda os fundamentos
2. **02_error_handling** - Domine o tratamento de erros
3. **03_resume** - Trabalhe com datasets grandes usando checkpoint
4. **04_custom_placeholder** - Controle onde o texto entra no prompt e qual coluna é lida
5. **05_advanced_legal** - Veja um caso real complexo
6. **06_polars** - Use com Polars se preferir
7. **07_multiple_data_types** - Conheça a flexibilidade de entrada
8. **08_rate_limiting** - Configure proteção contra rate limits
9. **example_09_web_search.py** - Enriqueça dados com busca web

## Dados de Exemplo

Todos os exemplos criam os próprios dados no código, sem arquivos externos.

## Dúvidas?

Consulte a [documentação](https://brunodcdo.com.br/dataframeit) para mais detalhes sobre parâmetros e funcionalidades.
