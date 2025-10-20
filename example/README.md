# Exemplos do DataFrameIt

Este diretório contém exemplos práticos de uso do DataFrameIt, organizados por nível de complexidade e funcionalidade.

## Índice de Exemplos

### Exemplos Básicos

#### 📝 [example_01_basic.py](example_01_basic.py) - Exemplo Básico
**Conceitos**: Fundamentos, análise de sentimento
- Criação de modelo Pydantic simples
- Template de prompt básico
- Processamento de dados sintéticos
- **Ideal para começar!**

#### 🤖 [example_02_openai.py](example_02_openai.py) - Usando OpenAI
**Conceitos**: Provider OpenAI, configurações específicas
- Configuração do cliente OpenAI
- Parâmetros `reasoning_effort` e `verbosity`
- Comparação com LangChain

#### ⚠️ [example_03_error_handling.py](example_03_error_handling.py) - Tratamento de Erros
**Conceitos**: Resiliência, retry, error tracking
- Verificação de status de processamento
- Análise de erros com `error_details`
- Configuração de retry customizado
- Filtragem de linhas com erro

#### 🔄 [example_04_resume.py](example_04_resume.py) - Processamento Incremental
**Conceitos**: Resume, datasets grandes, interrupção e retomada
- Uso de `resume=True`
- Salvamento de progresso
- Continuação de processamento interrompido

#### 🔧 [example_05_custom_placeholder.py](example_05_custom_placeholder.py) - Placeholder Customizado
**Conceitos**: Configuração avançada de template
- Uso de placeholder customizado (ex: `{meu_texto}` ao invés de `{documento}`)
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

## Como Executar os Exemplos

### Pré-requisitos

```bash
# Instalar DataFrameIt
pip install dataframeit

# Para exemplos com OpenAI
pip install openai

# Para exemplo com LangChain/Gemini
pip install langchain langchain-core langchain-google-genai

# Para exemplo com Polars
pip install polars
```

### Configurar Variáveis de Ambiente

```bash
# Para exemplos com Gemini (LangChain)
export GOOGLE_API_KEY="sua-chave-google"

# Para exemplos com OpenAI
export OPENAI_API_KEY="sua-chave-openai"
```

### Executar um Exemplo

```bash
cd example/
python3 example_01_basic.py
```

## Ordem Sugerida de Aprendizado

1. **example_01_basic.py** - Entenda os fundamentos
2. **example_02_openai.py** - Aprenda sobre diferentes providers
3. **example_03_error_handling.py** - Domine o tratamento de erros
4. **example_04_resume.py** - Aprenda a trabalhar com datasets grandes
5. **example_05_custom_placeholder.py** - Personalize seus templates
6. **example_06_advanced_legal.py** - Veja um caso real complexo
7. **example_07_polars.py** - Use com Polars se preferir

## Dados de Exemplo

- **sample_data.csv** - Dados sintéticos simples para exemplos básicos
- **clusters_saude_*.xlsx** - Dados reais para exemplo avançado de análise jurídica

## Dúvidas?

Consulte a [documentação principal](../README.md) para mais detalhes sobre parâmetros e funcionalidades.
