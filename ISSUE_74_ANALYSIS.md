# Análise da Issue 74 - Warnings com stacklevel inconsistente

## Problema Identificado

Os warnings no código do `dataframeit` não utilizam o parâmetro `stacklevel` de forma consistente, fazendo com que alguns warnings apontem para linhas internas do código da biblioteca ao invés de apontar para o código do usuário.

## Como funciona o stacklevel

O parâmetro `stacklevel` do `warnings.warn()` controla qual linha do código será reportada como origem do warning:

- **stacklevel=1 (padrão)**: aponta para a linha do próprio `warnings.warn()`
- **stacklevel=2**: aponta para quem chamou a função que contém `warnings.warn()`
- **stacklevel=3**: aponta para quem chamou o caller da função
- **stacklevel=N**: sobe N níveis na pilha de chamadas

**Melhor prática**: O warning deve apontar para o código do usuário, não para código interno da biblioteca.

## Warnings encontrados e stacklevel correto

### 1. core.py:216 - Conflito de colunas
**Localização**: Dentro de `dataframeit()`
**Cadeia**: `user_code -> dataframeit() -> warnings.warn()`
**Stacklevel atual**: Nenhum (usa padrão = 1)
**Stacklevel correto**: `2` (para apontar para user_code)

```python
# ANTES
warnings.warn(
    f"Colunas {existing_cols} já existem. Use resume=True para continuar ou renomeie-as."
)

# DEPOIS
warnings.warn(
    f"Colunas {existing_cols} já existem. Use resume=True para continuar ou renomeie-as.",
    stacklevel=2
)
```

---

### 2. core.py:584 - Falha ao processar linha (sequencial)
**Localização**: Dentro de `_process_rows()`
**Cadeia**: `user_code -> dataframeit() -> _process_rows() -> warnings.warn()`
**Stacklevel atual**: Nenhum (usa padrão = 1)
**Stacklevel correto**: `3` (para apontar para user_code)

```python
# ANTES
warnings.warn(f"Falha ao processar linha {idx}.")

# DEPOIS
warnings.warn(f"Falha ao processar linha {idx}.", stacklevel=3)
```

---

### 3. core.py:754-757 - Rate limit detectado
**Localização**: Dentro de `_process_rows_parallel()`
**Cadeia**: `user_code -> dataframeit() -> _process_rows_parallel() -> warnings.warn()`
**Stacklevel atual**: `2` ⚠️ **INCORRETO**
**Stacklevel correto**: `3` (para apontar para user_code)

```python
# ANTES
warnings.warn(
    f"Rate limit detectado! Reduzindo workers de {old_workers} para {current_workers}.",
    stacklevel=2
)

# DEPOIS
warnings.warn(
    f"Rate limit detectado! Reduzindo workers de {old_workers} para {current_workers}.",
    stacklevel=3
)
```

---

### 4. core.py:772 - Falha ao processar linha (paralelo)
**Localização**: Dentro de `_process_rows_parallel()`
**Cadeia**: `user_code -> dataframeit() -> _process_rows_parallel() -> warnings.warn()`
**Stacklevel atual**: Nenhum (usa padrão = 1)
**Stacklevel correto**: `3` (para apontar para user_code)

```python
# ANTES
warnings.warn(f"Falha ao processar linha {idx}.")

# DEPOIS
warnings.warn(f"Falha ao processar linha {idx}.", stacklevel=3)
```

---

### 5. core.py:802 - Erro inesperado no executor
**Localização**: Dentro de `_process_rows_parallel()`
**Cadeia**: `user_code -> dataframeit() -> _process_rows_parallel() -> warnings.warn()`
**Stacklevel atual**: Nenhum (usa padrão = 1)
**Stacklevel correto**: `3` (para apontar para user_code)

```python
# ANTES
warnings.warn(f"Erro inesperado no executor: {e}")

# DEPOIS
warnings.warn(f"Erro inesperado no executor: {e}", stacklevel=3)
```

---

### 6. errors.py:469-471 - Erro não-recuperável
**Localização**: Dentro de `retry_with_backoff()`
**Cadeia**: `user_code -> dataframeit() -> _process_rows*() -> call_langchain() -> retry_with_backoff() -> warnings.warn()`
**Stacklevel atual**: `3` ⚠️ **INCORRETO** (aponta para _process_rows)
**Stacklevel correto**: `5` (para apontar para user_code)

```python
# ANTES
warnings.warn(
    f"Erro não-recuperável detectado ({error_name}). Não será feito retry.",
    stacklevel=3
)

# DEPOIS
warnings.warn(
    f"Erro não-recuperável detectado ({error_name}). Não será feito retry.",
    stacklevel=5
)
```

---

### 7. errors.py:487-491 - Retry após falha
**Localização**: Dentro de `retry_with_backoff()`
**Cadeia**: `user_code -> dataframeit() -> _process_rows*() -> call_langchain() -> retry_with_backoff() -> warnings.warn()`
**Stacklevel atual**: `3` ⚠️ **INCORRETO** (aponta para _process_rows)
**Stacklevel correto**: `5` (para apontar para user_code)

```python
# ANTES
warnings.warn(
    f"Tentativa {attempt + 1}/{max_retries} falhou ({error_name}). "
    f"Aguardando {total_delay:.1f}s antes de tentar novamente...",
    stacklevel=3
)

# DEPOIS
warnings.warn(
    f"Tentativa {attempt + 1}/{max_retries} falhou ({error_name}). "
    f"Aguardando {total_delay:.1f}s antes de tentar novamente...",
    stacklevel=5
)
```

---

## Resumo das mudanças

| Arquivo | Linha | Stacklevel Atual | Stacklevel Correto | Status |
|---------|-------|------------------|-------------------|--------|
| core.py | 216 | Nenhum (1) | 2 | ❌ Precisa correção |
| core.py | 584 | Nenhum (1) | 3 | ❌ Precisa correção |
| core.py | 754 | 2 | 3 | ❌ Precisa correção |
| core.py | 772 | Nenhum (1) | 3 | ❌ Precisa correção |
| core.py | 802 | Nenhum (1) | 3 | ❌ Precisa correção |
| errors.py | 469 | 3 | 5 | ❌ Precisa correção |
| errors.py | 487 | 3 | 5 | ❌ Precisa correção |

## Benefícios da correção

1. **Melhor experiência do usuário**: Os warnings apontarão para o código do usuário, facilitando a identificação do problema
2. **Consistência**: Todos os warnings seguirão o mesmo padrão
3. **Conformidade com boas práticas**: Seguir a recomendação do Python de fazer warnings apontarem para o código do cliente
4. **Facilita debug**: Usuários saberão exatamente qual linha do código deles acionou o warning

## Impacto da mudança

- **Breaking change**: Não, apenas melhora a qualidade dos warnings
- **Testes**: Os testes que verificam warnings precisarão ser ajustados para considerar o novo stacklevel
- **Documentação**: Não requer mudanças na documentação do usuário
