"""Tratamento de erros e mensagens amigáveis para usuários iniciantes.

Este módulo contém funções para:
- Classificar erros como recuperáveis ou não-recuperáveis
- Gerar mensagens de erro amigáveis com instruções de resolução
- Validar dependências de providers
- Executar funções com retry e backoff exponencial
"""

from __future__ import annotations

import dataclasses
import functools
import importlib
import importlib.metadata
import json
import os
import random
import re
import time
import unicodedata
import warnings
from collections import deque
from collections.abc import Iterable, Mapping
from collections.abc import Set as AbstractSet
from contextlib import suppress
from functools import partial
from typing import TYPE_CHECKING, Any, cast, get_args, get_origin

from pydantic import BaseModel

from .search import SearchToolError, get_provider
from .utils import (
    _is_structure_class,
    _json_default,
    _members,
    _user_stacklevel,
    get_nested_pydantic_models,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterator, Sequence

    from pydantic import ValidationError
    from pydantic.fields import FieldInfo

    from .search import SearchProvider


CODEX_FILE_AUTH_LOGIN_COMMAND = "codex --config cli_auth_credentials_store='\"file\"' login"


# langchain-core >= 1.6 classifica os erros dos providers em ModelError, com
# is_retryable por classe. Versões anteriores não têm a hierarquia, e a
# classificação cai no status HTTP e nos padrões de mensagem.
try:
    from langchain_core.exceptions import ModelError as _ModelError
    from langchain_core.exceptions import ModelRateLimitError as _ModelRateLimitError
except ImportError:
    _ModelError = None
    _ModelRateLimitError = None


class ProviderError(RuntimeError):
    """Falha de execução reportada por um provider."""

    def __init__(self, *args: object, status_code: int | None = None) -> None:
        """Guarda, além da mensagem, o status HTTP que o provider informou como dado.

        A mensagem pode trazer texto do modelo, e por isso get_friendly_error_message
        escolhe a caixa pela classe e por `status_code`, e nunca pelo texto. Sem
        status informado, fica None.
        """
        super().__init__(*args)
        self.status_code = status_code


class ProviderTransientError(ProviderError):
    """Falha transitória que pode ser repetida sem reduzir o paralelismo."""


class ProviderOverloadedError(ProviderTransientError):
    """Falha transitória causada por sobrecarga ou limitação do provider."""


class ProviderRejectedOutputError(ProviderTransientError, ValueError):
    """Resposta recusada pela validação do modelo; ganha nova tentativa.

    Nos providers do LangChain, a tentativa seguinte pede a correção.

    É transitória por classe, e não pela mensagem: um número como '404' no texto
    analisado faria a classificação por texto tratá-la como erro HTTP definitivo.
    Continua ValueError para quem já capturava a falha de parsing.
    """


class ProviderAbortError(ProviderError):
    """Falha que impede o processamento de qualquer linha seguinte.

    A execução para de despachar linhas: a que recebeu o erro e as que faltam
    ficam sem status, o checkpoint é gravado, e `resume=True` as retoma quando a
    causa passar.
    """


class ProviderUsageLimitError(ProviderAbortError):
    """Cota de uso da conta esgotada."""


class ProviderConfigurationError(ValueError):
    """Configuração local incompatível com o contrato de um provider."""


class ProviderOutputError(ValueError):
    """Resposta definitiva incompatível com o contrato de saída."""


def dump_response(response: BaseModel, model: type[BaseModel] | None = None) -> dict[str, Any]:
    """O model_dump da resposta validada, pelo nome dos campos.

    O dump de um conjunto monta outro conjunto com o dump de cada item, e o
    item que vira dict, como o submodelo, não é hashable; a chave de dict
    que vira dict também não, e o model_dump cai com TypeError. O conjunto
    com submodelos sai como lista, com o dump de cada item, no campo e dentro
    de outro submodelo, de uma lista, de uma tupla ou dos valores de um dict
    (`_dump_with_model_sets`). O campo cuja anotação tem um conjunto de
    submodelos, nele ou em qualquer nível dentro dele, sai com esses conjuntos
    em lista com qualquer valor, também vazio (`_declared_sets_as_lists`): é
    como o checkpoint grava todo conjunto, e a validação da retomada, que
    passa a linha relida por esta mesma função, devolve a mesma lista.

    O que sobra, como a chave de dict que é submodelo, cai depois de a
    resposta ser paga, a cada resposta com um item desses, e por isso a falha
    interrompe a execução em vez de gastar as tentativas de cada linha.

    Com `model`, a classe declarada de um item que é instância de uma
    subclasse dela, o dump é o da classe declarada, como o pydantic faz com o
    item de uma lista: os campos que só a subclasse tem ficam de fora.
    """
    declared = model or type(response)
    dump: Callable[..., Any] = (
        response.model_dump
        if model is None
        else partial(model.__pydantic_serializer__.to_python, response)
    )
    written: Collection[str] = ()
    try:
        data = dump(by_alias=False)
    except _DUMP_ERRORS as error:
        if not _is_unhashable(error):
            raise
        data, written = _dump_with_model_sets(response, error, declared, dump)
    # O @model_serializer do modelo pode devolver outra coisa que um dict.
    if not isinstance(data, dict):
        return data
    for name in _fields_holding_any(declared).intersection(data):
        data[name] = _models_dumped(data[name], response)
    return _declared_sets_as_lists(declared, data, response, written)


@functools.cache
def _fields_holding_any(model: type[BaseModel]) -> frozenset[str]:
    """Os campos do modelo com `Any` ou `object` em algum nível, também dentro de um submodelo."""
    return frozenset(
        name
        for name, field in model.model_fields.items()
        if _holds_any(field.annotation, frozenset({model}))
    )


def _holds_any(annotation: object, seen: frozenset[type]) -> bool:
    """Se a anotação tem `Any` ou `object` em algum nível, também dentro de um submodelo.

    `seen` são os modelos já abertos no caminho, como em `_holds_model_set`.
    """
    for member in _members(annotation):
        if member is Any or member is object:
            return True
        if isinstance(member, type) and issubclass(member, BaseModel):
            if member not in seen and any(
                _holds_any(field.annotation, seen | {member})
                for field in member.model_fields.values()
            ):
                return True
        elif any(_holds_any(arg, seen) for arg in get_args(member)):
            return True
    return False


def _models_dumped(value: object, response: BaseModel) -> object:
    """O valor que o model_dump devolveu, com o dump dos submodelos que ele deixou como objeto.

    No pydantic 2.12, o model_dump não desce no `deque` nem na chave de dict
    que um campo `Any` recebeu, e o submodelo de dentro sai como objeto, sem
    erro; no 2.14, o item do `deque` sai pelo dump dele, e a chave e o conjunto
    que guardam um submodelo derrubam o dump. Esta função dá o resultado do
    2.14 nas duas versões: desce na lista, na tupla, no `deque` e nos valores
    do dict, escreve o submodelo pelo model_dump dele e levanta
    ProviderAbortError na chave e no conjunto que guardam um submodelo, e no
    submodelo cujo dump cai por um conjunto de submodelos.
    """
    if isinstance(value, BaseModel):
        try:
            dumped = value.model_dump(by_alias=False)
        except _DUMP_ERRORS as error:
            if not _is_unhashable(error):
                raise
            raise ProviderAbortError(_unhashable_dump_message(response, error)) from error
        return _models_dumped(dumped, response)
    kind = type(value)
    if kind is dict:
        pairs = cast("dict[object, object]", value)  # pragma: no mutate
        if any(_holds_model(key) for key in pairs):
            raise ProviderAbortError(_unhashable_dump_message(response, _UNHASHABLE_DICT))
        return {key: _models_dumped(item, response) for key, item in pairs.items()}
    if kind in (set, frozenset):
        if _holds_model(value):
            raise ProviderAbortError(_unhashable_dump_message(response, _UNHASHABLE_DICT))
        return value
    if kind in (list, tuple):
        items = cast("Iterable[object]", value)  # pragma: no mutate
        return kind(_models_dumped(item, response) for item in items)
    if isinstance(value, deque):
        return deque((_models_dumped(item, response) for item in value), value.maxlen)
    return value


def _holds_model(value: object) -> bool:
    """Se o valor é um submodelo, ou uma tupla ou um conjunto que guarda um."""
    if isinstance(value, BaseModel):
        return True
    return isinstance(value, (tuple, set, frozenset)) and any(_holds_model(item) for item in value)


# O erro do Python para o dict que entraria num conjunto ou viraria chave, que
# é o que o model_dump do pydantic 2.14 levanta no lugar de `_models_dumped`.
_UNHASHABLE_DICT = TypeError("unhashable type: 'dict'")


# O model_dump levanta o TypeError do Python, ou o PydanticSerializationError,
# que é um ValueError, em que o pydantic o embrulha quando ele sai de dentro de
# um @model_serializer(mode="wrap"). O texto do erro separa o caso
# (`_is_unhashable`).
_DUMP_ERRORS = (TypeError, ValueError)


def _is_unhashable(error: Exception) -> bool:
    """Se o erro do dump é o do Python para o valor que não entra em conjunto nem vira chave."""
    return "unhashable type" in str(error)


class _NoPlainDump(Exception):  # noqa: N818 (sinal interno, não é erro do usuário)
    """O valor tem uma forma que `_plain_dump` não sabe escrever sem o model_dump."""


def _dump_with_model_sets(
    response: BaseModel, error: Exception, model: type[BaseModel], dump: Callable[..., Any]
) -> tuple[dict[str, Any], Collection[str]]:
    """O model_dump com os conjuntos de submodelos que o derrubam como lista, e os campos trocados.

    O campo cujo dump, sozinho, cai por um valor que não é hashable sai por
    `_plain_dump`, que desce no valor e troca cada conjunto por lista. O que
    fica fora do dump, ou tem um serializador próprio que já devolve lista,
    não cai sozinho, e sai pelo model_dump como os demais campos. Sem campo
    que caia, com um valor que `_plain_dump` não escreve, como o dict com
    chave que é submodelo, ou com outro valor que não é hashable no resto,
    levanta ProviderAbortError com `error`, o TypeError do dump inteiro.
    """
    fields = model.model_fields
    plain = {}
    try:
        for name, field in fields.items():
            value = getattr(response, name)
            # O texto, o número e os demais valores sem itens não derrubam o dump.
            if not (isinstance(value, BaseModel) or _is_structure_class(type(value))):
                continue
            try:
                # Só importa se o dump cai; o que ele devolve é descartado.
                dump(include={name})
            except _DUMP_ERRORS as field_error:
                if not _is_unhashable(field_error):
                    raise
                plain[name] = _plain_dump(value, _declared_models(field))
    except _NoPlainDump:
        plain = {}
    if plain:
        try:
            data = dump(by_alias=False, exclude=set(plain))
        except _DUMP_ERRORS as rest_error:
            if not _is_unhashable(rest_error):
                raise
            error = rest_error
        else:
            data.update(plain)
            # Na ordem do modelo, como o model_dump; o que não é campo dele, no fim.
            ordered = {
                **{name: data[name] for name in fields if name in data},
                **{name: value for name, value in data.items() if name not in fields},
            }
            return ordered, plain
    raise ProviderAbortError(_unhashable_dump_message(response, error)) from error


def _declared_models(field: FieldInfo) -> list[type[BaseModel]]:
    """Os submodelos da anotação do campo, pelos quais `_plain_dump` escreve o item de subclasse.

    Com `SerializeAsAny` no campo ou em algum nível da anotação, nenhum: o
    pydantic escreve o item de subclasse pelos campos dela, e o item sai pelo
    dump da própria classe.
    """
    if _is_serialize_as_any(field.metadata) or _serializes_as_any(field.annotation):
        return []
    return get_nested_pydantic_models(field.annotation)


def _is_serialize_as_any(metadata: Iterable[object]) -> bool:
    """Se algum dos metadados é o `SerializeAsAny` do pydantic."""
    return any(type(item).__name__ == "SerializeAsAny" for item in metadata)


def _serializes_as_any(annotation: object) -> bool:
    """Se a anotação tem `SerializeAsAny` nos metadados de algum nível."""
    metadata = getattr(annotation, "__metadata__", ())  # pragma: no mutate
    return _is_serialize_as_any(metadata) or any(
        _serializes_as_any(arg) for arg in get_args(annotation)
    )


def _plain_dump(value: object, declared: Sequence[type[BaseModel]]) -> object:
    """O valor de um campo como o model_dump o escreveria, com cada conjunto em lista.

    Desce na lista, na tupla, no conjunto e nos valores do dict, e escreve o
    submodelo por `dump_response`, que faz o mesmo com os campos dele. O
    conjunto sai na ordem de `_as_sorted_list`. O submodelo que é instância de
    uma subclasse de uma das classes `declared`, as da anotação do campo, sai
    pelo dump da classe declarada. Levantam `_NoPlainDump` o dict com chave
    que é ou guarda um submodelo, que não tem dump, e a coleção de outra
    classe e o dataclass, que o model_dump escreve de um jeito que esta função
    não conhece.
    """
    if isinstance(value, BaseModel):
        base = next((cls for cls in declared if isinstance(value, cls)), None)
        return dump_response(value, None if type(value) in declared else base)
    kind = type(value)
    if kind in (set, frozenset):
        items = cast("AbstractSet[object]", value)  # pragma: no mutate
        return _as_sorted_list(_plain_dump(item, declared) for item in items)
    if kind in (list, tuple):
        items = cast("Iterable[object]", value)  # pragma: no mutate
        return kind(_plain_dump(item, declared) for item in items)
    if kind is dict:
        pairs = cast("dict[object, object]", value)  # pragma: no mutate
        if any(_holds_record(key) for key in pairs):
            raise _NoPlainDump
        return {key: _plain_dump(item, declared) for key, item in pairs.items()}
    if _is_structure_class(kind) or dataclasses.is_dataclass(value):
        raise _NoPlainDump
    return value


def _holds_record(value: object) -> bool:
    """Se o valor é um submodelo ou um dataclass, ou uma coleção que guarda um deles."""
    if isinstance(value, BaseModel) or dataclasses.is_dataclass(value):
        return True
    if not _is_structure_class(type(value)):
        return False
    # A chave é hashable, e a coleção dentro dela não é dict: os itens são os que ela percorre.
    return any(_holds_record(item) for item in value)  # ty: ignore[not-iterable]


def _declared_sets_as_lists(
    model: type[BaseModel], data: dict[str, Any], response: BaseModel, written: Collection[str]
) -> dict[str, Any]:
    """`data` com os conjuntos dos campos que declaram conjunto de submodelos em lista.

    O conjunto vazio e o que só tem itens que não são submodelo não derrubam o
    dump, e sairiam conjunto ao lado das listas das outras linhas, na mesma
    coluna. No campo declarado como conjunto de submodelos, o conjunto sai em
    lista. No que guarda um desses conjuntos mais fundo (`_nested_set_fields`),
    como o submodelo com um campo assim ou a lista de conjuntos, o valor sai
    por `_plain_dump`, que troca todo conjunto dele por lista; o que
    `_plain_dump` não escreve fica como o model_dump o deu. A anotação só é
    lida para o campo cujo valor é um conjunto, e o campo com serializador
    próprio fica como o serializador o deu, como todos os do modelo com
    `@model_serializer`, que monta o dump inteiro. Os campos de `written` já
    saíram por `_plain_dump` (`_dump_with_model_sets`).
    """
    nested = _nested_set_fields(model)
    for name, value in data.items():
        if name not in model.model_fields or _has_own_serializer(model, name):
            continue
        field = model.model_fields[name]
        if isinstance(value, (set, frozenset)):
            if _declares_model_set(field.annotation):
                data[name] = _as_sorted_list(value)
        elif name in written:  # pragma: no mutate (refazer o campo daria o mesmo)
            continue
        elif name in nested:
            with suppress(_NoPlainDump):
                data[name] = _plain_dump(getattr(response, name), _declared_models(field))
    return data


@functools.cache
def _nested_set_fields(model: type[BaseModel]) -> frozenset[str]:
    """Os campos do modelo que guardam um conjunto de submodelos abaixo do primeiro nível.

    Nenhum no modelo com `@model_serializer`, cujo dump não é o dos campos.
    """
    if model.__pydantic_decorators__.model_serializers:
        return frozenset()
    return frozenset(
        name
        for name, field in model.model_fields.items()
        if _holds_model_set(field.annotation, frozenset())
    )


def _holds_model_set(annotation: object, seen: frozenset[type]) -> bool:
    """Se a anotação tem um conjunto de submodelos em algum nível, também dentro de um submodelo.

    `seen` são os modelos já abertos no caminho: o que volta a um deles, como
    o modelo recursivo, não é aberto de novo.
    """
    if _declares_model_set(annotation):
        return True
    for member in _members(annotation):
        if isinstance(member, type) and issubclass(member, BaseModel):
            if member not in seen and any(
                _holds_model_set(field.annotation, seen | {member})
                for field in member.model_fields.values()
            ):
                return True
        elif any(_holds_model_set(arg, seen) for arg in get_args(member)):
            return True
    return False


def _has_own_serializer(model: type[BaseModel], name: str) -> bool:
    """Se o campo tem serializador próprio.

    É o `@field_serializer`, pelo nome ou pelo `"*"` de todos os campos, ou o
    `PlainSerializer` ou `WrapSerializer` no `Annotated` do campo inteiro. O
    que o serializador do usuário devolve é o que ele escolheu gravar.
    """
    return any(
        name in serializer.info.fields or "*" in serializer.info.fields
        for serializer in model.__pydantic_decorators__.field_serializers.values()
    ) or any(
        type(item).__name__ in {"PlainSerializer", "WrapSerializer"}
        # O serializador só do JSON não roda no dump em Python, que é o gravado.
        and getattr(item, "when_used", "always") in {"always", "unless-none"}  # pragma: no mutate
        for item in model.model_fields[name].metadata
    )


def _declares_model_set(annotation: object) -> bool:
    """Se a anotação, ou um membro da união, é um conjunto com submodelo entre os itens."""
    return any(
        isinstance(origin := get_origin(member), type)
        and issubclass(origin, AbstractSet)
        and bool(get_nested_pydantic_models(member))
        for member in _members(annotation)
    )


def _as_sorted_list(items: Iterable[object]) -> list[object]:
    """Os itens em lista, na ordem do texto JSON de cada um.

    A ordem em que um conjunto é percorrido muda de um processo para outro, e
    a retomada só reconhece a lista relida quando as duas saem na mesma ordem.
    É a ordem em que o checkpoint grava todo conjunto (`_json_default`). O item
    que o JSON não escreve, como o dict com chave de tupla, deixa a lista na
    ordem em que veio.
    """
    listed = list(items)
    try:
        return sorted(listed, key=lambda item: json.dumps(item, default=_json_default))
    except TypeError:
        return listed


def _unhashable_dump_message(response: BaseModel, error: Exception) -> str:
    """O motivo e a saída do model_dump que caiu com um valor que não é hashable."""
    model = type(response)
    reason = (
        f"O model_dump da resposta caiu com TypeError: {error}. O modelo {model.__name__} tem "
        "um set, um frozenset ou uma chave de dict cujos itens viram dict no dump, como os de "
        "submodelos, e o dict não é hashable. Repetir a linha não resolve: "
    )
    keyed = [
        name
        for name in model.model_fields
        if isinstance(value := getattr(response, name), Mapping)
        and any(isinstance(key, BaseModel) for key in value)
    ]
    if keyed:
        return (
            f"{reason}nos campos {keyed}, a chave do dict é um submodelo; declare a chave como "
            "texto ou número, ou o campo como lista de pares"
        )
    return (
        f"{reason}declare como texto ou número a chave de dict que é ou guarda um submodelo, "
        "e como list ou tuple o conjunto que fica num dataclass ou numa coleção que não é "
        "lista, tupla, conjunto nem dict"
    )


# Teto de erros de validação listados por resposta recusada, na mensagem da
# exceção e no pedido de correção do LangChain.
MAX_REJECTED_RULES = 20


def validation_error_summary(error: ValidationError) -> str:
    """Caminho e regra de cada erro de validação, sem o valor recusado (`input`).

    A mensagem vai para `_error_details`, e o valor recusado levaria dado da
    linha para a coluna. O caminho e a regra ainda trazem texto da resposta
    quando o Pydantic o põe ali: chave de dict, tag de união discriminada,
    mensagem de validador próprio. Por isso a classificação das exceções que
    carregam este resumo decide pela classe, e não pelo texto.
    """
    rules = [
        f"{'.'.join(str(part) for part in item['loc']) or '(resposta inteira)'}: {item['msg']}"
        for item in error.errors()[:MAX_REJECTED_RULES]
    ]
    return f"{error.error_count()} erro(s): {'; '.join(rules)}"


# Erros não-recuperáveis (não adianta tentar novamente)
NON_RECOVERABLE_ERRORS = (
    "AuthenticationError",
    "InvalidAPIKey",
    "PermissionDenied",
    "InvalidArgument",
    "NotFound",
    "401",
    "403",
    "404",
    # Tavily - erros de autenticação e argumentos
    "MissingAPIKeyError",
    "InvalidAPIKeyError",
    "BadRequestError",
    # Prompt maior que a janela de contexto: repetir não muda o tamanho
    "ContextOverflowError",
)

# Os padrões em minúsculas, como a mensagem com que is_recoverable_error os compara.
_NON_RECOVERABLE_PATTERNS = tuple(pattern.lower() for pattern in NON_RECOVERABLE_ERRORS)


# Auth via credenciais de SDK (sem env var de API key) compartilhada por
# todos os providers Bedrock. Linhas mantidas curtas para caber em caixas de 80 cols.
_BEDROCK_BASE = {
    "package": "langchain_aws",
    "install": "langchain-aws",
    "env_var": None,
    "auth_hint": (
        "aws configure\n"
        "OU exporte AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY\n"
        "(opcional: AWS_SESSION_TOKEN, AWS_REGION)"
    ),
}

# Providers cuja heurística simples (langchain_{provider} + {PROVIDER}_API_KEY) não bate com a realidade.
# env_var=None indica auth por SDK (ADC, AWS creds), não por API key.
_PROVIDER_OVERRIDES = {
    "claude_code": {
        "package": "claude_agent_sdk",
        "install": "dataframeit[claude-code]",
        "env_var": None,
        "name": "Claude Code",
        "auth_hint": "Autentique o Claude Code conforme a documentação do SDK.",
        "uses_langchain": False,
    },
    "codex": {
        "package": "openai_codex",
        "install": "dataframeit[codex]",
        "env_var": None,
        "name": "OpenAI Codex",
        "auth_hint": CODEX_FILE_AUTH_LOGIN_COMMAND,
        "uses_langchain": False,
    },
    "google_vertexai": {
        "package": "langchain_google_vertexai",
        "install": "langchain-google-vertexai",
        "env_var": None,
        "name": "Google Vertex AI",
        "auth_hint": (
            "gcloud auth application-default login\n"
            "OU exporte GOOGLE_APPLICATION_CREDENTIALS=/caminho/service-account.json"
        ),
    },
    "bedrock": {**_BEDROCK_BASE, "name": "AWS Bedrock"},
    "bedrock_converse": {**_BEDROCK_BASE, "name": "AWS Bedrock (Converse)"},
    "mistralai": {
        "package": "langchain_mistralai",
        "install": "langchain-mistralai",
        "env_var": "MISTRAL_API_KEY",
        "name": "Mistral AI",
    },
    "azure_openai": {
        "package": "langchain_openai",
        "install": "langchain-openai",
        "env_var": "AZURE_OPENAI_API_KEY",
        "name": "Azure OpenAI",
        "auth_hint": (
            'export AZURE_OPENAI_API_KEY="sua-chave-aqui"\n'
            'export AZURE_OPENAI_ENDPOINT="https://<recurso>.openai.azure.com/"\n'
            'export OPENAI_API_VERSION="2025-03-01-preview"'
        ),
    },
}


def _search_provider_info(search_provider: str) -> dict[str, Any]:
    """Nome e variável da chave do provedor de busca, no formato de _infer_provider_info."""
    provider = get_provider(search_provider)
    return {"name": provider.friendly_name, "env_var": provider.env_var}


def _infer_provider_info(provider: str | None) -> dict[str, Any]:
    """Infere informações do provider dinamicamente.

    Args:
        provider: Nome do provider (google_genai, openai, anthropic, etc).

    Returns:
        Dict com package, install, env_var e name inferidos.
    """
    if not bool(provider):
        return {"package": None, "install": None, "env_var": "API_KEY", "name": "LLM"}

    if provider in _PROVIDER_OVERRIDES:
        return dict(_PROVIDER_OVERRIDES[provider])

    # Inferir nome do pacote: provider -> langchain_{provider}
    package = f"langchain_{provider}"
    # Inferir nome para pip: langchain_{provider} -> langchain-{provider}
    install = package.replace("_", "-")

    # Inferir variável de ambiente
    # google_genai -> GOOGLE_API_KEY, openai -> OPENAI_API_KEY
    provider_upper = provider.replace("_genai", "").replace("_ai", "").upper()
    env_var = f"{provider_upper}_API_KEY"

    # Nome amigável
    name_map = {
        "google_genai": "Google Gemini",
        "openai": "OpenAI",
        "anthropic": "Anthropic Claude",
        "fireworks": "Fireworks AI",
        "together": "Together AI",
    }
    name = name_map.get(provider, provider.replace("_", " ").title())

    return {
        "package": package,
        "install": install,
        "env_var": env_var,
        "name": name,
    }


# O importlib.metadata normaliza o nome, e por isso a troca de caixa que o teste
# de mutação faria no literal dentro da função não mudaria o resultado.
_DISTRIBUTION_NAME = "dataframeit"
_EXTRA_ALL_MARKER = re.compile(r"""extra\s*==\s*["']all["']""")
_REQUIREMENT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _normalize_distribution_name(name: str) -> str:
    # Normalização da PEP 503: "langchain_openai" e "LangChain-OpenAI" são o
    # mesmo pacote, e é assim que o nome de instalação se compara ao requisito.
    return re.sub(r"[-_.]+", "-", name).lower()


def _all_extra_packages() -> frozenset[str]:
    """Nomes normalizados dos pacotes que o extra ``all`` instala.

    Lidos dos metadados do dataframeit instalado, que o build gera do
    ``pyproject.toml``, para que a dica de ``dataframeit[all]`` não divirja do
    extra. Sem os metadados, como ao rodar o código fonte sem instalar o
    pacote, devolve vazio: não há como saber o que o extra traz, e sugeri-lo
    poderia mandar instalar algo que não resolve.
    """
    try:
        requirements = importlib.metadata.requires(_DISTRIBUTION_NAME) or []
    except importlib.metadata.PackageNotFoundError:
        return frozenset()
    packages = set()
    for requirement in requirements:
        # O nome abre o requisito, e o extra só aparece no marcador de ambiente.
        match = _REQUIREMENT_NAME.match(requirement.strip())
        if match and _EXTRA_ALL_MARKER.search(requirement):
            packages.add(_normalize_distribution_name(match.group()))
    return frozenset(packages)


def _get_missing_package_message(
    package: str,
    install_name: str,
    friendly_name: str,
    alternative_install: str | None = None,
) -> str:
    """Gera mensagem amigável para pacote não instalado."""
    alternative = ""
    if bool(alternative_install):
        alternative = f"""║                                                                              ║
║  Ou, para instalar todas as dependências recomendadas:                       ║
║                                                                              ║
║      pip install {alternative_install:<62} ║
║                                                                              ║
"""
    return f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  BIBLIOTECA NÃO INSTALADA                                                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  A biblioteca '{package}' é necessária para usar {friendly_name}.            ║
║                                                                              ║
║  COMO RESOLVER:                                                              ║
║                                                                              ║
║  Execute o seguinte comando no terminal:                                     ║
║                                                                              ║
║      pip install {install_name:<62} ║
║                                                                              ║
{alternative}║  Após instalar, execute seu código novamente.                                ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""".strip()


def validate_provider_dependencies(provider: str | None) -> None:
    """Valida se as dependências do provider estão instaladas ANTES de iniciar.

    Args:
        provider: Nome do provider (google_genai, openai, anthropic, claude_code, etc).

    Raises:
        ImportError: Com mensagem amigável se dependência não estiver instalada.
    """
    provider_data = _infer_provider_info(provider)

    # Providers de SDK falam diretamente com seus runtimes, sem LangChain.
    if not provider_data.get("uses_langchain", True):
        try:
            importlib.import_module(provider_data["package"])
        except ImportError as err:
            raise ImportError(
                _get_missing_package_message(
                    provider_data["package"], provider_data["install"], provider_data["name"]
                )
            ) from err
        return

    # Validar LangChain base. O langchain e o langchain-core são dependências
    # obrigatórias, e por isso o extra all também os instala.
    try:
        importlib.import_module("langchain")
    except ImportError as err:
        raise ImportError(
            _get_missing_package_message("langchain", "langchain", "LangChain", "dataframeit[all]")
        ) from err

    try:
        importlib.import_module("langchain_core")
    except ImportError as err:
        raise ImportError(
            _get_missing_package_message(
                "langchain_core",
                "langchain-core",
                "LangChain Core",
                "dataframeit[all]",
            )
        ) from err

    # Validar provider específico (inferir dinamicamente)
    if bool(provider):
        package = provider_data["package"]
        install = provider_data["install"]
        name = provider_data["name"]
        try:
            importlib.import_module(package)
        except ImportError as err:
            # O extra all só traz os providers que têm extra próprio; para os
            # demais, como mistralai e cohere, sugeri-lo não resolveria.
            alternative = (
                "dataframeit[all]"
                if _normalize_distribution_name(install) in _all_extra_packages()
                else None
            )
            raise ImportError(
                _get_missing_package_message(package, install, name, alternative)
            ) from err


def validate_search_dependencies(search_provider: str = "tavily") -> None:
    """Valida se as dependências do provedor de busca estão instaladas e API key configurada.

    Args:
        search_provider: Nome do provedor de busca ("tavily" ou "exa").

    Raises:
        ImportError: Com mensagem amigável se pacote do provedor não estiver instalado.
        ValueError: Com mensagem amigável se API key não estiver configurada.
    """
    provider = get_provider(search_provider)

    try:
        importlib.import_module(provider.package_name)
    except ImportError as err:
        raise ImportError(
            _get_missing_package_message(
                provider.package_name,
                provider.install_name,
                provider.friendly_name,
            )
        ) from err

    if not bool(os.environ.get(provider.env_var)):
        raise ValueError(_get_missing_search_api_key_message(provider))


def _get_missing_search_api_key_message(provider: SearchProvider) -> str:
    """Gera mensagem amigável para API key de busca não configurada."""
    return f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  CHAVE DE API DO {provider.friendly_name.upper()} NÃO CONFIGURADA                                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Para usar busca web com {provider.friendly_name}, você precisa de uma API key.           ║
║                                                                              ║
║  COMO OBTER:                                                                 ║
║  1. Acesse {provider.signup_url:<58} ║
║  2. Crie uma conta ({provider.free_tier})                                          ║
║  3. Copie sua API key                                                        ║
║                                                                              ║
║  COMO CONFIGURAR:                                                            ║
║                                                                              ║
║  No Linux/Mac:                                                               ║
║      export {provider.env_var}="sua-chave-aqui"                                   ║
║                                                                              ║
║  No Windows (PowerShell):                                                    ║
║      $env:{provider.env_var}="sua-chave-aqui"                                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""".strip()


# Textos que só aparecem em erro de chave inválida, qualquer que seja o status.
_INVALID_KEY_MARKERS = (
    "api_key_invalid",
    "api key not valid",
    "invalid api key",
    "incorrect api key",
    "invalid x-api-key",
)


def get_friendly_error_message(  # noqa: C901, PLR0911 (uma saída por categoria de erro)
    error: Exception, provider: str | None = None
) -> str:
    """Converte erro técnico em mensagem amigável para usuários iniciantes.

    Args:
        error: Exceção original.
        provider: Nome do provider (google_genai, openai, etc).

    Returns:
        Mensagem de erro amigável com instruções de como resolver.
    """
    # As duas classes de resposta recusada decidem pela classe: a mensagem pode
    # trazer texto da resposta no caminho e na regra de um erro de validação
    # (ver validation_error_summary), e um número como '401' na chave de um dict
    # não pode virar erro de credencial.
    if isinstance(error, ProviderRejectedOutputError):
        return (
            "RESPOSTA RECUSADA PELA VALIDAÇÃO DO MODELO PYDANTIC\n"
            f"{error}\n"
            "Nenhuma tentativa produziu resposta válida. Veja as regras do modelo que "
            "falharam; se o texto é ambíguo, a instrução do campo pode ajudar."
        )
    if isinstance(error, ProviderOutputError):
        return (
            "RESPOSTA FORA DO CONTRATO DE SAÍDA\n"
            f"{error}\n"
            "O provider terminou sem uma resposta que sirva ao modelo Pydantic, e a linha "
            "não é tentada de novo."
        )

    # Obter informações do provider dinamicamente. A falha da ferramenta de busca
    # é do provedor de busca, e não do provider de LLM recebido: as caixas
    # apontam o nome e a chave dele.
    search_name = error.provider if isinstance(error, SearchToolError) else None
    provider_data = (
        _search_provider_info(search_name) if bool(search_name) else _infer_provider_info(provider)
    )
    provider_name = provider_data["name"]
    env_var = provider_data["env_var"]

    # A mensagem de um ProviderError pode trazer texto do modelo: o result do SDK
    # no claude_code, a mensagem do turno no codex. Um "API key", um "art. 403" ou
    # um "timeout" ali é conteúdo, e não diz nada de credencial, permissão ou
    # rede. Por isso, para ele, os padrões abaixo leem um texto vazio, e a caixa
    # sai da classe e do status estruturado (o status_code que o provider
    # preenche); sem classe nem status que decidam, cai na genérica, que mostra o
    # tipo e o começo da mensagem, e a de conexão, que só o texto decide, nunca
    # sai. Os demais erros, como os do LangChain, que embrulham os SDKs de vários
    # providers, são lidos pelo texto: sem status, em todas as caixas; com
    # status, o texto ainda decide a de chave inválida e, fora do SearchToolError,
    # as do Tavily e do Exa, e o status decide as outras.
    # Nenhum padrão casa com o texto vazio nem com o texto fixo que o teste de
    # mutação põe no lugar dele.
    error_str = (
        ""  # pragma: no mutate
        if isinstance(error, ProviderError)
        else f"{type(error).__name__}: {error}".lower()
    )
    error_name = type(error).__name__
    status = _http_error_status(error)
    # O erro da ferramenta de busca diz o serviço pela classe, e os demais, pelo
    # nome no texto: 'exa' como palavra, contando '_' como separador, casa
    # 'EXA_API_KEY' e 'exa-py', mas não 'hexagonal'.
    service = search_name or (
        "tavily"
        if "tavily" in error_str
        else "exa"
        if re.search(r"(?<![a-z0-9])exa(?![a-z0-9])", error_str)
        else None
    )
    is_tavily = service == "tavily"
    is_exa = service == "exa"

    def is_category(patterns: Iterable[str], codes: Iterable[int] = ()) -> bool:
        """Com status HTTP estruturado, só ele decide; sem, valem os padrões."""
        if status is not None:
            return status in codes
        return any(
            _matches_error_pattern(pattern, error_str) for pattern in (*patterns, *map(str, codes))
        )

    def is_search_category(patterns: Iterable[str], codes: Iterable[int] = ()) -> bool:
        """Caixa do Tavily ou do Exa: no SearchToolError, como is_category; nos demais, o texto.

        O status do SearchToolError é o do provedor de busca, e um "invalid" na
        mensagem de um 500 não é chave inválida. Os demais erros só chegam às
        caixas da busca pelo nome do serviço no texto, e o status deles pode ser
        de outra requisição.
        """
        if bool(search_name):
            return is_category(patterns, codes)
        return any(pattern in error_str for pattern in patterns)

    # === COTA DE USO ESGOTADA ===
    # Vem antes do rate limit: o insufficient_quota da OpenAI chega como 429, mas
    # retry_with_backoff o converte em ProviderUsageLimitError, que interrompe a
    # execução sem reduzir os workers. A caixa mostra a mensagem da conversão,
    # que nomeia o serviço sem cota.
    usage_limit = error if isinstance(error, ProviderUsageLimitError) else _usage_limit_error(error)
    if usage_limit is not None:
        return f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  COTA DE USO ESGOTADA                                                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
{_generic_box_details(str(usage_limit))}
║                                                                              ║
║  Repetir a linha não resolve, e a execução para.                             ║
║                                                                              ║
║  COMO RESOLVER:                                                              ║
║  1. Adicione crédito, mude de plano ou espere a renovação do período         ║
║  2. Rode de novo com resume=True para processar as linhas pendentes          ║
║  3. Se a cota é da busca web, use_search=False ou outro search_provider      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""".strip()

    # === ERROS DE RATE LIMIT ===
    # A caixa sai quando, e só quando, a execução reduz os workers: a decisão é a
    # mesma função que _process_rows_parallel chama, e não um padrão de texto
    # próprio. Por isso vem antes das caixas decididas pelo texto, que não
    # sombreiam a sobrecarga cuja mensagem cita uma chave ou um 403. Nos
    # ProviderError, is_rate_limit_error decide pela classe, e um '429' no texto,
    # que pode ser do modelo, não muda a caixa.
    if is_rate_limit_error(error):
        # O 429 do Tavily e do Exa chega como erro da ferramenta de busca, e a
        # caixa nomeia o serviço que recusou. A cota de uso esgotada, que não é
        # rate limit nem reduz os workers, saiu acima, na caixa de cota. O
        # ProviderError vem do provider de LLM, e o texto vazio dele mantém o
        # nome do provider.
        service = "Tavily" if is_tavily else "Exa" if is_exa else provider_name
        return f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  LIMITE DE REQUISIÇÕES ATINGIDO                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Você fez muitas requisições em pouco tempo para o {service}.          ║
║                                                                              ║
║  COMO RESOLVER:                                                              ║
║  1. Aguarde alguns minutos e tente novamente                                 ║
║  2. Use rate_limit_delay: cada worker espera esse tempo depois de cada       ║
║     linha processada, e o teto fica em                                       ║
║     parallel_requests * 60 / rate_limit_delay linhas por minuto:             ║
║                                                                              ║
║     dataframeit(..., rate_limit_delay=1.0)  # até 60 linhas/min por worker   ║
║                                                                              ║
║  3. Considere atualizar seu plano para limites maiores                       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""".strip()

    # === ERROS TAVILY ===
    if is_tavily and is_search_category(["apikey", "api_key", "missing", "invalid"], (401,)):
        return """
╔══════════════════════════════════════════════════════════════════════════════╗
║  ERRO DE AUTENTICAÇÃO TAVILY                                                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Sua chave de API do Tavily está inválida ou não configurada.                ║
║                                                                              ║
║  COMO RESOLVER:                                                              ║
║  1. Acesse https://app.tavily.com e obtenha sua API key                      ║
║  2. Configure a variável de ambiente:                                        ║
║                                                                              ║
║     export TAVILY_API_KEY="sua-chave-aqui"                                   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""".strip()

    # === ERROS EXA ===
    if is_exa and is_search_category(
        ["apikey", "api_key", "missing", "invalid", "unauthorized"], (401,)
    ):
        return """
╔══════════════════════════════════════════════════════════════════════════════╗
║  ERRO DE AUTENTICAÇÃO EXA                                                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Sua chave de API do Exa está inválida ou não configurada.                   ║
║                                                                              ║
║  COMO RESOLVER:                                                              ║
║  1. Acesse https://exa.ai e obtenha sua API key                              ║
║  2. Configure a variável de ambiente:                                        ║
║                                                                              ║
║     export EXA_API_KEY="sua-chave-aqui"                                      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""".strip()

    # === ERROS DE AUTENTICAÇÃO ===
    # Chave inválida decide pelo texto mesmo com status: o Google a devolve
    # como 400 INVALID_ARGUMENT, que o status sozinho não distingue. No
    # ProviderError, o texto vazio deixa a decisão só para o 401.
    invalid_key = any(marker in error_str for marker in _INVALID_KEY_MARKERS)
    if invalid_key or is_category(
        ["authenticationerror", "invalidapikey", "api_key", "api key"], (401,)
    ):
        if env_var is None:
            # Auth via credenciais de SDK (Vertex AI ADC, AWS creds, etc). Todo
            # provider sem env_var traz em _PROVIDER_OVERRIDES a dica de como
            # se autenticar.
            auth_hint = provider_data["auth_hint"]
            hint_lines = "\n".join(f"║     {line:<73}║" for line in auth_hint.split("\n"))
            msg = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ERRO DE AUTENTICAÇÃO - Credenciais não configuradas                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Provider: {provider_name:<66}║
║                                                                              ║
║  Este provider usa credenciais de SDK (não uma API key tradicional).         ║
║                                                                              ║
║  COMO RESOLVER:                                                              ║
║                                                                              ║
{hint_lines}
║                                                                              ║
║  Verifique também se região, projeto/conta e modelo estão corretos.          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
            return msg.strip()
        export_linux = f'export {env_var}="sua-chave-aqui"'
        export_win = f'$env:{env_var}="sua-chave-aqui"'
        msg = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ERRO DE AUTENTICAÇÃO - Chave de API inválida ou não configurada             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Provider: {provider_name:<66}║
║                                                                              ║
║  Sua chave de API não foi aceita.                                            ║
║                                                                              ║
║  COMO RESOLVER:                                                              ║
║                                                                              ║
║  1. Obtenha uma chave de API no site/console do provider.                    ║
║                                                                              ║
║  2. Configure no terminal (antes de executar seu código):                    ║
║                                                                              ║
║     No Linux/Mac:                                                            ║
║     {export_linux:<73}║
║                                                                              ║
║     No Windows (PowerShell):                                                 ║
║     {export_win:<73}║
║                                                                              ║
║  3. OU passe diretamente no código:                                          ║
║     dataframeit(..., api_key="sua-chave-aqui")                               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
        return msg.strip()

    # === ERROS DE PERMISSÃO ===
    if is_category(["permissiondenied", "forbidden"], (403,)):
        return f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ERRO DE PERMISSÃO - Sua chave não tem acesso a este recurso                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Sua chave de API do {provider_name} não tem permissão para usar este modelo.║
║                                                                              ║
║  POSSÍVEIS CAUSAS:                                                           ║
║  • A chave é de uma conta gratuita com acesso limitado                       ║
║  • O modelo solicitado requer um plano pago                                  ║
║  • A chave foi revogada ou expirou                                           ║
║                                                                              ║
║  COMO RESOLVER:                                                              ║
║  1. Verifique seu plano no site/console do {provider_name}                   ║
║  2. Tente usar um modelo mais básico                                         ║
║  3. Gere uma nova chave de API                                               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""".strip()

    # === ERROS DE TIMEOUT ===
    # A caixa sai só para o erro que is_recoverable_error classifica como
    # recuperável, o que retry_with_backoff repete enquanto houver tentativas, e
    # por isso manda aumentar max_retries. Ela não afirma que houve repetição,
    # porque com max_retries=1 não há. No ProviderError, recuperável é o
    # ProviderTransientError, e o timeout é só o status 408 ou 504.
    if is_recoverable_error(error) and is_category(["timeout", "deadlineexceeded"], (408, 504)):
        return f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  TEMPO ESGOTADO (TIMEOUT)                                                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  O {provider_name} demorou muito para responder.                             ║
║                                                                              ║
║  POSSÍVEIS CAUSAS:                                                           ║
║  • Servidor do {provider_name} sobrecarregado                                ║
║  • Conexão de internet instável                                              ║
║  • Texto muito longo para processar                                          ║
║                                                                              ║
║  COMO RESOLVER:                                                              ║
║  1. Aumente max_retries: o timeout ganha nova tentativa até esse limite      ║
║  2. Use resume=True para continuar de onde parou:                            ║
║                                                                              ║
║     df = dataframeit(df, ..., resume=True)                                   ║
║                                                                              ║
║  3. Tente novamente em alguns minutos                                        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""".strip()

    # === ERROS DE CONEXÃO ===
    if is_category(["connectionerror", "connectionreset", "sslerror", "network"]):
        return f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ERRO DE CONEXÃO                                                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Não foi possível conectar ao {provider_name}.                               ║
║                                                                              ║
║  POSSÍVEIS CAUSAS:                                                           ║
║  • Sem conexão com a internet                                                ║
║  • Firewall ou proxy bloqueando a conexão                                    ║
║  • Servidor do {provider_name} temporariamente indisponível                  ║
║                                                                              ║
║  COMO RESOLVER:                                                              ║
║  1. Verifique sua conexão com a internet                                     ║
║  2. Tente acessar google.com no navegador                                    ║
║  3. Se estiver em rede corporativa, consulte o suporte de TI                 ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""".strip()

    # === ERRO GENÉRICO ===
    return f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ERRO NO PROCESSAMENTO                                                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Tipo: {_pad_display(error_name, 69)} ║
║                                                                              ║
{_generic_box_details(str(error))}
║                                                                              ║
║  Se este erro persistir, você pode:                                          ║
║  1. Verificar se suas credenciais estão corretas                             ║
║  2. Tentar novamente com resume=True                                         ║
║  3. Reportar o problema em:                                                  ║
║     https://github.com/bdcdo/dataframeit/issues                              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""".strip()


# Largura do texto na linha 'Detalhes' da caixa genérica, que cabe entre o
# rótulo e a borda direita da moldura, e o teto de linhas que a mensagem ocupa.
_GENERIC_BOX_DETAIL_WIDTH = 65
_GENERIC_BOX_DETAIL_MAX_LINES = 3
_GENERIC_BOX_DETAIL_LABEL = "Detalhes: "


def _generic_box_details(message: str) -> str:
    """Linhas 'Detalhes' da caixa genérica, com a mensagem em até três linhas.

    Numa linha só, o prefixo que o provider põe na mensagem ocuparia a largura
    inteira: o do claude_code, com subtipo e status, passa de 65 caracteres, e a
    caixa não mostraria nada do texto que o SDK devolveu. A quebra, o corte da
    palavra maior que a linha e o preenchimento até a borda medem a largura
    exibida no terminal, e não o número de caracteres, para que emoji, CJK e
    marca combinante não desalinhem a moldura; a mensagem cortada no teto
    termina em reticências.
    """
    lines = _wrap_display(
        _box_text(message), _GENERIC_BOX_DETAIL_WIDTH, _GENERIC_BOX_DETAIL_MAX_LINES
    ) or [""]
    indent = " " * len(_GENERIC_BOX_DETAIL_LABEL)
    labels = [_GENERIC_BOX_DETAIL_LABEL] + [indent] * (len(lines) - 1)
    # Rótulos e linhas têm o mesmo tamanho, e o strict do zip não muda o resultado.
    return "\n".join(
        f"║  {label}{_pad_display(line, _GENERIC_BOX_DETAIL_WIDTH)} ║"
        for label, line in zip(labels, lines, strict=True)  # pragma: no mutate
    )


# Sequência de escape do terminal (ECMA-48): CSI (cor ANSI, como '\x1b[31m'),
# OSC (título, hiperlink) terminada em BEL ou ST, e as demais, de bytes
# intermediários e um byte final (como '\x1b(B' e '\x1b7'). O OSC sem
# terminador perde só o '\x1b]', e não o resto da mensagem.
_TERMINAL_ESCAPE = re.compile(
    r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[ -/]*[0-~])"
)


def _box_text(text: str) -> str:
    """Texto para a moldura: em NFC, sem escape do terminal e sem caractere de controle.

    O espaço Unicode, que inclui os separadores de linha que o splitlines parte
    (U+2028, U+2029, U+0085), fica, e o split de _wrap_display o troca pelo
    espaço entre as palavras; os demais caracteres de controle e de formatação,
    que o terminal não mostra ou mostra com largura imprevisível, saem, como o
    surrogate solto, que o print não codifica em UTF-8.
    """
    text = _TERMINAL_ESCAPE.sub("", unicodedata.normalize("NFC", text))
    return "".join(
        char
        for char in text
        if char.isspace() or unicodedata.category(char) not in {"Cc", "Cf", "Cs"}
    )


def _char_width(char: str) -> int:
    """Colunas que o caractere ocupa no terminal: largo e cheio 2, combinante 0."""
    if unicodedata.category(char) in {"Mn", "Me"}:
        return 0
    return 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1


def _display_width(text: str) -> int:
    return sum(map(_char_width, text))


def _pad_display(text: str, width: int) -> str:
    """Completa o texto com espaços até ``width`` colunas exibidas."""
    return text + " " * (width - _display_width(text))


def _cut_display(text: str, width: int) -> str:
    """Maior prefixo do texto que cabe em ``width`` colunas."""
    used = 0
    for index, char in enumerate(text):
        used += _char_width(char)
        if used > width:
            return text[:index]
    return text


def _wrap_display(text: str, width: int, max_lines: int) -> list[str]:
    """Quebra o texto em linhas de até ``width`` colunas, com reticências no corte.

    Como o textwrap, junta as palavras por um espaço e corta a palavra maior que
    a linha, mas mede a largura exibida. Passando de ``max_lines``, a última
    linha perde palavras até caber ' ...'. A quebra para quando passa do teto,
    e cada medida percorre no máximo uma linha, para que a mensagem enorme, ou
    com um token de centenas de milhares de caracteres, não trave a caixa.
    """
    placeholder = " ..."
    lines: list[str] = []
    # Qualquer valor falso serve de linha vazia, e os dois limites de linhas
    # marcados abaixo só encurtam o trabalho: passando de max_lines, a última
    # linha mostrada já está pronta. Ficam fora do teste de mutação, que
    # trocaria o "" por None e mudaria um limite em uma linha sem mudar o
    # resultado.
    line = ""  # pragma: no mutate
    for word in text.split():
        if len(lines) > max_lines:  # pragma: no mutate
            break
        candidate = f"{line} {word}" if line else word
        if _cut_display(candidate, width) == candidate:
            line = candidate
            continue
        if line:
            lines.append(line)
        line = word
        while len(lines) <= max_lines:  # pragma: no mutate
            head = _cut_display(line, width)
            if head == line:
                break
            lines.append(head)
            line = line[len(head) :]
    if line:
        lines.append(line)
    if len(lines) <= max_lines:
        return lines
    last = lines[max_lines - 1]
    room = width - len(placeholder)
    while " " in last and _display_width(last) > room:
        last = last[: last.rindex(" ")]
    return [*lines[: max_lines - 1], _cut_display(last, room) + placeholder]


# Status HTTP 4xx que indicam falha transitória; os demais 4xx são definitivos.
_RECOVERABLE_CLIENT_STATUSES = frozenset({408, 409, 429})
_HTTP_ERROR_MIN = 400
_HTTP_SERVER_ERROR_MIN = 500
_HTTP_ERROR_MAX = 599
# Status que reduzem os workers: o 429 e a sobrecarga 529 da Anthropic. O
# langchain-core classifica o 529 como ModelAPIError, e não como
# ModelRateLimitError, mas aliviar a carga ajuda como no 429.
_OVERLOADED_STATUSES = (429, 529)


def _http_error_status_or_none(value: object) -> int | None:
    """Aceita somente um status HTTP de erro entre 400 e 599."""
    if isinstance(value, int) and _HTTP_ERROR_MIN <= value <= _HTTP_ERROR_MAX:
        return value
    return None


def _is_recoverable_http_status(status: int) -> bool:
    """Diz se o status HTTP indica falha que vale repetir: 5xx e os 4xx transitórios.

    É a regra única dos três caminhos: o erro de SDK que chega pelo LangChain,
    classificado em is_recoverable_error, e os status que o claude_code e o codex
    recebem do próprio runtime e convertem nas classes Provider*Error.
    """
    return status >= _HTTP_SERVER_ERROR_MIN or status in _RECOVERABLE_CLIENT_STATUSES


# Atributos em que SDKs e clientes HTTP expõem o status da resposta:
# status_code (openai, anthropic, groq, mistral, cohere), code (google.genai,
# google.api_core, urllib) e http_status. O `code` do openai é textual e é
# descartado pela checagem de tipo em _own_http_status.
_HTTP_STATUS_ATTRIBUTES = ("status_code", "code", "http_status")


def _own_http_status(error: BaseException) -> int | None:
    """Status HTTP de erro (400-599) declarado pela própria exceção, se houver."""
    candidates = [getattr(error, name, None) for name in _HTTP_STATUS_ATTRIBUTES]
    candidates.append(getattr(getattr(error, "response", None), "status_code", None))
    for value in candidates:
        # A faixa descarta códigos que não são status HTTP de erro, inclusive bool.
        status = _http_error_status_or_none(value)
        if status is not None:
            return status
    return None


def _explicit_cause_chain(error: BaseException) -> Iterator[BaseException]:
    """A exceção e suas causas explícitas, cada uma uma vez.

    Wrappers como o ChatGoogleGenerativeAIError encadeiam o erro do SDK com
    `raise ... from`, por isso a busca segue `__cause__`. `__context__` fica de
    fora: é a exceção que estava sendo tratada quando esta surgiu, e não
    necessariamente a sua causa.
    """
    seen = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__


def _http_error_status(error: BaseException) -> int | None:
    """Status HTTP da exceção ou da primeira causa explícita que o declare.

    Wrappers como o ChatGoogleGenerativeAIError não carregam status próprio, e
    o status fica no erro do SDK que eles encadeiam.
    """
    for current in _explicit_cause_chain(error):
        status = _own_http_status(current)
        if status is not None:
            return status
    return None


def _matches_error_pattern(pattern: str, error_str: str) -> bool:
    """Casa um padrão com a mensagem, os dois já em minúsculas.

    Padrões numéricos casam só como número isolado, para que '401' não case com
    '4015 tokens'.
    """
    if pattern.isdigit():
        return re.search(rf"\b{pattern}\b", error_str) is not None
    return pattern in error_str


def is_recoverable_error(error: Exception) -> bool:
    """Verifica se um erro é recuperável (vale a pena fazer retry).

    A decisão segue esta precedência: as classes Provider*Error; o
    `is_retryable` do ModelError do langchain-core, quando existe; o status HTTP
    estruturado da exceção ou da sua causa (408, 409, 429 e 5xx são recuperáveis,
    os demais 4xx não); os padrões de NON_RECOVERABLE_ERRORS na mensagem; e, se
    nenhum casar, o erro é tratado como recuperável.

    Args:
        error: Exceção a ser analisada.

    Returns:
        True se o erro é recuperável, False caso contrário.
    """
    if isinstance(error, ProviderTransientError):
        return True
    if isinstance(
        error,
        (ProviderError, ProviderConfigurationError, ProviderOutputError),
    ):
        return False
    if _ModelError is not None and isinstance(error, _ModelError):
        return bool(error.is_retryable)

    status = _http_error_status(error)
    if status is not None:
        return _is_recoverable_http_status(status)

    error_str = f"{type(error).__name__}: {error}".lower()

    # Só o padrão não-recuperável muda a decisão; sem ele, o erro é transitório.
    return not any(
        _matches_error_pattern(pattern, error_str) for pattern in _NON_RECOVERABLE_PATTERNS
    )


def is_rate_limit_error(error: Exception) -> bool:
    """Verifica se um erro é especificamente de rate limit.

    É a decisão que reduz os workers na execução paralela e que escolhe a caixa
    de limite de requisições em get_friendly_error_message.

    Args:
        error: Exceção a ser analisada.

    Returns:
        True se o erro é de rate limit, False caso contrário.
    """
    if isinstance(error, ProviderOverloadedError):
        return True
    # Rate limit do pacote é só ProviderOverloadedError. Os demais ProviderError e
    # o ProviderOutputError decidem pela classe, sem ler o texto: a mensagem deles
    # pode trazer texto do modelo, como o resumo da validação e o campo result do
    # SDK no erro do claude_code, e um '429' ali seria tomado por rate limit. O
    # codex classifica o rate limit na origem, pelo código de erro do turno ou do
    # JSON-RPC, e o erro que traz "429 Too Many Requests" só no texto não reduz os
    # workers.
    if isinstance(error, (ProviderError, ProviderOutputError)):
        return False
    status = _http_error_status(error)
    if status in _OVERLOADED_STATUSES or (
        _ModelRateLimitError is not None and isinstance(error, _ModelRateLimitError)
    ):
        return True
    if _ModelError is not None and isinstance(error, _ModelError):
        return False

    if status is not None:
        return False

    error_str = f"{type(error).__name__}: {error}".lower()
    rate_limit_patterns = ("ratelimit", "resourceexhausted", "toomanyrequests", "429")
    return any(_matches_error_pattern(pattern, error_str) for pattern in rate_limit_patterns)


def _usage_limit_error(error: Exception) -> ProviderUsageLimitError | None:
    """Cota de uso esgotada, que só se resolve na conta e interrompe a execução.

    O `insufficient_quota` da OpenAI chega como 429: repetir a linha não traz
    crédito, e reduzir os workers não ajuda. O código é lido do atributo `code`
    que o SDK preenche com o corpo da resposta, na exceção ou numa causa
    explícita. O limite do plano do Tavily e o crédito esgotado do Exa são
    convertidos na própria ferramenta de busca, onde o status é lido.
    """
    if any(
        getattr(current, "code", None) == "insufficient_quota"
        for current in _explicit_cause_chain(error)
    ):
        return ProviderUsageLimitError(
            f"cota da conta no provider de LLM esgotada; adicione crédito ({error})"
        )
    return None


# Maior expoente do backoff. Com base_delay float, base_delay * 2**1024
# levanta OverflowError ao converter o int para float; com o expoente
# limitado, o produto chega no máximo a inf, e o min o corta em max_delay.
_MAX_BACKOFF_EXPONENT = 1023


def retry_with_backoff(
    func: Callable[[], dict[str, Any]],
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    *,
    on_failure: Callable[[Exception], None] | None = None,
) -> dict[str, Any]:
    """Executa função com retry e backoff exponencial.

    Args:
        func: Função a ser executada, que devolve o dicionário de resultado.
        max_retries: Número máximo de tentativas; o core já recusou valor < 1.
        base_delay: Delay base em segundos.
        max_delay: Delay máximo em segundos.
        on_failure: Anexa contexto à exceção terminal, depois da última tentativa.

    Returns:
        O dicionário devolvido por func, com as novas tentativas em '_retry_info'.

    Raises:
        ProviderAbortError: Repassado sem nova tentativa e sem aviso; a cota de
            uso esgotada chega como ProviderUsageLimitError.
        Exception: Última exceção após esgotar tentativas ou erro não-recuperável.
    """
    retry_info = {"retries": 0}

    attempt = 0
    try:
        while True:
            try:
                result = func()
                # Dentro do try: uma falha ao anotar o resultado também ganha nova tentativa.
                result["_retry_info"] = retry_info
                return result  # noqa: TRY300 (o else mudaria quais falhas são re-tentadas)
            except ProviderAbortError:
                # Parada deliberada: o aviso "Execução interrompida" a explica.
                raise
            except Exception as e:
                usage_limit = _usage_limit_error(e)
                if usage_limit is not None:
                    raise usage_limit from e
                error_name = type(e).__name__

                # Verificar se é erro não-recuperável
                if not is_recoverable_error(e):
                    warnings.warn(
                        f"Erro não-recuperável detectado ({error_name}). Não será feito retry.",
                        stacklevel=_user_stacklevel(),
                    )
                    raise

                # Última tentativa - não fazer mais retry
                if attempt >= max_retries - 1:
                    raise

                # Calcular delay com backoff exponencial
                delay = min(base_delay * (2 ** min(attempt, _MAX_BACKOFF_EXPONENT)), max_delay)
                jitter = random.uniform(0, 0.1) * delay  # noqa: S311 (jitter de espera, sem uso criptográfico)
                total_delay = delay + jitter

                retry_info["retries"] = attempt + 1

                # Warning informativo sobre o retry
                warnings.warn(
                    f"Tentativa {attempt + 1}/{max_retries} falhou ({error_name}). "
                    f"Aguardando {total_delay:.1f}s antes de tentar novamente...",
                    stacklevel=_user_stacklevel(),
                )

                time.sleep(total_delay)

            attempt += 1
    except Exception as error:
        if on_failure is not None:
            on_failure(error)
        raise
