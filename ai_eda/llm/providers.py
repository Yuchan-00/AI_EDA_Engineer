"""Provider routing: one :class:`~ai_eda.llm.client.LLMClient` over several providers, plus the provider table for the GUI.

Invariants enforced here:

* **A spec is resolved before anything is sent.** :class:`ProviderClient`
  takes ``provider:model`` specs (:func:`~ai_eda.llm.router.parse_model_spec`;
  an unprefixed id is the client's default provider) and delegates to the
  member with the *native* id. A reserved prefix (``anthropic:``) or a
  provider this client was not configured with is a ``ValueError`` raised
  before any member is called - nothing is sent, nothing is billed.
* **Every reply and error names the route that served it.** The response's
  ``via`` is the provider name and ``model_used`` is the *served spec*
  (``provider:<the id the provider reported>``), so the accounting never
  mixes a subscription call with a charged one; an
  :class:`~ai_eda.llm.client.LLMError` from a member is re-raised with the
  same ``via`` / prefixed ``model``. A bare (single-provider) client keeps
  returning native ids; only this client adds the prefix.
* **Billing is the members' billing.** ``paid`` is true when any member is
  paid; ``billing`` is ``"per_call"`` / ``"subscription"`` when every member
  agrees and ``"mixed"`` otherwise (the service then requires a budget for
  the per-call side and records the subscription use).
* :func:`describe_providers` is the pure view the GUI's run panel uses to
  offer exactly the providers that would work (OpenRouter: key set; Claude
  Code CLI: found and logged in). It never calls a model.

Nothing here decides design truth: a routed reply is the same proposal the
member returned.
"""

from __future__ import annotations

import inspect
import os
from typing import Any, Callable, Iterator, Mapping, Sequence

from pydantic import BaseModel

from ai_eda.errors import ToolUnavailableError
from ai_eda.llm.client import LLMClient, LLMError, LLMMessage, LLMResponse, ToolSpec, Usage
from ai_eda.llm.router import check_provider_name, split_model_spec

#: mirrors ``ai_eda.llm.openrouter.ENV_KEY`` without importing that module (it pulls in httpx)
OPENROUTER_ENV_KEY = "OPENROUTER_API_KEY"
#: the two routes a run panel can offer; ``script`` is the test double and takes a file, not a switch
REAL_PROVIDERS: tuple[str, ...] = ("openrouter", "claude")
#: how each provider bills (a scripted member is "per_call" by the client contract's default; it is also ``paid=False``)
PROVIDER_BILLING: Mapping[str, str] = {"openrouter": "per_call", "claude": "subscription", "script": "per_call"}
#: class names of the bare clients, for :func:`provider_names_of` without importing httpx or the CLI module
_CLIENT_CLASS_PROVIDERS: Mapping[str, str] = {
    "OpenRouterClient": "openrouter",
    "ClaudeCodeClient": "claude",
    "ScriptedLLMClient": "script",
}

ClientFactory = Callable[..., LLMClient]


# ---------------------------------------------------------------- routing client


class ProviderClient(LLMClient):
    """One client over several providers; see the module docstring."""

    def __init__(self, clients: Mapping[str, LLMClient], default: str) -> None:
        if not clients:
            raise ValueError("ProviderClient needs at least one member client")
        members: dict[str, LLMClient] = {}
        for name, client in clients.items():
            key = check_provider_name(name)
            if key is None or key in members:
                raise ValueError(f"duplicate or empty provider name {name!r}")
            members[key] = client
        default_name = check_provider_name(default)
        if default_name not in members:
            raise ValueError(f"default provider {default!r} is not a member (members: {', '.join(members)})")
        self._members = members
        self.default: str = default_name
        #: ``(provider, native model, member)`` of the most recent :meth:`stream`, for ``last_stream_*``
        self._last_stream: tuple[str, str, LLMClient] | None = None

    def __repr__(self) -> str:
        return f"ProviderClient(providers={self.providers!r}, default={self.default!r})"

    @property
    def providers(self) -> list[str]:
        """Member names in configuration order."""
        return list(self._members)

    @property
    def members(self) -> dict[str, LLMClient]:
        return dict(self._members)

    def member(self, provider: str) -> LLMClient:
        name = check_provider_name(provider)
        if name not in self._members:
            raise ValueError(f"provider {provider!r} is not configured (configured: {', '.join(self._members)})")
        return self._members[name]

    @property
    def paid(self) -> bool:  # type: ignore[override]
        return any(getattr(m, "paid", True) for m in self._members.values())

    @property
    def billing(self) -> str:  # type: ignore[override]
        kinds = {getattr(m, "billing", "per_call") for m in self._members.values()}
        return next(iter(kinds)) if len(kinds) == 1 else "mixed"

    def resolve(self, spec: str) -> tuple[str, str, LLMClient]:
        """``(provider, native id, member)`` for a spec; ``ValueError`` for a reserved or unconfigured provider."""
        provider, model = split_model_spec(spec)
        name = provider or self.default
        return name, model, self.member(name)

    # ------------------------------------------------------------ stamping

    @staticmethod
    def _served(provider: str, model: str, resp: LLMResponse) -> LLMResponse:
        used = resp.model_used or model
        if not used.startswith(provider + ":"):
            used = f"{provider}:{used}"
        return resp.model_copy(update={"model": f"{provider}:{model}", "model_used": used, "via": provider})

    @staticmethod
    def _failed(provider: str, model: str, err: LLMError) -> LLMError:
        err.via = provider
        native = err.model or model
        err.model = native if native.startswith(provider + ":") else f"{provider}:{native}"
        return err

    # ------------------------------------------------------------ LLMClient

    def complete(
        self,
        model: str,
        messages: list[LLMMessage],
        tools: list[ToolSpec] | None = None,
        response_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        provider, native, member = self.resolve(model)
        try:
            resp = member.complete(native, messages, tools=tools, response_schema=response_schema, temperature=temperature, max_tokens=max_tokens)
        except LLMError as e:
            raise self._failed(provider, native, e)
        return self._served(provider, native, resp)

    def stream(
        self,
        model: str,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        # resolved eagerly: an unconfigured provider fails at the call, before the consumer pulls the first piece
        provider, native, member = self.resolve(model)
        self._last_stream = (provider, native, member)
        return self._stream(provider, native, member, messages, temperature, max_tokens)

    @staticmethod
    def _stream(provider: str, native: str, member: LLMClient, messages: list[LLMMessage], temperature: float, max_tokens: int | None) -> Iterator[str]:
        try:
            yield from member.stream(native, messages, temperature=temperature, max_tokens=max_tokens)
        except LLMError as e:
            raise ProviderClient._failed(provider, native, e)

    @property
    def last_stream_usage(self) -> Usage | None:
        """The usage of the most recent stream, from the member that served it."""
        if self._last_stream is None:
            return None
        usage = getattr(self._last_stream[2], "last_stream_usage", None)
        return usage if isinstance(usage, Usage) else None

    @property
    def last_stream_response(self) -> LLMResponse | None:
        """The accounting view of the most recent stream, stamped with ``via`` and the served spec."""
        if self._last_stream is None:
            return None
        provider, native, member = self._last_stream
        final = getattr(member, "last_stream_response", None)
        return self._served(provider, native, final) if isinstance(final, LLMResponse) else None

    def close(self) -> None:
        for member in self._members.values():
            close = getattr(member, "close", None)
            if callable(close):
                close()


# ---------------------------------------------------------------- construction


def client_factory(provider: str) -> ClientFactory:
    """The bare client class of a provider, imported only when asked for (httpx / the CLI module stay optional)."""
    name = check_provider_name(provider)
    if name == "openrouter":
        from ai_eda.llm.openrouter import OpenRouterClient

        return OpenRouterClient
    if name == "claude":
        from ai_eda.llm.claude_cli import ClaudeCodeClient

        return ClaudeCodeClient
    if name == "script":
        from ai_eda.llm.fake import ScriptedLLMClient

        return ScriptedLLMClient
    raise ValueError(f"no client for provider {provider!r}")  # unreachable: check_provider_name refused it


def _accepted(factory: ClientFactory, kw: Mapping[str, Any]) -> dict[str, Any]:
    """The subset of ``kw`` the factory's signature takes (everything when it takes ``**kwargs`` or cannot be inspected)."""
    try:
        params = inspect.signature(factory).parameters
    except (TypeError, ValueError):
        return dict(kw)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kw)
    return {k: v for k, v in kw.items() if k in params}


def check_providers(providers: Sequence[str]) -> tuple[str, ...]:
    """The provider names, validated: at least one, each known, none repeated (``ValueError`` otherwise)."""
    names: list[str] = []
    for p in providers:
        name = check_provider_name(p)
        if name is None:
            raise ValueError("empty provider name")
        if name in names:
            raise ValueError(f"provider {name!r} is configured twice")
        names.append(name)
    if not names:
        raise ValueError("no provider configured")
    return tuple(names)


def build_client(
    providers: Sequence[str],
    *,
    factories: Mapping[str, ClientFactory] | None = None,
    soft_kw: Mapping[str, Any] | None = None,
    **client_kw: Any,
) -> LLMClient:
    """A bare client for one provider, a :class:`ProviderClient` (default = the first) for several.

    ``client_kw`` is routed to each member by its constructor signature (a
    name both accept, such as ``timeout``, reaches both); a name no
    configured factory accepts is a ``TypeError`` before any client is built.
    ``soft_kw`` entries are defaults applied only where accepted (e.g. the
    CLI's ``max_budget_usd`` from the budget). ``factories`` overrides the
    class per provider (tests pass a fake for ``claude``). A member that
    fails to build closes the members built before it.
    """
    names = check_providers(providers)
    chosen = {n: (factories or {}).get(n) or client_factory(n) for n in names}
    accepted_anywhere: set[str] = set()
    for f in chosen.values():
        accepted_anywhere |= set(_accepted(f, client_kw))
    unknown = sorted(set(client_kw) - accepted_anywhere)
    if unknown:
        raise TypeError(f"client option(s) {', '.join(unknown)} are not accepted by any configured provider ({', '.join(names)})")
    members: dict[str, LLMClient] = {}
    try:
        for name, factory in chosen.items():
            kw = {**_accepted(factory, soft_kw or {}), **_accepted(factory, client_kw)}
            members[name] = factory(**kw)
    except BaseException:
        for built in members.values():
            close = getattr(built, "close", None)
            if callable(close):
                close()
        raise
    if len(members) == 1:
        return next(iter(members.values()))
    return ProviderClient(members, default=names[0])


# ---------------------------------------------------------------- naming helpers (used by the service)


def provider_names_of(client: LLMClient) -> list[str]:
    """The provider name(s) behind a client: a :class:`ProviderClient`'s members, else the bare client's class."""
    if isinstance(client, ProviderClient):
        return client.providers
    cls = type(client).__name__
    if cls in _CLIENT_CLASS_PROVIDERS:
        return [_CLIENT_CLASS_PROVIDERS[cls]]
    for base in type(client).__mro__:
        if base.__name__ in _CLIENT_CLASS_PROVIDERS:
            return [_CLIENT_CLASS_PROVIDERS[base.__name__]]
    return [cls.lower()]


def members_of(client: LLMClient, names: Sequence[str] | None = None) -> list[tuple[str, LLMClient]]:
    """``(provider, client)`` pairs: the members of a :class:`ProviderClient`, else the bare client under its name."""
    if isinstance(client, ProviderClient):
        return list(client.members.items())
    name = list(names or provider_names_of(client))[0]
    return [(name, client)]


def cli_path_of(client: LLMClient) -> str | None:
    """The executable a CLI-backed client runs, when it says so (``cli`` / ``cli_path`` / ``executable``)."""
    for attr in ("cli", "cli_path", "executable"):
        value = getattr(client, attr, None)
        if isinstance(value, (str, os.PathLike)) and str(value):
            return str(value)
    return None


def subscription_detail(provider: str, client: LLMClient) -> str:
    """The approval detail of a subscription member: ``<provider> subscription via <cli path> (no per-call cost; usage shown)``."""
    return f"{provider} subscription via {cli_path_of(client) or provider} (no per-call cost; usage shown)"


# ---------------------------------------------------------------- GUI hook


class ProviderInfo(BaseModel):
    name: str
    available: bool
    #: why it is (not) available: the key state, or the CLI path with its version / login state, or the error
    reason: str
    #: "per_call" | "subscription"
    billing: str
    #: the CLI's login state when known (``claude`` only; ``None`` = not reported)
    logged_in: bool | None = None


def _field(state: Any, *names: str) -> Any:
    if isinstance(state, Mapping):
        for n in names:
            if n in state:
                return state[n]
        return None
    for n in names:
        if hasattr(state, n):
            return getattr(state, n)
    return None


def _login_fields(state: Any) -> tuple[bool | None, str | None]:
    """``(logged_in, auth_method)`` from whatever ``login_state()`` returned (mapping or object; missing -> None)."""
    logged = _field(state, "logged_in", "loggedIn")
    method = _field(state, "auth_method", "authMethod")
    return (logged if isinstance(logged, bool) else None), (str(method) if method else None)


def _describe_openrouter(environ: Mapping[str, str]) -> ProviderInfo:
    present = bool(environ.get(OPENROUTER_ENV_KEY, "").strip())
    return ProviderInfo(
        name="openrouter", available=present, billing=PROVIDER_BILLING["openrouter"],
        reason=f"{OPENROUTER_ENV_KEY} set" if present else f"{OPENROUTER_ENV_KEY} not set",
    )


def _describe_claude(factory: Callable[[], Any]) -> ProviderInfo:
    billing = PROVIDER_BILLING["claude"]
    try:
        client = factory()
    except ToolUnavailableError as e:
        return ProviderInfo(name="claude", available=False, billing=billing, reason=str(e))
    except Exception as e:  # noqa: BLE001 - the GUI must get a row, not a traceback
        return ProviderInfo(name="claude", available=False, billing=billing, reason=f"Claude Code CLI client unavailable: {e}")
    path = cli_path_of(client) or "claude"
    version: str | None = None
    logged_in: bool | None = None
    method: str | None = None
    problem: str | None = None
    try:
        get_version = getattr(client, "version", None)
        if callable(get_version):
            v = get_version()
            version = str(v) if v else None
        get_state = getattr(client, "login_state", None)
        if callable(get_state):
            state = get_state()
            logged_in, method = _login_fields(state)
            error = _field(state, "error")
            if isinstance(error, str) and error.strip():
                problem = error.strip()  # the probe ran but could not say (the client reports it instead of raising)
    except Exception as e:  # noqa: BLE001
        problem = str(e)
    state = "yes" if logged_in else ("no" if logged_in is False else "unknown")
    reason = f"{path} ({version or 'version unknown'}, logged in: {state}, auth: {method or 'unknown'})"
    if problem:
        reason += f"; {problem}"
    return ProviderInfo(name="claude", available=logged_in is not False and problem is None, billing=billing, reason=reason, logged_in=logged_in)


def describe_providers(
    *,
    environ: Mapping[str, str] | None = None,
    claude_factory: Callable[[], Any] | None = None,
) -> list[ProviderInfo]:
    """One row per real provider (:data:`REAL_PROVIDERS`) saying whether a run with it would work and why.

    OpenRouter: the key is set. Claude: the CLI is found (``claude_factory``,
    default the real client's discovery) and its ``login_state()`` does not
    say ``logged in: no``; ``version()`` / ``auth status`` are read, no model
    is called. ``environ`` defaults to ``os.environ``.
    """
    env = environ if environ is not None else os.environ
    factory = claude_factory or (lambda: client_factory("claude")())
    return [_describe_openrouter(env), _describe_claude(factory)]


__all__ = [
    "OPENROUTER_ENV_KEY",
    "PROVIDER_BILLING",
    "REAL_PROVIDERS",
    "ClientFactory",
    "ProviderClient",
    "ProviderInfo",
    "build_client",
    "check_providers",
    "cli_path_of",
    "client_factory",
    "describe_providers",
    "members_of",
    "provider_names_of",
    "subscription_detail",
]
