"""Task-based model routing, ``provider:model`` specs and explicit fallback.

Invariants enforced here:

* **A model spec names its provider.** ``<provider>:<native id>`` -
  ``openrouter:anthropic/claude-sonnet-5``, ``claude:claude-sonnet-5`` (the
  Claude Code CLI; its aliases ``claude:sonnet`` / ``claude:opus`` too),
  ``script:<anything>`` (the scripted client). :func:`parse_model_spec`
  splits on the first ``:`` only when the prefix is a name in
  :data:`KNOWN_PROVIDERS`, so a colon inside a native id survives; an
  unprefixed id means the *default provider* (the first provider the run
  configured), so every older command line keeps its meaning. ``anthropic:``
  is reserved for a direct Messages-API client that does not exist and is
  refused with a message that names the two routes that do.
* **Capabilities follow the provider, not the model.** The CLI route carries
  no caller tools (``supports_tools=False``) and requests structured output
  through ``--json-schema`` (``supports_structured=True``);
  :data:`PROVIDER_CAPABILITIES` is the table.
* **No fallback unless the user asked for one (decision 4).** One project
  uses one model where possible: :func:`default_router` has no fallback for
  any provider. ``--llm-fallback <spec>`` adds one explicitly and
  ``--llm-fallback same`` (:func:`same_model_fallback`) is the primary's
  model on the *other* configured provider. :data:`DEFAULT_FALLBACK_MODEL`
  stays only as the id a ``haiku``-style shorthand may name; nothing uses it
  by default.
* The router only *orders* candidates (:meth:`ModelRouter.candidates`,
  deduplicated by spec); whether a candidate is tried at all is the
  service's decision (budget, retryable error class, tool support).

Every request carries a completion cap: :attr:`ModelConfig.max_tokens` when
the config sets one, else :data:`DEFAULT_MAX_TOKENS` for the task
(:data:`DEFAULT_MAX_TOKENS_ANY` for tasks not listed). A request without a
cap would let a single call spend an unbounded amount against an approved
budget; the service additionally lowers the cap to the remaining token budget.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Mapping, NamedTuple, Sequence

from pydantic import BaseModel, Field

#: OpenRouter id used by :func:`default_router` when no model is named (the default provider is OpenRouter)
DEFAULT_PRIMARY_MODEL = "anthropic/claude-sonnet-5"
#: the OpenRouter id a ``haiku``-style ``--llm-fallback`` shorthand may name; NOT a default (decision 4: no fallback unless asked)
DEFAULT_FALLBACK_MODEL = "anthropic/claude-haiku-4.5"
#: the Claude Code CLI model used when the default provider is ``claude`` and no model is named
DEFAULT_CLAUDE_MODEL = "claude-sonnet-5"

#: provider names a spec may carry, in the order the CLI documents them
KNOWN_PROVIDERS: tuple[str, ...] = ("openrouter", "claude", "script")
#: prefixes reserved for routes that are not implemented; the value is the refusal text
RESERVED_PROVIDERS: Mapping[str, str] = {
    "anthropic": "direct Messages-API access is not implemented; use claude: (Claude Code CLI) or openrouter:",
}
#: OpenRouter names Anthropic's models ``anthropic/<id>``; the Claude Code CLI takes ``<id>`` - the only translation known here
OPENROUTER_ANTHROPIC_PREFIX = "anthropic/"
#: a full Claude Code CLI model name starts with this; anything else (``sonnet``, ``opus``, ``fable``) is a CLI-only alias
CLAUDE_CLI_FULL_NAME_PREFIX = "claude-"
#: the only id shape spelled the same on both routes: ``claude-<family>-<major>`` (``claude-sonnet-5``, measured on the
#: CLI and the repo's OpenRouter default). A minor version (OpenRouter ``claude-haiku-4.5`` vs the CLI's
#: ``claude-haiku-4-5``) or a date suffix (``-20251001``) is spelled differently or unknown there, so it is not translated.
SAME_SPELLING_ID_RE = re.compile(r"claude-[a-z]+-[0-9]+")


class ProviderCapabilities(NamedTuple):
    supports_tools: bool
    supports_structured: bool


#: what each route can carry; an unprefixed spec without a default provider gets the permissive row
PROVIDER_CAPABILITIES: Mapping[str, ProviderCapabilities] = {
    "openrouter": ProviderCapabilities(supports_tools=True, supports_structured=True),
    # the CLI route carries no caller tools; structured output goes through ``--json-schema``
    "claude": ProviderCapabilities(supports_tools=False, supports_structured=True),
    "script": ProviderCapabilities(supports_tools=True, supports_structured=True),
}
_ANY_CAPABILITIES = ProviderCapabilities(supports_tools=True, supports_structured=True)


class TaskKind(StrEnum):
    REQUIREMENT_ANALYSIS = "requirement_analysis"
    COMPONENT_PROPOSAL = "component_proposal"
    CIRCUIT_DESIGN = "circuit_design"
    REGULATORY_RESEARCH = "regulatory_research"
    RESULT_INTERPRETATION = "result_interpretation"
    REVIEW = "review"
    REPAIR_PLANNING = "repair_planning"
    CHAT = "chat"


#: completion-token cap sent with every request of a task unless the caller or the config says otherwise
DEFAULT_MAX_TOKENS: dict[TaskKind, int] = {
    TaskKind.REQUIREMENT_ANALYSIS: 4096,
    TaskKind.COMPONENT_PROPOSAL: 4096,
    TaskKind.CIRCUIT_DESIGN: 8192,
    TaskKind.REGULATORY_RESEARCH: 4096,
    TaskKind.RESULT_INTERPRETATION: 2048,
    TaskKind.REVIEW: 4096,
    TaskKind.REPAIR_PLANNING: 2048,
    TaskKind.CHAT: 2048,
}
#: cap for a task without an entry above
DEFAULT_MAX_TOKENS_ANY = 2048


def max_tokens_for(task: TaskKind, cfg: "ModelConfig | None" = None, requested: int | None = None) -> int:
    """The completion cap a request must carry: ``requested``, else ``cfg.max_tokens``, else the task default (never ``None``)."""
    if requested is not None:
        return max(1, int(requested))
    if cfg is not None and cfg.max_tokens is not None:
        return max(1, int(cfg.max_tokens))
    return DEFAULT_MAX_TOKENS.get(task, DEFAULT_MAX_TOKENS_ANY)


class ModelConfig(BaseModel):
    #: the provider's native model id, e.g. "anthropic/claude-sonnet-5" (OpenRouter) or "claude-sonnet-5" (Claude Code CLI)
    model: str
    #: the provider that serves ``model`` (a name in :data:`KNOWN_PROVIDERS`); ``None`` = the service's default provider
    provider: str | None = None
    temperature: float = 0.0
    #: completion cap for this model; ``None`` means "the task default" (:func:`max_tokens_for`), never "unlimited"
    max_tokens: int | None = None
    supports_tools: bool = True
    supports_structured: bool = True

    @property
    def spec(self) -> str:
        """``provider:model`` when the provider is set, else the bare model id."""
        return f"{self.provider}:{self.model}" if self.provider else self.model


def split_model_spec(text: str) -> tuple[str | None, str]:
    """``(provider, native id)`` of a spec; the provider is ``None`` for an unprefixed id.

    Splits on the first ``:`` only when the prefix (case-insensitive) is a
    known provider, so ``some/model:free`` stays one id and
    ``openrouter:some/model:free`` keeps its inner colon. ``ValueError`` for an
    empty spec, a known prefix with nothing after it, and a reserved prefix
    (``anthropic:``).
    """
    spec = (text or "").strip()
    if not spec:
        raise ValueError("model spec is empty")
    if ":" not in spec:
        return None, spec
    prefix, rest = spec.split(":", 1)
    name = prefix.strip().lower()
    if name in RESERVED_PROVIDERS:
        raise ValueError(f"model spec {spec!r}: {RESERVED_PROVIDERS[name]}")
    if name not in KNOWN_PROVIDERS:
        return None, spec
    model = rest.strip()
    if not model:
        raise ValueError(f"model spec {spec!r} names the provider {name!r} but no model")
    return name, model


def check_provider_name(provider: str | None) -> str | None:
    """``provider`` when it is ``None`` or a known provider; ``ValueError`` otherwise (reserved names get their own message)."""
    if provider is None:
        return None
    name = provider.strip().lower()
    if name in RESERVED_PROVIDERS:
        raise ValueError(f"provider {provider!r}: {RESERVED_PROVIDERS[name]}")
    if name not in KNOWN_PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}: use one of {', '.join(KNOWN_PROVIDERS)}")
    return name


def parse_model_spec(
    text: str,
    default_provider: str | None = None,
    *,
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> ModelConfig:
    """The :class:`ModelConfig` for a spec; an unprefixed id belongs to ``default_provider`` (``None`` keeps it provider-less).

    Tool / structured-output support comes from :data:`PROVIDER_CAPABILITIES`
    for the resolved provider (the permissive row when none is resolved).
    """
    provider, model = split_model_spec(text)
    resolved = provider or check_provider_name(default_provider)
    caps = PROVIDER_CAPABILITIES.get(resolved, _ANY_CAPABILITIES) if resolved else _ANY_CAPABILITIES
    return ModelConfig(
        model=model, provider=resolved, temperature=temperature, max_tokens=max_tokens,
        supports_tools=caps.supports_tools, supports_structured=caps.supports_structured,
    )


def default_model(provider: str | None) -> str:
    """The model :func:`default_router` names when the user names none: the CLI's Sonnet for ``claude``, OpenRouter's otherwise."""
    return DEFAULT_CLAUDE_MODEL if check_provider_name(provider) == "claude" else DEFAULT_PRIMARY_MODEL


class ModelRouter(BaseModel):
    default: ModelConfig
    by_task: dict[TaskKind, ModelConfig] = Field(default_factory=dict)
    fallbacks: list[ModelConfig] = Field(default_factory=list)

    def for_task(self, task: TaskKind) -> ModelConfig:
        return self.by_task.get(task, self.default)

    def candidates(self, task: TaskKind) -> list[ModelConfig]:
        """The task's primary, then the fallbacks in order; every spec appears once."""
        primary = self.for_task(task)
        out = [primary]
        seen = {primary.spec}
        for f in self.fallbacks:
            if f.spec not in seen:
                seen.add(f.spec)
                out.append(f)
        return out

    def configs(self) -> list[ModelConfig]:
        """Every config the router can name (default, task overrides, fallbacks) - for provider checks before any call."""
        return [self.default, *self.by_task.values(), *self.fallbacks]

    def providers(self) -> list[str | None]:
        """The distinct providers the configs name, in order (``None`` = the default provider)."""
        out: list[str | None] = []
        for c in self.configs():
            if c.provider not in out:
                out.append(c.provider)
        return out


def default_router(
    model: str | None = None,
    *,
    fallback: str | Sequence[str] | None = None,
    default_provider: str | None = None,
    by_task: Mapping[TaskKind | str, str] | None = None,
) -> ModelRouter:
    """The default routing: ``model`` (a spec; default Sonnet on the default provider) for every task, temperature 0, NO fallback.

    ``fallback`` is a spec or a list of specs the user asked for explicitly
    (``--llm-fallback``); a fallback equal to the primary is dropped by
    :meth:`ModelRouter.candidates`. ``by_task`` maps a task to a spec
    (``--llm-task-model TASK=SPEC``; an unknown task name is a
    ``ValueError`` listing the tasks). Unprefixed specs belong to
    ``default_provider``.
    """
    default_provider = check_provider_name(default_provider)
    primary = parse_model_spec(model or default_model(default_provider), default_provider)
    specs = [fallback] if isinstance(fallback, str) else list(fallback or [])
    fallbacks = [parse_model_spec(spec, default_provider) for spec in specs]
    overrides: dict[TaskKind, ModelConfig] = {TaskKind.REQUIREMENT_ANALYSIS: primary}
    for task, spec in (by_task or {}).items():
        try:
            kind = TaskKind(task)
        except ValueError:
            raise ValueError(f"unknown task {task!r}: use one of {', '.join(t.value for t in TaskKind)}") from None
        overrides[kind] = parse_model_spec(spec, default_provider)
    return ModelRouter(default=primary, by_task=overrides, fallbacks=fallbacks)


def translate_model(model: str, source: str, target: str) -> str:
    """``model`` (a native id of ``source``) as ``target`` names it, or ``ValueError`` when no counterpart is known.

    The only translation known: OpenRouter's ``anthropic/<id>`` <-> the Claude
    Code CLI's ``<id>``, and only for an ``<id>`` of the shape
    ``claude-<family>-<major>`` (:data:`SAME_SPELLING_ID_RE`) - the one shape
    spelled identically on both routes. An id with a minor version or a date
    (OpenRouter ``anthropic/claude-haiku-4.5`` is the CLI's
    ``claude-haiku-4-5``), a CLI alias (``sonnet``, ``opus``, ``fable``: no
    ``claude-`` prefix) and a non-Anthropic OpenRouter id have no known
    counterpart: the user names the fallback explicitly. The scripted provider
    answers any id, so it takes and gives ids verbatim.
    """
    source, target = check_provider_name(source) or "", check_provider_name(target) or ""
    if source == target or "script" in (source, target):
        return model
    if source == "openrouter" and target == "claude":
        if not model.startswith(OPENROUTER_ANTHROPIC_PREFIX):
            raise ValueError(f"{model!r} is not an Anthropic model on OpenRouter: no counterpart on the Claude Code CLI; name the fallback explicitly")
        native = model[len(OPENROUTER_ANTHROPIC_PREFIX):]
    elif source == "claude" and target == "openrouter":
        if not model.startswith(CLAUDE_CLI_FULL_NAME_PREFIX):
            raise ValueError(f"{model!r} is a Claude Code CLI alias, not a full model name: no counterpart on OpenRouter; name the fallback explicitly")
        native = model
    else:
        raise ValueError(f"no model translation from {source!r} to {target!r}")
    if not SAME_SPELLING_ID_RE.fullmatch(native):
        raise ValueError(
            f"{model!r}: only a claude-<family>-<major> id is known to be spelled the same on {source} and {target} "
            "(a minor version or a date is spelled differently, e.g. OpenRouter anthropic/claude-haiku-4.5 is the CLI's "
            "claude-haiku-4-5); name the fallback explicitly"
        )
    return native if target == "claude" else OPENROUTER_ANTHROPIC_PREFIX + native


def same_model_on(cfg: ModelConfig, provider: str) -> ModelConfig:
    """``cfg``'s model as a config of ``provider`` (capabilities from the table; temperature and cap kept).

    A provider-less ``cfg`` counts as OpenRouter (the historical default); :func:`same_model_fallback`
    resolves it against the run's configuration order first.
    """
    source = cfg.provider or "openrouter"
    return parse_model_spec(f"{provider}:{translate_model(cfg.model, source, provider)}", temperature=cfg.temperature, max_tokens=cfg.max_tokens)


def same_model_fallback(primary: ModelConfig, providers: Sequence[str]) -> ModelConfig:
    """``--llm-fallback same``: the primary's model on the other configured provider.

    ``providers`` is the run's configuration order (``--llm a,b``); the
    primary's provider (the first configured one when it is unprefixed) is
    skipped and the next one takes the translated id. ``ValueError`` when
    there is no other provider or the model has no counterpart there.
    """
    names = [check_provider_name(p) or "" for p in providers]
    if not names:
        raise ValueError("--llm-fallback same needs a configured provider")
    own = primary.provider or names[0]
    others = [n for n in names if n != own]
    if not others:
        raise ValueError(f"--llm-fallback same needs a second configured provider (configured: {', '.join(names)})")
    return same_model_on(primary.model_copy(update={"provider": own}), others[0])


__all__ = [
    "CLAUDE_CLI_FULL_NAME_PREFIX",
    "DEFAULT_CLAUDE_MODEL",
    "DEFAULT_FALLBACK_MODEL",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MAX_TOKENS_ANY",
    "DEFAULT_PRIMARY_MODEL",
    "KNOWN_PROVIDERS",
    "OPENROUTER_ANTHROPIC_PREFIX",
    "PROVIDER_CAPABILITIES",
    "RESERVED_PROVIDERS",
    "SAME_SPELLING_ID_RE",
    "ModelConfig",
    "ModelRouter",
    "ProviderCapabilities",
    "TaskKind",
    "check_provider_name",
    "default_model",
    "default_router",
    "max_tokens_for",
    "parse_model_spec",
    "same_model_fallback",
    "same_model_on",
    "split_model_spec",
    "translate_model",
]
