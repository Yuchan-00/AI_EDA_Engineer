"""Network policy: whether the system may go online at all, and which URLs it may fetch.

Invariants this module enforces:

* **The network is an outward action.** A :class:`NetworkPolicy` is
  ``approved`` only when the *user* created an online session
  (:meth:`NetworkPolicy.from_cli` with ``online=True`` grants and consumes
  :attr:`~ai_eda.security.ExternalAction.NETWORK_FETCH` with the detail
  :data:`ONLINE_SESSION_DETAIL` once, so the approval gate's audit shows
  both). Without it :meth:`NetworkPolicy.require_online` raises
  :class:`~ai_eda.errors.ApprovalRequiredError` - the archive calls it before
  any socket is opened.
* **Only trusted origins are fetched.** A URL is fetchable when (a)/(b) its
  host is in :attr:`NetworkPolicy.trusted_hosts` (the host of a KiCad
  library ``Datasheet`` URL, or a domain from the curated official-source
  list - callers add them with :meth:`NetworkPolicy.trust_host` and the
  reason is kept), or (c) it is *exactly* one of the user-supplied
  :attr:`NetworkPolicy.user_urls`. A URL proposed by a model is never
  fetched unless it satisfies one of these; (c) never widens to the host, so
  a model URL on the same host as a user URL is still refused.
* **Redirects stay on trusted ground.** A redirect target is allowed only on
  the same origin as the URL that was trusted, or on another trusted host
  (:meth:`NetworkPolicy.redirect_allowed`). ``ecfr.gov ->
  unblock.federalregister.gov`` is therefore refused before the second host
  is contacted.
* **Only https.** :func:`normalise_url` upgrades ``http://`` and scheme-less
  library URLs (``www.ti.com/lit/gpn/TLV767``) to ``https://`` and records
  that it did; any other scheme, an empty host or an unusable string
  (``~``, ``hhttps://...``) raises ``ValueError`` - a broken pointer is
  reported, never repaired by guessing.

Hosts compare case-insensitively with a leading ``www.`` ignored
(``www.ti.com`` and ``ti.com`` are one origin); other subdomains are
distinct (``assets.nexperia.com`` is not ``www.nexperia.com``), so an
allow-list must name each one.
"""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

from ai_eda.security.approval import ApprovalGate, ExternalAction, default_gate, require_approval

#: the approval detail of an online session; granted once by the user (CLI ``--online``)
ONLINE_SESSION_DETAIL = "online session"

_ALLOWED_SCHEMES = ("https", "http")


def host_key(host: str | None) -> str:
    """Canonical comparison key of a host: lower-case, trailing dot and leading ``www.`` removed."""
    h = (host or "").strip().lower().rstrip(".")
    if h.startswith("www."):
        h = h[4:]
    return h


def host_of(url: str) -> str:
    """The lower-case host of ``url`` (empty when it has none)."""
    return (urlsplit(url).hostname or "").lower()


def normalise_url(url: str) -> tuple[str, str | None]:
    """``(https URL, note)`` for a URL the archive may attempt, or ``ValueError`` when it is unusable.

    Surrounding whitespace and quotes are stripped (a KiCad library carries
    ``\\"https://assets.nexperia.com/...``), ``http://`` becomes ``https://``
    and a scheme-less ``www.host/path`` gets ``https://`` - each noted in the
    returned ``note``. The fragment is dropped, the host lower-cased. Any
    other scheme (``ftp``, ``file``, the ``hhttps`` typo), ``~``, an empty
    string or a URL without a host raises ``ValueError`` naming the problem.
    """
    raw = (url or "").strip().strip('"\'').strip()
    if not raw or raw == "~":
        raise ValueError("no URL")
    notes: list[str] = []
    if "://" not in raw:
        first = raw.split("/", 1)[0]
        if raw.startswith("//"):
            raw = "https:" + raw
        elif ":" in first and "." not in first.split(":", 1)[0]:
            raise ValueError(f"unsupported scheme {first.split(':', 1)[0]!r} (only https is fetched)")
        else:
            raw = "https://" + raw  # ``www.ti.com/lit/gpn/TLV767`` or ``host:port/path``
        notes.append("scheme-less URL taken as https")
    parts = urlsplit(raw)
    scheme = parts.scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise ValueError(f"unsupported scheme {parts.scheme!r} (only https is fetched)")
    if scheme == "http":
        notes.append("http upgraded to https (plain http is never used)")
    host = (parts.hostname or "").lower()
    if not host:
        raise ValueError(f"URL has no host: {raw!r}")
    netloc = host
    if parts.port is not None and parts.port != 443:
        netloc = f"{host}:{parts.port}"
    if parts.username or parts.password:
        raise ValueError("credentials in URLs are not supported")
    path = parts.path or "/"
    out = urlunsplit(("https", netloc, path, parts.query, ""))
    return out, ("; ".join(notes) or None)


class NetworkPolicy:
    """What the archive may fetch. See the module docstring for the rules."""

    def __init__(
        self,
        approved: bool,
        trusted_hosts: set[str] | None = None,
        user_urls: dict[str, str] | None = None,
        gate: ApprovalGate | None = None,
        *,
        approved_by: str = "user",
    ) -> None:
        self.gate = gate or default_gate()
        self.approved = False
        #: host key -> why it is trusted
        self.trust_reasons: dict[str, str] = {}
        for h in sorted(trusted_hosts or ()):
            self.trust_host(h, "trusted host given to the policy")
        #: normalised user URL -> the key the user gave it under (``R1``, ``eu_lvd``)
        self._user_urls: dict[str, str] = {}
        #: key -> user URL as given (kept for reporting)
        self.user_urls: dict[str, str] = {}
        #: key -> why the user's URL is unusable (never fetched, reported)
        self.invalid_user_urls: dict[str, str] = {}
        for key, u in (user_urls or {}).items():
            self.user_urls[key] = u
            try:
                norm, _ = normalise_url(u)
            except ValueError as e:
                self.invalid_user_urls[key] = str(e)
                continue
            self._user_urls[norm] = key
        if approved:
            # The online session *is* the user's approval of network fetches; record and consume it so the audit shows both.
            self.gate.grant(ExternalAction.NETWORK_FETCH, ONLINE_SESSION_DETAIL, approved_by=approved_by)
            require_approval(ExternalAction.NETWORK_FETCH, ONLINE_SESSION_DETAIL, self.gate)
            self.approved = True

    @classmethod
    def from_cli(
        cls,
        online: bool,
        trusted_hosts: set[str] | None = None,
        user_urls: dict[str, str] | None = None,
        gate: ApprovalGate | None = None,
        *,
        approved_by: str = "cli --online",
    ) -> NetworkPolicy:
        """A policy for one CLI run: ``--online`` grants (and consumes) ``NETWORK_FETCH`` once; without it nothing is fetched."""
        return cls(bool(online), trusted_hosts, user_urls, gate, approved_by=approved_by)

    # ------------------------------------------------------------ trust

    @property
    def trusted_hosts(self) -> set[str]:
        """Host keys trusted by (a)/(b)."""
        return set(self.trust_reasons)

    def trust_host(self, host: str, reason: str) -> None:
        """Trust every URL on ``host`` (a KiCad Datasheet URL's host, an official domain); the first reason is kept."""
        key = host_key(host_of(host) if "://" in host else host.split("/", 1)[0].split(":", 1)[0])
        if not key:
            raise ValueError(f"not a host: {host!r}")
        self.trust_reasons.setdefault(key, reason)

    def is_trusted_host(self, host: str) -> bool:
        return host_key(host) in self.trust_reasons

    def user_url_key(self, normalised_url: str) -> str | None:
        """The key under which the user supplied exactly this URL, else ``None``."""
        return self._user_urls.get(normalised_url)

    def trusted_origin(self, normalised_url: str) -> tuple[str | None, str | None]:
        """``(origin host key, why)`` when ``normalised_url`` may be fetched, else ``(None, why not)``."""
        host = host_key(host_of(normalised_url))
        if host in self.trust_reasons:
            return host, self.trust_reasons[host]
        key = self._user_urls.get(normalised_url)
        if key is not None:
            return host, f"URL supplied by the user under key {key!r}"
        return None, f"host {host!r} is not a trusted origin (not a KiCad library datasheet host, not in the official-domain allow-list, not a user-supplied URL)"

    def redirect_allowed(self, origin: str, target_url: str) -> str | None:
        """``None`` when a redirect from ``origin`` to ``target_url`` may be followed, else the reason it may not."""
        target = host_key(host_of(target_url))
        if not target:
            return f"redirect target has no host: {target_url!r}"
        if target == host_key(origin) or target in self.trust_reasons:
            return None
        return f"redirect to untrusted host {target!r} (from origin {host_key(origin)!r}) refused"

    # ------------------------------------------------------------ approval

    def require_online(self) -> None:
        """Raise :class:`~ai_eda.errors.ApprovalRequiredError` unless the user opened an online session.

        A policy created without approval still consults its gate: a
        ``NETWORK_FETCH`` / :data:`ONLINE_SESSION_DETAIL` approval granted on
        it directly counts (and is consumed, as every approval is); otherwise
        the gate records the denial and raises.
        """
        if self.approved:
            return
        require_approval(ExternalAction.NETWORK_FETCH, ONLINE_SESSION_DETAIL, self.gate)
        self.approved = True

    def describe(self) -> dict[str, object]:
        """A JSON-able summary for logs and validation details (no secrets involved)."""
        return {
            "approved": self.approved,
            "trusted_hosts": dict(sorted(self.trust_reasons.items())),
            "user_urls": dict(self.user_urls),
            "invalid_user_urls": dict(self.invalid_user_urls),
        }


__all__ = ["ONLINE_SESSION_DETAIL", "NetworkPolicy", "host_key", "host_of", "normalise_url"]
