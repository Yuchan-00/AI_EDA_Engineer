"""One run's source session: the network policy, the document archive, the catalog and the candidate list, built from the user's flags.

Invariants this module enforces (the CLI and the tests build their sessions
here, so they agree):

* **Offline by default.** Without ``online=True`` the policy is not approved:
  :meth:`~ai_eda.tools.sources.DocumentArchive.fetch` raises before any socket
  use, every datasheet and official text stays ``not fetched (offline)`` and
  ``NOT_VERIFIED``. ``online=True`` is the user's one approval of
  ``NETWORK_FETCH`` for the session (granted and consumed on the gate, so the
  audit shows both).
* **Trusted origins are enumerated, never inferred from a model.** The
  policy trusts (a) the hosts of the KiCad library ``Datasheet`` fields of
  the IR's parts (and an IR pointer on that same host), (b) every
  ``allowed_domains`` host of the regulatory candidate list, (c) the hosts
  the user named with ``--trust-host``; ``--datasheet-url REF=URL`` and
  ``--source-url ID=URL`` are trusted as exactly those URLs.
* **A catalog is the user's file with the user's date.** ``--catalog`` needs
  ``--catalog-date``; the file's sha256 becomes the source of every sourcing
  value it backs, never of an identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ai_eda.errors import AiEdaError
from ai_eda.ir import CircuitIR
from ai_eda.parts.catalog import CatalogError, CatalogSource
from ai_eda.parts.identity import SOURCES_DIRNAME
from ai_eda.parts.pointers import locate_datasheet
from ai_eda.regulatory.candidates import CandidateList, load_candidates
from ai_eda.security.approval import ApprovalGate
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.sources.archive import DocumentArchive
from ai_eda.tools.sources.policy import NetworkPolicy, host_key


class SessionError(AiEdaError):
    """A usage error in the session flags (the CLI exits 2 with the message)."""


def parse_key_urls(items: Iterable[str] | None, flag: str) -> dict[str, str]:
    """``KEY=URL`` pairs from a repeatable flag; :class:`SessionError` names the offending item."""
    out: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise SessionError(f"{flag} expects KEY=URL, got {item!r}")
        key, url = item.split("=", 1)
        if not key.strip() or not url.strip():
            raise SessionError(f"{flag} expects KEY=URL with a non-empty key and URL, got {item!r}")
        out[key.strip()] = url.strip()
    return out


def kicad_datasheet_hosts(ir: CircuitIR, library: KicadLibrary | None) -> dict[str, str]:
    """Host key -> reason for every datasheet pointer of the IR's parts that rule (a) trusts (the KiCad library's own ``Datasheet`` host)."""
    out: dict[str, str] = {}
    for c in ir.components:
        pointer = locate_datasheet(c, library).pointer
        if pointer is None or pointer.origin == "user" or not pointer.host or not pointer.trust_reason:
            continue
        out.setdefault(pointer.host, pointer.trust_reason)
    return out


@dataclass
class SourceSession:
    """What one run may fetch, where it archives, and the user's catalog / candidate list."""

    policy: NetworkPolicy
    archive: DocumentArchive
    candidates: CandidateList
    sources_dir: Path
    catalog: CatalogSource | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def online(self) -> bool:
        return self.policy.approved

    def tools(self) -> dict[str, Any]:
        """The ``AgentContext.tools`` entries the agents read (``archive``, ``policy``, ``regulatory_candidates`` and, when loaded, ``catalog``)."""
        tools: dict[str, Any] = {"archive": self.archive, "policy": self.policy, "regulatory_candidates": self.candidates}
        if self.catalog is not None:
            tools["catalog"] = self.catalog
        return tools

    def describe(self) -> dict[str, Any]:
        return {
            "online": self.online, "sources_dir": str(self.sources_dir), "policy": self.policy.describe(),
            "catalog": self.catalog.describe() if self.catalog is not None else None,
            "candidates": self.candidates.describe(), "notes": list(self.notes),
        }

    def summary(self) -> str:
        mode = "online (fetches allowed from the trusted hosts)" if self.online else "offline: nothing is fetched (pass --online to fetch datasheets and official texts)"
        line = f"sources: {self.sources_dir} - {mode}; trusted hosts: {len(self.policy.trusted_hosts)}; user URLs: {len(self.policy.user_urls)}"
        if self.catalog is not None:
            line += f"; catalog: {self.catalog.path.name} ({len(self.catalog.rows)} rows, exported {self.catalog.retrieved_at})"
        return line

    def close(self) -> None:
        self.archive.close()


def open_session(
    *,
    workdir: Path | str,
    ir: CircuitIR,
    library: KicadLibrary | None,
    online: bool,
    trust_hosts: Iterable[str] = (),
    datasheet_urls: dict[str, str] | None = None,
    source_urls: dict[str, str] | None = None,
    sources_dir: Path | str | None = None,
    catalog: Path | str | None = None,
    catalog_date: str | None = None,
    catalog_authority: str | None = None,
    catalog_supplier: str | None = None,
    candidates: CandidateList | Path | str | None = None,
    gate: ApprovalGate | None = None,
    client: Any = None,
    approved_by: str = "cli --online",
) -> SourceSession:
    """Build the session from the run's flags (see the module docstring); :class:`SessionError` for a usage error.

    ``client`` is an optional ``httpx.Client`` (tests route it to a loopback
    fake); ``gate`` the approval gate the online approval is recorded on
    (the default gate when ``None``).
    """
    notes: list[str] = []
    try:
        cands = candidates if isinstance(candidates, CandidateList) else load_candidates(candidates)
    except ValueError as e:
        raise SessionError(f"regulatory candidates: {e}") from e
    user_urls: dict[str, str] = {}
    for key, url in (datasheet_urls or {}).items():
        user_urls[key] = url
    for key, url in (source_urls or {}).items():
        if key in user_urls and user_urls[key] != url:
            raise SessionError(f"--datasheet-url and --source-url both name {key!r} with different URLs")
        user_urls[key] = url
    policy = NetworkPolicy.from_cli(bool(online), user_urls=user_urls, gate=gate, approved_by=approved_by)
    for key, why in policy.invalid_user_urls.items():
        notes.append(f"user URL for {key!r} is unusable and will not be fetched: {why}")
    for host, reason in kicad_datasheet_hosts(ir, library).items():
        policy.trust_host(host, reason)
    for host, ids in cands.allowed_hosts().items():
        policy.trust_host(host, f"official domain listed for {ids} in {Path(cands.source_path or 'candidates.json').name} ({cands.sha256})")
    for host in trust_hosts:
        h = host.strip()
        if not h:
            continue
        try:
            policy.trust_host(h, "host trusted by the user (--trust-host)")
        except ValueError as e:
            raise SessionError(f"--trust-host: {e}") from e
        if host_key(h) not in policy.trusted_hosts:  # pragma: no cover - trust_host either adds or raises
            raise SessionError(f"--trust-host: {host!r} is not a host")
    root = Path(sources_dir) if sources_dir is not None else Path(workdir) / SOURCES_DIRNAME
    archive = DocumentArchive(root, policy, client=client)
    loaded: CatalogSource | None = None
    if catalog is not None:
        if not catalog_date:
            archive.close()
            raise SessionError("--catalog needs --catalog-date <ISO 8601>: the date you exported the file (the system never invents one)")
        name = Path(catalog).name
        try:
            loaded = CatalogSource.load(catalog, catalog_date, catalog_authority or f"user-supplied catalog export {name}", supplier=catalog_supplier)
        except CatalogError as e:
            archive.close()
            raise SessionError(f"--catalog: {e}") from e
        notes.extend(f"catalog: {n}" for n in loaded.notes)
    return SourceSession(policy=policy, archive=archive, candidates=cands, sources_dir=root, catalog=loaded, notes=notes)


__all__ = ["SessionError", "SourceSession", "kicad_datasheet_hosts", "open_session", "parse_key_urls"]
