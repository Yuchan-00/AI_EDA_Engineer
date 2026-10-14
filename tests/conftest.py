from __future__ import annotations

from pathlib import Path

import pytest

import ai_eda.llm.claude_cli as _claude_cli
import ai_eda.report.stages as _stages
from ai_eda.ir import (
    CircuitDomain,
    CircuitIR,
    Component,
    LibraryRef,
    Net,
    NetKind,
    Pin,
    PinElectricalType,
    PinRef,
    ProjectMeta,
    Provenance,
    ProvenanceKind,
    SourceRef,
    Topology,
    authoritative,
    derived,
    user_requirement,
)

DS = SourceRef(title="Generic resistor datasheet", authority="Vendor", content_hash="sha256:abc")


def rawfile_command_ok(command: str, version: str) -> bool:
    """KiCad's ngspice-46 stamps ``Command: ngspice-46, Build ...`` into every rawfile; Debian/Ubuntu ``libngspice0``
    (ngspice-42) writes no ``Command:`` line at all, so the parsed command is empty (measured 2026-09-23). The line is
    evidence of the writer, not data."""
    if version == "ngspice-42":
        return command == ""
    return command.startswith(version + ", Build ")
AUTH = Provenance(kind=ProvenanceKind.AUTHORITATIVE, source=DS)


@pytest.fixture(autouse=True)
def stage_reports_find_no_browser(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """The stage-report writer discovers no browser in a test unless the test is marked ``browser``.

    A headless print costs about 1.5 s per report and every ``ai-eda run``
    that reaches RELEASE writes seven; the runs across the suite are not
    about PDFs. Only the writer's discovery (``ai_eda.report.stages.find_browser``)
    is disabled: ``ai_eda.report.pdf.find_browser`` itself, ``doctor`` and an
    explicit ``browser=`` / ``--browser`` are untouched, and a test marked
    ``@pytest.mark.browser`` gets the real discovery.
    """
    if request.node.get_closest_marker("browser") is None:
        monkeypatch.setattr(_stages, "find_browser", lambda: None)


@pytest.fixture(autouse=True)
def claude_cli_discovery_finds_only_the_fake(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """``find_claude_cli`` finds only the fake ``claude`` of ``tests/fake_claude_cli.py`` unless the test is marked ``real_claude_cli``.

    ``doctor`` and ``describe_providers`` probe whatever discovery returns
    (``claude --version``, ``claude auth status``), so without this a test
    that runs them without installing the fake would spawn the machine's real
    CLI and read its credential store. Discovery itself runs unchanged; a
    result that is not a fake (no :data:`~tests.fake_claude_cli.FAKE_MARKER`
    beside it) reads as "not found". An explicit ``ClaudeCodeClient(cli=...)``
    / ``--llm-claude-cli`` is untouched, and only
    ``tests/test_claude_cli_live.py`` is marked ``real_claude_cli``.
    """
    if request.node.get_closest_marker("real_claude_cli") is not None:
        return
    from tests.fake_claude_cli import is_fake_claude_cli

    real = _claude_cli.find_claude_cli

    def fake_only() -> str | None:
        found = real()
        return found if found is not None and is_fake_claude_cli(found) else None

    monkeypatch.setattr(_claude_cli, "find_claude_cli", fake_only)


def _pin(n: str) -> Pin:
    return Pin(number=n, name=f"~{n}", electrical_type=PinElectricalType.PASSIVE, provenance=AUTH)


def make_component(ref: str, value: str, verified_lib: bool = True) -> Component:
    return Component(
        ref=ref,
        value=value,
        description="resistor",
        mpn=authoritative(f"MPN-{value}", DS),
        package=authoritative("0603", DS),
        pins=[_pin("1"), _pin("2")],
        symbol=LibraryRef(library="Device", name="R", verified=verified_lib),
        footprint=LibraryRef(library="Resistor_SMD", name="R_0603_1608Metric", verified=verified_lib),
        provenance=Provenance(kind=ProvenanceKind.DERIVED, tool="test", note="fixture"),
        serves_requirements=["req.v_out"],
    )


@pytest.fixture
def divider_ir(tmp_path: Path) -> CircuitIR:
    """A two-resistor divider with authoritative parts, derived parameters, no artifacts."""
    ir = CircuitIR(project=ProjectMeta(id="t", name="divider", workdir=str(tmp_path)))
    ir.topology = Topology(name="resistive divider", domains=[CircuitDomain.ANALOG], provenance=AUTH)
    ir.components = [make_component("R1", "10k"), make_component("R2", "10k")]
    net_p = Provenance(kind=ProvenanceKind.DERIVED, tool="test")
    ir.nets = [
        Net(name="VIN", kind=NetKind.POWER, pins=[PinRef(component_ref="R1", pin_number="1")], provenance=net_p),
        Net(name="VOUT", pins=[PinRef(component_ref="R1", pin_number="2"), PinRef(component_ref="R2", pin_number="1")], provenance=net_p),
        Net(name="GND", kind=NetKind.GROUND, pins=[PinRef(component_ref="R2", pin_number="2")], provenance=net_p),
    ]
    ir.parameters["v_in"] = user_requirement(12.0, "V")
    ir.parameters["r1"] = authoritative(10_000.0, DS, "ohm")
    ir.parameters["r2"] = authoritative(10_000.0, DS, "ohm")
    ir.parameters["v_out"] = derived(6.0, tool="calc.divider.v_out", inputs={"v_in": "v_in", "r1": "r1", "r2": "r2"}, unit="V")
    return ir
