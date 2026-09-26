"""The one way a validator reads a SPICE run as evidence about the *current* IR.

Invariant: a validator that judges the design by a simulated operating point
may only read a run that (1) is the latest tool-backed ``spice`` result with an
``artifact_hash``, (2) ran on the ``SPICE_NETLIST`` artifact that was generated
from the current IR version and is unchanged on disk, (3) wrote a
``SPICE_RESULT`` (``results.json``) that is unchanged on disk and names that
same netlist hash. Anything else is a reason (a ``str``), never a run, and the
validator reports NOT_VERIFIED with it. The message strings are the ones
:class:`~ai_eda.validation.domain.AnalogBiasValidator` has always given, so a
refusal reads the same whichever validator gives it.

An operating point is usable only when its analysis ``succeeded`` and was not
marked ``unverifiable`` (the environment prevented the run; the stage reports
such runs NOT_VERIFIED, and a validator must not read their vectors as facts
about the design). A node voltage is the last sample of the vector named
after the net (lower-cased, as ngspice writes it); the ground net is node 0,
identically 0 V. Nothing here computes anything: it only locates evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_eda.ir import ArtifactKind, ArtifactRef, CircuitIR, Evidence, Net, NetKind, ValidationResult
from ai_eda.tools.spice.stage import CHECK_ID as SPICE_CHECK_ID, NGSPICE_DEFAULT_TEMP_C, read_results

#: the reason given when no usable ``spice`` result is attached; the caller prefixes what it needs it for
NO_SPICE_RESULT = "no tool-backed spice result attached"
NO_OP = "no successful operating-point (op) analysis in the simulation results"


@dataclass
class FreshSpiceRun:
    """The latest SPICE run, proven to be about the current IR (see the module docstring)."""

    data: dict[str, Any]
    netlist: ArtifactRef
    results: ArtifactRef
    #: the latest ``spice`` summary result the run belongs to
    spice: ValidationResult

    @property
    def engine(self) -> str:
        return str(self.data.get("engine"))

    @property
    def engine_version(self) -> str:
        return str(self.data.get("engine_version"))

    @property
    def engine_stamp(self) -> dict[str, Any]:
        info = self.data.get("engine_info") or {}
        return {k: info.get(k) for k in ("build", "codemodels_loaded", "settings_hash", "dll_path")}

    @property
    def assumptions(self) -> list[str]:
        report = self.data.get("netlist_report") or {}
        return [str(a) for a in report.get("assumptions") or []]

    @property
    def conditions(self) -> dict[str, Any]:
        return dict(self.data.get("conditions") or {})

    @property
    def temperature_c(self) -> float:
        """The analysis temperature ngspice used (``.temp`` card or its 27 degC default)."""
        t = self.conditions.get("temperature_c")
        return NGSPICE_DEFAULT_TEMP_C if t is None else float(t)

    @property
    def analysis_ids(self) -> list[str]:
        return sorted(self.data["analyses"])

    def op(self) -> tuple[str, dict[str, Any]] | str:
        """``(analysis id, result dict)`` of the first usable ``op`` analysis, or the reason there is none.

        Usable: ``kind == "op"``, ``result.succeeded`` and no ``result.unverifiable``
        cause (an op the environment could not run is not evidence).
        """
        for aid, a in self.data["analyses"].items():
            res = a.get("result") or {}
            if a.get("kind") == "op" and res.get("succeeded") and not res.get("unverifiable"):
                return aid, res
        return NO_OP

    def node_voltage(self, ir: CircuitIR, net_name: str) -> float | str:
        """The op voltage of the IR net ``net_name`` (V), or why it is not available."""
        picked = self.op()
        if isinstance(picked, str):
            return picked
        _, res = picked
        net = next((n for n in ir.nets if n.name == net_name), None)
        if net is None:
            return f"net {net_name!r} is not in the IR"
        if net.kind == NetKind.GROUND:
            return 0.0
        samples = (res.get("vectors") or {}).get(net.name.lower())
        if not samples:
            return f"net {net.name} has no op vector (the net touches only excluded parts)"
        try:
            v = float(samples[-1])
        except (TypeError, ValueError):
            return f"net {net.name} op vector is not numeric ({samples[-1]!r})"
        if not math.isfinite(v):
            return f"net {net.name} op voltage is not finite ({v!r})"
        return v

    def evidence(self, analysis_id: str) -> list[Evidence]:
        """What a human opens to confirm a verdict on ``analysis_id``: its rawfile (when written), then ``results.json``."""
        out = [Evidence(description="spice results.json", path=self.results.path, content_hash=self.results.content_hash)]
        res = (self.data["analyses"].get(analysis_id) or {}).get("result") or {}
        if res.get("raw_output_path"):
            out.insert(0, Evidence(description=f"ngspice rawfile of analysis {analysis_id} ({res.get('command')})", path=res["raw_output_path"], content_hash=res.get("raw_output_hash")))
        return out


def fresh_spice_run(ir: CircuitIR, *, needs: str) -> FreshSpiceRun | str:
    """The latest SPICE run as evidence about the current IR, or why there is none.

    ``needs`` says what the caller wants it for (``"analog bias needs an
    operating point from SPICE"``): it prefixes the reason when no usable
    ``spice`` result is attached at all; every other reason names the broken
    link of the chain by itself.
    """
    spice = ir.validation.latest(SPICE_CHECK_ID)
    if spice is None or not spice.is_tool_backed or spice.artifact_hash is None:
        return f"{needs}; {NO_SPICE_RESULT}"
    netlist = ir.artifacts.get(ArtifactKind.SPICE_NETLIST)
    if netlist is None:
        return "no SPICE netlist artifact"
    if netlist.is_stale(ir.content_hash()):
        return "the SPICE netlist was generated from a different IR version; regenerate and re-simulate"
    if not netlist.matches_disk():
        return "the SPICE netlist on disk does not match its recorded hash"
    if spice.artifact_hash != netlist.content_hash:
        return "the latest spice result ran on a different netlist than the current artifact"
    results = ir.artifacts.get(ArtifactKind.SPICE_RESULT)
    if results is None or not Path(results.path).is_file():
        return "no SPICE results artifact (results.json)"
    if not results.matches_disk():
        return "results.json on disk does not match its recorded hash"
    try:
        data = read_results(results.path)
    except ValueError as e:
        return f"unreadable results.json: {e}"
    if data.get("netlist_hash") != netlist.content_hash:
        return "results.json was produced for a different netlist than the current artifact"
    return FreshSpiceRun(data=data, netlist=netlist, results=results, spice=spice)


def pin_net(ir: CircuitIR, ref: str, pin: str) -> Net | None:
    """The net that connects pin ``pin`` of component ``ref``, or ``None`` when it is in no net."""
    for net in ir.nets:
        for p in net.pins:
            if p.component_ref == ref and p.pin_number == pin:
                return net
    return None


__all__ = ["NO_OP", "NO_SPICE_RESULT", "FreshSpiceRun", "fresh_spice_run", "pin_net"]
