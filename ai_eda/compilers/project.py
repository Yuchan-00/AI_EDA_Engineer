"""IR -> ``<project.id>.kicad_pro`` compiler: the fab minimums as KiCad design rules.

Invariants enforced here
------------------------
* **Derived, deterministic, locator-free.** The file is sorted-key JSON with
  LF line endings, built only from the IR: ``meta.filename`` is
  ``<project.id>.kicad_pro`` (no directory), there is no clock and no KiCad
  runtime version in it, so two workdirs give byte-identical files and the
  artifact is registered through :meth:`Compiler._write` with
  ``generated_from_ir_hash``.
* **Rules come from grounded limits only.** ``board.design_settings.rules``
  is :func:`~ai_eda.compilers.pcb.design_rules` - the authoritative
  ``ir.pcb.manufacturing`` values under KiCad's rule keys; an unverified
  limit never becomes a rule, and without ``ir.pcb`` the rules are empty
  (KiCad then applies its own defaults, which is what
  :func:`~ai_eda.tools.manufacturing.capability.check_capability` reports).
  When the fab clearance exceeds KiCad's default netclass clearance
  (:data:`KICAD_DEFAULT_NETCLASS_CLEARANCE_MM`) the ``Default`` netclass is
  written with that clearance too, so the effective constraint is never
  below the written rule whichever of the two KiCad resolves first.
* **No severities.** The compiler never writes ``rule_severities`` or
  exclusions: a project file that silences a check could not come from it,
  which is why the capability check accepts DRC evidence only when the
  report's ``project_hash`` equals this artifact's hash.
* **Unmeasured until the KiCad box says otherwise.** Whether kicad-cli
  applies these rules from a sibling project file is recorded in
  :data:`~ai_eda.tools.kicad.cli.PROJECT_RULES_MEASURED_VERSIONS`; this
  module only writes the file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai_eda.compilers.base import CompileContext, Compiler
from ai_eda.compilers.pcb import _BAD_STEM_RE, design_rules
from ai_eda.errors import CompileError
from ai_eda.ir import ArtifactKind, ArtifactRef, CircuitIR

#: KiCad's default ``Default`` netclass clearance (mm); a stricter fab clearance is written into the netclass as well
KICAD_DEFAULT_NETCLASS_CLEARANCE_MM = 0.2
#: KiCad's own board minimums (mm), which kicad-cli DRC applies where the project file states none: the minimum track width (a
#: ``min_track_width`` rule replaces it) and the copper-to-edge clearance (no IR limit maps onto it: :data:`MANUFACTURING_RULE_KEYS
#: <ai_eda.compilers.pcb.MANUFACTURING_RULE_KEYS>`). Measured 2026-11-13: kicad-cli 10.0.6 ``pcb drc`` on the STM32F407 example's
#: routed 6-layer board beside this compiler's project file with empty rules reported its tracks against "board setup constraint"
#: copper-to-edge 0.5000 mm and minimum width 0.2000 mm (``copper_edge_clearance`` / ``track_width`` violations at 0.475 / 0.15 mm)
KICAD_DEFAULT_MIN_TRACK_WIDTH_MM = 0.2
KICAD_DEFAULT_COPPER_EDGE_CLEARANCE_MM = 0.5
#: KiCad's own via minimums (mm) beside a project file that states none: the via diameter and the hole of a through or blind via
#: (a ``min_via_diameter`` / ``min_through_hole_diameter`` rule replaces it; a micro via has minimums of its own) and the gap
#: between two drilled holes (nothing writes ``min_hole_to_hole``). Measured 2026-11-13: kicad-cli 10.0.6 ``pcb drc`` on a 4-layer
#: probe board beside a project file with empty rules reported a 0.45 / 0.2 mm through via and a 0.45 / 0.2 mm blind via against
#: minimum diameter 0.5000 mm (``via_diameter``) and minimum hole 0.3000 mm (``drill_out_of_range``), left a 0.45 / 0.2 mm micro
#: via alone, and reported two same-net 0.6 / 0.3 mm through vias 0.5 mm apart against minimum 0.2500 mm (``hole_to_hole``, actual
#: 0.2000 mm; 0.55 mm apart passed); with written track and clearance rules alone the three still applied
KICAD_DEFAULT_MIN_VIA_DIAMETER_MM = 0.5
KICAD_DEFAULT_MIN_THROUGH_HOLE_MM = 0.3
KICAD_DEFAULT_MIN_HOLE_TO_HOLE_MM = 0.25
#: the ``.kicad_pro`` ``meta.version`` KiCad 8-10 write
PROJECT_FILE_META_VERSION = 3


def checked_rules(ir: CircuitIR) -> dict[str, float]:
    """The minimums kicad-cli DRC checks the compiled board against beside this compiler's project file (mm): ``min_track_width``
    (the written rule, else :data:`KICAD_DEFAULT_MIN_TRACK_WIDTH_MM`), ``min_clearance`` (the ``Default`` netclass clearance -
    :data:`KICAD_DEFAULT_NETCLASS_CLEARANCE_MM`, or the stricter fab clearance :class:`ProjectFileCompiler` writes into it - never
    below the written rule), ``min_copper_edge_clearance`` (:data:`KICAD_DEFAULT_COPPER_EDGE_CLEARANCE_MM`: nothing is written),
    ``min_via_diameter`` / ``min_through_hole_diameter`` (the written rule, else :data:`KICAD_DEFAULT_MIN_VIA_DIAMETER_MM` /
    :data:`KICAD_DEFAULT_MIN_THROUGH_HOLE_MM`) and ``min_hole_to_hole`` (:data:`KICAD_DEFAULT_MIN_HOLE_TO_HOLE_MM`: nothing is
    written). Pure: the same IR gives the same numbers."""
    rules = design_rules(ir)
    return {
        "min_track_width": rules.get("min_track_width", KICAD_DEFAULT_MIN_TRACK_WIDTH_MM),
        "min_clearance": max(KICAD_DEFAULT_NETCLASS_CLEARANCE_MM, rules.get("min_clearance", 0.0)),
        "min_copper_edge_clearance": KICAD_DEFAULT_COPPER_EDGE_CLEARANCE_MM,
        "min_via_diameter": rules.get("min_via_diameter", KICAD_DEFAULT_MIN_VIA_DIAMETER_MM),
        "min_through_hole_diameter": rules.get("min_through_hole_diameter", KICAD_DEFAULT_MIN_THROUGH_HOLE_MM),
        "min_hole_to_hole": KICAD_DEFAULT_MIN_HOLE_TO_HOLE_MM,
    }


def project_filename(ir: CircuitIR) -> str:
    """``<project.id>.kicad_pro``; :class:`~ai_eda.errors.CompileError` when the id is not a usable file stem (the PCB compiler's rule)."""
    if not ir.project.id or _BAD_STEM_RE.search(ir.project.id):
        raise CompileError(f"project id {ir.project.id!r} is not usable as a KiCad file stem (it names the project file)")
    return f"{ir.project.id}.kicad_pro"


class ProjectFileCompiler(Compiler):
    id = "compiler.kicad_pro"
    version = "0.1"
    kind = ArtifactKind.KICAD_PROJECT

    def build(self, ir: CircuitIR) -> dict[str, Any]:
        """The project JSON (pure: same IR -> same dict)."""
        rules = design_rules(ir)
        data: dict[str, Any] = {
            "meta": {"filename": project_filename(ir), "version": PROJECT_FILE_META_VERSION},
            "board": {"design_settings": {"rules": rules}},
        }
        clearance = rules.get("min_clearance")
        if clearance is not None and clearance > KICAD_DEFAULT_NETCLASS_CLEARANCE_MM:
            data["net_settings"] = {"classes": [{"name": "Default", "clearance": clearance}]}
        return data

    def compile(self, ir: CircuitIR, ctx: CompileContext) -> ArtifactRef:
        text = json.dumps(self.build(ir), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        return self._write(ir, Path(ctx.workdir) / project_filename(ir), text)


__all__ = [
    "KICAD_DEFAULT_COPPER_EDGE_CLEARANCE_MM", "KICAD_DEFAULT_MIN_HOLE_TO_HOLE_MM", "KICAD_DEFAULT_MIN_THROUGH_HOLE_MM", "KICAD_DEFAULT_MIN_TRACK_WIDTH_MM",
    "KICAD_DEFAULT_MIN_VIA_DIAMETER_MM", "KICAD_DEFAULT_NETCLASS_CLEARANCE_MM", "PROJECT_FILE_META_VERSION",
    "ProjectFileCompiler", "checked_rules", "project_filename",
]
