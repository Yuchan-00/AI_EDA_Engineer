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
#: the ``.kicad_pro`` ``meta.version`` KiCad 8-10 write
PROJECT_FILE_META_VERSION = 3


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


__all__ = ["KICAD_DEFAULT_NETCLASS_CLEARANCE_MM", "PROJECT_FILE_META_VERSION", "ProjectFileCompiler", "project_filename"]
