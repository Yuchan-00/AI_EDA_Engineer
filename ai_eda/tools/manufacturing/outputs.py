"""Basic format / completeness checks on Gerber and drill files.

These are intentionally simple: header sanity, required layer presence,
manifest consistency, non-empty files, zone copper present. They do not
replace a CAM review.

Gerber files are classified by their ``%TF.FileFunction`` header, never by
file name or extension: kicad-cli writes Protel extensions (``.gtl``,
``.gbl``, ``.gts`` ...) by default and ``.gbr`` with ``--no-protel-ext``,
and a fab does not care which. The ``.gbrjob`` manifest, when present, must
list only files that are in the set (it spells functions differently from
the plot headers - ``SolderMask,Top`` vs ``Soldermask,Top`` - so it is used
for completeness, not for classification).

Completeness is judged against the board the IR describes: ``layer_count``
copper plots ``Copper,L1 .. Copper,L<n>`` (KiCad numbers them top to bottom,
so ``B.Cu`` is ``L<n>``), both solder masks and the profile. A board whose
IR has a zone on a copper layer must have at least one filled region
(``G36``) in that layer's plot - the sign that the pour was really exported
(``--check-zones``) rather than silently omitted.

``check_output_artifact`` stamps the result with the hash of the files it
*read* (``ArtifactRef.disk_hash()``), never with the hash recorded at export
time, so a result always says which bytes were inspected.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from ai_eda.ir import ArtifactKind, ArtifactRef, CircuitIR, Evidence, ValidationResult, ValidationStatus

TOOL_ID = "mfg.output_check"
TOOL_VERSION = "0.3"

#: Gerber X2 file-function prefixes every board needs besides its copper plots.
_REQUIRED_NON_COPPER = ("Soldermask,Top", "Soldermask,Bot", "Profile")
#: Gerber region start (filled polygon): what a filled zone plots as.
_REGION_START = "G36*"


def required_functions(layer_count: int) -> set[str]:
    """``Copper,L1 .. Copper,L<n>`` plus masks and profile (prefix match against ``%TF.FileFunction``)."""
    if layer_count < 2 or layer_count % 2:
        raise ValueError(f"a KiCad board has an even copper layer count >= 2, got {layer_count}")
    return {f"Copper,L{i}" for i in range(1, layer_count + 1)} | set(_REQUIRED_NON_COPPER)


def copper_function(layer: str, layer_count: int) -> str:
    """``F.Cu`` -> ``Copper,L1``, ``In<k>.Cu`` -> ``Copper,L<k+1>``, ``B.Cu`` -> ``Copper,L<n>``."""
    if layer == "F.Cu":
        return "Copper,L1"
    if layer == "B.Cu":
        return f"Copper,L{layer_count}"
    if layer.startswith("In") and layer.endswith(".Cu") and layer[2:-3].isdigit():
        return f"Copper,L{int(layer[2:-3]) + 1}"
    raise ValueError(f"{layer!r} is not a copper layer name")


def _gerber_function(path: Path) -> str | None:
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()[:20]:
        if line.startswith("%TF.FileFunction,"):
            return line[len("%TF.FileFunction,"):].rstrip("*%")
    return None


def _manifest_problems(job: Path, names: set[str]) -> list[str]:
    try:
        manifest = json.loads(job.read_text(encoding="utf-8"))
        listed = [str(entry["Path"]) for entry in manifest["FilesAttributes"]]
    except (ValueError, KeyError, TypeError) as exc:
        return [f"{job.name}: not a valid gerber job manifest ({exc!r})"]
    missing = sorted(set(listed) - names)
    if missing:
        return [f"{job.name}: manifest lists files not in the set: {missing}"]
    return []


def check_gerber_set(files: list[Path], layer_count: int = 2, zone_layers: Iterable[str] = ()) -> ValidationResult:
    """Format / completeness check of a gerber set for a ``layer_count``-layer board.

    ``zone_layers`` are the copper layers on which the IR has zones; each of
    their plots must contain a filled region, otherwise the pour was not
    exported.
    """
    if not files:
        return ValidationResult(check_id="mfg.gerber", status=ValidationStatus.FAIL, message="no gerber files", tool=TOOL_ID, tool_version=TOOL_VERSION)
    problems: list[str] = []
    functions: dict[str, Path] = {}  # "Copper,L1" -> file (first two function fields)
    names = {f.name for f in files}
    for f in files:
        if not f.is_file():
            problems.append(f"{f.name}: missing")
            continue
        if f.stat().st_size == 0:
            problems.append(f"{f.name}: empty file")
            continue
        if f.suffix.lower() == ".gbrjob":
            problems.extend(_manifest_problems(f, names))
            continue
        fn = _gerber_function(f)
        if fn is None:
            problems.append(f"{f.name}: missing %TF.FileFunction header")
        else:
            functions.setdefault(",".join(fn.split(",")[:2]), f)
    try:
        required = required_functions(layer_count)
    except ValueError as exc:
        problems.append(str(exc))
        required = set()
    missing = {r for r in required if not any(fn.startswith(r) for fn in functions)}
    if missing:
        problems.append(f"missing layers: {sorted(missing)}")
    zones_checked: list[str] = []
    for layer in sorted(set(zone_layers)):
        try:
            fn = copper_function(layer, layer_count)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        plot = functions.get(fn)
        if plot is None:
            continue  # already reported as a missing layer
        if _REGION_START not in plot.read_text(encoding="utf-8", errors="ignore"):
            problems.append(f"zone on {layer}: {plot.name} ({fn}) contains no filled region - the pour was not exported")
        else:
            zones_checked.append(layer)
    status = ValidationStatus.FAIL if problems else ValidationStatus.PASS
    return ValidationResult(
        check_id="mfg.gerber",
        status=status,
        message="; ".join(problems) or f"{len(files)} files, {layer_count} copper layers, functions: {sorted(functions)}",
        tool=TOOL_ID,
        tool_version=TOOL_VERSION,
        evidence=[Evidence(description="gerber file", path=str(f)) for f in files],
        details={"functions": sorted(functions), "layer_count": layer_count, "zone_layers_with_copper": zones_checked},
    )


def check_drill_files(files: list[Path]) -> ValidationResult:
    if not files:
        return ValidationResult(check_id="mfg.drill", status=ValidationStatus.FAIL, message="no drill files", tool=TOOL_ID, tool_version=TOOL_VERSION)
    problems: list[str] = []
    for f in files:
        if not f.is_file():
            problems.append(f"{f.name}: missing")
            continue
        head = f.read_text(encoding="utf-8", errors="ignore")[:200]
        if "M48" not in head:
            problems.append(f"{f.name}: not an Excellon file (no M48 header)")
    status = ValidationStatus.FAIL if problems else ValidationStatus.PASS
    return ValidationResult(
        check_id="mfg.drill",
        status=status,
        message="; ".join(problems) or f"{len(files)} drill file(s)",
        tool=TOOL_ID,
        tool_version=TOOL_VERSION,
        evidence=[Evidence(description="drill file", path=str(f)) for f in files],
    )


#: check id per artifact kind, so the reviewer and the repair loop can re-run the right one
OUTPUT_CHECKS: dict[ArtifactKind, str] = {ArtifactKind.GERBER: "mfg.gerber", ArtifactKind.DRILL: "mfg.drill"}


def artifact_files(art: ArtifactRef) -> list[Path]:
    return [Path(f) for f in art.files] if art.files else [Path(art.path)]


def check_output_artifact(art: ArtifactRef, ir: CircuitIR | None = None) -> ValidationResult:
    """Run the format check that belongs to ``art.kind`` on its files, stamped with the hash of what was read.

    With ``ir`` the gerber completeness rules follow the board the IR
    describes (copper layer count, zone layers); without it a 2-layer board
    without zones is assumed and ``details["board_from_ir"]`` says so.
    ``artifact_hash`` is ``art.disk_hash()`` - the bytes actually inspected -
    so a check that ran on files edited after export can never be mistaken
    for a check of the recorded artifact (the reviewer compares it with
    ``content_hash``).
    """
    files = artifact_files(art)
    if art.kind == ArtifactKind.GERBER:
        layer_count, zone_layers = 2, ()
        if ir is not None and ir.pcb is not None:
            layer_count = len(ir.pcb.layers)
            zone_layers = sorted({z.layer for z in ir.pcb.zones})
        res = check_gerber_set(files, layer_count, zone_layers)
        res.details["board_from_ir"] = ir is not None and ir.pcb is not None
    elif art.kind == ArtifactKind.DRILL:
        res = check_drill_files(files)
    else:
        raise ValueError(f"no output check for artifact kind {art.kind}")
    res.artifact_hash = art.disk_hash()
    res.details["recorded_artifact_hash"] = art.content_hash
    if res.artifact_hash is None:
        res.status = ValidationStatus.FAIL
        res.message = "artifact file(s) missing on disk; " + res.message
    elif res.artifact_hash != art.content_hash:
        res.details["disk_differs_from_recorded"] = True
    return res
