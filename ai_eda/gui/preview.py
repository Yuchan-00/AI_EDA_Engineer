"""What the GUI shows of a project, built from the files a run wrote: pure functions returning text, bytes or plain data.

Invariant: a preview is a *view* under the same rule as ``report.html`` and
the stage reports. Every builder reads what is on disk in the project's
workdir (and the parsed IR it is given) and returns ``str`` / ``bytes`` /
lists and dicts of ``str`` / ``int`` / ``float`` / ``bool``; it computes no
status, registers no artifact, hashes nothing new (the only hashes are the
ones :func:`~ai_eda.report.data.artifact_disk_state` and
:func:`~ai_eda.tools.spice.evidence.fresh_spice_run` already compare),
saves nothing and never edits ``ir.json``. The labels it adds are facts
about hashes and files (an artifact on disk or changed, a ``results.json``
that is or is not the one the current IR's ``spice`` result names), never a
verdict.

* :func:`schematic_svg` draws the registered ``.kicad_sch`` with
  :mod:`ai_eda.gui.schematic_render` (a preview of the file the compiler
  wrote, drawn from its own ``lib_symbols``; not a KiCad render and no ERC is
  implied).
* :func:`board_svg` is :func:`ai_eda.report.figures.board_figure` with
  copper: pad geometry from the KiCad library on disk, never guessed; its
  layer groups carry the classes ``layer-F_Cu`` / ``layer-B_Cu`` / ``pads``
  / ``vias`` / ``outline`` / ``silk`` (``layer-F_SilkS`` / ``layer-B_SilkS``
  inside) / ``labels`` (and ``planes``: the hatched inner-plane zones of a
  4-layer board) so the page can toggle them; each track of a design with
  ``ir.si`` carries its net class (``data-class`` / ``data-nc``), which the
  page's 넷 클래스 색 toggle colours by, and :func:`board_classes` lists the
  classes for its legend (read from ``ir.si``, no status).
* :func:`model3d_scene` is the scene of the built-in 3D preview
  (:func:`ai_eda.tools.model3d.scene.build_scene` of the current IR, the
  KiCad library and the 3D model library this process finds - the same
  function the ``MODEL_3D`` compiler writes ``<project>.preview.glb`` from,
  so an unchanged IR on the same libraries gives the registered file's
  bytes); :func:`model3d_glb` is its glTF binary for the page's WebGL
  viewer, :func:`model3d_svg` its painter's-algorithm SVG (the fallback
  without WebGL; views of :data:`~ai_eda.tools.model3d.iso.VIEWS`), and
  :func:`model3d_summary` its caption, the body rule
  (:data:`~ai_eda.tools.model3d.scene.BODY_CAPTION`) and one row per part box.
  A picture of boxes, never a status. :func:`kicad_3d_files` lists KiCad's
  own STEP / GLB / render exports (``ArtifactKind.KICAD_STEP`` /
  ``KICAD_GLB`` / ``KICAD_RENDER``, real part shapes, written only where
  kicad-cli runs) with their artifact facts; :func:`kicad_render_file` is the
  one gate to a render PNG (a registered member, confined to the workdir).
* :func:`waveform_plan` / :func:`waveform_svgs` draw every recorded
  ``tran`` / ``dc`` / ``ac`` analysis of ``spice/results.json`` that
  succeeded (every vector as ngspice wrote it, voltages and currents on
  separate charts, at most four series each; an AC analysis by magnitude)
  through :func:`~ai_eda.report.figures.waveform_figure`; a missing file is
  no waveform. :func:`simulation_summary` says whether that file is the one
  the current IR's latest ``spice`` result names (read only through
  ``fresh_spice_run``) - a stale file is drawn and labelled, never hidden.
* :func:`bom_table` / :func:`cpl_table` read the CSVs; a free-text BOM cell
  (``Value`` / ``Description``) is decoded with
  :func:`~ai_eda.tools.manufacturing.csv_cells.bom_cell_text`, every other
  cell is shown exactly as written (``NOT_VERIFIED`` included).
* :func:`artifact_rows` lists ``ir.artifacts`` with their disk state and
  whether they were generated from the current design hash;
  :func:`artifact_state` is that row, shown next to the preview of the same
  file (the schematic, the BOM, the CPL, the board download), so a file
  compiled from an earlier design is drawn *and labelled*, never passed off
  as the current design; :func:`missing_sentence` says that a registered
  file is missing (the stage ran) rather than that its stage has not run.
  :func:`report_files` lists the stage reports and ``report.html`` that
  exist.
* :func:`confined_file` is the one gate between a URL and a file: a path
  inside the workdir, no ``..``, no symbolic link on the way, a regular file
  (never a directory listing). :func:`project_zip` packs ``ir.json``, the
  registered artifacts inside the workdir, ``pipeline.json``,
  ``report.html``, ``reports/*`` and ``spice/**`` (never ``sources/`` or the
  ``gui/`` run logs) with fixed timestamps, so the same tree gives the same
  bytes.

Every text these builders put into markup is escaped by the builder it
comes from; a builder that cannot draw raises (the server answers one line),
a preview whose input does not exist yet raises :class:`PreviewMissing` with
the Korean sentence the page shows.
"""

from __future__ import annotations

import csv
import io
import os
import re
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_eda.gui.projects import IR_FILE, safe_part
from ai_eda.gui.schematic_render import render_kicad_sch_file
from ai_eda.ir import ArtifactKind, CircuitIR
from ai_eda.report.data import FRESH, ON_DISK, STALE, artifact_disk_state
from ai_eda.report.figures import MAX_SERIES, board_figure, net_class_styles, vector_kind, waveform_figure
from ai_eda.report.pipeline_log import PIPELINE_FILE
from ai_eda.report.stages import REPORT_SUFFIXES, REPORT_TITLES, REPORTS_DIR, STAGE_REPORTS
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.manufacturing.csv_cells import bom_cell_text
from ai_eda.tools.model3d import BODY_CAPTION, VIEWS, Scene, build_scene, iso_svg, scene_caption, write_glb
from ai_eda.tools.spice.rawfile import COMPLEX_SUFFIXES
from ai_eda.tools.spice.stage import RESULTS_DIR, RESULTS_FILE, read_results

#: the files ``ai-eda report`` and the BOM / CPL compilers write into the workdir
REPORT_HTML_FILE = "report.html"
BOM_FILE = "bom.csv"
CPL_FILE = "cpl.csv"
#: the BOM columns written through ``free_text_cell`` (decoded with ``bom_cell_text``); every other cell is shown as written
BOM_FREE_TEXT_COLUMNS: tuple[str, ...] = ("Value", "Description")
#: the cell text the BOM writes for an identity / sourcing value nothing verified
NOT_VERIFIED_CELL = "NOT_VERIFIED"
#: the analysis kinds drawn as waveforms (an ``op`` is one point: its values are listed, not drawn)
WAVEFORM_KINDS: tuple[str, ...] = ("tran", "dc", "ac")
#: a waveform id (and the analysis id it starts with): what a URL segment ``<id>.svg`` can carry
WAVEFORM_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")
#: folders under the workdir the project zip never contains (the document archive, the GUI's run logs)
ZIP_EXCLUDED_DIRS: tuple[str, ...] = ("sources", "gui")
#: the timestamp of every zip entry (the zip format's epoch), so the same tree gives the same bytes
ZIP_DATE_TIME = (1980, 1, 1, 0, 0, 0)
#: freshness labels of a results.json (facts about hashes)
RESULTS_CURRENT = "현재 IR의 spice 결과가 가리키는 results.json입니다 (해시 일치)"
RESULTS_NOT_CURRENT = "현재 IR의 결과로 확인되지 않은 results.json입니다"
#: what the page says where a preview has no input: the missing file and the stage (``Stage`` name) that writes it - never
#: that the stage "has not run" (the stage table, copied from pipeline.json, says whether it ran and why nothing was written)
NO_SCHEMATIC = "회로도 없음: 등록된 .kicad_sch가 없습니다 (SCHEMATIC 단계가 IR의 부품으로 그려 등록합니다; 개요의 schematic 줄을 보십시오)"
NO_BOARD = "기판 없음: IR에 기판 외곽과 부품 배치가 없습니다 (PLACEMENT 단계가 배치합니다; 개요의 placement 줄을 보십시오)"
NO_RESULTS = "시뮬레이션 결과 없음: spice/results.json이 없습니다 (SPICE 단계가 씁니다; 개요의 spice 줄을 보십시오)"
NO_WAVEFORM = "파형 없음: 그 id의 파형이 spice/results.json에 없습니다"
NO_BOM = "BOM 없음: bom.csv가 없습니다 (MANUFACTURING_OUTPUTS 단계가 씁니다; 개요의 manufacturing_outputs 줄을 보십시오)"
NO_CPL = "CPL 없음: cpl.csv가 없습니다 (MANUFACTURING_OUTPUTS 단계가 씁니다; 개요의 manufacturing_outputs 줄을 보십시오)"
NO_REPORT_FILE = "보고서 파일 없음: 그 이름의 단계 보고서가 아직 쓰이지 않았습니다"
NO_MODEL3D = "3D 미리보기 없음: IR에 기판 외곽과 부품 배치가 없습니다 (PLACEMENT 단계가 배치합니다; 개요의 placement 줄을 보십시오)"
NO_MODEL3D_VIEW = "3D 미리보기 없음: 그런 보기가 없습니다 (iso, iso_bottom, top, bottom)"
NO_KICAD_RENDER = "KiCad 렌더 없음: 그 이름의 PNG가 등록된 KiCad 렌더(kicad_render)에 없습니다"
#: what KiCad's own 3D exports are called on the page (they carry the footprints' real 3D part models)
KICAD_3D_LABEL = "KiCad 3D 모델(실제 부품 모양)"
#: KiCad's own 3D exports, in the order the page lists them, with what each file is
KICAD_3D_KINDS: tuple[tuple[ArtifactKind, str], ...] = (
    (ArtifactKind.KICAD_STEP, "STEP (kicad-cli pcb export step)"),
    (ArtifactKind.KICAD_GLB, "GLB (kicad-cli pcb export glb)"),
    (ArtifactKind.KICAD_RENDER, "렌더 PNG (kicad-cli pcb render, 위·아래)"),
)
#: the width of the fallback SVG (the page scales it to the column)
MODEL3D_SVG_WIDTH = 960
#: what each previewed artifact is called in the sentence for a registered file that is not on disk
ARTIFACT_WORDS: dict[ArtifactKind, str] = {
    ArtifactKind.SCHEMATIC: "회로도", ArtifactKind.PCB: "기판", ArtifactKind.BOM: "BOM", ArtifactKind.CPL: "CPL", ArtifactKind.MODEL_3D: "3D 미리보기",
}

class PreviewMissing(LookupError):
    """The input of a preview does not exist yet (the server's 404); the message is the Korean sentence the page shows."""


# --------------------------------------------------------------------------- files inside the workdir


def confined_file(workdir: Path, parts: Sequence[str]) -> Path | None:
    """The regular file ``<workdir>/<parts...>`` (resolved), or ``None``.

    ``None`` for an empty or unsafe component (:func:`safe_part`), a symbolic
    link anywhere below the workdir, a directory (no listing), a missing
    file, or a resolved path outside the workdir. The workdir itself is
    resolved once (a link above it, such as ``/tmp`` on macOS, is the
    machine's layout, not the project's).
    """
    if not parts or not all(safe_part(p) for p in parts):
        return None
    try:
        root = Path(workdir).resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    current = root
    try:
        for part in parts:
            current = current / part
            if current.is_symlink():
                return None
        if not current.is_file():
            return None
        resolved = current.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    return resolved if resolved.is_relative_to(root) else None


def workdir_parts(path: str | Path, workdir: Path) -> tuple[str, ...] | None:
    """``path`` (absolute, or relative to the workdir) as components below the workdir, compared lexically; ``None`` when it lies outside."""
    p = Path(path)
    if not p.is_absolute():
        p = Path(workdir) / p
    base = Path(os.path.normpath(Path(workdir).absolute()))
    target = Path(os.path.normpath(p))
    if target == base or not target.is_relative_to(base):
        return None
    return target.relative_to(base).parts


def artifact_file(ir: CircuitIR, workdir: Path, kind: ArtifactKind) -> Path | None:
    """The file of the registered artifact ``kind`` when it is a regular file inside the workdir (:func:`confined_file`), else ``None``."""
    art = ir.artifacts.get(kind)
    if art is None:
        return None
    parts = workdir_parts(art.path, workdir)
    return confined_file(workdir, parts) if parts is not None else None


def missing_sentence(ir: CircuitIR, workdir: Path, kind: ArtifactKind, default: str) -> str:
    """The Korean sentence for a preview of ``kind`` whose file is not a regular file inside the workdir.

    ``default`` (what writes it) when ``ir.artifacts`` registers no ``kind``;
    otherwise that the registered file cannot be found - the stage did run
    and register it, so "not written yet" would be false.
    """
    art = ir.artifacts.get(kind)
    if art is None:
        return default
    parts = workdir_parts(art.path, workdir)
    shown = "/".join(parts) if parts is not None else Path(art.path).name
    return (
        f"{ARTIFACT_WORDS.get(kind, str(kind))} 파일 없음: IR에 등록된 파일({shown})이 작업 폴더 안의 일반 파일로 있지 않습니다 "
        "(단계는 이 파일을 쓰고 ir.artifacts에 등록했습니다; 파일 탭의 디스크 상태를 보십시오)"
    )


def artifact_download(ir: CircuitIR, workdir: Path, kind: ArtifactKind) -> str | None:
    """The workdir-relative path (``/``-joined) of the registered artifact ``kind`` when it can be downloaded, else ``None``."""
    art = ir.artifacts.get(kind)
    if art is None:
        return None
    parts = workdir_parts(art.path, workdir)
    if parts is None or confined_file(workdir, parts) is None:
        return None
    return "/".join(parts)


# --------------------------------------------------------------------------- schematic and board


def schematic_svg(sch_path: Path) -> str:
    """The SVG preview of a compiled ``.kicad_sch`` (:func:`ai_eda.gui.schematic_render.render_kicad_sch_file`)."""
    return render_kicad_sch_file(sch_path)


def project_schematic_svg(ir: CircuitIR, workdir: Path) -> str:
    """:func:`schematic_svg` of the project's registered schematic; :class:`PreviewMissing` (:func:`missing_sentence`) when there is none on disk."""
    path = artifact_file(ir, workdir, ArtifactKind.SCHEMATIC)
    if path is None:
        raise PreviewMissing(missing_sentence(ir, workdir, ArtifactKind.SCHEMATIC, NO_SCHEMATIC))
    return schematic_svg(path)


def board_available(ir: CircuitIR) -> bool:
    """Whether the IR holds what a board figure needs: an outline and at least one placement (drawing can still refuse, e.g. a footprint not in the library)."""
    pcb = ir.pcb
    return pcb is not None and pcb.outline is not None and bool(pcb.placements)


def board_svg(ir: CircuitIR, library: KicadLibrary) -> str:
    """The placed board with copper (:func:`ai_eda.report.figures.board_figure`); :class:`PreviewMissing` without an outline or placements."""
    if not board_available(ir):
        raise PreviewMissing(NO_BOARD)
    return board_figure(ir, library, copper=True, fig_id="board").svg


def board_classes(ir: CircuitIR) -> list[dict[str, Any]]:
    """The net classes of ``ir.si`` for the board tab's legend: name, colour token, default?, how many nets each routes, the promoted nets (``[]`` without ``ir.si``)."""
    return [
        {"name": st.name, "token": st.token, "default": st.default, "nets": len(st.nets), "promoted": list(st.promoted)}
        for st in net_class_styles(ir)
    ]


# --------------------------------------------------------------------------- the built-in 3D preview and KiCad's 3D exports


def model3d_scene(ir: CircuitIR, library: KicadLibrary) -> Scene:
    """The 3D preview scene of the current IR (:func:`~ai_eda.tools.model3d.scene.build_scene`, 3D library discovered as the compiler does).

    :class:`PreviewMissing` (:data:`NO_MODEL3D`) without an outline or
    placements - the board figure's input; a scene that cannot be built
    without guessing (a footprint not on disk ...) raises its own error.
    """
    if not board_available(ir):
        raise PreviewMissing(NO_MODEL3D)
    return build_scene(ir, library)


def model3d_glb(scene: Scene) -> bytes:
    """The scene as glTF 2.0 binary (:func:`~ai_eda.tools.model3d.glb.write_glb`: deterministic bytes, no path, no time)."""
    return write_glb(scene)


def model3d_svg(scene: Scene, view: str) -> str:
    """The scene seen from ``view`` (a key of :data:`~ai_eda.tools.model3d.iso.VIEWS`) as SVG without its in-figure caption (the page shows it)."""
    if view not in VIEWS:
        raise PreviewMissing(NO_MODEL3D_VIEW)
    return iso_svg(scene, view, width=MODEL3D_SVG_WIDTH, caption=False)


def model3d_summary(scene: Scene) -> dict[str, Any]:
    """What the scene shows, as the page lists it: the caption, the body rule, the board thickness (and whether it is assumed), the notes and one row per part box."""
    bodies = [
        {
            "ref": b.ref,
            "lib_id": b.lib_id,
            "side": b.side,
            "outline_source": b.outline_source,
            "models": list(b.models),
            "height_mm": b.height_mm,
            "reason": b.reason,
        }
        for b in scene.bodies
    ]
    return {
        "project": scene.project_id,
        "caption": scene_caption(scene),
        "body_caption": BODY_CAPTION,
        "thickness_mm": scene.thickness_mm,
        "thickness_assumed": not scene.thickness_grounded,
        "model_dir_found": scene.model_dir_found,
        "with_step": len(scene.bodies_with_step),
        "without_step": len(scene.bodies_without_step),
        "notes": list(scene.notes),
        "solids": scene.counts(),
        "bodies": bodies,
        "views": list(VIEWS),
    }


def kicad_3d_files(ir: CircuitIR, workdir: Path, rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """KiCad's own 3D exports registered in ``ir.artifacts`` (:data:`KICAD_3D_KINDS` order): per kind its label, its artifact facts
    (:func:`artifact_state`) and its files - each with the download path when it is a regular file inside the workdir, and whether it
    is a render PNG the page may show (:func:`kicad_render_file`). ``[]`` when kicad-cli never exported them (the stage message says why)."""
    out: list[dict[str, Any]] = []
    for kind, what in KICAD_3D_KINDS:
        art = ir.artifacts.get(kind)
        if art is None:
            continue
        files: list[dict[str, Any]] = []
        for path in art.files or [art.path]:
            parts = workdir_parts(path, workdir)
            download = "/".join(parts) if parts is not None and confined_file(workdir, parts) is not None else None
            name = Path(path).name
            image = kind is ArtifactKind.KICAD_RENDER and download is not None and name.lower().endswith(".png")
            files.append({"name": name, "download": download, "image": image})
        out.append({"kind": str(kind), "what": what, "label": KICAD_3D_LABEL, "files": files, "state": artifact_state(rows, kind)})
    return out


def kicad_render_file(ir: CircuitIR, workdir: Path, name: str) -> Path | None:
    """The render PNG ``name`` (a basename) when it is a member of the registered ``KICAD_RENDER`` artifact and a regular file inside the workdir."""
    art = ir.artifacts.get(ArtifactKind.KICAD_RENDER)
    if art is None or not safe_part(name) or not name.lower().endswith(".png"):
        return None
    for path in art.files or [art.path]:
        if Path(path).name != name:
            continue
        parts = workdir_parts(path, workdir)
        found = confined_file(workdir, parts) if parts is not None else None
        if found is not None:
            return found
    return None


# --------------------------------------------------------------------------- simulation


@dataclass(frozen=True)
class WaveformPlan:
    """One waveform chart: ``id`` (the URL's ``<id>.svg``), the analysis it draws, its vectors (one kind, at most :data:`MAX_SERIES`) and its title."""

    id: str
    analysis_id: str
    kind: str
    vectors: tuple[str, ...]
    title: str


def read_results_file(workdir: Path) -> dict[str, Any] | None:
    """``<workdir>/spice/results.json`` as the SPICE stage wrote it; ``None`` when there is no such file (``ValueError`` for another layout)."""
    path = confined_file(workdir, (RESULTS_DIR, RESULTS_FILE))
    if path is None:
        return None
    return read_results(path)


def _complex_part(name: str) -> bool:
    """Whether ``name`` is the phase / real / imaginary part of an AC vector (the magnitude is the vector's own name)."""
    head, sep, tail = name.rpartition(".")
    return bool(sep and head and tail in COMPLEX_SUFFIXES)


def waveform_plan(results: dict[str, Any]) -> list[WaveformPlan]:
    """The charts of a ``results.json``: per ``tran`` / ``dc`` / ``ac`` analysis that succeeded (in file order), its voltages, then its currents, four per chart.

    Ids follow :func:`~ai_eda.report.figures.waveform_figures`: ``<aid>``,
    ``<aid>_2`` ... for the voltages and ``<aid>_i``, ``<aid>_i2`` ... for the
    branch currents. An analysis the environment could not run
    (``unverifiable``), one without a scale vector, or whose id a URL
    cannot carry (:data:`WAVEFORM_ID_RE`) is not planned; an id already
    taken by an earlier chart is skipped, never overwritten.
    """
    analyses = results.get("analyses") if isinstance(results, dict) else None
    if not isinstance(analyses, dict):
        return []
    plans: list[WaveformPlan] = []
    taken: set[str] = set()
    for aid, entry in analyses.items():
        if not isinstance(aid, str) or not WAVEFORM_ID_RE.fullmatch(aid) or not isinstance(entry, dict):
            continue
        kind = entry.get("kind")
        result = entry.get("result")
        if kind not in WAVEFORM_KINDS or not isinstance(result, dict) or not result.get("succeeded") or result.get("unverifiable"):
            continue
        vectors = result.get("vectors")
        scale = result.get("scale")
        if not isinstance(vectors, dict) or not isinstance(scale, str) or scale not in vectors:
            continue
        groups: dict[str, list[str]] = {"voltage": [], "current": []}
        for name in vectors:
            if name != scale and not _complex_part(name):
                groups[vector_kind(result, name)].append(name)
        for group, id_suffix, title_suffix in (("voltage", "", ""), ("current", "_i", " (전류)")):
            names = groups[group]
            chunks = [names[i : i + MAX_SERIES] for i in range(0, len(names), MAX_SERIES)]
            for k, chunk in enumerate(chunks, start=1):
                fig_id = f"{aid}{id_suffix}" + (f"{'' if id_suffix else '_'}{k}" if k > 1 else "")
                if fig_id in taken or not WAVEFORM_ID_RE.fullmatch(fig_id):
                    continue
                taken.add(fig_id)
                title = f"시뮬레이션 파형 — 해석 {aid}{title_suffix}" + (f" ({k}/{len(chunks)})" if len(chunks) > 1 else "")
                plans.append(WaveformPlan(fig_id, aid, str(kind), tuple(chunk), title))
    return plans


def _draw(results: dict[str, Any], plan: WaveformPlan) -> str:
    return waveform_figure(results, plan.analysis_id, list(plan.vectors), title=plan.title, fig_id=plan.id).svg


def waveform_svgs(workdir: Path) -> list[tuple[str, str]]:
    """``(id, svg)`` of every chart of :func:`waveform_plan` on ``<workdir>/spice/results.json``; ``[]`` when there is no such file."""
    results = read_results_file(workdir)
    if results is None:
        return []
    return [(plan.id, _draw(results, plan)) for plan in waveform_plan(results)]


def waveform_svg(workdir: Path, fig_id: str) -> str:
    """The one chart ``fig_id`` of :func:`waveform_svgs`; :class:`PreviewMissing` without results.json or without that chart."""
    results = read_results_file(workdir)
    if results is None:
        raise PreviewMissing(NO_RESULTS)
    for plan in waveform_plan(results):
        if plan.id == fig_id:
            return _draw(results, plan)
    raise PreviewMissing(NO_WAVEFORM)


def _op_values(result: dict[str, Any]) -> dict[str, float | str]:
    """The last sample of every vector of an operating point, as ngspice recorded it (a non-number is shown as its text)."""
    out: dict[str, float | str] = {}
    for name, samples in (result.get("vectors") or {}).items():
        if isinstance(samples, list) and samples:
            value = samples[-1]
            out[str(name)] = value if isinstance(value, (int, float)) and not isinstance(value, bool) else str(value)
    return out


def simulation_summary(ir: CircuitIR, workdir: Path) -> dict[str, Any]:
    """What ``spice/results.json`` records (engine, conditions, every analysis with its command and outcome, the op values, the waveform charts) and whether it is the current IR's run.

    ``current`` is ``True`` only when :func:`~ai_eda.tools.spice.evidence.fresh_spice_run`
    accepts the latest ``spice`` result and its ``results.json`` is this
    file; otherwise ``freshness`` carries the reason. Nothing is judged.
    """
    from ai_eda.tools.spice.evidence import fresh_spice_run

    try:
        results = read_results_file(workdir)
    except (ValueError, OSError) as e:
        return {"file": f"{RESULTS_DIR}/{RESULTS_FILE}", "error": " ".join(str(e).split()), "analyses": [], "waveforms": [], "current": False, "freshness": ""}
    if results is None:
        return {"file": None, "missing": NO_RESULTS, "analyses": [], "waveforms": [], "current": False, "freshness": ""}
    run = fresh_spice_run(ir, needs="GUI preview")
    if isinstance(run, str):
        current, freshness = False, f"{RESULTS_NOT_CURRENT}: {run}"
    else:
        here = confined_file(workdir, (RESULTS_DIR, RESULTS_FILE))
        current = here is not None and Path(run.results.path).resolve() == here
        freshness = RESULTS_CURRENT if current else f"{RESULTS_NOT_CURRENT}: the current spice result names {Path(run.results.path).name} elsewhere"
    analyses: list[dict[str, Any]] = []
    for aid, entry in (results.get("analyses") or {}).items():
        entry = entry if isinstance(entry, dict) else {}
        result = entry.get("result") if isinstance(entry.get("result"), dict) else {}
        row: dict[str, Any] = {
            "id": str(aid),
            "kind": str(entry.get("kind") or ""),
            "command": str(entry.get("command") or result.get("command") or ""),
            "succeeded": bool(result.get("succeeded")),
            "unverifiable": str(result["unverifiable"]) if result.get("unverifiable") else None,
            "n_points": result.get("n_points") if isinstance(result.get("n_points"), int) else None,
        }
        if entry.get("kind") == "op" and result.get("succeeded"):
            row["op"] = _op_values(result)
        analyses.append(row)
    return {
        "file": f"{RESULTS_DIR}/{RESULTS_FILE}",
        "engine": str(results.get("engine") or ""),
        "engine_version": str(results.get("engine_version") or ""),
        "conditions": results.get("conditions") if isinstance(results.get("conditions"), dict) else {},
        "analyses": analyses,
        "waveforms": [{"id": p.id, "analysis_id": p.analysis_id, "kind": p.kind, "vectors": list(p.vectors), "title": p.title} for p in waveform_plan(results)],
        "current": current,
        "freshness": freshness,
    }


# --------------------------------------------------------------------------- BOM / CPL


def read_csv_table(path: Path, free_text_columns: Sequence[str] = ()) -> tuple[list[str], list[dict[str, str]]]:
    """``(columns, rows)`` of a CSV the compilers wrote (UTF-8); ``free_text_columns`` are decoded with :func:`bom_cell_text`. ``ValueError`` for a row whose width is not the header's."""
    text = Path(path).read_text(encoding="utf-8")
    reader = csv.reader(io.StringIO(text, newline=""))
    rows = list(reader)
    if not rows:
        return [], []
    header = rows[0]
    out: list[dict[str, str]] = []
    for number, cells in enumerate(rows[1:], start=2):
        if not cells:
            continue
        if len(cells) != len(header):
            raise ValueError(f"{Path(path).name} line {number} has {len(cells)} cells, the header {len(header)}")
        row = dict(zip(header, cells))
        for column in free_text_columns:
            if column in row:
                row[column] = bom_cell_text(row[column])
        out.append(row)
    return header, out


def _table(workdir: Path, name: str, free_text: Sequence[str]) -> dict[str, Any] | None:
    path = confined_file(workdir, (name,))
    if path is None:
        return None
    columns, rows = read_csv_table(path, free_text)
    return {"file": name, "columns": columns, "rows": rows}


def bom_table(workdir: Path) -> dict[str, Any] | None:
    """``{file, columns, rows, not_verified}`` of ``<workdir>/bom.csv`` (free-text cells decoded; ``not_verified`` lists per row the columns whose cell reads ``NOT_VERIFIED``); ``None`` without the file."""
    table = _table(workdir, BOM_FILE, BOM_FREE_TEXT_COLUMNS)
    if table is not None:
        table["not_verified"] = [[c for c, v in row.items() if v == NOT_VERIFIED_CELL] for row in table["rows"]]
    return table


def cpl_table(workdir: Path) -> dict[str, Any] | None:
    """``{file, columns, rows}`` of ``<workdir>/cpl.csv`` exactly as written; ``None`` without the file."""
    return _table(workdir, CPL_FILE, ())


def bom_rows(workdir: Path) -> list[dict[str, str]]:
    """The BOM rows as dicts (free-text cells decoded); ``[]`` without ``bom.csv``."""
    table = bom_table(workdir)
    return table["rows"] if table is not None else []


def cpl_rows(workdir: Path) -> list[dict[str, str]]:
    """The CPL rows as dicts, exactly as written; ``[]`` without ``cpl.csv``."""
    table = cpl_table(workdir)
    return table["rows"] if table is not None else []


# --------------------------------------------------------------------------- artifacts and reports


def artifact_rows(ir: CircuitIR, workdir: Path) -> list[dict[str, Any]]:
    """One row per registered artifact (by kind): the path relative to the workdir (as recorded when it lies outside), whether it can be
    downloaded, its content hash, its disk state (:func:`~ai_eda.report.data.artifact_disk_state`) and whether it was generated from the
    current design hash (``fresh`` / ``stale``, the report's labels; ``disk_matches`` is that label being ``on disk``)."""
    design_hash = ir.content_hash()
    rows: list[dict[str, Any]] = []
    for kind in sorted(ir.artifacts, key=str):
        art = ir.artifacts[kind]
        disk = artifact_disk_state(art)
        parts = workdir_parts(art.path, workdir)
        download = "/".join(parts) if parts is not None and confined_file(workdir, parts) is not None else None
        members = [workdir_parts(f, workdir) for f in art.files]
        rows.append(
            {
                "kind": str(kind),
                "path": "/".join(parts) if parts is not None else art.path,
                "inside_workdir": parts is not None,
                "download": download,
                "files": ["/".join(m) if m is not None else str(f) for m, f in zip(members, art.files)],
                "content_hash": art.content_hash or "",
                "generated_from_ir_hash": art.generated_from_ir_hash or "",
                "matches_design_hash": not art.is_stale(design_hash),
                "freshness": STALE if art.is_stale(design_hash) else FRESH,
                "disk": disk,
                # the disk label is ON_DISK: the file there has the registered content hash
                "disk_matches": disk == ON_DISK,
                "generator": art.generator or "",
                "generator_version": art.generator_version or "",
                "notes": list(art.notes),
            }
        )
    return rows


#: the facts of an artifact row a file preview carries beside it (copied from :func:`artifact_rows`, never recomputed)
ARTIFACT_STATE_KEYS: tuple[str, ...] = ("kind", "path", "download", "freshness", "matches_design_hash", "disk", "disk_matches")


def artifact_state(rows: Sequence[dict[str, Any]], kind: ArtifactKind) -> dict[str, Any] | None:
    """The row of :func:`artifact_rows` for ``kind`` cut to :data:`ARTIFACT_STATE_KEYS` (``None`` when ``kind`` is not registered):
    whether the file was generated from the current design hash, and its disk state - shown next to the file's preview."""
    for row in rows:
        if row["kind"] == str(kind):
            return {key: row[key] for key in ARTIFACT_STATE_KEYS}
    return None


def stage_report_names() -> dict[str, tuple[str, str]]:
    """Every file name a stage report can have (``01_이론_보고서.md`` / ``.html`` / ``.pdf`` ...) -> ``(stage, suffix)``."""
    out: dict[str, tuple[str, str]] = {}
    for stage, name in STAGE_REPORTS.items():
        stem = Path(name).stem
        for suffix in REPORT_SUFFIXES:
            out[stem + suffix] = (str(stage), suffix)
    return out


def stage_report_file(workdir: Path, name: str) -> Path | None:
    """``<workdir>/reports/<name>`` when ``name`` is a stage report's file name (:func:`stage_report_names`) and the file exists inside the workdir."""
    if name not in stage_report_names():
        return None
    return confined_file(workdir, (REPORTS_DIR, name))


def report_files(workdir: Path) -> dict[str, Any]:
    """The stage reports that exist (per report: stage, Korean title, and the ``.md`` / ``.html`` / ``.pdf`` names present, ``None`` for an absent one) and ``report.html`` when it was written."""
    reports: list[dict[str, Any]] = []
    for stage, name in STAGE_REPORTS.items():
        stem = Path(name).stem
        present = {suffix: (stem + suffix) if confined_file(workdir, (REPORTS_DIR, stem + suffix)) is not None else None for suffix in REPORT_SUFFIXES}
        if not any(present.values()):
            continue
        reports.append({"stage": str(stage), "title": REPORT_TITLES[stage], "md": present[".md"], "html": present[".html"], "pdf": present[".pdf"]})
    report_html = REPORT_HTML_FILE if confined_file(workdir, (REPORT_HTML_FILE,)) is not None else None
    return {"stage_reports": reports, "report_html": report_html}


# --------------------------------------------------------------------------- the project zip


def _tree(workdir: Path, top: str) -> list[tuple[str, ...]]:
    """Every regular file under ``<workdir>/<top>`` (no link followed, none listed), as components below the workdir."""
    base = confined_dir(workdir, top)
    if base is None:
        return []
    out: list[tuple[str, ...]] = []
    for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
        here = Path(dirpath)
        dirnames[:] = sorted(d for d in dirnames if not (here / d).is_symlink())
        for name in sorted(filenames):
            parts = (top, *here.relative_to(base).parts, name)
            if confined_file(workdir, parts) is not None:
                out.append(parts)
    return out


def confined_dir(workdir: Path, name: str) -> Path | None:
    """``<workdir>/<name>`` when it is a real directory (not a link) directly inside the workdir."""
    if not safe_part(name):
        return None
    try:
        root = Path(workdir).resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    path = root / name
    return path if path.is_dir() and not path.is_symlink() else None


def zip_members(ir: CircuitIR, workdir: Path) -> list[tuple[str, ...]]:
    """The workdir files a project zip holds besides ``ir.json`` (sorted): the registered artifacts inside the workdir (a multi-file artifact's members too),
    ``pipeline.json``, ``report.html``, ``reports/*`` and ``spice/**``; never anything under :data:`ZIP_EXCLUDED_DIRS`."""
    found: set[tuple[str, ...]] = set()
    for art in ir.artifacts.values():
        for path in (art.path, *art.files):
            parts = workdir_parts(path, workdir)
            if parts is not None and confined_file(workdir, parts) is not None:
                found.add(parts)
    for name in (PIPELINE_FILE, REPORT_HTML_FILE):
        if confined_file(workdir, (name,)) is not None:
            found.add((name,))
    reports = confined_dir(workdir, REPORTS_DIR)
    if reports is not None:
        for entry in sorted(reports.iterdir(), key=lambda p: p.name):
            if confined_file(workdir, (REPORTS_DIR, entry.name)) is not None:
                found.add((REPORTS_DIR, entry.name))
    found.update(_tree(workdir, RESULTS_DIR))
    return sorted(p for p in found if p[0] not in ZIP_EXCLUDED_DIRS and p != (IR_FILE,))


def project_zip(ir: CircuitIR, ir_bytes: bytes, workdir: Path, name: str) -> bytes:
    """A zip of the project built in memory: ``<name>/ir.json`` (``ir_bytes``, the file as read) and every :func:`zip_members` file under ``<name>/``.

    Entries are sorted, dated :data:`ZIP_DATE_TIME`, mode 0644, deflated at
    a fixed level: the same files give the same bytes. A file that vanishes
    between the listing and the read is left out.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        entries: list[tuple[str, bytes]] = [(f"{name}/{IR_FILE}", ir_bytes)]
        for parts in zip_members(ir, workdir):
            path = confined_file(workdir, parts)
            if path is None:
                continue
            try:
                data = path.read_bytes()
            except OSError:
                continue
            entries.append((f"{name}/{'/'.join(parts)}", data))
        for arcname, data in sorted(entries, key=lambda e: e[0]):
            info = zipfile.ZipInfo(arcname, date_time=ZIP_DATE_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            zf.writestr(info, data, compresslevel=6)
    return buf.getvalue()


__all__ = [
    "BOM_FILE",
    "BOM_FREE_TEXT_COLUMNS",
    "CPL_FILE",
    "KICAD_3D_KINDS",
    "KICAD_3D_LABEL",
    "MODEL3D_SVG_WIDTH",
    "NOT_VERIFIED_CELL",
    "NO_KICAD_RENDER",
    "NO_MODEL3D",
    "NO_MODEL3D_VIEW",
    "PreviewMissing",
    "REPORT_HTML_FILE",
    "WAVEFORM_KINDS",
    "WaveformPlan",
    "ZIP_DATE_TIME",
    "ZIP_EXCLUDED_DIRS",
    "artifact_download",
    "artifact_file",
    "artifact_rows",
    "artifact_state",
    "board_available",
    "board_classes",
    "board_svg",
    "bom_rows",
    "bom_table",
    "confined_dir",
    "confined_file",
    "cpl_rows",
    "cpl_table",
    "kicad_3d_files",
    "kicad_render_file",
    "missing_sentence",
    "model3d_glb",
    "model3d_scene",
    "model3d_summary",
    "model3d_svg",
    "project_schematic_svg",
    "project_zip",
    "read_csv_table",
    "read_results_file",
    "report_files",
    "safe_part",
    "schematic_svg",
    "simulation_summary",
    "stage_report_file",
    "stage_report_names",
    "waveform_plan",
    "waveform_svg",
    "waveform_svgs",
    "workdir_parts",
    "zip_members",
]
