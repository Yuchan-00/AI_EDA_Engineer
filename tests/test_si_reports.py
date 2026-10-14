"""The signal-integrity views: the board figure's class colours and plane hatches, the SI figures, the report sections, the 3D planes, the GUI toggles.

Synthetic boards (the fixture library of ``tests/test_routing.py``, the
synthetic SI boards of ``tests/test_si_checks.py``), no KiCad. What is proved:

* the board figure tags every track of a design with ``ir.si`` with its net
  class (``data-class`` / ``data-nc``), draws the inner-plane zones as
  hatched outlines under the tracks (never a fill), and its class view
  colours each track by class with a legend; a board without SI data or
  planes carries none of it;
* the Z0 curve exists only over a plane (the reason otherwise), marks the
  widths the copper carries and the class's target band; the delay bars copy
  the ``si.critical_length`` record; the step figure reads only a fresh
  rawfile (ngspice-gated);
* the circuit report's "임피던스·타이밍" section copies the stored statuses
  (a status changed in the record changes the report, nothing recomputed),
  names the promoted nets with their provenance, says why a 2-layer board has
  no impedance, and the theory report's section carries the formulas with
  the IR's numbers; both are deterministic;
* the 3D scene draws inner planes as translucent sheets at the stackup depth
  (a board without them is the scene it was: no new material in the GLB);
* the GUI lists the net classes for its legend, its page has the 평면 / 넷
  클래스 색 toggles and the SI filter, and app.css carries a rule per
  colour token.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from ai_eda.agents import AgentContext, PCBAgent
from ai_eda.design.board import PLANE_CLEARANCE_KEY
from ai_eda.ir import CircuitIR, ValidationStatus as S, Zone, user_requirement
from ai_eda.report.figures import CLASS_COLOURS, CLASS_KEY, NET_CLASS_DEFAULT, SERIES_COLOURS, THT_COLOUR, VIA_COLOUR, board_figure, net_class_styles
from ai_eda.report.si_figures import delay_bar_figure, delay_rows, fresh_si_rawfile, step_response_figure, z0_width_figure
from ai_eda.report.si_report import NO_SI, si_circuit_section, si_theory_section
from ai_eda.report.stages import ReportFigures, build_stage_document, circuit_report, stage_figures, theory_report
from ai_eda.tools.calc.tline import NO_STACKUP
from ai_eda.tools.model3d.glb import write_glb
from ai_eda.tools.model3d.iso import iso_svg
from ai_eda.tools.model3d.scene import build_scene
from ai_eda.tools.spice import NgspiceShared
from ai_eda.tools.spice.si_check import deck_stem, spice_si_results
from ai_eda.validation.si import si_results
from ai_eda.workflow import Orchestrator
from ai_eda.workflow.stages import Stage
from tests.test_routing import fixture_library
from tests.test_si_checks import default_classes, long_board, si_of

runner = NgspiceShared()
needs_ngspice = pytest.mark.skipif(not runner.available(), reason="ngspice shared library not found")
SVG_NS = "{http://www.w3.org/2000/svg}"


@pytest.fixture
def lib(tmp_path: Path):
    return fixture_library(tmp_path / "kicad")


def routed(tmp_path: Path, lib, layers: int | None, *, si: bool = True, validate: bool = True) -> CircuitIR:
    """The synthetic long board routed by the PCB agent (promotion, re-route, planes) with its si.* results recorded."""
    ir = long_board(tmp_path, lib, layers)
    if si:
        ir.si = si_of(*default_classes())
    if layers == 4:
        ir.parameters[PLANE_CLEARANCE_KEY] = user_requirement(0.5, "mm")
    res = PCBAgent().run(ir, AgentContext(workdir=tmp_path, tools={"kicad_library": lib}))
    Orchestrator.apply_proposals(ir, res.proposals)
    if validate and ir.si is not None:
        h = ir.content_hash()
        for r in si_results(ir):
            ir.validation.add(r.model_copy(update={"ir_hash": h}))
    return ir


def _svg(text: str):
    import xml.etree.ElementTree as ET

    return ET.fromstring(text)


# --------------------------------------------------------------------------- the board figure


def test_the_board_figure_tags_tracks_with_their_class_and_hatches_the_planes(tmp_path: Path, lib):
    ir = routed(tmp_path, lib, 4)
    styles = {st.name: st for st in net_class_styles(ir)}
    assert styles["DEFAULT"].token == "nc-default" and styles["DEFAULT"].colour == NET_CLASS_DEFAULT
    assert styles["Z50"].token == "nc-0" and styles["Z50"].colour == SERIES_COLOURS[0] and styles["Z50"].promoted == ("LONG",)
    fig = board_figure(ir, lib)
    root = _svg(fig.svg)
    tracks = [e for e in root.iter(f"{SVG_NS}line") if e.get("class") == "track"]
    assert len(tracks) == len(ir.pcb.tracks)
    assert {(t.get("data-net"), t.get("data-class"), t.get("data-nc")) for t in tracks} >= {("LONG", "Z50", "nc-0"), ("SHORT", "DEFAULT", "nc-default")}
    # the layer view keeps the layer colours; the planes are hatched outlines drawn before the tracks, never a solid fill
    assert {t.get("stroke") for t in tracks} <= {"#e34948", "#2a78d6"}
    order = [g.get("class") for g in root if g.tag == f"{SVG_NS}g"]
    assert order.index("planes") < order.index("tracks")
    planes = [p for p in root.iter(f"{SVG_NS}polygon") if p.get("class") == "plane"]
    assert [(p.get("data-net"), p.get("data-layer")) for p in planes] == [("GND", "In1.Cu"), ("GND", "In2.Cu")]
    assert all(p.get("fill", "").startswith("url(#hatch-board-") for p in planes) and len({p.get("fill") for p in planes}) == 2
    assert "내층 평면 영역 In1.Cu GND (45° 빗금), In2.Cu GND (-45° 빗금)" in fig.caption
    # the class view: each track in its class's colour (B.Cu lighter), neutral SMD pads, a legend of the classes with their net counts
    by_class = board_figure(ir, lib, colour_by="class", fig_id="si_board")
    croot = _svg(by_class.svg)
    ctracks = [e for e in croot.iter(f"{SVG_NS}line") if e.get("class") == "track"]
    assert {t.get("stroke") for t in ctracks if t.get("data-net") == "LONG"} == {SERIES_COLOURS[0]}
    assert {t.get("stroke") for t in ctracks if t.get("data-net") == "SHORT"} == {NET_CLASS_DEFAULT}
    assert all(t.get("stroke-opacity") == "0.5" for t in ctracks if t.get("data-layer") == "B.Cu")
    legend = next(g for g in croot.iter(f"{SVG_NS}g") if g.get("class") == "legend")
    assert [t.text for t in legend.iter(f"{SVG_NS}text")] == ["DEFAULT (2)", "Z50 (1)"]
    assert "url(#hatch-si_board-" in by_class.svg and by_class.title.endswith("보드 그림 (넷 클래스 색)")
    assert by_class.svg == board_figure(ir, lib, colour_by="class", fig_id="si_board").svg  # deterministic
    # no class hue is a colour the class view keeps for something else: the through-hole pads' yellow, the vias' / default class's gray
    assert THT_COLOUR not in CLASS_COLOURS and VIA_COLOUR not in CLASS_COLOURS and CLASS_COLOURS[0] == SERIES_COLOURS[0]
    vias = [g for g in croot.iter(f"{SVG_NS}g") if g.get("class") == "via"]
    assert all(g[0].get("fill") == "#ffffff" and g[0].get("stroke") for g in vias)  # a white disc in a gray ring, never the default class's gray
    ccap = [t.text for t in croot.iter(f"{SVG_NS}text") if t.get("class") == "caption"]
    assert "파랑 = B.Cu" not in " ".join(ccap)
    # the layer figure (the GUI recolours it by class with app.css) carries both keys: the class key hidden until the class view shows it
    caps = [t for t in root.iter(f"{SVG_NS}text") if t.get("class") == "caption"]
    layer_caps = [t for t in caps if t.get("data-view") == "layer"]
    class_caps = [t for t in caps if t.get("data-view") == "class"]
    assert layer_caps and class_caps and all(t.get("display") == "none" for t in class_caps) and all(t.get("display") is None for t in layer_caps)
    assert "파랑 = B.Cu" in " ".join(t.text for t in layer_caps) and "파랑 = B.Cu" not in " ".join(t.text for t in class_caps)
    assert "회색 테의 흰 원 = 비아" in " ".join(t.text for t in class_caps)
    with pytest.raises(ValueError, match="colour_by"):
        board_figure(ir, lib, colour_by="net")


def test_a_board_without_si_or_planes_carries_no_class_or_plane_markup(tmp_path: Path, lib):
    ir = routed(tmp_path, lib, 2, si=False)
    svg = board_figure(ir, lib).svg
    assert "data-nc" not in svg and "data-class" not in svg and 'class="planes"' not in svg and "hatch-" not in svg
    assert net_class_styles(ir) == []
    with pytest.raises(ValueError, match="needs ir.si"):
        board_figure(ir, lib, colour_by="class")


# --------------------------------------------------------------------------- the SI figures


def test_the_z0_curve_exists_only_over_a_plane_and_marks_the_routed_widths(tmp_path: Path, lib):
    four = routed(tmp_path / "4", lib, 4)
    fig = z0_width_figure(four)
    assert not isinstance(fig, str)
    assert "0.35 mm → 49.7 Ω (Z50)" in fig.caption and "0.4 mm →" in fig.caption and "(DEFAULT)" in fig.caption
    assert "Z50 목표 50 Ω ± 10%" in fig.svg and fig.svg.count('class="marker"') == 2
    series = [e for e in _svg(fig.svg).iter(f"{SVG_NS}polyline") if e.get("class") == "series"]
    assert len(series) == 1 and "F.Cu / B.Cu" in series[0].get("data-name")  # one curve: the symmetric stack's two layers agree
    assert fig.svg == z0_width_figure(four).svg
    two = z0_width_figure(routed(tmp_path / "2", lib, 2))
    assert isinstance(two, str) and "impedance is undefined without a reference plane" in two
    bare = long_board(tmp_path / "none", lib, None)
    assert z0_width_figure(bare) == NO_STACKUP


def test_the_delay_bars_copy_the_critical_length_record(tmp_path: Path, lib):
    ir = routed(tmp_path, lib, 2)
    crit = ir.validation.latest("si.critical_length")
    rows = delay_rows(crit)
    assert [r.net for r in rows] == ["LONG", "SHORT"] and [r.long for r in rows] == [True, False] and all(r.bound for r in rows)
    assert rows[0].threshold_s == pytest.approx(0.5e-9) and rows[0].delay_s == pytest.approx(crit.details["nets"][0]["delay_ps"] * 1e-12)
    fig = delay_bar_figure(rows, title="t")
    assert fig.svg.count('class="bar"') == 2 and 'data-net="LONG"' in fig.svg and 'data-long="true"' in fig.svg and fig.svg.count('class="threshold"') == 1
    assert "f × t_r = 500 ps" in fig.caption and "상한 sqrt(εr)/c0" in fig.caption
    assert delay_bar_figure(rows, title="t", limit=1).svg.count('class="bar"') == 1 and "나머지 1개는" in delay_bar_figure(rows, title="t", limit=1).caption
    with pytest.raises(ValueError):
        delay_bar_figure([], title="t")


def test_a_step_figure_needs_a_fresh_tool_backed_result(tmp_path: Path, lib):
    ir = routed(tmp_path, lib, 2)
    for r in spice_si_results(ir, {}, tmp_path / "w"):
        assert fresh_si_rawfile(ir, r) == f"{r.check_id} 은 NOT_VERIFIED 이라 파형이 없습니다"


@needs_ngspice
def test_the_step_figure_draws_the_fresh_rawfile_and_refuses_a_changed_one(tmp_path: Path, lib):
    ir = routed(tmp_path, lib, 4)
    res = {r.check_id: r for r in spice_si_results(ir, {"spice": runner}, tmp_path / "w")}
    r = res["spice.si.LONG"]
    raw = fresh_si_rawfile(ir, r)
    assert isinstance(raw, Path) and raw.name == f"{deck_stem('LONG')}.tran.raw"
    fig = step_response_figure(r, raw, fig_id="si_step_LONG")
    names = [e.get("data-name") for e in _svg(fig.svg).iter(f"{SVG_NS}polyline") if e.get("class") == "series"]
    assert names == ["먼 끝 v(B) (부하 쪽)", "가까운 끝 v(A) (드라이버 쪽)"] and f"→ {r.status}" in fig.caption
    assert f"오버슈트 {r.details['measured']['overshoot_rel']:.1%}" in fig.caption
    stale = r.model_copy(update={"ir_hash": "sha256:" + "0" * 64})
    assert fresh_si_rawfile(ir, stale) == "spice.si.LONG 은 이전 IR 버전의 결과입니다"
    raw.write_bytes(raw.read_bytes() + b"\n")
    assert "기록된 해시와 다릅니다" in fresh_si_rawfile(ir, r)
    # the report places one figure per fresh result
    ir.validation.add(res["spice.si.LONG"])
    raw.write_bytes(raw.read_bytes()[:-1])
    figures = stage_figures(Stage.PCB, ir, lib)
    assert figures.slots["si_step"] == ["si_step_LONG"]


# --------------------------------------------------------------------------- the report sections


def test_the_circuit_report_section_copies_the_record_over_a_plane(tmp_path: Path, lib):
    ir = routed(tmp_path, lib, 4)
    md = circuit_report(ir, lib, None)
    sec = md.split("## 임피던스·타이밍", 1)[1].split("\n## ", 1)[0]
    for text in ("| In1.Cu | 구리 | 17.5 µm | - | 평면: `GND` |", "| `Z50` | - | `LONG` | 1 | 50 Ω ± 10 % |", "`DEFAULT` → `Z50`",
                 "w = 0.35 mm (Z50): u = 1.75, Z0 = 49.71 Ω", "→ 배선 폭 0.35 mm", "- `si.impedance.Z50`: **PASS**", "- `si.critical_length`: **PASS**",
                 "![fig](fig:si_board)", "![fig](fig:si_z0)", "![fig](fig:si_delay)", "SPICE 계단 응답 그림이 없습니다", "KiCad DRC 의 넷 클래스 폭 검사는 측정되지 않았습니다"):
        assert text in sec, text
    promo = ir.si.promotion_of("LONG")
    assert promo.provenance.note in sec and f"| {promo.length_mm:.3f} |" in sec
    # a view: the status printed is the stored one - change the record and the report follows, nothing is recomputed
    imp = ir.validation.latest("si.impedance.Z50")
    ir.validation.add(imp.model_copy(update={"status": S.FAIL, "message": "stored verdict"}))
    assert "- `si.impedance.Z50`: **FAIL** (`si` v0.1) — stored verdict" in circuit_report(ir, lib, None)
    assert build_stage_document(Stage.PCB, ir, lib, None).html == build_stage_document(Stage.PCB, ir, lib, None).html
    # the 0.2 identity is a board-level fact copied from the copper's provenance, never a per-net promise
    assert "아무것도 선언·승격되지 않은 넷은 routing.maze 0.2 와 바이트 단위로 같게 배선됩니다" not in sec
    assert "이 보드는 1 개 넷(`LONG`)에 라우터 규칙이 있어, 모든 넷이 한 협상에서 배선되었고 모든 트랙이 routing.maze 0.3 으로 기록되어 있습니다" in sec
    assert "0.2 와의 바이트 동일성은 어떤 넷에도 규칙이 없는 보드에서만 성립하며, 이 보드는 그렇지 않습니다" in sec
    # the compiled board carries no (stackup): the report says so
    assert "컴파일된 `.kicad_pcb` 에는 `(stackup)` 절이 없습니다" in sec


def test_the_circuit_report_names_a_board_routed_without_rules_as_routing_maze_0_2(tmp_path: Path, lib):
    from ai_eda.report.si_report import router_version_text

    ir = long_board(tmp_path, lib, 2, length=30.0)  # every net short: nothing declared, nothing promoted, no rule
    ir.si = si_of(*default_classes())
    Orchestrator.apply_proposals(ir, PCBAgent().run(ir, AgentContext(workdir=tmp_path, tools={"kicad_library": lib})).proposals)
    assert router_version_text(ir).startswith("이 보드는 어떤 넷에도 라우터 규칙이 없어 routing.maze 0.2 와 바이트 단위로 같게 배선되었습니다")


def test_the_circuit_report_section_says_why_a_two_layer_board_has_no_impedance(tmp_path: Path, lib):
    ir = routed(tmp_path, lib, 2)
    sec = circuit_report(ir, lib, None).split("## 임피던스·타이밍", 1)[1].split("\n## ", 1)[0]
    assert "- **F.Cu / B.Cu**: 기준 평면이 없습니다" in sec and "t_pd ≤ √εr / c0 = √4.5 / 299792458 m/s = 7.0760 ps/mm" in sec
    assert "승격된 넷도 제어 폭을 받지 못했습니다" in sec and "use pcb_layers=4 or add a plane" in sec
    assert "fig:si_z0" not in sec and "기준 평면이 없어 임피던스가 정의되지 않으므로 Z0–폭 곡선이 없습니다" in sec
    assert "- **Z50 의 폭**: 제어 폭 없음" in sec


def test_a_design_without_si_gets_one_sentence_and_no_section_in_the_reports(tmp_path: Path, lib):
    ir = routed(tmp_path, lib, 2, si=False)
    assert si_circuit_section(ir, ReportFigures()) == ["## 임피던스·타이밍", "", NO_SI + ".", ""]
    assert "## 임피던스·타이밍" not in circuit_report(ir, lib, None) and "## 전송선로와 타이밍" not in theory_report(ir, lib)
    assert not any(slot.startswith("si_") for slot in stage_figures(Stage.PCB, ir, lib).slots)


def test_the_theory_section_carries_the_formulas_with_the_irs_numbers(tmp_path: Path, lib):
    ir = routed(tmp_path, lib, 4, validate=False)
    text = "\n".join(si_theory_section(ir))
    for needle in ("l_crit = f × t_r / t_pd,   f = 1/2", "Z0 = Z01(u + Δu_r)/√e_eff(u + Δu_r)", "w = 0.346357 mm", "Γ_s = (R_s − Z0)/(R_s + Z0)",
                   "w_min = A / t", "setup = t_lc + t_flight(clk)", "impedance is undefined without a reference plane - use pcb_layers=4 or add a plane"):
        assert needle in text, needle
    assert "## 전송선로와 타이밍 (신호 무결성 이론)" in theory_report(ir, lib)
    assert text == "\n".join(si_theory_section(ir))
    # over a plane the design's l_crit is the microstrip's, per routed width - not the no-plane bound sqrt(er)/c0 (70.66 mm)
    assert "상한 t_pd = √4.5 / c0" not in text and "70.66 mm" not in text
    assert "이 설계의 F.Cu 는 In1.Cu (GND 평면) 위의 마이크로스트립입니다" in text and "w = 0.35 mm: e_eff =" in text and "l_crit =" in text
    # the open-end first arrival is not an upper bound: a capacitive load can raise the peak above it
    assert "상한;" not in text and "유한한 t_r 과 부하" not in text and "이것은 **상한이 아닙니다**" in text and "판정은 `spice.si` 의 ngspice 과도 해석만 합니다" in text
    two = routed(tmp_path / "two", lib, 2, validate=False)
    t2 = "\n".join(si_theory_section(two))
    assert "이 설계의 F.Cu 에는 기준 평면이 없습니다: 상한 t_pd = √4.5 / c0 = 7.0760 ps/mm, 그래서 l_crit ≥ 0.5 × 1 ns / 7.0760 ps/mm = 70.66 mm" in t2
    assert "규칙은 이 하한 값을 그대로 써서" in t2


# --------------------------------------------------------------------------- the 3D planes


def test_inner_planes_are_translucent_sheets_at_their_stackup_depth(tmp_path: Path, lib):
    four = routed(tmp_path / "4", lib, 4, validate=False)
    scene = build_scene(four, lib, model_dir=None)
    planes = [s for s in scene.solids if s.kind == "plane"]
    slab = next(s for s in scene.solids if s.kind == "slab")
    assert len(planes) == 4 and {s.faces for s in planes} == {"top", "bottom"} and slab.material == "board_clear"
    # In1.Cu lies under F.Cu (35 um) and the 0.2 mm prepreg: its top face is 0.235 mm below the 1.6 mm board's top
    assert {(s.label, s.z1) for s in planes} >= {("GND In1.Cu", round(1.6 - 0.235, 6))}
    assert any(n.startswith("내층 평면 In1.Cu GND, In2.Cu GND") for n in scene.notes)
    doc = json.loads(write_glb(scene)[20:20 + int.from_bytes(write_glb(scene)[12:16], "little")])
    mats = {m["name"]: m for m in doc["materials"]}
    assert mats["plane"].get("alphaMode") == "BLEND" and mats["board_clear"].get("alphaMode") == "BLEND" and "board" not in mats
    assert 'class="k-plane g-copper"' in iso_svg(scene, "iso")
    two = build_scene(routed(tmp_path / "2", lib, 2, validate=False), lib, model_dir=None)
    assert not [s for s in two.solids if s.kind == "plane"] and next(s for s in two.solids if s.kind == "slab").material == "board"
    doc2 = json.loads(write_glb(two)[20:20 + int.from_bytes(write_glb(two)[12:16], "little")])
    assert {m["name"] for m in doc2["materials"]} & {"plane", "board_clear"} == set()
    # an inner zone without a stackup has no depth: not drawn, said in the notes
    bare = routed(tmp_path / "b", lib, 2, si=False, validate=False)
    bare.pcb.stackup = None
    bare.pcb.zones = [Zone(net="GND", layer="In1.Cu", polygon=[(1, 1), (5, 1), (5, 5), (1, 5)], provenance=four.pcb.zones[0].provenance)]
    s3 = build_scene(bare, lib, model_dir=None)
    assert not [s for s in s3.solids if s.kind == "plane"] and any("적층(ir.pcb.stackup)이 없어" in n for n in s3.notes)


# --------------------------------------------------------------------------- the GUI


def test_the_gui_lists_the_classes_and_its_page_has_the_si_toggles(tmp_path: Path, lib):
    from ai_eda.gui.page import APP_CSS, APP_HTML, APP_JS, NETCLASS_CSS
    from ai_eda.gui.preview import board_classes

    ir = routed(tmp_path, lib, 4, validate=False)
    assert board_classes(ir) == [{"name": "DEFAULT", "token": "nc-default", "default": True, "nets": 2, "promoted": []},
                                 {"name": "Z50", "token": "nc-0", "default": False, "nets": 1, "promoted": ["LONG"]}]
    assert board_classes(routed(tmp_path / "p", lib, 2, si=False, validate=False)) == []
    assert '<input type="checkbox" id="layer-planes" data-layer-class="planes" checked>' in APP_HTML
    assert '<input type="checkbox" id="board-netclass">' in APP_HTML and '<select id="val-group">' in APP_HTML
    assert "box.classList.toggle('by-class', byClass);" in APP_JS and "r.check_id.startsWith('spice.si.')" in APP_JS
    for i, colour in enumerate(CLASS_COLOURS):
        assert f'.board-view.by-class line.track[data-nc="nc-{i}"] {{ stroke: {colour}; }}' in NETCLASS_CSS
    assert all(f"stroke: {THT_COLOUR}" not in r for r in NETCLASS_CSS.splitlines() if "line.track" in r)
    # the class view swaps the layer caption for the class one, and draws vias as white discs in a gray ring
    assert '.board-view.by-class text.caption[data-view="layer"] { display: none; }' in NETCLASS_CSS
    assert '.board-view.by-class text.caption[data-view="class"] { display: inline; }' in NETCLASS_CSS
    assert ".board-view.by-class g.via > circle:not(.drill) { fill: #ffffff;" in NETCLASS_CSS
    assert NETCLASS_CSS in APP_CSS and "url(" not in APP_CSS and not re.search(r"style=", APP_HTML)
