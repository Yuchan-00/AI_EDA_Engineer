"""Typed vias (``Via.kind`` blind / buried / micro with a layer span) in the IR, the compiler, the drill export and every consumer.

No producer of typed vias exists yet (the router comes later), so every board here is built by hand:

* ``Via.kind`` defaults to ``None`` and is left out of every dump while ``None``: a via, an IR and its design hash serialise byte for
  byte as at c270797 (the strings below were printed by c270797's model), and the pinned pre-field IRs keep their hashes.
* :func:`~ai_eda.ir.pcb.via_span_problem` names each refused span; the model refuses a typed via whose names alone are illegal, the
  compiler and ``pcb.routing`` also refuse one the board cannot hold; a ``kind``-less via on any pair but ``F.Cu`` / ``B.Cu`` is a
  CompileError (KiCad would build it through).
* The compiler writes KiCad 10's token after ``via``; with kicad-cli the board survives ``pcb upgrade`` with every token and span, and
  ``export_drill`` returns ``<stem>.drl`` plus one file per span in :func:`~ai_eda.compilers.gerber.drill_pairs` order; a typed via
  ``F.Cu``-``B.Cu`` (a micro via on a 2-layer board) goes into ``<stem>.drl`` and names no pair file (measured).
* ``pcb.routing`` / ``pcb.keepout`` put a typed via's copper on its span only (stamped 0.2), ``mfg.capability`` adds a NOT_VERIFIED
  ``via_process`` row (stamped 0.2), SI counts each via's own barrel and extracts no path through a typed via (stamped si 0.2), the 3D
  scene says the span is not drawn - and every one of them treats a board without a typed via exactly as before.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ai_eda.compilers.base import CompileContext
from ai_eda.compilers.gerber import TYPED_VIA_VERSION, DrillExporter, drill_pairs
from ai_eda.compilers.pcb import PCBCompiler
from ai_eda.errors import CompileError, ToolExecutionError
from ai_eda.ir import (
    ArtifactKind,
    BoardOutline,
    CircuitIR,
    DielectricKind,
    Keepout,
    Layer,
    ManufacturingConstraints,
    PCBDesign,
    Placement,
    Provenance,
    ProvenanceKind,
    Stackup,
    StackupCopper,
    StackupDielectric,
    Track,
    ValidationStatus as S,
    Via,
    Zone,
    assumption,
    user_requirement,
)
from ai_eda.ir.pcb import MICRO_VIA_ADJACENT, copper_position, copper_stack_order, via_span, via_span_problem
from ai_eda.ir.provenance import design_data
from ai_eda.tools.calc.tline import propagation_delay
from ai_eda.tools.kicad import sexpr
from ai_eda.tools.kicad.cli import KicadCli
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.manufacturing.capability import TOOL_VERSION as CAPABILITY_VERSION, check_capability
from ai_eda.tools.model3d.scene import build_scene
from ai_eda.tools.si.measure import measure_nets
from ai_eda.tools.si.paths import TYPED_VIA_NOT_SPLIT, longest_path
from ai_eda.validation import ValidationContext, default_registry
from ai_eda.validation.layout import (
    CLEARANCE_CHECK,
    CONNECTIVITY_CHECK,
    KEEPOUT_CHECK,
    NOT_COMPARED,
    TOOL_ID,
    TOOL_VERSION,
    TYPED_VIA_TOOL_VERSION,
    TYPED_VIAS_NOT_COMPARED,
)
from tests.test_parts_existence import synthetic_library
from tests.test_pcb_agent import parts_ir
from tests.test_routing import board_ir, fixture_library

SIX = ["F.Cu", "In1.Cu", "In2.Cu", "In3.Cu", "In4.Cu", "B.Cu"]
P = Provenance(kind=ProvenanceKind.USER_REQUIREMENT, note="drawn by hand for the test")
W = 0.4
kicad = KicadCli()
needs_kicad = pytest.mark.skipif(not kicad.available(), reason="kicad-cli not installed")


def _via(x: float, y: float, layers: tuple[str, str] = ("F.Cu", "B.Cu"), kind: str | None = None, net: str = "N", d: float = 0.6, drill: float = 0.3) -> Via:
    return Via(net=net, x_mm=x, y_mm=y, drill_mm=drill, diameter_mm=d, layers=layers, kind=kind, provenance=P)


def _track(net: str, a: tuple[float, float], b: tuple[float, float], layer: str = "F.Cu") -> Track:
    return Track(net=net, layer=layer, start=a, end=b, width_mm=W, provenance=P)


# --------------------------------------------------------------------------- the IR: unchanged bytes, the span rules


def test_a_via_without_a_kind_serialises_exactly_as_at_c270797():
    """The strings were printed by c270797's model (``git archive c270797 ai_eda``) for the same vias."""
    prov = Provenance(kind=ProvenanceKind.DERIVED, tool="routing.maze", tool_version="0.8", derived_from=["net:N"], created_at=datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc))
    v = Via(net="N", x_mm=1.25, y_mm=-2.5, drill_mm=0.3, diameter_mm=0.6, provenance=prov)
    assert v.kind is None
    assert v.model_dump_json() == (
        '{"net":"N","x_mm":1.25,"y_mm":-2.5,"drill_mm":0.3,"diameter_mm":0.6,"layers":["F.Cu","B.Cu"],"provenance":{"kind":"derived","source":null,'
        '"derived_from":["net:N"],"inputs":{},"tool":"routing.maze","tool_version":"0.8","note":null,"created_at":"2026-01-02T03:04:05Z"}}'
    )
    assert design_data(v) == {"net": "N", "x_mm": 1.25, "y_mm": -2.5, "drill_mm": 0.3, "diameter_mm": 0.6, "layers": ["F.Cu", "B.Cu"],
                              "provenance": {"kind": "derived", "source": None, "derived_from": ["net:N"], "inputs": {}, "tool": "routing.maze", "tool_version": "0.8", "note": None}}
    assert "kind" not in v.model_dump() and "kind" not in v.model_dump(mode="json")
    typed = v.model_copy(update={"kind": "blind", "layers": ("F.Cu", "In2.Cu")})
    assert '"layers":["F.Cu","In2.Cu"],"kind":"blind","provenance"' in typed.model_dump_json() and design_data(typed)["kind"] == "blind"
    assert Via.model_validate_json(typed.model_dump_json()) == typed  # a saved typed via loads back


def test_the_pinned_ir_hashes_hold():
    from tests.test_rf_ir import PRE_KR447_HASH, PRE_KR447_IR
    from tests.test_stackup import PRE_SILK_HASH, PRE_SILK_IR, PRE_STACKUP_HASH, PRE_STACKUP_IR

    for path, expected in ((PRE_STACKUP_IR, PRE_STACKUP_HASH), (PRE_SILK_IR, PRE_SILK_HASH), (PRE_KR447_IR, PRE_KR447_HASH)):
        ir = CircuitIR.load(path)
        assert ir.pcb is not None and ir.pcb.vias and all(v.kind is None for v in ir.pcb.vias)
        assert ir.pcb.tracks and all(t.mid is None for t in ir.pcb.tracks)  # no arc: a track without ``mid`` hashes as before the field existed
        assert ir.content_hash() == expected


def test_span_rules_name_each_refusal():
    assert via_span_problem("blind", "F.Cu", "In2.Cu") is None and via_span_problem("blind", "In3.Cu", "B.Cu", SIX) is None
    assert via_span_problem("buried", "In2.Cu", "In3.Cu", SIX) is None and via_span_problem("micro", "F.Cu", "In1.Cu") is None
    assert via_span_problem("micro", "In4.Cu", "B.Cu", SIX) is None
    cases = {
        ("blind", "F.Cu", "Edge.Cuts"): "'Edge.Cuts' is not a copper layer name",
        ("blind", "In0.Cu", "B.Cu"): "'In0.Cu' is not a copper layer name",
        ("micro", "In2.Cu", "In2.Cu"): "both ends are the same layer",
        ("blind", "In2.Cu", "F.Cu"): "the upper layer comes first (KiCad writes it so)",
        ("blind", "B.Cu", "In3.Cu"): "the upper layer comes first (KiCad writes it so)",
        ("blind", "F.Cu", "B.Cu"): "a blind via has exactly one outer end",
        ("blind", "In1.Cu", "In3.Cu"): "a blind via has exactly one outer end",
        ("buried", "F.Cu", "In2.Cu"): "a buried via has no outer end",
        ("buried", "In3.Cu", "B.Cu"): "a buried via has no outer end",
        ("micro", "F.Cu", "In2.Cu"): MICRO_VIA_ADJACENT,
        ("micro", "In1.Cu", "In3.Cu"): MICRO_VIA_ADJACENT,
        ("through", "F.Cu", "B.Cu"): "'through' is not a via kind (blind, buried, micro)",
    }
    for (kind, top, bottom), why in cases.items():
        assert via_span_problem(kind, top, bottom) == why, (kind, top, bottom)
    assert MICRO_VIA_ADJACENT == "a micro via joins two adjacent copper layers (a stated rule of this version; KiCad 10.0.6 DRC does not check it)"
    # B.Cu's position needs the board: without it a micro via to B.Cu is not judged, with it it is
    assert via_span_problem("micro", "In3.Cu", "B.Cu") is None and via_span_problem("micro", "In3.Cu", "B.Cu", SIX) == MICRO_VIA_ADJACENT
    assert via_span_problem("blind", "F.Cu", "In7.Cu", SIX) == "'In7.Cu' is not a copper layer of the board"
    assert via_span_problem("micro", "F.Cu", "B.Cu", ["F.Cu", "B.Cu"]) is None  # two layers are adjacent: the rule is adjacency only
    # the model refuses what the names alone make illegal, and keeps a kind-less via permissive (a design error the compiler names)
    with pytest.raises(ValueError, match="via blind F.Cu-B.Cu: a blind via has exactly one outer end"):
        _via(1.0, 1.0, ("F.Cu", "B.Cu"), "blind")
    with pytest.raises(ValueError, match=r"via micro F.Cu-In2.Cu: a micro via joins two adjacent copper layers \(a stated rule"):
        _via(1.0, 1.0, ("F.Cu", "In2.Cu"), "micro")
    with pytest.raises(ValueError, match="kind"):
        Via(net="N", x_mm=0, y_mm=0, drill_mm=0.3, diameter_mm=0.6, kind="through")
    assert _via(1.0, 1.0, ("F.Cu", "In1.Cu")).kind is None


def test_positions_stack_order_and_spans():
    assert [copper_position(n, SIX) for n in SIX] == [0, 1, 2, 3, 4, 5]
    assert copper_position("In10.Cu") == 10 and copper_position("B.Cu") > copper_position("In99.Cu")
    with pytest.raises(ValueError):
        copper_position("F.SilkS")
    assert copper_stack_order(["B.Cu", "In10.Cu", "F.Cu", "In2.Cu", "Edge.Cuts", "In1.Cu"]) == ["F.Cu", "In1.Cu", "In2.Cu", "In10.Cu", "B.Cu"]
    assert via_span(_via(0, 0), SIX) == SIX
    assert via_span(_via(0, 0, ("F.Cu", "In2.Cu"), "blind"), SIX) == ["F.Cu", "In1.Cu", "In2.Cu"]
    assert via_span(_via(0, 0, ("In2.Cu", "In3.Cu"), "buried"), SIX) == ["In2.Cu", "In3.Cu"]
    assert via_span(_via(0, 0, ("In4.Cu", "B.Cu"), "micro"), SIX) == ["In4.Cu", "B.Cu"]


# --------------------------------------------------------------------------- the compiler and the drill export


def _six_layer_ir(tmp_path: Path, lib: KicadLibrary, vias: list[Via]) -> CircuitIR:
    ir = parts_ir(tmp_path, lib, ("R1", "R2"))
    ir.pcb = PCBDesign(
        outline=BoardOutline(width_mm=40.0, height_mm=20.0), layers=[Layer(name=n, kind="signal") for n in SIX], vias=vias,
        placements=[Placement(component_ref="R1", x_mm=8.0, y_mm=10.0, provenance=P), Placement(component_ref="R2", x_mm=32.0, y_mm=10.0, provenance=P)],
    )
    return ir


TYPED = [
    _via(12.0, 15.0, net="N0"),
    _via(16.0, 15.0, ("F.Cu", "In2.Cu"), "blind", net="N0"),
    _via(20.0, 15.0, ("In2.Cu", "In3.Cu"), "buried", net="N0"),
    _via(24.0, 15.0, ("In3.Cu", "B.Cu"), "blind", net="N0"),
    _via(28.0, 15.0, ("F.Cu", "In1.Cu"), "micro", net="N0", d=0.3, drill=0.1),
    _via(30.0, 15.0, ("In4.Cu", "B.Cu"), "micro", net="N0", d=0.3, drill=0.1),
    _via(34.0, 15.0, ("F.Cu", "In1.Cu"), "micro", net="N0", d=0.3, drill=0.1),  # a second via on a span: one drill file
]
#: (token, layers) of each via of :data:`TYPED` as the compiled board must carry them
EXPECTED_VIAS = [(None, ["F.Cu", "B.Cu"]), ("blind", ["F.Cu", "In2.Cu"]), ("buried", ["In2.Cu", "In3.Cu"]), ("blind", ["In3.Cu", "B.Cu"]),
                 ("micro", ["F.Cu", "In1.Cu"]), ("micro", ["In4.Cu", "B.Cu"]), ("micro", ["F.Cu", "In1.Cu"])]


def _board_vias(tree: list) -> list[tuple[str | None, list[str], str]]:
    """``(type token or None, layers, uuid)`` of every ``(via ...)`` of a board tree, in file order."""
    out = []
    for v in sexpr.find_all(tree, "via"):
        token = v[1] if isinstance(v[1], str) else None
        out.append((token, [str(x) for x in sexpr.args(sexpr.find(v, "layers"))], str(sexpr.get(v, "uuid"))))
    return out


def test_typed_vias_compile_to_kicad_10_tokens_and_drill_pairs(tmp_path: Path):
    lib = synthetic_library(tmp_path / "kicad")
    ir = _six_layer_ir(tmp_path, lib, [v.model_copy() for v in TYPED])
    ref = PCBCompiler().compile(ir, CompileContext(workdir=tmp_path / "out", tools={"kicad_library": lib}))
    assert ref.generator_version == "0.2"  # PCBCompiler.version is unchanged: the token appears only on a typed via
    tree = sexpr.parse_file(Path(ref.path))
    assert [(token, layers) for token, layers, _ in _board_vias(tree)] == EXPECTED_VIAS
    through = sexpr.find_all(tree, "via")[0]
    assert sexpr.head(through[1]) == "at"  # a through via keeps the token-less form
    assert drill_pairs(ir) == ["front-in1", "front-in2", "in2-in3", "in3-back", "in4-back"]
    big = ir.model_copy(deep=True)
    big.pcb.layers = [Layer(name=n, kind="signal") for n in ["F.Cu", *[f"In{k}.Cu" for k in range(1, 11)], "B.Cu"]]
    big.pcb.vias = [_via(5.0, 5.0, ("In10.Cu", "B.Cu"), "micro"), _via(6.0, 5.0, ("In2.Cu", "In9.Cu"), "buried"), _via(7.0, 5.0)]
    assert drill_pairs(big) == ["in10-back", "in2-in9"]  # sorted names, as kicad-cli 10.0.6 wrote them on a 12-layer probe
    assert drill_pairs(ir.model_copy(update={"pcb": ir.pcb.model_copy(update={"vias": [TYPED[0]]})})) == []


def test_the_compiler_refuses_what_kicad_would_build_differently(tmp_path: Path):
    lib = synthetic_library(tmp_path / "kicad")
    ctx = CompileContext(workdir=tmp_path / "out", tools={"kicad_library": lib})
    ir = _six_layer_ir(tmp_path, lib, [_via(12.0, 15.0, ("F.Cu", "In2.Cu"), net="N0")])
    with pytest.raises(CompileError) as e:
        PCBCompiler().compile(ir, ctx)
    assert str(e.value) == ("via #0 (N0): layers ['F.Cu', 'In2.Cu'] without a via kind - KiCad builds a via without a type token as a through via F.Cu-B.Cu "
                            "(kicad-cli 10.0.6), so the board would contradict the IR")
    ir.pcb.vias = [_via(12.0, 15.0, ("In3.Cu", "B.Cu"), "micro", net="N0")]  # legal by name, not on a 6-layer board
    with pytest.raises(CompileError, match=r"via #0 \(N0\): micro via In3.Cu-B.Cu: a micro via joins two adjacent copper layers"):
        PCBCompiler().compile(ir, ctx)
    ir.pcb.vias = [Via.model_construct(net="N0", x_mm=12.0, y_mm=15.0, drill_mm=0.3, diameter_mm=0.6, layers=("F.Cu", "B.Cu"), kind="blind", provenance=P)]
    with pytest.raises(CompileError, match="via #0 \\(N0\\): blind via F.Cu-B.Cu: a blind via has exactly one outer end"):
        PCBCompiler().compile(ir, ctx)  # a model the validator never saw is checked again
    ir.pcb.vias = [_via(12.0, 15.0, ("F.Cu", "In7.Cu"), "blind", net="N0")]
    with pytest.raises(CompileError, match="not a copper layer of this board"):
        PCBCompiler().compile(ir, ctx)
    ir.pcb.vias = [_via(12.0, 15.0, ("B.Cu", "F.Cu"), net="N0")]  # a kind-less via on the outer pair, either order, compiles as before
    assert PCBCompiler().compile(ir, ctx).path


class _FakeDrill(KicadCli):
    """A kicad-cli stand-in whose drill export writes the named files (and nothing else)."""

    def __init__(self, writes: list[str]) -> None:
        super().__init__(binary="fake-kicad-cli")
        self.writes = writes

    def _run(self, args, timeout=600, ok_codes=(0, 5)):
        out = Path(args[args.index("-o") + 1])
        for name in self.writes:
            (out / name).write_text("M48\n", encoding="utf-8")


def test_export_drill_requires_exactly_the_spans_the_ir_names(tmp_path: Path):
    pcb = tmp_path / "b.kicad_pcb"
    out = tmp_path / "drill"
    out.mkdir()
    (out / "b-front-in2.drl").write_text("stale", encoding="utf-8")  # an earlier run's span file is removed first
    assert _FakeDrill(["b.drl"]).export_drill(pcb, out) == [out / "b.drl"] and not (out / "b-front-in2.drl").exists()
    files = _FakeDrill(["b.drl", "b-in2-in3.drl", "b-front-in2.drl"]).export_drill(pcb, out, ["front-in2", "in2-in3"])
    assert files == [out / "b.drl", out / "b-front-in2.drl", out / "b-in2-in3.drl"]
    with pytest.raises(ToolExecutionError, match="drill export produced no b-in3-back.drl for the in3-back via span"):
        _FakeDrill(["b.drl", "b-front-in2.drl"]).export_drill(pcb, out, ["front-in2", "in3-back"])
    with pytest.raises(ToolExecutionError, match=r"drill export wrote \['b-in4-back.drl'\] beyond the via span\(s\) the IR names"):
        _FakeDrill(["b.drl", "b-front-in2.drl", "b-in4-back.drl"]).export_drill(pcb, out, ["front-in2"])


@needs_kicad
def test_kicad_keeps_the_tokens_and_writes_one_drill_file_per_span(tmp_path: Path):
    """kicad-cli ``pcb upgrade`` (load and save) keeps every token, span and the setup; the drill export returns the pair files in order."""
    lib = synthetic_library(tmp_path / "kicad")
    ir = _six_layer_ir(tmp_path, lib, [v.model_copy() for v in TYPED])
    ctx = CompileContext(workdir=tmp_path / "out", tools={"kicad_library": lib, "kicad_cli": kicad})
    ref = PCBCompiler().compile(ir, ctx)
    path = Path(ref.path)
    up = tmp_path / "up" / path.name
    up.parent.mkdir()
    shutil.copy(path, up)
    kicad._run(["pcb", "upgrade", "--force", str(up)], ok_codes=(0,))
    before, after = sexpr.parse_file(path), sexpr.parse_file(up)
    by_uuid = {u: (t, layers) for t, layers, u in _board_vias(after)}
    assert [by_uuid[u] for _, _, u in _board_vias(before)] == EXPECTED_VIAS  # KiCad re-sorts items on save; matched by uuid
    setup = lambda tree: [sexpr.head(c) for c in sexpr.find(tree, "setup")[1:] if isinstance(c, list)]  # noqa: E731
    assert setup(after) == setup(before)  # no board setup flag added for blind / buried / micro vias
    ir.artifacts[ArtifactKind.PCB] = ref
    art = DrillExporter().compile(ir, ctx)
    stem = path.stem
    assert [Path(f).name for f in art.files] == [f"{stem}.drl", *(f"{stem}-{p}.drl" for p in ["front-in1", "front-in2", "in2-in3", "in3-back", "in4-back"])]
    assert art.generator_version.startswith(f"{TYPED_VIA_VERSION}/kicad-cli ")
    heads = {Path(f).name: next(line for line in Path(f).read_text(encoding="utf-8").splitlines() if "TF.FileFunction" in line) for f in art.files}
    assert heads[f"{stem}.drl"].endswith("MixedPlating,1,6") and heads[f"{stem}-in2-in3.drl"].endswith("Plated,3,4,Buried")
    assert heads[f"{stem}-front-in1.drl"].endswith("Plated,1,2,Blind")  # the two micro vias F.Cu-In1.Cu share one file
    # a board without typed vias: exactly <stem>.drl, the exporter's old version
    plain = _six_layer_ir(tmp_path, lib, [TYPED[0].model_copy()])
    plain.artifacts[ArtifactKind.PCB] = PCBCompiler().compile(plain, ctx)
    art = DrillExporter().compile(plain, ctx)
    assert [Path(f).name for f in art.files] == [f"{stem}.drl"] and art.generator_version.startswith(f"{DrillExporter.version}/kicad-cli ")


# --------------------------------------------------------------------------- pcb.routing and pcb.keepout


@pytest.fixture
def lib(tmp_path: Path) -> KicadLibrary:
    return fixture_library(tmp_path / "kicad")


def _layout_ir(tmp_path: Path, lib: KicadLibrary, inner_layer: str, via_a: Via, via_b: Via) -> CircuitIR:
    """SMD pads R1.1 (3, 4) and R2.1 (11, 4) on F.Cu, net N: F.Cu to via A (5, 4), ``inner_layer`` to via B (9, 4), F.Cu to R2; a 6-layer board."""
    ir = board_ir(tmp_path, lib, [("R1", "SMD1", 3.0, 4.0), ("R2", "SMD1", 11.0, 4.0), ("R3", "SMD1", 8.0, 7.0)], {"N": [("R1", "1"), ("R2", "1")], "M": [("R3", "1")]}, (14.0, 8.0))
    ir.pcb.layers = [Layer(name=n, kind="signal") for n in SIX]
    ir.pcb.tracks = [_track("N", (3.0, 4.0), (5.0, 4.0)), _track("N", (5.0, 4.0), (9.0, 4.0), inner_layer), _track("N", (9.0, 4.0), (11.0, 4.0))]
    ir.pcb.vias = [via_a, via_b]
    return ir


def _routing(ir: CircuitIR, tmp_path: Path, lib: KicadLibrary) -> dict:
    results = default_registry.get(TOOL_ID).validate(ir, ValidationContext(workdir=tmp_path, tools={"kicad_library": lib}))
    return {r.check_id: r for r in results}


def test_a_typed_via_joins_only_the_layers_of_its_span(tmp_path: Path, lib: KicadLibrary):
    # blind F.Cu-In2.Cu does not reach an In3.Cu track: R2 is cut off (via B, blind F.Cu-In3.Cu, does reach it)
    ir = _layout_ir(tmp_path, lib, "In3.Cu", _via(5.0, 4.0, ("F.Cu", "In2.Cu"), "blind"), _via(9.0, 4.0, ("F.Cu", "In3.Cu"), "blind"))
    c = _routing(ir, tmp_path, lib)[CONNECTIVITY_CHECK]
    row = next(r for r in c.details["nets"] if r["net"] == "N")
    assert c.status is S.FAIL and row["status"] == "FAIL" and row["unconnected"] == ["R2.1"]
    assert all(r.tool_version == TYPED_VIA_TOOL_VERSION == "0.2" for r in _routing(ir, tmp_path, lib).values())
    # the same copper on In2.Cu: via A ends there and via B (F.Cu-In3.Cu) crosses it - one copper set
    ir = _layout_ir(tmp_path, lib, "In2.Cu", _via(5.0, 4.0, ("F.Cu", "In2.Cu"), "blind"), _via(9.0, 4.0, ("F.Cu", "In3.Cu"), "blind"))
    assert next(r for r in _routing(ir, tmp_path, lib)[CONNECTIVITY_CHECK].details["nets"] if r["net"] == "N")["status"] == "PASS"
    # through vias reach every layer, as before - and an IR without a typed via keeps 0.1
    ir = _layout_ir(tmp_path, lib, "In3.Cu", _via(5.0, 4.0), _via(9.0, 4.0))
    checks = _routing(ir, tmp_path, lib)
    assert next(r for r in checks[CONNECTIVITY_CHECK].details["nets"] if r["net"] == "N")["status"] == "PASS"
    assert all(r.tool_version == TOOL_VERSION == "0.1" for r in checks.values())


def test_a_typed_via_is_compared_only_on_its_span(tmp_path: Path, lib: KicadLibrary):
    """A foreign In3.Cu track through via A's position: a short for a through via, nothing for a blind F.Cu-In2.Cu via."""
    ir = _layout_ir(tmp_path, lib, "In2.Cu", _via(5.0, 4.0, ("F.Cu", "In2.Cu"), "blind"), _via(9.0, 4.0, ("F.Cu", "In3.Cu"), "blind"))
    ir.pcb.tracks.append(_track("M", (5.0, 1.0), (5.0, 7.0), "In3.Cu"))
    ir.pcb.manufacturing = ManufacturingConstraints(min_clearance_mm=assumption(0.2, note="a limit is a limit"))
    c = _routing(ir, tmp_path, lib)[CLEARANCE_CHECK]
    assert c.status is S.PASS and c.details["violations"] == [] and c.details["not_compared"] == [*NOT_COMPARED, TYPED_VIAS_NOT_COMPARED]
    ir.pcb.vias[0] = _via(5.0, 4.0)
    c = _routing(ir, tmp_path, lib)[CLEARANCE_CHECK]
    assert c.status is S.FAIL and [(v["a"], v["b"], v["layer"]) for v in c.details["violations"]] == [("track[3:M]", "via[0:N]", "In3.Cu")]
    ir.pcb.vias[1] = _via(9.0, 4.0)
    assert _routing(ir, tmp_path, lib)[CLEARANCE_CHECK].details["not_compared"] == NOT_COMPARED  # no typed via: the old list


def test_a_span_the_board_cannot_hold_is_a_fail_row_and_joins_nothing(tmp_path: Path, lib: KicadLibrary):
    ir = _layout_ir(tmp_path, lib, "In2.Cu", _via(5.0, 4.0, ("F.Cu", "In7.Cu"), "blind"), _via(9.0, 4.0, ("In3.Cu", "B.Cu"), "micro"))
    checks = _routing(ir, tmp_path, lib)
    c = checks[CONNECTIVITY_CHECK]
    assert c.status is S.FAIL and [r["message"] for r in c.details["items"]] == [
        "via[0:N] is a blind via from F.Cu to In7.Cu: 'In7.Cu' is not a copper layer of the board (the compiler refuses the same)",
        f"via[1:N] is a micro via from In3.Cu to B.Cu: {MICRO_VIA_ADJACENT} (the compiler refuses the same)",
    ]
    assert next(r for r in c.details["nets"] if r["net"] == "N")["unconnected"] == ["R2.1"]  # neither via joins anything
    assert checks[CLEARANCE_CHECK].status is S.FAIL and "2 track(s) / via(s) that cannot be copper" in checks[CLEARANCE_CHECK].message


def test_a_typed_via_reaches_a_plane_only_on_its_span(tmp_path: Path, lib: KicadLibrary):
    """Net N poured on In4.Cu, no inner track: R1's blind F.Cu-In2.Cu via cannot meet the plane (FAIL), a through via can (NOT_VERIFIED)."""
    ir = _layout_ir(tmp_path, lib, "In2.Cu", _via(5.0, 4.0, ("F.Cu", "In2.Cu"), "blind"), _via(9.0, 4.0))
    ir.pcb.tracks = [ir.pcb.tracks[0], ir.pcb.tracks[2]]
    ir.pcb.zones = [Zone(net="N", layer="In4.Cu", polygon=[(0.5, 0.5), (13.5, 0.5), (13.5, 7.5), (0.5, 7.5)], provenance=P)]
    row = next(r for r in _routing(ir, tmp_path, lib)[CONNECTIVITY_CHECK].details["nets"] if r["net"] == "N")
    assert row["status"] == "FAIL" and row["unconnected"] == ["R1.1"] and "only a via or a through-hole barrel reaches an inner plane" in row["message"]
    ir.pcb.vias[0] = _via(5.0, 4.0, ("F.Cu", "In4.Cu"), "blind")  # a span that holds the plane layer reaches it
    row = next(r for r in _routing(ir, tmp_path, lib)[CONNECTIVITY_CHECK].details["nets"] if r["net"] == "N")
    assert row["status"] == "NOT_VERIFIED" and row.get("plane") is True


def test_a_keepout_judges_a_typed_via_only_where_its_span_meets_the_layers(tmp_path: Path, lib: KicadLibrary):
    ir = _layout_ir(tmp_path, lib, "In2.Cu", _via(5.0, 4.0, ("F.Cu", "In2.Cu"), "blind"), _via(9.0, 4.0, ("F.Cu", "In3.Cu"), "blind"))
    ir.pcb.keepouts = [Keepout(id="K", layers=["In4.Cu"], rect=user_requirement([4.0, 3.0, 2.0, 2.0], "mm"), forbids=["vias"], reason="test area", provenance=P)]
    ctx = ValidationContext(workdir=tmp_path, tools={"kicad_library": lib})
    [res] = default_registry.get(KEEPOUT_CHECK).validate(ir, ctx)
    assert res.status is S.PASS and res.details["keepouts"][0]["compared"] == 0 and res.tool_version == TYPED_VIA_TOOL_VERSION
    ir.pcb.vias[0] = _via(5.0, 4.0, ("In3.Cu", "B.Cu"), "blind")  # its span meets In4.Cu
    [res] = default_registry.get(KEEPOUT_CHECK).validate(ir, ctx)
    assert res.status is S.FAIL and "keep-out K forbids vias; via[0:N] (N) reaches into it" in res.message
    ir.pcb.vias = [_via(5.0, 4.0), _via(9.0, 4.0)]  # a through via is on every layer, as before
    [res] = default_registry.get(KEEPOUT_CHECK).validate(ir, ctx)
    assert res.status is S.FAIL and res.tool_version == "0.1"


# --------------------------------------------------------------------------- mfg.capability, SI, 3D


def test_the_capability_row_appears_only_with_typed_vias(tmp_path: Path):
    lib = synthetic_library(tmp_path / "kicad")
    ir = _six_layer_ir(tmp_path, lib, [TYPED[0].model_copy()])
    res = check_capability(ir)
    assert "via_process" not in [r["limit"] for r in res.details["compared"]] and res.tool_version == CAPABILITY_VERSION == "0.1"
    ir.pcb.vias = [v.model_copy() for v in TYPED]
    res = check_capability(ir)
    [row] = [r for r in res.details["compared"] if r["limit"] == "via_process"]
    assert row["status"] == "NOT_VERIFIED" and res.status is S.NOT_VERIFIED and res.tool_version == "0.2"
    assert row["message"] == (
        "6 blind/buried/micro via(s) on span(s) ['blind F.Cu-In2.Cu', 'blind In3.Cu-B.Cu', 'buried In2.Cu-In3.Cu', 'micro F.Cu-In1.Cu', 'micro In4.Cu-B.Cu']: "
        "no grounded fab capability states controlled-depth, laser or sequential-lamination drilling; the spans are a generic stack's stated choice, not a fab's"
    )
    drill = next(r for r in res.details["compared"] if r["limit"] == "min_via_drill_mm")
    assert drill["status"] == "NOT_VERIFIED" and "no min_via_drill_mm" in drill["message"]  # still every via's row, typed or not


def _six_stack() -> Stackup:
    """A 6-layer test stack (numbers stated for the test, not a fab's): 35 / 17.5 um copper, dielectrics 0.1 / 0.2 / 0.86 / 0.2 / 0.1 mm."""
    copper = [StackupCopper(name=n, thickness_um=user_requirement(35.0 if n in ("F.Cu", "B.Cu") else 17.5, "um")) for n in SIX]
    diel = [StackupDielectric(kind=k, thickness_mm=user_requirement(h, "mm"), er=user_requirement(er))
            for k, h, er in ((DielectricKind.PREPREG, 0.1, 4.4), (DielectricKind.PREPREG, 0.2, 4.4), (DielectricKind.CORE, 0.86, 4.6),
                             (DielectricKind.PREPREG, 0.2, 4.4), (DielectricKind.PREPREG, 0.1, 4.4))]
    return Stackup(copper=copper, dielectrics=diel, provenance=P)


def test_si_counts_each_vias_own_barrel_and_extracts_no_path_through_a_typed_via(tmp_path: Path, lib: KicadLibrary):
    ir = _layout_ir(tmp_path, lib, "In2.Cu", _via(5.0, 4.0, ("F.Cu", "In2.Cu"), "blind"), _via(9.0, 4.0))
    stack = _six_stack()
    ir.pcb.stackup = stack
    ir.pcb.tracks.append(_track("M", (8.0, 7.0), (8.0, 5.5)))
    ir.pcb.vias.append(_via(8.0, 5.5, net="M"))
    m = measure_nets(ir)
    n, other = m["N"], m["M"]
    t_pd = propagation_delay(4.6)  # the whole stack's er_max bounds every barrel
    assert n.typed_vias == 1 and n.via_barrels_mm == stack.span_mm("F.Cu", "In2.Cu") + stack.span_mm("F.Cu", "B.Cu")
    assert n.length_mm == n.track_length_mm + n.via_barrels_mm and n.via_barrels_delay_s == n.via_barrels_mm / 1000.0 * t_pd
    assert n.summary()["via_barrels_mm"] == round(n.via_barrels_mm, 6) and n.summary()["typed_vias"] == 1
    # a net with through vias only: the old measure, the same float bits
    assert other.via_barrels_mm is None and other.length_mm == other.track_length_mm + other.vias * other.via_length_mm
    assert other.via_length_mm == stack.span_mm("F.Cu", "B.Cu") and "via_barrels_mm" not in other.summary()
    # the path model does not split a typed via's barrel
    assert longest_path("N", ir.pcb.tracks[:1], ir.pcb.vias[:1], {"A": [], "B": []}, stack, {}, t_pd) == (None, TYPED_VIA_NOT_SPLIT)
    assert TYPED_VIA_NOT_SPLIT == "a blind/buried/micro via's barrel is not split in this path model"


def test_si_results_and_promotions_on_a_board_with_a_typed_via_carry_their_own_version(tmp_path: Path, lib: KicadLibrary):
    """Review finding: the typed-via measure (each via its own barrel, no path through a typed via) is a mechanism of its own, so what it
    measures is stamped si 0.2; an IR without a typed via keeps si 0.1 (an older build reading the same IR measures every via as through)."""
    from ai_eda.tools.si.measure import SI_TYPED_VIA_VERSION, SI_VERSION, si_version
    from ai_eda.tools.si.promote import promote
    from ai_eda.validation.si import si_results
    from tests.test_si_checks import default_classes, long_board, si_of, track

    ir = long_board(tmp_path, lib, 4)
    ir.si = si_of(*default_classes())
    ir.pcb.tracks = [track("LONG", (5.0, 5.0), (95.0, 5.0)), track("SHORT", (5.0, 15.0), (15.0, 15.0))]
    assert (SI_VERSION, SI_TYPED_VIA_VERSION) == ("0.1", "0.2") and si_version(ir) == SI_VERSION
    assert {r.tool_version for r in si_results(ir, lib) if r.tool == "si"} == {SI_VERSION} and promote(ir)[1][0].provenance.tool_version == SI_VERSION
    ir.pcb.vias = [_via(20.0, 15.0, ("F.Cu", "In1.Cu"), "micro", net="GND", d=0.3, drill=0.1)]
    assert si_version(ir) == SI_TYPED_VIA_VERSION
    results = si_results(ir, lib)
    # si.crosstalk carries its own tool and version, which the typed-via stamp never touches
    assert results and {(r.tool, r.tool_version) for r in results} == {("si", SI_TYPED_VIA_VERSION), ("si.crosstalk", "0.2")}
    assert promote(ir)[1][0].provenance.tool_version == SI_TYPED_VIA_VERSION


def test_a_micro_via_joining_the_outer_layers_has_no_drill_pair_file(tmp_path: Path):
    """Review finding: kicad-cli writes every F.Cu-B.Cu hole, whatever its via type, into ``<stem>.drl`` (measured below), so a micro via
    F.Cu-B.Cu (legal on a 2-layer board) names no pair file and the export is stamped as any board without span files."""
    lib = synthetic_library(tmp_path / "kicad")
    ir = parts_ir(tmp_path, lib, ("R1", "R2"))
    ir.pcb = PCBDesign(outline=BoardOutline(width_mm=40.0, height_mm=20.0), layers=[Layer(name=n, kind="signal") for n in ("F.Cu", "B.Cu")],
                       vias=[_via(20.0, 15.0, ("F.Cu", "B.Cu"), "micro", net="N0", d=0.3, drill=0.1)],
                       placements=[Placement(component_ref="R1", x_mm=8.0, y_mm=10.0, provenance=P), Placement(component_ref="R2", x_mm=32.0, y_mm=10.0, provenance=P)])
    assert via_span_problem("micro", "F.Cu", "B.Cu", ["F.Cu", "B.Cu"]) is None and drill_pairs(ir) == []
    assert drill_pairs(_six_layer_ir(tmp_path, lib, [_via(12.0, 15.0, ("F.Cu", "In1.Cu"), "micro", net="N0")])) == ["front-in1"]  # any other span: its file


@needs_kicad
def test_kicad_writes_a_micro_via_joining_the_outer_layers_into_the_merged_drill_file(tmp_path: Path):
    lib = synthetic_library(tmp_path / "kicad")
    ir = parts_ir(tmp_path, lib, ("R1", "R2"))
    ir.pcb = PCBDesign(outline=BoardOutline(width_mm=40.0, height_mm=20.0), layers=[Layer(name=n, kind="signal") for n in ("F.Cu", "B.Cu")],
                       vias=[_via(20.0, 15.0, ("F.Cu", "B.Cu"), "micro", net="N0", d=0.3, drill=0.1)],
                       placements=[Placement(component_ref="R1", x_mm=8.0, y_mm=10.0, provenance=P), Placement(component_ref="R2", x_mm=32.0, y_mm=10.0, provenance=P)])
    ctx = CompileContext(workdir=tmp_path / "out", tools={"kicad_library": lib, "kicad_cli": kicad})
    ref = PCBCompiler().compile(ir, ctx)
    assert any(line.strip().startswith("(via micro") for line in Path(ref.path).read_text(encoding="utf-8").splitlines())
    ir.artifacts[ArtifactKind.PCB] = ref
    art = DrillExporter().compile(ir, ctx)
    stem = Path(ref.path).stem
    assert [Path(f).name for f in art.files] == [f"{stem}.drl"] and art.generator_version.startswith(f"{DrillExporter.version}/kicad-cli ")
    text = Path(art.files[0]).read_text(encoding="utf-8")
    assert "TF.FileFunction,MixedPlating,1,2" in text and "T1C0.100" in text  # the micro via's 0.1 mm hole, in the merged file


def test_the_3d_scene_says_typed_vias_are_drawn_full_depth(tmp_path: Path):
    lib = synthetic_library(tmp_path / "kicad")
    ir = _six_layer_ir(tmp_path, lib, [TYPED[0].model_copy()])
    assert not any("블라인드" in n for n in build_scene(ir, lib, model_dir=None).notes)
    ir.pcb.vias = [v.model_copy() for v in TYPED]
    notes = build_scene(ir, lib, model_dir=None).notes
    assert "블라인드/매립/마이크로 비아 6개는 보드 전체 깊이의 구멍으로 그림 (층 범위는 그리지 않음)" in notes
