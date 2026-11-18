"""The layer-escalation stack family (design.stackup.escalation 0.1) and the router's stack input type.

What is proven here, without KiCad or the network:

* The ladder is the project owner's (2, 4, 6, 8, 10, 12), checked at import;
  an odd count and a count above 12 are refused with the reason.
* Each escalation stack (4, 6, 8, 10, 12 copper layers) is 1.6 mm, has 2, 4,
  5, 6, 8 signal layers, and obeys the sourced stack rules, each checked here
  independently of the module's own checker: every signal layer next to a
  plane, at most two signal layers between planes, one per outer side, two
  adjacent signal layers with orthogonal preferred directions, a
  mirror-symmetric build.
* The via table per count is pinned, equals what the one generating rule
  gives, and every run is buildable: a drilled run goes through an even
  number of copper layers and never ends on a core's top side nor starts on
  its bottom side (Eurocircuits), any two runs are nested or disjoint, a micro
  via crosses exactly one dielectric and is at most 0.25 mm deep; a 0.3 mm
  drill keeps the through via within 8:1 and every micro via within 1:1.
* ``escalation_stackup`` stamps every number as an assumption with the owner's
  rule; ``RouterStack.from_stackup`` reads the 8-layer stack.
* The 2- and 4-layer template stacks, their texts and ``LAYER_COUNT_OPTIONS``
  are byte-identical to c270797.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from ai_eda.design.inputs import LAYER_COUNT_OPTIONS, LayerCountInput
from ai_eda.design.stackup import (
    ESCALATION_STACKS,
    ESCALATION_TOOL,
    ESCALATION_VERSION,
    GENERIC_NOTE,
    GENERIC_STACKS,
    LAYER_ESCALATION,
    STACKUP_VERSION,
    _check_ladder,
    board_layers,
    escalation_description,
    escalation_refusal,
    escalation_stack,
    escalation_stackup,
    generic_stackup,
    layer_count_choice_text,
    legal_spans,
    plane_zones,
    stack_description,
    stack_problems,
    stackup_choices,
    via_drill_problems,
    via_structures,
)
from ai_eda.ir import BoardOutline, DielectricKind, ProvenanceKind, Stackup
from ai_eda.tools.routing import RouterStack, ViaType
from ai_eda.tools.routing.stack import THROUGH

COUNTS = (4, 6, 8, 10, 12)
RULE_NOTE = " - chosen by the project owner's layer-escalation rule, not shown in a confirm_design table"

#: the via structures besides the through via, per copper count (the one rule's output, pinned)
VIAS = {
    4: [("micro", "F.Cu", "In1.Cu"), ("micro", "In2.Cu", "B.Cu")],
    6: [("micro", "F.Cu", "In1.Cu"), ("micro", "In4.Cu", "B.Cu"), ("blind", "F.Cu", "In3.Cu"), ("buried", "In2.Cu", "In3.Cu")],
    8: [("micro", "F.Cu", "In1.Cu"), ("micro", "In6.Cu", "B.Cu"), ("blind", "F.Cu", "In3.Cu"), ("blind", "In4.Cu", "B.Cu"), ("buried", "In2.Cu", "In3.Cu")],
    10: [("micro", "F.Cu", "In1.Cu"), ("micro", "In8.Cu", "B.Cu"), ("blind", "F.Cu", "In3.Cu"), ("blind", "In6.Cu", "B.Cu"),
         ("buried", "In2.Cu", "In3.Cu"), ("buried", "In6.Cu", "In7.Cu")],
    12: [("micro", "F.Cu", "In1.Cu"), ("micro", "In10.Cu", "B.Cu"), ("blind", "F.Cu", "In3.Cu"), ("blind", "In8.Cu", "B.Cu"),
         ("buried", "In2.Cu", "In3.Cu"), ("buried", "In4.Cu", "In7.Cu"), ("buried", "In8.Cu", "In9.Cu")],
}
#: the layer roles top to bottom (S signal, G ground plane, P power plane)
ROLES = {4: "SGPS", 6: "SGSSPS", 8: "SGSSPSGS", 10: "SGSSPGSSGS", 12: "SGSSPSSGSSPS"}


def _stack(n: int, power: str | None = "+3V3") -> Stackup:
    return escalation_stackup(n, ground_net="GND", power_net=power)


def _roles(s: Stackup) -> str:
    return "".join("S" if c.plane_net is None else ("G" if c.plane_net.value == "GND" else "P") for c in s.copper)


# --------------------------------------------------------------------------- the ladder


def test_the_ladder_is_the_owners_and_refuses_odd_and_too_many_layers():
    assert LAYER_ESCALATION == (2, 4, 6, 8, 10, 12)
    assert all(escalation_refusal(n) is None for n in LAYER_ESCALATION)
    assert escalation_refusal(5) == ("5 copper layers: refused - KiCad builds only even copper layer counts (kicad-cli 10.0.6 refused a 5-layer "
                                     "board: '5 is not a valid layer number'; compilers/pcb.py refuses odd counts)")
    assert escalation_refusal(14) == "14 copper layers: refused - no generic stack for 14 layers (the project owner's ladder stops at 12)"
    assert escalation_refusal(7).startswith("7 copper layers: refused - KiCad builds only even copper layer counts")
    assert escalation_refusal(0) == "0 copper layers: refused - a board needs at least 2 copper layers"
    assert escalation_refusal(True) == "True copper layers: refused - not a whole copper layer count" and "not a whole" in escalation_refusal("4")
    with pytest.raises(ValueError, match="5 copper layers: refused - KiCad builds only even"):
        escalation_stack(5)
    with pytest.raises(ValueError, match="first rung is the board as it is"):
        escalation_stack(2)
    with pytest.raises(ValueError, match="no generic stack for 14 layers"):
        escalation_stackup(14, ground_net="GND")
    assert escalation_stack(4) is GENERIC_STACKS[4] and escalation_stack(8) is ESCALATION_STACKS[8] and sorted(ESCALATION_STACKS) == [6, 8, 10, 12]
    # the import-time check names the problem; an odd count is left to the run-time refusal
    for bad, match in (((), "non-empty tuple"), ([2, 4], "non-empty tuple"), ((2, 4.0), "plain ints"), ((2, True), "plain ints"),
                       ((4, 6), "must start at 2"), ((2, 6, 4), "strictly ascending"), ((2, 4, 4), "strictly ascending")):
        with pytest.raises(ValueError, match=match):
            _check_ladder(bad)
    _check_ladder((2, 4, 5, 8, 10, 12))


# --------------------------------------------------------------------------- the stacks and their sourced rules


def test_each_escalation_stack_is_1_6_mm_with_the_stated_signal_layer_counts():
    assert len(GENERIC_STACKS[2].copper) == 2 and not GENERIC_STACKS[2].planes  # the 2-layer rung: both layers signal, no plane
    counts = [2] + [len(_stack(n).signal_layers()) for n in COUNTS]
    assert counts == [2, 2, 4, 5, 6, 8]
    for n in COUNTS:
        s = _stack(n)
        assert s.layer_count == n and abs(s.board_thickness_mm() - 1.6) < 1e-9 and abs(escalation_stack(n).thickness_mm - 1.6) < 1e-9
        assert _roles(s) == ROLES[n]
        g = escalation_stack(n)
        assert g.title == f"{n}-layer 1.6 mm FR-4" and g.roles_source
        assert {c.thickness_um.value for c in s.copper[1:-1]} == {17.5} and s.copper[0].thickness_um.value == s.copper[-1].thickness_um.value == 35.0
        assert {(float(d.er.value), float(d.er_frequency_hz.value)) for d in s.dielectrics} == {(4.5, 1.0e6)}
    assert "TI SPRAAR7J" in escalation_stack(8).roles_source and "Fig. 6" in escalation_stack(12).roles_source


def test_every_signal_layer_is_next_to_a_plane_with_at_most_two_between_planes_and_one_per_outer_side():
    for n in COUNTS:
        roles = _roles(_stack(n))
        for i, r in enumerate(roles):
            if r == "S":
                assert (i > 0 and roles[i - 1] != "S") or (i < n - 1 and roles[i + 1] != "S"), (n, i)
        runs = roles.replace("G", "|").replace("P", "|").split("|")
        assert len(runs[0]) == 1 and len(runs[-1]) == 1 and max(len(r) for r in runs) <= 2, (n, roles)


def test_adjacent_signal_layers_have_orthogonal_preferred_directions():
    for n in COUNTS:
        g = escalation_stack(n)
        s = _stack(n)
        assert [layer for layer, _ in g.directions] == s.signal_layers()
        assert {d for _, d in g.directions} <= {"horizontal", "vertical"}
        direction = dict(g.directions)
        names, roles = s.copper_names(), _roles(s)
        pairs = [(names[i], names[i + 1]) for i in range(n - 1) if roles[i] == roles[i + 1] == "S"]
        assert pairs and all(direction[a] != direction[b] for a, b in pairs) if n > 4 else not pairs


def test_the_build_is_mirror_symmetric():
    for n in COUNTS:
        s = _stack(n)
        assert [c.thickness_um.value for c in s.copper] == [c.thickness_um.value for c in reversed(s.copper)]
        layers = [(d.kind, d.thickness_mm.value, d.er.value, d.er_frequency_hz.value) for d in s.dielectrics]
        assert layers == layers[::-1], n
    # the outermost dielectric on each side is the 0.1 mm build-up prepreg above 4 layers
    assert all(_stack(n).dielectrics[0].thickness_mm.value == 0.1 and _stack(n).dielectrics[0].kind is DielectricKind.PREPREG for n in COUNTS[1:])


def test_the_via_table_is_pinned_and_generated_by_the_one_rule():
    for n in COUNTS:
        g = escalation_stack(n)
        assert list(g.vias) == VIAS[n] and via_structures(g) == g.vias, n
        assert stack_problems(g) == [] and via_drill_problems(g, 0.3) == []
    assert via_structures(GENERIC_STACKS[2]) == ()
    # the 6-layer B.Cu-side blind run is legal alone but overlaps F.Cu-In3.Cu without nesting, so the rule drops it
    six = escalation_stack(6)
    assert ("In2.Cu", "B.Cu") in legal_spans(six) and ("F.Cu", "In2.Cu") not in legal_spans(six)
    assert legal_spans(six) == [("F.Cu", "In1.Cu"), ("F.Cu", "In3.Cu"), ("F.Cu", "B.Cu"), ("In1.Cu", "In4.Cu"), ("In2.Cu", "In3.Cu"),
                                ("In2.Cu", "B.Cu"), ("In4.Cu", "B.Cu")]


def test_every_via_run_is_buildable_and_laminar():
    """Checked on the IR stack itself, not through the module's helpers."""
    for n in COUNTS:
        s = _stack(n)
        names = s.copper_names()
        core = [d.kind is DielectricKind.CORE for d in s.dielectrics]
        core_top = {i for i, c in enumerate(core) if c}
        core_bottom = {i + 1 for i, c in enumerate(core) if c}
        spans = [("through", 0, n - 1)] + [(k, names.index(t), names.index(b)) for k, t, b in VIAS[n]]
        for kind, a, b in spans:
            outer = (a == 0) + (b == n - 1)
            if kind == "micro":
                assert b == a + 1, (n, kind, a, b)  # exactly one dielectric
                depth = float(s.dielectrics[a].thickness_mm.value) + max(float(s.copper[a].thickness_um.value), float(s.copper[b].thickness_um.value)) / 1000.0
                assert depth <= 0.25 and depth / 0.3 <= 1.0
            else:
                assert (b - a + 1) % 2 == 0 and a not in core_bottom and b not in core_top, (n, kind, a, b)
                assert outer == {"through": 2, "blind": 1, "buried": 0}[kind]
        for i, (_, a, b) in enumerate(spans):
            for _, c, d in spans[i + 1:]:
                sa, sc = set(range(a, b + 1)), set(range(c, d + 1))
                assert not sa & sc or sa <= sc or sc <= sa, (n, (a, b), (c, d))
        # signal via types (spans holding two or more signal layers): through first
        signal = set(s.signal_layers())
        joined = [[names[i] for i in range(a, b + 1) if names[i] in signal] for _, a, b in spans]
        assert sum(len(j) >= 2 for j in joined) == {4: 1, 6: 3, 8: 4, 10: 5, 12: 6}[n]
    # through-via aspect ratio 1.6 mm : drill at most 8:1, micro depth : drill at most 1:1
    assert via_drill_problems(escalation_stack(12), 0.2) == [] and "above 8:1" in via_drill_problems(escalation_stack(12), 0.19)[0]
    micro = via_drill_problems(escalation_stack(4), 0.2)
    assert len(micro) == 2 and micro[0].startswith("micro via F.Cu-In1.Cu: aspect ratio 1.18:1 (0.235 mm deep, 0.2 mm drill) is above 1:1")
    for bad in (0.0, -0.3, float("nan"), True, "0.3"):
        with pytest.raises(ValueError, match="finite number > 0"):
            via_drill_problems(escalation_stack(6), bad)


def test_the_stack_checker_names_each_broken_rule():
    six, eight = escalation_stack(6), escalation_stack(8)

    def problems(g, **change) -> str:
        return " | ".join(stack_problems(dataclasses.replace(g, **change)))

    assert "In3.Cu is not adjacent to a reference plane" in problems(six, planes=(("In1.Cu", "ground"),))
    assert "4 signal layers ['In2.Cu', 'In3.Cu', 'In4.Cu', 'In5.Cu'] between two planes - at most two" in problems(
        eight, planes=(("In1.Cu", "ground"), ("In6.Cu", "ground")), directions=())
    assert "each outer side carries exactly one signal layer" in problems(six, planes=(("In2.Cu", "ground"), ("In4.Cu", "power")))
    assert "share the preferred direction horizontal" in problems(
        six, directions=(("F.Cu", "horizontal"), ("In2.Cu", "horizontal"), ("In3.Cu", "horizontal"), ("B.Cu", "vertical")))
    assert "must name the signal layers" in problems(six, directions=(("F.Cu", "horizontal"),))
    d = list(eight.dielectrics)
    d[1] = (DielectricKind.PREPREG, 0.25, 4.5, 1.0e6)
    assert "not mirror-symmetric: dielectric 1" in problems(eight, dielectrics=tuple(d))
    for extra, expect in (
        (("blind", "In2.Cu", "B.Cu"), "overlap without nesting"),
        (("blind", "F.Cu", "In2.Cu"), "goes through 3 copper layers - a via run must go through an even number"),
        (("buried", "In3.Cu", "In4.Cu"), "starts on In3.Cu, the bottom side of a core"),
        (("buried", "In1.Cu", "In2.Cu"), "ends on In2.Cu, the top side of a core"),
        (("blind", "In2.Cu", "In3.Cu"), "a blind via has exactly one outer end"),
        (("buried", "F.Cu", "In3.Cu"), "a buried via has no outer end"),
        (("micro", "F.Cu", "In2.Cu"), "a micro via crosses exactly one dielectric"),
        (("through", "F.Cu", "B.Cu"), "is implicit"),
        (("micro", "F.Cu", "In1.Cu"), "the span of micro F.Cu-In1.Cu again"),
        (("laser", "F.Cu", "In1.Cu"), "not one of through, blind, buried, micro"),
    ):
        assert expect in problems(six, vias=six.vias + (extra,)), extra
    thick = dataclasses.replace(GENERIC_STACKS[4], dielectrics=((DielectricKind.PREPREG, 0.3, 4.5, 1.0e6), (DielectricKind.CORE, 0.995, 4.5, 1.0e6),
                                                                (DielectricKind.PREPREG, 0.3, 4.5, 1.0e6)))
    assert "a micro via is at most 0.25 mm deep, this one 0.335 mm" in " | ".join(stack_problems(thick))
    assert stack_problems(GENERIC_STACKS[2]) == ["no plane layer: the stack rules are for a stack with reference planes"]


# --------------------------------------------------------------------------- the stackup the escalation applies


def test_the_escalation_stackup_is_an_assumption_chosen_by_the_owners_rule():
    s = _stack(8)
    description = escalation_description(8, "GND", "+3V3")
    assert description.startswith("8-layer 1.6 mm FR-4: copper F.Cu 35 um, In1.Cu 17.5 um") and description.endswith(GENERIC_NOTE)
    assert "planes In1.Cu = GND (ground), In4.Cu = +3V3 (power), In6.Cu = GND (ground)" in description
    assert "signal layers (preferred direction) F.Cu horizontal, In2.Cu vertical, In3.Cu horizontal, In5.Cu vertical, B.Cu horizontal" in description
    assert ("via structures through F.Cu-B.Cu, micro F.Cu-In1.Cu, micro In6.Cu-B.Cu, blind F.Cu-In3.Cu, blind In4.Cu-B.Cu, buried In2.Cu-In3.Cu "
            "(mechanical blind / buried runs follow Eurocircuits' rule") in description
    items = s.traced_items()
    assert len(items) == 8 + 3 + 7 * 3 and all(t.provenance.kind is ProvenanceKind.ASSUMPTION for _, t in items)
    assert {t.provenance.note for _, t in items} == {description + RULE_NOTE}
    assert s.provenance.kind is ProvenanceKind.ASSUMPTION and (s.provenance.tool, s.provenance.tool_version) == (ESCALATION_TOOL, ESCALATION_VERSION)
    assert s.provenance.note.startswith("layer count 8: a placeholder") and escalation_stackup(8, ground_net="GND", note="kept").provenance.note == "kept"
    # without a power net the ground net fills the power planes, and says so
    bare = _stack(10, power=None)
    assert {c.plane_net.value for c in bare.plane_layers()} == {"GND"}
    assert "In4.Cu = GND (a power plane position; the board has no power net)" in bare.copper[4].plane_net.provenance.note
    # deterministic, and the layers / plane zones follow the stack
    assert _stack(8).model_dump(mode="json", context={"view": "design"}) == s.model_dump(mode="json", context={"view": "design"})
    assert [(layer.name, layer.kind) for layer in board_layers(s)][:3] == [("F.Cu", "signal"), ("In1.Cu", "power"), ("In2.Cu", "signal")]
    zones = plane_zones(_stack(12), BoardOutline(width_mm=40.0, height_mm=30.0))
    assert [(z.net, z.layer) for z in zones] == [("GND", "In1.Cu"), ("+3V3", "In4.Cu"), ("GND", "In7.Cu"), ("+3V3", "In10.Cu")]


# --------------------------------------------------------------------------- the router's stack input


def test_router_stack_from_the_8_layer_stack():
    g = escalation_stack(8)
    rs = RouterStack.from_stackup(_stack(8), g.vias, g.title, directions=g.directions)
    assert rs.label == "8-layer 1.6 mm FR-4"
    assert rs.copper == ("F.Cu", "In1.Cu", "In2.Cu", "In3.Cu", "In4.Cu", "In5.Cu", "In6.Cu", "B.Cu")
    assert rs.signal == ("F.Cu", "In2.Cu", "In3.Cu", "In5.Cu", "B.Cu")
    assert rs.planes == (("In1.Cu", "GND"), ("In4.Cu", "+3V3"), ("In6.Cu", "GND"))
    assert rs.vias[0] == THROUGH == ViaType("through", "F.Cu", "B.Cu")
    assert rs.vias[1:] == (ViaType("micro", "F.Cu", "In1.Cu"), ViaType("micro", "In6.Cu", "B.Cu"), ViaType("blind", "F.Cu", "In3.Cu"),
                           ViaType("blind", "In4.Cu", "B.Cu"), ViaType("buried", "In2.Cu", "In3.Cu"))
    assert rs.directions == g.directions and hash(rs) == hash(RouterStack.from_stackup(_stack(8), g.vias, g.title, directions=g.directions))
    with pytest.raises(dataclasses.FrozenInstanceError):
        rs.label = "x"  # type: ignore[misc]
    assert RouterStack.from_stackup(_stack(4), escalation_stack(4).vias, "4").directions == ()
    for vias, directions, match in (
        ((("through", "F.Cu", "B.Cu"),), (), "the through via F.Cu-B.Cu is implicit"),
        ((("laser", "F.Cu", "In1.Cu"),), (), "the kind must be blind, buried or micro"),
        ((), (("In1.Cu", "horizontal"),), "In1.Cu: not a signal layer"),
        ((), (("F.Cu", "diagonal"),), "is not horizontal or vertical"),
    ):
        with pytest.raises(ValueError, match=match):
            RouterStack.from_stackup(_stack(8), vias, "x", directions=directions)
    # pure data: the module does not import the maze router
    tree = ast.parse((Path(__file__).parents[1] / "ai_eda" / "tools" / "routing" / "stack.py").read_text(encoding="utf-8"))
    imported = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)] + [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert not any("maze" in (m or "") for m in imported)


# --------------------------------------------------------------------------- the template stacks are unchanged


def _digest(value) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def test_the_2_and_4_layer_template_stacks_and_texts_are_byte_identical_to_c270797():
    assert LAYER_COUNT_OPTIONS == (2, 4) and STACKUP_VERSION == "0.1"
    assert stack_description(2) == ("2-layer 1.6 mm FR-4: copper F.Cu 35 um, B.Cu 35 um; dielectrics core 1.53 mm er 4.5 @ 1 MHz - a generic value, "
                                    "not a fab's (a fab's stack is not read into the IR in this version; only the fab's measured impedance is real)")
    assert stack_description(4) == ("4-layer 1.6 mm FR-4: copper F.Cu 35 um, In1.Cu 17.5 um, In2.Cu 17.5 um, B.Cu 35 um; dielectrics prepreg 0.2 mm "
                                    "er 4.5 @ 1 MHz, core 1.095 mm er 4.5 @ 1 MHz, prepreg 0.2 mm er 4.5 @ 1 MHz - a generic value, not a fab's (a fab's "
                                    "stack is not read into the IR in this version; only the fab's measured impedance is real)")
    pinned = {  # sha256 of the design view, captured at c270797
        (2, False, "+5V"): "f071ace2fe296e666e97148a10ec7854c8647f574dbe8449fd23b3b7eb3cfed1",
        (2, True, "+5V"): "70e5c61720f23338c8f7245c77f061c1e28f8f78a978a555151beaa5764c79f6",
        (4, False, "+5V"): "b2692c9a3056e14ad3b771250eb72082958390d6e532890070fcb81c133660d7",
        (4, False, "+3V3"): "109e3f4d9bc98df7589c9b8a151b29cd4b0de2ce06a89ab42fb2277894cb4f67",
        (4, True, "+5V"): "526cbdc704cd7b73e32662be0ef2f514e4506e8334c91d141b9909839388a448",
        (4, True, "+3V3"): "08ced55187e5b09e186ddf0cda84280f4888c52e16000f1a555eaa4f9f762dc4",
    }
    for (n, confirmed, power), digest in pinned.items():
        s = generic_stackup(n, "divider", confirmed=confirmed, ground_net="GND", power_net=power)
        assert _digest(json.dumps(s.model_dump(mode="json", context={"view": "design"}), sort_keys=True, separators=(",", ":"))) == digest, (n, confirmed, power)
    assert _digest("\n".join(c.text() for c in stackup_choices(LayerCountInput(value=2)))) == "20e71b5caf975f616f8fe21c268932c5e2d962f2a1b3975f19b6698d86f9f6a6"
    assert _digest("\n".join(c.text() for c in stackup_choices(LayerCountInput(value=4)))) == "37a2ecd16c1b92173e147890a6946a2cb97ce55e2ed93020195613cf016c2353"
    assert _digest(layer_count_choice_text()) == "19aa99b161d5769815ac56f61a427bab358571b34feb8565f706e437beb713cf"
    with pytest.raises(ValueError, match=r"no generic stack for 6 layers \(one of \[2, 4\]\)"):
        generic_stackup(6, "divider", confirmed=False)
    with pytest.raises(ValueError, match=r"no generic stack for 6 layers \(one of \[2, 4\]\)"):
        stack_description(6)
