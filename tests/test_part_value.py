"""Readable part values (``ai_eda.tools.calc.part_value``) and every template's use of them.

``Component.value`` is display text in KiCad's style (SI prefix, no unit, at
most 5 significant digits, trailing zeros dropped: ``100n``, ``1.5915k``);
the SPICE binding keeps the calculator's exact number and the netlist keeps
``format_spice_number``'s exact spelling (``1e-7``, ``1.5915494309189537k``).
Nothing here compares a part value with a design value as strings: the text
is parsed with the requirement quantity parser (``M`` = mega) and compared
within the 5-significant-digit display rounding.
"""

from __future__ import annotations

import csv
import random
import re
from pathlib import Path

import pytest

from ai_eda.compilers import BOMCompiler, CompileContext, SchematicCompiler
from ai_eda.design import TEMPLATE_VERSION
from ai_eda.ir import ArtifactKind, CircuitIR
from ai_eda.tools.calc import format_part_value as exported_format
from ai_eda.tools.calc.part_value import (
    AGREEMENT_RULE,
    PART_VALUE_DIGITS,
    display_tolerance,
    format_part_value,
    parse_part_value,
    part_value_agrees,
)
from ai_eda.tools.calc.si import format_spice_number, ngspice_reads, parse_spice_number
from ai_eda.tools.kicad import sexpr
from ai_eda.tools.kicad.board import read_board_footprints
from ai_eda.tools.manufacturing.csv_cells import bom_cell_text
from ai_eda.workflow import Stage
from tests.fixtures_atmega import atmega_library
from tests.test_circuit_templates import ASTABLE, DIVIDER, LED, RC, _confirm, _ir, _netlist, _present, template_library

ATMEGA = {"input_voltage": "9 V", "clock_frequency": "16 MHz"}
#: what a part value may look like: optional sign, 1..3 integer digits, an optional fraction, an optional KiCad prefix - or a plain exponent outside f..T
VALUE_RE = re.compile(r"^-?(?:\d{1,3}(?:\.\d*[1-9])?[fpnumkMGT]?|\d(?:\.\d*[1-9])?e-?\d+)$")


def _significant_digits(text: str) -> int:
    mantissa = re.match(r"^-?([\d.]+)", text).group(1).replace(".", "").lstrip("0")
    return len(mantissa)


# --------------------------------------------------------------------------- the formatter


@pytest.mark.parametrize(
    ("value", "text"),
    [
        # the digest's table
        (1e-7, "100n"), (1e-5, "10u"), (2.2e-11, "22p"), (1500.0, "1.5k"), (10000.0, "10k"), (6.48172677616823e-08, "64.817n"),
        # the templates' own numbers
        (1591.5494309189537, "1.5915k"), (14000.0, "14k"), (300.0, "300"), (1000.0, "1k"), (64.8172677616823e-9, "64.817n"),
        # prefixes, no unit letter, KiCad's M for mega and u for micro
        (1e6, "1M"), (4.7e-6, "4.7u"), (0.5, "500m"), (0.7, "700m"), (12.0, "12"), (3.3, "3.3"), (2.2e9, "2.2G"), (1e12, "1T"), (1e-15, "1f"),
        # rounding to 5 significant digits happens once, and the prefix follows the rounded number
        (999.996, "1k"), (99999.6, "100k"), (123456.0, "123.46k"), (1.00004, "1"), (999995000000.0, "1T"),
        # sign, zero, magnitudes outside f..T
        (-1500.0, "-1.5k"), (0.0, "0"), (-0.0, "0"), (1e-18, "1e-18"), (1.23456e-18, "1.2346e-18"), (1e15, "1e15"), (12345678e10, "1.2346e17"),
        # ints are numbers too
        (10_000, "10k"),
    ],
)
def test_format_part_value_table(value, text):
    assert format_part_value(value) == text
    assert exported_format is format_part_value


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_format_part_value_refuses_non_finite_numbers(bad):
    with pytest.raises(ValueError):
        format_part_value(bad)


@pytest.mark.parametrize("bad", [True, False, "10k", None, [1.0]])
def test_format_part_value_refuses_non_numbers(bad):
    with pytest.raises(TypeError):
        format_part_value(bad)  # type: ignore[arg-type]


def test_the_part_value_is_not_the_netlist_spelling():
    """The same number, two spellings: the part value for people, the netlist spelling for ngspice (unchanged)."""
    for value, part, netlist in ((1e-7, "100n", "1e-7"), (1e-5, "10u", "1e-5"), (2.2e-11, "22p", "22.0p"), (1591.5494309189537, "1.5915k", "1.5915494309189537k")):
        assert format_part_value(value) == part and format_spice_number(value) == netlist and parse_spice_number(netlist) == value
        # ngspice reads the netlist spelling exactly, or the model cannot vouch for it (17 digits: listed by the compiler as unmodelled)
        assert ngspice_reads(netlist) in (value, None)
    # a part value is never simulation input: ngspice's own parser reads KiCad's mega as milli
    assert parse_spice_number("1M") == 1e-3 and parse_part_value("1M", "ohm") == 1e6


def test_round_trip_parses_to_the_five_digit_rounding_and_formats_back_to_the_same_text():
    rng = random.Random(20260927)
    values = [m * 10.0**e for e in range(-17, 16) for m in (1.0, 1.2, 1.5, 2.2, 3.3, 4.7, 6.8, 9.99995, 9.999949)]
    values += [rng.uniform(1.0, 10.0) * 10.0 ** rng.randint(-18, 16) * rng.choice((1, -1)) for _ in range(3000)]
    for value in values:
        text = format_part_value(value)
        assert VALUE_RE.match(text), (value, text)
        assert _significant_digits(text) <= PART_VALUE_DIGITS, (value, text)
        parsed = parse_part_value(text, "F")
        # the quantity parser is correctly rounded: the text reads back as exactly the 5-significant-digit rounding of the value
        assert parsed == float(f"{value:.{PART_VALUE_DIGITS - 1}e}"), (value, text, parsed)
        assert part_value_agrees(text, value, "F") is True, (value, text)  # |parsed - value| <= display_tolerance(value) (+ a few ULPs)
        assert format_part_value(parsed) == text, (value, text)  # idempotent


def test_display_tolerance_is_half_a_unit_in_the_fifth_significant_digit():
    assert display_tolerance(1591.5494309189537) == pytest.approx(0.05) and display_tolerance(1e-7) == pytest.approx(5e-13)
    assert display_tolerance(999.996) == pytest.approx(0.005)  # the decade of the unrounded number, though it is shown as 1k
    assert display_tolerance(-1500.0) == pytest.approx(0.05) and display_tolerance(0.0) == 0.0


# --------------------------------------------------------------------------- reading a part value back


@pytest.mark.parametrize(
    ("text", "unit", "expected"),
    [
        ("100n", "F", 1e-7), ("10u", "H", 1e-5), ("22p", "F", 2.2e-11), ("1.5k", "ohm", 1500.0), ("1.5K", "ohm", 1500.0), ("1M", "ohm", 1e6),
        ("1m", "ohm", 1e-3), ("300", "ohm", 300.0), ("0", "F", 0.0), ("-1.5k", "ohm", -1500.0), ("1.2346e-18", "F", 1.2346e-18), ("1.5m", "A", 1.5e-3),
        # not one plain number in that unit: nothing is compared
        ("LM7805", "ohm", None), ("2N3904", "ohm", None), ("16MHz", "Hz", None), ("10kΩ", "ohm", None), ("10k 1%", "ohm", None), ("4k7", "ohm", None),
        ("", "ohm", None), (" 10k", "ohm", None), ("10k", None, None), ("10k", "", None),
    ],
)
def test_parse_part_value_reads_with_the_quantity_parser(text, unit, expected):
    assert parse_part_value(text, unit) == expected


def test_agreement_is_within_the_display_rounding_and_says_what_was_compared():
    assert part_value_agrees("10k", 10000.4, "ohm") is True and part_value_agrees("10k", 10000.6, "ohm") is False
    assert part_value_agrees("1.5915k", 1591.5494309189537, "ohm") is True and part_value_agrees("1.5916k", 1591.5494309189537, "ohm") is False
    assert part_value_agrees("100n", 1e-7, "F") is True and part_value_agrees("100n", 1e-6, "F") is False
    assert part_value_agrees("1M", 1e6, "ohm") is True and part_value_agrees("1M", 1e-3, "ohm") is False  # M is mega here
    assert part_value_agrees("LM7805", 5.0, "V") is None  # not compared, never agreement
    assert "quantity parser" in AGREEMENT_RULE and f"{PART_VALUE_DIGITS}-significant-digit display rounding" in AGREEMENT_RULE


@pytest.mark.parametrize(("text", "value", "unit"), [("1e400", 1000.0, "ohm"), ("1e309", 1e-7, "F"), ("-1e400", 1000.0, "ohm"), ("1e999k", 1000.0, "ohm")])
def test_a_part_value_that_overflows_is_no_number_and_never_agrees(text, value, unit):
    """The quantity parser reads ``1e400`` as infinity; that is no value, so nothing is compared (``math.ulp(inf)`` once made any
    difference fit the slack and every overflowing text 'agreed')."""
    assert parse_part_value(text, unit) is None
    assert part_value_agrees(text, value, unit) is None


def test_the_formatter_never_writes_a_value_that_reads_back_as_infinity():
    """``1.7976931348623157e308`` rounds to ``1.7977e308``, past the largest double: refused instead of written; the largest value whose
    rounding stays finite is written and read back."""
    for bad in (1.7976931348623157e308, -1.7976931348623157e308):
        with pytest.raises(ValueError, match="largest double"):
            format_part_value(bad)
    assert format_part_value(1.7976e308) == "1.7976e308" and parse_part_value("1.7976e308", "ohm") == 1.7976e308
    assert part_value_agrees("1.7976e308", 1.7976e308, "ohm") is True


# --------------------------------------------------------------------------- every template


def test_template_version_moved_with_the_value_spelling():
    """The same inputs now build different ``value`` text, so the template provenance names a new version."""
    assert TEMPLATE_VERSION == "0.2"


#: template -> (answers, library factory, the part values it must write)
TEMPLATE_VALUES = {
    "divider": (DIVIDER, template_library, {"R1": "14k", "R2": "10k"}),
    "led": (LED, template_library, {"R1": "300"}),
    "rc_lowpass": (RC, template_library, {"R1": "1.5915k", "C1": "100n"}),
    "astable": (ASTABLE, template_library, {"R1": "1k", "R2": "1k", "R3": "10k", "R4": "10k", "C1": "64.817n", "C2": "64.817n"}),
    "atmega128_devboard": (ATMEGA, atmega_library, {
        "C1": "10u", "C2": "10u", "C3": "100n", "R1": "1.5k", "C4": "100n", "C5": "100n", "L1": "10u", "C6": "100n", "C7": "100n",
        "R2": "10k", "C8": "100n", "R3": "10k", "C9": "22p", "C10": "22p",
    }),
}


@pytest.mark.parametrize("template", sorted(TEMPLATE_VALUES))
def test_every_template_writes_readable_values_and_keeps_the_exact_netlist(template: str, tmp_path: Path):
    answers, factory, expected = TEMPLATE_VALUES[template]
    lib = factory(tmp_path / "kicad")
    ir = _ir(tmp_path, "tpl")
    question = _present(ir, tmp_path, lib, answers)
    _confirm(ir, tmp_path, lib)
    numeric = {c.ref: c for c in ir.components if c.electrical}
    assert {ref: c.value for ref, c in numeric.items()} == expected
    netlist = {line.split()[0]: line for line in _netlist(ir, tmp_path, lib).splitlines()[1:] if line and not line.startswith(".")}
    for ref, c in numeric.items():
        assert VALUE_RE.match(c.value) and _significant_digits(c.value) <= PART_VALUE_DIGITS, (ref, c.value)
        assert f"{ref} " in question and f", value {c.value}" in question  # the confirm_design table shows the readable spelling
        for key, t in c.electrical.items():
            assert c.value == format_part_value(t.value), (ref, key)
            assert part_value_agrees(c.value, t.value, t.unit) is True, (ref, key, c.value, t.value)
        b = c.spice
        if b is not None and not b.exclude and b.value is not None:
            # the simulated number is the calculator's exact value, spelled in the netlist so ngspice reads exactly that double
            assert b.value.value == next(iter(c.electrical.values())).value
            spelled = format_spice_number(b.value.value)
            assert f" {spelled}" in netlist[ref] and parse_spice_number(spelled) == b.value.value, (ref, netlist[ref])
            assert ngspice_reads(spelled) in (b.value.value, None), (ref, spelled)  # None: 17 digits, reported by the compiler as unmodelled


def test_bom_schematic_and_board_show_the_readable_value_and_the_netlist_the_exact_one(tmp_path: Path):
    """The RC template through the whole offline pipeline: every derived view shows ``100n`` / ``1.5915k``, the netlist ``1e-7`` / the exact R."""
    lib = template_library(tmp_path / "kicad")
    ir = _ir(tmp_path, "rc")
    _present(ir, tmp_path, lib, RC)
    state, _ = _confirm(ir, tmp_path, lib, None)
    assert state.outcome(Stage.PCB) is not None and ArtifactKind.PCB in ir.artifacts
    want = {"R1": "1.5915k", "C1": "100n", "J1": "Conn_01x03"}
    # BOM (decoded by the single decoder: a positive number is never neutralised)
    with open(ir.artifacts[ArtifactKind.BOM].path, newline="", encoding="utf-8") as f:
        rows = {row["Reference"]: row for row in csv.DictReader(f)}
    assert {ref: bom_cell_text(row["Value"]) for ref, row in rows.items()} == want
    assert {ref: row["Value"] for ref, row in rows.items()} == want  # written as is: no apostrophe on a plain value
    # schematic: the placed symbols' Value fields
    tree = sexpr.parse_file(ir.artifacts[ArtifactKind.SCHEMATIC].path)
    placed = [s for s in sexpr.find_all(tree, "symbol") if sexpr.get(s, "lib_id") is not None]
    fields = {}
    for s in placed:
        props = {str(p[1]): str(p[2]) for p in sexpr.find_all(s, "property")}
        fields[props["Reference"]] = props["Value"]
    assert fields == want
    # board: the footprints' Value property
    assert {fp.ref: fp.value for fp in read_board_footprints(Path(ir.artifacts[ArtifactKind.PCB].path))} == want
    # netlist: the exact spellings, unchanged
    text = _netlist(ir, tmp_path, lib)  # the SPICE compiler itself (this offline run has no engine, so no netlist artifact)
    assert "C1 OUT 0 1e-7\n" in text and "R1 IN OUT 1.5915494309189537k\n" in text and "1.5915k" not in text and "100n" not in text
    # the two compilers are deterministic on the readable values too
    again = CircuitIR.load(ir.save(tmp_path / "ir.json"))
    for compiler, name in ((BOMCompiler(), "bom"), (SchematicCompiler(), "sch")):
        first = compiler.compile(again, CompileContext(workdir=tmp_path / f"{name}1", tools={"kicad_library": lib}))
        second = compiler.compile(again, CompileContext(workdir=tmp_path / f"{name}2", tools={"kicad_library": lib}))
        assert first.content_hash == second.content_hash


def test_a_report_claims_the_display_spelling_only_where_the_ir_carries_it(tmp_path: Path):
    """The astable theory quotes C1's value text from the IR with the 5-digit claim only while C1 and C2 carry the display spelling of
    their design number; a hand edit or a missing part gets the v0.1 sentence, never a claim about text the IR does not hold."""
    from ai_eda.design.templates import AstableTemplate, display_spelled

    lib = template_library(tmp_path / "kicad")
    ir = _ir(tmp_path, "osc")
    _present(ir, tmp_path, lib, ASTABLE)
    _confirm(ir, tmp_path, lib)
    body = lambda: "\n".join(s.body for s in AstableTemplate().theory(ir))  # noqa: E731
    assert "C 는 계산값 그대로입니다(E 계열 반올림 없음; C1·C2 의 부품 값 표기 `64.817n`: 유효숫자 5자리, 넷리스트는 정확한 값). " in body()
    assert display_spelled(ir.component("C1")) is True and display_spelled(ir.component("J1")) is None  # a header has no design number
    ir.component("C1").value = "68n`x"
    assert display_spelled(ir.component("C1")) is False
    assert "C 는 계산값 그대로입니다(E 계열 반올림 없음). " in body() and "유효숫자" not in body()
    ir.components = [c for c in ir.components if c.ref != "C1"]
    assert "C 는 계산값 그대로입니다(E 계열 반올림 없음). " in body() and "유효숫자" not in body()


def _as_template_v01(ir: CircuitIR) -> CircuitIR:
    """``ir`` as template v0.1 built it: every numeric part value in the netlist spelling, every template provenance at version 0.1."""
    for c in ir.components:
        numbers = [t.value for t in c.electrical.values()]
        if numbers:
            c.value = format_spice_number(numbers[0])
        if c.provenance.tool and c.provenance.tool.startswith("design.template."):
            c.provenance.tool_version = "0.1"
    if ir.topology is not None and ir.topology.provenance.tool:
        ir.topology.provenance.tool_version = "0.1"
    return ir


def test_a_v01_template_project_keeps_the_v01_wording_without_the_5_digit_claim(tmp_path: Path):
    """A project confirmed under template v0.1 keeps its netlist-spelled values (the circuit agent never rebuilds a design): its
    reports print v0.1's sentences - no value-column note, no '5 significant digits' next to ``64.8172677616823n`` or
    ``1.5915494309189537k`` - while a v0.2 build of the same inputs gets them."""
    from ai_eda.design.templates import VALUE_SPELLING_NOTE, AstableTemplate, RcLowpassTemplate
    from ai_eda.report.stages import VALUE_COLUMN_NOTE, parts_report, template_of

    lib = template_library(tmp_path / "kicad")
    ir = _ir(tmp_path / "osc", "osc")
    _present(ir, tmp_path / "osc", lib, ASTABLE)
    _confirm(ir, tmp_path / "osc", lib)
    assert VALUE_COLUMN_NOTE in parts_report(ir, lib)  # v0.2: the claim holds
    ir = _as_template_v01(ir)
    assert template_of(ir) == ("astable", "0.1") and ir.component("C1").value == ir.component("C2").value == "64.8172677616823n"
    theory = "\n".join(s.body for s in AstableTemplate().theory(ir))
    notes = AstableTemplate().part_notes(ir)
    parts = parts_report(ir, lib)
    assert "C 는 계산값 그대로입니다(E 계열 반올림 없음). 가장 가까운 E12 값" in theory
    assert "그대로(E 계열 반올림 없음). 매 주기 양극성 전압을 받으므로" in notes["C1"].why and "그대로(E 계열 반올림 없음). 매 주기" in notes["C2"].why
    assert VALUE_COLUMN_NOTE not in parts and "| `C1` | 64.8172677616823n |" in parts
    for text in (theory, parts, *(n.why for n in notes.values()), *(c for n in notes.values() for c in n.criteria)):
        assert "유효숫자" not in text and VALUE_SPELLING_NOTE not in text
    # the RC's calculated R: v0.1's resistor line, no spelling note
    rc = _ir(tmp_path / "rc", "rc")
    _present(rc, tmp_path / "rc", lib, RC)
    _confirm(rc, tmp_path / "rc", lib)
    assert f"저항값 1.5915 kΩ (계산값 그대로: E 계열 반올림은 하지 않았음; {VALUE_SPELLING_NOTE})" in RcLowpassTemplate().part_notes(rc)["R1"].criteria
    rc = _as_template_v01(rc)
    assert rc.component("R1").value == "1.5915494309189537k"
    assert "저항값 1.5915 kΩ (계산값 그대로: E 계열 반올림은 하지 않았음)" in RcLowpassTemplate().part_notes(rc)["R1"].criteria
    assert VALUE_COLUMN_NOTE not in parts_report(rc, lib) and VALUE_SPELLING_NOTE not in parts_report(rc, lib)
