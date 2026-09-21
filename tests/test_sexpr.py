"""Pure tests for the KiCad s-expression reader/writer (no KiCad installation needed)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_eda.tools.kicad import sexpr
from ai_eda.tools.kicad.sexpr import Q, QStr, S, SExprError

EFFECTS = S("effects", S("font", S("size", 1.27, 1.27)))


# --------------------------------------------------------------------------- quoting


def test_quoted_and_bare_atoms_are_distinct_types():
    node = sexpr.parse('(pad "1" smd roundrect (at -0.825 0) (layers "F.Cu" "F.Mask"))')
    assert node[0] == "pad" and type(node[0]) is str
    assert isinstance(node[1], QStr) and node[1] == "1"
    assert node[2] == "smd" and not isinstance(node[2], QStr)
    layers = sexpr.find(node, "layers")
    assert all(isinstance(layer, QStr) for layer in sexpr.args(layers))


def test_quoting_is_reproduced_not_inferred():
    # a quoted "yes" stays quoted, a bare 1.27 stays bare, a quoted number stays quoted
    node = sexpr.parse('(x "yes" 1.27 "1.27" passive "passive")')
    assert sexpr.dumps(node) == '(x "yes" 1.27 "1.27" passive "passive")\n'


def test_empty_string_stays_quoted():
    node = sexpr.parse('(property "Footprint" "")')
    assert isinstance(node[2], QStr) and node[2] == ""
    assert sexpr.dumps(node) == '(property "Footprint" "")\n'
    assert sexpr.dumps(S("name", Q(""))) == '(name "")\n'


def test_bare_atom_that_needs_quoting_is_refused():
    with pytest.raises(SExprError):
        sexpr.dumps(S("descr", "has space"))
    with pytest.raises(SExprError):
        sexpr.dumps(S("descr", ""))
    with pytest.raises(SExprError):
        sexpr.dumps(S("descr", 'quo"te'))


# --------------------------------------------------------------------------- escapes


def test_escapes_round_trip():
    text = '(text "say \\"hi\\"\\\\ path\\nline2\ttab Ω 한글")'
    node = sexpr.parse(text)
    assert node[1] == 'say "hi"\\ path\nline2\ttab Ω 한글'
    assert sexpr.dumps(node) == text + "\n"


def test_escape_unescape_pairs():
    assert sexpr.escape('a"b\\c\nd\re\tf') == 'a\\"b\\\\c\\nd\\re\tf'
    assert sexpr.unescape('a\\"b\\\\c\\nd\\re\\tf') == 'a"b\\c\nd\re\tf'
    assert sexpr.unescape("no escapes") == "no escapes"
    assert sexpr.unescape("\\q") == "\\q"  # unknown escape kept verbatim


def test_quote_inside_string_does_not_end_it():
    node = sexpr.parse('(a "x\\"(y)" b)')
    assert node[1] == 'x"(y)' and node[2] == "b"


# --------------------------------------------------------------------------- numbers


def test_numbers_are_kept_as_text_on_parse():
    node = sexpr.parse("(at 0 3.81 270)")
    assert node[1:] == ["0", "3.81", "270"]
    assert all(type(a) is str for a in node[1:])
    assert sexpr.to_float(node[2]) == 3.81
    assert sexpr.to_int(node[3]) == 270
    with pytest.raises(SExprError):
        sexpr.to_float("abc")
    with pytest.raises(SExprError):
        sexpr.to_int("1.5")


@pytest.mark.parametrize(
    "value, expected",
    [
        (0, "0"),
        (0.0, "0"),
        (-0.0, "0"),
        (1.0, "1"),
        (1.5, "1.5"),
        (1.27, "1.27"),
        (-0.825, "-0.825"),
        (12.5000, "12.5"),
        (0.0254, "0.0254"),
        (-0.237258, "-0.237258"),
        (1e-1, "0.1"),
        (1e-9, "0"),  # below 6 decimals -> 0, never an exponent, never -0
        (-1e-9, "0"),
        (20260206, "20260206"),
        (2.5e3, "2500"),
    ],
)
def test_fmt_num_matches_kicad_style(value, expected):
    assert sexpr.fmt_num(value) == expected


def test_fmt_num_decimals_cap_and_bool_rejection():
    assert sexpr.fmt_num(1.23456789) == "1.234568"
    assert sexpr.fmt_num(1.23456789, decimals=4) == "1.2346"
    with pytest.raises(TypeError):
        sexpr.fmt_num(True)


def test_builder_formats_python_values():
    node = S("pad", Q("1"), "smd", S("at", -0.825, 0.0, 180), S("hide", True), S("fill", False), S("v", 20260206))
    assert sexpr.dumps(node) == (
        '(pad "1" smd\n\t(at -0.825 0 180)\n\t(hide yes)\n\t(fill no)\n\t(v 20260206)\n)\n'
    )


def test_builder_skips_none_items():
    assert S("pad", Q("1"), None, S("drill", 1) if False else None) == ["pad", QStr("1")]


# --------------------------------------------------------------------------- layout


def test_layout_reproduces_kicad_pin_block():
    pin = S(
        "pin", "passive", "line",
        S("at", 0, 3.81, 270),
        S("length", 1.27),
        S("name", Q(""), EFFECTS),
        S("number", Q("1"), EFFECTS),
    )
    expected = (
        "(pin passive line\n"
        "\t(at 0 3.81 270)\n"
        "\t(length 1.27)\n"
        '\t(name ""\n'
        "\t\t(effects\n"
        "\t\t\t(font\n"
        "\t\t\t\t(size 1.27 1.27)\n"
        "\t\t\t)\n"
        "\t\t)\n"
        "\t)\n"
        '\t(number "1"\n'
        "\t\t(effects\n"
        "\t\t\t(font\n"
        "\t\t\t\t(size 1.27 1.27)\n"
        "\t\t\t)\n"
        "\t\t)\n"
        "\t)\n"
        ")\n"
    )
    assert sexpr.dumps(pin) == expected


def test_layout_packs_xy_runs_on_one_line():
    poly = S(
        "polyline",
        S("pts", S("xy", 0, 0), S("xy", 0, -1.27), S("xy", 1.27, -1.27)),
        S("stroke", S("width", 0), S("type", "default")),
    )
    assert sexpr.dumps(poly) == (
        "(polyline\n"
        "\t(pts\n"
        "\t\t(xy 0 0) (xy 0 -1.27) (xy 1.27 -1.27)\n"
        "\t)\n"
        "\t(stroke\n"
        "\t\t(width 0)\n"
        "\t\t(type default)\n"
        "\t)\n"
        ")\n"
    )


def test_layout_wraps_long_atom_lists_at_column_72():
    uuids = [Q(f"{i:08d}-0000-0000-0000-000000000000") for i in range(4)]
    text = sexpr.dumps(S("group", Q("g"), S("members", *uuids)))
    lines = text.split("\n")
    assert lines[0] == '(group "g"'
    assert lines[1].startswith('\t(members "00000000-')
    # head + two uuids reach col 87 (>= 72) -> wrap before the third; the third ends at col 40 so the fourth
    # still fits on that line (KiCad only wraps when the column is already past 72 when a space is due)
    assert lines[1] == '\t(members "00000000-0000-0000-0000-000000000000" "00000001-0000-0000-0000-000000000000"'
    assert lines[2] == '\t\t"00000002-0000-0000-0000-000000000000" "00000003-0000-0000-0000-000000000000"'
    assert lines[3] == "\t)"  # a wrapped list closes on its own line
    assert lines[4] == ")"
    assert sexpr.strict_equal(sexpr.parse(text), S("group", Q("g"), S("members", *uuids)))


def test_output_ends_with_single_newline_and_no_spaces_indent():
    text = sexpr.dumps(S("a", S("b", 1), S("c", S("d", Q("x")))))
    assert text.endswith(")\n") and not text.endswith("\n\n")
    assert all(line == line.lstrip(" ") for line in text.split("\n"))


# --------------------------------------------------------------------------- round trip


SAMPLE = """(kicad_symbol_lib
\t(version 20251024)
\t(generator "kicad_symbol_editor")
\t(symbol "R"
\t\t(pin_numbers
\t\t\t(hide yes)
\t\t)
\t\t(property "Description" "Resistor \\"1%\\"\\nline2"
\t\t\t(at 0 0 0)
\t\t\t(hide yes)
\t\t)
\t\t(symbol "R_0_1"
\t\t\t(rectangle
\t\t\t\t(start -1.016 -2.54)
\t\t\t\t(end 1.016 2.54)
\t\t\t)
\t\t\t(polyline
\t\t\t\t(pts
\t\t\t\t\t(xy 0 0) (xy 0 -1.27) (xy 1.27 -1.27)
\t\t\t\t)
\t\t\t)
\t\t)
\t\t(embedded_fonts no)
\t)
)
"""


def test_round_trip_inline_sample_is_byte_identical():
    tree = sexpr.parse(SAMPLE)
    assert sexpr.dumps(tree) == SAMPLE
    assert sexpr.strict_equal(sexpr.parse(sexpr.dumps(tree)), tree)


def test_parse_accepts_crlf_compact_and_bom():
    crlf = SAMPLE.replace("\n", "\r\n")
    compact = " ".join(SAMPLE.split())
    for variant in (crlf, compact, "﻿" + SAMPLE):
        assert sexpr.strict_equal(sexpr.parse(variant), sexpr.parse(SAMPLE))


def test_parse_errors():
    with pytest.raises(SExprError):
        sexpr.parse("((a)")
    with pytest.raises(SExprError):
        sexpr.parse("(a))")
    with pytest.raises(SExprError):
        sexpr.parse('(a "unterminated)')
    with pytest.raises(SExprError):
        sexpr.parse("(a) (b)")  # two roots
    with pytest.raises(SExprError):
        sexpr.parse("")
    assert sexpr.parse_all("(a) (b) c") == [["a"], ["b"], "c"]


def test_dump_file_writes_lf_utf8(tmp_path: Path):
    p = sexpr.dump_file(S("t", Q("Ω")), tmp_path / "sub" / "x.kicad_sym")
    raw = p.read_bytes()
    assert raw == '(t "Ω")\n'.encode("utf-8")
    assert b"\r" not in raw
    assert sexpr.strict_equal(sexpr.parse_file(p), S("t", Q("Ω")))


# --------------------------------------------------------------------------- helpers


def test_tree_helpers():
    tree = sexpr.parse(SAMPLE)
    assert sexpr.head(tree) == "kicad_symbol_lib"
    assert sexpr.head("atom") is None and sexpr.head([]) is None and sexpr.head([QStr("x")]) is None
    sym = sexpr.find(tree, "symbol")
    assert sym[1] == "R"
    assert sexpr.get(tree, "version") == "20251024"
    assert sexpr.get(tree, "missing", 1, "dflt") == "dflt"
    assert sexpr.get(tree, "version", 5) is None
    assert [s[1] for s in sexpr.find_all(sym, "symbol")] == ["R_0_1"]
    assert sexpr.find(sym, "nothing") is None
    assert sexpr.args(sexpr.find(sexpr.find(sym, "property"), "at")) == ["0", "0", "0"]
    assert sexpr.args(sym) == [QStr("R")]


def test_strict_equal_distinguishes_quoting_but_eq_does_not():
    a = S("x", Q("1"))
    b = S("x", "1")
    assert a == b
    assert not sexpr.strict_equal(a, b)
    assert sexpr.strict_equal(a, S("x", Q("1")))
    assert not sexpr.strict_equal(S("x"), "x")
    assert not sexpr.strict_equal(S("x", 1), S("x", 1, 2))


def test_deep_copy_keeps_qstr():
    src = S("property", Q("Reference"), Q("R"), S("at", 0, 0, 0))
    cp = sexpr.deep_copy(src)
    assert cp is not src and cp[3] is not src[3]
    assert isinstance(cp[1], QStr) and sexpr.strict_equal(cp, src)
    cp[3][1] = 9
    assert src[3][1] == 0
