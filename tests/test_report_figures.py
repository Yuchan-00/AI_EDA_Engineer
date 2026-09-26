"""The report figures (:mod:`ai_eda.report.figures`) are deterministic, escaped, valid SVG views.

Every builder is checked to parse as XML, to give identical bytes on two
calls, to refuse what the chart rules refuse (a 5th series, a non-positive
value on a log axis, mismatched lengths), to escape every string it is
given, and to draw exactly what the inputs hold: one element per pad /
track / via of the placed board on the synthetic library, one marker per
recorded measurement coloured by its recorded status, one label per bar.
Building a figure changes nothing in the IR (its hash and its artifacts).
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

import ai_eda.report.figures as figures
from ai_eda.ir import BoardSide, CircuitIR, LibraryRef, Track, ValidationResult, ValidationStatus, Via
from ai_eda.report.figures import (
    BAND_FILL,
    COLUMN_PX,
    COPPER_COLOURS,
    MAX_POINTS,
    MAX_SERIES,
    NO_MEASUREMENT,
    SERIES_COLOURS,
    STATUS_COLOURS,
    STATUS_OTHER,
    THT_COLOUR,
    ZERO_TOLERANCE,
    Band,
    Figure,
    Marker,
    Series,
    ToleranceRow,
    bar_figure,
    board_figure,
    downsample,
    esc,
    expectation_limit,
    log_ticks,
    nice_ticks,
    si_format,
    svg_line_chart,
    tolerance_figure,
    tolerance_rows,
    waveform_figure,
    waveform_figures,
)
from ai_eda.report.pdf import markdown_to_html
from ai_eda.tools.kicad.library import KicadLibrary
from tests.fixtures_kicad import divider_with_connector_ir
from tests.test_circuit_templates import template_library

SVG_NS = "{http://www.w3.org/2000/svg}"
ISO_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}")
THT_FOOTPRINT = LibraryRef(library="Resistor_THT", name="R_Axial_DIN0207_L6.3mm_D2.5mm_P7.62mm_Horizontal")


def _parse(svg: str) -> ET.Element:
    root = ET.fromstring(svg)
    assert root.tag == f"{SVG_NS}svg"
    return root


def _by_class(root: ET.Element, cls: str) -> list[ET.Element]:
    return [el for el in root.iter() if el.get("class") == cls]


def _text(root: ET.Element) -> str:
    return " ".join(t for t in root.itertext())


def _series(n: int = 3, name: str = "v(OUT)") -> Series:
    xs = [i * 1e-3 for i in range(n)]
    return Series(name, xs, [math.sin(x * 1000) for x in xs])


def _results(n: int = 3000, scale: str = "time") -> dict:
    """A ``spice/results.json`` in the stage's layout: one tran analysis with a voltage and a branch current."""
    t = [i * 1e-6 for i in range(n)]
    return {
        "format": "2",
        "engine": "ngspice-shared",
        "analyses": {
            "tran": {
                "kind": "tran",
                "command": "tran 1u 3m",
                "result": {
                    "scale": scale,
                    "vectors": {scale: t, "out": [5.0 * (1 - math.exp(-x / 1e-3)) for x in t], "vvin#branch": [-1e-3 * math.exp(-x / 1e-3) for x in t]},
                    "vector_types": {scale: scale, "out": "voltage", "vvin#branch": "current"},
                    "n_points": n,
                },
            }
        },
    }


# --------------------------------------------------------------------------- helpers


def test_escaping_numbers_and_ticks():
    assert esc('a<b&"c"') == "a&lt;b&amp;&quot;c&quot;"
    assert si_format(0.0015, "s") == "1.5 ms"
    assert si_format(1500, "Hz") == "1.5 kHz"
    assert si_format(6.48172677616823e-8, "F") == "64.82 nF"
    assert si_format(1e4, "ohm") == "10 kΩ"
    assert si_format(66.5, "mm") == "66.5 mm"  # no prefix on a non-SI-prefixed unit
    assert si_format(0.0, "V") == "0 V"
    assert si_format(0.5, "V", exponent=-3) == "500 mV" and si_format(0.0, "V", exponent=-3) == "0 mV"  # a shared axis prefix
    assert si_format(0.30000000000000004) == "0.3"
    assert nice_ticks(0, 0.005) == [0.0, 0.001, 0.002, 0.003, 0.004, 0.005]
    assert nice_ticks(-0.3, 5.2) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert nice_ticks(2.5, 2.5) == [2.5]
    assert all(4 <= len(nice_ticks(lo, hi)) <= 11 for lo, hi in ((0, 1), (0.01, 0.02), (-4.3, 5.1), (0, 1234), (1e-9, 7e-9)))
    assert log_ticks(100, 20000) == [100.0, 1000.0, 10000.0]
    assert log_ticks(80, 3000) == [100.0, 200.0, 500.0, 1000.0, 2000.0]
    assert log_ticks(300, 800) == [300.0, 400.0, 500.0, 600.0, 700.0, 800.0]  # under a decade: linear nice ticks
    with pytest.raises(ValueError, match="positive"):
        log_ticks(0, 10)


def test_downsampling_keeps_first_and_last_and_the_cap():
    xs = list(range(5000))
    ys = [x * 2 for x in xs]
    dx, dy = downsample(xs, ys, 2000)
    assert len(dx) <= 2000 and dx[0] == 0 and dx[-1] == 4999 and dy[-1] == 9998
    assert dx == sorted(dx) and len(set(dx)) == len(dx)
    assert downsample([1, 2, 3], [4, 5, 6]) == ([1, 2, 3], [4, 5, 6])
    assert downsample(xs, ys) == downsample(xs, ys)
    with pytest.raises(ValueError, match="length"):
        downsample([1, 2], [1])


# --------------------------------------------------------------------------- svg_line_chart


def test_line_chart_is_valid_xml_and_deterministic():
    kwargs = dict(title="파형", x_label="시간 (s)", y_label="전압 (V)", bands=[Band("y", 0.2, 0.8, "허용")], markers=[Marker(1e-3, 0.5, "설계점")])
    a = svg_line_chart([_series(50)], **kwargs)
    b = svg_line_chart([_series(50)], **kwargs)
    assert a == b
    root = _parse(a)
    assert root.get("width") == str(COLUMN_PX) == "672" and root.get("height") == "440"
    assert len(_by_class(root, "series")) == 1 and _by_class(root, "series")[0].get("data-name") == "v(OUT)"
    assert _by_class(root, "series")[0].get("stroke") == SERIES_COLOURS[0]
    assert len(_by_class(root, "band")) == 1 and _by_class(root, "band")[0].get("fill") == BAND_FILL
    assert len(_by_class(root, "marker")) == 1
    text = _text(root)
    assert "시간 (s)" in text and "전압 (V)" in text and "허용" in text and "설계점" in text
    assert "10 ms" in text and "0 ms" in text and "0 s" not in text  # SI-prefixed tick labels sharing the axis prefix
    assert not _by_class(root, "legend"), "one series: the title names it, no legend"
    assert not ISO_RE.search(a)


def test_legend_for_two_or_more_series_and_fixed_colour_order():
    svg = svg_line_chart([_series(20, "a"), _series(20, "b"), _series(20, "c")], title="t", x_label="x (s)", y_label="y (V)")
    root = _parse(svg)
    legend = _by_class(root, "legend")
    assert len(legend) == 1 and "a" in _text(legend[0]) and "c" in _text(legend[0])
    assert [s.get("stroke") for s in _by_class(root, "series")] == list(SERIES_COLOURS[:3])
    # legend text wears ink, never the series colour
    for t in legend[0].iter(f"{SVG_NS}text"):
        assert t.get("fill") not in SERIES_COLOURS


def test_a_fifth_series_raises():
    with pytest.raises(ValueError, match=f"at most {MAX_SERIES}"):
        svg_line_chart([_series(5, f"s{i}") for i in range(5)], title="t", x_label="x", y_label="y")
    svg_line_chart([_series(5, f"s{i}") for i in range(4)], title="t", x_label="x", y_label="y")  # four are fine


def test_text_is_escaped():
    svg = svg_line_chart([_series(5, "<b>&x")], title='t<1 "q"', x_label="x<y", y_label="y", bands=[Band("x", 0, 1e-3, "<band>")], markers=[Marker(0, 0, "<m>")])
    root = _parse(svg)
    assert "<b>" not in svg and "<band>" not in svg and "<m>" not in svg and "&lt;b&gt;&amp;x" in svg
    assert _by_class(root, "series")[0].get("data-name") == "<b>&x"
    assert root.find(f"{SVG_NS}title").text == 't<1 "q"'


def test_log_axes_refuse_non_positive_values_with_a_clear_message():
    with pytest.raises(ValueError, match=r"log x axis: series 'f' has a non-positive x value at index 0 \(0\)"):
        svg_line_chart([Series("f", [0, 1, 10], [1, 2, 3])], title="t", x_label="x", y_label="y", log_x=True)
    with pytest.raises(ValueError, match=r"log y axis: series 'f' has a non-positive y value at index 2 \(-1\)"):
        svg_line_chart([Series("f", [1, 2, 3], [1, 2, -1])], title="t", x_label="x", y_label="y", log_y=True)
    with pytest.raises(ValueError, match="non-positive"):
        svg_line_chart([Series("f", [1, 10], [1, 10])], title="t", x_label="x", y_label="y", log_y=True, bands=[Band("y", 0, 5, "")])
    # a valid log-log chart ticks at decades with SI prefixes
    cs = [1e-9 * 10 ** (i / 10) for i in range(41)]
    svg = svg_line_chart([Series("f", cs, [1 / c for c in cs])], title="t", x_label="C (F)", y_label="f (Hz)", log_x=True, log_y=True)
    text = _text(_parse(svg))
    assert "1 nF" in text and "1 µF" in text and "1 MHz" in text


def test_series_refusals():
    with pytest.raises(ValueError, match="differ in length"):
        svg_line_chart([Series("a", [1, 2], [1])], title="t", x_label="x", y_label="y")
    with pytest.raises(ValueError, match="no points"):
        svg_line_chart([Series("a", [], [])], title="t", x_label="x", y_label="y")
    with pytest.raises(ValueError, match="non-finite"):
        svg_line_chart([Series("a", [1, 2], [1, math.nan])], title="t", x_label="x", y_label="y")
    with pytest.raises(ValueError, match="at least one series"):
        svg_line_chart([], title="t", x_label="x", y_label="y")
    with pytest.raises(ValueError, match="axis must be"):
        svg_line_chart([_series(3)], title="t", x_label="x", y_label="y", bands=[Band("z", 0, 1, "")])


def test_chart_downsamples_to_the_cap_keeping_the_end_points():
    n = 7000
    svg = svg_line_chart([_series(n)], title="t", x_label="시간 (s)", y_label="전압 (V)")
    poly = _by_class(_parse(svg), "series")[0]
    pts = poly.get("points").split(" ")
    assert len(pts) == int(poly.get("data-points")) <= MAX_POINTS
    # the first point sits on the left edge of the plot and the last on the right edge
    x_first, x_last = float(pts[0].split(",")[0]), float(pts[-1].split(",")[0])
    assert x_first < x_last and x_last == max(float(p.split(",")[0]) for p in pts)


def test_each_chart_has_its_own_clip_path_id():
    """Inline SVGs share the page's id space and ``url(#id)`` resolves to the first match, so two charts on one page never share a clip id."""
    a = svg_line_chart([_series(20, "a"), _series(20, "b")], title="A", x_label="x (s)", y_label="y (V)")  # legend: a lower plot top
    b = svg_line_chart([_series(20)], title="B", x_label="x (s)", y_label="y (V)")
    ids = re.findall(r'<clipPath id="([^"]+)"', a + b)
    assert len(ids) == 2 and ids[0] != ids[1] and all(re.fullmatch(r"plot-[A-Za-z0-9_-]+", i) for i in ids)
    assert f'clip-path="url(#{ids[0]})"' in a and f'clip-path="url(#{ids[1]})"' in b
    html = markdown_to_html("![fig](fig:a)\n\n![fig](fig:b)\n", title="t", figures={"a": Figure("a", "A", "", a), "b": Figure("b", "B", "", b)})
    all_ids = re.findall(r'\sid="([^"]+)"', html)
    assert len(all_ids) == len(set(all_ids)) == 2
    # a caller's figure id names the clip path; an id with characters an id cannot carry is sanitised and made unique by a hash
    assert '<clipPath id="plot-waveform">' in waveform_figure(_results(), "tran", ["v(OUT)"], title="t").svg
    assert '<clipPath id="plot-waveform_ac">' in waveform_figure(_results(), "tran", ["v(OUT)"], title="t", fig_id="waveform_ac").svg
    odd = svg_line_chart([_series(5)], title="t", x_label="x", y_label="y", clip_id="a b)c")
    (cid,) = re.findall(r'<clipPath id="([^"]+)"', odd)
    assert re.fullmatch(r"plot-a-b-c-[0-9a-f]{8}", cid) and f"url(#{cid})" in odd
    assert cid != re.findall(r'<clipPath id="([^"]+)"', svg_line_chart([_series(5)], title="t", x_label="x", y_label="y", clip_id="a b c"))[0]
    # the default id is deterministic and follows the chart's text and rect
    assert svg_line_chart([_series(5)], title="t", x_label="x", y_label="y") == svg_line_chart([_series(5)], title="t", x_label="x", y_label="y")


def test_tick_labels_of_a_nearly_constant_series_are_distinct():
    """A 5 V node with 0.1 mV of ripple gets ticks 50 µV apart: the labels carry enough digits to differ (never five times '5 V')."""
    t = [i * 1e-5 for i in range(1001)]
    svg = svg_line_chart([Series("v(VCC)", t, [5.0 + 1e-4 * math.sin(i / 50) for i in range(1001)])], title="t", x_label="시간 (s)", y_label="전압 (V)")
    labels = [tx.text for tx in _by_class(_parse(svg), "ticks")[0].iter(f"{SVG_NS}text") if tx.text.endswith("V")]
    assert len(labels) >= 4 and len(set(labels)) == len(labels)
    assert labels == ["4.9999 V", "4.99995 V", "5 V", "5.00005 V", "5.0001 V"]
    # the usual ranges keep their short labels
    svg = svg_line_chart([_series(50)], title="t", x_label="시간 (s)", y_label="전압 (V)")
    assert "0 ms" in _text(_parse(svg)) and "10 ms" in _text(_parse(svg)) and "10.00 ms" not in svg


def test_linear_x_axis_starts_at_the_enclosing_tick_when_the_data_starts_a_hair_after_it():
    """ngspice stores the first transient point one step after tstart: 10.0007 ms .. 20 ms is drawn from the 10 ms tick, not from an unlabeled edge."""
    t = [0.010000728819574383 + i * (0.02 - 0.010000728819574383) / 2643 for i in range(2644)]
    svg = svg_line_chart([Series("v", t, [math.sin(x * 1e4) for x in t])], title="t", x_label="시간 (s)", y_label="전압 (V)")
    root = _parse(svg)
    ticks = [tx.text for tx in _by_class(root, "ticks")[0].iter(f"{SVG_NS}text") if tx.text.endswith("ms")]
    assert ticks == ["10 ms", "12 ms", "14 ms", "16 ms", "18 ms", "20 ms"]
    first_tick_x = float(next(tx for tx in _by_class(root, "ticks")[0].iter(f"{SVG_NS}text") if tx.text == "10 ms").get("x"))
    assert first_tick_x == float(_by_class(root, "axis")[1].get("x1"))  # the 10 ms tick sits on the axis' left end
    # an axis ending well short of the next tick is left alone (the astable's 0 .. 0.575 ms recovery curve keeps 0.1 ms steps)
    t = [i * 0.575e-3 / 400 for i in range(401)]
    svg = svg_line_chart([Series("v", t, [x for x in t])], title="t", x_label="시간 (s)", y_label="전압 (V)")
    assert [tx.text for tx in _by_class(_parse(svg), "ticks")[0].iter(f"{SVG_NS}text") if tx.text.endswith("s")][:6] == ["0 µs", "100 µs", "200 µs", "300 µs", "400 µs", "500 µs"]


def test_guide_and_marker_labels_avoid_each_other():
    """A threshold line and a marker on it at the right edge (the astable's V_BE crossing): both labels are kept apart."""
    ts = [i * 1e-6 for i in range(501)]
    vb = [5 - 9.3 * math.exp(-t / 6.5e-4) for t in ts]
    svg = svg_line_chart([Series("v_B(t)", ts, vb)], title="t", x_label="시간 (s)", y_label="전압 (V)", bands=[Band("y", 0.7, 0.7, "V_BE 문턱")], markers=[Marker(5e-4, 0.7, "T_half")])
    root = _parse(svg)
    guide = _by_class(root, "guide-label")[0]
    marker = _by_class(root, "marker-label")[0]
    assert guide.text == "V_BE 문턱" and marker.text == "T_half"
    same_row = abs(float(guide.get("y")) - float(marker.get("y"))) < 12
    assert not same_row or guide.get("text-anchor") != marker.get("text-anchor")


# --------------------------------------------------------------------------- waveform_figure


def test_waveform_figure_reads_results_json_vectors():
    res = _results()
    fig = waveform_figure(res, "tran", ["v(OUT)"], title="파형")
    assert isinstance(fig, Figure) and fig.id == "waveform" and fig.title == "파형"
    root = _parse(fig.svg)
    poly = _by_class(root, "series")
    assert len(poly) == 1 and poly[0].get("data-name") == "v(OUT)" and int(poly[0].get("data-points")) <= MAX_POINTS
    text = _text(root)
    assert "시간 (s)" in text and "전압 (V)" in text and "1 ms" in text
    assert "tran 1u 3m" in fig.caption and "3000" in fig.caption
    assert fig.svg == waveform_figure(res, "tran", ["v(OUT)"], title="파형").svg
    # plot-vector spellings are accepted too, and the current gets its own axis title
    assert _by_class(_parse(waveform_figure(res, "tran", ["out"], title="t").svg), "series")[0].get("data-name") == "out"
    cur = waveform_figure(res, "tran", ["i(VVIN)"], title="t")
    assert "전류 (A)" in _text(_parse(cur.svg)) and "mA" in _text(_parse(cur.svg))


def test_waveform_figure_refusals_name_what_is_available():
    res = _results()
    with pytest.raises(ValueError, match=r"analysis 'ac' is not in results.json \(analyses: \['tran'\]\)"):
        waveform_figure(res, "ac", ["v(OUT)"], title="t")
    with pytest.raises(ValueError, match=r"vector 'v\(NOPE\)' is not in analysis 'tran' \(available: \['out', 'time', 'vvin#branch'\]\)"):
        waveform_figure(res, "tran", ["v(NOPE)"], title="t")
    with pytest.raises(ValueError, match="never share one chart"):
        waveform_figure(res, "tran", ["v(OUT)", "i(VVIN)"], title="t")
    with pytest.raises(ValueError, match="no vectors"):
        waveform_figure(res, "tran", [], title="t")
    with pytest.raises(ValueError, match="not in results.json"):
        waveform_figure({"format": "2"}, "tran", ["v(OUT)"], title="t")
    res["analyses"]["tran"]["result"]["scale"] = None
    with pytest.raises(ValueError, match="no scale vector"):
        waveform_figure(res, "tran", ["v(OUT)"], title="t")


def test_waveform_figures_split_voltages_and_currents():
    figs = waveform_figures(_results(), "tran", ["v(OUT)", "i(VVIN)"], title="파형", fig_id="w")
    assert [f.id for f in figs] == ["w", "w_i"] and figs[1].title == "파형 (전류)"
    assert [f.id for f in waveform_figures(_results(), "tran", ["out"], title="t")] == ["waveform"]
    assert waveform_figures(_results(), "tran", ["i(VVIN)"], title="t")[0].id == "waveform_i"


def test_waveform_figures_split_more_than_four_vectors_into_several_charts():
    """A 5th voltage vector goes into a second chart (never a ValueError that loses the whole waveform); currents likewise."""
    res = _results(300)
    vectors = res["analyses"]["tran"]["result"]["vectors"]
    types = res["analyses"]["tran"]["result"]["vector_types"]
    for name in ("q1_b", "q2_b", "q1_c", "vcc"):
        vectors[name] = list(vectors["out"])
        types[name] = "voltage"
    vectors["vr1#branch"] = list(vectors["vvin#branch"])
    types["vr1#branch"] = "current"
    names = ["v(OUT)", "v(Q1_B)", "v(Q2_B)", "v(Q1_C)", "v(VCC)", "i(VVIN)", "i(VR1)"]
    figs = waveform_figures(res, "tran", names, title="파형", fig_id="w")
    assert [f.id for f in figs] == ["w", "w_2", "w_i"] and [f.title for f in figs] == ["파형 (1/2)", "파형 (2/2)", "파형 (전류)"]
    assert re.findall(r'data-name="([^"]+)"', figs[0].svg) == names[:4] and re.findall(r'data-name="([^"]+)"', figs[1].svg) == ["v(VCC)"]
    assert re.findall(r'data-name="([^"]+)"', figs[2].svg) == ["i(VVIN)", "i(VR1)"]
    assert 'class="legend"' in figs[0].svg and 'class="legend"' not in figs[1].svg
    assert len({re.findall(r'<clipPath id="([^"]+)"', f.svg)[0] for f in figs}) == 3
    for name in ("q1_c", "q2_b", "q1_b", "vcc"):
        vectors[name + "2"] = list(vectors["out"])
        types[name + "2"] = "voltage"
    nine = waveform_figures(res, "tran", names[:5] + ["q1_c2", "q2_b2", "q1_b2", "vcc2"], title="t")
    assert [f.id for f in nine] == ["waveform", "waveform_2", "waveform_3"] and nine[2].title == "t (3/3)"
    assert [f.id for f in waveform_figures(res, "tran", names[:4], title="t")] == ["waveform"]  # four still fit one chart


def test_waveform_on_a_frequency_scale_uses_a_log_axis():
    res = _results(200, scale="frequency")
    res["analyses"]["tran"]["result"]["vectors"]["frequency"] = [10 * 10 ** (i / 40) for i in range(200)]
    text = _text(_parse(waveform_figure(res, "tran", ["out"], title="t").svg))
    assert "주파수 (Hz)" in text and "100 Hz" in text and "1 kHz" in text


# --------------------------------------------------------------------------- board_figure


def _board(tmp_path: Path) -> tuple[CircuitIR, KicadLibrary]:
    """The divider on the synthetic library: R1 on a through-hole footprint, R2 on the bottom side, two tracks and a via."""
    lib = template_library(tmp_path / "kicad")
    ir = divider_with_connector_ir(tmp_path / "proj", library=lib)
    r1 = ir.component("R1")
    r1.footprint = lib.resolve_footprint(THT_FOOTPRINT)
    assert r1.footprint.verified
    ir.pcb.placement("R2").side = BoardSide.BOTTOM
    ir.pcb.tracks = [
        Track(net="VIN", layer="F.Cu", start=(5.0, 6.0), end=(14.0, 6.0), width_mm=0.25),
        Track(net="VOUT", layer="B.Cu", start=(5.0, 8.54), end=(10.0, 8.54), width_mm=0.3),
    ]
    ir.pcb.vias = [Via(net="VOUT", x_mm=10.0, y_mm=8.54, drill_mm=0.3, diameter_mm=0.6)]
    return ir, lib


def test_board_figure_draws_one_element_per_pad_track_and_via(tmp_path: Path):
    ir, lib = _board(tmp_path)
    before = ir.model_dump_json()
    fig = board_figure(ir, lib)
    assert ir.model_dump_json() == before and ir.artifacts == {}
    assert fig.id == "board"
    root = _parse(fig.svg)
    pads = _by_class(root, "pad")
    assert len(pads) == 7  # R1 2 (THT) + R2 2 + J1 3
    assert {(p.get("data-ref"), p.get("data-pad")) for p in pads} == {("R1", "1"), ("R1", "2"), ("R2", "1"), ("R2", "2"), ("J1", "1"), ("J1", "2"), ("J1", "3")}
    tht = [p for p in pads if p.get("data-tht") == "true"]
    assert {p.get("data-ref") for p in tht} == {"R1"}
    for p in tht:
        shape, drill = list(p)
        assert shape.get("fill") == THT_COLOUR and drill.get("class") == "drill" and drill.get("fill") == "#ffffff"
    fills = {p.get("data-ref"): list(p)[0].get("fill") for p in pads if p.get("data-tht") == "false"}
    assert fills["R2"] == COPPER_COLOURS["B.Cu"], "a bottom-side SMD pad is on B.Cu (pad_layers mirrors it)"
    assert fills["J1"] == COPPER_COLOURS["F.Cu"]
    tracks = _by_class(root, "track")
    assert len(tracks) == 2 and {t.get("data-layer") for t in tracks} == {"F.Cu", "B.Cu"}
    assert {t.get("stroke") for t in tracks} == set(COPPER_COLOURS.values())
    assert float(tracks[1].get("stroke-width")) == pytest.approx(0.25 * 18, abs=0.01)  # F.Cu drawn after B.Cu, width to scale
    vias = _by_class(root, "via")
    assert len(vias) == 1 and vias[0].get("data-net") == "VOUT" and list(vias[0])[1].get("class") == "drill"
    text = _text(root)
    assert "R1" in text and "R2" in text and "J1" in text and "10k" in text
    assert "30 × 20 mm" in text and "트랙 2개" in text and "비아 1개" in text and "F.Cu" in text
    assert "30 × 20 mm" in fig.caption and "트랙 2개" in fig.caption and "비아 1개" in fig.caption
    assert str(tmp_path) not in fig.svg and str(tmp_path) not in fig.caption and not ISO_RE.search(fig.svg)


def test_board_figure_placement_only_and_determinism(tmp_path: Path):
    ir, lib = _board(tmp_path)
    fig = board_figure(ir, lib, copper=False)
    assert fig.id == "placement"
    root = _parse(fig.svg)
    assert len(_by_class(root, "pad")) == 7 and not _by_class(root, "track") and not _by_class(root, "via")
    assert "동박 제외" in fig.caption and "동박 제외" in _text(root)
    assert fig.svg == board_figure(ir, lib, copper=False).svg
    assert board_figure(ir, lib).svg == board_figure(ir, KicadLibrary(roots=lib.roots)).svg
    # a wide board is scaled down to max_width, a narrow one keeps scale_px_per_mm
    ir.pcb.outline.width_mm = 200.0
    wide = _parse(board_figure(ir, lib).svg)
    assert int(wide.get("width")) == COLUMN_PX
    ir.pcb.outline.width_mm = 30.0
    assert int(_parse(board_figure(ir, lib, scale_px_per_mm=10).svg).get("width")) == 30 * 10 + 80


def test_board_figure_refuses_what_the_compiler_refuses(tmp_path: Path):
    ir, lib = _board(tmp_path)
    ir.pcb.placements = [p for p in ir.pcb.placements if p.component_ref != "R2"]
    with pytest.raises(ValueError, match="'R2' has no placement"):
        board_figure(ir, lib)
    ir, lib = _board(tmp_path)
    ir.component("R1").footprint = LibraryRef(library="Nope", name="Missing")
    with pytest.raises(ValueError, match="Nope:Missing of 'R1' was not found"):
        board_figure(ir, lib)
    ir, lib = _board(tmp_path)
    ir.component("R1").footprint = None
    with pytest.raises(ValueError, match="'R1' has no footprint"):
        board_figure(ir, lib)
    ir, lib = _board(tmp_path)
    ir.pcb.outline = None
    with pytest.raises(ValueError, match="outline"):
        board_figure(ir, lib)
    ir, lib = _board(tmp_path)
    ir.pcb.outline.height_mm = 0.0
    with pytest.raises(ValueError, match="positive size"):
        board_figure(ir, lib)


def test_long_values_are_respelled_or_shortened_on_the_board_but_kept_in_the_data_attribute(tmp_path: Path):
    """A long SPICE number keeps its scale letter (the one character that carries the unit) at 5 significant digits; a long non-number is cut; a short value is printed as written."""
    ir, lib = _board(tmp_path)
    ir.component("R1").value = "64.8172677616823n"
    ir.component("R2").value = "Conn_01x03_a_very_long_name"
    root = _parse(board_figure(ir, lib).svg)
    value = [t for t in _by_class(root, "value") if t.get("data-value") == "64.8172677616823n"]
    assert len(value) == 1 and value[0].text == "64.817n"
    long_name = [t for t in _by_class(root, "value") if t.get("data-value") == "Conn_01x03_a_very_long_name"]
    assert len(long_name) == 1 and long_name[0].text == "Conn_01x03_…"
    assert [t.text for t in _by_class(root, "value") if t.get("data-value") == "Conn_01x03"] == ["Conn_01x03"]
    ir.component("R1").value = "3.3000000000000003"
    root = _parse(board_figure(ir, lib).svg)
    assert [t.text for t in _by_class(root, "value") if t.get("data-value") == "3.3000000000000003"] == ["3.3"]


def test_board_caption_wraps_inside_the_svg_and_labels_wear_a_halo(tmp_path: Path):
    """On a 30 x 20 mm board the caption line is longer than the figure: it wraps into lines that fit, and the height grows with them; every ref / value label carries a surface-coloured halo so it stays legible over copper."""
    ir, lib = _board(tmp_path)
    for copper in (True, False):
        fig = board_figure(ir, lib, copper=copper)
        root = _parse(fig.svg)
        width = int(root.get("width"))
        captions = _by_class(root, "caption")
        assert len(captions) >= 2, "the caption needs more than one line at this width"
        for t in captions:
            assert figures._text_width(t.text, 11) <= width - 2 * figures._BOARD_MARGIN
            assert float(t.get("y")) < int(root.get("height"))
        assert " ".join(t.text for t in captions).startswith(f"{ir.project.id}: 30 × 20 mm, 부품 3개")
        ys = [float(t.get("y")) for t in captions]
        assert ys == sorted(ys) and all(b - a == 14 for a, b in zip(ys, ys[1:]))
        for label in _by_class(root, "ref") + _by_class(root, "value"):
            assert label.get("paint-order") == "stroke" and label.get("stroke") == figures.BOARD_FILL and label.get("stroke-width") == "3"
    assert board_figure(ir, lib).svg == board_figure(ir, lib).svg


# --------------------------------------------------------------------------- tolerance_figure


def test_tolerance_figure_colours_by_status_and_marks_missing_and_clamped_rows():
    rows = [
        ToleranceRow("f_osc", 1000.0, "Hz", 1025.17, 100.0, "PASS"),
        ToleranceRow("out_high", 5.0, "V", 5.5, 0.25, "FAIL"),
        ToleranceRow("v_out_tau", 3.1606, "V", 3.2, 0.0632, "NOT_VERIFIED"),
        ToleranceRow("v_mid", 3.0, "V", None, 0.03, None),
        ToleranceRow("no_tol", 2.0, "V", 2.1, None, "UNRESOLVED"),
        ToleranceRow("low", 0.0, "V", -0.9, 0.25, "FAIL"),
        ToleranceRow("edge", 1.0, "V", 1.15, 0.1, "PASS"),
    ]
    fig = tolerance_figure(rows)
    assert fig.svg == tolerance_figure(rows).svg and fig.id == "tolerance"
    root = _parse(fig.svg)
    by_label = {r.get("data-label"): r for r in _by_class(root, "row")}
    assert set(by_label) == {r.label for r in rows}

    def marker(label: str) -> ET.Element | None:
        found = _by_class(by_label[label], "marker")
        return found[0] if found else None

    assert marker("f_osc").get("fill") == STATUS_COLOURS["PASS"]
    assert marker("out_high").get("fill") == STATUS_COLOURS["FAIL"]
    assert marker("v_out_tau").get("fill") == STATUS_OTHER
    # no measurement: the band (the tolerance exists) but no marker; no tolerance: neither
    assert marker("v_mid") is None and NO_MEASUREMENT in _text(by_label["v_mid"]) and len(_by_class(by_label["v_mid"], "band")) == 1
    assert marker("no_tol") is None and "허용치 없음" in _text(by_label["no_tol"]) and not _by_class(by_label["no_tol"], "band")
    # the deviation is (measured - nominal) / tolerance; beyond +-1.5 the marker is clamped to the edge and the label carries a ">" mark
    assert marker("f_osc").get("data-deviation") == "0.2517" and marker("f_osc").get("data-clamped") == "false"
    assert marker("out_high").get("data-deviation") == "2" and marker("out_high").get("data-clamped") == "true"
    assert marker("low").get("data-deviation") == "-3.6" and marker("low").get("data-clamped") == "true"
    assert marker("edge").get("data-deviation") == "1.5" and marker("edge").get("data-clamped") == "false"
    assert float(marker("out_high").get("cx")) == float(marker("edge").get("cx"))  # clamped onto the +1.5 edge
    ticks = {t.get("data-value"): float(t.get("x")) for t in _by_class(root, "tick")}
    assert float(marker("low").get("cx")) == ticks["-1.5"] and float(marker("edge").get("cx")) == ticks["1.5"]
    labels = {r: _by_class(by_label[r], "marker-label")[0].text for r in ("f_osc", "out_high", "low", "edge")}
    assert labels["f_osc"] == "측정 1.025 kHz / 공칭 1 kHz"
    assert labels["out_high"] == "측정 5.5 V / 공칭 5 V (편차 > 1.5 × 허용치)"
    assert labels["low"] == "측정 -900 mV / 공칭 0 V (편차 > 1.5 × 허용치)"
    assert ">" not in labels["edge"]
    # every status word is printed as text beside its row (colour never carries it alone), and the legend names the colours
    for r in rows:
        assert (r.status or "기록 없음") in _text(by_label[r.label])
    legend = _text(_by_class(root, "legend")[0])
    assert "허용치" in legend and "PASS" in legend and "FAIL" in legend and "기타" in legend
    assert "허용치 대비 편차" in _text(root)
    with pytest.raises(ValueError, match="at least one row"):
        tolerance_figure([])
    # the nominal line is painted per row after the band (over it) and before the marker and labels (under them)
    for label, row in by_label.items():
        children = list(row)
        kinds = [c.get("class") for c in children]
        assert kinds.count("nominal") == 1, label
        if "band" in kinds:
            assert kinds.index("band") < kinds.index("nominal")
        if "marker" in kinds:
            assert kinds.index("nominal") < kinds.index("marker")
        nominal = children[kinds.index("nominal")]
        assert nominal.get("x1") == nominal.get("x2") == f"{ticks['0']:.2f}"
    assert 'class="nominal"' not in fig.svg.split('<g class="rows">')[0]


def test_tolerance_figure_treats_a_recorded_zero_tolerance_as_a_real_limit():
    rows = [ToleranceRow("exact", 0.0, "V", 0.0, 0.0, "PASS", True), ToleranceRow("off", 0.0, "V", 0.0295, 0.0, "FAIL", True), ToleranceRow("none", 2.0, "V", 2.1, None, "UNRESOLVED", True)]
    fig = tolerance_figure(rows)
    root = _parse(fig.svg)
    by_label = {r.get("data-label"): r for r in _by_class(root, "row")}
    exact = _by_class(by_label["exact"], "marker")[0]
    assert exact.get("data-deviation") == "0" and exact.get("data-clamped") == "false" and exact.get("fill") == STATUS_COLOURS["PASS"]
    off = _by_class(by_label["off"], "marker")[0]
    assert off.get("data-deviation") == "inf" and off.get("data-clamped") == "true" and off.get("fill") == STATUS_COLOURS["FAIL"]
    assert _by_class(by_label["off"], "marker-label")[0].text == f"측정 29.5 mV / 공칭 0 V ({ZERO_TOLERANCE}) (편차 > 1.5 × 허용치)"
    assert not _by_class(by_label["off"], "band") and not _by_class(by_label["exact"], "band")
    assert "허용치 없음" in _text(by_label["none"]) and "허용치 없음" not in _text(by_label["off"])
    assert "그대로이며 그림은 점의 위치 외에 아무것도 다시 계산하지 않습니다" in fig.caption
    # a row whose tolerance had to be computed from the IR says so in the caption
    mixed = tolerance_figure(rows + [ToleranceRow("ir", 1.0, "V", 1.05, 0.1, "PASS", False)])
    assert "기록이 없는 기대값만 SPICE 단계의 규칙" in mixed.caption


def test_tolerance_rows_come_from_the_recorded_spice_results(tmp_path: Path):
    lib = template_library(tmp_path / "kicad")
    ir = divider_with_connector_ir(tmp_path / "proj", library=lib)
    rows = tolerance_rows(ir)
    assert [r.label for r in rows] == ["v_out", "v_out_mid"]
    assert rows[0] == ToleranceRow("v_out", 6.0, "V", None, pytest.approx(0.06), None)  # 1 % of 6 V, nothing measured yet
    ir.validation.results.append(ValidationResult(check_id="spice.v_out", status=ValidationStatus.PASS, message="", details={"measured": 6.02}))
    ir.validation.results.append(ValidationResult(check_id="spice.v_out_mid", status=ValidationStatus.FAIL, message="", details={"measured": 3.5}))
    before = ir.model_dump_json()
    rows = tolerance_rows(ir)
    assert rows[0].measured == 6.02 and rows[0].status == "PASS" and rows[1].measured == 3.5 and rows[1].status == "FAIL"
    assert ir.model_dump_json() == before
    fig = tolerance_figure(rows)
    assert [m.get("fill") for m in _by_class(_parse(fig.svg), "marker")] == [STATUS_COLOURS["PASS"], STATUS_COLOURS["FAIL"]]
    # a relative tolerance on a nominal of 0 is no tolerance (the SPICE stage's rule)
    ir.simulation.expectations[0].nominal.value = 0.0
    assert tolerance_rows(ir)[0].tolerance is None
    ir.simulation = None
    assert tolerance_rows(ir) == []


def test_tolerance_rows_draw_the_recorded_limit_not_a_tolerance_edited_after_the_run(tmp_path: Path):
    """The band is the limit the verdict was judged against: a tol_rel tightened in ir.json after the run does not move it; a recorded 0 stays 0 and a recorded None stays None."""
    lib = template_library(tmp_path / "kicad")
    ir = divider_with_connector_ir(tmp_path / "proj", library=lib)
    e = ir.simulation.expectations[0]
    ir.validation.results.append(ValidationResult(check_id=f"spice.{e.id}", status=ValidationStatus.PASS, message="", details={"measured": 6.02, "nominal": 6.0, "tolerance": 0.06, "deviation": 0.02}))
    row = tolerance_rows(ir)[0]
    assert row == ToleranceRow(e.id, 6.0, "V", 6.02, 0.06, "PASS", True)
    e.tol_rel.value = 0.001  # a post-run edit: the stage's rule would now give 0.006 and a PASS marker clamped outside the band
    assert tolerance_rows(ir)[0] == row
    assert expectation_limit(e, {"tolerance": 0.06}) == (0.06, True) and expectation_limit(e, {"measured": 6.02}) == (pytest.approx(0.006), False)
    assert expectation_limit(e, None) == (pytest.approx(0.006), False) and expectation_limit(e, {"tolerance": None}) == (None, True)
    fig = tolerance_figure(tolerance_rows(ir))
    marker = _by_class(_parse(fig.svg), "marker")[0]
    assert marker.get("data-clamped") == "false" and marker.get("data-deviation") == "0.3333" and "그대로이며" in fig.caption
    # a recorded tolerance of 0 is a real limit (judge() keeps it), never "허용치 없음"; a recorded None is none
    ir.validation.results.append(ValidationResult(check_id=f"spice.{e.id}", status=ValidationStatus.FAIL, message="", details={"measured": 6.02, "nominal": 6.0, "tolerance": 0.0}))
    assert tolerance_rows(ir)[0].tolerance == 0.0 and ZERO_TOLERANCE in tolerance_figure(tolerance_rows(ir)).svg
    ir.validation.results.append(ValidationResult(check_id=f"spice.{e.id}", status=ValidationStatus.UNRESOLVED, message="", details={"measured": 6.02, "nominal": 6.0, "tolerance": None}))
    assert tolerance_rows(ir)[0].tolerance is None and tolerance_rows(ir)[0].tolerance_recorded
    # a recorded nominal wins over the IR's for the marker position; the IR's is used only without one
    ir.validation.results.append(ValidationResult(check_id=f"spice.{e.id}", status=ValidationStatus.PASS, message="", details={"measured": 6.02, "nominal": 6.01, "tolerance": 0.06}))
    assert tolerance_rows(ir)[0].nominal == 6.01
    ir.validation.results.append(ValidationResult(check_id=f"spice.{e.id}", status=ValidationStatus.PASS, message="", details={"measured": 6.02}))
    assert tolerance_rows(ir)[0] == ToleranceRow(e.id, 6.0, "V", 6.02, pytest.approx(0.006), "PASS", False)


# --------------------------------------------------------------------------- bar_figure


def test_bar_figure_labels_every_bar():
    labels = ["VCC", "Q1_C", "OUT & <x>", "GND"]
    values = [74.21, 51.8, 35.23, 0.0]
    fig = bar_figure(labels, values, title="네트별 동박 길이", y_label="동박 길이 (mm)", unit="mm")
    assert fig.id == "bars" and fig.svg == bar_figure(labels, values, title="네트별 동박 길이", y_label="동박 길이 (mm)", unit="mm").svg
    root = _parse(fig.svg)
    bars = _by_class(root, "bar")
    assert [b.get("data-label") for b in bars] == labels and all(b.get("fill") == SERIES_COLOURS[0] for b in bars)
    texts = [t.text for t in _by_class(root, "bar-label")]
    assert texts == ["74.21 mm", "51.8 mm", "35.23 mm", "0 mm"]
    assert [t.text for t in _by_class(root, "category")] == labels and "&lt;x&gt;" in fig.svg
    assert "동박 길이 (mm)" in _text(root) and not _by_class(root, "legend")
    assert "VCC 74.21 mm" in fig.caption
    with pytest.raises(ValueError, match="equal length"):
        bar_figure(["a"], [1.0, 2.0], title="t", y_label="y")
    with pytest.raises(ValueError, match="equal length"):
        bar_figure([], [], title="t", y_label="y")
    with pytest.raises(ValueError, match="non-finite"):
        bar_figure(["a"], [math.inf], title="t", y_label="y")
    # many bars still fit max_width
    many = bar_figure([f"N{i}" for i in range(30)], [float(i) for i in range(30)], title="t", y_label="y (mm)", unit="mm")
    assert int(_parse(many.svg).get("width")) == COLUMN_PX and len(_by_class(_parse(many.svg), "bar")) == 30


def test_figures_are_authored_at_the_column_width_with_readable_text(tmp_path: Path):
    """The page renders a figure at min(1, 672 / width): every default figure is at most 672 px wide, so its text is never scaled below the authored size, and nothing is authored below 10 px."""
    ir, lib = _board(tmp_path)
    ir.pcb.outline.width_mm = 200.0
    svgs = [
        svg_line_chart([_series(20, "a"), _series(20, "b")], title="t", x_label="시간 (s)", y_label="전압 (V)"),
        waveform_figure(_results(), "tran", ["v(OUT)"], title="t").svg,
        board_figure(ir, lib).svg,
        tolerance_figure([ToleranceRow("f_osc", 1000.0, "Hz", 1025.17, 100.0, "PASS")]).svg,
        bar_figure([f"N{i}" for i in range(30)], [float(i) for i in range(30)], title="t", y_label="y (mm)", unit="mm").svg,
    ]
    for svg in svgs:
        root = _parse(svg)
        width = int(root.get("width"))
        assert width <= COLUMN_PX
        scale = min(1.0, COLUMN_PX / width)
        for el in root.iter():
            if el.get("font-size") is not None:
                assert float(el.get("font-size")) * scale >= 10.0, el.get("class")
        assert float(root.get("font-size")) * scale >= 12.0  # the tick labels inherit 12 px
