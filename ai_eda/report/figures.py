"""Deterministic inline-SVG figures for the Korean stage reports (stdlib only, pure functions).

Invariant: a figure is a *view* under the same rule as the reports it goes
into. Every builder here reads its inputs (an IR, a KiCad library, the
``spice/results.json`` a run wrote, rows already computed from
``ir.validation``) and returns text; it computes no status, registers no
artifact, saves nothing and mutates nothing. The SVG carries no wall-clock
and no absolute path, and the same inputs give byte-identical output (no
randomness, fixed-precision numbers, dictionary order preserved from the
caller). Every string that reaches the markup passes through :func:`esc`.

Chart rules (from the data-visualization skill, applied throughout):

* the form follows the data's job - change over time is a line
  (:func:`svg_line_chart`, :func:`waveform_figure`), magnitude per category
  a bar (:func:`bar_figure`), a measurement against a tolerance a band with
  a marker (:func:`tolerance_figure`); one y axis per chart, never two
  (branch currents get their own chart, :func:`waveform_figures`);
* categorical hues in the fixed order :data:`SERIES_COLOURS`, never cycled,
  at most :data:`MAX_SERIES` series per chart (a 5th is a ``ValueError``);
  status colours (:data:`STATUS_COLOURS`) are reserved for a recorded status
  and always ship with the status word as text; text is never coloured in a
  series colour - a swatch beside it carries identity;
* surface :data:`SURFACE`, ink :data:`INK` / :data:`INK_SECONDARY`, a thin
  recessive grid behind the marks, 2 px lines, >= 8 px markers with a
  surface ring, thin bars, a legend for >= 2 series and none for one (the
  title names it), direct labels only where they matter, axis titles with
  units and SI-prefixed tick labels (:func:`si_format`, with enough digits
  to tell neighbouring ticks apart), 5-6 "nice" ticks per axis (1-2-5
  steps, decades on a log axis; a linear x axis whose limit lies a hair
  inside a tick is extended to that tick so the axis reads from a round
  number), log axes refusing non-positive values, series downsampled
  deterministically to at most :data:`MAX_POINTS` points (every k-th
  sample, first and last kept);
* every figure is authored at the report page's column width
  (:data:`COLUMN_PX`, 178 mm at 96 dpi) so the HTML / PDF renders it 1:1:
  the sizes above are what the reader sees, not a scaled-down copy;
* a page holds several figures in one id space, so the clip path of a
  chart carries a deterministic id of its own (``clip_id``, derived from
  the figure id or from the chart's own text and rect), never a shared one.

Board figures read pad geometry only through the KiCad library on disk
(:mod:`ai_eda.tools.kicad.geometry` places it), exactly as the PCB compiler
does: a footprint that is not in a library, a component without a placement
or a board without an outline is a ``ValueError``, never a guess. The
silkscreen on the board figure is the footprints' library silk graphics and
the IR's silk texts at their IR positions (:func:`board_figure`), the 3D
preview figure (:func:`model3d_figure`) the scene of
:mod:`ai_eda.tools.model3d` (body heights from the STEP files the footprints
name; a part without one is a flat outline, said in the caption).
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import EllipsisType
from typing import Any
from xml.sax.saxutils import escape as _xml_escape

from ai_eda.ir import CircuitIR, Expectation, SilkKind, ValidationStatus
from ai_eda.tools.calc.si import parse_spice_number
from ai_eda.tools.kicad import sexpr
from ai_eda.tools.kicad.geometry import footprint_bbox, mirrored_layer, pad_angle, pad_center, pad_copper_center, pad_layers, to_board
from ai_eda.tools.kicad.library import FootprintDef, KicadLibrary, Pad
from ai_eda.tools.model3d.iso import iso_svg
from ai_eda.tools.model3d.scene import GRAPHIC_HEADS, Scene, build_scene, graphic_paths, scene_caption, stroke_width
from ai_eda.tools.silkscreen.geometry import SILK_LAYERS, Shape, footprint_silk, ir_text_box, silk_text_problems
from ai_eda.tools.spice.rawfile import canonical_name
from ai_eda.tools.spice.stage import CHECK_ID as SPICE_CHECK

__all__ = [
    "AXIS",
    "BAND_FILL",
    "Band",
    "COLUMN_PX",
    "COPPER_COLOURS",
    "Figure",
    "GRID",
    "INK",
    "INK_SECONDARY",
    "MAX_POINTS",
    "MAX_SERIES",
    "Marker",
    "NO_MEASUREMENT",
    "SERIES_COLOURS",
    "SILK_COLOUR",
    "SILK_LAYER_ORDER",
    "NET_CLASS_DEFAULT",
    "NET_CLASS_OTHER",
    "NEUTRAL_PAD",
    "CLASS_COLOURS",
    "CLASS_KEY",
    "NetClassStyle",
    "net_class_styles",
    "STATUS_COLOURS",
    "STATUS_OTHER",
    "SURFACE",
    "Series",
    "THT_COLOUR",
    "ToleranceRow",
    "VIA_COLOUR",
    "bar_figure",
    "board_figure",
    "downsample",
    "esc",
    "expectation_limit",
    "layer_class",
    "log_ticks",
    "model3d_figure",
    "model3d_figure_from_scene",
    "nice_ticks",
    "plot_vector",
    "si_format",
    "svg_line_chart",
    "tolerance_figure",
    "tolerance_rows",
    "vector_kind",
    "waveform_figure",
    "waveform_figures",
]

# --------------------------------------------------------------------------- palette (light mode, validated)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
#: hairline grid, behind the marks
GRID = "#e6e4dc"
#: axis rules
AXIS = "#c3c2b7"
#: categorical hues in their fixed order: blue, orange, aqua, yellow, magenta, green, violet, red
SERIES_COLOURS: tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
#: series per chart; more must go into another chart
MAX_SERIES = 4
#: the lightest step of the sequential blue ramp: the tolerance band
BAND_FILL = "#cde2fb"
#: status colours are reserved for a recorded ValidationStatus and always come with the status word as text
STATUS_COLOURS: dict[str, str] = {ValidationStatus.PASS.value: "#008300", ValidationStatus.FAIL.value: "#e34948"}
STATUS_OTHER = "#8a8983"
#: copper by layer on the board figure
COPPER_COLOURS: dict[str, str] = {"F.Cu": "#e34948", "B.Cu": "#2a78d6"}
OTHER_COPPER = "#8a8983"
THT_COLOUR = "#eda100"
VIA_COLOUR = "#8a8983"
DRILL_COLOUR = "#ffffff"
BOARD_FILL = "#f3f1ea"
#: silkscreen strokes and texts on the board figure: ink on the light board, at their true stroke width / text height
SILK_COLOUR = INK
#: the board figure's class view (``colour_by="class"``) and the GUI's net-class toggle: the default class is neutral, every other
#: class of ``ir.si`` takes the next hue of :data:`CLASS_COLOURS` (a class beyond them: :data:`NET_CLASS_OTHER`)
NET_CLASS_DEFAULT = "#8a8983"
NET_CLASS_OTHER = INK_SECONDARY
#: SMD pads in the class view (the layer colours would read as classes)
NEUTRAL_PAD = "#b9b7ae"
#: the class hues: :data:`SERIES_COLOURS` in order without the colours the class view keeps for something else (the through-hole
#: pads' yellow; the vias and the default class share the neutral gray, so the class view draws a via as a white disc in a gray ring)
CLASS_COLOURS: tuple[str, ...] = tuple(c for c in SERIES_COLOURS if c not in (THT_COLOUR, VIA_COLOUR, NET_CLASS_DEFAULT, NEUTRAL_PAD))
#: a via in the class view: a white disc with this ring (never the default class's gray fill)
CLASS_VIA_RING = INK_SECONDARY
#: the class view's key (the report figure's caption, and the GUI's class caption over the layer figure)
CLASS_KEY = ("선 색 = 넷 클래스 (범례, 괄호 안은 넷 수), 진한 선 = F.Cu, 옅은 선 = B.Cu, 회색 패드 = SMD, 노랑 = 관통 패드(흰 원 = 드릴), "
             "회색 테의 흰 원 = 비아")
#: the hatch of an inner plane zone on the board figure: thin ink lines at a low opacity, one angle per inner layer (never a fill)
PLANE_HATCH = INK_SECONDARY
PLANE_HATCH_ANGLES: tuple[float, ...] = (45.0, -45.0, 0.0, 90.0)
#: the silk layer groups of the board figure, back first (the front is drawn over it)
SILK_LAYER_ORDER: tuple[str, ...] = ("B.SilkS", "F.SilkS")
#: points per series after :func:`downsample`
MAX_POINTS = 2000
#: the width every figure is authored at: the report page's 178 mm column at 96 dpi, so the HTML / PDF shows it 1:1
COLUMN_PX = 672
#: what the tolerance figure prints for a row without a measured value (the final report's wording)
NO_MEASUREMENT = "측정 없음"
NO_TOLERANCE = "허용치 없음"
#: what the tolerance figure prints for a recorded tolerance of exactly 0 (a real limit the stage judged against: only an exact match passes)
ZERO_TOLERANCE = "허용치 0"
#: a linear axis limit closer than this fraction of the span to the next tick outside is extended to that tick
_SNAP_FRACTION = 0.01
#: characters allowed in an SVG / HTML id as we write them (anything else is replaced and the id gets a hash suffix)
_ID_SAFE_RE = re.compile(r"[^A-Za-z0-9_-]")

_FONT = '"Noto Sans CJK KR", "Noto Sans KR", "Malgun Gothic", "Apple SD Gothic Neo", "WenQuanYi Zen Hei", system-ui, sans-serif'
_SVG_NS = "http://www.w3.org/2000/svg"
_TITLE_PX = 15
_TEXT_PX = 12
_SMALL_PX = 11

# --------------------------------------------------------------------------- data


@dataclass(frozen=True)
class Series:
    """One line of a chart: ``name`` (shown in the legend / data attribute), ``xs`` and ``ys`` of equal length."""

    name: str
    xs: Sequence[float]
    ys: Sequence[float]


@dataclass(frozen=True)
class Band:
    """A shaded range on one axis (``axis`` ``"x"`` or ``"y"``), e.g. a validity window; ``lo == hi`` draws a labelled guide line (a threshold)."""

    axis: str
    lo: float
    hi: float
    label: str = ""


@dataclass(frozen=True)
class Marker:
    """A labelled point (a design point, a crossing); ``colour`` defaults to ink so it stands out from every series."""

    x: float
    y: float
    label: str = ""
    colour: str | None = None


@dataclass(frozen=True)
class Figure:
    """A finished figure: ``id`` (the Markdown placeholder ``![fig](fig:<id>)``), ``title``, a Korean ``caption`` and the inline ``svg``."""

    id: str
    title: str
    caption: str
    svg: str


@dataclass(frozen=True)
class ToleranceRow:
    """One expectation of the tolerance figure.

    ``tolerance`` is the limit the SPICE stage recorded in
    ``details["tolerance"]`` when the recorded result carries that key
    (``None`` recorded means no usable tolerance and stays ``None``); an
    expectation without such a record falls back to the stage's own rule on
    the IR (``max(tol_abs, tol_rel * |nominal|)``, ``tolerance_recorded``
    ``False``) so the figure can still show the band. ``nominal`` is the
    recorded ``details["nominal"]`` when there is one, else the IR's.
    ``measured`` is the recorded ``details["measured"]`` (``None`` without a
    measurement), ``status`` the recorded status word (``None`` without a
    record). The figure computes nothing beyond the position of the marker
    from these numbers.
    """

    label: str
    nominal: float
    unit: str | None = None
    measured: float | None = None
    tolerance: float | None = None
    status: str | None = None
    tolerance_recorded: bool = False


# --------------------------------------------------------------------------- text and numbers


def esc(text: object) -> str:
    """The one escaping function for SVG text and attribute values (``&``, ``<``, ``>``, ``"``)."""
    return _xml_escape(str(text), {'"': "&quot;"})


def _f(value: float, decimals: int = 2) -> str:
    """A coordinate with fixed decimals; ``-0.00`` normalised to ``0.00`` so two builds never differ in a sign of zero."""
    text = f"{value:.{decimals}f}"
    if text.startswith("-") and float(text) == 0.0:
        text = text[1:]
    return text


def _g(value: float, digits: int = 6) -> str:
    """A number for a label: ``digits`` significant digits, no exponent notation below 1e6 (floating noise such as 0.30000000000000004 disappears)."""
    text = f"{value:.{digits}g}"
    if "e" in text:
        if abs(value) >= 1:
            text = f"{value:.0f}"
        else:
            text = f"{value:.{digits + 6}f}".rstrip("0").rstrip(".")
    if text.startswith("-") and float(text) == 0.0:
        text = text[1:]
    return text


_SI_PREFIXES: dict[int, str] = {-15: "f", -12: "p", -9: "n", -6: "µ", -3: "m", 0: "", 3: "k", 6: "M", 9: "G"}
#: units that take an SI prefix on an axis; ``mm``, ``%``, ``dB`` and a unitless number print plain
_PREFIXED_UNITS = frozenset({"s", "V", "A", "Hz", "F", "Ω", "ohm", "H", "W"})
_UNIT_DISPLAY = {"ohm": "Ω"}


def si_exponent(magnitude: float) -> int:
    """The engineering exponent (a multiple of 3 in [-15, 9]) for ``magnitude``; 0 for zero and non-finite values."""
    if magnitude == 0 or not math.isfinite(magnitude):
        return 0
    e3 = int(math.floor(math.log10(abs(magnitude)) / 3.0)) * 3
    return max(-15, min(9, e3))


def si_format(value: float, unit: str | None = "", *, exponent: int | None = None, digits: int = 4) -> str:
    """``value`` with an SI prefix and the unit (``0.0015, "s"`` -> ``1.5 ms``; ``1500, "Hz"`` -> ``1.5 kHz``).

    ``exponent`` fixes the prefix (an axis formats all its ticks with the
    prefix of its largest value, so ``0 ms``, ``0.5 ms``, ``1 ms`` line up).
    A unit outside :data:`_PREFIXED_UNITS` (``mm``, ``%``) and a unitless
    value print plain (``66.5 mm``); zero prints ``0`` with the unit.
    """
    u = _UNIT_DISPLAY.get(unit or "", unit or "")
    if not u or u not in _PREFIXED_UNITS or not math.isfinite(value):
        text = _g(value, digits)
        return f"{text} {u}" if u else text
    if value == 0:
        return f"0 {_SI_PREFIXES[exponent] if exponent is not None else ''}{u}"
    e3 = si_exponent(value) if exponent is None else exponent
    return f"{_g(value / 10 ** e3, digits)} {_SI_PREFIXES[e3]}{u}"


def _text_width(text: str, px: float) -> float:
    """A conservative width estimate (CJK characters count as one em, everything else 0.6 em) for placing labels."""
    return sum(px if ord(ch) > 0x2E7F else 0.6 * px for ch in text)


def _shorten(text: str, limit: int) -> str:
    """``text`` cut to ``limit`` characters with an ellipsis (a board label; the full value stays in the data attribute)."""
    return text if len(text) <= limit else text[: limit - 1] + "…"


_Box = tuple[float, float, float, float]


class _LabelPlacer:
    """Deterministic label placement: each label takes the candidate spot that overlaps no label placed before it and lies farthest from the drawn series.

    ``points`` are pixel positions of the series (thinned for speed);
    candidates are ``(x, y, text-anchor)``; a candidate that would leave the
    canvas is dropped (the first one is kept when all would). Ties keep the
    first candidate, so the output never depends on anything but the input.
    """

    def __init__(self, points: list[tuple[float, float]], width: int, height: int, px: float = _TEXT_PX) -> None:
        step = max(1, len(points) // 400)
        self.points = points[::step]
        self.width, self.height, self.px = width, height, px
        self.boxes: list[_Box] = []

    def _box(self, text: str, x: float, y: float, anchor: str) -> _Box:
        w = _text_width(text, self.px)
        x0 = x - w if anchor == "end" else (x - w / 2 if anchor == "middle" else x)
        return (x0, y - self.px, x0 + w, y + 3)

    @staticmethod
    def _overlaps(a: _Box, b: _Box) -> bool:
        return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]

    def _distance(self, box: _Box) -> float:
        best = math.inf
        for x, y in self.points:
            dx = max(box[0] - x, 0.0, x - box[2])
            dy = max(box[1] - y, 0.0, y - box[3])
            d = math.hypot(dx, dy)
            if d < best:
                best = d
                if best == 0.0:
                    break
        return best

    def place(self, text: str, candidates: Sequence[tuple[float, float, str]]) -> tuple[float, float, str]:
        inside = [c for c in candidates if (b := self._box(text, *c))[0] >= 2 and b[2] <= self.width - 2 and b[1] >= 0 and b[3] <= self.height]
        options = inside or list(candidates[:1])
        best: tuple[float, float, str] | None = None
        best_key: tuple[int, float] | None = None
        for cand in options:
            box = self._box(text, *cand)
            key = (sum(1 for b in self.boxes if self._overlaps(box, b)), -self._distance(box))
            if best_key is None or key < best_key:
                best, best_key = cand, key
        assert best is not None
        self.boxes.append(self._box(text, *best))
        return best


# --------------------------------------------------------------------------- ticks and sampling


def nice_ticks(lo: float, hi: float, target: int = 5) -> list[float]:
    """About ``target`` ticks in ``[lo, hi]`` at a 1-2-5 step (``0, 0.001, 0.002 ...``); ``lo == hi`` gives that one tick."""
    if not (math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError(f"tick range must be finite (got {lo!r}, {hi!r})")
    if hi < lo:
        lo, hi = hi, lo
    if hi == lo:
        return [lo + 0.0]
    span = hi - lo
    mag = 10.0 ** math.floor(math.log10(span))
    # the largest 1-2-5 step that still gives at least ``target - 1`` ticks (5-6 ticks for the usual span)
    candidates = sorted(m * mag * 10.0**k for k in (-2, -1, 0) for m in (1.0, 2.0, 5.0))
    step = candidates[0]
    for cand in candidates:
        count = math.floor(hi / cand + 1e-9) - math.ceil(lo / cand - 1e-9) + 1
        if count >= max(2, target - 1):
            step = cand
    first = math.ceil(lo / step - 1e-9)
    last = math.floor(hi / step + 1e-9)
    decimals = max(0, -int(math.floor(math.log10(step))) + 1)
    return [round(k * step, decimals + 6) + 0.0 for k in range(first, last + 1)]


def log_ticks(lo: float, hi: float) -> list[float]:
    """Decade ticks in ``[lo, hi]`` (``100, 1000, 10000``); the 1-2-5 points of those decades when fewer than three decades fall inside; linear :func:`nice_ticks` on a range narrower than that."""
    if lo <= 0 or hi <= 0:
        raise ValueError(f"a log axis needs positive limits (got {lo!r}, {hi!r})")
    if hi < lo:
        lo, hi = hi, lo
    d0, d1 = math.floor(math.log10(lo)), math.ceil(math.log10(hi))
    inside = lambda v: lo * (1 - 1e-9) <= v <= hi * (1 + 1e-9)  # noqa: E731
    decades = [10.0**d for d in range(d0, d1 + 1) if inside(10.0**d)]
    if len(decades) >= 3:
        return decades
    fine = [m * 10.0**d for d in range(d0, d1 + 1) for m in (1.0, 2.0, 5.0) if inside(m * 10.0**d)]
    return fine if len(fine) >= 3 else nice_ticks(lo, hi)


def downsample(xs: Sequence[float], ys: Sequence[float], cap: int = MAX_POINTS) -> tuple[list[float], list[float]]:
    """Every k-th sample so that at most ``cap`` remain; the first and the last sample are always kept. Deterministic."""
    n = len(xs)
    if n != len(ys):
        raise ValueError(f"xs and ys differ in length ({n} vs {len(ys)})")
    if cap < 2:
        raise ValueError("cap must be at least 2")
    if n <= cap:
        return list(xs), list(ys)
    k = math.ceil((n - 1) / (cap - 1))
    idx = list(range(0, n, k))
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return [xs[i] for i in idx], [ys[i] for i in idx]


# --------------------------------------------------------------------------- the line chart


def _check_series(series: Sequence[Series], log_x: bool, log_y: bool) -> None:
    if not series:
        raise ValueError("a chart needs at least one series")
    if len(series) > MAX_SERIES:
        raise ValueError(f"at most {MAX_SERIES} series per chart (got {len(series)}); put the rest into another chart")
    for s in series:
        if len(s.xs) != len(s.ys):
            raise ValueError(f"series {s.name!r}: xs and ys differ in length ({len(s.xs)} vs {len(s.ys)})")
        if not s.xs:
            raise ValueError(f"series {s.name!r} has no points")
        for i, (x, y) in enumerate(zip(s.xs, s.ys)):
            if not (math.isfinite(x) and math.isfinite(y)):
                raise ValueError(f"series {s.name!r} has a non-finite value at index {i} ({x!r}, {y!r})")
            if log_x and x <= 0:
                raise ValueError(f"log x axis: series {s.name!r} has a non-positive x value at index {i} ({x!r}); a log axis cannot show it")
            if log_y and y <= 0:
                raise ValueError(f"log y axis: series {s.name!r} has a non-positive y value at index {i} ({y!r}); a log axis cannot show it")


def _axis_range(values: list[float], log: bool, *, pad: float) -> tuple[float, float]:
    """``[lo, hi]`` around ``values`` with ``pad`` (a fraction of the span) on both ends; in log10 space on a log axis."""
    if log:
        bad = [v for v in values if v <= 0]
        if bad:
            raise ValueError(f"log axis: a band or marker has a non-positive value ({bad[0]!r}); a log axis cannot show it")
        values = [math.log10(v) for v in values]
    lo, hi = min(values), max(values)
    span = hi - lo
    if span == 0:
        span = abs(lo) * 0.2 if lo != 0 else (0.5 if log else 1.0)
    lo -= span * pad
    hi += span * pad
    if log:
        return 10.0 ** lo, 10.0 ** hi
    return lo, hi


class _Scale:
    """Pixel mapping of one axis (linear or log10)."""

    def __init__(self, lo: float, hi: float, p0: float, p1: float, log: bool) -> None:
        self.lo, self.hi, self.p0, self.p1, self.log = lo, hi, p0, p1, log
        self._f = math.log10 if log else (lambda v: v)
        self._flo = self._f(lo)
        self._span = self._f(hi) - self._flo or 1.0

    def __call__(self, v: float) -> float:
        return self.p0 + (self._f(v) - self._flo) / self._span * (self.p1 - self.p0)

    def ticks(self) -> list[float]:
        return log_ticks(self.lo, self.hi) if self.log else nice_ticks(self.lo, self.hi)


def _tick_labels(ticks: list[float], unit: str, log: bool) -> list[str]:
    """The tick texts of one axis: a shared SI prefix on a linear axis and enough significant digits that neighbouring ticks never read the same (``4.9999 V``, ``4.99995 V`` … on a 5 V axis with a 50 µV step)."""
    if log or not ticks:
        return [si_format(t, unit) for t in ticks]
    biggest = max(abs(t) for t in ticks)
    exponent = si_exponent(biggest or 1.0)
    digits = 4
    if len(ticks) > 1:
        step = min(b - a for a, b in zip(ticks, ticks[1:]))
        if step > 0 and biggest > 0:
            digits = max(4, math.ceil(math.log10(biggest / step)) + 1)
    return [si_format(t, unit, exponent=exponent, digits=digits) for t in ticks]


def _snap_to_ticks(lo: float, hi: float, ticks: list[float]) -> tuple[float, float, list[float]]:
    """Extend a linear axis limit outward to the adjacent tick when that tick lies within :data:`_SNAP_FRACTION` of the span; the tick step is kept.

    ngspice stores its first transient point one step after ``tstart``, so
    a 10 ms … 20 ms window starts at 10.0007 ms and would otherwise lose its
    10 ms tick and label; an axis that ends well short of the next tick is
    left alone.
    """
    if len(ticks) < 2:
        return lo, hi, ticks
    step = ticks[1] - ticks[0]
    span = hi - lo
    decimals = max(0, -int(math.floor(math.log10(step))) + 1) + 6
    ticks = list(ticks)
    below = round(ticks[0] - step, decimals) + 0.0
    if lo > below and lo - below <= _SNAP_FRACTION * span:
        lo, ticks = below, [below, *ticks]
    above = round(ticks[-1] + step, decimals) + 0.0
    if hi < above and above - hi <= _SNAP_FRACTION * span:
        hi, ticks = above, [*ticks, above]
    return lo, hi, ticks


def _clip_id(explicit: str | None, key: str) -> str:
    """A deterministic clip-path id: ``plot-<explicit>`` (sanitised, a hash suffix when a character had to be replaced) or ``plot-<hash of key>``.

    Several charts share one HTML id space on a report page, and
    ``url(#…)`` resolves to the first element with that id in the document,
    so a shared id would clip every later chart with the first chart's
    rectangle.
    """
    if explicit:
        safe = _ID_SAFE_RE.sub("-", explicit)
        if safe == explicit:
            return f"plot-{safe}"
        return f"plot-{safe}-{hashlib.sha256(explicit.encode('utf-8')).hexdigest()[:8]}"
    return "plot-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:10]


def _unit_of(label: str) -> str:
    """The unit an axis title carries in its trailing parentheses (``"시간 (s)"`` -> ``"s"``), used for the tick labels."""
    if label.endswith(")") and "(" in label:
        return label[label.rfind("(") + 1 : -1].strip()
    return ""


def _svg_open(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="{_SVG_NS}" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" '
        f'font-family=\'{_FONT}\' font-size="{_TEXT_PX}" fill="{INK}">',
        f"<title>{esc(title)}</title>",
        f'<rect class="surface" x="0" y="0" width="{width}" height="{height}" fill="{SURFACE}"/>',
    ]


def _title_text(title: str, x: float, y: float) -> str:
    return f'<text class="title" x="{_f(x)}" y="{_f(y)}" font-size="{_TITLE_PX}" font-weight="600">{esc(title)}</text>'


def _legend(entries: list[tuple[str, str, str]], x0: float, y: float, x_max: float) -> tuple[list[str], float]:
    """A legend row of ``(kind, colour, label)`` entries (``kind`` ``line`` / ``dot`` / ``box``) starting at ``x0``; wraps before ``x_max``. Returns markup and the next free y."""
    out = ['<g class="legend">']
    x = x0
    for kind, colour, label in entries:
        w = 24 + _text_width(label, _TEXT_PX) + 18
        if x > x0 and x + w > x_max:
            x = x0
            y += 18
        if kind == "line":
            out.append(f'<line x1="{_f(x)}" y1="{_f(y - 4)}" x2="{_f(x + 18)}" y2="{_f(y - 4)}" stroke="{colour}" stroke-width="2" stroke-linecap="round"/>')
        elif kind == "dot":
            out.append(f'<circle cx="{_f(x + 9)}" cy="{_f(y - 4)}" r="5" fill="{colour}" stroke="{SURFACE}" stroke-width="2"/>')
        else:
            out.append(f'<rect x="{_f(x)}" y="{_f(y - 10)}" width="18" height="12" fill="{colour}"/>')
        out.append(f'<text x="{_f(x + 24)}" y="{_f(y)}" fill="{INK_SECONDARY}">{esc(label)}</text>')
        x += w
    out.append("</g>")
    return out, y + 18


def svg_line_chart(
    series: Sequence[Series],
    *,
    title: str,
    x_label: str,
    y_label: str,
    log_x: bool = False,
    log_y: bool = False,
    bands: Sequence[Band] = (),
    markers: Sequence[Marker] = (),
    width: int = COLUMN_PX,
    height: int = 440,
    clip_id: str | None = None,
) -> str:
    """A line chart of at most :data:`MAX_SERIES` series as one deterministic SVG string.

    ``x_label`` / ``y_label`` are the axis titles; a trailing unit in
    parentheses (``"시간 (s)"``) is also used for the SI-prefixed tick labels.
    ``bands`` shade x- or y-ranges (a validity window) or, with ``lo == hi``,
    draw a labelled guide line; ``markers`` are labelled points. A legend is
    drawn for two or more series only. ``clip_id`` names the chart's clip
    path (callers embedding several charts in one page pass their figure
    id; the default is a hash of the chart's text and plot rectangle), see
    :func:`_clip_id`. ``ValueError`` for a 5th series, a series whose
    lengths differ, a non-finite value, or a non-positive value on a log
    axis.
    """
    _check_series(series, log_x, log_y)
    xs_all = [x for s in series for x in s.xs] + [b.lo for b in bands if b.axis == "x"] + [b.hi for b in bands if b.axis == "x"] + [m.x for m in markers]
    ys_all = [y for s in series for y in s.ys] + [b.lo for b in bands if b.axis == "y"] + [b.hi for b in bands if b.axis == "y"] + [m.y for m in markers]
    x_lo, x_hi = _axis_range(xs_all, log_x, pad=0.0 if not log_x else 0.02)
    y_lo, y_hi = _axis_range(ys_all, log_y, pad=0.06)
    x_ticks = log_ticks(x_lo, x_hi) if log_x else nice_ticks(x_lo, x_hi)
    if not log_x:
        x_lo, x_hi, x_ticks = _snap_to_ticks(x_lo, x_hi, x_ticks)
    x_unit, y_unit = _unit_of(x_label), _unit_of(y_label)
    # layout: the left margin fits the widest y tick label, the top the title (and the legend for >= 2 series)
    y_ticks_probe = _tick_labels(log_ticks(y_lo, y_hi) if log_y else nice_ticks(y_lo, y_hi), y_unit, log_y)
    ml = int(round(22 + max((_text_width(t, _TEXT_PX) for t in y_ticks_probe), default=30) + 10))
    mr = 28
    top = 44
    out = _svg_open(width, height, title)
    out.append(_title_text(title, 12, 24))
    if len(series) >= 2:
        legend, top = _legend([("line", SERIES_COLOURS[i], s.name) for i, s in enumerate(series)], 12, 46, width - mr)
        out += legend
    bottom = height - 50
    sx = _Scale(x_lo, x_hi, ml, width - mr, log_x)
    sy = _Scale(y_lo, y_hi, bottom, top, log_y)
    rect = f'x="{_f(ml)}" y="{_f(top)}" width="{_f(width - mr - ml)}" height="{_f(bottom - top)}"'
    cid = _clip_id(clip_id, "\0".join([title, x_label, y_label, *(s.name for s in series), rect]))
    out.append(f'<clipPath id="{esc(cid)}"><rect {rect}/></clipPath>')
    # the series in pixels first: label placement keeps every label off the drawn lines and off each other
    polylines: list[str] = []
    pixel_points: list[tuple[float, float]] = []
    for i, s in enumerate(series):
        xs, ys = downsample(s.xs, s.ys)
        pts = [(sx(x), sy(y)) for x, y in zip(xs, ys)]
        pixel_points += pts
        points = " ".join(f"{_f(px, 1)},{_f(py, 1)}" for px, py in pts)
        polylines.append(
            f'<polyline class="series" data-name="{esc(s.name)}" data-points="{len(xs)}" fill="none" stroke="{SERIES_COLOURS[i]}" '
            f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round" points="{points}"/>'
        )
    placer = _LabelPlacer(pixel_points, width, height, _SMALL_PX)
    # bands first (behind the grid), guide lines after the grid (in front of it), every label last (on top)
    guides: list[str] = []
    labels: list[str] = []

    def label(cls: str, text: str, candidates: list[tuple[float, float, str]]) -> None:
        x, y, anchor = placer.place(text, candidates)
        labels.append(f'<text class="{cls}" x="{_f(x)}" y="{_f(y)}" font-size="{_SMALL_PX}" fill="{INK_SECONDARY}" text-anchor="{anchor}">{esc(text)}</text>')

    for i, b in enumerate(bands):
        lo, hi = (b.lo, b.hi) if b.lo <= b.hi else (b.hi, b.lo)
        if b.axis == "x":
            x0, x1 = sx(lo), sx(hi)
            if lo == hi:
                guides.append(f'<line class="guide" x1="{_f(x0)}" y1="{_f(top)}" x2="{_f(x0)}" y2="{_f(bottom)}" stroke="{INK_SECONDARY}" stroke-width="1.5"/>')
                if b.label:
                    label("guide-label", b.label, [(x0 + 5, top + 14, "start"), (x0 - 5, top + 14, "end"), (x0 + 5, bottom - 6, "start"), (x0 - 5, bottom - 6, "end")])
            else:
                out.append(f'<rect class="band" x="{_f(x0)}" y="{_f(top)}" width="{_f(x1 - x0)}" height="{_f(bottom - top)}" fill="{BAND_FILL}"/>')
                if b.label:
                    label("band-label", b.label, [(x0 + 5, top + 14, "start"), (x1 - 5, top + 14, "end"), (x0 + 5, bottom - 6, "start"), (x1 - 5, bottom - 6, "end")])
        elif b.axis == "y":
            y1, y0 = sy(lo), sy(hi)
            if lo == hi:
                guides.append(f'<line class="guide" x1="{_f(ml)}" y1="{_f(y1)}" x2="{_f(width - mr)}" y2="{_f(y1)}" stroke="{INK_SECONDARY}" stroke-width="1.5"/>')
                if b.label:
                    label("guide-label", b.label, [(ml + 5, y1 - 5, "start"), (width - mr - 5, y1 - 5, "end"), (ml + 5, y1 + 14, "start"), (width - mr - 5, y1 + 14, "end")])
            else:
                out.append(f'<rect class="band" x="{_f(ml)}" y="{_f(y0)}" width="{_f(width - mr - ml)}" height="{_f(y1 - y0)}" fill="{BAND_FILL}"/>')
                if b.label:
                    label("band-label", b.label, [(ml + 5, y0 + 14, "start"), (width - mr - 5, y0 + 14, "end"), (ml + 5, y1 - 6, "start"), (width - mr - 5, y1 - 6, "end")])
        else:
            raise ValueError(f"band #{i}: axis must be 'x' or 'y' (got {b.axis!r})")
    # grid and ticks
    y_ticks = sy.ticks()
    out.append('<g class="grid">')
    for t in x_ticks:
        px = sx(t)
        out.append(f'<line x1="{_f(px)}" y1="{_f(top)}" x2="{_f(px)}" y2="{_f(bottom)}" stroke="{GRID}" stroke-width="1"/>')
    for t in y_ticks:
        py = sy(t)
        out.append(f'<line x1="{_f(ml)}" y1="{_f(py)}" x2="{_f(width - mr)}" y2="{_f(py)}" stroke="{GRID}" stroke-width="1"/>')
    out.append("</g>")
    out.append(f'<line class="axis" x1="{_f(ml)}" y1="{_f(bottom)}" x2="{_f(width - mr)}" y2="{_f(bottom)}" stroke="{AXIS}" stroke-width="1"/>')
    out.append(f'<line class="axis" x1="{_f(ml)}" y1="{_f(top)}" x2="{_f(ml)}" y2="{_f(bottom)}" stroke="{AXIS}" stroke-width="1"/>')
    out.append('<g class="ticks">')
    for t, tick_text in zip(x_ticks, _tick_labels(x_ticks, x_unit, log_x)):
        out.append(f'<text x="{_f(sx(t))}" y="{_f(bottom + 16)}" text-anchor="middle" fill="{INK_SECONDARY}">{esc(tick_text)}</text>')
    for t, tick_text in zip(y_ticks, _tick_labels(y_ticks, y_unit, log_y)):
        out.append(f'<text x="{_f(ml - 6)}" y="{_f(sy(t) + 4)}" text-anchor="end" fill="{INK_SECONDARY}">{esc(tick_text)}</text>')
    out.append("</g>")
    out.append(f'<text class="axis-title" x="{_f((ml + width - mr) / 2)}" y="{_f(height - 12)}" text-anchor="middle">{esc(x_label)}</text>')
    out.append(f'<text class="axis-title" x="14" y="{_f((top + bottom) / 2)}" text-anchor="middle" transform="rotate(-90 14 {_f((top + bottom) / 2)})">{esc(y_label)}</text>')
    out += guides
    # the marks
    out.append(f'<g class="series-group" clip-path="url(#{esc(cid)})">')
    out += polylines
    out.append("</g>")
    for m in markers:
        px, py = sx(m.x), sy(m.y)
        colour = m.colour or INK
        out.append(f'<circle class="marker" cx="{_f(px)}" cy="{_f(py)}" r="5" fill="{colour}" stroke="{SURFACE}" stroke-width="2"/>')
        if m.label:
            label("marker-label", m.label, [(px + 10, py - 8, "start"), (px - 10, py - 8, "end"), (px + 10, py + 16, "start"), (px - 10, py + 16, "end")])
    out += labels
    out.append("</svg>")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- waveforms from spice/results.json

_SCALE_AXES: dict[str, tuple[str, str, bool]] = {
    "time": ("시간 (s)", "s", False),
    "frequency": ("주파수 (Hz)", "Hz", True),
    "v-sweep": ("소스 전압 (V)", "V", False),
    "i-sweep": ("소스 전류 (A)", "A", False),
    "temp-sweep": ("온도 (°C)", "°C", False),
    "res-sweep": ("저항 (Ω)", "Ω", False),
}
_KIND_AXES: dict[str, str] = {"voltage": "전압 (V)", "current": "전류 (A)"}


def _analysis_result(results_json: dict[str, Any], analysis_id: str) -> dict[str, Any]:
    analyses = results_json.get("analyses") if isinstance(results_json, dict) else None
    if not isinstance(analyses, dict) or analysis_id not in analyses:
        raise ValueError(f"analysis {analysis_id!r} is not in results.json (analyses: {sorted(analyses) if isinstance(analyses, dict) else 'none'})")
    result = analyses[analysis_id].get("result")
    if not isinstance(result, dict) or not isinstance(result.get("vectors"), dict):
        raise ValueError(f"analysis {analysis_id!r} has no result vectors in results.json")
    return result


def plot_vector(vectors: dict[str, Any], name: str) -> str | None:
    """The plot vector key for ``name`` (``v(OUT)`` / ``OUT`` / ``out`` / ``i(VVIN)`` / ``vvin#branch``) in an analysis' ``vectors``, like ``SpiceResult.vector``; ``None`` when the analysis has no such vector."""
    for candidate in (name, name.lower(), canonical_name(name.lower())):
        if candidate in vectors:
            return candidate
    return None


_plot_vector = plot_vector


def vector_kind(result: dict[str, Any], plot_name: str) -> str:
    """``"voltage"`` or ``"current"``: the analysis' recorded ``vector_types`` entry when it is one of those, else ``#branch`` in the name means a current."""
    kind = (result.get("vector_types") or {}).get(plot_name)
    if isinstance(kind, str) and kind in _KIND_AXES:
        return kind
    return "current" if "#branch" in plot_name else "voltage"


_vector_kind = vector_kind


def waveform_figure(results_json: dict[str, Any], analysis_id: str, vectors: Sequence[str], *, title: str, fig_id: str = "waveform") -> Figure:
    """The named vectors of one analysis of a ``spice/results.json`` (:mod:`ai_eda.tools.spice.stage` layout) as a line chart.

    ``vectors`` may be spelled as in the IR (``v(OUT)``, ``i(VVIN)``) or as
    plot vector names (``out``, ``vvin#branch``); they are drawn against the
    analysis' scale vector (``time`` -> SI-prefixed seconds, ``frequency`` ->
    a log axis) and must all be of one kind: voltages in V or currents in A,
    never both on one chart (use :func:`waveform_figures` to split them).
    ``ValueError`` names a missing analysis / vector and the available ones.
    """
    if not vectors:
        raise ValueError("no vectors to draw")
    result = _analysis_result(results_json, analysis_id)
    plot = result["vectors"]
    scale = result.get("scale")
    if not scale or scale not in plot:
        raise ValueError(f"analysis {analysis_id!r} has no scale vector ({scale!r}) in results.json; nothing to draw against")
    scale_values = [float(v) for v in plot[scale]]
    x_label, _, log_x = _SCALE_AXES.get(str(scale), (f"{scale}", "", False))
    series: list[Series] = []
    kinds: set[str] = set()
    for name in vectors:
        key = _plot_vector(plot, name)
        if key is None:
            raise ValueError(f"vector {name!r} is not in analysis {analysis_id!r} (available: {sorted(plot)})")
        ys = [float(v) for v in plot[key]]
        if len(ys) != len(scale_values):
            raise ValueError(f"vector {name!r} has {len(ys)} samples, the scale {scale!r} {len(scale_values)}")
        kinds.add(_vector_kind(result, key))
        series.append(Series(name, scale_values, ys))
    if len(kinds) > 1:
        raise ValueError("voltages and currents never share one chart (one y axis per chart); use waveform_figures to draw them separately")
    kind = kinds.pop()
    svg = svg_line_chart(series, title=title, x_label=x_label, y_label=_KIND_AXES[kind], log_x=log_x, clip_id=fig_id)
    command = results_json["analyses"][analysis_id].get("command") or result.get("command") or analysis_id
    n = len(scale_values)
    sampled = f"; {n}점 중 균등 간격으로 최대 {MAX_POINTS}점만 그림" if n > MAX_POINTS else f"; {n}점"
    caption = f"시뮬레이션 결과(해석 `{analysis_id}`: `{command}`)의 {', '.join(vectors)} 파형, x축 = {scale}{sampled}. 값은 ngspice 가 기록한 그대로이며 그림은 판정하지 않습니다."
    return Figure(fig_id, title, caption, svg)


def waveform_figures(results_json: dict[str, Any], analysis_id: str, vectors: Sequence[str], *, title: str, fig_id: str = "waveform") -> list[Figure]:
    """:func:`waveform_figure` per vector kind, at most :data:`MAX_SERIES` series each.

    The voltages get id ``fig_id`` and, when any, the branch currents
    ``fig_id + "_i"`` (title + ``" (전류)"``). A kind with more than
    :data:`MAX_SERIES` vectors is drawn in several charts in the order
    given (``fig_id``, ``fig_id_2`` …; ``fig_id_i``, ``fig_id_i2`` …) with
    the title suffixed ``(k/n)``, so a 5th vector never loses the chart.
    """
    result = _analysis_result(results_json, analysis_id)
    plot = result["vectors"]
    groups: dict[str, list[str]] = {"voltage": [], "current": []}
    for name in vectors:
        key = plot_vector(plot, name)
        if key is None:
            raise ValueError(f"vector {name!r} is not in analysis {analysis_id!r} (available: {sorted(plot)})")
        groups[_vector_kind(result, key)].append(name)
    out: list[Figure] = []
    for kind, id_suffix, title_suffix in (("voltage", "", ""), ("current", "_i", " (전류)")):
        names = groups[kind]
        parts = [names[i : i + MAX_SERIES] for i in range(0, len(names), MAX_SERIES)]
        for k, part in enumerate(parts, start=1):
            part_title = f"{title}{title_suffix}" + (f" ({k}/{len(parts)})" if len(parts) > 1 else "")
            part_id = f"{fig_id}{id_suffix}" + (f"{'' if id_suffix else '_'}{k}" if k > 1 else "")
            out.append(waveform_figure(results_json, analysis_id, part, title=part_title, fig_id=part_id))
    return out


# --------------------------------------------------------------------------- the board

_BOARD_MARGIN = 40
#: the in-figure caption block: this much plus one line height per wrapped line
_BOARD_CAPTION_PAD = 12
_BOARD_CAPTION_LINE_H = 14
#: a value label longer than this is re-spelled (a SPICE number) or cut with an ellipsis on the board (the full value stays in ``data-value`` and in the parts table)
_VALUE_LABEL_CHARS = 12
#: ngspice's own scale-factor letters, for re-spelling a long SPICE number on the board
_SPICE_SUFFIXES: dict[int, str] = {-15: "f", -12: "p", -9: "n", -6: "u", -3: "m", 0: "", 3: "k", 6: "Meg", 9: "G"}
#: the surface-coloured halo behind a board label, so it stays legible over copper
_LABEL_HALO = f'paint-order="stroke" stroke="{BOARD_FILL}" stroke-width="3" stroke-linejoin="round"'


#: an upper-case ``M`` not followed by ``eg``: mega for KiCad and a part value, milli for SPICE
_AMBIGUOUS_MEGA_RE = re.compile(r"M(?![eE][gG])")


def _value_label(value: str, limit: int = _VALUE_LABEL_CHARS) -> str:
    """The board's value label: ``value`` as written when it fits ``limit``; a longer SPICE number is re-spelled to 5 significant digits with its scale letter (``64.8172677616823n`` -> ``64.817n``); anything else is cut with an ellipsis.

    A value with an upper-case ``M`` that is not ``Meg`` is never re-spelled:
    a part value reads it as mega (KiCad, :mod:`ai_eda.tools.calc.part_value`)
    and the SPICE parser as milli, so re-spelling would change the number by
    10^9 - it is cut like any other long text.
    """
    if len(value) <= limit:
        return value
    if _AMBIGUOUS_MEGA_RE.search(value):
        return _shorten(value, limit)
    try:
        number = parse_spice_number(value)
    except ValueError:
        return _shorten(value, limit)
    e3 = si_exponent(number)
    return _shorten(f"{_g(number / 10**e3, 5)}{_SPICE_SUFFIXES[e3]}", limit)


@dataclass(frozen=True)
class _PartBox:
    """One placed part as the board labels see it: its extent, its own pads (the square each pad can reach) and their drawn copper, in pixels."""

    ref: str
    value: str
    box: _Box
    pads: tuple[_Box, ...]
    copper: tuple[_Box, ...] = ()


def _label_layouts(part: _PartBox) -> dict[str, tuple[str, float, float, float, float]]:
    """The candidate spots of a part's two labels: ``name -> (text-anchor, ref x, ref baseline, value x, value baseline)``."""
    x1, y1, x2, y2 = part.box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    ref_w, value_w = _text_width(part.ref, _SMALL_PX), _text_width(_value_label(part.value), _SMALL_PX - 1)
    return {
        "inside": ("middle", cx, cy - 3, cx, cy + 11),
        "default": ("middle", cx, y1 - 4, cx, y2 + 11),
        "left": ("end", x1 - 3 - value_w - 4, cy + 4, x1 - 3, cy + 4),
        "right": ("start", x2 + 3, cy + 4, x2 + 3 + ref_w + 4, cy + 4),
        "above2": ("middle", cx, y1 - 17, cx, y1 - 4),
        "below2": ("middle", cx, y2 + 11, cx, y2 + 25),
        "left2": ("end", x1 - 3, cy - 2, x1 - 3, cy + 11),
        "right2": ("start", x2 + 3, cy - 2, x2 + 3, cy + 11),
    }


def _label_boxes(part: _PartBox, layout: tuple[str, float, float, float, float]) -> list[_Box]:
    anchor, rx, ry, vx, vy = layout

    def one(x: float, y: float, w: float, px: float) -> _Box:
        left = x - w / 2 if anchor == "middle" else (x - w if anchor == "end" else x)
        return (left, y - px, left + w, y + 3)

    return [one(rx, ry, _text_width(part.ref, _SMALL_PX), _SMALL_PX), one(vx, vy, _text_width(_value_label(part.value), _SMALL_PX - 1), _SMALL_PX - 1)]


def _board_labels(parts: Sequence[_PartBox], width: float) -> list[str]:
    """The ref / value labels of every placed part, each pair where it overlaps no label already placed.

    A part's labels go inside its extent when both fit there clear of its
    own pads (the empty middle of a QFP); otherwise the default is the
    reference centred above the part and the value centred below it. A part
    is *crowded* when those default labels would overlap another part's
    default labels (parts packed around a microcontroller). Three sweeps, each
    in IR order: (1) every part that is not crowded takes its inside spot,
    else its default, else a spot beside it - one line left / right, both
    lines above / below, two lines left / right - clear of every other
    part's extent; (2) a crowded part takes its inside spot or a spot beside
    it that is clear of every other part's extent *and* default labels;
    (3) the crowded parts
    still without a spot take the first spot beside them clear of the other
    parts' extents, else the default. Every spot must be free of the labels
    placed before it, the default and the spots beside a part must not
    cover another part's drawn pad copper (a label over a QFP's pads hides
    them), and a spot beside a part must lie inside the figure; when none is
    free the default stays - nothing is dropped. The labels are emitted in
    IR order. A board without crowded parts, without a part large enough to
    hold its labels and without a default label on another part's pads is
    drawn exactly as before these rules existed. Deterministic: the result
    depends on the parts and their order only.
    """
    layouts = [_label_layouts(p) for p in parts]
    defaults = [_label_boxes(p, lay["default"]) for p, lay in zip(parts, layouts)]
    overlaps = _LabelPlacer._overlaps
    crowded = [any(overlaps(a, b) for j, other in enumerate(defaults) if j != i for a in defaults[i] for b in other) for i in range(len(parts))]
    taken: list[_Box] = []
    chosen: dict[int, tuple[str, float, float, float, float]] = {}
    beside = ("left", "right", "above2", "below2", "left2", "right2")

    def first_free(i: int, spots: Sequence[str], strict: bool) -> tuple[str, float, float, float, float] | None:
        part, lay = parts[i], layouts[i]
        x1, y1, x2, y2 = part.box
        extents = [q.box for j, q in enumerate(parts) if j != i]
        others_default = [b for j, d in enumerate(defaults) if j != i for b in d] if strict else []
        others_copper = [c for j, q in enumerate(parts) if j != i for c in q.copper]
        for name in spots:
            boxes = _label_boxes(part, lay[name])
            if name == "inside" and not all(
                b[0] >= x1 + 2 and b[2] <= x2 - 2 and b[1] >= y1 + 2 and b[3] <= y2 - 2 and not any(overlaps(b, pad) for pad in part.pads) for b in boxes
            ):
                continue
            if name != "inside" and any(overlaps(b, c) for b in boxes for c in others_copper):
                continue
            if name in beside and any(b[0] < 2 or b[2] > width - 2 or b[1] < 2 or any(overlaps(b, o) for o in extents + others_default) for b in boxes):
                continue
            if not any(overlaps(b, t) for b in boxes for t in taken):
                return lay[name]
        return None

    def take(i: int, layout: tuple[str, float, float, float, float]) -> None:
        chosen[i] = layout
        taken.extend(_label_boxes(parts[i], layout))

    for i in range(len(parts)):
        if not crowded[i]:
            take(i, first_free(i, ("inside", "default", *beside), False) or layouts[i]["default"])
    for i in range(len(parts)):
        if crowded[i] and (layout := first_free(i, ("inside", *beside), True)) is not None:
            take(i, layout)
    for i in range(len(parts)):
        if crowded[i] and i not in chosen:
            take(i, first_free(i, (*beside, "default"), False) or layouts[i]["default"])
    out: list[str] = []
    for i, part in enumerate(parts):
        anchor, rx, ry, vx, vy = chosen[i]
        out += [
            f'<text class="ref" x="{_f(rx)}" y="{_f(ry)}" text-anchor="{anchor}" font-size="{_SMALL_PX}" font-weight="600" {_LABEL_HALO}>{esc(part.ref)}</text>',
            f'<text class="value" data-value="{esc(part.value)}" x="{_f(vx)}" y="{_f(vy)}" text-anchor="{anchor}" font-size="{_SMALL_PX - 1}" '
            f'fill="{INK_SECONDARY}" {_LABEL_HALO}>{esc(_value_label(part.value))}</text>',
        ]
    return out


def _wrap_caption(line: str, max_px: float, px: float) -> list[str]:
    """``line`` split after ``;`` / ``,`` into lines whose estimated width fits ``max_px``; a single piece wider than that is broken at character level, so nothing is ever clipped at the SVG edge."""
    out: list[str] = []
    current = ""
    for piece in re.split(r"(?<=[;,]) ", line):
        candidate = f"{current} {piece}" if current else piece
        if current and _text_width(candidate, px) > max_px:
            out.append(current)
            current = piece
        else:
            current = candidate
        while _text_width(current, px) > max_px and len(current) > 1:
            cut = next((k for k in range(len(current), 0, -1) if _text_width(current[:k], px) <= max_px), 1)
            out.append(current[:cut])
            current = current[cut:]
    if current:
        out.append(current)
    return out


def _pad_shape(pad: Pad, cx: float, cy: float, angle: float, scale: float, fill: str) -> str:
    """The pad's copper as one SVG element at pixel centre ``(cx, cy)`` (rotated ``angle`` degrees counter-clockwise on screen, KiCad's sense)."""
    if pad.shape == "circle":
        return f'<circle cx="{_f(cx)}" cy="{_f(cy)}" r="{_f(pad.size_w / 2 * scale)}" fill="{fill}"/>'
    w, h = pad.size_w * scale, pad.size_h * scale
    quarter = math.isclose(angle % 90.0, 0.0, abs_tol=1e-9)
    if quarter and not math.isclose(angle % 180.0, 0.0, abs_tol=1e-9):
        w, h = h, w
    if pad.shape == "oval":
        rx = min(w, h) / 2
    elif pad.shape == "roundrect":
        rx = (pad.roundrect_rratio if pad.roundrect_rratio is not None else 0.25) * min(w, h)
    else:
        rx = 0.0
    transform = "" if quarter else f' transform="rotate({_f(-angle)} {_f(cx)} {_f(cy)})"'
    rx_attr = f' rx="{_f(rx)}"' if rx > 0 else ""
    return f'<rect x="{_f(cx - w / 2)}" y="{_f(cy - h / 2)}" width="{_f(w)}" height="{_f(h)}"{rx_attr} fill="{fill}"{transform}/>'


def _pad_fill(pad: Pad, layers: list[str]) -> tuple[str, bool]:
    """``(fill colour, through-hole?)`` of a pad from its type and its placed layers."""
    if pad.pad_type in ("thru_hole", "np_thru_hole") or "*.Cu" in layers:
        return (THT_COLOUR if pad.pad_type != "np_thru_hole" else BOARD_FILL), True
    if "F.Cu" in layers:
        return COPPER_COLOURS["F.Cu"], False
    if "B.Cu" in layers:
        return COPPER_COLOURS["B.Cu"], False
    return OTHER_COPPER, False


#: SVG em per mm of KiCad text height: a sans-serif capital is about 0.7 em tall, so the drawn capitals are as tall as the KiCad size
_SILK_EM_PER_MM = 1.4


@dataclass
class _SilkDrawing:
    """The silkscreen of a board figure: markup per silk layer and what it counts."""

    by_layer: dict[str, list[str]]
    graphics: int = 0
    texts: int = 0  # IR silk texts drawn
    library_texts: int = 0  # library texts drawn (footprint silk texts, default-position references of undesigned footprints)
    designed: bool = False
    refused: int = 0  # IR texts the compiler refuses (not drawn; the pcb.silk.* checks name them)
    on_fab: int = 0  # IR references placed on the fab layer (not silk)


def _silk_text(text: str, box: Shape, angle: float, size: float, stroke: float, layer: str, X: Any, Y: Any, scale: float, *, kind: str, ref: str | None) -> str:
    """One silk text centred in its estimated box (the IR's / the library's anchor, justification and mirroring already applied), turned ``angle`` degrees counter-clockwise on screen, mirrored on a ``B.*`` layer."""
    cx = sum(p[0] for p in box.points) / len(box.points)
    cy = sum(p[1] for p in box.points) / len(box.points)
    transform = f"translate({_f(X(cx))} {_f(Y(cy))})"
    if not math.isclose(angle % 360.0, 0.0, abs_tol=1e-9) and not math.isclose(angle % 360.0, 360.0, abs_tol=1e-9):
        transform += f" rotate({_f(-angle)})"
    if layer.startswith("B."):
        transform += " scale(-1 1)"
    ref_attr = f' data-ref="{esc(ref)}"' if ref else ""
    return (
        f'<text class="silk-text" data-kind="{esc(kind)}"{ref_attr} data-size-mm="{_g(size)}" data-stroke-mm="{_g(stroke)}" transform="{transform}" '
        f'x="0" y="0" text-anchor="middle" dominant-baseline="central" font-size="{_f(size * scale * _SILK_EM_PER_MM)}" fill="{SILK_COLOUR}">{esc(text)}</text>'
    )


def _box_angle(box: Shape) -> float:
    """The text angle of an estimated text box: the direction of its first edge (the text's own +x), counter-clockwise on screen."""
    (x0, y0), (x1, y1) = box.points[0], box.points[1]
    return math.degrees(math.atan2(-(y1 - y0), x1 - x0))


def _board_silk(ir: CircuitIR, placed: Sequence[tuple[Any, Any, FootprintDef]], X: Any, Y: Any, scale: float) -> _SilkDrawing:
    """What the compiled board's silk layers carry, as markup per layer (:func:`board_figure`).

    Always: every footprint's own ``F.SilkS`` / ``B.SilkS`` graphics from the
    library (the strokes the 3D preview draws, :func:`~ai_eda.tools.model3d.scene.graphic_paths`,
    mirrored with the footprint), at their stroke width. With a designed
    silkscreen (``ir.pcb.silkscreen`` not empty) also every silk text: the
    IR texts at their IR positions (a reference on ``F.Fab`` / ``B.Fab`` is
    not silk and only counted; a text the compiler refuses is not drawn and
    counted), the library position of the ``Reference`` of a footprint
    without a designed one, and the footprints' visible library silk texts -
    each centred in the estimated box the ``pcb.silk.*`` checks measured
    (:mod:`ai_eda.tools.silkscreen.geometry`), in the browser's sans-serif,
    not KiCad's stroke font.
    """
    pcb = ir.pcb
    assert pcb is not None
    out = _SilkDrawing(by_layer={layer: [] for layer in SILK_LAYER_ORDER}, designed=bool(pcb.silkscreen))
    for comp, placement, fp in placed:
        for item in fp.node:
            if sexpr.head(item) not in GRAPHIC_HEADS:
                continue
            layer = sexpr.get(item, "layer")
            if layer not in SILK_LAYERS:
                continue
            board_layer = mirrored_layer(str(layer), placement.side)
            width = stroke_width(item)
            lines, fills = graphic_paths(item)
            for poly in fills:
                pts = " ".join(f"{_f(X(x))},{_f(Y(y))}" for x, y in (to_board(placement, u, v) for u, v in poly))
                out.by_layer[board_layer].append(f'<polygon class="silk-fill" data-ref="{esc(comp.ref)}" points="{pts}" fill="{SILK_COLOUR}"/>')
                out.graphics += 1
            if width <= 0:
                continue
            for path in lines:
                pts = " ".join(f"{_f(X(x))},{_f(Y(y))}" for x, y in (to_board(placement, u, v) for u, v in path))
                out.by_layer[board_layer].append(
                    f'<polyline class="silk-line" data-ref="{esc(comp.ref)}" points="{pts}" fill="none" stroke="{SILK_COLOUR}" '
                    f'stroke-width="{_f(width * scale)}" stroke-linecap="round" stroke-linejoin="round"/>'
                )
                out.graphics += 1
    if not out.designed:
        return out
    sides = {c.ref: (pl.side if (pl := pcb.placement(c.ref)) is not None else None) for c in ir.components}
    problems = silk_text_problems(pcb.silkscreen, sides)
    designed_refs: set[str] = set()
    for i, t in enumerate(pcb.silkscreen):
        if i in problems:
            out.refused += 1
            continue
        if t.kind == SilkKind.REFERENCE and t.component_ref is not None:
            designed_refs.add(t.component_ref)
        if t.layer not in SILK_LAYERS:
            out.on_fab += 1
            continue
        out.by_layer[t.layer].append(_silk_text(t.text, ir_text_box(t), t.rotation_deg, t.size_mm, t.thickness_mm, t.layer, X, Y, scale, kind=str(t.kind), ref=t.component_ref))
        out.texts += 1
    for comp, placement, fp in placed:
        values = {"Value": comp.value, "Datasheet": (comp.datasheet.url or "") if comp.datasheet is not None else ""}
        silk = footprint_silk(comp.ref, fp, placement, values)
        for item in silk.items:
            # a text item with a size (fp_text / a visible property); an fp_text_box has no single text size and is left out
            if item.kind != "text" or item.shape is None or not item.text or item.size is None or item.layer not in SILK_LAYERS or len(item.shape.points) != 4:
                continue
            out.by_layer[item.layer].append(_silk_text(item.text, item.shape, _box_angle(item.shape), item.size, item.width or 0.0, item.layer, X, Y, scale, kind="library", ref=comp.ref))
            out.library_texts += 1
        ref = silk.reference
        if comp.ref not in designed_refs and ref is not None and not ref.hidden and ref.layer in SILK_LAYERS:
            out.by_layer[ref.layer].append(_silk_text(ref.text, ref.box(), ref.rotation, ref.size, ref.thickness, ref.layer, X, Y, scale, kind="library_reference", ref=comp.ref))
            out.library_texts += 1
    return out


@dataclass(frozen=True)
class NetClassStyle:
    """One ``ir.si`` net class as the board figure / GUI colours it: ``token`` (``nc-default`` / ``nc-<i>`` / ``nc-other``), ``colour``, and its routed nets.

    ``nets`` are the nets the class routes (declared, promoted, and for the
    default class every net no class lists), in IR order; ``promoted`` the
    nets the critical-length rule moved into it. Read from ``ir.si`` only.
    """

    name: str
    token: str
    colour: str
    default: bool
    nets: tuple[str, ...]
    promoted: tuple[str, ...]


def net_class_styles(ir: CircuitIR) -> list[NetClassStyle]:
    """The colour of every ``ir.si`` net class in ``ir.si`` order (:class:`NetClassStyle`); ``[]`` for a design without ``ir.si``."""
    si = ir.si
    if si is None:
        return []
    members: dict[str, list[str]] = {c.name: [] for c in si.net_classes}
    for n in ir.nets:
        c = si.class_of(n.name)
        if c is not None:
            members[c.name].append(n.name)
    out: list[NetClassStyle] = []
    k = 0
    for c in si.net_classes:
        if c.default:
            token, colour = "nc-default", NET_CLASS_DEFAULT
        elif k < len(CLASS_COLOURS):
            token, colour = f"nc-{k}", CLASS_COLOURS[k]
            k += 1
        else:
            token, colour = "nc-other", NET_CLASS_OTHER
        out.append(NetClassStyle(c.name, token, colour, c.default, tuple(members[c.name]), tuple(p.net for p in c.promoted)))
    return out


def _plane_hatches(ir: CircuitIR, fig_id: str, X: Any, Y: Any) -> tuple[list[str], list[str]]:
    """``(markup, caption parts)`` of the inner-layer zones (the stackup's planes) as hatched outlines, never a fill that hides the tracks."""
    pcb = ir.pcb
    assert pcb is not None
    inner = [z for z in pcb.zones if re.match(r"^In[1-9][0-9]*\.Cu$", z.layer)]
    if not inner:
        return [], []
    layers = sorted({z.layer for z in inner}, key=lambda name: int(name[2:-3]))
    base = _ID_SAFE_RE.sub("-", fig_id)
    out = ['<g class="planes">', "<defs>"]
    for i, layer in enumerate(layers):
        angle = PLANE_HATCH_ANGLES[i % len(PLANE_HATCH_ANGLES)]
        out.append(
            f'<pattern id="hatch-{base}-{layer_class(layer)}" patternUnits="userSpaceOnUse" width="9" height="9" patternTransform="rotate({_f(angle)})">'
            f'<line x1="0" y1="0" x2="0" y2="9" stroke="{PLANE_HATCH}" stroke-width="1" stroke-opacity="0.35"/></pattern>'
        )
    out.append("</defs>")
    parts: list[str] = []
    for z in inner:
        points = " ".join(f"{_f(X(x))},{_f(Y(y))}" for x, y in z.polygon)
        out.append(
            f'<polygon class="plane" data-net="{esc(z.net)}" data-layer="{esc(z.layer)}" points="{points}" fill="url(#hatch-{base}-{layer_class(z.layer)})" '
            f'stroke="{PLANE_HATCH}" stroke-width="1" stroke-dasharray="6 4" stroke-opacity="0.6"/>'
        )
        angle = PLANE_HATCH_ANGLES[layers.index(z.layer) % len(PLANE_HATCH_ANGLES)]
        parts.append(f"{z.layer} {z.net} ({_g(angle)}° 빗금)")
    out.append("</g>")
    return out, parts


def layer_class(layer: str) -> str:
    """The class of a copper layer's track group in a board figure: ``F.Cu`` -> ``layer-F_Cu`` (every character outside ``[A-Za-z0-9_-]`` becomes ``_``)."""
    return "layer-" + _ID_SAFE_RE.sub("_", layer)


def board_figure(
    ir: CircuitIR, library: KicadLibrary, *, copper: bool = True, scale_px_per_mm: float = 18.0, max_width: int = COLUMN_PX, fig_id: str | None = None,
    colour_by: str = "layer",
) -> Figure:
    """The placed board of ``ir``: outline, every pad from the KiCad library, tracks and vias (``copper=True``), reference / value labels
    (inside a part whose extent holds them clear of its pads, else above / below it, beside it when crowded: :func:`_board_labels`)
    and a caption line.

    Reads exactly what the PCB compiler reads (``ir.pcb.outline``,
    ``ir.pcb.placement(ref)`` for every component, the footprint loaded
    from ``library``, ``ir.pcb.tracks`` / ``vias``) and refuses the same
    things with ``ValueError``: no ``ir.pcb`` / outline, a non-positive
    outline, a component without a footprint or placement, a footprint not
    found in the library. Colours: F.Cu red, B.Cu blue (tracks to scale,
    round caps; SMD pads by their placed layer), through-hole pads yellow
    with a white drill, vias gray with a white drill. ``scale_px_per_mm`` is
    reduced when the board would be wider than ``max_width`` px. Every pad
    is one ``<g class="pad">``, every track one ``<line class="track">``,
    every via one ``<g class="via">``. The layers are groups a viewer can
    toggle by class: ``outline`` (the board rect), one ``<g>`` per copper
    layer inside ``<g class="tracks">`` (:func:`layer_class`: ``layer-B_Cu``
    and ``layer-F_Cu`` always, empty when unused, plus any other layer that
    carries a track; ``copper=False`` draws no layer group), ``pads``,
    ``vias``, ``silk`` and ``labels``; the labels wear a surface-coloured
    halo so they stay legible over copper, and the caption line inside the
    figure wraps to the figure's width (one ``<text class="caption">`` per
    line). ``copper=False`` draws the placement only (id default
    ``placement``, else ``board``).

    Signal integrity (a design with ``ir.si``): every track carries
    ``data-class`` (the net class that routes it) and ``data-nc`` (its
    :func:`net_class_styles` token), so a viewer can colour the tracks by
    class; ``colour_by="class"`` draws that view itself - each track in its
    class's colour (F.Cu solid, B.Cu lighter), SMD pads neutral, a legend of
    the classes (with their net counts) above the board. The zones on inner
    copper layers (the stackup's planes) are drawn under the tracks as
    hatched, dashed outlines in ``<g class="planes">`` (one hatch angle per
    layer, never a fill that would hide the tracks), and the caption names
    them; KiCad computes their fill.

    The silkscreen (:func:`_board_silk`) is drawn over the copper in
    :data:`SILK_COLOUR` inside ``<g class="silk">``, one group per silk layer
    (:func:`layer_class`: ``layer-B_SilkS`` then ``layer-F_SilkS``, always
    both, empty when unused): the footprints' library silk graphics always
    (``silk-line`` / ``silk-fill``), and - when the IR carries a designed
    silkscreen - every silk text at its IR position (``silk-text``, with
    ``data-kind``); the reference designators then come from the IR, so the
    figure's own ref / value labels are drawn only for a board without
    designed silk (the ``labels`` group is always there, empty otherwise).
    The figure draws the silk; it judges nothing (``pcb.silk.*`` and KiCad's
    DRC do).
    """
    pcb = ir.pcb
    if pcb is None or pcb.outline is None:
        raise ValueError("ir.pcb.outline is None: a board figure needs the outline the PCB compiler needs")
    if colour_by not in ("layer", "class"):
        raise ValueError(f"colour_by must be 'layer' or 'class', got {colour_by!r}")
    by_class = colour_by == "class" and copper
    styles = net_class_styles(ir)
    if colour_by == "class" and not styles:
        raise ValueError("colour_by='class' needs ir.si net classes")
    style_of = {net: st for st in styles for net in st.nets}
    outline = pcb.outline
    if not (outline.width_mm > 0 and outline.height_mm > 0):
        raise ValueError(f"ir.pcb.outline must have positive size (got {outline.width_mm} x {outline.height_mm})")
    scale = min(float(scale_px_per_mm), (max_width - 2 * _BOARD_MARGIN) / outline.width_mm)
    if scale <= 0:
        raise ValueError("scale_px_per_mm / max_width leave no room for the board")
    width = int(round(outline.width_mm * scale + 2 * _BOARD_MARGIN))

    legend: list[str] = []
    top_pad = float(_BOARD_MARGIN)
    if by_class:
        entries = [("box", st.colour, f"{st.name} ({len(st.nets)})") for st in styles if st.nets]
        legend, next_y = _legend(entries, _BOARD_MARGIN, 22, width - _BOARD_MARGIN)
        top_pad = next_y + 6

    def X(x_mm: float) -> float:
        return _BOARD_MARGIN + (x_mm - outline.origin_x_mm) * scale

    def Y(y_mm: float) -> float:
        return top_pad + (y_mm - outline.origin_y_mm) * scale

    placed: list[tuple[Any, Any, FootprintDef]] = []
    for comp in ir.components:
        if comp.footprint is None:
            raise ValueError(f"component {comp.ref!r} has no footprint; the board figure never picks one")
        resolved = library.resolve_footprint(comp.footprint)
        if not resolved.verified:
            raise ValueError(f"footprint {comp.footprint.library}:{comp.footprint.name} of {comp.ref!r} was not found in a KiCad library; refusing to guess its pads")
        placement = pcb.placement(comp.ref)
        if placement is None:
            raise ValueError(f"component {comp.ref!r} has no placement in ir.pcb.placements")
        placed.append((comp, placement, library.load_footprint(comp.footprint)))

    kind = "board" if copper else "placement"
    what = ("보드 그림 (넷 클래스 색)" if by_class else "보드 그림 (동박 포함)") if copper else "배치도 (동박 제외)"
    title = f"{ir.project.id}: {what}"
    silk = _board_silk(ir, placed, X, Y, scale)
    tracks = list(pcb.tracks) if copper else []
    vias = list(pcb.vias) if copper else []
    other_layers = sorted({t.layer for t in tracks if t.layer not in COPPER_COLOURS})
    layer_order = ["B.Cu", *other_layers, "F.Cu"]
    # the caption line is wrapped to the figure's width before the height is known: every line must fit inside the SVG
    length_mm = sum(math.hypot(t.end[0] - t.start[0], t.end[1] - t.start[1]) for t in tracks)
    size = f"{_g(outline.width_mm)} × {_g(outline.height_mm)} mm"
    planes, plane_parts = _plane_hatches(ir, fig_id or kind, X, Y) if copper else ([], [])
    if by_class:
        stats = f"부품 {len(placed)}개, 트랙 {len(tracks)}개 (동박 {_g(length_mm, 4)} mm), 비아 {len(vias)}개"
        key = CLASS_KEY
    elif copper:
        stats = f"부품 {len(placed)}개, 트랙 {len(tracks)}개 (동박 {_g(length_mm, 4)} mm), 비아 {len(vias)}개"
        key = "빨강 = F.Cu, 파랑 = B.Cu, 노랑 = 관통 패드(흰 원 = 드릴), 회색 = 비아"
        if other_layers:
            key += f", 회색 선 = {', '.join(other_layers)}"
    else:
        stats = f"부품 {len(placed)}개, 동박 제외 (배치만)"
        key = "빨강 = F.Cu 패드, 파랑 = B.Cu 패드, 노랑 = 관통 패드(흰 원 = 드릴)"
    if silk.designed:
        stats += f", 실크 문자 {silk.texts}개 (IR 위치)"
        if silk.library_texts:
            stats += f" + 라이브러리 실크 문자 {silk.library_texts}개"
        if silk.on_fab:
            stats += f", F.Fab 참조 {silk.on_fab}개 (실크 아님)"
        if silk.refused:
            stats += f", 그리지 않은 실크 문자 {silk.refused}개 (컴파일러 거부)"
        key += ", 검정 = 실크스크린 (B.SilkS 문자는 거울상)"
    else:
        key += ", 검정 선 = 라이브러리 실크 (설계된 실크 문자 없음: 참조·값 라벨은 그림용)"
    if plane_parts:
        key += f", 빗금 + 점선 = 내층 평면 영역 {', '.join(plane_parts)} (KiCad 가 채움; 트랙을 가리지 않도록 빗금으로만 그림)"
    caption_lines = _wrap_caption(f"{ir.project.id}: {size}, {stats}; {key}", width - 2 * _BOARD_MARGIN, _SMALL_PX)
    class_lines: list[str] = []
    if copper and not by_class and styles:
        # the GUI recolours this layer figure by class (app.css): the layer key would then be false, so the class key rides along,
        # hidden (display="none") until the viewer's class view shows it instead of the layer key
        tail = key[key.index(", 검정"):] if ", 검정" in key else ""
        class_lines = _wrap_caption(f"{ir.project.id}: {size}, {stats}; {CLASS_KEY}{tail}", width - 2 * _BOARD_MARGIN, _SMALL_PX)
    caption_h = _BOARD_CAPTION_PAD + _BOARD_CAPTION_LINE_H * max(len(caption_lines), len(class_lines))
    height = int(round(outline.height_mm * scale + top_pad + _BOARD_MARGIN + caption_h))
    out = _svg_open(width, height, title)
    out += legend
    out.append(
        f'<rect class="outline" x="{_f(X(outline.origin_x_mm))}" y="{_f(Y(outline.origin_y_mm))}" width="{_f(outline.width_mm * scale)}" '
        f'height="{_f(outline.height_mm * scale)}" fill="{BOARD_FILL}" stroke="{INK_SECONDARY}" stroke-width="1.5"/>'
    )
    out += planes
    out.append('<g class="tracks">')
    for layer in layer_order:
        if copper:
            out.append(f'<g class="{layer_class(layer)}" data-layer="{esc(layer)}">')
        for i, t in enumerate(tracks):
            if t.layer != layer:
                continue
            st = style_of.get(t.net)
            cls_attr = f' data-class="{esc(st.name)}" data-nc="{st.token}"' if st is not None else ""
            if by_class:
                stroke = st.colour if st is not None else NET_CLASS_OTHER
                opacity = "" if t.layer == "F.Cu" else ' stroke-opacity="0.5"'
            else:
                stroke, opacity = COPPER_COLOURS.get(t.layer, OTHER_COPPER), ""
            out.append(
                f'<line class="track" data-net="{esc(t.net)}" data-layer="{esc(t.layer)}"{cls_attr} x1="{_f(X(t.start[0]))}" y1="{_f(Y(t.start[1]))}" '
                f'x2="{_f(X(t.end[0]))}" y2="{_f(Y(t.end[1]))}" stroke="{stroke}"{opacity} '
                f'stroke-width="{_f(t.width_mm * scale)}" stroke-linecap="round"/>'
            )
        if copper:
            out.append("</g>")
    out.append("</g>")
    # pads: bottom SMD first, then front SMD, then through-hole (visible on both sides) on top
    pad_items: list[tuple[int, str]] = []
    label_parts: list[_PartBox] = []
    for comp, placement, fp in placed:
        own_pads: list[_Box] = []
        own_copper: list[_Box] = []
        for pad in fp.pads:
            cx, cy = pad_copper_center(placement, pad)  # the copper; the hole is at pad_center (they differ by a drill offset)
            angle = pad_angle(placement, pad)
            reach = max(pad.size_w, pad.size_h) / 2 * scale
            own_pads.append((X(cx) - reach, Y(cy) - reach, X(cx) + reach, Y(cy) + reach))
            if math.isclose(angle % 90.0, 0.0, abs_tol=1e-9):  # the drawn copper: exact at quarter turns, the reach square otherwise
                hw, hh = (pad.size_w, pad.size_h) if math.isclose(angle % 180.0, 0.0, abs_tol=1e-9) else (pad.size_h, pad.size_w)
                own_copper.append((X(cx) - hw / 2 * scale, Y(cy) - hh / 2 * scale, X(cx) + hw / 2 * scale, Y(cy) + hh / 2 * scale))
            else:
                own_copper.append(own_pads[-1])
            layers = pad_layers(placement, pad)
            fill, tht = _pad_fill(pad, layers)
            if by_class and not tht:
                fill = NEUTRAL_PAD
            order = 2 if tht else (1 if "F.Cu" in layers else 0)
            parts = [f'<g class="pad" data-ref="{esc(comp.ref)}" data-pad="{esc(pad.number)}" data-tht="{"true" if tht else "false"}">']
            parts.append(_pad_shape(pad, X(cx), Y(cy), angle, scale, fill))
            if pad.drill:
                hx, hy = pad_center(placement, pad)
                parts.append(f'<circle class="drill" cx="{_f(X(hx))}" cy="{_f(Y(hy))}" r="{_f(pad.drill / 2 * scale)}" fill="{DRILL_COLOUR}"/>')
            parts.append("</g>")
            pad_items.append((order, "".join(parts)))
        box = footprint_bbox(placement, fp)
        if box is not None:
            label_parts.append(_PartBox(comp.ref, comp.value, (X(box.x1), Y(box.y1), X(box.x2), Y(box.y2)), tuple(own_pads), tuple(own_copper)))
    out.append('<g class="pads">')
    out += [markup for _, markup in sorted(pad_items, key=lambda item: item[0])]
    out.append("</g>")
    out.append('<g class="vias">')
    via_paint = f'fill="{DRILL_COLOUR}" stroke="{CLASS_VIA_RING}" stroke-width="1"' if by_class else f'fill="{VIA_COLOUR}"'
    for v in vias:
        out.append(
            f'<g class="via" data-net="{esc(v.net)}"><circle cx="{_f(X(v.x_mm))}" cy="{_f(Y(v.y_mm))}" r="{_f(v.diameter_mm / 2 * scale)}" {via_paint}/>'
            f'<circle class="drill" cx="{_f(X(v.x_mm))}" cy="{_f(Y(v.y_mm))}" r="{_f(v.drill_mm / 2 * scale)}" fill="{DRILL_COLOUR}"/></g>'
        )
    out.append("</g>")
    out.append('<g class="silk">')
    for layer in SILK_LAYER_ORDER:
        out.append(f'<g class="{layer_class(layer)}" data-layer="{esc(layer)}">')
        out += silk.by_layer[layer]
        out.append("</g>")
    out.append("</g>")
    out.append('<g class="labels">')  # always there (a viewer toggles it by class); empty when the silk carries the references
    if not silk.designed:
        out += _board_labels(label_parts, width)
    out.append("</g>")
    view = ' data-view="class"' if by_class else (' data-view="layer"' if class_lines else "")
    for i, line in enumerate(caption_lines):
        y = height - 10 - _BOARD_CAPTION_LINE_H * (len(caption_lines) - 1 - i)
        out.append(f'<text class="caption"{view} x="{_BOARD_MARGIN}" y="{_f(y)}" font-size="{_SMALL_PX}" fill="{INK_SECONDARY}">{esc(line)}</text>')
    for i, line in enumerate(class_lines):
        y = height - 10 - _BOARD_CAPTION_LINE_H * (len(class_lines) - 1 - i)
        out.append(f'<text class="caption" data-view="class" display="none" x="{_BOARD_MARGIN}" y="{_f(y)}" font-size="{_SMALL_PX}" '
                   f'fill="{INK_SECONDARY}">{esc(line)}</text>')
    out.append("</svg>")
    caption = (
        f"{what}: 외곽 {size}, {stats}. 패드 형상과 실크 선은 KiCad 라이브러리에서 읽은 것이고 위치는 IR 의 배치 그대로입니다"
        + (" (실크 문자는 IR 의 위치·크기·회전에 브라우저 글꼴로 그림; KiCad 의 스트로크 글꼴이 아님)" if silk.designed else "")
        + f". 색: {key}. "
        "그림은 보드의 유효성을 판정하지 않습니다 (DRC 결과는 별도 표)."
    )
    return Figure(fig_id or kind, title, caption, "\n".join(out) + "\n")


# --------------------------------------------------------------------------- the 3D preview


def model3d_figure(
    ir: CircuitIR, library: KicadLibrary, *, model_dir: Path | None | EllipsisType = ..., view: str = "iso", fig_id: str = "iso3d", width: int = COLUMN_PX,
) -> Figure:
    """The board's built-in 3D preview seen from ``view`` (:data:`~ai_eda.tools.model3d.iso.VIEWS`) as a figure.

    The scene is :func:`~ai_eda.tools.model3d.scene.build_scene` of the IR
    and the libraries on disk - the one the ``MODEL_3D`` artifact (the preview
    GLB) is written from - and the SVG is :func:`~ai_eda.tools.model3d.iso.iso_svg`
    at the column width without its in-figure caption: the caption is the
    figure's (:func:`~ai_eda.tools.model3d.scene.scene_caption`: size, the part
    boxes with / without a STEP height, the body rule
    :data:`~ai_eda.tools.model3d.scene.BODY_CAPTION`, the assumptions). The 3D
    library is ``model_dir`` when given (``None`` = none), else the one
    :func:`~ai_eda.tools.model3d.models.find_3dmodel_dir` finds. Refuses what
    the scene refuses (:class:`~ai_eda.tools.model3d.scene.SceneError` is a
    ``ValueError``); judges nothing.
    """
    return model3d_figure_from_scene(build_scene(ir, library, model_dir=model_dir), ir, view=view, fig_id=fig_id, width=width)


def model3d_figure_from_scene(scene: Scene, ir: CircuitIR, *, view: str = "iso", fig_id: str = "iso3d", width: int = COLUMN_PX) -> Figure:
    """:func:`model3d_figure` of a scene already built from ``ir`` (a report that also tabulates the scene's bodies builds it once)."""
    svg = iso_svg(scene, view, width=width, caption=False)
    title = f"{ir.project.id}: 3D 미리보기 ({'등각' if view == 'iso' else view})"
    return Figure(fig_id, title, scene_caption(scene), svg)


# --------------------------------------------------------------------------- tolerance figure

#: the x range of the tolerance figure in units of the tolerance; markers beyond are clamped and marked
_TOL_LIMIT = 1.5
_TOL_ROW_H = 44


def _finite_number(value: object) -> float | None:
    """``value`` as a float when it is a finite number (never a bool), else ``None``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    return float(value)


def expectation_limit(e: Expectation, details: dict[str, Any] | None) -> tuple[float | None, bool]:
    """``(limit, recorded)`` for expectation ``e``: the tolerance the SPICE stage judged against, as it recorded it.

    When ``details`` (of the recorded ``spice.<id>`` result) carries the key
    ``"tolerance"``, that value is the limit (a recorded ``None`` stays
    ``None``: the stage found no usable tolerance) and ``recorded`` is
    ``True``. Without such a record the stage's own rule is applied to the
    IR (``ai_eda.tools.spice.stage.judge``: ``max(tol_abs, tol_rel *
    |nominal|)``, a relative tolerance on a nominal of 0 counting for
    nothing, a non-finite tolerance being none) and ``recorded`` is
    ``False``. A tolerance of exactly 0 is a real limit either way.
    """
    if details is not None and "tolerance" in details:
        return _finite_number(details.get("tolerance")), True
    nominal = _finite_number(e.nominal.value)
    if nominal is None:
        return None, False
    limits: list[float] = []
    if e.tol_abs is not None:
        limits.append(abs(float(e.tol_abs.value)))
    if e.tol_rel is not None and nominal != 0.0:
        limits.append(abs(float(e.tol_rel.value)) * abs(nominal))
    if not limits or any(not math.isfinite(x) for x in limits):
        return None, False
    return max(limits), False


def tolerance_rows(ir: CircuitIR) -> list[ToleranceRow]:
    """One :class:`ToleranceRow` per expectation of ``ir.simulation``, from ``ir.validation.latest("spice.<id>")``: the recorded numbers, nothing recomputed.

    ``measured`` is ``details["measured"]`` of the latest recorded result
    (``None`` without one), ``status`` its status word, ``nominal`` the
    recorded ``details["nominal"]`` (the IR's nominal without one) and
    ``tolerance`` the recorded ``details["tolerance"]`` through
    :func:`expectation_limit`, which falls back to the stage's rule on the
    IR only for an expectation without a recorded tolerance
    (``tolerance_recorded`` says which). So a tolerance edited in the IR
    after the run does not move the band the recorded verdict was judged
    in. An expectation whose nominal is not a number is skipped. An IR
    without a simulation setup gives ``[]``.
    """
    sim = ir.simulation
    if sim is None:
        return []
    rows: list[ToleranceRow] = []
    for e in sim.expectations:
        nominal = _finite_number(e.nominal.value)
        if nominal is None:
            continue
        r = ir.validation.latest(f"{SPICE_CHECK}.{e.id}")
        details = r.details if r is not None and isinstance(r.details, dict) else None
        recorded_nominal = _finite_number(details.get("nominal")) if details is not None else None
        if recorded_nominal is not None:
            nominal = recorded_nominal
        measured = _finite_number(details.get("measured")) if details is not None else None
        tolerance, recorded = expectation_limit(e, details)
        rows.append(ToleranceRow(e.id, nominal, e.nominal.unit, measured, tolerance, r.status.value if r is not None else None, recorded))
    return rows


def tolerance_figure(rows: Sequence[ToleranceRow], *, title: str = "이론 공칭값 대 시뮬레이션 측정값", fig_id: str = "tolerance", width: int = COLUMN_PX) -> Figure:
    """One row per expectation: the tolerance band (−1 … +1, "허용치"), the nominal at 0 and the measured value as a marker at ``(measured − nominal) / tolerance``.

    The marker is coloured by the recorded status (PASS green, FAIL red,
    anything else gray) and the status word is printed beside the row label;
    its label reads ``측정 <value> / 공칭 <nominal>``. A row without a
    measurement draws no marker and says :data:`NO_MEASUREMENT`; a row
    without a usable tolerance draws no band (:data:`NO_TOLERANCE`); a row
    whose recorded tolerance is exactly 0 draws no band either but says
    :data:`ZERO_TOLERANCE` and puts the marker at 0 (an exact match) or on
    the edge (any deviation is beyond it). Markers beyond ±1.5 are clamped
    to the edge and their label carries a ``>`` mark (``편차 > 1.5 ×
    허용치``). The nominal line is painted per row over the band so it is
    visible inside it. ``ValueError`` without rows.
    """
    if not rows:
        raise ValueError("the tolerance figure needs at least one row")
    label_w = int(round(max(_text_width(f"{r.label} · {r.status or '기록 없음'}", _TEXT_PX) for r in rows))) + 16
    ml = min(max(120, label_w), 300)
    mr = 28
    top = 46
    legend_entries: list[tuple[str, str, str]] = [("box", BAND_FILL, "허용치 (±1)")]
    seen = {r.status for r in rows if r.measured is not None}
    for status, colour in STATUS_COLOURS.items():
        if status in seen:
            legend_entries.append(("dot", colour, status))
    if any(s not in STATUS_COLOURS for s in seen):
        legend_entries.append(("dot", STATUS_OTHER, "기타 (NOT_VERIFIED 등)"))
    legend, plot_top = _legend(legend_entries, 12, top, width - mr)
    plot_top += 6
    height = int(plot_top + len(rows) * _TOL_ROW_H + 52)
    out = _svg_open(width, height, title)
    out.append(_title_text(title, 12, 24))
    out += legend
    x_lo, x_hi = -_TOL_LIMIT, _TOL_LIMIT
    sx = _Scale(x_lo, x_hi, ml, width - mr, False)
    plot_bottom = plot_top + len(rows) * _TOL_ROW_H
    ticks = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    out.append('<g class="grid">')
    for t in ticks:
        out.append(f'<line x1="{_f(sx(t))}" y1="{_f(plot_top)}" x2="{_f(sx(t))}" y2="{_f(plot_bottom)}" stroke="{GRID}" stroke-width="1"/>')
    out.append("</g>")
    out.append(f'<text x="{_f(sx(0) + 5)}" y="{_f(plot_top - 4)}" font-size="{_SMALL_PX}" fill="{INK_SECONDARY}">공칭 (0)</text>')
    out.append('<g class="rows">')
    for i, r in enumerate(rows):
        y0 = plot_top + i * _TOL_ROW_H
        yc = y0 + _TOL_ROW_H / 2
        status_word = r.status or "기록 없음"
        out.append(f'<g class="row" data-label="{esc(r.label)}" data-status="{esc(status_word)}">')
        if i > 0:
            out.append(f'<line x1="{_f(ml)}" y1="{_f(y0)}" x2="{_f(width - mr)}" y2="{_f(y0)}" stroke="{GRID}" stroke-width="1"/>')
        out.append(f'<text x="{_f(ml - 10)}" y="{_f(yc + 4)}" text-anchor="end">{esc(r.label)}</text>')
        out.append(f'<text x="{_f(ml - 10)}" y="{_f(yc + 18)}" text-anchor="end" font-size="{_SMALL_PX}" fill="{INK_SECONDARY}">{esc(status_word)}</text>')
        if r.tolerance is not None and r.tolerance > 0:
            out.append(f'<rect class="band" x="{_f(sx(-1))}" y="{_f(y0 + 6)}" width="{_f(sx(1) - sx(-1))}" height="{_f(_TOL_ROW_H - 12)}" fill="{BAND_FILL}"/>')
        # the nominal line after the band (painted over it) and before the marker and the labels (painted under them)
        out.append(f'<line class="nominal" x1="{_f(sx(0))}" y1="{_f(y0)}" x2="{_f(sx(0))}" y2="{_f(y0 + _TOL_ROW_H)}" stroke="{INK_SECONDARY}" stroke-width="1.5"/>')
        nominal_text = si_format(r.nominal, r.unit)
        if r.measured is None:
            out.append(f'<text class="no-measurement" x="{_f(sx(0) + 12)}" y="{_f(yc + 4)}" fill="{INK_SECONDARY}">{esc(NO_MEASUREMENT)} / 공칭 {esc(nominal_text)}</text>')
        elif r.tolerance is None or r.tolerance < 0:
            out.append(f'<text class="no-tolerance" x="{_f(sx(0) + 12)}" y="{_f(yc + 4)}" fill="{INK_SECONDARY}">측정 {esc(si_format(r.measured, r.unit))} / 공칭 {esc(nominal_text)} ({NO_TOLERANCE})</text>')
        else:
            deviation = r.measured - r.nominal
            if r.tolerance > 0:
                ratio = deviation / r.tolerance
            else:  # a tolerance of 0: only an exact match is inside, anything else is beyond every edge
                ratio = 0.0 if deviation == 0 else math.copysign(math.inf, deviation)
            clamped = abs(ratio) > _TOL_LIMIT
            shown = max(-_TOL_LIMIT, min(_TOL_LIMIT, ratio))
            colour = STATUS_COLOURS.get(r.status or "", STATUS_OTHER)
            px = sx(shown)
            out.append(
                f'<circle class="marker" data-deviation="{_g(ratio, 4)}" data-clamped="{"true" if clamped else "false"}" cx="{_f(px)}" cy="{_f(yc)}" r="6" '
                f'fill="{colour}" stroke="{SURFACE}" stroke-width="2"/>'
            )
            text = f"측정 {si_format(r.measured, r.unit)} / 공칭 {nominal_text}"
            if r.tolerance == 0:
                text += f" ({ZERO_TOLERANCE})"
            if clamped:
                text += f" (편차 > {_g(_TOL_LIMIT)} × 허용치)"
            flip = px + 12 + _text_width(text, _TEXT_PX) > width - mr
            out.append(f'<text class="marker-label" x="{_f(px - 12 if flip else px + 12)}" y="{_f(yc + 4)}" text-anchor="{"end" if flip else "start"}">{esc(text)}</text>')
        out.append("</g>")
    out.append("</g>")
    out.append(f'<line class="axis" x1="{_f(ml)}" y1="{_f(plot_bottom)}" x2="{_f(width - mr)}" y2="{_f(plot_bottom)}" stroke="{AXIS}" stroke-width="1"/>')
    out.append('<g class="ticks">')
    for t in ticks:
        label = f"{'+' if t > 0 else ''}{_g(t)}"
        out.append(f'<text class="tick" data-value="{_g(t)}" x="{_f(sx(t))}" y="{_f(plot_bottom + 16)}" text-anchor="middle" fill="{INK_SECONDARY}">{label}</text>')
    out.append("</g>")
    out.append(f'<text class="axis-title" x="{_f((ml + width - mr) / 2)}" y="{_f(height - 12)}" text-anchor="middle">허용치 대비 편차 = (측정 − 공칭) / 허용치</text>')
    out.append("</svg>")
    n_measured = sum(1 for r in rows if r.measured is not None)
    if all(r.tolerance_recorded for r in rows if r.measured is not None):
        source = "판정·허용치·측정값은 SPICE 단계가 기록한 값 그대로이며 그림은 점의 위치 외에 아무것도 다시 계산하지 않습니다."
    else:
        source = ("판정과 측정값은 SPICE 단계가 기록한 값 그대로; 허용치는 기록된 값이고, 기록이 없는 기대값만 SPICE 단계의 규칙 max(tol_abs, tol_rel·|공칭값|) 로 "
                  "IR 에서 계산한 값입니다 (공칭값이 0 이면 tol_abs 만).")
    caption = (
        f"기대값 {len(rows)}개 중 측정값이 기록된 {n_measured}개의 편차를 허용치 단위로 표시. 띠 = 허용치 ±1, 점 = 시뮬레이션 측정값(색은 기록된 판정: 초록 PASS, 빨강 FAIL, 회색 그 외), "
        f"세로선 = 이론 공칭값. {source}"
    )
    return Figure(fig_id, title, caption, "\n".join(out) + "\n")


# --------------------------------------------------------------------------- bars

_BAR_SLOT = 64
_BAR_W = 24
#: a bar slot narrower than this (px) gets a vertical category label: slanted ones would overlap their neighbours
_NARROW_SLOT_PX = 16
#: the smallest text size a figure is authored with (px)
_MIN_TEXT_PX = 10


def bar_figure(labels: Sequence[str], values: Sequence[float], *, title: str, y_label: str, unit: str = "", fig_id: str = "bars", height: int = 360, max_width: int = COLUMN_PX) -> Figure:
    """Single-series vertical bars (blue) with a direct value label on every bar that has room for one, e.g. per-net copper length.

    ``unit`` labels the values (``mm``); the y axis title carries it too.
    Bars are 24 px wide in 64 px slots (the chart is as wide as its bars
    need, at most ``max_width``), rounded at the data end and square at the
    baseline, with a 2 px gap kept by the slot. When the slots get too
    narrow for the value labels (many bars at ``max_width``: the 64 nets of
    a dev board) no bar carries one - the caption lists every value and each
    bar keeps it in ``data-value`` - and when they are narrower than
    :data:`_NARROW_SLOT_PX` the category labels stand vertical at the 10 px
    minimum text size, the figure growing downwards to hold the longest.
    ``ValueError`` for empty input, mismatched lengths or a non-finite value.
    """
    if not labels or len(labels) != len(values):
        raise ValueError(f"labels and values must be non-empty and of equal length (got {len(labels)} and {len(values)})")
    for label, v in zip(labels, values):
        if not math.isfinite(v):
            raise ValueError(f"bar {label!r} has a non-finite value ({v!r})")
    n = len(values)
    exponent = si_exponent(max(abs(v) for v in values) or 1.0)
    value_texts = [si_format(v, unit, exponent=exponent) for v in values]
    y_lo = min(0.0, min(values))
    y_hi = max(0.0, max(values))
    if y_hi == y_lo:
        y_hi = y_lo + 1.0
    y_hi += (y_hi - y_lo) * 0.12  # room for the value labels
    y_probe = _tick_labels(nice_ticks(y_lo, y_hi), unit, False)
    ml = int(round(22 + max(_text_width(t, _TEXT_PX) for t in y_probe) + 10))
    mr = 24
    top = 44
    bottom = height - 50
    slot = _BAR_SLOT
    width = ml + n * slot + mr
    if width > max_width:
        width = max_width
        slot = (width - ml - mr) / n
    bar_w = min(_BAR_W, max(2.0, slot - 4))
    # value labels only when every one fits its slot (a partial set would read as missing data); vertical category labels in slots too narrow for slanted ones
    value_labels = all(_text_width(t, _SMALL_PX) <= slot - 2 for t in value_texts)
    vertical = slot < _NARROW_SLOT_PX
    svg_height = height
    if vertical:
        svg_height = height + max(0, int(math.ceil(max(_text_width(label, _MIN_TEXT_PX) for label in labels) + 12 - (height - bottom))))
    out = _svg_open(width, svg_height, title)
    out.append(_title_text(title, 12, 24))
    sy = _Scale(y_lo, y_hi, bottom, top, False)
    y_ticks = nice_ticks(y_lo, y_hi)
    out.append('<g class="grid">')
    for t in y_ticks:
        out.append(f'<line x1="{_f(ml)}" y1="{_f(sy(t))}" x2="{_f(width - mr)}" y2="{_f(sy(t))}" stroke="{GRID}" stroke-width="1"/>')
    out.append("</g>")
    out.append(f'<line class="axis" x1="{_f(ml)}" y1="{_f(sy(0.0))}" x2="{_f(width - mr)}" y2="{_f(sy(0.0))}" stroke="{AXIS}" stroke-width="1"/>')
    out.append(f'<line class="axis" x1="{_f(ml)}" y1="{_f(top)}" x2="{_f(ml)}" y2="{_f(bottom)}" stroke="{AXIS}" stroke-width="1"/>')
    out.append('<g class="ticks">')
    for t, text in zip(y_ticks, _tick_labels(y_ticks, unit, False)):
        out.append(f'<text x="{_f(ml - 6)}" y="{_f(sy(t) + 4)}" text-anchor="end" fill="{INK_SECONDARY}">{esc(text)}</text>')
    out.append("</g>")
    out.append('<g class="bars">')
    base = sy(0.0)
    for i, (label, v, text) in enumerate(zip(labels, values, value_texts)):
        cx = ml + slot * (i + 0.5)
        x0 = cx - bar_w / 2
        y_top = sy(v)
        top_px, bottom_px = (y_top, base) if v >= 0 else (base, y_top)
        h = bottom_px - top_px
        r = min(4.0, bar_w / 2, h)
        if h <= 0.5:
            path = f"M{_f(x0)},{_f(base)} h{_f(bar_w)} v-0.5 h-{_f(bar_w)} Z"
        elif v >= 0:
            path = (
                f"M{_f(x0)},{_f(base)} V{_f(top_px + r)} Q{_f(x0)},{_f(top_px)} {_f(x0 + r)},{_f(top_px)} H{_f(x0 + bar_w - r)} "
                f"Q{_f(x0 + bar_w)},{_f(top_px)} {_f(x0 + bar_w)},{_f(top_px + r)} V{_f(base)} Z"
            )
        else:
            path = (
                f"M{_f(x0)},{_f(base)} V{_f(bottom_px - r)} Q{_f(x0)},{_f(bottom_px)} {_f(x0 + r)},{_f(bottom_px)} H{_f(x0 + bar_w - r)} "
                f"Q{_f(x0 + bar_w)},{_f(bottom_px)} {_f(x0 + bar_w)},{_f(bottom_px - r)} V{_f(base)} Z"
            )
        out.append(f'<path class="bar" data-label="{esc(label)}" data-value="{_g(v)}" d="{path}" fill="{SERIES_COLOURS[0]}"/>')
        if value_labels:
            ly = top_px - 6 if v >= 0 else bottom_px + 14
            out.append(f'<text class="bar-label" x="{_f(cx)}" y="{_f(ly)}" text-anchor="middle" font-size="{_SMALL_PX}">{esc(text)}</text>')
        rotate = _text_width(label, _TEXT_PX) > slot - 6
        if vertical:
            lx, ly = cx + _MIN_TEXT_PX * 0.35, bottom + 6
            out.append(
                f'<text class="category" x="{_f(lx)}" y="{_f(ly)}" text-anchor="end" font-size="{_MIN_TEXT_PX}" fill="{INK_SECONDARY}" '
                f'transform="rotate(-90 {_f(lx)} {_f(ly)})">{esc(label)}</text>'
            )
        elif rotate:
            out.append(
                f'<text class="category" x="{_f(cx)}" y="{_f(bottom + 14)}" text-anchor="end" fill="{INK_SECONDARY}" '
                f'transform="rotate(-30 {_f(cx)} {_f(bottom + 14)})">{esc(label)}</text>'
            )
        else:
            out.append(f'<text class="category" x="{_f(cx)}" y="{_f(bottom + 16)}" text-anchor="middle" fill="{INK_SECONDARY}">{esc(label)}</text>')
    out.append("</g>")
    out.append(f'<text class="axis-title" x="14" y="{_f((top + bottom) / 2)}" text-anchor="middle" transform="rotate(-90 14 {_f((top + bottom) / 2)})">{esc(y_label)}</text>')
    out.append("</svg>")
    caption = f"{title}: {', '.join(f'{label} {text}' for label, text in zip(labels, value_texts))}."
    return Figure(fig_id, title, caption, "\n".join(out) + "\n")
