"""Approximate envelope (axis-aligned box) of a STEP model, read from the file on disk.

Invariant: the box is an **envelope, never a measurement**. The true solid
lies inside it; the box may be larger. It is read only from the STEP file
(ISO 10303-21) that a footprint's ``(model ...)`` reference names - nothing
comes from memory or from a table of package heights.

What is read (the ``DATA`` section's instances ``#N = ...;``; a record may
span several lines, strings are quoted with ``''`` escapes, ``/* */``
comments are ignored):

* every 3-coordinate ``CARTESIAN_POINT``. Two-coordinate points are the
  parameter-space points of pcurves and are skipped. B-spline control points
  bound their curve (convex hull), so they may lie *outside* the shape - one
  reason the box is an envelope. The origin of an ``AXIS2_PLACEMENT_3D`` is a
  point too.
* every ``CIRCLE`` / ``ELLIPSE`` placed by an ``AXIS2_PLACEMENT_3D``: the box
  of the **whole** circle (along axis *i* the centre +/- ``r * sqrt(1 - n_i^2)``,
  ``n`` the circle's unit axis; an ellipse counts as the circle of its larger
  semi-axis), even when only an arc of it bounds a face - the second reason.
  Without this a cylinder (a capacitor can, a pin) would contribute only its
  seam vertices.
* the **length unit** of every ``GLOBAL_UNIT_ASSIGNED_CONTEXT``: an
  ``SI_UNIT(prefix, .METRE.)`` (``.MILLI.`` = mm, ``$`` = m, ``.CENTI.``,
  ``.DECI.``, ``.MICRO.``, ``.NANO.``, ``.KILO.``) or a
  ``CONVERSION_BASED_UNIT`` whose ``LENGTH_MEASURE_WITH_UNIT`` resolves to one
  (``INCH`` = ``LENGTH_MEASURE(25.4)`` of millimetres, or ``0.0254`` of
  metres). Every coordinate and radius is converted to millimetres.

Refused - :attr:`StepEnvelope.box` is ``None`` and :attr:`StepEnvelope.reason`
says why (the caller draws no body rather than a wrong one): a file that is
not ISO-10303-21, no length unit, an unknown length unit, contexts that
disagree on the unit, no 3-D point, a non-finite number, and an assembly
whose parts are placed by a non-identity ``ITEM_DEFINED_TRANSFORMATION`` or
by ``MAPPED_ITEM`` (their points are in several frames; the transforms are
not applied here). Gzip-compressed STEP (``.stpz``, KiCad reads it) is
decompressed first. Full spheres / tori bounded by a single vertex are not
widened (not seen in the KiCad 10 library sample this was written against).
"""

from __future__ import annotations

import gzip
import math
import re
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Box3", "StepEnvelope", "read_step_envelope", "step_bbox", "parse_step_text", "METRE_PREFIX_MM"]

#: ``SI_UNIT`` prefix -> millimetres per unit of that prefix times metre
METRE_PREFIX_MM: dict[str, float] = {
    "$": 1000.0,
    ".KILO.": 1.0e6,
    ".DECI.": 100.0,
    ".CENTI.": 10.0,
    ".MILLI.": 1.0,
    ".MICRO.": 1.0e-3,
    ".NANO.": 1.0e-6,
}

_STRING = r"'(?:[^']|'')*'"
_COMMENT_OR_STRING_RE = re.compile(r"'(?:[^']|'')*'|/\*.*?\*/", re.S)
_RECORD_RE = re.compile(r"#(\d+)\s*=\s*((?:[^;']++|'(?:[^']|'')*+')*+);")
_WS_RE = re.compile(r"\s+")
_POINT_RE = re.compile(rf"^CARTESIAN_POINT\({_STRING},\(([^()]*)\)\)$")
_DIRECTION_RE = re.compile(rf"^DIRECTION\({_STRING},\(([^()]*)\)\)$")
_AXIS3_RE = re.compile(rf"^AXIS2_PLACEMENT_3D\({_STRING},#(\d+),(#\d+|\$),(#\d+|\$)\)$")
_CIRCLE_RE = re.compile(rf"^CIRCLE\({_STRING},#(\d+),([^,()]+)\)$")
_ELLIPSE_RE = re.compile(rf"^ELLIPSE\({_STRING},#(\d+),([^,()]+),([^,()]+)\)$")
_IDT_RE = re.compile(rf"^ITEM_DEFINED_TRANSFORMATION\({_STRING},{_STRING},#(\d+),#(\d+)\)$")
_GUAC_RE = re.compile(r"GLOBAL_UNIT_ASSIGNED_CONTEXT\(\(([^()]*)\)\)")
_SI_METRE_RE = re.compile(r"SI_UNIT\((\$|\.[A-Z]+\.),\.METRE\.\)")
_CONVERSION_RE = re.compile(rf"CONVERSION_BASED_UNIT\({_STRING},#(\d+)\)")
_LENGTH_MEASURE_RE = re.compile(r"LENGTH_MEASURE\(([^()]+)\),#(\d+)\)")
_REF_RE = re.compile(r"#(\d+)")


@dataclass(frozen=True, slots=True)
class Box3:
    """Axis-aligned box in millimetres (``x1 <= x2`` ...)."""

    x1: float
    y1: float
    z1: float
    x2: float
    y2: float
    z2: float

    @property
    def size(self) -> tuple[float, float, float]:
        return (self.x2 - self.x1, self.y2 - self.y1, self.z2 - self.z1)

    def corners(self) -> list[tuple[float, float, float]]:
        """The eight corners (x fastest, then y, then z)."""
        return [(x, y, z) for z in (self.z1, self.z2) for y in (self.y1, self.y2) for x in (self.x1, self.x2)]

    @staticmethod
    def around(points: list[tuple[float, float, float]]) -> "Box3":
        xs, ys, zs = zip(*points)
        return Box3(min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))

    def union(self, other: "Box3") -> "Box3":
        return Box3(min(self.x1, other.x1), min(self.y1, other.y1), min(self.z1, other.z1), max(self.x2, other.x2), max(self.y2, other.y2), max(self.z2, other.z2))


@dataclass(frozen=True, slots=True)
class StepEnvelope:
    """What :func:`read_step_envelope` found: the box (mm) or the reason there is none."""

    box: Box3 | None
    unit_mm: float | None  # millimetres per model length unit
    points: int  # 3-D CARTESIAN_POINTs read
    circles: int  # CIRCLE / ELLIPSE curves that widened the box
    reason: str = ""  # empty when box is not None


def step_bbox(path: Path | str) -> Box3 | None:
    """The approximate envelope of the STEP model at ``path`` in millimetres, or ``None`` (see :func:`read_step_envelope` for why)."""
    return read_step_envelope(path).box


def read_step_envelope(path: Path | str) -> StepEnvelope:
    """Read ``path`` and return its envelope; an unreadable file is a ``None`` box with the reason (never an exception)."""
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        return StepEnvelope(None, None, 0, 0, f"STEP 파일을 읽을 수 없음 ({type(exc).__name__})")
    if raw[:2] == b"\x1f\x8b":
        try:
            raw = gzip.decompress(raw)
        except (OSError, EOFError) as exc:
            return StepEnvelope(None, None, 0, 0, f"압축된 STEP 파일을 풀 수 없음 ({type(exc).__name__})")
    return parse_step_text(raw.decode("latin-1"))


def _floats(text: str) -> list[float] | None:
    try:
        return [float(v) for v in text.split(",")] if text else []
    except ValueError:
        return None


def _records(text: str) -> dict[int, str]:
    """``{N: body}`` of every instance, whitespace removed (names inside strings lose their spaces; nothing here reads them)."""
    cleaned = _COMMENT_OR_STRING_RE.sub(lambda m: m.group(0) if m.group(0).startswith("'") else " ", text)
    return {int(m.group(1)): _WS_RE.sub("", m.group(2)) for m in _RECORD_RE.finditer(cleaned)}


def _length_unit_mm(records: dict[int, str], ref: int, depth: int = 0) -> float | None:
    """Millimetres per unit of the length unit instance ``#ref`` (SI metre with a prefix, or a conversion to one); ``None`` = unknown."""
    body = records.get(ref)
    if body is None or depth > 4:
        return None
    si = _SI_METRE_RE.search(body)
    if si is not None:
        return METRE_PREFIX_MM.get(si.group(1))
    conv = _CONVERSION_RE.search(body)
    if conv is None:
        return None
    measure = records.get(int(conv.group(1)))
    lm = _LENGTH_MEASURE_RE.search(measure) if measure else None
    if lm is None:
        return None
    try:
        value = float(lm.group(1))
    except ValueError:
        return None
    base = _length_unit_mm(records, int(lm.group(2)), depth + 1)
    if base is None or not math.isfinite(value) or value <= 0:
        return None
    return value * base


def _unit_of(records: dict[int, str]) -> tuple[float | None, str]:
    factors: set[float] = set()
    for body in records.values():
        for m in _GUAC_RE.finditer(body):
            for ref in _REF_RE.findall(m.group(1)):
                unit = records.get(int(ref), "")
                if "LENGTH_UNIT" not in unit:
                    continue
                mm = _length_unit_mm(records, int(ref))
                if mm is None:
                    return None, f"알 수 없는 길이 단위 (#{ref})"
                factors.add(mm)
    if not factors:
        return None, "길이 단위(GLOBAL_UNIT_ASSIGNED_CONTEXT 의 LENGTH_UNIT)가 없음"
    if len(factors) > 1:
        return None, f"문맥마다 길이 단위가 다름 ({sorted(factors)} mm)"
    return factors.pop(), ""


def _placement(records: dict[int, str], ref: int) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]] | None:
    """(location, axis, ref_direction) of an ``AXIS2_PLACEMENT_3D`` (defaults z / x for ``$``), or ``None``."""
    m = _AXIS3_RE.match(records.get(ref, ""))
    if m is None:
        return None
    loc = _POINT_RE.match(records.get(int(m.group(1)), ""))
    coords = _floats(loc.group(1)) if loc else None
    if coords is None or len(coords) != 3:
        return None

    def direction(token: str, default: tuple[float, float, float]) -> tuple[float, ...] | None:
        if token == "$":
            return default
        d = _DIRECTION_RE.match(records.get(int(token[1:]), ""))
        values = _floats(d.group(1)) if d else None
        if values is None or len(values) != 3:
            return None
        norm = math.sqrt(sum(v * v for v in values))
        return tuple(v / norm for v in values) if norm > 0 and math.isfinite(norm) else None

    axis = direction(m.group(2), (0.0, 0.0, 1.0))
    ref_dir = direction(m.group(3), (1.0, 0.0, 0.0))
    if axis is None or ref_dir is None:
        return None
    return tuple(coords), axis, ref_dir


def _is_identity_pair(a: tuple | None, b: tuple | None) -> bool:
    if a is None or b is None:
        return False
    return all(math.isclose(x, y, abs_tol=1e-9) for va, vb in zip(a, b) for x, y in zip(va, vb))


def parse_step_text(text: str) -> StepEnvelope:
    """The envelope of the STEP model in ``text`` (the module docstring lists what is read and what is refused)."""
    if not text.lstrip("﻿ \t\r\n").startswith("ISO-10303-21"):
        return StepEnvelope(None, None, 0, 0, "ISO-10303-21 (STEP) 파일이 아님")
    records = _records(text)
    if not records:
        return StepEnvelope(None, None, 0, 0, "STEP DATA 영역에 개체가 없음")
    unit, why = _unit_of(records)
    if unit is None:
        return StepEnvelope(None, None, 0, 0, why)
    for body in records.values():
        if body.startswith("MAPPED_ITEM("):
            return StepEnvelope(None, unit, 0, 0, "MAPPED_ITEM 으로 배치된 부분이 있음 (변환을 적용하지 않으므로 상자를 만들지 않음)")
        idt = _IDT_RE.match(body)
        if idt is not None and not _is_identity_pair(_placement(records, int(idt.group(1))), _placement(records, int(idt.group(2)))):
            return StepEnvelope(None, unit, 0, 0, "조립체의 부분이 항등이 아닌 변환(ITEM_DEFINED_TRANSFORMATION)으로 배치됨 (적용하지 않으므로 상자를 만들지 않음)")
    lo = [math.inf, math.inf, math.inf]
    hi = [-math.inf, -math.inf, -math.inf]
    points = circles = 0

    def widen(c: tuple[float, ...], half: tuple[float, float, float]) -> None:
        for i in range(3):
            lo[i] = min(lo[i], c[i] - half[i])
            hi[i] = max(hi[i], c[i] + half[i])

    for body in records.values():
        if body.startswith("CARTESIAN_POINT("):
            m = _POINT_RE.match(body)
            coords = _floats(m.group(1)) if m else None
            if coords is None:
                return StepEnvelope(None, unit, points, circles, "CARTESIAN_POINT 를 읽을 수 없음")
            if len(coords) != 3:
                continue  # parameter-space point of a pcurve
            if not all(math.isfinite(v) for v in coords):
                return StepEnvelope(None, unit, points, circles, "CARTESIAN_POINT 에 유한하지 않은 수가 있음")
            points += 1
            widen(tuple(v * unit for v in coords), (0.0, 0.0, 0.0))
        elif body.startswith(("CIRCLE(", "ELLIPSE(")):
            m = _CIRCLE_RE.match(body) or _ELLIPSE_RE.match(body)
            if m is None:
                continue
            radii = _floats(",".join(m.groups()[1:]))
            placement = _placement(records, int(m.group(1)))
            if placement is None:
                continue  # a 2-D placement: a pcurve in parameter space
            if radii is None or not all(math.isfinite(r) and r >= 0 for r in radii):
                return StepEnvelope(None, unit, points, circles, "CIRCLE / ELLIPSE 반지름을 읽을 수 없음")
            centre, axis, _ = placement
            r = max(radii) * unit
            circles += 1
            widen(tuple(v * unit for v in centre), tuple(r * math.sqrt(max(0.0, 1.0 - n * n)) for n in axis))  # type: ignore[arg-type]
    if points == 0:
        return StepEnvelope(None, unit, 0, circles, "3차원 CARTESIAN_POINT 가 없음")
    return StepEnvelope(Box3(lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]), unit, points, circles)
