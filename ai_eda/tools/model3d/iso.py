"""Isometric (or top / bottom) SVG of a :class:`~ai_eda.tools.model3d.scene.Scene` by the painter's algorithm - deterministic, no status.

Invariant: the picture is a view of the scene only - it reads no file and
judges nothing; the same scene and view give the same bytes (fixed float
formatting, total-order sorts, no time, no path).

Painter's algorithm, with the order the board's structure allows (orthographic
view, eye on one side of the board; "near" = the side the eye is on):

1. bodies on the far side of the board (only their edges past the board can
   show), then 2. the board slab, then 3. the near side's flat layers (the
   decals: copper, mask, pads, silk, outlines, drill caps) by their height -
   an eye above a plane never sees a point above it hidden by something in
   it - then 4. the near side's bodies. Faces facing away from the eye are
   culled; decals on the far side are never drawn (the opaque slab hides
   them); a drill shows only the cap facing the eye.

Bodies (steps 1 and 4) are ordered pairwise: when the footprints of two
bodies are separated by a vertical plane (separating-axis test on their
outlines), the one on the far side of that plane is drawn first; overlapping
footprints, and any cycle, fall back to the depth of the footprint's
centroid. A convex prism's own front faces never overlap, so their order
within a body does not matter. This is a picture of boxes, not a hidden-line
proof: interpenetrating bodies (overlapping ``F.Fab`` boxes) may be drawn in
either order.

Shading is Lambert's with a fixed light, faces are filled with the scene's
material colours (the mask is translucent), bodies and the slab get a thin
darker edge. Each polygon has ``class="k-<kind> g-<group>"`` so a viewer can
toggle the scene's groups (``board`` / ``copper`` / ``silk`` / ``parts``) and a
body's faces carry ``data-ref``. The caption (Korean, wrapped inside the
figure) is :func:`~ai_eda.tools.model3d.scene.scene_caption`.
"""

from __future__ import annotations

import math
from xml.sax.saxutils import escape

from ai_eda.tools.model3d.scene import MATERIALS, Face, Scene, Solid, scene_caption, solid_faces

__all__ = ["VIEWS", "iso_svg"]

#: view name -> (azimuth, elevation) in degrees; azimuth 0 = the eye in front of the board (board +y side), positive = to the right
VIEWS: dict[str, tuple[float, float]] = {
    "iso": (35.0, 35.264),
    "iso_bottom": (35.0, -35.264),
    "top": (0.0, 90.0),
    "bottom": (0.0, -90.0),
}
_FONT = '"Noto Sans CJK KR", "Noto Sans KR", "Malgun Gothic", "Apple SD Gothic Neo", "WenQuanYi Zen Hei", system-ui, sans-serif'
_MARGIN = 16
_CAPTION_PX = 12
_CAPTION_LINE = 16
_AMBIENT = 0.55
_DIFFUSE = 0.45
_MATERIAL = {m.key: m for m in MATERIALS}
_MATERIAL_ORDER = {m.key: i for i, m in enumerate(MATERIALS)}
_EDGED = {"slab": "#3b3526", "body": "#111418"}

Vec = tuple[float, float, float]


def _norm(v: Vec) -> Vec:
    n = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    return (v[0] / n, v[1] / n, v[2] / n)


def _dot(a: Vec, b: Vec) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _camera(view: str) -> tuple[Vec, Vec, Vec]:
    """``(right, up, toward_eye)`` in KiCad's 3D frame (x, y up = -board y, z)."""
    if view not in VIEWS:
        raise ValueError(f"unknown view {view!r}; one of {sorted(VIEWS)}")
    az, el = VIEWS[view]
    if el >= 90.0:
        return (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
    if el <= -90.0:  # seen from below like a flipped board: x runs to the left
        return (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)
    a, e = math.radians(az), math.radians(el)
    d = (math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e))
    r = (math.cos(a), math.sin(a), 0.0)
    u = (-math.sin(a) * math.sin(e), math.cos(a) * math.sin(e), math.cos(e))
    return r, u, d


def _hex(rgb: tuple[float, float, float], shade: float) -> str:
    return "#" + "".join(f"{max(0, min(255, round(c * shade * 255))):02x}" for c in rgb)


def _separated(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> tuple[float, float] | None:
    """A unit axis along which the convex outlines ``a`` and ``b`` do not overlap, pointing from ``a`` to ``b``; ``None`` when they overlap."""
    for poly in (a, b):
        n = len(poly)
        for i in range(n):
            (x0, y0), (x1, y1) = poly[i], poly[(i + 1) % n]
            ax, ay = y1 - y0, x0 - x1
            length = math.hypot(ax, ay)
            if length < 1e-12:
                continue
            ax, ay = ax / length, ay / length
            pa = [x * ax + y * ay for x, y in a]
            pb = [x * ax + y * ay for x, y in b]
            if max(pa) <= min(pb) + 1e-9:
                return (ax, ay)
            if max(pb) <= min(pa) + 1e-9:
                return (-ax, -ay)
    return None


def _order_bodies(bodies: list[tuple[int, Solid]], d: Vec) -> list[tuple[int, Solid]]:
    """Far-to-near order of body prisms (module docstring, step 4); deterministic."""
    rings = [[(x, -y) for x, y in s.polygon] for _, s in bodies]
    depth = [_dot((sum(p[0] for p in r) / len(r), sum(p[1] for p in r) / len(r), 0.0), d) for r in rings]
    n = len(bodies)
    before: list[set[int]] = [set() for _ in range(n)]  # before[j] = bodies that must be drawn before j
    for i in range(n):
        for j in range(i + 1, n):
            axis = _separated(rings[i], rings[j])
            if axis is None:
                continue
            toward_eye = axis[0] * d[0] + axis[1] * d[1]
            if toward_eye > 1e-12:  # the eye is on j's side: i is behind
                before[j].add(i)
            elif toward_eye < -1e-12:
                before[i].add(j)
    done: list[int] = []
    left = set(range(n))
    while left:
        ready = [k for k in left if not (before[k] & left)]
        pick = min(ready or left, key=lambda k: (depth[k], bodies[k][0]))
        done.append(pick)
        left.discard(pick)
    return [bodies[k] for k in done]


def _wrap(text: str, max_px: float) -> list[str]:
    """``text`` in lines no wider than ``max_px`` (estimated: CJK 1 em, others 0.56 em), joining its ``"; "`` clauses where they fit."""
    def w(s: str) -> float:
        return sum(_CAPTION_PX * (1.0 if ord(ch) > 0x2E7F else 0.56) for ch in s)

    lines: list[str] = []
    for piece in text.split("; "):
        if lines and w(lines[-1] + "; " + piece) <= max_px:
            lines[-1] += "; " + piece
            continue
        while w(piece) > max_px and len(piece) > 1:  # a clause wider than the figure is cut where it reaches the edge
            cut = max(1, max(k for k in range(1, len(piece) + 1) if w(piece[:k]) <= max_px))
            lines.append(piece[:cut])
            piece = piece[cut:]
        lines.append(piece)
    return lines


def iso_svg(scene: Scene, view: str = "iso", *, width: int = 960, caption: bool = True) -> str:
    """The scene seen from ``view`` (:data:`VIEWS`) as a standalone SVG ``width`` px wide (module docstring)."""
    r, u, d = _camera(view)
    T = scene.thickness_mm
    eye_above = d[2] > 0
    light = _norm((0.35, -0.55, 0.76 if eye_above else -0.76))

    def near_height(z: float) -> float:
        return z if eye_above else T - z

    drawn: list[tuple[tuple, Face, Solid]] = []  # (sort key, face, solid)
    far_bodies: list[tuple[int, Solid]] = []
    near_bodies: list[tuple[int, Solid]] = []
    for index, s in enumerate(scene.solids):
        if s.kind == "body":
            near = (s.z0 >= T - 1e-9) if eye_above else (s.z1 <= 1e-9)
            (near_bodies if near else far_bodies).append((index, s))
            continue
        if s.kind == "slab":
            for f in solid_faces(s):
                if _dot(f.normal, d) > 1e-9:
                    drawn.append(((1, 0.0, 0, index), f, s))
            continue
        for f in solid_faces(s, sides=False):  # decals and drill caps: only the face toward the eye
            if _dot(f.normal, d) <= 1e-9:
                continue
            drawn.append(((2, near_height(f.points[0][2]), _MATERIAL_ORDER[s.material], index), f, s))
    for tier, group in ((0, far_bodies), (3, near_bodies)):
        for rank, (index, s) in enumerate(_order_bodies(group, d)):
            for f in solid_faces(s):
                if _dot(f.normal, d) > 1e-9:
                    drawn.append(((tier, float(rank), 0 if f.which == "side" else 1, index), f, s))
    drawn.sort(key=lambda item: item[0])

    projected: list[tuple[list[tuple[float, float]], Face, Solid]] = []
    xs: list[float] = []
    ys: list[float] = []
    for _, f, s in drawn:
        pts = [(_dot(p, r), -_dot(p, u)) for p in f.points]
        projected.append((pts, f, s))
        xs += [p[0] for p in pts]
        ys += [p[1] for p in pts]
    if not xs:
        xs, ys = [0.0, 1.0], [0.0, 1.0]
    x0, y0 = min(xs), min(ys)
    span_x, span_y = max(max(xs) - x0, 1e-9), max(max(ys) - y0, 1e-9)
    scale = (width - 2 * _MARGIN) / span_x
    view_h = span_y * scale + 2 * _MARGIN
    lines = _wrap(scene_caption(scene), width - 2 * _MARGIN) if caption else []
    height = int(math.ceil(view_h + (len(lines) * _CAPTION_LINE + 8 if lines else 0)))
    title = f"{scene.project_id}: 3D 미리보기 ({view})"
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" '
        f'font-family=\'{_FONT}\'>',
        f"<title>{escape(title)}</title>",
        f'<g class="scene" data-view="{escape(view)}">',
    ]
    for pts, f, s in projected:
        mat = _MATERIAL[s.material]
        shade = _AMBIENT + _DIFFUSE * max(0.0, _dot(f.normal, light))
        coords = " ".join(f"{_MARGIN + (x - x0) * scale:.2f},{_MARGIN + (y - y0) * scale:.2f}" for x, y in pts)
        attrs = f'class="k-{s.kind} g-{mat.group}" points="{coords}" fill="{_hex(mat.rgba[:3], shade)}"'
        if mat.rgba[3] < 1.0:
            attrs += f' fill-opacity="{mat.rgba[3]:.2f}"'
        if s.kind in _EDGED:
            attrs += f' stroke="{_EDGED[s.kind]}" stroke-width="0.6" stroke-linejoin="round" stroke-opacity="0.6"'
        if s.kind == "body":
            attrs += f' data-ref="{escape(s.label, {chr(34): "&quot;"})}"'
        out.append(f"<polygon {attrs}/>")
    out.append("</g>")
    for i, line in enumerate(lines):
        y = view_h + 4 + _CAPTION_LINE * (i + 1) - 4
        out.append(f'<text class="caption" x="{_MARGIN}" y="{y:.2f}" font-size="{_CAPTION_PX}" fill="#555555">{escape(line)}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"
