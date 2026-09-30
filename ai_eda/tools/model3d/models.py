"""3D model references of a library footprint: discovery of the 3D library, path resolution, KiCad's model transform.

Invariant: a body height comes only from a STEP file on disk that the
footprint's own ``(model "..." (offset ..) (scale ..) (rotate ..))`` names;
a reference that does not resolve to a file under the 3D library is *no
body*, with the reason - never a guessed height.

* :func:`find_3dmodel_dir`: ``$KICAD10_3DMODEL_DIR`` (used only when it is an
  existing directory), else ``$KICAD_3DMODEL_DIR`` (same rule), else
  ``<root>/3dmodels`` next to the ``symbols`` / ``footprints`` of the
  :class:`~ai_eda.tools.kicad.library.KicadLibrary` roots (``share/kicad`` of
  an installation). ``None`` when none exists.
* :func:`resolve_model_path`: ``${KICAD10_3DMODEL_DIR}/X.3dshapes/Y.step``
  (also KiCad's older ``$(VAR)`` spelling) resolves under that directory and
  must stay inside it; an absolute path is used when the file exists; any
  other variable or a relative path is unresolved (reason given).
* :func:`model_transform` / :func:`transformed_box`: KiCad's composition for a
  footprint model, as in its 3D viewer source (not measured here):
  ``T(offset) * Rz(-rz) * Ry(-ry) * Rx(-rx) * S(scale)`` applied to a model
  point, in KiCad's 3D frame (x right, **y up** = -board y, z up out of the
  footprint's side), offsets in millimetres, results rounded to 1e-6 mm. A legacy ``(at (xyz ..))`` is in
  inches and is converted (KiCad reads it that way).
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path

from ai_eda.tools.kicad import sexpr
from ai_eda.tools.kicad.library import FootprintDef, KicadLibrary
from ai_eda.tools.model3d.step_bbox import Box3

__all__ = [
    "MODEL_DIR_VARIABLE",
    "ModelRef",
    "footprint_models",
    "find_3dmodel_dir",
    "resolve_model_path",
    "model_transform",
    "transformed_box",
]

#: the path variable KiCad 10's footprint library writes in every ``(model ...)``
MODEL_DIR_VARIABLE = "KICAD10_3DMODEL_DIR"
_ENV_VARIABLES = (MODEL_DIR_VARIABLE, "KICAD_3DMODEL_DIR")
_VAR_RE = re.compile(r"^\$(?:\{([A-Za-z0-9_]+)\}|\(([A-Za-z0-9_]+)\))[\\/]?(.*)$")
_STEP_SUFFIXES = (".step", ".stp", ".stpz")


@dataclass(frozen=True, slots=True)
class ModelRef:
    """One ``(model ...)`` of a footprint, as written in the library."""

    path: str  # as written, e.g. "${KICAD10_3DMODEL_DIR}/Package_TO_SOT_THT.3dshapes/TO-92_Inline.step"
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)  # mm, KiCad 3D frame (y up)
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    rotate: tuple[float, float, float] = (0.0, 0.0, 0.0)  # degrees
    hidden: bool = False

    @property
    def is_step(self) -> bool:
        return self.path.lower().endswith(_STEP_SUFFIXES)


def _xyz(node: list | None, what: str, factor: float = 1.0) -> tuple[float, float, float] | None:
    if node is None:
        return None
    xyz = sexpr.find(node, "xyz")
    if xyz is None or len(xyz) < 4:
        raise ValueError(f"({what} ...) without (xyz x y z)")
    values = tuple(sexpr.to_float(v) * factor for v in xyz[1:4])
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f"({what} ...) has a non-finite value")
    return values  # type: ignore[return-value]


def footprint_models(fp: FootprintDef) -> list[ModelRef]:
    """Every ``(model ...)`` of the library footprint, in file order. A malformed transform is a ``ValueError`` naming the footprint."""
    out: list[ModelRef] = []
    for node in sexpr.find_all(fp.node, "model"):
        if len(node) < 2 or isinstance(node[1], list):
            raise ValueError(f"footprint {fp.lib_id}: (model ...) without a path")
        try:
            offset = _xyz(sexpr.find(node, "offset"), "offset")
            if offset is None:
                offset = _xyz(sexpr.find(node, "at"), "at", 25.4)  # legacy: inches
            scale = _xyz(sexpr.find(node, "scale"), "scale")
            rotate = _xyz(sexpr.find(node, "rotate"), "rotate")
        except ValueError as exc:
            raise ValueError(f"footprint {fp.lib_id}: model {str(node[1])!r}: {exc}") from exc
        hidden = "hide" in sexpr.args(node) or sexpr.get(node, "hide") == "yes"
        out.append(ModelRef(str(node[1]), offset or (0.0, 0.0, 0.0), scale or (1.0, 1.0, 1.0), rotate or (0.0, 0.0, 0.0), hidden))
    return out


def find_3dmodel_dir(library: KicadLibrary | None = None) -> Path | None:
    """The KiCad 3D model library directory (module docstring), or ``None`` when none exists."""
    for var in _ENV_VARIABLES:
        value = os.environ.get(var)
        if value and Path(value).is_dir():
            return Path(value)
    roots = library.roots if library is not None else KicadLibrary().roots
    for root in roots:
        candidate = root / "3dmodels"
        if candidate.is_dir():
            return candidate
    return None


def resolve_model_path(path: str, model_dir: Path | None) -> tuple[Path | None, str]:
    """``(file, "")`` for a model reference that names an existing file, else ``(None, reason)`` (module docstring)."""
    m = _VAR_RE.match(path.strip())
    if m is not None:
        var = m.group(1) or m.group(2)
        if var != MODEL_DIR_VARIABLE:
            return None, f"경로 변수 ${{{var}}} 는 KiCad 10 3D 라이브러리({MODEL_DIR_VARIABLE})가 아님"
        if model_dir is None:
            return None, "KiCad 3D 모델 라이브러리를 찾지 못함 (KICAD10_3DMODEL_DIR)"
        base = model_dir.resolve()
        target = (base / m.group(3)).resolve()
        if base != target and base not in target.parents:
            return None, "모델 경로가 3D 라이브러리 밖을 가리킴"
        if not target.is_file():
            return None, "3D 라이브러리에 STEP 파일이 없음"
        return target, ""
    candidate = Path(path)
    if candidate.is_absolute():
        return (candidate, "") if candidate.is_file() else (None, "절대 경로의 모델 파일이 없음")
    return None, "경로 변수 없는 상대 경로 (기준 디렉터리를 알 수 없음)"


def _q(v: float) -> float:
    """Rounded to 1e-6 mm (KiCad's resolution), ``-0.0`` -> ``0.0``: the same inputs give the same numbers everywhere."""
    return round(v, 6) + 0.0


def _rot(axis: int, deg: float) -> tuple[tuple[float, float, float], ...]:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    if axis == 0:
        return ((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c))
    if axis == 1:
        return ((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c))
    return ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))


def _apply(m: tuple[tuple[float, float, float], ...], p: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        m[0][0] * p[0] + m[0][1] * p[1] + m[0][2] * p[2],
        m[1][0] * p[0] + m[1][1] * p[1] + m[1][2] * p[2],
        m[2][0] * p[0] + m[2][1] * p[1] + m[2][2] * p[2],
    )


def model_transform(ref: ModelRef, point: tuple[float, float, float]) -> tuple[float, float, float]:
    """A model point (mm, model frame) in the footprint's 3D frame: scale, rotate x / y / z by the negated angles, then offset."""
    p = (point[0] * ref.scale[0], point[1] * ref.scale[1], point[2] * ref.scale[2])
    rx, ry, rz = ref.rotate
    if rx:
        p = _apply(_rot(0, -rx), p)
    if ry:
        p = _apply(_rot(1, -ry), p)
    if rz:
        p = _apply(_rot(2, -rz), p)
    return (_q(p[0] + ref.offset[0]), _q(p[1] + ref.offset[1]), _q(p[2] + ref.offset[2]))


def transformed_box(ref: ModelRef, box: Box3) -> Box3:
    """The box around the eight transformed corners of ``box`` (an envelope of the rotated envelope)."""
    return Box3.around([model_transform(ref, corner) for corner in box.corners()])
