"""Built-in 3D preview of a placed board (deterministic, stdlib only; no status is ever computed here).

* :mod:`~ai_eda.tools.model3d.step_bbox` - the approximate envelope of a STEP
  model on disk (points + full circles, SI / conversion-based length units).
* :mod:`~ai_eda.tools.model3d.models` - the 3D library directory, a
  footprint's ``(model ...)`` references and KiCad's model transform.
* :mod:`~ai_eda.tools.model3d.scene` - the scene from the IR + the libraries:
  board slab, mask, copper (pads / tracks / vias, both sides), drills, the
  footprints' own silkscreen strokes, and one body box per component
  (``F.Fab`` outline x the STEP envelope's height; no STEP -> flat outline).
* :mod:`~ai_eda.tools.model3d.glb` - :func:`write_glb`, glTF 2.0 binary.
* :mod:`~ai_eda.tools.model3d.iso` - :func:`iso_svg`, painter's-algorithm SVG.

KiCad's own 3D output (real part shapes, only where ``kicad-cli`` runs) is
:meth:`ai_eda.tools.kicad.cli.KicadCli.export_step` / ``export_glb`` /
``render``.
"""

from ai_eda.tools.model3d.glb import write_glb
from ai_eda.tools.model3d.iso import VIEWS, iso_svg
from ai_eda.tools.model3d.models import ModelRef, find_3dmodel_dir, footprint_models, model_transform, resolve_model_path, transformed_box
from ai_eda.tools.model3d.scene import BODY_CAPTION, BodyInfo, Scene, SceneError, Solid, build_scene, scene_caption
from ai_eda.tools.model3d.step_bbox import Box3, StepEnvelope, read_step_envelope, step_bbox

__all__ = [
    "BODY_CAPTION",
    "BodyInfo",
    "Box3",
    "ModelRef",
    "Scene",
    "SceneError",
    "Solid",
    "StepEnvelope",
    "VIEWS",
    "build_scene",
    "find_3dmodel_dir",
    "footprint_models",
    "iso_svg",
    "model_transform",
    "read_step_envelope",
    "resolve_model_path",
    "scene_caption",
    "step_bbox",
    "transformed_box",
    "write_glb",
]
