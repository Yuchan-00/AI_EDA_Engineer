"""glTF 2.0 binary (``.glb``) of a :class:`~ai_eda.tools.model3d.scene.Scene` - stdlib only, byte-deterministic.

Invariant: the same scene gives the same bytes. Nothing time- or
machine-dependent is written (no timestamp, no path, no random id); the JSON
chunk is ``json.dumps(sort_keys=True)`` with fixed separators, every float in
the binary chunk is a little-endian float32, and the accessor ``min`` /
``max`` are the float32 values themselves.

Layout (glTF 2.0, the Khronos specification's GLB container):

* 12-byte header (``glTF``, version 2, total length), a ``JSON`` chunk padded
  with spaces and a ``BIN`` chunk padded with zeros, both to 4 bytes.
* One mesh / node per material that has geometry (materials in
  :data:`~ai_eda.tools.model3d.scene.MATERIALS` order); each node carries
  ``extras.group`` (``board`` / ``copper`` / ``silk`` / ``parts``) and
  ``extras.label`` (the Korean toggle label) so a viewer can switch layers.
* Each primitive: ``POSITION`` + ``NORMAL`` (VEC3 float32, one vertex per face
  corner so faces are flat-shaded) and ``indices`` (``UNSIGNED_SHORT`` when
  the mesh has fewer than 65536 vertices, else ``UNSIGNED_INT``), mode 4
  (triangles), counter-clockwise front faces.
* Frame: glTF is y-up and in metres: ``X = board x``, ``Y = z`` (up, out of
  the top side), ``Z = board y`` (board y is down on screen, so +Z points
  toward a viewer looking at the top from the front) - a proper rotation of
  KiCad's 3D frame, so windings are kept.
* ``scenes[0].extras``: project id, the scene version, the board thickness
  and whether it is an assumption, and the scene's notes (Korean; no paths).
"""

from __future__ import annotations

import json
import struct
import sys
from array import array

from ai_eda.tools.model3d.scene import GROUP_LABELS, MATERIALS, Scene, solid_faces, triangulate

__all__ = ["GLB_GENERATOR", "write_glb", "GLB_MAGIC", "CHUNK_JSON", "CHUNK_BIN"]

GLB_MAGIC = 0x46546C67  # "glTF"
CHUNK_JSON = 0x4E4F534A  # "JSON"
CHUNK_BIN = 0x004E4942  # "BIN\0"
GLB_GENERATOR = "ai_eda.tools.model3d glb 0.1"
_ARRAY_BUFFER = 34962
_ELEMENT_ARRAY_BUFFER = 34963
_FLOAT = 5126
_UNSIGNED_SHORT = 5123
_UNSIGNED_INT = 5125
_MM_TO_M = 0.001


def _f32(values: list[float]) -> array:
    data = array("f", values)
    if sys.byteorder != "little":
        data.byteswap()
    return data


def _pad(blob: bytes, fill: bytes) -> bytes:
    return blob + fill * ((4 - len(blob) % 4) % 4)


def _mesh_arrays(scene: Scene, material: str) -> tuple[list[float], list[float], list[int]]:
    """Positions (m, glTF frame), normals and triangle indices of every solid of ``material``, in scene order."""
    positions: list[float] = []
    normals: list[float] = []
    indices: list[int] = []
    for solid in scene.solids:
        if solid.material != material:
            continue
        cap = triangulate(solid.polygon)
        for face in solid_faces(solid):
            base = len(positions) // 3
            nx, ny, nz = face.normal
            for x, y, z in face.points:  # KiCad 3D frame (x, y up, z) -> glTF (x, z, -y), metres
                positions += [x * _MM_TO_M + 0.0, z * _MM_TO_M + 0.0, -y * _MM_TO_M + 0.0]
                normals += [nx, nz, -ny + 0.0]
            if face.which == "side":
                indices += [base, base + 1, base + 2, base, base + 2, base + 3]
            elif face.which == "top":
                for a, b, c in cap:
                    indices += [base + a, base + b, base + c]
            else:  # bottom: the points are the ring reversed, so ring index i is point n-1-i
                n = len(face.points)
                for a, b, c in cap:
                    indices += [base + n - 1 - a, base + n - 1 - c, base + n - 1 - b]
    return positions, normals, indices


def write_glb(scene: Scene) -> bytes:
    """The scene as a glTF 2.0 binary (module docstring); the same scene always gives the same bytes."""
    binary = bytearray()
    accessors: list[dict] = []
    views: list[dict] = []
    meshes: list[dict] = []
    nodes: list[dict] = []
    materials: list[dict] = []

    def view(blob: bytes, target: int) -> int:
        offset = len(binary)
        binary.extend(_pad(blob, b"\x00"))
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(blob), "target": target})
        return len(views) - 1

    for mat in MATERIALS:
        positions, normals, indices = _mesh_arrays(scene, mat.key)
        if not indices:
            continue
        pos = _f32(positions)
        stored = pos.tolist()  # the float32 values, read back: min / max must be what the buffer holds
        mins = [min(stored[i::3]) for i in range(3)]
        maxs = [max(stored[i::3]) for i in range(3)]
        count = len(positions) // 3
        accessors.append({"bufferView": view(pos.tobytes(), _ARRAY_BUFFER), "componentType": _FLOAT, "count": count, "type": "VEC3", "min": mins, "max": maxs})
        pos_acc = len(accessors) - 1
        accessors.append({"bufferView": view(_f32(normals).tobytes(), _ARRAY_BUFFER), "componentType": _FLOAT, "count": count, "type": "VEC3"})
        nor_acc = len(accessors) - 1
        wide = count > 0xFFFF
        idx = array("I" if wide else "H", indices)
        if idx.itemsize != (4 if wide else 2):  # pragma: no cover - platforms where I is not 32-bit
            idx = array("L" if wide else "H", indices)
        if sys.byteorder != "little":
            idx.byteswap()
        accessors.append({"bufferView": view(idx.tobytes(), _ELEMENT_ARRAY_BUFFER), "componentType": _UNSIGNED_INT if wide else _UNSIGNED_SHORT, "count": len(indices), "type": "SCALAR"})
        idx_acc = len(accessors) - 1
        r, g, b, a = mat.rgba
        material = {
            "name": mat.key,
            "pbrMetallicRoughness": {"baseColorFactor": [r, g, b, a], "metallicFactor": mat.metallic, "roughnessFactor": mat.roughness},
        }
        if a < 1.0:
            material["alphaMode"] = "BLEND"
        materials.append(material)
        meshes.append({"name": mat.key, "primitives": [{"attributes": {"NORMAL": nor_acc, "POSITION": pos_acc}, "indices": idx_acc, "material": len(materials) - 1, "mode": 4}]})
        nodes.append({"name": mat.key, "mesh": len(meshes) - 1, "extras": {"group": mat.group, "label": GROUP_LABELS[mat.group]}})

    doc: dict = {
        "asset": {"version": "2.0", "generator": GLB_GENERATOR},
        "scene": 0,
        "scenes": [
            {
                "name": scene.project_id,
                "nodes": list(range(len(nodes))),
                "extras": {
                    "project": scene.project_id,
                    "scene_version": scene.version,
                    "units": "m",
                    "frame": "X = board x, Y = up (top side), Z = board y",
                    "board_thickness_mm": scene.thickness_mm,
                    "board_thickness_assumed": not scene.thickness_grounded,
                    "notes": list(scene.notes),
                },
            }
        ],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": len(binary)}],
    }
    if not meshes:  # glTF allows a scene without meshes; keep the arrays out rather than empty
        for key in ("nodes", "meshes", "materials", "accessors", "bufferViews", "buffers"):
            doc.pop(key)
    json_chunk = _pad(json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii"), b" ")
    chunks = struct.pack("<II", len(json_chunk), CHUNK_JSON) + json_chunk
    if binary:
        chunks += struct.pack("<II", len(binary), CHUNK_BIN) + bytes(binary)
    return struct.pack("<III", GLB_MAGIC, 2, 12 + len(chunks)) + chunks
