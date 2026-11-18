"""Small frozen geometry inputs for layout metric regression tests."""

import gzip
import json
from dataclasses import asdict
from pathlib import Path

from ai_eda.ir import CircuitIR
from ai_eda.tools.kicad.library import BBox, FootprintDef, Pad

DATA = Path(__file__).parent / "data" / "layout_baselines"


class FrozenFootprints:
    def __init__(self, rows):
        self.rows = rows

    def load_footprint(self, ref):
        row = self.rows[f"{ref.library}:{ref.name}"]
        return FootprintDef(lib_id=row["lib_id"], name=ref.name, node=[], attr=row["attr"],
                            pads=[Pad(**p) for p in row["pads"]], courtyard=BBox(**row["courtyard"]) if row["courtyard"] else None)


def freeze(ir, library):
    """Only fields the metric reader needs; no validation logs, paths or artifact payloads."""
    provenance = {"kind": "derived", "tool": "layout.baseline", "created_at": "1970-01-01T00:00:00Z"}
    components, footprints = [], {}
    for c in ir.components:
        ref = c.footprint.model_dump(include={"library", "name"})
        components.append({"ref": c.ref, "value": c.value, "footprint": ref, "provenance": provenance})
        fp = library.load_footprint(c.footprint)
        assert not any(p.primitives or p.unread_primitives for p in fp.pads), "extend the frozen format for custom pads"
        footprints[fp.lib_id] = {"lib_id": fp.lib_id, "attr": fp.attr, "pads": [asdict(p) for p in fp.pads],
                                 "courtyard": asdict(fp.courtyard) if fp.courtyard else None}
    pcb = ir.pcb.model_dump(mode="json", include={"layers", "outline", "placements", "tracks", "vias", "zones"})
    for field in ("placements", "tracks", "vias", "zones"):
        for row in pcb[field]:
            row["provenance"] = provenance
    # Keep plane assignments without carrying the material stack's unrelated facts.
    if ir.pcb.stackup:
        for c in ir.pcb.stackup.copper:
            if c.plane_net and not any(z["net"] == c.plane_net.value for z in pcb["zones"]):
                pcb["zones"].append({"net": c.plane_net.value, "layer": c.name, "polygon": [], "provenance": provenance})
    return {"ir": {"schema_version": ir.schema_version, "project": {"id": ir.project.id, "name": ir.project.name},
                   "components": components, "nets": [{"name": n.name, "kind": n.kind, "pins": [p.model_dump() for p in n.pins], "provenance": provenance} for n in ir.nets], "pcb": pcb},
            "footprints": footprints}


def load_geometry(name):
    data = json.loads(gzip.decompress((DATA / f"{name}.json.gz").read_bytes()))
    return CircuitIR.model_validate(data["ir"]), FrozenFootprints(data["footprints"])
