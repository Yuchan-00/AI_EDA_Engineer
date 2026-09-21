"""Deterministic identifiers for generated artifacts.

KiCad files are full of UUIDs. If they were random, two compilations of the
same IR would differ byte-for-byte, artifact hashes would be meaningless and
"regenerate from IR" would look like a change. All generated UUIDs are
therefore uuid5 of a stable key derived from the project id and the entity
they identify, so schematic symbols and PCB footprints for the same component
share one identity and KiCad's schematic parity check can link them.
"""

from __future__ import annotations

import uuid

NAMESPACE = uuid.UUID("6f2a2c6e-2a3d-5b1e-9c0a-0000a1eda100")


def stable_uuid(*parts: object) -> str:
    """uuid5 over the given parts, joined with ``|``. Same parts → same UUID, always."""
    return str(uuid.uuid5(NAMESPACE, "|".join(str(p) for p in parts)))


def sheet_uuid(project_id: str) -> str:
    return stable_uuid(project_id, "sheet", "root")


def symbol_uuid(project_id: str, ref: str) -> str:
    """UUID of the schematic symbol instance; the PCB footprint ``path`` points at it."""
    return stable_uuid(project_id, "symbol", ref)


def pin_uuid(project_id: str, ref: str, pin_number: str) -> str:
    return stable_uuid(project_id, "pin", ref, pin_number)


def footprint_uuid(project_id: str, ref: str) -> str:
    return stable_uuid(project_id, "footprint", ref)


def pad_uuid(project_id: str, ref: str, pad_number: str) -> str:
    return stable_uuid(project_id, "pad", ref, pad_number)


def net_item_uuid(project_id: str, kind: str, *key: object) -> str:
    """For wires, labels, junctions, tracks, vias, zones: ``kind`` + a stable key."""
    return stable_uuid(project_id, kind, *key)
