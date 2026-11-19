"""Deterministic identifiers for generated artifacts.

KiCad files are full of UUIDs. If they were random, two compilations of the
same IR would differ byte-for-byte, artifact hashes would be meaningless and
"regenerate from IR" would look like a change. All generated UUIDs are
therefore uuid5 of a stable key derived from the project id and the entity
they identify, so schematic symbols and PCB footprints for the same component
share one identity and KiCad's schematic parity check can link them.

Multi-unit symbols (a dual op-amp drawn as units A, B and C): every unit
instance is a schematic symbol with its own UUID. The lowest unit keeps
:func:`symbol_uuid` - the UUID the PCB footprint ``path`` points at, exactly
as for a single-unit part - and every other unit gets :func:`unit_uuid`.
Measured with kicad-cli 10.0.6 on an MCP6002-xSN board
(``tests/test_multi_unit.py``): the exported netlist gives the part one
component with every unit's pins and lists the unit UUIDs (``tstamps``) in an
order that is neither unit order nor UUID order (project ``dual_d``: units
3, 2, 1, whose UUIDs descend as units 2, 3, 1); ERC is clean and DRC
``--schematic-parity`` reports nothing with the footprint path on unit 1 while
unit 3 is listed first - and also with a path no unit carries, so parity
matches footprints by reference and is no evidence of the path. The path
stays on a unit that exists (the lowest), so the board keeps a link KiCad can
follow; KiCad's interactive board update was not part of the measurement.
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


def unit_uuid(project_id: str, ref: str, unit: int) -> str:
    """UUID of unit ``unit`` of a multi-unit symbol, for every unit but the lowest (which keeps :func:`symbol_uuid`)."""
    return stable_uuid(project_id, "symbol", ref, "unit", unit)


def pin_uuid(project_id: str, ref: str, pin_number: str) -> str:
    return stable_uuid(project_id, "pin", ref, pin_number)


def footprint_uuid(project_id: str, ref: str) -> str:
    return stable_uuid(project_id, "footprint", ref)


def pad_uuid(project_id: str, ref: str, pad_number: str) -> str:
    return stable_uuid(project_id, "pad", ref, pad_number)


def net_item_uuid(project_id: str, kind: str, *key: object) -> str:
    """For wires, labels, junctions, tracks, vias, zones: ``kind`` + a stable key."""
    return stable_uuid(project_id, kind, *key)
