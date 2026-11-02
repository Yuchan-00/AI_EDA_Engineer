"""The driver model of a net class: the edge, the source resistance and the far-end load the SI checks use.

Invariant: a value is either the driver component's *grounded* datasheet
fact (``Component.electrical[key]`` with ``authoritative`` or
``user_requirement`` provenance - the datasheet-facts flow puts it there only
after the user confirmed the grounded row) or the class's own traced value
(a template's confirmed choice or the user's). A model's unconfirmed fact is
never read. Missing both is ``None`` with the name of what is missing, and
the check that needs it is NOT_VERIFIED naming it.

A driver's fact describes the edges *it* launches, so it replaces the
class's value on a net only where the driver has a pin that can drive
(:data:`DRIVING_TYPES`: output, bidirectional, tri-state, open collector /
emitter) - never on a net where its only pins are inputs, passive or power
pins (a microcontroller's ``~RESET`` input, its ``AREF``), or a net it is not
on. There the class's own value stands, and the source says why.
"""

from __future__ import annotations

from dataclasses import dataclass

from ai_eda.ir import CircuitIR, NetClass, Pin, PinElectricalType, Traced

#: class field -> the datasheet fact key of the driver component that replaces it, and the unit both carry
DRIVER_FACTS: dict[str, tuple[str, str]] = {"t_rise_s": ("t_rise", "s"), "r_drive_ohm": ("r_out", "ohm"), "c_load_f": ("c_in", "F")}
#: pin electrical types that can launch an edge on a net
DRIVING_TYPES = frozenset({
    PinElectricalType.OUTPUT, PinElectricalType.BIDIRECTIONAL, PinElectricalType.TRI_STATE, PinElectricalType.OPEN_COLLECTOR, PinElectricalType.OPEN_EMITTER,
})


@dataclass(frozen=True)
class DriverValue:
    """One driver-model number and where it came from (``source`` is an id the report / check prints)."""

    value: float
    source: str
    traced: Traced


def pins_on_net(ir: CircuitIR, ref: str, net: str) -> list[Pin]:
    """The pins of component ``ref`` that net ``net`` joins (the IR's own pin records, in net order)."""
    n = ir.net(net)
    comp = ir.component(ref)
    if n is None or comp is None:
        return []
    out: list[Pin] = []
    for p in n.pins:
        if p.component_ref == ref:
            pin = comp.pin(p.pin_number)
            if pin is not None and pin not in out:
                out.append(pin)
    return out


def drives(ir: CircuitIR, ref: str, net: str) -> bool:
    """Whether component ``ref`` has a pin on ``net`` that can drive it (:data:`DRIVING_TYPES`)."""
    return any(p.electrical_type in DRIVING_TYPES for p in pins_on_net(ir, ref, net))


def driver_value(ir: CircuitIR, cls: NetClass, field: str, net: str | None = None) -> tuple[DriverValue | None, str]:
    """``(value, "")`` for ``field`` of ``cls`` (a key of :data:`DRIVER_FACTS`), or ``(None, what is missing)``.

    With ``net`` the driver's grounded fact applies only when the driver can
    drive that net (module docstring); without it, the class-wide value.
    """
    fact_key, unit = DRIVER_FACTS[field]
    not_driving = ""
    if cls.driver is not None:
        comp = ir.component(cls.driver)
        fact = comp.electrical.get(fact_key) if comp is not None else None
        if fact is not None and fact.provenance.is_authoritative and fact.unit == unit and isinstance(fact.value, (int, float)) and not isinstance(fact.value, bool):
            if net is None or drives(ir, cls.driver, net):
                return DriverValue(float(fact.value), f"{cls.driver}.{fact_key} (grounded datasheet fact)", fact), ""
            not_driving = f"; {cls.driver} has no pin that drives {net}, so its {fact_key} does not apply"
    own: Traced | None = getattr(cls, field)
    if own is not None:
        return DriverValue(float(own.value), f"si.net_classes[{cls.name}].{field} ({own.provenance.kind.value}{not_driving})", own), ""
    missing = f"si.net_classes[{cls.name}].{field}"
    if cls.driver is not None:
        missing += f" or the grounded datasheet fact {cls.driver}.{fact_key}" + (f" of a pin that drives {net}" if net is not None else "")
    return None, missing


__all__ = ["DRIVER_FACTS", "DRIVING_TYPES", "DriverValue", "driver_value", "drives", "pins_on_net"]
