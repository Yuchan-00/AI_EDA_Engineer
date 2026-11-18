"""The example design files under examples/designs: what a hand-written (or model-written) circuit must hold before the pipeline sees it.

No KiCad library is needed here: the file is checked against the design-file models and its own invariants. Building it
(symbols, footprints, ERC, PCB) needs the KiCad 10 libraries and is a pipeline run, recorded in the README change log.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_eda.design.design_file import DesignFile, _nonstandard_value

EXAMPLES = Path(__file__).parent.parent / "examples" / "designs"
STM32 = EXAMPLES / "stm32f407_display_devboard.json"


def _load(path: Path) -> DesignFile:
    return DesignFile.model_validate(json.loads(path.read_text(encoding="utf-8")))


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*.json")), ids=lambda p: p.name)
def test_every_example_is_a_design_file_with_standard_values_and_no_pin_in_two_nets(path: Path):
    design = _load(path)
    assert [p.ref for p in design.components if _nonstandard_value(p) is not None] == []  # the rule the loader enforces: every R / C / L value is sold
    seen: dict[str, str] = {}
    for net in design.nets:
        for pin in net.pins:
            assert pin not in seen, f"{pin} is in {seen[pin]} and {net.name}"
            seen[pin] = net.name
    refs = {p.ref for p in design.components}
    assert {pin.split(".")[0] for pin in seen} <= refs  # a net names only parts of the file
    # a part the simulation sees names the requirement it serves; an excluded one says why it is excluded
    for p in design.components:
        assert p.serves_requirements or (p.spice.exclude and p.spice.exclude_reason.strip()), p.ref


def test_the_stm32f407_board_wires_all_hundred_pins_and_the_three_display_connectors():
    design = _load(STM32)
    assert design.si is not None and design.si.model_dump(exclude_unset=True) == {}
    by_net = {n.name: n.pins for n in design.nets}
    mcu = {pin.split(".")[1]: name for name, pins in by_net.items() for pin in pins if pin.startswith("U1.")}
    assert sorted(mcu, key=int) == [str(n) for n in range(1, 101)]  # LQFP-100: the pipeline refuses a pin that is in no net
    # the FSMC data bus D0..D15 on the parallel connector J9 pins 3..18, in order, and its four control lines
    fsmc = ["PD14", "PD15", "PD0", "PD1", "PE7", "PE8", "PE9", "PE10", "PE11", "PE12", "PE13", "PE14", "PE15", "PD8", "PD9", "PD10"]
    for offset, name in enumerate(fsmc):
        assert f"J9.{3 + offset}" in by_net[name], name
    assert [next(n for n, pins in by_net.items() if f"J9.{pin}" in pins) for pin in (19, 20, 21, 22)] == ["PD7", "PD11", "PD5", "PD4"]  # NE1, A16, NWE, NOE
    # SPI1 on the SPI TFT header, I2C1 with a pull-up each on the OLED header
    assert [next(n for n, pins in by_net.items() if f"J10.{pin}" in pins) for pin in (3, 6, 7, 9)] == ["PA4", "PA7", "PA5", "PA6"]  # CS, MOSI, SCK, MISO
    for name, resistor in (("PB6", "R5"), ("PB7", "R6")):
        assert any(pin.startswith("J11.") for pin in by_net[name]) and f"{resistor}.2" in by_net[name] and f"{resistor}.1" in by_net["+3V3"]
    # supply: six VDD pins, one 100 nF each; the two VCAP pins have their own capacitor
    assert sum(1 for pin in by_net["+3V3"] if pin.startswith("U1.")) == 7  # 6 x VDD + VBAT
    values = {p.ref: p.value for p in design.components}
    assert sum(1 for v in values.values() if v == "100n") >= 6 and [values[r] for r in ("R5", "R6")] == ["4.7k", "4.7k"]
    assert {"VCAP_1", "VCAP_2"} <= set(by_net) and all(len(by_net[n]) == 2 for n in ("VCAP_1", "VCAP_2"))
    # every GPIO on the header of its chip side, in the chip's pin order: no two lines from one side cross on the way to its header
    for ref, first in (("J3", 1), ("J4", 26), ("J5", 51), ("J6", 76)):
        order = sorted((int(p.split(".")[1]), n) for n, pins in by_net.items() for p in pins if p.startswith(f"{ref}."))
        mcu = [int(next(p for p in by_net[n] if p.startswith("U1.")).split(".")[1]) for _k, n in order]
        assert [k for k, _n in order] == list(range(1, len(order) + 1)), ref
        assert mcu == sorted(mcu) and first <= mcu[0] and mcu[-1] <= first + 24, (ref, mcu)
    assert sum(1 for c in design.components if c.ref in ("J3", "J4", "J5", "J6")) == 4 and all(c.ref != "J7" for c in design.components)
    assert len(design.components) == 37 and len(design.nets) == 91


def test_stm32_loads_with_si_and_has_identical_router_rules_without_it(tmp_path: Path):
    from ai_eda.design.design_file import load_design_file
    from ai_eda.design.stackup import board_layers, escalation_stackup
    from ai_eda.ir import PCBDesign
    from ai_eda.tools.kicad.library import KicadLibrary
    from ai_eda.tools.routing.maze import RoutingParams
    from ai_eda.tools.si.rules import net_rules
    from tests.test_design_file_si import _apply

    lib = KicadLibrary()
    if lib.symbol_file("MCU_ST_STM32F4") is None:
        pytest.skip("KiCad STM32F4 library not found")
    with_si = _apply(load_design_file(STM32, lib))
    data = json.loads(STM32.read_text(encoding="utf-8"))
    data.pop("si")
    path = tmp_path / "without_si.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    without_si = _apply(load_design_file(path, lib))
    stack = escalation_stackup(6, ground_net="GND", power_net="+3V3")
    for ir in (with_si, without_si):
        ir.pcb = PCBDesign(stackup=stack, layers=board_layers(stack))
    params = RoutingParams.for_board(with_si, lib)
    a, b = net_rules(with_si, params), net_rules(without_si, params)
    assert len(with_si.components) == 37 and len(with_si.nets) == 91
    assert with_si.si.default_class().promote_to is None
    assert all(with_si.si.class_of(n.name).name == "DEFAULT" for n in with_si.nets)
    assert a.rules == b.rules == {} and a.signature() == b.signature() == ()
