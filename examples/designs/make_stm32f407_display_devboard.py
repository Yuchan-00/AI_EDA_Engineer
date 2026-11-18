"""Write the design file (ai-eda-design/1) of an STM32F407VET6 display-driver development board.

    python make_design.py OUT.json        (run with the Python of the ai-eda install: it reads pin names from the KiCad library)

The board: STM32F407VET6 (LQFP-100) on 3.3 V from an AMS1117-3.3, 8 MHz crystal, reset button, BOOT0 / BOOT1 straps, SWD header,
power LED, every GPIO on a header - one header per side of the LQFP, in the chip's pin order, so the lines from a side to its header do
not cross - and three display connectors that can be used one at a time or together:
  * J_LCD  2x13: 16-bit 8080 parallel TFT on the FSMC bus (D0..D15, NE1 = CS, A16 = RS, NWE = WR, NOE = RD, reset, backlight)
  * J_TFT  1x09: SPI colour TFT (ILI9341 / ST7789 module pin order) on SPI1
  * J_OLED 1x04: I2C OLED (SSD1306 module pin order) on I2C1, with 4.7 kohm pull-ups
Every pin of every part is in a net (the pipeline refuses a pin in no net), and every R / C / L value is an E series value.
Sources of the part facts (2026-11-12):
  * Pin numbers and names: the KiCad library symbol, which agrees pin for pin with ST's own pin data.
  * Pin functions used here - FSMC_D0..D15, FSMC_NE1 / A16 / NWE / NOE, SPI1_NSS / SCK / MISO / MOSI, I2C1_SCL / SDA, SWDIO / SWCLK,
    RCC_OSC_IN / OUT, TIM3_CH3 (PB0), TIM4_CH2 (PD13): each is listed for the pin used in ST's pin data,
    github.com/STMicroelectronics/STM32_open_pin_data, mcu/STM32F407V(E-G)Tx.xml (commit 7d1f1514ed, file sha256 3771e470e578...).
    `check_stm32f407_pins.py` repeats that comparison.
  * NOT checked against a datasheet (st.com could not be reached from the machine this was written on): PB2 as BOOT1, the 2.2 uF VCAP
    capacitors, 100 nF per VDD pin and the VDDA filter, the crystal's load capacitance (22 pF is a placeholder for an 18..20 pF crystal),
    the AMS1117-3.3 pinout (taken from the KiCad symbol) and its capacitors, the LED's 2.0 V forward voltage. Check them before building a board.
"""

from __future__ import annotations

import json
import math
import re
import sys

from ai_eda.tools.kicad.library import KicadLibrary, LibraryRef

LIB = KicadLibrary()
V33 = ["req.output_voltage"]
V5 = ["req.input_voltage"]


def pins_of(symbol: str) -> dict[str, list[str]]:
    """Pin name -> pin numbers of a library symbol (a name can sit on several pins: VDD, VSS)."""
    library, name = symbol.split(":")
    out: dict[str, list[str]] = {}
    for p in LIB.load_symbol(LibraryRef(library=library, name=name)).pins:
        out.setdefault(p.name, []).append(p.number)
    return out


MCU_SYMBOL = "MCU_ST_STM32F4:STM32F407VETx"
MCU = pins_of(MCU_SYMBOL)
LDO = pins_of("Regulator_Linear:AMS1117-3.3")
LED = pins_of("Device:LED")

components: list[dict] = []
nets: dict[str, dict] = {}
pin_associations: list[dict] = []


def associate(terminal: str, ic_pin: str, role: str) -> None:
    pin_associations.append({"terminal": terminal, "ic_pin": ic_pin, "role": role,
                             "rationale": "Declared layout intent using the existing wired pins; not a verified manufacturer placement rule"})


def net(name: str, *pins: str, kind: str = "signal", serves: list[str] | None = None) -> None:
    entry = nets.setdefault(name, {"name": name, "kind": kind, "pins": []})
    entry["pins"] += [p for p in pins if p not in entry["pins"]]
    if serves:
        entry["serves_requirements"] = sorted(set(entry.get("serves_requirements", [])) | set(serves))


def part(ref: str, value: str, description: str, symbol: str, footprint: str, spice: dict, serves: list[str] | None = None, electrical: dict | None = None) -> None:
    components.append({"ref": ref, "value": value, "description": description, "symbol": symbol, "footprint": footprint,
                       **({"electrical": electrical} if electrical else {}), **({"serves_requirements": serves} if serves else {}), "spice": spice})


def excluded(reason: str) -> dict:
    return {"exclude": True, "exclude_reason": reason}


def resistor(ref: str, ohm: float, text: str, description: str, a: str, b: str, serves: list[str]) -> None:
    part(ref, text, description, "Device:R", "Resistor_SMD:R_0603_1608Metric", {"device": "R", "value": {"value": ohm, "unit": "ohm"}}, serves)
    net(a, f"{ref}.1")
    net(b, f"{ref}.2")


def capacitor(ref: str, farad: float, text: str, description: str, a: str, b: str, serves: list[str] | None, footprint: str = "Capacitor_SMD:C_0603_1608Metric",
              spice: dict | None = None) -> None:
    part(ref, text, description, "Device:C", footprint, spice or {"device": "C", "value": {"value": farad, "unit": "F"}}, serves)
    net(a, f"{ref}.1")
    net(b, f"{ref}.2")


def header(ref: str, pins: int, description: str, rows: int = 1) -> None:
    if rows == 1:
        symbol, footprint = f"Connector_Generic:Conn_01x{pins:02d}", f"Connector_PinHeader_2.54mm:PinHeader_1x{pins:02d}_P2.54mm_Vertical"
    else:
        symbol, footprint = f"Connector_Generic:Conn_02x{pins // 2:02d}_Odd_Even", f"Connector_PinHeader_2.54mm:PinHeader_2x{pins // 2:02d}_P2.54mm_Vertical"
    part(ref, symbol.split(":")[1], description, symbol, footprint, excluded("connector, no electrical model"))


# ---- power: 5 V in, AMS1117-3.3, the 3.3 V rail, the filtered analog supply
net("+5V", kind="power", serves=V5)
net("+3V3", kind="power", serves=V33)
net("VDDA", kind="power", serves=V33)
net("GND", kind="ground")
header("J1", 2, "5 V input (1 = +5V, 2 = GND)")
net("+5V", "J1.1")
net("GND", "J1.2")
part("U2", "AMS1117-3.3", "3.3 V linear regulator", "Regulator_Linear:AMS1117-3.3", "Package_TO_SOT_SMD:SOT-223-3_TabPin2",
     excluded("no regulator model: the +3V3 rail is the ideal stimulus V3V3 in the simulation"))
net("+5V", f"U2.{LDO['VI'][0]}")
net("+3V3", f"U2.{LDO['VO'][0]}")
net("GND", f"U2.{LDO['GND'][0]}")
capacitor("C1", 10e-6, "10u", "regulator input capacitor", "+5V", "GND", V5, "Capacitor_SMD:C_0805_2012Metric")
capacitor("C2", 10e-6, "10u", "regulator output capacitor", "+3V3", "GND", V33, "Capacitor_SMD:C_0805_2012Metric")
part("L1", "10u", "analog supply filter inductor (+3V3 to VDDA)", "Device:L", "Inductor_SMD:L_0805_2012Metric", {"device": "L", "value": {"value": 10e-6, "unit": "H"}}, V33)
net("+3V3", "L1.1")
net("VDDA", "L1.2")
capacitor("C3", 1e-6, "1u", "VDDA bulk capacitor", "VDDA", "GND", V33)
capacitor("C4", 100e-9, "100n", "VDDA / VREF+ decoupling capacitor", "VDDA", "GND", V33)
associate("C1.1", f"U2.{LDO['VI'][0]}", "input decoupling")
associate("C2.1", f"U2.{LDO['VO'][0]}", "output decoupling")
associate("C3.1", f"U1.{MCU['VDDA'][0]}", "analog supply decoupling")
associate("C4.1", f"U1.{MCU['VREF+'][0]}", "reference supply decoupling")

# ---- MCU: supply pins, one 100 nF per VDD pin, the two VCAP capacitors
part("U1", "STM32F407VET6", "microcontroller, LQFP-100", MCU_SYMBOL, "Package_QFP:LQFP-100_14x14mm_P0.5mm",
     excluded("no MCU model: the simulation checks the passive networks around it"))
for number in MCU["VDD"]:
    net("+3V3", f"U1.{number}")
for name in ("VSS", "VSSA"):
    for number in MCU[name]:
        net("GND", f"U1.{number}")
net("+3V3", f"U1.{MCU['VBAT'][0]}")
net("VDDA", f"U1.{MCU['VDDA'][0]}", f"U1.{MCU['VREF+'][0]}")
VDD_CAPS = []
for i, number in enumerate(MCU["VDD"]):
    capacitor(f"C{5 + i}", 100e-9, "100n", f"VDD decoupling capacitor {i + 1} of {len(MCU['VDD'])}", "+3V3", "GND", V33)
    VDD_CAPS.append(f"C{5 + i}")
    associate(f"C{5 + i}.1", f"U1.{number}", "VDD decoupling")
next_c = 5 + len(MCU["VDD"])
CORE_CAPS = []
for i, name in enumerate(("VCAP_1", "VCAP_2")):
    net(name, f"U1.{MCU[name][0]}")
    capacitor(f"C{next_c + i}", 2.2e-6, "2.2u", f"{name} core regulator capacitor (value not checked against the datasheet)", name, "GND", None,
              spice=excluded("its node floats in the simulation once the MCU is excluded"))
    CORE_CAPS.append(f"C{next_c + i}")
    associate(f"C{next_c + i}.1", f"U1.{MCU[name][0]}", "core regulator capacitor")
next_c += 2

# ---- clock: 8 MHz crystal on PH0 / PH1 with two load capacitors
part("Y1", "8MHz", "HSE crystal", "Device:Crystal", "Crystal:Crystal_HC49-4H_Vertical", excluded("no crystal model"))
net("HSE_IN", f"U1.{MCU['PH0'][0]}", "Y1.1")
net("HSE_OUT", f"U1.{MCU['PH1'][0]}", "Y1.2")
associate("Y1.1", f"U1.{MCU['PH0'][0]}", "crystal input")
associate("Y1.2", f"U1.{MCU['PH1'][0]}", "crystal output")
CLOCK_CAPS = []
for i, node in enumerate(("HSE_IN", "HSE_OUT")):
    capacitor(f"C{next_c + i}", 22e-12, "22p", f"crystal load capacitor on {node}", node, "GND", None,
              spice=excluded("its node floats in the simulation once the crystal and the MCU are excluded"))
    CLOCK_CAPS.append(f"C{next_c + i}")
    associate(f"C{next_c + i}.1", f"U1.{MCU[('PH0', 'PH1')[i]][0]}", "crystal load capacitor")
next_c += 2

# ---- reset: pull-up, delay capacitor (starts discharged in the transient), button; boot straps
resistor("R1", 10e3, "10k", "NRST pull-up", "+3V3", "NRST", V33)
capacitor(f"C{next_c}", 100e-9, "100n", "NRST delay capacitor", "NRST", "GND", V33,
          spice={"device": "C", "value": {"value": 100e-9, "unit": "F"}, "params": {"ic": {"value": 0.0, "unit": "V"}}})
C_RESET = f"C{next_c}"
part("SW1", "SW_Push", "reset button (NRST to GND)", "Switch:SW_Push", "Button_Switch_THT:SW_PUSH_6mm", excluded("open switch, no model"))
net("NRST", f"U1.{MCU['NRST'][0]}", "SW1.1")
net("GND", "SW1.2")
resistor("R2", 10e3, "10k", "BOOT0 pull-down (run from flash)", "BOOT0", "GND", V33)
net("BOOT0", f"U1.{MCU['BOOT0'][0]}")
header("J2", 2, "BOOT0 jumper (1 = BOOT0, 2 = +3V3): fit to start the system bootloader")
net("BOOT0", "J2.1")
net("+3V3", "J2.2")
resistor("R3", 10e3, "10k", "PB2 pull-down (PB2 as BOOT1: not checked against the datasheet)", "PB2", "GND", V33)
for terminal in ("R1.2", f"{C_RESET}.1", "SW1.1"):
    associate(terminal, f"U1.{MCU['NRST'][0]}", "reset")
for terminal in ("R2.1", "J2.1"):
    associate(terminal, f"U1.{MCU['BOOT0'][0]}", "boot strap")
associate("R3.1", f"U1.{MCU['PB2'][0]}", "PB2 strap (BOOT1 function unverified)")

# ---- power LED: the LED is an ideal 2.0 V source in the simulation, so the resistor's current is checked
resistor("R4", 680.0, "680", "power LED series resistor", "+3V3", "LED_A", V33)
part("D1", "LED", "power indicator LED", "Device:LED", "LED_SMD:LED_0805_2012Metric",
     excluded("no LED model: an ideal 2.0 V source (VLED) stands in for its forward voltage"), V33)
net("LED_A", f"D1.{LED['A'][0]}")
net("GND", f"D1.{LED['K'][0]}")

# ---- every GPIO on a header, one per side of the LQFP-100 (the footprint's pad positions say which side a pin is on), each header in the
#      chip's pin order: J3 west (pins 1-25), J4 south (26-50), J5 east (51-75), J6 north (76-100). Measured 2026-11-12: with one header
#      per port (PA0..PA15 on J3 ...) every port left the chip on several sides, nearly every line crossed the others and the router's
#      negotiation did not converge on the 116 mm board.
MCU_FP = LIB.load_footprint(LibraryRef(library="Package_QFP", name="LQFP-100_14x14mm_P0.5mm"))
NAME_OF = {number: name for name, numbers in MCU.items() for number in numbers}
SIDE_GPIO: dict[str, list[str]] = {}
for pad in sorted((pad for pad in MCU_FP.pads if pad.number), key=lambda pad: int(pad.number)):
    side = ("west" if pad.x < 0 else "east") if abs(pad.x) > abs(pad.y) else ("north" if pad.y < 0 else "south")  # footprint frame, y down
    if re.fullmatch(r"P[A-E][0-9]+", NAME_OF[pad.number]):
        SIDE_GPIO.setdefault(side, []).append(NAME_OF[pad.number])
assert sum(len(names) for names in SIDE_GPIO.values()) == 80  # PA..PE, 16 each
for ref, side, first in (("J3", "west", 1), ("J4", "south", 26), ("J5", "east", 51), ("J6", "north", 76)):
    names = SIDE_GPIO[side]
    header(ref, len(names), f"{side} side GPIO header, LQFP pins {first}-{first + 24} in pin order: " + ", ".join(f"{k} = {n}" for k, n in enumerate(names, 1)))
    for k, name in enumerate(names, 1):
        net(name, f"U1.{MCU[name][0]}", f"{ref}.{k}")

# ---- SWD header
header("J8", 5, "SWD header (1 = +3V3, 2 = SWDIO, 3 = SWCLK, 4 = NRST, 5 = GND)")
for pin, name in enumerate(("+3V3", "PA13", "PA14", "NRST", "GND"), 1):
    net(name, f"J8.{pin}")

# ---- display 1: 16-bit 8080 parallel TFT on the FSMC bus
FSMC_DATA = ["PD14", "PD15", "PD0", "PD1", "PE7", "PE8", "PE9", "PE10", "PE11", "PE12", "PE13", "PE14", "PE15", "PD8", "PD9", "PD10"]  # FSMC_D0..D15
LCD = ["+3V3", "GND", *FSMC_DATA, "PD7", "PD11", "PD5", "PD4", "PD12", "PD13", "+5V", "GND"]  # ..., NE1 (CS), A16 (RS), NWE (WR), NOE (RD), reset, backlight, 5 V, GND
header("J9", 26, "parallel TFT, 16-bit 8080 on FSMC (1 +3V3, 2 GND, 3..18 D0..D15, 19 CS, 20 RS, 21 WR, 22 RD, 23 RST, 24 BL, 25 +5V, 26 GND)", rows=2)
for pin, name in enumerate(LCD, 1):
    net(name, f"J9.{pin}")

# ---- display 2: SPI colour TFT on SPI1 (module order VCC, GND, CS, RESET, DC, MOSI, SCK, LED, MISO)
header("J10", 9, "SPI TFT (1 +3V3, 2 GND, 3 CS = PA4, 4 RST = PC5, 5 DC = PC4, 6 MOSI = PA7, 7 SCK = PA5, 8 BL = PB0, 9 MISO = PA6)")
for pin, name in enumerate(("+3V3", "GND", "PA4", "PC5", "PC4", "PA7", "PA5", "PB0", "PA6"), 1):
    net(name, f"J10.{pin}")

# ---- display 3: I2C OLED on I2C1 with pull-ups (module order GND, VCC, SCL, SDA)
header("J11", 4, "I2C OLED (1 GND, 2 +3V3, 3 SCL = PB6, 4 SDA = PB7)")
for pin, name in enumerate(("GND", "+3V3", "PB6", "PB7"), 1):
    net(name, f"J11.{pin}")
resistor("R5", 4.7e3, "4.7k", "I2C SCL pull-up", "+3V3", "PB6", V33)
resistor("R6", 4.7e3, "4.7k", "I2C SDA pull-up", "+3V3", "PB7", V33)

# ---- simulation: the two rails are ideal sources; LED current, the reset delay, the filtered analog rail
TAU = 10e3 * 100e-9
simulation = {
    "stimuli": [
        {"id": "V5", "source": "voltage", "net": "+5V", "reference_net": "GND", "kind": "dc", "value": {"value": 5.0, "unit": "V"}, "serves_requirements": V5},
        {"id": "V3V3", "source": "voltage", "net": "+3V3", "reference_net": "GND", "kind": "dc", "value": {"value": 3.3, "unit": "V"}, "serves_requirements": V33},
        {"id": "VLED", "source": "voltage", "net": "LED_A", "reference_net": "GND", "kind": "dc", "value": {"value": 2.0, "unit": "V"}},
    ],
    "analyses": [
        {"id": "op", "kind": "op"},
        {"id": "tran", "kind": "tran", "params": {"step": {"value": TAU / 100, "unit": "s"}, "stop": {"value": 5 * TAU, "unit": "s"}, "uic": True}},
    ],
    "expectations": [
        {"id": "i_led", "analysis_id": "op", "vector": "i(VLED)", "nominal": {"value": (3.3 - 2.0) / 680.0, "unit": "A"}, "tol_rel": 0.02},
        {"id": "v_vdda", "analysis_id": "op", "vector": "v(VDDA)", "nominal": {"value": 3.3, "unit": "V"}, "tol_abs": {"value": 0.001, "unit": "V"}, "requirement_id": "req.output_voltage"},
        {"id": "v_nrst_tau", "analysis_id": "tran", "vector": "v(NRST)", "reduce": "at", "at": {"value": TAU, "unit": "s"},
         "nominal": {"value": 3.3 * (1 - math.exp(-1)), "unit": "V"}, "tol_rel": 0.02},
    ],
}

blocks = [
    {"id": "power", "function": "power input and regulator", "domain": "power", "component_refs": ["J1", "U2", "C1", "C2"], "input_nets": ["+5V"], "output_nets": ["+3V3"]},
    {"id": "mcu", "function": "MCU and supply support", "domain": "digital", "component_refs": ["U1", "L1", "C3", "C4", *VDD_CAPS, *CORE_CAPS], "input_nets": ["+3V3", "HSE_IN", "NRST", "BOOT0"]},
    {"id": "clock", "function": "HSE crystal and load capacitors", "component_refs": ["Y1", *CLOCK_CAPS], "input_nets": ["HSE_OUT"], "output_nets": ["HSE_IN"]},
    {"id": "reset", "function": "reset pull-up, delay and button", "component_refs": ["R1", C_RESET, "SW1"], "input_nets": ["+3V3"], "output_nets": ["NRST"]},
    {"id": "boot", "function": "boot and PB2 straps", "component_refs": ["R2", "R3", "J2"], "output_nets": ["BOOT0", "PB2"]},
    {"id": "indicator", "function": "power indicator", "component_refs": ["R4", "D1"], "input_nets": ["+3V3"]},
    *({"id": f"gpio_{side}", "function": f"{side} GPIO breakout", "component_refs": [ref]} for ref, side in (("J3", "west"), ("J4", "south"), ("J5", "east"), ("J6", "north"))),
    {"id": "debug", "function": "SWD interface", "component_refs": ["J8"], "output_nets": ["PA13", "PA14", "NRST"]},
    {"id": "parallel_display", "function": "parallel TFT interface", "component_refs": ["J9"], "input_nets": LCD},
    {"id": "spi_display", "function": "SPI TFT interface", "component_refs": ["J10"], "input_nets": ["PA4", "PC5", "PC4", "PA7", "PA5", "PB0"], "output_nets": ["PA6"]},
    {"id": "i2c_display", "function": "I2C OLED and pull-ups", "component_refs": ["J11", "R5", "R6"], "input_nets": ["PB6", "PB7"]},
]

design = {
    "format": "ai-eda-design/1",
    "title": "STM32F407VET6 display driver development board (FSMC parallel TFT, SPI TFT, I2C OLED)",
    "topology": {
        "name": "STM32F407 display development board", "domains": ["digital", "power"],
        "rationale": "MCU on a 3.3 V linear rail with an 8 MHz crystal, reset and boot straps, SWD, every GPIO on a header per chip side, and three display connectors (FSMC 16-bit parallel, SPI1, I2C1)",
        "blocks": blocks,
        "pin_associations": pin_associations,
    },
    "components": components,
    "nets": list(nets.values()),
    "simulation": simulation,
    # the board's SI net classes, validation only: the si.* / spice.si / spice.xtalk checks judge the routed copper, the routing is unchanged.
    # No value is stated: the driver edge, source resistance and load stay the template machinery's labelled choices (the STM32's I/O
    # rise times and levels are not read from its datasheet here, so the NM / 3 crosstalk budget stays NOT_VERIFIED).
    "si": {},
}

# ---- self-checks before writing: every MCU pin in exactly one net, no pin in two nets, the reset capacitor is the one with the initial condition
seen: dict[str, str] = {}
for entry in design["nets"]:
    for pin in entry["pins"]:
        assert pin not in seen, f"{pin} is in {seen[pin]} and {entry['name']}"
        seen[pin] = entry["name"]
mcu_numbers = {n for numbers in MCU.values() for n in numbers}
missing = sorted(mcu_numbers - {p.split(".")[1] for p in seen if p.startswith("U1.")}, key=int)
assert not missing, f"MCU pins in no net: {missing}"
assert len(mcu_numbers) == 100 and C_RESET in {c["ref"] for c in components}
with open(sys.argv[1], "w", encoding="utf-8", newline="\n") as f:
    json.dump(design, f, indent=1, ensure_ascii=False)
print(f"{sys.argv[1]}: {len(components)} components, {len(nets)} nets, {len(seen)} pins")
