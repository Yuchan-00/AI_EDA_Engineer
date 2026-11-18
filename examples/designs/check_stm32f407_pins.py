"""Compare the pin functions the STM32F407 example relies on with ST's own pin data.

    python check_stm32f407_pins.py <STM32F407V(E-G)Tx.xml> [stm32f407_display_devboard.json]

The XML is mcu/STM32F407V(E-G)Tx.xml of github.com/STMicroelectronics/STM32_open_pin_data (ST's machine-readable pin
data, the source of the CubeMX pin tables). Exit 0 when every function is listed for the pin the design uses, 1 otherwise.
This is a check against the manufacturer's pin data, not against the datasheet PDF.
"""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

NS = {"m": "http://dummy.com"}
#: MCU pin -> the signal the board uses it for, as ST's pin data spells it
WANTED = {
    "PD14": "FSMC_D0", "PD15": "FSMC_D1", "PD0": "FSMC_D2", "PD1": "FSMC_D3", "PE7": "FSMC_D4", "PE8": "FSMC_D5", "PE9": "FSMC_D6", "PE10": "FSMC_D7",
    "PE11": "FSMC_D8", "PE12": "FSMC_D9", "PE13": "FSMC_D10", "PE14": "FSMC_D11", "PE15": "FSMC_D12", "PD8": "FSMC_D13", "PD9": "FSMC_D14", "PD10": "FSMC_D15",
    "PD7": "FSMC_NE1", "PD11": "FSMC_A16", "PD5": "FSMC_NWE", "PD4": "FSMC_NOE",
    "PA4": "SPI1_NSS", "PA5": "SPI1_SCK", "PA6": "SPI1_MISO", "PA7": "SPI1_MOSI", "PB6": "I2C1_SCL", "PB7": "I2C1_SDA",
    "PA13": "SYS_JTMS-SWDIO", "PA14": "SYS_JTCK-SWCLK", "PB0": "TIM3_CH3", "PD13": "TIM4_CH2", "PH0": "RCC_OSC_IN", "PH1": "RCC_OSC_OUT",
}


def main() -> int:
    xml_path = sys.argv[1]
    design_path = sys.argv[2] if len(sys.argv) > 2 else str(Path(__file__).with_name("stm32f407_display_devboard.json"))
    st = {p.get("Name").split("-")[0]: (p.get("Position"), [s.get("Name") for s in p.findall("m:Signal", NS)]) for p in ET.parse(xml_path).getroot().findall("m:Pin", NS)}
    design = json.loads(Path(design_path).read_text(encoding="utf-8"))
    net_of = {pin.split(".")[1]: n["name"] for n in design["nets"] for pin in n["pins"] if pin.startswith("U1.")}
    bad = 0
    for pin, signal in WANTED.items():
        position, signals = st[pin]  # the exact pin name: PD1 is not PD10
        net = net_of.get(position)
        ok = signal in signals and net in (pin, "HSE_IN", "HSE_OUT")
        bad += not ok
        print(f"{'OK ' if ok else 'BAD'} {pin:<5} pin {position:>3} net {net:<8} {signal}")
    print(f"{len(WANTED) - bad} of {len(WANTED)} functions are listed by ST for the pin the design uses")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
