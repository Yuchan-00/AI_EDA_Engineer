"""The schematic layout corpus: every design the textbook layout gates are measured on.

The seven circuit templates (through :func:`tests.textbook_cases.circuit`, the astable among them), the TL081 inverting
amplifier fixture, the 35-part analog piano (``tests/data/analog_piano.design.json``), the ATmega128 development board
template (9 V in, 16 MHz clock) and the STM32F407 display board (``examples/designs``). :func:`corpus_ir` builds a fresh IR
for one name; nothing here compiles a schematic.
"""

from __future__ import annotations

from pathlib import Path

from ai_eda.agents import AgentContext
from ai_eda.design.design_file import load_design_file
from ai_eda.ir import BoardOutline, CircuitIR, PCBDesign, Placement, ProjectMeta
from ai_eda.tools.kicad import KicadLibrary
from ai_eda.workflow import Orchestrator, Stage
from tests.textbook_cases import CASES, amplifier, circuit

ROOT = Path(__file__).resolve().parent.parent
PIANO = Path(__file__).resolve().parent / "data" / "analog_piano.design.json"
STM32 = ROOT / "examples" / "designs" / "stm32f407_display_devboard.json"
ATMEGA_ANSWERS = {"input_voltage": "9 V", "clock_frequency": "16 MHz"}

#: every corpus design, smallest first
CORPUS: tuple[str, ...] = (*CASES, "amplifier", "analog_piano", "atmega128", "stm32f407")
#: the small analog designs (the templates and the amplifier fixture)
SMALL: tuple[str, ...] = (*CASES, "amplifier")


def design_file_ir(path: Path, project_id: str, library: KicadLibrary) -> CircuitIR:
    """The IR a design file describes: its topology, parts and nets (no simulation, no board)."""
    plan = load_design_file(path, library)
    payload = {c.target: c.payload for c in plan.changes if c.operation == "set"}
    return CircuitIR(project=ProjectMeta(id=project_id, name=project_id),
                     topology=payload["topology"],
                     components=[c.payload for c in plan.changes if c.target == "components"],
                     nets=[c.payload for c in plan.changes if c.target == "nets"])


def template_ir(project_id: str, workdir: Path, library: KicadLibrary, answers: dict[str, str]) -> CircuitIR:
    """A confirmed circuit template (the two-run confirm_design flow), like :func:`tests.textbook_cases.circuit`."""
    ir = CircuitIR(project=ProjectMeta(id=project_id, name=project_id, workdir=str(workdir)))
    tools = {"kicad_library": library}
    Orchestrator(AgentContext(workdir=workdir, tools=tools, answers={"application": "bench", "jurisdiction": "EU", **answers})).run(ir, stop_after=Stage.ARCHITECTURE)
    state = Orchestrator(AgentContext(workdir=workdir, tools=tools, answers={"confirm_design": "yes"})).run(ir, stop_after=Stage.ARCHITECTURE)
    assert ir.components and not state.blocked, state
    return ir


def corpus_ir(name: str, workdir: Path, library: KicadLibrary) -> CircuitIR:
    """A fresh IR of corpus design ``name``."""
    workdir.mkdir(parents=True, exist_ok=True)
    if name in CASES:
        return circuit(name, workdir, library)
    if name == "amplifier":
        return amplifier(workdir, library)
    if name == "analog_piano":
        return design_file_ir(PIANO, "analog_piano", library)
    if name == "atmega128":
        return template_ir("atmega128_devboard", workdir, library, ATMEGA_ANSWERS)
    if name == "stm32f407":
        return design_file_ir(STM32, "stm32f407_display", library)
    raise KeyError(name)


def grid_board(ir: CircuitIR, pitch_mm: float = 12.0) -> CircuitIR:
    """``ir`` with an unrouted board: every part on a square grid of ``pitch_mm`` inside one outline. Enough for DRC
    ``--schematic-parity``, which compares parts, footprints and nets, not placement or copper."""
    refs = sorted((c.ref for c in ir.components if c.footprint is not None), key=lambda r: (len(r), r))
    columns = max(1, int(len(refs) ** 0.5 + 0.999))
    placements = [Placement(component_ref=ref, x_mm=10.0 + (i % columns) * pitch_mm, y_mm=10.0 + (i // columns) * pitch_mm) for i, ref in enumerate(refs)]
    rows = (len(refs) + columns - 1) // columns
    ir.pcb = PCBDesign(outline=BoardOutline(width_mm=20.0 + columns * pitch_mm, height_mm=20.0 + rows * pitch_mm), placements=placements)
    return ir
