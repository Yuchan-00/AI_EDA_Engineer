"""Textbook 1.0 layout on random circuits: 200 seeded IRs of 3-60 parts (R, C, D, LED, 2N3904, LM358, TL081, Conn_01xN,
SW_Push). No layout reason ever raises, every fallback rung is named in the first note, the drawn sheet holds no hard
finding, and on 20 of them KiCad's ERC is clean and its netlist equals the IR.

The router's budget is lowered to 1,000 expansions per search here, so 200 boards take minutes: a connection it does
not make is joined by name labels and noted, exactly as with the full budget."""

from __future__ import annotations

import functools
import random

import pytest

from ai_eda.compilers import SchematicCompiler
from ai_eda.compilers.textbook import build, pipeline
from ai_eda.compilers.textbook.recipes import Ctx
from ai_eda.ir import CircuitIR, Component, LibraryRef, Net, Pin, PinElectricalType, PinRef, ProjectMeta, Provenance, ProvenanceKind
from ai_eda.tools.kicad import KicadCli, KicadLibrary, sexpr
from tests.test_schematic_compiler import netlist_nets

LIB = KicadLibrary()
CLI = KicadCli()
needs_libs = pytest.mark.skipif(LIB.symbol_file("Device") is None, reason="KiCad libraries not installed")
needs_kicad = pytest.mark.skipif(not CLI.available() or LIB.symbol_file("Device") is None, reason="KiCad CLI/libraries not installed")

SEEDS = range(200)
KICAD_SEEDS = range(0, 200, 10)  # 20 of them through kicad-cli
BUDGET = 1_000
#: (weight, library, symbol, reference prefix, value)
PARTS = ((6, "Device", "R", "R", "10k"), (4, "Device", "C", "C", "100n"), (2, "Device", "D", "D", "1N4148"),
         (2, "Device", "LED", "D", "LED"), (3, "Transistor_BJT", "2N3904", "Q", "2N3904"),
         (1, "Amplifier_Operational", "LM358", "U", "LM358"), (1, "Amplifier_Operational", "TL081", "U", "TL081"),
         (2, "Connector_Generic", "Conn_01x0{n}", "J", "Conn"), (2, "Switch", "SW_Push", "SW", "SW_Push"))


def random_ir(seed: int) -> CircuitIR:
    """A valid random circuit: every pin on a net or marked no-connect; supply pins of the op-amps on VCC / GND; each
    output pin on a net of its own with at least one passive pin, so KiCad finds nothing to report."""
    rng = random.Random(seed)
    prov = Provenance(kind=ProvenanceKind.USER_REQUIREMENT, note="fuzz")
    ir = CircuitIR(project=ProjectMeta(id=f"fuzz{seed}", name=f"fuzz{seed}"))
    count = 3 + int(57 * rng.random() ** 3)  # 3 to 60 parts, most of them small boards
    weights = [w for w, *_ in PARTS]
    refs: dict[str, int] = {}
    nets: dict[str, list[tuple[str, str]]] = {"VCC": [], "GND": []}
    signals: list[str] = []
    passive: dict[str, bool] = {}

    def new_net() -> str:
        name = f"N{len(signals) + 1}"
        signals.append(name)
        nets[name] = []
        passive[name] = False
        return name

    for _ in range(count):
        _, lib, name, prefix, value = rng.choices(PARTS, weights)[0]
        name = name.format(n=rng.randint(2, 6))
        libref = LIB.resolve_symbol(LibraryRef(library=lib, name=name))
        symbol = LIB.load_symbol(libref)
        refs[prefix] = refs.get(prefix, 0) + 1
        ref = f"{prefix}{refs[prefix]}"
        pins = []
        for p in symbol.pins:
            etype = p.ir_type
            if p.electrical_type == "no_connect" or (name == "TL081" and p.name in ("NULL", "NC")):
                pins.append(Pin(number=p.number, name=p.name, electrical_type=PinElectricalType.NO_CONNECT, provenance=prov))
                continue
            pins.append(Pin(number=p.number, name=p.name, electrical_type=etype, provenance=prov))
            if p.electrical_type == "power_in":
                net = "VCC" if p.name.startswith("V+") else "GND"
            elif p.electrical_type == "output":
                net = new_net()
            else:  # a rail, a small signal net already there (2-4 pins, like most nets of a real board) or a new one
                roll = rng.random()
                small = [n for n in signals if len(nets[n]) < rng.randint(2, 4)]
                net = "VCC" if roll < 0.12 else "GND" if roll < 0.27 else (rng.choice(small) if small and roll < 0.7 else new_net())
            nets[net].append((ref, p.number))
            if net in passive and p.electrical_type == "passive":
                passive[net] = True
        ir.components.append(Component(ref=ref, value=value, symbol=libref, provenance=prov, pins=pins))
    # a signal net needs two pins and a passive one: an output's net gets a load resistor to ground, any other short or
    # passive-free net joins the first good one (or ground)
    keep = [n for n in signals if passive[n] and len(nets[n]) >= 2]
    resistor = LIB.resolve_symbol(LibraryRef(library="Device", name="R"))
    for n in signals:
        if n in keep:
            continue
        if any(ir.component(r).pin(q).electrical_type is PinElectricalType.OUTPUT for r, q in nets[n]):
            refs["R"] = refs.get("R", 0) + 1
            ref = f"R{refs['R']}"
            ir.components.append(Component(ref=ref, value="10k", symbol=resistor, provenance=prov, pins=[
                Pin(number=q, name="", electrical_type=PinElectricalType.PASSIVE, provenance=prov) for q in ("1", "2")]))
            nets[n].append((ref, "1"))
            nets["GND"].append((ref, "2"))
            continue
        nets[keep[0] if keep else "GND"] += nets.pop(n)
    for name in sorted(nets, key=lambda n: (n not in ("VCC", "GND"), n)):
        if nets[name]:
            kind = "power" if name == "VCC" else "ground" if name == "GND" else "signal"
            ir.nets.append(Net(name=name, kind=kind, provenance=prov,
                               pins=[PinRef(component_ref=r, pin_number=p) for r, p in nets[name]]))
    return ir


@pytest.fixture
def quick(monkeypatch):
    monkeypatch.setattr(pipeline, "Ctx", functools.partial(Ctx, budget=BUDGET))


@needs_libs
@pytest.mark.parametrize("chunk", range(10))
def test_random_circuits_never_fail_for_a_layout_reason_and_name_every_fallback(chunk, quick):
    for seed in SEEDS[chunk::10]:
        ir = random_ir(seed)
        result = build(SchematicCompiler(), ir, LIB)  # a layout reason never raises
        head = result.notes[0]
        assert head.startswith("layout: textbook 1.0;"), (seed, head)
        listed = head.split("fallbacks: ", 1)[1]
        assert all(f in listed for f in result.fallbacks), (seed, head)
        assert (result.score is None) == ("whole-sheet table" in result.fallbacks), (seed, head)
        if result.score is not None:
            assert result.score[0] == 0, (seed, head)  # nothing KiCad would connect that the IR does not state


@needs_kicad
@pytest.mark.parametrize("seed", KICAD_SEEDS)
def test_random_circuits_pass_erc_and_keep_the_ir_netlist(tmp_path, seed, quick):
    ir = random_ir(seed)
    result = build(SchematicCompiler(), ir, LIB)
    path = tmp_path / f"fuzz{seed}.kicad_sch"
    path.write_text(sexpr.dumps(result.node), encoding="utf-8", newline="\n")
    erc = CLI.run_erc(path, path.with_suffix(".erc.json"))
    assert not erc.details["errors"] and not erc.details["warnings"], (seed, result.notes[0], erc.details)
    nets = netlist_nets(CLI.export_netlist(path, path.with_suffix(".net"), fmt="kicadsexpr"))
    got = {frozenset(p) for n, p in nets.items() if p and not n.startswith("unconnected-")}
    want = {frozenset((p.component_ref, p.pin_number) for p in n.pins) for n in ir.nets if n.pins}
    assert got == want, seed
