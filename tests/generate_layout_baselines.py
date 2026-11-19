"""Regenerate measured baselines: python -m tests.generate_layout_baselines --stm32-run PATH.

The saved STM32 run is read only. Templates use 9 V / 16 MHz and a 5 V ->
3.3 V unloaded divider, the current default placement/routing, fixed two
copper layers, and no silkscreen (not part of these measurements). The
corners are mitred (pcb.corners=miter45, routing.corners 0.1), the style the
recorded boards were routed with: these are baselines of the placer and the
router, not of the corner style, so the agent's later default (arc) does not
change them.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_eda.agents import AgentContext, PCBAgent
from ai_eda.compilers import CompileContext, SchematicCompiler
from ai_eda.ir import CircuitIR
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.layout_metrics import METRICS_VERSION, pcb_metrics, schematic_metrics
from ai_eda.workflow import Orchestrator
from tests.layout_baseline_data import DATA, freeze
from tests.test_circuit_templates import _confirm, _ir, _present


def digest(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stm32-run", type=Path, required=True)
    args = parser.parse_args()
    library = KicadLibrary()
    DATA.mkdir(parents=True, exist_ok=True)
    result = {"metrics_version": METRICS_VERSION, "boards": {}}

    def record(name, ir, schematic, source):
        raw = schematic.read_bytes()
        geometry = freeze(ir, library)
        packed = json.dumps(geometry, sort_keys=True, separators=(",", ":")).encode()
        (DATA / f"{name}.json.gz").write_bytes(gzip.compress(packed, mtime=0))
        (DATA / f"{name}.kicad_sch.gz").write_bytes(gzip.compress(raw, mtime=0))
        result["boards"][name] = {"source": source, "source_ir_hash": ir.content_hash(), "schematic_sha256": digest(raw),
                                 "geometry_sha256": digest(packed), "schematic": schematic_metrics(schematic), "pcb": pcb_metrics(ir, library)}
        print(name, json.dumps({k: v for k, v in result["boards"][name]["pcb"].items() if k != "airwire_mst_per_net_mm"}), flush=True)

    saved = CircuitIR.load(args.stm32_run / "ir.json")
    record("stm32_display", saved, args.stm32_run / "stm32_display.kicad_sch", "saved STM32 display run; original routing retained")
    with TemporaryDirectory() as tmp:
        for name, answers in (("atmega128_devboard", {"input_voltage": "9 V", "clock_frequency": "16 MHz"}),
                              ("divider", {"input_voltage": "5 V", "output_voltage": "3.3 V"})):
            workdir = Path(tmp) / name
            workdir.mkdir()
            ir = _ir(workdir, name)
            answers["pcb_layers"] = "2"
            _present(ir, workdir, library, answers)
            state, _ = _confirm(ir, workdir, library)
            assert not state.blocked
            res = PCBAgent().run(ir, AgentContext(workdir=workdir, tools={"kicad_library": library}, answers={"pcb.silkscreen": "skip", "pcb.corners": "miter45"}))
            Orchestrator.apply_proposals(ir, res.proposals)
            assert ir.pcb and ir.pcb.tracks, res.notes
            # the frozen baselines were measured on the legacy table: regenerate them with it (textbook 1.0 is the default)
            schematic = SchematicCompiler("legacy").compile(ir, CompileContext(workdir=workdir, tools={"kicad_library": library}))
            record(name, ir, Path(schematic.path), {"template_inputs": answers, "placement_notes": res.notes})
    (DATA / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
