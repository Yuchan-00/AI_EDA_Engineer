# AI EDA ENGINEER — project notes for Claude

Verification-first AI circuit design system. **The LLM never decides design truth.** The Universal Circuit IR (`ai_eda/ir/`) is the only original design data; every important value is `Traced[T]` with provenance; correctness claims come only from deterministic tools (calculators, SPICE, `kicad-cli` ERC/DRC, manufacturing checks) and the independent reviewer. Read `docs/ARCHITECTURE.md` for the invariants and the spec↔module map.

## Invariants (do not break)
1. Derived artifacts (`.kicad_sch`, `.kicad_pcb`, BOM, CPL, netlists, gerbers) are compiled from the IR and registered as `ArtifactRef` with `generated_from_ir_hash`. Compilers must be deterministic: same IR → byte-identical output (use `ai_eda.compilers.ids.stable_uuid`, never `uuid4()`/time).
2. A compiler never emits a symbol/footprint it did not find on disk in a KiCad library (`ai_eda.tools.kicad.library`). Pin numbers / pad geometry come from the library file, never from model memory. Mismatch between IR pins and library pins is an error, not a guess.
3. `ValidationStatus` is never boolean. No evidence → `NOT_VERIFIED`, never `PASS`. ERC/DRC results are produced only by `ai_eda/tools/kicad/cli.py`.
4. Agents return `IRProposal`s; only `Orchestrator.apply_proposals` mutates the IR.
5. Repairs only regenerate artifacts from the IR or re-run tools. Design changes need a human.
6. Outward-facing actions go through `ai_eda.security.require_approval`.

## Environment
- Windows 11, Korean locale. Python venv: `B:\Claude\.venv\Scripts\python.exe`. Run tests: `cd /b/Claude && .venv/Scripts/python -m pytest -q`.
- KiCad 10.0.6: `C:\Users\yc012\AppData\Local\Programs\KiCad\10.0\bin\kicad-cli.exe`; libraries at `C:\Users\yc012\AppData\Local\Programs\KiCad\10.0\share\kicad\symbols\*.kicad_sym`, `...\footprints\*.pretty\*.kicad_mod`, demos at `...\share\kicad\demos`.
- ngspice is NOT installed. No LLM key is set. Never rely on either in tests.
- Always pass `encoding="utf-8", errors="replace"` to `subprocess.run(text=True)`; the console code page is cp949.
- Bash-tool heredocs (`cat > f <<'EOF'`) are unreliable here — write files with the Write/Edit tools.
- Scratch/experiment files go in `C:\Users\yc012\AppData\Local\Temp\claude\B--Claude\abcacf4d-e090-437c-84fc-e48862689e37\scratchpad`, never in the repo. Do not touch `projects/`.

## Conventions
- Python 3.12, pydantic v2, `from __future__ import annotations`, type hints everywhere, small modules, docstrings that state the invariant the module enforces.
- Tests in `tests/`; tests that need `kicad-cli` use `pytest.mark.skipif(not KicadCli().available(), ...)` and must otherwise run against the real binary.
- Tool-backed `ValidationResult`s set `tool`, `tool_version`, `artifact_hash`, and attach `Evidence` (report path + hash).
