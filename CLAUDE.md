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
- There is no `ngspice.exe`; SPICE runs through KiCad's bundled `bin\ngspice.dll` (ngspice-46) via `ai_eda.tools.spice.NgspiceShared` (ctypes, in-process singleton). Tests that need it use `pytest.mark.skipif(not NgspiceShared().available(), ...)` and otherwise run against the real DLL. Measured facts the code depends on: analysis commands must be lowercase device names (`dc vvin 0 12 1`), decks carry no analysis cards, node names are limited to `NODE_RE`, `ngGet_Vec_Info` returns one static struct (copy before the next call), rawfiles carry a `Date` line (evidence, not deterministic artifacts), and ngspice's number parser is not correctly rounded (`10u` reads as 9.999999999999999e-06 - `ai_eda.tools.calc.si` models it).
- No LLM key is set. Never rely on one in tests.
- Always pass `encoding="utf-8", errors="replace"` to `subprocess.run(text=True)`; the console code page is cp949.
- Bash-tool heredocs (`cat > f <<'EOF'`) are unreliable here — write files with the Write/Edit tools.
- Scratch/experiment files go in `C:\Users\yc012\AppData\Local\Temp\claude\B--Claude\abcacf4d-e090-437c-84fc-e48862689e37\scratchpad`, never in the repo. Do not touch `projects/`.

## Linux / Claude Code web / any other machine
- The "Environment" section above describes the Windows PC this was built on; nothing there is a requirement. Every tool is discovered at runtime: `kicad-cli` via PATH (`ai_eda.tools.kicad.cli.find_kicad_cli`); KiCad symbol/footprint libraries from that binary's install root (`<root>/share/kicad`, e.g. `/usr/share/kicad` on Debian/Ubuntu) or `$KICAD10_SYMBOL_DIR` / `$KICAD_SYMBOL_DIR`; the ngspice shared library from `$NGSPICE_DLL` (path must exist) else `ngspice.dll` next to `kicad-cli`; XSPICE code models from `$NGSPICE_CODEMODEL_DIR` else `<root>/lib/ngspice`. `python -m ai_eda.cli doctor` prints what was found.
- Setup: `python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev,llm,parts]"`; add `cryptography` if datasheet PDFs are AES-encrypted (3 of 20 sampled manufacturer PDFs were). Run tests with `python -m pytest -q -p no:cacheprovider -rs`.
- Without KiCad / ngspice the tool-backed tests skip (`skipif`) and the pipeline reports those stages `NOT_VERIFIED`. A green run without them is not tool verification. Reference on this PC with both installed: 1040 passed, 5 skipped (2026-09-23).
- To get the tool-backed tests on Linux (NOT yet measured as of 2026-09-23 - verify, then update this note): install KiCad 10 (`kicad-cli` on PATH) and ngspice's shared library (`libngspice0` -> e.g. `NGSPICE_DLL=/usr/lib/x86_64-linux-gnu/libngspice.so.0`, `NGSPICE_CODEMODEL_DIR=<directory with the *.cm files>`). The loader uses `ctypes.CDLL` and guards the Windows-only `os.add_dll_directory`. The measured ngspice facts above come from KiCad's ngspice-46 build; re-measure another build with `tests/ngspice_parser_harness.py` before trusting `ai_eda.tools.calc.si`.
- Network-touching tests skip unless `AI_EDA_ONLINE=1` (read-only fetches of EUR-Lex, law.go.kr, eCFR); live OpenRouter tests need `OPENROUTER_API_KEY` and spend credits. Never enable either by default in CI. The archive never opens a socket without `--online`.
- Line endings are forced to LF by `.gitattributes`; write files with LF. The cp949 / heredoc notes above are Windows-only.
- Git: `origin` = https://github.com/Yuchan-00/AI_EDA_Engineer (branch `main`). Commit messages end with the Co-Authored-By line the session provides. Never force-push.

## Conventions
- Python 3.12, pydantic v2, `from __future__ import annotations`, type hints everywhere, small modules, docstrings that state the invariant the module enforces.
- Tests in `tests/`; tests that need `kicad-cli` use `pytest.mark.skipif(not KicadCli().available(), ...)` and must otherwise run against the real binary.
- Tool-backed `ValidationResult`s set `tool`, `tool_version`, `artifact_hash`, and attach `Evidence` (report path + hash).
