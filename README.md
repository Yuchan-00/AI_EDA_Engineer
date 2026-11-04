# AI EDA ENGINEER

**English** · [한국어](README.ko.md)

A **verification-first** AI engineering system for electronic circuit design. It takes a natural-language requirement through one pipeline:

`requirements` → `parts & regulatory research` → `Universal Circuit IR` → `calculations / SPICE` → `KiCad schematic & PCB` → `ERC / DRC` → `manufacturing outputs` → `independent review` → `auto-fix`

> **Core principle: the LLM never decides design truth.** It only proposes and orchestrates. Facts come from the IR, provenance, deterministic tools (calculators, SPICE, KiCad ERC/DRC, manufacturing-file checks) and an independent reviewer.

This page is an English overview. The full reference — every stage, check, option, measured result and the change history — is the Korean [README.ko.md](README.ko.md). The generated stage reports and the local web GUI are in Korean; the read-only HTML report (`ai-eda report`) is in English.

## Quick install

**macOS / Linux (Ubuntu, Debian, Arch, Fedora)** — in a terminal:

```bash
curl -fsSL https://raw.githubusercontent.com/Yuchan-00/AI_EDA_Engineer/main/install.sh | bash
```

**Windows 10 / 11** — in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -c "irm https://raw.githubusercontent.com/Yuchan-00/AI_EDA_Engineer/main/install.ps1 | iex"
```

If you already have the repository, run `./install.sh` (macOS / Linux) or `install.cmd` (Windows) in its folder.

- The installer sets up a uv-managed Python 3.12 environment, KiCad 10 (with libraries) and ngspice without asking. A browser for PDF reports is installed only on Linux when none is found; macOS and Windows use an existing Chrome or Edge.
- It puts `ai-eda` on your PATH and finishes with `ai-eda selftest`, which designs a 12 V → 5 V divider end to end to confirm the tools actually run.
- Do not run it with `sudo`; the script asks for an administrator password itself when it needs one (UAC on Windows).
- Debian stable has no KiCad 10 package: install with `--no-kicad`, then install KiCad 10 yourself. Options and per-OS details are in the Korean README (설치).

Then open a new terminal:

```bash
ai-eda doctor       # tools found (KiCad, ngspice, browser) and how to install what is missing
ai-eda gui --open   # local web GUI (Korean)
```

The install CI (GitHub Actions, 2026-10-14) installs, runs `doctor --strict` and `selftest` on ubuntu-24.04, ubuntu-22.04, macos-15, Arch Linux and windows-2025; on all five the selftest's divider board passed KiCad ERC/DRC with 0 errors / 0 warnings.

## Usage

```bash
ai-eda new demo --request "a circuit that converts 12 V input to 5 V"
ai-eda run projects/demo/ir.json
```

When required information is missing, the pipeline stops at `USER_INPUT_REQUIRED` and prints its questions. Answer with `--answer key=value` and run again:

```bash
ai-eda run projects/demo/ir.json --answer application="bench supply" --answer jurisdiction=EU \
  --answer input_voltage="12 V" --answer output_voltage="5 V"
```

A template's free choices (for example the divider's R2) are shown in a `confirm_design` table and applied only after you confirm it on a later run with `--answer confirm_design=yes`. Other commands: `review`, `report` / `serve` (read-only HTML report), `stage-reports` (Korean stage reports as Markdown, HTML and PDF), `relocate` (re-record a copied or moved project), `setup-kicad`.

## What it does

| Layer | What it is |
|---|---|
| **Universal Circuit IR** | Pydantic v2 model of the project, requirements, components, pins, nets, topology, constraints, PCB, regulatory scope, validation and artifacts. The design hash covers design content only, so the same design gets the same hash anywhere. |
| **Provenance** | Every important value is `Traced[T]`: `user_requirement`, `authoritative`, `assumption`, `derived` or `llm_generated`. |
| **Validation status** | `PASS / FAIL / NOT_VERIFIED / NOT_APPLICABLE / USER_INPUT_REQUIRED / UNRESOLVED`, aggregated conservatively. No evidence means `NOT_VERIFIED`, never `PASS`. |
| **Calculators** | Deterministic, registered calculators (Ohm's law, dividers, RC, LED resistors, power, junction temperature, transmission lines, RF matching and filters, ...) whose results are recomputed and checked. |
| **SPICE** | ngspice through KiCad's bundled library (ctypes): operating point, DC sweep, transient and AC analyses, judged against expected values with explicit tolerances or one-sided bounds. |
| **KiCad** | Compilers from the IR to `.kicad_sch`, `.kicad_pcb` and `.kicad_pro`, using symbols and footprints read from the installed KiCad libraries (never from model memory); ERC/DRC through `kicad-cli`, where any violation, warnings included, is a FAIL. |
| **Layout** | Deterministic placement (grid, core-ring, RF floorplan), a grid maze router with negotiated congestion, silkscreen placement, a built-in 3D preview, and need-driven signal-integrity checks (impedance, critical length, RF length). |
| **Manufacturing** | Gerber / drill export with format checks, BOM and CPL, and fab-capability limits grounded verbatim from an archived vendor page. |
| **Parts & regulatory** | Part existence checks against libraries and archived datasheets, datasheet-fact grounding, catalog import, and curated regulatory candidates (EU and KR) with deterministic applicability. Sources are fetched only with `--online` from trusted origins and archived with sha256. Regulatory *compliance* is never claimed (`NOT_VERIFIED`). |
| **Templates** | Deterministic circuit templates: unloaded resistor divider, LED + series resistor, first-order RC low-pass, BJT astable, ATmega128 dev board, and five KR 447 MHz radio boards (`radio_build`). |
| **Agents & LLM** | Ten agents that only propose; the orchestrator validates and applies proposals. LLMs (OpenRouter with a budget flag as approval, or your local Claude Code CLI) extract requirements and suggest parts; every model output must be grounded and then confirmed by you on a later run before it enters the IR. |
| **Review & repair** | An independent reviewer covering 14 areas, and an auto-repair loop that only regenerates artifacts or re-runs tools. Design changes need a human. |
| **Approval gate** | Outward-facing actions (paid model calls, network fetches, secrets, git push, orders, submissions) go through an explicit approval gate. |

## Known limits

- Component suitability is judged for electrical stress only (nominal values, one operating point, one temperature). Safety, regulatory compliance, environment, reliability, manufacturability, sourcing and cost are not evaluated by any code.
- Several KiCad canary tests (silkscreen DRC, 3D export, sibling project rules) have not yet run on a KiCad machine; until they do, the checks that depend on them stay `NOT_VERIFIED` by design. The only routed board whose real DRC has run is the selftest's divider in the install CI.
- Board impedance is computed quasi-statically and losslessly; only a fab's coupon measurement can confirm it.

The full list, with the reason each limit exists, is in [README.ko.md](README.ko.md).

## Project layout

```
ai_eda/
  ir/            Universal Circuit IR, provenance, validation status
  validation/    validators and their registry
  design/        deterministic circuit templates (rf/: KR 447 MHz radio family)
  tools/         calculators, ngspice runner, KiCad parser / libraries / kicad-cli wrapper,
                 placement, routing, silkscreen, 3D preview, manufacturing, source archive
  parts/         part existence, identity grounding, datasheet facts, catalog
  regulatory/    curated candidates, applicability, official-source research
  compilers/     IR → .kicad_sch / .kicad_pcb / .kicad_pro / BOM / CPL / SPICE netlist
  agents/        agents (propose only)
  llm/           model layer (OpenRouter, Claude Code CLI, budget, extraction)
  review/        independent reviewer (14 areas)
  repair/        deterministic repair strategies and closed loop
  report/, gui/  read-only HTML report, Korean stage reports, local web GUI
  workflow/      19-stage pipeline and orchestrator
  security/      approval gate for outward-facing actions
  cli.py         doctor / setup-kicad / selftest / new / run / review / relocate / report / serve / stage-reports / gui
tests/           test suite (tool tests skip when KiCad / ngspice are missing)
docs/ARCHITECTURE.md   spec ↔ module map and invariants (Korean)
install.sh, install.ps1, install.cmd   one-line installers
```

Run the tests with `python -m pytest -q -p no:cacheprovider -rs`. Without KiCad and ngspice the tool tests skip, and a green run without them is not tool verification.
