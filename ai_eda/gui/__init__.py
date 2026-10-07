"""Local web GUI (``ai-eda gui``): a loopback-only view of the projects under one root, and a launcher of CLI runs.

Invariants every module of this package keeps:

* **A view and a launcher, never a judge.** The GUI computes no status (every
  status it shows is copied from ``ir.validation`` / ``pipeline.json``
  through :func:`ai_eda.report.data.build_report_data`), registers no
  artifact, hashes nothing and never edits ``ir.json`` itself. The only
  writes it does on its own are a new project folder (through the same code
  path as ``ai-eda new``) and the run logs under ``<workdir>/gui/runs/``.
* **Every pipeline run is a subprocess** of the CLI (``python -P -m
  ai_eda.cli run ...``; ``-P`` keeps the project folder off its
  ``sys.path``, so a folder never supplies code), never in-process: its
  command line is built only from the
  CLI's own run options, so the approvals are exactly the CLI's (the LLM
  budget fields and the online checkbox become ``--llm-budget-*`` /
  ``--online`` flags whose gate the subprocess records). The GUI never adds a
  flag the user did not set, never runs a shell, and never reads or shows a
  key.
* **Loopback only.** The listener binds 127.0.0.1 and nothing else (no host
  option), checks the ``Host`` header on every request and the ``Origin``
  header on every POST, serves only files inside a project's workdir (no
  symlink, no directory listing) and opens no outbound connection: it is not
  an :class:`~ai_eda.security.ExternalAction`. There is no delete or rename
  endpoint (deleting would be ``SYSTEM_DELETE``).
* **Self-contained pages.** stdlib only; no framework, no CDN, no external
  resource; previews are deterministic inline SVG with every text escaped.
  :mod:`ai_eda.gui.schematic_render` draws the ``.kicad_sch`` the compiler
  wrote from its own ``lib_symbols`` - a preview, not a KiCad render, and no
  ERC is implied. The 3D tab's viewer is hand-written WebGL in the served
  ``app.js`` (no library): it draws the glTF binary of the built-in 3D
  preview (``/preview/<name>/board.glb``, part bodies are boxes of the
  ``F.Fab`` outline x the STEP height, not the parts' shapes) and falls back
  to the server's isometric SVG of the same scene where WebGL is missing;
  KiCad's own STEP / GLB / render files (real part shapes) are only linked /
  shown when a kicad-cli run registered them.
* The UI text is Korean; technical identifiers stay as they are and the
  embedded English ``report.html`` is shown unchanged.

Modules: :mod:`~ai_eda.gui.projects` (the projects root), :mod:`~ai_eda.gui.runs`
(the CLI subprocesses and their logs), :mod:`~ai_eda.gui.schematic_render`
(the schematic SVG), :mod:`~ai_eda.gui.preview` (every other preview, the
file gate and the project zip), :mod:`~ai_eda.gui.server` (the loopback HTTP
server and its routes; ``ai-eda gui``) and :mod:`~ai_eda.gui.page` (the page).
"""
