"""IR -> SPICE netlist compiler.

Invariants this module enforces:

* **Pure function of the IR.** :func:`build` returns deterministic text: the
  title line is ``project.id`` alone, model cards are sorted by name,
  elements are in sorted ref order, stimuli in sorted id order, LF newlines,
  ``.end`` last, no timestamps, no hashes, no comments.
* **Nothing is guessed.** Every component needs a :class:`SpiceBinding`
  (``None`` -> ``CompileError``; ``NothingToCompileError`` when no component
  has one yet, i.e. the mapping stage has not run); an included part needs a
  device, a value or a model, a ``pin_order`` that is a permutation of (a
  subset of) its pins, and every pin in a net - a pin the element does not
  use may only be left out of the nets when the IR types it ``no_connect``,
  the same rule the schematic compiler applies, and a pin that *is* in a net
  but not in ``pin_order`` must be listed in ``ignored_pins`` with a reason
  (the schematic connects it, the netlist does not; the report names it and
  the nets that lose their only connection that way). Exactly one
  ``NetKind.GROUND`` net becomes node ``0``; other nets keep their IR names,
  and two names that differ only by case (ngspice is case-insensitive), or a
  non-ground net called ``gnd``/``0`` (ngspice's ground aliases), are refused.
  A model a device references must be defined by a ``model_card`` in the IR:
  no ``.include``/``.lib`` is ever emitted, so the netlist is self-contained
  and its hash covers the models.
* **The simulated part is the shipped part.** An R / C / L value must equal
  the component's ``electrical["resistance" | "capacitance" | "inductance"]``
  when that entry exists (same unit, same number); a binding that would
  simulate 10 k while the BOM ships 20 k is a ``CompileError``. Which source
  each value was reconciled with is in the report (``value_sources``).
* **Nothing llm_generated reaches ngspice.** Component values, params, model
  cards, stimulus values/params, analysis params, expectation numbers *and
  the bindings, stimuli, analyses and expectations themselves* (which device
  a part gets, which net is stimulated, what runs, what is judged) with
  ``LLM_GENERATED`` provenance are ``CompileError``; model cards must be
  authoritative or user-supplied. The kinds that were accepted (and every
  value or decision that rests on an ``assumption``) are listed by
  :func:`build_report`; the SPICE stage reports results that rest on an
  assumption as NOT_VERIFIED.
* **A lossless line is fully stated.** A ``T`` element (``SpiceDevice.T``,
  four pins: port 1 +/-, port 2 +/-) carries exactly ``z0`` (ohm) and
  ``td`` (s), both positive, and no value or model; ngspice-42 simulates a
  ``T`` line without ``td`` with a delay of its own choosing and no error
  (measured), so such a binding is a ``CompileError`` and the runner's deck
  check refuses such a line too.
* **A tolerance must be a tolerance.** An expectation whose ``nominal`` is 0
  needs ``tol_abs``: ``tol_rel`` alone would be a zero tolerance. A one-sided
  ``bound`` (``at_least`` / ``at_most``: PASS on the passing side of
  ``nominal``) replaces the tolerance: it takes no ``tol_abs`` / ``tol_rel``
  (so it also waives the dB rule below) and cannot claim a requirement yet
  (``requirement_id``: the reviewer's nominal-vs-requirement rule and the
  tolerance chart know only tolerances).
* **A reduction must fit its analysis.** ``reduce=value`` only on ``op``,
  ``at`` / ``final`` / ``max`` / ``min`` only on a sweep, and
  ``reduce=frequency`` (rising mid-level crossings over time) only on a
  ``tran`` analysis: a dc or ac sweep has no time axis to count edges on.
  ``reduce=db_at`` only on an ``ac`` analysis, with ``at`` and a magnitude
  vector (never ``vp`` / ``vr`` / ``vi``); ``rms`` / ``db_rms`` /
  ``harmonic_dbc`` / ``am_depth`` only on a ``tran`` analysis, over a window
  ``[t_start, t_stop]`` inside the analysis' saved window (with a tran
  ``start`` > 0, ``t_start`` at least one step after it: ngspice-42 saves its
  first point up to one step after ``start`` - measured). A level reduction
  names exactly one reference (``reference_vector``, a magnitude of the same
  kind, or ``params["ref"]`` > 0); ``harmonic_dbc`` takes an integer ``k >= 2``
  and a window of whole periods of ``f0``; ``am_depth`` needs
  ``f_carrier >= 20 f_mod`` and a window of at least one modulation period;
  a dB level needs ``tol_abs`` and no ``tol_rel`` (or a one-sided ``bound``); a stated nominal unit must
  be the reduction's (``dB``, ``dBc``, ``percent``, the vector's for ``rms``).
  Every param is a finite traced number through the same llm_generated
  refusal; the other reductions take none. None of it reaches the deck: the
  reductions are Python arithmetic on the vectors ngspice wrote, so no
  ``.save`` / ``.meas`` / ``.four`` / ``sp`` / ``noise`` card or command
  exists here.
* **A modulated source is fully stated.** ``AM(VA VO MF FC TD)`` and
  ``SFFM(VO VA FC MDI FS)`` take every parameter, finite; ``mf`` / ``fc``
  (AM) and ``fc`` / ``fs`` (SFFM) must be > 0 and ``td`` >= 0 - ngspice-42
  silently replaces a zero MF or FS by a default of its own and simulates
  another waveform (measured, :class:`~ai_eda.ir.simulation.StimulusKind`).
* **No analysis in the netlist.** No ``.control`` block and no
  ``.op``/``.dc``/``.ac``/``.tran`` cards: :func:`analysis_command` renders
  an :class:`AnalysisSpec` as the interactive command the runner issues
  (``"op"``, ``"dc VVIN 0 12 1"``, ``"tran 1e-5 5m"``, ``"ac dec 10 1 1meg"``;
  the tran grammar is ``tran <step> <stop> [<start>] [uic]`` where ``uic``
  comes from the bool param of that name - ``True`` appends the keyword,
  ``False`` emits nothing, anything else is a ``CompileError``), numbers
  formatted by :func:`ai_eda.tools.calc.si.format_spice_number` so
  ngspice reads them back as the IR value; the few it would still read one
  ULP off (or that the model cannot vouch for) are listed by
  :func:`build_report` as ``inexact_numbers`` / ``unmodelled_numbers``.
  ``.temp`` is the only option card, emitted when
  ``SimulationSetup.temperature_c`` is set.
* **What compiles, runs.** Net names must match the runner's node rule
  (:data:`ai_eda.tools.spice.ngspice_shared.NODE_RE`: ngspice mangles the
  other characters) and the finished text must pass
  :func:`ai_eda.tools.spice.ngspice_shared.validate_deck` - the same judge
  the runner applies before ngspice.dll sees a deck - so a netlist this
  compiler wrote is never refused by the runner for its *content* (the
  runner loads the validated bytes through ``ngSpice_Circ``, so the file's
  path does not matter either; only where the rawfile can be written does,
  and that is reported as NOT_VERIFIED, not as a verdict). The title line
  (``project.id``) may not contain ``$`` (ngspice's inline-comment marker),
  quotes or backticks. Analysis and expectation ids are plain identifiers:
  they name the run directory, the rawfile and the ``spice.<id>`` check.

Netlist layout::

    <project.id>
    [.temp <t>]
    <model cards, verbatim, sorted by model name>
    <elements: "<NAME> <nodes in pin_order> <value|model_name> [k=v ...]">
               (a lossless line: "T<ref> <p1+> <p1-> <p2+> <p2-> td=<s> z0=<ohm>")
    <stimuli:  "V<id>|I<id> <net node> <reference node> DC <v> | PULSE(...) | SINE(...) | PWL(...) | AM(...) | SFFM(...) [AC <mag>]">
    .end

The element name is the IR ref when it already starts with the device
letter (``R1``, ``V1``) and ``<letter><ref>`` otherwise (``U1`` bound as a
subcircuit -> ``XU1``). :func:`netlist_elements` parses this layout back
(for tests and cross-checks) and :func:`spice_vector_name` maps an
:class:`Expectation` vector (``v(VOUT)``, ``i(VIN)``, and for ac analyses
``vp()`` / ``vr()`` / ``vi()`` phase / real / imaginary) to the ngspice
vector name (``vout``, ``vvin#branch``, ``vout.phase_deg`` ...) after
validating it against the IR.

:meth:`SpiceNetlistCompiler.compile` writes ``<workdir>/<project.id>.cir``
and, next to it, ``<project.id>.cir.report.json`` with :func:`build_report`
(excluded parts and their reasons, nets that touch only excluded parts,
accepted provenance kinds, the analysis commands and the vector names).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from ai_eda.compilers.base import CompileContext, Compiler
from ai_eda.errors import CompileError, NothingToCompileError
from ai_eda.ir import ArtifactKind, ArtifactRef, CircuitIR, Component, Net, NetKind, PinElectricalType, Provenance, ProvenanceKind, Traced
from ai_eda.ir.simulation import (
    AC_MAGNITUDE_PARAM,
    AC_VARIATIONS,
    ANALYSIS_OPTIONAL_PARAMS,
    ANALYSIS_PARAMS,
    DEVICE_PARAMS,
    ELECTRICAL_KEYS,
    MODEL_DEVICES,
    NODE_COUNTS,
    PARAM_DEVICES,
    DB_REDUCES,
    REDUCE_PARAM_UNITS,
    REDUCE_PARAMS,
    REDUCE_UNITS,
    REFERENCE_REDUCES,
    SINE_OPTIONAL_PARAMS,
    STIMULUS_NON_NEGATIVE_PARAMS,
    STIMULUS_PARAMS,
    STIMULUS_POSITIVE_PARAMS,
    VALUE_DEVICES,
    WINDOW_REDUCES,
    AnalysisSpec,
    Expectation,
    Reduce,
    SimulationSetup,
    SpiceDevice,
    Stimulus,
    StimulusKind,
)
from ai_eda.tools.calc.si import format_spice_number, ngspice_reads
from ai_eda.tools.spice.measure import MIN_CARRIER_TO_MOD_RATIO, MIN_HARMONIC_PERIODS, PERIOD_REL_TOL
from ai_eda.tools.spice.ngspice_shared import NODE_RE, validate_deck
from ai_eda.tools.spice.runner import SpiceAnalysis

#: element names, stimulus ids and analysis ids: they appear in commands, vector names, check ids and
#: file names, so keep them plain
_NAME = re.compile(r"^[A-Za-z0-9_]+$")
#: characters that would break a node token (whitespace, parentheses, ``=``, commas, quotes)
_BAD_NODE_CHARS = frozenset(" \t\r\n()=,\"'")
#: characters the title line may not contain: ngspice's inline comment marker and the quote / backtick
#: characters :func:`validate_deck` refuses on every line
_BAD_TITLE_CHARS = frozenset("$'\"`\r\n")
#: ``v(NET)`` / ``i(SRC)`` plus the complex parts of an ac result: ``vp`` phase, ``vr`` real, ``vi`` imaginary
_VECTOR = re.compile(r"^\s*([vViI])([pPrRiI]?)\s*\(\s*([^()\s]+)\s*\)\s*$")
#: complex-part letter -> the runner's vector suffix (:mod:`ai_eda.tools.spice.rawfile` convention)
_COMPLEX_SUFFIX = {"p": ".phase_deg", "r": ".real", "i": ".imag"}
_CARD_DEF = re.compile(r"^\.(model|subckt)\s+(\S+)(.*)$", re.IGNORECASE)
#: card lines that would make the netlist non-self-contained or smuggle an analysis in
#: dot-card prefixes a model card may never carry, matched the way ngspice matches them (``.inc`` is ``.include``,
#: ``.opt`` is ``.options``): files, control blocks, analyses, output control, initial conditions, options, globals
_FORBIDDEN_CARD_PREFIXES: tuple[str, ...] = (
    ".inc", ".lib", ".control", ".endc", ".end", ".title", ".op", ".dc", ".ac", ".tran", ".tf", ".noise", ".pz", ".sens", ".disto",
    ".sp", ".save", ".probe", ".print", ".plot", ".meas", ".opt", ".temp", ".ic", ".nodeset", ".global", ".four", ".width",
)


def element_name(ref: str, device: SpiceDevice) -> str:
    """``R1`` -> ``R1``; ``U1`` as a subcircuit -> ``XU1`` (ngspice reads the device from the first letter)."""
    return ref if ref[:1].upper() == device.value else f"{device.value}{ref}"


def stimulus_name(stimulus: Stimulus) -> str:
    return ("V" if stimulus.source == "voltage" else "I") + stimulus.id


# --------------------------------------------------------------------------- provenance bookkeeping


@dataclass
class _Ledger:
    """What the compiler accepted, for the report - and the one place the llm_generated refusal lives.

    It also formats every number, recording the ones ngspice will read one ULP
    off (``inexact``: no spelling below :data:`ai_eda.tools.calc.si.NGSPICE_EXACT_MANTISSA`
    is read exactly) and the ones the model cannot vouch for (``unmodelled``).
    """

    kinds: set[ProvenanceKind] = field(default_factory=set)
    assumptions: list[str] = field(default_factory=list)
    inexact: list[dict] = field(default_factory=list)
    unmodelled: list[dict] = field(default_factory=list)

    def number(self, value: float, what: str) -> str:
        value = float(value)
        text = format_spice_number(value)
        read = ngspice_reads(text)
        if read is None:
            self.unmodelled.append({"what": what, "text": text, "value": value})
        elif read != value:
            self.inexact.append({"what": what, "text": text, "value": value, "ngspice_reads": read})
        return text

    def accept_provenance(self, prov: Provenance, what: str, *, model_card: bool = False) -> None:
        """Refuse ``llm_generated``, list ``assumption`` - for a traced number and for the containers
        (binding, stimulus, analysis, expectation) that decide what is simulated and judged."""
        kind = prov.kind
        if kind == ProvenanceKind.LLM_GENERATED:
            raise CompileError(f"{what} has llm_generated provenance; nothing an LLM proposed reaches ngspice until it is verified")
        if model_card and not prov.is_authoritative:
            raise CompileError(f"{what}: a model card must have authoritative or user_requirement provenance, got {kind.value}")
        self.kinds.add(kind)
        if kind == ProvenanceKind.ASSUMPTION:
            self.assumptions.append(what)

    def accept(self, traced: Traced, what: str, *, model_card: bool = False) -> Traced:
        self.accept_provenance(traced.provenance, what, model_card=model_card)
        return traced


def _number(traced: Traced, what: str, ledger: _Ledger) -> str:
    value = traced.value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CompileError(f"{what} must be a number, got {type(value).__name__}")
    try:
        return ledger.number(value, what)
    except ValueError as e:
        raise CompileError(f"{what}: {e}") from e


def _param_tokens(params: dict[str, Traced], what: str, ledger: _Ledger) -> list[str]:
    tokens: list[str] = []
    for key in sorted(params):
        if not _NAME.match(key):
            raise CompileError(f"{what}: parameter name {key!r} is not a plain identifier")
        traced = ledger.accept(params[key], f"{what} param {key}")
        value = traced.value
        if isinstance(value, str):
            if not value or any(c in _BAD_NODE_CHARS for c in value):
                raise CompileError(f"{what} param {key}: {value!r} is not a single SPICE token")
            tokens.append(f"{key}={value}")
        else:
            tokens.append(f"{key}={_number(traced, f'{what} param {key}', ledger)}")
    return tokens


# --------------------------------------------------------------------------- nets -> nodes


@dataclass
class _Nodes:
    of_net: dict[str, str]
    of_pin: dict[tuple[str, str], str]
    ground: Net


def _nodes(ir: CircuitIR) -> _Nodes:
    grounds = [n for n in ir.nets if n.kind == NetKind.GROUND]
    if len(grounds) != 1:
        raise CompileError(
            f"exactly one NetKind.GROUND net is required (it becomes node 0), found {len(grounds)}: {[n.name for n in grounds]}"
        )
    seen: dict[str, str] = {}
    for n in ir.nets:
        if not n.name or any(c in _BAD_NODE_CHARS for c in n.name):
            raise CompileError(f"net {n.name!r} is not usable as a SPICE node name")
        if n.kind != NetKind.GROUND and not NODE_RE.fullmatch(n.name):
            # the runner refuses node names outside this set before ngspice sees them (measured: ngspice
            # mangles or drops the others), so the compiler must refuse them first
            raise CompileError(
                f"net {n.name!r} is not usable as a SPICE node name: ngspice keeps only "
                f"A-Z a-z 0-9 _ . / + - : # @ [ ] intact (the runner would reject the netlist)"
            )
        low = n.name.lower()
        if low in seen:
            raise CompileError(f"nets {seen[low]!r} and {n.name!r} differ only by case; ngspice is case-insensitive")
        seen[low] = n.name
        if n.kind != NetKind.GROUND and low in ("gnd", "0"):
            raise CompileError(f"net {n.name!r} is not the GROUND net but ngspice treats {n.name!r} as ground")
    of_net = {n.name: ("0" if n.kind == NetKind.GROUND else n.name) for n in ir.nets}
    components: dict[str, Component] = {}
    for c in ir.components:
        if c.ref.lower() in components:
            raise CompileError(f"component refs {components[c.ref.lower()].ref!r} and {c.ref!r} collide (ngspice is case-insensitive)")
        components[c.ref.lower()] = c
    of_pin: dict[tuple[str, str], str] = {}
    for n in ir.nets:
        for p in n.pins:
            c = components.get(p.component_ref.lower())
            if c is None or c.ref != p.component_ref:
                raise CompileError(f"net {n.name!r} references unknown component {p.component_ref!r}")
            if c.pin(p.pin_number) is None:
                raise CompileError(f"net {n.name!r} references {p.component_ref}.{p.pin_number}, not a pin of {p.component_ref}")
            key = (p.component_ref, p.pin_number)
            if key in of_pin and of_pin[key] != of_net[n.name]:
                raise CompileError(f"pin {p.component_ref}.{p.pin_number} is in two nets")
            of_pin[key] = of_net[n.name]
    return _Nodes(of_net=of_net, of_pin=of_pin, ground=grounds[0])


# --------------------------------------------------------------------------- model cards


def _join_continuations(lines: list[str]) -> list[str]:
    out: list[str] = []
    for raw in lines:
        if raw.startswith("+") and out:
            out[-1] += " " + raw[1:]
        else:
            out.append(raw)
    return out


def _normalise_card(text: str, what: str) -> str:
    lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    if not lines:
        raise CompileError(f"{what}: model_card is empty")
    # a whitelist, not a blacklist: a model card is ``.model`` / ``.subckt`` ... ``.ends`` text and nothing else.
    # Inside a subcircuit element lines, nested ``.model`` and ``.param`` are the body; outside one, an element
    # line would add a part the IR does not have, and any other dot card (``.inc`` - ngspice matches the prefix of
    # ``.include`` -, ``.opt``, ``.ic``, ``.nodeset``, ``.global``, ``.tf`` ...) would change what is simulated
    # without appearing in the report
    depth = 0
    for ln in lines:
        s = ln.strip()
        if not s or s[0] in "*+":
            continue
        first = s.split()[0].lower()
        if first.startswith("."):
            if first == ".subckt":
                depth += 1
            elif first == ".ends":
                depth = max(0, depth - 1)
            elif first == ".model" or (first == ".param" and depth > 0):
                pass
            elif any(first.startswith(tok) for tok in _FORBIDDEN_CARD_PREFIXES):
                raise CompileError(f"{what}: model_card line {s!r} is not allowed (cards must be self-contained .model/.subckt text)")
            else:
                raise CompileError(f"{what}: model_card line {s!r} is not allowed (only .model, .subckt/.ends and, inside a subcircuit, .param and element lines)")
        elif depth == 0:
            raise CompileError(f"{what}: model_card line {s!r} is an element outside any .subckt (a card adds models, never parts)")
    if depth != 0:
        raise CompileError(f"{what}: model_card has an unclosed .subckt")
    return "\n".join(lines)


def _card_definitions(card: str) -> dict[str, tuple[str, list[str]]]:
    """``{name.lower(): ("model" | "subckt", ports)}`` for every definition in a normalised card."""
    defs: dict[str, tuple[str, list[str]]] = {}
    for ln in _join_continuations(card.split("\n")):
        m = _CARD_DEF.match(ln.strip())
        if m is None:
            continue
        kind, name, rest = m.group(1).lower(), m.group(2), m.group(3).split()
        ports: list[str] = []
        if kind == "subckt":
            for tok in rest:
                if "=" in tok or tok.lower() == "params:":
                    break
                ports.append(tok)
        defs[name.lower()] = (kind, ports)
    return defs


# --------------------------------------------------------------------------- stimuli


def _pwl_points(traced: Traced, what: str, ledger: _Ledger) -> list[str]:
    raw = traced.value
    if not isinstance(raw, (list, tuple)) or not raw:
        raise CompileError(f"{what}: points must be a non-empty list of [t, v] pairs")
    if all(isinstance(p, (list, tuple)) for p in raw):
        pairs = [tuple(p) for p in raw]
    elif len(raw) % 2 == 0:
        pairs = [(raw[i], raw[i + 1]) for i in range(0, len(raw), 2)]
    else:
        raise CompileError(f"{what}: points must be [t, v] pairs or a flat even-length list")
    out: list[str] = []
    last_t: float | None = None
    for pair in pairs:
        if len(pair) != 2 or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in pair):
            raise CompileError(f"{what}: point {pair!r} is not a numeric [t, v] pair")
        t, v = float(pair[0]), float(pair[1])
        if last_t is not None and t <= last_t:
            raise CompileError(f"{what}: PWL times must increase (t={t} after {last_t})")
        last_t = t
        out += [ledger.number(t, f"{what} point t={t!r}"), ledger.number(v, f"{what} point v at t={t!r}")]
    return out


def _stimulus_spec(s: Stimulus, ledger: _Ledger) -> str:
    what = f"stimulus {s.id}"
    params = dict(s.params)
    ac = params.pop(AC_MAGNITUDE_PARAM, None)
    parts: list[str] = []
    if s.value is not None:
        ledger.accept(s.value, f"{what} value")
        parts.append(f"DC {_number(s.value, f'{what} value', ledger)}")
    if s.kind == StimulusKind.DC:
        if s.value is None:
            raise CompileError(f"{what}: a DC stimulus needs value")
        if params:
            raise CompileError(f"{what}: unexpected params {sorted(params)} for a DC stimulus")
    else:
        required = STIMULUS_PARAMS[s.kind]
        missing = [k for k in required if k not in params]
        if missing:
            raise CompileError(f"{what}: {s.kind.value} needs params {list(required)}, missing {missing}")
        optional: tuple[str, ...] = SINE_OPTIONAL_PARAMS if s.kind == StimulusKind.SINE else ()
        extra = sorted(set(params) - set(required) - set(optional))
        if extra:
            raise CompileError(f"{what}: unexpected params {extra} for a {s.kind.value} stimulus")
        if s.kind == StimulusKind.PWL:
            numbers = _pwl_points(ledger.accept(params["points"], f"{what} param points"), f"{what} param points", ledger)
        else:
            numbers = [_number(ledger.accept(params[k], f"{what} param {k}"), f"{what} param {k}", ledger) for k in required]
            for k in STIMULUS_POSITIVE_PARAMS.get(s.kind, ()):
                if float(params[k].value) <= 0.0:
                    raise CompileError(
                        f"{what}: {s.kind.value} param {k} must be > 0, got {params[k].value!r} "
                        "(ngspice-42 silently replaces a zero one by a default of its own and simulates another waveform - measured)"
                    )
            for k in STIMULUS_NON_NEGATIVE_PARAMS.get(s.kind, ()):
                if float(params[k].value) < 0.0:
                    raise CompileError(f"{what}: {s.kind.value} param {k} (a delay) must be >= 0, got {params[k].value!r}")
            given = [k for k in optional if k in params]
            if given != list(optional[: len(given)]):
                raise CompileError(f"{what}: optional sine params must be given in order {list(optional)}, got {given}")
            numbers += [_number(ledger.accept(params[k], f"{what} param {k}"), f"{what} param {k}", ledger) for k in given]
        parts.append(f"{s.kind.value.upper()}({' '.join(numbers)})")
    if ac is not None:
        parts.append(f"AC {_number(ledger.accept(ac, f'{what} param ac'), f'{what} param ac', ledger)}")
    return " ".join(parts)


# --------------------------------------------------------------------------- analyses


def _analysis_command(spec: AnalysisSpec, setup: SimulationSetup | None, ledger: _Ledger) -> str:
    what = f"analysis {spec.id}"
    allowed = ANALYSIS_PARAMS[spec.kind]
    optional = ANALYSIS_OPTIONAL_PARAMS[spec.kind]
    extra = sorted(set(spec.params) - set(allowed))
    if extra:
        raise CompileError(f"{what}: unexpected params {extra} for {spec.kind.value}")
    missing = [k for k in allowed if k not in optional and k not in spec.params]
    if missing:
        raise CompileError(f"{what}: {spec.kind.value} needs params {list(allowed)}, missing {missing}")
    p = {k: ledger.accept(t, f"{what} param {k}") for k, t in spec.params.items()}

    def num(key: str) -> float:
        v = p[key].value
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise CompileError(f"{what} param {key} must be a number, got {type(v).__name__}")
        return float(v)

    def fmt(value: float, key: str) -> str:
        return ledger.number(value, f"{what} param {key}")

    if spec.kind == SpiceAnalysis.OP:
        return "op"
    if spec.kind == SpiceAnalysis.DC:
        source = p["source"].value
        if not isinstance(source, str):
            raise CompileError(f"{what}: dc source must be a stimulus id string")
        stimulus = setup.stimulus(source) if setup is not None else None
        if stimulus is None:
            raise CompileError(f"{what}: dc sweeps unknown stimulus {source!r}")
        start, stop, step = num("start"), num("stop"), num("step")
        if step == 0 or (stop - start) * step <= 0:
            raise CompileError(f"{what}: dc sweep {start} -> {stop} with step {step} never terminates")
        return f"dc {stimulus_name(stimulus)} {fmt(start, 'start')} {fmt(stop, 'stop')} {fmt(step, 'step')}"
    if spec.kind == SpiceAnalysis.TRAN:
        step, stop = num("step"), num("stop")
        if step <= 0 or stop <= 0 or stop < step:
            raise CompileError(f"{what}: tran needs 0 < step <= stop, got step={step} stop={stop}")
        cmd = f"tran {fmt(step, 'step')} {fmt(stop, 'stop')}"
        if "start" in p:
            start = num("start")
            if start < 0 or start >= stop:
                raise CompileError(f"{what}: tran start must satisfy 0 <= start < stop, got {start}")
            cmd += f" {fmt(start, 'start')}"
        if "uic" in p:
            uic = p["uic"].value
            if not isinstance(uic, bool):
                raise CompileError(f"{what} param uic must be a bool (True appends 'uic', False emits nothing), got {type(uic).__name__} {uic!r}")
            if uic:
                cmd += " uic"
        return cmd
    if spec.kind == SpiceAnalysis.AC:
        variation = p["variation"].value
        if not isinstance(variation, str) or variation.lower() not in AC_VARIATIONS:
            raise CompileError(f"{what}: ac variation must be one of {list(AC_VARIATIONS)}, got {variation!r}")
        points = p["points"].value
        if isinstance(points, bool) or not isinstance(points, int) or points < 1:
            raise CompileError(f"{what}: ac points must be a positive integer, got {points!r}")
        fstart, fstop = num("fstart"), num("fstop")
        if fstart <= 0 or fstop < fstart:
            raise CompileError(f"{what}: ac needs 0 < fstart <= fstop, got {fstart} .. {fstop}")
        if setup is not None and not any(AC_MAGNITUDE_PARAM in s.params for s in setup.stimuli):
            raise CompileError(f"{what}: an ac analysis needs a stimulus with an {AC_MAGNITUDE_PARAM!r} magnitude param; none has one")
        return f"ac {variation.lower()} {points} {fmt(fstart, 'fstart')} {fmt(fstop, 'fstop')}"
    raise CompileError(f"{what}: unsupported analysis kind {spec.kind!r}")  # pragma: no cover


def analysis_command(spec: AnalysisSpec, setup: SimulationSetup | None = None) -> str:
    """The interactive ngspice command for ``spec`` (pure; ``CompileError`` on bad/llm_generated params).

    ``setup`` resolves ``dc.source`` to the stimulus element name and checks
    an ``ac`` analysis has a stimulus with an AC magnitude.
    """
    return _analysis_command(spec, setup, _Ledger())


# --------------------------------------------------------------------------- expectations


def vector_is_complex_part(vector: str) -> bool:
    """True for ``vp()`` / ``vr()`` / ``vi()`` / ``ip()`` / ``ir()`` / ``ii()``: a part of a complex (ac) vector."""
    m = _VECTOR.match(vector)
    return m is not None and bool(m.group(2))


def spice_vector_name(vector: str, ir: CircuitIR) -> str:
    """Validate an :class:`Expectation` vector against the IR and return ngspice's vector name.

    ``v(<NET>)`` -> the net name lower-cased (the ground net is refused: it is
    node 0, identically 0 V); ``i(<STIMULUS_ID>)`` -> ``v<id>#branch``;
    ``i(<REF>)`` for a component bound as a V device -> ``<element>#branch``.
    ngspice records branch currents only for voltage sources, so anything else
    is refused rather than mapped to a vector that will not exist. ``vp`` /
    ``vr`` / ``vi`` (and ``ip`` / ``ir`` / ``ii``) append the runner's
    ``.phase_deg`` / ``.real`` / ``.imag`` suffix; they exist only in ac
    results, which :func:`build` checks against the analysis kind.
    """
    m = _VECTOR.match(vector)
    if m is None:
        raise CompileError(f"vector {vector!r} is not of the form v(<NET>), i(<STIMULUS_ID or REF>) or vp/vr/vi(...) / ip/ir/ii(...) for an ac part")
    kind, part, name = m.group(1).lower(), m.group(2).lower(), m.group(3)
    suffix = _COMPLEX_SUFFIX.get(part, "")
    if kind == "v":
        net = next((n for n in ir.nets if n.name.lower() == name.lower()), None)
        if net is None:
            raise CompileError(f"vector {vector!r}: unknown net {name!r}")
        if net.kind == NetKind.GROUND:
            raise CompileError(f"vector {vector!r}: {net.name!r} is the ground net (node 0), identically 0 V")
        return net.name.lower() + suffix
    stimuli = ir.simulation.stimuli if ir.simulation is not None else []
    stimulus = next((s for s in stimuli if s.id.lower() == name.lower()), None)
    if stimulus is not None:
        if stimulus.source != "voltage":
            raise CompileError(f"vector {vector!r}: ngspice records branch currents only for voltage sources; stimulus {stimulus.id} is a current source")
        return f"{stimulus_name(stimulus).lower()}#branch{suffix}"
    comp = next((c for c in ir.components if c.ref.lower() == name.lower()), None)
    if comp is None:
        raise CompileError(f"vector {vector!r}: {name!r} is neither a stimulus id nor a component ref")
    b = comp.spice
    if b is None or b.exclude or b.device != SpiceDevice.V:
        raise CompileError(f"vector {vector!r}: i() needs a voltage source; {comp.ref} is not in the netlist as a V device")
    return f"{element_name(comp.ref, SpiceDevice.V).lower()}#branch{suffix}"


def _is_number(value) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def _check_param_device(ref: str, dev: SpiceDevice, b) -> None:
    """A :data:`~ai_eda.ir.simulation.PARAM_DEVICES` element (the lossless ``T`` line): exactly its params, each a positive number in its unit, no value, no model."""
    if b.value is not None or b.model_name is not None or b.model_card is not None:
        raise CompileError(f"{ref}: a {dev.value} element takes its params {sorted(DEVICE_PARAMS[dev])}, not a value or a model")
    required = DEVICE_PARAMS[dev]
    missing = sorted(set(required) - set(b.params))
    extra = sorted(set(b.params) - set(required))
    if missing or extra:
        raise CompileError(
            f"{ref}: a {dev.value} element needs exactly the params {sorted(required)} (missing {missing}, not allowed {extra}); "
            "ngspice would simulate a line without td with a delay of its own choosing"
        )
    for key, unit in required.items():
        traced = b.params[key]
        if not _is_number(traced.value) or not math.isfinite(float(traced.value)) or float(traced.value) <= 0.0:
            raise CompileError(f"{ref} param {key} must be a positive finite number, got {traced.value!r}")
        if traced.unit is not None and _unit_key(traced.unit) != _unit_key(unit):
            raise CompileError(f"{ref} param {key} carries unit {traced.unit!r}, a {dev.value} element takes {key} in {unit}")


def _check_expectation(exp: Expectation, ir: CircuitIR, setup: SimulationSetup, ledger: _Ledger) -> str:
    what = f"expectation {exp.id}"
    ledger.accept_provenance(exp.provenance, what)
    analysis = setup.analysis(exp.analysis_id)
    if analysis is None:
        raise CompileError(f"{what}: unknown analysis {exp.analysis_id!r}")
    vector = spice_vector_name(exp.vector, ir)
    if vector_is_complex_part(exp.vector) and analysis.kind != SpiceAnalysis.AC:
        raise CompileError(f"{what}: {exp.vector!r} names a part of a complex vector, which only an ac analysis produces ({exp.analysis_id} is {analysis.kind.value})")
    if exp.reduce == Reduce.AT and exp.at is None:
        raise CompileError(f"{what}: reduce=at needs 'at'")
    if exp.reduce == Reduce.VALUE and analysis.kind != SpiceAnalysis.OP:
        raise CompileError(f"{what}: reduce=value is for op results; use at/final/max/min on a {analysis.kind.value} analysis")
    if exp.reduce != Reduce.VALUE and analysis.kind == SpiceAnalysis.OP:
        raise CompileError(f"{what}: an op result is a single point; use reduce=value")
    if exp.reduce == Reduce.FREQUENCY and analysis.kind != SpiceAnalysis.TRAN:
        raise CompileError(f"{what}: reduce=frequency counts rising edges over time, which only a tran analysis produces ({exp.analysis_id} is {analysis.kind.value})")
    for label, traced in (("nominal", exp.nominal), ("tol_abs", exp.tol_abs), ("tol_rel", exp.tol_rel), ("at", exp.at)):
        if traced is not None:
            ledger.accept(traced, f"{what} {label}")
            if not _is_number(traced.value) or not math.isfinite(float(traced.value)):
                raise CompileError(f"{what} {label} must be a finite number")
    if float(exp.nominal.value) == 0.0 and exp.tol_abs is None and exp.tol_rel is not None:
        raise CompileError(f"{what}: nominal is 0 and only tol_rel is given - a relative tolerance on zero is no tolerance; give tol_abs")
    if exp.bound is not None:
        if exp.tol_abs is not None or exp.tol_rel is not None:
            raise CompileError(
                f"{what}: a one-sided bound ({exp.bound}) takes no tolerance - PASS is the measured value on the passing side of nominal; "
                "give either tol_abs / tol_rel or the bound"
            )
        if exp.requirement_id is not None:
            raise CompileError(
                f"{what}: a one-sided bound cannot claim a requirement yet (requirement_id {exp.requirement_id!r}): "
                "the reviewer's nominal-vs-requirement rule and the tolerance chart know only tolerances"
            )
    _check_reduction(exp, ir, analysis, what, ledger)
    return vector


def _vector_unit(vector: str) -> str:
    """``V`` for a ``v(...)`` vector, ``A`` for an ``i(...)`` one (the grammar is checked before this is asked)."""
    return "A" if vector.strip()[:1].lower() == "i" else "V"


def _check_reduction(exp: Expectation, ir: CircuitIR, analysis: AnalysisSpec, what: str, ledger: _Ledger) -> None:
    """The params, reference and window rules of the level / window reductions (``db_at`` ... ``am_depth``).

    Every other reduction takes no params and no ``reference_vector``. Each
    param is a finite number accepted through the ledger (``llm_generated`` is
    refused); a unit, when the param states one, must be the one
    :data:`~ai_eda.ir.simulation.REDUCE_PARAM_UNITS` names. The window of a
    tran reduction lies inside the analysis' saved window; an RMS needs
    ``f_max > 0``; a harmonic's window holds at least two whole periods of
    ``f0``; an AM depth needs ``f_carrier >= 20 f_mod``
    and a window of at least one modulation period; a dB level needs
    ``tol_abs`` and no ``tol_rel`` (a relative tolerance on a logarithm is no
    tolerance) unless it is a one-sided ``bound``, and a stated
    ``nominal.unit`` must be the reduction's unit.
    """
    kind = exp.reduce
    required, optional = REDUCE_PARAMS[kind]
    allowed = (*required, *optional)
    if not allowed and exp.params:
        raise CompileError(f"{what}: reduce={kind.value} takes no params, got {sorted(exp.params)}")
    extra = sorted(set(exp.params) - set(allowed))
    if extra:
        raise CompileError(f"{what}: unexpected params {extra} for reduce={kind.value} (it takes {list(allowed)})")
    missing = [k for k in required if k not in exp.params]
    if missing:
        raise CompileError(f"{what}: reduce={kind.value} needs params {list(required)}, missing {missing}")
    if kind not in REFERENCE_REDUCES and exp.reference_vector is not None:
        raise CompileError(f"{what}: reduce={kind.value} takes no reference_vector")
    values: dict[str, float] = {}
    for key in sorted(exp.params):
        traced = ledger.accept(exp.params[key], f"{what} param {key}")
        if not _is_number(traced.value) or not math.isfinite(float(traced.value)):
            raise CompileError(f"{what} param {key} must be a finite number, got {traced.value!r}")
        want = _vector_unit(exp.vector) if key == "ref" else REDUCE_PARAM_UNITS.get(key)
        if traced.unit is not None and _unit_key(traced.unit) != _unit_key(want):
            raise CompileError(f"{what} param {key} carries unit {traced.unit!r}, reduce={kind.value} takes {key} " + (f"in {want}" if want else "as a plain number"))
        values[key] = float(traced.value)
    if kind == Reduce.DB_AT:
        if analysis.kind != SpiceAnalysis.AC:
            raise CompileError(f"{what}: reduce=db_at reads a magnitude at one frequency, which only an ac analysis produces ({exp.analysis_id} is {analysis.kind.value})")
        if exp.at is None:
            raise CompileError(f"{what}: reduce=db_at needs 'at'")
        if vector_is_complex_part(exp.vector):
            raise CompileError(f"{what}: reduce=db_at needs a magnitude vector (v(...) / i(...)), got {exp.vector!r}: 20 log10 of a phase or of a real / imaginary part is not a level")
    if kind in WINDOW_REDUCES and analysis.kind != SpiceAnalysis.TRAN:
        raise CompileError(f"{what}: reduce={kind.value} reads a time window, which only a tran analysis produces ({exp.analysis_id} is {analysis.kind.value})")
    if kind in REFERENCE_REDUCES:
        if (exp.reference_vector is not None) == ("ref" in values):
            raise CompileError(f"{what}: reduce={kind.value} needs exactly one of reference_vector or params['ref'] (the level the vector is compared with)")
        if "ref" in values and values["ref"] <= 0.0:
            raise CompileError(f"{what} param ref must be > 0 (a level to divide by), got {values['ref']!r}")
        if exp.reference_vector is not None:
            spice_vector_name(exp.reference_vector, ir)  # the same grammar and IR checks as ``vector``
            if vector_is_complex_part(exp.reference_vector):
                raise CompileError(f"{what}: reference_vector {exp.reference_vector!r} names a part of a complex vector; a reference level is a magnitude (v(...) / i(...))")
            if _vector_unit(exp.reference_vector) != _vector_unit(exp.vector):
                raise CompileError(f"{what}: reference_vector {exp.reference_vector!r} and vector {exp.vector!r} are not the same kind of quantity (a voltage ratio or a current ratio)")
    if kind in WINDOW_REDUCES:
        t0, t1 = values["t_start"], values["t_stop"]
        start = float(analysis.params["start"].value) if "start" in analysis.params else 0.0
        stop = float(analysis.params["stop"].value)
        if not (0.0 <= t0 < t1) or t0 < start or t1 > stop:
            raise CompileError(f"{what}: the window [{t0:g}, {t1:g}] s must lie inside the tran window [{start:g}, {stop:g}] s of {exp.analysis_id} with 0 <= t_start < t_stop")
        step = float(analysis.params["step"].value)
        if start > 0.0 and t0 < start + step:
            # measured on ngspice-42: `tran 1u 5m 1m` saves its first point at 1.00028 ms, `tran 10u 5m 2m` at 2.0028 ms (a
            # breakpoint at the start is saved exactly): the first sample lies within one step after `start`, never before
            raise CompileError(
                f"{what}: t_start {t0:g} s must be at least one tran step after the analysis' start ({start:g} s + {step:g} s): "
                "ngspice saves its first point after start, up to one step later, and a window is never extrapolated"
            )
        if kind in (Reduce.RMS, Reduce.DB_RMS) and values["f_max"] <= 0.0:
            raise CompileError(f"{what} param f_max must be > 0 (the highest frequency the RMS must include; the grid guard is 1 / (20 f_max)), got {values['f_max']!r}")
        if kind == Reduce.HARMONIC_DBC:
            k = exp.params["k"].value
            if isinstance(k, bool) or float(k) != int(k) or int(k) < 2:
                raise CompileError(f"{what}: k must be an integer >= 2 (the harmonic's number; 1 is the fundamental itself), got {k!r}")
            f0 = values["f0"]
            if f0 <= 0.0:
                raise CompileError(f"{what} param f0 must be > 0, got {f0!r}")
            periods = (t1 - t0) * f0
            if round(periods) < 1 or abs(periods - round(periods)) > PERIOD_REL_TOL * max(1.0, periods):
                raise CompileError(f"{what}: the window must hold whole periods of f0 = {f0:g} Hz: (t_stop - t_start) f0 = {periods:.9g}")
            if round(periods) < MIN_HARMONIC_PERIODS:
                raise CompileError(f"{what}: the window must hold at least {MIN_HARMONIC_PERIODS} whole periods of f0 = {f0:g} Hz "
                                   "(the samples must show that the waveform repeats at f0), got 1")
        if kind == Reduce.AM_DEPTH:
            fc, fm = values["f_carrier"], values["f_mod"]
            if not fc > fm > 0.0:
                raise CompileError(f"{what}: reduce=am_depth needs f_carrier > f_mod > 0, got f_carrier={fc:g} f_mod={fm:g}")
            if fc < MIN_CARRIER_TO_MOD_RATIO * fm:
                raise CompileError(
                    f"{what}: carrier too slow for the envelope: f_carrier / f_mod = {fc / fm:.6g} < {MIN_CARRIER_TO_MOD_RATIO} "
                    "(each carrier period's amplitude averages the envelope over that period)"
                )
            if (t1 - t0) * fm < 1.0 - PERIOD_REL_TOL:
                raise CompileError(f"{what}: window shorter than one modulation period ({t1 - t0:g} s < 1 / f_mod = {1.0 / fm:g} s)")
    if kind in DB_REDUCES and exp.bound is None:  # a one-sided bound replaces the tolerance (checked in _check_expectation)
        if exp.tol_abs is None or exp.tol_rel is not None:
            raise CompileError(f"{what}: a level in dB needs tol_abs (and no tol_rel) or a one-sided bound: a relative tolerance on a logarithm is no tolerance")
    unit = REDUCE_UNITS.get(kind) or (_vector_unit(exp.vector) if kind == Reduce.RMS else None)
    if unit is not None and exp.nominal.unit is not None:
        same = {_unit_key(unit)} | ({"%"} if unit == "percent" else set())
        if _unit_key(exp.nominal.unit) not in same:
            raise CompileError(f"{what}: nominal unit {exp.nominal.unit!r} is not the unit of reduce={kind.value} ({unit})")


# --------------------------------------------------------------------------- the compile


def _unit_key(unit: str | None) -> str | None:
    return None if unit is None else unit.strip().lower().replace("ω", "ohm").replace("ohms", "ohm")


def _reconcile_value(c: Component, value: Traced, dev: SpiceDevice) -> str:
    """Where ``value`` was checked against: the component's electrical entry of the same quantity, or nothing.

    ``CompileError`` when the component records a resistance / capacitance /
    inductance and the binding would simulate a different number (or a
    different unit): the netlist must describe the part the BOM ships.
    """
    key = ELECTRICAL_KEYS.get(dev)
    electrical = c.electrical.get(key) if key else None
    if electrical is None:
        return "spice binding only (component records no " + (key or "electrical value") + ")"
    if not _is_number(electrical.value):
        raise CompileError(f"{c.ref}: electrical[{key!r}] is not a number ({electrical.value!r}); cannot reconcile it with the SPICE value")
    if value.unit is not None and electrical.unit is not None and _unit_key(value.unit) != _unit_key(electrical.unit):
        raise CompileError(
            f"{c.ref}: SPICE value unit {value.unit!r} differs from electrical[{key!r}] unit {electrical.unit!r}; the two numbers cannot be compared"
        )
    if float(value.value) != float(electrical.value):
        raise CompileError(
            f"{c.ref}: SPICE value {value.value!r} differs from electrical[{key!r}] = {electrical.value!r} "
            f"(the netlist would simulate a different part than the BOM ships; update the binding or the component, not one of them)"
        )
    return f"electrical.{key}"


@dataclass
class _Compiled:
    text: str
    report: dict


def _compile(ir: CircuitIR) -> _Compiled:
    if not ir.components:
        raise NothingToCompileError("no components: nothing to simulate")
    title = ir.project.id
    if not title.strip() or any(ch in title for ch in _BAD_TITLE_CHARS):
        raise CompileError(f"project.id {title!r} cannot be the netlist title line (empty, or contains one of $ ' \" ` or a newline)")
    if not title.isascii():
        raise CompileError(f"project.id {title!r} must be ASCII: it is the SPICE netlist's title line and ngspice rewrites non-ASCII text (create the project with an ASCII id)")
    if not any(c.spice is not None for c in ir.components):
        first = min(c.ref for c in ir.components)
        raise NothingToCompileError(f"no SPICE binding for {first} (no component has one: the SPICE model mapping has not been made yet)")
    nodes = _nodes(ir)
    ledger = _Ledger()
    claimed: dict[str, str] = {}

    def claim(name: str, what: str) -> None:
        if not _NAME.match(name):
            raise CompileError(f"{what}: element name {name!r} is not a plain identifier")
        if name.lower() in claimed:
            raise CompileError(f"{what}: element name {name!r} collides with {claimed[name.lower()]} (ngspice is case-insensitive)")
        claimed[name.lower()] = what

    excluded: list[dict[str, str]] = []
    excluded_refs: set[str] = set()
    ignored_pins: dict[str, dict[str, str]] = {}  # ref -> {pin: reason} for connected pins the element does not use
    value_sources: dict[str, str] = {}  # element -> where its value was reconciled
    element_lines: list[str] = []
    element_names: list[str] = []
    cards: dict[str, str] = {}  # model name (lower) -> normalised card text
    defined: dict[str, tuple[str, list[str], str]] = {}  # name (lower) -> (kind, ports, card key)
    referenced: list[tuple[str, SpiceDevice, str, int]] = []  # (ref, device, model_name, node count)

    for c in sorted(ir.components, key=lambda c: c.ref):
        b = c.spice
        if b is None:
            raise CompileError(f"no SPICE binding for {c.ref}")
        ledger.accept_provenance(b.provenance, f"{c.ref} SPICE binding")
        if b.exclude:
            excluded.append({"ref": c.ref, "reason": b.exclude_reason})
            excluded_refs.add(c.ref)
            continue
        if b.device is None:
            raise CompileError(f"{c.ref}: SPICE binding has no device and is not excluded")
        dev = b.device
        pins = [p.number for p in c.pins]
        if not b.pin_order:
            raise CompileError(f"{c.ref}: pin_order is empty")
        if len(set(b.pin_order)) != len(b.pin_order):
            raise CompileError(f"{c.ref}: pin_order {b.pin_order} repeats a pin")
        unknown = [p for p in b.pin_order if p not in pins]
        if unknown:
            raise CompileError(f"{c.ref}: pin_order {b.pin_order} names pins {unknown} that {c.ref} does not have (pins: {pins})")
        unknown = [p for p in b.ignored_pins if p not in pins]
        if unknown:
            raise CompileError(f"{c.ref}: ignored_pins names pins {unknown} that {c.ref} does not have (pins: {pins})")
        both = [p for p in b.ignored_pins if p in b.pin_order]
        if both:
            raise CompileError(f"{c.ref}: pins {both} are in pin_order and in ignored_pins")
        counts = NODE_COUNTS[dev]
        if counts is not None and len(b.pin_order) not in counts:
            raise CompileError(f"{c.ref}: a {dev.value} element takes {' or '.join(map(str, counts))} nodes, pin_order has {len(b.pin_order)}")
        node_list: list[str] = []
        for p in b.pin_order:
            node = nodes.of_pin.get((c.ref, p))
            if node is None:
                raise CompileError(f"{c.ref}.{p} is used by the SPICE element but is in no net")
            node_list.append(node)
        for ir_pin in c.pins:
            if ir_pin.number in b.pin_order:
                continue
            connected = (c.ref, ir_pin.number) in nodes.of_pin
            if ir_pin.number in b.ignored_pins and not connected:
                raise CompileError(f"{c.ref}.{ir_pin.number} is listed in ignored_pins but is in no net; ignored_pins is for a connected pin the element does not use - connect it or mark it no_connect")

            if ir_pin.number in b.ignored_pins:
                if not b.ignored_pins[ir_pin.number].strip():
                    raise CompileError(f"{c.ref}.{ir_pin.number} is in ignored_pins without a reason")
                ignored_pins.setdefault(c.ref, {})[ir_pin.number] = b.ignored_pins[ir_pin.number]
                continue
            if connected:
                raise CompileError(
                    f"{c.ref}.{ir_pin.number} is in a net but the SPICE binding does not use it (pin_order {b.pin_order}): "
                    "the schematic would connect it and the netlist would not; list it in ignored_pins with a reason"
                )
            if ir_pin.electrical_type is PinElectricalType.NO_CONNECT:
                continue
            raise CompileError(
                f"{c.ref}.{ir_pin.number} ({ir_pin.electrical_type.value}) is in no net; connect it or mark it no_connect in the IR"
            )
        name = element_name(c.ref, dev)
        claim(name, f"component {c.ref}")
        rest: list[str] = []
        if dev in PARAM_DEVICES:
            _check_param_device(c.ref, dev, b)
        elif dev in VALUE_DEVICES:
            if b.value is None and b.model_name is None:
                raise CompileError(f"{c.ref}: a {dev.value} element needs a value (or a model_name)")
            if b.value is not None:
                if dev in (SpiceDevice.R, SpiceDevice.C, SpiceDevice.L) and _is_number(b.value.value) and float(b.value.value) <= 0.0:
                    # ngspice simulates R=0 as ~1 mΩ and a negative passive without a word (measured on ngspice-42);
                    # a 0 V / negative V or I source is an ordinary source (a current probe, a negative rail)
                    raise CompileError(f"{c.ref}: a {dev.value} value must be positive; got {b.value.value!r} (ngspice would silently simulate a different part)")
                rest.append(_number(ledger.accept(b.value, f"{c.ref} value"), f"{c.ref} value", ledger))
                value_sources[name] = _reconcile_value(c, b.value, dev)
        else:
            if b.value is not None:
                raise CompileError(f"{c.ref}: a {dev.value} element takes a model, not a value")
            if not b.model_name:
                raise CompileError(f"{c.ref}: a {dev.value} element needs a model_name")
        if b.model_name is not None:
            if not b.model_name or any(ch in _BAD_NODE_CHARS for ch in b.model_name):
                raise CompileError(f"{c.ref}: model_name {b.model_name!r} is not a single token")
            rest.append(b.model_name)
            referenced.append((c.ref, dev, b.model_name, len(b.pin_order)))
        rest += _param_tokens(b.params, c.ref, ledger)
        if b.model_card is not None:
            if not b.model_name:
                raise CompileError(f"{c.ref}: model_card given without model_name")
            ledger.accept(b.model_card, f"{c.ref} model_card", model_card=True)
            card = _normalise_card(b.model_card.value, c.ref)
            defs = _card_definitions(card)
            key = b.model_name.lower()
            if key not in defs:
                raise CompileError(f"{c.ref}: model_card does not define {b.model_name!r} (it defines {sorted(defs) or 'nothing'})")
            if key in cards and cards[key] != card:
                raise CompileError(f"model {b.model_name!r} has two different model cards in the IR ({c.ref} disagrees with an earlier component)")
            if key not in cards:
                for dname, (dkind, ports) in defs.items():
                    if dname in defined and defined[dname][2] != key:
                        raise CompileError(f"{c.ref}: model_card defines {dname!r} which the card for {defined[dname][2]!r} also defines")
                    defined[dname] = (dkind, ports, key)
                cards[key] = card
        element_lines.append(" ".join([name, *node_list, *rest]))
        element_names.append(name)

    for ref, dev, model_name, n_nodes in referenced:
        entry = defined.get(model_name.lower())
        if entry is None:
            raise CompileError(f"{ref}: model {model_name!r} is not defined by any model_card in the IR (the netlist emits no .include)")
        dkind, ports, _ = entry
        if dev == SpiceDevice.X and dkind != "subckt":
            raise CompileError(f"{ref}: an X element needs a .subckt, but {model_name!r} is a .model")
        if dev != SpiceDevice.X and dkind != "model":
            raise CompileError(f"{ref}: a {dev.value} element needs a .model, but {model_name!r} is a .subckt")
        if dev == SpiceDevice.X and len(ports) != n_nodes:
            raise CompileError(f"{ref}: .subckt {model_name} has {len(ports)} ports {ports} but pin_order has {n_nodes} pins")

    setup = ir.simulation
    stimulus_lines: list[str] = []
    stimulus_ids: list[str] = []
    commands: dict[str, str] = {}
    vectors: dict[str, str] = {}
    temperature: str | None = None
    if setup is not None:
        for s in sorted(setup.stimuli, key=lambda s: s.id):
            what = f"stimulus {s.id}"
            ledger.accept_provenance(s.provenance, what)
            if not _NAME.match(s.id):
                raise CompileError(f"{what}: id must be a plain identifier (it becomes an element name)")
            for net_name in (s.net, s.reference_net):
                if net_name not in nodes.of_net:
                    raise CompileError(f"{what}: unknown net {net_name!r}")
            if s.net == s.reference_net:
                raise CompileError(f"{what}: net and reference_net are both {s.net!r}")
            name = stimulus_name(s)
            claim(name, what)
            stimulus_lines.append(f"{name} {nodes.of_net[s.net]} {nodes.of_net[s.reference_net]} {_stimulus_spec(s, ledger)}")
            stimulus_ids.append(s.id)
        seen_ids: set[str] = set()
        for a in setup.analyses:
            ledger.accept_provenance(a.provenance, f"analysis {a.id}")
            if not _NAME.match(a.id):
                raise CompileError(f"analysis id {a.id!r} must be a plain identifier (it names the run directory and the rawfile)")
            if a.id in seen_ids:
                raise CompileError(f"analysis id {a.id!r} is used twice")
            seen_ids.add(a.id)
            commands[a.id] = _analysis_command(a, setup, ledger)
        seen_ids = set()
        for e in setup.expectations:
            if not _NAME.match(e.id):
                raise CompileError(f"expectation id {e.id!r} must be a plain identifier (it becomes the check id spice.{e.id})")
            if e.id in seen_ids:
                raise CompileError(f"expectation id {e.id!r} is used twice")
            seen_ids.add(e.id)
            vectors[e.id] = _check_expectation(e, ir, setup, ledger)
        if setup.temperature_c is not None:
            temperature = _number(ledger.accept(setup.temperature_c, "temperature_c"), "temperature_c", ledger)

    lines = [ir.project.id]
    if temperature is not None:
        lines.append(f".temp {temperature}")
    for key in sorted(cards):
        lines.extend(cards[key].split("\n"))
    lines += element_lines
    lines += stimulus_lines
    lines.append(".end")
    text = "\n".join(lines) + "\n"
    # The runner validates every deck before ngspice.dll sees it (quotes, non-ASCII, node characters,
    # analysis cards, no non-ground node ...). Apply the same judge here so a compiled netlist is never
    # refused later: a refusal is a CompileError with the runner's own reasons.
    problems, _ = validate_deck(text)
    if problems:
        raise CompileError("netlist would be rejected by the SPICE runner: " + "; ".join(problems))

    ref_of = {c.ref: c for c in ir.components}

    def has_no_node(p) -> bool:
        return p.component_ref in excluded_refs or p.pin_number in ignored_pins.get(p.component_ref, {})

    # a net touching only excluded parts and ignored pins has no node in the netlist: the schematic
    # connects it, ngspice never sees it
    only_excluded = [n.name for n in ir.nets if n.pins and all(has_no_node(p) for p in n.pins)]
    stimulated = {s.net for s in setup.stimuli} | {s.reference_net for s in setup.stimuli} if setup is not None else set()
    only_excluded = [n for n in only_excluded if n not in stimulated]
    report = {
        "title": ir.project.id,
        "elements": element_names,
        "excluded": excluded,
        "ignored_pins": ignored_pins,
        "nets_touching_only_excluded": only_excluded,
        "nets_without_pins": [n.name for n in ir.nets if not n.pins],
        "ground_net": nodes.ground.name,
        "models": sorted(cards),
        "value_sources": value_sources,
        "accepted_provenance_kinds": sorted(k.value for k in ledger.kinds),
        "assumptions": list(ledger.assumptions),
        "inexact_numbers": list(ledger.inexact),
        "unmodelled_numbers": list(ledger.unmodelled),
        "stimuli": stimulus_ids,
        "analyses": commands,
        "vectors": vectors,
        "temperature_c": None if setup is None or setup.temperature_c is None else setup.temperature_c.value,
        "component_count": len(ref_of),
    }
    return _Compiled(text=text, report=report)


def build(ir: CircuitIR) -> str:
    """The netlist text for ``ir`` (pure, deterministic; ``CompileError`` rather than a guess)."""
    return _compile(ir).text


def build_report(ir: CircuitIR) -> dict:
    """What the compile accepted and left out: excluded refs (+reasons), nets touching only excluded
    parts, accepted provenance kinds, assumptions, numbers ngspice reads inexactly, analysis commands
    and expectation vector names."""
    return _compile(ir).report


def netlist_elements(text: str) -> list[tuple[str, list[str], str]]:
    """Parse a netlist in this compiler's layout into ``(name, nodes, rest)`` per element line.

    The title line, dot cards, comments and the inside of ``.subckt`` blocks
    are skipped; ``+`` continuations are joined. Two-terminal letters
    (R C L V I D) take two nodes, a lossless line (T) four (its ``rest`` is
    the ``td=... z0=...`` params); Q, M and X take every token up to the model
    / subcircuit name (trailing ``k=v`` params excluded).
    """
    lines = _join_continuations(text.splitlines())
    out: list[tuple[str, list[str], str]] = []
    depth = 0
    for i, line in enumerate(lines):
        s = line.strip()
        if i == 0 or not s or s[0] == "*":
            continue
        first = s.split()[0].lower()
        if first == ".subckt":
            depth += 1
            continue
        if first == ".ends":
            depth = max(0, depth - 1)
            continue
        if depth > 0 or s[0] == ".":
            continue
        toks = s.split()
        letter = toks[0][0].upper()
        if letter in "RCLVID":
            if len(toks) < 4:
                raise ValueError(f"malformed element line {s!r}")
            out.append((toks[0], toks[1:3], " ".join(toks[3:])))
        elif letter == "T":
            if len(toks) < 5:
                raise ValueError(f"malformed element line {s!r}")
            out.append((toks[0], toks[1:5], " ".join(toks[5:])))
        elif letter in "QMX":
            body = toks[1:]
            while body and "=" in body[-1]:
                body.pop()
            if len(body) < 2:
                raise ValueError(f"malformed element line {s!r}")
            n_nodes = len(body) - 1
            out.append((toks[0], body[:n_nodes], " ".join(toks[1 + n_nodes:])))
        else:
            raise ValueError(f"unknown element letter in {s!r}")
    return out


class SpiceNetlistCompiler(Compiler):
    """Writes ``<workdir>/<project.id>.cir`` (+ ``.cir.report.json``) and registers it as ``SPICE_NETLIST``."""

    id = "compiler.spice"
    #: 0.6: AM / SFFM stimuli and the level / window reductions' rules (an IR 0.5 compiled writes the same bytes)
    version = "0.6"
    kind = ArtifactKind.SPICE_NETLIST

    build = staticmethod(build)
    build_report = staticmethod(build_report)
    analysis_command = staticmethod(analysis_command)

    def compile(self, ir: CircuitIR, ctx: CompileContext) -> ArtifactRef:
        compiled = _compile(ir)
        path = Path(ctx.workdir) / f"{ir.project.id}.cir"
        ref = self._write(ir, path, compiled.text)
        report = {"generator": self.id, "generator_version": self.version, "netlist": path.name, **compiled.report}
        path.with_name(path.name + ".report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )
        return ref


__all__ = [
    "SpiceNetlistCompiler",
    "analysis_command",
    "build",
    "build_report",
    "element_name",
    "netlist_elements",
    "spice_vector_name",
    "stimulus_name",
    "vector_is_complex_part",
]
