"""The four verified circuit templates and the closed-world selection between them.

Invariant: a template is selected only by *confirmed* requirement values
(:mod:`ai_eda.design.inputs`), computes every number with a registered
calculator (:mod:`ai_eda.tools.calc.basic`, so ``calc.recompute`` re-derives
it), takes its parts from the KiCad library on disk
(:mod:`ai_eda.design.library_parts`) and presents its free choices for
confirmation (:mod:`ai_eda.design.base`). Selection is closed-world: a
template declares the keys it serves and may ignore; any other confirmed
design-category requirement refuses it with a note and a non-required
question under that requirement's own key - the agent never invents a key.
Two templates triggered at once is *ambiguous* and refused. A template
never authors a requirement: what it verifies is tied to the user's own
requirement (the RC's |H(f_c)| expectation names ``cutoff_frequency`` and the
reviewer compares the sweep point with it). A calculator that refuses its
inputs, overflows (``ValueError``) or divides by an underflowed product
(``ZeroDivisionError``) refuses the template with that sentence as the note -
a build never raises on a number the user typed.

Templates:

* ``divider`` - unloaded resistive divider from ``input_voltage`` and
  ``output_voltage`` (R2 chosen, R1 computed; ``output_current`` is served
  only when it is 0 A - a load refuses the template, no verified template
  supplies one). Its advisory load question is asked with the confirmation
  table, so a ``0 A`` answer can arrive before the build; one stated after
  the build is served by :func:`late_load_changes` (VOUT is proposed to
  serve it - a proposal the agent applies, never a silent edit) and a
  non-zero one is named as unservable;
* ``led`` - LED with series resistor from ``input_voltage``,
  ``led_forward_voltage`` and ``led_forward_current``; the LED is modelled
  as an ideal constant forward drop (a stimulus ``VLED`` = v_f, ``D1``
  excluded from the netlist - a modelling choice the user confirms) and the
  expectation measures the current through that ideal source;
* ``rc_lowpass`` - first-order RC low-pass from ``cutoff_frequency`` (C
  chosen, R computed), checked by an ac sweep at the corner;
* ``astable`` - collector-coupled BJT astable multivibrator from
  ``oscillation_frequency`` and ``input_voltage`` (R_c, R_b, V_BE and a
  generic NPN model chosen, C computed), checked by a transient run
  (``tran ... uic``, one capacitor's ``ic`` breaks the ideal circuit's
  symmetry) whose output frequency is *measured* from the rising edges of
  v(OUT) (``Reduce.FREQUENCY``) and whose swing is checked against the
  supply; the supply (3..6 V, the reverse base-emitter rating of small
  NPNs) and the frequency (100 Hz..20 kHz, non-polar timing capacitors and
  the model's missing switching times) are validity conditions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ai_eda.ir import (
    AnalysisSpec,
    Block,
    CircuitDomain,
    CircuitIR,
    Component,
    Constraint,
    ConstraintKind,
    Expectation,
    MissingInformation,
    Net,
    NetKind,
    PinRef,
    Provenance,
    ProvenanceKind,
    Reduce,
    SimulationSetup,
    SpiceBinding,
    SpiceDevice,
    Stimulus,
    StimulusKind,
    Topology,
    Traced,
)
from ai_eda.tools.calc.basic import (
    astable_c_for_frequency,
    astable_frequency,
    astable_tran_start,
    astable_tran_step,
    astable_tran_stop,
    astable_v_be_reverse,
    divider_r1_for_v_out,
    led_current,
    led_series_resistor,
    rc_ac_fstart,
    rc_ac_fstop,
    rc_lowpass_magnitude,
    rc_r_for_cutoff,
    rc_time_constant,
    voltage_divider_output,
)
from ai_eda.tools.calc.si import format_spice_number
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.spice import SpiceAnalysis

from ai_eda.design.base import (
    CHOICE_NOTE_PREFIX,
    NO_RECORD,
    TEMPLATE_VERSION,
    Choice,
    DesignChange,
    PartNote,
    Plan,
    Template,
    TheorySection,
    choice_provenance,
    number,
    parameter_value,
    quantity,
    structural_provenance,
    template_tool,
    unserved_requirements,
    unverified,
)
from ai_eda.design.inputs import KEY_ALIASES, UNIT_OF, DesignInput, read_inputs, read_value
from ai_eda.design.library_parts import TemplateRefusal, library_component, pin_by_name, require_pins, two_terminals

RESISTOR = (("Device", "R"), ("Resistor_SMD", "R_0603_1608Metric"))
CAPACITOR = (("Device", "C"), ("Capacitor_SMD", "C_0603_1608Metric"))
LED = (("Device", "LED"), ("LED_SMD", "LED_0603_1608Metric"))
HEADER_2 = (("Connector_Generic", "Conn_01x02"), ("Connector_PinHeader_2.54mm", "PinHeader_1x02_P2.54mm_Vertical"))
HEADER_3 = (("Connector_Generic", "Conn_01x03"), ("Connector_PinHeader_2.54mm", "PinHeader_1x03_P2.54mm_Vertical"))
NPN = (("Transistor_BJT", "2N3904"), ("Package_TO_SOT_THT", "TO-92_Inline"))
RESISTOR_THT = (("Device", "R"), ("Resistor_THT", "R_Axial_DIN0207_L6.3mm_D2.5mm_P7.62mm_Horizontal"))
CAPACITOR_THT = (("Device", "C"), ("Capacitor_THT", "C_Disc_D5.0mm_W2.5mm_P5.00mm"))


def _value_text(value: float) -> str:
    """The part's ``value`` string: the same spelling the netlist carries, so BOM and SPICE agree."""
    return format_spice_number(value)


def _part_line(c: Component) -> str:
    assert c.symbol is not None and c.footprint is not None
    where = f" (pins {', '.join(p.number for p in c.pins)} from {c.symbol.library_path})" if c.symbol.library_path else ""
    return f"{c.ref} {c.symbol.library}:{c.symbol.name} / {c.footprint.library}:{c.footprint.name}, value {c.value}{where}"


def _net(name: str, kind: NetKind, pins: list[tuple[str, str]], template: str, serves: list[str] | None = None) -> Net:
    return Net(name=name, kind=kind, pins=[PinRef(component_ref=r, pin_number=p) for r, p in pins], provenance=structural_provenance(template, f"{name} net"), serves_requirements=list(serves or []))


def _net_line(n: Net) -> str:
    return f"{n.name} ({n.kind.value}): {', '.join(f'{p.component_ref}.{p.pin_number}' for p in n.pins)}"


def _changes(template: str, title: str, topology: Topology, components: list[Component], nets: list[Net], params: dict[str, Traced], sim: SimulationSetup, constraints: list[Constraint]) -> list[DesignChange]:
    """Deterministic order: topology, components, nets, parameters (insertion order), simulation, constraints."""
    out = [DesignChange(description=f"topology: {title}", target="topology", operation="set", payload=topology, rationale=f"template {template} v{TEMPLATE_VERSION}")]
    out += [DesignChange(description=f"add {c.ref} ({c.description})", target="components", operation="append", payload=c) for c in components]
    out += [DesignChange(description=f"add net {n.name}", target="nets", operation="append", payload=n) for n in nets]
    out += [DesignChange(description=f"parameter {k}", target=f"parameters.{k}", operation="set", payload=v) for k, v in params.items()]
    out.append(DesignChange(description="simulation setup", target="simulation", operation="set", payload=sim))
    out += [DesignChange(description=f"constraint {c.id}", target="constraints", operation="append", payload=c) for c in constraints]
    return out


def _choice(template: str, key: str, value, unit: str | None, description: str, confirmed: bool) -> tuple[Choice, Traced]:
    return Choice(key, description, value, unit), Traced(value=value, unit=unit, provenance=choice_provenance(template, f"{key} = {value!r}{' ' + unit if unit else ''}: {description}", confirmed))


def _refused(plan: Plan, why: str) -> Plan:
    plan.notes.append(f"template {plan.template} not proposed: {why}")
    return plan


def _known(*values: float | None) -> bool:
    return all(v is not None for v in values)


def _div(a: float | None, b: float | None) -> float | None:
    """``a / b`` for the display numbers of a report; ``None`` when either is unknown or ``b`` is zero."""
    if a is None or b is None or b == 0:
        return None
    return a / b


def _mul(*values: float | None) -> float | None:
    out = 1.0
    for v in values:
        if v is None:
            return None
        out *= v
    return out


def _sub(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a - b


def _add(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a + b


def _model_card_text(ir: CircuitIR) -> str:
    """The first ``.model`` card the IR's SPICE bindings carry, as written; :data:`NO_RECORD` without one."""
    for c in ir.components:
        if c.spice is not None and c.spice.model_card is not None:
            return str(c.spice.model_card.value)
    return NO_RECORD


def _judging_line(nominal: float | None, unit: str | None, tol_rel: float | None = None, tol_abs: float | None = None) -> str:
    """The pass rule |measured − nominal| ≤ max(tol_abs, tol_rel·|nominal|) with this design's numbers."""
    if tol_rel is not None and nominal is not None:
        return f"|측정값 − {quantity(nominal, unit)}| ≤ {number(tol_rel * 100, 4)} % × {quantity(nominal, unit)} = {quantity(tol_rel * abs(nominal), unit)}"
    if tol_abs is not None:
        return f"|측정값 − {quantity(nominal, unit)}| ≤ {quantity(tol_abs, unit)}"
    return f"|측정값 − {quantity(nominal, unit)}| ≤ 허용치 ({NO_RECORD})"


#: the E12 series mantissas (IEC 60063)
E12_MANTISSAS: tuple[float, ...] = (1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2)


def nearest_e12(value: float | None) -> float | None:
    """The E12 series value nearest to ``value`` by ratio (so 9.5 rounds up to 10, not down to 8.2); ``None`` for an unknown or non-positive value.

    A display helper for the theory text: the IR keeps the calculator's
    exact value, the report only says what the nearest standard value would
    do to the formula.
    """
    if value is None or not math.isfinite(value) or value <= 0:
        return None
    exp = math.floor(math.log10(value))
    candidates = [m * 10.0 ** e for e in (exp - 1, exp, exp + 1) for m in E12_MANTISSAS]
    return min(candidates, key=lambda c: abs(math.log(c / value)))


def _is_choice(ir: CircuitIR, key: str) -> bool:
    """Whether ``ir.parameters[key]`` is a template choice the user confirmed (:func:`~ai_eda.design.base.choice_provenance`), not a calculator output."""
    t = ir.parameters.get(key)
    return t is not None and t.provenance.kind is ProvenanceKind.USER_REQUIREMENT and (t.provenance.note or "").startswith(CHOICE_NOTE_PREFIX)


def _resistor_criteria(value: float | None, dissipation: float | None, tolerance: str, *, chosen: bool = False) -> list[str]:
    """What any substitute resistor must satisfy: the value, twice the computed dissipation, the tolerance class.

    ``chosen`` says the value is a template choice the user confirmed (the
    parameter's provenance, see :func:`_is_choice`); otherwise it is a
    calculator output printed unrounded, and the line says which.
    """
    origin = "템플릿 선택값, 사용자 확인" if chosen else "계산값 그대로: E 계열 반올림은 하지 않았음"
    return [
        f"저항값 {quantity(value, 'ohm')} ({origin})",
        f"정격 전력 ≥ 2 × 계산 소비전력 = 2 × {quantity(dissipation, 'W')} = {quantity(_mul(2.0, dissipation), 'W')}",
        f"공차 {tolerance}",
    ]


def _resistor_substitutes(value: float | None, footprint: str) -> list[str]:
    v = quantity(value, "ohm")
    return [
        unverified(f"같은 값 {v}의 1/4 W 축형(axial) 저항", f"현재 풋프린트 {footprint}" if "THT" in footprint else "풋프린트를 Resistor_THT 로 바꾸어야 함"),
        unverified(f"같은 값 {v}의 0603 / 0805 SMD 저항", "현재 풋프린트 그대로" if "SMD" in footprint else "풋프린트를 Resistor_SMD 로 바꾸어야 함"),
    ]


def astable_drawing(r_c: str, r_b: str) -> str:
    """The collector-coupled astable as fixed-column ASCII art, wired as the template builds its nets.

    Columns, left to right: R1 over ``Q1_C`` (Q1's collector, C1's left
    end), R4 over ``Q2_B`` (Q2's base, C1's right end), R3 over ``Q1_B``
    (Q1's base, C2's left end), R2 over ``OUT`` (Q2's collector, C2's right
    end). Every row is placed from the same column positions, so the drop
    lines stay under their resistors whatever length the value strings have.
    """
    labels = [("R1", r_c), ("R4", r_b), ("R3", r_b), ("R2", r_c)]
    width = max(12, max(len(f"{lab} {val}") for lab, val in labels) + 2)
    col = [7 + k * width for k in range(4)]

    def row(pieces: list[tuple[int, str]]) -> str:
        line = ""
        for pos, text in pieces:
            line = line.ljust(pos) + text
        return line.rstrip()

    coupling = "+" + "---C1".ljust(width - 1, "-") + "+"
    coupling2 = "+" + "---C2".ljust(width - 1, "-") + "+"
    return "\n".join([
        "VCC ---+" + ("-" * (width - 1) + "+") * 3,
        row([(col[k], f"{lab} {val}") for k, (lab, val) in enumerate(labels)]),
        row([(c, "|") for c in col]),
        row([(3, "Q1_C"), (col[0], coupling), (col[1] + 1, "Q2_B"), (col[2] - 4, "Q1_B"), (col[2], coupling2), (col[3] + 1, "OUT (= Q2 컬렉터, J1.2)")]),
        row([(c, "|") for c in col]),
        row([(col[0] - 3, "Q1.C"), (col[1] - 3, "Q2.B"), (col[2] - 3, "Q1.B"), (col[3] - 3, "Q2.C")]),
        row([(col[0] - 3, "Q1.E = GND"), (col[3] - 3, "Q2.E = GND")]),
    ])


def _header_note(role: str, pins: str, req_text: str) -> PartNote:
    return PartNote(
        role=role,
        why=f"보드 밖과의 연결점: {pins}. 2.54 mm 핀 헤더는 브레드보드·점퍼선과 바로 맞고 KiCad 라이브러리에 심볼·풋프린트가 있음. {req_text}",
        criteria=[f"{pins}: 핀 수와 순서가 같은 1열 2.54 mm 헤더", "핀 번호 1.. 이 회로도의 넷 순서와 같아야 함 (넷 표 참조)", "정격 전류 ≥ 이 회로의 공급 전류 (mA 급이면 어떤 헤더든 충분)"],
        substitutes=[unverified("같은 핀 수의 1열 2.54 mm 수직 핀 헤더 (어느 제조사든)"), unverified("같은 핀 수의 직각(right-angle) 헤더", "풋프린트를 바꾸어야 함")],
    )


# --------------------------------------------------------------------------- divider


class DividerTemplate(Template):
    id = "divider"
    title = "unloaded resistive voltage divider"
    triggers = ("input_voltage", "output_voltage")
    needs = ("input_voltage", "output_voltage")
    serves = ("input_voltage", "output_voltage", "output_current")

    R2_OHM = 10_000.0
    TOL_REL = 0.01
    #: asked with the table when no load current is stated at all: an answer then enters the IR and is read before the build
    LOAD_QUESTION = (
        "Answer output_current=0 A if VOUT drives no load (a resistive divider is only valid unloaded, as a high-impedance reference); "
        "otherwise provide the circuit (components / nets) in the IR yourself - no verified template supplies a load."
    )
    #: the refusal when a load current *is* stated: the requirement exists, so an ``--answer`` for its key is not what changes it
    LOAD_REFUSAL = (
        "This template cannot serve that requirement: a resistive divider is only valid unloaded (a high-impedance reference) and no verified "
        "template supplies a load. Change the requirement {rid} in the IR (or correct the request) so it states 0 A, choose another design, "
        "or provide the circuit (components / nets) in the IR yourself."
    )

    def _load_question(self, why: str, requirement_id: str | None = None) -> MissingInformation:
        text = f"{why[0].upper()}{why[1:]}. "
        text += self.LOAD_QUESTION if requirement_id is None else self.LOAD_REFUSAL.format(rid=requirement_id)
        return MissingInformation(key="output_current", required=False, question=text, rationale=why)

    def refusals(self, ir: CircuitIR, inputs: dict[str, DesignInput], unusable: dict[str, str]) -> list[MissingInformation]:
        """The closed-world rule plus the divider's own: a stated load current must be 0 A."""
        out = super().refusals(ir, inputs, unusable)
        v_in, v_out = inputs["input_voltage"], inputs["output_voltage"]
        if "output_current" in unusable:
            ids = ", ".join(r.id for r in ir.requirements.requirements if r.key in KEY_ALIASES["output_current"]) or "req.output_current"
            out.append(self._load_question(f"output_current is stated but not readable ({unusable['output_current']}), and the divider is only valid unloaded", ids))
        elif "output_current" in inputs and inputs["output_current"].traced.value != 0.0:
            i_out = inputs["output_current"]
            out.append(self._load_question(
                f"the request needs {v_out.traced.value:.12g} V at {i_out.traced.value:.12g} A from {v_in.traced.value:.12g} V ({i_out.requirement.id}): "
                f"a resistive divider cannot supply a load",
                i_out.requirement.id,
            ))
        return out

    def build(self, ir: CircuitIR, inputs: dict[str, DesignInput], unusable: dict[str, str], library: KicadLibrary, *, confirmed: bool) -> Plan:
        t = self.id
        plan = Plan(template=t, title=self.title)
        v_in, v_out = inputs["input_voltage"], inputs["output_voltage"]
        plan.inputs = {"input_voltage": v_in, "output_voltage": v_out}
        serves_out = [v_out.requirement.id]
        if "output_current" in inputs:  # refusals() let it through: it is 0 A
            i_out = inputs["output_current"]
            plan.inputs["output_current"] = i_out
            serves_out.append(i_out.requirement.id)
            plan.notes.append(f"{i_out.requirement.id} = 0 A: VOUT is a high-impedance reference and serves it")
        if not 0.0 < v_out.traced.value < v_in.traced.value:
            return _refused(plan, f"output {v_out.traced.value:.12g} V is not between 0 and the input {v_in.traced.value:.12g} V ({v_out.requirement.id}, {v_in.requirement.id})")
        r2_choice, r2 = _choice(t, "r2", self.R2_OHM, "ohm", "lower resistor of the unloaded reference divider (R1 is solved from it)", confirmed)
        tol_choice, tol = _choice(t, "tol_rel", self.TOL_REL, None, "relative tolerance of the v(VOUT) expectation (1 %)", confirmed)
        plan.choices = [r2_choice, tol_choice]
        params: dict[str, Traced] = {"v_in": v_in.traced, "v_out_target": v_out.traced, "r2": r2}
        try:
            params["r1"] = divider_r1_for_v_out(params["v_in"], params["v_out_target"], params["r2"], ("v_in", "v_out_target", "r2"))
            params["v_out"] = voltage_divider_output(params["v_in"], params["r1"], params["r2"], ("v_in", "r1", "r2"))
        except (ValueError, ZeroDivisionError) as e:
            return _refused(plan, f"{e} ({v_in.requirement.id} = {v_in.traced.value:.12g} V, {v_out.requirement.id} = {v_out.traced.value:.12g} V)")
        params["tol_rel"] = tol
        plan.computed = [("r1", params["r1"]), ("v_out", params["v_out"])]
        try:
            r1 = library_component(library, "R1", _value_text(params["r1"].value), "upper divider resistor", *RESISTOR, structural_provenance(t, "upper divider resistor"), serves_out)
            r2c = library_component(library, "R2", _value_text(params["r2"].value), "lower divider resistor", *RESISTOR, structural_provenance(t, "lower divider resistor"), serves_out)
            j1 = library_component(library, "J1", "Conn_01x03", "VIN / VOUT / GND header", *HEADER_3, structural_provenance(t, "board header"), [v_in.requirement.id])
            r1a, r1b = two_terminals(r1)
            r2a, r2b = two_terminals(r2c)
            require_pins(j1, ("1", "2", "3"))
        except TemplateRefusal as e:
            return _refused(plan, str(e))
        for c, key in ((r1, "r1"), (r2c, "r2")):
            c.electrical["resistance"] = params[key]
            c.spice = SpiceBinding(device=SpiceDevice.R, value=params[key], pin_order=[p.number for p in c.pins], provenance=structural_provenance(t, "ideal resistor at the design value"))
        j1.spice = SpiceBinding(exclude=True, exclude_reason="connector, no electrical model", provenance=structural_provenance(t, "header excluded from the netlist"))
        nets = [
            _net("VIN", NetKind.POWER, [("J1", "1"), ("R1", r1a)], t, [v_in.requirement.id]),
            _net("VOUT", NetKind.SIGNAL, [("J1", "2"), ("R1", r1b), ("R2", r2a)], t, serves_out),
            _net("GND", NetKind.GROUND, [("J1", "3"), ("R2", r2b)], t),
        ]
        sim = SimulationSetup(
            stimuli=[Stimulus(id="VIN", source="voltage", net="VIN", reference_net="GND", kind=StimulusKind.DC, value=params["v_in"], provenance=structural_provenance(t, "the input supply at v_in"), serves_requirements=[v_in.requirement.id])],
            analyses=[AnalysisSpec(id="op", kind=SpiceAnalysis.OP, provenance=structural_provenance(t, "operating point"))],
            expectations=[Expectation(
                id="v_out", analysis_id="op", vector="v(VOUT)", reduce=Reduce.VALUE, nominal=params["v_out"], tol_rel=params["tol_rel"], requirement_id=v_out.requirement.id,
                provenance=structural_provenance(t, f"the divider output at the operating point verifies {v_out.requirement.id}"),
            )],
        )
        topology = Topology(
            name="resistive divider", domains=[CircuitDomain.ANALOG],
            rationale="unloaded reference divider from the confirmed input and output voltages; valid only with a high-impedance load on VOUT",
            provenance=structural_provenance(t, "selected by input_voltage + output_voltage without a load current"),
            blocks=[Block(id="divider", function="resistive voltage divider", domain=CircuitDomain.ANALOG, component_refs=["R1", "R2"], input_nets=["VIN"], output_nets=["VOUT"], provenance=structural_provenance(t, "block"))],
        )
        constraint = Constraint(
            id="c.divider.unloaded", kind=ConstraintKind.ELECTRICAL, target="VOUT",
            description="VOUT is a high-impedance reference: no load current was required; the output voltage holds only unloaded",
            provenance=structural_provenance(t, "template validity condition"),
        )
        plan.parts = [_part_line(c) for c in (r1, r2c, j1)]
        plan.nets = [_net_line(n) for n in nets]
        plan.simulation = [f"analysis op; expectation v_out: v(VOUT) = {params['v_out'].value:.12g} V +/- {self.TOL_REL:.0%} verifies {v_out.requirement.id}"]
        plan.changes = _changes(t, self.title, topology, [r1, r2c, j1], nets, params, sim, [constraint])
        if "output_current" not in inputs:
            plan.questions.append(self._load_question("no load current was stated: does VOUT need to supply one?"))
        return plan

    def theory(self, ir: CircuitIR) -> list[TheorySection]:
        v_in, v_target, r1, r2, v_out, tol = (parameter_value(ir, k) for k in ("v_in", "v_out_target", "r1", "r2", "v_out", "tol_rel"))
        total = _add(r1, r2)
        i = _div(v_in, total)
        p1, p2 = _mul(i, i, r1), _mul(i, i, r2)
        r_th = _div(_mul(r1, r2), total)
        r_load = _mul(10.0, r_th) if r_th is not None else None
        v_loaded = _mul(v_out, _div(r_load, _add(r_load, r_th))) if r_load is not None else None
        return [
            TheorySection("동작 원리", (
                "저항 두 개를 직렬로 두고 가운데에서 전압을 꺼내는 분압기입니다. 같은 전류 I 가 R1, R2 를 차례로 흐르므로 각 저항의 전압은 저항값에 비례합니다.\n\n"
                "    V_out = V_in·R2/(R1+R2)\n\n"
                f"이 설계에서 R2 = {quantity(r2, 'ohm')} 는 템플릿의 선택값(사용자 확인)이고, R1 은 목표 출력 V_out = {quantity(v_target, 'V')} 에서 계산기 `calc.divider.r1_for_v_out` 가 풉니다.\n\n"
                "    R1 = R2·(V_in − V_out)/V_out\n"
                f"       = {quantity(r2, 'ohm')}·({quantity(v_in, 'V')} − {quantity(v_target, 'V')})/{quantity(v_target, 'V')} = {quantity(r1, 'ohm')}\n\n"
                f"검산 (`calc.divider.v_out`): V_out = {quantity(v_in, 'V')}·{quantity(r2, 'ohm')}/({quantity(r1, 'ohm')} + {quantity(r2, 'ohm')}) = {quantity(v_out, 'V')}"
            )),
            TheorySection("이 설계의 수치", (
                "| 항목 | 식 | 값 |\n|---|---|---|\n"
                f"| 분압 전류 | I = V_in/(R1+R2) = {quantity(v_in, 'V')}/{quantity(total, 'ohm')} | {quantity(i, 'A')} |\n"
                f"| R1 소비전력 | P(R1) = I²·R1 | {quantity(p1, 'W')} |\n"
                f"| R2 소비전력 | P(R2) = I²·R2 | {quantity(p2, 'W')} |\n"
                f"| 테브냉 출력 저항 | R_th = R1‖R2 = R1·R2/(R1+R2) | {quantity(r_th, 'ohm')} |\n"
                f"| 출력 전압 (무부하) | V_out = V_in·R2/(R1+R2) | {quantity(v_out, 'V')} |"
            )),
            TheorySection("무부하 조건인 이유", (
                f"VOUT 에서 본 분압기는 전압 V_out, 직렬 저항 R_th = {quantity(r_th, 'ohm')} 인 테브냉 등가 전원입니다. 부하 R_L 을 달면 R2 와 병렬이 되어 출력이 떨어집니다.\n\n"
                "    V_out(부하) = V_out·R_L/(R_L + R_th)\n\n"
                f"예: R_L = 10·R_th = {quantity(r_load, 'ohm')} 만 달아도 V_out(부하) = {quantity(v_loaded, 'V')} 로 약 9 % 낮아집니다. "
                "그래서 이 템플릿은 부하 전류 요구사항이 0 A 일 때만(고임피던스 기준 전압) 적용되며, 부하가 있으면 설계를 거부합니다 (제약 조건 `c.divider.unloaded`)."
            )),
            TheorySection("시뮬레이션 판정", (
                f"동작점(op) 해석에서 v(VOUT) 를 읽어 공칭값 {quantity(v_out, 'V')} 과 비교합니다.\n\n"
                f"    {_judging_line(v_out, 'V', tol_rel=tol)}\n\n"
                "이상적인 저항 모델이므로 시뮬레이션은 위 식을 그대로 재현하며, 실물의 오차는 두 저항의 공차 합(비율 오차)이 지배합니다."
            )),
        ]

    def part_notes(self, ir: CircuitIR) -> dict[str, PartNote]:
        v_in, v_out, r1, r2, tol = (parameter_value(ir, k) for k in ("v_in", "v_out", "r1", "r2", "tol_rel"))
        i = _div(v_in, _add(r1, r2))
        tol_text = f"{number(tol * 100, 3)} % 이하 (판정 허용치가 {number(tol * 100, 3)} % 이므로 두 저항의 공차가 비율 오차로 그대로 들어감)" if tol is not None else "1 % (판정 허용치 기록 없음)"
        return {
            "R1": PartNote(
                role="R1: 상단 분압 저항 (VIN–VOUT), 출력 비율을 정함",
                why=f"R2 = {quantity(r2, 'ohm')} 선택값에서 R1 = R2·(V_in − V_out)/V_out = {quantity(r1, 'ohm')} 로 계산. 0603 SMD 는 이 SMD 템플릿의 기본 풋프린트(소형·자동 조립).",
                criteria=_resistor_criteria(r1, _mul(i, i, r1), tol_text, chosen=_is_choice(ir, "r1")),
                substitutes=_resistor_substitutes(r1, "Resistor_SMD"),
            ),
            "R2": PartNote(
                role="R2: 하단 분압 저항 (VOUT–GND)",
                why=f"템플릿 선택값 {quantity(r2, 'ohm')}: 분압 전류 {quantity(i, 'A')} 가 작아 소비전력이 무시할 만하고, 테브냉 저항이 kΩ 급이라 고임피던스 부하에 적합. 0603 SMD.",
                criteria=_resistor_criteria(r2, _mul(i, i, r2), tol_text, chosen=_is_choice(ir, "r2")),
                substitutes=_resistor_substitutes(r2, "Resistor_SMD"),
            ),
            "J1": _header_note("J1: VIN / VOUT / GND 헤더", "1 = VIN, 2 = VOUT, 3 = GND", "VOUT 은 고임피던스 기준 전압이므로 헤더 뒤에 부하를 달면 안 됨."),
        }


# --------------------------------------------------------------------------- LED


class LedTemplate(Template):
    id = "led"
    title = "LED indicator with series resistor"
    triggers = ("led_forward_voltage", "led_forward_current")
    all_triggers = False
    needs = ("input_voltage", "led_forward_voltage", "led_forward_current")
    serves = ("input_voltage", "led_forward_voltage", "led_forward_current")

    TOL_REL = 0.01
    MODEL = (
        "ideal constant-V_f LED: D1 is excluded from the netlist and replaced by the stimulus VLED = v_f between LED_A and GND; "
        "the expectation i(VLED) measures the current through that ideal source, not through a diode model (no authoritative model card)"
    )

    def build(self, ir: CircuitIR, inputs: dict[str, DesignInput], unusable: dict[str, str], library: KicadLibrary, *, confirmed: bool) -> Plan:
        t = self.id
        plan = Plan(template=t, title=self.title)
        missing = [k for k in self.needs if k not in inputs]
        if missing:
            return _missing_inputs(plan, "The LED indicator template", missing, unusable)
        v_in, v_f, i_f = inputs["input_voltage"], inputs["led_forward_voltage"], inputs["led_forward_current"]
        plan.inputs = {"input_voltage": v_in, "led_forward_voltage": v_f, "led_forward_current": i_f}
        params: dict[str, Traced] = {"v_in": v_in.traced, "v_f": v_f.traced, "i_f": i_f.traced}
        try:
            params["r_led"] = led_series_resistor(params["v_in"], params["v_f"], params["i_f"], ("v_in", "v_f", "i_f"))
            params["i_led"] = led_current(params["v_in"], params["v_f"], params["r_led"], ("v_in", "v_f", "r_led"))
        except (ValueError, ZeroDivisionError) as e:
            return _refused(plan, f"{e} ({v_in.requirement.id} = {v_in.traced.value:.12g} V, {v_f.requirement.id} = {v_f.traced.value:.12g} V, {i_f.requirement.id} = {i_f.traced.value:.12g} A)")
        tol_choice, tol = _choice(t, "tol_rel", self.TOL_REL, None, "relative tolerance of the i(VLED) expectation (1 %)", confirmed)
        params["tol_rel"] = tol
        plan.choices = [Choice("led_model", self.MODEL), tol_choice]
        plan.computed = [("r_led", params["r_led"]), ("i_led", params["i_led"])]
        req_v, req_i = v_f.requirement.id, i_f.requirement.id
        model_prov = choice_provenance(t, f"led_model: {self.MODEL}", confirmed)
        try:
            r1 = library_component(library, "R1", _value_text(params["r_led"].value), "LED series resistor", *RESISTOR, structural_provenance(t, "series resistor"), [req_i])
            d1 = library_component(library, "D1", "LED", "indicator LED", *LED, structural_provenance(t, "indicator LED"), [req_v, req_i])
            j1 = library_component(library, "J1", "Conn_01x02", "VCC / GND header", *HEADER_2, structural_provenance(t, "supply header"), [v_in.requirement.id])
            r1a, r1b = two_terminals(r1)
            require_pins(j1, ("1", "2"))
        except TemplateRefusal as e:
            return _refused(plan, str(e))
        anode, cathode = pin_by_name(d1, "A"), pin_by_name(d1, "K")
        if anode is None or cathode is None:
            return _refused(plan, f"Device:LED pins are not named A / K in this library (pins: {[(p.number, p.name) for p in d1.pins]}); the template will not guess the polarity")
        r1.electrical["resistance"] = params["r_led"]
        r1.spice = SpiceBinding(device=SpiceDevice.R, value=params["r_led"], pin_order=[p.number for p in r1.pins], provenance=structural_provenance(t, "ideal resistor at the design value"))
        d1.spice = SpiceBinding(exclude=True, exclude_reason=self.MODEL, provenance=model_prov)
        j1.spice = SpiceBinding(exclude=True, exclude_reason="connector, no electrical model", provenance=structural_provenance(t, "header excluded from the netlist"))
        nets = [
            _net("VCC", NetKind.POWER, [("J1", "1"), ("R1", r1a)], t, [v_in.requirement.id]),
            _net("LED_A", NetKind.SIGNAL, [("R1", r1b), ("D1", anode)], t, [req_i]),
            _net("GND", NetKind.GROUND, [("J1", "2"), ("D1", cathode)], t),
        ]
        sim = SimulationSetup(
            stimuli=[
                Stimulus(id="VIN", source="voltage", net="VCC", reference_net="GND", kind=StimulusKind.DC, value=params["v_in"], provenance=structural_provenance(t, "the supply at v_in"), serves_requirements=[v_in.requirement.id]),
                Stimulus(id="VLED", source="voltage", net="LED_A", reference_net="GND", kind=StimulusKind.DC, value=params["v_f"], provenance=model_prov, serves_requirements=[req_v]),
            ],
            analyses=[AnalysisSpec(id="op", kind=SpiceAnalysis.OP, provenance=structural_provenance(t, "operating point"))],
            expectations=[Expectation(
                id="i_led", analysis_id="op", vector="i(VLED)", reduce=Reduce.VALUE, nominal=params["i_led"], tol_rel=params["tol_rel"], requirement_id=req_i,
                provenance=structural_provenance(t, f"the current through the ideal source VLED that replaces D1 (ideal constant-V_f model), not through a diode model; verifies {req_i}"),
            )],
        )
        topology = Topology(
            name="LED indicator", domains=[CircuitDomain.ANALOG],
            rationale="series resistor sets the LED current from the confirmed supply, forward voltage and forward current; the LED is modelled as an ideal constant forward drop",
            provenance=structural_provenance(t, "selected by led_forward_voltage / led_forward_current with input_voltage"),
            blocks=[Block(id="led", function="current-limited LED indicator", domain=CircuitDomain.ANALOG, component_refs=["R1", "D1"], input_nets=["VCC"], output_nets=["LED_A"], provenance=structural_provenance(t, "block"))],
        )
        plan.parts = [_part_line(c) for c in (r1, d1, j1)]
        plan.nets = [_net_line(n) for n in nets]
        plan.simulation = [
            f"stimuli VIN = {params['v_in'].value:.12g} V on VCC, VLED = {params['v_f'].value:.12g} V on LED_A (the ideal LED); analysis op",
            f"expectation i_led: i(VLED) = {params['i_led'].value:.12g} A +/- {self.TOL_REL:.0%} verifies {req_i} (current through the ideal source, D1 excluded)",
        ]
        plan.changes = _changes(t, self.title, topology, [r1, d1, j1], nets, params, sim, [])
        return plan

    def theory(self, ir: CircuitIR) -> list[TheorySection]:
        v_in, v_f, i_f, r, i_led, tol = (parameter_value(ir, k) for k in ("v_in", "v_f", "i_f", "r_led", "i_led", "tol_rel"))
        drop = _sub(v_in, v_f)
        p_r, p_led = _mul(drop, i_led), _mul(v_f, i_led)
        return [
            TheorySection("동작 원리와 식", (
                "LED 는 순방향 전압 V_f 근처에서 전류가 급격히 늘어나는 비선형 소자라 전압원에 바로 달 수 없습니다. 직렬 저항이 남는 전압을 받아 전류를 정합니다.\n\n"
                "    R = (V_in − V_f)/I_f\n"
                f"      = ({quantity(v_in, 'V')} − {quantity(v_f, 'V')})/{quantity(i_f, 'A')} = {quantity(r, 'ohm')}   (`calc.led.R`)\n\n"
                "    I = (V_in − V_f)/R\n"
                f"      = {quantity(drop, 'V')}/{quantity(r, 'ohm')} = {quantity(i_led, 'A')}   (`calc.led.I`)"
            )),
            TheorySection("이 설계의 수치", (
                "| 항목 | 식 | 값 |\n|---|---|---|\n"
                f"| 저항 양단 전압 | V_in − V_f | {quantity(drop, 'V')} |\n"
                f"| 저항 소비전력 | P(R) = (V_in − V_f)·I = (V_in − V_f)²/R | {quantity(p_r, 'W')} |\n"
                f"| LED 소비전력 | P(LED) = V_f·I | {quantity(p_led, 'W')} |\n"
                f"| 공급 전류 | I | {quantity(i_led, 'A')} |"
            )),
            TheorySection("이상적 정전압 강하 모델과 그 한계", (
                f"시뮬레이션은 D1 을 넷리스트에서 빼고 LED_A–GND 사이에 정전압원 VLED = V_f = {quantity(v_f, 'V')} 를 둡니다(사용자가 확인한 모델링 선택). "
                "기대값 i(VLED) 는 그 이상 전원을 흐르는 전류이지 다이오드 모델의 전류가 아닙니다.\n\n"
                "실물 LED 의 V_f 는 전류와 온도에 따라 수백 mV 변하고 같은 부품끼리도 편차가 있습니다. 전류 오차는 dI/I ≈ −dV_f/(V_in − V_f) 이므로 "
                f"이 설계에서 V_f 가 0.1 V 커지면 전류는 약 {number(_mul(100.0, _div(0.1, drop)), 3)} % 줄어듭니다. 저항 양단 전압 {quantity(drop, 'V')} 이 클수록 이 민감도가 낮아집니다."
            )),
            TheorySection("시뮬레이션 판정", (
                f"동작점(op) 해석에서 i(VLED) 를 읽어 공칭값 {quantity(i_led, 'A')} 과 비교합니다.\n\n    {_judging_line(i_led, 'A', tol_rel=tol)}"
            )),
        ]

    def part_notes(self, ir: CircuitIR) -> dict[str, PartNote]:
        v_in, v_f, i_f, r, i_led, tol = (parameter_value(ir, k) for k in ("v_in", "v_f", "i_f", "r_led", "i_led", "tol_rel"))
        drop = _sub(v_in, v_f)
        tol_text = f"{number(tol * 100, 3)} % 이하 (판정 허용치)" if tol is not None else "1 % (판정 허용치 기록 없음)"
        return {
            "R1": PartNote(
                role="R1: LED 직렬 전류 제한 저항",
                why=f"R = (V_in − V_f)/I_f = {quantity(r, 'ohm')} (계산기 값 그대로). 0603 SMD 는 이 SMD 템플릿의 기본 풋프린트.",
                criteria=_resistor_criteria(r, _mul(drop, i_led), tol_text, chosen=_is_choice(ir, "r_led")),
                substitutes=_resistor_substitutes(r, "Resistor_SMD"),
            ),
            "D1": PartNote(
                role="D1: 표시 LED (시뮬레이션에서는 V_f 의 이상적 정전압 강하로 대체)",
                why=f"사용자 요구 V_f = {quantity(v_f, 'V')}, I_f = {quantity(i_f, 'A')} 를 만족하는 LED 라면 어느 것이든 이 회로가 맞음. KiCad 라이브러리 `Device:LED` 의 핀 이름 A / K 로 극성을 잡음. 0603 SMD.",
                criteria=[
                    f"데이터시트의 V_f(@ I_f = {quantity(i_f, 'A')}) 가 요구값 {quantity(v_f, 'V')} 과 같을 것 (다르면 R1 을 다시 계산)",
                    f"최대 순방향 전류 I_f(max) ≥ {quantity(i_led, 'A')} 에 여유",
                    "역전압 정격 ≥ V_in (역접속 보호는 없음)",
                    "0603 풋프린트, 애노드 / 캐소드 표시가 라이브러리 풋프린트의 극성 표시와 같을 것",
                ],
                substitutes=[unverified("같은 색·같은 V_f 급의 0603 LED (어느 제조사든)", "V_f 를 데이터시트에서 확인"), unverified("3 mm / 5 mm THT LED", "풋프린트를 LED_THT 로 바꾸어야 함")],
            ),
            "J1": _header_note("J1: VCC / GND 헤더", "1 = VCC, 2 = GND", ""),
        }


def _example(key: str) -> str:
    return {"input_voltage": "5 V", "led_forward_voltage": "2 V", "led_forward_current": "10 mA", "oscillation_frequency": "1 kHz"}.get(key, f"<value {UNIT_OF[key]}>")


def _missing_inputs(plan: Plan, who: str, missing: list[str], unusable: dict[str, str]) -> Plan:
    """A template input no requirement states is a required question; one stated but unreadable is a note (the requirement must change)."""
    for k in missing:
        if k in unusable:
            plan.notes.append(f"{k} not usable: {unusable[k]}")
        else:
            plan.questions.append(MissingInformation(
                key=k, question=f"{who} needs {k} in {UNIT_OF[k]}: answer {k}=<value {UNIT_OF[k]}> (e.g. {k}=\"{_example(k)}\")",
                rationale="template input",
            ))
    return _refused(plan, f"input(s) {missing} missing")


# --------------------------------------------------------------------------- RC low-pass


class RcLowpassTemplate(Template):
    id = "rc_lowpass"
    title = "first-order RC low-pass"
    triggers = ("cutoff_frequency",)
    needs = ("cutoff_frequency",)
    serves = ("cutoff_frequency",)

    C_FARAD = 100e-9
    TOL_REL = 0.02
    AC_PROBE_V = 1.0
    AC_VARIATION = "dec"
    AC_POINTS = 100

    def build(self, ir: CircuitIR, inputs: dict[str, DesignInput], unusable: dict[str, str], library: KicadLibrary, *, confirmed: bool) -> Plan:
        t = self.id
        plan = Plan(template=t, title=self.title)
        f_c = inputs["cutoff_frequency"]
        plan.inputs = {"cutoff_frequency": f_c}
        c_choice, c = _choice(t, "c", self.C_FARAD, "F", "shunt capacitor of the low-pass (R is solved from it and f_c)", confirmed)
        tol_choice, tol = _choice(t, "tol_rel", self.TOL_REL, None, "relative tolerance of the |H(f_c)| expectation (2 %: the ac grid is interpolated between 100 points per decade)", confirmed)
        probe_choice, probe = _choice(t, "ac_probe", self.AC_PROBE_V, "V", "small-signal probe amplitude of the ac sweep (unit input, so v(OUT) is |H|)", confirmed)
        var_choice, variation = _choice(t, "ac_variation", self.AC_VARIATION, None, "logarithmic ac sweep", confirmed)
        pts_choice, points = _choice(t, "ac_points", self.AC_POINTS, None, "points per decade of the ac sweep", confirmed)
        plan.choices = [c_choice, tol_choice, probe_choice, var_choice, pts_choice]
        params: dict[str, Traced] = {"f_c": f_c.traced, "c": c}
        try:
            params["r"] = rc_r_for_cutoff(params["f_c"], params["c"], ("f_c", "c"))
            params["tau"] = rc_time_constant(params["r"], params["c"], ("r", "c"))
            params["h_fc"] = rc_lowpass_magnitude(params["f_c"], params["tau"], ("f_c", "tau"))
            params["tol_rel"] = tol
            params["ac_fstart"] = rc_ac_fstart(params["f_c"], ("f_c",))
            params["ac_fstop"] = rc_ac_fstop(params["f_c"], ("f_c",))
        except (ValueError, ZeroDivisionError) as e:
            return _refused(plan, f"{e} ({f_c.requirement.id} = {f_c.traced.value:.12g} Hz)")
        params["ac_probe"] = probe
        plan.computed = [(k, params[k]) for k in ("r", "tau", "h_fc", "ac_fstart", "ac_fstop")]
        req = f_c.requirement.id
        try:
            r1 = library_component(library, "R1", _value_text(params["r"].value), "series resistor", *RESISTOR, structural_provenance(t, "series resistor"), [req])
            c1 = library_component(library, "C1", _value_text(params["c"].value), "shunt capacitor", *CAPACITOR, structural_provenance(t, "shunt capacitor"), [req])
            j1 = library_component(library, "J1", "Conn_01x03", "IN / OUT / GND header", *HEADER_3, structural_provenance(t, "board header"), [])
            r1a, r1b = two_terminals(r1)
            c1a, c1b = two_terminals(c1)
            require_pins(j1, ("1", "2", "3"))
        except TemplateRefusal as e:
            return _refused(plan, str(e))
        r1.electrical["resistance"] = params["r"]
        c1.electrical["capacitance"] = params["c"]
        r1.spice = SpiceBinding(device=SpiceDevice.R, value=params["r"], pin_order=[p.number for p in r1.pins], provenance=structural_provenance(t, "ideal resistor at the design value"))
        c1.spice = SpiceBinding(device=SpiceDevice.C, value=params["c"], pin_order=[p.number for p in c1.pins], provenance=structural_provenance(t, "ideal capacitor at the design value"))
        j1.spice = SpiceBinding(exclude=True, exclude_reason="connector, no electrical model", provenance=structural_provenance(t, "header excluded from the netlist"))
        nets = [
            _net("IN", NetKind.SIGNAL, [("J1", "1"), ("R1", r1a)], t),
            _net("OUT", NetKind.SIGNAL, [("J1", "2"), ("R1", r1b), ("C1", c1a)], t, [req]),
            _net("GND", NetKind.GROUND, [("J1", "3"), ("C1", c1b)], t),
        ]
        sim = SimulationSetup(
            stimuli=[Stimulus(
                id="VIN", source="voltage", net="IN", reference_net="GND", kind=StimulusKind.DC, value=params["ac_probe"], params={"ac": params["ac_probe"]},
                provenance=structural_provenance(t, "unit small-signal probe on IN"),
            )],
            analyses=[AnalysisSpec(
                id="ac", kind=SpiceAnalysis.AC, params={"variation": variation, "points": points, "fstart": params["ac_fstart"], "fstop": params["ac_fstop"]},
                provenance=structural_provenance(t, "frequency response two decades around the corner"),
            )],
            expectations=[Expectation(
                id="h_fc", analysis_id="ac", vector="v(OUT)", reduce=Reduce.AT, at=params["f_c"], nominal=params["h_fc"], tol_rel=params["tol_rel"], requirement_id=req,
                provenance=structural_provenance(t, f"|H| at the sweep point f_c = the value of {req}; the cutoff is where |H| = 1/sqrt(2)"),
            )],
        )
        topology = Topology(
            name="rc low-pass", domains=[CircuitDomain.ANALOG],
            rationale="first-order RC low-pass whose cutoff frequency f_c is where |H(f_c)| = 1/sqrt(2) (-3 dB); C chosen, R = 1 / (2 pi f_c C)",
            provenance=structural_provenance(t, "selected by cutoff_frequency"),
            blocks=[Block(id="lowpass", function="first-order RC low-pass", domain=CircuitDomain.ANALOG, component_refs=["R1", "C1"], input_nets=["IN"], output_nets=["OUT"], provenance=structural_provenance(t, "block"))],
        )
        constraint = Constraint(
            id="c.rc_lowpass.cutoff", kind=ConstraintKind.ELECTRICAL, target="OUT",
            description=f"the cutoff frequency ({req}) is the frequency where |H| = 1/sqrt(2); the expectation h_fc checks |H| at that sweep point",
            provenance=structural_provenance(t, "what 'cutoff frequency' means for this template"),
        )
        plan.parts = [_part_line(x) for x in (r1, c1, j1)]
        plan.nets = [_net_line(n) for n in nets]
        plan.simulation = [
            f"stimulus VIN: dc {self.AC_PROBE_V:.12g} V with ac {self.AC_PROBE_V:.12g} V on IN; analysis ac {self.AC_VARIATION} {self.AC_POINTS} {params['ac_fstart'].value:.12g} {params['ac_fstop'].value:.12g} (Hz)",
            f"expectation h_fc: v(OUT) at {params['f_c'].value:.12g} Hz = {params['h_fc'].value:.12g} +/- {self.TOL_REL:.0%} verifies {req}",
        ]
        plan.changes = _changes(t, self.title, topology, [r1, c1, j1], nets, params, sim, [constraint])
        return plan

    def theory(self, ir: CircuitIR) -> list[TheorySection]:
        f_c, c, r, tau, h_fc, tol, f_start, f_stop, probe = (parameter_value(ir, k) for k in ("f_c", "c", "r", "tau", "h_fc", "tol_rel", "ac_fstart", "ac_fstop", "ac_probe"))
        f_check = _div(1.0, _mul(2.0 * math.pi, r, c))
        variation = points = NO_RECORD
        if ir.simulation is not None:
            for a in ir.simulation.analyses:
                if "variation" in a.params and "points" in a.params:
                    variation, points = str(a.params["variation"].value), str(a.params["points"].value)
        h_dec_below = _div(1.0, math.sqrt(1.0 + 0.01))
        h_dec_above = _div(1.0, math.sqrt(1.0 + 100.0))
        return [
            TheorySection("동작 원리와 식", (
                "직렬 저항 R 과 병렬 커패시터 C 로 된 1차 저역통과 필터입니다. 커패시터의 임피던스 1/(j2πfC) 가 주파수에 반비례하므로 높은 주파수일수록 출력이 줄어듭니다.\n\n"
                "    H(f) = 1/(1 + j2πfRC),   τ = RC,   f_c = 1/(2πRC)\n\n"
                f"이 설계에서 C = {quantity(c, 'F')} 는 템플릿의 선택값(사용자 확인)이고 R 은 차단 주파수에서 계산기 `calc.rc.r_for_cutoff` 가 풉니다.\n\n"
                f"    R = 1/(2π·f_c·C) = 1/(2π·{quantity(f_c, 'Hz')}·{quantity(c, 'F')}) = {quantity(r, 'ohm')}\n"
                f"    τ = R·C = {quantity(tau, 's')}   (`calc.rc.tau`)\n"
                f"    검산: f_c = 1/(2π·R·C) = {quantity(f_check, 'Hz')}"
            )),
            TheorySection("주파수 응답", (
                "    |H(f)| = 1/√(1 + (f/f_c)²),   위상 φ(f) = −atan(f/f_c)\n\n"
                "| 주파수 | \\|H\\| | 위상 |\n|---|---|---|\n"
                f"| f_c/10 = {quantity(_div(f_c, 10.0), 'Hz')} | {number(h_dec_below, 4)} (−0.04 dB) | −5.7° |\n"
                f"| f_c = {quantity(f_c, 'Hz')} | 1/√2 = {number(h_fc, 5)} (−3.01 dB) | −45° |\n"
                f"| 10·f_c = {quantity(_mul(10.0, f_c), 'Hz')} | {number(h_dec_above, 4)} (−20.04 dB) | −84.3° |\n\n"
                "차단 주파수 위에서는 10배마다 −20 dB(1/10) 씩 줄어드는 1차 특성입니다. 계산기 `calc.rc.lowpass_magnitude` 가 f_c 에서의 |H| 를 계산해 기대값의 공칭값으로 씁니다."
            )),
            TheorySection("ac 스위프 격자와 판정", (
                f"프로브 VIN 은 dc {quantity(probe, 'V')} 에 ac 진폭 {quantity(probe, 'V')} 인 소신호 전원이므로 v(OUT) 의 크기가 곧 |H| 입니다. "
                f"스위프는 `{variation}` {points} 점/decade, f_start = f_c/100 = {quantity(f_start, 'Hz')} 부터 f_stop = 100·f_c = {quantity(f_stop, 'Hz')} 까지(코너 양쪽 두 decade)입니다.\n\n"
                f"기대값 h_fc 는 스위프 축에서 f_c = {quantity(f_c, 'Hz')} 의 값을 읽습니다(격자에 정확히 있으면 그 점, 아니면 이웃 두 표본 사이를 log-log 선형 보간). "
                "보간이면 두 이웃을 함께 판정합니다: 둘 다 허용치 안이면 PASS, 둘 다 같은 쪽으로 허용치 밖이면 FAIL, 그 밖(한쪽만 밖이거나 양쪽으로 걸침)은 격자가 허용치를 분해하지 못하므로 UNRESOLVED.\n\n"
                f"    {_judging_line(h_fc, None, tol_rel=tol)}\n\n"
                "허용치 2 % 는 100 점/decade 격자 사이의 보간 오차를 덮기 위한 선택값입니다."
            )),
        ]

    def part_notes(self, ir: CircuitIR) -> dict[str, PartNote]:
        f_c, c, r, tol, probe = (parameter_value(ir, k) for k in ("f_c", "c", "r", "tol_rel", "ac_probe"))
        p_r = _div(_mul(probe, probe), r)  # the probe amplitude across R (the whole probe at high frequency)
        return {
            "R1": PartNote(
                role="R1: 직렬 저항 (IN–OUT), C1 과 함께 τ = RC 를 정함",
                why=f"C = {quantity(c, 'F')} 선택값에서 R = 1/(2π·f_c·C) = {quantity(r, 'ohm')} 로 계산. 0603 SMD 는 이 SMD 템플릿의 기본 풋프린트.",
                criteria=_resistor_criteria(r, p_r, "1 % 이하 (Δf_c/f_c ≈ −ΔR/R: 저항 오차가 그대로 차단 주파수 오차가 됨)", chosen=_is_choice(ir, "r")),
                substitutes=_resistor_substitutes(r, "Resistor_SMD"),
            ),
            "C1": PartNote(
                role="C1: 병렬(shunt) 커패시터 (OUT–GND)",
                why=f"템플릿 선택값 {quantity(c, 'F')}: kHz 급 코너에서 R 이 kΩ 급이 되도록 고른 값. 0603 SMD.",
                criteria=[
                    f"정전용량 {quantity(c, 'F')}, 공차 5 % 이하 (Δf_c/f_c ≈ −ΔC/C: 커패시터 오차가 그대로 차단 주파수 오차가 됨)",
                    "유전체 C0G/NP0 권장 (X7R 은 전압·온도에 따라 용량이 수십 % 변해 f_c 가 움직임)",
                    f"정격 전압 ≥ 2 × 신호 진폭 (이 설계의 프로브 진폭 {quantity(probe, 'V')}; 실제 신호 진폭 요구사항은 {NO_RECORD})",
                ],
                substitutes=[unverified(f"같은 값 {quantity(c, 'F')}의 0603 C0G/NP0 MLCC", "현재 풋프린트 그대로"), unverified(f"같은 값의 필름 커패시터", "풋프린트를 Capacitor_THT 로 바꾸어야 함")],
            ),
            "J1": _header_note("J1: IN / OUT / GND 헤더", "1 = IN, 2 = OUT, 3 = GND", "OUT 은 R1 을 통한 출력이라 부하 임피던스는 R1 보다 훨씬 커야 함."),
        }


# --------------------------------------------------------------------------- BJT astable multivibrator


class AstableTemplate(Template):
    """Collector-coupled BJT astable multivibrator: a square wave at ``oscillation_frequency`` from ``input_voltage``.

    Q1 / Q2 are ``Transistor_BJT:2N3904`` (TO-92) wired by pin *name* (``E`` /
    ``B`` / ``C``, refused when the library spells them differently); R1 / R2
    are the collector loads (``r_c``), R3 / R4 the base resistors (``r_b``),
    C1 / C2 the timing capacitors (``c``, solved from the frequency). The
    simulation is a transient with ``uic`` started by C2's initial voltage
    (``c2_ic``): the ideal symmetric circuit has a metastable both-on state
    and a real one starts on mismatch and noise, so the start is a confirmed
    choice, never a hidden fact. The frequency expectation is *measured*
    (``Reduce.FREQUENCY`` on v(OUT)); the swing is judged by v(OUT)'s max
    against the supply and its min against 0 V (a saturated collector).

    Validity: 3 V <= ``input_voltage`` <= 6 V and 100 Hz <=
    ``oscillation_frequency`` <= 20 kHz. Below 3 V the drops V_BE / V_CE(sat)
    are no longer small against Vcc (neither the period expression nor the
    saturation margin holds); above 6 V the reverse base-emitter voltage
    Vcc - V_BE approaches the 6 V V_EBO absolute maximum that small-signal
    NPNs such as the 2N3904 specify (it exceeds it from 6.7 V; the template
    stops at 6 V for margin) - a family rating the template asserts
    conservatively, not a datasheet fact grounded in this IR - and the
    generic model has no B-E breakdown, so a passing simulation above it
    would not be evidence. Below 100 Hz C exceeds ~0.65 uF while seeing both
    polarities (non-polar parts only): impractical; above 20 kHz the
    transistors' switching time, absent from the model, and their storage
    time (its ``TR``) become a visible share of the period.

    The model is ngspice's default Gummel-Poon NPN with one storage-time
    parameter, ``TR = 200 ns``: with ``TR = 0`` (no charge storage) the
    saturated transistor turns off instantly and the regenerative switching
    of the ideal circuit is an algebraic jump that ngspice-42 cannot always
    integrate ("Timestep too small" at, measured, 3 V / 100 Hz, 6 V / 100 Hz,
    6 V / 200 Hz, 3 V / 200 Hz - at other points of the same range it
    completed), so the design's own verification could not run inside the
    stated validity. The storage time gives the switching a finite duration.
    Measured on ngspice-42 with it, over the 70-point grid
    {3, 3.5, 4, 4.5, 5, 5.5, 6} V x {100, 150, 200, 300, 500, 1k, 2k, 5k,
    10k, 20k} Hz: every transient completes with the template's own step
    (1 / (200 f)) and the measured frequency is within +1.6 % (6 V, 20 kHz)
    .. +4.9 % (3 V, 100 Hz) of the period expression, changing by less than
    0.05 % with a 10x finer step; the deviation grows as Vcc falls (V_CE(sat)
    is a larger share of the base swing) and shrinks as f rises (the storage
    time delays the edges). ``tol_rel`` = 10 % is chosen from that range:
    the worst admitted point uses half of it. The grid is what was measured;
    the range between grid points is stated as valid on the strength of the
    monotone behaviour across it, not measured pointwise.
    """

    id = "astable"
    title = "BJT astable multivibrator"
    triggers = ("oscillation_frequency",)
    needs = ("oscillation_frequency", "input_voltage")
    serves = ("oscillation_frequency", "input_voltage")

    R_C_OHM = 1_000.0
    R_B_OHM = 10_000.0
    V_BE_V = 0.7
    #: measured on ngspice-42 over the validity range: +1.6 % (6 V, 20 kHz) .. +4.9 % (3 V, 100 Hz); 10 % leaves the worst point half
    TOL_REL = 0.10
    #: the largest measured deviation of the simulated frequency from the period expression inside the validity range
    MEASURED_DEVIATION_MAX = 0.049
    C2_IC_V = -1.0
    SWING_TOL_ABS_V = 0.25
    V_IN_MIN, V_IN_MAX = 3.0, 6.0
    F_MIN, F_MAX = 100.0, 20_000.0
    #: the family V_EBO rating the supply bound is asserted against (conservative, not a datasheet fact in the IR)
    V_EBO_V = 6.0
    MODEL_NAME = "QNPN"
    MODEL_TR_S = 200e-9
    MODEL_CARD = ".model QNPN NPN (TR=200n)"
    MODEL = (
        "generic Gummel-Poon NPN with ngspice's default parameters plus a storage time (`.model QNPN NPN (TR=200n)`: IS = 1e-16 A, "
        "BF = 100, no junction capacitance, no B-E breakdown; TR = 200 ns is the reverse transit time of the small-signal switching "
        "class, not a datasheet fact) - not a 2N3904 vendor model; without TR the saturated transistor turns off instantly and "
        "ngspice-42 aborts the transient at some points of the validity range ('Timestep too small'), with it every point measured "
        "completes; the period depends on the model only through V_BE, V_CE(sat) and TR, and OUT's low level is its saturated "
        "collector (nominal 0 V)"
    )

    def build(self, ir: CircuitIR, inputs: dict[str, DesignInput], unusable: dict[str, str], library: KicadLibrary, *, confirmed: bool) -> Plan:
        t = self.id
        plan = Plan(template=t, title=self.title)
        missing = [k for k in self.needs if k not in inputs]
        if missing:
            return _missing_inputs(plan, "The BJT astable multivibrator template", missing, unusable)
        f_osc, v_in = inputs["oscillation_frequency"], inputs["input_voltage"]
        plan.inputs = {"oscillation_frequency": f_osc, "input_voltage": v_in}
        req_f, req_v = f_osc.requirement.id, v_in.requirement.id
        v, f = v_in.traced.value, f_osc.traced.value
        if not self.V_IN_MIN <= v <= self.V_IN_MAX:
            reverse = v - self.V_BE_V  # what calc.astable.v_be_reverse would report for this supply
            verb = "exceeds" if reverse > self.V_EBO_V else "approaches"
            return _refused(plan, (
                f"supply {v:.12g} V ({req_v}) is outside {self.V_IN_MIN:.12g}..{self.V_IN_MAX:.12g} V: below {self.V_IN_MIN:.12g} V the drops V_BE / V_CE(sat) are not small "
                f"against Vcc (the period expression and the saturation margin do not hold); above {self.V_IN_MAX:.12g} V the reverse base-emitter voltage Vcc - V_BE "
                f"(here {reverse:.12g} V) {verb} the {self.V_EBO_V:.12g} V V_EBO absolute maximum of small-signal NPNs such as the 2N3904 (exceeded from "
                f"{self.V_EBO_V + self.V_BE_V:.12g} V; the template stops at {self.V_IN_MAX:.12g} V for margin; a family rating asserted conservatively, not a datasheet fact "
                f"in this IR), and B-E breakdown is not in the model"
            ))
        if not self.F_MIN <= f <= self.F_MAX:
            return _refused(plan, (
                f"oscillation frequency {f:.12g} Hz ({req_f}) is outside {self.F_MIN:.12g}..{self.F_MAX:.12g} Hz: below {self.F_MIN:.12g} Hz the timing capacitor exceeds "
                f"~0.65 uF while seeing both polarities (non-polar parts only) - impractical; above {self.F_MAX:.12g} Hz the transistors' switching and storage times, "
                f"absent from the generic model, become a visible share of the period"
            ))
        r_c_choice, r_c = _choice(t, "r_c", self.R_C_OHM, "ohm", "collector load; sets the output drive and the ~Vcc/R_c saturation current", confirmed)
        r_b_choice, r_b = _choice(t, "r_b", self.R_B_OHM, "ohm", "base resistor, 10 x R_c so a transistor with beta >= 30 saturates with margin (forced beta ~10); C is solved from it and the frequency", confirmed)
        v_be_choice, v_be = _choice(t, "v_be", self.V_BE_V, "V", "base-emitter drop the period expression assumes for the switching threshold", confirmed)
        tol_choice, tol = _choice(t, "tol_rel", self.TOL_REL, None, (
            "relative tolerance of the frequency expectation: the expression neglects V_CE(sat), whose share of the base swing grows as Vcc falls, and "
            "the model's storage time, whose share of the period grows with f; measured on ngspice-42 over the validity range (3..6 V x 100 Hz..20 kHz, "
            "70-point grid) +1.6 % (6 V, 20 kHz) .. +4.9 % (3 V, 100 Hz), so 10 % leaves the worst admitted point half the tolerance"
        ), confirmed)
        model_choice = Choice("npn_model", self.MODEL)
        model_prov = choice_provenance(t, f"npn_model: {self.MODEL}", confirmed)
        ic_choice, c2_ic = _choice(t, "c2_ic", self.C2_IC_V, "V", "initial voltage across C2 for the transient start (uic: the run starts from the elements' initial conditions instead of the operating point, which would sit in the both-on state): breaks the symmetry of the ideal circuit so the simulated oscillation starts deterministically; a real circuit starts from mismatch and noise", confirmed)
        uic = Traced(value=True, provenance=c2_ic.provenance)  # the same confirmed decision as c2_ic: the flag is what makes the ic act
        swing_choice, swing_tol = _choice(t, "swing_tol_abs", self.SWING_TOL_ABS_V, "V", "absolute tolerance of the OUT high / low level expectations", confirmed)
        plan.choices = [r_c_choice, r_b_choice, v_be_choice, tol_choice, model_choice, ic_choice, swing_choice]
        params: dict[str, Traced] = {"v_in": v_in.traced, "f_osc": f_osc.traced, "r_c": r_c, "r_b": r_b, "v_be": v_be}
        try:
            params["c"] = astable_c_for_frequency(params["f_osc"], params["r_b"], params["v_in"], params["v_be"], ("f_osc", "r_b", "v_in", "v_be"))
            params["f_osc_design"] = astable_frequency(params["r_b"], params["c"], params["v_in"], params["v_be"], ("r_b", "c", "v_in", "v_be"))
            params["tran_step"] = astable_tran_step(params["f_osc"], ("f_osc",))
            params["tran_stop"] = astable_tran_stop(params["f_osc"], ("f_osc",))
            params["tran_start"] = astable_tran_start(params["f_osc"], ("f_osc",))
            params["v_be_reverse"] = astable_v_be_reverse(params["v_in"], params["v_be"], ("v_in", "v_be"))
        except (ValueError, ZeroDivisionError) as e:
            return _refused(plan, f"{e} ({req_f} = {f:.12g} Hz, {req_v} = {v:.12g} V)")
        params["tol_rel"] = tol
        params["c2_ic"] = c2_ic
        params["swing_tol_abs"] = swing_tol
        plan.computed = [(k, params[k]) for k in ("c", "f_osc_design", "tran_step", "tran_stop", "tran_start", "v_be_reverse")]
        c_text = _value_text(params["c"].value)
        try:
            q1 = library_component(library, "Q1", "2N3904", "NPN switching transistor", *NPN, structural_provenance(t, "first switching transistor"), [req_f])
            q2 = library_component(library, "Q2", "2N3904", "NPN switching transistor", *NPN, structural_provenance(t, "second switching transistor"), [req_f])
            r1 = library_component(library, "R1", _value_text(r_c.value), "Q1 collector load", *RESISTOR_THT, structural_provenance(t, "collector load"), [req_f])
            r2 = library_component(library, "R2", _value_text(r_c.value), "Q2 collector load (drives OUT)", *RESISTOR_THT, structural_provenance(t, "collector load"), [req_f])
            r3 = library_component(library, "R3", _value_text(r_b.value), "Q1 base resistor", *RESISTOR_THT, structural_provenance(t, "base resistor"), [req_f])
            r4 = library_component(library, "R4", _value_text(r_b.value), "Q2 base resistor", *RESISTOR_THT, structural_provenance(t, "base resistor"), [req_f])
            c1 = library_component(library, "C1", c_text, "timing capacitor Q1 collector to Q2 base", *CAPACITOR_THT, structural_provenance(t, "timing capacitor"), [req_f])
            c2 = library_component(library, "C2", c_text, "timing capacitor Q2 collector to Q1 base", *CAPACITOR_THT, structural_provenance(t, "timing capacitor"), [req_f])
            j1 = library_component(library, "J1", "Conn_01x03", "VCC / OUT / GND header", *HEADER_3, structural_provenance(t, "board header"), [req_v])
            r1a, r1b = two_terminals(r1)
            r2a, r2b = two_terminals(r2)
            r3a, r3b = two_terminals(r3)
            r4a, r4b = two_terminals(r4)
            c1a, c1b = two_terminals(c1)
            c2a, c2b = two_terminals(c2)
            require_pins(j1, ("1", "2", "3"))
        except TemplateRefusal as e:
            return _refused(plan, str(e))
        q_pins: dict[str, tuple[str, str, str]] = {}
        for q in (q1, q2):
            found = tuple(pin_by_name(q, name) for name in ("C", "B", "E"))
            if any(n is None for n in found) or len(q.pins) != 3:
                return _refused(plan, f"{NPN[0][0]}:{NPN[0][1]} pins are not named C / B / E in this library (pins: {[(p.number, p.name) for p in q.pins]}); the template will not guess the transistor's pinout")
            q_pins[q.ref] = found  # type: ignore[assignment]
        for r, key in ((r1, "r_c"), (r2, "r_c"), (r3, "r_b"), (r4, "r_b")):
            r.electrical["resistance"] = params[key]
            r.spice = SpiceBinding(device=SpiceDevice.R, value=params[key], pin_order=[p.number for p in r.pins], provenance=structural_provenance(t, "ideal resistor at the design value"))
        for cap in (c1, c2):
            cap.electrical["capacitance"] = params["c"]
        c1.spice = SpiceBinding(device=SpiceDevice.C, value=params["c"], pin_order=[p.number for p in c1.pins], provenance=structural_provenance(t, "ideal capacitor at the design value"))
        c2.spice = SpiceBinding(
            device=SpiceDevice.C, value=params["c"], pin_order=[p.number for p in c2.pins], params={"ic": params["c2_ic"]},
            provenance=structural_provenance(t, "ideal capacitor at the design value; its initial condition starts the transient (uic)"),
        )
        card = Traced(value=self.MODEL_CARD, provenance=model_prov)
        for q in (q1, q2):
            q.spice = SpiceBinding(
                device=SpiceDevice.Q, model_name=self.MODEL_NAME, model_card=card, pin_order=list(q_pins[q.ref]),
                provenance=structural_provenance(t, "NPN on the confirmed generic model; SPICE node order C B E from the library pin names"),
            )
        j1.spice = SpiceBinding(exclude=True, exclude_reason="connector, no electrical model", provenance=structural_provenance(t, "header excluded from the netlist"))
        (q1c, q1b, q1e), (q2c, q2b, q2e) = q_pins["Q1"], q_pins["Q2"]
        nets = [
            _net("VCC", NetKind.POWER, [("J1", "1"), ("R1", r1a), ("R2", r2a), ("R3", r3a), ("R4", r4a)], t, [req_v]),
            _net("Q1_C", NetKind.SIGNAL, [("R1", r1b), ("Q1", q1c), ("C1", c1a)], t),
            _net("Q2_B", NetKind.SIGNAL, [("C1", c1b), ("R4", r4b), ("Q2", q2b)], t),
            _net("OUT", NetKind.SIGNAL, [("R2", r2b), ("Q2", q2c), ("C2", c2a), ("J1", "2")], t, [req_f]),
            _net("Q1_B", NetKind.SIGNAL, [("C2", c2b), ("R3", r3b), ("Q1", q1b)], t),
            _net("GND", NetKind.GROUND, [("J1", "3"), ("Q1", q1e), ("Q2", q2e)], t),
        ]
        sim = SimulationSetup(
            stimuli=[Stimulus(id="VIN", source="voltage", net="VCC", reference_net="GND", kind=StimulusKind.DC, value=params["v_in"], provenance=structural_provenance(t, "the supply at v_in"), serves_requirements=[req_v])],
            analyses=[AnalysisSpec(
                id="tran", kind=SpiceAnalysis.TRAN,
                params={"step": params["tran_step"], "stop": params["tran_stop"], "start": params["tran_start"], "uic": uic},
                provenance=structural_provenance(t, "transient of 20 periods, the last 10 saved, 200 points per period"),
            )],
            expectations=[
                Expectation(
                    id="f_osc", analysis_id="tran", vector="v(OUT)", reduce=Reduce.FREQUENCY, nominal=params["f_osc_design"], tol_rel=params["tol_rel"], requirement_id=req_f,
                    provenance=structural_provenance(t, f"the frequency measured from v(OUT)'s rising mid-level crossings over the saved window verifies {req_f}"),
                ),
                Expectation(
                    id="out_high", analysis_id="tran", vector="v(OUT)", reduce=Reduce.MAX, nominal=params["v_in"], tol_abs=params["swing_tol_abs"], requirement_id=req_v,
                    provenance=structural_provenance(t, f"OUT's high level is the supply through R2 (Q2 off); verifies {req_v}"),
                ),
                Expectation(
                    id="out_low", analysis_id="tran", vector="v(OUT)", reduce=Reduce.MIN, nominal=Traced(value=0.0, unit="V", provenance=model_prov), tol_abs=params["swing_tol_abs"],
                    provenance=structural_provenance(t, "OUT's low level is Q2's saturated collector (nominal 0 V per the confirmed model choice; no requirement states it)"),
                ),
            ],
        )
        topology = Topology(
            name="BJT astable multivibrator", domains=[CircuitDomain.ANALOG],
            rationale="collector-coupled astable: two cross-coupled saturating NPN switches with RC timing; f = 1 / (2 R_b C ln((2 Vcc - V_BE) / (Vcc - V_BE))), C solved from the confirmed frequency",
            provenance=structural_provenance(t, "selected by oscillation_frequency with input_voltage"),
            blocks=[Block(
                id="astable", function="collector-coupled astable multivibrator", domain=CircuitDomain.ANALOG,
                component_refs=["Q1", "Q2", "R1", "R2", "R3", "R4", "C1", "C2"], input_nets=["VCC"], output_nets=["OUT"], provenance=structural_provenance(t, "block"),
            )],
        )
        constraints = [
            Constraint(
                id="c.astable.reverse_vbe", kind=ConstraintKind.ELECTRICAL, target="astable", parameters={"reverse_voltage": params["v_be_reverse"]},
                description=(
                    f"each base-emitter junction is reverse biased to about Vcc - V_BE = {params['v_be_reverse'].value:.12g} V once per period; the generic model does not simulate "
                    f"B-E breakdown; the fitted transistor's V_EBO rating must exceed it - not verified here"
                ),
                provenance=structural_provenance(t, "template validity condition"),
            ),
            Constraint(
                id="c.astable.startup", kind=ConstraintKind.ELECTRICAL, target="astable",
                description="the ideal symmetric circuit has a metastable both-on state; the simulation is started by the C2 initial condition (uic), real circuits start on mismatch and noise",
                provenance=structural_provenance(t, "what the simulated start-up means"),
            ),
            Constraint(
                id="c.astable.nonpolar_caps", kind=ConstraintKind.ELECTRICAL, target="astable",
                description="C1 / C2 see both polarities every period: non-polar ceramic / film parts only",
                provenance=structural_provenance(t, "part selection condition"),
            ),
        ]
        components = [q1, q2, r1, r2, r3, r4, c1, c2, j1]
        plan.parts = [_part_line(x) for x in components]
        plan.nets = [_net_line(n) for n in nets]
        plan.simulation = [
            f"stimulus VIN: dc {v:.12g} V on VCC; Q1 / Q2 on {self.MODEL_CARD!r}; C2 ic = {self.C2_IC_V:.12g} V; "
            f"analysis tran {params['tran_step'].value:.12g} {params['tran_stop'].value:.12g} {params['tran_start'].value:.12g} uic (s)",
            f"expectation f_osc: frequency of v(OUT) rising edges = {params['f_osc_design'].value:.12g} Hz +/- {self.TOL_REL:.0%} verifies {req_f} (fewer than 3 edges is 'no oscillation detected', FAIL)",
            f"expectation out_high: max v(OUT) = {v:.12g} V +/- {self.SWING_TOL_ABS_V:.12g} V verifies {req_v}",
            f"expectation out_low: min v(OUT) = 0 V +/- {self.SWING_TOL_ABS_V:.12g} V (saturated collector, no requirement)",
        ]
        plan.changes = _changes(t, self.title, topology, components, nets, params, sim, constraints)
        return plan

    # --- report hooks (views: numbers recomputed from ir.parameters with the formulas shown, nothing written) ---

    def _numbers(self, ir: CircuitIR) -> dict[str, float | None]:
        v_cc, v_be, r_b, r_c, c, f, f_design = (parameter_value(ir, k) for k in ("v_in", "v_be", "r_b", "r_c", "c", "f_osc", "f_osc_design"))
        ln_term = math.log((2.0 * v_cc - v_be) / (v_cc - v_be)) if _known(v_cc, v_be) and v_cc > v_be >= 0 else None
        return {
            "v_cc": v_cc, "v_be": v_be, "r_b": r_b, "r_c": r_c, "c": c, "f": f, "f_design": f_design, "ln": ln_term,
            "t_half": _mul(r_b, c, ln_term), "tau_b": _mul(r_b, c), "tau_c": _mul(r_c, c), "v_rev": _sub(v_cc, v_be),
            "i_c_sat": _div(v_cc, r_c), "i_b": _div(_sub(v_cc, v_be), r_b),
            "p_rc": _div(_mul(v_cc, v_cc), r_c), "p_rb": _div(_mul(_sub(v_cc, v_be), _sub(v_cc, v_be)), r_b),
            "tol_rel": parameter_value(ir, "tol_rel"), "c2_ic": parameter_value(ir, "c2_ic"), "swing_tol": parameter_value(ir, "swing_tol_abs"),
            "tran_step": parameter_value(ir, "tran_step"), "tran_stop": parameter_value(ir, "tran_stop"), "tran_start": parameter_value(ir, "tran_start"),
        }

    def _c_at(self, v_cc: float, f: float, r_b: float | None, v_be: float | None) -> float | None:
        if not _known(r_b, v_be) or v_cc <= v_be:
            return None
        return 1.0 / (2.0 * f * r_b * math.log((2.0 * v_cc - v_be) / (v_cc - v_be)))

    def theory(self, ir: CircuitIR) -> list[TheorySection]:
        n = self._numbers(ir)
        v_cc, v_be, r_b, r_c, c = n["v_cc"], n["v_be"], n["r_b"], n["r_c"], n["c"]
        beta_forced = _div(n["i_c_sat"], n["i_b"])
        t_rise = _mul(2.2, n["tau_c"])
        rise_share = _mul(100.0, _div(t_rise, n["t_half"]))
        ln2_gap = _mul(100.0, _div(_sub(n["ln"], math.log(2.0)), math.log(2.0)))
        bounds = [(self.V_IN_MIN, self.F_MIN), (self.V_IN_MAX, self.F_MIN), (self.V_IN_MIN, self.F_MAX), (self.V_IN_MAX, self.F_MAX)]
        bound_text = ", ".join(f"{quantity(v, 'V')}/{quantity(f, 'Hz')} → C = {quantity(self._c_at(v, f, r_b, v_be), 'F')}" for v, f in bounds)
        c_e12 = nearest_e12(c)
        f_e12 = _div(1.0, _mul(2.0, r_b, c_e12, n["ln"]))
        return [
            TheorySection("동작 원리: 컬렉터 결합 비안정 멀티바이브레이터", (
                "```\n" + astable_drawing(quantity(r_c, "ohm"), quantity(r_b, "ohm")) + "\n```\n\n"
                "두 NPN 트랜지스터가 교차 결합되어 있어 한쪽이 포화(ON)이면 다른 쪽은 차단(OFF)입니다. 안정 상태가 없으므로 두 상태를 번갈아 오가며 각 컬렉터에 구형파가 나옵니다.\n\n"
                "1. Q2가 막 켜졌다고 하자. Q2의 컬렉터(OUT)는 V_cc에서 V_CE(sat) ≈ 0 V로 떨어진다.\n"
                f"2. 그 직전 C2 양단 전압은 V_C2 = V(OUT) − V(Q1_B) = V_cc − V_BE 였다(OUT은 V_cc, Q1 베이스는 도통 중이라 V_BE). 커패시터 전압은 순간적으로 바뀌지 않으므로 Q1 베이스는 0 − (V_cc − V_BE) = −(V_cc − V_BE) = −{quantity(n['v_rev'], 'V')} 로 끌려 내려가고 Q1은 꺼진다.\n"
                f"3. Q1 베이스는 R3(= R_b)를 통해 V_cc를 향해 시정수 τ_b = R_b·C = {quantity(n['tau_b'], 's')} 로 지수적으로 충전된다.\n"
                f"4. 베이스 전압이 V_BE = {quantity(v_be, 'V')} 에 이르면 Q1이 켜지고, 같은 과정이 반대편에서 반복된다."
            )),
            TheorySection("반주기(half period) 유도", (
                "RC 충전의 일반해: v(t) = V_final − (V_final − V_start)·e^(−t/τ).\n"
                "V_start = −(V_cc − V_BE), V_final = V_cc, 목표 v = V_BE 를 넣으면\n\n"
                "    T_half = R_b · C · ln( (V_final − V_start) / (V_final − V_BE) )\n"
                "           = R_b · C · ln( (2·V_cc − V_BE) / (V_cc − V_BE) )\n\n"
                "대칭 회로이므로 주기 T = 2·T_half, 주파수는\n\n"
                "    f = 1 / ( 2 · R_b · C · ln((2·V_cc − V_BE)/(V_cc − V_BE)) )      … (식 1)\n\n"
                f"V_cc ≫ V_BE 이면 ln(2) = 0.693 이 되어 교과서식 T ≈ 1.386·R_b·C 가 됩니다. 이 설계는 근사식이 아니라 (식 1)을 씁니다. "
                f"{quantity(v_cc, 'V')} 에서 두 식의 ln 항 차이는 {number(ln2_gap, 3)} % ({number(n['ln'], 4)} 대 0.6931) 라 무시할 수 없기 때문입니다."
            )),
            TheorySection("이 설계의 수치", (
                "| 항목 | 식 | 값 |\n|---|---|---|\n"
                f"| ln 항 | ln((2·{number(v_cc)} − {number(v_be)})/({number(v_cc)} − {number(v_be)})) = ln({number(_sub(_mul(2.0, v_cc), v_be))}/{number(_sub(v_cc, v_be))}) | {number(n['ln'])} |\n"
                f"| 타이밍 커패시터 (계산기 `calc.astable.c_for_frequency`) | C = 1/(2·f·R_b·ln항) = 1/(2·{number(n['f'])}·{number(r_b)}·{number(n['ln'])}) | **{quantity(c, 'F')}** |\n"
                f"| 설계 주파수 (계산기 `calc.astable.f`) | (식 1) | {quantity(n['f_design'], 'Hz')} |\n"
                f"| 반주기 | T_half = R_b·C·ln항 | {quantity(n['t_half'], 's')} |\n"
                f"| 베이스 시정수 | τ_b = R_b·C | {quantity(n['tau_b'], 's')} |\n"
                f"| 베이스 역전압 (계산기 `calc.astable.v_be_reverse`) | V_cc − V_BE | **{quantity(n['v_rev'], 'V')}** |\n\n"
                f"C 는 계산값 그대로입니다(E 계열 반올림 없음). 가장 가까운 E12 값 {quantity(c_e12, 'F')} 를 쓰면 (식 1)로 f = {quantity(f_e12, 'Hz')} 가 됩니다."
            )),
            TheorySection("저항값 선택 근거(설계 선택값)", (
                f"- 컬렉터 저항 R_c = {quantity(r_c, 'ohm')}: 포화 시 컬렉터 전류 I_C(sat) ≈ V_cc/R_c = {quantity(n['i_c_sat'], 'A')}. 출력 구동 능력과 소비전력의 절충.\n"
                f"- 베이스 저항 R_b = {quantity(r_b, 'ohm')}: 베이스 전류 I_B ≈ (V_cc − V_BE)/R_b = {quantity(n['i_b'], 'A')}. "
                f"강제 β = I_C/I_B ≈ {number(beta_forced, 3)} 이므로 h_FE ≥ 30 인 트랜지스터라면 여유 있게 포화합니다(2N3904 계열의 h_FE 는 수십~수백: 계열 상식이며 이 IR 에 데이터시트로 근거를 둔 값은 아님).\n"
                f"- 소비전력: P(R_c) = V_cc²/R_c = {quantity(n['p_rc'], 'W')}, P(R_b) = (V_cc − V_BE)²/R_b = {quantity(n['p_rb'], 'W')}. 1/4 W 저항이면 충분합니다."
            )),
            TheorySection("출력 파형의 형태", (
                f"- 고레벨: 컬렉터가 R_c를 통해 V_cc까지 올라가므로 무부하에서 V_OH ≈ V_cc = {quantity(v_cc, 'V')}.\n"
                "- 저레벨: 포화 컬렉터 V_OL = V_CE(sat) (모델이 정하는 값, 공칭 0 V).\n"
                f"- 상승 에지가 둥근 이유: 컬렉터가 올라갈 때 반대편 타이밍 커패시터를 R_c를 통해 재충전하므로 τ_c = R_c·C = {quantity(n['tau_c'], 's')}, "
                f"10–90 % 상승시간 ≈ 2.2·τ_c ≈ {quantity(t_rise, 's')}. 반주기({quantity(n['t_half'], 's')})의 약 {number(rise_share, 2)} % 가 이 완만한 상승에 쓰입니다. 하강 에지는 트랜지스터가 켜지며 순간적으로 떨어집니다.\n"
                f"- 베이스 파형: 켜지는 순간 −{quantity(n['v_rev'], 'V')} 까지 내려갔다가 지수적으로 회복해 +{quantity(v_be, 'V')} 에서 스위칭합니다.\n\n"
                "실제 시뮬레이션에서 측정된 레벨과 주파수는 최종 보고서의 '이론값 대 시뮬레이션' 표에 있습니다."
            )),
            TheorySection("유효 범위(템플릿이 거부하는 조건)와 그 이유", (
                "| 조건 | 근거 |\n|---|---|\n"
                f"| {quantity(self.V_IN_MIN, 'V')} ≤ V_cc ≤ {quantity(self.V_IN_MAX, 'V')} | {quantity(self.V_IN_MIN, 'V')} 미만: V_BE·V_CE(sat)가 V_cc에 비해 커져 (식 1)과 포화 여유가 무너짐. "
                f"{quantity(self.V_IN_MAX, 'V')} 초과: 베이스-이미터 역전압 V_cc − V_BE 가 소신호 NPN(2N3904 계열)의 V_EBO 절대최대정격 {quantity(self.V_EBO_V, 'V')} 에 접근·초과"
                f"({quantity(self.V_EBO_V + self.V_BE_V, 'V')} 부터 초과). 범용 모델은 항복을 모사하지 않으므로 그 위에서의 PASS는 증거가 아님. "
                "이 V_EBO 는 계열 정격이며 이 IR 에 데이터시트로 근거를 둔 사실은 아님(검증되지 않음). |\n"
                f"| {quantity(self.F_MIN, 'Hz')} ≤ f ≤ {quantity(self.F_MAX, 'Hz')} | {quantity(self.F_MIN, 'Hz')} 미만: 0.6 µF 이상의 무극성 커패시터가 필요(타이밍 C는 양극성 전압을 받음). "
                f"{quantity(self.F_MAX, 'Hz')} 초과: 트랜지스터의 스위칭·저장시간이 주기의 무시 못 할 비율이 됨 |\n\n"
                f"경계값에서의 C (이 설계의 R_b, V_BE 로 (식 1)을 풀면): {bound_text}."
            )),
            TheorySection("모델과 시뮬레이션 이론", (
                f"**트랜지스터 모델** `{_model_card_text(ir)}` — Gummel-Poon NPN, ngspice 기본 파라미터(I_S = 1e-16 A, B_F = 100, 접합 용량 없음, B-E 항복 없음)에 "
                f"역방향 통과시간 T_R = {quantity(self.MODEL_TR_S, 's')}(저장시간)만 추가한 범용 모델이며 2N3904 벤더 모델이 아닙니다(사용자가 확인한 선택값). "
                "T_R이 없으면 포화 트랜지스터가 순간적으로 꺼져 ngspice-42가 유효 범위 일부에서 해석을 중단(\"Timestep too small\")했고, T_R을 넣으면 70개 격자점 전부 완주했습니다(템플릿 개발 시 측정).\n\n"
                f"**왜 uic(초기조건 시작)를 쓰는가** 이상적인 대칭 회로에는 두 트랜지스터가 모두 켜진 준안정 동작점이 존재합니다. SPICE의 DC 동작점 해석은 정확히 그 점을 찾고, 과도해석은 섭동이 없으면 거기 머뭅니다. "
                f"그래서 `tran … uic`로 동작점 계산을 건너뛰고 C2의 초기전압을 {quantity(n['c2_ic'], 'V')} 로 두어 대칭을 깹니다(확인된 시뮬레이션 선택값). 실물은 부품 불일치와 잡음으로 스스로 시작합니다.\n\n"
                "**과도해석 창** (계산기 `calc.astable.tran_*`)\n\n"
                f"    step  = 1/(200·f) = {quantity(n['tran_step'], 's')}   (주기당 200점)\n"
                f"    stop  = 20/f      = {quantity(n['tran_stop'], 's')}   (20주기)\n"
                f"    start = 10/f      = {quantity(n['tran_start'], 's')}   (앞 10주기는 시동 구간이라 저장하지 않음)\n\n"
                f"저장 구간 {quantity(_sub(n['tran_stop'], n['tran_start']), 's')} 에 약 {number(_mul(_sub(n['tran_stop'], n['tran_start']), n['f']), 3)} 주기가 들어갑니다."
            )),
            TheorySection("편차의 물리적 원인", (
                "(식 1)이 무시한 것 두 가지가 반대 방향으로 작용합니다.\n\n"
                f"- V_CE(sat) > 0: 베이스 시작 전압이 −(V_cc − V_BE) + V_CE(sat)로 덜 음수가 되고, 트랜지스터는 {quantity(v_be, 'V')} 보다 조금 낮은 V_BE에서 켜지기 시작하므로 반주기가 짧아진다 → 주파수 상승.\n"
                "- T_R(저장시간): 꺼짐이 지연되어 반주기가 길어진다 → 주파수 하강.\n\n"
                f"템플릿 개발 시 ngspice-42 로 유효 범위 70개 격자점을 측정한 결과 순효과는 +1.6 % ({quantity(self.V_IN_MAX, 'V')}, {quantity(self.F_MAX, 'Hz')}: 저장시간 비중 최대)에서 "
                f"+{number(self.MEASURED_DEVIATION_MAX * 100, 2)} % ({quantity(self.V_IN_MIN, 'V')}, {quantity(self.F_MIN, 'Hz')}: V_CE(sat) 비중 최대)까지였고, "
                f"그래서 허용오차를 {number(_mul(100.0, n['tol_rel']), 3)} % 로 잡았습니다(최악점도 허용치의 절반). 이 설계 자체의 측정 편차는 최종 보고서에 있습니다."
            )),
        ]

    def part_notes(self, ir: CircuitIR) -> dict[str, PartNote]:
        n = self._numbers(ir)
        v_cc, v_be, r_b, r_c, c = n["v_cc"], n["v_be"], n["r_b"], n["r_c"], n["c"]
        beta_forced = _div(n["i_c_sat"], n["i_b"])
        f_plus10 = _mul(n["f_design"], 1.0 / 1.1)
        q_note = lambda which, out: PartNote(  # noqa: E731
            role=f"{which}: {out}",
            why=(
                f"범용 소신호 NPN `Transistor_BJT:2N3904` (TO-92): 필요한 포화 전류 I_C(sat) ≈ V_cc/R_c = {quantity(n['i_c_sat'], 'A')}, 강제 β ≈ {number(beta_forced, 3)} 이라 h_FE ≥ 30 이면 충분히 포화. "
                "KiCad 심볼이 핀 이름 E / B / C 를 제공하므로 템플릿이 이름으로 배선(핀 번호로 추측하지 않음). TO-92 THT 는 손납땜·브레드보드 검증이 쉬운 풋프린트."
            ),
            criteria=[
                "NPN (PNP 는 극성이 반대라 불가)",
                f"V_CEO ≥ 2·V_cc = {quantity(_mul(2.0, v_cc), 'V')} (여유 2배)",
                f"I_C(max) ≥ 10 × V_cc/R_c = {quantity(_mul(10.0, n['i_c_sat']), 'A')}",
                f"h_FE ≥ 30 at I_C = {quantity(n['i_c_sat'], 'A')}",
                f"V_EBO ≥ V_cc − V_BE = {quantity(n['v_rev'], 'V')} (매 주기 베이스-이미터가 이만큼 역바이어스됨; 제약 조건 `c.astable.reverse_vbe`)",
                "TO-92 핀 순서 E-B-C (KiCad 심볼 `Q_NPN_EBC` 와 같은 순서; 다르면 심볼·풋프린트를 바꾸고 넷을 다시 확인)",
            ],
            substitutes=[
                unverified("PN2222A", "TO-92, 핀 순서 E-B-C 로 2N3904 와 같아 심볼 교체만으로 대체 가능"),
                unverified("2N2222A (금속 캔 TO-18)", "핀 순서 C-B-E 이므로 심볼과 풋프린트를 모두 바꾸어야 함"),
                unverified("BC547 / BC548", "TO-92 이지만 핀 순서 C-B-E: `Transistor_BJT:BC547` 심볼로 바꾸고 배선을 다시 확인"),
                unverified("2N4401", "TO-92, 핀 순서 E-B-C"),
            ],
        )
        r_c_note = lambda which, out: PartNote(  # noqa: E731
            role=f"{which}: {out}",
            why=f"R_c = {quantity(r_c, 'ohm')} (템플릿 선택값, 사용자 확인): I_C(sat) ≈ {quantity(n['i_c_sat'], 'A')} 로 출력 구동과 소비전력 {quantity(n['p_rc'], 'W')} 의 절충. 축형 THT 는 손납땜 기판용.",
            criteria=_resistor_criteria(r_c, n["p_rc"], "1 % 또는 5 % (컬렉터 저항은 주파수에 들어가지 않음)", chosen=_is_choice(ir, "r_c")),
            substitutes=_resistor_substitutes(r_c, "Resistor_THT"),
        )
        r_b_note = lambda which, out: PartNote(  # noqa: E731
            role=f"{which}: {out}",
            why=f"R_b = {quantity(r_b, 'ohm')} = 10 × R_c (템플릿 선택값): I_B ≈ {quantity(n['i_b'], 'A')}, 강제 β ≈ {number(beta_forced, 3)} 로 포화 여유 확보. 반주기 T_half = R_b·C·ln항 에 직접 들어가는 타이밍 저항. 축형 THT.",
            criteria=_resistor_criteria(r_b, n["p_rb"], "1 % 권장 (Δf/f ≈ −ΔR_b/R_b: 저항 오차가 그대로 주파수 오차가 됨)", chosen=_is_choice(ir, "r_b")),
            substitutes=_resistor_substitutes(r_b, "Resistor_THT"),
        )
        c_note = lambda which, out: PartNote(  # noqa: E731
            role=f"{which}: {out}",
            why=f"C = {quantity(c, 'F')} 는 계산기 `calc.astable.c_for_frequency` 의 값 그대로(E 계열 반올림 없음). 매 주기 양극성 전압을 받으므로 무극성이어야 하고(제약 조건 `c.astable.nonpolar_caps`), 5 mm 디스크 THT 는 손납땜 기판용.",
            criteria=[
                "무극성 (세라믹 / 필름; 전해 커패시터 불가 - 양극성 전압을 받음)",
                f"정전용량 {quantity(c, 'F')} ± 공차: Δf/f ≈ −ΔC/C 이므로 +10 % 이면 f ≈ {quantity(f_plus10, 'Hz')} (−9.1 %); 허용치 {number(_mul(100.0, n['tol_rel']), 3)} % 안에 들려면 5 % 이하 권장",
                f"정격 전압 ≥ 2·V_cc = {quantity(_mul(2.0, v_cc), 'V')}",
                "5 mm 피치 2핀 풋프린트 (Capacitor_THT:C_Disc_D5.0mm_W2.5mm_P5.00mm)",
            ],
            substitutes=[
                unverified("X7R 세라믹 5 mm 디스크", "용량이 온도·전압에 따라 변하므로 주파수 정확도가 필요하면 C0G/NP0 또는 필름"),
                unverified("C0G/NP0 세라믹 5 mm 디스크", "이 값은 C0G 로는 크므로 여러 개 병렬이 필요할 수 있음"),
                unverified("폴리에스터 / 폴리프로필렌 필름 박스 커패시터 (5 mm 피치)", "타이밍용으로 가장 안정적"),
            ],
        )
        return {
            "Q1": q_note("Q1", "첫 번째 스위칭 트랜지스터 (컬렉터 Q1_C 가 C1 을 통해 Q2 베이스를 구동)"),
            "Q2": q_note("Q2", "두 번째 스위칭 트랜지스터 (컬렉터가 OUT; C2 를 통해 Q1 베이스를 구동)"),
            "R1": r_c_note("R1", "Q1 컬렉터 부하 저항"),
            "R2": r_c_note("R2", "Q2 컬렉터 부하 저항, OUT 을 V_cc 로 끌어올림"),
            "R3": r_b_note("R3", "Q1 베이스 저항, 타이밍 (C2 와 함께 반주기를 정함)"),
            "R4": r_b_note("R4", "Q2 베이스 저항, 타이밍 (C1 과 함께 반주기를 정함)"),
            "C1": c_note("C1", "타이밍 커패시터 Q1 컬렉터 → Q2 베이스"),
            "C2": c_note("C2", "타이밍 커패시터 Q2 컬렉터 → Q1 베이스 (초기전압으로 시뮬레이션 시동)"),
            "J1": _header_note("J1: VCC / OUT / GND 헤더", "1 = VCC, 2 = OUT, 3 = GND", f"OUT 은 R2 = {quantity(r_c, 'ohm')} 를 통한 출력이라 무거운 부하를 달면 고레벨이 내려감."),
        }


TEMPLATES: list[Template] = [DividerTemplate(), LedTemplate(), RcLowpassTemplate(), AstableTemplate()]


def template_keys_text() -> str:
    return "; ".join(f"{t.id} needs {' + '.join(t.needs)}" for t in TEMPLATES)


@dataclass
class LateLoad:
    """What a divider-templated design does with ``output_current`` requirements stated after it was built."""

    #: the ``nets`` change that makes VOUT serve the 0 A requirement(s), when there is one to make
    changes: list[DesignChange] = field(default_factory=list)
    #: requirement ids the change serves
    served: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def late_load_changes(ir: CircuitIR) -> LateLoad:
    """The divider's rule applied after the build: a confirmed ``output_current`` that reads exactly 0 A is served by VOUT.

    Only a design whose ``VOUT`` net the divider template made qualifies
    (``provenance.tool`` = ``design.template.divider``); the change is a
    proposal on ``nets`` (the whole list, VOUT's ``serves_requirements``
    extended), so the orchestrator applies it like any other. A non-zero or
    unreadable load is named as one the design cannot serve - nothing is
    guessed and the reviewer still reports it.
    """
    out = LateLoad()
    vout = ir.net("VOUT")
    if vout is None or vout.provenance.tool != template_tool(DividerTemplate.id):
        return out
    reqs = [r for r in ir.requirements.requirements if r.key in KEY_ALIASES["output_current"] and r.id not in vout.serves_requirements]
    for r in reqs:
        traced, why = read_value(r, UNIT_OF["output_current"])
        if traced is None:
            if not (r.value is not None and r.value.provenance.needs_verification):  # an unconfirmed value is nobody's requirement yet
                out.notes.append(f"{r.id} not served by the divider: {why}")
            continue
        if traced.value != 0.0:
            out.notes.append(f"{r.id} = {traced.value:.12g} A: a resistive divider cannot supply a load; VOUT does not serve it "
                             f"(change the requirement to 0 A, choose another design, or provide the circuit yourself)")
            continue
        out.served.append(r.id)
    if out.served:
        nets = [n.model_copy(update={"serves_requirements": [*n.serves_requirements, *out.served]}) if n is vout else n for n in ir.nets]
        out.changes.append(DesignChange(
            description=f"VOUT serves {', '.join(out.served)} (0 A: a high-impedance reference)", target="nets", operation="set", payload=nets,
            rationale=f"template {DividerTemplate.id} v{TEMPLATE_VERSION}: output_current = 0 A stated after the build is served by the unloaded output",
        ))
        out.notes.append(f"{', '.join(out.served)} = 0 A stated after the build: VOUT is a high-impedance reference and is proposed to serve it")
    return out


def design_from_requirements(
    ir: CircuitIR,
    library: KicadLibrary,
    inputs: dict[str, DesignInput] | None = None,
    unusable: dict[str, str] | None = None,
    *,
    confirmed: bool = False,
) -> Plan | None:
    """The plan of the one template the confirmed requirements select, or ``None`` when none is triggered.

    Two or more triggered templates give a plan that refuses (ambiguous); a
    template that does not serve every confirmed design requirement gives a
    plan that refuses with a non-required question under each such
    requirement's own key.
    """
    if inputs is None or unusable is None:
        inputs, unusable = read_inputs(ir)
    triggered = [t for t in TEMPLATES if t.triggered(inputs)]
    if not triggered:
        return None
    if len(triggered) > 1:
        ids = [t.id for t in triggered]
        plan = Plan(template="+".join(ids), title="ambiguous")
        plan.notes.append(f"ambiguous: templates {ids} all match the confirmed requirements ({', '.join(sorted(inputs))}); refusing to guess - state the requirements of one circuit")
        return plan
    template = triggered[0]
    missing = [k for k in template.needs if k not in inputs]
    if not missing:
        refusals = template.refusals(ir, inputs, unusable)
        if refusals:
            plan = Plan(template=template.id, title=template.title, questions=refusals)
            return _refused(plan, "; ".join(q.rationale for q in refusals))
    return template.build(ir, inputs, unusable, library, confirmed=confirmed)


__all__ = [
    "CAPACITOR",
    "CAPACITOR_THT",
    "E12_MANTISSAS",
    "HEADER_2",
    "HEADER_3",
    "LED",
    "NPN",
    "RESISTOR",
    "RESISTOR_THT",
    "TEMPLATES",
    "AstableTemplate",
    "DividerTemplate",
    "LedTemplate",
    "RcLowpassTemplate",
    "LateLoad",
    "astable_drawing",
    "design_from_requirements",
    "late_load_changes",
    "nearest_e12",
    "template_keys_text",
]
