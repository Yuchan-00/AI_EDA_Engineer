"""The three verified circuit templates and the closed-world selection between them.

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
  chosen, R computed), checked by an ac sweep at the corner.
"""

from __future__ import annotations

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

from ai_eda.design.base import TEMPLATE_VERSION, Choice, DesignChange, Plan, Template, choice_provenance, structural_provenance, template_tool, unserved_requirements
from ai_eda.design.inputs import KEY_ALIASES, UNIT_OF, DesignInput, read_inputs, read_value
from ai_eda.design.library_parts import TemplateRefusal, library_component, pin_by_name, require_pins, two_terminals

RESISTOR = (("Device", "R"), ("Resistor_SMD", "R_0603_1608Metric"))
CAPACITOR = (("Device", "C"), ("Capacitor_SMD", "C_0603_1608Metric"))
LED = (("Device", "LED"), ("LED_SMD", "LED_0603_1608Metric"))
HEADER_2 = (("Connector_Generic", "Conn_01x02"), ("Connector_PinHeader_2.54mm", "PinHeader_1x02_P2.54mm_Vertical"))
HEADER_3 = (("Connector_Generic", "Conn_01x03"), ("Connector_PinHeader_2.54mm", "PinHeader_1x03_P2.54mm_Vertical"))


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
            for k in missing:
                if k in unusable:
                    plan.notes.append(f"{k} not usable: {unusable[k]}")
                else:
                    plan.questions.append(MissingInformation(
                        key=k, question=f"The LED indicator template needs {k} in {UNIT_OF[k]}: answer {k}=<value {UNIT_OF[k]}> (e.g. {k}=\"{_example(k)}\")",
                        rationale="template input",
                    ))
            return _refused(plan, f"input(s) {missing} missing")
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


def _example(key: str) -> str:
    return {"input_voltage": "5 V", "led_forward_voltage": "2 V", "led_forward_current": "10 mA"}.get(key, f"<value {UNIT_OF[key]}>")


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


TEMPLATES: list[Template] = [DividerTemplate(), LedTemplate(), RcLowpassTemplate()]


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
    "HEADER_2",
    "HEADER_3",
    "LED",
    "RESISTOR",
    "TEMPLATES",
    "DividerTemplate",
    "LedTemplate",
    "RcLowpassTemplate",
    "LateLoad",
    "design_from_requirements",
    "late_load_changes",
    "template_keys_text",
]
