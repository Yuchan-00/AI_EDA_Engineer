"""I2C pull-up resistors from the bus supply, the clock, the bus capacitance and the allowed rise time (D9 catalog, tier 1).

R1 (SDA) and R2 (SCL) pull the open-drain lines to VDD. The rise is an RC charge through the pull-up into the bus
capacitance; I2C measures the rise time between 0.3 V_DD and 0.7 V_DD, so t_r = R C_b ln(0.7 / 0.3) and the largest
pull-up that meets ``bus_rise_time`` is R_max = t_r / (0.8473 C_b). The smallest one lets the driver still reach
V_OL while sinking at most I_OL: R_min = (V_DD - V_OL) / I_OL. The design takes R = max(pullup_fraction x R_max, R_min)
(0.8: margin for the resistor tolerance and the capacitance estimate; R_min when the margin would undercut it) and
refuses when R_min > R_max (no value serves the bus).

Simulation (SDA only; SCL is the same R into the same bus): the bus capacitance and the driver are not board
parts, so they are simulation-only elements (the board itself has no ground pin: GND is a net without pins, the simulation's reference) (``SimulationSetup.elements``, PLAN decision 16 b): ``CBUS`` =
``bus_capacitance`` from SDA to GND, and the open-drain driver idealised as a diode ``DOD`` from SDA to the
private node ``DRV`` that a PULSE source holds at 0 V for half a clock period and then releases to V_DD. While
held, SDA sits one diode drop above ground: V_low = the generic model's forward drop at the current the pull-up
feeds it (``calc.i2c.v_low_diode``, about 0.66 V at 3.3 V / 1.9 kohm - the same order as I2C's 0.4 V V_OL, not
equal to it), which the expectation ``held_low`` measures at 0.9 t_low against that computed value. On release the
diode stops conducting and SDA rises through R1 into C_b. The expectation ``rise`` reads v(SDA) at t_check = t_low
+ t_30 + t_r and requires at least V_IH = 0.7 V_DD (a one-sided bound), where t_30 = R C_b ln((V_DD - V_low) /
(0.7 V_DD)) is the RC time from the release at V_low to the 0.3 V_DD point: with the start level computed and
verified, the check holds exactly when the simulated 30 % -> 70 % rise takes at most ``bus_rise_time``, i.e. when
the netlist's R C_b is at most t_r / 0.8473 - a pull-up above R_max fails it (up to one sample step of the
transient), the catalog test shows a part 30 % above the design value failing. Not verified: the real driver's
V_OL (the diode is an idealisation, not a device model) and SCL (not simulated).

Validity (engineering bounds, not measured): 2.5 V <= V_DD <= 5.5 V - the diode idealisation holds the line about
0.66 V up, which must be under the 30 % point (0.75 V at 2.5 V); a 1.8 V or 1.2 V bus needs a driver model, a later
catalog item - 1 kHz <= f_SCL <= 5 MHz, 1 pF <= C_b <= 10 nF, 10 ns <= t_r <= 10 us; R_min <= R_max or the design is
refused with both numbers.
"""

from __future__ import annotations

from ai_eda.ir import (
    AnalysisSpec,
    Block,
    CircuitDomain,
    CircuitIR,
    Constraint,
    ConstraintKind,
    Expectation,
    NetKind,
    Reduce,
    SimElement,
    SimulationSetup,
    SpiceBinding,
    SpiceDevice,
    Stimulus,
    StimulusKind,
    Topology,
    Traced,
)
from ai_eda.tools.calc.e_series import standard_value
from ai_eda.tools.calc.basic import (
    i2c_r_max,
    i2c_r_min,
    i2c_r_pullup,
    i2c_t_30,
    i2c_t_check,
    i2c_t_hold,
    i2c_t_low,
    i2c_tran_step,
    i2c_tran_stop,
    i2c_v_ih,
    i2c_v_low_diode,
)
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.spice import SpiceAnalysis

from ai_eda.design.base import Choice, PartNote, Plan, Template, TheorySection, choice_provenance, parameter_value, quantity, structural_provenance
from ai_eda.design.inputs import DesignInput
from ai_eda.design.library_parts import TemplateRefusal, library_component, require_pins, two_terminals
from ai_eda.design.templates import (
    HEADER_3,
    RESISTOR,
    _changes,
    _choice,
    _div,
    _header_note,
    _mul,
    _net,
    _net_line,
    _part_line,
    _refused,
    RESISTOR_SERIES,
    _placed_line,
    _resistor_criteria,
    _solved,
    standard_of,
    _resistor_substitutes,
    _spelled,
    _value_text,
)

class I2cPullupTemplate(Template):
    """I2C pull-ups; SDA simulated with the bus capacitance and an idealised open-drain driver as simulation-only elements (module docstring)."""

    id = "i2c_pullup"
    title = "I2C pull-up resistors"
    triggers = ("bus_capacitance", "bus_rise_time")
    needs = ("input_voltage", "clock_frequency", "bus_capacitance", "bus_rise_time")
    serves = ("input_voltage", "clock_frequency", "bus_capacitance", "bus_rise_time")

    V_OL_V = 0.4
    I_OL_A = 3e-3
    PULLUP_FRACTION = 0.8
    V_LOW_TOL_V = 0.05
    #: the diode idealisation holds the line about 0.66 V up: below 2.5 V that is not under the 30 % point (a low-voltage bus needs a driver model)
    V_MIN, V_MAX = 2.5, 5.5
    F_MIN, F_MAX = 1_000.0, 5_000_000.0
    C_MIN, C_MAX = 1e-12, 10e-9
    T_MIN, T_MAX = 10e-9, 10e-6
    MODEL_NAME, MODEL_CARD = "DOD", ".model DOD D"
    MODEL = (
        "open-drain driver idealised as generic diode `.model DOD D` (ngspice defaults: IS = 1e-14 A, N = 1) from SDA to the simulation-private node DRV, "
        "which a PULSE source holds at 0 V for half a clock period and then releases to V_DD: SDA is held one diode drop above ground (V_low from "
        "`calc.i2c.v_low_diode`, measured by held_low), then rises through R1 into CBUS = bus_capacitance (a simulation-only element: wiring, not a part); "
        "the real driver's V_OL is not verified, SCL is not simulated"
    )

    def build(self, ir: CircuitIR, inputs: dict[str, DesignInput], unusable: dict[str, str], library: KicadLibrary, *, confirmed: bool) -> Plan:
        t = self.id
        plan = Plan(template=t, title=self.title)
        v_dd, f_scl, c_b, t_r = (inputs[k] for k in self.needs)
        plan.inputs = dict(zip(self.needs, (v_dd, f_scl, c_b, t_r)))
        v, f, c, tr = (x.traced.value for x in (v_dd, f_scl, c_b, t_r))
        req_v, req_f, req_c, req_t = (x.requirement.id for x in (v_dd, f_scl, c_b, t_r))
        for value, lo, hi, req, what, unit in ((v, self.V_MIN, self.V_MAX, req_v, "bus supply", "V"), (f, self.F_MIN, self.F_MAX, req_f, "clock frequency", "Hz"),
                                                (c, self.C_MIN, self.C_MAX, req_c, "bus capacitance", "F"), (tr, self.T_MIN, self.T_MAX, req_t, "bus rise time", "s")):
            if not lo <= value <= hi:
                return _refused(plan, f"{what} {value:.12g} {unit} ({req}) is outside {lo:.12g}..{hi:.12g} {unit} (engineering bounds chosen for this template, not measured)")
        v_ol_choice, v_ol = _choice(t, "v_ol", self.V_OL_V, "V", "the low level the driver must still reach while sinking i_ol (I2C: 0.4 V); sets R_min and the 30 % point of the rise", confirmed)
        i_ol_choice, i_ol = _choice(t, "i_ol", self.I_OL_A, "A", "the driver's sink current at v_ol (I2C standard / fast mode: 3 mA); sets R_min", confirmed)
        frac_choice, fraction = _choice(t, "pullup_fraction", self.PULLUP_FRACTION, None, "the pull-up as a fraction of R_max (0.8: margin for the resistor tolerance and the capacitance estimate)", confirmed)
        model_choice = Choice("driver_model", self.MODEL)
        model_prov = choice_provenance(t, f"driver_model: {self.MODEL}", confirmed)
        tol_choice, v_low_tol = _choice(t, "v_low_tol", self.V_LOW_TOL_V, "V", "absolute tolerance of the held-low level expectation (the computed diode drop against the simulated one)", confirmed)
        plan.choices = [v_ol_choice, i_ol_choice, frac_choice, model_choice, tol_choice]
        params: dict[str, Traced] = {"v_dd": v_dd.traced, "f_scl": f_scl.traced, "c_b": c_b.traced, "t_r": t_r.traced, "v_ol": v_ol, "i_ol": i_ol, "pullup_fraction": fraction, "v_low_tol": v_low_tol}
        try:
            params["r_max"] = i2c_r_max(params["t_r"], params["c_b"], ("t_r", "c_b"))
            params["r_min"] = i2c_r_min(params["v_dd"], params["v_ol"], params["i_ol"], ("v_dd", "v_ol", "i_ol"))
        except (ValueError, ZeroDivisionError) as e:
            return _refused(plan, f"{e} ({req_v} = {v:.12g} V, {req_c} = {c:.12g} F, {req_t} = {tr:.12g} s)")
        r_max, r_min = params["r_max"].value, params["r_min"].value
        if r_min > r_max:
            return _refused(plan, (
                f"no pull-up satisfies this bus: the rise time {tr:.12g} s into {c:.12g} F allows at most R_max = {r_max:.12g} ohm, but the driver sinking "
                f"{self.I_OL_A:.12g} A to {self.V_OL_V:.12g} V needs at least R_min = {r_min:.12g} ohm; lower the bus capacitance, allow a longer rise time, "
                "or state a driver with more sink current (i_ol)"
            ))
        try:
            # the solved pull-up is not a part: R1 / R2 are an E series value inside the window R_min..R_max - the nearest one,
            # else the next one below, else the next one above, E24 before E96; every later number is computed from the placed value
            params["r_pullup_ideal"] = i2c_r_pullup(params["r_max"], params["r_min"], params["pullup_fraction"], ("r_max", "r_min", "pullup_fraction"))
            inside = (
                candidate
                for series in RESISTOR_SERIES for mode in ("nearest", "at_most", "at_least")
                for candidate in (standard_value(params["r_pullup_ideal"], "r_pullup_ideal", (series,), mode=mode),)
                if r_min <= candidate.value <= r_max
            )
            placed = next(inside, None)
            if placed is None:
                return _refused(plan, (
                    f"no standard resistor fits this bus: the window R_min = {r_min:.12g} ohm .. R_max = {r_max:.12g} ohm holds no E24 or E96 value; "
                    "lower the bus capacitance, allow a longer rise time, or state a driver with more sink current (i_ol)"
                ))
            params["r_pullup"] = placed
            params["v_ih"] = i2c_v_ih(params["v_dd"], ("v_dd",))
            params["t_low"] = i2c_t_low(params["f_scl"], ("f_scl",))
            params["v_low"] = i2c_v_low_diode(params["v_dd"], params["r_pullup"], ("v_dd", "r_pullup"))
            params["t_hold"] = i2c_t_hold(params["t_low"], ("t_low",))
            params["t_30"] = i2c_t_30(params["r_pullup"], params["c_b"], params["v_dd"], params["v_low"], ("r_pullup", "c_b", "v_dd", "v_low"))
            params["t_check"] = i2c_t_check(params["t_low"], params["t_30"], params["t_r"], ("t_low", "t_30", "t_r"))
            params["tran_step"] = i2c_tran_step(params["t_r"], ("t_r",))
            params["tran_stop"] = i2c_tran_stop(params["t_low"], params["r_pullup"], params["c_b"], ("t_low", "r_pullup", "c_b"))
        except (ValueError, ZeroDivisionError) as e:
            return _refused(plan, f"{e} ({req_v} = {v:.12g} V, {req_c} = {c:.12g} F, {req_t} = {tr:.12g} s)")
        r = params["r_pullup"].value
        plan.computed = [(k, params[k]) for k in ("r_max", "r_min", "r_pullup_ideal", "r_pullup", "v_ih", "t_low", "v_low", "t_hold", "t_30", "t_check", "tran_step", "tran_stop")]
        try:
            r1 = library_component(library, "R1", _value_text(r), "SDA pull-up", *RESISTOR, structural_provenance(t, "SDA pull-up"), [req_c, req_t])
            r2 = library_component(library, "R2", _value_text(r), "SCL pull-up", *RESISTOR, structural_provenance(t, "SCL pull-up"), [req_f])
            j1 = library_component(library, "J1", "Conn_01x03", "VDD / SDA / SCL header", *HEADER_3, structural_provenance(t, "bus header"), [req_v])
            r1a, r1b = two_terminals(r1)
            r2a, r2b = two_terminals(r2)
            require_pins(j1, ("1", "2", "3"))
        except TemplateRefusal as e:
            return _refused(plan, str(e))
        for part in (r1, r2):
            part.electrical["resistance"] = params["r_pullup"]
            part.spice = SpiceBinding(device=SpiceDevice.R, value=params["r_pullup"], pin_order=[p.number for p in part.pins], provenance=structural_provenance(t, "ideal resistor at the placed (E series) value"))
        j1.spice = SpiceBinding(exclude=True, exclude_reason="connector, no electrical model", provenance=structural_provenance(t, "header excluded from the netlist"))
        nets = [
            _net("VDD", NetKind.POWER, [("J1", "1"), ("R1", r1a), ("R2", r2a)], t, [req_v]),
            _net("SDA", NetKind.SIGNAL, [("R1", r1b), ("J1", "2")], t, [req_c, req_t]),
            _net("SCL", NetKind.SIGNAL, [("R2", r2b), ("J1", "3")], t, [req_f]),
            # the board has no ground pin (the pull-ups sit between VDD and the lines); GND is the simulation's reference, the bus system's ground
            _net("GND", NetKind.GROUND, [], t),
        ]
        drv_low = Traced(value=0.0, unit="V", provenance=choice_provenance(t, "drv_low = 0 V: the driver's active level", confirmed))
        sim = SimulationSetup(
            nodes=["DRV"],
            elements=[
                SimElement(id="CBUS", device=SpiceDevice.C, net="SDA", reference_net="GND", value=params["c_b"], reason=f"the bus wiring capacitance ({req_c}): a property of the wiring, not a part",
                           provenance=structural_provenance(t, "bus capacitance as a simulation-only element"), serves_requirements=[req_c]),
                SimElement(id="DOD", device=SpiceDevice.D, net="SDA", reference_net="DRV", model_name=self.MODEL_NAME, model_card=Traced(value=self.MODEL_CARD, provenance=model_prov),
                           reason="the open-drain driver idealised as a diode to the pulsed node DRV (conducts while DRV is low, releases when it is high)",
                           provenance=structural_provenance(t, "driver idealisation on the confirmed model")),
            ],
            stimuli=[
                Stimulus(id="VDD", source="voltage", net="VDD", reference_net="GND", kind=StimulusKind.DC, value=params["v_dd"], provenance=structural_provenance(t, "the bus supply at input_voltage"), serves_requirements=[req_v]),
                Stimulus(
                    id="DRV", source="voltage", net="DRV", reference_net="GND", kind=StimulusKind.PULSE,
                    params={"v1": drv_low, "v2": params["v_dd"], "td": params["t_low"], "tr": params["tran_step"], "tf": params["tran_step"], "pw": params["tran_stop"], "per": params["tran_stop"]},
                    provenance=structural_provenance(t, "the driver: low for half a clock period (t_low = 1 / (2 f_SCL)), then released"), serves_requirements=[req_f],
                ),
            ],
            analyses=[AnalysisSpec(id="tran", kind=SpiceAnalysis.TRAN, params={"step": params["tran_step"], "stop": params["tran_stop"]},
                                   provenance=structural_provenance(t, "the low half period and the release, up to six RC after it"))],
            expectations=[
                Expectation(
                    id="held_low", analysis_id="tran", vector="v(SDA)", reduce=Reduce.AT, at=params["t_hold"], nominal=params["v_low"], tol_abs=params["v_low_tol"],
                    provenance=structural_provenance(t, "the level the idealised driver holds SDA at (the diode drop at the pull-up's current) is the computed V_low: the start level t_30 is built on"),
                ),
                Expectation(
                    id="rise", analysis_id="tran", vector="v(SDA)", reduce=Reduce.AT, at=params["t_check"], nominal=params["v_ih"], bound="at_least",
                    provenance=structural_provenance(t, f"SDA has reached V_IH = 0.7 V_DD at t_low + t_30 + t_r: the 30 % -> 70 % rise takes at most the value of {req_t} (a bound: it names no requirement id; the SDA net serves it)"),
                ),
            ],
        )
        topology = Topology(
            name="I2C pull-ups", domains=[CircuitDomain.DIGITAL],
            rationale="open-drain bus pull-ups: R_max = t_r / (ln(0.7/0.3) C_b) from the rise time, R_min = (V_DD - V_OL) / I_OL from the driver's sink current; R = max(pullup_fraction x R_max, R_min)",
            provenance=structural_provenance(t, "selected by bus_capacitance + bus_rise_time"),
            blocks=[Block(id="pullups", function="I2C pull-up resistors", domain=CircuitDomain.DIGITAL, component_refs=["R1", "R2"], input_nets=["VDD"], output_nets=["SDA", "SCL"],
                          provenance=structural_provenance(t, "the one block of this template"))],
        )
        constraints = [
            Constraint(
                id="c.i2c_pullup.rise", kind=ConstraintKind.ELECTRICAL, target="SDA",
                description=f"the 30 % -> 70 % rise of SDA through R1 into the bus capacitance takes at most the value of {req_t} (expectation rise: v(SDA) at t_low + t_30 + t_r is at least 0.7 V_DD, a bound)",
                provenance=structural_provenance(t, "what 'bus rise time' means for this template"),
            ),
            Constraint(
                id="c.i2c_pullup.sink", kind=ConstraintKind.ELECTRICAL, target="SDA",
                description=f"R1 = R2 >= R_min = (V_DD - v_ol) / i_ol = {r_min:.12g} ohm so a driver sinking i_ol still reaches v_ol (checked by the calculators, not simulated: the driver is an idealisation)",
                provenance=structural_provenance(t, "the sink-current side of the pull-up window"),
            ),
        ]
        plan.parts = [_part_line(x) for x in (r1, r2, j1)]
        plan.nets = [_net_line(n) for n in nets]
        plan.simulation = [
            f"simulation-only elements: CBUS = {c:.12g} F SDA-GND (the bus capacitance, not a part); DOD = diode SDA-DRV on `{self.MODEL_CARD}` (the open-drain driver idealised; node DRV is simulation-only)",
            f"stimuli VDD = {v:.12g} V; DRV: pulse 0 -> {v:.12g} V after t_low = {params['t_low'].value:.12g} s (held low, then released); analysis tran {params['tran_step'].value:.12g} s .. {params['tran_stop'].value:.12g} s",
            f"expectation held_low: v(SDA) at t_hold = {params['t_hold'].value:.12g} s = v_low = {params['v_low'].value:.12g} V +/- {self.V_LOW_TOL_V:.12g} V (the diode drop the driver idealisation holds the line at; the start level of t_30)",
            f"expectation rise: v(SDA) at t_check = {params['t_check'].value:.12g} s at least v_ih = {params['v_ih'].value:.12g} V (= 0.7 V_DD; a bound) - the 30 % -> 70 % rise takes at most {tr:.12g} s ({req_t})",
            "not verified: V_OL of the real driver (the diode is an idealisation), SCL (same pull-up into the same bus)",
        ]
        plan.changes = _changes(t, self.title, topology, [r1, r2, j1], nets, params, sim, constraints)
        return plan

    def theory(self, ir: CircuitIR) -> list[TheorySection]:
        v_dd, c_b, t_r, v_ol, i_ol, frac, r_max, r_min, r, v_ih, t_low, v_low, t_30, t_check = (
            parameter_value(ir, k) for k in ("v_dd", "c_b", "t_r", "v_ol", "i_ol", "pullup_fraction", "r_max", "r_min", "r_pullup", "v_ih", "t_low", "v_low", "t_30", "t_check"))
        return [
            TheorySection("동작 원리와 식", (
                "I2C 의 SDA·SCL 은 개방 드레인이라 '1' 은 풀업 저항이 버스 용량 C_b 를 V_DD 로 충전해서 만듭니다. 상승은 RC 충전이고 I2C 는 상승 시간을 0.3·V_DD 에서 0.7·V_DD 까지로 정의하므로\n\n"
                "    t_r = R·C_b·ln(0.7/0.3) = 0.8473·R·C_b   →   R_max = t_r/(0.8473·C_b)\n"
                "    R_min = (V_DD − V_OL)/I_OL   (드라이버가 I_OL 을 흘리면서도 V_OL 에 닿아야 함)\n\n"
                f"이 설계: C_b = {quantity(c_b, 'F')}, t_r = {quantity(t_r, 's')} 요구사항에서 R_max = {quantity(r_max, 'ohm')} (`calc.i2c.r_max`), "
                f"V_DD = {quantity(v_dd, 'V')}, V_OL = {quantity(v_ol, 'V')}, I_OL = {quantity(i_ol, 'A')} 에서 R_min = {quantity(r_min, 'ohm')} (`calc.i2c.r_min`), "
                f"풀업 R = max({quantity(frac)} × R_max, R_min) = {quantity(_solved(ir, 'r_pullup'), 'ohm')}"
                + ("" if standard_of(ir, "r_pullup") is None else f" → 부품값 {quantity(r, 'ohm')} (창 R_min..R_max 안의 표준값; `{ir.parameters['r_pullup'].provenance.tool}`)") +
                f" (`calc.i2c.r_pullup`; 저항 공차·용량 추정 여유, R_min 아래로는 내려가지 않음). R_min > R_max 이면 어떤 값도 버스를 만족하지 못해 거부합니다."
            )),
            TheorySection("시뮬레이션 모델과 판정", (
                f"버스 용량과 드라이버는 보드 부품이 아니므로 시뮬레이션 전용 요소로 둡니다: CBUS = C_b (SDA–GND), DOD = 일반 다이오드 모델(SDA → 전용 노드 DRV). DRV 는 반 클럭 주기 t_low = 1/(2·f_SCL) = {quantity(t_low, 's')} 동안 0 V 로 잡혀 SDA 를 다이오드 한 개 전압 위에 묶고, 그 뒤 V_DD 로 풀려 다이오드가 끊기면 SDA 가 R1 을 통해 C_b 로 올라갑니다. "
                f"잡힌 레벨 V_low = {quantity(v_low, 'V')} 는 풀업이 흘리는 전류에서의 일반 다이오드 순방향 전압(`calc.i2c.v_low_diode`: IS = 1e-14 A, N = 1, 27 °C 의 고정점)이고 기대값 held_low 가 0.9·t_low 에서 그 값을 ±0.05 V 로 측정합니다.\n\n"
                f"    t_30 = R·C_b·ln((V_DD − V_low)/(0.7·V_DD)) = {quantity(t_30, 's')}  (풀린 뒤 0.3·V_DD 에 닿는 시각, `calc.i2c.t_30`)\n"
                f"    t_check = t_low + t_30 + t_r = {quantity(t_check, 's')}\n"
                f"    판정 rise: v(SDA)(t_check) ≥ V_IH = 0.7·V_DD = {quantity(v_ih, 'V')}  (한쪽 한계 `at_least`)\n\n"
                "출발 레벨을 계산하고 측정했으므로 이 판정은 넷리스트의 R·C_b 가 t_r/0.8473 이하일 때만(= 30→70 % 상승이 t_r 안일 때만) 통과하고, R_max 를 넘는 R 은 과도 해석의 한 표본 간격 안에서 FAIL 이 됩니다(카탈로그 테스트: 설계값의 1.3 배). "
                "실제 드라이버의 V_OL 과 SCL 은 검증하지 않습니다(드라이버는 소자 모델이 아니라 이상화; SCL 은 같은 R·같은 버스)."
            )),
        ]

    def part_notes(self, ir: CircuitIR) -> dict[str, PartNote]:
        v_dd, r, r_min, r_max = (parameter_value(ir, k) for k in ("v_dd", "r_pullup", "r_min", "r_max"))
        p_r = _div(_mul(v_dd, v_dd), r)  # the pull-up's dissipation while the line is held low (V_DD^2 / R)
        note = PartNote(
            role="풀업 저항 (VDD–SDA / VDD–SCL)",
            why=f"R = pullup_fraction × R_max = {quantity(_solved(ir, 'r_pullup'), 'ohm')} (`calc.i2c.r_pullup`)"
                + ("" if standard_of(ir, "r_pullup") is None else f", 부품은 창 안의 표준값 {quantity(r, 'ohm')}") +
                f"; 허용 창 R_min = {quantity(r_min, 'ohm')} .. R_max = {quantity(r_max, 'ohm')}. 0603 SMD.",
            criteria=_resistor_criteria(r, p_r, "5 % 이하 (R_max 와의 여유 20 % 안)", spelled=_spelled(ir, "R1", "R2"), standard=standard_of(ir, "r_pullup")),
            substitutes=_resistor_substitutes(r, "Resistor_SMD"),
        )
        return {
            "R1": PartNote(role="R1: SDA " + note.role, why=note.why, criteria=note.criteria, substitutes=note.substitutes),
            "R2": PartNote(role="R2: SCL " + note.role, why=note.why + " SDA 와 같은 값(시뮬레이션은 SDA 만).", criteria=note.criteria, substitutes=note.substitutes),
            "J1": _header_note("J1: VDD / SDA / SCL 헤더", "1 = VDD, 2 = SDA, 3 = SCL", f"버스 전원 {quantity(v_dd, 'V')} 과 두 개방 드레인 선을 잇는 곳(접지 핀 없음: 풀업은 VDD 와 선 사이에만 있고, 시뮬레이션의 GND 는 버스 시스템의 접지); 버스 용량은 배선·장치 핀의 합(요구사항 bus_capacitance)."),
        }
