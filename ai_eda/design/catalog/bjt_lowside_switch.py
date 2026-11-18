"""NPN low-side switch with a flyback diode, from the control level, the load supply, the load current and the
allowed ON drop (D9 catalog, tier 1).

Q1 (2N3904) switches the low side of an external load fed from VLOAD; R1 sets the base current from the control
level: I_B = I_C / beta_forced, R_B = (V_ctl - V_BE) / I_B. D1 (1N4001) across the load terminals (SW to VLOAD)
clamps the inductive turn-off.

Simulation: the load is a constant-current sink of ``output_current`` from VLOAD into SW - a simplified
steady-current load model, not a coil: its current never changes, so the OFF state is the clamped freewheeling
state and the ON state carries exactly the required current. A real coil's current ramps up through L / R at
turn-on and decays through the diode at turn-off; neither check needs those dynamics. A load with its own R / L
on the netlist but not in the BOM is a later catalog item (PLAN decision 16: simulation-only elements). The control is a PULSE on CTL: 0 -> V_ctl after ``t_delay``, ON
for ``t_on``, period ``t_period`` = the end of the transient.

Expectations are one-sided bounds (no tolerance): ``on_drop`` - the ON-state collector voltage, read as the mean
of v(SW) over the ON window, is at most ``max_voltage_drop``, the requirement's own value. ``max_voltage_drop``
is the allowed ON-state level, judged as that window mean (a windowed maximum is not a reduction the engine
has; ``MAX`` covers the whole transient, where the OFF state sits at the clamp); with the steady-current load
the ON level is flat, so mean and maximum agree inside the window, and the window excludes both edges; ``off_clamp`` - the maximum of v(SW) is at most V_load + v_f
(v_f: the confirmed clamp allowance of the diode model). A bound cannot claim a requirement (the compiler's
rule), so the SW net serves ``max_voltage_drop`` and the constraint names it. A base drive the model cannot
saturate with shows as a FAIL of ``on_drop``, not as a refusal: the design rule is beta_forced, the verdict is
ngspice's.

Models: ``.model QSW NPN (BF=100 TR=200n)`` and ``.model DFW D`` - ngspice's defaults with the current gain
spelled out and the switching class's reverse transit time (the astable template's measured reason: without TR
a saturated transistor turns off instantly and the transient can abort); not vendor models.

Validity (engineering bounds, not measured): 0 < output_current <= i_c_rating (0.2 A, the 2N3904 family's
continuous collector rating); output_voltage > 0 and output_voltage + v_f <= 40 V (the OFF-state collector
voltage against V_CEO); input_voltage >= V_BE + 0.5 V so the base
resistor is a positive, non-degenerate value; max_voltage_drop > 0.
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
    SimulationSetup,
    SpiceBinding,
    SpiceDevice,
    Stimulus,
    StimulusKind,
    Topology,
    Traced,
)
from ai_eda.tools.calc.basic import bjt_switch_base_current, bjt_switch_base_resistor, bjt_switch_clamp_max
from ai_eda.tools.calc.e_series import standard_value
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.spice import SpiceAnalysis

from ai_eda.design.base import NO_RECORD, Choice, PartNote, Plan, Template, TheorySection, choice_provenance, parameter_value, quantity, structural_provenance, unverified
from ai_eda.design.inputs import DesignInput
from ai_eda.design.library_parts import TemplateRefusal, library_component, pin_by_name, require_pins, two_terminals
from ai_eda.design.templates import (
    HEADER_2,
    HEADER_3,
    NPN,
    RESISTOR_THT,
    _changes,
    _choice,
    _div,
    _header_note,
    _mul,
    _net,
    _net_line,
    _part_line,
    _refused,
    _placed_line,
    _resistor_criteria,
    _solved,
    _resistor_substitutes,
    _spelled,
    _value_text,
    standard_of,
)

DIODE_FLYBACK = (("Diode", "1N4001"), ("Diode_THT", "D_DO-41_SOD81_P10.16mm_Horizontal"))


class BjtLowsideSwitchTemplate(Template):
    """NPN low-side switch with a flyback diode; the load is simulated as a steady-current sink (module docstring)."""

    id = "bjt_lowside_switch"
    title = "NPN low-side switch with flyback diode"
    triggers = ("output_current", "max_voltage_drop")
    needs = ("input_voltage", "output_voltage", "output_current", "max_voltage_drop")
    serves = ("input_voltage", "output_voltage", "output_current", "max_voltage_drop")

    V_BE_V = 0.7
    BETA_FORCED = 10.0
    V_F_V = 1.0
    I_C_RATING_A = 0.2
    #: the 2N3904 family's V_CEO; a load supply above it is refused (a rating of the chosen part, not measured here)
    V_CEO_V = 40.0
    #: the control level must exceed V_BE by this much, or R1 degenerates toward zero
    CTL_MARGIN_V = 0.5
    T_STEP_S = 10e-6
    T_DELAY_S = 1e-3
    T_ON_S = 2e-3
    T_PERIOD_S = 4e-3
    ON_WINDOW_S = (2e-3, 3e-3)
    MODEL_Q_NAME, MODEL_Q_CARD = "QSW", ".model QSW NPN (BF=100 TR=200n)"
    MODEL_D_NAME, MODEL_D_CARD = "DFW", ".model DFW D"
    MODELS = (
        "generic Gummel-Poon NPN `.model QSW NPN (BF=100 TR=200n)` (ngspice defaults: IS = 1e-16 A, no junction capacitance; BF = 100 is the "
        "forward gain the forced beta is compared with, TR = 200 ns the switching class's reverse transit time, the astable template's measured "
        "reason) and generic diode `.model DFW D` (IS = 1e-14 A, N = 1: about 0.75 V at 50 mA) - not 2N3904 / 1N4001 vendor models; the load is "
        "a constant-current sink of output_current from VLOAD into SW (a simplified steady-current load model, not a coil)"
    )

    def build(self, ir: CircuitIR, inputs: dict[str, DesignInput], unusable: dict[str, str], library: KicadLibrary, *, confirmed: bool) -> Plan:
        t = self.id
        plan = Plan(template=t, title=self.title)
        v_ctl, v_load, i_load, drop = (inputs[k] for k in self.needs)
        plan.inputs = dict(zip(self.needs, (v_ctl, v_load, i_load, drop)))
        vc, vl, il, vd = (x.traced.value for x in (v_ctl, v_load, i_load, drop))
        req_ctl, req_vload, req_i, req_drop = (x.requirement.id for x in (v_ctl, v_load, i_load, drop))
        if not 0 < il <= self.I_C_RATING_A:
            return _refused(plan, (
                f"load current {il:.12g} A ({req_i}) is outside 0..{self.I_C_RATING_A:.12g} A: the 2N3904's continuous collector rating (choice i_c_rating); "
                "a larger load needs another transistor, which is a later catalog item, not a hidden option"
            ))
        if vl <= 0:
            return _refused(plan, f"load supply {vl:.12g} V ({req_vload}) must be positive")
        if vl + self.V_F_V > self.V_CEO_V:
            return _refused(plan, (
                f"the OFF-state collector voltage V_load + v_f = {vl:.12g} V + {self.V_F_V:.12g} V = {vl + self.V_F_V:.12g} V ({req_vload}) exceeds the 2N3904's "
                f"V_CEO of {self.V_CEO_V:.12g} V; a higher load supply needs another transistor, which is a later catalog item"
            ))
        if vc < self.V_BE_V + self.CTL_MARGIN_V:
            return _refused(plan, (
                f"control level {vc:.12g} V ({req_ctl}) is below V_BE + {self.CTL_MARGIN_V:.12g} V = {self.V_BE_V + self.CTL_MARGIN_V:.12g} V: "
                "the base resistor R_B = (V_ctl - V_BE) / I_B would be zero or negative; drive the switch from a logic level"
            ))
        if vd <= 0:
            return _refused(plan, f"max voltage drop {vd:.12g} V ({req_drop}) must be positive: it is the bound the ON collector voltage is judged against")
        v_be_choice, v_be = _choice(t, "v_be", self.V_BE_V, "V", "base-emitter voltage the base resistor is designed with (a saturated 2N3904's is 0.7..0.9 V; a lower design value gives more drive)", confirmed)
        beta_choice, beta = _choice(t, "beta_forced", self.BETA_FORCED, None, "forced current gain I_C / I_B of the saturated switch (10: the usual margin against the datasheet's minimum h_FE at the switched current)", confirmed)
        v_f_choice, v_f = _choice(t, "v_f", self.V_F_V, "V", "forward drop allowed for the flyback diode at the load current: the off_clamp bound is V_load + v_f", confirmed)
        rating_choice, rating = _choice(t, "i_c_rating", self.I_C_RATING_A, "A", "continuous collector current the switch is rated for (2N3904 family: 200 mA); a load above it is refused", confirmed)
        model_choice = Choice("switch_models", self.MODELS)
        model_prov = choice_provenance(t, f"switch_models: {self.MODELS}", confirmed)
        step_choice, step = _choice(t, "tran_step", self.T_STEP_S, "s", "transient time step, also the control edge time (tr = tf)", confirmed)
        delay_choice, delay = _choice(t, "t_delay", self.T_DELAY_S, "s", "the control stays low this long first (the OFF state before the pulse)", confirmed)
        on_choice, t_on = _choice(t, "t_on", self.T_ON_S, "s", "the ON pulse width of the control", confirmed)
        period_choice, period = _choice(t, "t_period", self.T_PERIOD_S, "s", "the control period and the end of the transient: one ON pulse, then OFF again", confirmed)
        w0_choice, w_start = _choice(t, "on_window_start", self.ON_WINDOW_S[0], "s", "start of the window the ON collector voltage is averaged over (inside the ON pulse, after the edge)", confirmed)
        w1_choice, w_stop = _choice(t, "on_window_stop", self.ON_WINDOW_S[1], "s", "end of that window (before the turn-off edge)", confirmed)
        plan.choices = [v_be_choice, beta_choice, v_f_choice, rating_choice, model_choice, step_choice, delay_choice, on_choice, period_choice, w0_choice, w1_choice]
        params: dict[str, Traced] = {
            "v_ctl": v_ctl.traced, "v_load": v_load.traced, "i_load": i_load.traced, "max_drop": drop.traced,
            "v_be": v_be, "beta_forced": beta, "v_f": v_f, "i_c_rating": rating,
        }
        try:
            params["i_b"] = bjt_switch_base_current(params["i_load"], params["beta_forced"], ("i_load", "beta_forced"))
            # the solved R_B is not a part: R1 is the largest E24 value not above it, so the base drive is at least the designed one
            # (a larger resistor would leave the transistor short of the forced beta); the verdict stays ngspice's
            params["r_b_ideal"] = bjt_switch_base_resistor(params["v_ctl"], params["v_be"], params["i_b"], ("v_ctl", "v_be", "i_b"))
            params["r_b"] = standard_value(params["r_b_ideal"], "r_b_ideal", ("e24",), mode="at_most")
            params["clamp_max"] = bjt_switch_clamp_max(params["v_load"], params["v_f"], ("v_load", "v_f"))
        except (ValueError, ZeroDivisionError) as e:
            return _refused(plan, f"{e} ({req_ctl} = {vc:.12g} V, {req_i} = {il:.12g} A)")
        params.update({"tran_step": step, "t_delay": delay, "t_on": t_on, "t_period": period, "on_window_start": w_start, "on_window_stop": w_stop})
        plan.computed = [(k, params[k]) for k in ("i_b", "r_b_ideal", "r_b", "clamp_max")]
        try:
            r1 = library_component(library, "R1", _value_text(params["r_b"].value), "base resistor", *RESISTOR_THT, structural_provenance(t, "base resistor"), [req_ctl, req_i])
            q1 = library_component(library, "Q1", "2N3904", "NPN low-side switch", *NPN, structural_provenance(t, "switching transistor"), [req_i, req_drop])
            d1 = library_component(library, "D1", "1N4001", "flyback diode across the load (SW to VLOAD)", *DIODE_FLYBACK, structural_provenance(t, "flyback diode"), [req_vload])
            j1 = library_component(library, "J1", "Conn_01x02", "CTL / GND header", *HEADER_2, structural_provenance(t, "control header"), [req_ctl])
            j2 = library_component(library, "J2", "Conn_01x03", "VLOAD / SW / GND header (the load between VLOAD and SW)", *HEADER_3, structural_provenance(t, "load header"), [req_vload, req_i])
            r1a, r1b = two_terminals(r1)
            require_pins(j1, ("1", "2"))
            require_pins(j2, ("1", "2", "3"))
        except TemplateRefusal as e:
            return _refused(plan, str(e))
        qc, qb, qe = (pin_by_name(q1, name) for name in ("C", "B", "E"))
        if qc is None or qb is None or qe is None or len(q1.pins) != 3:
            return _refused(plan, f"{NPN[0][0]}:{NPN[0][1]} pins are not named C / B / E in this library (pins: {[(p.number, p.name) for p in q1.pins]}); the template will not guess the transistor's pinout")
        da, dk = pin_by_name(d1, "A"), pin_by_name(d1, "K")
        if da is None or dk is None or len(d1.pins) != 2:
            return _refused(plan, f"{DIODE_FLYBACK[0][0]}:{DIODE_FLYBACK[0][1]} pins are not named A / K in this library (pins: {[(p.number, p.name) for p in d1.pins]}); the template will not guess the diode's polarity")
        r1.electrical["resistance"] = params["r_b"]
        r1.spice = SpiceBinding(device=SpiceDevice.R, value=params["r_b"], pin_order=[p.number for p in r1.pins], provenance=structural_provenance(t, "ideal resistor at the placed (E series) value"))
        q1.spice = SpiceBinding(
            device=SpiceDevice.Q, model_name=self.MODEL_Q_NAME, model_card=Traced(value=self.MODEL_Q_CARD, provenance=model_prov), pin_order=[qc, qb, qe],
            provenance=structural_provenance(t, "NPN on the confirmed generic model; SPICE node order C B E from the library pin names"),
        )
        d1.spice = SpiceBinding(
            device=SpiceDevice.D, model_name=self.MODEL_D_NAME, model_card=Traced(value=self.MODEL_D_CARD, provenance=model_prov), pin_order=[da, dk],
            provenance=structural_provenance(t, "diode on the confirmed generic model; SPICE node order anode, cathode from the library pin names"),
        )
        for header in (j1, j2):
            header.spice = SpiceBinding(exclude=True, exclude_reason="connector, no electrical model", provenance=structural_provenance(t, "header excluded from the netlist"))
        nets = [
            _net("CTL", NetKind.SIGNAL, [("J1", "1"), ("R1", r1a)], t, [req_ctl]),
            _net("BASE", NetKind.SIGNAL, [("R1", r1b), ("Q1", qb)], t),
            _net("SW", NetKind.SIGNAL, [("Q1", qc), ("D1", da), ("J2", "2")], t, [req_i, req_drop]),
            _net("VLOAD", NetKind.POWER, [("J2", "1"), ("D1", dk)], t, [req_vload]),
            _net("GND", NetKind.GROUND, [("J1", "2"), ("Q1", qe), ("J2", "3")], t),
        ]
        ctl_low = Traced(value=0.0, unit="V", provenance=choice_provenance(t, "ctl_low = 0 V: the control's OFF level", confirmed))
        sim = SimulationSetup(
            stimuli=[
                Stimulus(
                    id="VCTL", source="voltage", net="CTL", reference_net="GND", kind=StimulusKind.PULSE,
                    params={"v1": ctl_low, "v2": params["v_ctl"], "td": delay, "tr": step, "tf": step, "pw": t_on, "per": period},
                    provenance=structural_provenance(t, "the control: 0 V, then one pulse to input_voltage"), serves_requirements=[req_ctl],
                ),
                Stimulus(id="VLOAD", source="voltage", net="VLOAD", reference_net="GND", kind=StimulusKind.DC, value=params["v_load"],
                         provenance=structural_provenance(t, "the load supply at output_voltage"), serves_requirements=[req_vload]),
                Stimulus(id="LOAD", source="current", net="VLOAD", reference_net="SW", kind=StimulusKind.DC, value=params["i_load"],
                         provenance=structural_provenance(t, "the load as a constant-current sink of output_current from VLOAD into SW (a simplified steady-current load model: its current never changes)"),
                         serves_requirements=[req_i]),
            ],
            analyses=[AnalysisSpec(id="tran", kind=SpiceAnalysis.TRAN, params={"step": step, "stop": period},
                                   provenance=structural_provenance(t, "one control period: OFF, the ON pulse, OFF again"))],
            expectations=[
                Expectation(
                    id="on_drop", analysis_id="tran", vector="v(SW)", reduce=Reduce.MEAN, params={"t_start": w_start, "t_stop": w_stop}, nominal=params["max_drop"], bound="at_most",
                    provenance=structural_provenance(t, f"the ON-state collector voltage, read as the mean of v(SW) over the ON window (flat with the steady-current load; both edges excluded), is at most the value of {req_drop} (a bound: it names no requirement id; the SW net serves it)"),
                ),
                Expectation(
                    id="off_clamp", analysis_id="tran", vector="v(SW)", reduce=Reduce.MAX, nominal=params["clamp_max"], bound="at_most",
                    provenance=structural_provenance(t, "the collector never exceeds V_load + v_f: the flyback diode clamps the OFF state"),
                ),
            ],
        )
        topology = Topology(
            name="BJT low-side switch", domains=[CircuitDomain.ANALOG],
            rationale="saturated NPN low-side switch: R_B = (V_ctl - V_BE) / (I_load / beta_forced); D1 across the load clamps the inductive turn-off to V_load + V_f",
            provenance=structural_provenance(t, "selected by output_current + max_voltage_drop"),
            blocks=[Block(id="switch", function="NPN low-side switch with flyback diode", domain=CircuitDomain.ANALOG, component_refs=["R1", "Q1", "D1"], input_nets=["CTL", "VLOAD"], output_nets=["SW"],
                          provenance=structural_provenance(t, "the one block of this template"))],
        )
        constraints = [
            Constraint(
                id="c.bjt_lowside_switch.on_drop", kind=ConstraintKind.ELECTRICAL, target="SW",
                description=f"with the control at the level of {req_ctl}, the ON-state collector voltage (the mean of v(SW) over the ON window, both edges excluded) is at most the value of {req_drop} (expectation on_drop, a bound)",
                provenance=structural_provenance(t, "what 'max voltage drop' means for this template"),
            ),
            Constraint(
                id="c.bjt_lowside_switch.off_clamp", kind=ConstraintKind.ELECTRICAL, target="SW",
                description="the collector never exceeds V_load + v_f: D1 carries the load current when the switch is off (expectation off_clamp, a bound)",
                provenance=structural_provenance(t, "the flyback clamp"),
            ),
        ]
        plan.parts = [_part_line(x) for x in (r1, q1, d1, j1, j2)]
        plan.nets = [_net_line(n) for n in nets]
        plan.simulation = [
            f"stimuli VCTL: pulse 0 -> {vc:.12g} V on CTL after {self.T_DELAY_S:.12g} s for {self.T_ON_S:.12g} s (period {self.T_PERIOD_S:.12g} s); VLOAD = {vl:.12g} V; "
            f"LOAD = {il:.12g} A constant-current sink from VLOAD into SW; analysis tran {self.T_STEP_S:.12g} s .. {self.T_PERIOD_S:.12g} s",
            f"expectation on_drop: ON-state level = mean of v(SW) over {self.ON_WINDOW_S[0]:.12g}..{self.ON_WINDOW_S[1]:.12g} s at most {vd:.12g} V (the value of {req_drop}; a bound)",
            f"expectation off_clamp: max of v(SW) at most {params['clamp_max'].value:.12g} V = V_load + v_f (a bound)",
        ]
        plan.changes = _changes(t, self.title, topology, [r1, q1, d1, j1, j2], nets, params, sim, constraints)
        return plan

    def theory(self, ir: CircuitIR) -> list[TheorySection]:
        v_ctl, v_load, i_load, drop, v_be, beta, v_f, i_b, r_b, clamp = (parameter_value(ir, k) for k in ("v_ctl", "v_load", "i_load", "max_drop", "v_be", "beta_forced", "v_f", "i_b", "r_b", "clamp_max"))
        return [
            TheorySection("동작 원리와 식", (
                "NPN 트랜지스터 Q1 이 부하의 아래쪽(로우사이드)을 GND 로 잇는 스위치입니다. 제어 전압이 R1 을 통해 베이스 전류 I_B 를 흘리면 Q1 이 포화되어 "
                "컬렉터 전압이 수십~수백 mV 로 내려가고, 제어가 0 V 이면 꺼집니다. 포화를 보장하려고 베이스 전류를 '강제 전류이득' β_forced 로 정합니다.\n\n"
                "    I_B = I_load / β_forced,   R_B = (V_ctl − V_BE) / I_B\n\n"
                f"이 설계에서 I_load = {quantity(i_load, 'A')} 와 V_ctl = {quantity(v_ctl, 'V')} 는 요구사항, β_forced = {quantity(beta)} 와 V_BE = {quantity(v_be, 'V')} 는 확인된 선택값입니다.\n\n"
                f"    I_B = {quantity(i_load, 'A')}/{quantity(beta)} = {quantity(i_b, 'A')}   (`calc.bjt_switch.base_current`)\n"
                f"    R_B = ({quantity(v_ctl, 'V')} − {quantity(v_be, 'V')})/{quantity(i_b, 'A')} = {quantity(_solved(ir, 'r_b'), 'ohm')}   (`calc.bjt_switch.base_resistor`)"
                + ("\n" + _placed_line(ir, "r_b", "ohm").rstrip("\n") if standard_of(ir, "r_b") is not None else "")
            )),
            TheorySection("환류 다이오드와 OFF 상태", (
                "유도성 부하(릴레이 코일)는 스위치가 꺼져도 전류를 계속 흘리려 하므로, 부하 양단의 D1 이 그 전류를 받아 컬렉터 전압을 V_load + V_F 로 묶습니다. "
                f"시뮬레이션은 부하를 VLOAD 에서 SW 로 흐르는 정전류 {quantity(i_load, 'A')} 로 둡니다(단순화한 정전류 부하 모델이지 코일이 아님: 전류가 변하지 않음). "
                "그래서 OFF 상태는 항상 환류 상태이고 ON 상태는 정확히 요구 전류를 흘립니다; 실제 코일은 턴온 때 L/R 로 전류가 올라가고 턴오프 때 다이오드를 통해 감쇠하지만, 두 판정에는 그 동역학이 필요 없어 모델에 없습니다.\n\n"
                f"    off_clamp: max v(SW) ≤ V_load + v_f = {quantity(v_load, 'V')} + {quantity(v_f, 'V')} = {quantity(clamp, 'V')}   (`calc.bjt_switch.clamp_max`)"
            )),
            TheorySection("ON 전압 강하 판정", (
                f"제어 펄스가 켜진 구간의 창 안에서 v(SW) 의 시간 평균(= ON 상태 레벨)을 구해 요구사항 max_voltage_drop = {quantity(drop, 'V')} 이하인지 봅니다(한쪽 한계 `at_most`, 허용차 없음). "
                "max_voltage_drop 은 이 템플릿에서 'ON 상태의 허용 레벨'이고 창 평균으로 판정합니다: 창 안의 최대값 축약은 엔진에 없고(MAX 는 전체 과도 — OFF 상태의 클램프 전압이 들어감), 정전류 부하에서는 ON 레벨이 평탄해 창 안의 평균과 최대가 같으며 창은 양쪽 에지를 뺍니다. "
                "한계 판정은 요구사항 id 를 직접 들 수 없으므로(컴파일러 규칙) SW 넷이 그 요구사항을 섬기고 제약 c.bjt_lowside_switch.on_drop 이 뜻을 적습니다. "
                "베이스 구동이 모자라면(β_forced 가 너무 크거나 V_ctl 이 낮으면) 컬렉터가 포화되지 못해 v(SW) 가 올라가고 이 판정이 FAIL 로 드러납니다: 설계 규칙은 β_forced, 판정은 ngspice 입니다."
            )),
        ]

    def part_notes(self, ir: CircuitIR) -> dict[str, PartNote]:
        v_ctl, v_load, i_load, v_be, i_b, r_b, clamp = (parameter_value(ir, k) for k in ("v_ctl", "v_load", "i_load", "v_be", "i_b", "r_b", "clamp_max"))
        p_r1 = _div(_mul(_sub(v_ctl, v_be), _sub(v_ctl, v_be)), r_b)  # (V_ctl - V_BE)^2 / R_B while ON
        return {
            "R1": PartNote(
                role="R1: 베이스 저항 (CTL–BASE)",
                why=(
                    f"R_B = (V_ctl − V_BE)/I_B = {quantity(_solved(ir, 'r_b'), 'ohm')} (`calc.bjt_switch.base_resistor`), I_B = {quantity(i_b, 'A')}. "
                    + ("" if standard_of(ir, "r_b") is None else f"부품은 그 값을 넘지 않는 표준값 {quantity(r_b, 'ohm')} (베이스 구동이 설계값보다 줄지 않게). ")
                    + "스루홀 축형 저항."
                ),
                criteria=_resistor_criteria(r_b, p_r1, "5 % 이하 (값이 크면 구동 부족, 작으면 제어단 부담)", spelled=_spelled(ir, "R1"), standard=standard_of(ir, "r_b")),
                substitutes=_resistor_substitutes(r_b, "Resistor_THT"),
            ),
            "Q1": PartNote(
                role="Q1: NPN 로우사이드 스위치 (C = SW, B = BASE, E = GND)",
                why=f"부하 전류 {quantity(i_load, 'A')} 를 포화 상태로 흘림; 핀은 라이브러리 심볼의 이름 C/B/E 로 배선.",
                criteria=[
                    f"연속 컬렉터 전류 정격 ≥ 2 × {quantity(i_load, 'A')} (2N3904: 200 mA)",
                    f"V_CEO ≥ {quantity(clamp, 'V')} (OFF 상태의 컬렉터 전압 = V_load + V_F)",
                    f"h_FE(min) at {quantity(i_load, 'A')} ≥ 3 × β_forced (포화 여유; 데이터시트로 확인)",
                    "ON 상태 손실 = V_CE(sat) × I_load 가 패키지 정격 안 (2N3904 TO-92: 625 mW)",
                ],
                substitutes=[unverified("BC337 (TO-92, 800 mA)", "핀 배열이 다를 수 있음: 심볼 핀 이름 C/B/E 로 배선되므로 라이브러리 심볼을 바꾸면 됨"),
                             unverified("2N2222A (TO-92/TO-18, 800 mA)", "같은 주의")],
            ),
            "D1": PartNote(
                role="D1: 환류 다이오드 (A = SW, K = VLOAD), 부하 양단",
                why=f"스위치가 꺼질 때 코일 전류 {quantity(i_load, 'A')} 를 받아 컬렉터를 V_load + V_F ≈ {quantity(clamp, 'V')} 로 묶음.",
                criteria=[
                    f"평균 정류 전류 ≥ {quantity(i_load, 'A')} (1N4001: 1 A)",
                    f"반복 역전압 ≥ 2 × {quantity(v_load, 'V')} (1N4001: 50 V)",
                    "PWM 구동이면 고속 회복형(1N4148·쇼트키)으로; 단순 ON/OFF 이면 1N4001 로 충분",
                ],
                substitutes=[unverified("1N4148 (DO-35, 200 mA)", "부하 전류 200 mA 이하일 때; 풋프린트를 Diode_THT:D_DO-35_SOD27_P7.62mm_Horizontal 로"),
                             unverified("1N5819 쇼트키 (DO-41, 1 A)", "낮은 V_F, 빠른 회복; 현재 풋프린트 그대로")],
            ),
            "J1": _header_note("J1: CTL / GND 헤더", "1 = CTL, 2 = GND", f"제어 신호 {quantity(v_ctl, 'V')} 를 넣는 곳 (MCU 핀·로직 출력)."),
            "J2": _header_note("J2: VLOAD / SW / GND 헤더", "1 = VLOAD, 2 = SW, 3 = GND", f"부하(릴레이 코일 등)는 1–2 사이에, 부하 전원 {quantity(v_load, 'V')} 은 1–3 사이에."),
        }


def _sub(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a - b
