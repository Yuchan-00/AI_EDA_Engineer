"""Second-order LC low-pass (Butterworth, Z0-terminated) from ``cutoff_frequency`` and ``system_impedance`` (D9 catalog, tier 1).

A series L and a shunt C between a source termination R1 = Z0 and a load R2 = Z0, both on the board, so the
simulated transfer is the doubly terminated prototype's: passband level v_pass = probe / 2, -3.01 dB at f_c.
L = sqrt(2) Z0 / (2 pi f_c), C = sqrt(2) / (2 pi f_c Z0) (the n = 2 Butterworth g values g1 = g2 = sqrt(2)).
The expectation is the level at the sweep point f_c relative to the passband (``db_at`` with ``ref`` = v_pass):
nominal -10 log10(1 + (f_c / f_c)^4) = -3.0103 dB, from the requirement's own frequency, so the check cannot
pass by construction (a wrong L or C moves the corner and the level at f_c).

The on-board source termination R1 is a confirmed choice (``source_termination = on_board``): the board expects
to be driven by a low-impedance source (a buffer, or a generator set to high-impedance-load mode). A source that
already presents Z0 would add to R1 (2 Z0 on the source side) and the doubly terminated response no longer
applies - the choice text says so, and a variant without R1 is a later catalog item, not a hidden option.

Validity (engineering bounds chosen for this template, not measured): 1 kHz <= f_c <= 30 MHz and 10 ohm <= Z0
<= 600 ohm. Below 1 kHz the inductor exceeds ~50 mH at 50 ohm (a part whose DC resistance and self-resonance the
ideal model leaves out); above 30 MHz the 0805 parts' parasitics and the board's trace inductance are a visible
share of the reactances. Every input inside the bounds builds exactly what it built before (TEMPLATE_VERSION
unchanged).
"""

from __future__ import annotations

import math

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
from ai_eda.tools.calc.basic import lc2_butterworth_level_db, lc2_series_l, lc2_shunt_c, rc_ac_fstart, rc_ac_fstop, voltage_divider_output
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.spice import SpiceAnalysis

from ai_eda.design.base import NO_RECORD, PartNote, Plan, Template, TheorySection, parameter_value, quantity, number, structural_provenance, unverified
from ai_eda.design.inputs import DesignInput
from ai_eda.design.library_parts import TemplateRefusal, library_component, require_pins, two_terminals
from ai_eda.design.templates import (
    CAPACITOR,
    HEADER_2,
    RESISTOR,
    _changes,
    _choice,
    _div,
    _header_note,
    _is_choice,
    _mul,
    _net,
    _net_line,
    _part_line,
    _refused,
    _resistor_criteria,
    _resistor_substitutes,
    _spelled,
    _value_text,
)

INDUCTOR = (("Device", "L"), ("Inductor_SMD", "L_0805_2012Metric"))


class LcLowpassTemplate(Template):
    """Second-order LC low-pass, Butterworth, terminated in ``system_impedance`` at both ends (module docstring)."""

    id = "lc_lowpass"
    title = "second-order LC low-pass (Butterworth, Z0-terminated)"
    triggers = ("cutoff_frequency", "system_impedance")
    needs = ("cutoff_frequency", "system_impedance")
    serves = ("cutoff_frequency", "system_impedance")

    TOL_DB = 0.2
    AC_PROBE_V = 1.0
    AC_VARIATION = "dec"
    #: 500 points per decade: the n = 2 corner is steep (-8.7 dB per ln-unit of frequency), so the two sweep samples that
    #: bracket f_c must sit inside the 0.2 dB tolerance for the interpolated level to be judged (otherwise UNRESOLVED)
    AC_POINTS = 500
    SOURCE_TERMINATION = "on_board"
    #: the cutoff frequencies (Hz) and impedances (ohm) the ideal LC model with 0805 parts is stated valid for (engineering bounds, not measured)
    F_MIN, F_MAX = 1_000.0, 30_000_000.0
    Z_MIN, Z_MAX = 10.0, 600.0

    def build(self, ir: CircuitIR, inputs: dict[str, DesignInput], unusable: dict[str, str], library: KicadLibrary, *, confirmed: bool) -> Plan:
        t = self.id
        plan = Plan(template=t, title=self.title)
        f_c, z0 = inputs["cutoff_frequency"], inputs["system_impedance"]
        plan.inputs = {"cutoff_frequency": f_c, "system_impedance": z0}
        f, z = f_c.traced.value, z0.traced.value
        if not self.F_MIN <= f <= self.F_MAX:
            return _refused(plan, (
                f"cutoff frequency {f:.12g} Hz ({f_c.requirement.id}) is outside {self.F_MIN:.12g}..{self.F_MAX:.12g} Hz: below {self.F_MIN:g} Hz the inductor "
                f"exceeds tens of mH, whose DC resistance and self-resonance the ideal model leaves out; above {self.F_MAX:g} Hz the 0805 parts' parasitics and the "
                f"board's trace inductance are a visible share of the reactances (engineering bounds chosen for this template, not measured)"
            ))
        if not self.Z_MIN <= z <= self.Z_MAX:
            return _refused(plan, (
                f"system impedance {z:.12g} ohm ({z0.requirement.id}) is outside {self.Z_MIN:.12g}..{self.Z_MAX:.12g} ohm: the terminations are 0603 resistors and "
                f"the element values stay in the 0805 inductor / capacitor ranges only inside those bounds (engineering bounds chosen for this template, not measured)"
            ))
        term_choice, _ = _choice(t, "source_termination", self.SOURCE_TERMINATION, None,
                                 "R1 = Z0 on the board: drive the IN header from a low-impedance source (a buffer, or a generator in high-Z-load mode); "
                                 "a source that already presents Z0 would add to R1 and the doubly terminated response no longer applies", confirmed)
        tol_choice, tol = _choice(t, "tol_db", self.TOL_DB, "dB",
                                  "absolute tolerance of the level at f_c (0.2 dB: the ac grid is interpolated between 500 points per decade, whose two samples around f_c stay inside it)", confirmed)
        probe_choice, probe = _choice(t, "ac_probe", self.AC_PROBE_V, "V", "small-signal probe amplitude of the ac sweep", confirmed)
        var_choice, variation = _choice(t, "ac_variation", self.AC_VARIATION, None, "logarithmic ac sweep", confirmed)
        pts_choice, points = _choice(t, "ac_points", self.AC_POINTS, None,
                                     "points per decade of the ac sweep (500: the samples bracketing f_c stay inside the 0.2 dB tolerance of the steep n = 2 corner)", confirmed)
        plan.choices = [term_choice, tol_choice, probe_choice, var_choice, pts_choice]
        params: dict[str, Traced] = {"f_c": f_c.traced, "z0": z0.traced}
        try:
            params["l"] = lc2_series_l(params["f_c"], params["z0"], ("f_c", "z0"))
            params["c"] = lc2_shunt_c(params["f_c"], params["z0"], ("f_c", "z0"))
            params["ac_probe"] = probe
            params["v_pass"] = voltage_divider_output(params["ac_probe"], params["z0"], params["z0"], ("ac_probe", "z0", "z0"))
            params["level_fc"] = lc2_butterworth_level_db(params["f_c"], params["f_c"], ("f_c", "f_c"))
            params["tol_db"] = tol
            params["ac_fstart"] = rc_ac_fstart(params["f_c"], ("f_c",))
            params["ac_fstop"] = rc_ac_fstop(params["f_c"], ("f_c",))
        except (ValueError, ZeroDivisionError) as e:
            return _refused(plan, f"{e} ({f_c.requirement.id} = {f:.12g} Hz, {z0.requirement.id} = {z:.12g} ohm)")
        plan.computed = [(k, params[k]) for k in ("l", "c", "v_pass", "level_fc", "ac_fstart", "ac_fstop")]
        req_f, req_z = f_c.requirement.id, z0.requirement.id
        try:
            r1 = library_component(library, "R1", _value_text(z), "source termination (Z0)", *RESISTOR, structural_provenance(t, "source termination"), [req_z])
            l1 = library_component(library, "L1", _value_text(params["l"].value), "series inductor", *INDUCTOR, structural_provenance(t, "series inductor"), [req_f])
            c1 = library_component(library, "C1", _value_text(params["c"].value), "shunt capacitor", *CAPACITOR, structural_provenance(t, "shunt capacitor"), [req_f])
            r2 = library_component(library, "R2", _value_text(z), "load termination (Z0)", *RESISTOR, structural_provenance(t, "load termination"), [req_z])
            j1 = library_component(library, "J1", "Conn_01x02", "IN / GND header", *HEADER_2, structural_provenance(t, "input header"), [])
            j2 = library_component(library, "J2", "Conn_01x02", "OUT / GND header", *HEADER_2, structural_provenance(t, "output header"), [])
            r1a, r1b = two_terminals(r1)
            l1a, l1b = two_terminals(l1)
            c1a, c1b = two_terminals(c1)
            r2a, r2b = two_terminals(r2)
            require_pins(j1, ("1", "2"))
            require_pins(j2, ("1", "2"))
        except TemplateRefusal as e:
            return _refused(plan, str(e))
        r1.electrical["resistance"] = params["z0"]
        r2.electrical["resistance"] = params["z0"]
        l1.electrical["inductance"] = params["l"]
        c1.electrical["capacitance"] = params["c"]
        for part, device, value, what in ((r1, SpiceDevice.R, params["z0"], "ideal resistor at Z0"), (r2, SpiceDevice.R, params["z0"], "ideal resistor at Z0"),
                                          (l1, SpiceDevice.L, params["l"], "ideal inductor at the calculator's value"), (c1, SpiceDevice.C, params["c"], "ideal capacitor at the calculator's value")):
            part.spice = SpiceBinding(device=device, value=value, pin_order=[p.number for p in part.pins], provenance=structural_provenance(t, what))
        for header in (j1, j2):
            header.spice = SpiceBinding(exclude=True, exclude_reason="connector, no electrical model", provenance=structural_provenance(t, "header excluded from the netlist"))
        nets = [
            _net("IN", NetKind.SIGNAL, [("J1", "1"), ("R1", r1a)], t),
            _net("FIN", NetKind.SIGNAL, [("R1", r1b), ("L1", l1a)], t),
            _net("OUT", NetKind.SIGNAL, [("L1", l1b), ("C1", c1a), ("R2", r2a), ("J2", "1")], t, [req_f, req_z]),
            _net("GND", NetKind.GROUND, [("J1", "2"), ("C1", c1b), ("R2", r2b), ("J2", "2")], t),
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
                id="level_fc", analysis_id="ac", vector="v(OUT)", reduce=Reduce.DB_AT, at=params["f_c"], nominal=params["level_fc"], tol_abs=params["tol_db"],
                params={"ref": params["v_pass"]}, requirement_id=req_f,
                provenance=structural_provenance(t, f"the level at the sweep point f_c = the value of {req_f}, relative to the passband level v_pass; the Butterworth corner is where it is -3.01 dB"),
            )],
        )
        topology = Topology(
            name="lc low-pass", domains=[CircuitDomain.ANALOG],
            rationale="second-order Butterworth LC low-pass between Z0 terminations: L = sqrt(2) Z0 / (2 pi f_c), C = sqrt(2) / (2 pi f_c Z0); -3.01 dB at f_c relative to the passband",
            provenance=structural_provenance(t, "selected by cutoff_frequency + system_impedance"),
            blocks=[Block(id="lowpass", function="second-order LC low-pass", domain=CircuitDomain.ANALOG, component_refs=["R1", "L1", "C1", "R2"], input_nets=["IN"], output_nets=["OUT"],
                          provenance=structural_provenance(t, "the one block of this template"))],
        )
        constraint = Constraint(
            id="c.lc_lowpass.cutoff", kind=ConstraintKind.ELECTRICAL, target="OUT",
            description=f"the cutoff frequency ({req_f}) is where the level relative to the passband is -3.01 dB (Butterworth n = 2); the expectation level_fc checks it at that sweep point",
            provenance=structural_provenance(t, "what 'cutoff frequency' means for this template"),
        )
        plan.parts = [_part_line(x) for x in (r1, l1, c1, r2, j1, j2)]
        plan.nets = [_net_line(n) for n in nets]
        plan.simulation = [
            f"stimulus VIN: dc {self.AC_PROBE_V:.12g} V with ac {self.AC_PROBE_V:.12g} V on IN; analysis ac {self.AC_VARIATION} {self.AC_POINTS} {params['ac_fstart'].value:.12g} Hz .. {params['ac_fstop'].value:.12g} Hz",
            f"expectation level_fc: v(OUT) at {f:.12g} Hz = {params['level_fc'].value:.12g} dB relative to v_pass = {params['v_pass'].value:.12g} V, +/- {self.TOL_DB:.12g} dB, verifies {req_f}",
        ]
        plan.changes = _changes(t, self.title, topology, [r1, l1, c1, r2, j1, j2], nets, params, sim, [constraint])
        return plan

    def theory(self, ir: CircuitIR) -> list[TheorySection]:
        f_c, z0, l, c, v_pass, level, tol, probe, f_start, f_stop = (parameter_value(ir, k) for k in ("f_c", "z0", "l", "c", "v_pass", "level_fc", "tol_db", "ac_probe", "ac_fstart", "ac_fstop"))
        f_check = _div(1.0, _mul(2.0 * math.pi * math.sqrt(2.0), None if l is None or c is None else math.sqrt(l * c)))
        return [
            TheorySection("동작 원리와 식", (
                "직렬 인덕터 L 과 병렬 커패시터 C 로 된 2차 저역통과 필터를 양쪽 Z0 저항(R1 전원 쪽, R2 부하 쪽)으로 종단한 Butterworth(최대 평탄) 프로토타입입니다. "
                "L 의 임피던스 j2πfL 은 주파수에 비례해 커지고 C 의 임피던스는 작아지므로 높은 주파수가 10배마다 −40 dB(2차)로 줄어듭니다.\n\n"
                "    g1 = g2 = √2 (n = 2 Butterworth),   L = g1·Z0/(2π·f_c),   C = g2/(2π·f_c·Z0)\n\n"
                f"이 설계에서 Z0 = {quantity(z0, 'ohm')} 와 f_c = {quantity(f_c, 'Hz')} 는 요구사항이고 L, C 는 계산기 `calc.lc.butterworth2.series_l` / `shunt_c` 가 풉니다.\n\n"
                f"    L = √2·{quantity(z0, 'ohm')}/(2π·{quantity(f_c, 'Hz')}) = {quantity(l, 'H')}\n"
                f"    C = √2/(2π·{quantity(f_c, 'Hz')}·{quantity(z0, 'ohm')}) = {quantity(c, 'F')}\n"
                f"    검산: f_c = 1/(2π·√2·√(L·C)) = {quantity(f_check, 'Hz')}"
            )),
            TheorySection("주파수 응답과 판정", (
                "    |H(f)|/|H(0)| = 1/√(1 + (f/f_c)⁴),   통과대역 |H(0)| = 1/2 (양단 종단: 전원 Z0 와 부하 Z0 의 분배)\n\n"
                f"프로브 VIN 은 dc {quantity(probe, 'V')} 에 ac 진폭 {quantity(probe, 'V')} 이므로 통과대역의 v(OUT) 은 v_pass = {quantity(v_pass, 'V')} (`calc.divider.v_out`, R1 = R2 = Z0) 이고, "
                f"기대값 level_fc 는 f_c 에서의 v(OUT) 을 v_pass 에 대한 dB 로 읽어 공칭값 {number(level, 5)} dB (`calc.lc.butterworth2.level_db`: −10·log10(1 + (f_c/f_c)⁴)) 와 비교합니다. "
                f"허용치 ±{quantity(tol, 'dB')} 는 사용자 확인 선택값입니다.\n\n"
                f"스위프는 `dec` 500 점/decade, f_start = f_c/100 = {quantity(f_start, 'Hz')} 부터 f_stop = 100·f_c = {quantity(f_stop, 'Hz')} 까지입니다. "
                "코너 기울기가 가파르므로(ln 단위 주파수당 −8.7 dB) f_c 를 감싸는 두 표본이 허용치 안에 들도록 100 점이 아니라 500 점을 씁니다; "
                "둘 중 하나라도 밖이면 보간값은 판정하지 않고 UNRESOLVED 로 둡니다."
            )),
            TheorySection("전원 종단에 관한 주의", (
                "R1 = Z0 가 보드 위에 있으므로 IN 헤더는 낮은 출력 임피던스의 전원(버퍼, 또는 고임피던스 부하 모드의 신호 발생기)으로 구동해야 합니다. "
                "이미 Z0 를 내는 전원을 연결하면 전원 쪽 종단이 2·Z0 가 되어 양단 종단 응답이 성립하지 않습니다(선택값 `source_termination` 참조)."
            )),
        ]

    def part_notes(self, ir: CircuitIR) -> dict[str, PartNote]:
        z0, l, c, probe = (parameter_value(ir, k) for k in ("z0", "l", "c", "ac_probe"))
        p_r = _div(_mul(probe, probe), _mul(4.0, z0))  # the probe's power in one termination at the passband (v_pass^2 / Z0)
        return {
            "R1": PartNote(
                role="R1: 전원 쪽 종단 저항 (IN–FIN), Z0",
                why=f"양단 종단 Butterworth 프로토타입의 전원 저항. 요구사항 system_impedance = {quantity(z0, 'ohm')} 그대로. 0603 SMD.",
                criteria=_resistor_criteria(z0, p_r, "1 % 이하 (종단 오차는 통과대역 리플과 코너 위치를 함께 움직임)", chosen=_is_choice(ir, "z0"), spelled=_spelled(ir, "R1")),
                substitutes=_resistor_substitutes(z0, "Resistor_SMD"),
            ),
            "L1": PartNote(
                role="L1: 직렬 인덕터 (FIN–OUT)",
                why=f"L = √2·Z0/(2π·f_c) = {quantity(l, 'H')} (`calc.lc.butterworth2.series_l`). 0805 SMD 는 이 템플릿의 기본 풋프린트.",
                criteria=[
                    f"인덕턴스 {quantity(l, 'H')}, 공차 5 % 이하 (Δf_c/f_c ≈ −ΔL/(2L))",
                    "자기 공진 주파수 ≥ 10·f_c, DC 저항 ≤ Z0/100 (이상 모델에는 둘 다 없음: 데이터시트로 확인)",
                    f"정격 전류 ≥ 2 × 통과대역 전류 (프로브 기준 {quantity(_div(probe, _mul(2.0, z0)), 'A')}; 실제 신호 진폭 요구사항은 {NO_RECORD})",
                ],
                substitutes=[unverified(f"같은 값 {quantity(l, 'H')}의 0805 권선형 인덕터", "현재 풋프린트 그대로"), unverified("같은 값의 1210 인덕터(더 높은 전류)", "풋프린트를 Inductor_SMD:L_1210_3225Metric 로")],
            ),
            "C1": PartNote(
                role="C1: 병렬(shunt) 커패시터 (OUT–GND)",
                why=f"C = √2/(2π·f_c·Z0) = {quantity(c, 'F')} (`calc.lc.butterworth2.shunt_c`). 0603 SMD.",
                criteria=[
                    f"정전용량 {quantity(c, 'F')}, 공차 5 % 이하 (Δf_c/f_c ≈ −ΔC/(2C))",
                    "유전체 C0G/NP0 권장 (X7R 은 전압·온도에 따라 용량이 수십 % 변해 f_c 가 움직임)",
                    f"정격 전압 ≥ 2 × 신호 진폭 (프로브 진폭 {quantity(probe, 'V')}; 실제 신호 진폭 요구사항은 {NO_RECORD})",
                ],
                substitutes=[unverified(f"같은 값 {quantity(c, 'F')}의 0603 C0G/NP0 MLCC", "현재 풋프린트 그대로")],
            ),
            "R2": PartNote(
                role="R2: 부하 쪽 종단 저항 (OUT–GND), Z0",
                why=f"양단 종단 프로토타입의 부하 저항 = system_impedance {quantity(z0, 'ohm')}. 0603 SMD.",
                criteria=_resistor_criteria(z0, p_r, "1 % 이하", chosen=_is_choice(ir, "z0"), spelled=_spelled(ir, "R2")),
                substitutes=_resistor_substitutes(z0, "Resistor_SMD"),
            ),
            "J1": _header_note("J1: IN / GND 헤더", "1 = IN, 2 = GND", "낮은 출력 임피던스의 전원으로 구동 (R1 이 보드 위의 종단)."),
            "J2": _header_note("J2: OUT / GND 헤더", "1 = OUT, 2 = GND", "R2 가 보드 위의 종단이므로 외부 부하는 Z0 보다 훨씬 커야 함."),
        }
