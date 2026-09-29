"""The RF requirement keys (``ai_eda.design.inputs``) and the LLM extraction's knowledge of them.

Typed answers are read the way the requirement agent records them
(``_answer_requirement``: the text as the user's value); a level becomes
watts / V/m only through the named ``calc.rf`` converter, and a copied input
re-reads identically (``design.inputs_vs_requirements`` PASS). No alias is a
bare physical word or merges two quantities (contract K2 of the RF foundation
design), and the extraction prompt names every canonical RF key.
"""

from __future__ import annotations

import pytest

from ai_eda.agents.requirement import _answer_requirement
from ai_eda.design import check_inputs_vs_requirements
from ai_eda.design.inputs import (
    BATTERY_ALIASES,
    KEY_ALIASES,
    MODULATION_ALIASES,
    MODULATIONS,
    PARSED_NOTE_PREFIX,
    RATIO_UNIT,
    RF_KEY_ALIASES,
    RF_UNIT_OF,
    SYMMETRIC_TOLERANCE_KEYS,
    UNIT_OF,
    canonical_key,
    is_template_input,
    read_inputs,
    read_modulation,
    read_value,
)
from ai_eda.ir import CircuitIR, ProjectMeta, Requirement, RequirementKind, ValidationStatus, llm_generated, user_requirement
from ai_eda.llm.extraction import EXTRACTION_VERSION, ExtractedValue, RequirementExtraction, ground_extraction, upgrade_confirmed
from ai_eda.llm.prompts import REQUIREMENT_EXTRACTION_SYSTEM

#: the bare physical words no key or alias may be (contract K2)
BARE_WORDS = ("frequency", "power", "bandwidth", "impedance", "sensitivity", "range", "gain", "deviation", "depth", "index", "voltage", "current")
#: near-bare aliases the design review removed: each would merge two different quantities
MERGING_ALIASES = ("center_frequency", "operating_frequency", "max_deviation", "if_bandwidth", "rx_bandwidth", "vbat", "v_bat", "v_batt")


def _ir(*answers: tuple[str, object]) -> CircuitIR:
    ir = CircuitIR(project=ProjectMeta(id="rf", name="rf"))
    for key, answer in answers:
        ir.requirements.requirements.append(_answer_requirement(key, answer))  # type: ignore[arg-type]
    return ir


def _numeric(key: str, value: object, unit: str | None) -> Requirement:
    """A confirmed numeric requirement, as a confirmed extraction stores it (number + unit as written)."""
    return Requirement(id=f"req.{key}", key=key, text=f"{key}: {value} {unit}", kind=RequirementKind.EXPLICIT, value=user_requirement(value, unit), category="electrical")


# --------------------------------------------------------------------------- typed answers


def test_typed_rf_answers_read_in_their_units() -> None:
    ir = _ir(
        ("carrier_frequency", "447.0125 MHz"), ("tx_power", "27 dBm"), ("rx_sensitivity", "-120 dBm"), ("modulation_depth", "80 %"),
        ("modulation_index", "5"), ("frequency_tolerance", "2.5 ppm"), ("link_range", "2 km"), ("antenna_gain", "2.15 dBi"),
        ("field_strength_limit", "50 mV/m"), ("channel_spacing", "12.5 kHz"), ("system_impedance", "50 ohm"),
    )
    found, unusable = read_inputs(ir)
    assert unusable == {}
    got = {k: (d.traced.value, d.traced.unit) for k, d in found.items()}
    assert got["carrier_frequency"] == (447_012_500.0, "Hz")
    assert got["tx_power"][1] == "W" and got["tx_power"][0] == pytest.approx(0.501187233627, rel=1e-12)
    assert got["rx_sensitivity"] == (-120.0, "dBm") and got["modulation_depth"] == (80.0, "percent")
    assert got["modulation_index"] == (5.0, RATIO_UNIT) and got["frequency_tolerance"] == (2.5, "ppm")
    assert got["link_range"] == (2000.0, "m") and got["antenna_gain"] == (2.15, "dBi")
    assert got["field_strength_limit"] == (pytest.approx(0.05, rel=1e-15), "V/m") and got["channel_spacing"] == (12500.0, "Hz")
    assert got["system_impedance"] == (50.0, "ohm")
    tx = found["tx_power"].traced
    assert tx.provenance.note == f"{PARSED_NOTE_PREFIX}req.tx_power: '27 dBm' -> 27 dBm -> 0.501187233627 W (calc.rf.dbm_to_w)"
    assert tx.provenance.derived_from == ["req.tx_power"] and is_template_input(tx)
    # a plain watt answer needs no calculator
    found, _ = read_inputs(_ir(("tx_power", "0.5 W")))
    assert found["tx_power"].traced.value == 0.5 and "calc.rf" not in (found["tx_power"].traced.provenance.note or "")


def test_a_sensitivity_in_volts_and_other_unit_mismatches_are_refused_with_the_reason() -> None:
    _, unusable = read_inputs(_ir(("rx_sensitivity", "0.25 uV")))
    assert unusable["rx_sensitivity"] == (
        "req.rx_sensitivity: '0.25 uV' is a voltage; a receiver sensitivity in volts needs the source impedance and the EMF / PD convention - state it in dBm"
    )
    assert "is a voltage" in (read_value(_numeric("rx_sensitivity", 0.25, "uV"), "dBm")[1] or "")
    _, unusable = read_inputs(_ir(("tx_power", "10 dBc"), ("modulation_index", "5 x"), ("carrier_frequency", "±2.5 MHz")))
    assert unusable["tx_power"] == "req.tx_power: '10 dBc' is a dBc quantity, not W"  # only dBm / dBW convert to W
    assert unusable["modulation_index"] == "req.modulation_index: '5 x' is not one plain number"
    assert "a tolerance (±2500000 Hz), not a value" in unusable["carrier_frequency"]
    _, unusable = read_inputs(_ir(("tx_power", "4000 dBm")))
    assert "calc.rf.dbm_to_w refused it" in unusable["tx_power"] and "overflows" in unusable["tx_power"]


def test_a_numeric_dbm_requirement_reads_as_watts_and_rereads_pass() -> None:
    """What a confirmed extraction stores: the number 27 with the unit 'dBm'."""
    req = _numeric("tx_power", 27.0, "dBm")
    t, why = read_value(req, "W")
    assert why is None and t is not None and t.value == pytest.approx(0.501187233627, rel=1e-12) and t.unit == "W"
    assert t.provenance.note == f"{PARSED_NOTE_PREFIX}req.tx_power: 27.0 dBm -> 0.501187233627 W (calc.rf.dbm_to_w)"
    t2, _ = read_value(_numeric("tx_power", -3.0, "dBW"), "W")
    assert t2 is not None and t2.value == pytest.approx(10 ** -0.3, rel=1e-15) and "calc.rf.dbw_to_w" in (t2.provenance.note or "")
    t3, _ = read_value(_numeric("field_strength_limit", 94.0, "dBuV/m"), "V/m")
    assert t3 is not None and round(t3.value, 7) == 0.0501187 and "calc.rf.dbuvm_to_vm" in (t3.provenance.note or "")
    t4, _ = read_value(_ir(("field_strength_limit", "94 dBuV/m")).requirements.requirements[0], "V/m")
    assert t4 is not None and t4.value == t3.value
    # the design copies the converted input; a later run re-reads the requirement the same way
    ir = CircuitIR(project=ProjectMeta(id="rf", name="rf"))
    ir.requirements.requirements.extend([req, _answer_requirement("modulation_index", "5")])
    ir.parameters["p_tx"] = t
    ratio, _ = read_value(ir.requirements.requirements[1], RATIO_UNIT)
    assert ratio is not None and ratio.unit == RATIO_UNIT  # never None: the re-read goes through read_value(req, t.unit)
    ir.parameters["beta"] = ratio
    res = check_inputs_vs_requirements(ir)
    assert res is not None and res.status is ValidationStatus.PASS and set(res.details["parameters"]) == {"p_tx", "beta"}
    # the requirement moved: the copy no longer equals it
    ir.requirements.requirements[0] = _numeric("tx_power", 30.0, "dBm")
    assert check_inputs_vs_requirements(ir).status is ValidationStatus.FAIL  # type: ignore[union-attr]


def test_a_symmetric_answer_is_a_value_only_under_frequency_tolerance_and_frequency_deviation() -> None:
    assert SYMMETRIC_TOLERANCE_KEYS == frozenset({"frequency_tolerance", "frequency_deviation"})
    found, unusable = read_inputs(_ir(("frequency_stability", "±2.5 ppm")))
    assert unusable == {} and found["frequency_tolerance"].traced.value == 2.5 and found["frequency_tolerance"].traced.unit == "ppm"
    # a confirmed extraction of "±2.5 ppm" stores the symmetric range [-2.5, 2.5]
    t, why = read_value(_numeric("frequency_tolerance", [-2.5, 2.5], "ppm"), "ppm")
    assert why is None and t is not None and t.value == 2.5
    assert read_value(_numeric("frequency_tolerance", [-2.5, 3.0], "ppm"), "ppm")[0] is None  # not symmetric: not one number
    assert read_value(_ir(("frequency_tolerance", "±2.5 ppm max")).requirements.requirements[0], "ppm")[0] is None
    _, unusable = read_inputs(_ir(("audio_bandwidth", "±3 kHz")))
    assert "a tolerance" in unusable["audio_bandwidth"]
    _, unusable = read_inputs(_ir(("carrier_frequency", "±446 MHz")))
    assert "a tolerance" in unusable["carrier_frequency"]


def test_a_plus_minus_peak_deviation_reads_as_its_magnitude() -> None:
    """``±2.5 kHz`` is how a peak FM deviation is written (PMR446 / NBFM): formerly refused as 'a tolerance, not a value'."""
    for key, answer, hz in (("frequency_deviation", "±2.5 kHz", 2500.0), ("peak_deviation", "±2.5 kHz", 2500.0), ("fm_deviation", "±5 kHz", 5000.0)):
        found, unusable = read_inputs(_ir((key, answer)))
        assert unusable == {}, (key, unusable)
        assert (found["frequency_deviation"].traced.value, found["frequency_deviation"].traced.unit) == (hz, "Hz")
    # the grounded extraction payload of a quoted "±2.5 kHz" is the symmetric range [-2500, 2500] Hz
    for value, unit in (([-2500.0, 2500.0], "Hz"), ([-2.5, 2.5], "kHz")):
        t, why = read_value(_numeric("frequency_deviation", value, unit), "Hz")
        assert why is None and t is not None and t.value == 2500.0 and "read as its magnitude" in (t.provenance.note or "")
    assert read_value(_numeric("frequency_deviation", [-2500.0, 3000.0], "Hz"), "Hz")[0] is None
    assert read_value(_ir(("frequency_deviation", "±2.5 kHz max")).requirements.requirements[0], "Hz")[0] is None
    # the copied value re-reads identically
    ir = _ir(("frequency_deviation", "±2.5 kHz"))
    t, _ = read_value(ir.requirements.requirements[0], "Hz")
    ir.parameters["delta_f"] = t  # type: ignore[assignment]
    res = check_inputs_vs_requirements(ir)
    assert res is not None and res.status is ValidationStatus.PASS
    # end to end through the grounding the user chose (LLM extraction): the quote "±2.5 kHz" becomes a usable input
    raw = "FM 무전기, ±2.5 kHz deviation"
    doc = {"requirements": [{"key": "frequency_deviation", "text": "deviation", "kind": "explicit", "category": "electrical", "quote": "±2.5 kHz deviation",
                             "value": {"quote": "±2.5 kHz", "number": 2.5, "unit": "kHz", "number_high": None}, "rationale": None}],
           "questions": [], "conflicts": [], "assumptions": [], "application": None, "jurisdictions": []}
    g = ground_extraction(raw, RequirementExtraction.model_validate(doc), "test/model")
    [r] = g.requirements
    assert g.demoted == [] and r.value is not None and r.value.value == [-2500.0, 2500.0] and r.value.unit == "Hz"
    ir = CircuitIR(project=ProjectMeta(id="rf", name="rf"))
    ir.requirements.requirements.extend(upgrade_confirmed(g.requirements))
    found, unusable = read_inputs(ir)
    assert unusable == {} and found["frequency_deviation"].traced.value == 2500.0


def test_system_and_antenna_impedance_and_the_three_bandwidths_are_different_keys() -> None:
    found, unusable = read_inputs(_ir(
        ("system_impedance", "50 ohm"), ("antenna_impedance", "36 ohm"),
        ("channel_bandwidth", "12.5 kHz"), ("occupied_bandwidth", "8.5 kHz"), ("channel_spacing", "12.5 kHz"),
    ))
    assert unusable == {}
    assert (found["system_impedance"].traced.value, found["antenna_impedance"].traced.value) == (50.0, 36.0)
    assert (found["channel_bandwidth"].traced.value, found["occupied_bandwidth"].traced.value, found["channel_spacing"].traced.value) == (12500.0, 8500.0, 12500.0)
    # two statements of ONE key that differ are ambiguous, as for every key (27 dBm is 0.501187 W, not 0.5 W)
    _, unusable = read_inputs(_ir(("tx_power", "0.5 W"), ("transmit_power", "27 dBm")))
    assert unusable["tx_power"].startswith("ambiguous: req.tx_power says 0.5 W, req.transmit_power says 0.501187233627 W")
    found, unusable = read_inputs(_ir(("tx_power", "27 dBm"), ("p_tx", "27 dBm")))
    assert unusable == {} and found["tx_power"].requirement.id == "req.tx_power"


def test_battery_voltage_is_the_only_battery_alias_and_rf_keys_carry_their_units() -> None:
    assert BATTERY_ALIASES == ("battery_voltage",) and canonical_key("battery_voltage") == "input_voltage"
    assert canonical_key("vbat") is None and canonical_key("v_bat") is None
    found, _ = read_inputs(_ir(("battery_voltage", "3.7 V")))
    assert found["input_voltage"].traced.value == 3.7
    assert canonical_key("modulation_type") == "modulation" and canonical_key("emission_mode") == "modulation"
    assert canonical_key("rf_frequency") == "carrier_frequency" and canonical_key("obw") == "occupied_bandwidth"
    for key, unit in RF_UNIT_OF.items():
        assert UNIT_OF[key] == unit and KEY_ALIASES[key] == RF_KEY_ALIASES[key]
    assert UNIT_OF["tx_power"] == UNIT_OF["erp"] == UNIT_OF["eirp"] == "W" and UNIT_OF["modulation_index"] == RATIO_UNIT


def test_no_rf_alias_is_a_bare_word_or_merges_two_quantities() -> None:
    every = [a for aliases in RF_KEY_ALIASES.values() for a in aliases] + list(MODULATION_ALIASES) + list(BATTERY_ALIASES)
    assert not set(BARE_WORDS) & set(every) and not set(MERGING_ALIASES) & set(every)
    assert not set(BARE_WORDS) & set(RF_KEY_ALIASES)
    # every alias means exactly one key (the tuples are pairwise disjoint across all keys)
    seen: dict[str, str] = {}
    for canon, aliases in (*KEY_ALIASES.items(), ("modulation", MODULATION_ALIASES)):
        for a in aliases:
            assert a not in seen, f"{a} is an alias of both {seen[a]} and {canon}"
            seen[a] = canon
    assert all(canonical_key(a) == canon for a, canon in seen.items())
    assert "tx_power" not in KEY_ALIASES["erp"] + KEY_ALIASES["eirp"] and "erp" not in KEY_ALIASES["tx_power"]


def test_the_bare_key_frequency_names_no_quantity() -> None:
    """Contract K2: part D removes ``frequency`` from ``oscillation_frequency``; no key, RF or not, may mean it."""
    for word in BARE_WORDS:
        assert canonical_key(word) is None, word
        assert all(word not in aliases for aliases in KEY_ALIASES.values()), word


# --------------------------------------------------------------------------- modulation


def test_read_modulation_takes_exactly_one_confirmed_name() -> None:
    assert MODULATIONS == ("am", "fm", "pm", "ssb", "dsb", "fsk", "gfsk", "ask", "ook", "psk", "lora")
    assert read_modulation(_ir()) == (None, None)  # not stated: a template asks
    for text, name in (("FM", "fm"), ("narrowband FM (NFM excluded)", "fm"), ("am", "am"), ("주파수 변조", "fm"), ("진폭변조", "am"),
                       ("GFSK", "gfsk"), ("LoRa", "lora"), ("DSB-SC", "dsb"), ("진폭 변조(AM)", "am")):
        assert read_modulation(_ir(("modulation", text))) == (name, None), text
    name, why = read_modulation(_ir(("modulation_type", "AM/FM")))
    assert name is None and "names several modulations (am, fm)" in (why or "")
    name, why = read_modulation(_ir(("modulation", "analog voice")))
    assert name is None and "names no modulation (one of am, fm, pm" in (why or "")
    name, why = read_modulation(_ir(("modulation", "FM"), ("emission_mode", "AM")))
    assert name is None and (why or "").startswith("ambiguous: req.modulation says fm, req.emission_mode says am")
    assert read_modulation(_ir(("modulation", "FM"), ("modulation_type", "fm"))) == ("fm", None)
    ir = _ir()
    ir.requirements.requirements.append(Requirement(id="req.modulation", key="modulation", text="modulation: FM", kind=RequirementKind.EXPLICIT, value=llm_generated("FM", model="m", note="quote: 'FM'"), category="electrical"))
    name, why = read_modulation(ir)
    assert name is None and "not yet the user's" in (why or "")
    name, why = read_modulation(_ir(("modulation", 5)))
    assert name is None and "is not a modulation name" in (why or "")


# --------------------------------------------------------------------------- the LLM extraction


def test_the_extraction_prompt_names_every_rf_key_and_the_level_units() -> None:
    assert EXTRACTION_VERSION == "0.4"
    for key in (*RF_KEY_ALIASES, "modulation", "battery_voltage"):
        assert key in REQUIREMENT_EXTRACTION_SYSTEM, key
    for unit in ("dBm", "dBW", "dBc", "dBi", "dBuV/m", "ppm", "V/m"):
        assert unit in REQUIREMENT_EXTRACTION_SYSTEM, unit
    assert "Never use a bare key such as frequency" in REQUIREMENT_EXTRACTION_SYSTEM
    assert "A power in dBm stays in dBm" in REQUIREMENT_EXTRACTION_SYSTEM
    assert "For a radio transmitter, receiver or transceiver request, ask (required)" in REQUIREMENT_EXTRACTION_SYSTEM
    assert "dBm" in ExtractedValue.model_fields["unit"].description  # type: ignore[operator]


def test_grounding_keeps_a_level_as_written_and_the_confirmed_value_reads_as_watts() -> None:
    raw = "447.0125 MHz 무전기, 출력 27 dBm, 수신 감도 -120 dBm, 50 ohm, FM"
    doc = {
        "requirements": [
            {"key": "carrier_frequency", "text": "carrier", "kind": "explicit", "category": "electrical", "quote": "447.0125 MHz 무전기",
             "value": {"quote": "447.0125 MHz", "number": 447.0125, "unit": "MHz", "number_high": None}, "rationale": None},
            {"key": "tx_power", "text": "tx power", "kind": "explicit", "category": "electrical", "quote": "출력 27 dBm",
             "value": {"quote": "27 dBm", "number": 27, "unit": "dBm", "number_high": None}, "rationale": None},
            {"key": "rx_sensitivity", "text": "sensitivity", "kind": "explicit", "category": "electrical", "quote": "수신 감도 -120 dBm",
             "value": {"quote": "-120 dBm", "number": -120, "unit": "dBm", "number_high": None}, "rationale": None},
            # a model that converts the level itself is caught: the quote says dBm
            {"key": "erp", "text": "erp", "kind": "explicit", "category": "electrical", "quote": "출력 27 dBm",
             "value": {"quote": "27 dBm", "number": 0.5, "unit": "W", "number_high": None}, "rationale": None},
            {"key": "modulation", "text": "FM", "kind": "explicit", "category": "electrical", "quote": "FM", "value": None, "rationale": None},
        ],
        "questions": [], "conflicts": [], "assumptions": [], "application": None, "jurisdictions": [],
    }
    g = ground_extraction(raw, RequirementExtraction.model_validate(doc), "test/model")
    by_key = {r.key: r for r in g.requirements}
    assert by_key["carrier_frequency"].value.value == 447_012_500.0 and by_key["carrier_frequency"].value.unit == "Hz"  # type: ignore[union-attr]
    assert (by_key["tx_power"].value.value, by_key["tx_power"].value.unit) == (27.0, "dBm")  # type: ignore[union-attr]
    assert (by_key["rx_sensitivity"].value.value, by_key["rx_sensitivity"].value.unit) == (-120.0, "dBm")  # type: ignore[union-attr]
    assert [k for k, _ in g.demoted] == ["erp"] and "unit mismatch" in dict(g.demoted)["erp"]
    confirmed = upgrade_confirmed(g.requirements)
    ir = CircuitIR(project=ProjectMeta(id="rf", name="rf"))
    ir.requirements.requirements.extend(confirmed)
    found, unusable = read_inputs(ir)
    assert found["carrier_frequency"].traced.value == 447_012_500.0
    assert found["tx_power"].traced.value == pytest.approx(0.501187233627, rel=1e-12) and "calc.rf.dbm_to_w" in (found["tx_power"].traced.provenance.note or "")
    assert found["rx_sensitivity"].traced.value == -120.0
    assert "erp" in unusable and "not yet the user's" in unusable["erp"]  # the demoted claim is an assumption, never an input
    assert read_modulation(ir) == ("fm", None)
