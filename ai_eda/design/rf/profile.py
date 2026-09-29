"""The KR 447 MHz licence-free walkie-talkie regulatory profile: placeholders the user confirms, never grounded facts.

Invariant: every number of this profile (band edges, channel raster, power,
emission, deviation, occupied bandwidth, frequency tolerance, spurious
limits, antenna condition, time-out, conformity route) is general knowledge
that no official text in this project grounds - law.go.kr was not reachable
when the design was written, and no quote from 「신고하지 아니하고 개설할 수 있는
무선국용 무선기기」 or 「무선설비규칙」 is archived. So each value is a **free
choice** of the RF templates (``kr447.*`` in ``ir.parameters``, listed in
``ir.rf.profile_keys``) whose description ends with
``[UNVERIFIED: <document>; general knowledge, law.go.kr not reachable]``; it
enters the IR only through the ``confirm_design`` table (``assumption`` until
confirmed, then ``user_requirement``), is **never** stamped ``authoritative``
and can never make a check PASS: ``rf.regulatory_profile`` compares the
requirements against it and is NOT_VERIFIED (FAIL on an exceedance) at best.
Transmitting needs KC conformity assessment first, including a self-built
unit (``kr447.conformity``).

The keys end with the suffixes ``rf.regulatory_profile`` matches
(``.max_power`` -> ``tx_power`` at most, ``.max_deviation`` ->
``frequency_deviation`` at most, ``.max_obw`` -> ``occupied_bandwidth`` at
most, ``.freq_tolerance`` -> ``frequency_tolerance`` at most, ``.band_low`` /
``.band_high`` / ``.channel_raster`` -> ``carrier_frequency`` inside the band
and on the raster). :func:`channel_plan` / :func:`channel_index` do the raster
arithmetic a template needs to refuse a carrier that is not a channel
(:func:`raster_refusal`): a band such as "447 MHz 대역" extracted as
``carrier_frequency = 447 MHz`` is not on the raster, and the refusal asks for
the channel instead of rounding to one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ai_eda.ir import Traced

from ai_eda.design.base import Choice, choice_provenance

#: every profile key starts with this
PROFILE_PREFIX = "kr447."
#: the tail of every profile description after the document name
UNVERIFIED_TAIL = "general knowledge, law.go.kr not reachable"
NOTICE = "「신고하지 아니하고 개설할 수 있는 무선국용 무선기기」"
RULES = "「무선설비규칙」"
CONFORMITY = "전파법 제58조의2, 「방송통신기자재등의 적합성평가에 관한 고시」"


@dataclass(frozen=True)
class ProfileEntry:
    """One profile placeholder: its key, value (a number, or a string for a condition), unit and the document that would ground it."""

    key: str
    value: float | str
    unit: str | None
    what: str
    document: str

    def __post_init__(self) -> None:
        if not self.key.startswith(PROFILE_PREFIX):
            raise ValueError(f"profile key {self.key!r} must start with {PROFILE_PREFIX!r}")

    @property
    def description(self) -> str:
        return f"{self.what} - a placeholder, not a grounded limit [UNVERIFIED: {self.document}; {UNVERIFIED_TAIL}]"


#: the KR 447 MHz profile (kr447 design §2.0 "KR 447 regulatory profile")
PROFILE: tuple[ProfileEntry, ...] = (
    ProfileEntry("kr447.band_low", 447.5625e6, "Hz", "lowest channel of the licence-exempt 447 MHz walkie-talkie band", NOTICE),
    ProfileEntry("kr447.band_high", 447.8625e6, "Hz", "highest channel of the licence-exempt 447 MHz walkie-talkie band", NOTICE),
    ProfileEntry("kr447.channel_raster", 12.5e3, "Hz", "channel raster (25 channels from the lowest to the highest)", NOTICE),
    ProfileEntry("kr447.max_power", 0.5, "W", "maximum transmitter power, assumed at the antenna port (conducted); if the notice states ERP, the erp requirement governs", NOTICE),
    ProfileEntry("kr447.emission", "F3E", None, "emission class (analog FM telephony)", f"{NOTICE} / {RULES}"),
    ProfileEntry("kr447.max_deviation", 2.5e3, "Hz", "maximum peak frequency deviation", RULES),
    ProfileEntry("kr447.max_obw", 8.5e3, "Hz", "maximum occupied (99 %) bandwidth; which test modulation it is measured with decides compliance", RULES),
    ProfileEntry("kr447.freq_tolerance", 2.5, "ppm", "frequency tolerance", RULES),
    ProfileEntry("kr447.spurious_max", -36.0, "dBm", "maximum transmitter spurious emission (the stricter of two values seen; 2.5 uW = -26 dBm also appears)", RULES),
    ProfileEntry("kr447.rx_spurious_max", -57.0, "dBm", "maximum receiver spurious emission (the stricter placeholder)", RULES),
    ProfileEntry("kr447.antenna", "integral (non-detachable)", None, "antenna condition", NOTICE),
    ProfileEntry("kr447.tx_timeout", "not known", None, "transmit time-out condition (the time-out block is built only when tx_timeout is stated)", NOTICE),
    ProfileEntry("kr447.conformity", "KC 적합인증 before any transmission (tier per the conformity notice)", None,
                 "conformity assessment route, needed also for a self-built unit before it transmits", CONFORMITY),
)
PROFILE_BY_KEY: dict[str, ProfileEntry] = {e.key: e for e in PROFILE}


def profile_keys() -> list[str]:
    """Every profile key, in table order: what ``ir.rf.profile_keys`` lists."""
    return [e.key for e in PROFILE]


def profile_choices(template_id: str, confirmed: bool) -> list[tuple[Choice, Traced]]:
    """The confirmation-table row and the parameter of every profile entry (an ``assumption`` until confirmed, then ``user_requirement``; never grounded)."""
    out: list[tuple[Choice, Traced]] = []
    for e in PROFILE:
        value = float(e.value) if isinstance(e.value, (int, float)) else e.value
        unit = f" {e.unit}" if e.unit else ""
        prov = choice_provenance(template_id, f"{e.key} = {value!r}{unit}: {e.description}", confirmed)
        out.append((Choice(e.key, e.description, value, e.unit), Traced(value=value, unit=e.unit, provenance=prov)))
    return out


def _hz(x: float, what: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0:
        raise ValueError(f"{what} must be a positive finite frequency, got {x!r}")
    return float(x)


def channel_plan(band_low: float, band_high: float, raster: float) -> list[float]:
    """The channel centres ``band_low + k raster`` up to ``band_high`` (Hz); refuses a band that is not a whole number of rasters."""
    lo, hi, step = _hz(band_low, "band_low"), _hz(band_high, "band_high"), _hz(raster, "channel_raster")
    if hi < lo:
        raise ValueError(f"band_high {hi:.12g} Hz is below band_low {lo:.12g} Hz")
    n = (hi - lo) / step
    k = round(n)
    if abs(n - k) > 1e-6:
        raise ValueError(f"the band {lo:.12g}..{hi:.12g} Hz is not a whole number of {step:.12g} Hz rasters ({n:.12g})")
    return [lo + i * step for i in range(k + 1)]


def channel_index(f_hz: float, band_low: float, band_high: float, raster: float) -> int | None:
    """The channel number (0 = ``band_low``) of ``f_hz``, or ``None`` when it is not a channel centre (to within 1e-6 of a raster step)."""
    f = _hz(f_hz, "carrier frequency")
    plan = channel_plan(band_low, band_high, raster)
    k = round((f - plan[0]) / raster)
    if 0 <= k < len(plan) and abs(f - plan[k]) <= 1e-6 * raster:
        return k
    return None


def _mhz(f: float) -> str:
    return f"{f / 1e6:.4f}"


def raster_refusal(f_hz: float, band_low: float, band_high: float, raster: float) -> str | None:
    """``None`` when ``f_hz`` is a channel of the (unverified) raster, else the refusal sentence naming the channel list."""
    if channel_index(f_hz, band_low, band_high, raster) is not None:
        return None
    plan = channel_plan(band_low, band_high, raster)
    listed = ", ".join(_mhz(c) for c in plan)
    return (
        f"carrier_frequency {f_hz / 1e6:.6g} MHz is not a channel of the KR 447 MHz raster [UNVERIFIED: {NOTICE}; {UNVERIFIED_TAIL}]: "
        f"the channels are {listed} MHz ({len(plan)} channels, {raster / 1e3:.12g} kHz raster). State the channel "
        f"(a band such as '447 MHz 대역' is not a carrier_frequency); nothing is rounded to a channel"
    )


__all__ = [
    "CONFORMITY",
    "NOTICE",
    "PROFILE",
    "PROFILE_BY_KEY",
    "PROFILE_PREFIX",
    "RULES",
    "UNVERIFIED_TAIL",
    "ProfileEntry",
    "channel_index",
    "channel_plan",
    "profile_choices",
    "profile_keys",
    "raster_refusal",
]
