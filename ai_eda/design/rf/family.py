"""The KR 447 MHz walkie-talkie family: which board each ``radio_build`` builds, what it needs and serves, and the selector question.

Invariant: which board is built is the *product*, so it is a categorical
**requirement** the user states (``radio_build``, given once and kept as
``user_requirement``), never a control answer and never inferred. When a
radio key is confirmed (:data:`SELECTOR_KEYS`) but ``radio_build`` is not,
the selector asks the required question :func:`selector_question` - the
table of :data:`BUILDS`, recommending ``rx_backend`` (the first RF stage)
first - instead of picking a board. The world of each build is closed: a
confirmed design requirement a build does not serve refuses it through the
template's ``refusals`` rule, and :func:`unserved_message` names the builds
that do serve it (``tx_power is served by radio_build=tx_exciter,
transceiver or transceiver_conducted``), so the reviewer never sees an
unserved requirement disguised as served.

The serve / need table is kr447 design §2.0 ("Requirement keys": ✓ served,
○ optional read, R refused); every build also serves ``radio_build`` itself
(added by :class:`BuildInfo`: the requirement that selected a board is never
unserved by it), ``pcb_layers`` is a board key every template serves through
its stack (the layer policy is :attr:`BuildInfo.layers`), and
``application`` / ``jurisdiction`` are ignored by every template. A
categorical need (``modulation``) counts as present only when its reader
reads it (:func:`ai_eda.design.inputs.present_keys`), and the closed-world
refusals run only once no need is missing. The two
stage-5 builds are one template with a variant: ``transceiver`` has the
integral antenna and no connector, ``transceiver_conducted`` a U.FL in place
of the antenna for bench and KC conducted samples (never fitted with an
antenna); they serve the same keys, because the conducted sample is the same
product measured at its antenna port.

:data:`SELECTOR_KEY` is the requirement key ``radio_build``
(``ai_eda.design.inputs.RADIO_BUILD_KEY``); the six build names equal
``ai_eda.design.inputs.RADIO_BUILDS`` in the same order (checked when this
module is imported).
"""

from __future__ import annotations

from dataclasses import dataclass

from ai_eda.ir import CircuitIR, MissingInformation

from ai_eda.design.inputs import RADIO_BUILD_KEY, RADIO_BUILDS, canonical_key

#: the categorical requirement key that selects the board (``ai_eda.design.inputs.RADIO_BUILD_KEY``)
SELECTOR_KEY = RADIO_BUILD_KEY
#: confirmed keys whose presence without ``radio_build`` makes the selector ask
SELECTOR_KEYS: tuple[str, ...] = ("carrier_frequency", "modulation")
#: the build the selector question recommends first (user decision 3A: D -> C -> B, the receive back-end first)
RECOMMENDED_FIRST = "rx_backend"


@dataclass(frozen=True)
class BuildInfo:
    """One ``radio_build``: its template, board, stage, the keys it needs / serves (``optional`` are read only when stated) and its layer policy."""

    build: str
    template_id: str
    title: str
    stage: int
    board: str
    needs: tuple[str, ...]
    serves: tuple[str, ...]
    optional: tuple[str, ...] = ()
    #: (allowed layer counts, default)
    layers: tuple[tuple[int, ...], int] = ((4,), 4)
    #: the build this one is a variant of (``transceiver_conducted`` -> ``transceiver``)
    variant_of: str | None = None
    #: the modulation the board implements (``modulation`` must name it when stated)
    modulation: str = "fm"

    def __post_init__(self) -> None:
        if SELECTOR_KEY not in self.serves:  # every build serves the requirement that selected it: its own radio_build is never unserved
            object.__setattr__(self, "serves", (SELECTOR_KEY, *self.serves))
        missing = [k for k in (*self.needs, *self.optional) if k not in self.serves]
        if missing:
            raise ValueError(f"build {self.build!r}: needed / optional keys {missing} must be served")


_TRANSCEIVER_SERVES = (
    "carrier_frequency", "modulation", "frequency_deviation", "audio_bandwidth", "channel_spacing", "occupied_bandwidth", "frequency_tolerance",
    "tx_power", "erp", "eirp", "system_impedance", "antenna_impedance", "antenna_gain", "link_range", "field_strength_limit", "rx_sensitivity",
    "input_voltage", "tx_timeout",
)
_TRANSCEIVER_OPTIONAL = ("erp", "eirp", "system_impedance", "antenna_impedance", "antenna_gain", "link_range", "field_strength_limit", "rx_sensitivity", "tx_timeout")

#: the six builds, in stage order (kr447 design §2.0 table)
BUILDS: dict[str, BuildInfo] = {b.build: b for b in (
    BuildInfo("audio_ptt", "kr447_audio_ptt", "KR447 audio / PTT bench board", 1,
              "power + PTT sequencer / TX interlock + TX audio processor + RX audio amplifier; no RF",
              needs=("input_voltage", "modulation"), serves=("modulation", "frequency_deviation", "audio_bandwidth", "input_voltage", "tx_timeout"),
              optional=("tx_timeout",), layers=((2, 4), 4)),
    BuildInfo("rx_backend", "kr447_rx_backend", "KR447 receiver IF back-end bench board", 2,
              "IF1 21.4 MHz U.FL input -> crystal ladder -> SA605 -> squelch -> audio",
              needs=("modulation", "input_voltage"),
              serves=("modulation", "frequency_deviation", "audio_bandwidth", "channel_spacing", "frequency_tolerance", "system_impedance", "input_voltage"),
              optional=("frequency_tolerance", "system_impedance")),
    BuildInfo("rx_frontend", "kr447_rx_frontend", "KR447 receiver front-end bench board", 3,
              "RF U.FL in -> BPF / LNA / BPF -> ADEX-10 -> diplexer -> post-amp -> IF1 U.FL out; LO1 chain",
              needs=("carrier_frequency", "input_voltage"),
              serves=("carrier_frequency", "modulation", "frequency_tolerance", "system_impedance", "rx_sensitivity", "input_voltage"),
              optional=("modulation", "system_impedance", "rx_sensitivity")),
    BuildInfo("tx_exciter", "kr447_tx_exciter", "KR447 transmitter exciter bench board (conducted into 50 ohm only)", 4,
              "TCXO 37.296875 MHz -> 2 buffered PM tanks -> x12 (3 x 2 x 2) -> BPF -> driver -> PA -> LPF -> U.FL, with PTT and TX audio",
              needs=("carrier_frequency", "modulation", "input_voltage"),
              serves=("carrier_frequency", "modulation", "frequency_deviation", "audio_bandwidth", "occupied_bandwidth", "frequency_tolerance",
                      "tx_power", "system_impedance", "input_voltage", "tx_timeout"),
              optional=("system_impedance", "tx_timeout")),
    BuildInfo("transceiver", "kr447_transceiver", "KR447 FM transceiver (integral antenna)", 5,
              "the composition of every block, integral straight lambda/4 antenna, no connector",
              needs=("carrier_frequency", "modulation", "input_voltage"), serves=_TRANSCEIVER_SERVES, optional=_TRANSCEIVER_OPTIONAL),
    BuildInfo("transceiver_conducted", "kr447_transceiver", "KR447 FM transceiver, conducted variant (U.FL instead of the antenna)", 5,
              "the same composition with a U.FL in place of the antenna, for bench and KC conducted samples; never fitted with an antenna",
              needs=("carrier_frequency", "modulation", "input_voltage"), serves=_TRANSCEIVER_SERVES, optional=_TRANSCEIVER_OPTIONAL,
              variant_of="transceiver"),
)}

if tuple(BUILDS) != RADIO_BUILDS:  # the requirement key's values and the family's builds are one list
    raise ValueError(f"the family's builds {tuple(BUILDS)} are not ai_eda.design.inputs.RADIO_BUILDS {RADIO_BUILDS}")


def serving_builds(key: str) -> tuple[str, ...]:
    """The builds that serve the canonical requirement ``key``, in stage order (empty when none does)."""
    return tuple(b.build for b in BUILDS.values() if key in b.serves)


def _or_list(items: tuple[str, ...]) -> str:
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} or {items[-1]}"


def unserved_message(key: str, build: str | None = None) -> str:
    """The closed-world sentence for a confirmed requirement ``key`` the build (if given) does not serve, naming the builds that do."""
    here = f"{key} is not served by radio_build={build}; " if build is not None else ""
    builds = serving_builds(key)
    if not builds:
        return f"{here}{key} is served by no radio_build of the KR 447 MHz family (the licence-exempt 447 MHz class is FM telephony [UNVERIFIED])"
    return f"{here}{key} is served by radio_build={_or_list(builds)}"


def selector_question(ir: CircuitIR) -> MissingInformation:
    """The required question ``radio_build``: the family table, ``rx_backend`` recommended first; the rationale names the confirmed radio keys seen."""
    seen = sorted({canonical_key(r.key) or r.key for r in ir.requirements.requirements
                   if (canonical_key(r.key) or r.key) in SELECTOR_KEYS and r.value is not None and r.value.provenance.is_authoritative})
    rows = "; ".join(
        f"{b.build} (stage {b.stage}{', recommended first' if b.build == RECOMMENDED_FIRST else ''}: {b.board}; needs {', '.join(b.needs)})"
        for b in BUILDS.values()
    )
    question = (
        f"Which board of the KR 447 MHz FM walkie-talkie family should be built? Answer {SELECTOR_KEY} with one of: {rows}. "
        f"Each stage is its own bench-board project; {RECOMMENDED_FIRST} is recommended first. Every KR regulatory number of these boards is an "
        f"UNVERIFIED placeholder, and KC conformity assessment is needed before any transmission."
    )
    why = (f"{', '.join(seen)} {'is' if len(seen) == 1 else 'are'} confirmed but {SELECTOR_KEY} is not" if seen
           else f"{SELECTOR_KEY} is not stated")
    return MissingInformation(key=SELECTOR_KEY, question=question, required=True, options=list(BUILDS), rationale=f"{why}: which board is built is the product, never guessed")


__all__ = [
    "BUILDS",
    "RECOMMENDED_FIRST",
    "SELECTOR_KEY",
    "SELECTOR_KEYS",
    "BuildInfo",
    "selector_question",
    "serving_builds",
    "unserved_message",
]
