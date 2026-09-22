"""Measure how ngspice.dll reads number spellings - the harness behind ``ai_eda.tools.calc.si.ngspice_reads``.

One deck with a ``V<i> n<i> 0 DC <spelling>`` source and a 1 k load per
spelling, one ``op``, then every node voltage read through
``ngGet_Vec_Info``: an ideal source into a resistor makes the node voltage
the parsed source value bit for bit (the MNA row for the source is
``v = E``), so the doubles below are exactly what ``INPevaluate`` produced.
The 20 values pinned in ``tests/test_si.py`` were measured this way on
2026-09-21; :func:`measure_spellings` re-measures any set on the DLL that is
installed now, and :func:`sample_spellings` draws a deterministic sample of
the spelling families the 13,975-spelling lab run covered (integer and
decimal mantissas, every scale suffix in both cases, ``meg`` / ``mil``,
trailing unit letters, exponent forms, signs).
"""

from __future__ import annotations

import random

from ai_eda.tools.spice import NgspiceShared

#: scale suffixes as ngspice spells them, plus the unit-letter tails it ignores
SUFFIXES: tuple[str, ...] = ("", "t", "g", "meg", "k", "m", "u", "n", "p", "f", "mil", "T", "G", "MEG", "Meg", "K", "M", "U", "N", "P", "F", "MIL")
TAILS: tuple[str, ...] = ("", "Ohm", "F", "H", "V", "A", "s", "Hz")
#: sources per deck: ngspice handles a few hundred independent nodes in one op without trouble
CHUNK = 250


def sample_spellings(n: int, seed: int = 20260921) -> list[str]:
    """``n`` distinct spellings drawn deterministically from the measured families."""
    rng = random.Random(seed)
    out: list[str] = []
    seen: set[str] = set()
    while len(out) < n:
        family = rng.randrange(6)
        if family == 0:  # short integer mantissa + suffix (10k, 22p, 1meg)
            text = f"{rng.randint(1, 999)}{rng.choice(SUFFIXES)}"
        elif family == 1:  # datasheet decimals (4.7u, 0.047, 3.3, 22.0p)
            digits = rng.randint(1, 5)
            mantissa = rng.randint(1, 10**digits - 1) / 10 ** rng.randint(0, digits)
            text = f"{mantissa:g}{rng.choice(SUFFIXES)}"
        elif family == 2:  # long mantissas (up to 15 significant digits: inside the modelled region)
            digits = rng.randint(6, 15)
            mantissa = rng.randint(10 ** (digits - 1), 10**digits - 1)
            point = rng.randint(0, digits)
            s = str(mantissa)
            text = (s[:point] + "." + s[point:]) if 0 < point < digits else s
            text = text.lstrip("0") or "0"
            if text.startswith("."):
                text = "0" + text
            text += rng.choice(SUFFIXES)
        elif family == 3:  # exponent forms
            text = f"{rng.randint(1, 99) / 10 ** rng.randint(0, 2):g}{rng.choice('eE')}{rng.choice(['', '-', '+'])}{rng.randint(0, 12)}"
        elif family == 4:  # trailing unit letters after the suffix (10kOhm, 4.7uF)
            text = f"{rng.randint(1, 999) / 10 ** rng.randint(0, 2):g}{rng.choice(SUFFIXES)}{rng.choice(TAILS)}"
        else:  # negative values
            text = f"-{rng.randint(1, 999)}{rng.choice(SUFFIXES)}"
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def measure_spellings(runner: NgspiceShared, spellings: list[str]) -> dict[str, float]:
    """``{spelling: the double ngspice.dll parsed}`` for every spelling, measured now."""
    eng = runner._engine()
    out: dict[str, float] = {}
    with eng.lock:
        for start in range(0, len(spellings), CHUNK):
            chunk = spellings[start : start + CHUNK]
            lines = ["parser measurement"]
            for i, text in enumerate(chunk):
                lines.append(f"V{i} n{i} 0 DC {text}")
                lines.append(f"R{i} n{i} 0 1k")
            lines.append(".end")
            eng.unload_all()
            eng.cap.clear()
            rc = eng._circ(lines)
            stderr = eng.cap.stderr()
            if rc != 0 or stderr:
                raise RuntimeError(f"ngspice refused the measurement deck (rc {rc}): {stderr}")
            eng.cap.clear()
            eng._cmd("op")
            stderr = eng.cap.stderr()
            if stderr:
                raise RuntimeError(f"ngspice reported errors during the measurement op: {stderr}")
            plot = eng.cur_plot()
            for i, text in enumerate(chunk):
                v = eng.read_vector(f"{plot}.n{i}")
                if v is None or v.length != 1:
                    raise RuntimeError(f"no operating point for spelling {text!r} (node n{i})")
                out[text] = float(v.data[0])
            eng.unload_all()
    return out


__all__ = ["CHUNK", "SUFFIXES", "TAILS", "measure_spellings", "sample_spellings"]
