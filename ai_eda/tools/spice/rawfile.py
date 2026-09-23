"""ngspice rawfile reader (ASCII and binary).

A rawfile is *evidence*: the file ngspice itself wrote with ``write`` (shared
library) or ``-r`` (batch binary). This module reports only what the file
says and never fills gaps: a truncated body, a point count that does not
match, or a missing ``Binary:``/``Values:`` section is a ``ValueError``.

Layout facts this parser relies on (all measured against the rawfiles that
KiCad 10.0.6's ``ngspice.dll`` (ngspice-46) writes, see
``docs/ARCHITECTURE.md`` and the SPICE stage notes):

* Header lines, in order: ``Title:``, ``Date:``, ``Command:``
  (``ngspice-46, Build ...`` - the engine version; Debian's ngspice-42 writes
  no ``Command:`` line at all, so it is optional and ``RawPlot.command`` is
  then empty), ``Plotname:``
  (``Operating Point`` / ``DC transfer characteristic`` /
  ``Transient Analysis`` / ``AC Analysis``), ``Flags: real|complex``,
  ``No. Variables:``, ``No. Points:``, then ``Variables:`` followed by one
  ``\\t<idx>\\t<name>\\t<type>[ <extra>...]`` line per variable (extras such as
  ``grid=3`` are space separated after the type word), then ``Values:``
  (ASCII) or ``Binary:`` (binary). Unknown ``Key: value`` lines are kept in
  :attr:`RawPlot.header`.
* Variable names: the scale first (``time``, ``frequency``, ``v(v-sweep)``,
  ``i(i-sweep)``, ``res-sweep``, ``temp-sweep``), then the rest alphabetically;
  node voltages are ``v(<node>)`` and branch currents ``i(<device>)``.
* ASCII: written in text mode, so CRLF on Windows; per point
  `` <idx>\\t<v0>`` then ``\\t<vi>`` per remaining variable, ``%.15e``
  (15 significant digits, i.e. ~1e-16 relative to the engine's doubles),
  complex values as ``re,im``, a blank line between points.
* Binary: the header is LF-only, the marker is exactly ``Binary:\\n`` and the
  body is ``No. Points x No. Variables`` little-endian float64 values
  interleaved per point (real: 8 bytes per value; complex: ``re, im`` = 16
  bytes per value). Bit-identical to the vectors read through the API.
* A file may hold several plots back to back (``ngspice -b -r`` with a deck
  that has more than one analysis): :func:`parse_all` returns them in file
  order, :func:`parse` the first one.

Complex vectors (AC analysis, where even ``frequency`` is complex) are
expanded with one convention shared with :class:`ai_eda.tools.spice.SpiceResult`:
``<name>`` = magnitude, ``<name>.phase_deg`` = phase in degrees,
``<name>.real`` / ``<name>.imag`` = the components.
"""

from __future__ import annotations

import math
import re
import struct
from pathlib import Path

from pydantic import BaseModel, Field

#: the ``Plotname:`` ngspice writes for each analysis kind
PLOTNAMES: dict[str, str] = {
    "op": "Operating Point",
    "dc": "DC transfer characteristic",
    "ac": "AC Analysis",
    "tran": "Transient Analysis",
}

#: suffixes of the complex convention (``<name>.<suffix>``)
COMPLEX_SUFFIXES: tuple[str, ...] = ("phase_deg", "real", "imag")

_TOKEN_RE = re.compile(rb"\S+")
_WS_RE = re.compile(rb"\s*")
_RAW_NAME_RE = re.compile(r"([vi])\((.+)\)")


class RawPlot(BaseModel):
    """One plot of an ngspice rawfile, exactly as written (names are the rawfile's variable names)."""

    title: str
    date: str
    command: str
    plotname: str
    flags: str
    #: ``(name, type)`` per variable, in file order (the first one is the scale when the plot has one)
    variables: list[tuple[str, str]]
    #: rawfile variable name -> samples; complex plots use the module's complex convention
    vectors: dict[str, list[float]] = Field(default_factory=dict)
    #: name of the sweep variable (first variable) or None for an operating point
    scale: str | None
    n_points: int
    #: True when the body was ``Binary:`` (bit-exact), False for ``Values:`` (15 significant digits)
    binary: bool
    #: every ``Key: value`` header line
    header: dict[str, str] = Field(default_factory=dict)

    @property
    def is_complex(self) -> bool:
        return "complex" in self.flags

    @property
    def n_variables(self) -> int:
        return len(self.variables)

    def as_plot_vectors(self) -> dict[str, list[float]]:
        """The vectors re-keyed by ngspice *plot* vector names (``v(vout)`` -> ``vout``, ``i(v1)`` -> ``v1#branch``).

        This is the naming :class:`ai_eda.tools.spice.SpiceResult.vectors` uses, so a
        rawfile round-trips against the in-memory result of the run that wrote it.
        """
        out: dict[str, list[float]] = {}
        for key, data in self.vectors.items():
            base, suffix = _split_suffix(key)
            name = canonical_name(base)
            out[name if suffix is None else f"{name}.{suffix}"] = data
        return out


def _split_suffix(key: str) -> tuple[str, str | None]:
    for suffix in COMPLEX_SUFFIXES:
        if key.endswith("." + suffix):
            return key[: -len(suffix) - 1], suffix
    return key, None


def canonical_name(raw_name: str) -> str:
    """Rawfile variable name -> ngspice plot vector name.

    ``v(<node>)`` -> ``<node>``; ``v(v-sweep)`` -> ``v-sweep``; ``i(i-sweep)`` -> ``i-sweep``;
    ``i(<dev>)`` -> ``<dev>#branch``; ``time`` / ``frequency`` / ``res-sweep`` / ``temp-sweep``
    unchanged. (XSPICE branch names such as ``a1#branch_1_0`` are written as ``i(a1)`` by
    ngspice, so they come back as ``a1#branch`` - the rawfile does not keep the suffix.)
    """
    m = _RAW_NAME_RE.fullmatch(raw_name)
    if not m:
        return raw_name
    kind, inner = m.group(1), m.group(2)
    if kind == "v":
        return inner
    if inner == "i-sweep":
        return inner
    return f"{inner}#branch"


def raw_variable_name(plot_vector_name: str, v_type: int) -> str:
    """ngspice plot vector name (+ ``v_type`` from ``ngGet_Vec_Info``) -> the name ``write`` puts in the rawfile.

    Measured: ``time``/``frequency`` unchanged; anything containing ``#branch`` -> ``i(<prefix>)``
    (``v1#branch`` -> ``i(v1)``, ``a1#branch_1_0`` -> ``i(a1)``); voltage-type (3) vectors ->
    ``v(<name>)`` (also ``v-sweep`` -> ``v(v-sweep)``); current-type (4) -> ``i(<name>)``
    (``i-sweep`` -> ``i(i-sweep)``); ``res-sweep`` (15) / ``temp-sweep`` (14) unchanged.
    """
    if plot_vector_name in ("time", "frequency"):
        return plot_vector_name
    if "#branch" in plot_vector_name:
        return f"i({plot_vector_name.split('#branch', 1)[0]})"
    if v_type == 3:
        return f"v({plot_vector_name})"
    if v_type == 4:
        return f"i({plot_vector_name})"
    return plot_vector_name


def complex_convention(name: str, values: list[tuple[float, float]]) -> dict[str, list[float]]:
    """Expand ``(re, im)`` samples into ``name`` (magnitude), ``name.phase_deg``, ``name.real``, ``name.imag``."""
    return {
        name: [math.hypot(re_, im_) for re_, im_ in values],
        f"{name}.phase_deg": [math.degrees(math.atan2(im_, re_)) for re_, im_ in values],
        f"{name}.real": [re_ for re_, _ in values],
        f"{name}.imag": [im_ for _, im_ in values],
    }


def parse(path: str | Path) -> RawPlot:
    """The first plot of the rawfile at ``path`` (see :func:`parse_all`)."""
    return parse_all(path)[0]


def parse_all(path: str | Path) -> list[RawPlot]:
    """Every plot in the rawfile at ``path``, in file order. ``ValueError`` on any structural problem."""
    path = Path(path)
    data = path.read_bytes()
    plots: list[RawPlot] = []
    pos = 0
    while True:
        pos = _WS_RE.match(data, pos).end()
        if pos >= len(data):
            break
        plot, pos = _parse_one(data, pos, path)
        plots.append(plot)
    if not plots:
        raise ValueError(f"{path}: no rawfile plot found (empty file)")
    return plots


def _parse_one(data: bytes, pos: int, path: Path) -> tuple[RawPlot, int]:
    header: dict[str, str] = {}
    variables: list[tuple[str, str]] = []
    in_variables = False
    binary: bool | None = None
    while binary is None:
        nl = data.find(b"\n", pos)
        if nl < 0:
            raise ValueError(f"{path}: header ends without a 'Binary:' or 'Values:' section")
        line = data[pos:nl].rstrip(b"\r").decode("utf-8", "replace")
        pos = nl + 1
        if line == "Binary:":
            binary = True
        elif line == "Values:":
            binary = False
        elif line == "Variables:":
            in_variables = True
        elif in_variables and line.startswith("\t"):
            parts = line.lstrip("\t").split("\t")
            if len(parts) < 3:
                raise ValueError(f"{path}: malformed variable line {line!r}")
            variables.append((parts[1], parts[2].split(" ", 1)[0]))
        else:
            in_variables = False
            key, sep, value = line.partition(":")
            if not sep:
                raise ValueError(f"{path}: unexpected header line {line!r}")
            header[key.strip()] = value.strip()
    try:
        n_vars = int(header["No. Variables"])
        n_points = int(header["No. Points"])
    except (KeyError, ValueError) as e:
        raise ValueError(f"{path}: missing/invalid 'No. Variables' or 'No. Points' header") from e
    if len(variables) != n_vars:
        raise ValueError(f"{path}: 'No. Variables: {n_vars}' but {len(variables)} variable lines")
    flags = header.get("Flags", "")
    is_complex = "complex" in flags
    if binary:
        columns, pos = _read_binary(data, pos, n_points, n_vars, is_complex, path)
    else:
        columns, pos = _read_ascii(data, pos, n_points, n_vars, is_complex, path)
    vectors: dict[str, list[float]] = {}
    for (name, _typ), col in zip(variables, columns):
        if is_complex:
            vectors.update(complex_convention(name, col))
        else:
            vectors[name] = col
    plotname = header.get("Plotname", "")
    scale = None if (not variables or plotname == PLOTNAMES["op"]) else variables[0][0]
    plot = RawPlot(
        title=header.get("Title", ""),
        date=header.get("Date", ""),
        command=header.get("Command", ""),
        plotname=plotname,
        flags=flags,
        variables=variables,
        vectors=vectors,
        scale=scale,
        n_points=n_points,
        binary=binary,
        header=header,
    )
    return plot, pos


def _read_binary(data: bytes, pos: int, n_points: int, n_vars: int, is_complex: bool, path: Path) -> tuple[list, int]:
    per = 16 if is_complex else 8
    n_bytes = n_points * n_vars * per
    if len(data) - pos < n_bytes:
        raise ValueError(f"{path}: binary body has {len(data) - pos} bytes, expected {n_bytes} ({n_points} points x {n_vars} variables x {per})")
    n_doubles = n_points * n_vars * (2 if is_complex else 1)
    flat = struct.unpack_from(f"<{n_doubles}d", data, pos) if n_doubles else ()
    columns: list[list] = [[] for _ in range(n_vars)]
    k = 0
    for _ in range(n_points):
        for c in range(n_vars):
            if is_complex:
                columns[c].append((flat[k], flat[k + 1]))
                k += 2
            else:
                columns[c].append(flat[k])
                k += 1
    return columns, pos + n_bytes


def _read_ascii(data: bytes, pos: int, n_points: int, n_vars: int, is_complex: bool, path: Path) -> tuple[list, int]:
    columns: list[list] = [[] for _ in range(n_vars)]
    tokens = _TOKEN_RE.finditer(data, pos)
    end = pos
    for i in range(n_points):
        m = next(tokens, None)
        if m is None:
            raise ValueError(f"{path}: ASCII body ends after {i} of {n_points} points")
        try:
            idx = int(m.group(0))
        except ValueError as e:
            raise ValueError(f"{path}: expected point index {i}, found {m.group(0)!r}") from e
        if idx != i:
            raise ValueError(f"{path}: expected point index {i}, found {idx}")
        for c in range(n_vars):
            m = next(tokens, None)
            if m is None:
                raise ValueError(f"{path}: ASCII body ends inside point {i}")
            tok = m.group(0).decode("ascii", "replace")
            try:
                if is_complex:
                    re_s, sep, im_s = tok.partition(",")
                    if not sep:
                        raise ValueError(tok)
                    columns[c].append((float(re_s), float(im_s)))
                else:
                    columns[c].append(float(tok))
            except ValueError as e:
                raise ValueError(f"{path}: bad value {tok!r} at point {i}, variable {c}") from e
        end = m.end()
    return columns, end


__all__ = [
    "COMPLEX_SUFFIXES",
    "PLOTNAMES",
    "RawPlot",
    "canonical_name",
    "complex_convention",
    "parse",
    "parse_all",
    "raw_variable_name",
]
