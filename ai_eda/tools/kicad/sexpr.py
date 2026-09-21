"""KiCad s-expression reader / writer.

Every ``.kicad_sym`` / ``.kicad_mod`` / ``.kicad_sch`` / ``.kicad_pcb`` file is
one s-expression. This module is the *only* place that knows the lexical
rules, so compilers and the library loader never touch raw text.

Invariants this module enforces
-------------------------------
* **Quoting is a property of the token, never inferred from content.**
  KiCad quotes string *fields* (names, numbers-as-strings such as pad "1",
  property values, layer names, uuids) and writes keywords / enums / numbers
  bare. A quoted token is a :class:`QStr`; a bare token is a plain ``str``.
  ``QStr("1")`` is written ``"1"``, ``"1"`` is written ``1`` and ``QStr("")``
  is written ``""`` - the writer never guesses.
* **Numbers are never reformatted on read.** Bare atoms keep their exact
  text (``"1.27"``, ``"20260206"``); convert on demand with :func:`to_float`.
  Values built in Python (``int`` / ``float`` / ``bool``) are formatted the way
  KiCad 10 writes them (:func:`fmt_num`: up to 6 decimals, trailing zeros
  stripped, never ``-0``, never an exponent; ``bool`` -> ``yes`` / ``no``).
* **Layout is KiCad's.** :func:`dumps` reproduces KiCad 10's ``Prettify``
  algorithm (one tab per depth, every nested list on its own line, ``(xy ..)``
  runs packed, ``)`` on its own line iff the last child was a list, long atom
  lists wrapped at column 72) so that ``dumps(parse(text))`` of a KiCad-saved
  file is byte-identical to it (with LF line endings).

Data model
----------
A *node* is a ``list`` whose items are atoms (``str`` / :class:`QStr`) or
nested nodes; by convention the first item is the head keyword. Nothing is
converted to numbers at parse time.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, Iterator

__all__ = [
    "QStr",
    "SExprError",
    "Node",
    "parse",
    "parse_all",
    "parse_file",
    "dumps",
    "dump_file",
    "fmt_num",
    "escape",
    "unescape",
    "S",
    "Q",
    "head",
    "find",
    "find_all",
    "get",
    "args",
    "to_float",
    "to_int",
    "strict_equal",
    "deep_copy",
]

Node = list
"""A parsed s-expression list: ``[head, *atoms_and_child_nodes]``."""


class QStr(str):
    """A quoted string token. Serialised as ``"..."`` with KiCad escapes.

    Behaves exactly like ``str`` for comparison / hashing (``QStr("R") == "R"``)
    so lookups stay convenient; only the writer treats it differently.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"QStr({str.__repr__(self)})"


class SExprError(ValueError):
    """Malformed s-expression text, or a node that cannot be serialised."""


# --------------------------------------------------------------------------- escapes

_UNESCAPES = {"n": "\n", "r": "\r", "t": "\t", '"': '"', "\\": "\\"}
_UNESCAPE_RE = re.compile(r"\\(.)", re.DOTALL)


def unescape(s: str) -> str:
    """Decode the escapes KiCad's reader understands (``\\"`` ``\\\\`` ``\\n`` ``\\t`` ``\\r``).

    An unknown escape is kept verbatim (backslash included).
    """
    if "\\" not in s:
        return s
    return _UNESCAPE_RE.sub(lambda m: _UNESCAPES.get(m.group(1), "\\" + m.group(1)), s)


def escape(s: str) -> str:
    """Encode a string body the way KiCad 10 writes it (tab stays literal)."""
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r")


# --------------------------------------------------------------------------- parser

# whitespace | ( | ) | "quoted" | bare | lone quote (error)
_TOKEN_RE = re.compile(r'[ \t\r\n]+|(\()|(\))|"((?:[^"\\]|\\.)*)"|([^ \t\r\n()"]+)|(")', re.DOTALL)


def parse_all(text: str) -> list[Any]:
    """Parse every top-level item of ``text`` (lists and/or atoms)."""
    if text.startswith("\ufeff"):
        text = text[1:]
    stack: list[list[Any]] = [[]]
    push = stack.append
    for m in _TOKEN_RE.finditer(text):
        if m.group(1):
            push([])
        elif m.group(2):
            if len(stack) == 1:
                raise SExprError(f"unexpected ')' at offset {m.start()}")
            node = stack.pop()
            stack[-1].append(node)
        elif m.group(3) is not None:
            stack[-1].append(QStr(unescape(m.group(3))))
        elif m.group(4):
            stack[-1].append(m.group(4))
        elif m.group(5):
            raise SExprError(f"unterminated string starting at offset {m.start()}")
        # else: whitespace
    if len(stack) != 1:
        raise SExprError(f"unbalanced parentheses: {len(stack) - 1} list(s) not closed")
    return stack[0]


def parse(text: str) -> Node:
    """Parse text that holds exactly one top-level list (every KiCad file does)."""
    items = parse_all(text)
    if len(items) != 1 or not isinstance(items[0], list):
        raise SExprError(f"expected exactly one top-level list, found {len(items)} item(s)")
    return items[0]


def parse_file(path: Path | str) -> Node:
    """Read a UTF-8 KiCad file and parse it. CRLF and LF are both accepted."""
    return parse(Path(path).read_bytes().decode("utf-8"))


# --------------------------------------------------------------------------- numbers


def fmt_num(x: float | int, decimals: int = 6) -> str:
    """Format a number the way KiCad writes it: ``0``, ``90``, ``1.27``, ``-0.825``.

    No trailing zeros, no ``.0`` on integral values, never ``-0``, never an
    exponent. ``decimals`` caps the precision (6 = PCB nanometre IU; symbol
    libraries only ever need 4).
    """
    if isinstance(x, bool):
        raise TypeError("bool is not a number here; it serialises as yes/no")
    if isinstance(x, int):
        return str(x)
    s = f"{x:.{decimals}f}".rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def to_float(atom: Any) -> float:
    """Numeric value of an atom (accepts the text forms KiCad's reader accepts)."""
    try:
        return float(atom)
    except (TypeError, ValueError) as exc:
        raise SExprError(f"not a number: {atom!r}") from exc


def to_int(atom: Any) -> int:
    v = to_float(atom)
    if v != int(v):
        raise SExprError(f"not an integer: {atom!r}")
    return int(v)


# --------------------------------------------------------------------------- builder


def Q(text: object) -> QStr:
    """Build a quoted-string atom (``Q("1")`` -> ``"1"``, ``Q("")`` -> ``""``)."""
    return QStr(str(text))


def S(head: str, *items: Any) -> Node:
    """Build a node: ``S("at", 0, 3.81, 270)`` -> ``(at 0 3.81 270)``.

    Items may be bare ``str``, :class:`QStr`, ``int``, ``float``, ``bool``
    (``yes``/``no``) or nested nodes. ``None`` items are skipped so optional
    children can be written inline: ``S("pad", ..., S("drill", d) if d else None)``.
    """
    return [head, *(it for it in items if it is not None)]


# --------------------------------------------------------------------------- serializer

_BARE_FORBIDDEN = re.compile(r'[ \t\r\n()"]')


def _atom_text(item: Any) -> str:
    if isinstance(item, QStr):
        return '"' + escape(item) + '"'
    if isinstance(item, bool):
        return "yes" if item else "no"
    if isinstance(item, str):
        if not item or _BARE_FORBIDDEN.search(item):
            raise SExprError(f"bare atom {item!r} is not writable; wrap it with Q() to quote it")
        return item
    if isinstance(item, int):
        return str(item)
    if isinstance(item, float):
        return fmt_num(item)
    raise SExprError(f"cannot serialise {type(item).__name__}: {item!r}")


class _Writer:
    """Port of KiCad's KICAD_FORMAT::Prettify (compact save off), driven by the tree."""

    __slots__ = ("out", "col", "last_open_xy", "multiline", "started")

    _WRAP_COL = 72
    _XY_COL = 99

    def __init__(self) -> None:
        self.out: list[str] = []
        self.col = 0  # byte column, like KiCad's std::string arithmetic
        self.last_open_xy = False
        self.multiline = False
        self.started = False

    def write(self, node: list, depth: int) -> None:
        out = self.out
        is_xy = len(node) > 1 and node[0] == "xy"
        if not self.started:
            out.append("(")
            self.col = 1
            self.started = True
        elif self.last_open_xy and is_xy and self.col < self._XY_COL:
            out.append(" (")
            self.col += 2
        else:
            out.append("\n" + "\t" * depth + "(")
            self.col = depth + 1
        self.last_open_xy = is_xy
        inner = depth + 1
        first = True
        last_was_list = False
        for item in node:
            if isinstance(item, list):
                self.write(item, inner)
                last_was_list = True
            else:
                text = _atom_text(item)
                if not first:
                    if self.last_open_xy or self.col < self._WRAP_COL:
                        out.append(" ")
                        self.col += 1
                    else:
                        out.append("\n" + "\t" * inner)
                        self.col = inner
                        self.multiline = True
                out.append(text)
                self.col += len(text) if text.isascii() else len(text.encode("utf-8"))
                last_was_list = False
            first = False
        if last_was_list or self.multiline:
            out.append("\n" + "\t" * depth + ")")
            self.col = depth + 1
            self.multiline = False
        else:
            out.append(")")
            self.col += 1


def dumps(node: Node) -> str:
    """Serialise a node in KiCad 10 layout. Ends with exactly one ``\\n`` (LF)."""
    if not isinstance(node, list):
        raise SExprError(f"top-level value must be a list, got {type(node).__name__}")
    w = _Writer()
    w.write(node, 0)
    w.out.append("\n")
    return "".join(w.out)


def dump_file(node: Node, path: Path | str) -> Path:
    """Write ``node`` as UTF-8 with LF line endings (deterministic across platforms)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(dumps(node))
    return p


# --------------------------------------------------------------------------- tree helpers


def head(node: Any) -> str | None:
    """Head keyword of a node, or None for atoms / empty lists."""
    if isinstance(node, list) and node and isinstance(node[0], str) and not isinstance(node[0], QStr):
        return node[0]
    return None


def _children(node: list, name: str) -> Iterator[list]:
    for item in node:
        if isinstance(item, list) and item and item[0] == name and not isinstance(item[0], QStr):
            yield item


def find(node: list, name: str) -> Node | None:
    """First child node whose head is ``name``, else None."""
    return next(_children(node, name), None)


def find_all(node: list, name: str) -> list[Node]:
    """All child nodes whose head is ``name`` (document order)."""
    return list(_children(node, name))


def get(node: list, name: str, index: int = 1, default: Any = None) -> Any:
    """``node`` -> first child ``(name ...)`` -> its item at ``index`` (default 1).

    ``get(pin, "length")`` -> ``"1.27"``; ``get(sym, "power", 1, None)`` -> ``"global"``.
    """
    child = find(node, name)
    if child is None or index >= len(child):
        return default
    return child[index]


def args(node: list) -> list[Any]:
    """The atoms (non-list items) after the head: ``args((layers "F.Cu" "F.Mask"))``."""
    return [it for it in node[1:] if not isinstance(it, list)]


def strict_equal(a: Any, b: Any) -> bool:
    """Structural equality that also requires identical quoting (``QStr`` vs bare).

    ``==`` on nodes treats ``QStr("1")`` and ``"1"`` as equal; round-trip tests
    must not.
    """
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(strict_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, list) or isinstance(b, list):
        return False
    return type(a) is type(b) and a == b


def deep_copy(node: Node) -> Node:
    """Deep copy of a node (``QStr`` tokens stay ``QStr``)."""
    return copy.deepcopy(node)
