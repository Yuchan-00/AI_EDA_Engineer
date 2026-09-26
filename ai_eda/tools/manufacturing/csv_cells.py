"""CSV cell rules shared by the BOM/CPL compilers, the catalog reader and the reviewer.

Invariant: a CSV cell that spreadsheet software would execute is never
written as it stands. Two rules live here so that every writer and reader
applies the same one:

* :func:`unsafe_cell` says *why* a (stripped) cell must not be copied
  anywhere: it starts with a formula / command prefix (``=``, ``+``, ``-``,
  ``@``) or carries a control character. Identity, sourcing and reference
  cells are refused on that verdict (the catalog skips the row, the BOM
  compiler raises).
* :func:`free_text_cell` encodes a free-text cell (the BOM's ``Value`` /
  ``Description``: the design's own words, where ``-12V`` is legitimate) so
  it no longer starts with a formula character: the original text with a
  leading :data:`TEXT_PREFIX` (an apostrophe). The encoding is injective - a
  text whose first character is already the prefix is prefixed again - and
  :func:`bom_cell_text` is its single decoder (strips exactly one leading
  apostrophe). A control character anywhere in the text (unstripped) is
  refused with :class:`ValueError`, because a prefix only defeats formula
  interpretation - a line-based reader would see the text after a line
  break as a new row that may itself start with ``=``.

What counts as a control character is one rule for both functions,
:func:`control_character`, by Unicode general category
(:data:`REFUSED_CATEGORIES`): ``Cc`` (the C0 and C1 controls: NUL, tab, LF,
CR, DEL and NEL U+0085), ``Zl`` / ``Zp`` (the line and paragraph separators
U+2028 / U+2029, which ``str.splitlines`` and Unicode-aware readers treat as
a line break) and ``Cf`` (format characters: the byte-order mark U+FEFF,
zero-width spaces, bidi overrides). A format character is not a line break,
but it is invisible and lets a cell look like something it is not
(``\ufeff=1`` starts with ``=`` to a reader that skips the mark), so the
simplest honest rule refuses it too instead of guessing what each reader
strips. Ordinary letters, digits, punctuation, spaces and the printable
non-ASCII characters of any script (``Lu``, ``Nd``, ``Po``, ``Zs`` ...) pass.

What is asserted here is only what the code checks: the written cell does
not start with a formula character and carries no character of a refused
category. Whether a given spreadsheet or fab importer hides or shows the
apostrophe is documented behaviour of those programs, not measured by this
project.
"""

from __future__ import annotations

import unicodedata

#: characters a spreadsheet reads as the start of a formula / command
FORMULA_PREFIXES = "=+-@"
#: the leading character that marks a free-text cell as text (the OWASP CSV-injection mitigation)
TEXT_PREFIX = "'"
#: Unicode general categories refused anywhere in a cell (module docstring): C0/C1 controls, line / paragraph separators, format characters
REFUSED_CATEGORIES = frozenset({"Cc", "Zl", "Zp", "Cf"})


def control_character(value: str) -> str | None:
    """Why ``value`` is refused as carrying a control character (the first offending code point and its Unicode category), or ``None``."""
    for ch in value:
        category = unicodedata.category(ch)
        if category in REFUSED_CATEGORIES:
            return f"contains a control character U+{ord(ch):04X} (Unicode category {category})"
    return None


def unsafe_cell(value: str | None) -> str | None:
    """Why a (stripped) cell must not be copied anywhere: a formula / command prefix or a control character; ``None`` when it is plain text."""
    if not value:
        return None
    if value[0] in FORMULA_PREFIXES:
        return f"starts with {value[0]!r} (a spreadsheet would read it as a formula or command)"
    return control_character(value)


def free_text_cell(text: str, where: str) -> tuple[str, str | None]:
    """``(cell, note)`` for a free-text cell: the text as written and, when it was altered, a note saying what and why.

    A text that (stripped) would execute, or that already starts with
    :data:`TEXT_PREFIX`, is written with a leading apostrophe; ``where``
    (``R1.Value``) opens the note so a stage summary can list the cells.
    :class:`ValueError` for a control character anywhere in ``text``.
    """
    why = control_character(text)
    if why is not None:
        raise ValueError(f"BOM cell {where} = {text!r} {why}; refusing to write a cell spreadsheet software would execute")
    why = unsafe_cell(text.strip())
    if why is None and text[:1] == TEXT_PREFIX:
        why = f"starts with {TEXT_PREFIX!r} (the text marker itself, which the reader would otherwise strip)"
    if why is None:
        return text, None
    return TEXT_PREFIX + text, f"{where}: {why}; written with a leading apostrophe so the cell no longer starts with a formula character"


def bom_cell_text(cell: str) -> str:
    """The text a free-text BOM cell stands for: exactly one leading :data:`TEXT_PREFIX` removed (the single decoder of :func:`free_text_cell`)."""
    return cell[1:] if cell[:1] == TEXT_PREFIX else cell


__all__ = ["FORMULA_PREFIXES", "REFUSED_CATEGORIES", "TEXT_PREFIX", "bom_cell_text", "control_character", "free_text_cell", "unsafe_cell"]
