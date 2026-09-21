"""KiCad integration.

- :mod:`ai_eda.tools.kicad.sexpr` reads and writes KiCad s-expression files
  (the only module that knows the lexical rules and KiCad's layout).
- :mod:`ai_eda.tools.kicad.library` resolves symbol/footprint references
  against the installed KiCad libraries and returns parsed pin / pad data.
- :mod:`ai_eda.tools.kicad.cli` wraps ``kicad-cli`` (ERC, DRC, exports) and
  converts its JSON reports into ValidationResults.
"""

from ai_eda.tools.kicad import sexpr
from ai_eda.tools.kicad.cli import KicadCli, find_kicad_cli
from ai_eda.tools.kicad import geometry
from ai_eda.tools.kicad.library import (
    KICAD_PIN_TYPES,
    BBox,
    FootprintDef,
    KicadLibrary,
    LibraryFormatError,
    LibraryLookupError,
    Pad,
    SymbolDef,
    SymbolPin,
    ir_pin_type_to_kicad,
    kicad_pin_type_to_ir,
)

__all__ = [
    "sexpr",
    "geometry",
    "KicadCli",
    "find_kicad_cli",
    "KicadLibrary",
    "LibraryLookupError",
    "LibraryFormatError",
    "SymbolDef",
    "SymbolPin",
    "FootprintDef",
    "Pad",
    "BBox",
    "KICAD_PIN_TYPES",
    "kicad_pin_type_to_ir",
    "ir_pin_type_to_kicad",
]
