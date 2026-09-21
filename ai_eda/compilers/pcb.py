"""IR -> ``.kicad_pcb`` compiler (KiCad 10, file version 20260206).

Invariants enforced here
------------------------
* **Nothing is guessed.** Every footprint is loaded from a KiCad library on
  disk through :class:`~ai_eda.tools.kicad.library.KicadLibrary` and embedded
  *verbatim* (geometry, ``attr``, graphics, 3D model); pad geometry never comes
  from memory. A footprint that does not resolve, a component without a
  placement, a net pin without a pad, a pad without an IR pin, an unknown
  layer or net, or a missing outline is a :class:`~ai_eda.errors.CompileError`.
* **Deterministic.** Every UUID is a uuid5 from :mod:`ai_eda.compilers.ids`;
  nets are numbered ``0 ""`` then IR nets sorted by name (:func:`net_numbers`);
  footprints follow IR order; the s-expression is written by
  :func:`ai_eda.tools.kicad.sexpr.dumps` with LF line endings. The same IR
  gives byte-identical output (tested).
* **Derived artifact.** The result is registered through
  :meth:`Compiler._write`, stamped with ``ir.content_hash()``. Validity of the
  board is *not* asserted here; only real ``kicad-cli`` DRC decides that.

What the file contains (verified with kicad-cli 10.0.6 ``pcb drc
--schematic-parity``, gerber and drill export)
-----------------------------------------------------------------
* ``(kicad_pcb (version 20260206) (generator "pcbnew") (generator_version "10.0")``
  - the numbers KiCad 10.0.6 itself writes. In this format there is **no**
  numeric ``(net N "NAME")`` table: pads, segments, vias and zones reference
  nets by name, ``(net "VIN")``. :func:`net_numbers` still provides the
  deterministic numbering (``0 ""`` then sorted IR nets from 1) for exporters
  that need net codes (e.g. a KiCad-9 ``20241229`` writer or netlists).
* Layers: copper layers from ``ir.pcb.layers`` (``F.Cu`` = 0, ``In<n>.Cu`` =
  ``2n + 2``, ``B.Cu`` = 2) followed by KiCad's standard non-copper set in the
  exact order KiCad writes it for an enabled 2-layer board.
* ``(setup)``: KiCad's defaults exactly as ``kicad-cli pcb upgrade --force``
  adds them (mask clearance 0, tented vias, default plot params). Design rules
  (minimum track width / clearance / via sizes) do **not** live in a
  ``.kicad_pcb`` - KiCad keeps them in the ``.kicad_pro`` project file - so
  without one kicad-cli applies its built-in defaults (clearance 0.2 mm, track
  0.2 mm, copper to edge 0.5 mm). :func:`design_rules` maps the authoritative
  ``ir.pcb.manufacturing`` values onto the ``.kicad_pro``
  ``board.design_settings.rules`` keys for a future project-file writer; the
  only fab value that *is* representable in the board file, the board
  thickness, is written to ``(general (thickness ..))``.
* Footprints (sorted by layer then uuid, as KiCad writes them): the library
  ``(footprint ...)`` tree with the modifications KiCad makes when embedding -
  library ``(version)``/``(generator)`` dropped, ``(layer)`` + ``(uuid)`` +
  ``(at x y [rot])`` inserted (rotation in ``(-180, 180]`` like KiCad),
  ``Reference`` / ``Value`` / ``Datasheet`` / ``Description`` properties from
  the IR component (``Description`` falls back to the library symbol's, the
  same rule the schematic compiler uses; KiCad 10 keeps no ``Footprint``
  property on a board footprint - the id is the header string),
  ``(path "/<schematic symbol uuid>")`` + ``(sheetname "/")`` +
  ``(sheetfile "<project>.kicad_sch")`` so ``--schematic-parity`` links the
  footprint to the schematic symbol written by the schematic compiler, a
  deterministic ``(uuid)`` on every graphic item, and on pads ``(net "NAME")``
  + ``(pintype ..)`` taken from the *verified library symbol's* pin
  (:mod:`ai_eda.compilers.pins`, the same check the schematic compiler runs,
  so both artifacts carry library pin types; an IR pin type that differs
  from the library is a CompileError, an IR ``no_connect`` becomes
  ``"<type>+no_connect"`` as KiCad writes it). Bottom-side placement follows
  KiCad's flip rule (see :mod:`ai_eda.tools.kicad.geometry`); for pads every
  child is handled by an explicit rule verified against kicad-cli 10.0.6
  ``lib_footprint_mismatch`` (coordinates and ``rect_delta`` mirrored in y,
  layers swapped, ``chamfer`` corners ``top_* <-> bottom_*``, scalars kept)
  and a pad child without a rule is refused. Library graphics keep library
  order (KiCad re-sorts them by layer/type/geometry on save; no semantic
  difference).
* Outline: one ``gr_rect`` on ``Edge.Cuts`` from ``ir.pcb.outline``.
* Tracks / vias / zones straight from ``ir.pcb`` (zones are emitted with a
  solid pad connection because thermal reliefs on 2.54 mm headers starve).
  Zones are written *unfilled* (no ``filled_polygon``): ``KicadCli.run_drc``
  passes ``--refill-zones`` and ``KicadCli.export_gerbers`` ``--check-zones``,
  so DRC judges and the fab files contain the filled pour, and the gerber
  check verifies the pour's copper is really in the plot.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ai_eda.compilers import ids
from ai_eda.compilers.base import CompileContext, Compiler
from ai_eda.compilers.pins import load_verified_symbol, pad_pin_types
from ai_eda.errors import CompileError, NothingToCompileError
from ai_eda.ir import ArtifactKind, ArtifactRef, BoardSide, CircuitIR, Component, Placement, Track, Via, Zone
from ai_eda.tools.kicad import sexpr
from ai_eda.tools.kicad.geometry import footprint_angle, mirrored_layer, normalize_angle, text_angle
from ai_eda.tools.kicad.library import FootprintDef, KicadLibrary
from ai_eda.tools.kicad.sexpr import Q, S

__all__ = [
    "PCBCompiler",
    "FILE_VERSION",
    "GENERATOR",
    "GENERATOR_VERSION",
    "NON_COPPER_LAYERS",
    "MANUFACTURING_RULE_KEYS",
    "net_numbers",
    "design_rules",
    "copper_layer_index",
]

FILE_VERSION = 20260206
GENERATOR = "pcbnew"
GENERATOR_VERSION = "10.0"

#: Non-copper layers of a default board, in the exact order and with the exact
#: indices / user names KiCad 10 writes them (only enabled layers are listed).
NON_COPPER_LAYERS: tuple[tuple[int, str, str | None], ...] = (
    (9, "F.Adhes", "F.Adhesive"),
    (11, "B.Adhes", "B.Adhesive"),
    (13, "F.Paste", None),
    (15, "B.Paste", None),
    (5, "F.SilkS", "F.Silkscreen"),
    (7, "B.SilkS", "B.Silkscreen"),
    (1, "F.Mask", None),
    (3, "B.Mask", None),
    (17, "Dwgs.User", "User.Drawings"),
    (19, "Cmts.User", "User.Comments"),
    (21, "Eco1.User", "User.Eco1"),
    (23, "Eco2.User", "User.Eco2"),
    (25, "Edge.Cuts", None),
    (27, "Margin", None),
    (31, "F.CrtYd", "F.Courtyard"),
    (29, "B.CrtYd", "B.Courtyard"),
    (35, "F.Fab", None),
    (33, "B.Fab", None),
)

_COPPER_KINDS = frozenset({"signal", "power", "mixed", "jumper"})

#: ``ManufacturingConstraints`` field -> ``.kicad_pro`` ``board.design_settings.rules`` key.
#: Not mappable: ``min_hole_to_edge_mm`` (KiCad only has copper-to-edge), ``copper_weight_oz``
#: (needs a full ``(stackup)``), ``layer_count_options``, ``fab``. ``board_thickness_mm`` goes
#: to ``(general (thickness ..))`` in the board file itself.
MANUFACTURING_RULE_KEYS: dict[str, str] = {
    "min_track_width_mm": "min_track_width",
    "min_clearance_mm": "min_clearance",
    "min_via_diameter_mm": "min_via_diameter",
    "min_via_drill_mm": "min_through_hole_diameter",
}

DEFAULT_BOARD_THICKNESS_MM = 1.6
EDGE_LINE_WIDTH_MM = 0.05
_BAD_STEM_RE = re.compile(r'[\\/:*?"<>|\s]')

# footprint children that are rebuilt rather than copied from the library tree
_FP_HEADER_HEADS = frozenset({"version", "generator", "generator_version", "layer", "uuid", "at", "tstamp", "tedit"})
_FP_MANAGED_HEADS = frozenset(
    {"descr", "tags", "property", "path", "sheetname", "sheetfile", "attr", "pad", "model", "embedded_fonts", "units"}
)
_FP_GRAPHIC_HEADS = frozenset(
    {"fp_line", "fp_rect", "fp_circle", "fp_arc", "fp_poly", "fp_curve", "fp_text", "fp_text_box", "dimension", "zone", "group", "image"}
)
#: graphics whose coordinates this compiler knows how to mirror for the bottom side
_MIRRORABLE_HEADS = frozenset({"fp_line", "fp_rect", "fp_circle", "fp_arc", "fp_poly", "fp_curve", "fp_text", "fp_text_box"})
_POINT_HEADS = frozenset({"start", "end", "center", "mid", "xy", "offset", "delta", "rect_delta"})
#: pad children KiCad writes *before* ``(net ..)``
_PAD_BEFORE_NET = frozenset(
    {"at", "size", "delta", "rect_delta", "drill", "property", "layers", "remove_unused_layers", "keep_end_layers",
     "zone_layer_connections", "roundrect_rratio", "chamfer_ratio", "chamfer"}
)
#: How ``PAD::Flip(TOP_BOTTOM)`` treats each pad child (evidence: kicad-cli 10.0.6 DRC ``lib_footprint_mismatch``
#: on Inductor_SMD:L_Bourns_SDR0604, Package_DFN_QFN:Analog_QFN-28-36-2EP, Package_LGA:AMS_LGA-10-1EP,
#: Battery:BatteryHolder_Keystone_1057, Button_Switch_SMD:SW_Push_1TS009 placed on B.Cu). A child not listed
#: here is refused on the bottom side rather than copied through unchanged.
_PAD_FLIP_MIRROR_Y = frozenset({"at", "drill", "primitives", "rect_delta"})  # y coordinates negated (drill: its offset)
_PAD_FLIP_LAYERS = frozenset({"layers"})  # F.* <-> B.*
_PAD_FLIP_CHAMFER = frozenset({"chamfer"})  # top_* <-> bottom_*
_PAD_FLIP_INVARIANT = frozenset(
    {
        "size", "property", "remove_unused_layers", "keep_end_layers", "roundrect_rratio", "chamfer_ratio", "options",
        "thermal_bridge_angle",  # PAD::Flip does not touch the spoke angle (KiCad source); DRC cannot tell either way
        "thermal_bridge_width", "thermal_gap", "zone_connect", "clearance", "solder_mask_margin", "solder_paste_margin",
        "solder_paste_margin_ratio", "die_length", "net", "pinfunction", "pintype", "uuid", "tstamp",
    }
)
_CHAMFER_FLIP = {"top_left": "bottom_left", "bottom_left": "top_left", "top_right": "bottom_right", "bottom_right": "top_right"}
_STANDARD_PROPERTIES = ("Reference", "Value", "Datasheet", "Description")
#: never copied from a library footprint onto a board footprint
_DROPPED_PROPERTIES = frozenset({"Footprint"})


# --------------------------------------------------------------------------- public helpers


def net_numbers(ir: CircuitIR) -> dict[str, int]:
    """Deterministic net codes: ``{"": 0, <IR nets sorted by name>: 1..n}``.

    KiCad 10's 20260206 format references nets by name, so these codes are not
    written into the board; they are the numbering any code-based exporter
    must use. Duplicate net names are a :class:`CompileError`.
    """
    names = [n.name for n in ir.nets]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise CompileError(f"duplicate net names in IR: {dupes}")
    if "" in names:
        raise CompileError("a net with an empty name cannot be compiled (KiCad reserves net 0 for 'no net')")
    table = {"": 0}
    for i, name in enumerate(sorted(names), start=1):
        table[name] = i
    return table


def design_rules(ir: CircuitIR) -> dict[str, float]:
    """``.kicad_pro`` ``board.design_settings.rules`` values from *authoritative* fab constraints.

    Only :data:`MANUFACTURING_RULE_KEYS` are mapped; values whose provenance is
    not authoritative / user-required are left out so an unverified limit never
    becomes a rule. The PCB compiler does not write a project file (yet), so
    this is exposed for the stage that will.
    """
    if ir.pcb is None:
        return {}
    rules: dict[str, float] = {}
    for field_name, key in MANUFACTURING_RULE_KEYS.items():
        traced = getattr(ir.pcb.manufacturing, field_name)
        if traced is not None and traced.provenance.is_authoritative:
            rules[key] = float(traced.value)
    return rules


def copper_layer_index(name: str) -> int:
    """KiCad 9/10 copper layer id: ``F.Cu`` 0, ``B.Cu`` 2, ``In<n>.Cu`` 2n+2; other names -> CompileError."""
    if name == "F.Cu":
        return 0
    if name == "B.Cu":
        return 2
    if name.startswith("In") and name.endswith(".Cu"):
        digits = name[2:-3]
        if digits.isdigit() and int(digits) >= 1:
            return 2 * int(digits) + 2
    raise CompileError(f"unknown copper layer name {name!r} (expected F.Cu, B.Cu or In<n>.Cu)")


# --------------------------------------------------------------------------- resolved component


@dataclass(slots=True)
class _Placed:
    component: Component
    placement: Placement
    footprint: FootprintDef
    pad_nets: dict[str, str]  # pad number -> net name (only connected pads)
    pin_types: dict[str, str]  # pad number -> KiCad pintype, verified against the library symbol
    description: str  # value of the footprint's Description property


# --------------------------------------------------------------------------- the compiler


class PCBCompiler(Compiler):
    id = "compiler.kicad_pcb"
    version = "0.2"
    kind = ArtifactKind.PCB

    def compile(self, ir: CircuitIR, ctx: CompileContext) -> ArtifactRef:
        node = self.build(ir, ctx)
        path = Path(ctx.workdir) / f"{ir.project.id}.kicad_pcb"
        return self._write(ir, path, sexpr.dumps(node))

    # --- tree construction (pure: same IR -> same tree) -------------------------

    def build(self, ir: CircuitIR, ctx: CompileContext) -> list:
        """The complete ``(kicad_pcb ...)`` tree for ``ir``; raises CompileError instead of guessing."""
        if ir.pcb is None:
            raise NothingToCompileError("ir.pcb is None: nothing to lay out (the PCB stage needs an outline and placements)")
        if not ir.components:
            raise NothingToCompileError("IR has no components: nothing to place")
        if not ir.project.id or _BAD_STEM_RE.search(ir.project.id):
            raise CompileError(f"project id {ir.project.id!r} is not usable as a KiCad file stem (it names the board and schematic files)")
        library = self._library(ctx)
        net_numbers(ir)  # validates net names
        placed = self._resolve_components(ir, library)
        copper = self._copper_layers(ir)
        sheetfile = f"{ir.project.id}.kicad_sch"
        node = S(
            "kicad_pcb",
            S("version", FILE_VERSION),
            S("generator", Q(GENERATOR)),
            S("generator_version", Q(GENERATOR_VERSION)),
            self._general(ir),
            S("paper", Q("A4")),
            S("title_block", S("title", Q(ir.project.name))) if ir.project.name else None,
            self._layers(copper),
            self._setup(),
        )
        # KiCad writes footprints sorted by (layer, uuid); tracks keep IR order (KiCad re-sorts those by
        # net code on save, which depends on load order - not replicated, no semantic effect).
        footprints = [(0 if p.placement.side == BoardSide.TOP else 2, ids.footprint_uuid(ir.project.id, p.component.ref), p) for p in placed]
        for _, _, p in sorted(footprints, key=lambda t: (t[0], t[1])):
            node.append(self._footprint(ir.project.id, p, sheetfile))
        node.append(self._outline(ir))
        net_names = {n.name for n in ir.nets}
        copper_names = [name for _, name, _ in copper]
        for i, track in enumerate(ir.pcb.tracks):
            node.append(self._segment(ir.project.id, i, track, net_names, copper_names))
        for i, via in enumerate(ir.pcb.vias):
            node.append(self._via(ir.project.id, i, via, net_names, copper_names))
        for i, zone in enumerate(ir.pcb.zones):
            node.append(self._zone(ir.project.id, i, zone, net_names, copper_names))
        node.append(S("embedded_fonts", False))
        return node

    # --- inputs --------------------------------------------------------------------

    @staticmethod
    def _library(ctx: CompileContext) -> KicadLibrary:
        lib = ctx.tools.get("kicad_library")
        if lib is None:
            lib = KicadLibrary()
        if not isinstance(lib, KicadLibrary):
            raise CompileError(f"ctx.tools['kicad_library'] is not a KicadLibrary: {type(lib).__name__}")
        return lib

    @staticmethod
    def _resolve_components(ir: CircuitIR, library: KicadLibrary) -> list[_Placed]:
        assert ir.pcb is not None
        refs = [c.ref for c in ir.components]
        dupes = sorted({r for r in refs if refs.count(r) > 1})
        if dupes:
            raise CompileError(f"duplicate component references in IR: {dupes}")
        known = set(refs)
        for pl in ir.pcb.placements:
            if pl.component_ref not in known:
                raise CompileError(f"placement for unknown component {pl.component_ref!r}")
        # pin -> net, validated once
        pin_nets: dict[tuple[str, str], str] = {}
        for net in ir.nets:
            for pin in net.pins:
                comp = ir.component(pin.component_ref)
                if comp is None:
                    raise CompileError(f"net {net.name!r} references unknown component {pin.component_ref!r}")
                if comp.pin(pin.pin_number) is None:
                    raise CompileError(f"net {net.name!r} references {comp.ref}.{pin.pin_number}, which is not an IR pin of {comp.ref!r}")
                key = (comp.ref, pin.pin_number)
                if key in pin_nets and pin_nets[key] != net.name:
                    raise CompileError(f"pin {comp.ref}.{pin.pin_number} is on two nets: {pin_nets[key]!r} and {net.name!r}")
                pin_nets[key] = net.name
        placed: list[_Placed] = []
        for comp in ir.components:
            if comp.footprint is None:
                raise CompileError(f"component {comp.ref!r} has no footprint; the PCB compiler never picks one")
            resolved = library.resolve_footprint(comp.footprint)
            if not resolved.verified:
                raise CompileError(
                    f"footprint {comp.footprint.library}:{comp.footprint.name} of {comp.ref!r} was not found in a KiCad "
                    f"library (searched {[str(r) for r in library.roots]}); refusing to guess"
                )
            fp = library.load_footprint(comp.footprint)
            placement = ir.pcb.placement(comp.ref)
            if placement is None:
                raise CompileError(f"component {comp.ref!r} has no placement in ir.pcb.placements")
            pad_nets = PCBCompiler._pad_nets(comp, fp, pin_nets)
            # pad pin types come from the verified library symbol - the same check the schematic compiler runs
            symbol = load_verified_symbol(comp, library)
            pin_types = pad_pin_types(comp, symbol)
            description = comp.description or symbol.properties.get("Description", "")
            placed.append(_Placed(comp, placement, fp, pad_nets, pin_types, description))
        return placed

    @staticmethod
    def _pad_nets(comp: Component, fp: FootprintDef, pin_nets: dict[tuple[str, str], str]) -> dict[str, str]:
        """IR pins vs library pads: a mismatch is an error, not a guess."""
        pad_numbers = {p.number for p in fp.pads if p.number}
        electrical = {p.number for p in fp.pads if p.number and p.pad_type != "np_thru_hole"}
        ir_pins = {p.number for p in comp.pins}
        fp_id = f"{fp.lib_id}"
        without_pad = sorted(ir_pins - pad_numbers)
        if without_pad:
            raise CompileError(f"{comp.ref!r}: IR pins {without_pad} have no pad in footprint {fp_id} (pads: {sorted(pad_numbers)})")
        extra = sorted(electrical - ir_pins)
        if extra:
            raise CompileError(f"{comp.ref!r}: footprint {fp_id} has pads {extra} that are not IR pins (pins: {sorted(ir_pins)})")
        pad_nets: dict[str, str] = {}
        for (ref, pin_number), net_name in pin_nets.items():
            if ref != comp.ref:
                continue
            if pin_number not in pad_numbers:
                raise CompileError(f"net {net_name!r} references {ref}.{pin_number} but footprint {fp_id} has no pad {pin_number!r}")
            pad_nets[pin_number] = net_name
        return pad_nets

    @staticmethod
    def _copper_layers(ir: CircuitIR) -> list[tuple[int, str, str]]:
        assert ir.pcb is not None
        layers = ir.pcb.layers
        if not layers:
            raise CompileError("ir.pcb.layers is empty; a board needs at least F.Cu and B.Cu")
        names = [layer.name for layer in layers]
        if len(set(names)) != len(names):
            raise CompileError(f"duplicate copper layers in ir.pcb.layers: {names}")
        for layer in layers:
            if layer.kind not in _COPPER_KINDS:
                raise CompileError(f"layer {layer.name!r}: kind {layer.kind!r} is not one of {sorted(_COPPER_KINDS)}")
        indexed = sorted((copper_layer_index(layer.name), layer.name, layer.kind) for layer in layers)
        if "F.Cu" not in names or "B.Cu" not in names:
            raise CompileError(f"ir.pcb.layers must contain F.Cu and B.Cu (got {names})")
        inner = sorted(idx for idx, name, _ in indexed if name not in ("F.Cu", "B.Cu"))
        if inner != [2 * n + 2 for n in range(1, len(inner) + 1)]:
            raise CompileError(f"inner copper layers must be In1.Cu..In<n>.Cu without gaps (got {names})")
        if len(indexed) % 2:
            raise CompileError(f"KiCad needs an even copper layer count (got {len(indexed)}: {names})")
        # KiCad's file order: F.Cu, In1.Cu .. In<n>.Cu, B.Cu
        return [entry for entry in indexed if entry[1] == "F.Cu"] + [
            entry for entry in indexed if entry[1] not in ("F.Cu", "B.Cu")
        ] + [entry for entry in indexed if entry[1] == "B.Cu"]

    # --- board-level nodes ---------------------------------------------------------

    @staticmethod
    def _general(ir: CircuitIR) -> list:
        assert ir.pcb is not None
        thickness = DEFAULT_BOARD_THICKNESS_MM
        traced = ir.pcb.manufacturing.board_thickness_mm
        if traced is not None and traced.provenance.is_authoritative:
            thickness = float(traced.value)
        return S("general", S("thickness", thickness), S("legacy_teardrops", False))

    @staticmethod
    def _setup() -> list:
        """KiCad 10's defaults, exactly as ``kicad-cli pcb upgrade --force`` adds them to a bare board.

        Design rules are *not* here (they live in ``.kicad_pro``, see :func:`design_rules`).
        Vias are tented (mask covers them) and neither filled nor capped; ``pcbplotparams``
        is the GUI plot dialog state and only matters with ``--board-plot-params``.
        """
        return S(
            "setup",
            S("pad_to_mask_clearance", 0),
            S("allow_soldermask_bridges_in_footprints", False),
            S("tenting", S("front", True), S("back", True)),
            S("covering", S("front", False), S("back", False)),
            S("plugging", S("front", False), S("back", False)),
            S("capping", False),
            S("filling", False),
            S(
                "pcbplotparams",
                S("layerselection", "0x00000000_00000000_55555555_5755f5ff"),
                S("plot_on_all_layers_selection", "0x00000000_00000000_00000000_00000000"),
                S("disableapertmacros", False),
                S("usegerberextensions", False),
                S("usegerberattributes", True),
                S("usegerberadvancedattributes", True),
                S("creategerberjobfile", True),
                S("dashed_line_dash_ratio", 12),
                S("dashed_line_gap_ratio", 3),
                S("svgprecision", 4),
                S("plotframeref", False),
                S("mode", 1),
                S("useauxorigin", False),
                S("pdf_front_fp_property_popups", True),
                S("pdf_back_fp_property_popups", True),
                S("pdf_metadata", True),
                S("pdf_single_document", False),
                S("dxfpolygonmode", True),
                S("dxfimperialunits", True),
                S("dxfusepcbnewfont", True),
                S("psnegative", False),
                S("psa4output", False),
                S("plot_black_and_white", True),
                S("sketchpadsonfab", False),
                S("plotpadnumbers", False),
                S("hidednponfab", False),
                S("sketchdnponfab", True),
                S("crossoutdnponfab", True),
                S("subtractmaskfromsilk", False),
                S("outputformat", 1),
                S("mirror", False),
                S("drillshape", 1),
                S("scaleselection", 1),
                S("outputdirectory", Q("")),
            ),
        )

    @staticmethod
    def _layers(copper: list[tuple[int, str, str]]) -> list:
        node = S("layers")
        for idx, name, kind in copper:
            node.append(S(str(idx), Q(name), kind))
        for idx, name, user_name in NON_COPPER_LAYERS:
            node.append(S(str(idx), Q(name), "user", Q(user_name) if user_name else None))
        return node

    @staticmethod
    def _outline(ir: CircuitIR) -> list:
        assert ir.pcb is not None
        outline = ir.pcb.outline
        if outline is None:
            raise CompileError("ir.pcb.outline is None; a board without an Edge.Cuts outline fails DRC (invalid_outline)")
        if outline.width_mm <= 0 or outline.height_mm <= 0:
            raise CompileError(f"ir.pcb.outline must have positive size (got {outline.width_mm} x {outline.height_mm})")
        return S(
            "gr_rect",
            S("start", outline.origin_x_mm, outline.origin_y_mm),
            S("end", outline.origin_x_mm + outline.width_mm, outline.origin_y_mm + outline.height_mm),
            S("stroke", S("width", EDGE_LINE_WIDTH_MM), S("type", "solid")),
            S("fill", False),
            S("layer", Q("Edge.Cuts")),
            S("uuid", Q(ids.net_item_uuid(ir.project.id, "outline"))),
        )

    @staticmethod
    def _check_net_and_layer(what: str, net: str, layer: str, net_names: set[str], copper_names: list[str]) -> None:
        if net not in net_names:
            raise CompileError(f"{what} references unknown net {net!r}")
        if layer not in copper_names:
            raise CompileError(f"{what} is on layer {layer!r}, which is not a copper layer of this board ({copper_names})")

    @classmethod
    def _segment(cls, project_id: str, index: int, track: Track, net_names: set[str], copper_names: list[str]) -> list:
        cls._check_net_and_layer(f"track #{index}", track.net, track.layer, net_names, copper_names)
        if track.width_mm <= 0:
            raise CompileError(f"track #{index} ({track.net}) has non-positive width {track.width_mm}")
        return S(
            "segment",
            S("start", float(track.start[0]), float(track.start[1])),
            S("end", float(track.end[0]), float(track.end[1])),
            S("width", float(track.width_mm)),
            S("layer", Q(track.layer)),
            S("net", Q(track.net)),
            S("uuid", Q(ids.net_item_uuid(project_id, "track", index))),
        )

    @classmethod
    def _via(cls, project_id: str, index: int, via: Via, net_names: set[str], copper_names: list[str]) -> list:
        for layer in via.layers:
            cls._check_net_and_layer(f"via #{index}", via.net, layer, net_names, copper_names)
        if via.drill_mm <= 0 or via.diameter_mm <= via.drill_mm:
            raise CompileError(f"via #{index} ({via.net}): diameter {via.diameter_mm} must exceed drill {via.drill_mm} > 0")
        return S(
            "via",
            S("at", float(via.x_mm), float(via.y_mm)),
            S("size", float(via.diameter_mm)),
            S("drill", float(via.drill_mm)),
            S("layers", Q(via.layers[0]), Q(via.layers[1])),
            S("net", Q(via.net)),
            S("uuid", Q(ids.net_item_uuid(project_id, "via", index))),
        )

    @classmethod
    def _zone(cls, project_id: str, index: int, zone: Zone, net_names: set[str], copper_names: list[str]) -> list:
        cls._check_net_and_layer(f"zone #{index}", zone.net, zone.layer, net_names, copper_names)
        if len(zone.polygon) < 3:
            raise CompileError(f"zone #{index} ({zone.net}) needs at least 3 polygon points")
        clearance = 0.5 if zone.clearance_mm is None else float(zone.clearance_mm)
        pts = S("pts", *[S("xy", float(x), float(y)) for x, y in zone.polygon])
        return S(
            "zone",
            S("net", Q(zone.net)),
            S("layer", Q(zone.layer)),
            S("uuid", Q(ids.net_item_uuid(project_id, "zone", index))),
            S("hatch", "edge", 0.5),
            S("connect_pads", "yes", S("clearance", clearance)),
            S("min_thickness", 0.25),
            S("fill", "yes", S("thermal_gap", 0.5), S("thermal_bridge_width", 0.5)),
            S("polygon", pts),
        )

    # --- footprint embedding ---------------------------------------------------------

    @classmethod
    def _footprint(cls, project_id: str, p: _Placed, sheetfile: str) -> list:
        comp, placement, fp = p.component, p.placement, p.footprint
        lib = fp.node
        side = placement.side
        rot = footprint_angle(placement.rotation_deg)  # KiCad stores footprint orientation in (-180, 180]
        node = S(
            "footprint",
            Q(fp.lib_id),
            S("layer", Q("B.Cu" if side == BoardSide.BOTTOM else "F.Cu")),
            S("uuid", Q(ids.footprint_uuid(project_id, comp.ref))),
            S("at", float(placement.x_mm), float(placement.y_mm), rot if rot else None),
        )
        for head_name in ("descr", "tags"):
            child = sexpr.find(lib, head_name)
            if child is not None:
                node.append(sexpr.deep_copy(child))
        node.extend(cls._properties(project_id, comp, placement, lib, p.description))
        node.append(S("path", Q("/" + ids.symbol_uuid(project_id, comp.ref))))
        node.append(S("sheetname", Q("/")))
        node.append(S("sheetfile", Q(sheetfile)))
        attr = sexpr.find(lib, "attr")
        if attr is not None:
            node.append(sexpr.deep_copy(attr))
        # footprint-level settings KiCad writes between attr and the graphics (copied verbatim, library order)
        for child in lib:
            h = sexpr.head(child)
            if h is None or h in _FP_HEADER_HEADS or h in _FP_MANAGED_HEADS or h in _FP_GRAPHIC_HEADS:
                continue
            node.append(sexpr.deep_copy(child))
        gfx_index = 0
        for child in lib:
            h = sexpr.head(child)
            if h in _FP_GRAPHIC_HEADS:
                node.append(cls._graphic(project_id, comp.ref, gfx_index, child, placement))
                gfx_index += 1
        for child in sexpr.find_all(lib, "pad"):
            node.append(cls._pad(project_id, comp, placement, child, p.pad_nets, p.pin_types))
        fonts = sexpr.find(lib, "embedded_fonts")
        if fonts is not None:
            node.append(sexpr.deep_copy(fonts))
        for model in sexpr.find_all(lib, "model"):
            node.append(sexpr.deep_copy(model))
        return node

    @classmethod
    def _properties(cls, project_id: str, comp: Component, placement: Placement, lib: list, description: str) -> list[list]:
        lib_props: dict[str, list] = {}
        for prop in sexpr.find_all(lib, "property"):
            lib_props.setdefault(str(prop[1]), prop)
        values = {
            "Reference": comp.ref,
            "Value": comp.value,
            "Datasheet": (comp.datasheet.url or "") if comp.datasheet is not None else "",
            "Description": description,
        }
        out: list[list] = []
        for key in _STANDARD_PROPERTIES:
            template = lib_props.get(key)
            if template is None:
                template = cls._default_property(key)
            out.append(cls._property(project_id, comp.ref, key, values[key], template, placement))
        for key, prop in lib_props.items():
            if key in _STANDARD_PROPERTIES or key in _DROPPED_PROPERTIES:
                continue
            out.append(cls._property(project_id, comp.ref, key, str(prop[2]), prop, placement))
        return out

    @staticmethod
    def _default_property(key: str) -> list:
        if key == "Reference":
            return S("property", Q(key), Q(""), S("at", 0, 0, 0), S("layer", Q("F.SilkS")),
                     S("effects", S("font", S("size", 1, 1), S("thickness", 0.15))))
        if key == "Value":
            return S("property", Q(key), Q(""), S("at", 0, 0, 0), S("layer", Q("F.Fab")),
                     S("effects", S("font", S("size", 1, 1), S("thickness", 0.15))))
        return S("property", Q(key), Q(""), S("at", 0, 0, 0), S("layer", Q("F.Fab")), S("hide", True),
                 S("effects", S("font", S("size", 1.27, 1.27))))

    @classmethod
    def _property(cls, project_id: str, ref: str, key: str, value: str, template: list, placement: Placement) -> list:
        node = sexpr.deep_copy(template)
        node[1] = Q(key)
        node[2] = Q(value)
        if placement.side == BoardSide.BOTTOM:
            cls._mirror_points(node)
            cls._mirror_layers(node, placement.side)
        cls._transform_text(node, placement)
        cls._insert_uuid(node, ids.net_item_uuid(project_id, "fp_property", ref, key))
        return node

    @classmethod
    def _graphic(cls, project_id: str, ref: str, index: int, lib_item: list, placement: Placement) -> list:
        node = sexpr.deep_copy(lib_item)
        h = sexpr.head(node)
        if placement.side == BoardSide.BOTTOM:
            if h not in _MIRRORABLE_HEADS:
                raise CompileError(f"{ref!r}: cannot place footprint with a ({h} ...) item on the bottom side (no mirror rule)")
            cls._mirror_points(node)
            cls._mirror_layers(node, placement.side)
        if h in ("fp_text", "fp_text_box"):
            cls._transform_text(node, placement)
        cls._insert_uuid(node, ids.net_item_uuid(project_id, "fp_graphic", ref, index))
        return node

    @classmethod
    def _pad(
        cls, project_id: str, comp: Component, placement: Placement, lib_pad: list, pad_nets: dict[str, str], pin_types: dict[str, str]
    ) -> list:
        node = sexpr.deep_copy(lib_pad)
        number = str(node[1])
        at = sexpr.find(node, "at")
        if at is None or len(at) < 3:
            raise CompileError(f"{comp.ref!r}: pad {number!r} has no (at x y)")
        lib_angle = sexpr.to_float(at[3]) if len(at) > 3 else 0.0
        if placement.side == BoardSide.BOTTOM:
            cls._flip_pad(comp.ref, number, node, placement.side)
            angle = normalize_angle(placement.rotation_deg - lib_angle)  # PAD::Flip negates the pad orientation
        else:
            angle = normalize_angle(placement.rotation_deg + lib_angle)
        at[:] = S("at", at[1], at[2], angle if angle else None)
        # (net "NAME") goes after the last "before-net" child; then pintype; uuid last
        net_name = pad_nets.get(number)
        pintype_atom = pin_types.get(number) if number else None
        insert_at = 0
        for i, child in enumerate(node):
            if sexpr.head(child) in _PAD_BEFORE_NET:
                insert_at = i + 1
        extras: list[list] = []
        if net_name is not None:
            extras.append(S("net", Q(net_name)))
        if pintype_atom is not None:
            existing = sexpr.find(node, "pintype")
            if existing is not None:
                node.remove(existing)
            extras.append(S("pintype", Q(pintype_atom)))
        node[insert_at:insert_at] = extras
        existing_uuid = sexpr.find(node, "uuid")
        if existing_uuid is not None:
            node.remove(existing_uuid)
        node.append(S("uuid", Q(ids.pad_uuid(project_id, comp.ref, number))))
        return node

    # --- node surgery ----------------------------------------------------------------

    @classmethod
    def _flip_pad(cls, ref: str, number: str, node: list, side: BoardSide) -> None:
        """Apply ``PAD::Flip(TOP_BOTTOM)`` to a library pad copied onto the bottom side.

        Every child must be covered by one of the rule sets above; anything
        else is refused so a construct KiCad would rewrite is never copied
        through unchanged (that is exactly what DRC reports as
        ``lib_footprint_mismatch`` and what the fab would build wrongly).
        """
        for child in node:
            h = sexpr.head(child)
            if h is None:
                continue
            if h in _PAD_FLIP_CHAMFER:
                corners = sexpr.args(child)
                unknown = [c for c in corners if c not in _CHAMFER_FLIP]
                if unknown:
                    raise CompileError(f"{ref!r}: pad {number!r} has unknown chamfer corner(s) {unknown}")
                child[1:] = [_CHAMFER_FLIP[c] for c in corners]
            elif h not in _PAD_FLIP_MIRROR_Y and h not in _PAD_FLIP_LAYERS and h not in _PAD_FLIP_INVARIANT:
                raise CompileError(
                    f"{ref!r}: cannot place pad {number!r} on the bottom side: no flip rule for its ({h} ...) child"
                )
        cls._mirror_points(node)  # negates y of (at), (offset), (rect_delta) and custom-shape primitives
        cls._mirror_layers(node, side)

    @staticmethod
    def _insert_uuid(node: list, uuid: str) -> None:
        """Insert ``(uuid ..)`` where KiCad writes it: after ``(layer ..)`` / ``(hide ..)``, before ``(effects ..)``."""
        existing = sexpr.find(node, "uuid")
        if existing is not None:
            node.remove(existing)
        pos = len(node)
        for i, child in enumerate(node):
            if sexpr.head(child) in ("layer", "hide", "unlocked", "locked"):
                pos = i + 1
        effects = sexpr.find(node, "effects")
        if effects is not None:
            pos = min(pos, node.index(effects))
        node.insert(pos, S("uuid", Q(uuid)))

    @staticmethod
    def _transform_text(node: list, placement: Placement) -> None:
        """Set the stored angle of a property / fp_text ``(at x y a)`` and, on the bottom, ``(justify mirror)``.

        Coordinates and layers must already have been mirrored by the caller
        (:meth:`_mirror_points` / :meth:`_mirror_layers`) for bottom-side items.
        """
        at = sexpr.find(node, "at")
        if at is None or len(at) < 3:
            raise CompileError(f"({sexpr.head(node)} ...) without (at x y [angle])")
        lib_angle = sexpr.to_float(at[3]) if len(at) > 3 else 0.0
        at[:] = S("at", at[1], at[2], text_angle(placement, lib_angle))
        if placement.side == BoardSide.BOTTOM:
            effects = sexpr.find(node, "effects")
            if effects is None:
                effects = S("effects")
                node.append(effects)
            justify = sexpr.find(effects, "justify")
            if justify is None:
                effects.append(S("justify", "mirror"))
            elif "mirror" not in sexpr.args(justify):
                justify.append("mirror")

    @classmethod
    def _mirror_points(cls, node: list) -> None:
        """Negate the y of every coordinate child (start/end/center/mid/xy/offset/delta), recursively.

        ``(at x y a)`` nodes are handled by the callers (angle rules differ per item type).
        """
        for child in node:
            if not isinstance(child, list):
                continue
            h = sexpr.head(child)
            if h in _POINT_HEADS and len(child) >= 3 and not isinstance(child[2], list):
                child[2] = float(-sexpr.to_float(child[2]))
            elif h == "at" and len(child) >= 3 and node[0] != "model":
                child[2] = float(-sexpr.to_float(child[2]))
            elif h not in ("effects", "stroke", "fill", "font", "model"):
                cls._mirror_points(child)

    @staticmethod
    def _mirror_layers(node: list, side: BoardSide) -> None:
        for child in node:
            if not isinstance(child, list):
                continue
            h = sexpr.head(child)
            if h == "layer" and len(child) > 1:
                child[1] = Q(mirrored_layer(str(child[1]), side))
            elif h == "layers":
                child[1:] = [Q(mirrored_layer(str(a), side)) if not isinstance(a, list) else a for a in child[1:]]
