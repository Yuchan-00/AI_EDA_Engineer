"""The router's stack input: which copper layers carry signals, which are planes of which net, and which via structures join them.

Pure data for the N-layer router (``routing.maze`` 0.9, stack mode): this
module knows nothing of the maze router and the router reads nothing else of
the stack. A :class:`RouterStack` is built from a board's
:class:`~ai_eda.ir.Stackup` (:meth:`RouterStack.from_stackup`) plus the via
structures the stack allows - for the layer escalation, the ``vias`` of the
generic stack (:data:`ai_eda.design.stackup.ESCALATION_STACKS`), each a legal
run of that stack's core / prepreg map (``design.stackup.span_problem``).
Whether a stack is routable (its layers match the board, its first via type is
the through via, every span is legal) is the router's call: it refuses with
the reason, so this module stores what it is given.

Layer names and orders are KiCad's: ``F.Cu``, ``In1.Cu`` .. ``In<n>.Cu``,
``B.Cu`` top to bottom; a via type names its upper layer first.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from ai_eda.ir import Stackup

#: what a via structure is: a through via (F.Cu-B.Cu), a blind via (one outer end), a buried via (no outer end) or a laser micro via (one dielectric)
ViaKind = Literal["through", "blind", "buried", "micro"]
VIA_KINDS: tuple[str, ...] = ("through", "blind", "buried", "micro")
#: the directions a signal layer may prefer (a wrong-way step costs more, it is never banned)
DIRECTIONS: tuple[str, ...] = ("horizontal", "vertical")


@dataclass(frozen=True)
class ViaType:
    """One via structure of a stack: its kind and the copper layers it joins, upper one first."""

    kind: ViaKind
    top: str
    bottom: str


#: the via every stack has, always type 0
THROUGH = ViaType("through", "F.Cu", "B.Cu")


@dataclass(frozen=True)
class RouterStack:
    """A board stack as the router sees it.

    ``copper`` is every copper layer in stack order; ``signal`` the routing
    layers in stack order (``F.Cu`` and ``B.Cu`` included); ``planes`` the
    ``(layer, net)`` of each plane layer; ``vias`` the via types, the through
    via first; ``directions`` the ``(signal layer, horizontal / vertical)``
    preferred direction of each signal layer (empty: none stated); ``label``
    names the stack in notes and provenance (the generic stack's title).
    """

    label: str
    copper: tuple[str, ...]
    signal: tuple[str, ...]
    planes: tuple[tuple[str, str], ...]
    vias: tuple[ViaType, ...]
    directions: tuple[tuple[str, str], ...] = ()

    @classmethod
    def from_stackup(cls, stackup: Stackup, vias: Sequence[tuple[str, str, str]], label: str, *,
                     directions: Sequence[tuple[str, str]] = ()) -> RouterStack:
        """The router's view of ``stackup``: its copper, signal and plane layers, the through via and then ``vias``.

        ``vias`` are ``(kind, top, bottom)`` besides the through via, which is
        implicit (``GenericStack.vias``). ``ValueError`` for a via kind other
        than blind, buried or micro (a listed through via included), and for a
        direction that names no signal layer of the stack or is neither
        horizontal nor vertical. The spans themselves are judged by the router.
        """
        signal = tuple(stackup.signal_layers())
        types = [THROUGH]
        for kind, top, bottom in vias:
            if kind not in VIA_KINDS or kind == "through":
                raise ValueError(f"via type {kind} {top}-{bottom}: the kind must be blind, buried or micro (the through via F.Cu-B.Cu is implicit)")
            types.append(ViaType(kind, top, bottom))
        for layer, direction in directions:
            if layer not in signal:
                raise ValueError(f"preferred direction for {layer}: not a signal layer of the stack {list(signal)}")
            if direction not in DIRECTIONS:
                raise ValueError(f"preferred direction for {layer}: {direction!r} is not horizontal or vertical")
        return cls(
            label=label,
            copper=tuple(stackup.copper_names()),
            signal=signal,
            planes=tuple((c.name, str(c.plane_net.value)) for c in stackup.plane_layers()),
            vias=tuple(types),
            directions=tuple((layer, direction) for layer, direction in directions),
        )


__all__ = ["DIRECTIONS", "THROUGH", "VIA_KINDS", "RouterStack", "ViaKind", "ViaType"]
