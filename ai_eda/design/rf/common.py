"""The KR 447 MHz family's common design choices that more than one block declares (kr447 design §2.0, "Common design choices").

Invariant: one key, one confirmation-table row. A choice several blocks
declare (the TX audio block designs its integrator with ``rf.n_mult``, the
TX modulator / multipliers and the RX LO chain build it) is declared from
the one value and the one text here, so the rows are identical and
:func:`~ai_eda.design.rf.blocks.base.merge_results` joins them on any board
that composes those blocks (the transceiver composes all three). A block
that finds the key already shared by an earlier block of its board
(``BlockContext.shared``) uses that one; the text here is what either
writes.
"""

from __future__ import annotations

#: the multiplication factor from the reference oscillator to the carrier (TX) and to LO1 (RX)
N_MULT_KEY = "rf.n_mult"
N_MULT = 12.0
N_MULT_TEXT = ("multiplication factor N = 3 x 2 x 2 (decision 1B) from the reference oscillator to the carrier (TX: f_c / 12 = 37.296875 MHz; "
               "the TX audio block's integrator is designed with it) and to LO1 (RX: (f_c - IF1) / 12 = 35.5135417 MHz); fewer stages and wider "
               "relative spur offsets than x36")

__all__ = ["N_MULT", "N_MULT_KEY", "N_MULT_TEXT"]
