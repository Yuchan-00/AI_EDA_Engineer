"""The blocks of the RF templates (power, PTT, audio, IF back-end, front-end, LO / TX chains, PA, T/R) and their shared builder API.

Invariant: every block is built through :mod:`ai_eda.design.rf.blocks.base`
- parts only from the parts table and the library on disk, names only from
its prefix, structure checked before the block is used - so a stage board and
the transceiver that composes it name the same circuit the same way. The
block modules themselves are added by the kr447 wave-2 parts; this package
starts with the API.
"""

from ai_eda.design.rf.blocks.base import (
    GROUND_NET,
    Block,
    BlockBuilder,
    BlockContext,
    BlockPrefix,
    BlockResult,
    Floating,
    FloatingReport,
    apply_prefix,
    exclude_floating,
    merge_results,
    rename_vector,
)

__all__ = [
    "GROUND_NET",
    "Block",
    "BlockBuilder",
    "BlockContext",
    "BlockPrefix",
    "BlockResult",
    "Floating",
    "FloatingReport",
    "apply_prefix",
    "exclude_floating",
    "merge_results",
    "rename_vector",
]
