"""Exception hierarchy shared across the system."""


class AiEdaError(Exception):
    """Base class for all project errors."""


class ToolUnavailableError(AiEdaError):
    """A required external tool (kicad-cli, ngspice, ...) is not installed or not found."""


class ToolExecutionError(AiEdaError):
    """An external tool ran but failed or produced unparseable output."""


class ProvenanceError(AiEdaError):
    """A value was used in a context that requires stronger provenance than it has.

    Example: an ``llm_generated`` MPN being written into a BOM.
    """


class ApprovalRequiredError(AiEdaError):
    """An outward-facing action was attempted without explicit user approval."""

    def __init__(self, action: str, detail: str = "") -> None:
        self.action = action
        self.detail = detail
        super().__init__(f"approval required for external action '{action}'" + (f": {detail}" if detail else ""))


class NotRepairableError(AiEdaError):
    """A finding cannot be safely fixed automatically and needs a human decision."""


class ConsistencyError(AiEdaError):
    """Two derived artifacts (or an artifact and the IR) disagree."""


class IRSchemaError(AiEdaError):
    """An IR file can not be loaded as the design it claims to be.

    Raised by :meth:`ai_eda.ir.CircuitIR.load` for a ``schema_version`` this
    code does not know and for keys the models would silently drop (a typo in
    a hand-edited ``ir.json`` must not delete design or traceability data).
    """


class CompileError(AiEdaError):
    """The IR cannot be compiled into an artifact without guessing.

    Raised, for example, when a component's symbol/footprint is not verified
    in a KiCad library, or IR pin numbers do not match the library symbol.
    The compiler refuses rather than inventing data. The pipeline records
    such a compile as FAIL: the IR is inconsistent and a human must fix it.
    """


class NothingToCompileError(CompileError):
    """The part of the IR this compiler works from does not exist yet.

    Examples: no components (nothing to draw), ``ir.pcb is None`` (nothing to
    lay out), no PCB artifact (nothing to export gerbers from). This is not an
    inconsistency, so the pipeline records the stage as NOT_VERIFIED rather
    than FAIL - and never as PASS, because an empty file that a tool accepts
    is not evidence about a design.
    """
