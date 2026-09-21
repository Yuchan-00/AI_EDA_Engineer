from enum import StrEnum


class ReviewArea(StrEnum):
    """The 14 review areas from the project spec, in order."""

    REQUIREMENTS_VS_IR = "review.requirements_vs_ir"
    IR_VS_SCHEMATIC = "review.ir_vs_schematic"
    IR_VS_PCB = "review.ir_vs_pcb"
    SCHEMATIC_VS_PCB = "review.schematic_vs_pcb"
    PCB_VS_BOM = "review.pcb_vs_bom"
    PCB_VS_CPL = "review.pcb_vs_cpl"
    MANUFACTURING_OUTPUTS = "review.manufacturing_outputs"
    CALCULATIONS_VS_DESIGN = "review.calculations_vs_design"
    SPICE_VS_REQUIREMENTS = "review.spice_vs_requirements"
    ERC = "review.erc"
    DRC = "review.drc"
    REGULATORY_PROVENANCE = "review.regulatory_provenance"
    COMPONENT_PROVENANCE = "review.component_provenance"
    MANUFACTURING_CAPABILITIES = "review.manufacturing_capabilities"
