from enum import StrEnum


class Stage(StrEnum):
    REQUIREMENT_ANALYSIS = "requirement_analysis"
    MISSING_INFORMATION = "missing_information"
    REGULATORY_RESEARCH = "regulatory_research"
    ARCHITECTURE = "architecture"
    COMPONENT_SELECTION = "component_selection"
    PLACEMENT = "placement"
    IR_BUILD = "ir_build"
    CALCULATION = "calculation"
    SPICE = "spice"
    SCHEMATIC = "schematic"
    ERC = "erc"
    PCB = "pcb"
    DRC = "drc"
    MANUFACTURABILITY = "manufacturability"
    MANUFACTURING_OUTPUTS = "manufacturing_outputs"
    INDEPENDENT_REVIEW = "independent_review"
    REPAIR = "repair"
    RELEASE = "release"


STAGE_ORDER: list[Stage] = list(Stage)
