"""Signal-integrity tools: routed-copper measurements, the net-class -> router rule mapping, the critical-length promotion.

* :mod:`ai_eda.tools.si.measure` - each net's routed length (tracks + via
  barrels), propagation delay and per-segment impedance from the IR copper,
  the stackup and the registered transmission-line calculators;
* :mod:`ai_eda.tools.si.rules` - ``ir.si`` net classes -> the router's
  per-net :class:`~ai_eda.tools.routing.maze.NetRule` (widths from
  ``calc.tline.width_for_z0`` over the stackup's plane, minimum widths,
  delay budgets converted with t_pd, declared pairs);
* :mod:`ai_eda.tools.si.paths` - the longest pad-to-pad path through a
  net's copper (the line the critical-length rule and ``spice.si`` measure);
* :mod:`ai_eda.tools.si.driver` - a class's driver edge / resistance / load
  (grounded datasheet fact of a driver that drives the net, or the class's
  confirmed choice);
* :mod:`ai_eda.tools.si.promote` - the critical-length rule and the
  promotion of electrically long nets to a controlled-impedance class.

None of it is a verdict: :mod:`ai_eda.validation.si` and ``spice.si``
(:mod:`ai_eda.tools.spice.si_check`) judge.
"""

from ai_eda.tools.si.driver import DRIVER_FACTS, DRIVING_TYPES, DriverValue, driver_value, drives, pins_on_net
from ai_eda.tools.si.measure import NOT_DRC, SI_TOOL, SI_VERSION, LineModel, NetLine, NetMeasure, Segment, line_model, measure_nets, via_model
from ai_eda.tools.si.paths import NetPath, longest_path, net_pads
from ai_eda.tools.si.promote import PROMOTE_TOOL, UNDRIVEN_KINDS, CriticalRow, critical_rows, promote
from ai_eda.tools.si.rules import WIDTH_STEP_MM, ClassRule, SIRules, class_rule, controlled_width, net_rules

__all__ = [
    "DRIVER_FACTS",
    "DRIVING_TYPES",
    "NOT_DRC",
    "PROMOTE_TOOL",
    "SI_TOOL",
    "SI_VERSION",
    "UNDRIVEN_KINDS",
    "WIDTH_STEP_MM",
    "ClassRule",
    "CriticalRow",
    "DriverValue",
    "LineModel",
    "NetLine",
    "NetMeasure",
    "NetPath",
    "SIRules",
    "Segment",
    "class_rule",
    "controlled_width",
    "critical_rows",
    "driver_value",
    "drives",
    "line_model",
    "longest_path",
    "measure_nets",
    "net_pads",
    "net_rules",
    "pins_on_net",
    "promote",
    "via_model",
]
