# Functional intent and layout measurements

`ai-eda-design/1` accepts two optional keys under `topology`:

```json
{
  "blocks": [
    {
      "id": "core",
      "function": "controller and supply support",
      "domain": "digital",
      "component_refs": ["U1", "C1"],
      "input_nets": ["VCC"],
      "output_nets": []
    }
  ],
  "pin_associations": [
    {
      "terminal": "C1.1",
      "ic_pin": "U1.1",
      "role": "supply decoupling",
      "rationale": "The support terminal assigned to this supply pin"
    }
  ]
}
```

These example references must exist in the actual design. `function` is the
existing block model's role. Associations name existing library pins on the
same net; they add intent without changing connectivity. Each declaration is
shown in the confirmation table. Blocks and associations receive
`user_requirement` provenance tied to the design file's hash. An association's
note also preserves its rationale. The file cannot claim authoritative
provenance. Invalid references are refused with `intent.component_ref`,
`intent.net_ref` or `intent.pin_ref`; a pair on different nets is refused with
`intent.association_net`.

Files without these keys keep their plan, confirmation table and IR. Empty
`Topology.pin_associations` and false `Block.inferred` are omitted from
serialization, including the design view, preserving existing IR hashes.
Declaring blocks opts into the existing compiler's block ordering; no new
schematic or placement policy is selected here.

`ai_eda.tools.functional_intent.resolve_intent(ir)` is a separate, pure,
opt-in view. It returns declared blocks when present, otherwise connected
components after excluding nets whose kind is `power` or `ground`.
Isolated components form singleton blocks. Inferred blocks carry
`inferred=true` and versioned `topology.connectivity` provenance. Block IDs
sort first; references and pin numbers use natural ordering. The view never
changes the IR or the confirmation table. It does not infer signal direction,
functional roles beyond connectivity, or pin associations. Its `unassociated`
list reports rail-only capacitors with no declared served pin: a shared rail
cannot establish that assignment. Rail classification is the IR's explicit
net kind, not a guess based on a net's spelling.

`ai_eda.tools.layout_metrics.schematic_metrics(path)` reads the compiled
sheet and its embedded symbol definitions. It reports symbol/symbol,
symbol/text and text/text bounding-box overlaps, global/local labels, wire
segments, wire length, crossings and bends. Bounds include pin text within
the symbol and visible properties as separate text objects. Text bounds use
the preview font estimate; bounding-box overlap is a geometric proxy, not
proof that two visible strokes collide. A crossing is a unique interior
intersection without a junction dot; a bend is a non-collinear degree-two
wire vertex. A hierarchical design requires one measurement per sheet.

`pcb_metrics(ir, library)` reports rectangular outline area, summed placed
courtyard/pad bounding-box areas, Euclidean pad-centre MST length per net,
its total and plane/non-plane subtotals, track length, vias and copper layers.
Plane nets are those assigned by a stackup or copper zone; the non-plane
subtotal also includes power nets routed as tracks. MST length ignores
obstacles and return paths. Missing placements or pad geometry are errors,
not zero-length measurements. All distances are mm and areas mm², rounded
to six decimals. Neither API produces an ERC/DRC or validation verdict.

The measured values are in `tests/data/layout_baselines/metrics.json`.
Compressed sheet and PCB geometry inputs total about 115 KiB and permit
offline remeasurement without installed libraries. The STM32 input retains
the saved run's six-layer routing. The ATmega128 template uses 9 V / 16 MHz;
the unloaded divider uses 5 V to 3.3 V. Both template baselines explicitly
request two copper layers and use the current placer/router with mitred
corners. Silkscreen is skipped because it is outside these metrics.

Regenerate with the installed libraries and a read-only saved STM32 run:

```text
python -m tests.generate_layout_baselines --stm32-run PATH_TO_SAVED_RUN
```

The generator writes only test fixtures and temporary template workdirs.
The baseline JSON records source IR and geometry hashes, template inputs and
placement/routing notes. It does not copy the archived validation logs or
assert that a new ERC/DRC run passed.
