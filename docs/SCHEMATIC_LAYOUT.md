# Small-circuit schematic layout

The default schematic compiler remains version **0.8**, with the existing
table placement, wires and labels. The opt-in **textbook** policy emits
version **0.9** artifacts. It does not alter the electrical IR.

Policy selection, highest priority first:

1. `topology.schematic_layout` in the IR or design file (`legacy` or `textbook`).
2. `SchematicCompiler(layout="textbook")` for direct compilation.
3. `--answer schematic.layout=textbook` when running the pipeline.

An absent policy retains the previous serialization, design hash and output.
Explicit `legacy` overrides a lower-priority textbook selection. The pipeline
records the effective compiler version on the artifact and compilation result.

```json
"topology": {
  "name": "RC filter",
  "domains": ["analog"],
  "schematic_layout": "textbook"
}
```

**CHOICE:** textbook applies to at most **20 components including connectors**,
with at most **one non-connector having more than three pins**. Designs beyond
either limit compile with the unchanged 0.8 implementation, even when textbook
was requested. Existing library verification, pin-type, annotated-reference,
single-unit-symbol and intentional-no-connect requirements still apply.

The policy derives motifs from library pin geometry and net connectivity,
independently of template or topology names. Supply nets form horizontal top
rails. Ground symbols terminate individual branches. Series parts follow
supply-to-ground columns, while passive signal chains run from left to right.
Declared block inputs select the direction; without them, natural connector
reference and pin order provide the fixed tie-break. A header carrying both
input and output retains its real pins, with an additional input terminal at
the left edge. Feedback between an IC input and output sits above the IC.

Matching collector-coupled transistor halves use mirrored columns. The column
swap checks symbol identities and displayed values, excluding external
connectors. Their cross-coupling uses two diagonals with one undotted crossing.
T joints receive junction dots. All placements, fields and wire vertices use
the existing 1.27 mm grid, with upright reference/value fields.

No local or global net labels are emitted. Ground and supply symbols and the
input terminal are resolved from installed libraries, with generated `#`
references excluded from physical netlist components. Power flags are added
only where their electrical types allow them. They state that a rail is
externally powered; they do not verify the power source.

Routing avoids measured symbol and field bounds, unrelated pins and foreign
wire endpoints. It first searches without crossings; difficult general IC
wiring may use straight interior crossings at a high cost. This bounded search
does **not** establish a global minimum crossing number for arbitrary graphs.
Blocked routes, overlaps and accidental contacts are compilation errors;
there is no silent label fallback. Layout measurements are geometry evidence,
not electrical verification.

`tests/test_schematic_textbook.py` compares real KiCad netlist pin partitions
and ERC for both policies on divider, LED, RC/LC low-pass, BJT switch, astable
and I2C pull-up templates when KiCad is installed. Names of unlabeled signal
nets are assigned by KiCad, so connectivity is compared by `(reference, pin)`
membership. Golden geometry pins the mirrored astable, divider, horizontal RC
signal path and single-column BJT switch. Template metrics require zero
overlaps and labels, and zero crossings except the astable's single crossing.
An additional op-amp fixture checks feedback placement and connectivity.
