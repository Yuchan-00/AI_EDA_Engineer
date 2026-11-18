# Third-party notices

This repository is licensed under MIT + Commons Clause (see `LICENSE`). The parts listed below were ported from projects under
other permissive licenses; their copyright lines and license texts are kept here and in the ported file's header.

## VTR / VPR (Verilog-to-Routing)

- Source: https://github.com/verilog-to-routing/vtr-verilog-to-routing (root `LICENSE.md`, "VTR License", MIT; opened 2026-11-13).
- Ported: the PathFinder negotiation schedule of `vpr/src/route/route_common.h` (`get_single_rr_cong_cost`,
  `get_single_rr_cong_pres_cost`: cost = base x acc x pres, pres = 1 + pres_fac x (occ + 1 - capacity)),
  `vpr/src/route/route_common.cpp` (`pathfinder_update_acc_cost_and_overuse_info`: acc += overuse x acc_fac) and the pres_fac
  update of `vpr/src/route/route.cpp` (first iteration at first_iter_pres_fac, then initial_pres_fac, then x pres_fac_mult capped at
  max_pres_fac), with VPR's documented defaults.
- Into: `ai_eda/tools/routing/maze.py` (`StackSearchParams`, `_NegotiationN`; routing.maze 0.9, N signal layers), translated from
  C++ to Python onto the router's halo counts.

```
Copyright 2012 VTR Developers

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

Ideas only (no code copied): nextpnr router2 (ISC) - rerouting the failed nets in a shuffled order each iteration, realised here as
a deterministic sha256 order.
