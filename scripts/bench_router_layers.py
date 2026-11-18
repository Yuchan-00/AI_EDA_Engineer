"""Bench the N-layer router (routing.maze 0.9, multilayer plan P5 / P7): route a placed IR on an escalation stack for a fixed number of
negotiation iterations and print each iteration's progress, the time it took, the search effort and the process's peak memory.

A measuring tool, not a test: it reads the clock (the router never does) and pytest does not collect it (``scripts/`` is outside
``testpaths``). Run it from the repository root so ``ai_eda`` is imported from this checkout::

    python scripts/bench_router_layers.py PLACED_IR.json --layers 8 --iterations 3 [--ground GND] [--power +3V3] [--profile 25] [--corners arc]

``PLACED_IR.json`` is a placed :class:`~ai_eda.ir.CircuitIR` (``CircuitIR.load``), e.g. the STM32F407 example design
(``examples/designs/stm32f407_display_devboard.json``) taken through the PLACEMENT stage. The board gets the ``--layers`` copper
escalation stack (``ai_eda.design.stackup.escalation_stackup``: 4, 6, 8, 10 or 12) with its via structures and preferred directions
(P2), its ground planes on the ground net and its power planes on the power net, each plane zone the outline inset by ``--inset`` mm.
The plane nets follow the plan's rule unless named: the IR's single ground-kind net, and the power-kind net with the most pins (ties:
the name; none: the ground net). The negotiation cap is ``--iterations`` in place of ``RoutingParams.for_board``'s, so the run stops
there and the end game (N signal layers, P8) / drop / recovery that follows a capped negotiation is timed as the last step; the
last line names the nets the end game moved. ``--layers 2`` routes the board as it is
(no stack), for comparison. ``--corners arc`` (the PCB agent's default style) straightens the router's sidesteps and rounds the
routed corners into tangent arcs, mitring where no arc fits (routing.corners 0.2), ``--corners miter45`` mitres them
(routing.corners 0.1); either prints the pass's counts - the arcs and the sidesteps straightened too under ``arc``. The default
``orthogonal`` routes as the router alone does.

Per iteration it prints the elapsed and iteration seconds, the nets rerouted, the over-used cells, the conflicting nets and, on an
N-layer board, the present factor, the search states touched and the over-used vias per type. ``--profile N`` runs the whole route
under cProfile and prints the N most expensive functions and the ``heappop`` / ``heappush`` call counts. The peak working set is
read from the OS at the end (Windows: ``GetProcessMemoryInfo``; elsewhere ``getrusage``).
"""

from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import sys
import time
from collections import Counter
from dataclasses import replace

sys.path.insert(0, os.getcwd())

from ai_eda.design.stackup import PLANE_EDGE_CLEARANCE_MM, board_layers, escalation_stack, escalation_stackup  # noqa: E402
from ai_eda.ir import CircuitIR, NetKind  # noqa: E402
from ai_eda.tools.kicad.library import KicadLibrary  # noqa: E402
from ai_eda.tools.routing import maze  # noqa: E402
from ai_eda.tools.routing.stack import RouterStack  # noqa: E402


def peak_memory_mb() -> float | None:
    """The process's peak working set (MB), or ``None`` when the OS does not say."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        psapi = ctypes.WinDLL("psapi")
        kernel = ctypes.WinDLL("kernel32")
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return None
        return counters.PeakWorkingSetSize / 2 ** 20
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2 ** 20 if sys.platform == "darwin" else peak / 2 ** 10  # bytes on macOS, KiB on Linux


def plane_nets_of(ir: CircuitIR, ground: str | None, power: str | None) -> tuple[str, str | None]:
    """The ground and power plane nets: as given, else the single ground-kind net and the power-kind net with the most pins."""
    if ground is None:
        grounds = [net.name for net in ir.nets if net.kind == NetKind.GROUND]
        if len(grounds) != 1:
            raise SystemExit(f"name the ground plane net with --ground: the IR has {len(grounds)} ground-kind net(s) {grounds}")
        ground = grounds[0]
    if power is None:
        powers = sorted((net for net in ir.nets if net.kind == NetKind.POWER), key=lambda net: (-len(net.pins), net.name))
        power = powers[0].name if powers else None
    return ground, power


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ir", help="a placed CircuitIR (JSON)")
    ap.add_argument("--layers", type=int, default=8, help="copper layers: 2 (no stack), 4, 6, 8, 10 or 12 (default 8)")
    ap.add_argument("--iterations", type=int, default=3, help="the negotiation cap (default 3)")
    ap.add_argument("--ground", help="the ground plane net (default: the single ground-kind net)")
    ap.add_argument("--power", help="the power plane net (default: the power-kind net with the most pins)")
    ap.add_argument("--inset", type=float, default=PLANE_EDGE_CLEARANCE_MM, help="plane zone inset from the outline, mm")
    ap.add_argument("--profile", type=int, default=0, metavar="N", help="run under cProfile and print the N most expensive functions")
    ap.add_argument("--corners", choices=("orthogonal", "miter45", "arc"), default="orthogonal",
                    help="corner style: arc straightens the sidesteps and rounds the routed corners (routing.corners 0.2, the PCB agent's "
                         "default), miter45 mitres them (routing.corners 0.1); default orthogonal")
    args = ap.parse_args()

    t0 = time.perf_counter()
    ir = CircuitIR.load(args.ir)
    library = KicadLibrary()
    params = replace(maze.RoutingParams.for_board(ir, library), max_iterations=args.iterations)
    stack: RouterStack | None = None
    plane_nets: dict[str, list[tuple[str, list[tuple[float, float]]]]] | None = None
    if args.layers != 2:
        ground, power = plane_nets_of(ir, args.ground, args.power)
        stackup = escalation_stackup(args.layers, ground_net=ground, power_net=power)
        ir = ir.model_copy(update={"pcb": ir.pcb.model_copy(update={"layers": board_layers(stackup), "stackup": stackup})})
        g = escalation_stack(args.layers)
        stack = RouterStack.from_stackup(stackup, g.vias, g.title, directions=g.directions)
        o, d = ir.pcb.outline, args.inset
        poly = [(o.origin_x_mm + d, o.origin_y_mm + d), (o.origin_x_mm + o.width_mm - d, o.origin_y_mm + d),
                (o.origin_x_mm + o.width_mm - d, o.origin_y_mm + o.height_mm - d), (o.origin_x_mm + d, o.origin_y_mm + o.height_mm - d)]
        plane_nets = {}
        for layer, net in stack.planes:
            plane_nets.setdefault(net, []).append((layer, poly))
        print(f"stack: {stack.label}; signal {', '.join(stack.signal)}; planes {', '.join(f'{layer}={net}' for layer, net in stack.planes)}; "
              f"vias {', '.join(f'{v.kind} {v.top}-{v.bottom}' for v in stack.vias)}", flush=True)
    print(f"board: {maze.routed_net_count(ir)} nets to route, grid {params.grid_mm} mm, cap {params.max_iterations} iteration(s); "
          f"loaded in {time.perf_counter() - t0:.1f} s", flush=True)

    marks = {"start": time.perf_counter(), "last": time.perf_counter()}
    seconds: list[float] = []  # each iteration's
    init = maze._Board.__init__

    def timed_init(self, *a, **k):
        init(self, *a, **k)
        now = time.perf_counter()
        print(f"[{now - marks['start']:8.1f}s] static phase done ({now - marks['last']:.1f} s): grid {self.nx} x {self.ny} = {self.n} cells, "
              f"routing layers {', '.join(self.layer_names)}, signal via types {self.T}, search states "
              f"{self.R * self.n * maze._STATES_PER_CELL + self.barrels}", flush=True)
        marks["last"] = now

    maze._Board.__init__ = timed_init

    def progress(row: dict) -> None:
        now = time.perf_counter()
        seconds.append(now - marks["last"])
        extra = ""
        if "present_factor" in row:
            vias = {label: count for label, count in row["overused_vias"].items() if count}
            extra = f", present factor {row['present_factor']:g}, states touched {row['states_touched']}, over-used vias {vias}"
        print(f"[{now - marks['start']:8.1f}s] iteration {row['iteration']} ({now - marks['last']:.1f} s): rerouted {row['rerouted']}, "
              f"over-used cells {row['overused_cells']}, conflicting {len(row['conflicting'])}{extra}", flush=True)
        marks["last"] = now

    prof = cProfile.Profile() if args.profile else None
    if prof is not None:
        prof.enable()
    result = maze.route_board(ir, library, params, progress=progress, plane_nets=plane_nets, stack=stack, corner_style=args.corners)
    if prof is not None:
        prof.disable()
    now = time.perf_counter()
    s = result.stats
    kinds = Counter("through" if v.kind is None else f"{v.kind} {v.layers[0]}-{v.layers[1]}" for v in result.vias)
    end_game = s.get("stack", {}).get("search", {}).get("end_game")
    moved = "" if end_game is None else f"end game moved {[(row['net'], row['how'], row['ripped_up']) for row in end_game['moved']]}, "
    print(f"[{now - marks['start']:8.1f}s] done ({now - marks['last']:.1f} s after the last iteration: end game / drop / recovery / emission): "
          f"version {result.version}, iterations {s.get('iterations')}, legal {s.get('legal')}, routed {s.get('routed_nets')}, "
          f"unrouted {len(result.unrouted)}, {moved}dropped {len(s.get('dropped', []))}, recovered {len(s.get('recovered', []))}, "
          f"tracks {len(result.tracks)}, vias {dict(kinds)}", flush=True)
    if "corners" in s:
        c = s["corners"]
        arcs = f"arcs {c['arcs']}, " if "arcs" in c else ""  # routing.corners 0.2 (arc style)
        print(f"corners ({c['tool']} {c['version']} {c['style']}): found {c['found']}, {arcs}mitred {c['mitred']}, left {c['left']} "
              f"{c['left_reasons']}, skipped {c['skipped']} {c['skipped_reasons']}")
        if "sidesteps" in c:  # routing.corners 0.2: the straightening before the rounding
            print(f"sidesteps: {c['sidesteps']}, straightened {c['straightened']}, left {c['sidesteps_left']} {c['sidestep_reasons']}")
    if seconds:
        print(f"seconds per iteration: {', '.join(f'{x:.1f}' for x in seconds)} (mean {sum(seconds) / len(seconds):.1f})")
    peak = peak_memory_mb()
    print(f"peak working set: {'not reported by the OS' if peak is None else f'{peak:.0f} MB'}")
    if prof is not None:
        st = pstats.Stats(prof)
        calls = {func[2]: st.stats[func][1] for func in st.stats if func[2] in ("<built-in method _heapq.heappop>", "<built-in method _heapq.heappush>")}
        print(f"heappop calls {calls.get('<built-in method _heapq.heappop>', 0)}, heappush calls {calls.get('<built-in method _heapq.heappush>', 0)}")
        st.sort_stats("tottime").print_stats(args.profile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
