#!/usr/bin/env python3
"""Per-phase instruction table of one run from the exact event counters.

  report.py LOG TOTAL_INSTR [--bg]

LOG is stderr of an instrumented run with OXIMOND_JIT_TRACE=1 (inline compile,
OXIMOND_JIT_SYNC=1, unless --bg). TOTAL_INSTR is the benchmark's instruction
count from benchmarks/icount.sh. Prints, for all compiles of the run together,
the instructions of each stage in the compiling thread, in absolute terms and
as a share of the compile and of the run.
"""
import sys
from collections import defaultdict

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from events import parse, compiles, summarize, kv

GROUPS = [
    ("verify (compile_with_plan entry)", ["tree:start -> tree:verified"]),
    ("verify (optimizer entry)", ["opt:verify_in:b -> opt:verify_in:e"]),
    ("Draft::new + finish (normalise)", ["opt:verify_in:e -> opt:normalised"]),
    ("verify (optimizer exit)", ["opt:rounds_done -> opt:verify_out:e"]),
    ("snapshot/guard renumber and member setup", ["tree:member_opt:e", "tree:ir", "side:opt:e"]),
    ("plan_exits (exit plan, snapshot descriptors)", ["tree:opt_done -> tree:exits_planned", "side:opt_done -> side:exits_planned"]),
    ("Oximond lowering: start to JITBuilder ready", ["lw:start -> lw:builder_ready"]),
    ("JITModule::new", ["lw:builder_ready -> lw:module_new"]),
    ("declare functions, signatures", ["lw:module_new -> lw:declared"]),
    ("CLIF build (Lowerer + FunctionBuilder)", ["lw:declared -> lw:clif_built"]),
    ("Cranelift define_function", ["lw:define:b -> lw:define:e"]),
    ("finalize_definitions + module memory", ["lw:define:e -> lw:finalized"]),
]


def main():
    log, total = sys.argv[1], int(sys.argv[2])
    evs = parse(log)
    cs = [summarize(c) for c in compiles(evs)]
    ctotal = sum(c["total"] for c in cs)
    stage = defaultdict(int)
    passes = defaultdict(int)
    for c in cs:
        for k, v in c["stages"].items():
            stage[k] += v
        for k, v in c["passes"].items():
            passes[k] += v
    rows = []
    used = set()
    for label, keys in GROUPS:
        v = 0
        for k, n in stage.items():
            if any(k == key or k.startswith(key) for key in keys):
                v += n
                used.add(k)
        rows.append((label, v))
    pass_total = sum(passes.values())
    for k in stage:
        if k.startswith("pass:") or "member_opt" in k:
            used.add(k)
    rows.append(("optimizer passes (all rounds, all members)", pass_total))
    rest = ctotal - sum(v for _, v in rows)
    rows.append(("other glue inside the compile", rest))
    print(f"compiles: {len(cs)} ({sum(1 for c in cs if not c['side'])} trees, {sum(1 for c in cs if c['side'])} sides), "
          f"compile instructions {ctotal} = {100*ctotal/total:.1f}% of the run's {total}")
    print(f"IR instructions: before optimiser {sum(c.get('n_pre',0) for c in cs)}, after {sum(c.get('n_ir',0) for c in cs)}; "
          f"CLIF instructions {sum(c.get('clif',0) for c in cs)}")
    print(f"\n{'stage':<52}{'instr':>11}{'% compile':>11}{'% run':>8}")
    for label, v in rows:
        print(f"{label:<52}{v:>11}{100*v/ctotal:>10.1f}%{100*v/total:>7.2f}%")
    print(f"\n{'optimizer pass':<52}{'instr':>11}{'% compile':>11}{'% run':>8}")
    for k, v in sorted(passes.items(), key=lambda kv: -kv[1]):
        print(f"{k:<52}{v:>11}{100*v/ctotal:>10.1f}%{100*v/total:>7.2f}%")
    # VM-side: install
    inst = 0
    stack = {}
    for e in evs:
        if e["name"] == "complete:b":
            stack[e["tid"]] = e["ic"]
        elif e["name"] == "complete:e":
            inst += e["ic"] - stack.pop(e["tid"], e["ic"])
    print(f"\ninstall (complete_compile, VM thread): {inst} = {100*inst/total:.2f}% of the run")


if __name__ == "__main__":
    main()
