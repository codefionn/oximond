#!/usr/bin/env python3
"""Allocations per phase and per trace IR instruction.

  allocs.py ALLOC_PHASES_JSON SYNC_LOG

ALLOC_PHASES_JSON is written by a build with --features alloc-phases under
OXIMOND_ALLOC_PHASES=FILE. SYNC_LOG is stderr of the instrumented build for the
same run (OXIMOND_JIT_SYNC=1 OXIMOND_JIT_TRACE=1), which gives the trace IR and
CLIF instruction counts of the compiles. Allocations are counted once for each
alloc, alloc_zeroed and realloc call.
"""
import json
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from events import parse, compiles, summarize


def main():
    data = json.load(open(sys.argv[1]))
    cs = [summarize(c) for c in compiles(parse(sys.argv[2]))]
    n_pre = sum(c.get("n_pre", 0) for c in cs)
    n_ir = sum(c.get("n_ir", 0) for c in cs)
    clif = sum(c.get("clif", 0) for c in cs)
    ph = data["phases"]
    total = data["totals"]["alloc_count"]
    print(f"total {total} allocations, {data['totals']['alloc_bytes']} bytes; "
          f"compiles {len(cs)}, trace IR before optimizer {n_pre}, after {n_ir}, CLIF {clif}")
    print(f"{'phase':<14}{'allocs':>8}{'share':>8}{'bytes':>11}{'per IR (post)':>15}{'per IR (pre)':>14}{'per CLIF':>10}")
    for name in ("unclassified", "recording", "optimization", "codegen", "installation", "execution", "exit_recovery", "report"):
        p = ph[name]
        n = p["alloc_count"]
        print(f"{name:<14}{n:>8}{100*n/total:>7.1f}%{p['alloc_bytes']:>11}{n/n_ir:>15.1f}{n/n_pre:>14.1f}{n/clif:>10.2f}")
    comp = ph["optimization"]["alloc_count"] + ph["codegen"]["alloc_count"]
    inst = ph["installation"]["alloc_count"]
    print(f"compile (optimization + codegen): {comp}: {comp/n_ir:.1f} per IR instruction after the optimizer, "
          f"{comp/n_pre:.1f} before it, {comp/clif:.2f} per CLIF instruction, {comp/len(cs):.0f} per submission")
    print(f"installation: {inst}: {inst/n_ir:.1f} per IR instruction after the optimizer, {inst/len(cs):.0f} per submission")


if __name__ == "__main__":
    main()
