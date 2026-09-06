#!/usr/bin/env python3
"""Cycles and memory events per loop iteration of the stage 1 kernels, MIR code
against Cranelift code, for the spill-heavy kernels in particular.

    cargo build --release -p oximond-mir --example bench
    scripts/heavy.sh -x python3 benchmarks/mir_cycles.py OUT.json [--repeats 7]

Each number is the difference of two runs of `bench mir-run K FUEL` (100,000 and
500,000 iterations) over the 400,000 extra iterations, so the compile and the
start-up cancel. Runs are pinned with taskset to a CPU whose SMT sibling is
idle, under `env -i`, and repeated; the median and the range are reported.
Events (store-to-load forwards, stores, loads, dispatch stalls) are the AMD
Zen 4 names of `perf list`.
"""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))
import mir_backend  # noqa: E402

KERNELS = ["pressure", "guard_restore", "nbody", "swap", "int_side"]
EVENT_KERNELS = ["pressure", "guard_restore"]
EVENT_GROUPS = [
    ["ls_stlf", "ls_bad_status2.stli_other", "ls_dispatch.store_dispatch", "ls_dispatch.ld_dispatch"],
    [
        "de_dis_dispatch_token_stalls1.store_queue_rsrc_stall",
        "de_dis_dispatch_token_stalls1.load_queue_rsrc_stall",
        "de_no_dispatch_per_slot.backend_stalls",
        "op_cache_hit_miss.op_cache_miss",
    ],
    ["ex_ret_brn_misp", "ex_ret_brn_tkn", "de_src_op_disp.op_cache"],
]
SMALL, LARGE = 100_000, 500_000


def perf(bench, cpu, mode, kernel, fuel, events):
    run = subprocess.run(
        ["perf", "stat", "-x,", "-e", ",".join(e + ":u" for e in events),
         "taskset", "-c", str(cpu), "env", "-i", bench, mode, kernel, str(fuel)],
        capture_output=True, text=True,
    )
    values = {}
    for line in run.stderr.splitlines():
        fields = line.split(",")
        if len(fields) > 2 and fields[0].isdigit():
            values[fields[2].split(":")[0]] = int(fields[0])
    return values, run.stdout.strip()


def per_iteration(bench, cpu, mode, kernel, events):
    a, _ = perf(bench, cpu, mode, kernel, SMALL, events)
    b, out = perf(bench, cpu, mode, kernel, LARGE, events)
    return {e: (b[e] - a[e]) / (LARGE - SMALL) for e in b}, out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out")
    parser.add_argument("--bin", default=str(ROOT / "target" / "release" / "examples" / "bench"))
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    cpu = mir_backend.idle_pair()
    result = {"cpu": cpu, "load": open("/proc/loadavg").read().split()[:3], "rows": [], "events": []}
    for kernel in KERNELS:
        for mode in ("mir-run", "clif-run"):
            runs = []
            for _ in range(args.repeats):
                v, out = per_iteration(args.bin, cpu, mode, kernel, ["cycles", "instructions"])
                runs.append({"cycles": v["cycles"], "instr": v["instructions"], "out": out})
            cycles = [r["cycles"] for r in runs]
            result["rows"].append({
                "kernel": kernel, "mode": mode, "runs": runs,
                "median_cycles": statistics.median(cycles), "min": min(cycles), "max": max(cycles),
            })
    for kernel in EVENT_KERNELS:
        for mode in ("mir-run", "clif-run"):
            merged = {}
            for group in EVENT_GROUPS:
                v, _ = per_iteration(args.bin, cpu, mode, kernel, group)
                merged.update(v)
            result["events"].append({"kernel": kernel, "mode": mode, "per_iter": merged})
    json.dump(result, open(args.out, "w"), indent=1)
    print("load", result["load"], "cpu", cpu)
    for row in result["rows"]:
        print(f"{row['kernel']:14} {row['mode']:9} {row['median_cycles']:7.1f} cycles/iter ({row['min']:.1f} to {row['max']:.1f})")


if __name__ == "__main__":
    main()
