#!/usr/bin/env python3
"""Per-phase instruction table from `perf script` output with call chains.

  perf script -i perf.data -F comm,pid,tid,time,period,ip,sym,dso > samples.txt
  phases.py [--burn MS] [--total INSTR] [--within SUBSTR] samples.txt [samples.txt...]

Many short runs are merged (see collect.sh). The kernel throttles perf above
about 2000 samples a second (see sample.sh), so runs use a long period and
are merged. --burn drops the first MS milliseconds of each run (default 0). Shares are sample-weighted. With --total the table also gives
absolute instructions (share times the exact count of one run). With --within only samples
whose stack has a frame containing SUBSTR count, so the shares are those of
that call (for example `define_function` for Cranelift's compile).

Each sample carries the period (instructions:u since the last sample). The
sample goes to the phase of the first matching frame from the leaf (see
classify.py), and to the thread class "vm" (tid == pid) or "compile" (any
other tid). Prints absolute instructions and shares per thread class.
"""
import re
import sys
from collections import defaultdict

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from classify import phase_of, is_alloc

HEAD = re.compile(r"^\S.*?\s+(\d+)/(\d+)\s+([\d.]+):\s+(\d+)\s")
FRAME = re.compile(r"^\s+([0-9a-f]+)\s+(.*?)\s+\((.*)\)\s*$")


def samples(path):
    cur = None
    for line in open(path, errors="replace"):
        m = HEAD.match(line)
        if m:
            if cur:
                yield cur
            cur = {"pid": int(m[1]), "tid": int(m[2]), "time": float(m[3]),
                   "period": int(m[4]), "frames": [], "dsos": []}
            continue
        m = FRAME.match(line)
        if m and cur is not None:
            sym = re.sub(r"\+0x[0-9a-f]+$", "", m[2])
            cur["frames"].append(sym)
            cur["dsos"].append(m[3])
    if cur:
        yield cur


def main():
    args = sys.argv[1:]
    burn, total_exact, within = 0.0, None, None
    paths = []
    while args:
        a = args.pop(0)
        if a == "--burn":
            burn = float(args.pop(0))
        elif a == "--total":
            total_exact = float(args.pop(0))
        elif a == "--within":
            within = args.pop(0)
        else:
            paths.append(a)
    table = defaultdict(lambda: defaultdict(int))
    alloc = defaultdict(lambda: defaultdict(int))
    total = defaultdict(int)
    for path in paths:
        t0 = None
        for s in samples(path):
            if t0 is None:
                t0 = s["time"]
            if s["time"] - t0 < burn / 1000:
                continue
            thread = "vm" if s["tid"] == s["pid"] else "compile"
            frames = s["frames"]
            if within and not any(within in f for f in frames):
                continue
            # A sample with no frames at all has its IP in anonymous memory:
            # perf prints no line for it. That is Cranelift's generated code.
            if not frames:
                phase = "generated_code"
            else:
                phase = phase_of(frames)
            table[thread][phase] += s["period"]
            total[thread] += s["period"]
            if is_alloc(frames):
                alloc[thread][phase] += s["period"]
    grand = sum(total.values())
    print(f"samples weight {grand} over {len(paths)} runs")
    for thread in ("vm", "compile"):
        if not total[thread]:
            continue
        print(f"\n## thread {thread}: {100*total[thread]/grand:.1f}% of all samples")
        print(f"{'phase':<22}{'share of all':>14}{'of which alloc':>16}" + (f"{'instr/run':>14}" if total_exact else ""))
        for phase, n in sorted(table[thread].items(), key=lambda kv: -kv[1]):
            line = f"{phase:<22}{100*n/grand:>13.2f}%{100*alloc[thread][phase]/grand:>15.2f}%"
            if total_exact:
                line += f"{total_exact*n/grand:>14.0f}"
            print(line)


if __name__ == "__main__":
    main()
