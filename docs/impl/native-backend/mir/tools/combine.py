#!/usr/bin/env python3
"""One per-phase table from flat and call-chain samples of the same runs.

  combine.py --total INSTR --flat 'DIR/*.flat.txt' --dwarf 'DIR/*.data.txt'

Why two inputs. A dwarf call chain is empty for a sample in generated code and
also when unwinding fails, and about half of the empty ones on this build are
failures, so dwarf alone overstates generated code. A flat sample always has its
IP, so it gives the reliable split of the run into generated code, libc and the
loader, and Oximond or Cranelift code. The call chains then split the Oximond
and Cranelift code into phases (leaf-first rules in classify.py, callers
charged for allocator and library frames). Samples whose chain is empty or
whose leaf is in libc are left out of that second split.

Output: share of the run and instructions per run for each category, then each
phase's share of the code samples scaled by the code share.
"""
import glob
import re
import sys
from collections import defaultdict

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from classify import phase_of, is_alloc
from flat import LINE as FLAT_LINE, category
import phases as ph


def main():
    args = sys.argv[1:]
    total = float(args[args.index("--total") + 1])
    flat_files = sorted(glob.glob(args[args.index("--flat") + 1]))
    dwarf_files = sorted(glob.glob(args[args.index("--dwarf") + 1]))
    threads = "--by-thread" in args
    cat = defaultdict(int)
    weight = 0
    for path in flat_files:
        for line in open(path, errors="replace"):
            m = FLAT_LINE.match(line)
            if not m:
                continue
            c = category(m[6], m[7])
            if c.startswith(("cl.", "ox.", "vm.")) or c == "other":
                c = "code"
            elif c == "libc.malloc_free?":
                c = "libc.malloc_free"
            key = (("vm" if m[1] == m[2] else "compile") + "/" if threads else "") + c
            cat[key] += int(m[4])
            weight += int(m[4])
    print(f"flat samples: {len(flat_files)} runs, weight {weight}")
    print(f"\n{'category':<28}{'share of run':>13}{'instr/run':>13}")
    for c, n in sorted(cat.items(), key=lambda kv: -kv[1]):
        print(f"{c:<28}{100*n/weight:>12.2f}%{total*n/weight:>13.0f}")
    code = sum(n for c, n in cat.items() if c.endswith("code") and "generated" not in c)
    phase = defaultdict(int)
    pw = 0
    for path in dwarf_files:
        for s in ph.samples(path):
            frames = s["frames"]
            if not frames:
                continue
            leaf = frames[0]
            if "libc" in s["dsos"][0] or "ld-linux" in s["dsos"][0]:
                continue
            p = phase_of(frames)
            phase[(("vm" if s["tid"] == s["pid"] else "compile") + "/" if threads else "") + p] += s["period"]
            pw += s["period"]
    print(f"\ncode samples with a call chain: weight {pw}")
    print(f"{'phase':<28}{'% of code':>10}{'% of run':>10}{'instr/run':>13}")
    for p, n in sorted(phase.items(), key=lambda kv: -kv[1]):
        share = n / pw
        print(f"{p:<28}{100*share:>9.1f}%{100*share*code/weight:>9.2f}%{total*share*code/weight:>13.0f}")


if __name__ == "__main__":
    main()
