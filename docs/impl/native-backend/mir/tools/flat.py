#!/usr/bin/env python3
"""Category shares from flat (no call chain) perf samples.

  perf script -i X.flat -F pid,tid,time,period,ip,sym,dso > X.flat.txt
  flat.py [--total INSTR] X.flat.txt [...]

Each sample has the exact instruction it stopped at, so generated code (an IP
in anonymous memory, which perf names `/tmp/perf-PID.map`), libc, the loader and
Oximond code are told apart without unwinding. Inside Oximond code the leaf
function is classified with classify.py. Allocator and memcpy samples are
reported on their own: their caller is not known here (see phases.py and
malloc_report.py for callers).
"""
import re
import sys
from collections import defaultdict

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from classify import phase_of

LINE = re.compile(r"^\s*(\d+)/(\d+)\s+([\d.]+):\s+(\d+)\s+([0-9a-f]+)\s+(.*)\s+\((.*)\)\s*$")
ALLOC = re.compile(r"^(_?_?(libc_)?(malloc|free|calloc|realloc|memalign|posix_memalign|aligned_alloc)|_int_.*|cfree|malloc_consolidate|"
                   r"tcache_.*|unlink_chunk|sysmalloc|checked_request2size|arena_for_chunk|alloc_perturb|_mid_memalign)$")
MEM = re.compile(r"^(__)?(memcpy|memmove|memset|memcmp|memchr|strlen|bcmp).*|__.*(memcpy|memmove|memset|memcmp|strlen).*")


def category(sym, dso):
    base = sym.split("+")[0]
    if "perf-" in dso and dso.endswith(".map") or dso in ("[unknown]", "//anon"):
        return "generated_code"
    if "ld-linux" in dso or "ld.so" in dso:
        return "startup.ld.so"
    if "libc" in dso or "libm" in dso:
        if ALLOC.match(base):
            return "libc.malloc_free"
        if MEM.match(base):
            return "libc.mem"
        if base == "[unknown]":
            return "libc.malloc_free?"  # an unnamed libc IP: the allocator's static helpers
        return "libc.other"
    return phase_of([sym])


def main():
    args = sys.argv[1:]
    total = None
    if args and args[0] == "--total":
        args.pop(0)
        total = float(args.pop(0))
    table = defaultdict(lambda: defaultdict(int))
    weight = defaultdict(int)
    for path in args:
        for line in open(path, errors="replace"):
            m = LINE.match(line)
            if not m:
                continue
            thread = "vm" if m[1] == m[2] else "compile"
            cat = category(m[6], m[7])
            table[thread][cat] += int(m[4])
            weight[thread] += int(m[4])
    grand = sum(weight.values())
    print(f"samples weight {grand} over {len(args)} runs")
    for thread in ("vm", "compile"):
        if not weight[thread]:
            continue
        print(f"\n## thread {thread}: {100*weight[thread]/grand:.1f}% of all samples")
        print(f"{'category':<24}{'share of all':>14}" + (f"{'instr/run':>14}" if total else ""))
        for cat, n in sorted(table[thread].items(), key=lambda kv: -kv[1]):
            line = f"{cat:<24}{100*n/grand:>13.2f}%"
            if total:
                line += f"{total*n/grand:>14.0f}"
            print(line)


if __name__ == "__main__":
    main()
