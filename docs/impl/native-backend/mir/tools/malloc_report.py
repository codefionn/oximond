#!/usr/bin/env python3
"""Summarise a mallocprof.so dump.

  malloc_report.py DUMP [--top N] [--json OUT]

Prints allocator calls (malloc, calloc, realloc, memalign) and frees by phase
and thread, and the top calling functions. A caller is the first frame outside
the allocator, std/alloc/core generics and the Rust allocator shims, so a
`Vec::push` is charged to the function that owns the Vec.
"""
import json
import re
import subprocess
import sys
from collections import defaultdict

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from classify import phase_of, ALLOC_LEAVES



def load(path):
    base = None
    binary = None
    rows = []
    for line in open(path):
        parts = line.split()
        if parts[0] == "base":
            base, binary = int(parts[1], 16), parts[2]
            continue
        kind, thread, calls, nbytes = parts[0], int(parts[1]), int(parts[2]), int(parts[3])
        rows.append((kind, thread, calls, nbytes, [int(x, 16) for x in parts[4:]]))
    return base, binary, rows


def main():
    path = sys.argv[1]
    top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 25
    base, binary, rows = load(path)
    addrs = [a for r in rows for a in r[4]]
    # addr2line per address, one call per chunk; keep order by asking one at a time per chunk
    names = {}
    uniq = sorted(set(addrs))
    chunk = 300
    for i in range(0, len(uniq), chunk):
        part = uniq[i:i + chunk]
        args = [f"0x{a - base - 1:x}" for a in part]
        out = subprocess.run(["addr2line", "-a", "-f", "-C", "-i", "-e", binary, *args],
                             capture_output=True, text=True).stdout.splitlines()
        idx = -1
        j = 0
        func = None
        for line in out:
            if re.fullmatch(r"0x[0-9a-f]+", line):
                idx += 1
                names[part[idx]] = []
                j = 0
            elif j % 2 == 0:
                func = line
                j += 1
            else:
                where = re.sub(r"\s*\(discriminator \d+\)", "", line)
                names[part[idx]].append((func, where))
                j += 1
    by_phase = defaultdict(lambda: [0, 0, 0])  # (thread,phase) -> mallocs, bytes, frees
    by_caller = defaultdict(lambda: [0, 0, 0])
    by_file = defaultdict(lambda: [0, 0])
    totals = defaultdict(lambda: [0, 0])
    for kind, thread, calls, nbytes, stack in rows:
        pairs = []
        for a in stack:
            pairs.extend(names.get(a, [("??", "??")]))
        frames = pairs
        thr = "vm" if thread == 0 else "worker"
        phase = phase_of(frames)
        caller = next((f"{f} ({w.split('/src/')[-1].split('/')[-1]})" for f, w in pairs
                       if not re.search(r"/rustc/|/library/|\.rustup|hashbrown|smallvec|rustc-hash", w)
                       and not ALLOC_LEAVES.match(f) and not f.startswith("__rustc")), "??")
        key = (thr, phase)
        crate_file = next((w for f, w in pairs
                           if not re.search(r"/rustc/|/library/|\.rustup|hashbrown|smallvec|rustc-hash", w)
                           and not ALLOC_LEAVES.match(f) and not f.startswith("__rustc")), "??:0")
        crate_file = re.sub(r":\d+$", "", crate_file)
        crate_file = re.sub(r"^.*/(registry/src/[^/]+/|crates/oximond/src/)", "", crate_file)
        if kind == "m":
            by_file[crate_file][0] += calls
            by_file[crate_file][1] += nbytes
        if kind == "m":
            by_phase[key][0] += calls
            by_phase[key][1] += nbytes
            by_caller[(thr, phase, caller)][0] += calls
            by_caller[(thr, phase, caller)][1] += nbytes
            totals[thr][0] += calls
        else:
            by_phase[key][2] += calls
            by_caller[(thr, phase, caller)][2] += calls
            totals[thr][1] += calls
    print(f"binary {binary}")
    for thr in ("vm", "worker"):
        if thr not in totals:
            continue
        print(f"\n## thread {thr}: {totals[thr][0]} allocations, {totals[thr][1]} frees")
        print(f"{'phase':<22}{'allocs':>9}{'bytes':>12}{'frees':>9}")
        for (t, phase), (m, b, f) in sorted(by_phase.items(), key=lambda kv: -kv[1][0]):
            if t == thr:
                print(f"{phase:<22}{m:>9}{b:>12}{f:>9}")
    print("\n## allocations by source file of the first non-library frame")
    for f, (m, b) in sorted(by_file.items(), key=lambda kv: -kv[1][0])[:20]:
        print(f"{m:>7} {b:>10}  {f}")
    print(f"\n## top callers (allocs, bytes)")
    for (thr, phase, caller), (m, b, f) in sorted(by_caller.items(), key=lambda kv: -kv[1][0])[:top]:
        print(f"{m:>7} {b:>10}  {thr:<6} {phase:<18} {caller[:110]}")
    if "--json" in sys.argv:
        out = sys.argv[sys.argv.index("--json") + 1]
        json.dump({"phase": {f"{t}/{p}": v for (t, p), v in by_phase.items()},
                   "callers": {f"{t}/{p}/{c}": v for (t, p, c), v in by_caller.items()}}, open(out, "w"))


if __name__ == "__main__":
    main()
