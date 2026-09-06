#!/usr/bin/env python3
"""Where the cycles of a stage 1 kernel go, instruction by instruction, for MIR
code and Cranelift code alike.

    cargo build --release -p oximond-mir --example bench
    scripts/heavy.sh -x python3 benchmarks/mir_profile.py KERNEL [--fuel N] [--cpu N]

The code of both backends lives in an anonymous mapping, which `perf annotate`
cannot read. `bench mir-run` and `bench clif-run` write their machine code to
the file named by `BENCH_DUMP` and print the address it ran at. This script
samples `cycles:u` of the run, subtracts the address, and prints every
instruction of the dump (objdump, Intel syntax) with the samples that landed on
it, as a share of the samples in the kernel's code. A line marks the loop top
(the target of the last backward jump), so the loop body is easy to read.

Runs are pinned with taskset to a CPU whose SMT sibling is idle and use
`env -i`, as the other measurement scripts do.
"""
import argparse
import collections
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))
import mir_backend  # noqa: E402


def profile(bench, cpu, mode, kernel, fuel):
    with tempfile.TemporaryDirectory(prefix="mir-profile-") as tmp:
        dump = Path(tmp) / "code.bin"
        data = Path(tmp) / "perf.data"
        run = subprocess.run(
            ["perf", "record", "-q", "-e", "cycles:u", "-c", "2000", "-o", str(data),
             "taskset", "-c", str(cpu), "env", "-i", f"BENCH_DUMP={dump}",
             bench, mode, kernel, str(fuel)],
            capture_output=True, text=True,
        )
        info = dict(kv.split("=") for kv in run.stdout.split() if "=" in kv)
        base = int(info["code_at"], 16)
        length = int(info["code_len"])
        script = subprocess.run(
            ["perf", "script", "-i", str(data), "-F", "ip"],
            capture_output=True, text=True,
        ).stdout
        hist = collections.Counter()
        total = 0
        for line in script.split():
            ip = int(line, 16)
            total += 1
            if base <= ip < base + length:
                hist[ip - base] += 1
        asm = subprocess.run(
            ["objdump", "-D", "-b", "binary", "-mi386:x86-64", "-M", "intel", str(dump)],
            capture_output=True, text=True,
        ).stdout
        return hist, total, asm, length


def report(name, hist, total, asm):
    inside = sum(hist.values())
    print(f"## {name}: {inside} of {total} samples in the kernel code")
    rows = []
    for line in asm.splitlines():
        m = re.match(r"\s*([0-9a-f]+):\s+((?:[0-9a-f]{2} )+)\s*(.*)", line)
        if m:
            rows.append((int(m.group(1), 16), m.group(3).strip()))
    # The loop top: the target of the last backward jump.
    top = None
    for at, text in rows:
        m = re.match(r"j\w+\s+(?:0x)?([0-9a-f]+)", text)
        if m and int(m.group(1), 16) < at:
            top = int(m.group(1), 16)
    for at, text in rows:
        count = hist.get(at, 0)
        share = 100.0 * count / inside if inside else 0.0
        mark = "  <- loop top" if at == top else ""
        print(f"{share:5.1f}% {at:5x}: {text}{mark}")
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("kernel")
    p.add_argument("--fuel", type=int, default=2_000_000)
    p.add_argument("--cpu", type=int)
    p.add_argument("--bin", default=str(ROOT / "target/release/examples/bench"))
    args = p.parse_args()
    cpu = args.cpu if args.cpu is not None else mir_backend.idle_pair()
    print(f"load average: {open('/proc/loadavg').read().split()[:3]}, cpu {cpu}")
    for mode, name in (("clif-run", "Cranelift"), ("mir-run", "MIR")):
        hist, total, asm, _ = profile(args.bin, cpu, mode, args.kernel, args.fuel)
        report(f"{name} {args.kernel}", hist, total, asm)


if __name__ == "__main__":
    main()
