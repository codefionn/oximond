#!/usr/bin/env python3
"""Compare store-to-load interlocks and cycles between two Oximond binaries.

A load that cannot be satisfied from an older store still in the store buffer
(a 16-byte load of a register written as a tag byte and a payload word, a
four-byte load of the upper half of an eight-byte store, a spill reloaded a
few instructions after it was stored) waits until the store leaves the buffer.
On Zen 4 `ls_bad_status2.stli_other` counts these. The tool runs the same
benchmark under `perf stat` with BEFORE and AFTER in turn, pinned to one CPU,
and reports the minimum of each count over --repeat runs:

    scripts/heavy.sh -x python3 benchmarks/stli.py --cpu 14 \\
        stage2-oximond target/release/oximond calls/recursive_fib calls/method

Without benchmark names it runs calls/recursive_fib, calls/method and
calls/closure. `--require-ratio R` exits with status 1 unless every
benchmark's interlock count dropped by a factor of at least R. The CPU should
have an idle SMT sibling, and the cycle counts shift with the machine's load,
so compare BEFORE and AFTER from one invocation only.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
EVENTS = ("ls_bad_status2.stli_other:u", "cycles:u", "instructions:u")
DEFAULT_BENCHMARKS = ("calls/recursive_fib", "calls/method", "calls/closure")


def parse_stat(text: str) -> dict[str, int]:
    """Counts by event name from `perf stat -x,` output."""
    counts: dict[str, int] = {}
    for line in text.splitlines():
        fields = line.split(",")
        if len(fields) < 3 or not fields[0].strip().isdigit():
            continue
        counts[fields[2]] = int(fields[0])
    missing = [event for event in EVENTS if event not in counts]
    if missing:
        raise RuntimeError(f"perf did not count {missing}: {text!r}")
    return counts


def measure(binary: str, benchmark: str, cpu: int, scale: str) -> dict[str, int]:
    program = ROOT / "lua" / f"{benchmark}.lua"
    command = [
        "perf", "stat", "-x,", "-e", ",".join(EVENTS), "--",
        "taskset", "-c", str(cpu), binary, str(program), scale,
    ]
    done = subprocess.run(command, capture_output=True, text=True, check=True)
    return parse_stat(done.stderr)


def best(runs: list[dict[str, int]]) -> dict[str, int]:
    return {event: min(run[event] for run in runs) for event in EVENTS}


def ratio(before: int, after: int) -> float:
    return before / after if after else float("inf")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("benchmarks", nargs="*", default=list(DEFAULT_BENCHMARKS))
    parser.add_argument("--cpu", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--scale", default="3")
    parser.add_argument("--require-ratio", type=float, default=0.0)
    arguments = parser.parse_args(argv)

    failed = False
    print(f"{'benchmark':<24} {'stli before':>12} {'after':>12} {'ratio':>6}"
          f" {'cycles':>8} {'instrs':>8}")
    for benchmark in arguments.benchmarks:
        before, after = [], []
        for _ in range(arguments.repeat):
            before.append(measure(arguments.before, benchmark, arguments.cpu, arguments.scale))
            after.append(measure(arguments.after, benchmark, arguments.cpu, arguments.scale))
        before, after = best(before), best(after)
        stli = ratio(before[EVENTS[0]], after[EVENTS[0]])
        cycles = after["cycles:u"] / before["cycles:u"] - 1
        instructions = after["instructions:u"] / before["instructions:u"] - 1
        print(f"{benchmark:<24} {before[EVENTS[0]]:>12} {after[EVENTS[0]]:>12} {stli:>5.2f}x"
              f" {cycles:>+7.1%} {instructions:>+7.1%}")
        if stli < arguments.require_ratio:
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
