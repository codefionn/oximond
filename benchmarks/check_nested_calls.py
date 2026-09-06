#!/usr/bin/env python3
"""Check retired instructions for the Lua 5.1 outer-call loop regression."""

import argparse
import csv
import os
from pathlib import Path
import statistics
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parent.parent


def measure(binary: Path, cpu: int, mode: str) -> int:
    with tempfile.NamedTemporaryFile(mode="r+") as counters:
        result = subprocess.run(
            [
                "perf", "stat", "-x", ";", "-o", counters.name,
                "-e", "instructions:u", "--", "taskset", "-c", str(cpu),
                str(binary), "--compat=lua51", f"--execution={mode}", "-E",
                str(ROOT / "benchmarks/lua/control/nested_calls.lua"),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        if result.stdout.strip() != "119999880":
            raise RuntimeError(f"{mode} returned unexpected output: {result.stdout!r}")
        for row in csv.reader(counters, delimiter=";"):
            if len(row) >= 3 and row[2] == "instructions:u":
                return int(row[0])
    raise RuntimeError("perf did not report retired user instructions")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "target/release/oximond")
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    if not os.environ.get("OXIMOND_HEAVY_ACTIVE"):
        parser.error("run through scripts/heavy.sh -x")
    print("load average:", ", ".join(f"{load:.2f}" for load in os.getloadavg()))
    samples = {mode: [] for mode in ("interpreter", "tracing")}
    for iteration in range(args.repeat):
        modes = list(samples)
        if iteration % 2:
            modes.reverse()
        for mode in modes:
            samples[mode].append(measure(args.binary.resolve(), args.cpu, mode))
    medians = {mode: statistics.median(counts) for mode, counts in samples.items()}
    for mode, count in medians.items():
        print(f"{mode}: {count:,.0f} instructions")
    ratio = medians["tracing"] / medians["interpreter"]
    print(f"tracing / interpreter: {ratio:.3f}")
    if ratio >= 1:
        raise SystemExit("tracing retires more instructions than the interpreter")


if __name__ == "__main__":
    main()
