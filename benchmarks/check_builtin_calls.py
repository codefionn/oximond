#!/usr/bin/env python3
"""Check retired instructions of library calls inside traces.

Each benchmark runs in the interpreter and in tracing mode (compile on the
main thread, so the compile counts) under Lua 5.1 and Lua 5.4. Tracing must
retire fewer instructions than the interpreter, and `stdlib/native_calls`
at most half as many.
"""

import argparse
import csv
import os
from pathlib import Path
import statistics
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parent.parent

# Benchmark, and the most tracing may retire as a share of the interpreter.
BENCHMARKS = [
    ("stdlib/native_calls", 0.5),
    ("string/table_concat", 1.0),
    ("string/naive_concat", 1.0),
    ("string/format", 1.0),
    ("string/plain_find", 1.0),
    ("calls/builtin", 1.0),
]


def measure(binary: Path, cpu: int, compat: str, mode: str, name: str) -> int:
    env = dict(os.environ, OXIMOND_JIT_SYNC="1")
    with tempfile.NamedTemporaryFile(mode="r+") as counters:
        subprocess.run(
            [
                "perf", "stat", "-x", ";", "-o", counters.name,
                "-e", "instructions:u", "--", "taskset", "-c", str(cpu),
                str(binary), f"--compat={compat}", f"--execution={mode}", "-E",
                str(ROOT / "benchmarks/lua" / f"{name}.lua"),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
            env=env,
        )
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
    failures = []
    for name, limit in BENCHMARKS:
        for compat in ("lua51", "lua54"):
            medians = {}
            for mode in ("interpreter", "tracing"):
                counts = [
                    measure(args.binary.resolve(), args.cpu, compat, mode, name)
                    for _ in range(args.repeat)
                ]
                medians[mode] = statistics.median(counts)
            ratio = medians["tracing"] / medians["interpreter"]
            print(
                f"{name:22s} {compat}: interpreter {medians['interpreter']:>14,.0f}"
                f"  tracing {medians['tracing']:>14,.0f}  ratio {ratio:.3f} (limit {limit})"
            )
            if ratio > limit:
                failures.append(f"{name} {compat}: ratio {ratio:.3f} over {limit}")
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
