#!/usr/bin/env python3
"""Compare CLI file startup and optional delayed-reader compilation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import tempfile
import time


def checksum(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for label in ("original", "sequential", "pipeline"):
        parser.add_argument("--" + label, type=Path, required=True)
    parser.add_argument("--cpus", required=True, help="Idle physical CPU IDs, without their SMT siblings")
    parser.add_argument("--repeat", type=int, default=15)
    parser.add_argument("--slow-reader", type=Path, help="Compiler load_pipeline release example")
    parser.add_argument("--json", type=Path, required=True)
    options = parser.parse_args()
    if options.repeat < 3:
        parser.error("use at least three repetitions")
    binaries = {label: getattr(options, label).resolve() for label in ("original", "sequential", "pipeline")}
    environment = dict(os.environ)
    for name in ("LUA_INIT", "LUA_INIT_5_5", "LUA_PATH", "LUA_PATH_5_5", "OXIMOND_COUNTERS", "OXIMOND_INTERP_PROFILE"):
        environment.pop(name, None)
    report = {"cpus": options.cpus, "load_average": os.getloadavg(), "repeat": options.repeat,
              "binaries": {label: {"path": str(path), "sha256": checksum(path)} for label, path in binaries.items()}, "rows": []}
    randomizer = random.Random(20261008)
    with tempfile.TemporaryDirectory(prefix="file-loading.") as directory:
        root = Path(directory)
        cases = {}
        for label, count in (("small", 100), ("medium", 10000), ("large", 100000)):
            cases[label] = ("local x=0;\n" + "x=x+1;\n" * count + "print(x)\n", f"{count}\n")
        cases["late-closure"] = ("local x=0;\n" + "x=x+1;\n" * 10000 + "local function f() x=x+1 end; print(x)\n", "10000\n")
        count = 4000
        cases["functions"] = ("local functions={}\n" + "".join(
            f"function functions.f{i}(x) return x+{i} end;\n" for i in range(count))
            + f"print(functions.f{count-1}(1))\n", f"{count}\n")
        paths = {}
        for label, (source, _) in cases.items():
            path = root / (label + ".lua")
            path.write_text(source)
            paths[label] = path

        def collect(name: str, commands: dict[str, list[str]], expected: str, size: int) -> None:
            samples = {label: [] for label in commands}
            for _ in range(2):
                for command in commands.values():
                    subprocess.run(command, env=environment, check=True, capture_output=True, timeout=120)
            for _ in range(options.repeat):
                order = list(commands)
                randomizer.shuffle(order)
                for label in order:
                    started = time.perf_counter_ns()
                    result = subprocess.run(commands[label], env=environment, capture_output=True, text=True, timeout=120)
                    elapsed = (time.perf_counter_ns() - started) / 1e6
                    if result.returncode or result.stdout != expected:
                        raise RuntimeError(f"{name}/{label}: {result.returncode}: {result.stdout!r} {result.stderr}")
                    samples[label].append(elapsed)
            row = {"name": name, "bytes": size, "median_ms": {label: statistics.median(values) for label, values in samples.items()},
                   "samples_ms": samples, "instructions": {}}
            for label, command in commands.items():
                counts = []
                log = root / "perf.csv"
                for _ in range(3):
                    subprocess.run(["perf", "stat", "--no-big-num", "-x,", "-e", "instructions:u", "-o", str(log), "--", *command],
                                   env=environment, check=True, capture_output=True, timeout=120)
                    for line in log.read_text().splitlines():
                        fields = line.split(",")
                        if len(fields) > 2 and fields[2] == "instructions:u":
                            counts.append(int(fields[0]))
                if len(counts) != 3:
                    raise RuntimeError("perf did not supply instruction counts")
                row["instructions"][label] = statistics.median(counts)
            report["rows"].append(row)
            print(json.dumps({key: value for key, value in row.items() if key != "samples_ms"}), flush=True)

        for label, path in paths.items():
            commands = {variant: ["taskset", "-c", options.cpus, str(binary), "--compat=lua54", "--execution=interpreter", "-E", str(path)]
                        for variant, binary in binaries.items()}
            collect("cli-" + label, commands, cases[label][1], path.stat().st_size)
        if options.slow_reader:
            path = paths["large"]
            fingerprints = []
            for mode in ("complete", "sequential", "pipeline"):
                result = subprocess.run([str(options.slow_reader), mode, str(path), "0", "fingerprint"],
                                        check=True, capture_output=True, text=True, timeout=120)
                fingerprints.append(result.stdout)
            if len(set(fingerprints)) != 1:
                raise RuntimeError("delayed-reader modes produced different bytecode")
            commands = {mode: ["taskset", "-c", options.cpus, str(options.slow_reader), mode, str(path), "1000"]
                        for mode in ("complete", "sequential", "pipeline")}
            collect("compiler-simulated-1ms-per-8KiB-read", commands, "", path.stat().st_size)
    options.json.parent.mkdir(parents=True, exist_ok=True)
    options.json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
