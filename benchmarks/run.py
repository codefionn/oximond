#!/usr/bin/env python3
"""Run deterministic Lua benchmarks across Lua, LuaJIT, and Oximond."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import hashlib
import os
from pathlib import Path
import platform
import random
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Sequence


ROOT = Path(__file__).resolve().parent.parent
SUITE_PATH = ROOT / "benchmarks" / "suite.json"


@dataclass(frozen=True)
class Runtime:
    name: str
    command: tuple[str, ...]
    # Commands that replace `command` for entries of a compat profile, such
    # as Lua 5.4 for lua54 entries.
    compat_commands: tuple[tuple[str, tuple[str, ...]], ...] = ()


@dataclass(frozen=True)
class Benchmark:
    name: str
    path: Path
    # Oximond compatibility profile; the suite runs Lua 5.1 unless an entry
    # names another, such as luajit21 for LuaJIT's 64-bit integer literals.
    compat: str = "lua51"
    # Runtimes that cannot run the program, such as LuaJIT for a Lua 5.4
    # program. The runner leaves them out of that program's row.
    skip: frozenset[str] = frozenset()


def runs(selected_runtimes: Sequence[Runtime], benchmark: Benchmark) -> list[Runtime]:
    return [runtime for runtime in selected_runtimes if runtime.name not in benchmark.skip]


def executable(env_name: str, *fallbacks: str) -> str:
    configured = os.environ.get(env_name)
    if configured:
        return configured
    for fallback in fallbacks:
        found = shutil.which(fallback)
        if found:
            return found
    raise RuntimeError(f"cannot find {' or '.join(fallbacks)}; set {env_name}")


def runtimes(include_luajit_interpreter: bool, compats: set[str]) -> list[Runtime]:
    lua = executable("LUA_BIN", "lua5.1", "lua")
    lua_compat_commands = ()
    if "lua54" in compats:
        lua_compat_commands = (("lua54", (executable("LUA54_BIN", "lua5.4"),)),)
    luajit = executable("LUAJIT_BIN", "luajit")
    oximond = os.environ.get("OXIMOND_BIN", str(ROOT / "target" / "release" / "oximond"))
    if not Path(oximond).is_file() and shutil.which(oximond) is None:
        raise RuntimeError(
            f"cannot find {oximond}; run cargo build --release -p oximond-cli "
            "or set OXIMOND_BIN"
        )
    result = [
        Runtime("lua", (lua,), lua_compat_commands),
        Runtime("luajit", (luajit,)),
    ]
    if include_luajit_interpreter:
        result.append(Runtime("luajit-interpreter", (luajit, "-joff")))
    result.extend(
        [
            Runtime(
                "oximond-interpreter",
                (oximond, "--compat=lua51", "--execution=interpreter", "-E"),
            ),
            Runtime(
                "oximond-tracing",
                (oximond, "--compat=lua51", "--execution=tracing", "-E"),
            ),
        ]
    )
    return result


def load_suite(pattern: str | None) -> list[Benchmark]:
    entries = json.loads(SUITE_PATH.read_text(encoding="utf-8"))
    matcher = re.compile(pattern) if pattern else None
    selected = []
    for entry in entries:
        name = entry["name"]
        if matcher and not matcher.search(name):
            continue
        selected.append(
            Benchmark(
                name,
                ROOT / "benchmarks" / entry["path"],
                entry.get("compat", "lua51"),
                frozenset(entry.get("skip", ())),
            )
        )
    if not selected:
        raise RuntimeError("benchmark filter selected no programs")
    return selected


def clean_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in (
        "LUA_INIT",
        "LUA_INIT_5_1",
        "LUA_INIT_5_2",
        "LUA_INIT_5_3",
        "LUA_INIT_5_4",
        "LUA_INIT_5_5",
        "LUA_PATH",
        "LUA_CPATH",
    ):
        environment.pop(name, None)
    environment["LC_ALL"] = "C"
    return environment


def command_for(runtime: Runtime, benchmark: Benchmark, scale: int, cpu: int | None) -> list[str]:
    compat_command = dict(runtime.compat_commands).get(benchmark.compat)
    runtime_command = list(compat_command) if compat_command else [
        f"--compat={benchmark.compat}" if argument == "--compat=lua51" else argument
        for argument in runtime.command
    ]
    command = [*runtime_command, str(benchmark.path), str(scale)]
    if cpu is not None:
        command = ["taskset", "-c", str(cpu), *command]
    return command


def invoke(
    runtime: Runtime,
    benchmark: Benchmark,
    scale: int,
    cpu: int | None,
    timeout: float,
) -> tuple[str, float]:
    command = command_for(runtime, benchmark, scale, cpu)
    started = time.perf_counter_ns()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=clean_environment(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", "replace").strip()
        rendered = shlex.join(command)
        raise RuntimeError(
            f"{runtime.name} failed {benchmark.name} with status "
            f"{completed.returncode}\ncommand: {rendered}\nstderr: {stderr}"
        )
    stdout = completed.stdout.decode("utf-8", "replace").strip()
    return stdout, elapsed_ms


def verify_checksums(
    selected_runtimes: Sequence[Runtime],
    benchmarks: Sequence[Benchmark],
    scale: int,
    cpu: int | None,
    timeout: float,
) -> None:
    for benchmark in benchmarks:
        checksums = {}
        running = runs(selected_runtimes, benchmark)
        for runtime in running:
            checksums[runtime.name] = invoke(runtime, benchmark, scale, cpu, timeout)[0]
        expected = checksums[running[0].name]
        disagreements = {name: value for name, value in checksums.items() if value != expected}
        if disagreements:
            details = "\n".join(f"  {name}: {value!r}" for name, value in checksums.items())
            raise RuntimeError(f"checksum mismatch for {benchmark.name}:\n{details}")


def measure(
    selected_runtimes: Sequence[Runtime],
    benchmarks: Sequence[Benchmark],
    scale: int,
    warmup: int,
    repeat: int,
    cpu: int | None,
    timeout: float,
) -> dict[str, dict[str, list[float]]]:
    timings = {
        benchmark.name: {runtime.name: [] for runtime in runs(selected_runtimes, benchmark)}
        for benchmark in benchmarks
    }
    randomizer = random.Random(0x0A11CE)
    for benchmark in benchmarks:
        for _ in range(warmup):
            order = runs(selected_runtimes, benchmark)
            randomizer.shuffle(order)
            for runtime in order:
                invoke(runtime, benchmark, scale, cpu, timeout)
        for _ in range(repeat):
            order = runs(selected_runtimes, benchmark)
            randomizer.shuffle(order)
            for runtime in order:
                _, elapsed_ms = invoke(runtime, benchmark, scale, cpu, timeout)
                timings[benchmark.name][runtime.name].append(elapsed_ms)
    return timings


def summarize(
    selected_runtimes: Sequence[Runtime],
    benchmarks: Sequence[Benchmark],
    timings: dict[str, dict[str, list[float]]],
) -> list[dict[str, object]]:
    rows = []
    for benchmark in benchmarks:
        medians = {
            name: statistics.median(samples)
            for name, samples in timings[benchmark.name].items()
        }
        references = [medians[name] for name in ("lua", "luajit") if name in medians]
        reference = min(references) if references else None
        tracing = medians.get("oximond-tracing")
        interpreter = medians.get("oximond-interpreter")
        rows.append(
            {
                "benchmark": benchmark.name,
                "median_ms": medians,
                "oximond_gap": tracing / reference if tracing is not None and reference is not None else None,
                "gap_vs_lua": tracing / medians["lua"]
                if tracing is not None and "lua" in medians
                else None,
                "gap_vs_luajit": tracing / medians["luajit"]
                if tracing is not None and "luajit" in medians
                else None,
                "gap_vs_luajit_interpreter": tracing / medians["luajit-interpreter"]
                if tracing is not None and "luajit-interpreter" in medians
                else None,
                "tracing_speedup": interpreter / tracing
                if interpreter is not None and tracing is not None
                else None,
            }
        )
    return rows


def runtime_version(command: Sequence[str]) -> str:
    completed = subprocess.run(
        [command[0], "-v"],
        cwd=ROOT,
        env=clean_environment(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=10,
        check=False,
    )
    return completed.stdout.decode("utf-8", "replace").strip().splitlines()[0]


def command_metadata(command: Sequence[str]) -> dict[str, object]:
    executable_path = Path(shutil.which(command[0]) or command[0])
    return {
        "command": list(command),
        "version": runtime_version(command),
        "sha256": hashlib.sha256(executable_path.read_bytes()).hexdigest(),
    }


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def git_state() -> dict[str, object]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        ).stdout
    )
    return {"revision": revision or None, "dirty": dirty}


def markdown(selected_runtimes: Sequence[Runtime], rows: Sequence[dict[str, object]]) -> str:
    names = [runtime.name for runtime in selected_runtimes]
    header = ["Benchmark", *names, "Oximond gap", "Tracing speedup"]
    alignment = [":--", *["--:" for _ in names], "--:", "--:"]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(alignment) + " |",
    ]
    for row in rows:
        medians = row["median_ms"]
        values = [f"{medians[name]:.3f} ms" if name in medians else "n/a" for name in names]
        gap = row["oximond_gap"]
        speedup = row["tracing_speedup"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["benchmark"]),
                    *values,
                    f"{gap:.2f}x" if gap is not None else "n/a",
                    f"{speedup:.2f}x" if speedup is not None else "n/a",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filter", help="regular expression matched against benchmark names")
    parser.add_argument("--list", action="store_true", help="list selected benchmarks and exit")
    parser.add_argument("--quick", action="store_true", help="use no warmup and three repeats")
    parser.add_argument("--scale", type=int, default=None, help="positive workload multiplier")
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=None)
    parser.add_argument("--cpu", type=int, help="pin each process with taskset")
    parser.add_argument("--timeout", type=float, default=120.0, help="per-process timeout")
    parser.add_argument("--include-luajit-interpreter", action="store_true")
    parser.add_argument(
        "--runtime",
        action="append",
        choices=(
            "lua",
            "luajit",
            "luajit-interpreter",
            "oximond-interpreter",
            "oximond-tracing",
        ),
        help="runtime to include; repeat to select more than one",
    )
    parser.add_argument("--before", type=Path, help="compare a preserved Oximond binary in the same shuffled run")
    parser.add_argument("--fail-gap", type=float, help="fail if any Oximond gap exceeds this")
    parser.add_argument("--json", type=Path, help="write machine-readable results")
    parser.add_argument("--markdown", type=Path, help="write the result table")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.scale is not None and args.scale < 1:
        raise RuntimeError("--scale must be positive")
    if args.repeat is not None and args.repeat < 1:
        raise RuntimeError("--repeat must be positive")
    if args.warmup is not None and args.warmup < 0:
        raise RuntimeError("--warmup cannot be negative")
    benchmarks = load_suite(args.filter)
    if args.list:
        print("\n".join(benchmark.name for benchmark in benchmarks))
        return 0
    include_joff = args.include_luajit_interpreter or bool(
        args.runtime and "luajit-interpreter" in args.runtime
    )
    selected_runtimes = runtimes(include_joff, {benchmark.compat for benchmark in benchmarks})
    if args.runtime:
        selected = set(args.runtime)
        selected_runtimes = [runtime for runtime in selected_runtimes if runtime.name in selected]
    if args.before:
        before = str(args.before.resolve(strict=True))
        for mode in ("interpreter", "tracing"):
            if any(runtime.name == f"oximond-{mode}" for runtime in selected_runtimes):
                selected_runtimes.append(Runtime(
                    f"before-{mode}",
                    (before, "--compat=lua51", f"--execution={mode}", "-E"),
                ))
    scale = args.scale if args.scale is not None else (1 if args.quick else 3)
    warmup = args.warmup if args.warmup is not None else (0 if args.quick else 1)
    repeat = args.repeat if args.repeat is not None else (3 if args.quick else 7)
    verify_checksums(selected_runtimes, benchmarks, scale, args.cpu, args.timeout)
    timings = measure(
        selected_runtimes,
        benchmarks,
        scale,
        warmup,
        repeat,
        args.cpu,
        args.timeout,
    )
    rows = summarize(selected_runtimes, benchmarks, timings)
    rendered = markdown(selected_runtimes, rows)
    print(rendered, end="")
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "host": {"platform": platform.platform(), "cpu": cpu_model(), "cpu_affinity": args.cpu},
        "git": git_state(),
        "scale": scale,
        "warmup": warmup,
        "repeat": repeat,
        "runtimes": [
            {
                "name": runtime.name,
                **command_metadata(runtime.command),
                **(
                    {
                        "compat_commands": {
                            compat: command_metadata(command)
                            for compat, command in runtime.compat_commands
                        }
                    }
                    if runtime.compat_commands
                    else {}
                ),
            }
            for runtime in selected_runtimes
        ],
        "results": rows,
        "samples_ms": timings,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(rendered, encoding="utf-8")
    if args.fail_gap is not None:
        offenders = [
            row for row in rows if row["oximond_gap"] is not None and row["oximond_gap"] > args.fail_gap
        ]
        if offenders:
            names = ", ".join(
                f"{row['benchmark']} ({row['oximond_gap']:.2f}x)" for row in offenders
            )
            print(f"Oximond gaps above {args.fail_gap:.2f}x: {names}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"benchmark error: {error}", file=sys.stderr)
        raise SystemExit(1)
