#!/usr/bin/env python3
"""Per-opcode executed-instruction tracer for the interpreter's fast loop.

The fast loop (`run_fast_activations::<false, false, false>`) is bound by the
number of instructions it executes, not by branch misses or cache misses. This
tool shows what each opcode costs in that currency. It finds the loop's
dispatch `jmp *%reg`, runs the binary under gdb, skips a number of dispatches
to get past start-up, then single-steps the next N dispatches. Each stretch
from one dispatch target to the next dispatch is a segment. The report has one
row per target address: the number of segments, the instructions executed per
segment (the dispatch itself included) and the source line from addr2line.

    benchmarks/optrace.py target/release/oximond numeric/for_sum
    benchmarks/optrace.py target/release/oximond numeric/for_sum --dump 0x1a2b3c

Needs gdb, nm, objdump and addr2line, and a release binary with line tables.
It runs gdb with ASLR off, so a PIE binary loads at 0x555555554000.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Mapping, NoReturn, Sequence

try:  # Only present when this file runs as a gdb script.
    import gdb  # type: ignore[import-not-found]
except ImportError:
    gdb = None

ROOT = Path(__file__).resolve().parent.parent
# The interpreter's instantiation: no recording, no hooks, direct Lua calls
# (`<RECORD, HOOKED, COUNT_ONLY, DIRECT>` in fast_path.rs).
LOOP_SYMBOL = "run_fast_activations::<false, false, false, true>"
LOAD_BASE = 0x555555554000
SEGMENT_LIMIT = 2000


def gdb_record() -> None:
    """Runs inside gdb: records the instruction addresses of each segment."""
    dispatch = int(os.environ["OPTRACE_DISPATCH"], 16)
    skip = int(os.environ["OPTRACE_SKIP"])
    count = int(os.environ["OPTRACE_COUNT"])
    gdb.execute("set pagination off")
    gdb.execute("set confirm off")
    gdb.execute(f"break *{dispatch:#x}")
    gdb.execute("run")
    gdb.execute(f"ignore 1 {skip}")
    gdb.execute("continue")
    gdb.execute("delete")
    segments = []
    for _ in range(count):
        path = []
        gdb.execute("stepi", to_string=True)
        while True:
            pc = int(gdb.selected_frame().pc())
            path.append(pc)
            if pc == dispatch or len(path) > SEGMENT_LIMIT:
                break
            gdb.execute("stepi", to_string=True)
        segments.append(path)
    with open(os.environ["OPTRACE_OUT"], "w") as out:
        json.dump(segments, out)
    gdb.execute("kill")


def fail(message: str) -> NoReturn:
    sys.exit(f"optrace: {message}")


def find_dispatch(binary: Path) -> int:
    """Offset of the first indirect `jmp *%reg` in the fast loop."""
    for tool in ("nm", "objdump"):
        if shutil.which(tool) is None:
            fail(f"{tool} not found; install binutils")
    symbols = subprocess.run(["nm", "-C", "-S", str(binary)], capture_output=True, text=True)
    start = size = None
    for line in symbols.stdout.splitlines():
        parts = line.split(maxsplit=3)
        if len(parts) == 4 and parts[3].endswith("::" + LOOP_SYMBOL):
            start, size = int(parts[0], 16), int(parts[1], 16)
            break
    if start is None:
        fail(f"symbol {LOOP_SYMBOL} not found in {binary}; build with "
             f"`cargo build --release -p oximond-cli` and do not strip it")
    dis = subprocess.run(
        ["objdump", "-d", "--no-show-raw-insn", f"--start-address={start}", f"--stop-address={start + size}",
         str(binary)], capture_output=True, text=True).stdout
    for line in dis.splitlines():
        match = re.match(r"\s*([0-9a-f]+):\s+jmp\s+\*%r", line)
        if match:
            return int(match.group(1), 16)
    fail(f"no `jmp *%reg` dispatch found in {LOOP_SYMBOL} of {binary}")


def resolve_target(target: str, profile: str | None) -> tuple[Path, str]:
    """(Lua file, profile) for a suite benchmark name or a Lua file path."""
    path = Path(target)
    if path.is_file():
        return path.resolve(), profile or "lua51"
    for entry in json.loads((ROOT / "benchmarks" / "suite.json").read_text()):
        if entry["name"] == target:
            return ROOT / "benchmarks" / entry["path"], profile or entry.get("compat", "lua51")
    fail(f"{target} is neither a Lua file nor a benchmark name in benchmarks/suite.json")


def record(binary: Path, script: Path, profile: str, execution: str, scale: str, dispatch: int,
           skip: int, count: int) -> list[list[int]]:
    if shutil.which("gdb") is None:
        fail("gdb not found; install it (this tool single-steps the binary under gdb)")
    with tempfile.TemporaryDirectory(prefix="optrace-") as tmp:
        out = Path(tmp) / "segments.json"
        env = dict(os.environ, OPTRACE_DISPATCH=hex(LOAD_BASE + dispatch), OPTRACE_SKIP=str(skip),
                   OPTRACE_COUNT=str(count), OPTRACE_OUT=str(out), OXIMOND_JIT_SYNC="1")
        cmd = ["gdb", "-q", "-batch", "-ex", "set disable-randomization on", "-x", str(Path(__file__).resolve()),
               "--args", str(binary), f"--compat={profile}", f"--execution={execution}", "-E", str(script), scale]
        result = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if not out.exists():
            fail(f"gdb recorded nothing (exit {result.returncode}). The benchmark may finish before "
                 f"--skip={skip} dispatches, or the dispatch address is wrong. gdb output:\n{result.stdout[-1500:]}")
        return json.loads(out.read_text())


def aggregate(segments: Sequence[Sequence[int]], base: int = LOAD_BASE) -> list[dict[str, object]]:
    """One row per dispatch target, largest share of executed instructions first.

    `lengths` maps each distinct segment length to how often it occurred, since
    one target can take several paths (a branch inside the opcode)."""
    by_target: dict[int, list[int]] = collections.defaultdict(list)
    for path in segments:
        if path:
            by_target[path[0] - base].append(len(path))
    total = sum(len(path) for path in segments)
    rows = []
    for target, lengths in by_target.items():
        rows.append({
            "target": target, "segments": len(lengths), "mean": sum(lengths) / len(lengths),
            "min": min(lengths), "max": max(lengths), "lengths": dict(collections.Counter(lengths)),
            "share": 100.0 * sum(lengths) / total if total else 0.0,
        })
    rows.sort(key=lambda row: (-row["share"], row["target"]))
    return rows


def source_lines(binary: Path, targets: Sequence[int]) -> dict[int, str]:
    """Outermost source line of each target, which is the fast_path.rs line even
    when the opcode's first instruction comes from an inlined helper."""
    if not targets or shutil.which("addr2line") is None:
        return {}
    out = subprocess.run(["addr2line", "-a", "-i", "-e", str(binary)] + [hex(t) for t in targets],
                         capture_output=True, text=True).stdout.splitlines()
    lines: dict[int, str] = {}
    current = None
    for line in out:
        if re.fullmatch(r"0x[0-9a-f]+", line):
            current = int(line, 16)
        elif current is not None:
            lines[current] = line.split("/")[-1]  # later lines are outer frames
    return lines


def format_report(segments: Sequence[Sequence[int]], lines: Mapping[int, str], base: int = LOAD_BASE) -> str:
    rows = aggregate(segments, base)
    total = sum(len(path) for path in segments)
    out = [f"segments={len(segments)} mean={total / len(segments) if segments else 0:.1f} instructions"]
    out.append(f"{'target':>10} {'n':>6} {'instrs':>7} {'share':>6}  source")
    for row in rows:
        instrs = f"{row['min']}" if row["min"] == row["max"] else f"{row['mean']:.1f}"
        out.append(f"{row['target']:#10x} {row['segments']:6d} {instrs:>7} {row['share']:5.1f}%  "
                   f"{lines.get(row['target'], '?')}")
    return "\n".join(out)


def disassembly(binary: Path, lo: int, hi: int) -> dict[int, str]:
    dis = subprocess.run(["objdump", "-d", "--no-show-raw-insn", f"--start-address={lo:#x}",
                          f"--stop-address={hi:#x}", str(binary)], capture_output=True, text=True).stdout
    table = {}
    for line in dis.splitlines():
        match = re.match(r"\s*([0-9a-f]+):\s+(.*)", line)
        if match:
            table[int(match.group(1), 16)] = match.group(2).strip()
    return table


def format_dump(segments: Sequence[Sequence[int]], target: int, binary: Path, base: int = LOAD_BASE) -> str:
    """The executed path of the first segment that starts at offset `target`."""
    for path in segments:
        if path and path[0] - base == target:
            table = disassembly(binary, min(path) - base, max(path) - base + 16)
            out = [f"segment starting {target:#x} ({len(path)} instructions)"]
            out += [f"  {pc - base:#x}: {table.get(pc - base, '?')[:90]}" for pc in path]
            return "\n".join(out)
    known = ", ".join(f"{row['target']:#x}" for row in aggregate(segments, base))
    fail(f"no recorded segment starts at {target:#x}; recorded targets: {known}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("binary", type=Path, help="release oximond binary")
    parser.add_argument("target", nargs="?", help="benchmark name from suite.json, or a Lua file")
    parser.add_argument("--profile", help="compat profile (default: the suite entry's, else lua51)")
    parser.add_argument("--execution", default="interpreter")
    parser.add_argument("--scale", default="1", help="scale argument passed to the script")
    parser.add_argument("--skip", type=int, default=200000, help="dispatches to run before recording")
    parser.add_argument("--record", type=int, default=400, help="dispatches to record")
    parser.add_argument("--dump", metavar="ADDR", type=lambda text: int(text, 16),
                        help="print the executed path of the segment at this target (hex)")
    parser.add_argument("--save", type=Path, help="write the recorded segments as JSON")
    parser.add_argument("--load", type=Path, help="report from segments saved with --save instead of running")
    args = parser.parse_args()
    if args.target is None and args.load is None:
        parser.error("a target or --load is required")
    binary = args.binary.resolve()
    if not binary.is_file():
        fail(f"{binary} does not exist")
    if args.load:
        segments = json.loads(args.load.read_text())
    else:
        if shutil.which("gdb") is None:
            fail("gdb not found; install it (this tool single-steps the binary under gdb)")
        script, profile = resolve_target(args.target, args.profile)
        dispatch = find_dispatch(binary)
        print(f"dispatch jump at {dispatch:#x}", file=sys.stderr)
        segments = record(binary, script, profile, args.execution, args.scale, dispatch, args.skip, args.record)
    if args.save:
        args.save.write_text(json.dumps(segments))
    if args.dump is not None:
        print(format_dump(segments, args.dump, binary))
    else:
        targets = [row["target"] for row in aggregate(segments)]
        print(format_report(segments, source_lines(binary, targets)))


if gdb is not None:
    gdb_record()
elif __name__ == "__main__":
    main()
