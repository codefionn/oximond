#!/usr/bin/env python3
"""Count instructions, table allocations and trace activity for Oximond.

run.py compares wall time across runtimes. This tool answers a narrower
question: what did one change do to the work Oximond itself performs? For each
selected benchmark and each Oximond mode it reports

  * retired user instructions from `perf stat -e instructions:u`, pinned with
    taskset, as the median of --repeat runs;
  * real tables allocated, elided (phantom) table charges and their bytes, and
    the trace statistics, all read from the OXIMOND_COUNTERS line the CLI
    prints to stderr at exit;
  * with --warmed, the marginal instructions of one more scale unit, which
    leaves out start-up, recording and compilation; with --scales, the same
    at several scales, checked for a stable trace plan;
  * with --op-stats-oximond, the operation counts of a diagnostic build;
  * with --alloc-phases-oximond, allocation counts split by phase (startup,
    recording, compiler, installation, guest execution, exit recovery,
    report) from an alloc-phases diagnostic build.

It also records SHA-256 hashes of the binaries and the machine load. Run it
through the gate when the numbers matter:

    scripts/heavy.sh -x python3 benchmarks/measure.py --cpu 4 --json out.json
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run  # noqa: E402  (shares the suite loader, environment and metadata helpers)

COUNTERS_PREFIX = "oximond-counters:"
MODES = ("interpreter", "tracing")
# Every OXIMOND_COUNTERS line from a current binary has these keys. A line
# without one comes from an older binary, and its numbers are not comparable.
REQUIRED_COUNTER_KEYS = (
    "tables_created", "elided_tables", "elided_table_bytes", "traces_compiled", "native_entries",
    "iterations", "recording_starts", "traces_recorded", "loads_forwarded", "loads_merged",
    "loads_hoisted",
)
# The line prints these keys only when the count is nonzero, so a missing key
# means zero by design, not missing data.
NONZERO_ONLY_PREFIXES = ("abort.", "exit.", "side_exit.")
OPTIMIZER_KEYS = ("loads_forwarded", "loads_merged", "loads_hoisted")
ALLOC_KEYS = ("alloc_count", "alloc_bytes", "alloc_peak_bytes")
GATE_EXIT_STATUS = 3

OP_STATS_SCHEMA = "oximond-op-stats/2"
# Reports of these schemas still load with allow_legacy. They lack the counters
# in V2_OP_COUNTERS, which then read as unavailable (None, never 0), and they
# never satisfy a fresh run's requirements.
OP_STATS_LEGACY_SCHEMAS = ("oximond-op-stats/1",)
OP_STATS_FEATURE = "op-stats"

ALLOC_PHASES_SCHEMA = "oximond-alloc-phases/1"
ALLOC_PHASES_FEATURE = "alloc-phases"
# `Phase::name` in crates/oximond/src/runtime/alloc_phase.rs, in index order.
ALLOC_PHASE_NAMES = ("unclassified", "recording", "optimization", "codegen", "installation",
                     "execution", "exit_recovery", "report")
ALLOC_PHASE_TOTAL_KEYS = ("alloc_count", "alloc_bytes", "alloc_peak_bytes", "live_bytes")
ALLOC_PHASE_KEYS = ("alloc_count", "alloc_bytes", "alloc_count_vm", "alloc_count_other",
                    "alloc_bytes_vm", "alloc_bytes_other", "active_peak_bytes",
                    "origin_live_bytes", "origin_peak_bytes")
OP_STATS_TOP_PAIRS = 40
# The counters of crates/oximond/src/runtime/op_stats.rs, by hypothesis. A
# report without one of them comes from an older diagnostic build.
REQUIRED_OP_COUNTERS: dict[str, tuple[str, ...]] = {
    "h1": (
        "h1.ir.array_sites", "h1.ir.array_sites.const_key", "h1.ir.array_sites.float_key",
        "h1.ir.array_sites.induction", "h1.ir.array_sites.range_once", "h1.ir.array_sites.bytes",
        "h1.ir.type_guards", "h1.native.bound_checks", "h1.native.integrality_checks",
        "h1.native.bound_checks.induction", "h1.native.bound_checks.range_once",
        "h1.interp.get_table.integer_key", "h1.interp.get_table.float_key",
        "h1.interp.set_table.integer_key", "h1.interp.set_table.float_key",
        "h1.ir.array_sites.float_invariant_induction", "h1.ir.array_sites.float_invariant_range_once",
        "h1.native.bound_checks.float_invariant_induction", "h1.native.bound_checks.float_invariant_range_once",
    ),
    "h2": (
        "h2.fast.dispatches", "h2.fast.scalar_misses", "h2.fast.arithmetic_slow",
        "h2.fast.exits.fallback", "h2.general.steps",
    ),
    "h3": (
        "h3.ir.members", "h3.ir.snapshots", "h3.ir.snapshot_frames", "h3.ir.snapshot_entries",
        "h3.ir.snapshot_entries.value", "h3.ir.snapshot_entries.const",
        "h3.ir.snapshot_entries.unchanged", "h3.ir.snapshot_entries.root",
        "h3.ir.snapshot_entries.virtual", "h3.ir.snapshot_entries.other", "h3.ir.snapshot_virtuals",
        "h3.ir.snapshot_recipe_entries", "h3.ir.snapshot_recipe_entries.value",
        "h3.ir.snapshot_recipe_entries.const", "h3.ir.snapshot_recipe_entries.root",
        "h3.ir.snapshot_recipe_values.feeding", "h3.ir.snapshot_recipe_values.recovery",
        "h3.ir.snapshot_recipes_unknown", "h3.ir.snapshot_values", "h3.ir.snapshot_only_values",
        "h3.ir.snapshot_only_values.cheap_pure", "h3.ir.snapshot_only_values.instructions",
        "h3.ir.guards", "h3.ir.exit_descriptors", "h3.ir.exit_area_slots",
        "h3.ir.exit_area_slots.max", "h3.ir.trees_compiled", "h3.native.exits",
        "h3.native.exit_slots_stored",
    ),
    "h4": (
        "h4.activations", "h4.activations.argument_slots", "h4.activations.argument_slots.stack",
        "h4.activations.string_arguments", "h4.activations.cleared_slots",
        "h4.call_sites.closure_calls", "h4.call_sites.arguments",
        "h4.call_sites.arguments.caller_dead", "h4.call_sites.arguments.caller_dead_string",
        "h4.call_sites.vararg_callee", "h4.call_sites.captured_caller", "h4.general.calls_hooked",
        "h4.inline_plan_calls", "h4.inline_table_plan_calls",
    ),
    "h5": (
        "h5.clear_range.instructions", "h5.clear_range.slots", "h5.clear_range.slots.nil",
        "h5.clear_range.slots.scalar", "h5.clear_range.slots.string",
        "h5.clear_range.slots.reference", "h5.gc.root_scans", "h5.gc.stack_slots",
        "h5.gc.frame_slots", "h5.gc.frame_slots.reference", "h5.gc.frame_slots.reference_dead",
        "h5.gc.frame_slots.string", "h5.gc.frame_slots.string_dead", "h5.gc.frames_unknown_pc",
    ),
    "h6": (
        "h6.calls.same_closure", "h6.calls.same_prototype_other_closure", "h6.calls.other_prototype",
        "h6.ir.pinned_closures", "h6.closures_created",
    ),
}
# Counters schema 2 added; a schema 1 report has none of them.
V2_OP_COUNTERS = frozenset({
    "h3.ir.snapshot_entries.virtual", "h3.ir.snapshot_virtuals", "h3.ir.snapshot_recipe_entries",
    "h3.ir.snapshot_recipe_entries.value", "h3.ir.snapshot_recipe_entries.const",
    "h3.ir.snapshot_recipe_entries.root", "h3.ir.snapshot_recipe_values.feeding",
    "h3.ir.snapshot_recipe_values.recovery", "h3.ir.snapshot_recipes_unknown",
})
# Counters both schemas name whose meaning schema 2 changed: the snapshot walk
# now visits virtual recipes, so `.other` no longer holds virtual entries and
# the snapshot value classes count recipe operands. A legacy report's values
# of these are not comparable and read as unavailable too.
V2_CHANGED_OP_COUNTERS = frozenset({
    "h3.ir.snapshot_entries.other", "h3.ir.snapshot_values", "h3.ir.snapshot_only_values",
    "h3.ir.snapshot_only_values.cheap_pure", "h3.ir.snapshot_only_values.instructions",
})
# `Kind::name` in crates/oximond/src/runtime/op_stats.rs.
OP_COUNTER_KINDS = frozenset({"static_ir", "dynamic_fast_interp", "dynamic_general_interp", "dynamic_interp",
                              "dynamic_native", "gc_event"})
OP_HISTOGRAMS = ("fast_opcodes", "general_opcodes", "fast_opcode_pairs", "operand_classes")
OP_SITE_TABLES = ("arithmetic", "field", "call")
HYPOTHESES = {
    "h1": "loop range proofs",
    "h2": "adaptive bytecode",
    "h3": "recovery recipes",
    "h4": "direct call windows and dead-argument moves",
    "h5": "live-root maps and lazy dead-slot clearing",
    "h6": "closure-family specialization",
}
# The counters op_stats_markdown shows per hypothesis; the JSON keeps all.
KEY_OP_COUNTERS: dict[str, tuple[str, ...]] = {
    "h1": ("h1.ir.array_sites", "h1.ir.array_sites.induction", "h1.ir.array_sites.range_once",
           "h1.ir.array_sites.float_key", "h1.native.bound_checks", "h1.native.bound_checks.range_once",
           "h1.native.integrality_checks", "h1.interp.get_table.integer_key", "h1.interp.get_table.float_key",
           "h1.ir.array_sites.float_invariant_range_once", "h1.native.bound_checks.float_invariant_range_once"),
    "h2": ("h2.fast.dispatches", "h2.fast.scalar_misses", "h2.fast.arithmetic_slow",
           "h2.fast.exits.fallback", "h2.general.steps"),
    "h3": ("h3.ir.snapshots", "h3.ir.snapshot_entries", "h3.ir.snapshot_entries.virtual",
           "h3.ir.snapshot_virtuals", "h3.ir.snapshot_recipes_unknown",
           "h3.ir.snapshot_values", "h3.ir.snapshot_only_values",
           "h3.ir.snapshot_only_values.cheap_pure", "h3.ir.exit_area_slots", "h3.native.exits",
           "h3.native.exit_slots_stored"),
    "h4": ("h4.activations", "h4.activations.argument_slots", "h4.activations.string_arguments",
           "h4.activations.cleared_slots", "h4.call_sites.arguments", "h4.call_sites.arguments.caller_dead",
           "h4.call_sites.vararg_callee", "h4.call_sites.captured_caller", "h4.general.calls_hooked",
           "h4.inline_plan_calls", "h4.inline_table_plan_calls"),
    "h5": ("h5.clear_range.instructions", "h5.clear_range.slots", "h5.clear_range.slots.reference",
           "h5.clear_range.slots.string", "h5.gc.root_scans", "h5.gc.frame_slots.reference",
           "h5.gc.frame_slots.reference_dead", "h5.gc.frames_unknown_pc"),
    "h6": ("h6.calls.same_closure", "h6.calls.same_prototype_other_closure", "h6.calls.other_prototype",
           "h6.ir.pinned_closures", "h6.closures_created"),
}
DIAGNOSTIC_LABEL = "diagnostic build overhead"

# The stock interpreter that checks each compatibility profile.
ORACLES = {
    "lua51": ("lua5.1", "LUA_BIN"),
    "lua54": ("lua5.4", "LUA54_BIN"),
    "luajit21": ("luajit", "LUAJIT_BIN"),
}


@dataclass(frozen=True)
class Mode:
    name: str
    binary: Path
    execution: str


@dataclass
class Sample:
    instructions: int
    counters: dict[str, int]
    stdout: str


@dataclass(frozen=True)
class Oracle:
    """What checked a benchmark's output: a stock interpreter, or nothing and why."""
    name: str | None
    checksum: str | None
    path: str | None = None
    reason: str | None = None


def parse_perf(text: str) -> int:
    """Returns the instructions:u count from `perf stat -x,` output."""
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(",")
        if len(fields) >= 3 and fields[2].startswith("instructions"):
            value = fields[0]
            if not value.isdigit():
                raise RuntimeError(f"perf could not count instructions: {line!r}")
            return int(value)
    raise RuntimeError(f"no instructions:u line in perf output: {text!r}")


def parse_counters(stderr: str, binary: Path | str = "oximond") -> dict[str, int]:
    """Parses the `oximond-counters:` line into integer counters.

    Raises RuntimeError naming `binary` when the line is absent or lacks one
    of REQUIRED_COUNTER_KEYS: such a binary predates the counters this tool
    compares, and a zero in their place would be a made-up number. Keys with
    a NONZERO_ONLY_PREFIXES prefix are printed only when nonzero.
    """
    for line in stderr.splitlines():
        if line.startswith(COUNTERS_PREFIX):
            counters = {}
            for item in line[len(COUNTERS_PREFIX):].split():
                key, _, value = item.partition("=")
                if not value.lstrip("-").isdigit():
                    raise RuntimeError(f"bad counter {item!r} in {line!r}")
                counters[key] = int(value)
            missing = [key for key in REQUIRED_COUNTER_KEYS if key not in counters]
            if missing:
                raise RuntimeError(f"{binary} predates counters {', '.join(missing)}: its OXIMOND_COUNTERS line "
                                   f"lacks them; rebuild it from a current revision")
            return counters
    raise RuntimeError(f"{binary} printed no {COUNTERS_PREFIX} line: the binary predates OXIMOND_COUNTERS "
                       f"or did not exit cleanly")


def prefixed(counters: Mapping[str, int], prefix: str) -> int:
    return sum(value for key, value in counters.items() if key.startswith(prefix))


def aggregate(samples: Sequence[Sample]) -> dict[str, object]:
    """Median instructions and median of each counter over the samples.

    A counter that some samples report and others do not is an error, except
    for the nonzero-only keys, whose absence means zero.
    """
    keys = sorted({key for sample in samples for key in sample.counters})
    for key in keys:
        if key.startswith(NONZERO_ONLY_PREFIXES):
            continue
        lacking = sum(1 for sample in samples if key not in sample.counters)
        if lacking:
            raise RuntimeError(f"counter {key} is missing from {lacking} of {len(samples)} samples")
    return {
        "instructions": int(statistics.median(s.instructions for s in samples)),
        "instructions_samples": [s.instructions for s in samples],
        "counters": {
            key: int(statistics.median(s.counters.get(key, 0) for s in samples))
            for key in keys
        },
    }


def oximond_environment(jit_sync: bool, extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment for one Oximond run: counters on, JIT sync as requested."""
    environment = run.clean_environment()
    environment["OXIMOND_COUNTERS"] = "1"
    environment.pop("OXIMOND_OP_STATS", None)
    environment.pop("OXIMOND_ALLOC_PHASES", None)
    if jit_sync:
        environment["OXIMOND_JIT_SYNC"] = "1"
    else:
        environment.pop("OXIMOND_JIT_SYNC", None)
    environment.update(extra or {})
    return environment


def invoke(mode: Mode, benchmark: run.Benchmark, scale: int, cpu: int | None, timeout: float,
           jit_sync: bool = False, extra_env: Mapping[str, str] | None = None) -> Sample:
    with tempfile.TemporaryDirectory(prefix="measure-") as directory:
        output = Path(directory) / "perf.txt"
        command = [
            "perf", "stat", "-x,", "-e", "instructions:u", "-o", str(output), "--",
            str(mode.binary), f"--compat={benchmark.compat}", f"--execution={mode.execution}",
            "-E", str(benchmark.path), str(scale),
        ]
        if cpu is not None:
            command = ["taskset", "-c", str(cpu), *command]
        environment = oximond_environment(jit_sync, extra_env)
        completed = subprocess.run(
            command,
            cwd=run.ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        stderr = completed.stderr.decode("utf-8", "replace")
        if completed.returncode != 0:
            raise RuntimeError(
                f"{mode.name} failed {benchmark.name} with status {completed.returncode}\n"
                f"command: {' '.join(command)}\nstderr: {stderr.strip()}"
            )
        instructions = parse_perf(output.read_text(encoding="utf-8"))
    return Sample(
        instructions,
        parse_counters(stderr, mode.binary),
        completed.stdout.decode("utf-8", "replace").strip(),
    )


def oracle_for(compat: str) -> tuple[str | None, str | None, str | None]:
    """(oracle name, executable, reason when there is none) for a profile."""
    if compat not in ORACLES:
        return None, None, f"no stock oracle for compat {compat}"
    name, variable = ORACLES[compat]
    configured = os.environ.get(variable)
    path = shutil.which(configured or name)
    if path is None:
        return None, None, f"{configured or name} not found (set {variable})"
    return name, path, None


def reference_checksum(benchmark: run.Benchmark, scale: int, timeout: float) -> Oracle:
    """Output of the stock interpreter for the benchmark's profile: Lua 5.1
    for lua51, Lua 5.4 for lua54, LuaJIT for luajit21."""
    name, path, reason = oracle_for(benchmark.compat)
    if path is None:
        return Oracle(None, None, None, reason)
    completed = subprocess.run(
        [path, str(benchmark.path), str(scale)],
        cwd=run.ROOT,
        env=run.clean_environment(),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        error = completed.stderr.decode("utf-8", "replace").strip().splitlines()
        return Oracle(None, None, path, f"{name} exited with status {completed.returncode}: "
                                        f"{error[-1] if error else ''}".rstrip(": "))
    return Oracle(name, completed.stdout.decode("utf-8", "replace").strip(), path)


def trace_plan(counters: Mapping[str, int]) -> dict[str, int]:
    """The counts of recording and compilation activity a run did, with each
    abort reason separately.

    Equal counts at two scales do not prove the same traces: there are no
    trace fingerprints. A difference does prove the plans differ.
    """
    plan = {
        "traces_compiled": counters["traces_compiled"],
        "recording_starts": counters["recording_starts"],
        "traces_recorded": counters["traces_recorded"],
        "aborts": prefixed(counters, "abort."),
    }
    plan.update({key: value for key, value in counters.items() if key.startswith("abort.")})
    return plan


def scale_series(points: Sequence[tuple[int, Mapping[str, object]]], tolerance: float) -> dict[str, object]:
    """Marginal instructions per scale unit between consecutive scales.

    `points` pairs each scale, ascending, with its aggregate. The estimate is
    inconclusive when the trace plan differs between two scales, when a
    marginal is not positive, or when two consecutive marginals differ by
    more than `tolerance` percent.
    """
    reasons = []
    per_scale = [{"scale": scale, "instructions": data["instructions"],
                  "trace_plan": trace_plan(data["counters"])} for scale, data in points]
    marginal = []
    for low, high in zip(per_scale, per_scale[1:]):
        units = high["scale"] - low["scale"]
        value = (high["instructions"] - low["instructions"]) / units
        marginal.append({"from": low["scale"], "to": high["scale"], "instructions_per_scale": value})
        if value <= 0:
            reasons.append(f"marginal from scale {low['scale']} to {high['scale']} is not positive ({value:,.0f})")
        keys = sorted(set(low["trace_plan"]) | set(high["trace_plan"]))
        changed = [f"{key} {low['trace_plan'].get(key, 0)}->{high['trace_plan'].get(key, 0)}"
                   for key in keys if low["trace_plan"].get(key, 0) != high["trace_plan"].get(key, 0)]
        if changed:
            reasons.append(f"trace counts differ between scale {low['scale']} and {high['scale']}: "
                           + ", ".join(changed))
    for first, second in zip(marginal, marginal[1:]):
        a, b = first["instructions_per_scale"], second["instructions_per_scale"]
        spread = abs(b - a) / max(abs(a), abs(b)) * 100 if max(abs(a), abs(b)) else 0.0
        if spread > tolerance:
            reasons.append(f"marginals {first['from']}->{first['to']} and {second['from']}->{second['to']} "
                           f"differ by {spread:.1f}% (tolerance {tolerance:g}%)")
    return {
        "points": per_scale,
        "marginal": marginal,
        "status": "inconclusive" if reasons else "ok",
        "reasons": reasons,
    }


def measure_mode(
    mode: Mode,
    benchmark: run.Benchmark,
    scale: int,
    repeat: int,
    cpu: int | None,
    timeout: float,
    warmed: bool,
    jit_sync: bool = False,
    scales: Sequence[int] | None = None,
    tolerance: float = 10.0,
) -> tuple[dict[str, object], str]:
    """Measures one mode. With `scales`, the first scale is the reported one."""
    base = scales[0] if scales else scale
    samples = [invoke(mode, benchmark, base, cpu, timeout, jit_sync) for _ in range(repeat)]
    checksums = {sample.stdout for sample in samples}
    if len(checksums) != 1:
        raise RuntimeError(f"{mode.name} printed varying output for {benchmark.name}: {checksums}")
    checksum = next(iter(checksums))
    result = aggregate(samples)
    extra_scales = list(scales[1:]) if scales else ([2 * scale] if warmed else [])
    points = [(base, result)]
    outputs = {base: checksum}
    for other in extra_scales:
        runs = [invoke(mode, benchmark, other, cpu, timeout, jit_sync) for _ in range(repeat)]
        printed = {s.stdout for s in runs}
        if len(printed) != 1:
            raise RuntimeError(f"{mode.name} printed varying output for {benchmark.name} at scale {other}")
        output = next(iter(printed))
        same = [s for s, text in outputs.items() if text == output]
        if same:
            raise RuntimeError(f"{benchmark.name} prints the same output at scales {same[0]} and {other}")
        outputs[other] = output
        points.append((other, aggregate(runs)))
    if warmed and not scales:
        twice = points[1][1]
        result["instructions_double_scale"] = twice["instructions"]
        result["warmed_instructions_per_scale"] = twice["instructions"] - result["instructions"]
        result["counters_double_scale"] = twice["counters"]
        check = scale_series(points, tolerance)
        result["warmed_check"] = {"status": check["status"], "reasons": check["reasons"]}
    if scales:
        result["scales"] = scale_series(points, tolerance)
    return result, checksum


def validate_op_stats(document: object, binary: Path | str, execution: str | None = None,
                      jit_sync: bool | None = None, allow_legacy: bool = False) -> dict[str, object]:
    """Checks a parsed op-stats report and returns the part measure.py keeps.

    Raises RuntimeError naming `binary` for a wrong schema, a build without
    the op-stats feature, or a missing counter, histogram or site table.
    A report of a legacy schema is refused unless `allow_legacy`; then it loads
    with `legacy` set and its V2_OP_COUNTERS unavailable (None), listed in
    `unavailable_counters`. A legacy report never satisfies the current schema.
    """
    if not isinstance(document, dict):
        raise RuntimeError(f"{binary} wrote an op-stats report that is not a JSON object")
    schema = document.get("schema")
    legacy = schema in OP_STATS_LEGACY_SCHEMAS
    if schema != OP_STATS_SCHEMA and not (legacy and allow_legacy):
        raise RuntimeError(f"{binary} wrote op-stats schema {schema!r}, expected {OP_STATS_SCHEMA!r}")
    build = document.get("build")
    if not isinstance(build, list) or OP_STATS_FEATURE not in build:
        raise RuntimeError(f"{binary} reports build features {build!r} without {OP_STATS_FEATURE!r}")
    counters = document.get("counters")
    info = document.get("counter_info")
    if not isinstance(counters, dict) or not isinstance(info, dict):
        raise RuntimeError(f"{binary} wrote an op-stats report without counters or counter_info")
    required = [name for names in REQUIRED_OP_COUNTERS.values() for name in names
                if not (legacy and name in V2_OP_COUNTERS)]
    missing = [name for name in required if name not in counters]
    if missing:
        raise RuntimeError(f"{binary} predates op-stats counters {', '.join(missing)}; rebuild it with "
                           f"--features op-stats from a current revision")
    bad = [name for name in required if not isinstance(counters[name], int) or isinstance(counters[name], bool)]
    if bad:
        raise RuntimeError(f"{binary} wrote non-integer op-stats counters {', '.join(bad)}")
    unlabelled = [name for name in required if not isinstance(info.get(name), dict) or "kind" not in info[name]]
    if unlabelled:
        raise RuntimeError(f"{binary} wrote op-stats counters without a kind: {', '.join(unlabelled)}")
    histograms = document.get("histograms")
    if not isinstance(histograms, dict) or any(not isinstance(histograms.get(h), dict) for h in OP_HISTOGRAMS):
        raise RuntimeError(f"{binary} wrote an op-stats report without the histograms {', '.join(OP_HISTOGRAMS)}")
    sites = document.get("sites")
    if not isinstance(sites, dict) or any(not isinstance(sites.get(t), dict) for t in OP_SITE_TABLES):
        raise RuntimeError(f"{binary} wrote an op-stats report without the site tables {', '.join(OP_SITE_TABLES)}")
    kinds = [name for name in required if info[name]["kind"] not in OP_COUNTER_KINDS]
    if kinds:
        raise RuntimeError(f"{binary} wrote op-stats counters with an unknown kind: {', '.join(kinds)}")
    if execution is not None and document.get("execution") != execution:
        raise RuntimeError(f"{binary} reports execution {document.get('execution')!r}, expected {execution!r}")
    if jit_sync is not None and document.get("jit_sync") != ("on" if jit_sync else "off"):
        raise RuntimeError(f"{binary} reports jit_sync {document.get('jit_sync')!r}, expected "
                           f"{'on' if jit_sync else 'off'}")
    pairs = sorted(histograms["fast_opcode_pairs"].items(), key=lambda item: (-item[1], item[0]))
    unavailable = sorted(V2_OP_COUNTERS | V2_CHANGED_OP_COUNTERS) if legacy else []
    return {
        "schema": schema,
        "legacy": legacy,
        "unavailable_counters": unavailable,
        "build": list(build),
        "report_fields": {key: value for key, value in document.items() if isinstance(value, str) and key != "schema"},
        "counters": {**{name: counters[name] for name in counters if name not in unavailable},
                     **{name: None for name in unavailable}},
        "counter_kinds": {name: info[name]["kind"] for name in info if isinstance(info[name], dict)},
        "histograms": {
            "fast_opcodes": dict(histograms["fast_opcodes"]),
            "general_opcodes": dict(histograms["general_opcodes"]),
            "operand_classes": dict(histograms["operand_classes"]),
            "fast_opcode_pairs": dict(pairs[:OP_STATS_TOP_PAIRS]),
            "fast_opcode_pairs_total": sum(histograms["fast_opcode_pairs"].values()),
            "fast_opcode_pairs_distinct": len(pairs),
        },
        "sites": sites,
    }


def load_op_stats(path: Path, binary: Path | str, execution: str | None = None,
                  jit_sync: bool | None = None, allow_legacy: bool = False) -> dict[str, object]:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"{binary} printed no op-stats report; build it with --features op-stats")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{binary} wrote an op-stats report that is not JSON: {error}") from error
    return validate_op_stats(document, binary, execution, jit_sync, allow_legacy)


def invoke_op_stats(mode: Mode, benchmark: run.Benchmark, scale: int, cpu: int | None, timeout: float,
                    jit_sync: bool) -> tuple[Sample, dict[str, object]]:
    """Runs a diagnostic op-stats build once and returns its sample and report."""
    with tempfile.TemporaryDirectory(prefix="measure-op-stats-") as directory:
        report = Path(directory) / "op-stats.json"
        sample = invoke(mode, benchmark, scale, cpu, timeout, jit_sync, {"OXIMOND_OP_STATS": str(report)})
        stats = load_op_stats(report, mode.binary, mode.execution, jit_sync)
    return sample, stats


def is_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def validate_alloc_phases(document: object, binary: Path | str, execution: str | None = None,
                          jit_sync: bool | None = None) -> dict[str, object]:
    """Checks a parsed alloc-phases report and returns the part measure.py keeps.

    Raises RuntimeError naming `binary` for a wrong schema, a build without the
    alloc-phases feature, a missing or non-count field, a missing phase, or
    phase counters that do not add up to the report's own totals.
    """
    if not isinstance(document, dict):
        raise RuntimeError(f"{binary} wrote an alloc-phases report that is not a JSON object")
    schema = document.get("schema")
    if schema != ALLOC_PHASES_SCHEMA:
        raise RuntimeError(f"{binary} wrote alloc-phases schema {schema!r}, expected {ALLOC_PHASES_SCHEMA!r}")
    build = document.get("build")
    if not isinstance(build, list) or ALLOC_PHASES_FEATURE not in build:
        raise RuntimeError(f"{binary} reports build features {build!r} without {ALLOC_PHASES_FEATURE!r}")
    context = document.get("context")
    if not isinstance(context, dict):
        raise RuntimeError(f"{binary} wrote an alloc-phases report without context")
    if execution is not None and context.get("execution") != execution:
        raise RuntimeError(f"{binary} reports execution {context.get('execution')!r}, expected {execution!r}")
    if jit_sync is not None and context.get("jit_sync") != ("on" if jit_sync else "off"):
        raise RuntimeError(f"{binary} reports jit_sync {context.get('jit_sync')!r}, expected "
                           f"{'on' if jit_sync else 'off'}")
    totals = document.get("totals")
    if not isinstance(totals, dict) or not all(is_count(totals.get(key)) for key in ALLOC_PHASE_TOTAL_KEYS):
        raise RuntimeError(f"{binary} wrote alloc-phases totals without counts for {', '.join(ALLOC_PHASE_TOTAL_KEYS)}")
    instrumentation = document.get("instrumentation")
    if not isinstance(instrumentation, dict) or not is_count(instrumentation.get("header_bytes")):
        raise RuntimeError(f"{binary} wrote an alloc-phases report without instrumentation header_bytes")
    phases = document.get("phases")
    if not isinstance(phases, dict):
        raise RuntimeError(f"{binary} wrote an alloc-phases report without phases")
    missing = [name for name in ALLOC_PHASE_NAMES if not isinstance(phases.get(name), dict)]
    if missing:
        raise RuntimeError(f"{binary} wrote an alloc-phases report without the phases {', '.join(missing)}")
    unknown = sorted(set(phases) - set(ALLOC_PHASE_NAMES))
    if unknown:
        raise RuntimeError(f"{binary} wrote unknown alloc phases {', '.join(unknown)}")
    for name in ALLOC_PHASE_NAMES:
        phase = phases[name]
        bad = [key for key in ALLOC_PHASE_KEYS if not is_count(phase.get(key))]
        if bad:
            raise RuntimeError(f"{binary} wrote alloc phase {name} without counts for {', '.join(bad)}")
        released = phase.get("released_by")
        if not isinstance(released, dict) or any(
                not isinstance(released.get(other), dict)
                or not is_count(released[other].get("count")) or not is_count(released[other].get("bytes"))
                for other in ALLOC_PHASE_NAMES):
            raise RuntimeError(f"{binary} wrote alloc phase {name} without released_by counts for every phase")
        if phase["alloc_count"] != phase["alloc_count_vm"] + phase["alloc_count_other"] or \
                phase["alloc_bytes"] != phase["alloc_bytes_vm"] + phase["alloc_bytes_other"]:
            raise RuntimeError(f"{binary} wrote alloc phase {name} whose thread split does not add up")
    count = sum(phases[name]["alloc_count"] for name in ALLOC_PHASE_NAMES)
    allocated = sum(phases[name]["alloc_bytes"] for name in ALLOC_PHASE_NAMES)
    live = sum(phases[name]["origin_live_bytes"] for name in ALLOC_PHASE_NAMES)
    released_bytes = sum(phases[name]["released_by"][other]["bytes"]
                         for name in ALLOC_PHASE_NAMES for other in ALLOC_PHASE_NAMES)
    if (count, allocated, live) != (totals["alloc_count"], totals["alloc_bytes"], totals["live_bytes"]) \
            or released_bytes + live != totals["alloc_bytes"]:
        raise RuntimeError(f"{binary} wrote alloc phases that do not reconcile with its totals "
                           f"(count {count} vs {totals['alloc_count']}, bytes {allocated} vs "
                           f"{totals['alloc_bytes']}, live {live} vs {totals['live_bytes']}, "
                           f"released+live {released_bytes + live})")
    if document.get("reconciled") is not True:
        raise RuntimeError(f"{binary} says its alloc phases do not reconcile")
    return {
        "schema": schema,
        "build": list(build),
        "context": {key: value for key, value in context.items() if isinstance(value, str)},
        "totals": {key: totals[key] for key in ALLOC_PHASE_TOTAL_KEYS},
        "instrumentation": dict(instrumentation),
        "phases": {name: {**{key: phases[name][key] for key in ALLOC_PHASE_KEYS},
                          "released_by": {other: dict(phases[name]["released_by"][other])
                                          for other in ALLOC_PHASE_NAMES}}
                   for name in ALLOC_PHASE_NAMES},
    }


def load_alloc_phases(path: Path, binary: Path | str, execution: str | None = None,
                      jit_sync: bool | None = None) -> dict[str, object]:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"{binary} printed no alloc-phases report; build it with --no-default-features --features alloc-phases")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{binary} wrote an alloc-phases report that is not JSON: {error}") from error
    return validate_alloc_phases(document, binary, execution, jit_sync)


def invoke_alloc_phases(mode: Mode, benchmark: run.Benchmark, scale: int, cpu: int | None, timeout: float,
                        jit_sync: bool) -> tuple[Sample, dict[str, object]]:
    """Runs an alloc-phases diagnostic build once and returns its sample and report."""
    with tempfile.TemporaryDirectory(prefix="measure-alloc-phases-") as directory:
        report = Path(directory) / "alloc-phases.json"
        sample = invoke(mode, benchmark, scale, cpu, timeout, jit_sync, {"OXIMOND_ALLOC_PHASES": str(report)})
        phases = load_alloc_phases(report, mode.binary, mode.execution, jit_sync)
    return sample, phases


def sibling_of(cpu: int) -> int | None:
    path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    numbers: list[int] = []
    for part in text.split(","):
        low, _, high = part.partition("-")
        numbers.extend(range(int(low), int(high or low) + 1))
    others = [n for n in numbers if n != cpu]
    return others[0] if others else None


def cpu_busy_percent(cpu: int, interval: float = 0.25) -> float | None:
    def read() -> tuple[int, int] | None:
        try:
            for line in Path("/proc/stat").read_text(encoding="utf-8").splitlines():
                if line.startswith(f"cpu{cpu} "):
                    fields = [int(x) for x in line.split()[1:]]
                    return sum(fields), fields[3] + fields[4]
        except OSError:
            pass
        return None

    first = read()
    time.sleep(interval)
    second = read()
    if first is None or second is None or second[0] == first[0]:
        return None
    return 100.0 * (1 - (second[1] - first[1]) / (second[0] - first[0]))


def machine_state(cpu: int | None) -> dict[str, object]:
    uptime = subprocess.run(["uptime"], capture_output=True, text=True, check=False).stdout.strip()
    try:
        load = [float(x) for x in Path("/proc/loadavg").read_text(encoding="utf-8").split()[:3]]
    except OSError:
        load = []
    state: dict[str, object] = {
        "platform": platform.platform(),
        "cpu": run.cpu_model(),
        "cpu_affinity": cpu,
        "uptime": uptime,
        "loadavg": load,
    }
    if cpu is not None:
        sibling = sibling_of(cpu)
        state["sibling_cpu"] = sibling
        state["pinned_cpu_busy_percent"] = cpu_busy_percent(cpu)
        state["sibling_cpu_busy_percent"] = cpu_busy_percent(sibling) if sibling is not None else None
    return state


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binary_metadata(path: Path, features: Sequence[str] | None = None,
                    features_source: str | None = None) -> dict[str, object]:
    entry: dict[str, object] = {"path": str(path), "sha256": sha256(path)}
    if features is not None:
        entry["features"] = list(features)
        entry["features_source"] = features_source
    return entry


def ratio(numerator: int | float | None, denominator: int | float | None) -> float | None:
    if numerator is None or not denominator:
        return None
    return numerator / denominator


def summarize_counters(counters: Mapping[str, int]) -> dict[str, object]:
    missing = [key for key in REQUIRED_COUNTER_KEYS if key not in counters]
    if missing:
        raise RuntimeError(f"counters lack {', '.join(missing)}")
    exits = {k[5:]: v for k, v in counters.items() if k.startswith("exit.")}
    side = {k[10:]: v for k, v in counters.items() if k.startswith("side_exit.")}
    aborts = {k[6:]: v for k, v in counters.items() if k.startswith("abort.")}
    extra = {key: counters[key] for key in ALLOC_KEYS if key in counters}
    return {
        **extra,
        **{key: counters[key] for key in REQUIRED_COUNTER_KEYS},
        "exits": exits,
        "side_exits": side,
        "aborts": aborts,
    }


def fmt_count(value: int | None) -> str:
    return "n/a" if value is None else f"{value:,}"


def fmt_map(values: Mapping[str, int]) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(values.items())) or "-"


def fmt_ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def markdown(rows: Sequence[dict[str, object]], mode_names: Sequence[str], warmed: bool) -> str:
    header = ["Benchmark", "Mode", "instructions:u", *(["warmed per scale"] if warmed else []),
              "vs interpreter", "tables", "elided", "elided bytes",
              "traces", "entries", "iterations", "exits", "side exits", "aborts"]
    alignment = [":--", ":--"] + ["--:"] * (len(header) - 5) + [":--"] * 3
    extra = [key for key in OPTIMIZER_KEYS + ALLOC_KEYS
             if any(key in row["modes"][name].get("summary", {}) or key in row["modes"][name].get("alloc", {})
                    for row in rows for name in mode_names)]
    header += extra
    alignment += ["--:"] * len(extra)
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(alignment) + " |"]
    for row in rows:
        interpreter = row["modes"].get("interpreter", {}).get("instructions")
        for name in mode_names:
            data = row["modes"][name]
            summary = data.get("summary", {})
            speed = ratio(data["instructions"], interpreter)
            warmed_cell = fmt_count(data.get("warmed_instructions_per_scale"))
            if data.get("warmed_check", {}).get("status") == "inconclusive":
                warmed_cell += " (inconclusive)"
            lines.append("| " + " | ".join([
                str(row["benchmark"]),
                name,
                fmt_count(data["instructions"]),
                *([warmed_cell] if warmed else []),
                f"{speed:.3f}" if speed is not None else "n/a",
                fmt_count(summary.get("tables_created")),
                fmt_count(summary.get("elided_tables")),
                fmt_count(summary.get("elided_table_bytes")),
                fmt_count(summary.get("traces_compiled")),
                fmt_count(summary.get("native_entries")),
                fmt_count(summary.get("iterations")),
                fmt_map(summary.get("exits", {})),
                fmt_map(summary.get("side_exits", {})),
                fmt_map(summary.get("aborts", {})),
                *(fmt_count(summary.get(key, data.get("alloc", {}).get(key))) for key in extra),
            ]) + " |")
    return "\n".join(lines) + "\n"


def warmed_notes_markdown(rows: Sequence[dict[str, object]], mode_names: Sequence[str]) -> str:
    """Why each inconclusive --warmed estimate is inconclusive."""
    lines = []
    for row in rows:
        for name in mode_names:
            check = row["modes"][name].get("warmed_check")
            if check and check["status"] == "inconclusive":
                lines.append(f"- {row['benchmark']} {name}: warmed estimate inconclusive: "
                             + "; ".join(check["reasons"]))
    return "\n".join(lines) + "\n" if lines else ""


def scales_markdown(rows: Sequence[dict[str, object]], mode_names: Sequence[str]) -> str:
    """Per-scale instructions and marginal instructions per scale unit."""
    lines = ["| Benchmark | Mode | instructions:u by scale | marginal per scale unit | status |",
             "| :-- | :-- | :-- | :-- | :-- |"]
    for row in rows:
        for name in mode_names:
            series = row["modes"][name].get("scales")
            if not series:
                continue
            points = ", ".join(f"{p['scale']}: {p['instructions']:,}" for p in series["points"])
            marginal = ", ".join(f"{m['from']}->{m['to']}: {m['instructions_per_scale']:,.0f}"
                                 for m in series["marginal"])
            status = series["status"] if not series["reasons"] else (
                f"{series['status']}: " + "; ".join(series["reasons"]))
            lines.append(f"| {row['benchmark']} | {name} | {points} | {marginal} | {status} |")
    return "\n".join(lines) + "\n" if len(lines) > 2 else ""


def oracle_markdown(rows: Sequence[dict[str, object]]) -> str:
    """Which stock interpreter checked each row; rows without one say why."""
    checked: dict[str, list[str]] = {}
    unchecked = []
    for row in rows:
        if "oracle" not in row:
            continue
        if row["oracle"]:
            checked.setdefault(str(row["oracle"]), []).append(str(row["benchmark"]))
        else:
            unchecked.append(f"{row['benchmark']} ({row.get('oracle_reason') or 'no reason recorded'})")
    lines = [f"- checked against {name}: {', '.join(names)}" for name, names in sorted(checked.items())]
    if unchecked:
        lines.append("- NO ORACLE, outputs only agree among Oximond builds: " + ", ".join(unchecked))
    return "\n".join(lines) + "\n" if lines else ""


def op_ratios(stats: Mapping[str, object]) -> dict[str, dict[str, float | None]]:
    """Derived ratios per hypothesis from one op-stats report."""
    c = stats["counters"]
    sites = stats["sites"]
    calls = c["h6.calls.same_closure"] + c["h6.calls.same_prototype_other_closure"] + c["h6.calls.other_prototype"]
    arithmetic = sites["arithmetic"]
    call_sites = sites["call"]
    hot_stable = arithmetic.get("hot_by_change", {}).get("stable", {}).get("executions")
    stable_a = arithmetic.get("stable_a_executions") or []
    return {
        "h1": {
            "range-once sites / array sites": ratio(c["h1.ir.array_sites.range_once"], c["h1.ir.array_sites"]),
            "induction sites / array sites": ratio(c["h1.ir.array_sites.induction"], c["h1.ir.array_sites"]),
            "float-key sites / array sites": ratio(c["h1.ir.array_sites.float_key"], c["h1.ir.array_sites"]),
            "range-once bound checks / bound checks": ratio(c["h1.native.bound_checks.range_once"],
                                                            c["h1.native.bound_checks"]),
            "integrality checks / bound checks": ratio(c["h1.native.integrality_checks"], c["h1.native.bound_checks"]),
        },
        "h2": {
            "scalar misses / fast dispatches": ratio(c["h2.fast.scalar_misses"], c["h2.fast.dispatches"]),
            "slow arithmetic / fast dispatches": ratio(c["h2.fast.arithmetic_slow"], c["h2.fast.dispatches"]),
            "general steps / all dispatches": ratio(c["h2.general.steps"],
                                                    c["h2.fast.dispatches"] + c["h2.general.steps"]),
            "hot arithmetic executions at stable-class sites": ratio(hot_stable, arithmetic.get("hot_executions")),
            "arithmetic executions at stable float_float sites": ratio(stable_a[0] if stable_a else None,
                                                                       arithmetic.get("executions")),
        },
        "h3": {
            "snapshot-only / snapshot values": ratio(c["h3.ir.snapshot_only_values"], c["h3.ir.snapshot_values"]),
            "snapshot-only cheap / snapshot values": ratio(c["h3.ir.snapshot_only_values.cheap_pure"],
                                                           c["h3.ir.snapshot_values"]),
            "snapshot-only cheap / snapshot-only": ratio(c["h3.ir.snapshot_only_values.cheap_pure"],
                                                         c["h3.ir.snapshot_only_values"]),
            "entries per snapshot": ratio(c["h3.ir.snapshot_entries"], c["h3.ir.snapshots"]),
            "exit slots stored per native exit": ratio(c["h3.native.exit_slots_stored"], c["h3.native.exits"]),
        },
        "h4": {
            "caller-dead argument share": ratio(c["h4.call_sites.arguments.caller_dead"], c["h4.call_sites.arguments"]),
            "caller-dead string share of caller-dead": ratio(c["h4.call_sites.arguments.caller_dead_string"],
                                                             c["h4.call_sites.arguments.caller_dead"]),
            "argument slots per activation": ratio(c["h4.activations.argument_slots"], c["h4.activations"]),
            "cleared slots per activation": ratio(c["h4.activations.cleared_slots"], c["h4.activations"]),
            "vararg-callee share of closure calls": ratio(c["h4.call_sites.vararg_callee"],
                                                          c["h4.call_sites.closure_calls"]),
            "captured-caller share of closure calls": ratio(c["h4.call_sites.captured_caller"],
                                                            c["h4.call_sites.closure_calls"]),
        },
        "h5": {
            "slots per CLEAR_RANGE": ratio(c["h5.clear_range.slots"], c["h5.clear_range.instructions"]),
            "reference share of cleared slots": ratio(c["h5.clear_range.slots.reference"], c["h5.clear_range.slots"]),
            "nil share of cleared slots": ratio(c["h5.clear_range.slots.nil"], c["h5.clear_range.slots"]),
            "dead reference slots share": ratio(c["h5.gc.frame_slots.reference_dead"],
                                                c["h5.gc.frame_slots.reference"]),
            "dead string slots share": ratio(c["h5.gc.frame_slots.string_dead"], c["h5.gc.frame_slots.string"]),
        },
        "h6": {
            "same_prototype_other_closure share of closure calls": ratio(
                c["h6.calls.same_prototype_other_closure"], calls),
            "other_prototype share of closure calls": ratio(c["h6.calls.other_prototype"], calls),
            "call-site executions where only the closure changed": ratio(
                call_sites.get("by_change", {}).get("b_changed", {}).get("executions"), call_sites.get("executions")),
        },
    }


def op_stats_markdown(rows: Sequence[dict[str, object]]) -> str:
    """Per-hypothesis counters and ratios for each benchmark and mode with op-stats."""
    sections = []
    for row in rows:
        for name, data in row["modes"].items():
            stats = data.get("op_stats")
            if not stats:
                continue
            kinds = stats.get("counter_kinds", {})
            lines = [f"### {row['benchmark']} {name} (op-stats diagnostic build)", "",
                     f"Diagnostic run instructions:u {fmt_count(stats.get('instructions'))} "
                     f"({DIAGNOSTIC_LABEL}; not comparable with the normal build). "
                     f"Build: {', '.join(stats.get('build', []))}.", "",
                     "| Hypothesis | Counter | Kind | Value |", "| :-- | :-- | :-- | --: |"]
            ratios = op_ratios(stats)
            for hypothesis, keys in KEY_OP_COUNTERS.items():
                label = f"{hypothesis} {HYPOTHESES[hypothesis]}"
                for key in keys:
                    lines.append(f"| {label} | {key} | {kinds.get(key, '?')} | {fmt_count(stats['counters'][key])} |")
                for title, value in ratios[hypothesis].items():
                    lines.append(f"| {label} | {title} | derived | {fmt_ratio(value)} |")
            pairs = stats["histograms"]["fast_opcode_pairs"]
            total = stats["histograms"].get("fast_opcode_pairs_total") or sum(pairs.values())
            top = sorted(pairs.items(), key=lambda item: (-item[1], item[0]))[:10]
            if top:
                lines += ["", "Top fast-loop opcode pairs (dynamic_fast_interp):", "",
                          "| Pair | Count | Share |", "| :-- | --: | --: |"]
                lines += [f"| {pair} | {count:,} | {fmt_ratio(ratio(count, total))} |" for pair, count in top]
            sections.append("\n".join(lines) + "\n")
    return "\n".join(sections)


def alloc_phases_markdown(rows: Sequence[dict[str, object]]) -> str:
    """Per-phase allocation tables for each benchmark and mode with an alloc-phases report."""
    sections = []
    for row in rows:
        for name, data in row["modes"].items():
            report = data.get("alloc_phases")
            if not report:
                continue
            totals = report["totals"]
            lines = [f"### {row['benchmark']} {name} (alloc-phases diagnostic build)", "",
                     f"Totals: {fmt_count(totals['alloc_count'])} allocations, {fmt_count(totals['alloc_bytes'])} "
                     f"bytes, global peak {fmt_count(totals['alloc_peak_bytes'])} bytes, "
                     f"live at the snapshot {fmt_count(totals['live_bytes'])} bytes. "
                     f"Header overhead {fmt_count(report['instrumentation']['header_bytes'])} bytes, not counted.",
                     "", "Peaks are per phase and do not sum to the global peak.", "",
                     "| Phase | allocations | other threads | bytes | active peak | origin peak | "
                     "origin live at snapshot | freed by itself | freed by other phases |",
                     "| :-- | --: | --: | --: | --: | --: | --: | --: | --: |"]
            for phase in ALLOC_PHASE_NAMES:
                entry = report["phases"][phase]
                released = entry["released_by"]
                own = released[phase]["bytes"]
                others = sum(released[other]["bytes"] for other in ALLOC_PHASE_NAMES if other != phase)
                lines.append(f"| {phase} | {fmt_count(entry['alloc_count'])} | "
                             f"{fmt_count(entry['alloc_count_other'])} | {fmt_count(entry['alloc_bytes'])} | "
                             f"{fmt_count(entry['active_peak_bytes'])} | {fmt_count(entry['origin_peak_bytes'])} | "
                             f"{fmt_count(entry['origin_live_bytes'])} | {fmt_count(own)} | {fmt_count(others)} |")
            sections.append("\n".join(lines) + "\n")
    return "\n".join(sections)


def comparison_markdown(rows: Sequence[dict[str, object]]) -> str:
    lines = ["| Benchmark | Mode | before | after | change |", "| :-- | :-- | --: | --: | --: |"]
    for row in rows:
        for mode in MODES:
            before, after = row["modes"].get(f"before-{mode}"), row["modes"].get(mode)
            if before and after:
                change = after["instructions"] / before["instructions"] - 1
                lines.append(f"| {row['benchmark']} | {mode} | {before['instructions']:,} | "
                             f"{after['instructions']:,} | {change:+.2%} |")
    return "\n".join(lines) + "\n"


def alloc_regressions(before: Mapping[str, int], after: Mapping[str, int]) -> list[tuple[str, int, int]]:
    """alloc_count and alloc_peak_bytes entries that grew from before to after."""
    return [(key, before[key], after[key]) for key in ("alloc_count", "alloc_peak_bytes")
            if key in before and key in after and after[key] > before[key]]


def gate_violations(rows: Sequence[dict[str, object]], percent: float, require_alloc: bool = False) -> list[str]:
    """Rows that fail --gate: instructions up by more than percent, or any alloc rise.

    A mode measured on one side only, a missing instruction count, or (with
    `require_alloc`) a missing allocation count is a violation too: the gate
    never passes on data it does not have.
    """
    problems = []
    for row in rows:
        for mode in MODES:
            before, after = row["modes"].get(f"before-{mode}"), row["modes"].get(mode)
            if not before and not after:
                continue
            if not before or not after:
                problems.append(f"{row['benchmark']} {mode}: measured only "
                                f"{'after' if after else 'before'}, nothing to compare")
                continue
            if not isinstance(before.get("instructions"), int) or not isinstance(after.get("instructions"), int):
                problems.append(f"{row['benchmark']} {mode}: instructions:u missing")
                continue
            if before["instructions"] and after["instructions"] > before["instructions"] * (1 + percent / 100):
                change = after["instructions"] / before["instructions"] - 1
                problems.append(f"{row['benchmark']} {mode}: instructions:u {before['instructions']:,} -> "
                                f"{after['instructions']:,} ({change:+.2%}, limit {percent:g}%)")
            if require_alloc:
                for key in ("alloc_count", "alloc_peak_bytes"):
                    for side, data in (("before", before), ("after", after)):
                        if key not in data.get("alloc", {}):
                            problems.append(f"{row['benchmark']} {mode}: {key} missing {side}")
            for key, old, new in alloc_regressions(before.get("alloc", {}), after.get("alloc", {})):
                problems.append(f"{row['benchmark']} {mode}: {key} {old:,} -> {new:,}")
    return problems


def parse_scales(text: str) -> list[int]:
    try:
        scales = [int(part) for part in text.split(",") if part.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"bad --scales {text!r}: {error}") from error
    if len(scales) < 2 or any(s < 1 for s in scales) or scales != sorted(set(scales)):
        raise argparse.ArgumentTypeError("--scales needs two or more positive, strictly increasing scales")
    return scales


def runner_git_state(source_revision: str | None = None) -> dict[str, object]:
    """The revision of the runner and its workloads.

    A runner exported with `git archive` sits inside some other checkout,
    whose state says nothing about it: then the revision is the one given
    as `source_revision`, or None.
    """
    top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=run.ROOT, capture_output=True,
                         text=True, check=False).stdout.strip()
    state: dict[str, object] = {"runner_root": str(run.ROOT)}
    if top and Path(top).resolve() == Path(run.ROOT).resolve():
        state.update(run.git_state())
        state["source"] = "checkout"
    else:
        state.update({"revision": source_revision, "dirty": None,
                      "source": "given by --source-revision" if source_revision else "export without a revision"})
    return state


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--filter", help="regular expression matched against benchmark names")
    parser.add_argument("--list", action="store_true", help="list selected benchmarks and exit")
    parser.add_argument("--scale", type=int, default=1, help="positive workload multiplier")
    parser.add_argument("--repeat", type=int, default=5, help="runs per benchmark and mode; the median is reported")
    parser.add_argument("--quick", action="store_true", help="three repeats, no warmed or multi-scale measurement")
    parser.add_argument("--cpu", type=int, help="pin each process with taskset")
    scaling = parser.add_mutually_exclusive_group()
    scaling.add_argument("--warmed", action="store_true",
                         help="also run at twice the scale and report the marginal instructions of one scale unit")
    scaling.add_argument("--scales", type=parse_scales, metavar="S1,S2,...",
                         help="run every mode at each scale (ascending; the first is the reported one) and report "
                              "marginal instructions per scale unit between consecutive scales")
    parser.add_argument("--scale-tolerance", type=float, default=10.0, metavar="PCT",
                        help="marginals of consecutive scale pairs may differ by this many percent before the "
                             "estimate is inconclusive (default 10)")
    parser.add_argument("--oximond", type=Path, default=None, help="binary under test (default: OXIMOND_BIN or target/release/oximond)")
    parser.add_argument("--before", type=Path, help="preserved binary measured in the same invocation")
    parser.add_argument("--jit-sync", action=argparse.BooleanOptionalAction, default=True,
                        help="set OXIMOND_JIT_SYNC=1 for every Oximond run so tracing counts are deterministic (default: on)")
    parser.add_argument("--alloc-oximond", type=Path,
                        help="alloc-stats build of the binary under test; run once per benchmark and mode for allocation counts")
    parser.add_argument("--alloc-before", type=Path,
                        help="alloc-stats build of the --before binary")
    parser.add_argument("--op-stats-oximond", type=Path,
                        help="op-stats diagnostic build of the binary under test; run once per benchmark and mode "
                             "for operation counts")
    parser.add_argument("--op-stats-before", type=Path,
                        help="op-stats diagnostic build of the --before binary")
    parser.add_argument("--alloc-phases-oximond", type=Path,
                        help="alloc-phases diagnostic build of the binary under test; run once per benchmark and mode "
                             "with OXIMOND_ALLOC_PHASES")
    parser.add_argument("--alloc-phases-before", type=Path,
                        help="alloc-phases diagnostic build of the --before binary")
    parser.add_argument("--gate", type=float, metavar="PCT",
                        help="exit 3 when median instructions:u rises more than PCT percent against --before, "
                             "or alloc_count/alloc_peak_bytes rise against --alloc-before")
    parser.add_argument("--mode", action="append", choices=MODES, help="mode to include; repeat for more")
    parser.add_argument("--no-reference", action="store_true", help="skip the stock Lua/LuaJIT checksum comparison")
    parser.add_argument("--timeout", type=float, default=120.0, help="per-process timeout")
    parser.add_argument("--source-revision",
                        help="revision the workloads and runner were exported from, recorded when the runner "
                             "does not run from a git checkout of its own")
    parser.add_argument("--json", type=Path, help="write machine-readable results")
    parser.add_argument("--markdown", type=Path, help="write the result table")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.scale < 1 or args.repeat < 1:
        raise RuntimeError("--scale and --repeat must be positive")
    if args.scales and args.scale != 1:
        raise RuntimeError("--scales replaces --scale; give the base scale as the first of --scales")
    if args.scale_tolerance < 0:
        raise RuntimeError("--scale-tolerance must not be negative")
    if args.gate is not None:
        if args.gate < 0:
            raise RuntimeError("--gate must not be negative")
        if not args.before:
            raise RuntimeError("--gate needs --before (and optionally --alloc-oximond with --alloc-before)")
    if args.alloc_before and not (args.before and args.alloc_oximond):
        raise RuntimeError("--alloc-before needs --before and --alloc-oximond")
    if args.op_stats_before and not (args.before and args.op_stats_oximond):
        raise RuntimeError("--op-stats-before needs --before and --op-stats-oximond")
    if args.alloc_phases_before and not (args.before and args.alloc_phases_oximond):
        raise RuntimeError("--alloc-phases-before needs --before and --alloc-phases-oximond")
    # Taken before any run, so edits made while it measures cannot leak in.
    git = runner_git_state(args.source_revision)
    benchmarks = run.load_suite(args.filter)
    if args.list:
        print("\n".join(b.name for b in benchmarks))
        return 0
    if shutil.which("perf") is None:
        raise RuntimeError("perf is not installed")
    binary = (args.oximond or Path(os.environ.get("OXIMOND_BIN", run.ROOT / "target" / "release" / "oximond"))).resolve(strict=True)
    repeat = 3 if args.quick else args.repeat
    warmed = args.warmed and not args.quick
    scales = args.scales if not args.quick else None
    selected = args.mode or list(MODES)
    modes = [Mode(m, binary, m) for m in selected]
    if args.before:
        before = args.before.resolve(strict=True)
        modes += [Mode(f"before-{m}", before, m) for m in selected]
    alloc_binaries = {
        False: args.alloc_oximond.resolve(strict=True) if args.alloc_oximond else None,
        True: args.alloc_before.resolve(strict=True) if args.alloc_before else None,
    }
    op_binaries = {
        False: args.op_stats_oximond.resolve(strict=True) if args.op_stats_oximond else None,
        True: args.op_stats_before.resolve(strict=True) if args.op_stats_before else None,
    }
    op_features: dict[bool, list[str]] = {}
    phase_binaries = {
        False: args.alloc_phases_oximond.resolve(strict=True) if args.alloc_phases_oximond else None,
        True: args.alloc_phases_before.resolve(strict=True) if args.alloc_phases_before else None,
    }
    phase_features: dict[bool, list[str]] = {}
    base_scale = scales[0] if scales else args.scale
    state = machine_state(args.cpu)
    rows = []
    for benchmark in benchmarks:
        if args.no_reference:
            oracle = Oracle(None, None, None, "--no-reference")
        else:
            oracle = reference_checksum(benchmark, base_scale, args.timeout)
        expected = oracle.checksum
        row: dict[str, object] = {"benchmark": benchmark.name, "compat": benchmark.compat,
                                  "oracle": oracle.name, "modes": {}}
        if oracle.name:
            row["oracle_path"] = oracle.path
        else:
            row["oracle_reason"] = oracle.reason
        for mode in modes:
            data, checksum = measure_mode(mode, benchmark, args.scale, repeat, args.cpu, args.timeout, warmed,
                                          args.jit_sync, scales, args.scale_tolerance)
            if expected is None:
                expected = checksum
            if checksum != expected:
                source = oracle.name or "the first Oximond mode"
                raise RuntimeError(f"checksum mismatch for {benchmark.name} in {mode.name}: {checksum!r} != "
                                   f"{expected!r} ({source})")
            is_before = mode.name.startswith("before-")
            alloc_binary = alloc_binaries[is_before]
            if alloc_binary is not None:
                sample = invoke(Mode(mode.name, alloc_binary, mode.execution), benchmark, base_scale,
                                args.cpu, args.timeout, args.jit_sync)
                if sample.stdout != checksum:
                    raise RuntimeError(f"alloc-stats build printed {sample.stdout!r} for {benchmark.name} in {mode.name}")
                data["alloc"] = {key: sample.counters[key] for key in ALLOC_KEYS if key in sample.counters}
                if not data["alloc"]:
                    raise RuntimeError(f"{alloc_binary} printed no alloc_* counters; build it with --no-default-features --features alloc-stats")
                missing = [key for key in ALLOC_KEYS if key not in data["alloc"]]
                if missing:
                    raise RuntimeError(f"{alloc_binary} printed no {', '.join(missing)}; rebuild it")
            op_binary = op_binaries[is_before]
            if op_binary is not None:
                sample, stats = invoke_op_stats(Mode(mode.name, op_binary, mode.execution), benchmark, base_scale,
                                                args.cpu, args.timeout, args.jit_sync)
                if sample.stdout != checksum:
                    raise RuntimeError(f"op-stats build printed {sample.stdout!r} for {benchmark.name} in "
                                       f"{mode.name}, the normal build {checksum!r}")
                known = op_features.setdefault(is_before, stats["build"])
                if known != stats["build"]:
                    raise RuntimeError(f"{op_binary} reported build features {stats['build']} after {known}")
                stats["instructions"] = sample.instructions
                stats["instructions_label"] = (f"{DIAGNOSTIC_LABEL}: instructions:u of the instrumented build, "
                                               "never compared with normal-build numbers")
                stats["oximond_counters"] = sample.counters
                data["op_stats"] = stats
            phase_binary = phase_binaries[is_before]
            if phase_binary is not None:
                sample, phases = invoke_alloc_phases(Mode(mode.name, phase_binary, mode.execution), benchmark,
                                                     base_scale, args.cpu, args.timeout, args.jit_sync)
                if sample.stdout != checksum:
                    raise RuntimeError(f"alloc-phases build printed {sample.stdout!r} for {benchmark.name} in "
                                       f"{mode.name}, the normal build {checksum!r}")
                known = phase_features.setdefault(is_before, phases["build"])
                if known != phases["build"]:
                    raise RuntimeError(f"{phase_binary} reported build features {phases['build']} after {known}")
                phases["instructions"] = sample.instructions
                phases["instructions_label"] = (f"{DIAGNOSTIC_LABEL}: instructions:u of the instrumented build, "
                                                "never compared with normal-build numbers")
                phases["oximond_counters"] = sample.counters
                data["alloc_phases"] = phases
            data["summary"] = summarize_counters(data["counters"])
            row["modes"][mode.name] = data
            row["checksum"] = checksum
        rows.append(row)
        print(f"measured {benchmark.name}", file=sys.stderr)
    mode_names = [m.name for m in modes]
    rendered = markdown(rows, mode_names, warmed)
    oracles = oracle_markdown(rows)
    if oracles:
        rendered += "\nOracles:\n\n" + oracles
    if warmed:
        notes = warmed_notes_markdown(rows, mode_names)
        rendered += notes and "\n" + notes
    if scales:
        rendered += "\n" + scales_markdown(rows, mode_names)
    if args.before:
        rendered += "\n" + comparison_markdown(rows)
    op_section = op_stats_markdown(rows)
    if op_section:
        rendered += "\n## Operation statistics (op-stats diagnostic builds)\n\n" + op_section
    phase_section = alloc_phases_markdown(rows)
    if phase_section:
        rendered += "\n## Allocation phases (alloc-phases diagnostic builds)\n\n" + phase_section
    print(rendered, end="")
    binaries: dict[str, object] = {"oximond": binary_metadata(binary)}
    if args.before:
        binaries["before"] = binary_metadata(args.before.resolve())
    for key, path in (("alloc_oximond", alloc_binaries[False]), ("alloc_before", alloc_binaries[True])):
        if path:
            binaries[key] = binary_metadata(path, ["alloc-stats"], "inferred: the binary printed alloc_* counters")
    for key, path, is_before in (("op_stats_oximond", op_binaries[False], False),
                                 ("op_stats_before", op_binaries[True], True)):
        if path:
            binaries[key] = binary_metadata(path, op_features.get(is_before, []),
                                            "reported: the build field of its op-stats report")
    for key, path, is_before in (("alloc_phases_oximond", phase_binaries[False], False),
                                 ("alloc_phases_before", phase_binaries[True], True)):
        if path:
            binaries[key] = binary_metadata(path, phase_features.get(is_before, []),
                                            "reported: the build field of its alloc-phases report")
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "host": state,
        "git": git,
        "scale": base_scale,
        "scales": scales,
        "scale_tolerance": args.scale_tolerance,
        "repeat": repeat,
        "warmed": warmed,
        "jit_sync": args.jit_sync,
        "gate": args.gate,
        "binaries": binaries,
        "results": rows,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        header = (f"Load {state['uptime']}; cpu {args.cpu} (sibling {state.get('sibling_cpu')}); "
                  f"oximond sha256 {metadata['binaries']['oximond']['sha256']}\n\n")
        args.markdown.write_text(header + rendered, encoding="utf-8")
    if args.gate is not None:
        problems = gate_violations(rows, args.gate, require_alloc=bool(alloc_binaries[True]))
        if problems:
            print(f"gate failed ({len(problems)} problems, limit {args.gate:g}%):", file=sys.stderr)
            for problem in problems:
                print(f"  {problem}", file=sys.stderr)
            return GATE_EXIT_STATUS
        print(f"gate passed ({args.gate:g}%)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"measure error: {error}", file=sys.stderr)
        raise SystemExit(1)
