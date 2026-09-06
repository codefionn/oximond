#!/usr/bin/env python3
"""Census of the trace IR that native compiles lower, for the MIR research.

Two steps. `collect` runs suite benchmarks with OXIMOND_JIT_CENSUS=1 and keeps
the `jit-census:` lines the compiler prints per compile. `report` aggregates a
collection into markdown tables.

    scripts/heavy.sh python3 benchmarks/trace_census.py collect OUT_DIR
    python3 benchmarks/trace_census.py report OUT_DIR [--ops NAME ...]
    python3 -m unittest benchmarks/test_trace_census.py

collect options: --bin BIN (default target/release/oximond), --scale N
(default 3), --timeout S (default 120), and benchmark names (default the whole
suite, which is what run.py's oximond-tracing runtime runs). Collection counts
nothing in time, so it needs no pinning. Compiles run on a compile thread, so
which trees get compiled before the program ends varies a little per run.

Per compile the compiler reports, per member (root first, then sides): the
trace IR operations by variant (`Variant.Sub` where the op has an `op:`
field), and per guard [window, value, const, unchanged, root, virtual, slice,
distinct snapshot values, values live across the guard, snapshot values that
only the snapshot keeps alive]. It also reports the CLIF instructions each
operation kind lowered to (children excluded) and the CLIF total of the
function. Trees are counted once per root: the last non-detached compile wins,
because a tier upgrade or a new side recompiles the whole tree.
"""
import argparse
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PREFIX = "jit-census: "
GUARD_FIELDS = ["window", "value", "const", "unchanged", "root", "virtual", "slice",
                "snap_values", "live_across", "weak_extra"]


def parse_lines(text):
    """The census records in `text`, a stderr capture."""
    records = []
    for line in text.splitlines():
        if line.startswith(PREFIX):
            try:
                records.append(json.loads(line[len(PREFIX):]))
            except json.JSONDecodeError:
                pass
    return records


def final_trees(records):
    """One record per root: the last compile that was not a detached side."""
    trees = {}
    for record in records:
        if not record.get("detached"):
            trees[record["root"]] = record
    return list(trees.values())


def variant(name):
    return name.split(".", 1)[0]


def percentile(values, fraction):
    """Nearest-rank percentile of a non-empty list."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def summarize(records):
    """Aggregate the census records of one benchmark."""
    trees = final_trees(records)
    summary = {
        "compiles": len(records),
        "detached": sum(1 for r in records if r.get("detached")),
        "trees": len(trees),
        "sides": 0,
        "ir": 0,
        "ops": Counter(),
        "guards": [],
        "loop_params": 0,
        "peak_gpr": 0,
        "peak_fpr": 0,
        "clif_total": 0,
        "clif_ops": defaultdict(lambda: [0, 0]),
        "tiers": Counter(),
    }
    for tree in trees:
        summary["tiers"][tree["tier"]] += 1
        summary["clif_total"] += tree["clif_total"]
        summary["sides"] += len(tree["members"]) - 1
        for name, (count, clif) in tree["clif_ops"].items():
            row = summary["clif_ops"][variant(name)]
            row[0] += count
            row[1] += clif
        for member in tree["members"]:
            summary["ir"] += member["pre"] + member["body"]
            summary["loop_params"] += member["loop_params"]
            summary["peak_gpr"] = max(summary["peak_gpr"], member["peak_gpr"])
            summary["peak_fpr"] = max(summary["peak_fpr"], member["peak_fpr"])
            for name, count in member["ops"].items():
                summary["ops"][variant(name)] += count
            summary["guards"].extend(member["guards"])
    return summary


def stats(values):
    if not values:
        return "-"
    return f"{sum(values) / len(values):.1f} / {percentile(values, 0.95)} / {max(values)}"


def guard_column(summary, field):
    return [g[GUARD_FIELDS.index(field)] for g in summary["guards"]]


def merge(summaries):
    total = summarize([])
    for s in summaries:
        for key in ("compiles", "detached", "trees", "sides", "ir", "loop_params", "clif_total"):
            total[key] += s[key]
        total["peak_gpr"] = max(total["peak_gpr"], s["peak_gpr"])
        total["peak_fpr"] = max(total["peak_fpr"], s["peak_fpr"])
        total["ops"].update(s["ops"])
        total["guards"].extend(s["guards"])
        total["tiers"].update(s["tiers"])
        for name, (count, clif) in s["clif_ops"].items():
            total["clif_ops"][name][0] += count
            total["clif_ops"][name][1] += clif
    return total


def coverage(ops):
    """Op variants by count, descending, with the cumulative share, and the
    number of variants that reach 95% and 99%."""
    total = sum(ops.values())
    rows, running, at95, at99 = [], 0, None, None
    for index, (name, count) in enumerate(sorted(ops.items(), key=lambda kv: (-kv[1], kv[0])), 1):
        running += count
        share = running / total
        rows.append((name, count, count / total, share))
        if at95 is None and share >= 0.95:
            at95 = index
        if at99 is None and share >= 0.99:
            at99 = index
    return rows, at95, at99


def op_table(summary):
    lines = ["| TraceOp | IR count | share | cumulative | CLIF emitted | CLIF per op |",
             "|---|---:|---:|---:|---:|---:|"]
    rows, _, _ = coverage(summary["ops"]) if summary["ops"] else ([], None, None)
    for name, count, share, cumulative in rows:
        lowered, clif = summary["clif_ops"].get(name, (0, 0))
        per = f"{clif / lowered:.1f}" if lowered else "-"
        lines.append(f"| {name} | {count} | {share:.1%} | {cumulative:.1%} | {clif} | {per} |")
    return "\n".join(lines)


def summary_table(named):
    head = ("| benchmark | compiles | trees | sides | IR insts | guards | window entries/guard "
            "(mean/p95/max) | value entries/guard | live across guard | peak gpr/fpr | CLIF | CLIF/IR |")
    lines = [head, "|---|---:|---:|---:|---:|---:|---|---|---|---|---:|---:|"]
    for name, s in named:
        ratio = f"{s['clif_total'] / s['ir']:.1f}" if s["ir"] else "-"
        lines.append(
            f"| {name} | {s['compiles']} | {s['trees']} | {s['sides']} | {s['ir']} | {len(s['guards'])} "
            f"| {stats(guard_column(s, 'window'))} | {stats(guard_column(s, 'value'))} "
            f"| {stats(guard_column(s, 'live_across'))} | {s['peak_gpr']}/{s['peak_fpr']} "
            f"| {s['clif_total']} | {ratio} |")
    return "\n".join(lines)


def load(directory):
    named = []
    for path in sorted(Path(directory).glob("*.jsonl")):
        records = [json.loads(line) for line in path.read_text().splitlines() if line]
        named.append((path.stem.replace("__", "/"), summarize(records)))
    return named


def report(directory, ops_for):
    named = load(directory)
    total = merge([s for _, s in named])
    print(f"Benchmarks: {len(named)}. Trees (last compile per root): {total['trees']}, "
          f"sides: {total['sides']}, trace IR instructions: {total['ir']}, "
          f"guards: {len(total['guards'])}, tiers: {dict(total['tiers'])}.\n")
    print("## Per benchmark\n")
    print(summary_table(named + [("all", total)]))
    print("\n## Guard snapshot detail, all benchmarks (mean / p95 / max)\n")
    print("| field | mean / p95 / max |\n|---|---|")
    for field in GUARD_FIELDS:
        print(f"| {field} | {stats(guard_column(total, field))} |")
    print("\n## All benchmarks, by TraceOp\n")
    print(op_table(total))
    _, at95, at99 = coverage(total["ops"])
    attributed = sum(clif for _, clif in total["clif_ops"].values())
    print(f"\nCLIF attributed to operations: {attributed} of {total['clif_total']} "
          f"({attributed / max(1, total['clif_total']):.1%}); the rest is entry, loop, "
          f"exit and stub code outside any operation.")
    print(f"\nVariants for 95% of trace IR instructions: {at95}. For 99%: {at99}. "
          f"Of {len(total['ops'])} variants seen.")
    # Coverage where each benchmark counts equally, so a few large ones do not decide.
    shares = defaultdict(float)
    live = [s for _, s in named if s["ir"]]
    for s in live:
        for name, count in s["ops"].items():
            shares[name] += count / s["ir"]
    equal = Counter({k: v / len(live) for k, v in shares.items()})
    rows, e95, e99 = coverage(equal)
    print(f"With each benchmark weighted equally: 95% needs {e95} variants, 99% needs {e99}.")
    for name in ops_for:
        for label, s in named:
            if label == name:
                print(f"\n## {label}, by TraceOp\n")
                print(op_table(s))
    return 0


def collect(args):
    suite = json.load(open(ROOT / "benchmarks" / "suite.json"))
    wanted = args.names or [e["name"] for e in suite]
    by_name = {e["name"]: e for e in suite}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in wanted:
        entry = by_name[name]
        compat = entry.get("compat", "lua51")
        command = [args.bin, f"--compat={compat}", "--execution=tracing", "-E",
                   str(ROOT / "benchmarks" / entry["path"]), str(args.scale)]
        env = {"PATH": "/usr/bin:/bin", "OXIMOND_JIT_CENSUS": "1"}
        try:
            run = subprocess.run(command, env=env, capture_output=True, timeout=args.timeout,
                                 cwd=ROOT, text=True, errors="replace")
            text, status = run.stderr, run.returncode
        except subprocess.TimeoutExpired as expired:
            text, status = (expired.stderr or b"").decode(errors="replace"), "timeout"
        lines = [l for l in text.splitlines() if l.startswith(PREFIX)]
        (out / (name.replace("/", "__") + ".jsonl")).write_text(
            "\n".join(l[len(PREFIX):] for l in lines) + ("\n" if lines else ""))
        print(f"{name}: {len(lines)} compiles, status {status}", file=sys.stderr)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("out")
    c.add_argument("names", nargs="*")
    c.add_argument("--bin", default=str(ROOT / "target/release/oximond"))
    c.add_argument("--scale", type=int, default=3)
    c.add_argument("--timeout", type=int, default=120)
    r = sub.add_parser("report")
    r.add_argument("dir")
    r.add_argument("--ops", nargs="*", default=[])
    args = parser.parse_args(argv)
    return collect(args) if args.cmd == "collect" else report(args.dir, args.ops)


if __name__ == "__main__":
    sys.exit(main())
