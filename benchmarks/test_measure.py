#!/usr/bin/env python3
"""Tests for measure.py.

The unit tests run no binaries. Two groups run real programs and are off by
default:

  * MEASURE_CHECKSUMS=1 runs the opt/ workloads under stock Lua and an
    Oximond binary (debug or release) and compares their output;
  * MEASURE_LIVE=1 runs measure.py under perf against target/release/oximond.
    It must run through the gate, which sets OXIMOND_HEAVY_ACTIVE:
    MEASURE_LIVE=1 scripts/heavy.sh -x python3 -m unittest benchmarks/test_measure.py
"""

from pathlib import Path
import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BINARY = ROOT / "target" / "release" / "oximond"
OP_STATS_SOURCE = ROOT / "crates" / "oximond" / "src" / "runtime" / "op_stats.rs"

PERF = "# started on Mon\n\n361161081,,instructions:u,17313175,100.00,,\n"
BASE_COUNTERS = {
    "tables_created": 24, "elided_tables": 3, "elided_table_bytes": 960, "traces_compiled": 1,
    "native_entries": 2, "iterations": 99, "recording_starts": 1, "traces_recorded": 1,
    "loads_forwarded": 4, "loads_merged": 2, "loads_hoisted": 1,
}
COUNTERS = ("oximond-counters: " + " ".join(f"{k}={v}" for k, v in BASE_COUNTERS.items())
            + " abort.UnsupportedOperand=3 exit.LoopComplete=1 side_exit.Type=4")
NEW_COUNTERS = COUNTERS + " alloc_count=500 alloc_bytes=40000 alloc_peak_bytes=9000"


def counters(**overrides):
    return {**BASE_COUNTERS, **overrides}


class ParsingTests(unittest.TestCase):
    def test_perf_line(self):
        self.assertEqual(measure.parse_perf(PERF), 361161081)

    def test_perf_not_counted(self):
        with self.assertRaises(RuntimeError):
            measure.parse_perf("<not counted>,,instructions:u,0,100.00,,\n")

    def test_counters_line(self):
        parsed = measure.parse_counters("noise\n" + COUNTERS + "\n")
        self.assertEqual(parsed["tables_created"], 24)
        self.assertEqual(parsed["abort.UnsupportedOperand"], 3)
        summary = measure.summarize_counters(parsed)
        self.assertEqual(summary["exits"], {"LoopComplete": 1})
        self.assertEqual(summary["side_exits"], {"Type": 4})
        self.assertEqual(summary["elided_table_bytes"], 960)
        self.assertEqual(summary["recording_starts"], 1)

    def test_missing_line_names_binary(self):
        with self.assertRaisesRegex(RuntimeError, r"/old/oximond printed no oximond-counters: line.*predates"):
            measure.parse_counters("nothing here\n", "/old/oximond")

    def test_missing_required_keys(self):
        line = COUNTERS.replace(" recording_starts=1", "").replace(" loads_hoisted=1", "")
        with self.assertRaises(RuntimeError) as caught:
            measure.parse_counters(line, "/old/oximond")
        message = str(caught.exception)
        self.assertIn("/old/oximond predates", message)
        self.assertIn("recording_starts", message)
        self.assertIn("loads_hoisted", message)
        self.assertNotIn("tables_created", message)

    def test_line_without_tables_created(self):
        with self.assertRaisesRegex(RuntimeError, "tables_created"):
            measure.parse_counters(COUNTERS.replace("tables_created=24 ", ""), "b")

    def test_summarize_refuses_missing_keys(self):
        with self.assertRaisesRegex(RuntimeError, "tables_created"):
            measure.summarize_counters({})

    def test_median(self):
        samples = [measure.Sample(i, counters(tables_created=i // 10), "x") for i in (100, 300, 200)]
        result = measure.aggregate(samples)
        self.assertEqual(result["instructions"], 200)
        self.assertEqual(result["counters"]["tables_created"], 20)

    def test_partial_key_across_samples(self):
        samples = [measure.Sample(1, counters(alloc_count=5), "x"), measure.Sample(1, counters(), "x")]
        with self.assertRaisesRegex(RuntimeError, "alloc_count is missing from 1 of 2"):
            measure.aggregate(samples)

    def test_nonzero_only_keys_default_to_zero(self):
        samples = [measure.Sample(1, counters(**{"abort.X": 4}), "x"), measure.Sample(1, counters(), "x"),
                   measure.Sample(1, counters(**{"abort.X": 2}), "x")]
        self.assertEqual(measure.aggregate(samples)["counters"]["abort.X"], 2)


class NewKeyTests(unittest.TestCase):
    def test_optimizer_and_alloc_keys(self):
        summary = measure.summarize_counters(measure.parse_counters(NEW_COUNTERS))
        self.assertEqual(summary["loads_forwarded"], 4)
        self.assertEqual(summary["loads_merged"], 2)
        self.assertEqual(summary["loads_hoisted"], 1)
        self.assertEqual(summary["alloc_count"], 500)
        self.assertEqual(summary["alloc_peak_bytes"], 9000)

    def test_alloc_keys_absent_stay_absent(self):
        summary = measure.summarize_counters(measure.parse_counters(COUNTERS))
        for key in measure.ALLOC_KEYS:
            self.assertNotIn(key, summary)

    def test_markdown_columns(self):
        def row(parsed, alloc=None):
            data = {"instructions": 10, "summary": measure.summarize_counters(parsed)}
            if alloc:
                data["alloc"] = alloc
            return {"benchmark": "b", "modes": {"interpreter": data}}

        plain = measure.markdown([row(measure.parse_counters(COUNTERS))], ["interpreter"], False)
        self.assertIn("loads_hoisted", plain)
        self.assertNotIn("alloc_count", plain)
        full = measure.markdown([row(measure.parse_counters(NEW_COUNTERS))], ["interpreter"], False)
        header = full.splitlines()[0]
        self.assertTrue(header.rstrip(" |").endswith("alloc_peak_bytes"))
        self.assertEqual(len(header.split("|")), len(full.splitlines()[1].split("|")))
        self.assertEqual(len(header.split("|")), len(full.splitlines()[2].split("|")))
        alloc_only = measure.markdown([row(counters(), {"alloc_count": 7})], ["interpreter"], False)
        self.assertIn("| 7 |", alloc_only.splitlines()[2] + " ")


class EnvironmentTests(unittest.TestCase):
    def test_jit_sync_on(self):
        with mock.patch.dict(os.environ, {"OXIMOND_JIT_SYNC": "0"}):
            self.assertEqual(measure.oximond_environment(True)["OXIMOND_JIT_SYNC"], "1")

    def test_jit_sync_off_clears_inherited_value(self):
        with mock.patch.dict(os.environ, {"OXIMOND_JIT_SYNC": "1"}):
            environment = measure.oximond_environment(False)
        self.assertNotIn("OXIMOND_JIT_SYNC", environment)
        self.assertEqual(environment["OXIMOND_COUNTERS"], "1")

    def test_inherited_op_stats_path_is_cleared(self):
        with mock.patch.dict(os.environ, {"OXIMOND_OP_STATS": "/tmp/x"}):
            self.assertNotIn("OXIMOND_OP_STATS", measure.oximond_environment(True))
            self.assertEqual(measure.oximond_environment(True, {"OXIMOND_OP_STATS": "/r"})["OXIMOND_OP_STATS"], "/r")

    def test_flag_default_and_negation(self):
        self.assertTrue(measure.parse_args([]).jit_sync)
        self.assertFalse(measure.parse_args(["--no-jit-sync"]).jit_sync)
        self.assertTrue(measure.parse_args(["--jit-sync"]).jit_sync)

    def test_invoke_passes_environment(self):
        seen = {}

        def fake_run(command, **kwargs):
            seen.update(kwargs["env"])
            output = Path(command[command.index("-o") + 1])
            output.write_text(PERF)
            return subprocess.CompletedProcess(command, 0, b"1\n", COUNTERS.encode())

        benchmark = measure.run.load_suite("^opt/invariant_field$")[0]
        mode = measure.Mode("tracing", Path("/bin/true"), "tracing")
        with mock.patch.object(measure.subprocess, "run", fake_run):
            measure.invoke(mode, benchmark, 1, None, 1.0, True)
            self.assertEqual(seen["OXIMOND_JIT_SYNC"], "1")
            seen.clear()
            with mock.patch.dict(os.environ, {"OXIMOND_JIT_SYNC": "1"}):
                measure.invoke(mode, benchmark, 1, None, 1.0, False)
            self.assertNotIn("OXIMOND_JIT_SYNC", seen)

    def test_invoke_refuses_old_binary(self):
        def fake_run(command, **kwargs):
            Path(command[command.index("-o") + 1]).write_text(PERF)
            return subprocess.CompletedProcess(command, 0, b"1\n", b"")

        benchmark = measure.run.load_suite("^opt/invariant_field$")[0]
        with mock.patch.object(measure.subprocess, "run", fake_run):
            with self.assertRaisesRegex(RuntimeError, "/old/bin printed no"):
                measure.invoke(measure.Mode("tracing", Path("/old/bin"), "tracing"), benchmark, 1, None, 1.0)


def legacy_op_stats_document():
    """A schema 1 report: none of the schema 2 counters."""
    document = op_stats_document(schema="oximond-op-stats/1")
    for name in measure.V2_OP_COUNTERS:
        del document["counters"][name]
        del document["counter_info"][name]
    return document


def op_stats_document(**overrides):
    names = [name for group in measure.REQUIRED_OP_COUNTERS.values() for name in group]
    values = {name: 0 for name in names}
    values.update({
        "h1.ir.array_sites": 4, "h1.ir.array_sites.range_once": 1,
        "h4.call_sites.arguments": 10, "h4.call_sites.arguments.caller_dead": 4,
        "h3.ir.snapshot_values": 20, "h3.ir.snapshot_only_values": 8,
        "h3.ir.snapshot_only_values.cheap_pure": 5,
        "h6.calls.same_closure": 6, "h6.calls.same_prototype_other_closure": 3, "h6.calls.other_prototype": 1,
        "h5.gc.frame_slots.reference": 8, "h5.gc.frame_slots.reference_dead": 2,
    })
    site = {"kind": "dynamic_fast_interp", "sites": 1, "executions": 10, "overflow": 0, "hot_sites": 0,
            "hot_executions": 0, "a_changes": 0, "b_changes": 0,
            "by_change": {k: {"sites": 0, "executions": 0} for k in ("stable", "b_changed", "a_changed", "both_changed")},
            "hot_by_change": {k: {"sites": 0, "executions": 0} for k in ("stable", "b_changed", "a_changed", "both_changed")},
            "stable_a_executions": [0] * 16, "flag_executions": [0] * 8}
    document = {
        "schema": "oximond-op-stats/2",
        "build": ["op-stats"],
        "execution": "interpreter",
        "jit_sync": "on",
        "counters": values,
        "counter_info": {name: {"kind": "static_ir" if ".ir." in name else "dynamic_fast_interp", "help": "h"}
                         for name in names},
        "histograms": {
            "fast_opcodes": {"Add": 50, "ForLoop": 30},
            "general_opcodes": {},
            "fast_opcode_pairs": {f"Op{i}>Op{i + 1}": 100 - i for i in range(60)},
            "operand_classes": {"Add.float_float": 50},
        },
        "histogram_info": {},
        "sites": {"arithmetic": dict(site), "field": dict(site), "call": dict(site)},
    }
    document.update(overrides)
    return document


class OpStatsValidationTests(unittest.TestCase):
    def test_good_document(self):
        stats = measure.validate_op_stats(op_stats_document(), "/diag", "interpreter", True)
        self.assertEqual(stats["build"], ["op-stats"])
        self.assertEqual(stats["counters"]["h1.ir.array_sites"], 4)
        self.assertEqual(stats["counter_kinds"]["h1.ir.array_sites"], "static_ir")
        pairs = stats["histograms"]["fast_opcode_pairs"]
        self.assertEqual(len(pairs), measure.OP_STATS_TOP_PAIRS)
        self.assertIn("Op0>Op1", pairs)
        self.assertNotIn("Op59>Op60", pairs)
        self.assertEqual(stats["histograms"]["fast_opcode_pairs_distinct"], 60)
        self.assertEqual(stats["histograms"]["fast_opcodes"], {"Add": 50, "ForLoop": 30})
        self.assertEqual(stats["report_fields"]["execution"], "interpreter")

    def test_wrong_schema(self):
        with self.assertRaisesRegex(RuntimeError, "/diag wrote op-stats schema 'oximond-op-stats/0'"):
            measure.validate_op_stats(op_stats_document(schema="oximond-op-stats/0"), "/diag")

    def test_missing_counter(self):
        document = op_stats_document()
        del document["counters"]["h3.ir.snapshot_only_values.cheap_pure"]
        with self.assertRaisesRegex(RuntimeError, "/diag predates op-stats counters h3.ir.snapshot_only_values.cheap_pure"):
            measure.validate_op_stats(document, "/diag")

    def test_missing_build_feature(self):
        with self.assertRaisesRegex(RuntimeError, "without 'op-stats'"):
            measure.validate_op_stats(op_stats_document(build=["alloc-stats"]), "/diag")

    def test_missing_kind(self):
        document = op_stats_document()
        del document["counter_info"]["h6.closures_created"]
        with self.assertRaisesRegex(RuntimeError, "without a kind: h6.closures_created"):
            measure.validate_op_stats(document, "/diag")

    def test_missing_histogram_and_sites(self):
        document = op_stats_document()
        del document["histograms"]["fast_opcode_pairs"]
        with self.assertRaisesRegex(RuntimeError, "histograms"):
            measure.validate_op_stats(document, "/diag")
        document = op_stats_document()
        del document["sites"]["call"]
        with self.assertRaisesRegex(RuntimeError, "site tables"):
            measure.validate_op_stats(document, "/diag")

    def test_execution_and_jit_sync_must_match(self):
        with self.assertRaisesRegex(RuntimeError, "execution 'interpreter', expected 'tracing'"):
            measure.validate_op_stats(op_stats_document(), "/diag", "tracing", True)
        with self.assertRaisesRegex(RuntimeError, "jit_sync 'on', expected off"):
            measure.validate_op_stats(op_stats_document(), "/diag", "interpreter", False)

    def test_requested_execution_and_jit_sync_must_be_reported(self):
        for field in ("execution", "jit_sync"):
            document = op_stats_document()
            del document[field]
            with self.assertRaisesRegex(RuntimeError, f"reports {field} None"):
                measure.validate_op_stats(document, "/diag", "interpreter", True)

    def test_unknown_kind(self):
        document = op_stats_document()
        document["counter_info"]["h6.closures_created"]["kind"] = "dynamic_guess"
        with self.assertRaisesRegex(RuntimeError, "unknown kind: h6.closures_created"):
            measure.validate_op_stats(document, "/diag")

    def test_kinds_match_registry(self):
        if not OP_STATS_SOURCE.is_file():
            self.skipTest("needs crates/oximond/src/runtime/op_stats.rs")
        names = set(re.findall(r'Self::\w+ => "([a-z_]+)"', OP_STATS_SOURCE.read_text(encoding="utf-8")))
        self.assertLessEqual(measure.OP_COUNTER_KINDS, names)

    def test_missing_report_file(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "/diag printed no op-stats report; build it with --features op-stats"):
                measure.load_op_stats(Path(directory) / "none.json", "/diag")
            bad = Path(directory) / "bad.json"
            bad.write_text("{not json")
            with self.assertRaisesRegex(RuntimeError, "not JSON"):
                measure.load_op_stats(bad, "/diag")

    def test_schema_two_requires_the_recipe_counters(self):
        self.assertEqual(measure.OP_STATS_SCHEMA, "oximond-op-stats/2")
        for name in sorted(measure.V2_OP_COUNTERS):
            document = op_stats_document()
            del document["counters"][name]
            with self.assertRaisesRegex(RuntimeError, "predates op-stats counters.*" + re.escape(name)):
                measure.validate_op_stats(document, "/diag")
            document = op_stats_document()
            document["counter_info"][name] = {"kind": "static_ir", "help": "h"}
            document["counters"][name] = 0
            measure.validate_op_stats(document, "/diag")
            del document["counter_info"][name]
            with self.assertRaisesRegex(RuntimeError, "without a kind"):
                measure.validate_op_stats(document, "/diag")

    def test_legacy_report_fails_the_current_requirements(self):
        document = legacy_op_stats_document()
        with self.assertRaisesRegex(RuntimeError, "schema 'oximond-op-stats/1', expected 'oximond-op-stats/2'"):
            measure.validate_op_stats(document, "/old", allow_legacy=False)
        # A schema 2 label does not excuse the missing counters either.
        relabelled = legacy_op_stats_document()
        relabelled["schema"] = "oximond-op-stats/2"
        with self.assertRaisesRegex(RuntimeError, "predates op-stats counters"):
            measure.validate_op_stats(relabelled, "/old", allow_legacy=True)

    def test_legacy_report_loads_with_new_counters_unavailable(self):
        stats = measure.validate_op_stats(legacy_op_stats_document(), "/old", allow_legacy=True)
        self.assertTrue(stats["legacy"])
        self.assertEqual(stats["schema"], "oximond-op-stats/1")
        changed = measure.V2_OP_COUNTERS | measure.V2_CHANGED_OP_COUNTERS
        self.assertEqual(stats["unavailable_counters"], sorted(changed))
        for name in changed:
            self.assertIsNone(stats["counters"][name], name)
        # Schema 2 changed what the snapshot value classes count, so the
        # legacy 20 does not carry over.
        self.assertIsNone(stats["counters"]["h3.ir.snapshot_values"])
        self.assertIsNotNone(stats["counters"]["h3.ir.snapshots"])
        fresh = measure.validate_op_stats(op_stats_document(), "/diag")
        self.assertFalse(fresh["legacy"])
        self.assertEqual(fresh["unavailable_counters"], [])
        self.assertEqual(fresh["counters"]["h3.ir.snapshot_virtuals"], 0)
        # It renders without a number the report does not have.
        row = {"benchmark": "b", "modes": {"interpreter": {"instructions": 1, "op_stats": stats}}}
        text = measure.op_stats_markdown([row])
        self.assertIn("| h3 recovery recipes | h3.ir.snapshot_virtuals | ? | n/a |", text)

    def test_load_op_stats_reads_legacy_only_on_request(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.json"
            path.write_text(json.dumps(legacy_op_stats_document()))
            with self.assertRaisesRegex(RuntimeError, "expected 'oximond-op-stats/2'"):
                measure.load_op_stats(path, "/old")
            self.assertTrue(measure.load_op_stats(path, "/old", allow_legacy=True)["legacy"])

    def test_required_counters_match_registry(self):
        if not OP_STATS_SOURCE.is_file():
            self.skipTest("needs crates/oximond/src/runtime/op_stats.rs")
        registry = set(re.findall(r'=\s*"(h[1-6]\.[a-z_.]+)"', OP_STATS_SOURCE.read_text(encoding="utf-8")))
        required = {name for group in measure.REQUIRED_OP_COUNTERS.values() for name in group}
        self.assertEqual(required - registry, set(), "measure.py requires counters op_stats.rs does not define")
        self.assertEqual(registry - required, set(), "op_stats.rs defines counters measure.py does not require")
        for hypothesis, names in measure.REQUIRED_OP_COUNTERS.items():
            self.assertTrue(all(name.startswith(hypothesis + ".") for name in names), hypothesis)
        for hypothesis, names in measure.KEY_OP_COUNTERS.items():
            self.assertTrue(set(names) <= set(measure.REQUIRED_OP_COUNTERS[hypothesis]), hypothesis)


class OpStatsMarkdownTests(unittest.TestCase):
    def test_renders_counters_ratios_and_label(self):
        stats = measure.validate_op_stats(op_stats_document(), "/diag")
        stats["instructions"] = 123456
        row = {"benchmark": "opt/call_args", "modes": {"interpreter": {"instructions": 1, "op_stats": stats},
                                                      "tracing": {"instructions": 1}}}
        text = measure.op_stats_markdown([row])
        self.assertIn("### opt/call_args interpreter", text)
        self.assertNotIn("opt/call_args tracing", text)
        self.assertIn("123,456 (diagnostic build overhead", text)
        self.assertIn("| h1 loop range proofs | h1.ir.array_sites | static_ir | 4 |", text)
        self.assertIn("| caller-dead argument share | derived | 0.400 |", text)
        self.assertIn("| range-once sites / array sites | derived | 0.250 |", text)
        self.assertIn("| snapshot-only cheap / snapshot values | derived | 0.250 |", text)
        self.assertIn("| same_prototype_other_closure share of closure calls | derived | 0.300 |", text)
        self.assertIn("| dead reference slots share | derived | 0.250 |", text)
        self.assertIn("| Op0>Op1 | 100 |", text)
        self.assertEqual(text.count("| Op"), 10)
        # Zero denominators print n/a, never a made-up zero.
        self.assertIn("| integrality checks / bound checks | derived | n/a |", text)

    def test_no_section_without_op_stats(self):
        self.assertEqual(measure.op_stats_markdown([{"benchmark": "b", "modes": {"tracing": {"instructions": 1}}}]), "")


def alloc_phases_document(**overrides):
    """A consistent report: 10 allocations of 100 bytes per phase, half freed by execution."""
    phases = {}
    for name in measure.ALLOC_PHASE_NAMES:
        released = {other: {"count": 0, "bytes": 0} for other in measure.ALLOC_PHASE_NAMES}
        released["execution"] = {"count": 5, "bytes": 500}
        phases[name] = {"alloc_count": 10, "alloc_bytes": 1000, "alloc_count_vm": 8, "alloc_count_other": 2,
                        "alloc_bytes_vm": 800, "alloc_bytes_other": 200, "active_peak_bytes": 4000,
                        "origin_live_bytes": 500, "origin_peak_bytes": 900, "released_by": released}
    count = 10 * len(phases)
    document = {
        "schema": "oximond-alloc-phases/1",
        "build": ["alloc-stats", "alloc-phases"],
        "context": {"execution": "tracing", "jit_sync": "on"},
        "totals": {"alloc_count": count, "alloc_bytes": 100 * count, "alloc_peak_bytes": 4000,
                   "live_bytes": 50 * count},
        "reconciled": True,
        "instrumentation": {"header_bytes": 16 * count, "header_bytes_min": 16},
        "phases": phases,
    }
    document.update(overrides)
    return document


class AllocPhasesValidationTests(unittest.TestCase):
    def test_good_document(self):
        report = measure.validate_alloc_phases(alloc_phases_document(), "/diag", "tracing", True)
        self.assertEqual(report["build"], ["alloc-stats", "alloc-phases"])
        self.assertEqual(report["totals"]["alloc_count"], 80)
        self.assertEqual(report["phases"]["codegen"]["alloc_bytes"], 1000)
        self.assertEqual(report["phases"]["recording"]["released_by"]["execution"], {"count": 5, "bytes": 500})
        self.assertEqual(report["instrumentation"]["header_bytes"], 1280)

    def test_wrong_schema(self):
        with self.assertRaisesRegex(RuntimeError, "/diag wrote alloc-phases schema 'oximond-alloc-phases/0'"):
            measure.validate_alloc_phases(alloc_phases_document(schema="oximond-alloc-phases/0"), "/diag")

    def test_missing_build_feature(self):
        with self.assertRaisesRegex(RuntimeError, "without 'alloc-phases'"):
            measure.validate_alloc_phases(alloc_phases_document(build=["alloc-stats"]), "/diag")

    def test_execution_and_jit_sync_must_match(self):
        with self.assertRaisesRegex(RuntimeError, "execution 'tracing', expected 'interpreter'"):
            measure.validate_alloc_phases(alloc_phases_document(), "/diag", "interpreter")
        with self.assertRaisesRegex(RuntimeError, "jit_sync 'on', expected off"):
            measure.validate_alloc_phases(alloc_phases_document(), "/diag", jit_sync=False)

    def test_missing_phase(self):
        document = alloc_phases_document()
        del document["phases"]["exit_recovery"]
        with self.assertRaisesRegex(RuntimeError, "without the phases exit_recovery"):
            measure.validate_alloc_phases(document, "/diag")

    def test_unknown_phase(self):
        document = alloc_phases_document()
        document["phases"]["parsing"] = dict(document["phases"]["report"])
        with self.assertRaisesRegex(RuntimeError, "unknown alloc phases parsing"):
            measure.validate_alloc_phases(document, "/diag")

    def test_counts_must_be_non_negative_integers(self):
        for value in (True, -1, 1.5, "3", None):
            document = alloc_phases_document()
            document["phases"]["codegen"]["origin_peak_bytes"] = value
            with self.assertRaisesRegex(RuntimeError, "codegen without counts for origin_peak_bytes"):
                measure.validate_alloc_phases(document, "/diag")
        document = alloc_phases_document()
        del document["totals"]["live_bytes"]
        with self.assertRaisesRegex(RuntimeError, "totals without counts"):
            measure.validate_alloc_phases(document, "/diag")

    def test_released_by_needs_every_phase(self):
        document = alloc_phases_document()
        del document["phases"]["codegen"]["released_by"]["report"]
        with self.assertRaisesRegex(RuntimeError, "codegen without released_by"):
            measure.validate_alloc_phases(document, "/diag")

    def test_thread_split_must_add_up(self):
        document = alloc_phases_document()
        document["phases"]["recording"]["alloc_count_other"] = 3
        with self.assertRaisesRegex(RuntimeError, "recording whose thread split"):
            measure.validate_alloc_phases(document, "/diag")

    def test_phases_must_reconcile_with_totals(self):
        document = alloc_phases_document()
        document["totals"]["alloc_count"] += 1
        with self.assertRaisesRegex(RuntimeError, "do not reconcile with its totals"):
            measure.validate_alloc_phases(document, "/diag")
        document = alloc_phases_document()
        document["phases"]["report"]["released_by"]["report"]["bytes"] = 7
        with self.assertRaisesRegex(RuntimeError, "do not reconcile"):
            measure.validate_alloc_phases(document, "/diag")
        with self.assertRaisesRegex(RuntimeError, "says its alloc phases do not reconcile"):
            measure.validate_alloc_phases(alloc_phases_document(reconciled=False), "/diag")

    def test_missing_report_file(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "printed no alloc-phases report; build it with --no-default-features --features alloc-phases"):
                measure.load_alloc_phases(Path(directory) / "none.json", "/diag")
            bad = Path(directory) / "bad.json"
            bad.write_text("{not json")
            with self.assertRaisesRegex(RuntimeError, "not JSON"):
                measure.load_alloc_phases(bad, "/diag")

    def test_phase_names_match_registry(self):
        source = Path(__file__).resolve().parents[1] / "crates/oximond/src/runtime/alloc_phase.rs"
        if not source.is_file():
            self.skipTest("needs crates/oximond/src/runtime/alloc_phase.rs")
        names = tuple(re.findall(r'Self::\w+ => "([a-z_]+)"', source.read_text(encoding="utf-8")))
        self.assertEqual(names, measure.ALLOC_PHASE_NAMES)

    def test_inherited_report_path_is_cleared(self):
        with mock.patch.dict(os.environ, {"OXIMOND_ALLOC_PHASES": "/tmp/x"}):
            self.assertNotIn("OXIMOND_ALLOC_PHASES", measure.oximond_environment(True))

    def test_before_needs_both_binaries(self):
        with self.assertRaisesRegex(RuntimeError, "--alloc-phases-before needs --before and --alloc-phases-oximond"):
            measure.main(["--alloc-phases-before", "/x"])

    def test_markdown(self):
        report = measure.validate_alloc_phases(alloc_phases_document(), "/diag")
        row = {"benchmark": "alloc/hash_read", "modes": {"tracing": {"instructions": 1, "alloc_phases": report},
                                                         "interpreter": {"instructions": 1}}}
        text = measure.alloc_phases_markdown([row])
        self.assertIn("### alloc/hash_read tracing (alloc-phases diagnostic build)", text)
        self.assertNotIn("hash_read interpreter", text)
        self.assertIn("do not sum to the global peak", text)
        self.assertIn("| codegen | 10 | 2 | 1,000 | 4,000 | 900 | 500 | 0 | 500 |", text)
        self.assertIn("| execution | 10 | 2 | 1,000 | 4,000 | 900 | 500 | 500 | 0 |", text)
        self.assertEqual(measure.alloc_phases_markdown([{"benchmark": "b", "modes": {"tracing": {}}}]), "")


def aggregate_at(instructions, **plan):
    return {"instructions": instructions, "counters": counters(**plan)}


class GitStateTests(unittest.TestCase):
    def test_an_export_inside_another_checkout_takes_the_given_revision(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(measure.run, "ROOT", Path(directory)):
            state = measure.runner_git_state("abc123")
        self.assertEqual(state["revision"], "abc123")
        self.assertIsNone(state["dirty"])
        self.assertEqual(state["source"], "given by --source-revision")

    def test_the_checkout_reports_its_own_state(self):
        top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=measure.run.ROOT,
                             capture_output=True, text=True, check=False).stdout.strip()
        if not top or Path(top).resolve() != Path(measure.run.ROOT).resolve():
            self.skipTest("the runner is not a git checkout of its own (an export)")
        state = measure.runner_git_state("ignored")
        self.assertEqual(state["source"], "checkout")
        self.assertNotEqual(state["revision"], "ignored")


class ScaleTests(unittest.TestCase):
    def test_linear_series(self):
        series = measure.scale_series([(1, aggregate_at(1100)), (2, aggregate_at(2100)), (4, aggregate_at(4100))], 10)
        self.assertEqual(series["status"], "ok")
        self.assertEqual([m["instructions_per_scale"] for m in series["marginal"]], [1000, 1000])
        self.assertEqual(series["points"][2]["trace_plan"]["traces_compiled"], 1)

    def test_trace_plan_change_is_inconclusive(self):
        series = measure.scale_series([(1, aggregate_at(1100)), (2, aggregate_at(2100, traces_compiled=2))], 10)
        self.assertEqual(series["status"], "inconclusive")
        self.assertIn("traces_compiled 1->2", series["reasons"][0])

    def test_abort_total_change_is_inconclusive(self):
        series = measure.scale_series([(1, aggregate_at(1100)), (2, aggregate_at(2100, **{"abort.ShortLoop": 1}))], 10)
        self.assertIn("aborts 0->1", series["reasons"][0])

    def test_abort_reasons_that_trade_places_are_inconclusive(self):
        series = measure.scale_series([(1, aggregate_at(1100, **{"abort.ShortLoop": 1})),
                                       (2, aggregate_at(2100, **{"abort.InnerLoop": 1}))], 10)
        self.assertEqual(series["status"], "inconclusive")
        self.assertIn("abort.InnerLoop 0->1, abort.ShortLoop 1->0", series["reasons"][0])

    def test_marginal_spread_is_inconclusive(self):
        series = measure.scale_series([(1, aggregate_at(1100)), (2, aggregate_at(2100)), (3, aggregate_at(3300))], 10)
        self.assertEqual(series["status"], "inconclusive")
        self.assertIn("differ by 16.7%", series["reasons"][0])
        self.assertEqual(measure.scale_series(
            [(1, aggregate_at(1100)), (2, aggregate_at(2100)), (3, aggregate_at(3300))], 20)["status"], "ok")

    def test_non_positive_marginal(self):
        series = measure.scale_series([(1, aggregate_at(1100)), (2, aggregate_at(1000))], 10)
        self.assertIn("not positive", series["reasons"][0])

    def test_parse_scales(self):
        self.assertEqual(measure.parse_args(["--scales", "1,2,4"]).scales, [1, 2, 4])
        for bad in ("1", "2,1", "1,1", "0,1", "a,b"):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                measure.parse_args(["--scales", bad])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            measure.parse_args(["--scales", "1,2", "--warmed"])

    def fake_invoke(self, plan_by_scale=None, output=None):
        def invoke(mode, benchmark, scale, cpu, timeout, jit_sync=False, extra_env=None):
            plan = (plan_by_scale or {}).get(scale, {})
            return measure.Sample(100 + 1000 * scale, counters(**plan), output or str(scale))
        return invoke

    def test_measure_mode_scales(self):
        benchmark = measure.run.load_suite("^opt/invariant_field$")[0]
        mode = measure.Mode("tracing", Path("/bin/true"), "tracing")
        with mock.patch.object(measure, "invoke", self.fake_invoke()):
            result, checksum = measure.measure_mode(mode, benchmark, 1, 2, None, 1, False, True, [1, 2, 4])
        self.assertEqual(checksum, "1")
        self.assertEqual(result["instructions"], 1100)
        self.assertEqual(result["scales"]["status"], "ok")
        self.assertEqual([p["instructions"] for p in result["scales"]["points"]], [1100, 2100, 4100])
        self.assertNotIn("warmed_instructions_per_scale", result)

    def test_measure_mode_same_output_at_two_scales(self):
        benchmark = measure.run.load_suite("^opt/invariant_field$")[0]
        mode = measure.Mode("tracing", Path("/bin/true"), "tracing")
        with mock.patch.object(measure, "invoke", self.fake_invoke(output="same")):
            with self.assertRaisesRegex(RuntimeError, "same output at scales 1 and 2"):
                measure.measure_mode(mode, benchmark, 1, 1, None, 1, False, True, [1, 2])

    def test_warmed_marks_plan_change(self):
        benchmark = measure.run.load_suite("^opt/invariant_field$")[0]
        mode = measure.Mode("tracing", Path("/bin/true"), "tracing")
        with mock.patch.object(measure, "invoke", self.fake_invoke({2: {"recording_starts": 3}})):
            result, _ = measure.measure_mode(mode, benchmark, 1, 1, None, 1, True, True)
        self.assertEqual(result["warmed_instructions_per_scale"], 1000)
        self.assertEqual(result["warmed_check"]["status"], "inconclusive")
        result["summary"] = measure.summarize_counters(result["counters"])
        rows = [{"benchmark": "b", "modes": {"tracing": result}}]
        self.assertIn("1,000 (inconclusive)", measure.markdown(rows, ["tracing"], True))
        self.assertIn("recording_starts 1->3", measure.warmed_notes_markdown(rows, ["tracing"]))

    def test_scales_markdown(self):
        series = measure.scale_series([(1, aggregate_at(1100)), (2, aggregate_at(2100, traces_compiled=2))], 10)
        text = measure.scales_markdown([{"benchmark": "b", "modes": {"tracing": {"scales": series}}}], ["tracing"])
        self.assertIn("| b | tracing | 1: 1,100, 2: 2,100 | 1->2: 1,000 | inconclusive: trace counts differ", text)


def fake_row(name, before, after, before_alloc=None, after_alloc=None):
    def mode(instructions, alloc):
        data = {"instructions": instructions, "summary": {}, "counters": {}}
        if alloc:
            data["alloc"] = alloc
        return data

    return {"benchmark": name, "modes": {
        "interpreter": mode(after, after_alloc), "before-interpreter": mode(before, before_alloc),
    }}


class GateLogicTests(unittest.TestCase):
    def test_rise_over_limit(self):
        problems = measure.gate_violations([fake_row("a", 1000, 1011)], 1.0)
        self.assertEqual(len(problems), 1)
        self.assertIn("a interpreter", problems[0])

    def test_rise_within_limit_and_drop(self):
        self.assertEqual(measure.gate_violations([fake_row("a", 1000, 1010)], 1.0), [])
        self.assertEqual(measure.gate_violations([fake_row("a", 1000, 500)], 1.0), [])

    def test_alloc_rise(self):
        row = fake_row("a", 1000, 1000, {"alloc_count": 5, "alloc_peak_bytes": 100},
                       {"alloc_count": 6, "alloc_peak_bytes": 100})
        problems = measure.gate_violations([row], 1.0)
        self.assertEqual(len(problems), 1)
        self.assertIn("alloc_count", problems[0])
        row = fake_row("a", 1000, 1000, {"alloc_count": 5, "alloc_peak_bytes": 100},
                       {"alloc_count": 5, "alloc_peak_bytes": 101})
        self.assertIn("alloc_peak_bytes", measure.gate_violations([row], 1.0)[0])

    def test_alloc_bytes_alone_not_gated(self):
        row = fake_row("a", 1000, 1000, {"alloc_count": 5, "alloc_bytes": 1}, {"alloc_count": 5, "alloc_bytes": 9})
        self.assertEqual(measure.gate_violations([row], 1.0), [])

    def test_one_side_missing_fails(self):
        row = fake_row("a", 1000, 1000)
        del row["modes"]["before-interpreter"]
        self.assertIn("measured only after", measure.gate_violations([row], 50.0)[0])

    def test_missing_instructions_fails(self):
        row = fake_row("a", 1000, None)
        self.assertIn("instructions:u missing", measure.gate_violations([row], 50.0)[0])

    def test_missing_alloc_fails_when_required(self):
        row = fake_row("a", 1000, 1000, {"alloc_count": 5, "alloc_peak_bytes": 1}, {"alloc_count": 5})
        self.assertEqual(measure.gate_violations([row], 1.0), [])
        problems = measure.gate_violations([row], 1.0, require_alloc=True)
        self.assertEqual(problems, ["a interpreter: alloc_peak_bytes missing after"])


class OracleTests(unittest.TestCase):
    def which(self, available):
        return lambda name: f"/usr/bin/{name}" if name in available else None

    def test_selection_by_compat(self):
        environment = {k: v for k, v in os.environ.items() if k not in ("LUA_BIN", "LUA54_BIN", "LUAJIT_BIN")}
        with mock.patch.dict(os.environ, environment, clear=True), \
                mock.patch.object(measure.shutil, "which", self.which({"lua5.1", "lua5.4", "luajit"})):
            self.assertEqual(measure.oracle_for("lua51"), ("lua5.1", "/usr/bin/lua5.1", None))
            self.assertEqual(measure.oracle_for("lua54"), ("lua5.4", "/usr/bin/lua5.4", None))
            self.assertEqual(measure.oracle_for("luajit21"), ("luajit", "/usr/bin/luajit", None))
            name, path, reason = measure.oracle_for("lua55")
            self.assertIsNone(name)
            self.assertIn("no stock oracle for compat lua55", reason)
        with mock.patch.dict(os.environ, environment, clear=True), \
                mock.patch.object(measure.shutil, "which", self.which({"lua5.1"})):
            name, path, reason = measure.oracle_for("lua54")
            self.assertIsNone(path)
            self.assertIn("lua5.4 not found (set LUA54_BIN)", reason)
        with mock.patch.dict(os.environ, {**environment, "LUA54_BIN": "my-lua"}, clear=True), \
                mock.patch.object(measure.shutil, "which", self.which({"my-lua"})):
            self.assertEqual(measure.oracle_for("lua54"), ("lua5.4", "/usr/bin/my-lua", None))

    def test_reference_runs_profile_oracle(self):
        commands = []

        def fake_run(command, **kwargs):
            commands.append(command)
            return subprocess.CompletedProcess(command, 0, b"42\n", b"")

        benchmark = measure.run.Benchmark("x", Path("/x.lua"), "lua54")
        with mock.patch.object(measure, "oracle_for", lambda compat: ("lua5.4", "/usr/bin/lua5.4", None)), \
                mock.patch.object(measure.subprocess, "run", fake_run):
            oracle = measure.reference_checksum(benchmark, 3, 1.0)
        self.assertEqual(commands, [["/usr/bin/lua5.4", "/x.lua", "3"]])
        self.assertEqual((oracle.name, oracle.checksum), ("lua5.4", "42"))

    def test_failing_oracle_is_recorded(self):
        def fake_run(command, **kwargs):
            return subprocess.CompletedProcess(command, 1, b"", b"lua: x.lua:1: syntax error\n")

        benchmark = measure.run.Benchmark("x", Path("/x.lua"), "lua51")
        with mock.patch.object(measure, "oracle_for", lambda compat: ("lua5.1", "/usr/bin/lua5.1", None)), \
                mock.patch.object(measure.subprocess, "run", fake_run):
            oracle = measure.reference_checksum(benchmark, 1, 1.0)
        self.assertIsNone(oracle.name)
        self.assertIn("lua5.1 exited with status 1: lua: x.lua:1: syntax error", oracle.reason)

    def test_markdown_names_rows_without_oracle(self):
        text = measure.oracle_markdown([
            {"benchmark": "a", "oracle": "lua5.1"}, {"benchmark": "b", "oracle": "lua5.4"},
            {"benchmark": "c", "oracle": None, "oracle_reason": "luajit not found"},
        ])
        self.assertIn("checked against lua5.1: a", text)
        self.assertIn("checked against lua5.4: b", text)
        self.assertIn("NO ORACLE, outputs only agree among Oximond builds: c (luajit not found)", text)


class MainTests(unittest.TestCase):
    """Drives main() with invoke and the oracle replaced by canned results."""

    def run_main(self, extra, instructions=(1000, 1000), alloc_counts=(10, 10), alloc=True, op=False,
                 op_stdout=None, op_document=None, filter_="^opt/invariant_field$", oracle=None):
        with tempfile.TemporaryDirectory() as directory:
            names = ("new", "old", "anew", "aold", "onew", "oold")
            new, old, alloc_new, alloc_old, op_new, op_old = (Path(directory) / n for n in names)
            for path in (new, old, alloc_new, alloc_old, op_new, op_old):
                path.write_bytes(path.name.encode())
            seen = []

            def fake_invoke(mode, benchmark, scale, cpu, timeout, jit_sync=False, extra_env=None):
                seen.append((mode.binary, dict(extra_env or {})))
                if mode.binary in (alloc_new, alloc_old):
                    count = alloc_counts[0] if mode.binary == alloc_old else alloc_counts[1]
                    return measure.Sample(1, counters(alloc_count=count, alloc_bytes=1, alloc_peak_bytes=5), "7")
                if mode.binary in (op_new, op_old):
                    if op_document is not False:
                        document = op_document or op_stats_document(execution=mode.execution)
                        Path(extra_env["OXIMOND_OP_STATS"]).write_text(json.dumps(document))
                    return measure.Sample(999999, counters(), op_stdout or "7")
                value = instructions[0] if mode.binary == old else instructions[1]
                return measure.Sample(value * scale, counters(), "7" if scale == 1 else f"7x{scale}")

            argv = ["--filter", filter_, "--mode", "interpreter", "--repeat", "1",
                    "--oximond", str(new), "--before", str(old), *extra]
            if alloc:
                argv += ["--alloc-oximond", str(alloc_new), "--alloc-before", str(alloc_old)]
            if op:
                argv += ["--op-stats-oximond", str(op_new), "--op-stats-before", str(op_old)]
            if oracle is None:
                argv.append("--no-reference")
            out = Path(directory) / "r.json"
            markdown_out = Path(directory) / "r.md"
            err = io.StringIO()
            with mock.patch.object(measure, "invoke", fake_invoke), \
                    mock.patch.object(measure, "reference_checksum",
                                      lambda b, s, t: oracle or measure.Oracle(None, None, None, "--no-reference")), \
                    mock.patch.object(measure.shutil, "which", lambda name: "/usr/bin/" + name), \
                    mock.patch.object(measure, "machine_state", lambda cpu: {"uptime": "u"}), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
                status = measure.main([*argv, "--json", str(out), "--markdown", str(markdown_out)])
            self.seen = seen
            self.paths = dict(zip(names, (new, old, alloc_new, alloc_old, op_new, op_old)))
            self.hashes = {n: hashlib.sha256(n.encode()).hexdigest() for n in names}
            return status, err.getvalue(), json.loads(out.read_text()), markdown_out.read_text()

    def test_instruction_rise_exits_3(self):
        status, err, _, _ = self.run_main(["--gate", "1.0"], (1000, 1020))
        self.assertEqual(status, 3)
        self.assertIn("opt/invariant_field interpreter", err)

    def test_instruction_rise_within_gate(self):
        status, _, report, _ = self.run_main(["--gate", "5.0"], (1000, 1020))
        self.assertEqual(status, 0)
        self.assertEqual(report["gate"], 5.0)

    def test_alloc_rise_exits_3(self):
        status, err, report, _ = self.run_main(["--gate", "50"], (1000, 1000), alloc_counts=(10, 11))
        self.assertEqual(status, 3)
        self.assertIn("alloc_count", err)
        self.assertEqual(report["results"][0]["modes"]["interpreter"]["alloc"]["alloc_count"], 11)
        self.assertEqual(report["results"][0]["modes"]["before-interpreter"]["alloc"]["alloc_count"], 10)

    def test_without_gate_unchanged(self):
        status, err, report, _ = self.run_main([], (1000, 5000), alloc_counts=(10, 99))
        self.assertEqual(status, 0)
        self.assertNotIn("gate", err)
        self.assertIsNone(report["gate"])
        self.assertTrue(report["jit_sync"])
        self.assertIn("alloc_oximond", report["binaries"])

    def test_no_jit_sync_recorded(self):
        _, _, report, _ = self.run_main(["--no-jit-sync"], (1, 1))
        self.assertFalse(report["jit_sync"])

    def test_gate_without_before_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(measure.shutil, "which", lambda n: n):
            binary = Path(directory) / "b"
            binary.write_bytes(b"x")
            with self.assertRaisesRegex(RuntimeError, "--gate needs --before"):
                measure.main(["--gate", "1", "--oximond", str(binary)])

    def test_metadata_hashes_and_features(self):
        _, _, report, _ = self.run_main([], op=True)
        binaries = report["binaries"]
        self.assertEqual(binaries["oximond"]["sha256"], self.hashes["new"])
        self.assertEqual(binaries["before"]["sha256"], self.hashes["old"])
        self.assertEqual(binaries["alloc_before"]["sha256"], self.hashes["aold"])
        self.assertEqual(binaries["alloc_oximond"]["features"], ["alloc-stats"])
        self.assertEqual(binaries["op_stats_oximond"]["sha256"], self.hashes["onew"])
        self.assertEqual(binaries["op_stats_oximond"]["features"], ["op-stats"])
        self.assertIn("reported", binaries["op_stats_oximond"]["features_source"])
        self.assertEqual(binaries["op_stats_before"]["path"], str(self.paths["oold"]))

    def test_op_stats_results(self):
        status, _, report, text = self.run_main(["--gate", "1"], op=True)
        self.assertEqual(status, 0)
        mode = report["results"][0]["modes"]["interpreter"]
        self.assertEqual(mode["instructions"], 1000)
        self.assertEqual(mode["op_stats"]["instructions"], 999999)
        self.assertIn("diagnostic build overhead", mode["op_stats"]["instructions_label"])
        self.assertEqual(mode["op_stats"]["counters"]["h1.ir.array_sites"], 4)
        self.assertIn("before-interpreter", report["results"][0]["modes"])
        self.assertIn("op_stats", report["results"][0]["modes"]["before-interpreter"])
        reports = [env["OXIMOND_OP_STATS"] for binary, env in self.seen if binary == self.paths["onew"]]
        self.assertEqual(len(reports), 1)
        self.assertTrue(all("OXIMOND_OP_STATS" not in env for binary, env in self.seen
                            if binary not in (self.paths["onew"], self.paths["oold"])))
        self.assertIn("## Operation statistics", text)
        self.assertIn("999,999 (diagnostic build overhead", text)

    def test_op_stats_checksum_mismatch(self):
        with self.assertRaisesRegex(RuntimeError, "op-stats build printed '8'"):
            self.run_main([], op=True, op_stdout="8")

    def test_op_stats_report_missing(self):
        with self.assertRaisesRegex(RuntimeError, "printed no op-stats report; build it with --features op-stats"):
            self.run_main([], op=True, op_document=False)

    def test_op_stats_wrong_schema(self):
        with self.assertRaisesRegex(RuntimeError, "schema"):
            self.run_main([], op=True, op_document=op_stats_document(schema="other"))

    def test_op_stats_before_needs_before(self):
        with self.assertRaisesRegex(RuntimeError, "--op-stats-before needs --before and --op-stats-oximond"):
            measure.main(["--op-stats-before", "/x"])

    def test_oracle_recorded(self):
        _, _, report, text = self.run_main([], oracle=measure.Oracle("lua5.1", "7", "/usr/bin/lua5.1"))
        self.assertEqual(report["results"][0]["oracle"], "lua5.1")
        self.assertIn("checked against lua5.1: opt/invariant_field", text)
        _, _, report, text = self.run_main([])
        self.assertIsNone(report["results"][0]["oracle"])
        self.assertEqual(report["results"][0]["oracle_reason"], "--no-reference")
        self.assertIn("NO ORACLE", text)

    def test_oracle_mismatch(self):
        with self.assertRaisesRegex(RuntimeError, r"checksum mismatch .* \(lua5.4\)"):
            self.run_main([], oracle=measure.Oracle("lua5.4", "8", "/usr/bin/lua5.4"))

    def test_scales_flag(self):
        _, _, report, text = self.run_main(["--scales", "1,2,4"], alloc=False)
        self.assertEqual(report["scales"], [1, 2, 4])
        series = report["results"][0]["modes"]["interpreter"]["scales"]
        self.assertEqual(series["status"], "ok")
        self.assertEqual([m["instructions_per_scale"] for m in series["marginal"]], [1000, 1000])
        self.assertIn("| opt/invariant_field | interpreter | 1: 1,000, 2: 2,000, 4: 4,000 |", text)
        with self.assertRaisesRegex(RuntimeError, "--scales replaces --scale"):
            self.run_main(["--scales", "1,2", "--scale", "3"], alloc=False)


class OptBenchmarkTests(unittest.TestCase):
    NAMES = ["opt/field_reload", "opt/store_load", "opt/alias_holder", "opt/invariant_field",
             "opt/range_dense", "opt/range_float_index", "opt/range_offset", "opt/range_fail",
             "opt/arith_stable", "opt/arith_mixed", "opt/mod_int", "opt/mod_float", "opt/mod_mixed",
             "opt/cold_recovery", "opt/call_args",
             "opt/call_args_hook", "opt/dead_slots", "opt/retained_slots", "opt/closure_family",
             "opt/closure_protos"]

    def test_listed(self):
        names = [b.name for b in measure.run.load_suite("opt/")]
        self.assertEqual(names, self.NAMES)

    def test_integer_workloads_use_lua54(self):
        compat = {b.name: b.compat for b in measure.run.load_suite("opt/")}
        self.assertEqual(compat["opt/range_dense"], "lua54")
        self.assertEqual(compat["opt/arith_mixed"], "lua54")
        for name in ("opt/mod_int", "opt/mod_float", "opt/mod_mixed"):
            self.assertEqual(compat[name], "lua54")
        self.assertEqual(compat["opt/range_float_index"], "lua51")

    @unittest.skipUnless(os.environ.get("MEASURE_CHECKSUMS") == "1",
                         "runs every opt/ workload in both modes; set MEASURE_CHECKSUMS=1")
    def test_checksums_match_oracle(self):
        binary = next((p for p in (ROOT / "target" / "release" / "oximond", ROOT / "target" / "debug" / "oximond")
                       if p.is_file()), None)
        if binary is None:
            self.skipTest("needs an oximond binary")
        for benchmark in measure.run.load_suite("opt/"):
            name, oracle, reason = measure.oracle_for(benchmark.compat)
            if oracle is None:
                self.fail(f"{benchmark.name}: {reason}")
            expected = subprocess.run([oracle, str(benchmark.path), "1"], capture_output=True, text=True,
                                      check=True, env=measure.run.clean_environment()).stdout.strip()
            self.assertTrue(expected, benchmark.name)
            for execution in measure.MODES:
                completed = subprocess.run(
                    [str(binary), f"--compat={benchmark.compat}", f"--execution={execution}", "-E",
                     str(benchmark.path), "1"],
                    capture_output=True, text=True, timeout=120, check=True,
                    env={**measure.run.clean_environment(), "OXIMOND_JIT_SYNC": "1"})
                self.assertEqual(completed.stdout.strip(), expected, f"{benchmark.name} {execution} vs {name}")


@unittest.skipUnless(
    os.environ.get("MEASURE_LIVE") == "1" and os.environ.get("OXIMOND_HEAVY_ACTIVE") == "1"
    and BINARY.is_file() and shutil.which("perf"),
    "live perf run: needs MEASURE_LIVE=1, target/release/oximond and perf, and must run through the gate "
    "(MEASURE_LIVE=1 scripts/heavy.sh -x python3 -m unittest benchmarks/test_measure.py)")
class LiveTests(unittest.TestCase):
    def test_quick_run_writes_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            status = measure.main([
                "--filter", "^allocation/ctor_array$", "--quick", "--repeat", "1",
                "--json", str(out / "r.json"), "--markdown", str(out / "r.md"),
            ])
            self.assertEqual(status, 0)
            report = json.loads((out / "r.json").read_text())
            modes = report["results"][0]["modes"]
            self.assertGreater(modes["interpreter"]["instructions"], 0)
            self.assertGreaterEqual(modes["interpreter"]["summary"]["tables_created"], 150000)
            self.assertIn("sha256", report["binaries"]["oximond"])
            self.assertIn("loadavg", report["host"])
            self.assertIn("allocation/ctor_array", (out / "r.md").read_text())


if __name__ == "__main__":
    unittest.main()
