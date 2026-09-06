#!/usr/bin/env python3
"""Tests for optrace.py. They run no binary and no gdb.

OPTRACE_LIVE=1 also runs the tracer against target/release/oximond for
numeric/for_sum, which needs gdb.
"""

from pathlib import Path
import os
import subprocess
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import optrace  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BASE = optrace.LOAD_BASE


def segment(target, length):
    """A recorded segment: `length` addresses starting at `target`."""
    return [BASE + target + i for i in range(length)]


SEGMENTS = [segment(0x100, 31)] * 3 + [segment(0x200, 66)] * 3 + [segment(0x300, 10), segment(0x300, 14)]


class AggregateTests(unittest.TestCase):
    def test_one_row_per_target_ordered_by_share(self):
        rows = optrace.aggregate(SEGMENTS)
        self.assertEqual([row["target"] for row in rows], [0x200, 0x100, 0x300])
        self.assertEqual([row["segments"] for row in rows], [3, 3, 2])

    def test_instruction_counts_and_shares(self):
        rows = {row["target"]: row for row in optrace.aggregate(SEGMENTS)}
        self.assertEqual((rows[0x100]["min"], rows[0x100]["max"]), (31, 31))
        self.assertEqual(rows[0x300]["mean"], 12)
        self.assertEqual(rows[0x300]["lengths"], {10: 1, 14: 1})
        total = 3 * 31 + 3 * 66 + 24
        self.assertAlmostEqual(rows[0x200]["share"], 100 * 198 / total)
        self.assertAlmostEqual(sum(row["share"] for row in rows.values()), 100)

    def test_empty(self):
        self.assertEqual(optrace.aggregate([]), [])
        self.assertIn("segments=0", optrace.format_report([], {}))

    def test_report_rows(self):
        text = optrace.format_report(SEGMENTS, {0x100: "fast_path.rs:923"})
        lines = text.splitlines()
        self.assertTrue(lines[0].startswith("segments=8"))
        self.assertIn("0x200", lines[2])
        self.assertIn("66", lines[2])
        self.assertIn("fast_path.rs:923", lines[3])
        self.assertIn("12.0", lines[4])
        self.assertTrue(lines[4].rstrip().endswith("?"))


class DumpTests(unittest.TestCase):
    def test_dump_prints_path_with_disassembly(self):
        table = {0x100: "mov %rax,%rbx", 0x101: "jmp *%rcx"}
        with mock.patch.object(optrace, "disassembly", return_value=table):
            text = optrace.format_dump([segment(0x100, 2)], 0x100, Path("bin"))
        self.assertEqual(text.splitlines(), ["segment starting 0x100 (2 instructions)",
                                             "  0x100: mov %rax,%rbx", "  0x101: jmp *%rcx"])

    def test_dump_unknown_target_lists_known_ones(self):
        with self.assertRaises(SystemExit) as raised:
            optrace.format_dump(SEGMENTS, 0x999, Path("bin"))
        self.assertIn("0x100", str(raised.exception))


class FailureTests(unittest.TestCase):
    def test_missing_symbol_is_reported(self):
        done = subprocess.CompletedProcess([], 0, stdout="0000000000001000 0000000000000010 t other_fn\n")
        with mock.patch.object(optrace.subprocess, "run", return_value=done):
            with self.assertRaises(SystemExit) as raised:
                optrace.find_dispatch(Path("bin"))
        self.assertIn("not found", str(raised.exception))

    def test_missing_dispatch_jump_is_reported(self):
        symbol = "0000000000001000 0000000000000100 t <vm>::run_fast_activations::<false, false, false, true>\n"
        calls = [subprocess.CompletedProcess([], 0, stdout=symbol),
                 subprocess.CompletedProcess([], 0, stdout="  1000:\tret\n")]
        with mock.patch.object(optrace.subprocess, "run", side_effect=calls):
            with self.assertRaises(SystemExit) as raised:
                optrace.find_dispatch(Path("bin"))
        self.assertIn("no `jmp *%reg`", str(raised.exception))

    def test_dispatch_found_inside_loop_only(self):
        symbol = ("0000000000001000 0000000000000100 t <vm>::run_fast_activations::<false, true, false>\n"
                  "0000000000002000 0000000000000100 t <vm>::run_fast_activations::<false, false, false, true>\n")
        dis = "  2010:\tmov %rax,%rbx\n  2014:\tjmp    *%rcx\n"
        calls = [subprocess.CompletedProcess([], 0, stdout=symbol), subprocess.CompletedProcess([], 0, stdout=dis)]
        with mock.patch.object(optrace.subprocess, "run", side_effect=calls) as run:
            self.assertEqual(optrace.find_dispatch(Path("bin")), 0x2014)
        self.assertIn("--start-address=8192", run.call_args_list[1].args[0])

    def test_missing_gdb_is_reported(self):
        with mock.patch.object(optrace.shutil, "which", return_value=None):
            with self.assertRaises(SystemExit) as raised:
                optrace.record(Path("bin"), Path("x.lua"), "lua51", "interpreter", "1", 0, 0, 1)
        self.assertIn("gdb not found", str(raised.exception))

    def test_unknown_benchmark_is_reported(self):
        with self.assertRaises(SystemExit) as raised:
            optrace.resolve_target("no/such_benchmark", None)
        self.assertIn("suite.json", str(raised.exception))

    def test_suite_benchmark_resolves(self):
        script, profile = optrace.resolve_target("numeric/for_sum", None)
        self.assertTrue(script.is_file())
        self.assertTrue(profile)


@unittest.skipUnless(os.environ.get("OPTRACE_LIVE") == "1", "set OPTRACE_LIVE=1 to run gdb")
class LiveTests(unittest.TestCase):
    def test_for_sum_has_rows_and_dump(self):
        binary = ROOT / "target" / "release" / "oximond"
        tool = ROOT / "benchmarks" / "optrace.py"
        report = subprocess.run([sys.executable, str(tool), str(binary), "numeric/for_sum", "--record", "40"],
                                capture_output=True, text=True, check=True).stdout
        rows = report.splitlines()[2:]
        self.assertGreaterEqual(len(rows), 1)
        target = rows[0].split()[0]
        dump = subprocess.run([sys.executable, str(tool), str(binary), "numeric/for_sum", "--record", "40",
                               "--dump", target], capture_output=True, text=True, check=True).stdout
        self.assertIn("jmp", dump.splitlines()[-1])


if __name__ == "__main__":
    unittest.main()
