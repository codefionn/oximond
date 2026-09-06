"""Focused aggregate-runner tests using temporary harnesses, never real workloads."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

MODULE = importlib.util.spec_from_file_location("integration_runner", Path(__file__).with_name("test.py"))
runner = importlib.util.module_from_spec(MODULE)
sys.modules[MODULE.name] = runner
MODULE.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.report = self.root / "output/report.json"
        self.gate = self.root / "scripts/heavy.sh"
        self.gate.parent.mkdir()
        self.gate.write_text('#!/bin/sh\nprintf "gate\\n"\nexec "$@"\n')
        self.gate.chmod(0o755)

    def spec(self, name, status=0, available=True):
        harness = self.root / f"{name}.py"
        harness.write_text(f"print('stub {name}'); raise SystemExit({status})\n")
        required = [harness] if available else [self.root / "missing-binary"]
        return runner.Integration(name, [sys.executable, str(harness)], required, [])

    def execute(self, specs, names, strict=False, already_gated=False):
        with contextlib.redirect_stdout(io.StringIO()), patch.dict(os.environ, {"OXIMOND_HEAVY_ACTIVE": "0"}):
            status = runner.run_selected(specs, names, strict, self.root, self.report, "smoke", already_gated)
        return status, json.loads(self.report.read_text())

    def test_default_missing_prerequisites_are_skipped_without_launching(self):
        specs = {"one": self.spec("one", available=False)}
        with patch.object(runner.subprocess, "run", side_effect=AssertionError("must not run")):
            status, report = self.execute(specs, ["one"])
        self.assertEqual(status, 0)
        self.assertEqual(report["summary"]["skipped"], 1)
        self.assertEqual(report["results"][0]["missing"], [str(self.root / "missing-binary")])

    def test_explicit_missing_prerequisites_fail(self):
        status, report = self.execute({"one": self.spec("one", available=False)}, ["one"], strict=True)
        self.assertEqual(status, 1)
        self.assertEqual(report["summary"]["unavailable"], 1)

    def test_failure_does_not_prevent_other_selected_tests_and_preserves_logs(self):
        specs = {"one": self.spec("one", 7), "two": self.spec("two")}
        status, report = self.execute(specs, ["one", "two"])
        self.assertEqual(status, 1)
        self.assertEqual(report["summary"]["failed"], 1)
        self.assertEqual(report["summary"]["passed"], 1)
        self.assertEqual(report["results"][0]["exit_code"], 7)
        log = Path(report["results"][1]["log"]).read_text()
        self.assertEqual(log.count("gate\n"), 1)
        self.assertIn("stub two", log)
        self.assertEqual(report["results"][1]["command"][0], str(self.gate))

    def test_existing_gate_marker_and_explicit_flag_avoid_nested_gate(self):
        command = [sys.executable, "harness.py"]
        with patch.dict(os.environ, {"OXIMOND_HEAVY_ACTIVE": "1"}):
            self.assertEqual(runner.gated_command(self.root, command), command)
        with patch.dict(os.environ, {"OXIMOND_HEAVY_ACTIVE": "0"}):
            self.assertEqual(runner.gated_command(self.root, command, True), command)
        spec = self.spec("one")
        self.gate.unlink()
        status, report = self.execute({"one": spec}, ["one"], already_gated=True)
        self.assertEqual(status, 0)
        self.assertEqual(report["results"][0]["command"], spec.command)

    def test_missing_gate_is_reported_without_ungated_execution(self):
        self.gate.unlink()
        status, report = self.execute({"one": self.spec("one")}, ["one"], strict=True)
        self.assertEqual(status, 1)
        self.assertIn(f"executable {self.gate}", report["results"][0]["missing"])

    def test_process_launch_error_is_recorded_as_failure(self):
        with patch.object(runner.subprocess, "run", side_effect=OSError("launch failed")):
            status, report = self.execute({"one": self.spec("one")}, ["one"])
        self.assertEqual(status, 1)
        self.assertEqual(report["results"][0]["error"], "launch failed")

    def test_named_selection_is_deduplicated_and_strict(self):
        specs = {name: self.spec(name) for name in runner.NAMES}
        with patch.object(runner, "ROOT", self.root), patch.object(runner, "catalog", return_value=specs), \
                patch.object(runner, "run_selected", return_value=0) as run:
            self.assertEqual(runner.main(["valkey", "valkey", "--test", "full"]), 0)
            self.assertEqual(run.call_args.args[1:3], (["valkey"], True))
            self.assertEqual(run.call_args.args[5], "full")
            self.assertEqual(runner.main([]), 0)
            self.assertEqual(run.call_args.args[1:3], (list(runner.NAMES), False))
            self.assertEqual(runner.main(["--all"]), 0)
            self.assertEqual(run.call_args.args[1:3], (list(runner.NAMES), True))

    def test_list_has_no_execution_or_generated_report(self):
        specs = {name: self.spec(name) for name in runner.NAMES}
        with patch.object(runner, "ROOT", self.root), patch.object(runner, "catalog", return_value=specs), \
                patch.object(runner, "run_selected", side_effect=AssertionError("must not run")), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(runner.main(["neovim", "--list"]), 0)
        self.assertIn("neovim (smoke): available", output.getvalue())
        self.assertNotIn("valkey", output.getvalue())
        self.assertFalse(self.report.exists())

    def test_every_real_command_is_a_test_and_smoke_matrix_is_bounded(self):
        specs = runner.catalog(runner.ROOT, "smoke", self.report.parent)
        for spec in specs.values():
            self.assertNotIn("prepare", spec.command)
            self.assertNotIn("build", spec.command)
        self.assertEqual(specs["lua-libraries"].command.count("--case"), 4)
        self.assertIn("--profile", specs["valkey"].command)
        full = runner.catalog(runner.ROOT, "full", self.report.parent)
        self.assertNotIn("--profile", full["lua-libraries"].command)
        self.assertNotIn("--profile", full["valkey"].command)
        self.assertIn("world-smoke", full["luanti"].command)
        self.assertIn("full-smoke", full["neovim"].command)


if __name__ == "__main__":
    unittest.main()
