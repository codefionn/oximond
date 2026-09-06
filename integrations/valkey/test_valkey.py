"""Build command tests. No Valkey archive, Cargo build, or compiler is needed."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


MODULE = importlib.util.spec_from_file_location("valkey_integration", Path(__file__).with_name("valkey.py"))
valkey = importlib.util.module_from_spec(MODULE)
MODULE.loader.exec_module(valkey)


class BuildTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "output"
        self.shared = self.root / "target/release/liboximond_capi.a"
        self.shared.parent.mkdir(parents=True)
        self.shared.write_bytes(b"shared library without bundled bridge")
        self.calls = []

    def prepare(self, base, variant):
        (valkey.source(base, variant) / "src").mkdir(parents=True)
        return []

    def call(self, command, log, cwd=None):
        command = list(map(str, command))
        self.calls.append(command)
        log.append({"command": command})
        if "build" in command and command[0] == "cargo":
            target = Path(command[command.index("--target-dir") + 1])
            archive = target / "release/liboximond_capi.a"
            archive.parent.mkdir(parents=True)
            archive.write_bytes(b"private Lua 5.1 bundled bridge")
        elif command[0] == "make":
            tree = Path(command[command.index("-C") + 1])
            (tree / "src/valkey-server").write_bytes(b"test binary")
        elif command[0] == "nm":
            return "00000000 T oximond_newstate_with_options\n" if "oximond" in command[1] else ""
        return ""

    def build(self, *options):
        argv = ["valkey.py", "--dir", str(self.output), "build", *options]
        with patch.object(sys, "argv", argv), patch.object(valkey, "ROOT", self.root), \
                patch.object(valkey, "prepare", side_effect=self.prepare), \
                patch.object(valkey, "call", side_effect=self.call), \
                patch.object(valkey, "rust_source_hashes", return_value={"Cargo.toml": "hash"}):
            self.assertEqual(valkey.main(), 0)

    def test_default_build_uses_private_bundled_bridge_before_linking(self):
        self.build()
        cargo = [command for command in self.calls if command[0] == "cargo"]
        self.assertEqual(len(cargo), 1)
        self.assertEqual(cargo[0][1:7], ["build", "--release", "-p", "oximond-capi",
                                        "--no-default-features", "--features"])
        self.assertEqual(cargo[0][7], "lua51,bundled-bridge")
        self.assertEqual(cargo[0][8:], ["--target-dir", str(self.output / "cargo")])
        oximond_make = next(command for command in self.calls
                            if command[0] == "make" and "oximond" in command[2])
        self.assertLess(self.calls.index(cargo[0]), self.calls.index(oximond_make))
        linked = next(item.split("=", 1)[1] for item in oximond_make
                      if item.startswith("OXIMOND_LIBRARY="))
        self.assertTrue(linked.startswith(str(self.output / "artifacts/liboximond_capi-")))
        self.assertNotEqual(linked, str(self.shared))
        self.assertEqual(Path(linked).read_bytes(), b"private Lua 5.1 bundled bridge")
        report = json.loads((self.output / "build-oximond.json").read_text())
        self.assertEqual(report["oximond_library"], linked)

    def test_explicit_library_skips_cargo(self):
        library = self.root / "selected.a"
        library.write_bytes(b"selected library")
        self.build("oximond", "--library", str(library))
        self.assertFalse(any(command[0] == "cargo" for command in self.calls))
        make = next(command for command in self.calls if command[0] == "make")
        self.assertIn(f"OXIMOND_LIBRARY={library}", make)

    def test_stock_only_skips_cargo(self):
        self.build("stock")
        self.assertFalse(any(command[0] == "cargo" for command in self.calls))
        self.assertEqual(len([command for command in self.calls if command[0] == "make"]), 1)

    def test_build_library_forces_private_build_with_explicit_library(self):
        library = self.root / "selected.a"
        library.write_bytes(b"selected library")
        self.build("oximond", "--library", str(library), "--build-library")
        self.assertTrue(any(command[0] == "cargo" for command in self.calls))
        make = next(command for command in self.calls if command[0] == "make")
        self.assertNotIn(f"OXIMOND_LIBRARY={library}", make)
        self.assertTrue(any(item.startswith(f"OXIMOND_LIBRARY={self.output / 'artifacts'}")
                            for item in make))

    def test_cargo_failure_keeps_original_error_and_build_report(self):
        def failing_call(command, log, cwd=None):
            if command[0] == "cargo":
                log.append({"command": list(map(str, command)), "exit": 1,
                            "stderr": "cargo failed"})
                raise RuntimeError("cargo failed")
            return self.call(command, log, cwd)

        argv = ["valkey.py", "--dir", str(self.output), "build", "oximond"]
        with patch.object(sys, "argv", argv), patch.object(valkey, "ROOT", self.root), \
                patch.object(valkey, "prepare", side_effect=self.prepare), \
                patch.object(valkey, "call", side_effect=failing_call), \
                patch.object(valkey, "rust_source_hashes", return_value={"Cargo.toml": "hash"}):
            with self.assertRaisesRegex(RuntimeError, "^cargo failed$"):
                valkey.main()
        report = json.loads((self.output / "build-oximond.json").read_text())
        self.assertEqual(report["commands"][-1]["stderr"], "cargo failed")
        self.assertNotIn("oximond_library", report)


if __name__ == "__main__":
    unittest.main()
