#!/usr/bin/env python3
"""Run existing integration builds. Default smoke checks skip missing prerequisites.

Named integrations and --all require their prerequisites. --test full selects
complete library/Valkey matrices, Luanti devtest worlds, and full Neovim config.
This runner never prepares or builds integrations. Full Neovim config may let
its existing plugin manager fetch missing plugins. Each child runs through the
heavy gate unless the caller already holds it.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("lua-libraries", "valkey", "luanti", "neovim")
PROFILES = ("lua51", "lua52", "lua53", "lua54", "lua55", "luajit21")


@dataclass
class Integration:
    name: str
    command: list[str]
    required: list[Path]
    executables: list[Path]
    tools: tuple[str, ...] = ()
    detail_report: Path | None = None

    def missing(self):
        reasons = [str(path) for path in self.required if not path.exists()]
        reasons += [f"executable {path}" for path in self.executables
                    if not path.is_file() or not os.access(path, os.X_OK)]
        reasons += [f"tool {tool}" for tool in self.tools if not shutil.which(tool)]
        if not sys.platform.startswith("linux"):
            reasons.append("Linux host")
        return reasons


def catalog(root, test, output, oximond=None):
    """Keep preparation separate from discovery: no harness imports or builds."""
    scripts = root / "integrations"
    target = root / "target"
    python = sys.executable
    specs = {}
    base = target / "lua-libraries"
    manifest = scripts / "lua-libraries/sources.json"
    pins = json.loads(manifest.read_text())["sources"]
    required = [base / "build-report.json", manifest]
    required += [base / "downloads" / pin["archive"] for pin in pins.values()]
    required += [base / "source" / pin["directory"] for pin in pins.values()]
    profiles = PROFILES if test == "full" else ("lua54",)
    oximond = (oximond or target / "release/oximond").resolve()
    executables = [oximond]
    for profile in dict.fromkeys((*profiles, "lua51")):
        executables.append(base / "source" / pins[profile]["directory"] / "src"
                           / ("luajit" if profile == "luajit21" else "lua"))
    for variant in ("oximond", *profiles):
        for module in ("lpeg.so", "socket/core.so", "mime/core.so"):
            required.append(base / "modules" / variant / module)
        for wrapper in ("socket.lua", "mime.lua", "ltn12.lua"):
            required.append(base / "modules" / variant / wrapper)
    required += [base / "modules/redis" / module for module in
                 ("cjson.so", "cjson/safe.so", "struct.so", "cmsgpack.so", "cmsgpack_safe.so")]
    detail = output / f"lua-libraries-{test}.json"
    command = [python, str(scripts / "lua-libraries/libraries.py"), "run", "--report", str(detail)]
    command += ["--oximond", str(oximond)]
    if test == "smoke":
        command += ["--profile", "lua54", "--engine", "interpreter", "--timeout", "60"]
        for case in ("lpeg-fixture", "socket-loopback", "teal-compiler", "bundled-modules"):
            command += ["--case", case]
    specs["lua-libraries"] = Integration("lua-libraries", command, required, executables,
                                         detail_report=detail)

    base = target / "valkey"
    pin = json.loads((scripts / "valkey/source.json").read_text())
    detail = output / f"valkey-{test}.json"
    command = [python, str(scripts / "valkey/valkey.py"), "run", "--report", str(detail)]
    if test == "smoke":
        command += ["--profile", "lua54", "--engine", "interpreter"]
    specs["valkey"] = Integration(
        "valkey", command, [base / f"build-{variant}.json" for variant in ("stock", "oximond")],
        [base / "trees" / variant / pin["directory"] / "src/valkey-server"
         for variant in ("stock", "oximond")], detail_report=detail)

    base = target / "luanti"
    required = [base / "source"]
    if test == "full":
        required.append(base / "source/games/devtest")
    specs["luanti"] = Integration(
        "luanti", [python, str(scripts / "luanti/luanti.py"),
                   "world-smoke" if test == "full" else "smoke", "both"],
        required, [base / f"build-{variant}/bin/luanti" for variant in ("luajit", "oximond")],
        ("nm", "ldd") if test == "smoke" else ())

    base = target / "neovim"
    variants = ("lua", "luajit", "oximond")
    required = [base / "source/runtime", base / "kickstart/init.lua"]
    required += [base / f"user-{variant}/config/nvim/init.lua" for variant in variants]
    specs["neovim"] = Integration(
        "neovim", [python, str(scripts / "neovim/neovim.py"),
                   "full-smoke" if test == "full" else "smoke", "all"],
        required, [base / f"build-{variant}/bin/nvim" for variant in variants], ("nm", "ldd"))
    for spec in specs.values():
        spec.required.insert(0, Path(spec.command[1]))
    return specs


def gated_command(root, command, already_gated=False):
    if already_gated or os.environ.get("OXIMOND_HEAVY_ACTIVE") == "1":
        return command
    return [str(root / "scripts/heavy.sh"), *command]


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")


def run_selected(specs, names, strict, root, report_path, test, already_gated=False):
    report = {"format": 1, "started_at": datetime.now(timezone.utc).isoformat(),
              "test": test, "selection": names, "strict_prerequisites": strict,
              "results": [], "summary": {}}
    output = report_path.parent
    output.mkdir(parents=True, exist_ok=True)
    for name in names:
        spec = specs[name]
        missing = spec.missing()
        command = gated_command(root, spec.command, already_gated)
        if not already_gated and os.environ.get("OXIMOND_HEAVY_ACTIVE") != "1":
            gate = root / "scripts/heavy.sh"
            if not gate.is_file() or not os.access(gate, os.X_OK):
                missing.append(f"executable {gate}")
        row = {"integration": name, "test": test, "command": command}
        if missing:
            row.update(status="unavailable" if strict else "skipped", missing=missing)
            print(f"{name}: {row['status']} ({len(missing)} missing prerequisites)", flush=True)
        else:
            log = output / f"{name}-{test}.log"
            row["log"] = str(log)
            if spec.detail_report is not None:
                # These are this runner's own generated artifacts. Remove old
                # detail output so a failed startup cannot report stale success.
                spec.detail_report.unlink(missing_ok=True)
            print(f"{name}: running {test}, log {log}", flush=True)
            started = time.monotonic()
            try:
                with log.open("w") as stream:
                    result = subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT)
                row.update(status="passed" if result.returncode == 0 else "failed",
                           exit_code=result.returncode)
            except OSError as error:
                row.update(status="failed", error=str(error))
            except KeyboardInterrupt:
                row.update(status="interrupted", error="interrupted by user")
            row["duration_seconds"] = round(time.monotonic() - started, 3)
            if spec.detail_report is not None and spec.detail_report.exists():
                row["detail_report"] = str(spec.detail_report)
            print(f"{name}: {row['status']}", flush=True)
            if row["status"] == "failed" and log.exists():
                print(log.read_text(errors="replace")[-4000:], flush=True)
        report["results"].append(row)
        report["summary"] = {status: sum(r["status"] == status for r in report["results"])
                             for status in ("passed", "failed", "skipped", "unavailable", "interrupted")}
        write_report(report_path, report)
        if row["status"] == "interrupted":
            return 130
    print("Results: " + ", ".join(f"{count} {status}" for status, count in report["summary"].items())
          + f". Report: {report_path}")
    return int(bool(report["summary"]["failed"] or report["summary"]["unavailable"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("integrations", nargs="*", choices=NAMES, metavar="INTEGRATION")
    parser.add_argument("--all", action="store_true", help="require every integration to be available")
    parser.add_argument("--list", action="store_true", help="list prerequisites and commands without running tests")
    parser.add_argument("--test", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--report", type=Path, default=ROOT / "target/integrations/report.json")
    parser.add_argument("--oximond", type=Path, help="CLI binary used by Lua library checks")
    parser.add_argument("--already-gated", action="store_true", help="caller already holds an older heavy gate")
    args = parser.parse_args(argv)
    if args.all and args.integrations:
        parser.error("choose named integrations or --all")
    names = list(dict.fromkeys(args.integrations)) or list(NAMES)
    report = args.report.resolve()
    specs = catalog(ROOT, args.test, report.parent, args.oximond)
    if args.list:
        for name in names:
            missing = specs[name].missing()
            print(f"{name} ({args.test}): {'unavailable' if missing else 'available'}")
            print("  command: " + " ".join(gated_command(ROOT, specs[name].command, args.already_gated)))
            for reason in missing:
                print("  missing: " + reason)
        return 0
    return run_selected(specs, names, bool(args.integrations or args.all), ROOT, report,
                        args.test, args.already_gated)


if __name__ == "__main__":
    raise SystemExit(main())
