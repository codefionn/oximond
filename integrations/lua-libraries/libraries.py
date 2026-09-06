#!/usr/bin/env python3
"""Pinned native Lua libraries and Teal against all Oximond profiles."""

import argparse
from contextlib import contextmanager, nullcontext
import fcntl
import hashlib
import http.server
import json
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import tarfile
import threading
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROFILES = ("lua51", "lua52", "lua53", "lua54", "lua55", "luajit21")
ENGINES = ("interpreter", "tracing")
SOURCES = json.loads((HERE / "sources.json").read_text())["sources"]
MARKER = "OXIMOND_LIBRARY_OK"
CASES = ("lpeg-fixture", "lpeg-upstream", "socket-loopback", "socket-http",
         "socket-except", "socket-ltn12", "socket-mime", "socket-url",
         "teal-compiler", "teal-bootstrap", "bundled-modules", "bundled-edge")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(base, name):
    return base / "source" / SOURCES[name]["directory"]


def call(command, log):
    print("+", " ".join(map(str, command)), flush=True)
    result = subprocess.run(list(map(str, command)), capture_output=True, text=True)
    log.append({"command": list(map(str, command)), "exit": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr})
    if result.returncode:
        raise RuntimeError(result.stderr or result.stdout)
    return result.stdout


def prepare(base):
    for name, pin in SOURCES.items():
        archive = base / "downloads" / pin["archive"]
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            print("download", pin["url"], flush=True)
            data = urlopen(pin["url"], timeout=60).read()
            if hashlib.sha256(data).hexdigest() != pin["sha256"]:
                raise RuntimeError(f"checksum mismatch downloading {name}")
            archive.write_bytes(data)
        if sha(archive) != pin["sha256"]:
            raise RuntimeError(f"checksum mismatch: {archive}")
        # Only this harness's generated source directories are replaced.
        checkout = source(base, name)
        if checkout.exists():
            shutil.rmtree(checkout)
        with tarfile.open(archive) as bundle:
            if any(not member.name.startswith(pin["directory"] + "/")
                   and member.name != pin["directory"] for member in bundle.getmembers()):
                raise RuntimeError(f"unexpected archive root: {archive}")
            bundle.extractall(base / "source", filter="data")


def verify_sources(base):
    for name, pin in SOURCES.items():
        archive = base / "downloads" / pin["archive"]
        if sha(archive) != pin["sha256"]:
            raise RuntimeError(f"checksum mismatch: {archive}")
        with tarfile.open(archive) as bundle:
            for member in bundle.getmembers():
                if member.isfile():
                    extracted = base / "source" / member.name
                    if extracted.read_bytes() != bundle.extractfile(member).read():
                        raise RuntimeError(f"modified pinned source: {extracted}; build again")


def build_modules(base, variant, include, log):
    output = base / "modules" / variant
    (output / "socket").mkdir(parents=True, exist_ok=True)
    (output / "mime").mkdir(exist_ok=True)
    cc = os.environ.get("CC", "cc")
    # The vendored bridge intentionally omits lualib.h; use the pinned 5.1 copy.
    common = [cc, "-O2", "-fPIC", "-shared", "-I", include,
              "-I", source(base, "lua51") / "src"]
    lpeg = source(base, "lpeg")
    call([*common, "-o", output / "lpeg.so",
          *[lpeg / (name + ".c") for name in ("lpcap", "lpcode", "lpprint", "lptree", "lpvm", "lpcset")]], log)
    socket = source(base, "luasocket") / "src"
    for wrapper in socket.glob("*.lua"):
        destination = output if wrapper.stem in ("socket", "mime", "ltn12") else output / "socket"
        shutil.copyfile(wrapper, destination / wrapper.name)
    names = ("luasocket", "timeout", "buffer", "io", "auxiliar", "compat",
             "options", "inet", "usocket", "except", "select", "tcp", "udp")
    call([*common, "-DLUASOCKET_NODEBUG", "-o", output / "socket/core.so",
          *[socket / (name + ".c") for name in names]], log)
    call([*common, "-o", output / "mime/core.so", socket / "mime.c", socket / "compat.c"], log)
    compat = source(base, "compat53")
    (output / "compat53").mkdir(exist_ok=True)
    for wrapper in (compat / "compat53").glob("*.lua"):
        shutil.copyfile(wrapper, output / "compat53" / wrapper.name)
    if variant in ("lua51", "lua52", "luajit21", "oximond"):
        call([*common, "-o", output / "compat53/utf8.so", compat / "lutf8lib.c"], log)
    # A native module must resolve Lua symbols from its host, never another Lua shared library.
    for module in output.rglob("*.so"):
        linked = call(["ldd", module], log)
        if re.search(r"liblua(?:jit|[\d.])", linked):
            raise RuntimeError(f"module links another Lua runtime: {module}")
        call(["nm", "-D", "--undefined-only", module], log)


def build(base, jobs):
    prepare(base)
    log = []
    try:
        for profile in PROFILES:
            src = source(base, profile)
            if profile == "luajit21":
                call(["make", "-C", src, f"-j{jobs}", "BUILDMODE=static"], log)
            else:
                call(["make", "-C", src, f"-j{jobs}", "linux"], log)
            build_modules(base, profile, src / "src", log)
        build_modules(base, "oximond", ROOT / "crates/oximond-capi/vendor/lua51", log)
        # Redis's modules define the fixed binary/JSON formats used by the Rust modules.
        # Redis embeds Lua 5.1; this separate oracle never masquerades as a modern ABI build.
        redis = source(base, "redis") / "deps/lua/src"
        output = base / "modules/redis"
        (output / "cjson").mkdir(parents=True, exist_ok=True)
        common = [os.environ.get("CC", "cc"), "-O2", "-fPIC", "-shared", "-I", source(base, "lua51") / "src"]
        call([*common, "-o", output / "cjson.so", redis / "lua_cjson.c", redis / "strbuf.c", redis / "fpconv.c", "-lm"], log)
        shutil.copyfile(output / "cjson.so", output / "cjson/safe.so")
        for module in ("struct", "cmsgpack"):
            call([*common, "-o", output / (module + ".so"), redis / ("lua_" + module + ".c")], log)
        shutil.copyfile(output / "cmsgpack.so", output / "cmsgpack_safe.so")
        for module in output.rglob("*.so"):
            linked = call(["ldd", module], log)
            if re.search(r"liblua(?:jit|[\d.])", linked):
                raise RuntimeError(f"module links another Lua runtime: {module}")
            call(["nm", "-D", "--undefined-only", module], log)
    finally:
        (base / "build-report.json").write_text(json.dumps(log, indent=2) + "\n")


def lua_quote(value):
    return json.dumps(str(value))


def prelude(base, variant):
    socket = source(base, "luasocket") / "src"
    paths = [str(base / "modules" / variant / "?.lua"), str(source(base, "lpeg") / "?.lua"),
             str(source(base, "teal") / "?.lua")]
    cpaths = [str(base / "modules" / variant / "?.so")]
    # Deliberately exclude system search paths, startup files, and LuaRocks loaders.
    return (f"package.path = {lua_quote(';'.join(paths))}\n"
            f"package.cpath = {lua_quote(';'.join(cpaths))}\n"
            "math.randomseed(271828)\n"
            "local function integration_kernel()\n"
            "   local sum = 0.0\n   for i = 1.0, 20000.0 do sum = sum + i end\n"
            "   return sum\nend\n"
            "local checksum = 0.0\nfor j = 1, 256 do checksum = checksum + integration_kernel() end\n"
            "assert(checksum == 51202560000)\n")


def script(base, case, variant, profile, http_port):
    body = prelude(base, variant)
    if case in ("lpeg-fixture", "socket-loopback", "socket-http", "teal-compiler", "bundled-modules", "bundled-edge"):
        body += f"local integration_http_port = {http_port or 0}\n"
        body += (HERE / "fixtures" / (case + ".lua")).read_text()
    elif case == "lpeg-upstream":
        body += f"dofile({lua_quote(source(base, 'lpeg') / 'test.lua')})\n"
    elif case.startswith("socket-"):
        filename = {"socket-except": "excepttest.lua", "socket-ltn12": "ltn12test.lua",
                    "socket-mime": "mimetest.lua", "socket-url": "urltest.lua"}[case]
        upstream = source(base, "luasocket") / "test"
        body += f"dofile({lua_quote(upstream / filename)})\n"
    elif case == "teal-bootstrap":
        body += (HERE / "fixtures/teal-bootstrap.lua").read_text().replace(
            "TEAL_SOURCE", lua_quote(source(base, "teal") / "tl.tl"))
    body += f"\nprint({lua_quote(MARKER)})\n"
    return body


@contextmanager
def http_fixture():
    payload = b"loopback\x00http\xff\r\n"

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            if self.path != "/binary?x=1":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("X-Oximond-Fixture", "local")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def limits(memory_mib):
    resource.setrlimit(resource.RLIMIT_AS, (memory_mib * 1024 ** 2,) * 2)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def execute(base, case, profile, variant, command, timeout, memory_mib, http_port=None, engine="reference"):
    work = base / "runs" / profile / variant / engine / case
    work.mkdir(parents=True, exist_ok=True)
    # Upstream LuaSocket tests use relative files and write generated MIME payloads.
    if case.startswith("socket-"):
        shutil.copyfile(source(base, "luasocket") / "test/testsupport.lua", work / "testsupport.lua")
    runner = work / "run.lua"
    runner.write_text(script(base, case, variant, profile, http_port))
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("LUA_", "LUA_INIT", "LUA_PATH", "LUA_CPATH"))}
    if variant == "oximond":
        env["OXIMOND_JIT_STATS"] = "1"
    invocation = [*map(str, command), str(runner)]
    try:
        result = subprocess.run(invocation, cwd=work, env=env, capture_output=True,
                                timeout=timeout, preexec_fn=lambda: limits(memory_mib))
        stdout = result.stdout.decode("utf-8", errors="replace")
        stderr = result.stderr.decode("utf-8", errors="replace")
        status = result.returncode
    except subprocess.TimeoutExpired as error:
        stdout = (error.stdout or b"").decode("utf-8", errors="replace")
        stderr = (error.stderr or b"").decode("utf-8", errors="replace") + "\nprocess deadline exceeded"
        status = "timeout"
    record = {"command": invocation, "exit": status, "stdout": stdout, "stderr": stderr,
              "passed": status == 0 and MARKER in stdout.splitlines()}
    generated = work / "bootstrap-tl.lua"
    if case == "teal-bootstrap" and record["passed"]:
        record["generated_sha256"] = sha(generated)
    (work / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def run(args):
    base = args.dir.resolve()
    verify_sources(base)
    binary = args.oximond.resolve()
    report = {"sources": SOURCES, "oximond": str(binary), "oximond_sha256": sha(binary),
              "manifest_sha256": sha(HERE / "sources.json"),
              "kind": "reference" if args.reference_only else "differential",
              "jit_evidence": "bounded numeric Lua kernel in each workload process",
              "results": []}
    if (base / "build-report.json").exists():
        report["build_report_sha256"] = sha(base / "build-report.json")
    report["module_sha256"] = {str(path.relative_to(base)): sha(path)
                               for path in sorted((base / "modules").rglob("*.so"))}
    report["fixture_sha256"] = {path.name: sha(path) for path in sorted((HERE / "fixtures").glob("*.lua"))}
    report["wrapper_sha256"] = {str(path.relative_to(base)): sha(path)
                                for path in sorted((base / "modules").rglob("*.lua"))}
    report["capi_header_sha256"] = {path.name: sha(path)
                                   for path in sorted((ROOT / "crates/oximond-capi/vendor/lua51").glob("*.h"))}
    profiles = args.profile or PROFILES
    cases = args.case or CASES
    engines = args.engine or ENGINES
    for profile in profiles:
        reference = source(base, profile) / "src" / ("luajit" if profile == "luajit21" else "lua")
        for case in cases:
            if case in ("bundled-modules", "bundled-edge"):
                reference = source(base, "lua51") / "src/lua"
                reference_variant = "redis"
            else:
                reference = source(base, profile) / "src" / ("luajit" if profile == "luajit21" else "lua")
                reference_variant = profile
            context = http_fixture() if case == "socket-http" else nullcontext(None)
            with context as port:
                baseline = execute(base, case, profile, reference_variant, [reference], args.timeout,
                                   args.memory_mib, port)
                if args.reference_only:
                    row = {"profile": profile, "engine": "reference", "case": case,
                           "passed": baseline["passed"], "reference": baseline}
                    report["results"].append(row)
                    print(f"{profile:8} reference   {case:18} {'PASS' if row['passed'] else 'FAIL'}", flush=True)
                    args.report.parent.mkdir(parents=True, exist_ok=True)
                    args.report.write_text(json.dumps(report, indent=2) + "\n")
                    continue
                for engine in engines:
                    result = execute(base, case, profile, "oximond", [binary, "--compat", profile,
                                     "--execution", engine], args.timeout, args.memory_mib, port, engine)
                    reasons = []
                    if not baseline["passed"]:
                        reasons.append("reference failed")
                    if not result["passed"]:
                        reasons.append("Oximond failed")
                    # Upstream logs contain timing/random chunk details; their assertions decide success.
                    differential = case.endswith("fixture") or case in ("socket-loopback", "socket-http", "teal-compiler", "teal-bootstrap", "bundled-modules", "bundled-edge")
                    if differential and result["stdout"] != baseline["stdout"]:
                        reasons.append("stdout differs from matching reference")
                    if case == "teal-bootstrap" and result.get("generated_sha256") != baseline.get("generated_sha256"):
                        reasons.append("bootstrap compiler bytes differ from matching reference")
                    stats = re.search(r"\bnative_entries: (\d+)", result["stderr"])
                    native_entries = int(stats.group(1)) if stats else None
                    if engine == "tracing" and (native_entries is None or native_entries == 0):
                        reasons.append("no measured native JIT executions")
                    row = {"profile": profile, "engine": engine, "case": case,
                           "passed": not reasons, "reasons": reasons, "native_entries": native_entries,
                           "reference": baseline, "oximond": result}
                    report["results"].append(row)
                    print(f"{profile:8} {engine:11} {case:18} {'PASS' if row['passed'] else 'FAIL'} "
                          + "; ".join(reasons), flush=True)
                    args.report.parent.mkdir(parents=True, exist_ok=True)
                    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if all(row["passed"] for row in report["results"]) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=ROOT / "target/lua-libraries")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--jobs", type=int, default=2)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--profile", action="append", choices=PROFILES)
    run_parser.add_argument("--engine", action="append", choices=ENGINES)
    run_parser.add_argument("--case", action="append", choices=CASES)
    run_parser.add_argument("--reference-only", action="store_true")
    run_parser.add_argument("--oximond", type=Path, default=ROOT / "target/release/oximond")
    run_parser.add_argument("--report", type=Path, default=ROOT / "target/lua-libraries/report.json")
    run_parser.add_argument("--timeout", type=int, default=300)
    run_parser.add_argument("--memory-mib", type=int, default=2048)
    args = parser.parse_args()
    args.dir = args.dir.resolve()
    args.dir.mkdir(parents=True, exist_ok=True)
    # Builds replace pinned source directories and runs write deterministic artifact paths.
    with (args.dir / ".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.command == "prepare":
            prepare(args.dir)
            return 0
        if args.command == "build":
            if args.jobs < 1:
                parser.error("--jobs must be positive")
            build(args.dir, args.jobs)
            return 0
        if args.timeout < 1 or args.memory_mib < 64:
            parser.error("--timeout must be positive and --memory-mib must be at least 64")
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
