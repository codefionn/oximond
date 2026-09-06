#!/usr/bin/env python3
"""Build pinned Valkey and compare real embedded Lua scripting locally."""

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import socket
import subprocess
import tarfile
import threading
import time
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PIN = json.loads((HERE / "source.json").read_text())
PATCH = HERE / "valkey-8.1.3-oximond.patch"
PROFILES = ("lua51", "lua52", "lua53", "lua54", "lua55", "luajit21")
ENGINES = ("interpreter", "tracing")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rust_source_hashes(root):
    files = [root / "Cargo.toml", root / "Cargo.lock"]
    files.extend(path for path in (root / "crates").rglob("*")
                 if path.is_file() and "target" not in path.parts)
    return {str(path.relative_to(root)): sha(path) for path in sorted(files)}


def call(command, log, cwd=None):
    print("+", " ".join(map(str, command)), flush=True)
    result = subprocess.run(list(map(str, command)), cwd=cwd, text=True, capture_output=True)
    log.append({"command": list(map(str, command)), "exit": result.returncode,
                "cwd": str(cwd) if cwd is not None else str(Path.cwd()),
                "stdout": result.stdout, "stderr": result.stderr})
    if result.returncode:
        raise RuntimeError(result.stderr[-12000:] or result.stdout[-12000:])
    return result.stdout


def source(base, variant):
    return base / "trees" / variant / PIN["directory"]


def prepare(base, variant):
    archive = base / "downloads" / PIN["archive"]
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        data = urlopen(PIN["url"], timeout=60).read()
        if hashlib.sha256(data).hexdigest() != PIN["sha256"]:
            raise RuntimeError("Valkey archive checksum mismatch")
        archive.write_bytes(data)
    if sha(archive) != PIN["sha256"]:
        raise RuntimeError("Valkey archive checksum mismatch")
    parent = source(base, variant).parent
    if source(base, variant).exists():
        shutil.rmtree(source(base, variant))
    parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as bundle:
        bundle.extractall(parent, filter="data")
    log = []
    call(["git", "apply", "--check", PATCH], log, cwd=source(base, variant))
    call(["git", "apply", PATCH], log, cwd=source(base, variant))
    return log


def build(args):
    variants = ("stock", "oximond") if args.variant == "all" else (args.variant,)
    for variant in variants:
        log = prepare(args.dir, variant)
        source_hashes = None
        try:
            library = args.library.resolve() if args.library is not None else None
            if variant == "oximond" and (args.build_library or library is None):
                source_hashes = rust_source_hashes(args.source_root.resolve())
                cargo_target = args.dir / "cargo"
                cargo = ["cargo"] + (["+" + args.toolchain] if args.toolchain else [])
                call([*cargo, "build", "--release", "-p", "oximond-capi",
                      "--no-default-features", "--features", "lua51,bundled-bridge",
                      "--target-dir", cargo_target], log, cwd=args.source_root.resolve())
                library = cargo_target / "release/liboximond_capi.a"
                frozen_library = args.dir / "artifacts" / ("liboximond_capi-" + sha(library) + ".a")
                frozen_library.parent.mkdir(parents=True, exist_ok=True)
                if not frozen_library.exists():
                    shutil.copyfile(library, frozen_library)
                library = frozen_library
            command = ["make", "-C", source(args.dir, variant), f"-j{args.jobs}",
                       "MALLOC=libc", "OPT=-O2", "BUILD_TLS=no", "BUILD_RDMA=no", "USE_SYSTEMD=no"]
            if variant == "oximond":
                if not library.is_file():
                    raise RuntimeError(f"build the Lua 5.1 ABI bundled-bridge static library first: {library}")
                command.append(f"OXIMOND_LIBRARY={library}")
            command.append("valkey-server")
            call(command, log)
            binary = source(args.dir, variant) / "src/valkey-server"
            symbols = call(["nm", binary], log)
            if variant == "oximond" and " oximond_newstate_with_options" not in symbols:
                raise RuntimeError("server is missing Oximond's constructor")
            if variant == "stock" and " oximond_newstate_with_options" in symbols:
                raise RuntimeError("stock server unexpectedly contains Oximond")
            call([binary, "--version"], log)
        finally:
            report = {"source": PIN, "patch_sha256": sha(PATCH), "commands": log}
            if source_hashes is not None:
                report["oximond_source_sha256"] = source_hashes
            if variant == "oximond" and library is not None and library.is_file():
                report["oximond_library"] = str(library)
                report["oximond_library_sha256"] = sha(library)
                report["oximond_source_root"] = str(args.source_root.resolve())
            (args.dir / f"build-{variant}.json").write_text(json.dumps(report, indent=2) + "\n")


class ReplyError(Exception):
    pass


class Client:
    def __init__(self, path, timeout=5):
        self.connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.connection.settimeout(timeout)
        self.connection.connect(str(path))
        self.stream = self.connection.makefile("rb")

    def close(self):
        self.stream.close()
        self.connection.close()

    def reply(self):
        line = self.stream.readline()
        if not line or not line.endswith(b"\r\n"):
            raise RuntimeError("server closed or returned an incomplete RESP reply")
        marker, value = line[:1], line[1:-2]
        if marker == b"-":
            raise ReplyError(value.decode("utf-8", errors="replace"))
        if marker == b"+":
            return value
        if marker == b":":
            return int(value)
        if marker == b"$":
            length = int(value)
            if length == -1:
                return None
            data = self.stream.read(length)
            if len(data) != length or self.stream.read(2) != b"\r\n":
                raise RuntimeError("incomplete RESP bulk string")
            return data
        if marker == b"*":
            length = int(value)
            return None if length == -1 else [self.reply() for _ in range(length)]
        raise RuntimeError(f"unsupported RESP marker {marker!r}")

    def command(self, *parts):
        parts = [part if isinstance(part, bytes) else str(part).encode() for part in parts]
        request = b"*" + str(len(parts)).encode() + b"\r\n"
        for part in parts:
            request += b"$" + str(len(part)).encode() + b"\r\n" + part + b"\r\n"
        self.connection.sendall(request)
        return self.reply()


def normalize(value):
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, list):
        return list(map(normalize, value))
    if isinstance(value, dict):
        return {key: normalize(item) for key, item in value.items()}
    return value


def limit_process():
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024 ** 3,) * 2)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


@contextmanager
def server(base, variant, profile, engine):
    work = base / "runs" / variant / profile / engine
    work.mkdir(parents=True, exist_ok=True)
    # Keep UNIX socket names below Linux's 108-byte sockaddr_un path limit.
    socket_dir = Path("/tmp") / ("oximond-vk-" + hashlib.sha256(str(work).encode()).hexdigest()[:16])
    socket_dir.mkdir(mode=0o700, exist_ok=True)
    path = socket_dir / "server.sock"
    if path.exists():
        path.unlink()
    env = os.environ.copy()
    env["OXIMOND_VALKEY_PROFILE"] = profile
    env["OXIMOND_VALKEY_EXECUTION"] = engine
    binary = source(base, variant) / "src/valkey-server"
    command = [str(binary), "--port", "0", "--unixsocket", str(path), "--unixsocketperm", "700",
               "--save", "", "--appendonly", "no", "--daemonize", "no", "--protected-mode", "yes",
               "--busy-reply-threshold", "50", "--dir", str(work), "--logfile", str(work / "server.log")]
    with (work / "process.log").open("wb") as output:
        process = subprocess.Popen(command, cwd=work, env=env, stdout=output, stderr=subprocess.STDOUT,
                                   preexec_fn=limit_process)
        try:
            deadline = time.monotonic() + 15
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"{variant} server exited {process.returncode}; inspect {work}")
                try:
                    client = Client(path)
                    if client.command("PING") != b"PONG":
                        raise RuntimeError("unexpected readiness reply")
                    break
                except (OSError, RuntimeError):
                    if time.monotonic() >= deadline:
                        raise RuntimeError(f"server readiness deadline; inspect {work}")
                    time.sleep(0.05)
            yield client, path
        finally:
            if "client" in locals():
                client.close()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            if path.exists():
                path.unlink()


def expected_error(client, fragment, *command):
    try:
        client.command(*command)
    except ReplyError as error:
        assert fragment.lower() in str(error).lower(), str(error)
        return fragment
    raise AssertionError("command unexpectedly succeeded")


def workload(client, path, variant, profile, engine):
    checks = {}
    client.command("FLUSHALL")
    binary = b"a\0b\xff"
    checks["binary-call"] = client.command("EVAL", "server.call('SET',KEYS[1],ARGV[1]); return server.call('GET',KEYS[1])", 1, "fixture:key", binary)
    assert checks["binary-call"] == binary, ("binary-call", checks["binary-call"])
    script = "return {ARGV[1], tonumber(ARGV[2]) + 2, server.call('GET',KEYS[1])}"
    digest = client.command("SCRIPT", "LOAD", script)
    assert digest == hashlib.sha1(script.encode()).hexdigest().encode(), ("script digest", digest)
    checks["evalsha"] = client.command("EVALSHA", digest, 1, "fixture:key", binary, 40)
    assert checks["evalsha"] == [binary, 42, binary], ("evalsha", checks["evalsha"])
    checks["script-exists"] = client.command("SCRIPT", "EXISTS", digest, "0" * 40)
    assert checks["script-exists"] == [1, 0], ("script-exists", checks["script-exists"])
    assert client.command("SCRIPT", "FLUSH") == b"OK"
    checks["script-flush"] = client.command("SCRIPT", "EXISTS", digest)
    assert checks["script-flush"] == [0], ("script-flush", checks["script-flush"])
    expected_error(client, "NOSCRIPT", "EVALSHA", digest, 0)
    checks["errors"] = expected_error(client, "fixture failure", "EVAL", "error('fixture failure')", 0)
    checks["command-error"] = client.command("EVAL", "local result=server.pcall('INCR','fixture:key'); return {type(result.err),string.sub(result.err,1,3)}", 0)
    assert checks["command-error"] == [b"string", b"ERR"], ("command-error", checks["command-error"])
    checks["sandbox-globals"] = client.command("EVAL", "local names={'io','package','require','ffi','jit'}; for _,name in ipairs(names) do assert(rawget(_G,name)==nil) end; return 'restricted'", 0)
    assert checks["sandbox-globals"] == b"restricted", ("sandbox-globals", checks["sandbox-globals"])
    checks["sandbox-os"] = client.command("EVAL", "assert(type(os.clock)=='function'); for name in pairs(os) do assert(name=='clock') end; return 'clock only'", 0)
    assert checks["sandbox-os"] == b"clock only", ("sandbox-os", checks["sandbox-os"])
    checks["sandbox-write"] = expected_error(client, "readonly", "EVAL", "math.abs = 1; return 1", 0)
    checks["sandbox-rawset"] = expected_error(client, "readonly", "EVAL", "rawset(math,'abs',1); return 1", 0)
    checks["serialization"] = client.command("EVAL", "local json=cjson.decode('[true,null,42]'); assert(json[2]==cjson.null); local bytes=struct.pack('>I2i2',0x1234,-2); local data=cmsgpack.unpack(cmsgpack.pack({ARGV[1],42})); return {cjson.encode(json),bytes,data[1],data[2],bit.bxor(6,3)}", 0, binary)
    assert checks["serialization"] == [b"[true,null,42]", bytes.fromhex("1234fffe"), binary, 42, 5], ("serialization", checks["serialization"])
    library = "#!lua name=fixture\nserver.register_function('fixture_add',function(keys,args) return tonumber(args[1])+tonumber(args[2]) end)"
    assert client.command("FUNCTION", "LOAD", "REPLACE", library) == b"fixture"
    checks["fcall"] = client.command("FCALL", "fixture_add", 0, 20, 22)
    assert checks["fcall"] == 42, ("fcall", checks["fcall"])
    assert client.command("FUNCTION", "FLUSH") == b"OK"
    checks["function-flush"] = client.command("FUNCTION", "LIST")
    assert checks["function-flush"] == [], ("function-flush", checks["function-flush"])
    expected_error(client, "Function not found", "FCALL", "fixture_add", 0)
    # A separate connection must observe BUSY and kill a genuinely running script.
    outcome = []
    def run_busy():
        busy = Client(path, timeout=10)
        try:
            busy.command("EVAL", "local n=0; while true do n=n+1 end", 0)
            outcome.append("unexpected success")
        except ReplyError as error:
            outcome.append(str(error))
        except Exception as error:
            outcome.append(repr(error))
        finally:
            busy.close()
    thread = threading.Thread(target=run_busy, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    saw_busy = False
    while time.monotonic() < deadline:
        try:
            client.command("PING")
        except ReplyError as error:
            if "BUSY" in str(error):
                saw_busy = True
                break
            raise
        time.sleep(0.01)
    assert saw_busy, "instruction hook did not let the server report BUSY"
    assert client.command("SCRIPT", "KILL") == b"OK"
    thread.join(timeout=5)
    assert not thread.is_alive() and outcome and "killed" in outcome[0].lower(), outcome
    assert client.command("PING") == b"PONG"
    checks["timeout-kill"] = "busy script killed; server remained responsive"
    if variant == "oximond":
        version = "Lua 5.1" if profile == "luajit21" else "Lua 5." + profile[-1]
        assert client.command("EVAL", "return _VERSION", 0) == version.encode()
        # Oximond preserves instruction-hook semantics by interpreting hooked code.
        # The unbounded kill test above needs that hook; this bounded JIT probe does not.
        assert client.command("CONFIG", "SET", "busy-reply-threshold", 0) == b"OK"
        kernel = "local function kernel() local n=0.0; for i=1.0,20000.0 do n=n+i end; return n end; local total=0.0; for i=1,256 do total=total+kernel() end; assert(total==51202560000); return server.oximond_native_runs()"
        native = client.command("EVAL", kernel, 0)
        if engine == "tracing":
            assert native > 0, "numeric Lua kernel never entered native JIT code"
        else:
            assert native == 0, "interpreter process unexpectedly executed native JIT code"
        checks["native_runs"] = native
    return checks


def run(args):
    report = {"source": PIN, "patch_sha256": sha(PATCH), "results": [],
              "harness_sha256": sha(Path(__file__).resolve()),
              "manifest_sha256": sha(HERE / "source.json"),
              "binary_sha256": {variant: sha(source(args.dir, variant) / "src/valkey-server")
                                for variant in (("stock",) if args.reference_only else ("stock", "oximond"))}}
    report["build_report_sha256"] = {variant: sha(args.dir / f"build-{variant}.json")
                                     for variant in (("stock",) if args.reference_only else ("stock", "oximond"))}
    try:
        with server(args.dir, "stock", "lua51", "interpreter") as (client, path):
            baseline = workload(client, path, "stock", "lua51", "interpreter")
    except Exception as error:
        report["reference"] = {"passed": False, "error": str(error)}
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print("stock reference FAIL", str(error), flush=True)
        return 1
    report["reference"] = normalize(baseline)
    report["reference_passed"] = True
    if args.reference_only:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print("stock reference PASS", flush=True)
        return 0
    for profile in args.profile or PROFILES:
        for engine in args.engine or ENGINES:
            row = {"profile": profile, "engine": engine, "passed": False}
            try:
                with server(args.dir, "oximond", profile, engine) as (client, path):
                    result = workload(client, path, "oximond", profile, engine)
                common = {key: value for key, value in result.items() if key != "native_runs"}
                assert common == baseline, "workload replies differ from stock Valkey"
                row.update(passed=True, checks=normalize(result))
            except Exception as error:
                row["error"] = str(error)
            report["results"].append(row)
            print(profile, engine, "PASS" if row["passed"] else "FAIL", row.get("error", ""), flush=True)
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if all(row["passed"] for row in report["results"]) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=ROOT / "target/valkey")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("variant", choices=("stock", "oximond", "all"), default="all", nargs="?")
    build_parser = commands.add_parser("build")
    build_parser.add_argument("variant", choices=("stock", "oximond", "all"), default="all", nargs="?")
    build_parser.add_argument("--jobs", type=int, default=2)
    build_parser.add_argument("--library", type=Path,
                              help="use an existing Lua 5.1 ABI bundled-bridge static library instead of building")
    build_parser.add_argument("--build-library", action="store_true",
                              help="force a private Lua 5.1 ABI C API static library build (default unless --library is supplied)")
    build_parser.add_argument("--toolchain", help="select a named installed Rust toolchain")
    build_parser.add_argument("--source-root", type=Path, default=ROOT,
                              help="build the static library from another source checkout")
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--profile", action="append", choices=PROFILES)
    run_parser.add_argument("--engine", action="append", choices=ENGINES)
    run_parser.add_argument("--reference-only", action="store_true")
    run_parser.add_argument("--report", type=Path, default=ROOT / "target/valkey/report.json")
    args = parser.parse_args()
    args.dir = args.dir.resolve()
    args.dir.mkdir(parents=True, exist_ok=True)
    with (args.dir / ".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.command == "prepare":
            for variant in ("stock", "oximond") if args.variant == "all" else (args.variant,):
                prepare(args.dir, variant)
            return 0
        if args.command == "build":
            if args.jobs < 1:
                parser.error("--jobs must be positive")
            build(args)
            return 0
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
