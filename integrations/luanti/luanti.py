#!/usr/bin/env python3
"""Prepare, build, and run Luanti 5.17.0 against LuaJIT and Oximond."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "target" / "luanti"
SOURCE = BASE / "source"
PATCH = Path(__file__).with_name("luanti-5.17.0-oximond.patch")
REVISION = "c0e6812b1a4260bb25a1f606f70f55f4962bb97d"
URL = "https://github.com/luanti-org/luanti.git"
VARIANTS = ("luajit", "oximond")
GAMES_DIR = BASE / "games"
# Third-party games smoke-tested besides Luanti's bundled devtest. Each is a
# pinned shallow clone: (repository, tag, commit).
EXTERNAL_GAMES = {
    "mineclonia": ("https://codeberg.org/mineclonia/mineclonia.git", "0.123.1",
                   "cdb1329ef4f46988592f400a8a09d71c6a8878db"),
}
GAMES = ("devtest", *EXTERNAL_GAMES)
SHUTDOWN_TIMEOUT = 60


def call(*args, cwd=None, env=None, capture=False):
    command = [str(a) for a in args]
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, cwd=cwd, env=env, check=True,
                          text=True, capture_output=capture)


def prepare():
    BASE.mkdir(parents=True, exist_ok=True)
    if not (SOURCE / ".git").exists():
        call("git", "clone", "--depth", "1", "--branch", "5.17.0",
             "--single-branch", URL, SOURCE)
        call("git", "checkout", "--detach", REVISION, cwd=SOURCE)
    actual = call("git", "rev-parse", "HEAD", cwd=SOURCE, capture=True).stdout.strip()
    if actual != REVISION:
        raise RuntimeError(f"Luanti source is at {actual}, expected {REVISION}; preserve it and use a fresh source directory")
    reverse = subprocess.run(["git", "apply", "--reverse", "--check", str(PATCH)],
                             cwd=SOURCE, capture_output=True)
    if reverse.returncode == 0:
        print("Luanti patch already applied")
        return
    if call("git", "status", "--porcelain", cwd=SOURCE, capture=True).stdout.strip():
        raise RuntimeError("Luanti source has unrelated edits; inspect them before applying the patch")
    call("git", "apply", "--check", PATCH, cwd=SOURCE)
    call("git", "apply", PATCH, cwd=SOURCE)


def game_checkout(game):
    """Returns the pinned checkout of an external game, cloning it if missing."""
    url, tag, revision = EXTERNAL_GAMES[game]
    path = GAMES_DIR / game
    if not (path / ".git").exists():
        GAMES_DIR.mkdir(parents=True, exist_ok=True)
        call("git", "clone", "--depth", "1", "--branch", tag, url, path)
    actual = call("git", "rev-parse", "HEAD", cwd=path, capture=True).stdout.strip()
    if actual != revision:
        raise RuntimeError(f"{game} is at {actual}, expected {revision}; preserve it and use a fresh checkout")
    return path


def bench_game_path(game):
    """Like game_checkout, but a worktree reuses the main checkout's clone."""
    if not (GAMES_DIR / game / ".git").exists():
        common = subprocess.run(["git", "rev-parse", "--git-common-dir"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip()
        if common:
            shared = (ROOT / common).resolve().parent / "target" / "luanti" / "games" / game
            head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=shared,
                                  capture_output=True, text=True).stdout.strip()
            if head == EXTERNAL_GAMES[game][2]:
                return shared
    return game_checkout(game)


def prepare_game(game):
    """Clones a pinned external game once and links it into both user dirs."""
    if game not in EXTERNAL_GAMES:
        return
    path = game_checkout(game)
    for variant in VARIANTS:
        games = BASE / f"user-{variant}" / "games"
        games.mkdir(parents=True, exist_ok=True)
        link = games / game
        if not link.exists():
            link.symlink_to(path)


def binary(variant):
    return BASE / f"build-{variant}" / "bin" / "luanti"


def check_version(variant):
    exe = binary(variant)
    result = call(exe, "--version", capture=True)
    output = result.stdout + result.stderr
    marker = "Using LuaJIT" if variant == "luajit" else "Using Oximond"
    release = f"Luanti 5.17.0-{variant}"
    if marker not in output or release not in output:
        raise RuntimeError(f"{exe} did not identify itself as {release} with {marker}:\n{output}")
    print(output.strip())


def build_one(variant, jobs, capi_profile="luajit"):
    prepare()
    build_dir = BASE / f"build-{variant}"
    opts = ["-S", SOURCE, "-B", build_dir, "-DCMAKE_BUILD_TYPE=Release",
            f"-DVERSION_EXTRA={variant}",
            "-DBUILD_CLIENT=ON", "-DBUILD_SERVER=OFF", "-DBUILD_UNITTESTS=ON",
            "-DBUILD_DOCUMENTATION=OFF", "-DENABLE_SOUND=OFF",
            "-DENABLE_GETTEXT=OFF", "-DENABLE_CURL=OFF", "-DENABLE_LTO=OFF",
            "-DRUN_IN_PLACE=OFF", "-DOXIMOND_SIDE_BY_SIDE=ON",
            f"-DCUSTOM_SHAREDIR={SOURCE}"]
    if variant == "luajit":
        opts += ["-DENABLE_LUAJIT=ON", "-DREQUIRE_LUAJIT=ON", "-DENABLE_OXIMOND=OFF"]
    else:
        call("cargo", "build", "--release", "-p", "oximond-capi",
             "--no-default-features", "--features", f"{capi_profile},small-blocks",
             cwd=ROOT)
        library = ROOT / "target" / "release" / "liboximond_capi.a"
        if not library.is_file():
            raise RuntimeError(f"Missing Rust static library: {library}")
        # A Rust static library does not carry its native dependencies;
        # oximond's ffi_native module needs libffi and libdl after the archive.
        opts += ["-DENABLE_OXIMOND=ON", "-DENABLE_LUAJIT=OFF",
                 "-DCMAKE_CXX_STANDARD_LIBRARIES=-lffi -ldl",
                 f"-DOXIMOND_LIBRARY={library}",
                 f"-DOXIMOND_BRIDGE={ROOT / 'crates' / 'oximond-capi' / 'bridge.cpp'}"]
    call("cmake", *opts)
    call("cmake", "--build", build_dir, "--parallel", jobs)
    check_version(variant)


def launch(variant, extra):
    user_dir = BASE / f"user-{variant}"
    user_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["LUANTI_USER_PATH"] = str(user_dir)
    return subprocess.Popen([str(binary(variant)), *extra], cwd=SOURCE, env=env)


def stop_processes(processes):
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def smoke(variant):
    check_version(variant)
    if variant == "oximond":
        symbols = call("nm", binary(variant), capture=True).stdout
        if " oximond_invoke" not in symbols:
            raise RuntimeError("Oximond binary has no oximond_invoke bridge symbol")
    elif sys.platform.startswith("linux"):
        libraries = call("ldd", binary(variant), capture=True).stdout
        if "libluajit" not in libraries.lower():
            raise RuntimeError("LuaJIT binary is not linked to libluajit")
    user_dir = BASE / f"user-{variant}"
    user_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["LUANTI_USER_PATH"] = str(user_dir)
    call(binary(variant), "--run-unittests", "--test-module", "TestLua",
         cwd=SOURCE, env=env)


def world_smoke(variant, port, timeout, game="devtest"):
    check_version(variant)
    prepare_game(game)
    user_dir = BASE / f"user-{variant}"
    world = user_dir / "worlds" / ("smoke" if game == "devtest" else f"smoke-{game}")
    world.mkdir(parents=True, exist_ok=True)
    world_config = world / "world.mt"
    if not world_config.exists():
        world_config.write_text(
            f"gameid = {game}\nbackend = sqlite3\nplayer_backend = sqlite3\n"
            "auth_backend = sqlite3\nmod_storage_backend = sqlite3\n")
    server_config = user_dir / "smoke.conf"
    server_config.write_text("bind_address = 127.0.0.1\nserver_announce = false\n")
    env = os.environ.copy()
    env["LUANTI_USER_PATH"] = str(user_dir)
    command = [str(binary(variant)), "--server", "--world", str(world),
               "--gameid", game, "--config", str(server_config),
               "--port", str(port)]
    print("+", " ".join(command), flush=True)
    suffix = "" if game == "devtest" else f"-{game}"
    log_path = BASE / f"world-smoke-{variant}{suffix}.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=SOURCE, env=env,
                                   stdout=log, stderr=subprocess.STDOUT)
        ready = False
        alive_after_tick = False
        started_at = time.monotonic()
        deadline = started_at + timeout
        ready_after = None
        try:
            while time.monotonic() < deadline:
                output = log_path.read_text(errors="replace")
                if f'Server for gameid="{game}" listening' in output:
                    ready = True
                    ready_after = time.monotonic() - started_at
                    time.sleep(1)
                    alive_after_tick = process.poll() is None
                    break
                if process.poll() is not None:
                    break
                time.sleep(0.25)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    # Shutdown waits for emerge threads to finish loading the
                    # game's mapgen scripts; killing earlier loses mod storage.
                    process.wait(timeout=SHUTDOWN_TIMEOUT)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    output = log_path.read_text(errors="replace")
    print(output)
    if not ready or not alive_after_tick or "ERROR[" in output or "FATAL" in output:
        raise RuntimeError(f"{variant} did not start the {game} world")
    print(f"{variant} started the {game} world in {ready_after:.1f}s")


def bench(args):
    sys.path.insert(0, str(Path(__file__).with_name("bench")))
    import bench as lib
    area = lib.parse_area(args.area)
    variants = VARIANTS if args.variant == "both" else (args.variant,)
    workloads = list(lib.WORKLOADS) if args.workload == "all" else [args.workload]
    env = lib.Env(BASE, SOURCE, binary, check_version, bench_game_path)
    for variant in variants:
        if not binary(variant).is_file():
            raise RuntimeError(f"{binary(variant)} is missing; run: luanti.py build {variant}")
    results = []
    for _ in range(args.reps):
        for workload in workloads:
            probes = {}
            for variant in variants:
                artifacts = {}
                res, text, probe, report = lib.run_one(
                    env, variant, workload, area, args.cpus, args.timeout,
                    jit_trace=args.jit_trace, profile=args.profile,
                    lua_profile=args.lua_profile, attribution=args.attribution,
                    artifacts=artifacts)
                print(json.dumps(res), flush=True)
                results.append(res)
                probes[variant] = probe
                print(f"probe {variant}: {sorted(probe)}", flush=True)
                if res["rc"] != 0 or res.get("errors"):
                    print(f"warning: {variant} {workload} rc={res['rc']} errors={res.get('errors')}",
                          file=sys.stderr)
                if args.jit_trace and variant == "oximond":
                    print(lib.format_aborts(text, int(os.environ.get("BENCH_ABORT_ROWS", "40"))))
                if args.attribution and variant == "oximond":
                    print(lib.format_attribution(artifacts.get("attribution", ""),
                                                 artifacts.get("perf", ""), args.top))
                if args.profile:
                    print(lib.categorize(report))
                if args.lua_profile:
                    print(lib.format_lprof(text))
            lib.compare_probes(probes)
    table = lib.ratio_table(results)
    if table:
        print(table)
    return 1 if any(r["rc"] != 0 or r.get("errors") for r in results) else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    build_parser = sub.add_parser("build")
    build_parser.add_argument("variant", choices=(*VARIANTS, "both"))
    build_parser.add_argument("--jobs", type=int, default=min(os.cpu_count() or 2, 4))
    build_parser.add_argument("--capi-profile", choices=("luajit", "lua51"), default="luajit",
                              help="Lua dialect of the Oximond C API (a Cargo feature)")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("variant", choices=(*VARIANTS, "both"))
    run_parser.add_argument("engine_args", nargs=argparse.REMAINDER)
    smoke_parser = sub.add_parser("smoke")
    smoke_parser.add_argument("variant", choices=(*VARIANTS, "both"))
    world_parser = sub.add_parser("world-smoke")
    world_parser.add_argument("variant", choices=(*VARIANTS, "both"))
    world_parser.add_argument("--timeout", type=float, default=90,
                              help="seconds to wait for server readiness")
    world_parser.add_argument("--game", choices=(*GAMES, "all"), default="devtest",
                              help="game to start; external games are cloned at a pinned tag")
    bench_parser = sub.add_parser("bench", help="compare instructions:u on fixed workloads")
    bench_parser.add_argument("--variant", choices=(*VARIANTS, "both"), default="both")
    bench_parser.add_argument("--workload", default="mineclonia-emerge",
                              choices=("devtest-startup", "mineclonia-startup",
                                       "mineclonia-emerge", "all"))
    bench_parser.add_argument("--reps", type=int, default=1)
    bench_parser.add_argument("--area", default="64,-64,63",
                              help="emerge area RADIUS,YLO,YHI: x,z in [-R,R-1], y in [YLO,YHI]")
    bench_parser.add_argument("--cpus", default="4-7",
                              help="taskset list; CPUs N and N+8 share a core")
    bench_parser.add_argument("--timeout", type=float, default=1200)
    bench_parser.add_argument("--jit-trace", action="store_true",
                              help="oximond: print JIT aborts grouped by opcode, reason, source:line")
    bench_parser.add_argument("--profile", action="store_true",
                              help="perf record -g and print a per-thread category breakdown")
    bench_parser.add_argument("--lua-profile", action="store_true",
                              help="count-hook line sampler; prints top Lua lines per environment")
    bench_parser.add_argument("--attribution", action="store_true",
                              help="oximond: interpreter sampler plus perf map; print the top "
                                   "interpreted functions and lines with their trace state "
                                   "and JIT samples per trace header")
    bench_parser.add_argument("--top", type=int, default=30,
                              help="rows per table of the --attribution report")
    args = parser.parse_args()
    if args.command == "bench":
        return bench(args)
    if args.command == "prepare":
        prepare()
    elif args.command == "build":
        for variant in VARIANTS if args.variant == "both" else (args.variant,):
            build_one(variant, args.jobs, args.capi_profile)
    elif args.command == "smoke":
        for variant in VARIANTS if args.variant == "both" else (args.variant,):
            smoke(variant)
    elif args.command == "world-smoke":
        games = GAMES if args.game == "all" else (args.game,)
        for game in games:
            for variant in VARIANTS if args.variant == "both" else (args.variant,):
                world_smoke(variant, 30041 if variant == "luajit" else 30042,
                            args.timeout, game)
    else:
        extra = args.engine_args
        if extra and extra[0] == "--":
            extra = extra[1:]
        variants = VARIANTS if args.variant == "both" else (args.variant,)
        for variant in variants:
            check_version(variant)
        processes = []
        try:
            for variant in variants:
                variant_args = list(extra)
                if args.variant == "both" and not any(
                        item == "--port" or item.startswith("--port=") for item in extra):
                    variant_args += ["--port", "30041" if variant == "luajit" else "30042"]
                processes.append(launch(variant, variant_args))
            for process in processes:
                process.wait()
        except KeyboardInterrupt:
            stop_processes(processes)
            return 130
        except BaseException:
            stop_processes(processes)
            raise
        if any(process.returncode for process in processes):
            return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
