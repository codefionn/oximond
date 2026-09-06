#!/usr/bin/env python3
"""Build and compare Neovim with Lua, LuaJIT, and Oximond using kickstart.nvim."""

import argparse
import hashlib
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import tarfile
import time
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "target" / "neovim"
SOURCE = BASE / "source"
KICKSTART = BASE / "kickstart"
PATCH = Path(__file__).with_name("neovim-0.12.5-oximond.patch")
KICKSTART_PATCH = Path(__file__).with_name("kickstart-compat.patch")
WORKLOAD = Path(__file__).with_name("workload.lua")
NEOVIM_URL = "https://github.com/neovim/neovim.git"
NEOVIM_TAG = "v0.12.5"
NEOVIM_REV = "5885a30e1e1225349079e7a1c4a3848aa8e43e42"
KICKSTART_URL = "https://github.com/nvim-lua/kickstart.nvim.git"
KICKSTART_REV = "80743df53d8f7058fc5b60e41f1081d11df9c880"
LUA_URL = "https://www.lua.org/ftp/lua-5.1.5.tar.gz"
LUA_SHA256 = "2640fc56a795f29d28ef15e13c34a47e223960b0240e8cb0a82d9b0738695333"
VARIANTS = ("lua", "luajit", "oximond")
MARKER = "OXIMOND_NVIM_WORKLOAD_OK"


def call(*command, cwd=None, env=None, capture=False):
    command = [str(part) for part in command]
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, cwd=cwd, env=env, check=True, text=True,
                          capture_output=capture)


def revision(path):
    return call("git", "rev-parse", "HEAD", cwd=path, capture=True).stdout.strip()


def clone_pinned(url, path, rev, branch=None):
    if not (path / ".git").is_dir():
        path.parent.mkdir(parents=True, exist_ok=True)
        command = ["git", "clone", "--depth", "1"]
        if branch:
            command += ["--branch", branch]
        call(*command, url, path)
    if revision(path) != rev:
        # Fetch the selected commit even when the branch has advanced.
        call("git", "fetch", "--depth", "1", "origin", rev, cwd=path)
        if call("git", "status", "--porcelain", cwd=path, capture=True).stdout.strip():
            raise RuntimeError(f"{path} has local changes; preserve them before checkout")
        call("git", "checkout", "--detach", rev, cwd=path)
    if revision(path) != rev:
        raise RuntimeError(f"{path} is not at the pinned commit {rev}")


def prepare():
    clone_pinned(NEOVIM_URL, SOURCE, NEOVIM_REV, NEOVIM_TAG)
    clone_pinned(KICKSTART_URL, KICKSTART, KICKSTART_REV)
    for checkout, patch in ((SOURCE, PATCH), (KICKSTART, KICKSTART_PATCH)):
        if not patch.is_file():
            raise RuntimeError(f"Missing integration patch: {patch}")
        reverse = subprocess.run(["git", "apply", "--reverse", "--check", str(patch)],
                                 cwd=checkout, capture_output=True)
        if reverse.returncode != 0:
            if call("git", "status", "--porcelain", cwd=checkout, capture=True).stdout.strip():
                raise RuntimeError(f"{checkout} has unrelated edits; inspect them before patching")
            call("git", "apply", "--check", patch, cwd=checkout)
            call("git", "apply", patch, cwd=checkout)
    for variant in VARIANTS:
        config = BASE / f"user-{variant}" / "config" / "nvim"
        config.parent.mkdir(parents=True, exist_ok=True)
        if config.is_symlink():
            if config.resolve() != KICKSTART.resolve():
                raise RuntimeError(f"Unexpected config link: {config}")
        elif config.exists():
            raise RuntimeError(f"Config path already exists: {config}")
        else:
            config.symlink_to(KICKSTART, target_is_directory=True)


def lua_aux_source():
    source = BASE / "lua-5.1.5" / "src" / "lauxlib.c"
    headers = ("lua.h", "luaconf.h", "lauxlib.h", "lualib.h")
    if source.is_file() and all((source.parent / header).is_file() for header in headers):
        return source
    archive = BASE / "lua-5.1.5.tar.gz"
    if not archive.is_file():
        print(f"+ download {LUA_URL}", flush=True)
        archive.write_bytes(urlopen(LUA_URL, timeout=60).read())
    if hashlib.sha256(archive.read_bytes()).hexdigest() != LUA_SHA256:
        raise RuntimeError(f"Lua 5.1.5 archive checksum mismatch: {archive}")
    with tarfile.open(archive, "r:gz") as bundle:
        required = {f"lua-5.1.5/src/{name}" for name in ("lauxlib.c", *headers)}
        members = [member for member in bundle.getmembers() if member.name in required]
        if {member.name for member in members} != required:
            raise RuntimeError("Lua archive lacks required C API sources or headers")
        bundle.extractall(BASE, members=members, filter="data")
    return source


def binary(variant):
    return BASE / f"build-{variant}" / "bin" / "nvim"


def build(variant, jobs):
    prepare()
    options = ["-S", SOURCE, "-B", BASE / f"build-{variant}", "-G", "Ninja",
               "-DCMAKE_BUILD_TYPE=Release", "-DCOMPILE_LUA=OFF"]
    if variant == "lua":
        options += ["-DPREFER_LUA=ON"]
    elif variant == "luajit":
        options += ["-DPREFER_LUA=OFF"]
    else:
        cargo_dir = BASE / "cargo"
        call("cargo", "build", "--release", "-p", "oximond-capi",
             "--target-dir", cargo_dir, cwd=ROOT)
        options += ["-DPREFER_LUA=ON", "-DENABLE_OXIMOND=ON",
                    f"-DOXIMOND_LIBRARY={cargo_dir / 'release/liboximond_capi.a'}",
                    f"-DOXIMOND_BRIDGE={ROOT / 'crates/oximond-capi/bridge.cpp'}",
                    f"-DOXIMOND_LAUXLIB_SOURCE={lua_aux_source()}",
                    f"-DOXIMOND_LUA_INCLUDE_DIR={BASE / 'lua-5.1.5/src'}"]
    call("cmake", *options)
    env = os.environ.copy()
    env["CCACHE_DIR"] = str(BASE / "ccache")
    call("cmake", "--build", BASE / f"build-{variant}", "-j", jobs, env=env)
    check_binary(variant)


def check_binary(variant):
    exe = binary(variant)
    if not exe.is_file():
        raise RuntimeError(f"Build {variant} first: {exe}")
    call(exe, "--version")
    if variant == "oximond":
        symbols = call("nm", exe, capture=True).stdout
        if " oximond_invoke" not in symbols:
            raise RuntimeError("Neovim binary is missing the Oximond C API bridge")
    elif os.name == "posix":
        linked = call("ldd", exe, capture=True).stdout.lower()
        library = "libluajit" if variant == "luajit" else "liblua5.1"
        if library not in linked:
            raise RuntimeError(f"{exe} is not linked to {library}")


def environment(variant, full_config=False):
    user = BASE / f"user-{variant}"
    env = os.environ.copy()
    for name, folder in (("XDG_CONFIG_HOME", "config"), ("XDG_DATA_HOME", "data"),
                         ("XDG_STATE_HOME", "state"), ("XDG_CACHE_HOME", "cache")):
        path = user / folder
        path.mkdir(parents=True, exist_ok=True)
        env[name] = str(path)
    env["VIMRUNTIME"] = str(SOURCE / "runtime")
    if not full_config:
        env["OXIMOND_NVIM_COMPARE"] = "1"
    if variant == "oximond":
        env["OXIMOND_NVIM_ENGINE"] = "1"
    return env


def workload_command(variant):
    script = str(WORKLOAD).replace("\\", "\\\\").replace("]", "\\]")
    return [str(binary(variant)), "--headless", "-c", f"lua dofile([[{script}]])", "-c", "qa!"]


def run_workload(variant, full_config=False):
    start = time.perf_counter()
    result = subprocess.run(workload_command(variant), cwd=ROOT,
                            env=environment(variant, full_config=full_config),
                            text=True, capture_output=True, timeout=120)
    elapsed = time.perf_counter() - start
    output = result.stdout + result.stderr
    startup_error = ("Error in " in output
                     or "Error executing " in output
                     or "Error detected while processing" in output
                     or re.search(r"\bE\d{3,4}:", output) is not None)
    if result.returncode != 0 or MARKER not in output or startup_error:
        mode = "full kickstart" if full_config else "core kickstart"
        raise RuntimeError(f"{variant} {mode} workload failed (exit {result.returncode}):\n{output}")
    return elapsed


def smoke(variant):
    check_binary(variant)
    elapsed = run_workload(variant)
    print(f"{variant}: workload passed in {elapsed:.3f}s")


def benchmark(variants, samples):
    if samples < 1:
        raise ValueError("samples must be positive")
    for variant in variants:
        smoke(variant)
    times = {variant: [] for variant in variants}
    randomizer = random.Random(20260930)
    for _ in range(samples):
        order = list(variants)
        randomizer.shuffle(order)
        for variant in order:
            times[variant].append(run_workload(variant))
    for variant in variants:
        print(f"{variant:8} median {statistics.median(times[variant]):.3f}s  "
              f"samples {[round(value, 3) for value in times[variant]]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    build_command = commands.add_parser("build")
    build_command.add_argument("variant", choices=(*VARIANTS, "all"))
    build_command.add_argument("--jobs", type=int, default=min(os.cpu_count() or 2, 4))
    smoke_command = commands.add_parser("smoke")
    smoke_command.add_argument("variant", choices=(*VARIANTS, "all"))
    full_smoke_command = commands.add_parser("full-smoke")
    full_smoke_command.add_argument("variant", choices=(*VARIANTS, "all"))
    run_command = commands.add_parser("run")
    run_command.add_argument("--full-config", action="store_true",
                             help="run the complete kickstart config")
    run_command.add_argument("variant", choices=VARIANTS)
    run_command.add_argument("editor_args", nargs=argparse.REMAINDER)
    bench_command = commands.add_parser("benchmark")
    bench_command.add_argument("variant", choices=(*VARIANTS, "all"), default="all", nargs="?")
    bench_command.add_argument("--samples", type=int, default=7)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "build":
        for variant in VARIANTS if args.variant == "all" else (args.variant,):
            build(variant, args.jobs)
    elif args.command == "smoke":
        for variant in VARIANTS if args.variant == "all" else (args.variant,):
            smoke(variant)
    elif args.command == "full-smoke":
        for variant in VARIANTS if args.variant == "all" else (args.variant,):
            check_binary(variant)
            elapsed = run_workload(variant, full_config=True)
            print(f"{variant}: full kickstart passed in {elapsed:.3f}s")
    elif args.command == "run":
        check_binary(args.variant)
        extra = args.editor_args[1:] if args.editor_args[:1] == ["--"] else args.editor_args
        raise SystemExit(subprocess.call([str(binary(args.variant)), *extra],
                                         cwd=ROOT,
                                         env=environment(args.variant, full_config=args.full_config)))
    else:
        benchmark(VARIANTS if args.variant == "all" else (args.variant,), args.samples)


if __name__ == "__main__":
    main()
