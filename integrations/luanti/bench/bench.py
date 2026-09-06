"""Fixed Luanti workloads for comparing LuaJIT and Oximond.

Each run starts a headless server in a fresh world under
target/luanti/bench/user-<variant>, with a fixed seed and one emerge thread,
under `perf stat -e instructions:u,task-clock` pinned with taskset. The
oxbench worldmod (bench/oxbench) measures the workload and shuts the server
down. The driver in ../luanti.py supplies paths and checks through `Env`.
"""

import collections
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import tempfile
import time

MOD = Path(__file__).resolve().parent / "oxbench"
SEED = 1234567
DEFAULT_AREA = "64,-64,63"  # radius,ylo,yhi: x,z in [-64,63], y in [-64,63]
DEFAULT_CPUS = "4-7"
WORKLOADS = {
    "devtest-startup": ("devtest", "startup"),
    "mineclonia-startup": ("mineclonia", "startup"),
    "mineclonia-emerge": ("mineclonia", "emerge"),
}
PORTS = {"luajit": 30061, "oximond": 30062}


def server_port(preferred):
    """Returns `preferred` if no process holds that UDP port, else a free one.

    Several sessions and worktrees run benches at the same time; two servers on
    one port make the second exit with "Failed to bind socket". Ephemeral ports
    have five digits like the preferred ones, so the command line keeps its
    length."""
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET6, socket.SOCK_DGRAM) as probe:
            try:
                probe.bind(("::", port))
            except OSError:
                continue
            return probe.getsockname()[1]
    raise OSError("no free UDP port for the server")
PROBE_KEYS = ("jit", "ffi_global", "require_ffi", "ull")


def parse_area(text):
    """Parses "RADIUS,YLO,YHI" into ints; the area is x,z in [-R, R-1]."""
    try:
        radius, ylo, yhi = (int(part) for part in text.split(","))
    except ValueError:
        raise ValueError(f"--area wants RADIUS,YLO,YHI integers, got {text!r}")
    if radius <= 0 or ylo > yhi:
        raise ValueError(f"--area {text!r} is empty")
    return radius, ylo, yhi


def parse_perf_stat(text):
    """Reads `perf stat -x,` output: instructions and task-clock in ms."""
    out = {}
    for line in text.splitlines():
        parts = line.split(",")
        if len(parts) < 3:
            continue
        try:
            value = float(parts[0])
        except ValueError:
            continue  # "<not supported>" and similar
        if parts[2].startswith("instructions"):
            out["instructions"] = int(value)
        elif parts[2].startswith("task-clock"):
            out["cpu_ms"] = round(value, 1)
    return out


def parse_probe(log_text):
    """Returns the set of (jit, ffi_global, require_ffi, ull) tuples logged."""
    seen = set()
    for line in log_text.splitlines():
        if "[oxbench-probe]" not in line:
            continue
        fields = dict(re.findall(r"(\w+(?:\.\w+)?)=(\S+)", line.split("]", 2)[-1]))
        seen.add(tuple(fields.get(key, "?") for key in PROBE_KEYS))
    return seen


def compare_probes(probes):
    """Raises if the engines disagree on jit/ull/ffi availability."""
    for variant, seen in probes.items():
        if not seen:
            raise RuntimeError(f"{variant} logged no [oxbench-probe] line")
    first = None
    for variant, seen in probes.items():
        if first is None:
            first = (variant, seen)
        elif seen != first[1]:
            raise RuntimeError(
                f"probe mismatch ({'/'.join(PROBE_KEYS)}): "
                f"{first[0]}={sorted(first[1])} {variant}={sorted(seen)}")


ABORT_RE = re.compile(
    r"jit-trace: (?:side )?recording of the loop at pc \S+ (?:aborted|restarted) "
    r"at pc \d+ \((.*?)\) (.*?):(\d+): (\w+)\s*$")


def short_source(source):
    """Drops the "@" and the checkout prefix: builtin/... or mods/..."""
    source = source.lstrip("@")
    for marker in ("/luanti/source/", "/games/mineclonia/"):
        if marker in source:
            return source.split(marker, 1)[1]
    return source


def opcode_name(text):
    """"Some(Ok(TestEqual))" -> "TestEqual"."""
    words = re.findall(r"\w+", text)
    return words[-1] if words else text


def group_aborts(log_text, top=40):
    """Groups jit-trace abort lines by (opcode, reason, source:line)."""
    counts = collections.Counter()
    compiled = 0
    for line in log_text.splitlines():
        match = ABORT_RE.search(line)
        if match:
            op, source, lineno, reason = match.groups()
            counts[(opcode_name(op), reason, f"{short_source(source)}:{lineno}")] += 1
        elif "jit-trace: compiled " in line:
            compiled += 1
    return counts.most_common(top), compiled, sum(counts.values())


def format_aborts(log_text, top=40):
    rows, compiled, aborts = group_aborts(log_text, top)
    lines = [f"jit-trace: {compiled} compiled, {aborts} aborted/restarted recordings"]
    # The per-reason totals over every row, not only the top ones.
    every, _, _ = group_aborts(log_text, None)
    reasons = collections.Counter()
    for (_, reason, _), n in every:
        reasons[reason] += n
    lines.append("  totals: " + ", ".join(
        f"{reason}={reasons[reason]}" for reason in ("InnerLoop", "LoopNotClosed")))
    for (op, reason, where), n in rows:
        lines.append(f"  {n:6d}  {op:<14s} {reason:<22s} {where}")
    return "\n".join(lines)


CATEGORY_RULES = [
    ("parse/compile Lua source", r"syntax::|compiler::|bytecode::|Compiler|Lexer|Parser|ChunkMachine|PrattMachine|ConstantKey|validate|btree"),
    ("trace JIT compile/record", r"cranelift|regalloc|jit_ir|recorder|jit::trace::(build|compile|lower|optimize)"),
    ("JIT machine code", r"^\[JIT\]|^\[unknown\]|^0x"),
    ("gc", r"gc::|::collect|mark_|sweep"),
    ("C API boundary", r"oximond_capi|^lua_|^luaL_|^oxi_|bridge|call_native|native_open_call|protected_call"),
    ("interpreter", r"oximond::vm|run_fast|interpret|scalar_instruction|for_in"),
    ("runtime (tables/strings/values)", r"oximond::|table_storage|StringStorage|LuaString|Runtime|drop"),
    ("LuaJIT VM (lib)", r"LJ_|lj_|^libluajit"),
    ("libc", r"^libc"),
    ("luanti C++ / other", r"."),
]
REPORT_RE = re.compile(r"\s*([\d.]+)%\s+(\S+)\s+(\S+(?: tid \d+)?)\s+\[.\]\s+(.*)")


def categorize(report_text, threads=("luanti", "Emerge-0"), floor=0.3):
    """Sums `perf report --sort comm,dso,sym` percentages per thread and category."""
    totals = collections.Counter()
    shares = collections.Counter()
    for line in report_text.splitlines():
        match = REPORT_RE.match(line)
        if not match:
            continue
        pct, comm, dso, sym = float(match[1]), match[2], match[3], match[4]
        if dso.startswith("[JIT]"):
            key = "[JIT]"
        elif "luajit" in dso:
            key = "libluajit"
        elif dso.startswith("libc"):
            key = "libc"
        else:
            key = sym
        for name, rx in CATEGORY_RULES:
            if re.search(rx, key):
                totals[(comm, name)] += pct
                break
        shares[comm] += pct
    lines = []
    for comm in threads:
        lines.append(f"{comm} thread ({shares[comm]:.1f}% of samples)")
        for (c, name), v in sorted(totals.items(), key=lambda kv: -kv[1]):
            if c == comm and v >= floor:
                lines.append(f"  {name:34s} {v:5.1f}%")
    return "\n".join(lines)


def parse_lprof(log_text):
    """Last cumulative dump per Lua environment: {tag: (total, [(count, loc)])}."""
    blocks = {}
    for line in log_text.splitlines():
        match = re.search(r"\[lprof (\w+)\] (total )?(\d+)(?: (\S+))?\s*$", line)
        if not match:
            continue
        tag, is_total, number, loc = match.groups()
        if is_total:
            blocks[tag] = (int(number), [])
        elif loc and tag in blocks:
            blocks[tag][1].append((int(number), loc))
    return blocks


def format_lprof(log_text, top=25):
    blocks = parse_lprof(log_text)
    if not blocks:
        return "lua-profile: no [lprof] output in the log"
    lines = []
    for tag, (total, rows) in sorted(blocks.items()):
        lines.append(f"lua-profile {tag}: {total} samples (1 per 1000 VM instructions)")
        for count, loc in rows[:top]:
            lines.append(f"  {count:7d} {100 * count / max(total, 1):5.1f}%  {loc}")
    return "\n".join(lines)


def result_line(variant, game, workload, rc, wall, stat, fields, errors):
    res = {"variant": variant, "game": game, "workload": workload, "rc": rc,
           "wall_s": round(wall, 2),
           "instructions": stat.get("instructions"), "cpu_ms": stat.get("cpu_ms"),
           "emerge_us": int(fields["emerge_us"]) if "emerge_us" in fields else None,
           "blocks": int(fields["blocks"]) if "blocks" in fields else None}
    if errors:
        res["errors"] = errors[:3]
    return res


def ratio_table(results):
    """Oximond/LuaJIT ratios per workload from the last result of each variant."""
    by = {}
    for r in results:
        by[(r["workload"], r["variant"])] = r
    lines = []
    for workload in dict.fromkeys(r["workload"] for r in results):
        a, b = by.get((workload, "luajit")), by.get((workload, "oximond"))
        if not (a and b):
            continue

        def ratio(key):
            if a.get(key) and b.get(key):
                return f"{b[key] / a[key]:.2f}x"
            return "n/a"
        lines.append((workload, a["instructions"], b["instructions"],
                      ratio("instructions"), ratio("cpu_ms"), ratio("emerge_us")))
    if not lines:
        return ""
    out = [f"{'workload':<20s} {'luajit instr':>16s} {'oximond instr':>16s} {'instr':>7s} {'cpu':>7s} {'emerge':>7s}"]
    for w, ai, bi, ri, rc, re_ in lines:
        fmt = lambda n: f"{n:,d}" if n else "n/a"
        out.append(f"{w:<20s} {fmt(ai):>16s} {fmt(bi):>16s} {ri:>7s} {rc:>7s} {re_:>7s}")
    return "\n".join(out)


class Env:
    """Paths and hooks the driver supplies."""

    def __init__(self, base, source, binary, check_version, game_path):
        self.base = Path(base)  # target/luanti
        self.source = Path(source)
        self.binary = binary  # variant -> Path
        self.check_version = check_version
        self.game_path = game_path  # game -> Path of a checked-out game or None


def write_world(env, variant, game, workload, area, extra_conf):
    user = env.base / "bench" / f"user-{variant}"
    games = user / "games"
    games.mkdir(parents=True, exist_ok=True)
    if game != "devtest":
        link = games / game
        path = env.game_path(game)
        if link.is_symlink() and link.resolve() != Path(path).resolve():
            link.unlink()
        if not link.exists():
            link.symlink_to(path)
    world = user / "worlds" / workload
    shutil.rmtree(world, ignore_errors=True)
    (world / "worldmods").mkdir(parents=True)
    (world / "worldmods" / "oxbench").symlink_to(MOD)
    (world / "world.mt").write_text(
        f"gameid = {game}\nbackend = sqlite3\nplayer_backend = sqlite3\n"
        "auth_backend = sqlite3\nmod_storage_backend = sqlite3\n")
    radius, ylo, yhi = area
    mode = WORKLOADS[workload][1]
    conf = {"bind_address": "127.0.0.1", "server_announce": "false",
            "fixed_map_seed": SEED, "num_emerge_threads": 1,
            "max_block_generate_distance": 1000, "oxbench_mode": mode,
            "oxbench_radius": radius, "oxbench_ylo": ylo, "oxbench_yhi": yhi}
    conf.update(extra_conf)
    (user / "bench.conf").write_text("".join(f"{k} = {v}\n" for k, v in conf.items()))
    return user, world


def run_one(env, variant, workload, area, cpus, timeout, jit_trace=False,
            profile=False, lua_profile=False, attribution=False, artifacts=None):
    """Runs one workload; returns (result dict, log text, probe set, profile text).

    `attribution` (oximond only) runs the server with OXIMOND_ATTRIBUTION and
    OXIMOND_PERF_MAP under `perf record`; the sampler report and the perf
    report go into the `artifacts` dict as "attribution" and "perf"."""
    game = WORKLOADS[workload][0]
    env.check_version(variant)
    extra = {}
    if lua_profile:
        extra = {"oxbench_lprof": "true", "secure.trusted_mods": "oxbench"}
    user, world = write_world(env, variant, game, workload, area, extra)
    log = user / f"{workload}.log"
    child_env = dict(os.environ, LUANTI_USER_PATH=str(user), LC_ALL="C")  # perf stat -x, prints decimal commas otherwise
    child_env.pop("OXIMOND_JIT_TRACE", None)
    for name in ("OXIMOND_ATTRIBUTION", "OXIMOND_PERF_MAP"):
        child_env.pop(name, None)
    if jit_trace and variant == "oximond":
        child_env["OXIMOND_JIT_TRACE"] = "1"
    attributing = attribution and variant == "oximond"
    profile = profile or attributing
    scratch = Path(tempfile.mkdtemp(prefix="luanti-bench-"))
    try:
        stat = scratch / "stat.txt"
        data = scratch / "perf.data"
        attr_file = scratch / "attribution.tsv"
        if attributing:
            child_env["OXIMOND_ATTRIBUTION"] = str(attr_file)
            child_env["OXIMOND_PERF_MAP"] = "1"
        if profile:
            # OXI_PERF_CALLGRAPH=dwarf,16384 unwinds without frame pointers (per-caller shares
            # of the C API symbols); OXI_PERF_KEEP=<path> copies the perf.data out.
            # OXI_PERF_CALLGRAPH=none records flat user-space samples, for hosts where
            # perf_event_paranoid=2 refuses call chains.
            graph = os.environ.get("OXI_PERF_CALLGRAPH")
            if graph == "none":
                # OXI_PERF_EVENT=instructions:u samples retired instructions instead,
                # and OXI_PERF_FREQ raises the rate for line-level annotation.
                # OXI_PERF_EXTRA adds record flags, such as `-b` for branch
                # stacks.
                event = os.environ.get("OXI_PERF_EVENT", "cycles:u")
                freq = os.environ.get("OXI_PERF_FREQ", "999")
                extra = os.environ.get("OXI_PERF_EXTRA", "").split()
                perf = ["perf", "record", "-F", freq, "-e", event, *extra,
                        "-o", str(data)]
            else:
                perf = ["perf", "record", "-F", "499" if graph else "999",
                        f"--call-graph={graph}" if graph else "-g", "-o", str(data)]
        else:
            perf = ["perf", "stat", "-x,", "-e", "instructions:u,task-clock", "-o", str(stat)]
        cmd = ["timeout", str(timeout), *perf, "taskset", "-c", cpus,
               str(env.binary(variant)), "--server", "--world", str(world),
               "--gameid", game, "--config", str(user / "bench.conf"),
               "--port", str(server_port(PORTS[variant]))]
        t0 = time.monotonic()
        with log.open("w") as handle:
            rc = subprocess.run(cmd, cwd=env.source, env=child_env, stdout=handle,
                                stderr=subprocess.STDOUT).returncode
        wall = time.monotonic() - t0
        stats = parse_perf_stat(stat.read_text()) if stat.exists() else {}
        report = ""
        if profile and data.exists():
            report = subprocess.run(
                ["perf", "report", "-i", str(data), "--no-children", "--stdio",
                 "-g", "none", "--sort", "comm,dso,sym"],
                capture_output=True, text=True).stdout
        if profile and data.exists() and os.environ.get("OXI_PERF_KEEP"):
            shutil.copy(data, os.environ["OXI_PERF_KEEP"])
        if artifacts is not None:
            artifacts["perf"] = report
            if attributing and attr_file.exists():
                artifacts["attribution"] = attr_file.read_text(errors="replace")
        # The perf maps this run wrote; only ones that name traces are ours.
        for pid in set(re.findall(r"\[JIT\] tid (\d+)", report)):
            path = Path(f"/tmp/perf-{pid}.map")
            try:
                if "oximond trace#" in path.read_text(errors="replace"):
                    path.unlink()
            except OSError:
                pass
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    fields = {}
    out = world / "oxbench.txt"
    if out.exists():
        for line in out.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                fields[k] = v
    text = log.read_text(errors="replace")
    errors = [l for l in text.splitlines() if "ERROR[" in l]
    res = result_line(variant, game, workload, rc, wall, stats, fields, errors)
    return res, text, parse_probe(text), report


def parse_attribution(text):
    """Reads the sampler report (OXIMOND_ATTRIBUTION) into per-thread dicts."""
    threads = {}
    exits = collections.Counter()
    hz = 0
    for line in text.splitlines():
        f = line.split("\t")
        kind = f[0]
        if kind == "oximond-attribution":
            hz = float(f[2])
        elif kind == "exits":
            exits[(f[1], f[2])] += int(f[3])
        elif kind == "thread":
            threads[f[1]] = {"interp": int(f[2]), "native": int(f[3]), "stale": int(f[4]),
                             "host": int(f[5]) if len(f) > 5 else 0,
                             "funcs": {}, "lines": {}, "loops": {}}
        elif kind == "func":
            threads[f[1]]["funcs"][(f[2], int(f[3]))] = {
                "own": int(f[4]), "incl": int(f[5]), "native": int(f[6]),
                "state": f[7], "reason": f[8], "loops": int(f[9]),
                "via": (f[10], int(f[11]), int(f[12])) if len(f) > 12 and f[10] else None}
        elif kind == "line":
            threads[f[1]]["lines"][(f[2], int(f[3]))] = {
                "n": int(f[4]), "loop": f[5] if len(f) > 5 else ""}
        elif kind == "loop":
            threads[f[1]]["loops"][(f[2], int(f[3]), int(f[4]))] = {
                "line": int(f[5]), "inner": int(f[6]), "incl": int(f[7]), "native": int(f[8]),
                "state": f[9], "reason": f[10], "attempts": int(f[11]), "exits": int(f[12]),
                "entries": int(f[13]), "iters": int(f[14]),
                "kinds": f[15] if len(f) > 15 else "",
                "nearest": int(f[16]) if len(f) > 16 and f[16] else 0}
    return threads, exits, hz


def loop_state(loop):
    text = loop["state"]
    if loop["reason"]:
        text += f" ({short_reason(loop['reason'])})"
    if loop["state"] in ("compiled", "compiled-exited"):
        text += f" entries={loop['entries']} iters={loop['iters']} exits={loop['exits']}"
        if loop.get("kinds"):
            text += f" interpreter exits {loop['kinds'].replace(',', ' ')}"
    return text


def short_reason(reason):
    """"InnerLoop at @/x/builtin/a.lua:5" -> "InnerLoop at builtin/a.lua:5"."""
    head, _, where = reason.partition(" at ")
    return f"{head} at {short_source(where)}" if where else reason


def compiled_note(loop):
    """Why interpreter samples sit under a loop that has a trace."""
    if loop["state"] not in ("compiled", "compiled-exited", "covered"):
        return ""
    return (f" -> trace compiled but this code ran interpreted: {loop['exits']} counted exits "
            f"/ {loop['entries']} entries")


def function_rows(thread):
    """One row per sampled function: shares, the state of its hottest loop."""
    rows = []
    interp = max(thread["interp"], 1)
    for (source, defined), func in thread["funcs"].items():
        if func["own"] == 0 and func["native"] == 0:
            continue
        loops = [(key, loop) for key, loop in thread["loops"].items()
                 if key[0] == source and key[1] == defined]
        loops.sort(key=lambda kv: -(kv[1]["inner"] + kv[1]["native"]))
        if loops:
            loop = loops[0][1]
            state = f"loop@{loop['line']}: {loop_state(loop)}"
            state += compiled_note(loop)
        else:
            state = "no sampled loop"
            via = thread["loops"].get(func["via"]) if func["via"] else None
            if via:
                # Not in a loop of its own: the cost is the caller loop's.
                source_c, defined_c, _ = func["via"]
                state = (f"called from loop@{via['line']} of {short_source(source_c)}:{defined_c}: "
                         f"{loop_state(via)}{compiled_note(via)}")
            else:
                state += f"; entry {func['state']}"
                if func["reason"]:
                    state += f" ({short_reason(func['reason'])})"
        rows.append({"source": source, "defined": defined, "own": func["own"],
                     "share": 100.0 * func["own"] / interp,
                     "incl": 100.0 * func["incl"] / interp, "native": func["native"],
                     "state": state})
    rows.sort(key=lambda r: -r["own"])
    return rows


def perf_jit_rows(perf_text):
    """JIT samples per trace header from `perf report --sort comm,dso,sym`."""
    rows = collections.defaultdict(float)
    for line in perf_text.splitlines():
        match = REPORT_RE.match(line)
        if match and match[3].startswith("[JIT]") and "oximond trace#" in match[4]:
            rows[(match[2], match[4].strip())] += float(match[1])
    return rows


def format_attribution(text, perf_text="", top=30):
    """The attribution report: top interpreted functions and lines with their
    trace state, and JIT samples per trace header from perf."""
    threads, exits, hz = parse_attribution(text)
    if not threads:
        return "attribution: no samples (is OXIMOND_ATTRIBUTION honoured by this build?)"
    out = []
    jit = perf_jit_rows(perf_text)
    for comm, thread in sorted(threads.items(), key=lambda kv: -(kv[1]["interp"] + kv[1]["native"])):
        total = thread["interp"] + thread["native"]
        if total < 20:
            continue
        out.append(f"== thread {comm}: {thread['interp']} interpreter samples, "
                   f"{thread['native']} native, {thread['stale']} stale, "
                   f"{thread['host']} host ({hz:.0f} Hz)")
        out.append("  (the sampler sees the next backedge or call after a tick; a line is "
                   "the line of that backedge or call)")
        rows = function_rows(thread)
        out.append(f"  top {top} interpreted functions (own% = share of interpreter samples)")
        out.append(f"    {'own%':>5s} {'incl%':>6s}  function and trace state")
        for r in rows[:top]:
            if r["own"] == 0:
                continue
            out.append(f"    {r['share']:5.1f} {r['incl']:6.1f}  "
                       f"{short_source(r['source'])}:{r['defined']}  {r['state']}")
        interp = max(thread["interp"], 1)
        out.append(f"  top {top} interpreted lines")
        lines = sorted(thread["lines"].items(), key=lambda kv: -kv[1]["n"])[:top]
        for (source, line), info in lines:
            state = ""
            if info["loop"]:
                defined, pc = info["loop"].split(":")
                loop = thread["loops"].get((source, int(defined), int(pc)))
                if loop:
                    state = f"loop@{loop['line']}: {loop_state(loop)}"
            out.append(f"    {100.0 * info['n'] / interp:5.1f}  "
                       f"{short_source(source)}:{line}  {state}")
        uncovered = [r for r in rows if r["own"]]
        out.append("  top uncovered functions (every function with interpreter samples), "
                   "by interpreter share")
        for r in uncovered[:10]:
            out.append(f"    {r['share']:5.1f}  {short_source(r['source'])}:{r['defined']}  {r['state']}")
        # A function's row names one caller loop (the last sampled), and
        # functions of different chunks with one name and line (`=(load):12`)
        # share a row, so a loop's own cost is its inclusive count here.
        by_loop = sorted(((key, loop) for key, loop in thread["loops"].items()
                          if loop["nearest"] or loop["native"]),
                         key=lambda kv: -(kv[1]["nearest"] + kv[1]["native"]))[:top]
        if by_loop:
            out.append("  interpreter samples per loop (near% = the loop is the nearest loop on "
                       "the stack: its code and the loopless calls it makes; incl% = anywhere "
                       "on the stack; then native samples)")
            out.append(f"    {'near%':>5s} {'incl%':>6s} {'native':>6s}  loop and trace state")
            for (source, defined, pc), loop in by_loop:
                out.append(f"    {100.0 * loop['nearest'] / interp:5.1f} "
                           f"{100.0 * loop['incl'] / interp:6.1f} {loop['native']:6d}  "
                           f"{short_source(source)}:{loop['line']}  {loop_state(loop)}")
        native = sorted(((key, loop) for key, loop in thread["loops"].items() if loop["native"]),
                        key=lambda kv: -kv[1]["native"])[:10]
        if native:
            out.append(f"  native samples per trace header (sampler; {thread['native']} total)")
            for (source, defined, pc), loop in native:
                out.append(f"    {loop['native']:6d}  {short_source(source)}:{loop['line']}  "
                           f"{loop_state(loop)}")
        mine = sorted(((pct, name) for (c, name), pct in jit.items() if c == comm), reverse=True)
        if mine:
            out.append("  JIT samples per trace header (perf, % of all samples)")
            for pct, name in mine[:15]:
                out.append(f"    {pct:5.2f}%  {name}")
    if exits:
        out.append("== exits that went back to the interpreter, by kind and guard")
        for (kind, guard), n in exits.most_common(12):
            out.append(f"    {n:10d}  {kind}/{guard}")
    return "\n".join(out)
