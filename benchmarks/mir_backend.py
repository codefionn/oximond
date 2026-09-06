#!/usr/bin/env python3
"""Compare the Cranelift and MIR trace backends on the benchmark suite.

    scripts/heavy.sh -m 16G python3 benchmarks/mir_backend.py diff OUT.json
    scripts/heavy.sh -m 16G python3 benchmarks/mir_backend.py coverage OUT.json
    scripts/heavy.sh -x python3 benchmarks/mir_backend.py icount OUT.json [--cpu N]
    scripts/heavy.sh -x python3 benchmarks/mir_backend.py kernels OUT.json [--cpu N]
    python3 benchmarks/mir_backend.py report OUT.json

Every mode runs the suite entries (all of benchmarks/suite.json, or the names
given) with `--execution=tracing`, once per backend, switching only
OXIMOND_JIT_BACKEND on the same binary, so the path and the environment have
the same length.

diff      stdout and exit status must be equal under both backends.
coverage  runs under mir with OXIMOND_JIT_TRACE=1 and OXIMOND_JIT_SYNC=1 and
          counts the compiles the MIR backend accepted and the ones it
          handed to Cranelift, in trace IR instructions, by reason.
icount    pinned `perf stat -e instructions:u` of the whole process under each
          backend, and with OXIMOND_JIT_ICOUNT=1 the instructions of the
          compile thread per tree (`jit-icount:` lines). Run it through the
          gate with -x, and note the load average it prints.

kernels   steady-state cost of the code the Optimized tier runs: one loop per
          kernel, run for 300k and for 1.5M iterations (under the 2M a loop
          may run before the Peak tier recompiles it with Cranelift), the
          difference of the two instruction counts over the difference of
          the iterations, per backend, pinned like icount.

Options: --bin BIN (default target/release/oximond), --probe 0|1 (set
OXIMOND_JIT_MIR_PROBE under both backends; the probe is on without it, so 0
is the probe-off run), --opt-flags BITS (replace
the optimization flags, such as 0xb to run without range proofs), --scale N (default 3),
--timeout S (default 300), --cpu N (icount; default: a CPU below 8 whose SMT
sibling N+8 is idle), and benchmark names.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKENDS = ("cranelift", "mir")


def pad(backend):
    """A value for `PAD` that makes the environment as long under every
    backend: the initial stack alignment follows its size and moves counts
    by a few percent."""
    return "x" * (max(map(len, BACKENDS)) - len(backend))


def entries(names):
    suite = json.load(open(ROOT / "benchmarks" / "suite.json"))
    by_name = {e["name"]: e for e in suite}
    return [by_name[n] for n in names] if names else suite


def command(args, entry):
    return [
        args.bin,
        f"--compat={entry.get('compat', 'lua51')}",
        "--execution=tracing",
        "-E",
        *([f"--opt-flags={args.opt_flags}"] if args.opt_flags else []),
        str(ROOT / "benchmarks" / entry["path"]),
        str(args.scale),
    ]


def environment(args, backend):
    """The environment of one run: the same variables under every backend, so
    its length does not depend on the backend. `--probe` sets
    OXIMOND_JIT_MIR_PROBE under both (it has no effect under cranelift); `0`
    turns the MIR probe off, which is on when the variable is not set."""
    env = {"PATH": "/usr/bin:/bin", "OXIMOND_JIT_BACKEND": backend, "PAD": pad(backend)}
    if args.probe is not None:
        env["OXIMOND_JIT_MIR_PROBE"] = args.probe
    return env


def run(args, entry, backend, extra_env=None, wrap=()):
    env = environment(args, backend)
    env.update(extra_env or {})
    try:
        done = subprocess.run(
            [*wrap, *command(args, entry)],
            env=env,
            capture_output=True,
            timeout=args.timeout,
            cwd=ROOT,
        )
    except subprocess.TimeoutExpired:
        return None
    return done


def diff(args):
    result = {}
    for entry in entries(args.names):
        outputs = {}
        for backend in BACKENDS:
            done = run(args, entry, backend)
            outputs[backend] = None if done is None else (done.returncode, done.stdout)
        same = outputs["cranelift"] == outputs["mir"] and outputs["mir"] is not None
        result[entry["name"]] = {
            "same": same,
            "timeout": any(v is None for v in outputs.values()),
            "status": {b: (o[0] if o else None) for b, o in outputs.items()},
        }
        print(f"{entry['name']:<36} {'same' if same else 'DIFFERENT'}", flush=True)
    return result


FALLBACK = re.compile(r"jit-trace: mir backend fallback to cranelift \((\d+) IR instructions\): (?:instruction \d+: )?(.*)")
COMPILED = re.compile(r"jit-trace: mir backend compiled tree: (\d+) IR instructions, (\d+) MIR instructions, (\d+) bytes")


def coverage(args):
    result = {}
    env = {"OXIMOND_JIT_TRACE": "1", "OXIMOND_JIT_SYNC": "1"}
    for entry in entries(args.names):
        done = run(args, entry, "mir", env)
        if done is None:
            result[entry["name"]] = {"timeout": True}
            continue
        accepted = [0, 0, 0, 0]  # compiles, ir, mir insts, bytes
        fallback = [0, 0]
        reasons = defaultdict(lambda: [0, 0])
        for line in done.stderr.decode(errors="replace").splitlines():
            m = COMPILED.match(line)
            if m:
                accepted[0] += 1
                for k in range(3):
                    accepted[k + 1] += int(m.group(k + 1))
                continue
            m = FALLBACK.match(line)
            if m:
                fallback[0] += 1
                fallback[1] += int(m.group(1))
                reasons[m.group(2)][0] += 1
                reasons[m.group(2)][1] += int(m.group(1))
        result[entry["name"]] = {
            "accepted": accepted,
            "fallback": fallback,
            "reasons": reasons,
        }
        print(f"{entry['name']:<36} accepted {accepted[0]:>3} fallback {fallback[0]:>3}", flush=True)
    return result


ICOUNT = re.compile(r"jit-icount: (tree|side|job tree|job side) backend=(\S+) ir=(\d+) instructions=(\d+)")


def total_instructions(args, entry, backend, pin):
    done = run(
        args,
        entry,
        backend,
        wrap=[*pin, "perf", "stat", "-x,", "-e", "instructions:u"],
    )
    if done is None:
        return None
    for line in done.stderr.decode(errors="replace").splitlines():
        if "instructions" in line:
            return int(line.split(",")[0])
    return None


def idle_pair():
    """A CPU N below 8 whose SMT sibling N+8 is idle too, by two samples of
    /proc/stat a second apart. The least busy pair if none is idle."""
    import time

    def sample():
        rows = {}
        for line in open("/proc/stat"):
            if line.startswith("cpu") and line[3].isdigit():
                f = line.split()
                values = list(map(int, f[1:]))
                rows[int(f[0][3:])] = (values[3] + values[4], sum(values))
        return rows

    a = sample()
    time.sleep(1)
    b = sample()

    def busy(cpu):
        idle = b[cpu][0] - a[cpu][0]
        total = b[cpu][1] - a[cpu][1]
        return 1 - idle / total if total else 1.0

    return min(range(8), key=lambda n: busy(n) + busy(n + 8))


def icount(args):
    print("load average:", open("/proc/loadavg").read().split()[:3], file=sys.stderr)
    if args.cpu < 0:
        args.cpu = idle_pair()
        print(f"pinned to CPU {args.cpu} (sibling {args.cpu + 8})", file=sys.stderr)
    pin = ["taskset", "-c", str(args.cpu)]
    result = {}
    base_scale = args.scale
    for entry in entries(args.names):
        row = {}
        for backend in BACKENDS:
            args.scale = base_scale
            total = total_instructions(args, entry, backend, pin)
            # The same run again at twice the scale: the difference is the
            # steady-state cost of the extra work, with warmup and compile
            # cancelled.
            args.scale = base_scale * 2
            doubled = total_instructions(args, entry, backend, pin)
            args.scale = base_scale
            compiles = []
            done = run(args, entry, backend, {"OXIMOND_JIT_ICOUNT": "1"}, wrap=pin)
            if done is not None:
                for line in done.stderr.decode(errors="replace").splitlines():
                    m = ICOUNT.match(line)
                    if m:
                        compiles.append((m.group(1), m.group(2), int(m.group(3)), int(m.group(4))))
            row[backend] = {"instructions": total, "doubled": doubled, "compiles": compiles}
        result[entry["name"]] = row
        a, b = row["cranelift"]["instructions"], row["mir"]["instructions"]
        print(f"{entry['name']:<36} cranelift {a} mir {b}", flush=True)
    result["_load"] = open("/proc/loadavg").read().split()[:3]
    result["_cpu"] = args.cpu
    return result


KERNELS = {
    "float arithmetic": ("lua51", "local x, y = 0, 1 for i = 1, N do x = x * 0.999 + i * 0.5 - y / 3 y = y * 1.0000001 end print(x, y)"),
    "float modulo and floor": ("lua51", "local s = 0 for i = 1, N do s = s + (i % 7) + math.floor(i / 3) end print(s)"),
    "integer arithmetic": ("lua54", "local a, b = 0, 1 for i = 1, N do a = a + (i % 8) b = (b * 3 + i) & 0xffff a = a + (i // 4) end print(a, b)"),
    "integer division by 7": ("lua54", "local a = 0 for i = 1, N do a = a + (i % 7) + (i // 7) end print(a)"),
    "integer modulo by a value": ("lua54", "local acc, h = 0, 0 local d = { 7, -7, 13, -13, 1, -1, 1000003, -64 } for i = 1, N do h = (h + 2654435761) % 4294967296 local n = h - 2147483648 acc = (acc + n % d[i % 8 + 1] + i % 97) % 1000000007 end print(acc)"),
    "array sum, float keys": ("lua51", "local t = {} for i = 1, 1000 do t[i] = i * 0.5 end local s = 0 for i = 1, N do s = s + t[(i % 1000) + 1] end print(s)"),
    "array sum, integer keys": ("lua54", "local t = {} for i = 1, 1000 do t[i] = i * 0.5 end local s = 0.0 for i = 1, N do s = s + t[(i % 1000) + 1] end print(s)"),
    "array update": ("lua54", "local t = {} for i = 1, 1000 do t[i] = i end for i = 1, N do local k = (i % 1000) + 1 t[k] = t[k] + 1 end print(t[1], t[1000])"),
    "table fields": ("lua51", "local o = {x = 1, y = 2, z = 3} for i = 1, N do o.x = o.x + o.y * 0.5 o.z = o.z + o.x end print(o.x, o.z)"),
    # The string-index probe.
    "hash lookup, key and table share the string": ("lua54", "local keys, t = {}, {} for i = 1, 200 do keys[i] = 'key_' .. i t[keys[i]] = i end local s = 0 for i = 1, N do s = s + t[keys[i % 200 + 1]] end print(s)"),
    "hash lookup, equal strings in two allocations": ("lua54", "local t, probes = {}, {} for i = 1, 200 do t['key_' .. i] = i probes[i] = 'key_' .. i end local s = 0 for i = 1, N do s = s + t[probes[i % 200 + 1]] end print(s)"),
    "hash lookup of an absent key": ("lua54", "local t, probes = {}, {} for i = 1, 200 do t['key_' .. i] = i probes[i] = 'gone_' .. i end local s = 0 for i = 1, N do if t[probes[i % 200 + 1]] == nil then s = s + 1 end end print(s)"),
    "hash lookup in 2000 keys": ("lua54", "local t, keys = {}, {} for i = 1, 2000 do keys[i] = 'key_' .. i t[keys[i]] = i % 107 end local s = 0 for i = 1, N do s = s + t[keys[i % 2000 + 1]] end print(s)"),
    "hash store over a present key": ("lua54", "local t, keys = {}, {} for i = 1, 2000 do keys[i] = 'key_' .. i t[keys[i]] = 0 end for i = 1, N do t[keys[i % 2000 + 1]] = i % 101 end print(t[keys[1]])"),
    "hash delete and revive": ("lua54", "local t, keys = {}, {} for i = 1, 2000 do keys[i] = 'key_' .. i t[keys[i]] = 0 end for i = 1, N do local k = keys[i % 2000 + 1] t[k] = nil t[k] = i % 101 end print(t[keys[1]])"),
    "branches and select": ("lua51", "local a, b = 0, 0 for i = 1, N do if i > 0 then a = a + 1 end local m = (a > b) and a or b b = b + m * 0.001 end print(a, b)"),
}


def kernels(args):
    import tempfile

    print("load average:", open("/proc/loadavg").read().split()[:3], file=sys.stderr)
    if args.cpu < 0:
        args.cpu = idle_pair()
        print(f"pinned to CPU {args.cpu} (sibling {args.cpu + 8})", file=sys.stderr)
    pin = ["taskset", "-c", str(args.cpu)]
    small, large = 300_000, 1_500_000
    result = {}
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as scratch:
        for name, (compat, source) in KERNELS.items():
            row = {}
            for backend in BACKENDS:
                counts = {}
                for n in (small, large):
                    path = Path(scratch) / "kernel.lua"
                    path.write_text(source.replace("N", str(n)))
                    env = environment(args, backend)
                    done = subprocess.run(
                        [*pin, "perf", "stat", "-x,", "-e", "instructions:u", args.bin,
                         f"--compat={compat}", "--execution=tracing", "-E", str(path)],
                        env=env, capture_output=True, timeout=args.timeout, cwd=ROOT,
                    )
                    for line in done.stderr.decode(errors="replace").splitlines():
                        if "instructions" in line:
                            counts[n] = int(line.split(",")[0])
                per_iteration = (counts[large] - counts[small]) / (large - small)
                row[backend] = {"counts": counts, "per_iteration": per_iteration}
            result[name] = row
            a, b = row["cranelift"]["per_iteration"], row["mir"]["per_iteration"]
            print(f"{name:<28} cranelift {a:7.2f} mir {b:7.2f}  {100 * (b / a - 1):+.1f}%", flush=True)
    result["_load"] = open("/proc/loadavg").read().split()[:3]
    result["_cpu"] = args.cpu
    return result


def report(args):
    data = json.load(open(args.out))
    print(json.dumps(data, indent=1)[:2000])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["diff", "coverage", "icount", "kernels", "report"])
    parser.add_argument("out")
    parser.add_argument("names", nargs="*")
    parser.add_argument("--bin", default=str(ROOT / "target" / "release" / "oximond"))
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--cpu", type=int, default=-1)
    parser.add_argument("--probe", choices=["0", "1"], help="set OXIMOND_JIT_MIR_PROBE to this under both backends (the probe is on without it; 0 turns it off)")
    parser.add_argument("--opt-flags", default="", help="pass --opt-flags=BITS to every run")
    args = parser.parse_args()
    if args.mode == "report":
        args.out = args.out
        report(args)
        return
    result = {"diff": diff, "coverage": coverage, "icount": icount, "kernels": kernels}[args.mode](args)
    json.dump(result, open(args.out, "w"), indent=1, default=lambda o: dict(o))
    if args.mode == "diff" and not all(v["same"] for v in result.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
