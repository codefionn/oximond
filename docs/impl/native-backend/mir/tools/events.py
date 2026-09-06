#!/usr/bin/env python3
"""Parse the `jit-lat: ev` lines of the instrumented build.

  events.py LOG [LOG...]          per-compile breakdown tables (markdown)
  events.py --fit LOG [LOG...]    one row per compile: n_ir, clif, instructions
  events.py --latency LOG         submission to first native entry timeline

Each line is `jit-lat: ev NAME tid=T ic=N t=Instant { tv_sec: S, tv_nsec: NS }`.
`ic` is the thread's exact user instruction count (perf_event_open, own thread,
exclude_kernel), minus what earlier events spent printing.
"""
import re
import sys
from collections import defaultdict

LINE = re.compile(r"^jit-lat: ev (.*?) tid=(\d+) ic=(\d+)(?: nat=(\d+))?(?: cpu=(\d+) vcs=(\d+) ivcs=(\d+))? t=Instant \{ tv_sec: (\d+), tv_nsec: (\d+) \}")


def parse(path):
    evs = []
    for line in open(path, errors="replace"):
        m = LINE.match(line)
        if m:
            evs.append(dict(name=m[1], tid=int(m[2]), ic=int(m[3]), nat=int(m[4] or 0), cpu=int(m[5] or 0),
                            vcs=int(m[6] or 0), ivcs=int(m[7] or 0), t=int(m[8]) + int(m[9]) / 1e9))
    return evs


def base(name):
    """The event name without its key=value details and spaces."""
    return re.sub(r"\s\w+=\S+", "", name).replace(" ", "_")


def kv(name):
    return {k: v for k, v in re.findall(r"(\w+)=(\w+)", name)}


def compiles(evs):
    """Split into compiles: compile_begin ... compile_end on one thread."""
    out = []
    open_ = {}
    for e in evs:
        if e["name"].startswith("compile_begin"):
            open_[e["tid"]] = dict(id=int(kv(e["name"])["id"]), tid=e["tid"], evs=[e])
        elif e["tid"] in open_:
            c = open_[e["tid"]]
            c["evs"].append(e)
            if e["name"] == "compile_end":
                out.append(open_.pop(e["tid"]))
    return out


def summarize(c):
    evs = c["evs"]
    first, last = evs[0], evs[-1]
    r = dict(id=c["id"], tid=c["tid"], total=last["ic"] - first["ic"], t_us=(last["t"] - first["t"]) * 1e6)
    r["side"] = any(e["name"].startswith("side:") for e in evs)
    # ir sizes
    for e in evs:
        if e["name"].startswith("tree:ir "):
            d = kv(e["name"])
            r.update(members=int(d["members"]), n_pre=int(d["pre_opt"]), n_ir=int(d["post_opt"]),
                     guards=int(d["guards"]), snapshots=int(d["snapshots"]))
        if e["name"].startswith("side:opt:b"):
            r["n_pre"] = int(kv(e["name"])["n"])
        if e["name"].startswith("side:opt:e"):
            r["n_ir"] = int(kv(e["name"])["post_opt"])
        if e["name"].startswith("side:exits_planned"):
            d = kv(e["name"])
            r.update(guards=int(d["guards"]), snapshots=int(d["snapshots"]))
        if e["name"].startswith(("tree:lower:e", "side:lower:e")):
            r["clif"] = int(kv(e["name"])["clif"])
    # consecutive-event stages
    stages = defaultdict(int)
    passes = defaultdict(int)
    stack = {}
    prev = first
    for e in evs[1:]:
        stages[f"{base(prev['name'])} -> {base(e['name'])}"] += e["ic"] - prev["ic"]
        n = e["name"]
        if n.startswith("pass:"):
            _, pname, rest = n.split(":", 2)
            if rest.startswith("b"):
                stack[pname] = e["ic"]
            else:
                passes[pname] += e["ic"] - stack.pop(pname)
        prev = e
    def stage_sum(*keys):
        return sum(v for k, v in stages.items() if any(k.startswith(x) for x in keys))
    r["define"] = stage_sum("lw:define:b")
    r["clif_build"] = stage_sum("lw:declared -> lw:clif_built")
    r["module"] = stage_sum("lw:start", "lw:builder_ready", "lw:module_new", "lw:define:e")
    r["exits"] = stage_sum("tree:opt_done", "side:opt_done")
    # Optimizer: from the first member's start to its end, per member.
    opt = 0
    b = None
    for e in evs:
        if e["name"].startswith(("tree:member_opt:b", "side:opt:b")):
            b = e["ic"]
        elif e["name"].startswith(("tree:member_opt:e", "side:opt:e")) and b is not None:
            opt += e["ic"] - b
            b = None
    r["opt"] = opt
    r["stages"] = dict(stages)
    r["passes"] = dict(passes)
    return r


def latency(path):
    """Submission to first native entry, per compile, with VM-thread instructions."""
    evs = parse(path)
    subs = [(i, e) for i, e in enumerate(evs) if e["name"].startswith("submit ")]
    print(f"{'id':>3} {'kind':<5}{'tier':<10}{'await':<6}{'queue us':>9}{'compile us':>11}{'wake us':>8}{'install us':>11}"
          f"{'to entry us':>12}{'VM instr sub->done':>20}{'VM instr sub->entry':>21}{'compile instr':>14}")
    for i, e in subs:
        d = kv(e["name"])
        cid = d["id"]
        vm = e["tid"]
        cb = next((x for x in evs[i:] if x["name"] == f"compile_begin id={cid}"), None)
        ce = next((x for x in evs[i:] if x["name"] == "compile_end" and cb and x["tid"] == cb["tid"] and x["t"] >= cb["t"]), None)
        done_b = next((x for x in evs[i:] if x["name"] == "complete:b" and x["tid"] == vm and ce and x["t"] >= ce["t"]), None)
        done_e = next((x for x in evs[i:] if x["name"] == "complete:e" and x["tid"] == vm and done_b and x["t"] >= done_b["t"]), None)
        entry = next((x for x in evs[i:] if x["name"] == "first_entry" and done_e and x["t"] >= done_e["t"]), None)
        if not (cb and ce and done_b and done_e):
            continue
        sync = cb["tid"] == vm
        compile_ic = ce["ic"] - cb["ic"]
        # VM-thread instructions from submission to the install call, minus the compile when it ran inline.
        vm_done = done_b["ic"] - e["ic"] - (compile_ic if sync else 0)
        vm_entry = (entry["ic"] - e["ic"] - (compile_ic if sync else 0) - (done_e["ic"] - done_b["ic"]) * 0) if entry else None
        print(f"{cid:>3} {'side' if d['side']=='true' else 'tree':<5}{d['tier']:<10}{d['awaited']:<6}"
              f"{(cb['t']-e['t'])*1e6:>9.0f}{(ce['t']-cb['t'])*1e6:>11.0f}{(done_b['t']-ce['t'])*1e6:>8.0f}"
              f"{(done_e['t']-done_b['t'])*1e6:>11.0f}"
              f"{((entry['t']-e['t'])*1e6 if entry else float('nan')):>12.0f}{vm_done:>20}"
              f"{(vm_entry if vm_entry is not None else -1):>21}{compile_ic:>14}")


def main():
    args = sys.argv[1:]
    mode = "table"
    if args and args[0] in ("--fit", "--latency"):
        mode = args.pop(0)[2:]
    rows = []
    for p in args:
        evs = parse(p)
        for c in compiles(evs):
            s = summarize(c)
            s["log"] = p
            rows.append(s)
    if mode == "latency":
        latency(args[0])
        return
    if mode == "fit":
        keys = ("log", "id", "side", "n_pre", "n_ir", "clif", "guards", "snapshots", "total", "opt", "exits", "clif_build", "define", "module")
        print(",".join(keys))
        for r in rows:
            print(",".join(str(r.get(k, "")) for k in keys))
        return
    for r in rows:
        print(f"\n### compile {r['id']} ({'side' if r['side'] else 'tree'}) tid {r['tid']}: {r['total']} instr, "
              f"{r['t_us']:.0f} us, n_pre={r.get('n_pre')} n_ir={r.get('n_ir')} clif={r.get('clif')}")
        for k, v in sorted(r["stages"].items(), key=lambda kv: -kv[1])[:14]:
            print(f"  {k:<48}{v:>10}{100*v/r['total']:>6.1f}%")


if __name__ == "__main__":
    main()
