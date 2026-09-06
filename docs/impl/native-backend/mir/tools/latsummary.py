#!/usr/bin/env python3
"""Median submission-to-first-entry numbers over repeated background runs.

  latsummary.py LOG [LOG...]

For every compile id and kind in the logs: the medians (and min to max) of the
queue wait, the compile wall time, the wake-up delay after compile end, the
install time, the time to the first native entry, and the VM thread's
instructions from submission to install. A compile that never completed within
a run (the process ended first) is counted in the `done` column.
"""
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from events import parse, kv


def rows_of(path):
    evs = parse(path)
    out = []
    for i, e in enumerate(evs):
        if not e["name"].startswith("submit "):
            continue
        d = kv(e["name"])
        cid, vm = d["id"], e["tid"]
        cb = next((x for x in evs[i:] if x["name"] == f"compile_begin id={cid}"), None)
        ce = next((x for x in evs[i:] if cb and x["name"] == "compile_end" and x["tid"] == cb["tid"] and x["t"] >= cb["t"]), None)
        db = next((x for x in evs[i:] if ce and x["name"] == "complete:b" and x["tid"] == vm and x["t"] >= ce["t"]), None)
        de = next((x for x in evs[i:] if db and x["name"] == "complete:e" and x["tid"] == vm and x["t"] >= db["t"]), None)
        en = next((x for x in evs[i:] if de and x["name"] == "first_entry" and x["t"] >= de["t"]), None)
        sync = bool(cb) and cb["tid"] == vm
        r = dict(id=int(cid), kind="side" if d["side"] == "true" else "tree", tier=d["tier"], awaited=d["awaited"],
                 sync=sync, done=bool(de))
        if cb:
            r["queue_us"] = (cb["t"] - e["t"]) * 1e6
        if cb and ce:
            r["compile_us"] = (ce["t"] - cb["t"]) * 1e6
            r["compile_cpu_us"] = ce["cpu"] - cb["cpu"] if ce["cpu"] else None
            r["compile_ivcs"] = ce["ivcs"] - cb["ivcs"] if ce["cpu"] else None
            r["compile_vcs"] = ce["vcs"] - cb["vcs"] if ce["cpu"] else None
            r["compile_ic"] = ce["ic"] - cb["ic"]
        if ce and db:
            r["wake_us"] = (db["t"] - ce["t"]) * 1e6
        if db and de:
            r["install_us"] = (de["t"] - db["t"]) * 1e6
            r["install_ic"] = de["ic"] - db["ic"]
        if en:
            r["entry_us"] = (en["t"] - e["t"]) * 1e6
        if db:
            r["vm_ic_to_done"] = db["ic"] - e["ic"] - (r.get("compile_ic", 0) if sync else 0)
            r["vm_native_to_done"] = db["nat"] - e["nat"]
            r["vm_interp_to_done"] = r["vm_ic_to_done"] - r["vm_native_to_done"]
        if en:
            r["vm_ic_to_entry"] = en["ic"] - e["ic"] - (r.get("compile_ic", 0) if sync else 0)
            r["vm_interp_to_entry"] = r["vm_ic_to_entry"] - (en["nat"] - e["nat"])
        out.append(r)
    return out


def fmt(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    return f"{statistics.median(vals):.0f} ({min(vals):.0f}-{max(vals):.0f})"


def main():
    groups = defaultdict(list)
    for p in sys.argv[1:]:
        for r in rows_of(p):
            groups[(r["id"], r["kind"], r["tier"], r["awaited"], r["sync"])].append(r)
    cols = [("queue_us", "queue us"), ("compile_us", "compile us"), ("compile_cpu_us", "worker cpu us"),
            ("compile_ivcs", "invol. switches"), ("compile_vcs", "vol. switches"), ("wake_us", "wake us"),
            ("install_us", "install us"), ("entry_us", "to entry us"), ("vm_ic_to_done", "VM instr to install"),
            ("vm_native_to_done", "of which native"), ("vm_interp_to_done", "of which interp+runtime"),
            ("vm_interp_to_entry", "interp+runtime to entry"),
            ("compile_ic", "compile instr")]
    print("id kind tier awaited sync runs done | " + " | ".join(c[1] for c in cols))
    for key in sorted(groups):
        rs = groups[key]
        print(f"{key[0]} {key[1]} {key[2]} {key[3]} {'sync' if key[4] else 'bg'} {len(rs)} {sum(r['done'] for r in rs)} | "
              + " | ".join(fmt([r.get(c[0]) for r in rs]) for c in cols))


if __name__ == "__main__":
    main()
