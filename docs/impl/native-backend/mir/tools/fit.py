#!/usr/bin/env python3
"""Fixed and per-instruction compile cost, fitted over many compiles.

  events.py --fit fit/*.log > fit.csv ; fit.py fit.csv

Least squares of instructions = a + b * n over trees and sides separately, for
n = trace IR instructions after the optimiser and n = CLIF instructions, for
the whole compile and for each stage. Reports a, b and r^2.
"""
import csv
import sys


def ols(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx if sxx else 0.0
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    return a, b, 1 - ss_res / ss_tot if ss_tot else 0.0


def main():
    rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["n_ir"] and r["clif"]]
    for kind in ("False", "True"):
        sel = [r for r in rows if r["side"] == kind]
        if len(sel) < 3:
            continue
        print(f"\n## {'sides' if kind == 'True' else 'trees'}: {len(sel)} compiles, "
              f"n_ir {min(int(r['n_ir']) for r in sel)}..{max(int(r['n_ir']) for r in sel)}, "
              f"clif {min(int(r['clif']) for r in sel)}..{max(int(r['clif']) for r in sel)}")
        for r in sel:
            r["non_cranelift"] = str(int(r["total"]) - int(r["define"]))
        print(f"{'stage':<26}{'x':<8}{'fixed a':>12}{'per-x b':>12}{'r2':>7}")
        for col in ("total", "non_cranelift", "opt", "exits", "clif_build", "define", "module"):
            for x in ("n_pre", "n_ir", "clif"):
                a, b, r2 = ols([int(r[x]) for r in sel], [int(r[col]) for r in sel])
                print(f"{col:<26}{x:<8}{a:>12.0f}{b:>12.0f}{r2:>7.2f}")
        print("n_pre is trace IR instructions before the optimizer, n_ir after it, clif the CLIF instructions.")
    # medians
    import statistics
    for kind in ("False", "True"):
        sel = [r for r in rows if r["side"] == kind]
        if sel:
            print(f"\nmedian total per compile ({'sides' if kind == 'True' else 'trees'}): "
                  f"{statistics.median(int(r['total']) for r in sel):.0f} instr, "
                  f"median n_ir {statistics.median(int(r['n_ir']) for r in sel)}, "
                  f"median instr per IR {statistics.median(int(r['total'])/int(r['n_ir']) for r in sel):.0f}, "
                  f"min total {min(int(r['total']) for r in sel)}")


if __name__ == "__main__":
    main()
