import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trace_census as tc  # noqa: E402


def member(ops, guards=(), pre=0):
    return {"pre": pre, "body": sum(ops.values()) - pre, "values": 5, "loop_params": 1,
            "snapshots": len(guards), "peak_gpr": 3, "peak_fpr": 2, "ops": ops,
            "guards": list(guards)}


def record(root, members, detached=False, clif=30):
    return {"root": root, "tier": "Baseline", "detached": detached, "clif_total": clif,
            "clif_ops": {"Arithmetic.Add": [2, 2], "Arithmetic.Mul": [1, 1], "Move": [1, 0]},
            "members": members}


G1 = [10, 4, 3, 2, 1, 0, 0, 4, 2, 1]
G2 = [20, 8, 6, 4, 2, 0, 0, 8, 5, 0]


class CensusTests(unittest.TestCase):
    def test_parse_lines_skips_other_stderr(self):
        text = "noise\njit-census: " + json.dumps({"a": 1}) + "\njit-census: {broken\n"
        self.assertEqual(tc.parse_lines(text), [{"a": 1}])

    def test_last_compile_per_root_wins_and_detached_is_ignored(self):
        old = record("t1", [member({"Move": 1})])
        new = record("t1", [member({"Move": 2})])
        det = record("t1", [member({"Move": 9})], detached=True)
        trees = tc.final_trees([old, new, det])
        self.assertEqual(len(trees), 1)
        self.assertEqual(trees[0]["members"][0]["ops"], {"Move": 2})

    def test_summarize_merges_sub_ops_and_counts_sides(self):
        root = member({"Arithmetic.Add": 2, "Arithmetic.Mul": 1, "GuardType": 1}, [G1, G2])
        side = member({"Move": 1})
        s = tc.summarize([record("t1", [root, side])])
        self.assertEqual(s["trees"], 1)
        self.assertEqual(s["sides"], 1)
        self.assertEqual(s["ir"], 5)
        self.assertEqual(s["ops"]["Arithmetic"], 3)
        self.assertEqual(s["clif_ops"]["Arithmetic"], [3, 3])
        self.assertEqual(tc.guard_column(s, "window"), [10, 20])
        self.assertEqual(tc.stats([10, 20]), "15.0 / 20 / 20")

    def test_percentile_is_nearest_rank(self):
        self.assertEqual(tc.percentile(list(range(1, 101)), 0.95), 95)
        self.assertEqual(tc.percentile([7], 0.95), 7)

    def test_coverage_counts_variants_to_reach_each_share(self):
        rows, at95, at99 = tc.coverage({"a": 90, "b": 6, "c": 3, "d": 1})
        self.assertEqual((at95, at99), (2, 3))
        self.assertEqual(rows[0][0], "a")

    def test_report_runs_on_a_collection(self):
        with tempfile.TemporaryDirectory() as d:
            rec = record("t1", [member({"Arithmetic.Add": 3, "Move": 1}, [G1])])
            (Path(d) / "algorithm__nbody.jsonl").write_text(json.dumps(rec) + "\n")
            import contextlib
            import io
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                tc.report(d, ["algorithm/nbody"])
            text = out.getvalue()
            self.assertIn("| algorithm/nbody | 1 | 1 | 0 | 4 |", text)
            self.assertIn("Variants for 95%", text)


if __name__ == "__main__":
    unittest.main()
