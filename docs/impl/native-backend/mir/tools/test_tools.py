"""Tests for the latency study tools and a check of latency.md against its
acceptance criteria.

  python3 -m unittest docs/impl/native-backend/mir/tools/test_tools.py
"""
import csv
import io
import os
import re
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import classify  # noqa: E402
import events  # noqa: E402
import fit  # noqa: E402,F401  (import runs nothing: main is guarded below)

DOC = os.path.join(HERE, "..", "latency.md")


def line(name, tid, ic, t_ns, extra=""):
    return (f"jit-lat: ev {name} tid={tid} ic={ic}{extra} "
            f"t=Instant {{ tv_sec: 10, tv_nsec: {t_ns} }}\n")


FIXTURE = "".join([
    line("submit id=0 side=false tier=Optimized awaited=true", 1, 100, 1_000_000),
    line("compile_begin id=0", 2, 0, 1_100_000),
    line("tree:start", 2, 50, 1_100_100),
    line("tree:verified", 2, 150, 1_100_200),
    line("tree:member_opt:b n=10", 2, 160, 1_100_300),
    line("pass:copy elimination:b", 2, 200, 1_100_400),
    line("pass:copy elimination:e changed=true", 2, 700, 1_100_500),
    line("tree:member_opt:e", 2, 900, 1_100_600),
    line("tree:ir members=1 pre_opt=10 post_opt=8 guards=2 snapshots=2", 2, 910, 1_100_700),
    line("tree:opt_done", 2, 920, 1_100_800),
    line("tree:exits_planned", 2, 1000, 1_100_900),
    line("lw:start", 2, 1010, 1_101_000),
    line("lw:define:b", 2, 1500, 1_101_100),
    line("lw:define:e", 2, 9500, 1_101_200),
    line("tree:lower:e clif=40", 2, 9600, 1_101_300),
    line("compile_end", 2, 9700, 1_101_400),
    line("complete:b", 1, 200, 1_200_000),
    line("complete:e", 1, 700, 1_210_000),
    line("first_entry", 1, 800, 1_220_000),
])


class EventsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile("w", suffix=".log", delete=False)
        self.tmp.write(FIXTURE)
        self.tmp.close()

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_compile_summary(self):
        evs = events.parse(self.tmp.name)
        cs = events.compiles(evs)
        self.assertEqual(len(cs), 1)
        s = events.summarize(cs[0])
        self.assertEqual(s["total"], 9700)
        self.assertEqual(s["n_pre"], 10)
        self.assertEqual(s["n_ir"], 8)
        self.assertEqual(s["clif"], 40)
        self.assertEqual(s["define"], 8000)
        self.assertEqual(s["passes"]["copy elimination"], 500)
        self.assertEqual(s["opt"], 740)

    def test_latency_columns(self):
        out = io.StringIO()
        old = sys.stdout
        sys.stdout = out
        try:
            events.latency(self.tmp.name)
        finally:
            sys.stdout = old
        row = out.getvalue().splitlines()[1].split()
        # queue 100 us, compile 1.4 us, 98.6 us from compile end to install,
        # 10 us install, first entry 220 us after submission.
        self.assertEqual(row[0], "0")
        self.assertEqual(row[4], "100")
        self.assertEqual(row[5], "1")
        self.assertEqual(row[7], "10")
        self.assertEqual(row[8], "220")
        self.assertEqual(row[9], "100")


class ClassifyTest(unittest.TestCase):
    def test_generic_argument_is_not_the_crate(self):
        frames = [("drop_glue<alloc::vec::Vec<cranelift_codegen::ir::BlockArg>>", "library/core/src/ptr/mod.rs:844"),
                  ("eliminate_copies", "crates/oximond/src/jit_ir/trace/optimize.rs:971")]
        self.assertEqual(classify.phase_of(frames), "ox.opt.copies")

    def test_regalloc2_by_path(self):
        frames = [("run", "/x/regalloc2-0.15.2/src/ion/process.rs:10")]
        self.assertEqual(classify.phase_of(frames), "cl.regalloc2")

    def test_allocator_leaf_goes_to_caller(self):
        frames = ["malloc", "alloc::alloc::alloc", "oximond::jit_ir::trace::optimize::fold_constants"]
        self.assertEqual(classify.phase_of(frames), "ox.opt.fold")
        self.assertTrue(classify.is_alloc(frames))

    def test_compile_region_skips_recorder_rules(self):
        frames = ["oximond::jit_ir::trace::ir::TraceIr::clone", "oximond::jit::trace::compile::compile_with_plan"]
        self.assertNotEqual(classify.phase_of(frames), "ox.recording")

    def test_plan_in_recording_is_plan_tree(self):
        frames = ["Snapshot::clone", "oximond::jit_ir::trace::tree::plan_tree_with", "TracePolicy>::record"]
        self.assertEqual(classify.phase_of(frames), "ox.plan_tree")


class FitTest(unittest.TestCase):
    def test_ols_exact_line(self):
        a, b, r2 = fit.ols([1, 2, 3, 4], [12, 22, 32, 42])
        self.assertAlmostEqual(a, 2)
        self.assertAlmostEqual(b, 10)
        self.assertAlmostEqual(r2, 1)


class DocTest(unittest.TestCase):
    """latency.md carries every acceptance criterion."""

    @classmethod
    def setUpClass(cls):
        cls.text = open(DOC, encoding="utf-8").read()

    def has(self, pattern):
        self.assertRegex(self.text, pattern)

    def test_commit_and_machine_stated(self):
        self.has(r"b33edd2e")
        self.has(r"load average")

    def test_per_phase_breakdown_by_thread_for_each_benchmark(self):
        for bench in ("algorithm/nbody", "algorithm/mandelbrot", "algorithm/spectral_norm",
                      "opt/arith_mixed", "table/hash_read_indexed"):
            self.assertIn(bench, self.text)
        self.has(r"(?i)compile thread")
        self.has(r"(?i)VM thread")
        self.has(r"(?i)share of (the )?run")

    def test_fixed_versus_per_instruction(self):
        self.has(r"(?i)fixed cost")
        self.has(r"(?i)per trace IR instruction")

    def test_malloc_by_caller_and_per_instruction(self):
        self.has(r"(?i)malloc")
        self.has(r"(?i)allocations per trace IR instruction")
        self.has(r"snapshot\.rs")

    def test_ranked_pass_list_has_verbs(self):
        self.has(r"(?i)remove")
        self.has(r"(?i)fuse")
        self.has(r"(?i)keep")

    def test_layout_section_has_sources(self):
        self.has(r"(?m)^## IR data layout for compile speed")
        for needle in ("lj_ir.h", "lj_asm.c", "maglev-ir.h", "AirGenerate.cpp", "v8.dev/blog", "webkit.org/blog"):
            self.assertIn(needle, self.text)

    def test_latency_section(self):
        self.has(r"(?i)first native entry")
        self.has(r"(?i)interpreter instructions")

    def test_regeneration_commands_exist(self):
        self.has(r"(?m)^## Regenerating the numbers")
        for script in re.findall(r"docs/impl/native-backend/mir/tools/([\w.]+)", self.text):
            self.assertTrue(os.path.exists(os.path.join(HERE, script)), script)

    def test_house_style(self):
        self.assertNotIn("—", self.text, "em dash")
        self.assertNotIn("–", self.text, "en dash")
        self.assertIn("Inference.", self.text)
        for h in re.findall(r"(?m)^#{2,4} (.+)$", self.text):
            words = h.split()
            # Sentence case: no word after the first starts uppercase unless it is
            # an acronym, a code span or a proper name from this short list.
            allowed = {"IR", "MIR", "CLIF", "ISLE", "JIT", "LuaJIT", "V8", "Maglev", "Liftoff", "Sparkplug",
                       "Winch", "JavaScriptCore", "Air", "Makarov's", "Cranelift", "Oximond", "VM", "ISA", "RA",
                       "Oximond's", "SSA", "ABI", "I"}
            for w in words[1:]:
                bare = w.strip("`():,")
                if bare and bare[0].isupper() and bare not in allowed and not bare.isupper():
                    self.fail(f"heading not sentence case: {h!r} ({bare})")


if __name__ == "__main__":
    unittest.main()
