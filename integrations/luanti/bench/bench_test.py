import importlib.util
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bench  # noqa: E402

spec = importlib.util.spec_from_file_location("luanti_driver", HERE.parent / "luanti.py")
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class BenchTest(unittest.TestCase):
    def test_area(self):
        self.assertEqual(bench.parse_area("64,-64,63"), (64, -64, 63))
        with self.assertRaises(ValueError):
            bench.parse_area("0,1,2")
        with self.assertRaises(ValueError):
            bench.parse_area("x")

    def test_perf_stat(self):
        text = "93341304453,,instructions:u,4000,100.00,,\n5800.5,msec,task-clock,5800,100.00,,\n<not counted>,,foo,0,0,,\n"
        self.assertEqual(bench.parse_perf_stat(text),
                         {"instructions": 93341304453, "cpu_ms": 5800.5})

    def test_probe_compare(self):
        a = "x [oxbench-probe] env=main jit=true jit.version=LuaJIT 2.1 ffi_global=false require_ffi=nil ull=true/true\n"
        b = a.replace("LuaJIT 2.1", "Oximond").replace("env=main", "env=mapgen")
        bench.compare_probes({"luajit": bench.parse_probe(a), "oximond": bench.parse_probe(b)})
        c = a.replace("ull=true/true", "ull=false/nil")
        with self.assertRaises(RuntimeError):
            bench.compare_probes({"luajit": bench.parse_probe(a), "oximond": bench.parse_probe(c)})
        with self.assertRaises(RuntimeError):
            bench.compare_probes({"luajit": bench.parse_probe(a), "oximond": set()})

    def test_abort_grouping(self):
        line = ("jit-trace: recording of the loop at pc Some(3) aborted at pc 9 (Some(Ok(GetField))) "
                "@/x/games/mineclonia/mcl_util/shape.lua:107: UnsupportedOperand")
        side = line.replace("jit-trace: ", "jit-trace: side ").replace("aborted", "restarted")
        text = "\n".join([line, line, side, "jit-trace: compiled trace tier=X body=3 in 1ms"])
        rows, compiled, aborts = bench.group_aborts(text)
        self.assertEqual(rows[0], (("GetField", "UnsupportedOperand", "mcl_util/shape.lua:107"), 3))
        self.assertEqual((compiled, aborts), (1, 3))

    def test_categories(self):
        report = ("  50.00%  Emerge-0  luanti  [.] oxi_rawseti\n"
                  "  20.00%  Emerge-0  [JIT] tid 5  [.] 0x1234\n"
                  "  10.00%  Emerge-0  luanti  [.] oximond::vm::run_fast\n"
                  "   5.00%  luanti  libc.so.6  [.] memcpy\n")
        out = bench.categorize(report)
        self.assertIn("C API boundary", out)
        self.assertIn("JIT machine code", out)
        self.assertRegex(out, r"Emerge-0 thread \(80.0%")

    def test_lprof_last_dump_wins(self):
        log = ("[lprof main] total 10\n[lprof main] 6 a.lua:1\n"
               "[lprof main] total 20\n[lprof main] 12 b.lua:2\n[lprof mapgen] total 4\n[lprof mapgen] 3 c.lua:3\n")
        blocks = bench.parse_lprof(log)
        self.assertEqual(blocks["main"], (20, [(12, "b.lua:2")]))
        self.assertIn("c.lua:3", bench.format_lprof(log))

    def test_attribution_thread_rows(self):
        old, _, _ = bench.parse_attribution(
            "oximond-attribution\t1\t1000\nthread\tEmerge-0\t5\t2\t1\n")
        self.assertEqual(old["Emerge-0"]["host"], 0)
        new, _, _ = bench.parse_attribution(
            "oximond-attribution\t1\t1000\nthread\tEmerge-0\t5\t2\t1\t7\n")
        self.assertEqual(new["Emerge-0"]["host"], 7)
        self.assertEqual(new["Emerge-0"]["interp"], 5)

    def test_result_and_ratio(self):
        a = bench.result_line("luajit", "g", "w", 0, 1.0, {"instructions": 100, "cpu_ms": 10.0},
                              {"emerge_us": "5", "blocks": "2"}, [])
        b = bench.result_line("oximond", "g", "w", 0, 1.0, {"instructions": 400, "cpu_ms": 20.0},
                              {"emerge_us": "10", "blocks": "2"}, ["ERROR[x]"])
        self.assertEqual(set(a), {"variant", "game", "workload", "rc", "wall_s", "instructions",
                                  "cpu_ms", "emerge_us", "blocks"})
        self.assertIn("errors", b)
        self.assertIn("4.00x", bench.ratio_table([a, b]))

    def test_cli(self):
        self.assertTrue((bench.MOD / "init.lua").is_file())
        self.assertTrue((bench.MOD / "mod.conf").is_file())
        import subprocess
        out = subprocess.run([sys.executable, str(HERE.parent / "luanti.py"), "bench", "--help"],
                             capture_output=True, text=True)
        self.assertEqual(out.returncode, 0)
        for flag in ("--variant", "--workload", "--reps", "--area", "--cpus",
                     "--jit-trace", "--profile", "--lua-profile"):
            self.assertIn(flag, out.stdout)


if __name__ == "__main__":
    unittest.main()
