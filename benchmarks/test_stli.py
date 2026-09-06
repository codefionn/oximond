#!/usr/bin/env python3
"""Tests for stli.py. They run no binaries."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stli  # noqa: E402

STAT = """\
212532
6590974,,ls_bad_status2.stli_other:u,2130000,100.00,,
278213287,,instructions:u,2130000,100.00,,
76275313,,cycles:u,2130000,100.00,,
"""


class ParseStat(unittest.TestCase):
    def test_reads_every_event_and_skips_the_program_output(self):
        counts = stli.parse_stat(STAT)
        self.assertEqual(counts["ls_bad_status2.stli_other:u"], 6590974)
        self.assertEqual(counts["instructions:u"], 278213287)
        self.assertEqual(counts["cycles:u"], 76275313)

    def test_a_missing_event_is_an_error(self):
        with self.assertRaises(RuntimeError):
            stli.parse_stat("1,,cycles:u,1,100.00,,\n")

    def test_best_takes_the_minimum_of_each_event(self):
        runs = [
            {"ls_bad_status2.stli_other:u": 5, "cycles:u": 9, "instructions:u": 7},
            {"ls_bad_status2.stli_other:u": 3, "cycles:u": 11, "instructions:u": 7},
        ]
        self.assertEqual(
            stli.best(runs),
            {"ls_bad_status2.stli_other:u": 3, "cycles:u": 9, "instructions:u": 7},
        )

    def test_ratio_of_zero_interlocks_is_infinite(self):
        self.assertEqual(stli.ratio(10, 5), 2.0)
        self.assertEqual(stli.ratio(10, 0), float("inf"))


if __name__ == "__main__":
    unittest.main()
