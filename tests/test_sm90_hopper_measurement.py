"""A concurrent GPU job must invalidate timing, never become SKIP or PASS."""
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("hopper_bench", ROOT / "benchmarks/bench_sm90_hopper.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class Measurement(unittest.TestCase):
    def test_specialization_denominator_negative(self):
        spec = importlib.util.spec_from_file_location("hopper_cases", ROOT / "tests/run_sm90_hopper_cases.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.validate_cases(module.CASES)
        with self.assertRaisesRegex(ValueError, "omits"):
            module.validate_cases([row for row in module.CASES if row[0] != "fp32-no-initial"])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            module.validate_cases(module.CASES + [module.CASES[0]])

    def test_process_parser_fails_closed(self):
        self.assertEqual(bench.process_ids(""), set())
        self.assertEqual(bench.process_ids("10\n20\n"), {10, 20})
        for text in ("N/A", "unknown", "12\nERROR"):
            with self.assertRaises(RuntimeError):
                bench.process_ids(text)

    def test_foreign_job_and_busy_idle_are_red(self):
        telemetry = "GPU-test, 0, 1590, 70, 0\n"
        watch = bench.DeviceWatch(0)
        with patch.object(bench.subprocess, "check_output", side_effect=["", telemetry]):
            watch.sample(idle=True)
        with patch.object(bench.subprocess, "check_output", side_effect=[str(os.getpid()), telemetry]):
            watch.sample()
        for pids, report, idle in (("99999999", telemetry, False),
                                   ("", telemetry.replace(", 0,", ", 25,", 1), True)):
            with patch.object(bench.subprocess, "check_output", side_effect=[pids, report]):
                with self.assertRaises(RuntimeError):
                    watch.sample(idle=idle)
                self.assertEqual(watch.records[-1]["pids"], sorted(bench.process_ids(pids)))


if __name__ == "__main__":
    unittest.main()
