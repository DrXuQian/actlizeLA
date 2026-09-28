"""CPU source, route and executable runner contracts; never a device PASS."""
import json
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4
from actlize_la import gdn_residual_interface as api

ROOT = Path(__file__).resolve().parents[1]


class FullChunkContract(unittest.TestCase):
    def test_candidate_is_explicit_and_has_the_measured_control(self):
        self.assertEqual(api.RESIDUAL_VARIANTS["residual-full-chunk"], "full-chunk")
        self.assertEqual(api.RESIDUAL_CONTROLS["residual-full-chunk"], "residual-gate-cache-solve-static")
        self.assertEqual(api.RESIDUAL_ENTRYPOINTS["full-chunk"], "residual_full_chunk")
        self.assertIsNone(api.PROFILE_VARIANTS["residual-full-chunk"])

    def test_default_does_not_change_and_missing_candidate_fails(self):
        tensors = [Mock() for _ in range(5)]
        for tensor in tensors:
            tensor.contiguous.return_value = tensor
        control, candidate = Mock(return_value=("C", "H")), Mock(return_value=("O", "H"))
        with patch.object(api, "_backend", return_value=SimpleNamespace(residual=control, residual_full_chunk=candidate)):
            self.assertEqual(api.gdn_chunk_residual(*tensors), ("C", "H"))
            self.assertEqual(api.gdn_chunk_residual(*tensors, delivery="full-chunk", output_final_state=False), ("O", None))
        candidate.assert_called_once_with(*tensors, None, False)
        control.assert_called_once_with(*tensors, None, True)
        with patch.object(api, "_backend", return_value=SimpleNamespace(residual=control)):
            with self.assertRaises(AttributeError):
                api.gdn_chunk_residual(*tensors, delivery="full-chunk")
        control.assert_called_once()

    def test_actual_source_negatives(self):
        run = subprocess.run([sys.executable, str(ROOT / "dev/ppu/check_full_chunk.py"), "--self-test"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertEqual(run.stdout.count("EXPECTED-RED/PASS"), 7)
        for label in ("tail-eligible", "missing-beta-row", "wrong-fallback", "wrong-launch"):
            self.assertIn(label + " EXPECTED-RED/PASS", run.stdout)

    def test_device_suite_denominator_and_nonzero_multichunk_cases(self):
        # Execute the actual suite enumeration with device/math work mocked,
        # so an inflated printed denominator cannot hide a missing case.
        sys.path.insert(0, str(ROOT / "tests"))
        import test_ppu_residual_backend as gate
        for deliveries, expected in (([], 30), (["full-chunk"], 36)):
            argv = ["gate", "--extension", __file__]
            if deliveries:
                argv += ["--deliveries", *deliveries]
            with patch.object(sys, "argv", argv), patch.dict(os.environ), \
                 patch.object(gate, "admit") as admit, \
                 patch.object(gate.torch.cuda, "set_device"), \
                 patch.object(gate.torch.cuda, "get_device_properties", return_value=SimpleNamespace(name="PPU")), \
                 patch.object(gate.torch, "set_num_threads"), redirect_stdout(io.StringIO()) as log:
                gate.main()
            self.assertEqual(admit.call_count, expected)
            self.assertIn(f"cases={expected}", log.getvalue())
            if deliveries:
                for shape in ((2, 128, 1, 2), (1, 2048, 16, 32)):
                    for decay in (-.1, -1.):
                        admit.assert_any_call(shape, decay, True, deliveries=deliveries)

    def test_runner_builds_once_profiles_both_gates_and_stops_on_failure(self):
        directory = Path("/workspace") / ("actlizeLA-full-chunk-runner-" + uuid4().hex)
        directory.mkdir()
        fake = directory / "bash"
        fake.write_text(f"#!{sys.executable}\nimport os,sys,json\n" +
            "with open(os.environ['CAPTURE_LOG'],'a') as f:\n"
            " f.write(json.dumps(dict(argv=sys.argv[1:],out=os.environ['OUT'],"
            "device=os.environ['DEVICE'],gate=os.environ.get('GATE'),"
            "candidate=os.environ.get('CANDIDATE')))+'\\n')\n"
            "sys.exit(int(os.environ['CHILD_RC']))\n")
        fake.chmod(0o755)
        for rc in (0, 43):
            log, out = directory / f"commands-{rc}.jsonl", directory / f"run-{rc}"
            env = os.environ | dict(PATH=str(directory) + os.pathsep + os.environ["PATH"],
                CAPTURE_LOG=str(log), OUT=str(out), DEVICE="3", CHILD_RC=str(rc))
            run = subprocess.run(["/bin/bash", str(ROOT / "tools/run_ppu10_full_chunk_box.sh")],
                                 env=env, capture_output=True, text=True)
            self.assertEqual(run.returncode, rc, run.stdout + run.stderr)
            commands = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(len(commands), 1 if rc else 2)
            self.assertEqual(commands[0]["candidate"], "residual-full-chunk")
            self.assertEqual(commands[0]["gate"], "-1.0")
            self.assertEqual(commands[0]["device"], "3")
            if rc == 0:
                self.assertEqual(commands[1]["out"], str(out / "acu-weak"))
                self.assertEqual(commands[1]["device"], "3")
                self.assertEqual(commands[1]["argv"][1:], ["--wy-run", str(out),
                    "--wy-control", "residual-gate-cache-solve-static", "--wy-delivery",
                    "residual-full-chunk", "--gate", "-0.1"])
            else:
                self.assertNotIn("captures complete", run.stdout)


if __name__ == "__main__":
    unittest.main()
