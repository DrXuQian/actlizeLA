"""CPU-only binding/runner admission tests; no PPU device certificate."""
from pathlib import Path
import json
import os
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace
from uuid import uuid4
from actlize_la import gdn_residual_interface as api

ROOT=Path(__file__).resolve().parents[1]


class GateCacheContract(unittest.TestCase):
    def test_two_gate_script_builds_once_and_stops_when_admission_fails(self):
        directory = Path("/workspace") / f"actlizeLA-ppu10-runner-test-{uuid4().hex}"
        directory.mkdir()
        fake_bash = directory / "bash"
        fake_bash.write_text(f"#!{sys.executable}\n" +
            "import json,os,sys\n"
            "with open(os.environ['CAPTURE_LOG'],'a') as log:\n"
            " log.write(json.dumps(dict(argv=sys.argv[1:],out=os.environ.get('OUT'),"
            "gate=os.environ.get('GATE'),candidate=os.environ.get('CANDIDATE'),"
            "device=os.environ.get('DEVICE'),acu=os.environ.get('ACU')))+'\\n')\n"
            "sys.exit(int(os.environ['CHILD_RC']))\n")
        fake_bash.chmod(0o755)
        for rc in (0, 43):
            log = directory / f"commands-{rc}.jsonl"
            out = directory / f"result-{rc}"
            env = os.environ | dict(PATH=str(directory) + os.pathsep + os.environ["PATH"],
                CAPTURE_LOG=str(log), CHILD_RC=str(rc), OUT=str(out), DEVICE="3",
                ACU="/sim/eec/shared/junfu.qx/asight/bin/acu")
            result = subprocess.run(["/bin/bash", str(ROOT / "tools/run_ppu10_solve_gate_cache_box.sh")],
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, rc, result.stdout + result.stderr)
            commands = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(len(commands), 1 if rc else 2)
            first = commands[0]
            self.assertEqual(first["gate"], "-1.0")
            self.assertEqual(first["candidate"], "residual-gate-cache-solve-static")
            self.assertEqual(first["device"], "3")
            if rc == 0:
                second = commands[1]
                self.assertEqual(second["out"], str(out / "acu-weak"))
                self.assertEqual(second["device"], "3")
                self.assertEqual(second["argv"][1:], ["--wy-run", str(out),
                    "--wy-control", "residual-gate-cache", "--wy-delivery",
                    "residual-gate-cache-solve-static", "--gate", "-0.1"])
            else:
                self.assertNotIn("both captures complete", result.stdout)

    def test_composed_candidate_uses_gate_cache_control_not_hv(self):
        name = "residual-gate-cache-solve-static"
        self.assertEqual(api.RESIDUAL_VARIANTS[name], "gate-cache-solve-static")
        self.assertEqual(api.RESIDUAL_CONTROLS[name], "residual-gate-cache")
        self.assertEqual(api.RESIDUAL_ENTRYPOINTS["gate-cache-solve-static"], "residual_gate_cache_solve_static")
        self.assertIsNone(api.PROFILE_VARIANTS[name])

    def test_composed_dispatch_and_state_are_not_gate_cache_fallback(self):
        tensors = [Mock() for _ in range(5)]
        state = Mock()
        for tensor in (*tensors, state):
            tensor.contiguous.return_value = tensor
        control = Mock()
        subject = Mock(return_value=("O", "H"))
        with patch.object(api, "_backend", return_value=SimpleNamespace(
            residual_gate_cache=control, residual_gate_cache_solve_static=subject
        )):
            self.assertEqual(api.gdn_chunk_residual(*tensors, initial_state=state,
                output_final_state=False, delivery="gate-cache-solve-static"), ("O", None))
        subject.assert_called_once_with(*tensors, state, False)
        control.assert_not_called()
        with patch.object(api, "_backend", return_value=SimpleNamespace(residual_gate_cache=control)):
            with self.assertRaises(AttributeError):
                api.gdn_chunk_residual(*tensors, delivery="gate-cache-solve-static")
        control.assert_not_called()

    def test_composition_source_and_negatives_use_actual_launcher(self):
        run = subprocess.run([sys.executable, str(ROOT / "dev/ppu/check_gate_cache_solve.py"), "--self-test"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        for label in ("old-solve", "wrong-configure", "inverse-alias", "wrong-state", "wrong-output", "wrong-binding"):
            self.assertIn(label + " EXPECTED-RED/PASS", run.stdout)

    def test_composition_runner_retains_both_math_proofs(self):
        text = (ROOT / "tools/run_ppu_residual_delivery_acu_box.sh").read_text()
        self.assertIn('residual-gate-cache|residual-gate-cache-solve-static)', text)
        self.assertIn('residual-gate-cache || "$CANDIDATE" == residual-gate-cache-solve-static', text)
        self.assertIn('residual-solve-static || "$CANDIDATE" == residual-gate-cache-solve-static', text)
        self.assertIn('check_gate_cache_solve.py" --self-test', text)
        self.assertIn('libgdn_wy_ppu.so', text)
        paired = (ROOT / "tools/run_ppu10_solve_gate_cache_box.sh").read_text()
        self.assertEqual(paired.count('bash "$ROOT/tools/run_ppu_residual_delivery_acu_box.sh"'), 1)
        self.assertIn('GATE=-1.0', paired)
        self.assertIn('--wy-run "$RUN"', paired)
        self.assertIn('--wy-control residual-gate-cache --wy-delivery residual-gate-cache-solve-static --gate -0.1', paired)

    def test_exact_variant_is_explicit_and_control_is_same_geometry(self):
        self.assertEqual(api.RESIDUAL_VARIANTS["residual-gate-cache"],"gate-cache")
        self.assertEqual(api.RESIDUAL_CONTROLS["residual-gate-cache"],"residual-warps8-hvlayout")
        self.assertEqual(api.RESIDUAL_ENTRYPOINTS["gate-cache"],"residual_gate_cache")
        self.assertIsNone(api.PROFILE_VARIANTS["residual-gate-cache"])

    def test_default_and_candidate_do_not_alias(self):
        tensors=[Mock() for _ in range(5)]
        for t in tensors: t.contiguous.return_value=t
        control=Mock(return_value=("C","H"))
        candidate=Mock(return_value=("O","H"))
        with patch.object(api,"_backend",return_value=SimpleNamespace(residual=control,residual_gate_cache=candidate)):
            self.assertEqual(api.gdn_chunk_residual(*tensors),("C","H"))
            self.assertEqual(api.gdn_chunk_residual(*tensors,delivery="gate-cache"),("O","H"))
        control.assert_called_once()
        candidate.assert_called_once()

    def test_missing_candidate_fails_instead_of_falling_back(self):
        tensors=[Mock() for _ in range(5)]
        with patch.object(api,"_backend",return_value=SimpleNamespace(residual=Mock())):
            with self.assertRaises(AttributeError):
                api.gdn_chunk_residual(*tensors,delivery="gate-cache")

    def test_runner_keeps_site_acu_and_raw_bit_admission(self):
        text=(ROOT/"tools/run_ppu_residual_delivery_acu_box.sh").read_text()
        self.assertIn("/sim/eec/shared/junfu.qx/asight/bin/acu",text)
        self.assertIn('CANDIDATE" == residual-gate-cache',text)
        self.assertIn("l033_wy_gate_cache",text)
        self.assertIn("check_gate_cache.py",text)
        self.assertIn("test_ppu_residual_backend.py",text)


if __name__=="__main__": unittest.main()
