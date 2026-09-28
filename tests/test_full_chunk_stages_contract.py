"""CPU-only selection, negative-control and executable runner contracts."""
from contextlib import redirect_stdout
import io
import json
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
STAGES = ("solve", "output", "both")
DELIVERIES = ["full-chunk", *("full-chunk-" + stage for stage in STAGES)]


class FullChunkStagesContract(unittest.TestCase):
    def test_inventory_and_actual_python_dispatch(self):
        tensors = [Mock() for _ in range(5)]
        for tensor in tensors:
            tensor.contiguous.return_value = tensor
        default = Mock(return_value=("baseline", "state"))
        backend = SimpleNamespace(residual=default)
        for stage in STAGES:
            variant, delivery = "residual-full-chunk-" + stage, "full-chunk-" + stage
            self.assertEqual(api.RESIDUAL_VARIANTS[variant], delivery)
            self.assertEqual(api.RESIDUAL_CONTROLS[variant], "residual-full-chunk")
            self.assertIsNone(api.PROFILE_VARIANTS[variant])
            name = "residual_full_chunk_" + stage
            self.assertEqual(api.RESIDUAL_ENTRYPOINTS[delivery], name)
            call = Mock(return_value=(stage, "state"))
            setattr(backend, name, call)
            with patch.object(api, "_backend", return_value=backend):
                self.assertEqual(api.gdn_chunk_residual(*tensors, delivery=delivery), (stage, "state"))
            call.assert_called_once_with(*tensors, None, True)
            delattr(backend, name)
            with patch.object(api, "_backend", return_value=backend), self.assertRaises(AttributeError):
                api.gdn_chunk_residual(*tensors, delivery=delivery)
        default.assert_not_called()
        with patch.object(api, "_backend", return_value=backend):
            self.assertEqual(api.gdn_chunk_residual(*tensors), ("baseline", "state"))
        default.assert_called_once()

    def test_real_source_negatives(self):
        run = subprocess.run([sys.executable, str(ROOT / "dev/ppu/check_full_chunk_stages.py"), "--self-test"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertEqual(run.stdout.count("EXPECTED-RED/PASS"), 12)
        for label in ("output-lost-causal", "wrong-shared-state-launch", "ignored-choice", "tail-admitted"):
            self.assertIn(label + " EXPECTED-RED/PASS", run.stdout)

    def test_each_new_arm_gets_the_complete_device_enumeration(self):
        sys.path.insert(0, str(ROOT / "tests"))
        import test_ppu_residual_backend as gate
        for deliveries in ([name] for name in DELIVERIES[1:]):
            with patch.object(sys, "argv", ["gate", "--extension", __file__, "--deliveries", *deliveries]), \
                 patch.dict(os.environ), patch.object(gate, "admit") as admit, \
                 patch.object(gate.torch.cuda, "set_device"), patch.object(gate.torch, "set_num_threads"), \
                 patch.object(gate.torch.cuda, "get_device_properties", return_value=SimpleNamespace(name="PPU")), \
                 redirect_stdout(io.StringIO()) as log:
                gate.main()
            self.assertEqual(admit.call_count, 36)
            self.assertIn("cases=36", log.getvalue())
            for shape in ((2, 128, 1, 2), (1, 2048, 16, 32)):
                for decay in (-.1, -1.):
                    admit.assert_any_call(shape, decay, True, deliveries=deliveries)

    def test_runner_builds_once_admits_four_and_captures_six_pairs(self):
        self.run_runner_case("none", 0)

    def test_runner_build_failure_is_not_a_pass(self):
        self.run_runner_case("build", 43)

    def test_runner_numeric_failure_prevents_every_capture(self):
        self.run_runner_case("admission", 43)

    def test_runner_capture_failure_stops_remaining_candidates(self):
        self.run_runner_case("capture", 43)

    def run_runner_case(self, fail_at, expected_rc):
        directory = Path("/workspace") / ("actlizeLA-full-stages-runner-" + uuid4().hex)
        directory.mkdir()
        driver = f"#!{sys.executable}\n" + r'''
import json, os, pathlib, sys
kind = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
out = pathlib.Path(os.environ['OUT'])
row = dict(kind=kind, args=args, out=str(out), device=os.environ.get('DEVICE'),
           sdk=os.environ.get('PPU_SDK'), root=os.environ.get('PPU_SDK_ROOT'),
           perf=os.environ.get('PERF'), visible=os.environ.get('CUDA_VISIBLE_DEVICES'),
           samples=os.environ.get('SAMPLES'))
with open(os.environ['CAPTURE_LOG'], 'a') as f:
    f.write(json.dumps(row) + '\n')
build = kind == 'bash' and args[0].endswith('/run_ppu_wy_fla_box.sh')
admission = kind == 'python' and args[0].endswith('/test_ppu_residual_backend.py')
capture = kind == 'bash' and args[0].endswith('/run_ppu_gdn_fla_acu_box.sh')
if (os.environ['FAIL_AT'] == 'build' and build or
    os.environ['FAIL_AT'] == 'admission' and admission or
    os.environ['FAIL_AT'] == 'capture' and capture):
    sys.exit(43)
if build:
    (out / 'build').mkdir(parents=True)
    (out / 'build/_gdn_wy_ppu.fake.so').write_text('fixture, not a real binary')
    for name in ('l032_wy_solve_static', 'l033_wy_gate_cache', 'l034_wy_full_chunk', 'l035_wy_full_chunk_stages'):
        path = out / 'build' / name
        path.write_text(pathlib.Path(sys.argv[0]).read_text())
        path.chmod(0o755)
'''
        for name in ("bash", "cmake", "python"):
            path = directory / name
            path.write_text(driver)
            path.chmod(0o755)
        log, out = directory / "calls.jsonl", directory / "run"
        env = os.environ | dict(PATH=str(directory) + os.pathsep + os.environ["PATH"],
            CAPTURE_LOG=str(log), OUT=str(out), DEVICE="3", FAIL_AT=fail_at,
            PPU_SDK="/explicit/sdk", PPU_SDK_ROOT="/must-not-win", ACU="/bin/true", SAMPLES="99")
        run = subprocess.run(["/bin/bash", str(ROOT / "tools/run_ppu10_full_chunk_stages_box.sh")],
                             env=env, capture_output=True, text=True)
        self.assertEqual(run.returncode, expected_rc, run.stdout + run.stderr)
        rows = [json.loads(line) for line in log.read_text().splitlines()]
        builds = [r for r in rows if r["kind"] == "bash" and r["args"][0].endswith("run_ppu_wy_fla_box.sh")]
        self.assertEqual(len(builds), 1)
        self.assertEqual(builds[0]["perf"], "0")
        self.assertIsNone(builds[0]["samples"])
        self.assertEqual(builds[0]["sdk"], "/explicit/sdk")
        captures = [r for r in rows if r["kind"] == "bash" and r["args"][0].endswith("run_ppu_gdn_fla_acu_box.sh")]
        expected_captures = 6 if fail_at == "none" else int(fail_at == "capture")
        self.assertEqual(len(captures), expected_captures)
        for r, (stage, gate) in zip(captures, ((s, g) for s in STAGES for g in ("-1.0", "-0.1"))):
            self.assertEqual(r["args"][1:], ["--wy-run", str(out), "--wy-control", "residual-full-chunk",
                "--wy-delivery", "residual-full-chunk-" + stage, "--gate", gate])
            self.assertEqual(r["out"], str(out / f"acu-{stage}-g{gate}"))
            self.assertEqual(r["sdk"], "/explicit/sdk")
            self.assertEqual(r["visible"], "3")
        if fail_at == "none":
            admissions = [r for r in rows if r["kind"] == "python" and
                          r["args"][0].endswith(("test_ppu_residual_backend.py", "admit_ppu_residual_fla.py"))]
            self.assertEqual(len(admissions), 2)
            for r in admissions:
                index = r["args"].index("--deliveries") + 1
                self.assertEqual(r["args"][index:index + 4], DELIVERIES)
                self.assertEqual(r["visible"], "3")
            self.assertIn("six captures complete", run.stdout)
        else:
            self.assertNotIn("six captures complete", run.stdout)


if __name__ == "__main__":
    unittest.main()
