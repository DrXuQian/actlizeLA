"""Selection, coverage, source negatives and executable handoff contracts."""
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


class MixedTailContract(unittest.TestCase):
    def test_explicit_dispatch_and_default_unchanged(self):
        self.assertEqual(api.RESIDUAL_VARIANTS["residual-mixed-tail"], "mixed-tail")
        self.assertEqual(api.RESIDUAL_CONTROLS["residual-mixed-tail"], "residual-full-chunk")
        self.assertIsNone(api.PROFILE_VARIANTS["residual-mixed-tail"])
        tensors = [Mock() for _ in range(5)]
        for tensor in tensors:
            tensor.contiguous.return_value = tensor
        backend = SimpleNamespace(residual=Mock(return_value=(1, 2)), residual_mixed_tail=Mock(return_value=(3, 4)))
        with patch.object(api, "_backend", return_value=backend):
            self.assertEqual(api.gdn_chunk_residual(*tensors, delivery="mixed-tail"), (3, 4))
            self.assertEqual(api.gdn_chunk_residual(*tensors), (1, 2))
        backend.residual_mixed_tail.assert_called_once_with(*tensors, None, True)
        backend.residual.assert_called_once_with(*tensors, None, True)
        del backend.residual_mixed_tail
        with patch.object(api, "_backend", return_value=backend), self.assertRaises(AttributeError):
            api.gdn_chunk_residual(*tensors, delivery="mixed-tail")

    def test_source_and_ten_constructive_negatives(self):
        run = subprocess.run([sys.executable, str(ROOT / "dev/ppu/check_mixed_tail.py"), "--self-test"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertEqual(run.stdout.count("EXPECTED-RED/PASS"), 10)

    def test_mixed_tail_alone_gets_all_36_cases_not_only_full_no_initial(self):
        sys.path.insert(0, str(ROOT / "tests"))
        import test_ppu_residual_backend as gate
        with patch.object(sys, "argv", ["gate", "--extension", __file__, "--deliveries", "mixed-tail"]), \
             patch.dict(os.environ), patch.object(gate, "admit") as admit, \
             patch.object(gate.torch.cuda, "set_device"), patch.object(gate.torch, "set_num_threads"), \
             patch.object(gate.torch.cuda, "get_device_properties", return_value=SimpleNamespace(name="PPU")), \
             redirect_stdout(io.StringIO()) as log:
            gate.main()
        self.assertEqual(admit.call_count, 36)
        self.assertIn("cases=36", log.getvalue())
        admit.assert_any_call((1, 2048, 16, 32), -.1, True, deliveries=["mixed-tail"])
        admit.assert_any_call((2, 65, 1, 2), -1., False, deliveries=["mixed-tail"])

    def test_runner_success_and_fail_closed_paths(self):
        for failure in ("none", "build", "admission", "edges", "capture"):
            with self.subTest(failure=failure):
                self.runner_case(failure)

    def test_every_tail_length_and_long_case_reaches_existing_numeric_gate(self):
        sys.path.insert(0, str(ROOT / "tests"))
        import test_ppu_mixed_tail_backend as gate
        with patch.object(sys, "argv", ["gate", "--extension", __file__]), \
             patch.dict(os.environ), patch.object(gate, "admit") as admit, \
             patch.object(gate.torch.cuda, "set_device"), patch.object(gate.torch, "set_num_threads"), \
             patch.object(gate.torch.cuda, "get_device_properties", return_value=SimpleNamespace(name="PPU")), \
             redirect_stdout(io.StringIO()) as log:
            gate.main()
        self.assertEqual(admit.call_count, 261)
        self.assertIn("cases=261", log.getvalue())
        for length in range(65, 128):
            for decay in (-.1, -1.):
                for initial in (False, True):
                    admit.assert_any_call((1, length, 1, 2), decay, initial, deliveries=["full-chunk", "mixed-tail"])

    def runner_case(self, failure):
        directory = Path("/workspace") / ("actlizeLA-mixed-runner-" + uuid4().hex)
        directory.mkdir()
        driver = f"#!{sys.executable}\n" + r'''
import json, os, pathlib, sys
kind, args = pathlib.Path(sys.argv[0]).name, sys.argv[1:]
out = pathlib.Path(os.environ['OUT'])
row = dict(kind=kind, args=args, out=str(out), sdk=os.environ.get('PPU_SDK'),
           perf=os.environ.get('PERF'), samples=os.environ.get('SAMPLES'),
           visible=os.environ.get('CUDA_VISIBLE_DEVICES'))
with open(os.environ['CAPTURE_LOG'], 'a') as f:
    f.write(json.dumps(row) + '\n')
build = kind == 'bash' and args[0].endswith('/run_ppu_wy_fla_box.sh')
admission = kind == 'python' and args[0].endswith('/test_ppu_residual_backend.py')
edges = kind == 'python' and args[0].endswith('/test_ppu_mixed_tail_backend.py')
capture = kind == 'bash' and args[0].endswith('/run_ppu_gdn_fla_acu_box.sh')
if any(os.environ['FAIL_AT'] == name and active for name, active in
       [('build', build), ('admission', admission), ('edges', edges), ('capture', capture)]):
    sys.exit(43)
if build:
    (out / 'build').mkdir(parents=True)
    (out / 'build/_gdn_wy_ppu.fake.so').write_text('not a device binary')
    path = out / 'build/l037_wy_mixed_tail'
    path.write_text(pathlib.Path(sys.argv[0]).read_text())
    path.chmod(0o755)
'''
        for name in ("bash", "cmake", "python"):
            path = directory / name
            path.write_text(driver)
            path.chmod(0o755)
        out, log = directory / "run", directory / "calls.jsonl"
        env = os.environ | dict(PATH=str(directory) + os.pathsep + os.environ["PATH"],
            CAPTURE_LOG=str(log), OUT=str(out), DEVICE="3", FAIL_AT=failure, SAMPLES="99",
            PPU_SDK="/explicit/sdk", PPU_SDK_ROOT="/wrong/sdk", ACU="/bin/true")
        run = subprocess.run(["/bin/bash", str(ROOT / "tools/run_ppu10_mixed_tail_box.sh")],
                             env=env, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0 if failure == "none" else 43, run.stdout + run.stderr)
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        select = lambda suffix: [r for r in calls if r["args"] and r["args"][0].endswith(suffix)]
        build, = select("/run_ppu_wy_fla_box.sh")
        self.assertEqual(build["perf"], "0")
        self.assertIsNone(build["samples"])
        self.assertEqual(build["sdk"], "/explicit/sdk")
        captures = select("/run_ppu_gdn_fla_acu_box.sh")
        self.assertEqual(len(captures), 4 if failure == "none" else int(failure == "capture"))
        self.assertEqual(len(select("/pack_ppu10_mixed_tail.sh")), int(failure == "none"))
        pairs = [(s, g) for s in ("2049", "2111") for g in ("-1.0", "-0.1")]
        for row, (sequence, gate) in zip(captures, pairs):
            self.assertEqual(row["args"][1:], ["--wy-run", str(out), "--wy-control", "residual-full-chunk",
                                              "--wy-delivery", "residual-mixed-tail", "--sequence", sequence, "--gate", gate])
            self.assertEqual(row["out"], str(out / ("acu-mixed-s" + sequence + "-g" + gate)))
            self.assertEqual(row["visible"], "3")
            self.assertEqual(row["sdk"], "/explicit/sdk")
        if failure == "none":
            for suffix in ("/test_ppu_residual_backend.py", "/admit_ppu_residual_fla.py"):
                row, = select(suffix)
                index = row["args"].index("--deliveries")
                self.assertEqual(row["args"][index + 1:index + 3], ["full-chunk", "mixed-tail"])
            self.assertEqual(len(select("/test_ppu_mixed_tail_backend.py")), 1)
            admission, = select("/admit_ppu_residual_fla.py")
            i = admission["args"].index("--sequences")
            self.assertEqual(admission["args"][i + 1:i + 3], ["2049", "2111"])
            self.assertIn("four captures complete", run.stdout)
        else:
            self.assertNotIn("four captures complete", run.stdout)


if __name__ == "__main__":
    unittest.main()
