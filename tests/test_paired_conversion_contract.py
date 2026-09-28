"""Explicit dispatch, complete denominators and executable handoff negatives."""
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


class PairedConversionContract(unittest.TestCase):
    def test_candidate_is_explicit_default_and_math_unchanged(self):
        self.assertEqual(api.RESIDUAL_VARIANTS["residual-paired-conversion"], "paired-conversion")
        self.assertEqual(api.RESIDUAL_CONTROLS["residual-paired-conversion"], "residual-full-chunk")
        self.assertIsNone(api.PROFILE_VARIANTS["residual-paired-conversion"])
        self.assertEqual(api.math_contract("residual-paired-conversion"), api.math_contract("residual-full-chunk"))
        tensors = [Mock() for _ in range(5)]
        for tensor in tensors:
            tensor.contiguous.return_value = tensor
        backend = SimpleNamespace(residual=Mock(return_value=(1, 2)), residual_paired_conversion=Mock(return_value=(3, 4)))
        with patch.object(api, "_backend", return_value=backend):
            self.assertEqual(api.gdn_chunk_residual(*tensors, delivery="paired-conversion"), (3, 4))
            self.assertEqual(api.gdn_chunk_residual(*tensors), (1, 2))
        backend.residual_paired_conversion.assert_called_once_with(*tensors, None, True)
        backend.residual.assert_called_once_with(*tensors, None, True)
        del backend.residual_paired_conversion
        with patch.object(api, "_backend", return_value=backend), self.assertRaises(AttributeError):
            api.gdn_chunk_residual(*tensors, delivery="paired-conversion")

    def test_source_eleven_constructive_negatives(self):
        run = subprocess.run([sys.executable, str(ROOT / "dev/ppu/check_paired_conversion.py"), "--self-test"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertEqual(run.stdout.count("EXPECTED-RED/PASS"), 11)

    def test_resource_comparison_excludes_next_elf_inventory_not_real_resources(self):
        sys.path.insert(0, str(ROOT / "dev/ppu"))
        from check_wy_binary import compare_resources
        record = lambda i: f"Func 1 kernel{i} RESOURCE INFO:\nSTACK SIZE:0\nvreg_number:84\n"
        before = "".join(record(i) + f"\nELF FILE {i + 1}\nFunc 1: kernel{i + 1}\n" for i in range(44))
        after = record(44) + record(45) + "\nELF FILE 1\nFunc 1: kernel0\n" + before
        compare_resources(before, after)
        with self.assertRaisesRegex(AssertionError, "resources changed"):
            compare_resources(before, after.replace("kernel42 RESOURCE INFO:\nSTACK SIZE:0", "kernel42 RESOURCE INFO:\nSTACK SIZE:32"))

    def run_enumeration(self, gate, argv):
        with patch.object(sys, "argv", argv), patch.dict(os.environ), \
             patch.object(gate, "admit") as admit, patch.object(gate.torch.cuda, "set_device"), \
             patch.object(gate.torch.cuda, "get_device_properties", return_value=SimpleNamespace(name="PPU")), \
             patch.object(gate.torch, "set_num_threads"), redirect_stdout(io.StringIO()) as log:
            gate.main()
        return admit, log.getvalue()

    def test_candidate_alone_gets_retained_36_case_gate(self):
        sys.path.insert(0, str(ROOT / "tests"))
        import test_ppu_residual_backend as gate
        admit, log = self.run_enumeration(gate, ["gate", "--extension", __file__, "--deliveries", "paired-conversion"])
        self.assertEqual(admit.call_count, 36)
        self.assertIn("cases=36", log)
        admit.assert_any_call((1, 2048, 16, 32), -.1, True, deliveries=["paired-conversion"])

    def test_all64_extents_both_gates_and_initial_states_reach_oracle(self):
        sys.path.insert(0, str(ROOT / "tests"))
        import test_ppu_paired_conversion_backend as gate
        admit, log = self.run_enumeration(gate, ["gate", "--extension", __file__])
        self.assertEqual(admit.call_count, 256)
        self.assertIn("cases=256", log)
        for length in range(1, 65):
            for decay in (-.1, -1.):
                for initial in (False, True):
                    admit.assert_any_call((1, length, 1, 2), decay, initial,
                                          deliveries=["full-chunk", "paired-conversion"])

    def test_runner_success_and_fail_closed_paths(self):
        for failure in ("none", "build", "native", "admission", "edges", "capture"):
            with self.subTest(failure=failure):
                self.runner_case(failure)

    def runner_case(self, failure):
        directory = Path("/workspace") / ("actlizeLA-paired-runner-" + uuid4().hex)
        directory.mkdir()
        driver = f"#!{sys.executable}\n" + r'''
import json, os, pathlib, sys
kind, args = pathlib.Path(sys.argv[0]).name, sys.argv[1:]
out = pathlib.Path(os.environ['OUT'])
row = dict(kind=kind, args=args, out=str(out), sdk=os.environ.get('PPU_SDK'),
           perf=os.environ.get('PERF'), samples=os.environ.get('SAMPLES'),
           visible=os.environ.get('CUDA_VISIBLE_DEVICES'), target=os.environ.get('GDN_QSA_TARGET'))
with open(os.environ['CAPTURE_LOG'], 'a') as f:
    f.write(json.dumps(row) + '\n')
suffixes = dict(build='/run_ppu_wy_fla_box.sh', native='/check_paired_conversion.py',
                admission='/test_ppu_residual_backend.py', edges='/test_ppu_paired_conversion_backend.py',
                capture='/run_ppu_gdn_fla_acu_box.sh')
if args and os.environ['FAIL_AT'] in suffixes and args[0].endswith(suffixes[os.environ['FAIL_AT']]):
    sys.exit(43)
if args and args[0].endswith(suffixes['build']):
    (out / 'build').mkdir(parents=True)
    (out / 'build/_gdn_wy_ppu.fake.so').write_text('not a device binary')
    path = out / 'build/l039_wy_paired_conversion'
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
            PPU_SDK="/explicit/sdk", PPU_SDK_ROOT="/wrong/sdk", GDN_QSA_TARGET="sm90", ACU="/bin/true")
        run = subprocess.run(["/bin/bash", str(ROOT / "tools/run_ppu10_paired_conversion_box.sh")],
                             env=env, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0 if failure == "none" else 43, run.stdout + run.stderr)
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        select = lambda suffix: [r for r in calls if r["args"] and r["args"][0].endswith(suffix)]
        build, = select("/run_ppu_wy_fla_box.sh")
        self.assertEqual(build["perf"], "0")
        self.assertIsNone(build["samples"])
        self.assertEqual(build["sdk"], "/explicit/sdk")
        self.assertEqual(build["target"], "ppu10")
        captures = select("/run_ppu_gdn_fla_acu_box.sh")
        self.assertEqual(len(captures), 2 if failure == "none" else int(failure == "capture"))
        self.assertEqual(len(select("/pack_ppu10_paired_conversion.sh")), int(failure == "none"))
        for row, decay in zip(captures, ("-1.0", "-0.1")):
            self.assertEqual(row["args"][1:], ["--wy-run", str(out), "--wy-control", "residual-full-chunk",
                "--wy-delivery", "residual-paired-conversion", "--gate", decay])
            self.assertEqual(row["out"], str(out / ("acu-paired-g" + decay)))
            self.assertEqual(row["visible"], "3")
        if failure == "none":
            native, = select("/check_paired_conversion.py")
            self.assertIn("--host", native["args"])
            self.assertIn("--binding", native["args"])
            for suffix in ("/test_ppu_residual_backend.py", "/admit_ppu_residual_fla.py"):
                row, = select(suffix)
                i = row["args"].index("--deliveries")
                self.assertEqual(row["args"][i + 1:i + 3], ["full-chunk", "paired-conversion"])
            self.assertIn("two captures complete", run.stdout)
        else:
            self.assertNotIn("two captures complete", run.stdout)


if __name__ == "__main__":
    unittest.main()
