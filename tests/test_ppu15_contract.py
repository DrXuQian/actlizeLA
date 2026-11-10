"""Run real dispatch/build/loader boundaries, with fail-closed negative controls."""
import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from actlize_la import gdn_forward, get_device_profile
from actlize_la.backends import loading
from actlize_la.backends.registry import require_implementation


class Ppu15Contract(unittest.TestCase):
    def tearDown(self):
        loading.clear_backend_cache()

    def test_unified_auto_uses_native_forward_and_preserves_state(self):
        inputs = tuple(object() for _ in range(5))
        initial = object()
        result = (object(), object())
        call = Mock(return_value=result)
        with patch("actlize_la.gdn_interface.import_module", return_value=SimpleNamespace(gdn_chunk_ppu15=call)) as load:
            got = gdn_forward(*inputs, initial_state=initial, output_final_state=False, backend="ppu15")
        self.assertIs(got,result)
        load.assert_called_once_with(".gdn_ppu15_interface","actlize_la")
        call.assert_called_once_with(*inputs,initial_state=initial,output_final_state=False,delivery="native")

    def test_native_choice_does_not_change_old_backend(self):
        old = require_implementation("residual","ppu10")
        new = require_implementation("residual","ppu15")
        self.assertEqual(old["module"],"gdn_residual_interface")
        self.assertEqual(old["default_delivery"],"full-chunk")
        self.assertEqual(new["module"],"gdn_ppu15_interface")
        self.assertEqual(old["numerical_contract"],new["numerical_contract"])

    def test_perfmodel_does_not_query_hardware(self):
        with patch("actlize_la.device.device_profile",side_effect=AssertionError("device query")):
            result = get_device_profile(mode="perfmodel",backend="ppu15",sm_count=20)
        self.assertEqual((result.source,result.sm_count,result.capability),("configured",20,None))

    def test_loader_cannot_accept_old_ppu_or_sm90_image(self):
        for arch in (100,170,900,None):
            module = SimpleNamespace(target_arch=lambda: arch) if arch else object()
            with patch.object(loading,"_explicit_path",return_value="/fixture.so"), patch.object(loading,"_load",return_value=module):
                with self.assertRaisesRegex(RuntimeError,"not a native PPU1.5"):
                    loading.load_ppu15()
        with patch.object(loading,"_explicit_path",return_value="/fixture.so"), patch.object(
            loading,"_load",return_value=SimpleNamespace(target_arch=lambda:150)) as native:
            loading.load_ppu15()
            native.assert_called_once_with("_gdn_residual_ppu15","/fixture.so")

    def test_missing_ppu15_selection_cannot_borrow_ppu10_path(self):
        with patch.dict(os.environ,{"GDN_QSA_WY_EXTENSION":"/old.so"},clear=True):
            with self.assertRaisesRegex(RuntimeError,"GDN_QSA_PPU15_EXTENSION"):
                loading.load_ppu15()

    def test_legacy_delivery_cannot_be_ignored(self):
        from actlize_la.gdn_ppu15_interface import gdn_chunk_ppu15
        with self.assertRaisesRegex(ValueError,"PPU1.0 delivery"):
            gdn_chunk_ppu15(*(object() for _ in range(5)),delivery="full-chunk")

    def test_cmake_target_conflicts_fail_before_sdk_discovery(self):
        work=Path("/workspace")/f"actlize-la-ppu15-build-negatives-{uuid4().hex}"
        for i, flag in enumerate(("-DGDN_QSA_ENABLE_PPU=ON", "-DCUTLASS_PPU_ARCHS=ppu0010", "-DACOMPUTE_VERSION=10700")):
            result=subprocess.run(["cmake","-S",str(ROOT),"-B",str(work/str(i)),
                "-DGDN_QSA_TARGET=ppu15","-DGDN_QSA_BUILD_PPU_TESTS=OFF",flag],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertRegex(" ".join(result.stderr.split()),"legacy PPU1.0 alias|conflicts|different backend")
            self.assertFalse((work/str(i)/"actlize").exists())

    def test_source_graph_is_explicit_and_no_legacy_primitive_import(self):
        source=(ROOT/"cmake/backends/ppu15.cmake").read_text()
        for unit in ("prepare","state","output","launch"):
            self.assertIn(f"csrc/backends/ppu15/{unit}.cu",source)
        self.assertNotIn("csrc/gdn_chunk/",source)
        for path in (ROOT/"csrc/backends/ppu15").glob("*"):
            text=path.read_text()
            self.assertNotIn("PPU0010_",text)
            self.assertNotIn("mma_sm90",text)
        self.assertIn("__HGGC_ARCH__ != 150",(ROOT/"csrc/backends/ppu15/primitives.cuh").read_text())

    def test_native_checker_rejects_missing_math_or_missing_body(self):
        spec=importlib.util.spec_from_file_location("ppu15_native_check",ROOT/"dev/ppu15/check_native.py")
        checker=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)
        names=("prefix_kernel","solve_kernel","state_kernel_full","state_kernel_tail","output_kernel")
        resources="PPU 1.5\n"+"\n".join(f"Func {i}: {name}\nvreg_number:32\nSTACK SIZE:0" for i,name in enumerate(names))
        opcodes="v.mm.f32.bf16.m16n16k16 v.mm.f32.tf32.m16n16k8 vmem.aiu.ld.tsm tsm.ld.swzl"
        isa="\n".join(f"Disassembly of section .text.kernel.{name}:\n{opcodes}" for name in names)
        self.assertEqual(checker.inspect_dump(isa,resources)["status"],"PASS")
        with self.assertRaisesRegex(ValueError,"missing native opcode"):
            checker.inspect_dump(isa.replace("v.mm.f32.tf32.m16n16k8","compatibility_arithmetic"),resources)
        with self.assertRaisesRegex(ValueError,"denominator"):
            checker.inspect_dump(isa,resources.replace("state_kernel_tail","missing_body"))
        with self.assertRaisesRegex(ValueError,"stack/spill"):
            checker.inspect_dump(isa,resources.replace("STACK SIZE:0","STACK SIZE:16",1))
        # Other kernels still contain all opcodes: cannot lend their evidence
        # to an accidentally empty state body.
        with self.assertRaisesRegex(ValueError,"missing native opcode"):
            checker.inspect_dump(isa.replace(f"state_kernel_full:\n{opcodes}","state_kernel_full:\ns.ret"),resources)

    def test_public_device_and_build_handoff_are_fail_closed(self):
        runner=(ROOT/"tools/run_ppu15_gdn_box.sh").read_text()
        self.assertLess(runner.index('"$OUT/build/test_ppu15_primitives" 2>&1'),runner.index('"$ROOT/tests/test_ppu15_device.py"'))
        self.assertIn("results.tar.gz",runner)
        self.assertNotIn("mktemp",runner)
        self.assertNotIn("/tmp/",runner)
        device=(ROOT/"tests/test_ppu15_device.py").read_text()
        self.assertIn("len(cases)!=47",device)
        self.assertIn("assert_pair(reference,expected)",device)
        self.assertLess(device.index('device_admission="PASS"'),device.index('result["performance"]=measure'))

    def test_correctness_grid_retains_old_cases_and_covers_model_heads(self):
        from test_ppu15_device import correctness_cases
        cases = correctness_cases()
        self.assertEqual(len(cases), 47)
        self.assertEqual(len(set(cases)), 47)
        for length in (1, 63, 64, 65, 127, 128, 129):
            for gate in (-.1, -1.):
                for initial in (False, True):
                    self.assertIn(((2, length, 1, 2), gate, initial, False), cases)
        for length in (2048, 8192):
            for hv in (32, 48):
                for gate in (-.1, -1.):
                    self.assertIn(((1, length, 16, hv), gate, True, False), cases)
        self.assertIn(((1, 129, 2, 4), -.01, True, True), cases)
        self.assertIn(((1, 2048, 16, 64), -.1, False, False), cases)

    def test_no_perf_flag_cannot_enter_measurement(self):
        import test_ppu15_device as device_test
        work = Path('/workspace') / f'actlize-la-ppu15-no-perf-host-{uuid4().hex}'
        work.mkdir()
        extension = work / 'host-only-never-loaded.so'
        extension.write_bytes(b'HOST_ONLY_COMMAND_CONTRACT_NOT_A_DEVICE_IMAGE')
        argv = ['test_ppu15_device.py', '--extension', str(extension), '--out', str(work), '--no-perf']
        props = SimpleNamespace(name='PPU-ZW890', multi_processor_count=1)
        with patch.object(sys, 'argv', argv), patch.dict(os.environ, {}, clear=False), patch.object(
                device_test.torch.cuda, 'set_device'), patch.object(
                device_test.torch.cuda, 'get_device_properties', return_value=props), patch.object(
                device_test, 'run_case', return_value={'host_only': True}) as case, patch.object(
                device_test, 'measure', side_effect=AssertionError('timing forbidden')) as measure:
            device_test.main()
        self.assertEqual(case.call_count, 47)
        measure.assert_not_called()
        result = json.loads((work/'device.json').read_text())
        self.assertEqual(result['performance_status'], 'NOT_RUN')
        self.assertEqual(result['performance'], [])
        # This proves command control flow only; mock results are never admission.
        self.assertTrue(all(row == {'host_only': True} for row in result['cases']))
        runner = (ROOT/'tools/run_ppu15_gdn_box.sh').read_text()
        self.assertIn('case "${NO_PERF:-0}" in', runner)
        self.assertIn('EXTRA_ARGS+=(--no-perf)', runner)
        self.assertIn('"${EXTRA_ARGS[@]}" 2>&1', runner)


if __name__ == "__main__":
    unittest.main()
