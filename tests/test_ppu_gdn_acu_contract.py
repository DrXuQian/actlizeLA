"""Host-only contract/negative tests; no device, ACU or installed FLA required."""
import copy
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
import profile_ppu_gdn_fla as profile

spec = importlib.util.spec_from_file_location("collect_acu", ROOT / "tools/collect_ppu_gdn_acu.py")
collect = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collect)


class ACUContract(unittest.TestCase):
    def test_capture_keeps_moved_backend_sources_and_catalog(self):
        directory = self.directory()
        status, _, bundle, _ = self.run_mock_capture(directory)
        self.assertEqual(status['status'], 'PASS', status['errors'])
        root = bundle / 'sources/ours'
        collect.validate_backend_source_snapshot(root)
        for name in ('csrc/backends/ppu_aiu/primitives.cuh',
                     'csrc/backends/cuda_sm80/primitives.cuh',
                     'actlize_la/backends/targets.json'):
            self.assertEqual((root / name).read_bytes(), (ROOT / name).read_bytes())
        # Actual omission of either a moved header or the JSON is red.
        real_is_file = Path.is_file
        for omitted in ('csrc/backends/ppu_aiu/primitives.cuh',
                        'actlize_la/backends/targets.json'):
            def missing(path):
                return path != root / omitted and real_is_file(path)
            with patch.object(Path, 'is_file', missing), self.assertRaisesRegex(
                    RuntimeError, 'backend source snapshot incomplete'):
                collect.validate_backend_source_snapshot(root)

    def test_residual_delivery_capture_requires_matching_control(self):
        for delivery in sorted(collect.RESIDUAL_DELIVERIES):
            with self.subTest(delivery=delivery):
                control = collect.RESIDUAL_CONTROLS[delivery]
                arms=collect.capture_arms(Path('/bundle'),'wy',delivery,control)
                self.assertEqual([a.delivery for a in arms],[control,delivery,delivery])
                for wrong in {None,'scalar','state-pipeline',delivery,'residual','residual-warps8'} - {control}:
                    with self.assertRaises(ValueError):
                        collect.capture_arms(Path('/bundle'),'wy',delivery,wrong)

    def test_residual_delivery_numeric_close_is_not_raw_bit_admission(self):
        baseline=dict(math_contract=collect.MATH_CONTRACT,delivery_mask=None,
                      scalar_raw_bit_equal=None,errors=[.008,.004],fingerprint='same')
        candidate=baseline | dict(residual_raw_bit_equal=True,residual_fingerprint='same')
        collect.validate_residual_delivery_admission(candidate,baseline,.02)
        for field,value in (('fingerprint','different'),('residual_raw_bit_equal',False),
                            ('residual_fingerprint','different'),('math_contract','old')):
            with self.subTest(field=field),self.assertRaises(ValueError):
                collect.validate_residual_delivery_admission(candidate|{field:value},baseline,.02)

    def test_complete_residual_deliveries_and_changed_output_negative(self):
        directory=self.directory()
        for delivery in sorted(collect.RESIDUAL_DELIVERIES):
            control = collect.RESIDUAL_CONTROLS[delivery]
            d=directory/delivery;d.mkdir()
            status,commands,bundle,_=self.run_mock_capture(d,control=control,subject_delivery=delivery)
            self.assertEqual(status['status'],'PASS',status['errors'])
            d=directory/(delivery+'-bad');d.mkdir()
            status,_,_,_=self.run_mock_capture(d,control=control,subject_delivery=delivery,
                                              plant='changed-control-output')
            self.assertEqual(status['status'],'INCOMPLETE')

    def test_residual_public_entrypoints_are_distinct_and_default_is_old(self):
        import torch
        from actlize_la import gdn_residual_interface as api
        from unittest.mock import Mock
        inputs=tuple(torch.zeros(1) for _ in range(5))
        backend=SimpleNamespace(**{name:Mock(return_value=('out','state')) for name in
            ('residual','residual_prefetch','residual_operands','residual_v16','residual_blayout',
             'residual_warps8','residual_warps8_blayout','residual_warps8_operands','residual_warps8_hlayout',
             'residual_warps8_hvlayout','residual_warps8_metadata','residual_solve_static')})
        with patch.object(api,'_backend',return_value=backend):
            for delivery,name in (('scalar','residual'),('prefetch','residual_prefetch'),
                                  ('operands','residual_operands'),('v16','residual_v16'),
                                  ('blayout','residual_blayout'),('warps8','residual_warps8'),
                                  ('warps8-blayout','residual_warps8_blayout'),
                                  ('warps8-operands','residual_warps8_operands'),
                                  ('warps8-hlayout','residual_warps8_hlayout'),
                                  ('warps8-hvlayout','residual_warps8_hvlayout'),
                                  ('warps8-metadata','residual_warps8_metadata'),
                                  ('solve-static','residual_solve_static')):
                self.assertEqual(api.gdn_chunk_residual(*inputs,delivery=delivery),('out','state'))
                getattr(backend,name).assert_called_once()
            with self.assertRaises(ValueError): api.gdn_chunk_residual(*inputs,delivery='unknown')

    def test_residual_default_accepts_legacy_backend_without_candidates(self):
        import torch
        from unittest.mock import Mock
        from actlize_la import gdn_residual_interface as api
        inputs = tuple(torch.zeros(1) for _ in range(5))
        backend = SimpleNamespace(residual=Mock(return_value=('out', 'state')))
        with patch.object(api, '_backend', return_value=backend):
            self.assertEqual(api.gdn_chunk_residual(*inputs), ('out', 'state'))
        backend.residual.assert_called_once()

    def test_residual_runner_rejects_bad_selector_or_missing_acu_before_build(self):
        directory = self.directory()
        runner = ROOT / 'tools/run_ppu_residual_delivery_acu_box.sh'
        for candidate in ('not-a-candidate', 'residual-v16', 'residual-blayout', 'residual-warps8',
                          'residual-warps8-blayout','residual-warps8-operands','residual-warps8-hlayout',
                          'residual-warps8-hvlayout','residual-warps8-metadata','residual-solve-static'):
            out = directory / candidate
            env = os.environ | {'CANDIDATE': candidate, 'OUT': str(out),
                                'ACU': str(directory / 'missing-site-acu')}
            run = subprocess.run(['bash', str(runner)], env=env, text=True,
                                 capture_output=True, timeout=10)
            self.assertNotEqual(run.returncode, 0)
            self.assertIn('FAIL:', run.stderr)
            self.assertFalse(out.exists(), 'failed admission still entered build')

    def test_blayout_cannot_silently_use_scalar_backend(self):
        import torch
        from unittest.mock import Mock
        from actlize_la import gdn_residual_interface as api
        inputs=tuple(torch.zeros(1) for _ in range(5))
        backend=SimpleNamespace(residual=Mock(return_value=('out','state')))
        with patch.object(api,'_backend',return_value=backend),self.assertRaises(AttributeError):
            api.gdn_chunk_residual(*inputs,delivery='blayout')
        backend.residual.assert_not_called()

    def test_static_solve_cannot_silently_fall_back_to_hv(self):
        import torch
        from unittest.mock import Mock
        from actlize_la import gdn_residual_interface as api
        inputs=tuple(torch.zeros(1) for _ in range(5))
        backend=SimpleNamespace(residual=Mock(),residual_warps8_hvlayout=Mock())
        with patch.object(api,'_backend',return_value=backend),self.assertRaises(AttributeError):
            api.gdn_chunk_residual(*inputs,delivery='solve-static')
        backend.residual.assert_not_called()
        backend.residual_warps8_hvlayout.assert_not_called()

    def test_static_solve_native_helper_and_capture_are_bound(self):
        from actlize_la import gdn_residual_interface as api
        self.assertEqual(api.RESIDUAL_CONTROLS['residual-solve-static'],'residual-warps8-hvlayout')
        run=subprocess.run([sys.executable,str(ROOT/'dev/ppu/check_solve_static.py'),'--self-test'],
                           capture_output=True,text=True)
        self.assertEqual(run.returncode,0,run.stdout+run.stderr)
        self.assertIn('wrong-binding EXPECTED-RED/PASS',run.stdout)
        runner=(ROOT/'tools/run_ppu_residual_delivery_acu_box.sh').read_text()
        self.assertIn('CANDIDATE" == residual-solve-static',runner)
        self.assertIn('--host "$RUN/build/l032_wy_solve_static"',runner)

    def test_warps8_cannot_silently_use_scalar_backend(self):
        import torch
        from unittest.mock import Mock
        from actlize_la import gdn_residual_interface as api
        inputs=tuple(torch.zeros(1) for _ in range(5))
        backend=SimpleNamespace(residual=Mock(return_value=('out','state')))
        with patch.object(api,'_backend',return_value=backend),self.assertRaises(AttributeError):
            api.gdn_chunk_residual(*inputs,delivery='warps8')
        backend.residual.assert_not_called()

    def test_warps8_blayout_cannot_silently_use_control_backend(self):
        import torch
        from unittest.mock import Mock
        from actlize_la import gdn_residual_interface as api
        inputs = tuple(torch.zeros(1) for _ in range(5))
        backend = SimpleNamespace(residual=Mock(), residual_warps8=Mock())
        with patch.object(api, '_backend', return_value=backend), self.assertRaises(AttributeError):
            api.gdn_chunk_residual(*inputs, delivery='warps8-blayout')
        backend.residual.assert_not_called()
        backend.residual_warps8.assert_not_called()

    def test_warps8_blayout_same_geometry_receipt_cannot_be_four_warps(self):
        control, _ = self.records()
        control.update(role='wy', implementation='wy', wy_delivery='residual-warps8',
                       math_contract=collect.MATH_CONTRACT)
        subject = control | dict(wy_delivery='residual-warps8-blayout')
        collect.validate_wy_control(control, subject)
        with self.assertRaises(ValueError):
            collect.validate_wy_control(control | dict(wy_delivery='residual'), subject)

    def test_warps8_blayout_profile_selects_exact_api_and_paired_capture(self):
        inputs = (1,2,3,4,5)
        with patch('actlize_la.gdn_chunk_residual', return_value='new-layout') as call:
            subject, _ = profile.subject_call('wy','wy',Path('/binding.so'),inputs,
                                               'residual-warps8-blayout')
            self.assertEqual(subject(), 'new-layout')
            call.assert_called_once_with(*inputs, output_final_state=True, delivery='warps8-blayout')
        directory = self.directory()
        status, commands, _, _ = self.run_mock_capture(
            directory, control='residual-warps8', subject_delivery='residual-warps8-blayout')
        self.assertEqual(status['status'], 'PASS', status['errors'])
        captures = [cmd for cmd in commands if '--phase' in cmd and cmd[cmd.index('--phase')+1] == 'subject']
        self.assertEqual([cmd[cmd.index('--wy-delivery')+1] for cmd in captures],
                         ['residual-warps8','residual-warps8-blayout','residual-warps8-blayout'])

    def test_warps8_operands_requires_its_backend_and_blayout_control(self):
        import torch
        from unittest.mock import Mock
        from actlize_la import gdn_residual_interface as api
        inputs = tuple(torch.zeros(1) for _ in range(5))
        backend = SimpleNamespace(residual=Mock(), residual_warps8_blayout=Mock())
        with patch.object(api, '_backend', return_value=backend), self.assertRaises(AttributeError):
            api.gdn_chunk_residual(*inputs, delivery='warps8-operands')
        backend.residual.assert_not_called()
        backend.residual_warps8_blayout.assert_not_called()
        control, _ = self.records()
        control.update(role='wy', implementation='wy', wy_delivery='residual-warps8-blayout',
                       math_contract=collect.MATH_CONTRACT)
        subject = control | dict(wy_delivery='residual-warps8-operands')
        collect.validate_wy_control(control, subject)
        for wrong in ('residual', 'residual-warps8', 'residual-warps8-operands'):
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                collect.validate_wy_control(control | dict(wy_delivery=wrong), subject)

    def test_warps8_operands_profile_and_capture_are_exact(self):
        inputs = (1,2,3,4,5)
        with patch('actlize_la.gdn_chunk_residual', return_value='new-schedule') as call:
            subject, _ = profile.subject_call('wy','wy',Path('/binding.so'),inputs,
                                               'residual-warps8-operands')
            self.assertEqual(subject(), 'new-schedule')
            call.assert_called_once_with(*inputs, output_final_state=True, delivery='warps8-operands')
        directory = self.directory()
        status, commands, _, _ = self.run_mock_capture(
            directory, control='residual-warps8-blayout', subject_delivery='residual-warps8-operands')
        self.assertEqual(status['status'], 'PASS', status['errors'])
        captures = [cmd for cmd in commands if '--phase' in cmd and cmd[cmd.index('--phase')+1] == 'subject']
        self.assertEqual([cmd[cmd.index('--wy-delivery')+1] for cmd in captures],
                         ['residual-warps8-blayout','residual-warps8-operands','residual-warps8-operands'])

    def test_hlayout_requires_exact_backend_and_blayout_control(self):
        import torch
        from unittest.mock import Mock
        from actlize_la import gdn_residual_interface as api
        inputs = tuple(torch.zeros(1) for _ in range(5))
        backend = SimpleNamespace(residual=Mock(), residual_warps8_blayout=Mock(),
                                  residual_warps8_operands=Mock())
        with patch.object(api, '_backend', return_value=backend), self.assertRaises(AttributeError):
            api.gdn_chunk_residual(*inputs, delivery='warps8-hlayout')
        for call in vars(backend).values(): call.assert_not_called()
        control, _ = self.records()
        control.update(role='wy', implementation='wy', wy_delivery='residual-warps8-blayout',
                       math_contract=collect.MATH_CONTRACT)
        subject = control | dict(wy_delivery='residual-warps8-hlayout')
        collect.validate_wy_control(control, subject)
        for wrong in ('residual', 'residual-warps8', 'residual-warps8-operands'):
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                collect.validate_wy_control(control | dict(wy_delivery=wrong), subject)

    def test_hlayout_profile_and_capture_select_paired_api(self):
        inputs = (1,2,3,4,5)
        with patch('actlize_la.gdn_chunk_residual', return_value='paired-H') as call:
            subject, _ = profile.subject_call('wy','wy',Path('/binding.so'),inputs,
                                               'residual-warps8-hlayout')
            self.assertEqual(subject(), 'paired-H')
            call.assert_called_once_with(*inputs, output_final_state=True, delivery='warps8-hlayout')
        directory = self.directory()
        status, commands, _, _ = self.run_mock_capture(
            directory, control='residual-warps8-blayout', subject_delivery='residual-warps8-hlayout')
        self.assertEqual(status['status'], 'PASS', status['errors'])
        captures = [cmd for cmd in commands if '--phase' in cmd and cmd[cmd.index('--phase')+1] == 'subject']
        self.assertEqual([cmd[cmd.index('--wy-delivery')+1] for cmd in captures],
                         ['residual-warps8-blayout','residual-warps8-hlayout','residual-warps8-hlayout'])

    def test_hvlayout_requires_exact_backend_and_hlayout_control(self):
        import torch
        from unittest.mock import Mock
        from actlize_la import gdn_residual_interface as api
        inputs=tuple(torch.zeros(1) for _ in range(5))
        backend=SimpleNamespace(residual=Mock(),residual_warps8_hlayout=Mock())
        with patch.object(api,'_backend',return_value=backend),self.assertRaises(AttributeError):
            api.gdn_chunk_residual(*inputs,delivery='warps8-hvlayout')
        for call in vars(backend).values(): call.assert_not_called()
        control,_=self.records()
        control.update(role='wy',implementation='wy',wy_delivery='residual-warps8-hlayout',
                       math_contract=collect.MATH_CONTRACT)
        subject=control|dict(wy_delivery='residual-warps8-hvlayout')
        collect.validate_wy_control(control,subject)
        for wrong in ('residual','residual-warps8','residual-warps8-blayout','residual-warps8-hvlayout'):
            with self.subTest(wrong=wrong),self.assertRaises(ValueError):
                collect.validate_wy_control(control|dict(wy_delivery=wrong),subject)

    def test_hvlayout_profile_and_capture_select_both_paired_endpoints(self):
        inputs=(1,2,3,4,5)
        with patch('actlize_la.gdn_chunk_residual',return_value='paired-HV') as call:
            subject,_=profile.subject_call('wy','wy',Path('/binding.so'),inputs,
                                           'residual-warps8-hvlayout')
            self.assertEqual(subject(),'paired-HV')
            call.assert_called_once_with(*inputs,output_final_state=True,delivery='warps8-hvlayout')
        directory=self.directory()
        status,commands,_,_=self.run_mock_capture(directory,control='residual-warps8-hlayout',
                                                 subject_delivery='residual-warps8-hvlayout')
        self.assertEqual(status['status'],'PASS',status['errors'])
        captures=[cmd for cmd in commands if '--phase' in cmd and cmd[cmd.index('--phase')+1]=='subject']
        self.assertEqual([cmd[cmd.index('--wy-delivery')+1] for cmd in captures],
                         ['residual-warps8-hlayout','residual-warps8-hvlayout','residual-warps8-hvlayout'])

    def test_metadata_requires_new_backend_and_exact_hv_control(self):
        import torch
        from unittest.mock import Mock
        from actlize_la import gdn_residual_interface as api
        inputs=tuple(torch.zeros(1) for _ in range(5))
        backend=SimpleNamespace(residual=Mock(),residual_warps8_hvlayout=Mock())
        with patch.object(api,'_backend',return_value=backend),self.assertRaises(AttributeError):
            api.gdn_chunk_residual(*inputs,delivery='warps8-metadata')
        for call in vars(backend).values(): call.assert_not_called()
        control,_=self.records()
        control.update(role='wy',implementation='wy',wy_delivery='residual-warps8-hvlayout',
                       math_contract=collect.MATH_CONTRACT)
        subject=control|dict(wy_delivery='residual-warps8-metadata')
        collect.validate_wy_control(control,subject)
        for wrong in ('residual','residual-warps8-hlayout','residual-warps8-metadata'):
            with self.subTest(wrong=wrong),self.assertRaises(ValueError):
                collect.validate_wy_control(control|dict(wy_delivery=wrong),subject)

    def test_metadata_profile_selects_new_api_and_same_binary_hv(self):
        inputs=(1,2,3,4,5)
        with patch('actlize_la.gdn_chunk_residual',return_value='metadata-lookahead') as call:
            subject,_=profile.subject_call('wy','wy',Path('/binding.so'),inputs,
                                           'residual-warps8-metadata')
            self.assertEqual(subject(),'metadata-lookahead')
            call.assert_called_once_with(*inputs,output_final_state=True,delivery='warps8-metadata')
        directory=self.directory()
        status,commands,_,_=self.run_mock_capture(directory,control='residual-warps8-hvlayout',
                                                 subject_delivery='residual-warps8-metadata')
        self.assertEqual(status['status'],'PASS',status['errors'])
        captures=[cmd for cmd in commands if '--phase' in cmd and cmd[cmd.index('--phase')+1]=='subject']
        self.assertEqual([cmd[cmd.index('--wy-delivery')+1] for cmd in captures],
                         ['residual-warps8-hvlayout','residual-warps8-metadata','residual-warps8-metadata'])

    @classmethod
    def setUpClass(cls):
        # Small synthetic artifacts are retained, entirely under /workspace.
        cls.root = Path("/workspace") / f"gdn-acu-contract-{uuid4().hex}"
        cls.root.mkdir()

    def directory(self):
        directory = self.root / self._testMethodName
        directory.mkdir()
        return directory

    def test_portable_streaming_sha_matches_full_digest(self):
        directory = self.directory()
        path = directory / "non-ascii-证据.bin"
        data = bytes(range(256)) * 5000  # spans the streaming chunk boundary
        path.write_bytes(data)
        self.assertEqual(collect.sha(path), hashlib.sha256(data).hexdigest())

    def records(self):
        ours = dict(status="PASS", role="ours", phase="subject", warmup=0, public_api_calls=1, gate=-0.1,
                    shape=dict(B=1, S=2048, Hk=16, Hv=32, K=128, V=128), input_sha="input", reference_sha="ref",
                    device=dict(name="PPU", uuid="fixture", cu=72), torch="vendor",
                    torch_cuda="13.0", initial_state="zero", output_final_state=True,
                    fla_heads="native", max_relative_error_limit=0.02, protocol="single-forward",
                    cache_control="all", extension_sha256="binary", library_sha256="device-binary",
                    implementation="original", output_sha="output", fla={},
                    output_dtype="torch.bfloat16", state_dtype="torch.float32")
        fla = copy.deepcopy(ours)
        fla["role"] = "fla"
        return ours, fla

    def test_site_acu_outranks_sdk_and_path(self):
        directory = self.directory()
        sdk = directory / "sdk"
        matched = sdk / "asight/bin/acu"
        matched.parent.mkdir(parents=True)
        matched.write_text("synthetic executable identity only\n")
        matched.chmod(0o755)
        site = Path("/sim/eec/shared/junfu.qx/asight/bin/acu")
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(collect.shutil, "which", return_value=str(directory / "older-acu")), \
                patch.object(Path, "is_file", return_value=True), \
                patch.object(collect.os, "access", return_value=True):
            self.assertEqual(collect.find_acu(), site)
            self.assertNotEqual(collect.find_acu(), matched)

    def test_missing_site_acu_does_not_fall_back_to_sdk_or_path(self):
        site = Path("/sim/eec/shared/junfu.qx/asight/bin/acu")
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(collect.shutil, "which", return_value="/sdk/asight/bin/acu"), \
                patch.object(Path, "is_file", side_effect=lambda p: p != site, autospec=True), \
                patch.object(collect.os, "access", return_value=True):
            self.assertFalse(site.is_file())
            self.assertTrue(Path("/sdk/asight/bin/acu").is_file())
            self.assertEqual(collect.find_acu(), site)

    def test_explicit_acu_is_not_silently_replaced(self):
        directory = self.directory()
        absent = directory / "missing-explicit-acu"
        with patch.dict(os.environ, {"ACU": str(absent)}, clear=True):
            self.assertEqual(collect.find_acu(), absent.resolve())

    def test_direct_subject_calls_once_without_profiler_hooks(self):
        order = []
        def step(name):
            return lambda: order.append(name)
        profile.capture_one(step("call"), step("sync"))
        self.assertEqual(order, ["sync", "call", "sync"])

    def test_subject_call_failure_is_not_swallowed(self):
        order = []
        def bad():
            raise RuntimeError("planted launch failure")
        with self.assertRaisesRegex(RuntimeError, "launch failure"):
            profile.capture_one(bad, lambda: order.append("sync"))
        self.assertEqual(order, ["sync"])

    def test_native_exception_keeps_failed_receipt_and_is_not_retried(self):
        directory = self.directory()
        receipt = directory / "wy.json"
        calls = []
        thrown = IndexError("map::at")
        def bad():
            calls.append("forward")
            raise thrown
        with patch.object(profile, "loaded_library_hashes", return_value={"/sdk/libparser.so": "hash"}):
            with self.assertRaises(IndexError) as error:
                profile.run_with_failure_receipt(bad, receipt, dict(role="wy", phase="subject"))
        self.assertIs(error.exception, thrown)
        self.assertEqual(calls, ["forward"])
        saved = json.loads(receipt.read_text())
        self.assertEqual(saved["status"], "FAIL")
        self.assertEqual(saved["error"], "map::at")
        self.assertEqual(saved["loaded_libraries"], {"/sdk/libparser.so": "hash"})
        _, fla = self.records()
        with self.assertRaises(ValueError):
            collect.validate_pair(saved, fla, "wy")

    def test_profiler_libraries_join_identity_without_loading_anything(self):
        directory = self.directory()
        paths = {directory / name for name in ("libhgBinaryAnalysis.so.13", "libhggc_injection.so",
                 "libperfworks.so", "_gdn_wy_ppu.so", "unrelated.so")}
        for path in paths:
            path.write_bytes(b"synthetic DSO")
        with patch.object(profile, "loaded_library_paths", return_value=paths):
            actual = profile.loaded_library_hashes()
        self.assertEqual(len(actual), 4)
        self.assertNotIn(str(directory / "unrelated.so"), actual)

    def test_wy_is_not_an_alias_for_original(self):
        import actlize_la
        inputs = (object(),) * 5
        extension = self.root / "_gdn_wy_ppu.so"
        with patch.object(actlize_la, "gdn_chunk_wy", return_value="wy-result") as wy, \
                patch.object(profile.bench.admission, "gdn_chunk", side_effect=AssertionError("wrong API")), \
                patch.dict(os.environ, {}, clear=True):
            call, identity = profile.subject_call("wy", "wy", extension, inputs)
            self.assertEqual(call(), "wy-result")
            wy.assert_called_once_with(*inputs, output_final_state=True)
            self.assertEqual(os.environ["GDN_QSA_WY_EXTENSION"], str(extension.resolve()))
            self.assertNotIn("GDN_QSA_PPU_EXTENSION", os.environ)
            self.assertEqual(identity, {})
            for role, implementation in (("ours", "wy"), ("wy", "original"), ("fla", "unknown")):
                with self.assertRaises(ValueError):
                    profile.subject_call(role, implementation, extension, inputs)

    def test_no_profiler_library_or_api_dependency_remains(self):
        source = (ROOT / "benchmarks/profile_ppu_gdn_fla.py").read_text()
        for forbidden in ("PPUProfiler", "ProfilerStart", "ProfilerStop", "ctypes", "cuda.profiler"):
            self.assertNotIn(forbidden, source)

    def test_residual_dispatch_is_new_api_not_a_delivery_alias(self):
        inputs = (object(),) * 5
        with patch("actlize_la.gdn_chunk_residual", return_value="new-algorithm") as new, \
                patch("actlize_la.gdn_chunk_wy", side_effect=AssertionError("wrong math API")):
            call, _ = profile.subject_call("wy", "wy", Path("/binding.so"), inputs, "residual")
            self.assertEqual(call(), "new-algorithm")
            new.assert_called_once_with(*inputs, output_final_state=True, delivery="scalar")
        self.assertNotIn("residual", collect.DELIVERIES)
        self.assertIsNone(collect.PROFILE_VARIANTS["residual"])

    def test_residual_admission_cannot_fake_raw_equality_or_loosen_tolerance(self):
        arm = dict(math_contract=collect.MATH_CONTRACT, delivery_mask=None,
                   scalar_raw_bit_equal=None, errors=[.008, .004])
        collect.validate_residual_admission(arm, .02)
        for key, value in (("math_contract", "old"), ("delivery_mask", 62960),
                           ("scalar_raw_bit_equal", True), ("errors", [.02, .004]),
                           ("errors", [float("nan"), .004]), ("errors", [])):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                collect.validate_residual_admission(arm | {key: value}, .02)
        for key in arm:
            bad = arm.copy()
            del bad[key]
            with self.assertRaises(ValueError):
                collect.validate_residual_admission(bad, .02)
        with self.assertRaises(ValueError):
            collect.validate_residual_admission(arm, .03)

    def test_old_delivery_raw_gate_is_not_weakened_by_new_algorithm(self):
        control, _ = self.records()
        control.update(role="wy", implementation="wy", wy_delivery="split-prepare")
        subject = control | dict(wy_delivery="state-pipeline", output_sha="different", errors=[.001, .001])
        with self.assertRaisesRegex(ValueError, "output_sha"):
            collect.validate_wy_control(control, subject)
        for variant in (None, "scalar", "split-prepare"):
            with self.assertRaises(ValueError):
                collect.capture_arms(Path("/bundle"), "wy", "residual", variant)

    def test_complete_residual_capture_allows_only_explicit_new_rounding(self):
        status, commands, bundle, _ = self.run_mock_capture(
            self.directory(), control="state-pipeline", subject_delivery="residual")
        self.assertEqual(status["status"], "PASS", status["errors"])
        self.assertEqual(status["capture_order"], ["wy-control", "wy", "fla"])
        calls = [cmd for cmd in commands if "--set" in cmd]
        self.assertEqual(len(calls), 3)
        self.assertEqual([cmd[cmd.index("--wy-delivery")+1] for cmd in calls],
                         ["state-pipeline", "residual", "residual"])
        before = json.loads((bundle / "wy-control/wy.json").read_text())
        after = json.loads((bundle / "wy.json").read_text())
        self.assertNotEqual(before["output_sha"], after["output_sha"])
        self.assertNotEqual(before["math_contract"], after["math_contract"])

    def test_residual_bad_numeric_or_lost_contract_fails_before_profile(self):
        root = self.directory()
        for plant in ("residual-wrong-contract", "residual-bad-error"):
            directory = root / plant
            directory.mkdir()
            status, commands, _, _ = self.run_mock_capture(
                directory, control="state-pipeline", subject_delivery="residual", plant=plant)
            self.assertEqual(status["status"], "INCOMPLETE")
            self.assertTrue(status["errors"])
            self.assertFalse(any("--set" in cmd for cmd in commands))

    def test_only_preflight_warms_up_and_comparison_runs_on_cpu(self):
        import torch
        want = (torch.tensor([1.0]), torch.tensor([2.0]))
        copies, calls = [], []
        class DeviceResult:
            def __init__(self, cpu):
                self.value = cpu
            def detach(self):
                return self
            def cpu(self):
                copies.append("D2H")
                return self.value
            def float(self):
                raise AssertionError("must not launch device verification casts")
        def call():
            calls.append("forward")
            return tuple(DeviceResult(x) for x in want)
        subject = profile.run_phase(call, lambda: None, want, "subject", 5)
        self.assertEqual(len(calls), 1)
        self.assertEqual(copies, ["D2H", "D2H"])
        self.assertEqual((subject["public_api_calls"], subject["warmup"]), (1, 0))
        calls.clear()
        preflight = profile.run_phase(call, lambda: None, want, "preflight", 5)
        self.assertEqual(len(calls), 6)
        self.assertEqual(preflight["output_sha"], subject["output_sha"])
        with self.assertRaises(ValueError):
            profile.run_phase(call, lambda: None, want, "unknown", 5)

    def test_preflight_and_subject_must_match(self):
        subject, _ = self.records()
        preflight = subject | dict(phase="preflight", warmup=5, public_api_calls=6)
        collect.validate_preflight(preflight, subject)
        for key, value in (("output_sha", "changed"), ("device", {}), ("input_sha", "changed"),
                           ("warmup", 5), ("phase", "preflight"), ("public_api_calls", 6)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                collect.validate_preflight(preflight, subject | {key: value})
        with self.assertRaises(ValueError):
            collect.validate_preflight({}, subject)

    def test_all_kernels_profiled_not_warmup_or_truncated_subset(self):
        cmd = collect.acu_command(Path("/acu"), Path("/report"), Path("/binding.so"),
                                  "ours", -0.1, Path("/bundle"))
        self.assertEqual(cmd[:6], ["/acu", "-f", "-o", "/report", "--set", "full"])
        for flag, value in (("--set", "full"), ("--phase", "subject"),
                            ("--check-exit-code", "yes"), ("--gate", "-0.1")):
            self.assertEqual(cmd[cmd.index(flag) + 1], value)
        for forbidden in ("--profile-from-start", "--launch-count", "--kernel-name",
                          "--disable-profiler-start-stop", "--csv"):
            self.assertNotIn(forbidden, cmd)
        preflight = collect.child_command(Path("/binding.so"), "ours", -0.1, Path("/bundle"), "preflight")
        self.assertEqual(preflight[preflight.index("--phase") + 1], "preflight")
        self.assertNotIn("/acu", preflight)
        wy = collect.acu_command(Path("/acu"), Path("/report"), Path("/_gdn_wy_ppu.so"),
                                 "wy", -0.1, Path("/bundle"), "wy")
        self.assertEqual(wy[wy.index("--implementation") + 1], "wy")
        self.assertEqual(wy[wy.index("--role") + 1], "wy")

    def test_changed_input_device_or_missing_receipt_cannot_pass(self):
        ours, fla = self.records()
        collect.validate_pair(ours, fla)
        for key, value in (("input_sha", "different"), ("device", dict(cu=32)),
                           ("gate", -1.0), ("public_api_calls", 2), ("extension_sha256", "stale"),
                           ("output_final_state", False), ("status", "FAIL"), ("role", "ours")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                collect.validate_pair(ours, fla | {key: value})
        with self.assertRaises(ValueError):
            collect.validate_pair(ours, {})
        with self.assertRaises(ValueError):
            collect.validate_pair(ours | dict(input_sha=""), fla | dict(input_sha=""))

    def test_delivery_reaches_subject_and_receipts_cannot_cross_variants(self):
        for delivery in collect.DELIVERIES:
            if delivery == "scalar":
                continue
            cmd = collect.acu_command(Path("/acu"), Path("/report"), Path("/_gdn_wy_ppu.so"),
                                      "wy", -.1, Path("/bundle"), "wy", delivery)
            self.assertEqual(cmd[cmd.index("--wy-delivery") + 1], delivery)
            with patch("actlize_la.gdn_chunk_wy", return_value=("out", "state")) as forward:
                call, _ = profile.subject_call("wy", "wy", Path("/_gdn_wy_ppu.so"), (1, 2, 3, 4, 5), delivery)
                self.assertEqual(call(), ("out", "state"))
                self.assertEqual(forward.call_args.kwargs["delivery"], delivery)
        ours, fla = self.records()
        ours.update(role="wy", implementation="wy", wy_delivery="all")
        fla.update(implementation="wy", wy_delivery="all")
        collect.validate_pair(ours, fla, "wy")
        with self.assertRaisesRegex(ValueError, "delivery"):
            collect.validate_pair(ours, fla | dict(wy_delivery="scalar"), "wy")
        pre = ours | dict(phase="preflight", warmup=5, public_api_calls=6)
        collect.validate_preflight(pre, ours)
        with self.assertRaisesRegex(ValueError, "wy_delivery"):
            collect.validate_preflight(pre | dict(wy_delivery="state"), ours)

    def test_wy_receipt_cannot_use_original_role_or_other_device_library(self):
        ours, fla = self.records()
        wy = ours | dict(role="wy", implementation="wy")
        fla["implementation"] = "wy"
        collect.validate_pair(wy, fla, "wy")
        for plant in (wy | dict(role="ours"), wy | dict(implementation="original"),
                      wy | dict(library_sha256="stale")):
            with self.assertRaises(ValueError):
                collect.validate_pair(plant, fla, "wy")
        with self.assertRaises(ValueError):
            collect.validate_pair(wy, fla)  # default still means ORIGINAL

    def make_wy_run(self, directory):
        build = directory / "build"
        build.mkdir()
        binding, library = build / "_gdn_wy_ppu.cpython312.so", build / "libgdn_wy_ppu.so"
        binding.write_bytes(b"WY binding")
        library.write_bytes(b"WY device library")
        (directory / "sha.txt").write_text("d" * 40 + "\n")
        (directory / "source.diff").write_text("")
        (directory / "binaries.sha256").write_text("".join(
            f"{collect.sha(path)}  {path}\n" for path in (binding, library)))
        comparison = dict(protocol="full-public-api-event-span", cases=[dict(g=-0.1)],
                          binary_sha256={str(binding): collect.sha(binding)})
        (directory / "comparison.json").write_text(json.dumps(comparison))
        return binding, library

    def test_reused_binding_and_device_library_are_both_bound_to_old_run(self):
        directory = self.directory()
        binding, library = self.make_wy_run(directory)
        chosen, _, origin = collect.read_wy_run(directory)
        self.assertEqual(chosen, binding.resolve())
        self.assertEqual(origin["source_sha"], "d" * 40)
        for binary in (binding, library):
            original = binary.read_bytes()
            binary.write_bytes(b"planted replacement")
            with self.subTest(binary=binary), self.assertRaisesRegex(ValueError, "manifest"):
                collect.read_wy_run(directory)
            binary.write_bytes(original)
        record = json.loads((directory / "comparison.json").read_text())
        record["binary_sha256"][str(binding)] = "planted-other-run"
        (directory / "comparison.json").write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "comparison.json"):
            collect.read_wy_run(directory)

    def test_reuse_missing_or_ambiguous_binding_is_red(self):
        directory = self.directory()
        with self.assertRaisesRegex(ValueError, "one reused WY"):
            collect.read_wy_run(directory)
        binding, _ = self.make_wy_run(directory)
        (binding.parent / "_gdn_wy_ppu.old.so").write_bytes(b"ambiguous")
        with self.assertRaisesRegex(ValueError, "one reused WY"):
            collect.read_wy_run(directory)

    def test_numeric_only_receipt_keeps_repeats_and_raw_bit_admission(self):
        directory = self.directory()
        self.make_wy_run(directory)
        path = directory / "comparison.json"
        record = json.loads(path.read_text())
        record.update(protocol="numeric-admission-for-acu", samples=0)
        record["cases"][0].update(timing="NOT_RUN", arms={
            role: dict(admitted_repeats=8, samples_us=[], scalar_raw_bit_equal=True)
            for role in ("wy", "wy-aiu-state-output", "wy-split-prepare", "fla")})
        path.write_text(json.dumps(record))
        _, accepted, _ = collect.read_wy_run(directory)
        self.assertEqual(accepted["protocol"], "numeric-admission-for-acu")
        for field, value in (("admitted_repeats", 7), ("samples_us", [10.]),
                             ("scalar_raw_bit_equal", False)):
            wrong = copy.deepcopy(record)
            wrong["cases"][0]["arms"]["wy-split-prepare"][field] = value
            path.write_text(json.dumps(wrong))
            with self.subTest(field=field), self.assertRaises(ValueError):
                collect.read_wy_run(directory)
        wrong = copy.deepcopy(record)
        wrong["samples"] = 10
        path.write_text(json.dumps(wrong))
        with self.assertRaisesRegex(ValueError, "must not claim API timing"):
            collect.read_wy_run(directory)

    def test_comparison_rebinding_rejects_changed_fla_input_or_output(self):
        ours, fla = self.records()
        ours.update(role="wy", implementation="wy")
        fla["implementation"] = "wy"
        ours["device"]["properties"] = "synthetic properties, no UUID in old run"
        comparison = dict(cases=[dict(g=-0.1, input_sha="input", arms={
            "wy": dict(fingerprint="output", state_dtype="torch.float32"),
            "fla": dict(fingerprint="output", state_dtype="torch.float32")})],
            initial_state="zero", final_state=True, qk_norm=False, scale="1/sqrt(128)",
            dtype="bf16", torch="vendor", fla={}, limit=0.02,
            device=ours["device"]["properties"])
        collect.validate_comparison(comparison, ours, fla)
        packed = ours | dict(wy_delivery="all")
        with self.assertRaisesRegex(ValueError, "not admitted"):
            collect.validate_comparison(comparison, packed, fla)
        comparison["delivery_ab"] = True
        with self.assertRaisesRegex(ValueError, "not admitted"):
            collect.validate_comparison(comparison, packed, fla)
        comparison["cases"][0]["arms"]["wy-all"] = dict(fingerprint="output", state_dtype="torch.float32", delivery_mask=7)
        collect.validate_comparison(comparison, packed, fla)
        # Identical numerical fingerprints must not let a legacy capture
        # impersonate the new compute-tile kernel family.
        tiled = ours | dict(wy_delivery="tiled-all")
        with self.assertRaisesRegex(ValueError, "not admitted"):
            collect.validate_comparison(comparison, tiled, fla)
        comparison["cases"][0]["arms"]["wy-tiled-all"] = dict(fingerprint="output", state_dtype="torch.float32", delivery_mask=56)
        collect.validate_comparison(comparison, tiled, fla)
        # Same numerical answer is not evidence that mask48 was measured:
        # neither all-tiled nor state-only is the new mixed-stage combination.
        pair = ours | dict(wy_delivery="tiled-state-output")
        comparison["cases"][0]["arms"]["wy-tiled-state"] = dict(fingerprint="output", state_dtype="torch.float32")
        with self.assertRaisesRegex(ValueError, "not admitted"):
            collect.validate_comparison(comparison, pair, fla)
        comparison["cases"][0]["arms"]["wy-tiled-state-output"] = dict(fingerprint="output", state_dtype="torch.float32", delivery_mask=48)
        collect.validate_comparison(comparison, pair, fla)
        for suffix in ("address", "gates", "both"):
            name = f"tiled-state-output-{suffix}"
            subject = ours | dict(wy_delivery=name)
            with self.assertRaisesRegex(ValueError, "not admitted"):
                collect.validate_comparison(comparison, subject, fla)
            comparison["cases"][0]["arms"][f"wy-{name}"] = dict(fingerprint="output", state_dtype="torch.float32", delivery_mask=collect.DELIVERIES[name])
            collect.validate_comparison(comparison, subject, fla)
        for suffix in ("prepare", "output", "both"):
            name = f"stage-address-{suffix}"
            subject = ours | dict(wy_delivery=name)
            with self.assertRaisesRegex(ValueError, "not admitted"):
                collect.validate_comparison(comparison, subject, fla)
            comparison["cases"][0]["arms"][f"wy-{name}"] = dict(fingerprint="output", state_dtype="torch.float32", delivery_mask=collect.DELIVERIES[name])
            collect.validate_comparison(comparison, subject, fla)
        for suffix in ("shared", "warp"):
            name = f"prepare-rows-{suffix}"
            subject = ours | dict(wy_delivery=name)
            with self.assertRaisesRegex(ValueError, "not admitted"):
                collect.validate_comparison(comparison, subject, fla)
            comparison["cases"][0]["arms"][f"wy-{name}"] = dict(fingerprint="output", state_dtype="torch.float32", delivery_mask=collect.DELIVERIES[name])
            collect.validate_comparison(comparison, subject, fla)
        for role, key, value in (("wy", "input_sha", "other"), ("wy", "output_sha", "other"),
                                  ("fla", "fla", dict(entry_sha256="changed")),
                                  ("wy", "state_dtype", "torch.bfloat16"),
                                  ("fla", "output_sha", "other")):
            with self.subTest(role=role, key=key), self.assertRaises(ValueError):
                collect.validate_comparison(comparison, ours | ({key: value} if role == "wy" else {}),
                                             fla | ({key: value} if role == "fla" else {}))

    def run_mock_capture(self, directory, control=None, plant=None, subject_delivery=None, sequence=2048):
        """Run the real orchestration with synthetic tool receipts, not a GPU."""
        prior = directory / "preceding"
        prior.mkdir()
        binding, library = self.make_wy_run(prior)
        ours, fla = self.records()
        ours["shape"]["S"] = sequence
        ours.update(role="wy", implementation="wy", extension_sha256=collect.sha(binding),
                    library_sha256=collect.sha(library),
                    loaded_libraries={str(p.resolve()): collect.sha(p) for p in (binding, library)})
        ours["device"]["properties"] = "synthetic PPU properties"
        fla = ours | dict(role="fla", loaded_libraries={})
        comparison_path = prior / "comparison.json"
        comparison = json.loads(comparison_path.read_text())
        comparison.update(initial_state="zero", final_state=True, qk_norm=False,
                          scale="1/sqrt(128)", dtype="bf16", torch="vendor", fla={}, limit=0.02,
                          device=ours["device"]["properties"])
        comparison["cases"] = [dict(g=-0.1, input_sha="input", arms={
            role: dict(fingerprint="output", state_dtype="torch.float32") for role in ("wy", "fla")})]
        if sequence != 2048 and plant != "missing-case-shape":
            comparison["cases"][0]["shape"] = dict(ours["shape"])
        delivery = subject_delivery or ("aiu-state-output" if control is not None else "scalar")
        if control is not None:
            # Reproduce the already-shipped AIU JSON exactly at the disputed
            # boundary: generic flag false, no aiu_ab flag, correct arm masks.
            comparison["delivery_ab"] = False
            comparison["cases"][0]["arms"].update({f"wy-{name}": dict(
                fingerprint="output", state_dtype="torch.float32", delivery_mask=mask)
                for name, mask in ((control, collect.PROFILE_VARIANTS[control]), (delivery, collect.PROFILE_VARIANTS[delivery]))})
        if delivery == "residual":
            comparison["cases"][0]["arms"]["wy-residual"].update(
                fingerprint="residual-output", math_contract=collect.MATH_CONTRACT,
                errors=[.008, .004], scalar_raw_bit_equal=None)
        if delivery in collect.RESIDUAL_DELIVERIES:
            for variant in {"residual",delivery,control}:
                comparison["cases"][0]["arms"].setdefault(f"wy-{variant}", dict(delivery_mask=None)).update(
                    fingerprint="residual-output",math_contract=collect.MATH_CONTRACT,
                    errors=[.008,.004],scalar_raw_bit_equal=None)
                if variant != "residual":
                    comparison["cases"][0]["arms"][f"wy-{variant}"].update(
                        residual_raw_bit_equal=True,residual_fingerprint="residual-output")
        if plant == "missing-comparison-arm":
            del comparison["cases"][0]["arms"]["wy-prepare-rows-shared"]
        if plant == "wrong-comparison-mask":
            comparison["cases"][0]["arms"]["wy-aiu-state-output"]["delivery_mask"] = 1520
        if plant == "missing-comparison-mask":
            del comparison["cases"][0]["arms"]["wy-aiu-state-output"]["delivery_mask"]
        comparison_path.write_text(json.dumps(comparison))
        original_comparison_hash = collect.sha(comparison_path)
        bundle = directory / "bundle"
        bundle.mkdir()
        args = SimpleNamespace(wy_run=prior, extension=None, sdk=directory,
                               acu=Path(sys.executable), gate=-0.1, device="0", sequence=sequence,
                               wy_delivery=delivery, wy_control=control)
        commands = []
        def fake_run(command, log, env, **kwargs):
            command = [str(x) for x in command]
            commands.append(command)
            self.assertNotIn("bash", command)
            self.assertNotIn("cmake", command)
            log.write_text("SYNTHETIC TOOL OUTPUT; not device evidence\n")
            if "--role" in command:
                role = command[command.index("--role") + 1]
                phase = command[command.index("--phase") + 1]
                selected = command[command.index("--wy-delivery") + 1]
                is_control = role == "wy" and selected == control
                self.assertIn(role, ("wy", "fla"))
                self.assertEqual(command[command.index("--implementation") + 1], "wy")
                receipt = dict(ours if role == "wy" else fla)
                if sequence != 2048:
                    self.assertEqual(command[command.index("--sequence") + 1], str(sequence))
                if plant == "ignored-sequence":
                    receipt["shape"] = dict(receipt["shape"], S=2048)
                receipt["wy_delivery"] = selected
                if delivery == "residual" and role == "wy":
                    receipt.update(math_contract=collect.MATH_CONTRACT if selected == "residual" else collect.WY_MATH_CONTRACT,
                                   errors=[.008, .004])
                    if selected == "residual":
                        receipt["output_sha"] = "residual-output"
                        if plant == "residual-wrong-contract":
                            receipt["math_contract"] = collect.WY_MATH_CONTRACT
                        if plant == "residual-bad-error":
                            receipt["errors"] = [.03, .004]
                if delivery in collect.RESIDUAL_DELIVERIES and role == "wy":
                    receipt.update(math_contract=collect.MATH_CONTRACT,errors=[.008,.004],output_sha="residual-output")
                if phase == "preflight":
                    receipt.update(phase=phase, warmup=5, public_api_calls=6)
                if is_control and plant == "changed-control-device":
                    receipt["device"] = ours["device"] | dict(uuid="different-card")
                if is_control and plant == "ignored-control-delivery":
                    receipt["wy_delivery"] = delivery
                if is_control and phase == "subject":
                    if plant == "failed-control-profile":
                        raise RuntimeError("planted ACU failure")
                    if plant == "wrong-control-loaded-library":
                        receipt["loaded_libraries"] = {str(binding.resolve()): collect.sha(binding)}
                    if plant == "changed-control-output":
                        receipt["output_sha"] = "changed-output"
                Path(command[command.index("--receipt") + 1]).write_text(json.dumps(receipt))
                if "--set" in command and not (is_control and plant == "missing-control-report"):
                    Path(command[command.index("-o") + 1] + ".acurep").write_bytes(b"synthetic report")
            return dict(status="COLLECTED", returncode=0)
        with patch.object(collect, "run", side_effect=fake_run), \
                patch.object(collect.shutil, "which", return_value=None):
            status = collect.collect(args, bundle, {"PATH": ""})
        self.assertEqual(collect.sha(comparison_path), original_comparison_hash)
        return status, commands, bundle, library

    def test_mixed_tail_sequence_reaches_every_actual_child_and_receipt(self):
        root = self.directory()
        for sequence in (2049, 2111):
            directory = root / str(sequence)
            directory.mkdir()
            status, commands, _, _ = self.run_mock_capture(
                directory,
                control="residual-full-chunk", subject_delivery="residual-mixed-tail", sequence=sequence)
            self.assertEqual(status["status"], "PASS", status["errors"])
            children = [c for c in commands if "--role" in c]
            self.assertEqual(len(children), 6)
            self.assertTrue(all(c[c.index("--sequence") + 1] == str(sequence) for c in children))

    def test_mixed_tail_rejects_ignored_sequence_and_unshaped_old_admission(self):
        root = self.directory()
        for plant in ("ignored-sequence", "missing-case-shape"):
            directory = root / plant
            directory.mkdir()
            status, commands, _, _ = self.run_mock_capture(directory,
                control="residual-full-chunk", subject_delivery="residual-mixed-tail",
                sequence=2049, plant=plant)
            self.assertNotEqual(status["status"], "PASS")
            self.assertFalse(any("--set" in c for c in commands))

    def test_pipeline_capture_keeps_same_binary_split_control_and_all_fla_kernels(self):
        status, commands, bundle, _ = self.run_mock_capture(
            self.directory(),control="split-prepare",subject_delivery="state-pipeline")
        self.assertEqual(status["status"],"PASS")
        subjects = [cmd for cmd in commands if "--set" in cmd]
        self.assertEqual(len(subjects),3)
        self.assertEqual([(cmd[cmd.index("--role")+1],cmd[cmd.index("--wy-delivery")+1]) for cmd in subjects],
                         [("wy","split-prepare"),("wy","state-pipeline"),("fla","state-pipeline")])
        for cmd in subjects:
            for forbidden in ("--launch-count","--kernel-name","--csv"):
                self.assertNotIn(forbidden,cmd)
        self.assertEqual(status["capture_order"],["wy-control","wy","fla"])
        control=json.loads((bundle/"wy-control/wy.json").read_text())
        subject=json.loads((bundle/"wy.json").read_text())
        collect.validate_wy_control(control,subject)
        with self.assertRaises(ValueError):
            collect.validate_wy_control(control,subject | dict(wy_delivery="split-prepare"))

    def test_complete_reused_wy_capture_never_builds_or_selects_original(self):
        status, commands, bundle, library = self.run_mock_capture(self.directory())
        self.assertEqual(status["status"], "PASS", status)
        self.assertEqual(status["implementation"], "wy")
        self.assertEqual(status["comparison_origin"]["source_sha"], "d" * 40)
        self.assertTrue((bundle / "binaries" / library.name).is_file())
        self.assertTrue((bundle / "preceding-comparison/comparison.json").is_file())
        stages = [cmd[cmd.index("--phase") + 1] for cmd in commands if "--phase" in cmd]
        self.assertEqual(stages, ["preflight", "preflight", "subject", "subject"])

    def test_three_arm_aiu_capture_accepts_measured_masks_despite_old_flag(self):
        status, commands, bundle, library = self.run_mock_capture(
            self.directory(), control="prepare-rows-shared")
        self.assertEqual(status["status"], "PASS", status)
        self.assertEqual(status["capture_order"], ["wy-control", "wy", "fla"])
        calls = [cmd for cmd in commands if "--phase" in cmd]
        self.assertEqual([cmd[cmd.index("--phase") + 1] for cmd in calls],
                         ["preflight"] * 3 + ["subject"] * 3)
        captures = [cmd for cmd in calls if "--set" in cmd]
        self.assertEqual([(cmd[cmd.index("--role") + 1], cmd[cmd.index("--wy-delivery") + 1])
                         for cmd in captures], [("wy", "prepare-rows-shared"),
                         ("wy", "aiu-state-output"), ("fla", "aiu-state-output")])
        reports = list(bundle.rglob("*.acurep"))
        self.assertEqual(len(reports), 3)
        self.assertEqual(len(set(cmd[cmd.index("--receipt") + 1] for cmd in calls)), 6)
        self.assertTrue((bundle / "wy-control/wy.json").is_file())
        archive = collect.pack(bundle, status)
        with tarfile.open(archive) as tar:
            self.assertEqual(len([name for name in tar.getnames() if name.endswith(".acurep")]), 3)

    def test_three_arm_missing_wrong_mask_device_delivery_or_report_is_red(self):
        directory = self.directory()
        for plant in ("missing-comparison-arm", "wrong-comparison-mask", "missing-comparison-mask", "changed-control-device",
                      "ignored-control-delivery", "failed-control-profile", "missing-control-report",
                      "wrong-control-loaded-library", "changed-control-output"):
            trial = directory / plant
            trial.mkdir()
            with self.subTest(plant=plant):
                status, commands, _, _ = self.run_mock_capture(trial, "prepare-rows-shared", plant)
                self.assertEqual(status["status"], "INCOMPLETE", status)
                self.assertTrue(status["errors"])
                if plant in ("missing-comparison-arm", "wrong-comparison-mask", "missing-comparison-mask", "changed-control-device",
                             "ignored-control-delivery"):
                    self.assertFalse(any("--set" in cmd for cmd in commands))

    def test_control_inventory_rejects_duplicates_unknown_and_original(self):
        directory = self.directory()
        for implementation, delivery, control in (("wy", "aiu-state-output", "aiu-state-output"),
                ("original", "scalar", "prepare-rows-shared"), ("wy", "aiu-state-output", "typo")):
            with self.subTest(control=control), self.assertRaises(ValueError):
                collect.capture_arms(directory, implementation, delivery, control)

    def test_report_extension_variants_missing_empty_ambiguous(self):
        directory = self.directory()
        for suffix in ("", ".acurep"):
            base = directory / ("exact.report" if not suffix else "appended.report")
            report = Path(str(base) + suffix)
            report.write_bytes(b"synthetic report")
            self.assertEqual(collect.report_file(base), report)
        for base in (directory / "absent", directory / "empty", directory / "duplicate"):
            if base.name == "empty":
                base.touch()
            if base.name == "duplicate":
                base.write_bytes(b"one")
                Path(str(base) + ".acurep").write_bytes(b"two")
            with self.subTest(base=base), self.assertRaises(RuntimeError):
                collect.report_file(base)

    def test_loaded_device_library_stale_is_red(self):
        directory = self.directory()
        binding, library = directory / "_gdn_chunk_ppu.so", directory / "libgdn_qsa_ppu.so"
        binding.write_bytes(b"binding")
        library.write_bytes(b"device")
        loaded = {str(path): collect.sha(path) for path in (binding, library)}
        collect.validate_loaded_binary(dict(loaded_libraries=loaded), binding, library)
        for plant in (loaded | {str(library): "stale"}, {str(binding): collect.sha(binding)}, {}):
            with self.assertRaises(ValueError):
                collect.validate_loaded_binary(dict(loaded_libraries=plant), binding, library)

    def test_command_failure_not_mistaken_for_optional_probe(self):
        directory = self.directory()
        with self.assertRaisesRegex(RuntimeError, "rc=7"):
            collect.run([sys.executable, "-c", "raise SystemExit(7)"], directory / "required.log",
                        os.environ.copy(), console=False)
        optional = collect.run([sys.executable, "-c", "raise SystemExit(7)"], directory / "optional.log",
                               os.environ.copy(), console=False, optional=True)
        self.assertEqual(optional["status"], "UNAVAILABLE")

    def test_tar_contents_and_checksums_include_failed_capture(self):
        directory = self.directory()
        bundle = directory / "bundle"
        bundle.mkdir()
        (bundle / "ours.report.acurep").write_bytes(b"synthetic native report")
        (bundle / "fla-acu.log").write_text("planted FLA failure\n")
        archive = collect.pack(bundle, dict(status="INCOMPLETE", errors=["FLA failed"]))
        with tarfile.open(archive) as tar:
            names = tar.getnames()
            self.assertTrue(all(not n.startswith("/") and ".." not in Path(n).parts for n in names))
            status = json.load(tar.extractfile(f"{directory.name}/STATUS.json"))
            self.assertEqual(status["status"], "INCOMPLETE")
            sums = tar.extractfile(f"{directory.name}/SHA256SUMS").read().decode().splitlines()
            for row in sums:
                expected, name = row.split("  ", 1)
                self.assertEqual(expected, collect.sha(bundle / name))
            self.assertIn(f"{directory.name}/ours.report.acurep", names)

    def test_archive_cannot_follow_source_links(self):
        directory = self.directory()
        bundle = directory / "bundle"
        bundle.mkdir()
        outside = directory / "not-for-upload"
        outside.write_text("synthetic private file")
        (bundle / "planted-link").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "symlinks"):
            collect.pack(bundle, dict(status="INCOMPLETE"))


if __name__ == "__main__":
    unittest.main()
