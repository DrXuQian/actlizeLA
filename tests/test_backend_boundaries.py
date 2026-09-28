"""Actual dispatcher/build-selector negatives. Host-only; no GPU or downloads."""
import os
import copy
import json
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from actlize_la import gdn_forward, backend_inventory
from actlize_la.backends.registry import require_backend, require_implementation


class Dispatch(unittest.TestCase):
    def setUp(self):
        self.inputs = tuple(object() for _ in range(5))

    def run_call(self, algorithm="original", backend=None, env=None, **kwargs):
        outputs = (object(), object())
        call = Mock(return_value=outputs)
        entry = {"original": "gdn_chunk", "wy": "gdn_chunk_wy", "residual": "gdn_chunk_residual"}[algorithm]
        with patch.dict(os.environ, env or {}, clear=True), patch(
            "actlize_la.gdn_interface.import_module",
            return_value=SimpleNamespace(**{entry: call}),
        ) as importer:
            got = gdn_forward(*self.inputs, algorithm=algorithm, backend=backend, **kwargs)
        self.assertIs(got, outputs)
        self.assertEqual(call.call_count, 1)
        self.assertEqual(call.call_args.args, self.inputs)
        return call, importer

    def test_default_preserves_original_cuda_entry(self):
        call, importer = self.run_call()
        self.assertEqual(call.call_args.kwargs, {"output_final_state": True})
        importer.assert_called_once_with(".gdn_chunk_interface", "actlize_la")

    def test_original_explicit_ppu_keeps_original_auto_algorithm(self):
        call, _ = self.run_call(backend="ppu10", env={"GDN_QSA_PPU_EXTENSION": "/bound/control.so"}, output_final_state=False)
        self.assertEqual(call.call_args.kwargs, {"output_final_state": False})

    def test_complete_wy_and_residual_keep_state_and_private_delivery(self):
        for algorithm, delivery in (("wy", "aiu-state-output"), ("residual", "solve-static")):
            with self.subTest(algorithm=algorithm):
                state = object()
                call, _ = self.run_call(algorithm, "ppu10", initial_state=state,
                                        output_final_state=False, delivery=delivery)
                self.assertEqual(call.call_args.kwargs, dict(initial_state=state,
                    output_final_state=False, delivery=delivery))

    def test_no_candidate_promotion_when_delivery_omitted(self):
        call, _ = self.run_call("residual")
        self.assertEqual(call.call_args.kwargs["delivery"], "scalar")

    def test_unsupported_and_incompatible_targets_fail_before_import(self):
        with patch.dict(os.environ, {}, clear=True), patch("actlize_la.gdn_interface.import_module") as importer:
            for target in ("ppu15", "ppu17", "cuda_sm90"):
                with self.subTest(target=target), self.assertRaisesRegex(RuntimeError, "not implemented|no .* implementation"):
                    gdn_forward(*self.inputs, backend=target)
            with self.assertRaisesRegex(RuntimeError, "no cuda_sm80 implementation"):
                gdn_forward(*self.inputs, algorithm="wy", backend="cuda_sm80")
            importer.assert_not_called()

    def test_unknown_algorithm_is_not_wy_or_original_fallback(self):
        with self.assertRaisesRegex(ValueError, "unknown GDN algorithm"):
            gdn_forward(*self.inputs, algorithm="kda", backend="ppu10")

    def test_initial_state_and_delivery_are_not_silently_dropped(self):
        with self.assertRaisesRegex(ValueError, "must not be dropped"):
            gdn_forward(*self.inputs, initial_state=object())
        with self.assertRaisesRegex(ValueError, "no WY/residual delivery"):
            gdn_forward(*self.inputs, delivery="scalar")

    def test_conflicting_explicit_backend_and_legacy_environment_are_red(self):
        for env, requested in (({}, "ppu10"), ({"GDN_QSA_PPU_EXTENSION": "control.so"}, "cuda_sm80")):
            with patch.dict(os.environ, env, clear=True), self.assertRaisesRegex(RuntimeError, "no fallback"):
                gdn_forward(*self.inputs, backend=requested)

    def test_inventory_is_not_mutable_authority_or_device_admission(self):
        inventory = backend_inventory()
        self.assertFalse(inventory["targets"]["ppu15"]["implemented"])
        inventory["targets"]["ppu15"]["implemented"] = True
        with self.assertRaisesRegex(RuntimeError, "not implemented"):
            require_backend("ppu15")
        self.assertEqual(require_backend("ppu17")["device_admission"],"UNVERIFIED")
        with self.assertRaises(ValueError):
            require_backend("typo")

    def test_math_contracts_remain_distinct_and_bind_existing_interfaces(self):
        from actlize_la.gdn_residual_interface import MATH_CONTRACT, WY_MATH_CONTRACT
        self.assertEqual(require_implementation("residual", "ppu10")["numerical_contract"], MATH_CONTRACT)
        self.assertEqual(require_implementation("wy", "ppu10")["numerical_contract"], WY_MATH_CONTRACT)
        self.assertNotEqual(MATH_CONTRACT, WY_MATH_CONTRACT)
        self.assertEqual(require_implementation("original", "cuda_sm80")["final_state_dtype"], "bf16")
        self.assertEqual(require_implementation("residual", "ppu10")["final_state_dtype"], "fp32")

    def test_inventory_import_has_no_torch_or_native_side_effect(self):
        code = "from actlize_la import backend_inventory; import sys; backend_inventory(); assert 'torch' not in sys.modules; assert not any(k.endswith(('_gdn_chunk','_gdn_chunk_ppu','_gdn_wy_ppu')) for k in sys.modules)"
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class BuildBoundary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = Path("/workspace") / f"gdn-backend-contract-{uuid4().hex}"
        cls.work.mkdir()

    def configure(self, label, *flags):
        return subprocess.run(["cmake", "-S", str(ROOT), "-B", str(self.work / label),
            "-DBUILD_TESTING=OFF", "-DGDN_QSA_BUILD_PPU_TESTS=OFF", *flags],
            capture_output=True, text=True, timeout=60)

    def test_default_configure_imports_no_native_dependency_graph(self):
        result = self.configure("host")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("source graph: none", result.stdout)
        self.assertFalse((self.work / "host/actlize").exists())

    def test_new_architecture_is_not_successful_legacy_build(self):
        for target in ("ppu15", "ppu17"):
            with self.subTest(target=target):
                result = self.configure(target, f"-DGDN_QSA_TARGET={target}")
                self.assertNotEqual(result.returncode, 0)
                self.assertRegex(" ".join(result.stderr.split()),"not implemented|PPU_CUTLASS_ROOT")
                self.assertFalse((self.work / target / "actlize").exists())
        result=self.configure("new-cuda", "-DGDN_QSA_TARGET=cuda_sm90")
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertFalse((self.work/"new-cuda/actlize").exists())

    def test_unknown_target_and_conflicting_legacy_alias_are_red(self):
        result = self.configure("typo", "-DGDN_QSA_TARGET=pp17")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unknown GDN target", result.stderr)
        result = self.configure("conflict", "-DGDN_QSA_TARGET=ppu17", "-DGDN_QSA_ENABLE_PPU=ON")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("legacy PPU1.0 alias", result.stderr)

    def test_build_directory_cannot_change_target(self):
        result = self.configure("reuse")
        self.assertEqual(result.returncode, 0, result.stderr)
        result = self.configure("reuse", "-DGDN_QSA_TARGET=ppu10")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("separate target build directory", " ".join(result.stderr.split()))

    def test_legacy_target_cannot_override_explicit_new_dependency_arch(self):
        for label, flag in (("arch17", "-DACOMPUTE_VERSION=10700"),
                            ("arch15", "-DCUTLASS_PPU_ARCHS=ppu0015")):
            result = self.configure(label, "-DGDN_QSA_TARGET=ppu10", flag)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("conflicts with", result.stderr)
            self.assertFalse((self.work / label / "actlize").exists())

    def test_setuptools_rejects_non_cuda_before_torch_discovery(self):
        for target in ("ppu10", "ppu17", "cuda_sm90", "ppu15", "typo"):
            with self.subTest(target=target):
                result = subprocess.run([sys.executable, "setup.py", "--name"], cwd=ROOT,
                    env=os.environ | {"GDN_QSA_TARGET": target}, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("GDN target", result.stderr)
                self.assertNotIn("CUDA_HOME", result.stderr)

    def test_marking_sm90_implemented_cannot_borrow_sm80_source_graph(self):
        # Same catalog/dispatcher, only the planted declaration changes.
        base = backend_inventory()
        for entry in (None, "csrc/backends/cuda_sm80/build.py"):
            data = copy.deepcopy(base)
            data['targets']['cuda_sm90']['implemented'] = True
            data['targets']['cuda_sm90']['build'] = 'setuptools'
            data['targets']['cuda_sm90'].pop('build_entry',None)
            if entry:
                data['targets']['cuda_sm90']['build_entry'] = entry
            with patch.dict(os.environ, {'GDN_QSA_TARGET': 'cuda_sm90'}, clear=True), patch(
                'json.loads', return_value=data
            ), self.assertRaisesRegex(RuntimeError, 'no registered build entry|SM80 builder rejects'):
                runpy.run_path(str(ROOT / 'setup.py'))

    def test_cuda_setup_enters_exact_sm80_builder_and_keeps_sources(self):
        extension = Mock(side_effect=lambda name, sources, **kwargs: dict(name=name, sources=sources, **kwargs))
        fake = SimpleNamespace(BuildExtension=Mock(), CUDAExtension=extension)
        with patch.dict(os.environ, {'GDN_QSA_TARGET': 'cuda_sm80', 'GDN_QSA_BUILD_GDN_ONLY': '1'}, clear=True), patch.dict(
            sys.modules, {'torch.utils.cpp_extension': fake}
        ), patch('setuptools.setup') as setup:
            runpy.run_path(str(ROOT / 'setup.py'))
        self.assertEqual(extension.call_count, 1)
        result = setup.call_args.kwargs['ext_modules'][0]
        self.assertEqual(result['name'], 'actlize_la._gdn_chunk')
        self.assertEqual({Path(p).name for p in result['sources']}, {
            'gdn_kernel.cu', 'gdn_scan_stage1.cu', 'gdn_scan_stage1_reset.cu',
            'gdn_scan_stage2.cu', 'gdn_scan_stage2_blelloch.cu',
            'gdn_scan_stage3_reset.cu', 'gdn_ops.cu'})
        self.assertIn('-arch=sm_80', result['extra_compile_args']['nvcc'])
        self.assertEqual(result['include_dirs'], [str(ROOT / 'third_party/cutlass/include')])

    def test_marking_ppu17_implemented_cannot_configure_an_empty_or_legacy_graph(self):
        for entry in (None, 'cmake/backends/ppu_aiu.cmake'):
            directory = self.work / ('false-ppu17-legacy' if entry else 'false-ppu17-empty')
            (directory / 'cmake/backends').mkdir(parents=True)
            (directory / 'actlize_la/backends').mkdir(parents=True)
            data = backend_inventory()
            data['targets']['ppu17']['implemented'] = True
            data['targets']['ppu17'].pop('build_entry',None)
            if entry:
                data['targets']['ppu17']['build_entry'] = entry
            # Execute the real selector/leaf, not a parallel model of them.
            for name in ('cmake/GdnBackend.cmake', 'cmake/backends/ppu_aiu.cmake'):
                (directory / name).write_bytes((ROOT / name).read_bytes())
            (directory / 'actlize_la/backends/targets.json').write_text(json.dumps(data))
            (directory / 'CMakeLists.txt').write_text(
                'cmake_minimum_required(VERSION 3.19)\nproject(negative NONE)\n'
                'include(cmake/GdnBackend.cmake)\ninclude("${GDN_QSA_BACKEND_ENTRY}")\n')
            result = subprocess.run(['cmake', '-S', str(directory), '-B', str(directory / 'build'),
                '-DGDN_QSA_TARGET=ppu17'], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertRegex(' '.join(result.stderr.split()),
                'no registered build entry|PPU1.0 source graph rejects')

    def test_legacy_header_rejects_hopper_or_conflicting_macros(self):
        for defines, expected in (([], True), (["GDN_QSA_PPU=1"], True),
            (["GDN_QSA_TARGET_CUDA_SM90=1"], False),
            (["GDN_QSA_TARGET_PPU17=1", "GDN_QSA_PPU=1"], False),
            (["ACOMPUTE_VERSION=10700", "GDN_QSA_PPU=1"], False),
            (["GDN_QSA_PPU=1", "GDN_QSA_TARGET_CUDA_SM80=1"], False),
            (["GDN_QSA_TARGET_PPU10=1"], False)):
            with self.subTest(defines=defines):
                result = subprocess.run(["c++", "-E", "-x", "c++",
                    *(f"-D{x}" for x in defines), "-"], input=
                    f'#include "{ROOT}/include/gdn_qsa/backend_target.h"\n',
                    capture_output=True, text=True)
                self.assertEqual(result.returncode == 0, expected, result.stderr)
                if not expected:
                    self.assertIn("error:", result.stderr)

    def test_private_primitives_do_not_import_each_other(self):
        cuda = (ROOT / "csrc/backends/cuda_sm80/primitives.cuh").read_text()
        ppu = (ROOT / "csrc/backends/ppu_aiu/primitives.cuh").read_text()
        self.assertNotIn("hggc_runtime", cuda)
        self.assertNotIn("gdn_qsa/ppu/shared_copy", cuda)
        self.assertNotIn("cute/arch/mma_sm80", ppu)
        self.assertNotIn("cute/atom/mma_traits_sm90_gmma", ppu)
        # Every device object tracks the moved header, not just its former shim.
        cmake = (ROOT / "cmake/backends/ppu_aiu.cmake").read_text()
        self.assertEqual(cmake.count("csrc/backends/ppu_aiu/primitives.cuh"), 2)
        self.assertEqual(cmake.count("include/gdn_qsa/backend_target.h"), 2)


class LoaderBoundary(unittest.TestCase):
    def tearDown(self):
        from actlize_la.backends.loading import clear_backend_cache
        clear_backend_cache()

    def test_cached_cuda_does_not_hide_a_later_explicit_ppu_error(self):
        from actlize_la.backends.loading import load_original
        sentinel = object()
        with patch.dict(os.environ, {}, clear=True), patch(
            "actlize_la.backends.loading.import_module", return_value=sentinel
        ) as importer:
            self.assertIs(load_original(), sentinel)
            self.assertIs(load_original(), sentinel)
            importer.assert_called_once()
            os.environ["GDN_QSA_PPU_EXTENSION"] = ""
            with self.assertRaisesRegex(RuntimeError, "GDN_QSA_PPU_EXTENSION"):
                load_original()

    def test_wy_requires_explicit_file_and_never_uses_original_loader(self):
        from actlize_la.backends.loading import load_wy
        for env in ({}, {"GDN_QSA_WY_EXTENSION": ""}, {"GDN_QSA_WY_EXTENSION": str(ROOT)}):
            with patch.dict(os.environ, env, clear=True), patch(
                "actlize_la.backends.loading.import_module"
            ) as importer, self.assertRaisesRegex(RuntimeError, "GDN_QSA_WY_EXTENSION"):
                load_wy()
            importer.assert_not_called()

    def test_cache_identity_includes_module_and_path(self):
        from actlize_la.backends.loading import _load
        modules = [object(), object(), object()]
        with patch("actlize_la.backends.loading.util.spec_from_file_location") as spec, patch(
            "actlize_la.backends.loading.util.module_from_spec", side_effect=modules
        ) as create:
            self.assertIs(_load("_gdn_chunk_ppu", "/first.so"), modules[0])
            self.assertIs(_load("_gdn_chunk_ppu", "/first.so"), modules[0])
            self.assertIs(_load("_gdn_chunk_ppu", "/second.so"), modules[1])
            self.assertIs(_load("_gdn_wy_ppu", "/second.so"), modules[2])
            self.assertEqual(create.call_count, 3)
            self.assertEqual(spec.return_value.loader.exec_module.call_count, 3)

    def test_file_validation_is_not_repeated_on_the_hot_path(self):
        from actlize_la.backends.loading import _explicit_path
        with patch.dict(os.environ, {"GDN_QSA_WY_EXTENSION": str(ROOT / "setup.py")}), patch(
            "actlize_la.backends.loading.Path.is_file", return_value=True
        ) as check:
            self.assertEqual(_explicit_path("GDN_QSA_WY_EXTENSION"), str(ROOT / "setup.py"))
            self.assertEqual(_explicit_path("GDN_QSA_WY_EXTENSION"), str(ROOT / "setup.py"))
            check.assert_called_once()


if __name__ == "__main__":
    unittest.main()
