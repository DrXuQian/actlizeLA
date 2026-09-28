"""CPU-only admission for FLA A/B wiring; no PPU or FLA import needed."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "ppu_fla_bench", ROOT / "benchmarks/bench_ppu_gdn_fla.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class FLAContract(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.inputs = bench.admission.fixture(1, 5, 1, 2, -0.1)

    def test_native_same_inputs_and_math(self):
        calls = []

        def modern(q, k, v, g, beta, scale=None, initial_state=None,
                   output_final_state=False, use_qk_l2norm_in_kernel=False,
                   state_v_first=False, **kwargs):
            calls.append((q, k, v, g, beta))
            self.assertEqual(scale, 128 ** -0.5)
            self.assertIsNone(initial_state)
            self.assertTrue(output_final_state)
            self.assertFalse(use_qk_l2norm_in_kernel)
            self.assertFalse(state_v_first)
            self.assertNotIn("head_first", kwargs)
            self.assertEqual(kwargs["chunk_size"], 64)
            return bench.admission.reference((q, k, v, g, beta))

        result = bench.fla_call(modern, self.inputs, "native")()
        self.assertTrue(all(a is b for a, b in zip(calls[0], self.inputs)))
        self.assertEqual(bench.checked_pair(result, bench.admission.reference(self.inputs)), [0, 0])

    def test_expansion_once_before_timing_old_head_first(self):
        calls = []

        def legacy(q, k, v, g, beta, head_first=True, **kwargs):
            self.assertFalse(head_first)
            self.assertTrue(kwargs["output_final_state"])
            self.assertEqual(q.shape[2], 2)
            self.assertTrue(torch.equal(q, self.inputs[0].repeat_interleave(2, 2)))
            self.assertTrue(torch.equal(k, self.inputs[1].repeat_interleave(2, 2)))
            calls.append((q, k))

        call = bench.fla_call(legacy, self.inputs, "expanded")
        self.assertEqual(calls, [])
        call()
        call()
        self.assertIs(calls[0][0], calls[1][0])
        self.assertIs(calls[0][1], calls[1][1])

    def test_wrong_output_state_and_missing_state_are_red(self):
        want = bench.admission.reference(self.inputs)
        plants = [(torch.zeros_like(want[0]), want[1]),
                  (want[0], torch.zeros_like(want[1])),
                  (want[0], None), (want[0],),
                  (want[0], want[1][..., :64]),
                  (torch.full_like(want[0], float("nan")), want[1])]
        for plant in plants:
            with self.subTest(shapes=[None if x is None else tuple(x.shape) for x in plant]):
                with self.assertRaises(AssertionError):
                    bench.checked_pair(plant, want)

    def test_interleaved_order_and_both_verdicts(self):
        self.assertEqual(bench.sample_order(0), ("ours", "fla"))
        self.assertEqual(bench.sample_order(1), ("fla", "ours"))
        self.assertEqual(bench.verdict([1, 2], [3, 4]), "OURS-WINS")
        self.assertEqual(bench.verdict([3, 4], [1, 2]), "FLA-WINS")
        self.assertEqual(bench.verdict([1, 3], [2, 4]), "UNRESOLVED")
        self.assertEqual(bench.verdict([1, 2], [2, 3]), "UNRESOLVED")
        for invalid in ([], [0], [-1], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                bench.verdict(invalid, [1, 2])

    def test_unavailable_fla_never_becomes_ours_only_pass(self):
        import builtins
        original_import = builtins.__import__

        for missing in ("fla", "triton"):
            def unavailable(name, *args, **kwargs):
                if name == missing:
                    raise ModuleNotFoundError(f"planted missing {missing}")
                if name == "triton":
                    return SimpleNamespace(__version__="contract-stub")
                if name == "triton.backends.nvidia":
                    return SimpleNamespace(compiler=SimpleNamespace(ptx_get_version=lambda _: 90))
                return original_import(name, *args, **kwargs)

            with self.subTest(missing=missing):
                with patch.object(builtins, "__import__", side_effect=unavailable):
                    with self.assertRaisesRegex(RuntimeError, "no comparison or speedup is valid"):
                        bench.load_fla()

    def test_cuda13_parser_backport_is_narrow_and_idempotent(self):
        def old(version):
            if version == "12.9":
                return 88
            raise RuntimeError("Triton only support CUDA 10.0 or higher, but got CUDA version: " + version)

        compiler = SimpleNamespace(ptx_get_version=old)
        self.assertEqual(bench.cuda13_ptx_compat(compiler)["status"],
                         "process-local-upstream-backport")
        self.assertEqual(compiler.ptx_get_version("13.0"), 90)
        self.assertEqual(compiler.ptx_get_version("12.9"), 88)
        for unknown in ("9.0", "13.1", "14.0"):
            with self.assertRaisesRegex(RuntimeError, "got CUDA version: " + unknown):
                compiler.ptx_get_version(unknown)
        patched = compiler.ptx_get_version
        bench.cuda13_ptx_compat(compiler)
        self.assertIs(compiler.ptx_get_version, patched)

    def test_existing_vendor_mapping_is_not_overwritten(self):
        # Even a vendor-specific PTX cap is not ours to change.
        native = lambda version: 86
        compiler = SimpleNamespace(ptx_get_version=native)
        self.assertEqual(bench.cuda13_ptx_compat(compiler)["status"],
                         "native-mapping-unchanged")
        self.assertIs(compiler.ptx_get_version, native)

    def test_unrelated_compiler_failure_is_not_swallowed(self):
        def broken(version):
            raise RuntimeError("planted missing assembler")

        compiler = SimpleNamespace(ptx_get_version=broken)
        with self.assertRaisesRegex(RuntimeError, "planted missing assembler"):
            bench.cuda13_ptx_compat(compiler)
        self.assertIs(compiler.ptx_get_version, broken)


if __name__ == "__main__":
    unittest.main()
