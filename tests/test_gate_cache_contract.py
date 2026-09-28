"""CPU-only binding/runner admission tests; no PPU device certificate."""
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace
from actlize_la import gdn_residual_interface as api

ROOT=Path(__file__).resolve().parents[1]


class GateCacheContract(unittest.TestCase):
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
