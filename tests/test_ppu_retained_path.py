"""CPU API contracts for the retained PPU path, not device numerics."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from actlize_la import gdn_forward
from actlize_la import gdn_residual_interface as residual


def tensor():
    result = Mock()
    result.contiguous.return_value = result
    return result


class RetainedPpuPath(unittest.TestCase):
    def setUp(self):
        self.inputs = tuple(tensor() for _ in range(5))
        self.state = tensor()
        self.native = SimpleNamespace(
            residual=Mock(return_value=("scalar-output", "scalar-state")),
            residual_gate_cache_solve_static=Mock(return_value=("tail-output", "tail-state")),
            residual_full_chunk=Mock(return_value=("output", "state")),
        )

    def test_unified_entry_passes_state_and_selects_retained_implementation(self):
        with patch.object(residual, "_backend", return_value=self.native):
            result = gdn_forward(*self.inputs, initial_state=self.state,
                                 backend="ppu10", algorithm="residual", delivery="full-chunk")
        self.assertEqual(result, ("output", "state"))
        self.native.residual_full_chunk.assert_called_once_with(*self.inputs, self.state, True)
        self.native.residual.assert_not_called()
        self.native.residual_gate_cache_solve_static.assert_not_called()

    def test_output_only_preserves_native_flag_and_hides_final_state(self):
        with patch.object(residual, "_backend", return_value=self.native):
            result = gdn_forward(*self.inputs, backend="ppu10", algorithm="residual",
                                 delivery="full-chunk", output_final_state=False)
        self.assertEqual(result, ("output", None))
        self.native.residual_full_chunk.assert_called_once_with(*self.inputs, None, False)

    def test_existing_residual_default_is_unchanged(self):
        with patch.object(residual, "_backend", return_value=self.native):
            result = gdn_forward(*self.inputs, backend="ppu10", algorithm="residual")
        self.assertEqual(result, ("scalar-output", "scalar-state"))
        self.native.residual.assert_called_once_with(*self.inputs, None, True)
        self.native.residual_full_chunk.assert_not_called()

    def test_missing_new_symbol_is_an_error_not_silent_fallback(self):
        for delivery, symbol in (("full-chunk", "residual_full_chunk"),
                                 ("gate-cache-solve-static", "residual_gate_cache_solve_static")):
            with self.subTest(delivery=delivery):
                with patch.object(residual, "_backend", return_value=SimpleNamespace(residual=Mock())):
                    with self.assertRaisesRegex(AttributeError, symbol):
                        gdn_forward(*self.inputs, backend="ppu10", algorithm="residual", delivery=delivery)

    def test_unselected_geometry_is_not_exposed_on_main(self):
        with patch.object(residual, "_backend") as loader:
            with self.assertRaisesRegex(ValueError, "unknown residual delivery"):
                gdn_forward(*self.inputs, backend="ppu10", algorithm="residual", delivery="geometry-v16-w4")
            loader.assert_not_called()

    def test_retained_identity_is_residual_not_materialized_wy(self):
        self.assertEqual(residual.RESIDUAL_ENTRYPOINTS["full-chunk"], "residual_full_chunk")
        self.assertEqual(residual.math_contract("residual-full-chunk"), residual.MATH_CONTRACT)
        self.assertNotEqual(residual.MATH_CONTRACT, residual.WY_MATH_CONTRACT)


if __name__ == "__main__":
    unittest.main()
