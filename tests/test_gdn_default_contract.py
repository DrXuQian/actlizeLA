"""Default-admission denominator and non-symmetric contract negative controls."""
import unittest
import torch

from test_ppu_default_device import cases
from test_ppu_gdn_backend import fixture, assert_pair
from test_ppu_wy_backend import oracle


def check_denominator(rows):
    expected = {(t, hv, g) for t in (1, 63, 64, 65, 127, 128, 129, 2048, 8192)
                for hv in (32, 48) for g in (-.1, -1.)}
    if len(rows) != 36 or len(set(rows)) != 36 or set(rows) != expected:
        raise AssertionError("device admission is missing/duplicating a required geometry")


class DefaultContract(unittest.TestCase):
    def test_exact_device_denominator(self):
        check_denominator(cases())

    def test_missing_or_duplicate_geometry_is_red(self):
        rows = cases()
        for wrong in (rows[:-1], rows[:-1] + rows[:1]):
            with self.assertRaisesRegex(AssertionError, "missing/duplicating"):
                check_denominator(wrong)

    def test_state_transpose_double_scale_and_head_mapping_negatives(self):
        torch.set_num_threads(1)
        for hv in (32, 48):
            for gate in (-.1, -1.):
                inputs = fixture(1, 65, 16, hv, gate)
                state = torch.randn(1, hv, 128, 128, generator=torch.Generator().manual_seed(7241)) * .01
                want = oracle(inputs, state)
                wrong_heads = (inputs[0].roll(1, 2), inputs[1].roll(1, 2), *inputs[2:])
                double_scale = (inputs[0] / (128 ** .5), *inputs[1:])
                for plant, bad_inputs, bad_state in (
                    ("state-transpose", inputs, state.transpose(-1, -2).contiguous()),
                    ("query-scale-twice", double_scale, state),
                    ("wrong-key-head", wrong_heads, state),
                ):
                    with self.subTest(hv=hv, gate=gate, plant=plant):
                        with self.assertRaisesRegex(AssertionError, "2% criterion"):
                            assert_pair(oracle(bad_inputs, bad_state), want)


if __name__ == "__main__":
    unittest.main()
