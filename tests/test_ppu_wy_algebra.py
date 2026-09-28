#!/usr/bin/env python3
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wy_cpu import wy_forward, recurrent_double, block_inverse
from test_ppu_gdn_backend import fixture, assert_pair, MAX_RELATIVE_ERROR


def main():
    torch.set_num_threads(1)
    count = 0
    for length in (1, 16, 63, 64, 65, 129, 2048):
        for gate in (0., -0.1, -1.):
            inputs = fixture(1, length, 2, 4, gate)
            if length == 129:
                inputs = (*inputs[:3], torch.linspace(-0.4, 0., length)[None, :, None].expand(1, length, 4).clone(), inputs[4])
            initial = torch.randn(1, 4, 128, 128, generator=torch.Generator().manual_seed(98)) * 0.005
            for state in (None, initial):
                want = recurrent_double(inputs, state)
                exact = wy_forward(inputs, state, rounded=False)
                for a, b in zip(exact, want):
                    torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-11)
                rounded = wy_forward(inputs, state)
                errors = assert_pair(rounded, want)
                count += 1
                print(f"[WY algebra] S={length} g={gate} initial={state is not None} errors={errors} PASS")
    # The actual target (not only a small-head proxy), both 35B and 122B heads.
    for hv in (32, 64):
        inputs = fixture(1, 2048, 16, hv, -0.1)
        got, want = wy_forward(inputs), recurrent_double(inputs)
        print(f"[WY target] B1/S2048/Hk16/Hv{hv} errors={assert_pair(got, want)} PASS")
        count += 1
    inputs = fixture(1, 129, 2, 4, -0.01)
    # Larger normalized keys make inverse/sign and causal negatives observable.
    q, k, v, g, beta = inputs
    k = torch.nn.functional.normalize(k.float(), dim=-1).to(k.dtype)
    inputs = q, k, v, g, beta
    want = recurrent_double(inputs)
    print(f"[WY normalized keys] errors={assert_pair(wy_forward(inputs), want)} PASS")
    for plant in ("inverse-sign", "gva", "reset", "causal", "snapshot-after-update"):
        wrong = wy_forward(inputs, plant=plant)
        try:
            assert_pair(wrong, want)
        except AssertionError:
            print(f"[WY negative] plant={plant} EXPECTED-RED/PASS")
        else:
            raise AssertionError(f"escaped negative: {plant}")
    gen = torch.Generator().manual_seed(18)
    lower = torch.tril(torch.randn(3, 64, 64, generator=gen, dtype=torch.float64) * .02, -1)
    torch.testing.assert_close(block_inverse(lower, False) @ (torch.eye(64) + lower),
                               torch.eye(64).expand(3, 64, 64).double(), rtol=1e-12, atol=1e-12)
    print(f"[WY algebra] PASS cases={count + 1} tolerance={MAX_RELATIVE_ERROR} device_execution=NOT_RUN")


if __name__ == "__main__":
    main()
