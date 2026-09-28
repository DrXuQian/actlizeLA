"""Reuse the admitted rounded residual algebra, not a new parallel GEMM model."""
import torch
from residual_cpu import residual_forward
from test_ppu_gdn_backend import fixture, assert_pair
from wy_cpu import recurrent_double


def identical(a, b):
    return all(torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8))
               for x, y in zip(a, b))


def main():
    torch.set_num_threads(1)
    cases = 0
    policy = lambda chunk, initial: initial or chunk != 0
    # Extra tails exercise the mathematical identity independently; public
    # first-chunk dispatch deliberately leaves those on the old device body.
    for length in (1, 63, 64, 65, 128, 129, 2048):
        for gate in (0., -.001, -.1, -1.):
            for nonzero in (False, True):
                inputs = list(fixture(1, length, 1, 2, gate))
                initial = torch.full((1, 2, 128, 128), .0078125) if nonzero else None
                want = residual_forward(inputs, initial)
                got = residual_forward(inputs, initial, history_policy=policy)
                if not identical(got, want):
                    raise AssertionError(f"first-chunk identity changed bits: {length}/{gate}/{nonzero}")
                assert_pair(got, recurrent_double(inputs, initial))
                cases += 1
    # Distinct signed zeros plus finite signed values. Multiplication by known
    # H0 must produce the +0 accumulator identity without changing V's sign.
    for length in (64, 128):
        inputs = list(fixture(1, length, 1, 2, -.1))
        flat = inputs[2].view(-1)
        flat[0::7] = 0.
        flat[1::7] = -0.
        if not identical(residual_forward(inputs), residual_forward(inputs, history_policy=policy)):
            raise AssertionError("first-chunk signed-zero mismatch")
        cases += 1
    inputs = fixture(1, 128, 1, 2, -.001)
    initial = torch.full((1, 2, 128, 128), .25)
    negatives = (("ignore-supplied-state", initial, lambda ct, init: ct != 0),
                 ("skip-second-chunk", None, lambda ct, init: ct > 1),
                 ("reset-every-chunk", None, lambda ct, init: False))
    for label, state, wrong in negatives:
        want = residual_forward(inputs, state)
        if identical(residual_forward(inputs, state, history_policy=wrong), want):
            raise AssertionError("negative escaped: " + label)
        print("[first chunk algebra negative]", label, "EXPECTED-RED/PASS", flush=True)
    if cases != 58:
        raise AssertionError("first-chunk algebra denominator changed")
    print("[first chunk algebra] PASS cases=58 negatives=3 RAW-BIT+independent-oracle device=NOT_RUN")


if __name__ == "__main__":
    main()
