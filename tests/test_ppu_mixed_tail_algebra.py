"""Reuse the admitted residual algebra to prove full/tail FP32 state continuity."""
import torch
from residual_cpu import residual_forward
from test_ppu_first_chunk_algebra import identical
from test_ppu_gdn_backend import fixture, assert_pair
from wy_cpu import recurrent_double


def split_at_full(inputs, initial=None, plant=None):
    length = inputs[0].shape[1]
    full = length // 64 * 64
    if not full or full == length:
        return residual_forward(inputs, initial)
    prefix = tuple(t[:, :full].contiguous() for t in inputs)
    tail = tuple(t[:, full:].contiguous() for t in inputs)
    first, carried = residual_forward(prefix, initial)
    if plant == "reset":
        carried.zero_()
    elif plant == "round-carried-state":
        carried = carried.bfloat16().float()
    last, final = residual_forward(tail, carried)
    if plant == "missing-tail":
        last.zero_()
    return torch.cat((first, last), dim=1), final


def main():
    torch.set_num_threads(1)
    cases = 0
    for length in [*range(65, 128), 1, 64, 128, 2048, 2049, 2111]:
        for gate in (-.1, -1.):
            for nonzero in (False, True):
                inputs = fixture(1, length, 1, 2, gate)
                initial = torch.full((1, 2, 128, 128), .0078125) if nonzero else None
                want = residual_forward(inputs, initial)
                got = split_at_full(inputs, initial)
                if not identical(got, want):
                    raise AssertionError(f"mixed-tail changed residual bits {length}/{gate}/{nonzero}")
                assert_pair(got, recurrent_double(inputs, initial))
                cases += 1
    inputs = fixture(1, 129, 1, 2, -.001)
    initial = torch.full((1, 2, 128, 128), .03)
    want = residual_forward(inputs, initial)
    for plant in ("reset", "round-carried-state", "missing-tail"):
        if identical(split_at_full(inputs, initial, plant), want):
            raise AssertionError("mixed-tail negative escaped: " + plant)
        print("[mixed tail algebra negative]", plant, "EXPECTED-RED/PASS")
    if cases != 276:
        raise AssertionError("mixed-tail algebra denominator changed")
    print(f"[mixed tail algebra] PASS cases={cases} all63tails+fallbacks+long negatives=3 RAW-BIT+independent-oracle device=NOT_RUN")


if __name__ == "__main__":
    main()
