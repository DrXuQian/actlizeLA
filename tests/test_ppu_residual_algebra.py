#!/usr/bin/env python3
"""Reassociation proof + explicit BF16-boundary evidence + semantic negatives."""
import torch
from residual_cpu import residual_forward
from wy_cpu import recurrent_double, wy_forward
from test_ppu_gdn_backend import fixture, assert_pair


def main():
    torch.set_num_threads(1)
    cases, changed = 0, 0
    maximum = [0., 0.]
    for length in (1, 16, 63, 64, 65, 129, 2048):
        for gate in (0., -0.001, -0.1, -1.):
            inputs = fixture(1, length, 2, 4, gate)
            for nonzero in (False, True):
                initial = (torch.randn(1, 4, 128, 128, generator=torch.Generator().manual_seed(98)) * .005
                           if nonzero else None)
                want = recurrent_double(inputs, initial)
                exact = residual_forward(inputs, initial, rounded=False)
                for actual, expected in zip(exact, want):
                    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-11)
                got = residual_forward(inputs, initial)
                errors = assert_pair(got, want)
                maximum = [max(a, b) for a, b in zip(maximum, errors)]
                old = wy_forward(inputs, initial)
                changed += any(not torch.equal(a, b) for a, b in zip(got, old))
                print(f"[residual algebra] S={length} g={gate} initial={nonzero} errors={errors} PASS", flush=True)
                cases += 1
    for gate in (-0.1, -1.):
        for hv in (32, 64):
            inputs = fixture(1, 2048, 16, hv, gate)
            errs = assert_pair(residual_forward(inputs), recurrent_double(inputs))
            print(f"[residual target] B1/S2048/Hk16/Hv{hv} g={gate} errors={errs} PASS", flush=True)
            maximum = [max(a, b) for a, b in zip(maximum, errs)]
            cases += 1
    # Normalized K, varying beta, nonuniform gates and nonzero H expose the
    # noncommuting P/B and cross-chunk memory. No constant-beta shortcut.
    inputs = list(fixture(1, 129, 2, 4, -.01))
    inputs[1] = torch.nn.functional.normalize(inputs[1].float(), dim=-1).bfloat16()
    inputs[3] = torch.linspace(-.06, -.001, 129)[None, :, None].expand(1, 129, 4).clone()
    inputs[4] = torch.linspace(.05, .95, 129)[None, :, None].expand(1, 129, 4).bfloat16().contiguous()
    initial = torch.randn(1, 4, 128, 128, generator=torch.Generator().manual_seed(81)) * .02
    want = recurrent_double(inputs, initial)
    errs = assert_pair(residual_forward(inputs, initial), want)
    print(f"[residual stress] normalized-K/variable-beta/fp32-gate/nonzero-H errors={errs} PASS")
    for plant in ("reset", "gate-omitted", "beta-outside-inverse", "inverse-identity"):
        try:
            assert_pair(residual_forward(inputs, initial, plant=plant), want)
        except AssertionError:
            print(f"[residual algebra negative] {plant} EXPECTED-RED/PASS")
        else:
            raise AssertionError(f"escaped semantic negative: {plant}")
    if cases != 60 or not changed:
        raise AssertionError(f"coverage/rounding contract lost: cases={cases}, differing={changed}")
    print(f"[residual algebra] PASS cases={cases+1} changed-vs-WY={changed}/56 max_errors={maximum} "
          "old-RAW-BIT-gate=UNCHANGED new-association=EXPLICIT device=NOT_RUN")


if __name__ == "__main__":
    main()
