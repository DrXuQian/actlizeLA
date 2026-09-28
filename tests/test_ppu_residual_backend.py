#!/usr/bin/env python3
"""New-association admission. Old WY RAW-BIT delivery admission is separate."""
import argparse
import os
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from actlize_la import gdn_chunk_residual
from actlize_la.gdn_residual_interface import RESIDUAL_ENTRYPOINTS, MATH_CONTRACT
from test_ppu_gdn_backend import fixture, digest, assert_pair, print_failure
from test_ppu_wy_backend import oracle


def admit(shape, gate, nonzero, *, stress=False, deliveries=()):
    cpu = list(fixture(*shape, gate))
    if nonzero:
        cpu[3] = cpu[3].float()
    if stress:
        cpu[1] = torch.nn.functional.normalize(cpu[1].float(), dim=-1).bfloat16()
        cpu[3] = torch.linspace(-.06, -.001, shape[1])[None, :, None].expand(shape[0], shape[1], shape[3]).clone()
        cpu[4] = torch.linspace(.05, .95, shape[1])[None, :, None].expand(shape[0], shape[1], shape[3]).bfloat16().contiguous()
    initial = (torch.randn(shape[0], shape[3], 128, 128, generator=torch.Generator().manual_seed(18)) * .005
               if nonzero else None)
    want = oracle(cpu, initial)
    inputs = tuple(t.cuda() for t in cpu)
    state = initial.cuda() if initial is not None else None
    got = gdn_chunk_residual(*inputs, initial_state=state)
    torch.cuda.synchronize()
    try:
        errs = assert_pair(got, want)
    except AssertionError:
        print_failure(f"residual/{shape}/{gate}/{nonzero}", got, want)
        raise
    if got[0].dtype != torch.bfloat16 or got[1].dtype != torch.float32:
        raise AssertionError("residual output/state precision changed")
    fingerprint = digest(got)
    for delivery in deliveries:
        for _ in range(8):
            candidate = gdn_chunk_residual(*inputs, initial_state=state, delivery=delivery)
            assert_pair(candidate, want)
            if not all(torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a,b in zip(candidate,got)):
                raise AssertionError(f"residual {delivery} is not RAW-BIT equal to residual control")
        expanded_inputs = (inputs[0].repeat_interleave(shape[3]//shape[2],2),
                           inputs[1].repeat_interleave(shape[3]//shape[2],2), *inputs[2:])
        expanded_candidate = gdn_chunk_residual(*expanded_inputs, initial_state=state, delivery=delivery)
        if digest(expanded_candidate) != fingerprint:
            raise AssertionError(f"{delivery} GVA differs from control")
        output, absent = gdn_chunk_residual(*inputs, initial_state=state, delivery=delivery, output_final_state=False)
        if absent is not None or not torch.equal(output.view(torch.uint8),got[0].view(torch.uint8)):
            raise AssertionError(f"{delivery} output-only differs from control")
    for _ in range(7):
        repeat = gdn_chunk_residual(*inputs, initial_state=state)
        assert_pair(repeat, want)
        if digest(repeat) != fingerprint:
            raise AssertionError("residual replay is not RAW-BIT stable")
    q, k, v, g, beta = inputs
    ratio = shape[3] // shape[2]
    expanded = gdn_chunk_residual(q.repeat_interleave(ratio, 2), k.repeat_interleave(ratio, 2),
                                 v, g, beta, initial_state=state)
    if digest(expanded) != fingerprint:
        raise AssertionError("residual GVA head mapping differs from expansion")
    out, absent = gdn_chunk_residual(*inputs, initial_state=state, output_final_state=False)
    if absent is not None or not torch.equal(out, got[0]):
        raise AssertionError("residual output-only contract differs")
    if digest(inputs) != digest(cpu) or (state is not None and not torch.equal(state.cpu(), initial)):
        raise AssertionError("residual modified an input")
    for index in (0, 1):
        wrong = list(got)
        wrong[index] = torch.zeros_like(wrong[index])
        try:
            assert_pair(wrong, want)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"zero-{index} negative escaped")
    print(f"[residual device] shape={shape} g={gate} initial={nonzero} stress={stress} "
          f"errors={errs} contract={MATH_CONTRACT} fingerprint={fingerprint} repeat=8/8 "
          "GVA/output-only=RAW-BIT old-WY-equality=NOT_REQUIRED_NUMERICALLY_REASSOCIATED NUMERIC/PASS", flush=True)
    for delivery in deliveries:
        print(f"[residual delivery device] delivery={delivery} shape={shape} g={gate} initial={nonzero} stress={stress} "
              "control=residual byte-equal=PASS repeat=8/8 GVA/output-only=RAW-BIT/PASS", flush=True)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--extension", type=Path, required=True)
    p.add_argument("--deliveries", nargs="+", choices=tuple(k for k in RESIDUAL_ENTRYPOINTS if k != "scalar"), default=[])
    args = p.parse_args()
    if not args.extension.is_file():
        p.error("residual extension missing")
    os.environ["GDN_QSA_WY_EXTENSION"] = str(args.extension.resolve())
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    if "PPU" not in torch.cuda.get_device_properties(0).name.upper():
        raise RuntimeError("residual gate requires a PPU")
    count = 0
    for length in (1, 16, 63, 64, 65, 129):
        for gate in (-.1, -1.):
            for state in (False, True):
                admit((2, length, 1, 2), gate, state, deliveries=args.deliveries)
                count += 1
    for gate in (0., -.001, -.1, -1.):
        admit((1, 2048, 16, 32), gate, False, deliveries=args.deliveries)
        count += 1
    admit((1, 2048, 16, 64), -.1, False, deliveries=args.deliveries)
    admit((1, 129, 2, 4), -.01, True, stress=True, deliveries=args.deliveries)
    count += 2
    if count != 30:
        raise AssertionError("residual device coverage denominator changed")
    print(f"[residual device] PASS cases={count} repeats=8 original-2%-gate=UNCHANGED routing=UNCHANGED")


if __name__ == "__main__":
    main()
