#!/usr/bin/env python3
"""Extra zero-sign/explicit-zero admission, alongside the complete residual gate."""
import argparse
import os
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from actlize_la import gdn_chunk_residual
from test_ppu_gdn_backend import fixture, assert_pair, digest
from test_ppu_wy_backend import oracle


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--extension", type=Path, required=True)
    args = p.parse_args()
    if not args.extension.is_file():
        p.error("first-chunk extension missing")
    os.environ["GDN_QSA_WY_EXTENSION"] = str(args.extension.resolve())
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    if "PPU" not in torch.cuda.get_device_properties(0).name.upper():
        raise RuntimeError("first-chunk admission requires a PPU")
    cases = 0
    for length in (64, 65, 128, 2048):
        for gate in (-.1, -1.):
            cpu = list(fixture(1, length, 1, 2, gate))
            for tensor in cpu[:3]:
                tensor.view(-1)[0::13] = 0.
                tensor.view(-1)[1::13] = -0.
            inputs = tuple(t.cuda() for t in cpu)
            fingerprints = []
            for explicit in (False, True):
                initial = torch.zeros(1, 2, 128, 128) if explicit else None
                state = initial.cuda() if explicit else None
                want = oracle(cpu, initial)
                control = gdn_chunk_residual(*inputs, initial_state=state, delivery="full-chunk")
                assert_pair(control, want)
                fingerprint = digest(control)
                for _ in range(8):
                    got = gdn_chunk_residual(*inputs, initial_state=state, delivery="first-chunk")
                    assert_pair(got, want)
                    if digest(got) != fingerprint:
                        raise AssertionError("first-chunk zero-sign/explicit-state RAW-BIT mismatch")
                if digest(inputs) != digest(cpu) or (explicit and not torch.equal(state.cpu(), initial)):
                    raise AssertionError("first-chunk mutated inputs")
                fingerprints.append(fingerprint)
                cases += 1
                print(f"[first chunk edge] S={length} g={gate} explicit_zero={explicit} "
                      f"signed_zero=1 fingerprint={fingerprint} repeat=8/8 RAW-BIT+NUMERIC/PASS", flush=True)
            if len(set(fingerprints)) != 1:
                raise AssertionError("None and explicit positive-zero states differ")
    if cases != 16:
        raise AssertionError("first-chunk edge denominator changed")
    print("[first chunk edge] PASS cases=16 repeats=8 None-vs-explicit-zero=RAW-BIT routing=UNCHANGED")


if __name__ == "__main__":
    main()
