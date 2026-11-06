#!/usr/bin/env python3
"""PPU1.0 default-vs-retained admission, including real model head geometries.

No timing or FLA speed claim: independently validate recurrence, default
dispatch, carried state and repeated raw bytes before the performance run.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from actlize_la import gdn_forward, gdn_chunk_residual, get_gdn_route_info
from actlize_la.serving.device import current_device_identity
from test_ppu_gdn_backend import fixture, assert_pair, digest, MAX_RELATIVE_ERROR
from test_ppu_wy_backend import oracle


def cases():
    # Deliberately independent of the routing implementation/catalog.
    return [(length, hv, gate) for hv in (32, 48)
            for length in (1, 63, 64, 65, 127, 128, 129, 2048, 8192)
            for gate in (-0.1, -1.0)]


def bit_equal(a, b):
    return all(x.dtype == y.dtype and torch.equal(x.view(torch.uint8), y.view(torch.uint8))
               for x, y in zip(a, b))


def admit(length, hv, gate):
    cpu = fixture(1, length, 16, hv, gate)
    # Distinct K/V axes, distinct heads: shape equality cannot hide transpose.
    initial = (torch.randn(1, hv, 128, 128, generator=torch.Generator().manual_seed(7241)) * .01)
    want = oracle(cpu, initial)
    tensors = tuple(t.cuda() for t in cpu)
    state = initial.cuda()
    input_digest, state_digest = digest(tensors), digest((state,))
    explicit = gdn_forward(*tensors, initial_state=state, backend="ppu10",
                           algorithm="residual", delivery="full-chunk")
    assert_pair(explicit, want)
    error = None
    for _ in range(8):
        default = gdn_forward(*tensors, initial_state=state, backend="ppu10")
        error = assert_pair(default, want)
        if not bit_equal(default, explicit):
            raise AssertionError("default != explicit retained output/state bytes")
        if default[0].dtype != torch.bfloat16 or default[1].dtype != torch.float32:
            raise AssertionError("default output/state precision is not BF16/FP32")
    for call in (
        lambda: gdn_forward(*tensors, initial_state=state, backend="ppu10", algorithm="residual"),
        lambda: gdn_chunk_residual(*tensors, initial_state=state),
    ):
        if not bit_equal(call(), explicit):
            raise AssertionError("another public residual default chose a different implementation")
    only, absent = gdn_forward(*tensors, initial_state=state, backend="ppu10", output_final_state=False)
    if absent is not None or not bit_equal((only,), (explicit[0],)):
        raise AssertionError("default output-only changed results")
    if (digest(tensors), digest((state,))) != (input_digest, state_digest):
        raise AssertionError("default path modified inputs or initial state")
    # Same input, wrong initial layout: this must fail the independent oracle.
    wrong_state = state.transpose(-1, -2).contiguous()
    wrong = gdn_forward(*tensors, initial_state=wrong_state, backend="ppu10")
    try:
        assert_pair(wrong, want)
    except AssertionError:
        pass
    else:
        raise AssertionError("non-symmetric initial-state transpose negative escaped")
    result = dict(shape=[1, length, 16, hv, 128], gate=gate, initial="nonzero-asymmetric",
                  errors=error, fingerprint=digest(explicit), repeats=8,
                  default_explicit="RAW-BIT/PASS", oracle="NUMERIC/PASS",
                  wrong_state_transpose="EXPECTED-RED/PASS")
    print("[PPU default GDN] " + json.dumps(result), flush=True)
    return result


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    if not args.extension.is_file():
        parser.error("missing _gdn_wy_ppu extension; no scalar fallback")
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        print("[PPU default GDN] SKIP: no device; device admission NOT_RUN", flush=True)
        return 3
    torch.cuda.set_device(0)
    identity = current_device_identity("ppu10")
    os.environ["GDN_QSA_WY_EXTENSION"] = str(args.extension.resolve())
    report = dict(schema=1, device=identity, cases=[], status="RUNNING",
                  criterion=MAX_RELATIVE_ERROR, timing="NOT_RUN", scope="PUBLIC_DEFAULT_NOT_SERVING")
    try:
        sample = tuple(t.cuda() for t in fixture(1, 64, 16, 32, -.1))
        report["route"] = get_gdn_route_info(*sample, backend="ppu10")
        if report["route"]["native_entry"] != "residual_full_chunk":
            raise AssertionError("the default route is not the retained native entry")
        print("[PPU default route] " + json.dumps(report["route"]), flush=True)
        for row in cases():
            report["cases"].append(admit(*row))
        if len(report["cases"]) != 36:
            raise AssertionError("default admission denominator must be 36")
        report["status"] = "PASS"
        print("[PPU default GDN] PASS cases=36 repeat=8 numeric+default-bits; performance=NOT_RUN", flush=True)
    except Exception as exc:
        report.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        args.results.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
