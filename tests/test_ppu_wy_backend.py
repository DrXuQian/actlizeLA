#!/usr/bin/env python3
"""New WY device admission; unchanged original tolerance and recurrent oracle."""
import argparse
import os
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from test_ppu_gdn_backend import fixture, assert_pair, digest, print_failure, MAX_RELATIVE_ERROR
from actlize_la import gdn_chunk_wy
from actlize_la.reference.gdn_chunk_ref import torch_recurrent_gated_delta_rule


def oracle(inputs, initial):
    q, k, v, g, beta = inputs
    ratio = v.shape[2] // q.shape[2]
    return torch_recurrent_gated_delta_rule(
        q.repeat_interleave(ratio, 2), k.repeat_interleave(ratio, 2), v, g, beta,
        initial_state=initial, output_final_state=True)


def run_case(shape, gate, nonzero, device, deliveries=()):
    cpu = fixture(*shape, gate)
    if nonzero:
        # Also exercise the FP32 gate ABI without changing its values.
        cpu = (*cpu[:3], cpu[3].float(), cpu[4])
    initial = (torch.randn(shape[0], shape[3], 128, 128,
                          generator=torch.Generator().manual_seed(18)) * .005) if nonzero else None
    want = oracle(cpu, initial)
    inputs = tuple(x.to(device) for x in cpu)
    state = initial.to(device) if initial is not None else None
    def call():
        return gdn_chunk_wy(*inputs, initial_state=state)
    got = call()
    torch.cuda.synchronize()
    try:
        errs = assert_pair(got, want)
    except AssertionError:
        print_failure(f"wy/{shape}/g={gate}/initial={nonzero}", got, want)
        raise
    fingerprint = digest(got)
    for _ in range(7):
        if digest(call()) != fingerprint:
            raise AssertionError("WY repeated launches changed output/state bits")
    q, k, v, g, beta = inputs
    ratio = shape[3] // shape[2]
    expanded = gdn_chunk_wy(q.repeat_interleave(ratio, 2), k.repeat_interleave(ratio, 2),
                            v, g, beta, initial_state=state)
    if digest(expanded) != fingerprint:
        raise AssertionError("native GVA differs from materialized expansion")
    out_only, absent = gdn_chunk_wy(*inputs, initial_state=state, output_final_state=False)
    if absent is not None or not torch.equal(out_only, got[0]):
        raise AssertionError("output-only path changed output or returned state")
    if digest(inputs) != digest(cpu) or (state is not None and not torch.equal(state.cpu(), initial)):
        raise AssertionError("WY modified an input")
    for delivery in deliveries:
        call_variant = lambda: gdn_chunk_wy(*inputs, initial_state=state, delivery=delivery)
        for _ in range(8):
            candidate = call_variant()
            assert_pair(candidate, want)
            if digest(candidate) != fingerprint:
                print_failure(f"wy/{delivery}/{shape}", candidate, got)
                raise AssertionError(f"{delivery} differs from scalar WY bits")
        candidate_expanded = gdn_chunk_wy(
            q.repeat_interleave(ratio, 2), k.repeat_interleave(ratio, 2), v, g, beta,
            initial_state=state, delivery=delivery)
        if digest(candidate_expanded) != fingerprint:
            raise AssertionError(f"{delivery} native GVA differs from expanded")
        candidate_out, candidate_state = gdn_chunk_wy(
            *inputs, initial_state=state, output_final_state=False, delivery=delivery)
        if candidate_state is not None or not torch.equal(candidate_out, got[0]):
            raise AssertionError(f"{delivery} output-only mismatch")
        if digest(inputs) != digest(cpu) or (state is not None and not torch.equal(state.cpu(), initial)):
            raise AssertionError(f"{delivery} modified an input")
        print(f"[WY delivery admission] delivery={delivery} shape={shape} g={gate} "
              f"initial={nonzero} scalar-raw-bit=PASS repeat=8/8 GVA/output-only=PASS", flush=True)
    for role in (0, 1):
        wrong = list(got)
        wrong[role] = torch.zeros_like(wrong[role])
        try:
            assert_pair(wrong, want)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"zero-{role} negative escaped")
    print(f"[WY device] shape={shape} g={gate} initial={nonzero} input_sha={digest(cpu)} "
          f"errors={errs} limit={MAX_RELATIVE_ERROR} state_dtype={got[1].dtype} "
          f"fingerprint={fingerprint} replay=8/8 GVA=RAW-BIT output-only=PASS "
          "zero-output/state=EXPECTED-RED NUMERIC/PASS", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wy-extension", type=Path, required=True)
    p.add_argument("--device", type=int, default=0)
    family = p.add_mutually_exclusive_group()
    family.add_argument("--delivery-ab", action="store_true")
    family.add_argument("--tile-ab", action="store_true")
    family.add_argument("--state-ab", action="store_true")
    family.add_argument("--stage-ab", action="store_true")
    family.add_argument("--prepare-rows-ab", action="store_true")
    family.add_argument("--aiu-ab", action="store_true")
    family.add_argument("--split-prepare-ab", action="store_true")
    family.add_argument("--state-pipeline-ab", action="store_true")
    args = p.parse_args()
    if not args.wy_extension.is_file():
        p.error("WY extension missing")
    os.environ["GDN_QSA_WY_EXTENSION"] = str(args.wy_extension.resolve())
    torch.set_num_threads(1)
    torch.cuda.set_device(args.device)
    props = torch.cuda.get_device_properties(args.device)
    if "PPU" not in props.name.upper():
        raise RuntimeError(f"not a PPU: {props.name}")
    device = torch.device("cuda", args.device)
    from actlize_la.gdn_wy_interface import (PACKED_DELIVERIES, TILED_DELIVERIES,
                                              STATE_DELIVERIES, STAGE_DELIVERIES, PREPARE_ROWS_DELIVERIES, AIU_DELIVERIES,
                                              SPLIT_PREPARE_DELIVERIES, STATE_PIPELINE_DELIVERIES)
    deliveries = (STATE_PIPELINE_DELIVERIES if args.state_pipeline_ab else
                  SPLIT_PREPARE_DELIVERIES if args.split_prepare_ab else AIU_DELIVERIES if args.aiu_ab else PREPARE_ROWS_DELIVERIES if args.prepare_rows_ab else
                  STAGE_DELIVERIES if args.stage_ab else STATE_DELIVERIES if args.state_ab else
                  TILED_DELIVERIES if args.tile_ab else PACKED_DELIVERIES if args.delivery_ab else ())
    for length in (1, 16, 63, 64, 65, 129):
        for initial in (False, True):
            run_case((2, length, 1, 2), -.1, initial, device, deliveries)
    for gate in (0., -.1, -1.):
        run_case((1, 2048, 16, 32), gate, False, device, deliveries)
    run_case((1, 2048, 16, 64), -.1, False, device, deliveries)
    print(f"[WY device] PASS cases=16 delivery_variants={1+len(deliveries)} original-numerical-gate=UNCHANGED")


if __name__ == "__main__":
    main()
