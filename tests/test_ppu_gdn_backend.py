#!/usr/bin/env python3
"""Device admission for the ORIGINAL kernels and dispatch, with fixed criteria.

The 2% max/max tolerance is the repository's existing test_gdn_chunk.py
criterion. Reset is approximate; do not label it RAW-BIT vs the reference.
Repeated runs and GVA vs explicit head expansion must be bit-identical.
"""
import argparse
import hashlib
import os
from pathlib import Path
import statistics

import torch

from actlize_la.gdn_chunk_interface import gdn_chunk, gdn_chunk_twolevel
from actlize_la.reference.gdn_chunk_ref import torch_recurrent_gated_delta_rule

MAX_RELATIVE_ERROR = 2e-2


def fixture(batch, sequence, q_heads, value_heads, gate, seed=0x6A09E667):
    # CPU-generated fixture makes the input independent of vendor RNG kernels.
    gen = torch.Generator().manual_seed(seed)
    def rand(shape):
        return (torch.randn(shape, generator=gen) * 0.05).to(torch.bfloat16)
    q = rand((batch, sequence, q_heads, 128))
    k = rand(q.shape)
    v = rand((batch, sequence, value_heads, 128))
    g = torch.full((batch, sequence, value_heads), gate, dtype=torch.bfloat16)
    beta = torch.rand(g.shape, generator=gen).sigmoid().to(torch.bfloat16)
    return q, k, v, g, beta


def digest(tensors):
    h = hashlib.sha256()
    for t in tensors:
        h.update(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()[:16]


def error(got, want):
    got, want = got.float().cpu(), want.float().cpu()
    if not torch.isfinite(got).all():
        return float("inf")
    return ((got - want).abs().max() / (want.abs().max() + 1e-9)).item()


def assert_pair(got, want):
    errs = [error(a, b) for a, b in zip(got, want)]
    if max(errs) >= MAX_RELATIVE_ERROR:
        raise AssertionError(f"original 2% criterion failed: output/state={errs}")
    return errs


def print_failure(label, got, want):
    # Read the failing launch, never launch another variant or mutate output
    # while diagnosing. Near-100% relative error alone does not establish zero.
    for role, actual, expected in zip(("output", "state"), got, want):
        a, b = actual.detach().float().cpu(), expected.detach().float().cpu()
        worst = int((a - b).abs().reshape(-1).argmax())
        print(f"[PPU GDN failure] case={label} role={role} "
              f"finite={int(torch.isfinite(a).sum())}/{a.numel()} "
              f"nonzero={int(torch.count_nonzero(a))}/{a.numel()} "
              f"got_max={a.abs().max().item():.9g} want_max={b.abs().max().item():.9g} "
              f"worst_flat={worst} got={a.reshape(-1)[worst].item():.9g} "
              f"want={b.reshape(-1)[worst].item():.9g}", flush=True)


def reference(inputs):
    q, k, v, g, beta = inputs
    ratio = v.shape[2] // q.shape[2]
    return torch_recurrent_gated_delta_rule(
        q.repeat_interleave(ratio, 2), k.repeat_interleave(ratio, 2),
        v, g, beta, output_final_state=True)


def run_case(label, shape, gate, group_chunks, expected_route, device):
    cpu = fixture(*shape, gate)
    want = reference(cpu)
    print(f"[PPU GDN fixture] case={label} input_sha={digest(cpu)} "
          f"reference_sha={digest(want)} limit={MAX_RELATIVE_ERROR}", flush=True)
    inputs = tuple(x.to(device) for x in cpu)
    def launch():
        if group_chunks is None:
            return gdn_chunk(*inputs)
        return gdn_chunk_twolevel(*inputs, group_chunks=group_chunks)
    got = launch()
    torch.cuda.synchronize()
    if group_chunks is not None:
        info = got[2].cpu().tolist()
        use_shift, exclusive, count, groups, metric = info
        route = "reset" if use_shift else ("blelloch" if exclusive else "hillis-steele")
        if route != expected_route:
            raise AssertionError(f"{label}: expected {expected_route}, got {route}: {info}")
        print(f"[PPU GDN route] case={label} route={route} "
              f"groups={int(groups)} nonreset_groups={int(count)} metric={metric:g}")
    try:
        errs = assert_pair(got[:2], want)
    except AssertionError:
        print_failure(label, got[:2], want)
        raise
    initial_hash = digest(got[:2])
    for _ in range(3):
        again = launch()
        if not all(torch.equal(a.view(torch.int16), b.view(torch.int16))
                   for a, b in zip(got[:2], again[:2])):
            raise AssertionError(f"{label}: replay not bit-stable")
    # A correct numerical oracle must reject a genuinely wrong output.
    try:
        assert_pair((torch.zeros_like(got[0]), got[1]), want)
    except AssertionError:
        negative = "EXPECTED-RED/PASS"
    else:
        raise AssertionError("zero-output negative control escaped")
    # Metamorphic GVA check: the same logical input through the expanded path.
    if shape[2] != shape[3]:
        q, k, v, g, beta = inputs
        ratio = shape[3] // shape[2]
        expanded = q.repeat_interleave(ratio, 2), k.repeat_interleave(ratio, 2), v, g, beta
        equivalent = (gdn_chunk(*expanded) if group_chunks is None else
                      gdn_chunk_twolevel(*expanded, group_chunks=group_chunks))
        if not all(torch.equal(a.view(torch.int16), b.view(torch.int16))
                   for a, b in zip(got[:2], equivalent[:2])):
            raise AssertionError(f"{label}: fused GVA differs from explicit expansion")
    print(f"[PPU GDN original] case={label} B,S,Hk,Hv={shape} g={gate} "
          f"input_sha={digest(cpu)} max_relative_output={errs[0]:.8f} "
          f"max_relative_state={errs[1]:.8f} limit={MAX_RELATIVE_ERROR} "
          f"output_sha={initial_hash} repeat=4/4 RAW-BIT/STABLE "
          f"zero-output={negative} NUMERIC/PASS")


def perf(device, samples, launches, warmup):
    # The target Qwen-like shape; include both decay domains because reset is
    # data-dependent. This measures the complete original public dispatch,
    # INCLUDING its allocations, head/gate preprocessing and host decision sync.
    for gate, expected in ((-1.0, "reset-GC8"), (-0.1, "serial")):
        cpu = fixture(1, 2048, 16, 32, gate)
        inputs = tuple(x.to(device) for x in cpu)
        call = lambda: gdn_chunk(*inputs)
        first = call()
        assert_pair(first, reference(cpu))
        for _ in range(warmup):
            call()
        times = []
        for _ in range(samples):
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            torch.cuda.synchronize()
            start.record()
            for _ in range(launches):
                out = call()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) * 1000 / launches)
        if digest(out) != digest(first):
            raise AssertionError("performance replay changed output")
        print(f"[PPU GDN original perf] shape=B1,S2048,Hk16,Hv32,D128 g={gate} "
              f"expected_route={expected} protocol=full-public-api-event-span "
              f"includes_host_dispatch_sync=1 warmup={warmup} samples={samples} "
              f"launches_per_sample={launches} median_us={statistics.median(times):.3f} "
              f"range=[{min(times):.3f},{max(times):.3f}] output_sha={digest(out)}",
              flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--extension", type=Path, required=True)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--perf", action="store_true")
    p.add_argument("--samples", type=int, default=7)
    p.add_argument("--launches", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    args = p.parse_args()
    if not args.extension.is_file():
        p.error("PPU binding not found")
    if min(args.samples, args.launches) < 1 or args.warmup < 0:
        p.error("samples/launches must be positive and warmup nonnegative")
    os.environ["GDN_QSA_PPU_EXTENSION"] = str(args.extension.resolve())
    torch.set_num_threads(1)
    torch.cuda.set_device(args.device)
    props = torch.cuda.get_device_properties(args.device)
    if "PPU" not in props.name.upper():
        raise RuntimeError(f"not a PPU device: {props.name}")
    print(f"[PPU GDN original device] name={props.name} cu={props.multi_processor_count} "
          f"torch={torch.__version__} extension={args.extension}", flush=True)
    device = torch.device("cuda", args.device)
    run_case("tail-gva-hillis", (2, 65, 1, 2), 0.0, 2, "hillis-steele", device)
    run_case("blelloch", (1, 64, 1, 2), 0.0, 2, "blelloch", device)
    run_case("reset-register-replay", (1, 64, 1, 2), -1.0, 2, "reset", device)
    run_case("serial-tail", (1, 65, 16, 32), -0.1, None, "serial", device)
    run_case("target-reset", (1, 2048, 16, 32), -1.0, 8, "reset", device)
    run_case("target-auto", (1, 2048, 16, 32), -1.0, None, "auto", device)
    if args.perf:
        perf(device, args.samples, args.launches, args.warmup)
    print("[PPU GDN original] PASS: original reset/scan/register-replay + serial + GVA/tail")


if __name__ == "__main__":
    main()
