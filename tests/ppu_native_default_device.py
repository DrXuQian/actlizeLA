#!/usr/bin/env python3
"""Product-entry routing/RAW-BIT/graph gate; no performance measurement.

The same DSO's unmodified retained C entry is the control. Its independent
recurrence admission remains separate; equality alone is not that proof.
"""
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from actlize_la import gdn_forward, get_gdn_route_info
from actlize_la.serving import NativeLibrary, PreparedGDNCore, gdn_core_workspace_specs
from actlize_la.serving.gdn_core import gdn_core_route
from actlize_la.serving.device import current_device_identity
from serving_device import GdnCase, allocate, raw_equal
from test_ppu_gdn_backend import fixture, digest


def expected_route(b, t, hk, hv, fp32):
    # Independent test denominator; do not import the production policy.
    return 3 if (b, t, hk, hv, fp32) in {
        (1, 2048, 16, 32, True), (1, 2048, 16, 48, True),
        (1, 8192, 16, 32, True), (1, 8192, 16, 48, True)} else 0


def retained(lib, inputs, initial):
    q, k, v, g, beta = inputs
    b, t, hk, _ = q.shape
    hv = v.shape[2]
    work = allocate(gdn_core_workspace_specs(b, t, hk, hv), q.device)
    output = torch.empty_like(v)
    final = torch.empty(b, hv, 128, 128, dtype=torch.float32, device=q.device)
    fn = lib.lib.gdn_wy_forward_residual_full_chunk
    fn.argtypes, fn.restype = [C.c_void_p] * 12 + [C.c_int] * 4 + [C.c_bool, C.c_void_p], C.c_int
    rc = fn(*(x.data_ptr() for x in inputs), initial.data_ptr() if initial is not None else None,
            output.data_ptr(), final.data_ptr(), *(work[n].data_ptr() for n in work),
            b, t, hk, hv, g.dtype == torch.float32, torch.cuda.current_stream().cuda_stream)
    if rc:
        raise RuntimeError(f"retained control launch failed: {rc}")
    torch.cuda.synchronize()
    assert torch.isfinite(output).all() and torch.isfinite(final).all()
    return output, final


def same(got, want):
    assert len(got) == len(want) and all(raw_equal(a, b) for a, b in zip(got, want)), "output/state RAW-BIT mismatch"


def product_case(lib, b, t, hk, hv, gate, fp32):
    cpu = list(fixture(b, t, hk, hv, gate))
    if fp32:
        cpu[3] = torch.full(cpu[3].shape, gate, dtype=torch.float32)
    inputs = [x.cuda() for x in cpu]
    initial = (torch.randn(b, hv, 128, 128, generator=torch.Generator().manual_seed(7241)) * .01).cuda()
    want = retained(lib, inputs, initial)
    route = get_gdn_route_info(*inputs, initial_state=initial, backend="ppu10")
    selected = expected_route(b, t, hk, hv, fp32)
    assert route["core_route"]["id"] == selected
    assert route["native_entry"] == "actlize_la_ppu10_forward_prepared"
    work = allocate(gdn_core_workspace_specs(b, t, hk, hv), inputs[0].device)
    output, final = torch.empty_like(want[0]), torch.empty_like(want[1])
    plan = PreparedGDNCore(lib, q=inputs[0], k=inputs[1], v=inputs[2], g=inputs[3], beta=inputs[4],
                           initial_state=initial, output=output, final_state=final, workspace=work)
    assert plan.core_route["id"] == selected
    before = digest((*inputs, initial))
    for _ in range(8):
        plan()
        same((output, final), want)
        same(gdn_forward(*inputs, initial_state=initial, backend="ppu10"), want)
    assert digest((*inputs, initial)) == before, "read-only inputs changed"
    only, absent = gdn_forward(*inputs, initial_state=initial, backend="ppu10", output_final_state=False)
    assert absent is None
    same((only,), want[:1])
    same(gdn_forward(*inputs, backend="ppu10"), retained(lib, inputs, None))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        plan()
    for i in range(8):
        inputs[2].copy_(cpu[2] * (-1 if i % 2 else 1))
        control = retained(lib, inputs, initial)
        before_alloc = torch.cuda.memory_stats()["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == before_alloc
        same((output, final), control)
    graph.reset()
    return dict(shape=[b, t, hk, hv, 128], gate=gate, gate_fp32=fp32,
                route=plan.core_route, public_prepared="RAW-BIT/PASS", initial="zero+asymmetric-nonzero",
                repeats=8, graph_changed_input=8, graph_allocation="STABLE")


def serving_case(lib, t, hv):
    c = GdnCase(lib, t, "prefill", requests=1, hk=16, hv=hv)
    assert gdn_core_route(lib, 1, t, 16, hv)["id"] == 3
    c.plan()
    c.ok()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        c.plan()
    for source, length, live in ((0, t, 1), (2, t, 1), (2, t - 1, 1), (2, t, 0)):
        c.slots.fill_(source)
        c.starts[1] = length
        c.live.fill_(live)
        c.state.copy_(c.initial)
        before = c.state.clone()
        before_alloc = torch.cuda.memory_stats()["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == before_alloc
        c.ok()  # Diagnostics may allocate; keep them outside the replay envelope.
        if not live:
            same((c.state,), (before,))
            continue
        inputs = [c.work[n] for n in ("q", "k", "v", "log_g", "beta")]
        want = retained(lib, inputs, c.work["initial_kv"])
        same((c.work["packed_output"], c.work["final_kv"]), want)
        same((c.out[:length].unsqueeze(0),), (want[0][:, :length],))
        same((c.state[2].transpose(-1, -2).contiguous().unsqueeze(0),), want[1:])
        same((c.state[:2], c.state[3:]), (before[:2], before[3:]))
    graph.reset()
    return dict(shape=[1, t, 16, hv, 128], serving_core="RAW-BIT/PASS",
                state_slot_layout="VK-to-KV-to-VK/PASS", guards="PASS",
                graph="changed-slot+shorter-live-length+inactive/PASS", allocation="STABLE")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = dict(status="INCOMPLETE", cases=[], serving=[], performance="NOT_RUN")
    try:
        if not torch.cuda.is_available():
            report.update(status="SKIP", reason="no PPU device; not PASS")
            print("[native default] SKIP: no PPU device; not PASS", flush=True)
            return 3
        report["device"] = current_device_identity("ppu10")
        path = args.library.resolve(strict=True)
        report["library"] = str(path)
        report["binary_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        os.environ.pop("GDN_QSA_WY_EXTENSION", None)
        os.environ["ACTLIZE_LA_SERVING_LIBRARY"] = str(path)
        lib = NativeLibrary(path, target="ppu10")
        cases = [(1, t, 16, hv, g, True) for t in (2048, 8192)
                 for hv in (32, 48) for g in (-.1, -1.)]
        cases += [(2, 2048, 16, 32, -.1, True), (1, 2049, 16, 32, -.1, True),
                  (1, 2048, 16, 32, -.1, False), (1, 2048, 16, 64, -.1, True),
                  (1, 65, 2, 6, -.1, True)]
        for cell in cases:
            row = product_case(lib, *cell)
            report["cases"].append(row)
            print("[native default] " + json.dumps(row), flush=True)
        for t in (2048, 8192):
            for hv in (32, 48):
                row = serving_case(lib, t, hv)
                report["serving"].append(row)
                print("[native default serving] " + json.dumps(row), flush=True)
        assert len(report["cases"]) == 13 and len(report["serving"]) == 4
        assert sum(row["route"]["id"] == 3 for row in report["cases"]) == 8
        report["status"] = "PASS"
        print("[native default] PASS: 8 native + 5 fallback; public/prepared/graphs + 4 serving geometries; performance=NOT_RUN", flush=True)
    except Exception as exc:
        report.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        args.results.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
