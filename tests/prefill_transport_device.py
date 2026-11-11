#!/usr/bin/env python3
"""Product transport graph/parity gate; no profiling dependencies or timings.

The staged control uses the same library and identical first-request operands,
with an extra inactive request to select the ordinary prepare+pack path. This
is a correctness control, not a performance A/B (capacity and core route may
differ). The full serving suite separately checks independent recurrence.
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from actlize_la.serving import NativeLibrary
from actlize_la.serving.device import current_device_identity
from serving_device import GdnCase, raw_equal
from aux_device import graph_stats

PACKED_FIELDS = ("q", "k", "v", "beta", "log_g", "initial_kv",
                 "final_kv", "packed_output", "errors")
UNUSED_DIRECT_FIELDS = ("queries", "keys", "decays", "betas")


def cases():
    return [(1, t, hv, True) for t in (2048, 8192) for hv in (32, 48, 64)] + [
        (2, 2048, 32, False), (1, 512, 48, False), (1, 2049, 64, False)]


def compare(actual, expected):
    denominator = {*PACKED_FIELDS, "output", "state"}
    assert set(actual) == set(expected) == denominator, "comparison field denominator changed"
    for name in sorted(denominator):
        assert raw_equal(actual[name], expected[name]), f"bit mismatch: {name}"


def reset(case, iteration):
    case.state.copy_(case.initial)
    case.out.fill_(-99)
    for name in (*PACKED_FIELDS, *UNUSED_DIRECT_FIELDS):
        case.work[name].fill_(37)
    # One primary request; all remaining request capacity is inactive/empty.
    length = (case.width, case.width - 1, case.width // 2, 0,
              case.width, case.width, case.width, case.width)[iteration]
    case.starts.zero_()
    case.starts[1:].fill_(length)
    case.live.fill_(0 if iteration == 4 else 1)
    case.slots.zero_()
    case.slots[0] = 0 if iteration == 1 else 2
    case.final_slots.zero_()
    case.final_slots[0] = 0 if iteration == 5 else (3 if iteration == 1 else 2)
    case.checkpoint_slots.zero_()
    case.checkpoint_offsets.zero_()
    if iteration == 6:
        case.checkpoint_slots[0] = 1  # Unsupported checkpoint: errors=7, no writes.
    if iteration == 7:
        case.final_slots[0] = case.state.shape[0]  # Out-of-bounds: errors=2.


def snapshot(case, selected, iteration):
    result = {name: case.work[name][:1].cpu() for name in PACKED_FIELDS}
    result.update(output=case.out[:case.width].cpu(), state=case.state.cpu())
    expected_error = 7 if iteration == 6 else 2 if iteration == 7 else 0
    assert result["errors"].item() == expected_error
    assert not case.work["errors"][1:].any().item(), "inactive request error changed"
    assert bool((case.out[case.width:] == -99).all()), "inactive output changed"
    if iteration in (3, 4, 5, 6, 7):
        assert raw_equal(result["state"], case.initial), "empty/inactive/rejected request wrote state"
    for name, guard in case.guards:
        assert bool((guard == 37).all()), f"workspace canary changed: {name}"
    if selected:
        for name in UNUSED_DIRECT_FIELDS:
            assert bool((case.work[name] == 37).all()), f"unused direct-pack scratch was written: {name}"
    return result


def run_case(lib, requests, width, hv, selected):
    public = GdnCase(lib, width, "prefill", requests=requests, hk=16, hv=hv)
    control = GdnCase(lib, width, "prefill", requests=requests + 1, hk=16, hv=hv)
    control.initial.copy_(public.initial)
    for name in ("mixed", "ad", "bd"):
        getattr(control, name)[:width].copy_(getattr(public, name)[:width])
    assert public.plan.transport_route == {"entry": "ala_prefill_run", "direct_pack": selected}
    assert control.plan.transport_route == {"entry": "ala_prefill_run", "direct_pack": False}
    graphs, stats = [], []
    for case, direct in ((public, selected), (control, False)):
        reset(case, 0)
        case.plan()
        case.ok()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            case.plan()
        observed = graph_stats(lib, graph)
        assert observed["kernels"] == (8 if direct else 9), "native graph disagrees with transport route"
        graph.instantiate()
        graphs.append(graph)
        stats.append(observed)
    for iteration in range(8):
        reset(public, iteration)
        reset(control, iteration)
        torch.cuda.synchronize()
        allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
        for graph in graphs:
            graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations, "replay allocated"
        actual = snapshot(public, selected, iteration)
        expected = snapshot(control, False, iteration)
        compare(actual, expected)
        if iteration == 0:
            wrong = {**actual, "q": actual["q"].clone()}
            wrong["q"].view(torch.int16).reshape(-1)[0] ^= 1
            try:
                compare(wrong, expected)
            except AssertionError as error:
                assert "q" in str(error)
            else:
                raise AssertionError("one-bit packed operand mutation escaped")
    for graph in graphs:
        graph.reset()
    return dict(requests=requests, width=width, hk=16, hv=hv, direct_pack=selected,
                public_graph=stats[0], staged_graph=stats[1], replays=8,
                packed_output_state="RAW-BIT/PASS", allocation="STABLE",
                metadata="length/slot/live/empty/null/checkpoint/invalid-slot",
                negative="one-bit-packed-q:EXPECTED_RED")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    result = dict(status="INCOMPLETE", cases=[], performance="NOT_RUN")
    try:
        if not torch.cuda.is_available():
            result.update(status="SKIP", reason="no PPU device; not PASS")
            return 3
        torch.cuda.set_device(0)
        result["device"] = current_device_identity("ppu10")
        path = args.library.resolve(strict=True)
        result.update(library=str(path), binary_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        lib = NativeLibrary(path, target="ppu10")
        query = lib.lib.ala_prefill_token_parallel
        query.argtypes, query.restype = [], C.c_int
        assert query() == 1, "library has not integrated token-parallel transport"
        for cell in cases():
            row = run_case(lib, *cell)
            result["cases"].append(row)
            print("[prefill transport] " + json.dumps(row), flush=True)
        assert len(result["cases"]) == 9 and sum(x["direct_pack"] for x in result["cases"]) == 6
        assert hashlib.sha256(path.read_bytes()).hexdigest() == result["binary_sha256"]
        result["status"] = "PASS"
        return 0
    except Exception as error:
        result.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        args.results.write_text(json.dumps(result, indent=2) + "\n")
        print(f"[prefill transport] {result['status']}: {args.results}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
