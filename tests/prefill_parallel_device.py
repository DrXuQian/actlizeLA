"""Compare token-parallel prefill against an immutable admitted native DSO.

Not a timing harness. Full serving/auxiliary gates separately check independent
recurrence, checked dependency errors and decode/verify/commit. This gate adds
exact intermediate-buffer, inactive-row and changing-metadata graph parity.
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from actlize_la.serving import NativeLibrary
from actlize_la.serving.device import current_device_identity
from collect_serving_prefill_acu import read_run
from serving_device import GdnCase, raw_equal

FIELDS = ("queries", "keys", "decays", "betas", "q", "k", "v", "beta",
          "log_g", "initial_kv", "final_kv", "packed_output", "errors")


def schedule(lib):
    fn = getattr(lib.lib, "ala_prefill_token_parallel", None)
    if fn is None:
        return 0  # Only allowed for the separately hash-bound historical DSO.
    fn.argtypes, fn.restype = [], C.c_int
    return fn()


def reset(case, iteration):
    case.state.copy_(case.initial)
    case.out.fill_(-99)
    for name in FIELDS:
        case.work[name].fill_(37)
    width = case.width
    requests = case.starts.numel() - 1
    # Capacity remains fixed in the graph. Change length, slots and live rows.
    length = max(0, width - (iteration % 3))
    starts = [0, length]
    if requests == 3:
        starts += [length + width // 2, length + width // 2]
    case.starts.copy_(torch.tensor(starts, dtype=torch.int32, device=case.device))
    case.live.fill_(requests if iteration % 2 else 1)
    src = [0 if iteration % 2 else 2] + ([4, 0] if requests == 3 else [])
    dst = [1 if iteration % 2 else 2] + ([3, 0] if requests == 3 else [])
    if iteration == 7:
        dst[0] = 0  # Null final slot zeros active outputs; state must not change.
    case.slots.copy_(torch.tensor(src, device=case.device))
    case.final_slots.copy_(torch.tensor(dst, device=case.device))


def snapshot(case):
    case.ok()
    return {"output": case.out.cpu(), "state": case.state.cpu(),
            **{name: case.work[name].cpu() for name in FIELDS}}


def compare(actual, expected):
    assert set(actual) == set(expected), "comparison field denominator changed"
    for name in expected:
        assert raw_equal(actual[name], expected[name]), f"bit mismatch: {name}"


def run_case(old, new, width, hv):
    requests = 1 if width >= 2048 else 3
    baseline = GdnCase(old, width, "prefill", hv=hv, requests=requests)
    candidate = GdnCase(new, width, "prefill", hv=hv, requests=requests)
    graphs = []
    for case in (baseline, candidate):
        reset(case, 0)
        case.plan()
        case.ok()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            case.plan()
        graphs.append(graph)
    # Prepare resets before the allocation-stability observation: CPU checks
    # below do not launch device kernels or change persistent device buffers.
    for iteration in range(8):
        for case in (baseline, candidate):
            reset(case, iteration)
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        for graph in graphs:
            graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated
        expected, actual = snapshot(baseline), snapshot(candidate)
        compare(actual, expected)
        if iteration == 0:
            # A real one-bit change in an observed packed operand must be red.
            planted = {**actual, "q": actual["q"].clone()}
            planted["q"].view(torch.int16).flatten()[0] ^= 1
            try:
                compare(planted, expected)
            except AssertionError as error:
                assert "q" in str(error)
            else:
                raise AssertionError("one-bit negative escaped")
    return dict(width=width, hk=16, hv=hv, requests=requests, replays=8,
                buffers=len(FIELDS) + 2, parity="RAW-BIT/PASS",
                metadata="length/live/initial-slot/final-slot/null/empty",
                negative="one-bit-packed-q:EXPECTED_RED")


def direct_pack_gate(old, new):
    """H1公共默认与未登记shape回退；不能以实验fn替换代替实际默认命中。"""
    sys.path.insert(0, str(ROOT / "benchmarks"))
    from profile_serving_prefill import ServingCall, raw_fixture, capture, digest
    from aux_device import graph_stats
    query = getattr(new.lib, "ala_prefill_direct_pack_route", None)
    if query is None:
        raise RuntimeError("H1候选缺少公共transport route查询")
    query.argtypes, query.restype = [C.c_int] * 4, C.c_int
    old_query = getattr(old.lib, "ala_prefill_direct_pack_route", None)
    if old_query is not None:
        old_query.argtypes, old_query.restype = [C.c_int] * 4, C.c_int
    cases = [(1, t, hv, True) for hv in (32, 48, 64) for t in (2048, 8192)]
    cases += [(2, 2048, 32, False), (1, 512, 48, False), (1, 2049, 64, False)]
    rows, retained = [], []
    for batch, length, hv, selected in cases:
        fixture = raw_fixture(length, hv, -.1, batch)
        baseline, public = ServingCall(old, fixture), ServingCall(new, fixture)
        assert old_query is None or old_query(batch, length, 16, hv) == 0, "基线不可已启用H1"
        assert query(batch, length, 16, hv) == int(selected)
        assert public.plan.transport_route == {"entry": "ala_prefill_run", "direct_pack": selected}
        want = capture(baseline)
        assert digest(capture(public)) == digest(want)
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            public()
        stats = graph_stats(new, graph)
        assert stats["kernels"] == (8 if selected else 9), "实际默认图未命中登记路由"
        graph.instantiate()
        retained.append((baseline, public, graph))
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        for _ in range(8):
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == allocated
            result = (public.output, public.raw["state"][public.final])
            assert digest(public.cpu_pair(result)) == digest(want)
        rows.append(dict(B=batch, T=length, Hv=hv, direct_pack=selected,
                         graph=stats, replays=8, raw_bit="PASS"))
    assert rows[-1]["graph"]["kernels"] != 8, "未登记shape的错误快路径声明应红"
    # H1不再写旧FP32归一化scratch；只排除这4个明确废弃的中间buffer。
    global FIELDS
    original_fields = FIELDS
    unused = ("queries", "keys", "decays", "betas")
    try:
        FIELDS = tuple(n for n in FIELDS if n not in unused)
        mutable = [run_case(old, new, 2048, hv) for hv in (32, 48, 64)]
    finally:
        FIELDS = original_fields
    return dict(cases=rows, false_route_claim_red=True, mutable_graph=mutable,
                compared_fields=[n for n in original_fields if n not in unused] + ["output", "state"],
                unused_intermediate_fields=list(unused))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--direct-pack", action="store_true", help="对比当前token-parallel基线与H1公共默认")
    args = parser.parse_args()
    result = dict(status="INCOMPLETE", cases=[], performance="NOT_RUN")
    try:
        if not torch.cuda.is_available():
            result.update(status="SKIP", reason="no PPU device")
            return 3
        torch.cuda.set_device(0)
        result["device"] = current_device_identity("ppu10")
        result["baseline"] = read_run(args.baseline_run)
        result["candidate_sha256"] = hashlib.sha256(args.candidate.read_bytes()).hexdigest()
        old = NativeLibrary(result["baseline"]["library"], target="ppu10")
        new = NativeLibrary(args.candidate, target="ppu10")
        if args.direct_pack:
            if schedule(old) != 1 or schedule(new) != 1:
                raise ValueError("H1必须对比当前token-parallel基线，不得使用旧串行路径")
            result["direct_pack"] = direct_pack_gate(old, new)
            result.update(status="PASS", cases=result["direct_pack"]["cases"])
            return 0
        if schedule(old) != 0 or schedule(new) != 1:
            raise ValueError("require serial baseline and token-parallel candidate, not two identical routes")
        for hv in (32, 48, 64):
            for width in (31, 32, 33, 63, 64, 65, 127, 128, 129, 2048, 8192):
                row = run_case(old, new, width, hv)
                result["cases"].append(row)
                print("[prefill transport parity] " + json.dumps(row), flush=True)
        result["status"] = "PASS"
        return 0
    except Exception as error:
        result.update(status="FAIL", error=repr(error))
        raise
    finally:
        args.results.write_text(json.dumps(result, indent=2) + "\n")
        print(f"[prefill transport parity] {result['status']}: {args.results}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
