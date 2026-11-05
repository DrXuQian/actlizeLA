"""Real-device auxiliary numerics and composed graphs; no b12x runtime.

Run after serving_device.py. CPU references and observations are outside graph.
"""

import argparse
import ctypes as C
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import sys
import torch
from actlize_la.serving import (
    NativeLibrary,
    ServingRows,
    PreparedPointwise,
    PreparedShortConv,
    PreparedConvHistoryV1,
    PreparedPLEHash,
    ple_geometry,
    PreparedGDNVerifyV2,
    PreparedGDNCommitV2,
    PreparedGDNCheckedV2,
)
from aux_reference import (
    bf,
    grouped,
    gated,
    hc_silu,
    hc_combine,
    ple_gate,
    convolution,
    encoded_history,
    hash_packed,
    assert_raw_unchanged,
)
from serving_reference import gdn_reference
from actlize_la.serving.device import current_device_identity
from serving_device import GdnCase, close

torch.set_num_threads(1)
CASES = []
PERF = []
DEVICE = None


def record(name, **kwargs):
    row = dict(case=name, status="PASS", **kwargs)
    CASES.append(row)
    print("[FlashNext device] " + json.dumps(row, sort_keys=True), flush=True)


def gpu(x):
    return x.to(DEVICE)


def errors(n):
    return torch.empty(n, dtype=torch.int32, device=DEVICE)


def check_bf(got, want, ulps=1):
    got = got.detach().cpu().to(torch.bfloat16)
    want = want.detach().cpu().to(torch.bfloat16)
    assert torch.isfinite(got).all() and torch.isfinite(want).all()

    def ordered(t):
        b = t.view(torch.int16).int()
        return torch.where(b < 0, -32768 - b, b)

    diff = (ordered(got) - ordered(want)).abs()
    maximum = int(diff.max()) if diff.numel() else 0
    assert maximum <= ulps, (
        f"BF16 max ulps {maximum} > {ulps}, bad={int((diff > ulps).sum())}"
    )
    return int((diff != 0).sum())


def checked(plan, error):
    plan()
    torch.cuda.synchronize()
    assert not error.any().item(), error.cpu().tolist()


def measure(name, plan, reset=None):
    # Timings describe this operator/graph, not full-model performance or no-bubble admission.
    for _ in range(3):
        if reset:
            reset()
        plan()
    torch.cuda.synchronize()
    values = []
    for _ in range(7):
        if reset:
            reset()
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        plan()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000)
    row = dict(
        name=name,
        median_us=statistics.median(values),
        samples_us=values,
        scope="event-launch-span-not-device-only",
    )
    PERF.append(row)
    print("[FlashNext perf] " + json.dumps(row), flush=True)


def pointwise_tests(lib):
    # MTP hidden feedback is a full 10240-wide norm, not four 2560-wide norms.
    full = torch.randn(2, 1, 10240).bfloat16()
    full[:, :, :2560] *= 8
    fw = torch.randn(1, 10240) * 0.1 + 1
    fo = gpu(torch.zeros_like(full))
    fe = errors(2)
    fp = PreparedPointwise(
        lib, "grouped_norm", groups=1, x=gpu(full), weight=gpu(fw), output=fo, errors=fe
    )
    checked(fp, fe)
    check_bf(fo, grouped(full, fw))
    assert not torch.equal(
        fo.cpu(), grouped(full.view(2, 4, 2560), fw.view(4, 2560)).view_as(full)
    )
    for T in (1, 2, 4, 8, 16, 32, 65):
        torch.manual_seed(19 + T)
        x = torch.randn(T, 4, 2560).bfloat16()
        value = torch.randn(T, 1, 2560).bfloat16()
        logits = torch.randn_like(x)
        # Inputs are noncompact column views, outputs have row-padding canaries.
        xp = torch.empty(T, 10256, dtype=torch.bfloat16, device=DEVICE)
        xd = xp[:, :10240].view(T, 4, 2560)
        xd.copy_(x)
        gatep = torch.empty_like(xp)
        gate = gatep[:, :10240].view(T, 4, 2560)
        gate.copy_(logits)
        backing = torch.full((T + 2, 10256), 17, dtype=torch.bfloat16, device=DEVICE)
        out = backing[1 : T + 1, :10240].view(T, 4, 2560)
        live = gpu(torch.tensor([max(1, T - 1)], dtype=torch.int32))
        n = int(live.cpu()[0])
        rows = ServingRows(T, num_tokens=live)
        err = errors(T)
        for dtype in (torch.bfloat16, torch.float32):
            w = (torch.randn(4, 2560) * 0.2 + 0.8).to(dtype)
            wd = gpu(w)
            for zero in (False, True):
                p = PreparedPointwise(
                    lib,
                    "grouped_norm",
                    x=xd,
                    weight=wd,
                    output=out,
                    errors=err,
                    rows=rows,
                    zero_centered=zero,
                )
                checked(p, err)
                check_bf(out[:n], grouped(x, w, zero=zero)[:n])
                assert not out[n:].any().item()
            combined = torch.empty_like(out)
            inj = torch.randn(T, 4).bfloat16()
            p = PreparedPointwise(
                lib,
                "hc_combine_norm",
                x=xd,
                weight=wd,
                gate=gpu(inj),
                value=gpu(value),
                combined=combined,
                output=out,
                errors=err,
                rows=rows,
                zero_centered=True,
            )
            checked(p, err)
            check_bf(combined[:n], hc_combine(x, value, inj)[:n])
            # Isolate the norm boundary using ACTUAL rounded combined values.
            check_bf(out[:n], grouped(combined.cpu(), w, zero=True)[:n])
        assert (backing[0] == 17).all().item() and (backing[-1] == 17).all().item()
        assert (backing[1 : T + 1, 10240:] == 17).all().item()
        trace = torch.empty(T, 4, 5120, device=DEVICE)
        mean = torch.empty(T, 1, 2560, dtype=torch.bfloat16, device=DEVICE)
        p = PreparedPointwise(
            lib,
            "hc_gate_mean",
            x=xd,
            gate=gate,
            output=mean,
            trace=trace,
            errors=err,
            rows=rows,
        )
        checked(p, err)
        native = trace[:n].cpu()
        check_bf(native[..., :2560], bf(logits[:n].float().sigmoid()))
        products = bf(native[..., :2560] * x[:n].float())
        assert torch.equal(native[..., 2560:], products)
        total = torch.zeros_like(products[:, 0])
        for s in range(4):
            total += products[:, s]
        check_bf(mean[:n], (total / 4).bfloat16()[:, None], 0)
        pletrace = torch.empty(T, 4, 6, device=DEVICE)
        p = PreparedPointwise(
            lib,
            "ple_gate",
            x=xd,
            gate=gate,
            value=gpu(value),
            output=out,
            trace=pletrace,
            errors=err,
            rows=rows,
        )
        checked(p, err)
        expected, stages = ple_gate(x, logits, value)
        check_bf(pletrace[:n, :, 0], stages[:n, :, 0])
        # Downstream BF16 stages reconstructed independently from the observed sum.
        native = pletrace[:n].cpu()
        v = bf(native[..., 0:1] / (2560**0.5))
        clamp = bf(v.abs().clamp_min(float(torch.tensor(1e-6, dtype=torch.bfloat16))))
        root = bf(clamp.sqrt())
        signed = bf(root * v.sign())
        prob = bf(signed.sigmoid())
        check_bf(native[..., 1:5], torch.cat((v, clamp, root, signed), -1), 0)
        check_bf(native[..., 5:6], prob)
        check_bf(out[:n], (native[..., 5:6] * value[:n].float()).bfloat16(), 0)
        assert not out[n:].any().item() and not pletrace[n:].any().item()
        if T == 8:
            measure("ple-gate-T8", p)
        for D in (80, 160, 320):
            down = torch.randn(T, 1, D).bfloat16()
            bottleneck = torch.empty_like(gpu(down))
            tr = torch.empty(T, 1, D, device=DEVICE)
            p = PreparedPointwise(
                lib,
                "hc_scaled_silu",
                groups=1,
                x=gpu(down),
                output=bottleneck,
                trace=tr,
                errors=err,
                rows=rows,
            )
            checked(p, err)
            want, scaled = hc_silu(down)
            assert torch.equal(tr[:n].cpu(), scaled[:n])
            check_bf(bottleneck[:n], want[:n])
        record(
            "norm-hc-ple",
            tokens=T,
            logical_width=2560,
            weights="bf16/fp32",
            canary="PASS",
            intermediates="CHECKED",
        )
    for H in (16, 48):
        for dtype in (torch.bfloat16, torch.float32):
            for per_head in (False, True):
                for rounded, before, silu in itertools.product((False, True), repeat=3):
                    x = torch.randn(4, H, 128).bfloat16()
                    w = torch.randn(H if per_head else 1, 128).to(dtype)
                    gp = torch.randn(4, H * 128 + 19).bfloat16()
                    g = gp[:, : H * 128].view_as(x)
                    gd = gpu(gp)[:, : H * 128].view_as(g)
                    out = torch.empty_like(gpu(x))
                    tr = torch.empty(4, H, 256, device=DEVICE)
                    err = errors(4)
                    p = PreparedPointwise(
                        lib,
                        "gated_norm",
                        groups=H,
                        x=gpu(x),
                        gate=gd,
                        weight=gpu(w),
                        output=out,
                        trace=tr,
                        errors=err,
                        round_normalized=rounded,
                        norm_before_gate=before,
                        activation="silu" if silu else "sigmoid",
                    )
                    checked(p, err)
                    want, trace = gated(
                        x, g, w, rounded=rounded, before=before, silu=silu
                    )
                    native = tr.cpu()
                    n = native[..., :128]
                    if rounded:
                        check_bf(n, trace[..., :128])
                    else:
                        torch.testing.assert_close(
                            n, trace[..., :128], rtol=4e-6, atol=3e-6
                        )
                    weighted = n * w.float()
                    if rounded and dtype == torch.bfloat16:
                        weighted = bf(weighted)
                    assert torch.equal(native[..., 128:], weighted), (
                        "affine weight precision/rounding"
                    )
                    activation = g.float().sigmoid() * (g.float() if silu else 1)
                    check_bf(out, (weighted * (activation if before else 1)).bfloat16())
    record(
        "gated-rms",
        heads="16/48",
        modes="norm-before/after+rounded/unrounded+sigmoid/silu",
    )


class ConvCase:
    def __init__(self, lib, dilation, width, channels=67, mode="verify"):
        self.lib, self.d, self.w, self.c = lib, dilation, width, channels
        self.H = 3 * dilation
        self.L = self.H + 7
        self.x = torch.randn(3 * width, channels).bfloat16()
        self.weight = torch.randn(channels, 4).bfloat16() * 0.125
        self.initial = torch.randn(4, channels, self.L).bfloat16()
        # Position-major pool, channel stride1, position strideC. No hardcoded channel pitch.
        self.backing = torch.full(
            (4, self.L, channels + 3), 13, dtype=torch.bfloat16, device=DEVICE
        )
        self.state = self.backing[:, :, :channels].transpose(1, 2)
        self.state.copy_(self.initial)
        self.starts = gpu(torch.tensor([0, width, width, 2 * width], dtype=torch.int32))
        self.slots = gpu(
            torch.tensor([0, 0, 2], dtype=torch.int64)
        )  # empty row shares slot0
        self.ns = gpu(torch.tensor([3], dtype=torch.int32))
        self.nt = gpu(torch.tensor([2 * width], dtype=torch.int32))
        self.fresh = gpu(torch.tensor([False, True, True]))
        self.prev = gpu(torch.ones(3, dtype=torch.int32))
        self.rows = ServingRows(3 * width, self.nt, self.starts, self.slots, self.ns)
        self.xd = gpu(self.x)
        self.wd = gpu(self.weight)
        self.out = torch.empty_like(self.xd)
        self.err = errors(3)
        self.plan = PreparedShortConv(
            lib,
            x=self.xd,
            weight=self.wd,
            state=self.state,
            output=self.out,
            rows=self.rows,
            fresh=self.fresh,
            previous_accepted=self.prev,
            errors=self.err,
            dilation=dilation,
            mode=mode,
        )
        self.counts = gpu(torch.zeros(3, dtype=torch.int32))
        self.snap = torch.empty(
            3, channels, self.L, dtype=torch.bfloat16, device=DEVICE
        )
        self.stamps = torch.empty(3, dtype=torch.int64, device=DEVICE)
        self.snaperr = errors(3)
        self.restoreerr = errors(3)
        args = dict(
            state=self.state,
            snapshot=self.snap,
            stamps=self.stamps,
            consumed_prefix=self.counts,
            rows=self.rows,
            dilation=dilation,
        )
        self.capture = PreparedConvHistoryV1(lib, errors=self.snaperr, **args)
        self.restore = PreparedConvHistoryV1(
            lib, errors=self.restoreerr, restore=True, **args
        )


def conv_tests(lib):
    for d in (1, 3):
        for width in (1, 2, 4, 8):
            c = ConvCase(lib, d, width)
            for accepted in range(width + 1):
                c.state.copy_(c.initial)
                c.prev.fill_(1)
                c.fresh.copy_(torch.tensor([False, True, True]))
                checked(c.capture, c.snaperr)
                checked(c.plan, c.err)
                for row, first, slot in ((0, 0, 0), (2, width, 2)):
                    history = (
                        c.initial[slot, :, : c.H]
                        if row == 0
                        else torch.zeros(c.c, c.H).bfloat16()
                    )
                    want, _ = convolution(
                        c.x[first : first + width], c.weight, history, d
                    )
                    check_bf(c.out[first : first + width], want)
                    assert torch.equal(
                        c.state[slot].cpu(),
                        encoded_history(c.x[first : first + width], history, 7),
                    )
                assert not c.out[2 * width :].any().item()
                c.counts.fill_(accepted)
                checked(c.restore, c.restoreerr)
                if accepted == 0:
                    assert torch.equal(c.state.cpu(), c.initial)
                previous = (
                    c.initial[0, :, : c.H]
                    if accepted == 0
                    else c.state[0, :, accepted - 1 : accepted - 1 + c.H].cpu()
                )
                _, want = convolution(
                    c.x[:accepted], c.weight, c.initial[0, :, : c.H], d
                )
                assert torch.equal(previous, want)
            # Multiple transactions: do not double-shift a positive prefix; restore zero.
            c.state.copy_(c.initial)
            c.prev.fill_(1)
            c.fresh.zero_()
            expected = c.initial.clone()
            for iteration, k in enumerate((width, 0, 1, width // 2, width, 0)):
                oldprev = c.prev.cpu().tolist()
                c.x = (torch.randn_like(c.x) * 0.1).bfloat16()
                c.xd.copy_(c.x)
                checked(c.capture, c.snaperr)
                checked(c.plan, c.err)
                for row, first, slot in ((0, 0, 0), (2, width, 2)):
                    hist = expected[slot, :, oldprev[row] - 1 : oldprev[row] - 1 + c.H]
                    want, _ = convolution(c.x[first : first + width], c.weight, hist, d)
                    check_bf(c.out[first : first + width], want)
                    if k:
                        expected[slot] = encoded_history(
                            c.x[first : first + width], hist, 7
                        )
                c.counts.fill_(k)
                checked(c.restore, c.restoreerr)
                if k:
                    c.prev.fill_(k)
                assert torch.equal(c.state.cpu(), expected)
            assert (c.backing[:, :, c.c :] == 13).all().item()
            record(
                "shortconv-prefix",
                dilation=d,
                width=width,
                prefixes=width + 1,
                multiround=6,
                slot_zero="VALID",
                stride="position-major",
            )
        for length in (1, 5, 17, 65):
            c = ConvCase(lib, d, length, mode="prefill")
            checked(c.plan, c.err)
            want, state = convolution(c.x[:length], c.weight, c.initial[0, :, : c.H], d)
            check_bf(c.out[:length], want)
            assert torch.equal(c.state[0, :, : c.H].cpu(), state)
            assert not c.state[[0, 2], :, c.H :].any().item()
            record("shortconv-prefill", dilation=d, length=length)
        for plant in ("slot", "prefix", "conflict", "range", "overlap"):
            c = ConvCase(lib, d, 4)
            before = c.state.clone()
            if plant == "slot":
                c.slots[0] = -2
            if plant == "prefix":
                c.prev[0] = 0
            if plant == "conflict":
                c.slots[2] = 0
            if plant == "range":
                c.starts[1] = -1
            if plant == "overlap":
                c.starts.copy_(torch.tensor([0, 4, 1, 5], dtype=torch.int32))
            c.plan()
            torch.cuda.synchronize()
            assert c.err[0].item() != 0 and torch.equal(c.state[0], before[0])
            record("shortconv-negative", dilation=d, plant=plant, result="EXPECTED_RED")


def hash_tests(lib):
    geom = ple_geometry(vocab_size=151936, base_size=100, dense_layer_ordinal=7)
    T = 32
    token = torch.tensor([1, 2, 9, 4, 9, 9, 3, 7] * 4, dtype=torch.int64)
    history = torch.tensor([[3, 7], [9, 2], [1, 9], [9, 9]], dtype=torch.int64)
    starts = gpu(torch.tensor([0, 5, 5, 13, 24], dtype=torch.int32))
    ns = gpu(torch.tensor([4], dtype=torch.int32))
    nt = gpu(torch.tensor([24], dtype=torch.int32))
    out = torch.empty(T, 16, dtype=torch.int64, device=DEVICE)
    ids = torch.empty(T, dtype=torch.int64, device=DEVICE)
    err = errors(T)
    td, hd = gpu(token), gpu(history)
    rows = ServingRows(T, nt, starts, None, ns)
    params = [
        gpu(torch.tensor(v, dtype=torch.int64))
        for v in (geom.multipliers, geom.prime_sizes, geom.table_offsets)
    ]
    plan = PreparedPLEHash(
        lib,
        tokens=td,
        history=hd,
        multipliers=params[0],
        primes=params[1],
        offsets=params[2],
        hashes=out,
        request_ids=ids,
        errors=err,
        rows=rows,
        geometry=geom,
        eos=9,
    )
    checked(plan, err)
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        plan()
    graph.instantiate()
    stats = graph_stats(lib, graph)
    memory = torch.cuda.memory_allocated()
    for n in range(1, 5):
        ns.fill_(n)
        graph.replay()
        torch.cuda.synchronize()
        assert not err.any().item()
        want, mapping = hash_packed(
            token, [0, 5, 5, 13, 24], history, geom, 9, live_requests=n, live_tokens=24
        )
        assert torch.equal(out.cpu(), want) and torch.equal(ids.cpu(), mapping)
        assert torch.equal(hd.cpu(), history)
        assert torch.cuda.memory_allocated() == memory
    measure("packed-ple-hash-graph", graph.replay)
    params[1][0] = 0
    plan()
    torch.cuda.synchronize()
    assert err.ne(0).all().item() and out.eq(-1).all().item()
    record(
        "hash-graph",
        replays=4,
        geometry="pinned-SplitMix64",
        eos="history/current/packed",
        negative="zero-prime",
        graph=stats,
        external_scratch_bytes=0,
    )


def graph_stats(lib, graph):
    prefix = "hggc" if lib.target.startswith("ppu") else "cuda"
    get_nodes = getattr(lib.lib, prefix + "GraphGetNodes")
    get_type = getattr(lib.lib, prefix + "GraphNodeGetType")
    get_nodes.argtypes = [C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_size_t)]
    get_type.argtypes = [C.c_void_p, C.POINTER(C.c_int)]
    count = C.c_size_t()
    assert get_nodes(C.c_void_p(graph.raw_cuda_graph()), None, C.byref(count)) == 0
    nodes = (C.c_void_p * count.value)()
    assert get_nodes(C.c_void_p(graph.raw_cuda_graph()), nodes, C.byref(count)) == 0
    histogram = {}
    for node in nodes:
        kind = C.c_int()
        assert get_type(node, C.byref(kind)) == 0
        histogram[kind.value] = histogram.get(kind.value, 0) + 1
    # CUDA-compatible enum: kernel0, memcpy1, memset2, host3, child4,
    # memalloc10/memfree11. Inspect the captured graph, not a source estimate.
    assert not any(histogram.get(k, 0) for k in (3, 4, 10, 11)), histogram
    return dict(
        nodes=count.value,
        kernels=histogram.get(0, 0),
        types=histogram,
        host_callbacks=0,
        allocation_nodes=0,
    )


class GdnChain:
    def __init__(self, lib):
        self.c = GdnCase(lib, 8, "verify")
        c = self.c
        R, T, Cc = 3, 24, 10240
        self.conv_initial = torch.randn(3, Cc, 10).bfloat16() * 0.03
        self.conv_state = gpu(self.conv_initial)
        self.weight = torch.randn(Cc, 4).bfloat16() * 0.1
        self.conv_slots = gpu(torch.tensor([0, 1, -1], dtype=torch.int64))
        self.prev = gpu(torch.ones(R, dtype=torch.int32))
        self.fresh = gpu(torch.zeros(R, dtype=torch.bool))
        self.nt = gpu(torch.tensor([T], dtype=torch.int32))
        self.conv_error = errors(R)
        self.rows = ServingRows(T, self.nt, c.starts, self.conv_slots, c.live)
        self.mixed = torch.empty_like(c.mixed)
        self.conv = PreparedShortConv(
            lib,
            x=c.mixed,
            weight=gpu(self.weight),
            state=self.conv_state,
            output=self.mixed,
            rows=self.rows,
            fresh=self.fresh,
            previous_accepted=self.prev,
            errors=self.conv_error,
            dilation=1,
        )
        self.epochs = gpu(torch.zeros(R, dtype=torch.int64))
        self.ready = torch.zeros_like(self.epochs)
        self.consumed = torch.zeros_like(self.epochs)
        # First null-slot verification must initialize stamps even though it
        # must preserve an already-outstanding window on later inactive runs.
        c.work["recorded_lengths"].fill_(123)
        c.work["recorded_slots"].fill_(123)
        self.selected = gpu(torch.zeros(R, dtype=torch.int32))
        ep = dict(
            epochs=self.epochs, ready_epochs=self.ready, consumed_epochs=self.consumed
        )
        self.verify = PreparedGDNVerifyV2(
            lib,
            **ep,
            upstream_errors=self.conv_error,
            mixed=self.mixed,
            a=c.ad,
            b=c.bd,
            a_log=c.plan.inputs[3],
            dt_bias=c.plan.inputs[4],
            state=c.state,
            starts=c.starts,
            live=c.live,
            initial_slots=c.slots,
            final_slots=c.slots,
            output=c.out,
            workspace=c.work,
            width=8,
            key_heads=16,
            value_heads=48,
        )
        self.commit = PreparedGDNCommitV2(
            lib,
            **ep,
            upstream_errors=c.work["errors"],
            state=c.state,
            slots=c.slots,
            consumed_prefix=c.counts,
            live=c.live,
            records=c.work,
            errors=c.commit_errors,
        )
        self.snapshot = torch.empty(R, Cc, 10, dtype=torch.bfloat16, device=DEVICE)
        self.stamps = torch.empty(R, dtype=torch.int64, device=DEVICE)
        self.snap_error = errors(R)
        self.restore_error = errors(R)
        args = dict(
            state=self.conv_state,
            snapshot=self.snapshot,
            stamps=self.stamps,
            consumed_prefix=c.counts,
            rows=self.rows,
            dilation=1,
        )
        self.capture = PreparedConvHistoryV1(lib, errors=self.snap_error, **args)
        self.restore = PreparedConvHistoryV1(
            lib, errors=self.restore_error, restore=True, **args
        )
        self.norm_gate = gpu(torch.randn(T, 48, 128).bfloat16())
        self.norm_weight = gpu(torch.ones(128))
        self.normalized = torch.empty_like(c.out)
        self.norm_error = errors(T)
        nrows = ServingRows(
            T, self.nt, c.starts, self.conv_slots, c.live, c.work["errors"]
        )
        self.norm = PreparedPointwise(
            lib,
            "gated_norm",
            groups=48,
            x=c.out,
            gate=self.norm_gate,
            weight=self.norm_weight,
            output=self.normalized,
            errors=self.norm_error,
            rows=nrows,
        )

    def run(self):
        self.epochs.add_(1)  # device counter node; no host .item() or Python policy
        self.capture()
        self.conv()
        self.verify()
        self.norm()
        self.c.counts.copy_(
            self.selected
        )  # device-to-device stand-in for acceptance output
        self.commit()
        self.restore()

    def reset(self):
        self.c.state.copy_(self.c.initial)
        self.conv_state.copy_(self.conv_initial)
        self.epochs.zero_()
        self.ready.zero_()
        self.consumed.zero_()
        self.prev.fill_(1)


def chain_tests(lib):
    a = GdnChain(lib)
    c = a.c
    a.run()
    torch.cuda.synchronize()
    assert (
        c.work["recorded_lengths"][2].item() == 0
        and c.work["recorded_slots"][2].item() == 0
    )
    a.reset()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        a.run()
    graph.instantiate()
    stats = graph_stats(lib, graph)
    state = c.initial.clone()
    conv = a.conv_initial.clone()
    previous = [1, 1, 1]
    memory = torch.cuda.memory_allocated()
    for iteration, count in enumerate((0, 1, 4, 8, 2, 0, 8, 3)):
        counts = [count, 8 - count, 0]
        a.selected.copy_(torch.tensor(counts, dtype=torch.int32))
        # Snapshot the actual device state, NOT the independent CPU recurrence
        # after earlier rounds. The latter is a numerical, not bitwise, oracle.
        before = c.state.detach().cpu().clone()
        conv_before = a.conv_state.detach().cpu().clone()
        expected_outputs = []
        for r, slot in ((0, 2), (1, 4)):
            h = conv[r, :, previous[r] - 1 : previous[r] + 2]
            raw = c.x[r * 8 : (r + 1) * 8]
            convolved, _ = convolution(raw, a.weight, h, 1)
            # Independent recurrence consumes convolution output and original logits.
            want, _, *_ = gdn_reference(
                convolved,
                c.a[r * 8 : (r + 1) * 8],
                c.b[r * 8 : (r + 1) * 8],
                c.alog,
                c.bias,
                state[slot],
                16,
            )
            expected_outputs.append(want)
            _, state[slot], *_ = gdn_reference(
                convolved,
                c.a[r * 8 : (r + 1) * 8],
                c.b[r * 8 : (r + 1) * 8],
                c.alog,
                c.bias,
                state[slot],
                16,
                counts[r],
            )
            if counts[r]:
                conv[r] = encoded_history(raw, h, 7)
                previous[r] = counts[r]
        graph.replay()
        torch.cuda.synchronize()
        for err in (
            a.snap_error,
            a.conv_error,
            c.work["errors"],
            a.norm_error,
            c.commit_errors,
            a.restore_error,
        ):
            assert not err.any().item(), err.cpu().tolist()
        # Native convolution/rsqrt differ only at admitted BF16 boundaries;
        # recurrence arithmetic itself is tested separately by serving_device.
        close(c.state[2], state[2])
        close(c.state[4], state[4])
        for r, want in enumerate(expected_outputs):
            close(c.out[r * 8 : (r + 1) * 8], want)
        norm_want, _ = gated(
            c.out[:16].cpu(), a.norm_gate[:16].cpu(), a.norm_weight.cpu()
        )
        check_bf(a.normalized[:16], norm_want)
        assert not a.normalized[16:].any().item()
        assert torch.equal(a.conv_state.cpu(), conv)
        for r, slot in ((0, 2), (1, 4)):
            if counts[r] == 0:
                label = f"zero-prefix replay={iteration} request={r} slot={slot}"
                assert_raw_unchanged(
                    c.state[slot], before[slot], label=label + " recurrent"
                )
                assert_raw_unchanged(
                    a.conv_state[r], conv_before[r], label=label + " convolution"
                )
                print(
                    "[FlashNext zero-prefix] " + label + " raw_bitdiff=0/PASS",
                    flush=True,
                )
        a.prev.copy_(torch.tensor(previous, dtype=torch.int32))
        assert torch.cuda.memory_allocated() == memory
    before = c.state.clone()
    a.commit()
    torch.cuda.synchronize()
    assert c.commit_errors[:2].eq(5).all().item() and torch.equal(c.state, before)
    # An outstanding record survives a temporarily inactive replay. It cannot
    # be consumed under a different slot, even when the requested prefix is zero.
    a.reset()
    a.epochs.fill_(1)
    a.capture()
    a.conv()
    a.verify()
    torch.cuda.synchronize()
    ready = a.ready.clone()
    lengths = c.work["recorded_lengths"].clone()
    stamps = c.work["recorded_slots"].clone()
    c.live.zero_()
    a.verify()
    torch.cuda.synchronize()
    assert torch.equal(a.ready, ready) and torch.equal(
        c.work["recorded_lengths"], lengths
    )
    assert torch.equal(c.work["recorded_slots"], stamps)
    c.live.fill_(3)
    c.slots[0] = 3
    c.counts.zero_()
    a.commit()
    torch.cuda.synchronize()
    assert c.commit_errors[0].item() == 5
    c.slots[0] = 2
    a.commit()
    torch.cuda.synchronize()
    assert not c.commit_errors[0].item() and a.consumed[0].item() == 1
    # Invalid ShortConv slot: GDN consumes its error, not a successful zero output.
    a.conv_slots[0] = -2
    a.selected.fill_(1)
    before = c.state.clone()
    graph.replay()
    torch.cuda.synchronize()
    assert (
        a.conv_error[0].item() == 2
        and c.work["errors"][0].item() == 2
        and c.commit_errors[0].item() == 2
    )
    assert torch.equal(c.state[2], before[2]) and not a.normalized[:8].any().item()
    a.conv_slots[0] = 0
    a.reset()
    a.selected.copy_(torch.tensor([1, 1, 0], dtype=torch.int32))
    measure("shortconv-gdn-verify-norm-commit-graph", graph.replay, reset=a.reset)
    assert not c.commit_errors.any().item()
    retained = sum(
        t.numel() * t.element_size()
        for n, t in c.work.items()
        if n in ("delta", "keys", "decays", "recorded_slots", "recorded_lengths")
    )
    record(
        "gdn-composed-graph",
        replays=8,
        metadata="device-count/epochs",
        retained_bytes=retained,
        graph=stats,
        stale_commit="EXPECTED_RED",
        upstream_error="PROPAGATED_NOT_CLEARED",
        scope="OPERATOR_HARNESS_NOT_MODEL_MTP",
    )


def checked_gdn_tests(lib):
    for mode, width in (("decode", 1), ("prefill", 65)):
        c = GdnCase(lib, width, mode)
        dep = gpu(torch.tensor([2, 0, 0], dtype=torch.int32))
        checked_plan = PreparedGDNCheckedV2(
            lib,
            upstream_errors=dep,
            mixed=c.mixed,
            a=c.ad,
            b=c.bd,
            a_log=c.plan.inputs[3],
            dt_bias=c.plan.inputs[4],
            state=c.state,
            starts=c.starts,
            live=c.live,
            initial_slots=c.slots,
            final_slots=c.final_slots,
            output=c.out,
            workspace=c.work,
            width=width,
            mode=mode,
            key_heads=16,
            value_heads=48,
        )
        checked_plan()
        torch.cuda.synchronize()
        assert c.work["errors"].cpu().tolist() == [2, 0, 0]
        assert torch.equal(c.state[2].cpu(), c.initial[2])
        # A bad request does not suppress a different valid request.
        close(
            c.state[4],
            c.golden(1)[1],
            chunk=(mode == "prefill" and lib.target == "ppu15"),
        )
        dep.zero_()
        c.state.copy_(c.initial)
        checked_plan()
        c.ok()
        for r in (0, 1):
            close(
                c.out[r * width : (r + 1) * width],
                c.golden(r)[0],
                chunk=(mode == "prefill" and lib.target == "ppu15"),
            )
        record(
            "gdn-checked-" + mode,
            upstream_error="PROPAGATED/STATE_UNTOUCHED",
            other_request="CONTINUES",
        )


def independent_streams(lib):
    a, b = ConvCase(lib, 3, 8), ConvCase(lib, 3, 8)
    sa, sb = torch.cuda.Stream(), torch.cuda.Stream()
    current = torch.cuda.current_stream()
    sa.wait_stream(current)
    sb.wait_stream(current)
    with torch.cuda.stream(sa):
        a.plan()
    with torch.cuda.stream(sb):
        b.plan()
    current.wait_stream(sa)
    current.wait_stream(sb)
    torch.cuda.synchronize()
    for c in (a, b):
        assert not c.err.any().item()
        want, _ = convolution(c.x[:8], c.weight, c.initial[0, :, :9], 3)
        check_bf(c.out[:8], want)
    record("concurrent-instances", streams=2, storage="DISJOINT", state="INDEPENDENT")


def ple_hc_graph(lib):
    T, R, D = 16, 2, 2560
    torch.manual_seed(918)
    x = torch.randn(T, 4, D).bfloat16()
    key = torch.randn_like(x)
    value = torch.randn(T, 1, D).bfloat16()
    injection = torch.randn(T, 4).bfloat16()
    gain = torch.randn(4, D) * 0.1 + 1
    kw = torch.randn(4, D) * 0.1
    cw = torch.randn(4, D) * 0.1
    convw = torch.randn(4 * D, 4).bfloat16() * 0.1
    starts = gpu(torch.tensor([0, 4, 8], dtype=torch.int32))
    ns = gpu(torch.tensor([2], dtype=torch.int32))
    nt = gpu(torch.tensor([8], dtype=torch.int32))
    slots = gpu(torch.tensor([0, 1], dtype=torch.int64))
    rows = ServingRows(T, nt, starts, slots, ns)
    combined = torch.empty_like(gpu(x))
    query_norm = torch.empty_like(combined)
    key_norm = torch.empty_like(combined)
    gated_out = torch.empty_like(combined)
    normed = torch.empty_like(combined)
    convout = torch.empty(T, 4 * D, dtype=torch.bfloat16, device=DEVICE)
    contribution = torch.empty_like(combined)
    state = torch.zeros(3, 4 * D, 16, dtype=torch.bfloat16, device=DEVICE)
    error_vectors = [errors(T) for _ in range(5)] + [errors(R)]
    combine = PreparedPointwise(
        lib,
        "hc_combine_norm",
        x=gpu(x),
        gate=gpu(injection),
        value=gpu(value),
        weight=gpu(gain),
        combined=combined,
        output=query_norm,
        errors=error_vectors[0],
        rows=rows,
    )
    normk = PreparedPointwise(
        lib,
        "grouped_norm",
        x=gpu(key),
        weight=gpu(kw),
        output=key_norm,
        zero_centered=True,
        errors=error_vectors[1],
        rows=rows,
    )
    gate = PreparedPointwise(
        lib,
        "ple_gate",
        x=key_norm,
        gate=query_norm,
        value=gpu(value),
        output=gated_out,
        errors=error_vectors[2],
        rows=rows,
    )
    normconv = PreparedPointwise(
        lib,
        "grouped_norm",
        x=gated_out,
        weight=gpu(cw),
        output=normed,
        zero_centered=True,
        errors=error_vectors[3],
        rows=rows,
    )
    conv = PreparedShortConv(
        lib,
        x=normed.flatten(1),
        weight=gpu(convw),
        state=state,
        output=convout,
        rows=rows,
        fresh=gpu(torch.ones(R, dtype=torch.bool)),
        previous_accepted=gpu(torch.ones(R, dtype=torch.int32)),
        errors=error_vectors[5],
        dilation=3,
    )
    finish = PreparedPointwise(
        lib,
        "ple_contribution",
        x=gated_out,
        value=convout,
        output=contribution,
        errors=error_vectors[4],
        rows=rows,
    )

    def sequence():
        combine()
        normk()
        gate()
        normconv()
        conv()
        finish()

    sequence()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        sequence()
    graph.instantiate()
    stats = graph_stats(lib, graph)
    memory = torch.cuda.memory_allocated()
    for iteration in range(8):
        a = 1 + iteration % 4
        b = 4 - iteration % 4
        starts.copy_(torch.tensor([0, a, a + b], dtype=torch.int32))
        nt.fill_(a + b)
        slots.copy_(torch.tensor([iteration % 2, 1 - iteration % 2], dtype=torch.int64))
        state.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert all(not e.any().item() for e in error_vectors)
        check_bf(combined[: a + b], hc_combine(x, value, injection)[: a + b])
        check_bf(query_norm[: a + b], grouped(combined.cpu(), gain)[: a + b])
        check_bf(key_norm[: a + b], grouped(key, kw, zero=True)[: a + b])
        # Check each fused seam against the actual upstream BF16 tensor.
        want_gate, _ = ple_gate(key_norm.cpu(), query_norm.cpu(), value)
        check_bf(gated_out[: a + b], want_gate[: a + b])
        check_bf(normed[: a + b], grouped(gated_out.cpu(), cw, zero=True)[: a + b])
        for first, end in ((0, a), (a, a + b)):
            want, _ = convolution(
                normed[first:end].cpu().flatten(1),
                convw,
                torch.zeros(4 * D, 9).bfloat16(),
                3,
            )
            check_bf(convout[first:end], want)
        check_bf(
            contribution[: a + b],
            (
                gated_out[: a + b].cpu() + convout[: a + b].cpu().view(-1, 4, D)
            ).bfloat16(),
            0,
        )
        assert not contribution[a + b :].any().item() and not state[2].any().item()
        assert torch.cuda.memory_allocated() == memory
    measure("hc-combine-norm-ple-gate-dilated-conv-graph", graph.replay)
    record(
        "ple-hc-composed-graph",
        replays=8,
        graph=stats,
        metadata="lengths/slots",
        scope="projected-inputs-no-embedding-IO",
    )


def operator_performance(lib):
    x = gpu(torch.randn(8, 48, 128).bfloat16())
    gate = torch.randn_like(x)
    weight = gpu(torch.ones(128, dtype=torch.float32))
    output = torch.empty_like(x)
    err = errors(8)
    norm = PreparedPointwise(
        lib,
        "gated_norm",
        groups=48,
        x=x,
        gate=gate,
        weight=weight,
        output=output,
        errors=err,
        activation="silu",
    )
    c = ConvCase(lib, 3, 8, channels=10240)
    for name, plan, error, reset in (
        ("gated-norm", norm, err, None),
        ("ple-shortconv-d3", c.plan, c.err, lambda: c.state.copy_(c.initial)),
    ):
        if reset:
            reset()
        checked(plan, error)
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            plan()
        graph.instantiate()
        stats = graph_stats(lib, graph)
        measure(name + "-graph", graph.replay, reset=reset)
        storage = sum(t.numel() * t.element_size() for t in plan.buffers)
        record(
            "operator-resources",
            operator=name,
            graph=stats,
            external_scratch_bytes=0,
            writable_logical_bytes=storage,
            diagnostic_trace_bytes=0,
        )


def main():
    global DEVICE
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument(
        "--target", required=True, choices=("ppu10", "ppu15", "cuda_sm120")
    )
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    report = dict(
        target=args.target,
        cases=CASES,
        performance=PERF,
        scope="AUXILIARY_OPERATOR_ADMISSION_NOT_FULL_MODEL_EXPORT",
    )
    try:
        if not torch.cuda.is_available():
            report.update(
                status="SKIP", reason="device unavailable; no numerical execution"
            )
            print(json.dumps(report))
            return 3
        DEVICE = torch.device("cuda", torch.cuda.current_device())
        if args.target == "cuda_sm120" and torch.cuda.get_device_capability() != (
            12,
            0,
        ):
            raise RuntimeError("requires native SM120 validation, no SM90a reuse")
        report["device"] = str(torch.cuda.get_device_properties(DEVICE))
        report["torch"] = torch.__version__
        report["binary_sha256"] = hashlib.sha256(
            Path(args.library).read_bytes()
        ).hexdigest()
        report["target_admission"] = current_device_identity(args.target)
        lib = NativeLibrary(args.library, target=args.target)
        pointwise_tests(lib)
        conv_tests(lib)
        hash_tests(lib)
        checked_gdn_tests(lib)
        chain_tests(lib)
        ple_hc_graph(lib)
        independent_streams(lib)
        operator_performance(lib)
        report["status"] = "PASS"
        return 0
    except Exception as exc:
        report.update(status="FAIL", error=repr(exc))
        raise
    finally:
        args.results.parent.mkdir(parents=True, exist_ok=True)
        args.results.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    sys.exit(main())
