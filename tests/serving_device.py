"""Real-device serving admission, deliberately independent of any profiler.

No fake CUDA/PPU backend. References execute on CPU outside graph capture.
"""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from actlize_la.serving import (
    NativeLibrary,
    PreparedGDN,
    PreparedGDNCommit,
    PreparedQSA,
    PreparedConvHistory,
    gdn_workspace_specs,
    qsa_workspace_specs,
)
from serving_reference import (
    gdn_reference,
    replay_records,
    compressed_history,
    index_encode,
    select_reference,
    attention_reference,
)

torch.set_num_threads(1)
RESULTS = []


def record(name, **fields):
    entry = dict(case=name, status="PASS", **fields)
    RESULTS.append(entry)
    print("[serving device] " + json.dumps(entry, sort_keys=True), flush=True)


def allocate(specs, device):
    return {
        name: torch.empty(shape, dtype=dtype, device=device)
        for name, (shape, dtype) in specs.items()
    }


def close(got, want, *, chunk=False):
    got = got.detach().cpu().float()
    want = want.detach().cpu().float()
    assert torch.isfinite(got).all(), "nonfinite output/state"
    # BF16 output rounding and chunk residual's BF16 intermediate boundary
    # differ from FP32 recurrence. The original 2% relative-L2 gate is NOT
    # relaxed; incremental state has an additional tight absolute check.
    relative = float(
        torch.linalg.vector_norm(got - want)
        / torch.linalg.vector_norm(want).clamp_min(1e-12)
    )
    assert relative < (0.02 if chunk else 0.006), f"relative L2={relative}"
    if not chunk:
        torch.testing.assert_close(got, want, rtol=0.01, atol=2e-5)
    return relative


class GdnCase:
    def __init__(self, lib, width, mode, requests=3, hk=16, hv=48):
        torch.manual_seed(701 + width)
        self.lib, self.width, self.mode, self.hk, self.hv = lib, width, mode, hk, hv
        r, rows = requests, requests * width
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.x = torch.randn(rows, (2 * hk + hv) * 128).bfloat16()
        self.a = (torch.randn(rows, hv) - 1).bfloat16()
        self.b = torch.randn(rows, hv).bfloat16()
        self.alog = torch.linspace(-3, -1, hv)
        self.bias = torch.linspace(-0.2, 0.1, hv)
        self.initial = torch.randn(5, hv, 128, 128) * 0.02
        self.initial[0] = 0
        # Square state is deliberately asymmetric and strided by pool row.
        backing = torch.empty((5, hv * 16384 + 64), device=self.device)
        self.state = backing[:, : hv * 16384].view(5, hv, 128, 128)
        self.state.copy_(self.initial)
        projection = torch.empty(
            rows, self.x.shape[1] + 16, dtype=torch.bfloat16, device=self.device
        )
        self.mixed = projection[:, : self.x.shape[1]]
        self.mixed.copy_(self.x)
        ab = torch.empty(rows, 2 * hv, dtype=torch.bfloat16, device=self.device)
        self.ad = ab[:, :hv]
        self.bd = ab[:, hv:]
        self.ad.copy_(self.a)
        self.bd.copy_(self.b)
        self.starts = torch.arange(r + 1, dtype=torch.int32, device=self.device) * width
        self.live = torch.tensor([r], dtype=torch.int32, device=self.device)
        # View matches Flare state_indices[:,0], not a contiguous copied index.
        table = torch.zeros(r, 8, dtype=torch.int64, device=self.device)
        table[:, 0] = torch.tensor(([2, 4, 0] + [0] * r)[:r], device=self.device)
        self.slots = table[:, 0]
        self.final_slots = self.slots if mode != "prefill" else self.slots.clone()
        if mode == "prefill":
            self.slots[0] = 0  # zero -> nonzero final, plus nonzero continuation
        self.out = torch.full(
            (rows, hv, 128), -99, dtype=torch.bfloat16, device=self.device
        )
        self.checkpoint_slots = torch.zeros(r, dtype=torch.int64, device=self.device)
        self.checkpoint_offsets = torch.zeros(r, dtype=torch.int32, device=self.device)
        self.work = allocate(
            gdn_workspace_specs(r, width, hk, hv, mode=mode, target=lib.target),
            self.device,
        )
        self.plan = PreparedGDN(
            lib,
            mixed=self.mixed,
            a=self.ad,
            b=self.bd,
            a_log=self.alog.to(self.device),
            dt_bias=self.bias.to(self.device),
            state=self.state,
            starts=self.starts,
            live=self.live,
            initial_slots=self.slots,
            final_slots=self.final_slots,
            output=self.out,
            workspace=self.work,
            width=width,
            mode=mode,
            key_heads=hk,
            value_heads=hv,
            checkpoint_slots=self.checkpoint_slots,
            checkpoint_offsets=self.checkpoint_offsets,
        )
        self.counts = torch.zeros(r, dtype=torch.int32, device=self.device)
        self.commit_errors = torch.empty(r, dtype=torch.int32, device=self.device)
        self.commit = None
        if mode == "verify":
            self.commit = PreparedGDNCommit(
                lib,
                state=self.state,
                slots=self.slots,
                consumed_prefix=self.counts,
                live=self.live,
                records=self.work,
                errors=self.commit_errors,
            )

    def golden(self, row, count=None, initial=None):
        first = int(self.starts[row].cpu())
        last = int(self.starts[row + 1].cpu())
        slot = int(self.slots[row].cpu())
        state = self.initial[slot] if initial is None else initial
        return gdn_reference(
            self.x[first:last],
            self.a[first:last],
            self.b[first:last],
            self.alog,
            self.bias,
            state,
            self.hk,
            count,
        )

    def ok(self):
        torch.cuda.synchronize()
        assert not self.work["errors"].any().item(), self.work["errors"].cpu().tolist()


def test_gdn(lib):
    c = GdnCase(lib, 1, "decode")
    c.starts.copy_(torch.tensor([0, 1, 1, 1], dtype=torch.int32, device=c.device))
    c.slots.fill_(2)  # empty rows retain the active slot from a previous replay
    c.plan()
    c.ok()
    want, state, *_ = c.golden(0)
    close(c.out[:1], want)
    close(c.state[2], state)
    c.state.copy_(c.initial)
    c.starts.copy_(torch.tensor([0, 1, 2, 2], dtype=torch.int32, device=c.device))
    c.plan()  # same metadata, now both conflicting rows really consume a token
    torch.cuda.synchronize()
    assert c.work["errors"][:2].eq(3).all().item()
    assert torch.equal(c.state.cpu(), c.initial)
    record("gdn-empty-row-slot", empty="IGNORED", active_collision="EXPECTED_RED")

    for width in (1, 2, 4, 8):
        c = GdnCase(lib, width, "verify")
        c.plan()
        c.ok()
        assert torch.equal(c.state.cpu(), c.initial), "verify wrote committed state"
        for row in (0, 1):
            close(c.out[row * width : (row + 1) * width], c.golden(row)[0])
        assert not c.out[2 * width :].any().item()
        for count in range(width + 1):
            c.state.copy_(c.initial)
            c.plan()
            c.counts.copy_(
                torch.tensor(
                    [count, width - count, 0], device=c.device, dtype=torch.int32
                )
            )
            c.commit()
            torch.cuda.synchronize()
            assert not c.commit_errors.any().item()
            for row, accepted in ((0, count), (1, width - count)):
                slot = int(c.slots[row].cpu())
                close(c.state[slot], c.golden(row, accepted)[1])
                # Exact replay of GPU-generated compact records isolates commit
                # from independent exp/rsqrt/reduction order differences.
                d = c.work["delta"][row].cpu().transpose(0, 1)
                k = c.work["keys"][row].cpu().transpose(0, 1)
                decay = c.work["decays"][row].cpu().transpose(0, 1)
                assert torch.equal(
                    c.state[slot].cpu(),
                    replay_records(c.initial[slot], d, k, decay, accepted),
                )
            assert torch.equal(c.state[[0, 1, 3]].cpu(), c.initial[[0, 1, 3]])
        # Wrong round binding and an out-of-range accepted prefix are errors,
        # not truncated counts or a stale-state commit.
        for plant in ("slot", "count", "duplicate"):
            c.state.copy_(c.initial)
            c.slots.copy_(torch.tensor([2, 4, 0], device=c.device))
            c.plan()
            c.counts.copy_(torch.tensor([1, 1, 0], dtype=torch.int32, device=c.device))
            if plant == "slot":
                c.slots[0] = 1
            if plant == "count":
                c.counts[0] = width + 1
            if plant == "duplicate":
                c.slots[1] = 2
            c.commit()
            torch.cuda.synchronize()
            assert c.commit_errors[0].item() != 0
            assert torch.equal(c.state[2].cpu(), c.initial[2])
        record(
            "gdn-verify-prefix",
            width=width,
            prefixes=width + 1,
            head_mapping="16:48",
            negative_controls=3,
        )

    for length in (63, 64, 65, 127, 128, 129, 2048):
        c = GdnCase(lib, length, "prefill")
        c.plan()
        c.ok()
        for row in (0, 1):
            want, state, *_ = c.golden(row)
            close(c.out[row * length : (row + 1) * length], want, chunk=True)
            close(c.state[int(c.final_slots[row].cpu())], state, chunk=True)
        assert torch.equal(c.state[0].cpu(), c.initial[0])
        if length in (65, 129):
            c.slots.copy_(c.final_slots)
            before = c.state.cpu().clone()
            c.plan()
            c.ok()
            for row in (0, 1):
                slot = int(c.slots[row].cpu())
                want, state, *_ = c.golden(row, initial=before[slot])
                close(c.out[row * length : (row + 1) * length], want, chunk=True)
                close(c.state[slot], state, chunk=True)
        # Continue through the actual prepared decode entry, not chunk(T=1).
        d = GdnCase(lib, 1, "decode")
        d.state.copy_(c.state)
        before = d.state.cpu().clone()
        d.plan()
        d.ok()
        for row in (0, 1):
            want, state, *_ = d.golden(row, initial=before[int(d.slots[row].cpu())])
            close(d.out[row : row + 1], want)
            close(d.state[int(d.slots[row].cpu())], state)
        record(
            "gdn-prefill-decode",
            length=length,
            initial="zero+asymmetric-nonzero",
            native_chunk=lib.target == "ppu15",
        )


def test_gdn_graph(lib):
    c = GdnCase(lib, 8, "verify")
    c.counts.copy_(torch.tensor([4, 3, 0], device=c.device, dtype=torch.int32))
    for _ in range(2):
        c.plan()
        c.commit()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        c.plan()
        c.commit()
    memory = torch.cuda.memory_allocated()
    fingerprints = []
    for iteration in range(8):
        c.state.copy_(c.initial)
        live = 1 + iteration % 2
        slots = [1 + iteration % 2, 3 + iteration % 2, 0]
        c.slots.copy_(torch.tensor(slots, device=c.device, dtype=torch.int64))
        l0 = iteration + 1
        l1 = 8 - iteration
        c.starts.copy_(
            torch.tensor(
                [0, l0, l0 + l1, l0 + l1 + 1], device=c.device, dtype=torch.int32
            )
        )
        lengths = [iteration, min(l1, 3), 0]
        c.counts.copy_(torch.tensor(lengths, device=c.device, dtype=torch.int32))
        c.live.fill_(live)
        graph.replay()
        c.ok()
        assert not c.commit_errors.any().item()
        for row in range(live):
            close(c.state[slots[row]], c.golden(row, lengths[row])[1])
        untouched = [slot for slot in range(5) if slot not in slots[:live]]
        assert torch.equal(c.state[untouched].cpu(), c.initial[untouched])
        assert torch.cuda.memory_allocated() == memory, (
            "graph replay allocated device memory"
        )
    # Identical inputs/start state repeated eight times must be raw-bit stable.
    for _ in range(8):
        c.state.copy_(c.initial)
        graph.replay()
        torch.cuda.synchronize()
        fingerprints.append(hashlib.sha256(c.state.cpu().numpy().tobytes()).hexdigest())
    assert len(set(fingerprints)) == 1
    # Custom-op fake/mutation tracing gate: compiled wrapper must not elide
    # the native call as a no-return pure Python function.
    compiled = torch.compile(c.plan, backend="eager", fullgraph=True)
    c.state.copy_(c.initial)
    c.out.fill_(float("nan"))
    compiled()
    c.ok()
    first, end = int(c.starts[0].cpu()), int(c.starts[1].cpu())
    close(c.out[first:end], c.golden(0)[0])
    record(
        "gdn-graph",
        replays=8,
        changed="slots/live/counts/starts",
        stable=True,
        memory="STABLE",
        torch_compile="PASS",
    )


def test_prefill_graph(lib):
    c = GdnCase(lib, 65, "prefill")
    c.plan()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        c.plan()
    memory = torch.cuda.memory_allocated()
    for iteration, length in enumerate((63, 64, 65, 64, 63, 65, 64, 63)):
        c.state.copy_(c.initial)
        c.starts.copy_(
            torch.tensor(
                [0, length, length + 64, length + 64],
                dtype=torch.int32,
                device=c.device,
            )
        )
        c.live.fill_(2)
        c.slots[0] = 0 if iteration % 2 else 2
        graph.replay()
        c.ok()
        for row in (0, 1):
            first = int(c.starts[row].cpu())
            end = int(c.starts[row + 1].cpu())
            want, state, *_ = c.golden(row)
            close(c.out[first:end], want, chunk=True)
            close(c.state[int(c.final_slots[row].cpu())], state, chunk=True)
        assert torch.cuda.memory_allocated() == memory
    before = c.state.clone()
    c.checkpoint_slots[0] = 1
    graph.replay()
    torch.cuda.synchronize()
    assert c.work["errors"][0].item() == 7
    assert torch.equal(c.state[2], before[2])
    record(
        "prefill-graph",
        replays=8,
        boundaries="63/64/65",
        checkpoint_negative="EXPECTED_RED",
    )


def test_shared_scratch(lib):
    c = GdnCase(lib, 1, "decode")
    d = GdnCase(lib, 1, "decode")
    # Bind the next layer to the same temporary buffers, not the same state
    # or output. Captured calls are ordered on the same current stream.
    shared = PreparedGDN(
        lib,
        mixed=d.mixed,
        a=d.ad,
        b=d.bd,
        a_log=d.plan.inputs[3],
        dt_bias=d.plan.inputs[4],
        state=d.state,
        starts=d.starts,
        live=d.live,
        initial_slots=d.slots,
        final_slots=d.final_slots,
        output=d.out,
        workspace=c.work,
        width=1,
        mode="decode",
    )
    c.plan()
    shared()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        c.plan()
        shared()
    c.state.copy_(c.initial)
    d.state.copy_(d.initial * 0.5)
    expected_c = c.initial.clone()
    expected_d = d.initial * 0.5
    for _ in range(8):
        graph.replay()
        for row in (0, 1):
            slot = int(c.slots[row].cpu())
            expected_c[slot] = c.golden(row, initial=expected_c[slot])[1]
            expected_d[slot] = d.golden(row, initial=expected_d[slot])[1]
    torch.cuda.synchronize()
    close(c.state, expected_c)
    close(d.state, expected_d)
    record("multi-layer-shared-scratch", layers=2, replays=8, state_alias="DISJOINT")


class QsaCase:
    def __init__(
        self, lib, context, width=8, *, ties=False, rope_bf16=False, page_size=64
    ):
        torch.manual_seed(1700 + context)
        self.lib, self.context, self.width = lib, context, width
        self.device = torch.device("cuda", torch.cuda.current_device())
        total = context + 2 * width + 16
        self.page_size = page_size
        pages = (total + page_size - 1) // page_size + 2
        self.raw = torch.randn(total, 128).bfloat16()
        self.key = torch.randn(total, 2, 256).bfloat16()
        self.value = torch.randn(total, 2, 256).bfloat16()
        self.query = torch.randn(total, 24, 256).bfloat16()
        self.iq = (
            torch.zeros(total, 4, 128) if ties else torch.randn(total, 4, 128)
        ).bfloat16()
        self.qw = torch.randn(128) * 0.1
        self.kw = torch.randn(128) * 0.1
        self.rope = torch.arange(total)[:, None].expand(-1, 3).clone()
        phase = torch.arange(total + 32)[:, None] * torch.exp(
            -torch.arange(32)[None, :] / 5
        )
        self.co = phase.cos()
        self.si = phase.sin()
        if rope_bf16:
            self.co = self.co.bfloat16()
            self.si = self.si.bfloat16()
        perm = torch.arange(pages).roll(1)
        dev = self.device
        b = torch.bfloat16
        i = torch.int32
        l = torch.int64
        self.caches = {
            "key_cache": torch.zeros(pages, page_size, 2, 256, dtype=b, device=dev),
            "value_cache": torch.zeros(pages, page_size, 2, 256, dtype=b, device=dev),
            "compressed_cache": torch.zeros(
                pages, page_size // 4, 128, dtype=b, device=dev
            ),
            "raw_ring": torch.zeros(3, 16, 128, dtype=b, device=dev),
            "raw_tags": torch.full((3, 16), -1, dtype=l, device=dev),
            "raw_rope": torch.full((3, 16, 3), -1, dtype=l, device=dev),
            "anchors": torch.full((3,), -1, dtype=l, device=dev),
        }
        old = compressed_history(
            self.raw[:context], self.kw, self.rope[:context], self.co, self.si
        )
        for t in range(context):
            self.caches["key_cache"][perm[t // page_size], t % page_size] = self.key[
                t
            ].to(dev)
            self.caches["value_cache"][perm[t // page_size], t % page_size] = (
                self.value[t].to(dev)
            )
        for group in range(context // 4):
            self.caches["compressed_cache"][
                perm[group // (page_size // 4)], group % (page_size // 4)
            ] = old[group].to(dev)
        for t in range(max(0, context - 16), context):
            self.caches["raw_ring"][1, t % 16] = self.raw[t].to(dev)
            self.caches["raw_tags"][1, t % 16] = t
            self.caches["raw_rope"][1, t % 16] = self.rope[t].to(dev)
        self.caches["anchors"][1] = context - 1
        rows = width + 1
        self.inputs = {
            "query": torch.zeros(rows, 24, 256, dtype=b, device=dev),
            "new_key": torch.zeros(rows, 2, 256, dtype=b, device=dev),
            "new_value": torch.zeros(rows, 2, 256, dtype=b, device=dev),
            "index_query": torch.zeros(rows, 4, 128, dtype=b, device=dev),
            "raw_key": torch.zeros(rows, 128, dtype=b, device=dev),
            "query_weight": self.qw.to(dev),
            "key_weight": self.kw.to(dev),
            "cosine": self.co.to(dev),
            "sine": self.si.to(dev),
            "request_ids": torch.tensor([0] * width + [-1], dtype=l, device=dev),
            "positions": torch.full((rows,), -1, dtype=l, device=dev),
            "rope_positions": torch.zeros(rows, 3, dtype=l, device=dev),
            "raw_slots": torch.tensor([1, -1], dtype=l, device=dev),
            "slot_mapping": torch.full((rows,), -1, dtype=l, device=dev),
            "starts": torch.tensor([0, width, rows], dtype=i, device=dev),
            "live": torch.tensor([2], dtype=i, device=dev),
            "lengths": torch.tensor([context + width, 0], dtype=i, device=dev),
            "accepted_previous": torch.tensor([1, 0], dtype=i, device=dev),
            "prefilling": torch.tensor([False, False], device=dev),
            "main_table": torch.stack((perm, torch.full_like(perm, -1))).int().to(dev),
            "compressed_table": torch.stack((perm, torch.full_like(perm, -1)))
            .int()
            .to(dev),
        }
        self.perm = perm
        self.work = allocate(qsa_workspace_specs(rows, 2), dev)
        self.plan = PreparedQSA(
            lib, inputs=self.inputs, caches=self.caches, workspace=self.work
        )
        self.set_interval(context, 1)

    def set_interval(self, first, accepted):
        w = self.width
        dev = self.device
        rows = slice(first, first + w)
        for dst, src in (
            ("query", self.query),
            ("new_key", self.key),
            ("new_value", self.value),
            ("index_query", self.iq),
            ("raw_key", self.raw),
        ):
            self.inputs[dst][:w].copy_(src[rows])
        self.inputs["positions"][:w].copy_(torch.arange(first, first + w, device=dev))
        self.inputs["rope_positions"][:w].copy_(self.rope[rows])
        self.inputs["lengths"][0] = first + w
        self.inputs["accepted_previous"][0] = accepted
        p = torch.arange(first, first + w)
        self.inputs["slot_mapping"][:w].copy_(
            self.perm[p // self.page_size] * self.page_size + p % self.page_size
        )

    def check(self, first, inactive=True):
        torch.cuda.synchronize()
        assert not self.work["errors"].any().item(), self.work["errors"].cpu().tolist()
        comp = compressed_history(
            self.raw[: first + self.width],
            self.kw,
            self.rope[: first + self.width],
            self.co,
            self.si,
        )
        encoded = index_encode(
            self.iq[first : first + self.width],
            self.qw,
            self.rope[first : first + self.width],
            self.co,
            self.si,
        )
        for t in range(self.width):
            pos = first + t
            selected, groups, scores = select_reference(
                encoded[t], comp, pos, first + self.width
            )
            actual = self.work["selected"][t].cpu()
            assert torch.equal(actual, selected), (
                f"QSA selection differs: position={pos}, bad={int((actual != selected).sum())}"
            )
            close(
                self.work["output"][t],
                attention_reference(self.query[pos], self.key, self.value, selected),
            )
        if inactive:
            assert not self.work["output"][-1].any().item(), (
                "inactive QSA output not zero"
            )


def test_qsa(lib):
    c = QsaCase(lib, 63)
    c.inputs["starts"][-1] = c.width
    c.inputs["raw_slots"][1] = 1
    c.inputs["main_table"][1].copy_(c.inputs["main_table"][0])
    c.inputs["compressed_table"][1].copy_(c.inputs["compressed_table"][0])
    c.inputs["lengths"][1] = c.inputs["lengths"][0]
    c.plan()
    c.check(63)
    record("qsa-empty-row-alias", stale_slot_and_pages="IGNORED")

    for context in (0, 127, 128, 129):
        c = QsaCase(lib, context, width=1, page_size=128)
        c.plan()
        c.check(context)
        record("qsa-single-token", context=context, page_size=128)

    for context in (0, 1, 2, 3, 63, 64, 65, 2048, 2051, 4096):
        c = QsaCase(lib, context, ties=context >= 2048)
        c.plan()
        c.check(context)
        record(
            "qsa-page-compression-tail",
            context=context,
            width=8,
            heads="24:2",
            selection_capacity=2051,
        )
    c = QsaCase(lib, 63, rope_bf16=True)
    c.plan()
    c.check(63)
    record("qsa-bf16-rope-rounding")
    c = QsaCase(lib, 63)
    # A second active request has disjoint writable pages and a different
    # context length. Its single-token attention must equal its value exactly.
    c.inputs["request_ids"][-1] = 1
    c.inputs["positions"][-1] = 0
    c.inputs["raw_slots"][1] = 2
    c.inputs["lengths"][1] = 1
    c.inputs["accepted_previous"][1] = 1
    c.inputs["main_table"][1, 0] = c.perm[-1]
    c.inputs["compressed_table"][1, 0] = c.perm[-1]
    c.inputs["slot_mapping"][-1] = int(c.perm[-1]) * 64
    c.inputs["new_value"][-1].copy_(c.value[0])
    c.inputs["new_key"][-1].copy_(c.key[0])
    c.plan()
    c.check(63, inactive=False)
    assert torch.equal(c.work["output"][-1].cpu(), c.value[0].repeat_interleave(12, 0))
    record("qsa-two-requests", contexts="63+8/0+1", pages="disjoint")
    for accepted in (0, 1, 4, 8):
        c = QsaCase(lib, 63)
        c.plan()
        c.check(63)
        first = 63 + accepted
        # Replace rejected candidate values, not merely their position tags.
        c.raw[first : first + 8] *= -1
        c.key[first : first + 8] *= -1
        c.value[first : first + 8] *= -1
        c.set_interval(first, accepted)
        c.plan()
        c.check(first)
        record("qsa-rejected-continuation", consumed_prefix=accepted)
    for plant in ("page", "tag", "anchor", "duplicate-page"):
        c = QsaCase(lib, 63)
        if plant == "page":
            c.inputs["main_table"][0, 0] = -1
        if plant == "tag":
            c.caches["raw_tags"][1, 62 % 16] = -1
        if plant == "anchor":
            c.caches["anchors"][1] = 0
        if plant == "duplicate-page":
            c.inputs["main_table"][0, 1] = c.inputs["main_table"][0, 0]
        before = {n: t.clone() for n, t in c.caches.items()}
        c.plan()
        torch.cuda.synchronize()
        assert c.work["errors"][0].item() != 0
        assert all(torch.equal(c.caches[n], v) for n, v in before.items()), (
            f"invalid {plant} mutated cache"
        )
        record("qsa-negative", plant=plant, verdict="EXPECTED_RED")


def test_qsa_graph(lib):
    c = QsaCase(lib, 63)
    initial = {n: t.clone() for n, t in c.caches.items()}
    original_perm = c.perm.clone()
    c.plan()
    torch.cuda.synchronize()
    for n, t in initial.items():
        c.caches[n].copy_(t)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        c.plan()
    memory = torch.cuda.memory_allocated()
    for iteration in range(8):
        for n, t in initial.items():
            c.caches[n].copy_(t)
        c.perm = original_perm.roll(iteration % 2)
        c.inputs["main_table"][0].copy_(c.perm)
        c.inputs["compressed_table"][0].copy_(c.perm)
        for lp in range(len(c.perm)):
            for name in ("key_cache", "value_cache", "compressed_cache"):
                c.caches[name][c.perm[lp]].copy_(initial[name][original_perm[lp]])
        slot = 1 + iteration % 2
        c.inputs["raw_slots"][0] = slot
        for name in ("raw_ring", "raw_tags", "raw_rope", "anchors"):
            c.caches[name][slot].copy_(initial[name][1])
        c.set_interval(63, 1)
        graph.replay()
        c.check(63)
        first = 63 + iteration
        c.set_interval(first, iteration)
        graph.replay()
        c.check(first)
        assert torch.cuda.memory_allocated() == memory
    record(
        "qsa-graph",
        replays=16,
        changed="pages/slots/lengths/positions/accepted",
        memory="STABLE",
    )


def test_conv_transaction(lib):
    dev = torch.device("cuda", torch.cuda.current_device())
    state = torch.arange(5 * 32 * 10, device=dev).reshape(5, 32, 10).bfloat16()
    original = state.clone()
    slots = torch.tensor([2, 4, 0], device=dev, dtype=torch.int64)
    live = torch.tensor([3], device=dev, dtype=torch.int32)
    counts = torch.tensor([0, 4, 0], device=dev, dtype=torch.int32)
    snapshot = torch.empty(3, 32, 10, device=dev, dtype=torch.bfloat16)
    stamps = torch.empty(3, device=dev, dtype=torch.int64)
    errors = torch.empty(3, device=dev, dtype=torch.int32)
    args = dict(
        state=state,
        snapshot=snapshot,
        slots=slots,
        recorded_slots=stamps,
        live=live,
        consumed_prefix=counts,
        errors=errors,
    )
    capture = PreparedConvHistory(lib, **args, restore=False)
    rollback = PreparedConvHistory(lib, **args, restore=True)
    capture()
    state[2:5].fill_(-3)
    rollback()
    torch.cuda.synchronize()
    assert not errors.any().item()
    assert torch.equal(state[2], original[2])
    assert (state[4] == -3).all().item(), (
        "positive prefix unexpectedly canonicalized twice"
    )
    assert torch.equal(state[0], original[0])
    record(
        "shortconv-zero-consumption-transport",
        arithmetic="REUSED_BY_CALLER",
        positive_prefix="DEFERRED_AS_FLARE",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument(
        "--target", required=True, choices=("ppu10", "ppu15", "cuda_sm120")
    )
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    report = dict(
        target=args.target,
        scope="OPERATOR_NUMERICS_AND_CAPTURE_NOT_FULL_MODEL_ADMISSION",
        cases=RESULTS,
    )
    try:
        if not torch.cuda.is_available():
            report.update(
                status="SKIP",
                reason="no CUDA-compatible device; execution NOT VERIFIED",
            )
            print(json.dumps(report), flush=True)
            return 3
        report["device"] = str(
            torch.cuda.get_device_properties(torch.cuda.current_device())
        )
        report["torch"] = torch.__version__
        report["library_sha256"] = hashlib.sha256(
            Path(args.library).read_bytes()
        ).hexdigest()
        lib = NativeLibrary(args.library, target=args.target)
        if args.target == "cuda_sm120" and torch.cuda.get_device_capability() != (
            12,
            0,
        ):
            raise RuntimeError(
                "SM120 validation requires an SM120 device, not SM90a or another architecture"
            )
        test_gdn(lib)
        test_gdn_graph(lib)
        test_prefill_graph(lib)
        test_shared_scratch(lib)
        test_conv_transaction(lib)
        test_qsa(lib)
        test_qsa_graph(lib)
        report["status"] = "PASS"
        print(
            f"[serving device] PASS cases={len(RESULTS)} results={args.results}",
            flush=True,
        )
        return 0
    except Exception as exc:
        report.update(status="FAIL", error=repr(exc))
        raise
    finally:
        args.results.parent.mkdir(parents=True, exist_ok=True)
        args.results.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    sys.exit(main())
