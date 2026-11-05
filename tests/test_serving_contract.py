import ctypes as C
import math
import random

import pytest
import torch

from actlize_la.serving import gdn_workspace_specs, qsa_workspace_specs
from actlize_la.serving._native import reject_aliases
from actlize_la.serving.gdn import _Gdn, _Commit, _Prefill
from actlize_la.serving.qsa import _Qsa
from actlize_la.serving.conv_history import _Conv
from serving_reference import (
    gdn_reference,
    replay_records,
    select_reference,
    merged_topk,
    index_encode,
)

torch.set_num_threads(1)


def test_target_device_mismatch_before_launch():
    from actlize_la.serving.device import validate_device_target

    validate_device_target("ppu10", "PPU-ZW810")
    validate_device_target("ppu15", "PPU-ZW890")
    validate_device_target("cuda_sm120", "RTX PRO 6000 Blackwell", (12, 0))
    for target, name, cap in (
        ("ppu15", "PPU-ZW810", None),
        ("ppu10", "PPU-ZW890", None),
        ("ppu15", "unknown PPU", None),
        ("ppu10", "ZW8100", None),
        ("cuda_sm120", "PPU-ZW810", (12, 0)),
        ("cuda_sm120", "H800", (9, 0)),
    ):
        with pytest.raises(ValueError):
            validate_device_target(target, name, cap)


@pytest.mark.parametrize("width", [1, 2, 4, 8])
def test_gdn_all_accepted_prefixes(width):
    torch.manual_seed(10 + width)
    hk, hv = 16, 48
    x = torch.randn(width, (2 * hk + hv) * 128).bfloat16()
    a = torch.randn(width, hv).bfloat16()
    b = torch.randn(width, hv).bfloat16()
    alog = torch.linspace(-3, -1, hv)
    bias = torch.linspace(-0.1, 0.2, hv)
    h = torch.randn(hv, 128, 128) * 0.02
    out, final, d, k, decay = gdn_reference(x, a, b, alog, bias, h, hk)
    for accepted in range(width + 1):
        _, want, *_ = gdn_reference(x, a, b, alog, bias, h, hk, accepted)
        # Independent Torch exp/sigmoid may select different CPU vector paths
        # for a shortened batch. Test arithmetic, not their final fp32 ulp.
        torch.testing.assert_close(
            replay_records(h, d, k, decay, accepted), want, rtol=1e-6, atol=1e-7
        )
    # Equal square extents MUST NOT conceal KV/VK reinterpretation.
    wrong = gdn_reference(x, a, b, alog, bias, h.transpose(-1, -2), hk)[0]
    assert not torch.equal(out, wrong), "transpose negative did not turn red"
    # Third value head belongs to key head 0, not key head 2.
    permuted = x.clone()
    permuted[:, hk * 128 : 2 * hk * 128] = (
        x[:, hk * 128 : 2 * hk * 128]
        .reshape(width, hk, 128)
        .roll(1, 1)
        .reshape(width, -1)
    )
    assert not torch.equal(gdn_reference(permuted, a, b, alog, bias, h, hk)[0], out)


@pytest.mark.parametrize("length", [63, 64, 65, 127, 128, 129, 2048])
def test_continuous_reference(length):
    torch.manual_seed(length)
    hk, hv = 1, 3
    x = torch.randn(length, 640).bfloat16()
    a = torch.full((length, hv), -2.0).bfloat16()
    b = torch.randn(length, hv).bfloat16()
    alog = torch.zeros(hv)
    bias = torch.zeros(hv)
    initial = torch.randn(hv, 128, 128) * 0.01
    full, state, *_ = gdn_reference(x, a, b, alog, bias, initial, hk)
    current = initial
    outs = []
    for first in range(0, length, 63):
        o, current, *_ = gdn_reference(
            x[first : first + 63],
            a[first : first + 63],
            b[first : first + 63],
            alog,
            bias,
            current,
            hk,
        )
        outs.append(o)
    assert torch.equal(torch.cat(outs), full)
    assert torch.equal(current, state)


def test_workspace_and_pool_aliases():
    specs = gdn_workspace_specs(4, 8, mode="verify", target="ppu15")
    assert specs["delta"][0] == (4, 48, 8, 128)
    assert (
        sum(
            math.prod(s) * torch.tensor([], dtype=d).element_size()
            for s, d in specs.values()
        )
        < 4 * 48 * 8 * 128 * 128 * 4
    )
    assert qsa_workspace_specs(8, 2)["selected"][0] == (8, 2051)
    pool = torch.empty(4, 100)
    # Different strided regions in one backing pool are legal.
    reject_aliases([pool[:, :20]], [pool[:, 20:40], pool[:, 40:]])
    with pytest.raises(ValueError, match="disjoint"):
        reject_aliases([pool[:, :21]], [pool[:, 20:40]])
    with pytest.raises(ValueError, match="invalid GDN"):
        gdn_workspace_specs(2, 2, mode="decode", target="ppu15")


@pytest.mark.parametrize("count", [0, 1, 511, 512, 513, 1023, 1024, 1025, 32768])
def test_bounded_selection(count):
    rng = random.Random(count)
    scores = [rng.randrange(7) for _ in range(count)]  # deliberate exact ties
    expected = sorted(range(count), key=lambda i: (-scores[i], i))[:512]
    assert merged_topk(scores) == expected
    if count > 512:
        assert merged_topk(scores, 511) != expected, (
            "missing selection denominator must fail"
        )


@pytest.mark.parametrize(
    "pos", [0, 2, 3, 4, 62, 63, 64, 2046, 2047, 2048, 2049, 2050, 2051, 4095]
)
def test_qsa_causality_tail_ties(pos):
    q = torch.zeros(4, 128)
    keys = torch.ones((pos + 1) // 4 + 8, 128)
    selected, groups, _ = select_reference(q, keys, pos, pos + 1)
    valid = selected[selected >= 0]
    assert valid.unique().numel() == valid.numel()
    assert (valid <= pos).all()
    assert groups.tolist() == list(range(min(512, (pos + 1) // 4)))
    assert valid.numel() == min(512, (pos + 1) // 4) * 4 + (pos + 1) % 4
    if pos % 4 == 3:
        # Old qsa_expand always appended current block, duplicating it.
        wrong = torch.cat((valid, torch.arange(pos // 4 * 4, pos + 1)))
        assert wrong.unique().numel() != wrong.numel() or wrong.numel() > 2051


def test_indexer_weight_rounding():
    torch.manual_seed(9)
    x = torch.randn(3, 4, 128).bfloat16()
    weight = torch.randn(128) * 0.1
    rope = torch.tensor([[0, 1, 2], [2, 3, 4], [4, 5, 6]])
    cos = torch.randn(16, 32).bfloat16()
    sin = torch.randn(16, 32).bfloat16()
    got = index_encode(x, weight, rope, cos, sin)
    assert got.dtype == torch.bfloat16
    assert not torch.equal(
        got, index_encode(x, torch.zeros_like(weight), rope, cos, sin)
    )


def test_schema_declares_all_writers():
    schema = str(torch.ops.actlize_la.serving_out.default._schema)
    assert "Tensor(a1!)[] buffers" in schema and "-> ()" in schema


def test_fake_trace_preserves_mutation_node():
    from torch.fx.experimental.proxy_tensor import make_fx

    def run(x, output):
        torch.ops.actlize_la.serving_out([x], [output], 987654321)
        return output

    # Metadata-only trace, not a fake numerical backend or device admission.
    graph = make_fx(run, tracing_mode="fake")(torch.empty(2), torch.empty(2))
    calls = [
        node
        for node in graph.graph.nodes
        if node.target == torch.ops.actlize_la.serving_out.default
    ]
    assert len(calls) == 1
    with pytest.raises(RuntimeError, match="serving plan expired"):
        run(torch.empty(2), torch.empty(2))


def test_export_functionalization_keeps_output_mutation():
    class Run(torch.nn.Module):
        def forward(self, x, output):
            torch.ops.actlize_la.serving_out([x], [output], 987654321)
            return output

    # Use the actual export/AOT functionalization path. torch.func.functionalize
    # has different custom-op restrictions and is not this integration's path.
    graph = torch.export.export(Run(), (torch.empty(2), torch.empty(2)))
    functional = graph.run_decompositions()
    mutations = [
        spec.target
        for spec in functional.graph_signature.output_specs
        if spec.kind.name == "USER_INPUT_MUTATION"
    ]
    assert mutations == ["output"]
    assert "actlize_la.serving_out" in functional.graph_module.code


def test_native_abi_sm120_when_built():
    import os

    path = os.environ.get("ALA_SERVING_TEST_LIBRARY")
    if not path:
        pytest.skip(
            "no compiled native library supplied; device validation is a separate runner"
        )
    lib = C.CDLL(path)
    for i, struct in enumerate((_Gdn, _Commit, _Prefill, _Conv)):
        assert lib.ala_serving_struct_size(i) == C.sizeof(struct)
    assert lib.ala_qsa_struct_size() == C.sizeof(_Qsa)
    for name in ("ala_gdn_run", "ala_gdn_commit", "ala_qsa_run", "ala_conv_capture"):
        fn = getattr(lib, name)
        fn.argtypes = [C.c_void_p, C.c_void_p]
        assert fn(None, None) == -1, "null ABI must fail before any device call"


def test_flare_fp32_rope_is_not_bf16_rope():
    torch.manual_seed(17)
    x = torch.randn(2, 4, 128).bfloat16()
    w = torch.randn(128) * 0.1
    positions = torch.tensor([[0, 1, 2], [3, 4, 5]])
    cos = torch.randn(8, 32)
    sin = torch.randn(8, 32)
    fp = index_encode(x, w, positions, cos, sin)
    bf = index_encode(x, w, positions, cos.bfloat16(), sin.bfloat16())
    assert fp.dtype == torch.float32 and bf.dtype == torch.bfloat16
    assert not torch.equal(fp.bfloat16(), bf), (
        "wrong rotary boundary negative stayed green"
    )


@pytest.mark.parametrize("width", [1, 2, 4, 8])
def test_shortconv_prefix_contract(width):
    # The existing speculative ShortConv stores history AFTER input0 followed
    # by candidate inputs1..W-1. Reuse this encoding, do not shift twice.
    history = torch.arange(3 * 11).reshape(11, 3)
    inputs = torch.arange(width * 11).reshape(width, 11) + 100
    joined = torch.cat((history, inputs.T), 1)
    encoded = joined[:, 1:]
    for count in range(width + 1):
        selected = history if count == 0 else encoded[:, count - 1 : count + 2]
        assert torch.equal(selected, joined[:, count : count + 3])
    if width > 1:
        assert not torch.equal(encoded[:, :3], joined[:, 2:5]), (
            "accepted draft count mistaken for target input count"
        )
