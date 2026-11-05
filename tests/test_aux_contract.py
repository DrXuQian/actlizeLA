import ctypes as C
import os
import pytest
import torch
from actlize_la.serving.auxiliary import (
    TensorV1,
    RowsV1,
    PointwiseV1,
    ShortConvV1,
    HashV1,
    HeaderV1,
    tensor_descriptor,
)
from actlize_la.serving.gdn_transaction import TransactionV2
from actlize_la.serving.geometry import ple_geometry, splitmix64
from actlize_la.serving import qsa_cache_specs
from aux_reference import (
    bf,
    grouped,
    gated,
    hc_silu,
    hc_mean,
    hc_combine,
    ple_gate,
    convolution,
    encoded_history,
    hash_packed,
    assert_raw_unchanged,
)

torch.set_num_threads(1)


def test_zero_prefix_uses_device_snapshot_not_mathematical_oracle():
    reference = torch.ones(2, 128, 128, dtype=torch.float32)
    before = reference.clone()
    before[0, 0, 0] = torch.nextafter(reference[0, 0, 0], torch.tensor(2.0))
    after = before.clone()  # the operation did NOT modify any bit
    torch.testing.assert_close(after, reference, rtol=1e-6, atol=0)
    # Reproduce the old false red after a previously admitted rounded update.
    assert not torch.equal(after, reference)
    assert_raw_unchanged(after, before, label="unchanged FP32 state")
    after.view(torch.int32)[0, 0, 0] ^= 1
    # The old CPU-reference equality would now turn GREEN despite this write.
    assert torch.equal(after, reference)
    with pytest.raises(AssertionError, match="raw_bitdiff=1/"):
        assert_raw_unchanged(after, before, label="one-bit accidental state write")


def test_raw_snapshot_sees_bf16_and_signed_zero_writes():
    before = torch.zeros(2, 9, dtype=torch.bfloat16)
    after = before.clone()
    assert_raw_unchanged(after, before, label="conv snapshot")
    after.view(torch.int16)[0, 0] = -32768
    assert torch.equal(after, before)  # float equality misses this -0.0 bit change
    with pytest.raises(AssertionError, match="raw_bitdiff=1/"):
        assert_raw_unchanged(after, before, label="signed-zero mutation")


def test_whole_width_is_not_grouped():
    x = torch.ones(2, 4, 2560, dtype=torch.bfloat16)
    x[:, 0] = 8
    w = torch.ones(4, 2560)
    whole = grouped(x.view(2, 1, 10240), w.view(1, 10240))
    assert not torch.equal(whole.view_as(x), grouped(x, w))


def test_aux_export_all_mutations_and_fake_negative():
    from actlize_la.serving.auxiliary import PreparedAuxiliary
    from actlize_la.serving._native import _plans, _handles

    # Metadata-only owner, NOT a fake device implementation. Export must retain
    # all declared writes; no native launch or device result is manufactured.
    owner = PreparedAuxiliary.__new__(PreparedAuxiliary)
    x = torch.empty(2, 4, 2560, dtype=torch.bfloat16)
    y = torch.empty_like(x)
    combined = torch.empty_like(x)
    error = torch.empty(2, dtype=torch.int32)
    owner.inputs = [x]
    owner.buffers = [y, combined, error]
    owner.device = x.device
    owner.signatures = [(tuple(t.shape), t.dtype) for t in (x, y, combined, error)]
    handle = next(_handles)
    _plans[handle] = owner

    class Run(torch.nn.Module):
        def forward(self, x, y, combined, error):
            torch.ops.actlize_la.serving_out([x], [y, combined, error], handle)
            return y

    ep = torch.export.export(Run(), (x, y, combined, error)).run_decompositions()
    changed = [
        s.target
        for s in ep.graph_signature.output_specs
        if s.kind.name == "USER_INPUT_MUTATION"
    ]
    assert changed == ["y", "combined", "error"]
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode() as mode:
        fx, fy, fc, fe = (mode.from_tensor(t) for t in (x, y, combined, error))
        torch.ops.actlize_la.serving_out([fx], [fy, fc, fe], handle)
        with pytest.raises(ValueError, match="static tensor"):
            torch.ops.actlize_la.serving_out([fx], [fy, fc, fe.to(torch.int64)], handle)


def test_flare_bindings_keep_metadata_semantics(monkeypatch):
    from types import SimpleNamespace
    from actlize_la.serving import flare

    def capture(*args, **kwargs):
        return kwargs

    for name in (
        "PreparedShortConv",
        "PreparedConvHistoryV1",
        "PreparedPointwise",
        "PreparedPLEHash",
        "PreparedGDN",
        "PreparedGDNCheckedV2",
        "PreparedGDNVerifyV2",
        "PreparedGDNCommitV2",
    ):
        monkeypatch.setattr(flare, name, capture)
    meta = SimpleNamespace(
        num_tokens=torch.tensor([1], dtype=torch.int32),
        num_seqs=torch.tensor([1], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        convolution_slots=torch.tensor([0], dtype=torch.int64),
        fresh=torch.tensor([False]),
        accepted_tokens=torch.tensor([1], dtype=torch.int32),
        state_indices=torch.tensor([[2]], dtype=torch.int64),
        initial_state_indices=torch.tensor([2]),
        final_state_indices=torch.tensor([2]),
        checkpoint_state_indices=torch.tensor([0]),
        checkpoint_offsets=torch.tensor([0]),
    )
    x = torch.empty(1, 10240, dtype=torch.bfloat16)
    e = torch.zeros(1, dtype=torch.int32)
    conv = flare.bind_shortconv(
        None,
        x=x,
        weight=None,
        state=None,
        metadata=meta,
        output=x,
        errors=e,
        dilation=3,
        mode="decode",
    )
    assert conv["rows"].slots is meta.convolution_slots and conv["rows"].slots[0] == 0
    assert conv["mode"] == "verify" and conv["dilation"] == 3
    assert conv["previous_accepted"] is meta.accepted_tokens
    history = flare.bind_shortconv_history(
        None,
        state=None,
        metadata=meta,
        tokens=1,
        snapshot=None,
        stamps=None,
        consumed_prefix=e,
        errors=e,
        dilation=3,
        restore=True,
    )
    assert history["rows"].slots is meta.convolution_slots and history["restore"]
    gated = flare.bind_gated_norm(
        None,
        hidden=torch.empty(1, 48, 128),
        gate=None,
        weight=None,
        output=None,
        errors=e,
        value_heads=48,
        epsilon=1e-6,
        silu_gate=False,
    )
    assert gated["activation"] == "sigmoid" and gated["round_normalized"]
    kw = dict(
        mixed_qkv=x,
        a=None,
        b=None,
        A_log=None,
        dt_bias=None,
        state=None,
        metadata=meta,
        output=torch.empty(1, 48, 128),
        workspace={},
        width=1,
    )
    checked = flare.bind_gdn(None, **kw, mode="decode", upstream_errors=e)
    assert checked["upstream_errors"] is e and checked["initial_slots"][0] == 2
    with pytest.raises(ValueError, match="epochs"):
        flare.bind_gdn(None, **kw, mode="verify", upstream_errors=e)
    with pytest.raises(NotImplementedError, match="checkpoint"):
        flare.bind_gdn(None, **kw, mode="prefill", checkpoint_export=True)
    epoch = {
        n: torch.zeros(1, dtype=torch.int64)
        for n in ("epochs", "ready_epochs", "consumed_epochs")
    }
    verify = flare.bind_gdn(
        None, **kw, mode="verify", transaction=epoch, upstream_errors=e
    )
    assert verify["ready_epochs"] is epoch["ready_epochs"]


@pytest.mark.parametrize("t", [1, 2, 4, 8, 16, 32, 65])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_rounding_witnesses(t, dtype):
    torch.manual_seed(t)
    x = torch.randn(t, 4, 2560).bfloat16()
    w = (torch.randn(4, 2560) * 0.3 + 0.7).to(dtype)
    exact = grouped(x, w)
    assert not torch.equal(exact, grouped(x, w[:1].expand_as(w)))
    assert not torch.equal(exact, grouped(x, w, zero=True))
    if dtype == torch.float32:
        assert not torch.equal(exact, grouped(x, w.bfloat16()))
    projected = torch.randn(t, 1, 320).bfloat16() * 3
    z, trace = hc_silu(projected)
    assert torch.equal(trace, bf(projected.float() / 4))
    # Division by four is an exact power-of-two scale for these BF16 inputs;
    # test its observable boundary directly, do not invent a false witness.
    logits = torch.randn_like(x)
    mean, stages = hc_mean(x, logits)
    wrong = (logits.float().sigmoid() * x.float()).mean(1, keepdim=True).bfloat16()
    assert not torch.equal(mean, wrong)
    assert torch.equal(stages[..., :2560], bf(logits.float().sigmoid()))
    value = torch.randn(t, 1, 2560).bfloat16()
    inj = torch.randn(t, 4).bfloat16()
    combined = hc_combine(x, value, inj)
    normalized = grouped(combined, w)
    unrounded = (
        x.float() + 2 * (inj.float() / 4).sigmoid().unsqueeze(-1) * value.float()
    )
    assert not torch.equal(normalized, grouped(unrounded, w))
    out, steps = ple_gate(x, logits, value)
    wronggate = (x.float() * logits.float()).sum(-1, keepdim=True) / (2560**0.5)
    wrong = (
        wronggate.sign() * wronggate.abs().clamp_min(1e-6).sqrt()
    ).sigmoid() * value.float()
    assert not torch.equal(out, wrong.bfloat16())
    assert steps.shape == (t, 4, 6)
    zero, zero_steps = ple_gate(torch.zeros_like(x), logits, value)
    assert not zero_steps[..., 4].any()
    assert torch.equal(zero, (value.float() * 0.5).expand_as(x).bfloat16())


@pytest.mark.parametrize("rounded", [False, True])
@pytest.mark.parametrize("before", [False, True])
@pytest.mark.parametrize("silu", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_gated_modes(rounded, before, silu, dtype):
    torch.manual_seed(5)
    x = torch.randn(3, 48, 128).bfloat16()
    g = torch.randn_like(x)
    w = torch.randn(48, 128).to(dtype)
    y, trace = gated(x, g, w, rounded=rounded, before=before, silu=silu)
    assert torch.isfinite(y).all()
    if rounded:
        assert torch.equal(trace[..., :128], bf(trace[..., :128]))
        assert not torch.equal(
            y, gated(x, g, w, rounded=False, before=before, silu=silu)[0]
        )
    if dtype == torch.bfloat16 and rounded:
        assert torch.equal(trace[..., 128:], bf(trace[..., 128:]))


@pytest.mark.parametrize("dilation", [1, 3])
@pytest.mark.parametrize("width", [1, 2, 4, 8])
def test_shortconv_prefixes(dilation, width):
    torch.manual_seed(width)
    c, h = 17, 3 * dilation
    history = torch.randn(c, h).bfloat16()
    w = torch.randn(c, 4).bfloat16()
    x = torch.randn(width, c).bfloat16()
    next_x = torch.randn(2, c).bfloat16()
    out, final = convolution(x, w, history, dilation)
    encoded = encoded_history(x, history, width - 1)
    for accepted in range(width + 1):
        selected = (
            history if accepted == 0 else encoded[:, accepted - 1 : accepted - 1 + h]
        )
        _, want = convolution(x[:accepted], w, history, dilation)
        assert torch.equal(selected, want)
        after, _ = convolution(next_x, w, selected, dilation)
        all_out, _ = convolution(
            torch.cat((x[:accepted], next_x)), w, history, dilation
        )
        assert torch.equal(after, all_out[-2:])
    if dilation == 3:
        assert not torch.equal(out, convolution(x, w, history[:, -3:], 1)[0])
    current = history
    outputs = []
    for token in x:
        one, current = convolution(token[None], w, current, dilation)
        outputs.append(one)
    assert torch.equal(torch.cat(outputs), out) and torch.equal(current, final)


@pytest.mark.parametrize(
    "ordinal,first,last,factors",
    [
        (0, 101, 179, (15411308609579, 2523213358983, 11275943025685)),
        (1, 181, 269, (27696880803969, 9426225827651, 20661530663711)),
        (7, 787, 883, (38816308503819, 37780860329989, 8063966781539)),
    ],
)
def test_geometry_pinned_b12x_vectors(ordinal, first, last, factors):
    g = ple_geometry(
        vocab_size=151936, base_size=100, dense_layer_ordinal=ordinal, alignment=128
    )
    assert (
        g.prime_sizes[0] == first
        and g.prime_sizes[-1] == last
        and g.multipliers == factors
    )
    assert g.nbytes == 280 and g.padded_vocab_size % 128 == 0
    assert g == ple_geometry(
        vocab_size=g.vocab_size,
        prime_sizes=g.prime_sizes,
        table_offsets=g.table_offsets,
        multipliers=g.multipliers,
        alignment=128,
    )
    assert splitmix64(0) == 0xE220A8397B1DCDAF
    with pytest.raises(ValueError):
        ple_geometry(
            vocab_size=g.vocab_size,
            prime_sizes=g.prime_sizes,
            table_offsets=(1, *g.table_offsets[1:]),
        )
    with pytest.raises(ValueError):
        ple_geometry(
            vocab_size=g.vocab_size,
            prime_sizes=g.prime_sizes,
            table_offsets=g.table_offsets,
            multipliers=(2, 3, 5),
        )


def test_hash_eos_and_partition():
    g = ple_geometry(vocab_size=200, base_size=100)
    tokens = torch.tensor([3, 9, 4, 5, 8, 1, 9, 2])
    starts = [0, 3, 3, 8]
    history = torch.tensor([[1, 2], [9, 9], [4, 9]])
    got, ids = hash_packed(tokens, starts, history, g, 9)
    assert ids.tolist() == [0, 0, 0, 2, 2, 2, 2, 2]
    # Before/after EOS, packed boundary and reversed lag all have witnesses.
    assert not torch.equal(got[1], got[2])
    combined, _ = hash_packed(tokens, [0, 8], history[:1], g, 9)
    assert not torch.equal(got[3], combined[3])
    masked, mapping = hash_packed(tokens, starts, history, g, 9, live_requests=2)
    assert (masked[3:] == -1).all() and (mapping[3:] == -1).all()
    assert qsa_cache_specs(4, 3, page_size=128)["compressed_cache"][0] == (4, 32, 128)


def test_native_aux_descriptors():
    path = os.environ.get("ALA_SERVING_TEST_LIBRARY")
    if not path:
        pytest.skip("native library not built; this is metadata validation only")
    lib = C.CDLL(path)
    for i, cls in enumerate((TensorV1, RowsV1, PointwiseV1, ShortConvV1, HashV1)):
        assert lib.ala_aux_v1_struct_size(i) == C.sizeof(cls)
    assert lib.ala_gdn_transaction_v2_size() == C.sizeof(TransactionV2)
    validate = lib.ala_aux_v1_validate
    validate.argtypes = [C.c_int, C.c_void_p]
    t = torch.empty(3, 7, 9).transpose(1, 2)
    d = tensor_descriptor(t)
    assert validate(0, C.byref(d)) == 0
    d.nbytes = 2
    assert validate(0, C.byref(d)) == -1
    pool = torch.empty(2, 4, 5120, dtype=torch.bfloat16)
    x = pool[:, :, ::2]
    out = pool[:, :, 1::2]
    weight = torch.ones(4, 2560)
    errors = torch.empty(2, dtype=torch.int32)
    p = PointwiseV1(
        header=HeaderV1(1, C.sizeof(PointwiseV1)),
        rows=RowsV1(tokens=2),
        x=tensor_descriptor(x),
        output=tensor_descriptor(out),
        weight=tensor_descriptor(weight),
        errors=tensor_descriptor(errors),
        kind=0,
        groups=4,
        width=2560,
        weight_per_group=1,
        epsilon=1e-6,
    )
    # Keep referenced allocations alive; the native validator never reads them.
    assert validate(2, C.byref(p)) == 0
    p.output = tensor_descriptor(x)
    assert validate(2, C.byref(p)) == -1
    p.header.version = 99
    assert validate(2, C.byref(p)) == -1
