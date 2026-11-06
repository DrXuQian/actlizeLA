"""Native prefill workspace, capability and arithmetic boundaries; host only."""
import math
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from actlize_la.serving import gdn_workspace_specs
from actlize_la.serving.gdn import _workspace, _require_native_prefill
from residual_cpu import residual_forward
from serving_reference import gdn_reference


@pytest.mark.parametrize("width", [1, 63, 64, 65, 127, 128, 129, 2048, 8192])
@pytest.mark.parametrize("hv", [32, 48])
def test_padded_inverse_covers_every_native_group_not_just_logical_elements(width, hv):
    groups = 3 * hv * ((width + 63) // 64)
    ours = gdn_workspace_specs(3, width, 16, hv, mode="prefill", target="ppu10")
    other = gdn_workspace_specs(3, width, 16, hv, mode="prefill", target="ppu15")
    assert ours["inverse"][0] == (groups, 128, 128)
    assert other["inverse"][0] == (groups, 64, 64)
    # Independent physical footprint; most of each padded inverse slot is a hole.
    end_of_last_write = (groups - 1) * 16384 + 4096
    assert math.prod(ours["inverse"][0]) >= end_of_last_write
    assert math.prod(other["inverse"][0]) < end_of_last_write
    assert ours["snapshots"][0] == (groups, 128, 128)
    assert math.prod(ours["vnew"][0]) == groups * 8192


def test_wrong_target_inverse_shape_is_rejected_before_launch():
    specs = gdn_workspace_specs(1, 65, 1, 3, mode="prefill", target="ppu10")
    tensors = {name: torch.empty(shape, dtype=dtype) for name, (shape, dtype) in specs.items()}
    _workspace(tensors, specs, torch.device("cpu"))
    tensors["inverse"] = torch.empty(6, 64, 64, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="inverse"):
        _workspace(tensors, specs, torch.device("cpu"))


def test_old_semantic_library_cannot_masquerade_as_native_prefill():
    for missing in ("actlize_la_ppu10_configure", "actlize_la_ppu10_forward_prepared"):
        exports = dict(actlize_la_ppu10_configure=Mock(), actlize_la_ppu10_forward_prepared=Mock())
        del exports[missing]
        with pytest.raises(RuntimeError, match="no semantic fallback"):
            _require_native_prefill(SimpleNamespace(target="ppu10", lib=SimpleNamespace(**exports)))
    _require_native_prefill(SimpleNamespace(target="ppu15", lib=SimpleNamespace()))
    _require_native_prefill(SimpleNamespace(target="cuda_sm120", lib=SimpleNamespace()))


@pytest.mark.parametrize("length", [63, 64, 65, 127, 128, 129])
def test_normalized_packed_prefill_and_neutral_padding_follow_recurrence(length):
    torch.set_num_threads(1)
    torch.manual_seed(114 + length)
    hk, hv = 2, 6
    x = torch.randn(length, (2 * hk + hv) * 128).bfloat16()
    a, b = (torch.randn(length, hv).bfloat16() for _ in range(2))
    alog, bias = torch.linspace(-3, -1, hv), torch.linspace(-.2, .1, hv)
    vk = torch.randn(hv, 128, 128) * .02
    want_o, want_vk, *_ = gdn_reference(x, a, b, alog, bias, vk, hk)
    q, k, v = x.split((hk * 128, hk * 128, hv * 128), -1)
    def normalize(value):
        value = value.reshape(1, length, hk, 128).float()
        return (value * torch.rsqrt((value * value).sum(-1, keepdim=True) + 1e-6)).bfloat16()
    q, k = normalize(q), normalize(k)
    v = v.reshape(1, length, hv, 128)
    log_g = -alog.exp() * torch.nn.functional.softplus(a.float() + bias, threshold=20)
    beta = b.float().sigmoid().bfloat16()
    packed = (q, k, v, log_g[None], beta[None])
    initial = vk.transpose(-1, -2).contiguous()[None]
    out, final = residual_forward(packed, initial)
    for actual, want in ((out[0].float(), want_o.float()), (final[0].transpose(-1, -2), want_vk)):
        relative = torch.linalg.vector_norm(actual - want) / torch.linalg.vector_norm(want)
        assert relative < .02, float(relative)
    # Keep the same ABI capacity but a shorter live range; zero K/V/beta and
    # log-g=0 must preserve the final live state through the padded remainder.
    padded = tuple(torch.nn.functional.pad(t, (0, 0, 0, 0, 0, 64)) if t.ndim == 4
                   else torch.nn.functional.pad(t, (0, 0, 0, 64)) for t in packed)
    padded_out, padded_final = residual_forward(padded, initial)
    assert torch.equal(padded_out[:, :length], out)
    assert torch.equal(padded_final, final)
