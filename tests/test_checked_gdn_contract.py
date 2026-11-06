"""Checked-entry oracle routing, not a replacement for native device admission."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import aux_device
from residual_cpu import residual_forward
from serving_device import close
from serving_reference import gdn_reference


@pytest.fixture(scope="module")
def reported_prefill_pair():
    """Exact host fixture seed/geometry from GdnCase(width=65), valid row 1.

    Matrix residual algebra is independent of the device implementation. Its
    BF16 boundaries need not produce the exact same device mismatch count.
    """
    torch.set_num_threads(1)
    width, hk, hv = 65, 16, 48
    torch.manual_seed(701 + width)
    x = torch.randn(3 * width, (2 * hk + hv) * 128).bfloat16()
    a = (torch.randn(3 * width, hv) - 1).bfloat16()
    b = torch.randn(3 * width, hv).bfloat16()
    alog, bias = torch.linspace(-3, -1, hv), torch.linspace(-.2, .1, hv)
    initial = torch.randn(5, hv, 128, 128) * .02
    initial[0] = 0
    x, a, b = x[width:2 * width], a[width:2 * width], b[width:2 * width]
    want_o, want_s, *_ = gdn_reference(x, a, b, alog, bias, initial[4], hk)
    q, k, v = x.split((hk * 128, hk * 128, hv * 128), -1)

    def normalize(value):
        value = value.reshape(1, width, hk, 128).float()
        return (value * torch.rsqrt(value.square().sum(-1, keepdim=True) + 1e-6)).bfloat16()

    logs = -alog.exp() * torch.nn.functional.softplus(a.float() + bias, threshold=20)
    beta = b.float().sigmoid().bfloat16()
    out, state = residual_forward(
        (normalize(q), normalize(k), v.reshape(1, width, hv, 128), logs[None], beta[None]),
        initial[4].transpose(-1, -2).contiguous()[None],
    )
    return ((out[0], want_o), (state[0].transpose(-1, -2), want_s))


def test_actual_residual_rounding_is_not_a_recurrence_pointwise_contract(reported_prefill_pair):
    for got, want in reported_prefill_pair:
        # Same failure seam as the reported traceback: the 0.6% L2 check
        # succeeds, then the strict per-element recurrence check rejects it.
        relative = torch.linalg.vector_norm(got.float() - want.float()) / torch.linalg.vector_norm(want.float())
        assert relative < .006
        with pytest.raises(AssertionError, match="Tensor-likes are not close"):
            close(got, want, chunk=False)
        assert close(got, want, chunk=True) < .02


def test_real_corruption_remains_red_under_chunk_contract(reported_prefill_pair):
    (output, want_output), (state, want_state) = reported_prefill_pair
    for bad, want in (
        (output.float() * 1.1, want_output),
        (state * 1.1, want_state),
        (state.transpose(-1, -2), want_state),
        (state.roll(1, 0), want_state),
        (torch.full_like(state, float("nan")), want_state),
    ):
        with pytest.raises(AssertionError):
            close(bad, want, chunk=True)


@pytest.mark.parametrize("target", ["ppu10", "ppu15", "cuda_sm120"])
@pytest.mark.parametrize("plant", [None, "bad-output", "bad-state", "blocked-state-write", "checked-only-one-bit"])
def test_checked_device_test_itself_uses_the_correct_contract(monkeypatch, target, plant):
    """CPU transport stand-in exercises the actual auxiliary test control flow.

    This tests the checker, not kernel execution. A tiny near-zero rounding
    witness passes relative L2 but fails recurrence's pointwise check.
    """
    current = []
    contracts = []
    want_state = torch.tensor([[1e-4, 1.0], [2.0, 3.0]])

    class Case:
        def __init__(self, lib, width, mode):
            self.width, self.mode = width, mode
            self.initial = torch.zeros(5, 2, 2)
            self.state = self.initial.clone()
            self.out = torch.zeros(3 * width, 2, 2)
            self.final_slots = torch.tensor([2, 4, 0])
            self.work = {"errors": torch.zeros(3, dtype=torch.int32)}
            self.plan = Mock(inputs=[None] * 5, side_effect=lambda: prepare(
                lib, upstream_errors=torch.zeros(3, dtype=torch.int32), unchecked=True)())
            for name in ("mixed", "ad", "bd", "starts", "live", "slots", "checkpoint_slots", "checkpoint_offsets"):
                setattr(self, name, None)
            current[:] = [self]

        def golden(self, row):
            return want_state.expand(self.width, 2, 2), want_state

        def ok(self):
            assert not self.work["errors"].any()

    def prepare(lib, **kwargs):
        case = current[0]

        def run():
            dep = kwargs["upstream_errors"]
            case.work["errors"].copy_(dep)
            for row, slot in ((0, 2), (1, 4)):
                if dep[row]:
                    if case.mode == "prefill" and plant == "blocked-state-write":
                        case.state[slot, 0, 0] += 1
                    continue
                out, state = case.golden(row)
                out, state = out.clone(), state.clone()
                if case.mode == "prefill" and target in ("ppu10", "ppu15"):
                    out[:, 0, 0] += 2e-4
                    state[0, 0] += 2e-4
                if case.mode == "prefill" and plant == "bad-output":
                    out *= 1.1
                if case.mode == "prefill" and plant == "bad-state":
                    state *= 1.1
                if case.mode == "prefill" and plant == "checked-only-one-bit" and not kwargs.get("unchecked", False):
                    state.view(torch.int32)[0, 0] ^= 1
                case.out[row * case.width:(row + 1) * case.width].copy_(out)
                case.state[slot].copy_(state)
        return run

    def observe(got, want, *, chunk=False):
        contracts.append((current[0].mode, chunk))
        return close(got, want, chunk=chunk)

    monkeypatch.setattr(aux_device, "GdnCase", Case)
    monkeypatch.setattr(aux_device, "PreparedGDNCheckedV2", prepare)
    monkeypatch.setattr(aux_device, "gpu", lambda value: value)
    monkeypatch.setattr(aux_device.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(aux_device, "close", observe)
    monkeypatch.setattr(aux_device, "record", lambda *args, **kwargs: None)
    if plant:
        with pytest.raises(AssertionError):
            aux_device.checked_gdn_tests(SimpleNamespace(target=target))
    else:
        aux_device.checked_gdn_tests(SimpleNamespace(target=target))
        assert contracts == [("decode", False)] * 5 + [
            ("prefill", target in ("ppu10", "ppu15"))] * 5
