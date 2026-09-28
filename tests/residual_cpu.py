"""Independent matrix algebra/rounding reference, NOT a device execution model."""
import torch
from wy_cpu import block_inverse


def residual_forward(inputs, initial=None, *, rounded=True, plant=None):
    q, k, v, g, beta = inputs
    b, length, hk, d = q.shape
    hv = v.shape[2]
    dtype = torch.float32 if rounded else torch.float64
    cast = (lambda x: x.to(torch.bfloat16).float()) if rounded else (lambda x: x)
    q, k = (x.to(dtype).repeat_interleave(hv // hk, 2).transpose(1, 2) for x in (q, k))
    v, g, beta = (x.to(dtype).transpose(1, 2) for x in (v, g, beta))
    state = torch.zeros(b, hv, d, d, dtype=dtype) if initial is None else initial.to(dtype).clone()
    outputs = []
    for first in range(0, length, 64):
        count = min(64, length - first)
        def pad(x):
            return torch.nn.functional.pad(x, (0, 0, 0, 64-count))
        qi, ki, vi = (pad(x[..., first:first+count, :]) for x in (q, k, v))
        gi = torch.nn.functional.pad(g[..., first:first+count], (0, 64-count)).cumsum(-1)
        be = torch.nn.functional.pad(beta[..., first:first+count], (0, 64-count))
        lower_mask = torch.ones(64, 64, dtype=torch.bool).tril(-1)
        causal = torch.ones(64, 64, dtype=torch.bool).tril()
        differences = gi[..., :, None] - gi[..., None, :]
        lower = torch.where(lower_mask, (ki @ ki.transpose(-1, -2)) * be[..., :, None] *
                            differences.masked_fill(~lower_mask, 0).exp(), 0)
        inverse = cast(block_inverse(lower, rounded))
        if plant == "reset":
            state.zero_()
        old = cast(state)
        kh = ki @ old
        residual = cast(be[..., :, None] * (vi - gi.exp()[..., :, None] * kh))
        if plant == "gate-omitted":
            residual = cast(be[..., :, None] * (vi - kh))
        new = inverse @ residual
        if plant == "beta-outside-inverse":
            new = be[..., :, None] * (inverse @ cast(vi - gi.exp()[..., :, None] * kh))
        if plant == "inverse-identity":
            new = residual
        new[..., count:, :] = 0
        p = cast(torch.where(causal, (qi @ ki.transpose(-1, -2)) *
                              differences.masked_fill(~causal, 0).exp(), 0))
        out = ((qi @ old) * gi.exp()[..., :, None] + p @ cast(new)) * d ** -0.5
        outputs.append(out[..., :count, :])
        scaled = cast(new * (gi[..., count-1, None] - gi).exp()[..., :, None])
        state = state * gi[..., count-1].exp()[..., None, None] + ki.transpose(-1, -2) @ scaled
    out = torch.cat(outputs, -2).transpose(1, 2).contiguous()
    return (out.to(inputs[0].dtype) if rounded else out), state
