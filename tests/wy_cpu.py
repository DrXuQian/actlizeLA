"""CPU arithmetic contract, independent of native load/fragment delivery.

This is NOT device validation. Rounded mode models the candidate's explicit
BF16 boundaries and TF32x3 block merges; double mode proves the WY identity.
"""
import torch
import torch.nn.functional as F


def tf32(x):
    bits = x.contiguous().view(torch.int32)
    return ((bits + 4096) & -8192).view(torch.float32)


def tf32x3(a, b):
    ah, bh = tf32(a), tf32(b)
    al, bl = tf32(a - ah), tf32(b - bh)
    return al @ bh + ah @ bl + ah @ bh


def block_inverse(lower, rounded):
    """The kernel's four diagonal solves + gap-ordered block substitution."""
    inverse = torch.zeros_like(lower)
    for start in range(0, 64, 16):
        a = lower[..., start:start + 16, start:start + 16]
        r = inverse[..., start:start + 16, start:start + 16]
        for row in range(16):
            r[..., row, row] = 1
            for k in range(row):
                r[..., row, :] -= a[..., row, k, None] * r[..., k, :]
    product = tf32x3 if rounded else torch.matmul
    for gap in range(1, 4):
        for col in range(4 - gap):
            row = col + gap
            rs, cs = slice(row * 16, (row + 1) * 16), slice(col * 16, (col + 1) * 16)
            tmp = torch.zeros_like(lower[..., rs, cs])
            for k in range(col, row):
                ks = slice(k * 16, (k + 1) * 16)
                tmp += product(lower[..., rs, ks], inverse[..., ks, cs])
            inverse[..., rs, cs] = -product(inverse[..., rs, rs], tmp)
    return inverse


def wy_forward(inputs, initial=None, *, rounded=True, plant=None):
    q, k, v, g, beta = inputs
    batch, length, hk, dim = q.shape
    hv = v.shape[2]
    dtype = torch.float32 if rounded else torch.float64
    cast = (lambda x: x.to(torch.bfloat16).float()) if rounded else (lambda x: x)
    mapping = torch.arange(hv) // (hv // hk)
    if plant == "gva":
        mapping = (mapping + 1) % hk
    q, k = q[:, :, mapping], k[:, :, mapping]
    q, k, v = [F.pad(x.transpose(1, 2).to(dtype), (0, 0, 0, (-length) % 64))
               .reshape(batch, hv, -1, 64, dim) for x in (q, k, v)]
    g, beta = [F.pad(x.transpose(1, 2).to(dtype), (0, (-length) % 64))
               .reshape(batch, hv, -1, 64) for x in (g, beta)]
    prefix = g.cumsum(-1)
    mask = torch.ones(64, 64, dtype=torch.bool).tril(-1)
    decay = torch.where(mask, (prefix[..., :, None] - prefix[..., None, :]), 0).exp()
    lower = torch.where(mask, (k @ k.transpose(-1, -2)) * beta[..., :, None] * decay, 0)
    if plant == "inverse-sign":
        lower = -lower
    inverse = cast(block_inverse(lower, rounded))
    w = cast(inverse @ cast(k * beta[..., :, None] * prefix.exp()[..., :, None]))
    u = cast(inverse @ cast(v * beta[..., :, None]))
    state = torch.zeros(batch, hv, dim, dim, dtype=dtype) if initial is None else initial.to(dtype).clone()
    output = torch.zeros_like(v)
    causal = torch.ones(64, 64, dtype=torch.bool).tril()
    for chunk in range(q.shape[2]):
        valid = min(64, length - chunk * 64)
        qi, ki, gi = q[:, :, chunk], k[:, :, chunk], prefix[:, :, chunk]
        if plant == "reset":
            state.zero_()
        old = cast(state)
        new = u[:, :, chunk] - w[:, :, chunk] @ old
        if plant == "snapshot-after-update":
            old = cast(state * gi[..., valid - 1].exp()[..., None, None] +
                       ki.transpose(-1, -2) @ cast(new * (gi[..., valid - 1, None] - gi).exp()[..., :, None]))
        p = torch.where(causal, (qi @ ki.transpose(-1, -2)) *
                        torch.where(causal, gi[..., :, None] - gi[..., None, :], 0).exp(), 0)
        if plant == "causal":
            p = (qi @ ki.transpose(-1, -2)) * (gi[..., :, None] - gi[..., None, :]).exp()
        output[:, :, chunk] = ((qi @ old) * gi.exp()[..., :, None] + cast(p) @ cast(new)) * dim ** -0.5
        state = state * gi[..., valid - 1].exp()[..., None, None] + ki.transpose(-1, -2) @ cast(
            new * (gi[..., valid - 1, None] - gi).exp()[..., :, None])
    out = output.reshape(batch, hv, -1, dim)[:, :, :length].transpose(1, 2).contiguous()
    return (out.to(inputs[0].dtype) if rounded else out), state


def recurrent_double(inputs, initial=None):
    q, k, v, g, beta = (x.double() for x in inputs)
    batch, length, hk, dim = q.shape
    hv = v.shape[2]
    q, k = (x.repeat_interleave(hv // hk, 2) for x in (q, k))
    h = torch.zeros(batch, hv, dim, dim, dtype=torch.float64) if initial is None else initial.double().clone()
    output = torch.empty_like(v)
    for t in range(length):
        h *= g[:, t].exp()[..., None, None]
        new = (v[:, t] - (k[:, t, :, :, None] * h).sum(-2)) * beta[:, t, :, None]
        h += k[:, t, :, :, None] * new[..., None, :]
        output[:, t] = (q[:, t, :, :, None] * h).sum(-2) * dim ** -0.5
    return output, h
