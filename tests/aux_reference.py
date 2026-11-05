"""Independent CPU/PyTorch arithmetic. No product kernels or b12x imports."""

import math
import torch


def bf(x):
    return x.to(torch.bfloat16).float()


def grouped(x, w, eps=1e-6, zero=False):
    f = x.float()
    n = f * torch.rsqrt((f * f).mean(-1, keepdim=True) + eps)
    return (n * (w.float() + (1 if zero else 0))).bfloat16()


def gated(x, gate, w, eps=1e-6, rounded=True, before=True, silu=False):
    activation = gate.float().sigmoid()
    if silu:
        activation *= gate.float()
    f = x.float() if before else x.float() * activation
    n = f * torch.rsqrt(f.square().mean(-1, keepdim=True) + eps)
    if rounded:
        n = bf(n)
    weighted = n * w.float()
    if rounded and w.dtype == torch.bfloat16:
        weighted = bf(weighted)
    out = weighted * activation if before else weighted
    return out.bfloat16(), torch.cat((n, weighted), -1)


def hc_silu(x):
    scaled = bf(x.float() / 4)
    return (scaled * scaled.sigmoid()).bfloat16(), scaled


def hc_mean(x, logits):
    gate = bf(logits.float().sigmoid())
    product = bf(x.float() * gate)
    total = torch.zeros_like(product[:, 0])
    for s in range(4):
        total += product[:, s]
    return (total / 4).bfloat16().unsqueeze(1), torch.cat((gate, product), -1)


def hc_combine(x, value, logits):
    scale = 2 * (logits.float() / 4).sigmoid()
    return (x.float() + scale.unsqueeze(-1) * value.float()).bfloat16()


def ple_gate(key, query, value):
    product = bf(key.float() * query.float())
    summed = bf(product.sum(-1, keepdim=True))
    divided = bf(summed / math.sqrt(key.shape[-1]))
    clamped = bf(
        divided.abs().clamp_min(float(torch.tensor(1e-6, dtype=torch.bfloat16)))
    )
    root = bf(clamped.sqrt())
    signed = bf(root * divided.sign())
    prob = bf(signed.sigmoid())
    return (prob * value.float()).bfloat16(), torch.cat(
        (summed, divided, clamped, root, signed, prob), -1
    )


def convolution(x, w, history, dilation):
    """Full sequential oracle. history is chronological unshifted input data."""
    seq = torch.cat((history.T, x), 0).float()
    h = history.shape[1]
    out = []
    for t in range(x.shape[0]):
        acc = torch.zeros(x.shape[1])
        for tap in range(4):
            acc = acc + seq[t + tap * dilation] * w[:, tap].float()
        rounded = bf(acc)
        out.append((rounded * rounded.sigmoid()).bfloat16())
    output = torch.stack(out) if out else torch.empty_like(x)
    return output, seq[-h:].T.bfloat16()


def encoded_history(x, history, drafts, prefill=False):
    if not x.shape[0]:
        return None
    h = history.shape[1]
    joined = torch.cat((history, x.T), 1)
    if prefill:
        return torch.cat(
            (joined[:, -h:], torch.zeros(x.shape[1], drafts).bfloat16()), 1
        )
    out = torch.zeros(x.shape[1], h + drafts).bfloat16()
    out[:, :h] = joined[:, 1 : 1 + h]
    out[:, h : h + min(drafts, x.shape[0] - 1)] = x[1 : 1 + drafts].T
    return out


def hash_packed(
    tokens,
    starts,
    history,
    geometry,
    eos,
    *,
    live_requests=None,
    live_tokens=None,
    slots=None,
):
    nr = len(starts) - 1 if live_requests is None else live_requests
    nt = len(tokens) if live_tokens is None else live_tokens
    out = torch.full((len(tokens), 16), -1, dtype=torch.int64)
    ids = torch.full((len(tokens),), -1, dtype=torch.int64)
    for r in range(nr):
        if slots is not None and slots[r] < 0:
            continue
        seq = list(map(int, history[r]))
        segment = 0
        for j, token in enumerate(seq):
            if token == eos:
                segment = j + 1
        for t in range(starts[r], min(starts[r + 1], nt)):
            seq.append(int(tokens[t]))
            here = len(seq) - 1
            for head in range(16):
                mixed = 0
                for lag in range(2 + head // 8):
                    pos = here - lag
                    value = seq[pos] if pos >= segment else eos
                    mixed ^= (value * geometry.multipliers[lag]) & ((1 << 64) - 1)
                if mixed >= 1 << 63:
                    mixed -= 1 << 64
                out[t, head] = (
                    mixed % geometry.prime_sizes[head] + geometry.table_offsets[head]
                )
            ids[t] = r
            if seq[-1] == eos:
                segment = len(seq)
    return out, ids
