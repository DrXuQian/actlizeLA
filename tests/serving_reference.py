"""Independent, allocation-heavy CPU references, NEVER a production fallback."""

import math
import torch


def gdn_reference(mixed, a, b, a_log, bias, initial, hk, count=None):
    """Sequential mathematical recurrence, VxK state and raw a/b inputs."""
    hv = initial.shape[0]
    count = mixed.shape[0] if count is None else count
    x = mixed[:count].float()
    query = x[:, : hk * 128].reshape(count, hk, 128)
    key = x[:, hk * 128 : 2 * hk * 128].reshape(count, hk, 128)
    value = x[:, 2 * hk * 128 :].reshape(count, hv, 128)
    query = (
        query / torch.sqrt(query.square().sum(-1, keepdim=True) + 1e-6) / math.sqrt(128)
    )
    key = key / torch.sqrt(key.square().sum(-1, keepdim=True) + 1e-6)
    logs = -a_log.float().exp() * torch.nn.functional.softplus(
        a[:count].float() + bias.float(), threshold=20
    )
    decay = logs.exp()
    beta = b[:count].float().sigmoid().bfloat16().float()
    head = torch.arange(hv) // (hv // hk)
    state = initial.float().clone()
    outputs, delta = [], []
    for t in range(count):
        state = state * decay[t, :, None, None]
        residual = (value[t] - torch.einsum("hvk,hk->hv", state, key[t, head])) * beta[
            t, :, None
        ]
        state = state + residual[:, :, None] * key[t, head, None, :]
        outputs.append(torch.einsum("hvk,hk->hv", state, query[t, head]).bfloat16())
        delta.append(residual)
    empty = mixed.new_empty((0, hv, 128))
    return (
        torch.stack(outputs) if outputs else empty,
        state,
        torch.stack(delta) if delta else empty.float(),
        key,
        decay,
    )


def replay_records(initial, delta, keys, decay, count):
    state = initial.clone()
    head = torch.arange(state.shape[0]) // (state.shape[0] // keys.shape[1])
    for t in range(count):
        state = (
            decay[t, :, None, None] * state
            + delta[t, :, :, None] * keys[t, head, None, :]
        )
    return state


def index_encode(x, weight, positions, cosine, sine, eps=1e-6):
    """Zero-centered weighted RMSNorm; BF16 products of partial mRoPE."""
    z = (
        x.float()
        * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
        * (1 + weight.float())
    ).bfloat16()
    pair = torch.arange(32)
    axes = torch.where(
        (pair % 3 == 1) & (pair < 33),
        1,
        torch.where((pair % 3 == 2) & (pair < 30), 2, 0),
    )
    co = cosine[positions[..., axes], pair]
    si = sine[positions[..., axes], pair]
    while co.ndim < z.ndim:
        co = co.unsqueeze(-2)
        si = si.unsqueeze(-2)
    lo, hi = z[..., :32], z[..., 32:64]
    if cosine.dtype == torch.float32:
        return torch.cat(
            (
                lo.float() * co - hi.float() * si,
                hi.float() * co + lo.float() * si,
                z[..., 64:].float(),
            ),
            -1,
        )
    a = (
        (lo.float() * co.float()).bfloat16().float()
        + (-hi.float() * si.float()).bfloat16().float()
    ).bfloat16()
    b = (
        (hi.float() * co.float()).bfloat16().float()
        + (lo.float() * si.float()).bfloat16().float()
    ).bfloat16()
    return torch.cat((a, b, z[..., 64:]), -1)


def compressed_history(raw, weight, rope, cosine, sine):
    groups = raw.shape[0] // 4
    pooled = raw[: groups * 4].reshape(groups, 4, 128).float().mean(1).bfloat16()
    return index_encode(
        pooled, weight, rope[torch.arange(groups) * 4], cosine, sine
    ).bfloat16()


def select_reference(encoded_query, compressed, position, length):
    count = min((position + 1) // 4, length // 4, compressed.shape[0])
    scores = torch.einsum(
        "hd,bd->bh", encoded_query.float(), compressed[:count].float()
    ).relu().sum(-1) / math.sqrt(128)
    groups = torch.argsort(scores, descending=True, stable=True)[:512]
    selected = (groups[:, None] * 4 + torch.arange(4)[None, :]).flatten()
    tail = torch.arange((position + 1) // 4 * 4, position + 1)
    positions = torch.full((2051,), -1, dtype=torch.int32)
    joined = torch.cat((selected, tail))
    positions[: joined.numel()] = joined.int()
    return positions, groups, scores[groups]


def attention_reference(q, k, v, selected):
    positions = selected[selected >= 0].long()
    if not positions.numel():
        return torch.zeros_like(q)
    h = q.shape[0]
    kv = k.shape[1]
    head = torch.arange(h) // (h // kv)
    logits = torch.einsum(
        "hd,thd->ht", q.float(), k[positions][:, head].float()
    ) / math.sqrt(q.shape[-1])
    return torch.einsum(
        "ht,thd->hd", logits.softmax(-1), v[positions][:, head].float()
    ).bfloat16()


def merged_topk(scores, window=512):
    """Independent proof of bounded carry + new candidates vs full stable sort."""
    carry = []
    for first in range(0, len(scores), window):
        carry = sorted(
            carry + list(range(first, min(first + window, len(scores)))),
            key=lambda i: (-scores[i], i),
        )[:window]
    return carry
