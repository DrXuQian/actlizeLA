"""Public Python API for the GDN (gated delta rule) chunked linear attention.

Wraps the compiled CUDA extension `actlize_la._gdn_chunk` with shape checks
and input normalization. Forward-only (no backward in this release).
"""

from __future__ import annotations

import torch
from .backends.loading import load_original, clear_backend_cache


def _backend():
    """Explicit PPU selection; never silently execute an NVIDIA fallback."""
    return load_original()


_backend.cache_clear = clear_backend_cache

__all__ = ["gdn_chunk", "gdn_chunk_twolevel", "gdn_chunk_reference", "GDN_D", "GDN_CHUNK"]

GDN_D = 128          # head dim (compile-time constant in the kernels)
GDN_CHUNK = 16       # intra-chunk size (compile-time constant)


def _check_inputs(q, k, v, g, beta):
    if q.dim() != 4 or q.size(3) != GDN_D:
        raise ValueError(f"q must be [B,S,Hk,{GDN_D}], got {tuple(q.shape)}")
    if k.shape != q.shape:
        raise ValueError(f"k must match q, got q={tuple(q.shape)} k={tuple(k.shape)}")
    B, S, Hk, _ = q.shape
    if v.dim() != 4 or v.size(3) != GDN_D or v.shape[:2] != (B, S):
        raise ValueError(f"v must be [B,S,Hv,{GDN_D}], got {tuple(v.shape)}")
    Hv = v.size(2)
    if min(B, S, Hk, Hv) <= 0:
        raise ValueError("B, S and both head counts must be positive")
    if g.shape != (B, S, Hv) or beta.shape != (B, S, Hv):
        raise ValueError("g and beta must be [B,S,Hv]")
    if Hv % Hk != 0:
        raise ValueError(f"Hv must be a multiple of Hk, got Hk={Hk} Hv={Hv}")
    if any(x.dtype != torch.bfloat16 for x in (q, k, v, g, beta)):
        raise ValueError("the native GDN kernels require BF16 q/k/v/g/beta")
    if any(x.device != q.device or not x.is_cuda for x in (q, k, v, g, beta)):
        raise ValueError("all inputs must be on the same CUDA/PPU device")
    return B, S, Hk, Hv


def gdn_chunk(q, k, v, g, beta, output_final_state=True):
    """Production GDN forward with automatic serial/reset-fastpath dispatch.

    Args:
        q, k: [B, S, Hk, D] bf16 — QK heads.
        v:    [B, S, Hv, D] bf16 — V heads (Hv must be a multiple of Hk).
        g, beta: [B, S, Hv] bf16 — log-decay and update gate.
        output_final_state: whether to return the final state.

    Returns:
        out [B, S, Hv, D], and final_state [B, Hv, D, D] if requested.
    """
    q, k, v, g, beta = (x.contiguous() for x in (q, k, v, g, beta))
    _check_inputs(q, k, v, g, beta)
    with torch.cuda.device(q.device):
        r = _backend().forward_gdn_chunk_auto(q, k, v, g, beta, output_final_state)
    return r[0], r[1]


def gdn_chunk_twolevel(q, k, v, g, beta, group_chunks=64, eps=1e-6, frac=1.0, gt_eps=1e-2):
    """Explicit two-level superchunk scan (reset/scan dispatch).

    Lower-level entry that surfaces the group size. See `gdn_chunk` for the
    production auto-dispatching path.
    """
    q, k, v, g, beta = (x.contiguous() for x in (q, k, v, g, beta))
    _check_inputs(q, k, v, g, beta)
    if not isinstance(group_chunks, int) or group_chunks <= 0:
        raise ValueError("group_chunks must be a positive integer")
    if not (eps >= 0 and gt_eps >= 0 and 0 < frac <= 1):
        raise ValueError("require eps >= 0, gt_eps >= 0, 0 < frac <= 1")
    with torch.cuda.device(q.device):
        r = _backend().forward_gdn_chunk_twolevel(q, k, v, g, beta, group_chunks, eps, frac, gt_eps)
    return r[0], r[1], r[2]


def gdn_chunk_reference(q, k, v, g, beta, chunk_size=64, mode="chunk"):
    """Pure-torch reference (correctness anchor). Returns (out, final_state)."""
    from .reference.gdn_chunk_ref import (
        torch_chunk_gated_delta_rule,
        torch_recurrent_gated_delta_rule,
    )
    fn = torch_chunk_gated_delta_rule if mode == "chunk" else torch_recurrent_gated_delta_rule
    return fn(q, k, v, g, beta, chunk_size=chunk_size, output_final_state=True)
