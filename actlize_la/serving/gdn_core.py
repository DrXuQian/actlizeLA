"""Prepared PPU1.0 residual on processed operands, in public KV state order.

Unlike PreparedGDN this entry does not normalize Q/K, transform gates, or move
slot state. It shares the product native dispatch with public and serving calls.
"""

import ctypes as C
import torch

from ._native import PreparedNative, tensor
from .gdn import _require_native_prefill, gdn_workspace_specs


def gdn_core_workspace_specs(batch, length, key_heads, value_heads):
    """Only the four retained-core buffers; inverse keeps its 128x128 pitch."""
    if (min(batch, length, key_heads, value_heads) < 1 or
            max(batch, length, key_heads * 128, value_heads * 128) > 2**31 - 1 or
            batch * value_heads * max(4, (length + 63) // 64) > 2**31 - 1):
        raise ValueError("retained GDN geometry exceeds native grid limits")
    specs = gdn_workspace_specs(batch, length, key_heads, value_heads,
                                mode="prefill", target="ppu10")
    return {name: specs[name] for name in ("inverse", "snapshots", "vnew", "gates")}


def gdn_core_route(library, batch, length, key_heads, value_heads, *, gate_fp32=True):
    """Inspect the same native shape/type policy, without executing kernels."""
    names = {0: "retained", 1: "solve8", 2: "solve8-value64-warps8", 3: "native-bf16"}
    query = getattr(library.lib, "actlize_la_ppu10_core_route_v2", None)
    if query is not None:
        query.argtypes, query.restype = [C.c_int] * 4 + [C.c_bool], C.c_int
        selected = query(batch, length, key_heads, value_heads, gate_fp32)
    else:
        legacy = getattr(library.lib, "actlize_la_ppu10_core_route", None)
        selected = 0
        if legacy is not None:
            legacy.argtypes, legacy.restype = [C.c_int] * 4, C.c_int
            selected = legacy(batch, length, key_heads, value_heads)
    if selected not in names:
        raise RuntimeError("unknown native GDN core route")
    return {"id": selected, "name": names[selected], "entry": "actlize_la_ppu10_forward_prepared"}


def _geometry(q, k, v, g, beta, initial, output, final):
    if q.ndim != 4 or v.ndim != 4:
        raise ValueError("Q/K/V must have [B,T,H,128] geometry")
    batch, length, hk, dim = q.shape
    hv = v.shape[2]
    if (dim != 128 or min(batch, length, hk, hv) < 1 or hv % hk or
            max(batch, length, hk * 128, hv * 128) > 2**31 - 1 or
            batch * hv * max(4, (length + 63) // 64) > 2**31 - 1):
        raise ValueError("unsupported retained GDN geometry or native grid overflow")
    bf16, fp32 = torch.bfloat16, torch.float32
    for name, value, shape, dtype in (
        ("q", q, (batch, length, hk, 128), bf16),
        ("k", k, (batch, length, hk, 128), bf16),
        ("v", v, (batch, length, hv, 128), bf16),
        ("beta", beta, (batch, length, hv), bf16),
        ("g", g, (batch, length, hv), g.dtype),
        ("output", output, (batch, length, hv, 128), bf16),
    ):
        tensor(value, shape=shape, dtype=dtype, device=q.device, name=name)
    if g.dtype not in (bf16, fp32):
        raise ValueError("g must be BF16/FP32 natural-log decay, not raw logits")
    for name, value in (("initial_state", initial), ("final_state", final)):
        if value is not None:
            tensor(value, shape=(batch, hv, 128, 128), dtype=fp32,
                   device=q.device, name=name)
    return batch, length, hk, hv


def _configure(library, device):
    """Attributes are context/device-specific; configure once, before capture."""
    configured = getattr(library, "_gdn_core_configured_devices", set())
    key = str(device)
    if key not in configured:
        with torch.cuda.device(device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("prepare/warm up the PPU1.0 GDN library before capture")
            fn = library.lib.actlize_la_ppu10_configure
            fn.argtypes, fn.restype = [], C.c_int
            PreparedNative.check(fn())
        configured.add(key)
        library._gdn_core_configured_devices = configured


class PreparedGDNCore(PreparedNative):
    """Caller-owned processed-input forward. Keep this owner alive with its graph.

    Q/K/V/beta are BF16; g is natural-log BF16/FP32. Q scale is applied once by
    the retained kernel. Initial/final state are FP32 [B,Hv,K,V], NOT slot VK.
    A missing initial state means zero; missing final state suppresses export.
    Inputs are read-only; output, optional final and all scratch are disjoint.
    """

    def __init__(self, library, *, q, k, v, g, beta, output, workspace,
                 initial_state=None, final_state=None):
        if library.target != "ppu10":
            raise ValueError("PreparedGDNCore currently supports native PPU1.0 only")
        _require_native_prefill(library)
        self.geometry = _geometry(q, k, v, g, beta, initial_state, output, final_state)
        specs = gdn_core_workspace_specs(*self.geometry)
        if set(workspace) != set(specs):
            raise ValueError("retained GDN workspace keys mismatch")
        for name, (shape, dtype) in specs.items():
            tensor(workspace[name], shape=shape, dtype=dtype, device=q.device, name=name)
        self.has_initial = initial_state is not None
        self.has_final = final_state is not None
        inputs = [q, k, v, g, beta] + ([initial_state] if self.has_initial else [])
        buffers = [output] + ([final_state] if self.has_final else [])
        buffers += [workspace[name] for name in specs]
        self.bind(library, inputs, buffers)
        self.fn = library.lib.actlize_la_ppu10_forward_prepared
        self.fn.argtypes = [C.c_void_p] * 12 + [C.c_int] * 4 + [C.c_bool, C.c_void_p]
        self.fn.restype = C.c_int
        _configure(library, self.device)
        self.core_route = gdn_core_route(library, *self.geometry, gate_fp32=g.dtype == torch.float32)

    def launch(self, inputs, buffers):
        initial = inputs[5].data_ptr() if self.has_initial else None
        final = buffers[1].data_ptr() if self.has_final else None
        scratch = buffers[1 + int(self.has_final):]
        self.check(self.fn(
            *(value.data_ptr() for value in inputs[:5]), initial,
            buffers[0].data_ptr(), final, *(value.data_ptr() for value in scratch),
            *self.geometry, inputs[3].dtype == torch.float32, self.stream()))
