"""Paged BF16 QSA with Flare-compatible compression and speculative history.

Main q/k arrive norm+RoPE applied; raw index q/key arrive before the indexer's
weighted norm/partial RoPE. No FP8 reinterpretation, full score matrix or
history recompression. Main MTP compression_ratio=0 must use FlashAttention.
"""

from __future__ import annotations
import ctypes as C
import torch
from ._native import PreparedNative, tensor

_READ = (
    "query",
    "new_key",
    "new_value",
    "index_query",
    "raw_key",
    "query_weight",
    "key_weight",
    "cosine",
    "sine",
    "request_ids",
    "positions",
    "rope_positions",
    "raw_slots",
    "slot_mapping",
    "starts",
    "live",
    "lengths",
    "accepted_previous",
    "prefilling",
    "main_table",
    "compressed_table",
)
_WRITE = (
    "key_cache",
    "value_cache",
    "compressed_cache",
    "raw_ring",
    "raw_tags",
    "raw_rope",
    "anchors",
    "encoded_query",
    "selected",
    "selected_count",
    "selected_groups",
    "selected_scores",
    "output",
    "errors",
)


class _Qsa(C.Structure):
    _fields_ = [(n, C.c_void_p) for n in (*_READ, *_WRITE)]
    _fields_ += [
        (n, C.c_int64)
        for n in (
            "query_stride",
            "key_stride",
            "value_stride",
            "index_stride",
            "raw_stride",
            "ring_slot_stride",
            "tag_slot_stride",
            "rope_slot_stride",
            "anchor_stride",
            "raw_slot_stride",
        )
    ]
    _fields_ += [
        (n, C.c_int32)
        for n in (
            "requests",
            "tokens",
            "pages",
            "raw_slot_count",
            "table_width",
            "page_size",
            "ring_size",
            "rope_rows",
            "query_heads",
            "kv_heads",
            "rope_bf16",
        )
    ]
    _fields_ += [("norm_epsilon", C.c_float)]


def qsa_workspace_specs(tokens, requests, query_heads=24):
    if min(tokens, requests, query_heads) <= 0:
        raise ValueError("QSA capacities must be positive")
    return {
        "encoded_query": ((tokens, 4, 128), torch.float32),
        "selected": ((tokens, 2051), torch.int32),
        "selected_count": ((tokens,), torch.int32),
        "selected_groups": ((tokens, 512), torch.int32),
        "selected_scores": ((tokens, 512), torch.float32),
        "output": ((tokens, query_heads, 256), torch.bfloat16),
        "errors": ((requests,), torch.int32),
    }


class PreparedQSA(PreparedNative):
    """Append → compress completed groups → retain ring → top-k → attention.

    All dictionaries contain caller-owned tensors. raw ring/anchors are
    speculative: next interval supplies its first logical position and the
    previous target-consumed count; rejected cache entries are invisible.
    Page64/128, BF16 KV, ratio4, selection2048+3, mRoPE(11,11,10).
    """

    def __init__(
        self,
        library,
        *,
        inputs,
        caches,
        workspace,
        norm_epsilon=1e-6,
        compression_ratio=4,
    ):
        if compression_ratio != 4:
            raise ValueError(
                "this QSA bridge is ratio4 only; ratio0 MTP attention belongs to FlashAttention"
            )
        if set(inputs) != set(_READ):
            raise ValueError(f"QSA inputs must be {sorted(_READ)}")
        if set(caches) != set(_WRITE[:7]):
            raise ValueError("QSA cache fields mismatch")
        q = inputs["query"]
        rows, hq, dim = q.shape
        dev = q.device
        pages, ps, hkv, cdim = caches["key_cache"].shape
        slots, ring, rdim = caches["raw_ring"].shape
        requests, table = inputs["main_table"].shape
        rope = inputs["cosine"].shape[0]
        rope_dtype = inputs["cosine"].dtype
        if (
            dim != 256
            or cdim != 256
            or rdim != 128
            or ps not in (64, 128)
            or ring < 12
            or ps % ring
            or min(rows, hq, hkv, pages, slots, requests, table, rope) <= 0
            or hq % hkv
            or not norm_epsilon > 0
            or rope_dtype not in (torch.bfloat16, torch.float32)
        ):
            raise ValueError("unsupported paged QSA geometry")
        b, f, i, l = torch.bfloat16, torch.float32, torch.int32, torch.int64
        specifications = {
            "query": ((rows, hq, 256), b),
            "new_key": ((rows, hkv, 256), b),
            "new_value": ((rows, hkv, 256), b),
            "index_query": ((rows, 4, 128), b),
            "raw_key": ((rows, 128), b),
            "query_weight": ((128,), f),
            "key_weight": ((128,), f),
            "cosine": ((rope, 32), rope_dtype),
            "sine": ((rope, 32), rope_dtype),
            "request_ids": ((rows,), l),
            "positions": ((rows,), l),
            "rope_positions": ((rows, 3), l),
            "raw_slots": ((requests,), l),
            "slot_mapping": ((rows,), l),
            "starts": ((requests + 1,), i),
            "live": ((1,), i),
            "lengths": ((requests,), i),
            "accepted_previous": ((requests,), i),
            "prefilling": ((requests,), torch.bool),
            "main_table": ((requests, table), i),
            "compressed_table": ((requests, table), i),
            "key_cache": ((pages, ps, hkv, 256), b),
            "value_cache": ((pages, ps, hkv, 256), b),
            "compressed_cache": ((pages, ps // 4, 128), b),
            "raw_ring": ((slots, ring, 128), b),
            "raw_tags": ((slots, ring), l),
            "raw_rope": ((slots, ring, 3), l),
            "anchors": ((slots,), l),
        }
        specs = qsa_workspace_specs(rows, requests, hq)
        if set(workspace) != set(specs):
            raise ValueError(f"QSA workspace must be {sorted(specs)}")
        specifications.update(specs)
        all_values = {**inputs, **caches, **workspace}
        strided = {
            "query",
            "new_key",
            "new_value",
            "index_query",
            "raw_key",
            "raw_slots",
            "raw_ring",
            "raw_tags",
            "raw_rope",
            "anchors",
        }
        for name, (shape, dtype) in specifications.items():
            tensor(
                all_values[name],
                shape=shape,
                dtype=dtype,
                device=dev,
                name=name,
                contiguous=name not in strided,
            )
        self.config = dict(
            requests=requests,
            tokens=rows,
            pages=pages,
            raw_slot_count=slots,
            table_width=table,
            page_size=ps,
            ring_size=ring,
            rope_rows=rope,
            query_heads=hq,
            kv_heads=hkv,
            rope_bf16=int(rope_dtype == torch.bfloat16),
            norm_epsilon=float(norm_epsilon),
        )
        self.strides = {
            "query_stride": "query",
            "key_stride": "new_key",
            "value_stride": "new_value",
            "index_stride": "index_query",
            "raw_stride": "raw_key",
            "ring_slot_stride": "raw_ring",
            "tag_slot_stride": "raw_tags",
            "rope_slot_stride": "raw_rope",
            "anchor_stride": "anchors",
            "raw_slot_stride": "raw_slots",
        }
        self.config.update(
            {field: all_values[name].stride(0) for field, name in self.strides.items()}
        )
        # Inner dimensions may not be transposed merely because their extents
        # happen to match. Pool views can have a larger outer stride.
        for name in strided:
            t = all_values[name]
            pitch = 1
            for n, s in reversed(list(zip(t.shape[1:], t.stride()[1:]))):
                if s != pitch:
                    raise ValueError(f"{name}: noncontiguous inner layout")
                pitch *= n
            if t.stride(0) < pitch:
                raise ValueError(f"{name}: overlapping rows")
        self.bind(
            library, [all_values[n] for n in _READ], [all_values[n] for n in _WRITE]
        )
        size = library.lib.ala_qsa_struct_size
        size.restype = C.c_int
        if size() != C.sizeof(_Qsa):
            raise RuntimeError("QSA C/Python ABI size mismatch")
        self.fn = library.lib.ala_qsa_run
        self.fn.restype = C.c_int
        self.fn.argtypes = [C.POINTER(_Qsa), C.c_void_p]

    def launch(self, inputs, buffers):
        args = _Qsa(**self.config)
        values = dict(zip((*_READ, *_WRITE), (*inputs, *buffers)))
        for name, t in values.items():
            setattr(args, name, t.data_ptr())
        for field, name in self.strides.items():
            setattr(args, field, values[name].stride(0))
        self.check(self.fn(C.byref(args), self.stream()))
