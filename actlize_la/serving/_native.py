"""Ahead-of-time C ABI and mutation-visible graph bridge; no JIT loader."""

from __future__ import annotations

import ctypes as C
import itertools
from pathlib import Path
from weakref import WeakValueDictionary

import torch

_plans = WeakValueDictionary()
_handles = itertools.count(1)


@torch.library.custom_op("actlize_la::serving_out", mutates_args=("buffers",))
def _serving_out(
    inputs: list[torch.Tensor], buffers: list[torch.Tensor], plan: int
) -> None:
    try:
        owner = _plans[plan]
    except KeyError as exc:
        raise RuntimeError(
            "serving plan expired; retain the prepared object through graph lifetime"
        ) from exc
    owner.check_operands(inputs, buffers)
    with torch.cuda.device(owner.device):
        owner.launch(inputs, buffers)


@_serving_out.register_fake
def _fake_serving_out(inputs, buffers, plan):
    return None


class NativeLibrary:
    """Load once during preparation. The target is explicit, not device-profile selected."""

    def __init__(self, path, *, target):
        expected = {"ppu10": 100, "ppu15": 150, "cuda_sm120": 1200}
        if target not in expected:
            raise ValueError(f"unsupported serving target {target!r}")
        self.path = Path(path).resolve(strict=True)
        self.lib = C.CDLL(str(self.path))
        for name in (
            "ala_serving_version",
            "ala_serving_target",
            "ala_prefill_configure",
        ):
            getattr(self.lib, name).restype = C.c_int
        self.lib.ala_serving_struct_size.argtypes = [C.c_int]
        self.lib.ala_serving_struct_size.restype = C.c_int
        if (
            self.lib.ala_serving_version() != 1
            or self.lib.ala_serving_target() != expected[target]
        ):
            raise ValueError("serving ABI/target mismatch; no binary fallback")
        self.target = target

    def function(self, name, struct, which):
        if self.lib.ala_serving_struct_size(which) != C.sizeof(struct):
            raise RuntimeError(f"C/Python ABI size mismatch for {struct.__name__}")
        fn = getattr(self.lib, name)
        fn.argtypes = [C.POINTER(struct), C.c_void_p]
        fn.restype = C.c_int
        return fn


def tensor(value, *, shape, dtype, device, name, contiguous=True):
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != tuple(shape)
        or value.dtype != dtype
        or value.device != device
        or (contiguous and not value.is_contiguous())
    ):
        raise ValueError(
            f"{name}: expected {dtype} {shape} on {device}"
            + (" contiguous" if contiguous else "")
        )
    return value


def reject_aliases(inputs, buffers):
    """Conservative byte-span check at prepare, including strided pool views.

    Read/read aliasing is fine. Every writer must be disjoint from other
    operands. An in-place state is passed ONCE, as a mutable buffer.
    """

    def spans(t):
        if not t.numel():
            return []
        lo = t.data_ptr()
        item = t.element_size()
        if t.is_contiguous():
            return [(lo, lo + t.numel() * item)]
        inner = 1
        for n, s in reversed(list(zip(t.shape[1:], t.stride()[1:]))):
            if s != inner:
                raise ValueError("serving views may stride the outer row only")
            inner *= n
        if t.stride(0) < inner:
            raise ValueError("overlapping serving rows are unsupported")
        return [
            (lo + i * t.stride(0) * item, lo + i * t.stride(0) * item + inner * item)
            for i in range(t.shape[0])
        ]

    def overlap(a, b):
        i = j = 0
        while i < len(a) and j < len(b):
            if a[i][0] < b[j][1] and b[j][0] < a[i][1]:
                return True
            if a[i][1] <= b[j][0]:
                i += 1
            else:
                j += 1
        return False

    read = [spans(t) for t in inputs]
    written = []
    for value in buffers:
        current = spans(value)
        if any(overlap(current, prior) for prior in (*read, *written)):
            raise ValueError(
                "serving mutable buffers must be disjoint from input and other mutable byte spans"
            )
        written.append(current)


class PreparedNative:
    def bind(self, library, inputs, buffers):
        if not inputs or not inputs[0].is_cuda:
            raise ValueError(
                "prepared serving execution requires device tensors (CUDA-compatible PPU or NVIDIA)"
            )
        self.library, self.inputs, self.buffers = library, list(inputs), list(buffers)
        self.device = inputs[0].device
        if any(t.device != self.device for t in (*inputs, *buffers)):
            raise ValueError("all serving operands must be on one device")
        reject_aliases(inputs, buffers)
        self.signatures = [
            (tuple(t.shape), t.dtype, t.stride()[1:], t.is_contiguous())
            for t in (*inputs, *buffers)
        ]
        self.pointers = tuple(t.data_ptr() for t in (*inputs, *buffers))
        self.handle = next(_handles)
        _plans[self.handle] = self

    def __call__(self):
        _serving_out(self.inputs, self.buffers, self.handle)

    def check_operands(self, inputs, buffers):
        values = (*inputs, *buffers)
        if len(inputs) != len(self.inputs) or len(buffers) != len(self.buffers):
            raise ValueError("serving operand denominator changed")
        for value, (shape, dtype, inner, contiguous) in zip(values, self.signatures):
            if (
                tuple(value.shape) != shape
                or value.dtype != dtype
                or value.device != self.device
                or value.stride()[1:] != inner
                or (contiguous and not value.is_contiguous())
            ):
                raise ValueError(
                    "serving operand shape/dtype/inner layout changed after preparation"
                )
        if tuple(t.data_ptr() for t in values) != self.pointers:
            reject_aliases(inputs, buffers)

    def stream(self):
        # Stream lookup is not a device-profile query, readback or synchronize.
        return C.c_void_p(torch.cuda.current_stream(self.device).cuda_stream)

    @staticmethod
    def check(rc):
        if rc:
            raise RuntimeError(f"actlizeLA serving launch failed: status={rc}")
