"""Prepared non-GEMM helpers. All tensors, errors and optional traces are caller-owned.

No b12x, Triton, projection or execution allocation. The versioned C header is
csrc/serving/aux_v1.h; the existing serving ABI and wrappers remain unchanged.
"""

from __future__ import annotations
import ctypes as C
from dataclasses import dataclass
import torch
from ._native import PreparedNative, _plans, _handles


class TensorV1(C.Structure):
    _fields_ = [
        ("data", C.c_void_p),
        ("nbytes", C.c_uint64),
        ("shape", C.c_int64 * 3),
        ("stride", C.c_int64 * 3),
        ("dtype", C.c_int32),
        ("rank", C.c_int32),
    ]


class HeaderV1(C.Structure):
    _fields_ = [("version", C.c_uint32), ("struct_bytes", C.c_uint32)]


class RowsV1(C.Structure):
    _fields_ = [
        (n, TensorV1)
        for n in ("starts", "slots", "num_seqs", "num_tokens", "upstream_errors")
    ]
    _fields_ += [("requests", C.c_int32), ("tokens", C.c_int32)]


class PointwiseV1(C.Structure):
    _fields_ = [("header", HeaderV1), ("rows", RowsV1)]
    _fields_ += [
        (n, TensorV1)
        for n in (
            "x",
            "weight",
            "gate",
            "value",
            "output",
            "combined",
            "trace",
            "errors",
        )
    ]
    _fields_ += [
        (n, C.c_int32)
        for n in (
            "kind",
            "groups",
            "width",
            "zero_centered",
            "weight_per_group",
            "round_normalized",
            "norm_before_gate",
            "silu_gate",
        )
    ]
    _fields_ += [("epsilon", C.c_float)]


class ShortConvV1(C.Structure):
    _fields_ = [("header", HeaderV1), ("rows", RowsV1)]
    _fields_ += [
        (n, TensorV1)
        for n in (
            "x",
            "weight",
            "state",
            "fresh",
            "previous_accepted",
            "output",
            "snapshot",
            "stamps",
            "consumed_prefix",
            "errors",
        )
    ]
    _fields_ += [(n, C.c_int32) for n in ("kind", "channels", "dilation", "drafts")]


class HashV1(C.Structure):
    _fields_ = [("header", HeaderV1), ("rows", RowsV1)]
    _fields_ += [
        (n, TensorV1)
        for n in (
            "tokens",
            "history",
            "multipliers",
            "primes",
            "offsets",
            "hashes",
            "request_ids",
            "errors",
        )
    ]
    _fields_ += [(n, C.c_int64) for n in ("eos", "vocab_size", "padded_vocab_size")]


_DTYPE = {
    torch.bfloat16: 1,
    torch.float32: 2,
    torch.int32: 3,
    torch.int64: 4,
    torch.uint8: 5,
    torch.bool: 5,
}


def tensor_descriptor(t):
    if t is None:
        return TensorV1()
    if not isinstance(t, torch.Tensor) or t.dtype not in _DTYPE or not 1 <= t.ndim <= 3:
        raise ValueError("auxiliary tensor requires BF16/FP32/I32/I64/U8 and rank1..3")
    return TensorV1(
        data=t.data_ptr(),
        nbytes=t.untyped_storage().nbytes() - t.storage_offset() * t.element_size(),
        shape=tuple(t.shape) + (0,) * (3 - t.ndim),
        stride=tuple(t.stride()) + (0,) * (3 - t.ndim),
        dtype=_DTYPE[t.dtype],
        rank=t.ndim,
    )


@dataclass(frozen=True)
class ServingRows:
    """Flat live rows or packed request metadata; values change only on device.

    Packed slots use -1 inactive / 0 valid. upstream_errors is READ-only; each
    operation owns a separate error output, never overwriting earlier failures.
    """

    tokens: int
    num_tokens: torch.Tensor | None = None
    starts: torch.Tensor | None = None
    slots: torch.Tensor | None = None
    num_seqs: torch.Tensor | None = None
    upstream_errors: torch.Tensor | None = None

    @property
    def requests(self):
        return 0 if self.starts is None else self.starts.numel() - 1

    def tensors(self):
        return {
            "rows_" + n: getattr(self, n)
            for n in ("starts", "slots", "num_seqs", "num_tokens", "upstream_errors")
            if getattr(self, n) is not None
        }

    def descriptor(self, values):
        return RowsV1(
            **{
                n: tensor_descriptor(values.get("rows_" + n))
                for n in (
                    "starts",
                    "slots",
                    "num_seqs",
                    "num_tokens",
                    "upstream_errors",
                )
            },
            requests=self.requests,
            tokens=self.tokens,
        )


class PreparedAuxiliary(PreparedNative):
    """Thin owner of a validated descriptor, not an allocator or tuner."""

    def _bind(self, library, cls, kind, symbol, rows, values, writes, config):
        self.library, self.cls, self.rows, self.config = library, cls, rows, config
        values = {
            **rows.tensors(),
            **{n: t for n, t in values.items() if t is not None},
        }
        self.write_names = tuple(n for n in values if n in writes)
        self.read_names = tuple(n for n in values if n not in writes)
        self.inputs = [values[n] for n in self.read_names]
        self.buffers = [values[n] for n in self.write_names]
        self.device = next(iter(values.values())).device
        if self.device.type != "cuda" or any(
            t.device != self.device for t in values.values()
        ):
            raise ValueError(
                "prepared auxiliary execution requires one CUDA-compatible device"
            )
        self.signatures = [
            (tuple(t.shape), t.dtype) for t in (*self.inputs, *self.buffers)
        ]
        sizes = library.lib.ala_aux_v1_struct_size
        sizes.argtypes = [C.c_int]
        sizes.restype = C.c_int
        for index, structure in ((0, TensorV1), (1, RowsV1), (kind, cls)):
            if sizes(index) != C.sizeof(structure):
                raise RuntimeError("auxiliary C/Python ABI size mismatch")
        self.validate = library.lib.ala_aux_v1_validate
        self.validate.argtypes = [C.c_int, C.c_void_p]
        self.validate.restype = C.c_int
        self.fn = getattr(library.lib, symbol)
        self.fn.argtypes = [C.POINTER(cls), C.c_void_p]
        self.fn.restype = C.c_int
        args = self._args(self.inputs, self.buffers)
        if self.validate(kind, C.byref(args)):
            raise ValueError(
                f"{symbol}: invalid dtype/shape/stride/capacity/alias contract"
            )
        self.handle = next(_handles)
        _plans[self.handle] = self

    def _args(self, inputs, buffers):
        values = dict(zip((*self.read_names, *self.write_names), (*inputs, *buffers)))
        args = self.cls(
            header=HeaderV1(1, C.sizeof(self.cls)),
            rows=self.rows.descriptor(values),
            **self.config,
        )
        for name, t in values.items():
            if not name.startswith("rows_"):
                setattr(args, name, tensor_descriptor(t))
        return args

    def check_metadata(self, inputs, buffers):
        if len(inputs) != len(self.inputs) or len(buffers) != len(self.buffers):
            raise ValueError("auxiliary operand denominator changed")
        for t, (shape, dtype) in zip((*inputs, *buffers), self.signatures):
            if tuple(t.shape) != shape or t.dtype != dtype or t.device != self.device:
                raise ValueError("auxiliary static tensor contract changed")

    def check_operands(self, inputs, buffers):
        self.check_metadata(inputs, buffers)
        # Exact stride/capacity/alias validation is performed by the native
        # launch on the concrete descriptors, never on fake device addresses.

    def launch(self, inputs, buffers):
        args = self._args(inputs, buffers)
        self.check(self.fn(C.byref(args), self.stream()))


def _logical(x, groups):
    if x is None:
        return None
    if x.ndim == 2:
        if x.shape[1] % groups:
            raise ValueError("logical groups do not divide width")
        return x.unflatten(1, (groups, x.shape[1] // groups))
    if x.ndim != 3 or x.shape[1] != groups:
        raise ValueError("expected logical [tokens,groups,width]")
    return x


class PreparedPointwise(PreparedAuxiliary):
    """Named Norm/HC/PLE primitive. Disjoint outputs only, no hidden GEMM.

    Optional trace exposes rounded intermediates in FP32 storage. It is an
    explicit caller-owned output and can be omitted in the serving chain.
    """

    KINDS = {
        "grouped_norm": 0,
        "gated_norm": 1,
        "hc_scaled_silu": 2,
        "hc_gate_mean": 3,
        "hc_combine": 4,
        "hc_combine_norm": 5,
        "ple_gate": 6,
        "ple_contribution": 7,
    }

    def __init__(
        self,
        library,
        operation,
        *,
        x,
        output,
        errors,
        rows=None,
        groups=4,
        weight=None,
        gate=None,
        value=None,
        combined=None,
        trace=None,
        epsilon=1e-6,
        zero_centered=False,
        round_normalized=True,
        norm_before_gate=True,
        activation="sigmoid",
    ):
        if operation not in self.KINDS or activation not in ("sigmoid", "silu"):
            raise ValueError("unknown auxiliary operation/activation")
        kind = self.KINDS[operation]
        x = _logical(x, groups)
        T, G, D = x.shape
        if rows is None:
            rows = ServingRows(T)
        if rows.tokens != T:
            raise ValueError("row capacity differs from input")
        output = _logical(output, 1 if kind == 3 else G)
        if gate is not None and kind not in (4, 5):
            gate = _logical(gate, G)
        if value is not None:
            value = _logical(value, G if kind == 7 else 1)
        if combined is not None:
            combined = _logical(combined, G)
        per_group = True
        if weight is not None:
            if kind == 1 and weight.numel() == D:
                per_group = False
            weight = weight.view(G if per_group else 1, D)
        values = dict(
            x=x,
            weight=weight,
            gate=gate,
            value=value,
            output=output,
            combined=combined,
            trace=trace,
            errors=errors,
        )
        config = dict(
            kind=kind,
            groups=G,
            width=D,
            epsilon=float(epsilon),
            zero_centered=int(zero_centered),
            weight_per_group=int(per_group),
            round_normalized=int(round_normalized),
            norm_before_gate=int(norm_before_gate),
            silu_gate=int(activation == "silu"),
        )
        self._bind(
            library,
            PointwiseV1,
            2,
            "ala_pointwise_v1_run",
            rows,
            values,
            {"output", "combined", "trace", "errors"},
            config,
        )


class PreparedShortConv(PreparedAuxiliary):
    """kernel4 dilation1/3. Verify commits input0 and retains later inputs.

    Plain decode is verify width1. previous_accepted must stay positive after
    a zero-consumption rollback; restore does not rewrite framework counters.
    """

    def __init__(
        self,
        library,
        *,
        x,
        weight,
        state,
        output,
        rows,
        fresh,
        previous_accepted,
        errors,
        dilation,
        mode="verify",
    ):
        if mode not in ("prefill", "verify"):
            raise ValueError("unsupported convolution mode")
        if state.ndim != 3 or dilation not in (1, 3):
            raise ValueError("invalid convolution state")
        channels = state.shape[1]
        drafts = state.shape[2] - 3 * dilation
        config = dict(
            kind=0 if mode == "prefill" else 1,
            channels=channels,
            dilation=dilation,
            drafts=drafts,
        )
        values = dict(
            x=x,
            weight=weight,
            state=state,
            output=output,
            fresh=fresh,
            previous_accepted=previous_accepted,
            errors=errors,
        )
        self._bind(
            library,
            ShortConvV1,
            3,
            "ala_shortconv_v1_run",
            rows,
            values,
            {"state", "output", "errors"},
            config,
        )


class PreparedConvHistoryV1(PreparedAuxiliary):
    """Versioned history transport: slot0 valid, arbitrary pitches, dilation3.

    Legacy PreparedConvHistory remains unchanged (its slot0-null GDN contract).
    """

    def __init__(
        self,
        library,
        *,
        state,
        snapshot,
        stamps,
        consumed_prefix,
        errors,
        rows,
        dilation,
        restore=False,
    ):
        channels = state.shape[1]
        drafts = state.shape[2] - 3 * dilation
        config = dict(
            kind=3 if restore else 2,
            channels=channels,
            dilation=dilation,
            drafts=drafts,
        )
        values = dict(
            state=state,
            snapshot=snapshot,
            stamps=stamps,
            consumed_prefix=consumed_prefix,
            errors=errors,
        )
        writes = {"state", "errors"} if restore else {"snapshot", "stamps", "errors"}
        self._bind(
            library,
            ShortConvV1,
            3,
            "ala_shortconv_v1_run",
            rows,
            values,
            writes,
            config,
        )


class PreparedPLEHash(PreparedAuxiliary):
    def __init__(
        self,
        library,
        *,
        tokens,
        history,
        multipliers,
        primes,
        offsets,
        hashes,
        request_ids,
        errors,
        rows,
        geometry,
        eos,
    ):
        # Checkpoint parameters are immutable weights. Check their 280 bytes
        # ONCE during preparation; this readback never occurs in launch().
        for name, value, expected in (
            ("multipliers", multipliers, geometry.multipliers),
            ("primes", primes, geometry.prime_sizes),
            ("offsets", offsets, geometry.table_offsets),
        ):
            if value.dtype != torch.int64 or tuple(value.shape) != (len(expected),):
                raise ValueError(f"{name}: expected fixed int64 geometry")
            if tuple(value.detach().cpu().tolist()) != tuple(expected):
                raise ValueError(
                    f"{name}: device checkpoint differs from host geometry"
                )
        config = dict(
            eos=eos,
            vocab_size=geometry.vocab_size,
            padded_vocab_size=geometry.padded_vocab_size,
        )
        values = dict(
            tokens=tokens,
            history=history,
            multipliers=multipliers,
            primes=primes,
            offsets=offsets,
            hashes=hashes,
            request_ids=request_ids,
            errors=errors,
        )
        self._bind(
            library,
            HashV1,
            4,
            "ala_ple_hash_v1_run",
            rows,
            values,
            {"hashes", "request_ids", "errors"},
            config,
        )
