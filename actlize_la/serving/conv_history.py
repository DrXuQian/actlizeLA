"""Transport for zero-consumption rollback; does not compute ShortConv."""

import ctypes as C
import torch
from ._native import PreparedNative, tensor


class _Conv(C.Structure):
    _fields_ = [
        (n, C.c_void_p)
        for n in (
            "state",
            "snapshot",
            "slots",
            "recorded_slots",
            "live",
            "counts",
            "errors",
        )
    ]
    _fields_ += [(n, C.c_int64) for n in ("state_stride", "slot_stride")]
    _fields_ += [
        (n, C.c_int32) for n in ("requests", "state_slots", "channels", "columns")
    ]


class PreparedConvHistory(PreparedNative):
    """Bind one of capture-before-conv / restore-zero-after-acceptance.

    Positive consumption retains Flare's encoded history+tail: its existing
    next-call `accepted_tokens - 1` selection still applies. For count zero,
    restore the entire pre-verify convolution row; Flare's previous_accepted
    must remain unchanged, as PreparedVerificationState currently specifies.
    """

    def __init__(
        self,
        library,
        *,
        state,
        snapshot,
        slots,
        recorded_slots,
        live,
        consumed_prefix,
        errors,
        restore,
    ):
        n, ch, cols = state.shape
        r = slots.numel()
        dev = state.device
        if (
            n < 2
            or ch < 1
            or cols not in range(3, 11)
            or state.stride()[1:] != (cols, 1)
            or state.stride(0) < ch * cols
        ):
            raise ValueError(
                "convolution state requires [slot,channels,3+speculative] with dense inner rows"
            )
        for name, t, shape, dtype, contiguous in (
            ("state", state, (n, ch, cols), torch.bfloat16, False),
            ("snapshot", snapshot, (r, ch, cols), torch.bfloat16, True),
            ("slots", slots, (r,), torch.int64, False),
            ("recorded_slots", recorded_slots, (r,), torch.int64, True),
            ("live", live, (1,), torch.int32, True),
            ("counts", consumed_prefix, (r,), torch.int32, True),
            ("errors", errors, (r,), torch.int32, True),
        ):
            tensor(
                t,
                shape=shape,
                dtype=dtype,
                device=dev,
                name=name,
                contiguous=contiguous,
            )
        self.config = dict(
            state_stride=state.stride(0),
            slot_stride=slots.stride(0),
            requests=r,
            state_slots=n,
            channels=ch,
            columns=cols,
        )
        self.restore = restore
        if restore:
            self.read_names = ("snapshot", "slots", "recorded_slots", "live", "counts")
            self.write_names = ("state", "errors")
            self.bind(
                library,
                [snapshot, slots, recorded_slots, live, consumed_prefix],
                [state, errors],
            )
        else:
            self.read_names = ("state", "slots", "live", "counts")
            self.write_names = ("snapshot", "recorded_slots", "errors")
            self.bind(
                library,
                [state, slots, live, consumed_prefix],
                [snapshot, recorded_slots, errors],
            )
        self.fn = library.function(
            "ala_conv_rollback_zero" if restore else "ala_conv_capture", _Conv, 3
        )

    def launch(self, inputs, buffers):
        args = _Conv(**self.config)
        values = dict(zip((*self.read_names, *self.write_names), (*inputs, *buffers)))
        for n, t in values.items():
            setattr(args, n, t.data_ptr())
        args.state_stride = values["state"].stride(0)
        args.slot_stride = values["slots"].stride(0)
        self.check(self.fn(C.byref(args), self.stream()))
