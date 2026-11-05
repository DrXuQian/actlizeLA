"""Fixed-address GDN transactions for Flare's value-by-key state pools.

Auxiliary ShortConv/gated norm remain caller-owned. Inputs are convolved BF16
q|k|v and RAW a/b logits, not the processed g/beta accepted by gdn_forward.
"""

from __future__ import annotations

import ctypes as C
import torch
from ._native import PreparedNative, tensor


class _Gdn(C.Structure):
    _fields_ = [
        (n, C.c_void_p)
        for n in (
            "mixed",
            "a",
            "b",
            "a_log",
            "dt_bias",
            "state",
            "starts",
            "live",
            "initial_slots",
            "final_slots",
            "output",
            "queries",
            "keys",
            "decays",
            "betas",
            "delta",
            "recorded_lengths",
            "recorded_slots",
            "errors",
            "checkpoint_slots",
            "checkpoint_offsets",
        )
    ]
    _fields_ += [
        (n, C.c_int64)
        for n in (
            "mixed_stride",
            "a_stride",
            "b_stride",
            "state_stride",
            "initial_stride",
            "final_stride",
        )
    ]
    _fields_ += [
        (n, C.c_int32)
        for n in ("requests", "tokens", "slots", "hk", "hv", "width", "mode")
    ]


class _Commit(C.Structure):
    _fields_ = [
        (n, C.c_void_p)
        for n in (
            "state",
            "delta",
            "keys",
            "decays",
            "slots",
            "recorded_slots",
            "counts",
            "live",
            "recorded_lengths",
            "errors",
        )
    ]
    _fields_ += [(n, C.c_int64) for n in ("state_stride", "slot_stride")]
    _fields_ += [
        (n, C.c_int32) for n in ("requests", "state_slots", "hk", "hv", "width")
    ]


class _Prefill(C.Structure):
    _fields_ = [("gdn", _Gdn)] + [
        (n, C.c_void_p)
        for n in (
            "q",
            "k",
            "v",
            "beta",
            "output",
            "log_g",
            "initial_kv",
            "final_kv",
            "inverse",
            "snapshots",
            "vnew",
            "gates",
        )
    ]


def gdn_workspace_specs(requests, width, key_heads=16, value_heads=48, *, mode, target):
    """Shapes/dtypes, including compact records. Allocate once, never in run()."""
    if (
        min(requests, width, key_heads, value_heads) <= 0
        or value_heads % key_heads
        or mode not in ("decode", "verify", "prefill")
        or target not in ("ppu10", "ppu15", "cuda_sm120")
        or (mode == "decode" and width != 1)
        or (mode == "verify" and width not in (1, 2, 4, 8))
    ):
        raise ValueError("invalid GDN capacity/mode/target")
    r, w, hk, hv = requests, width, key_heads, value_heads
    f, b, i = torch.float32, torch.bfloat16, torch.int32
    specs = {
        "queries": ((r, hk, w, 128), f),
        "keys": ((r, hk, w, 128), f),
        "decays": ((r, hv, w), f),
        "betas": ((r, hv, w), b),
        "errors": ((r,), i),
    }
    if mode == "verify":
        specs.update(
            delta=((r, hv, w, 128), f),
            recorded_lengths=((r,), i),
            recorded_slots=((r,), torch.int64),
        )
    if mode == "prefill" and target == "ppu15":
        groups = r * hv * ((w + 63) // 64)
        specs.update(
            q=((r, w, hk, 128), b),
            k=((r, w, hk, 128), b),
            v=((r, w, hv, 128), b),
            beta=((r, w, hv), b),
            log_g=((r, w, hv), f),
            packed_output=((r, w, hv, 128), b),
            initial_kv=((r, hv, 128, 128), f),
            final_kv=((r, hv, 128, 128), f),
            inverse=((groups, 64, 64), b),
            snapshots=((groups, 128, 128), b),
            vnew=((groups, 128, 64), b),
            gates=((groups, 64), f),
        )
    return specs


def _workspace(workspace, specs, device):
    if set(workspace) != set(specs):
        raise ValueError(f"workspace keys mismatch: expected {sorted(specs)}")
    for name, (shape, dtype) in specs.items():
        tensor(workspace[name], shape=shape, dtype=dtype, device=device, name=name)


def _state(state, hv, device):
    if (
        state.ndim != 4
        or state.shape[0] < 2
        or tuple(state.shape[1:]) != (hv, 128, 128)
        or state.stride()[1:] != (16384, 128, 1)
        or state.stride(0) < hv * 16384
        or state.dtype != torch.float32
        or state.device != device
    ):
        raise ValueError(
            "state must be FP32 [slot,Hv,V,K], contiguous within a possibly strided pool row"
        )


class PreparedGDN(PreparedNative):
    """decode=in-place; verify=read-only state + records; prefill=initial→final.

    Width is capacity, starts/live/slots are device values and may change at
    replay. Null decode/verify rows produce zero and never mutate slot zero.
    Prefill may read slot zero as a fresh zero state. No positional snapshots.
    """

    def __init__(
        self,
        library,
        *,
        mixed,
        a,
        b,
        a_log,
        dt_bias,
        state,
        starts,
        live,
        initial_slots,
        final_slots,
        output,
        workspace,
        width,
        mode,
        key_heads=16,
        value_heads=48,
        checkpoint_slots=None,
        checkpoint_offsets=None,
    ):
        hk, hv, rows, requests = (
            key_heads,
            value_heads,
            mixed.shape[0],
            starts.numel() - 1,
        )
        specs = gdn_workspace_specs(
            requests, width, hk, hv, mode=mode, target=library.target
        )
        dev = mixed.device
        for name, x, shape, dtype, contiguous in (
            ("mixed", mixed, (rows, (2 * hk + hv) * 128), torch.bfloat16, False),
            ("a", a, (rows, hv), torch.bfloat16, False),
            ("b", b, (rows, hv), torch.bfloat16, False),
            ("a_log", a_log, (hv,), torch.float32, True),
            ("dt_bias", dt_bias, (hv,), torch.float32, True),
            ("starts", starts, (requests + 1,), torch.int32, True),
            ("live", live, (1,), torch.int32, True),
            ("initial_slots", initial_slots, (requests,), torch.int64, False),
            ("final_slots", final_slots, (requests,), torch.int64, False),
            ("output", output, (rows, hv, 128), torch.bfloat16, True),
        ):
            tensor(
                x,
                shape=shape,
                dtype=dtype,
                device=dev,
                name=name,
                contiguous=contiguous,
            )
        for x in (mixed, a, b):
            if x.stride(1) != 1 or x.stride(0) < x.shape[1]:
                raise ValueError(
                    "projection views require contiguous columns and nonoverlapping rows"
                )
        _state(state, hv, dev)
        _workspace(workspace, specs, dev)
        self.config = dict(
            requests=requests,
            tokens=rows,
            slots=state.shape[0],
            hk=hk,
            hv=hv,
            width=width,
            mode={"decode": 0, "verify": 1, "prefill": 2}[mode],
            mixed_stride=mixed.stride(0),
            a_stride=a.stride(0),
            b_stride=b.stride(0),
            state_stride=state.stride(0),
            initial_stride=initial_slots.stride(0),
            final_stride=final_slots.stride(0),
        )
        self.mode, self.names = mode, tuple(specs)
        self.workspace, self.output = workspace, output
        inputs = [mixed, a, b, a_log, dt_bias, starts, live, initial_slots, final_slots]
        self.input_names = [
            "mixed",
            "a",
            "b",
            "a_log",
            "dt_bias",
            "starts",
            "live",
            "initial_slots",
            "final_slots",
        ]
        if (checkpoint_slots is None) != (checkpoint_offsets is None):
            raise ValueError("checkpoint fields must be supplied together")
        if checkpoint_slots is not None:
            tensor(
                checkpoint_slots,
                shape=(requests,),
                dtype=torch.int64,
                device=dev,
                name="checkpoint_slots",
            )
            tensor(
                checkpoint_offsets,
                shape=(requests,),
                dtype=torch.int32,
                device=dev,
                name="checkpoint_offsets",
            )
            inputs.extend((checkpoint_slots, checkpoint_offsets))
            self.input_names.extend(("checkpoint_slots", "checkpoint_offsets"))
        buffers = [output, *[workspace[n] for n in self.names]]
        if mode == "verify":
            inputs.append(state)
        else:
            buffers.append(state)
        self.bind(library, inputs, buffers)
        self.fn = library.function(
            "ala_prefill_run" if mode == "prefill" else "ala_gdn_run",
            _Prefill if mode == "prefill" else _Gdn,
            2 if mode == "prefill" else 0,
        )
        if mode == "prefill":
            with torch.cuda.device(self.device):
                self.check(library.lib.ala_prefill_configure())

    def arguments(self, inputs, buffers):
        p = _Gdn(**self.config)
        for name, t in zip(self.input_names, inputs):
            setattr(p, name, t.data_ptr())
        state = inputs[-1] if self.mode == "verify" else buffers[-1]
        p.state = state.data_ptr()
        p.state_stride = state.stride(0)
        p.mixed_stride, p.a_stride, p.b_stride = (t.stride(0) for t in inputs[:3])
        p.initial_stride, p.final_stride = inputs[7].stride(0), inputs[8].stride(0)
        p.output = buffers[0].data_ptr()
        work = dict(zip(self.names, buffers[1 : 1 + len(self.names)]))
        for name in (
            "queries",
            "keys",
            "decays",
            "betas",
            "delta",
            "recorded_lengths",
            "recorded_slots",
            "errors",
        ):
            setattr(p, name, work[name].data_ptr() if name in work else None)
        if self.mode == "prefill":
            args = _Prefill(gdn=p)
            for name, _ in _Prefill._fields_[1:]:
                key = "packed_output" if name == "output" else name
                setattr(args, name, work[key].data_ptr() if key in work else None)
        else:
            args = p
        return args

    def launch(self, inputs, buffers):
        args = self.arguments(inputs, buffers)
        self.check(self.fn(C.byref(args), self.stream()))


class PreparedGDNCommit(PreparedNative):
    """Commit *target-consumed* prefix, NOT accepted draft count.

    Flare currently passes its clipped commit.emitted_count because verification
    input zero is the old pending token. Sampling/count policy stays in Flare.
    Records must belong to this round, be read once, and share request ordering.
    """

    def __init__(
        self, library, *, state, slots, consumed_prefix, live, records, errors
    ):
        r, hv, w, d = records["delta"].shape
        hk = records["keys"].shape[1]
        if d != 128 or hv % hk or w not in (1, 2, 4, 8):
            raise ValueError("invalid correction record geometry")
        dev = state.device
        _state(state, hv, dev)
        for name, x, shape, dtype, contiguous in (
            ("slots", slots, (r,), torch.int64, False),
            ("counts", consumed_prefix, (r,), torch.int32, True),
            ("live", live, (1,), torch.int32, True),
            ("errors", errors, (r,), torch.int32, True),
            ("delta", records["delta"], (r, hv, w, 128), torch.float32, True),
            ("keys", records["keys"], (r, hk, w, 128), torch.float32, True),
            ("decays", records["decays"], (r, hv, w), torch.float32, True),
            ("recorded_lengths", records["recorded_lengths"], (r,), torch.int32, True),
            ("recorded_slots", records["recorded_slots"], (r,), torch.int64, True),
        ):
            tensor(
                x,
                shape=shape,
                dtype=dtype,
                device=dev,
                name=name,
                contiguous=contiguous,
            )
        self.config = dict(
            requests=r,
            state_slots=state.shape[0],
            hk=hk,
            hv=hv,
            width=w,
            state_stride=state.stride(0),
            slot_stride=slots.stride(0),
        )
        self.input_names = (
            "delta",
            "keys",
            "decays",
            "slots",
            "recorded_slots",
            "counts",
            "live",
            "recorded_lengths",
        )
        self.bind(
            library,
            [
                records["delta"],
                records["keys"],
                records["decays"],
                slots,
                records["recorded_slots"],
                consumed_prefix,
                live,
                records["recorded_lengths"],
            ],
            [state, errors],
        )
        self.fn = library.function("ala_gdn_commit", _Commit, 1)

    def arguments(self, inputs, buffers):
        p = _Commit(
            **self.config, state=buffers[0].data_ptr(), errors=buffers[1].data_ptr()
        )
        for name, t in zip(self.input_names, inputs):
            setattr(p, name, t.data_ptr())
        p.state_stride = buffers[0].stride(0)
        p.slot_stride = inputs[3].stride(0)
        return p

    def launch(self, inputs, buffers):
        p = self.arguments(inputs, buffers)
        self.check(self.fn(C.byref(p), self.stream()))
