"""Additive single-use GDN records and explicit upstream error dependencies."""

import ctypes as C
import torch
from .auxiliary import HeaderV1, TensorV1, tensor_descriptor
from ._native import tensor
from .gdn import PreparedGDN, PreparedGDNCommit, _Gdn, _Commit, _Prefill


class TransactionV2(C.Structure):
    _fields_ = (
        [("header", HeaderV1)]
        + [
            (n, TensorV1)
            for n in ("epochs", "ready_epochs", "consumed_epochs", "upstream_errors")
        ]
        + [("requests", C.c_int32)]
    )


class _Transaction:
    def _bind_transaction(
        self, library, epochs, ready_epochs, consumed_epochs, upstream_errors, *, verify
    ):
        r = self.config["requests"]
        for name, t in (
            ("epochs", epochs),
            ("ready_epochs", ready_epochs),
            ("consumed_epochs", consumed_epochs),
        ):
            tensor(t, shape=(r,), dtype=torch.int64, device=self.device, name=name)
        if upstream_errors is not None:
            tensor(
                upstream_errors,
                shape=(r,),
                dtype=torch.int32,
                device=self.device,
                name="upstream_errors",
            )
        self.base_inputs, self.base_buffers = len(self.inputs), len(self.buffers)
        self.verifying = verify
        read = [epochs, consumed_epochs if verify else ready_epochs]
        if upstream_errors is not None:
            read.append(upstream_errors)
        self.has_dependency = upstream_errors is not None
        self.bind(
            library,
            [*self.inputs, *read],
            [*self.buffers, ready_epochs if verify else consumed_epochs],
        )
        size = library.lib.ala_gdn_transaction_v2_size
        size.restype = C.c_int
        if size() != C.sizeof(TransactionV2):
            raise RuntimeError("GDN transaction ABI mismatch")
        self.fn = getattr(
            library.lib, "ala_gdn_verify_v2" if verify else "ala_gdn_commit_v2"
        )
        self.fn.argtypes = [
            C.POINTER(_Gdn if verify else _Commit),
            C.POINTER(TransactionV2),
            C.c_void_p,
        ]
        self.fn.restype = C.c_int

    def launch(self, inputs, buffers):
        base = self.arguments(inputs[: self.base_inputs], buffers[: self.base_buffers])
        extra = inputs[self.base_inputs :]
        guard = TransactionV2(
            header=HeaderV1(2, C.sizeof(TransactionV2)),
            requests=self.config["requests"],
            epochs=tensor_descriptor(extra[0]),
            ready_epochs=tensor_descriptor(buffers[-1] if self.verifying else extra[1]),
            consumed_epochs=tensor_descriptor(
                extra[1] if self.verifying else buffers[-1]
            ),
            upstream_errors=tensor_descriptor(
                extra[2] if self.has_dependency else None
            ),
        )
        self.check(self.fn(C.byref(base), C.byref(guard), self.stream()))


class PreparedGDNVerifyV2(_Transaction, PreparedGDN):
    """Epochs must advance after EACH commit (including consumed_prefix=0).

    ready/consumed start at zero; no unconsumed record window may be overwritten.
    upstream_errors normally points at ShortConv errors, on the same stream.
    """

    def __init__(
        self,
        library,
        *,
        epochs,
        ready_epochs,
        consumed_epochs,
        upstream_errors=None,
        **kwargs,
    ):
        if kwargs.pop("mode", "verify") != "verify":
            raise ValueError("V2 transaction requires verify mode")
        super().__init__(library, mode="verify", **kwargs)
        self._bind_transaction(
            library, epochs, ready_epochs, consumed_epochs, upstream_errors, verify=True
        )


class PreparedGDNCommitV2(_Transaction, PreparedGDNCommit):
    """Reject wrong-round and twice-consumed records without touching state."""

    def __init__(
        self,
        library,
        *,
        epochs,
        ready_epochs,
        consumed_epochs,
        upstream_errors=None,
        **kwargs,
    ):
        super().__init__(library, **kwargs)
        self._bind_transaction(
            library,
            epochs,
            ready_epochs,
            consumed_epochs,
            upstream_errors,
            verify=False,
        )


class PreparedGDNCheckedV2(PreparedGDN):
    """Decode/prefill with a separate read-only per-request upstream error vector."""

    def __init__(self, library, *, upstream_errors, **kwargs):
        if kwargs.get("mode") == "verify":
            raise ValueError("verification requires the epoch-bearing V2 transaction")
        super().__init__(library, **kwargs)
        tensor(
            upstream_errors,
            shape=(self.config["requests"],),
            dtype=torch.int32,
            device=self.device,
            name="upstream_errors",
        )
        self.bind(library, [*self.inputs, upstream_errors], self.buffers)
        size = library.lib.ala_gdn_transaction_v2_size
        size.restype = C.c_int
        if size() != C.sizeof(TransactionV2):
            raise RuntimeError("GDN transaction ABI mismatch")
        self.fn = getattr(
            library.lib,
            "ala_prefill_checked_v2"
            if self.mode == "prefill"
            else "ala_gdn_checked_v2",
        )
        self.fn.argtypes = [
            C.POINTER(_Prefill if self.mode == "prefill" else _Gdn),
            C.POINTER(TransactionV2),
            C.c_void_p,
        ]
        self.fn.restype = C.c_int

    def launch(self, inputs, buffers):
        base = self.arguments(inputs[:-1], buffers)
        guard = TransactionV2(
            header=HeaderV1(2, C.sizeof(TransactionV2)),
            requests=self.config["requests"],
            upstream_errors=tensor_descriptor(inputs[-1]),
        )
        self.check(self.fn(C.byref(base), C.byref(guard), self.stream()))
