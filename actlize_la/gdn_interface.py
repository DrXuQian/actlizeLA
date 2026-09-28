"""Common forward contract, without imposing an intermediate/kernel ABI.

SM90 automatically selects a prebuilt configuration from tensor/device metadata.
Other families retain their original default; WY/residual remain explicit.
Selection never examines gate values or compiles a kernel during a call.
"""
from importlib import import_module
import os

from .backends.registry import require_implementation
from .device import validate_profile_options


def gdn_forward(q, k, v, g, beta, initial_state=None, output_final_state=True,
                *, algorithm="auto", backend=None, delivery=None, configuration=None,
                mode="device", sm_count=None):
    """Execute one complete implementation with its existing numerical gate.

    backend names identify compiled execution targets, not a measured device.
    Auto identifies NVIDIA SM90 from tensor-device metadata and uses the
    installed shape-dispatched bundle. PPU generation must remain explicit.
    Explicit original uses GDN_QSA_PPU_EXTENSION when supplied, otherwise
    CUDA SM80; WY/residual require their opt-in PPU extension.
    Gate dtypes/normalization and state semantics are those of the selected
    implementation; this wrapper performs no hidden conversions.

    mode='perfmodel' requires an explicit backend and positive sm_count.
    It skips our device-profile query, not the forward computation. Configured
    metadata cannot claim a measured H800 winner or alter native launch geometry.
    """
    validate_profile_options(mode, backend, sm_count)
    legacy_ppu = "GDN_QSA_PPU_EXTENSION" in os.environ
    if algorithm == "auto":
        if configuration is not None:
            raise ValueError("auto selects its configuration; use explicit fused_sm90 for diagnostics")
        if backend is None:
            from .sm90_policy import tensor_backend
            backend = tensor_backend(q)
        if backend == "cuda_sm90":
            if delivery is not None:
                raise ValueError("SM90 auto does not accept legacy delivery selectors")
            from .sm90_auto import forward
            return forward(q, k, v, g, beta, initial_state, output_final_state,
                           mode=mode, sm_count=sm_count)
        algorithm = "original"
    if configuration is not None and algorithm != "fused_sm90":
        raise ValueError("named SM90 configurations do not apply to another backend")
    if algorithm == "fused_sm90" and backend is None:
        raise ValueError("fused_sm90 requires an explicit cuda_sm90 or ppu17 backend")
    if backend is None:
        backend = "ppu10" if algorithm != "original" or legacy_ppu else "cuda_sm80"
    implementation = require_implementation(algorithm, backend)
    if initial_state is not None and not implementation["initial_state"]:
        raise ValueError(f"{algorithm} does not accept initial_state; it must not be dropped")
    if algorithm == "original":
        if delivery is not None:
            raise ValueError("original has no WY/residual delivery selector")
        expected = "ppu10" if legacy_ppu else "cuda_sm80"
        if backend != expected:
            raise RuntimeError(
                f"requested {backend}, but GDN_QSA_PPU_EXTENSION selects {expected}; "
                "set/unset the extension explicitly, no fallback")
    module = import_module(f".{implementation['module']}", __package__)
    call = getattr(module, implementation["entry"])
    if algorithm == "fused_sm90":
        if delivery is not None:
            raise ValueError("fused_sm90 owns its pipeline; legacy delivery selectors do not apply")
        return call(q,k,v,g,beta,initial_state=initial_state,
                    output_final_state=output_final_state,backend=backend,
                    **({"configuration": configuration} if configuration is not None else {}))
    if algorithm == "original":
        return call(q, k, v, g, beta, output_final_state=output_final_state)
    return call(q, k, v, g, beta, initial_state=initial_state,
                output_final_state=output_final_state,
                delivery="scalar" if delivery is None else delivery)
