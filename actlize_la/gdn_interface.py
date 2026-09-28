"""Common forward contract, without imposing an intermediate/kernel ABI.

Existing public APIs are retained. This entry does not promote an experimental
algorithm or choose one from decay values: original remains the default, WY
and residual are explicit. Each implementation owns allocation and all launches.
"""
from importlib import import_module
import os

from .backends.registry import require_implementation


def gdn_forward(q, k, v, g, beta, initial_state=None, output_final_state=True,
                *, algorithm="original", backend=None, delivery=None, configuration=None):
    """Execute one complete implementation with its existing numerical gate.

    backend names identify compiled execution targets, not a measured device.
    None preserves existing selection: original uses GDN_QSA_PPU_EXTENSION
    when supplied, otherwise CUDA SM80; WY/residual require their opt-in PPU
    extension. No new architecture is guessed from CUDA-compatible APIs.
    Gate dtypes/normalization and state semantics are those of the selected
    implementation; this wrapper performs no hidden conversions.
    """
    legacy_ppu = "GDN_QSA_PPU_EXTENSION" in os.environ
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
