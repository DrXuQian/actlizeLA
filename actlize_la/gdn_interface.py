"""Common forward contract, without imposing an intermediate/kernel ABI.

SM90 automatically selects a prebuilt configuration from tensor/device metadata.
PPU1.0 selects retained full-chunk residual; PPU1.5 selects native residual.
Selection never examines gate values or compiles a kernel during a call.
"""
from importlib import import_module
import os

from .backends.registry import require_implementation
from .device import validate_profile_options


def _resolve_forward(q, initial_state=None, *, algorithm="auto", backend=None,
                     delivery=None, configuration=None, mode="device", sm_count=None):
    """Single policy authority for execution and preparation-time inspection."""
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
            return dict(backend=backend, algorithm="auto", module="sm90_auto", entry="forward",
                        delivery=None, configuration=None)
        algorithm = "residual" if backend in ("ppu10", "ppu15") else "original"
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
    if algorithm == "fused_sm90" and delivery is not None:
        raise ValueError("fused_sm90 owns its pipeline; legacy delivery selectors do not apply")
    if algorithm not in ("original", "fused_sm90") and delivery is None:
        delivery = implementation.get("default_delivery", "scalar")
    return dict(implementation, backend=backend, algorithm=algorithm,
                delivery=delivery, configuration=configuration)


def gdn_forward(q, k, v, g, beta, initial_state=None, output_final_state=True,
                *, algorithm="auto", backend=None, delivery=None, configuration=None,
                mode="device", sm_count=None):
    """Complete forward; input/state numerical contracts belong to its route.

    PPU generation must remain explicit. No hidden Q/K normalization, gate
    activation, query rescaling or state conversion takes place here.
    Use get_gdn_route_info with the same inputs/options before timing/capture
    to inspect the selected PPU1.0 implementation and loaded native images.
    mode='perfmodel' skips device-profile queries, not computation.
    """
    route = _resolve_forward(q, initial_state, algorithm=algorithm, backend=backend,
                             delivery=delivery, configuration=configuration,
                             mode=mode, sm_count=sm_count)
    if route["algorithm"] == "auto":
        from .sm90_auto import forward
        return forward(q, k, v, g, beta, initial_state, output_final_state,
                       mode=mode, sm_count=sm_count)
    module = import_module(f".{route['module']}", __package__)
    call = getattr(module, route["entry"])
    if route["algorithm"] == "fused_sm90":
        return call(q,k,v,g,beta,initial_state=initial_state,
                    output_final_state=output_final_state,backend=route["backend"],
                    **({"configuration": configuration} if configuration is not None else {}))
    if route["algorithm"] == "original":
        return call(q, k, v, g, beta, output_final_state=output_final_state)
    return call(q, k, v, g, beta, initial_state=initial_state,
                output_final_state=output_final_state,
                delivery=route["delivery"])
