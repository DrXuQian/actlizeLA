"""Residual GDN with the retained PPU1.0 delivery as the public default.

Its BF16 association differs from materialized WY/original; FP32 carried state
and the independent numerical gate remain part of the public contract.
"""
import os
from .gdn_wy_interface import _backend as _legacy_backend, DELIVERIES
from .backends.registry import require_implementation


def _backend():
    # An explicit development extension stays explicit. A broken selected
    # extension must never silently fall back to the installed wheel.
    if "GDN_QSA_WY_EXTENSION" in os.environ:
        if "ACTLIZE_LA_SERVING_LIBRARY" in os.environ:
            raise ValueError("select only one of GDN_QSA_WY_EXTENSION and ACTLIZE_LA_SERVING_LIBRARY")
        return _legacy_backend()
    if "ACTLIZE_LA_SERVING_LIBRARY" in os.environ:
        from ._gdn_ppu10 import explicit_backend
        path = os.environ["ACTLIZE_LA_SERVING_LIBRARY"]
        return explicit_backend(path, "" if os.path.isabs(path) else os.getcwd())
    from ._gdn_ppu10 import packaged_backend
    return packaged_backend()

MATH_CONTRACT = "gated-inverse-residual-bf16-v1"
WY_MATH_CONTRACT = "materialized-wu-bf16-v1"
# Profiling may select a new algorithm; the public WY delivery inventory
# deliberately does not. None is NOT a secretly reused delivery mask.
RESIDUAL_VARIANTS = {"residual": "scalar", "residual-prefetch": "prefetch",
                     "residual-operands": "operands", "residual-v16": "v16",
                     "residual-blayout": "blayout", "residual-warps8": "warps8",
                     "residual-warps8-blayout": "warps8-blayout",
                     "residual-warps8-operands": "warps8-operands",
                     "residual-warps8-hlayout": "warps8-hlayout",
                     "residual-warps8-hvlayout": "warps8-hvlayout",
                     "residual-warps8-metadata": "warps8-metadata",
                     "residual-solve-static": "solve-static",
                     "residual-gate-cache": "gate-cache",
                     "residual-gate-cache-solve-static": "gate-cache-solve-static",
                     "residual-full-chunk": "full-chunk"}
RESIDUAL_ENTRYPOINTS = {"scalar": "residual", "prefetch": "residual_prefetch",
                        "operands": "residual_operands", "v16": "residual_v16",
                        "blayout": "residual_blayout", "warps8": "residual_warps8",
                        "warps8-blayout": "residual_warps8_blayout",
                        "warps8-operands": "residual_warps8_operands",
                        "warps8-hlayout": "residual_warps8_hlayout",
                        "warps8-hvlayout": "residual_warps8_hvlayout",
                        "warps8-metadata": "residual_warps8_metadata",
                        "solve-static": "residual_solve_static",
                        "gate-cache": "residual_gate_cache",
                        "gate-cache-solve-static": "residual_gate_cache_solve_static",
                        "full-chunk": "residual_full_chunk"}
# Same-geometry performance controls; scalar residual remains the numeric
# anchor for every delivery. Do not profile the combination against four warps.
RESIDUAL_CONTROLS = {name: "residual" for name in RESIDUAL_VARIANTS if name != "residual"}
RESIDUAL_CONTROLS["residual-warps8-blayout"] = "residual-warps8"
RESIDUAL_CONTROLS["residual-warps8-operands"] = "residual-warps8-blayout"
RESIDUAL_CONTROLS["residual-warps8-hlayout"] = "residual-warps8-blayout"
RESIDUAL_CONTROLS["residual-warps8-hvlayout"] = "residual-warps8-hlayout"
RESIDUAL_CONTROLS["residual-warps8-metadata"] = "residual-warps8-hvlayout"
RESIDUAL_CONTROLS["residual-solve-static"] = "residual-warps8-hvlayout"
RESIDUAL_CONTROLS["residual-gate-cache"] = "residual-warps8-hvlayout"
RESIDUAL_CONTROLS["residual-gate-cache-solve-static"] = "residual-gate-cache"
RESIDUAL_CONTROLS["residual-full-chunk"] = "residual-gate-cache-solve-static"
PROFILE_VARIANTS = {**DELIVERIES, **dict.fromkeys(RESIDUAL_VARIANTS)}


def math_contract(variant):
    if variant not in PROFILE_VARIANTS:
        raise ValueError(f"unknown algorithm/delivery: {variant}")
    return MATH_CONTRACT if variant in RESIDUAL_VARIANTS else WY_MATH_CONTRACT


def gdn_chunk_residual(q, k, v, g, beta, initial_state=None, output_final_state=True, *, delivery=None):
    """P[beta*(V-exp(prefix)*K@H)] with separate BF16 residual rounding.

    Same shape/type/scale contract as WY, FP32 state, no reset/truncation/CP.
    Do not use old-WY RAW-BIT equality to admit this different association.
    """
    if delivery is None:
        delivery = require_implementation("residual", "ppu10")["default_delivery"]
    if delivery not in RESIDUAL_ENTRYPOINTS:
        raise ValueError(f"unknown residual delivery: {delivery}")
    inputs = tuple(x.contiguous() for x in (q, k, v, g, beta))
    state = initial_state.contiguous() if initial_state is not None else None
    backend = _backend()
    call = getattr(backend, RESIDUAL_ENTRYPOINTS[delivery], None)
    if call is None:
        raise RuntimeError(
            f"selected backend has no {RESIDUAL_ENTRYPOINTS[delivery]}; "
            "non-default diagnostics require an explicit GDN_QSA_WY_EXTENSION; no fallback")
    output, final = call(*inputs, state, output_final_state)
    return output, final if output_final_state else None
