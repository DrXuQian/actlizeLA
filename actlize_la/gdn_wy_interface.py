"""Explicit experimental WY backend. Does not change original auto routing."""
from .backends.loading import load_wy, clear_backend_cache


def _backend():
    return load_wy()


_backend.cache_clear = clear_backend_cache


PACKED_DELIVERIES = ("prepare", "state", "output", "all")
TILED_DELIVERIES = ("tiled-prepare", "tiled-state", "tiled-output",
                    "tiled-state-output", "tiled-all")
STATE_DELIVERIES = ("tiled-state-output", "tiled-all", "tiled-state-output-address",
                    "tiled-state-output-gates", "tiled-state-output-both")
STAGE_DELIVERIES = ("tiled-state-output", "tiled-state-output-both", "stage-address-prepare",
                    "stage-address-output", "stage-address-both")
PREPARE_ROWS_DELIVERIES = ("tiled-state-output", "tiled-state-output-both", "stage-address-prepare",
                           "prepare-rows-shared", "prepare-rows-warp")
AIU_DELIVERIES = ("prepare-rows-shared", "aiu-state", "aiu-output", "aiu-state-output")
SPLIT_PREPARE_DELIVERIES = ("aiu-state-output", "split-prepare")
STATE_PIPELINE_DELIVERIES = ("split-prepare", "state-pipeline")
DELIVERIES = {"scalar": 0, "prepare": 1, "state": 2, "output": 4, "all": 7,
              "tiled-prepare": 8, "tiled-state": 16, "tiled-output": 32,
              "tiled-state-output": 48, "tiled-all": 56,
              "tiled-state-output-address": 112, "tiled-state-output-gates": 176,
              "tiled-state-output-both": 240, "stage-address-prepare": 496,
              "stage-address-output": 752, "stage-address-both": 1008,
              "prepare-rows-shared": 1520, "prepare-rows-warp": 2544,
              "aiu-state": 5616, "aiu-output": 9712, "aiu-state-output": 13808,
              "split-prepare": 30192, "state-pipeline": 62960}


def gdn_chunk_wy(q, k, v, g, beta, initial_state=None, output_final_state=True, *, delivery="scalar"):
    """C64 WY forward, BF16 Q/K/V/beta, BF16/FP32 log gate, FP32 state.

    Native GVA, D=128, scale=1/sqrt(128). No QK normalization, gate transform,
    reset approximation, decay-based host synchronization or fallback.
    Delivery variants are opt-in; scalar preserves the admitted control.
    """
    tensors = tuple(x.contiguous() for x in (q, k, v, g, beta))
    state = initial_state.contiguous() if initial_state is not None else None
    if delivery not in DELIVERIES:
        raise ValueError(f"unknown WY delivery {delivery!r}; choose {tuple(DELIVERIES)}")
    # Keep the old 7-argument call for archived scalar bindings.
    args = (*tensors, state, output_final_state)
    if delivery != "scalar":
        args += (DELIVERIES[delivery],)
    output, final = _backend().forward(*args)
    return output, final if output_final_state else None
