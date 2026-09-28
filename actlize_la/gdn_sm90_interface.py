"""Explicit complete SM90 algorithm, not an SM80 primitive replacement.

BF16 Q/K/V/beta; natural-log per-token g is BF16/FP32. No normalization,
conversion, gate materialization, synchronization or routing is hidden here.
Numerical device admission is required before promoting this candidate.
"""
from .backends.loading import _load, _explicit_path

MATH_CONTRACT = "cula-scalar-gdn-fused-bf16-v1"


def gdn_chunk_sm90(q,k,v,g,beta,initial_state=None,output_final_state=True,*,backend,source_check=False,configuration="control"):
    if backend not in ("cuda_sm90","ppu17"):
        raise ValueError(f"fused_sm90 cannot execute on {backend}")
    module = _load("_gdn_fused_sm90", _explicit_path("GDN_QSA_SM90_EXTENSION"))
    if source_check and backend != "ppu17":
        raise ValueError("source_check is only the explicit PPU-fork CUDA simulation-input build")
    expected = backend + ("-source-check" if source_check else "")
    if module.target != expected or module.math_contract != MATH_CONTRACT:
        raise RuntimeError(f"SM90 binary identity mismatch: requested {backend}, got {module.target}")
    if configuration not in ("control", "value64", "value64-local-inverse", "value128-paired"):
        raise ValueError(f"unknown SM90 configuration: {configuration}")
    # Old experimental modules also lacked this field. They must not be
    # mislabeled control merely because their binary predates the registry.
    actual = getattr(module, "configuration", None)
    if actual != configuration:
        raise RuntimeError(f"SM90 configuration mismatch: requested {configuration}, got {actual}")
    out, state = module.forward(q,k,v,g,beta,initial_state,output_final_state)
    return out, state if output_final_state else None
