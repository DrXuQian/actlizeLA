"""actlizeLA: forward linear attention with explicit native backends."""

from importlib import import_module

# A GDN-only installation must not need the three unrelated NVIDIA extensions.
# Import only contracts here; load the selected implementation lazily.
_MODULES = {
    "load_sm90": "sm90",
    "gdn_forward": "gdn_interface",
    "backend_inventory": "backends",
    "gdn_chunk": "gdn_chunk_interface",
    "gdn_chunk_reference": "gdn_chunk_interface",
    "gdn_chunk_twolevel": "gdn_chunk_interface",
    "gdn_chunk_wy": "gdn_wy_interface",
    "gdn_chunk_residual": "gdn_residual_interface",
    "out_proj_gemm": "output_gate_interface",
    "out_proj_gemm_cutlass": "output_gate_interface",
    "rmsnorm_gated": "output_gate_interface",
    "qsa_expand": "qsa_core_interface",
    "qsa_pass2_tc": "qsa_core_interface",
    "qsa_pass2_tc_reuse": "qsa_core_interface",
    "qsa_sparse_core_attention": "qsa_core_interface",
    "qsa_indexer": "qsa_indexer_interface",
    "qsa_indexer_reference": "qsa_indexer_interface",
    "qsa_indexer_topk_only": "qsa_indexer_interface",
}


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(name)
    value = getattr(import_module(f".{_MODULES[name]}", __name__), name)
    globals()[name] = value
    return value

__version__ = "0.1.0"
__all__ = [
    "load_sm90",
    # GDN
    "gdn_forward",
    "backend_inventory",
    "gdn_chunk",
    "gdn_chunk_twolevel",
    "gdn_chunk_wy",
    "gdn_chunk_residual",
    "gdn_chunk_reference",
    # QSA indexer
    "qsa_indexer",
    "qsa_indexer_topk_only",
    "qsa_indexer_reference",
    # QSA sparse-core attention
    "qsa_sparse_core_attention",
    "qsa_expand",
    "qsa_pass2_tc",
    "qsa_pass2_tc_reuse",
    # output gate
    "rmsnorm_gated",
    "out_proj_gemm",
    "out_proj_gemm_cutlass",
]
