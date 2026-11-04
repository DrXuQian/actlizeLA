"""Preparation-only bindings to Flare metadata objects; no scheduler policy.

This module does not import Flare/b12x or patch its module globals. The caller
explicitly chooses these operators while preparing its existing graph family.
"""

from .gdn import PreparedGDN, PreparedGDNCommit
from .qsa import PreparedQSA


def bind_gdn(
    library,
    *,
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    state,
    metadata,
    output,
    workspace,
    mode,
    width,
    key_heads=16,
    value_heads=48,
    checkpoint_export=False,
):
    """Bind after Flare's existing ShortConv, before its gated RMSNorm.

    mode='verify' uses record capacity as width; starts determines the current
    1/2/4/8-token interval. mode='decode' consumes one pending token in-place.
    Prefill uses explicit initial/final slots (fresh input selects null slot).
    Intermediate prefix-checkpoint exports are not part of this ABI.
    """
    if checkpoint_export:
        raise NotImplementedError(
            "intermediate prefill checkpoint exports require a separate binding"
        )
    if mode == "prefill":
        initial = metadata.initial_state_indices
        final = metadata.final_state_indices
    else:
        initial = metadata.state_indices[:, 0]
        final = initial
    return PreparedGDN(
        library,
        mixed=mixed_qkv,
        a=a,
        b=b,
        a_log=A_log,
        dt_bias=dt_bias,
        state=state,
        starts=metadata.query_start_loc,
        live=metadata.num_seqs,
        initial_slots=initial,
        final_slots=final,
        output=output.view(mixed_qkv.shape[0], value_heads, 128),
        workspace=workspace,
        width=width,
        mode=mode,
        key_heads=key_heads,
        value_heads=value_heads,
        checkpoint_slots=metadata.checkpoint_state_indices,
        checkpoint_offsets=metadata.checkpoint_offsets,
    )


def bind_gdn_commit(library, *, state, metadata, records, consumed_prefix, errors):
    """Pass commit.emitted_count in the current Flare graph, not accepted_drafts.

    See docs/FLARE_SERVING.md for the pending-token and EOS/budget contract.
    Each layer retains its own compact records until this operation finishes.
    """
    return PreparedGDNCommit(
        library,
        state=state,
        slots=metadata.state_indices[:, 0],
        consumed_prefix=consumed_prefix,
        live=metadata.num_seqs,
        records=records,
        errors=errors,
    )


def bind_qsa(
    library,
    *,
    query,
    key,
    value,
    index_query,
    raw_index_key,
    index_q_norm_weight,
    index_k_norm_weight,
    rotary,
    state,
    metadata,
    live_rows,
    workspace,
    norm_epsilon=1e-6,
    compression_ratio=4,
):
    """Replaces the QSA cache write+selector+attention portion only.

    Main q/key already underwent Flare's norm/partial RoPE; its output gate
    and projection remain outside. Do not ALSO call paged_kv_write for this
    binding. BF16 KV must be selected in the caller's cache allocator.
    """
    inputs = dict(
        query=query,
        new_key=key,
        new_value=value,
        index_query=index_query,
        raw_key=raw_index_key,
        query_weight=index_q_norm_weight,
        key_weight=index_k_norm_weight,
        cosine=rotary.cos,
        sine=rotary.sin,
        request_ids=metadata.request_ids,
        positions=metadata.query_positions,
        rope_positions=metadata.rope_positions,
        raw_slots=metadata.raw_state_slot_ids,
        slot_mapping=metadata.slot_mapping,
        starts=metadata.query_start_loc,
        live=live_rows,
        lengths=metadata.sequence_lengths,
        accepted_previous=metadata.num_accepted_tokens,
        prefilling=metadata.is_prefilling,
        main_table=metadata.main_block_table,
        compressed_table=metadata.compressed_block_table,
    )
    caches = dict(
        key_cache=state.main_k_cache,
        value_cache=state.main_v_cache,
        compressed_cache=state.compressed_k_cache,
        raw_ring=state.raw_k_ring,
        raw_tags=state.raw_logical_positions,
        raw_rope=state.raw_rope_positions,
        anchors=state.raw_interval_start_positions,
    )
    return PreparedQSA(
        library,
        inputs=inputs,
        caches=caches,
        workspace=workspace,
        norm_epsilon=norm_epsilon,
        compression_ratio=compression_ratio,
    )
