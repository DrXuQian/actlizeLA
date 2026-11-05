"""Preparation-only bindings to Flare metadata objects; no scheduler policy.

This module does not import Flare/b12x or patch its module globals. The caller
explicitly chooses these operators while preparing its existing graph family.
"""

from .gdn import PreparedGDN, PreparedGDNCommit
from .gdn_transaction import (
    PreparedGDNCheckedV2,
    PreparedGDNVerifyV2,
    PreparedGDNCommitV2,
)
from .auxiliary import (
    ServingRows,
    PreparedShortConv,
    PreparedConvHistoryV1,
    PreparedPointwise,
    PreparedPLEHash,
)
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
    upstream_errors=None,
    transaction=None,
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
    if transaction is not None and mode != "verify":
        raise ValueError("record transactions apply only to verify/commit")
    if mode == "verify" and upstream_errors is not None and transaction is None:
        raise ValueError("checked verification requires caller-owned record epochs")
    cls, extra = PreparedGDN, {}
    if transaction is not None:
        cls = PreparedGDNVerifyV2
        extra = {**transaction, "upstream_errors": upstream_errors}
    elif upstream_errors is not None:
        cls, extra = PreparedGDNCheckedV2, {"upstream_errors": upstream_errors}
    return cls(
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
        **extra,
    )


def bind_gdn_commit(
    library,
    *,
    state,
    metadata,
    records,
    consumed_prefix,
    errors,
    transaction=None,
    upstream_errors=None,
):
    """Pass commit.emitted_count in the current Flare graph, not accepted_drafts.

    See docs/FLARE_SERVING.md for the pending-token and EOS/budget contract.
    Each layer retains its own compact records until this operation finishes.
    """
    if upstream_errors is not None and transaction is None:
        raise ValueError("checked commit requires caller-owned record epochs")
    cls = PreparedGDNCommitV2 if transaction is not None else PreparedGDNCommit
    extra = (
        {}
        if transaction is None
        else {**transaction, "upstream_errors": upstream_errors}
    )
    return cls(
        library,
        state=state,
        slots=metadata.state_indices[:, 0],
        consumed_prefix=consumed_prefix,
        live=metadata.num_seqs,
        records=records,
        errors=errors,
        **extra,
    )


def auxiliary_rows(metadata, tokens, *, upstream_errors=None):
    """DeltaMetadata/PLEMetadata use raw ShortConv slots: -1 null, 0 VALID.

    Never pass recurrence state_indices here. Those use a different null slot.
    """
    return ServingRows(
        tokens,
        num_tokens=metadata.num_tokens,
        starts=metadata.query_start_loc,
        slots=metadata.convolution_slots,
        num_seqs=metadata.num_seqs,
        upstream_errors=upstream_errors,
    )


def bind_shortconv(
    library,
    *,
    x,
    weight,
    state,
    metadata,
    output,
    errors,
    dilation,
    mode,
    upstream_errors=None,
):
    """Replace b12x short_conv. Decode is width-one verification history.

    GDN dilation=1; PLE dilation=3. Positive acceptance is deferred, NOT shifted
    by commit. The caller retains previous accepted count after zero rollback.
    """
    return PreparedShortConv(
        library,
        x=x,
        weight=weight,
        state=state,
        fresh=metadata.fresh,
        previous_accepted=metadata.accepted_tokens,
        output=output,
        errors=errors,
        dilation=dilation,
        mode="verify" if mode == "decode" else mode,
        rows=auxiliary_rows(metadata, x.shape[0], upstream_errors=upstream_errors),
    )


def bind_shortconv_history(
    library,
    *,
    state,
    metadata,
    tokens,
    snapshot,
    stamps,
    consumed_prefix,
    errors,
    dilation,
    restore=False,
    upstream_errors=None,
):
    """Replace the old slot0-null history binding when using raw conv slots."""
    return PreparedConvHistoryV1(
        library,
        state=state,
        snapshot=snapshot,
        stamps=stamps,
        consumed_prefix=consumed_prefix,
        errors=errors,
        dilation=dilation,
        restore=restore,
        rows=auxiliary_rows(metadata, tokens, upstream_errors=upstream_errors),
    )


def bind_gated_norm(
    library,
    *,
    hidden,
    gate,
    weight,
    output,
    errors,
    value_heads,
    epsilon,
    num_valid_tokens=None,
    rows=None,
    round_normalized=True,
    norm_before_gate=True,
    silu_gate=True,
    trace=None,
):
    """No projection/gate activation beforehand; preserves gate column stride."""
    return PreparedPointwise(
        library,
        operation="gated_norm",
        x=hidden,
        weight=weight,
        gate=gate,
        output=output,
        errors=errors,
        groups=value_heads,
        epsilon=epsilon,
        round_normalized=round_normalized,
        norm_before_gate=norm_before_gate,
        activation="silu" if silu_gate else "sigmoid",
        trace=trace,
        rows=rows
        if rows is not None
        else ServingRows(hidden.shape[0], num_tokens=num_valid_tokens),
    )


def bind_ple_hash(
    library,
    *,
    metadata,
    geometry,
    multipliers,
    primes,
    offsets,
    hashes,
    request_ids,
    errors,
    eos,
    upstream_errors=None,
):
    """Only hashes; CPU table/gather/readiness/history commit stay in Flare."""
    return PreparedPLEHash(
        library,
        tokens=metadata.token_ids,
        history=metadata.committed_history,
        geometry=geometry,
        multipliers=multipliers,
        primes=primes,
        offsets=offsets,
        hashes=hashes,
        request_ids=request_ids,
        errors=errors,
        eos=eos,
        rows=auxiliary_rows(
            metadata, metadata.token_ids.shape[0], upstream_errors=upstream_errors
        ),
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
