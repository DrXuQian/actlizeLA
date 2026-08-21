#!/usr/bin/env python3
"""One-call FLA/Triton GDN subject matching actlizeLA's PPU perf fixture.

Run once outside ACU to populate the Triton cache, then run the same command
under ACU.  ``--scope=full`` emits cumsum plus the four forward stages;
``--scope=post-cumsum`` starts at the same public boundary as actlizeLA ABI v4
and emits exactly the four directly comparable stages.
"""

from __future__ import annotations

import argparse
import hashlib
import math

import torch

from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from fla.ops.common.chunk_o import chunk_fwd_o
from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd
from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--sequences", type=int, default=1)
    p.add_argument("--length", type=int, default=2048)
    p.add_argument("--qk-heads", type=int, default=16)
    p.add_argument("--v-heads", type=int, default=32)
    p.add_argument("--scope", choices=("full", "post-cumsum"), default="post-cumsum")
    return p.parse_args()


def sync(device: str) -> None:
    kind = device.split(":", 1)[0]
    backend = getattr(torch, kind, None)
    if backend is None or not hasattr(backend, "synchronize"):
        raise RuntimeError(f"torch backend {kind!r} has no synchronize()")
    backend.synchronize()


def payload_hash(x: torch.Tensor) -> str:
    raw = x.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def device_identity(device: torch.device) -> str:
    try:
        props = torch.cuda.get_device_properties(device)
        return "name={!r} ordinal={} multiprocessors={} total_memory={}".format(
            props.name,
            device.index if device.index is not None else torch.cuda.current_device(),
            getattr(props, "multi_processor_count", "UNKNOWN"),
            props.total_memory,
        )
    except Exception as exc:  # pragma: no cover - backend-specific diagnostics
        return f"identity=UNAVAILABLE({type(exc).__name__}:{exc})"


def main() -> int:
    a = parse_args()
    if (a.sequences, a.length, a.qk_heads, a.v_heads) != (1, 2048, 16, 32):
        raise SystemExit("the registered comparison shape is B1,T2048,H16,HV32")
    device = torch.device(a.device)

    q_count = a.sequences * a.length * a.qk_heads * 128
    v_count = a.sequences * a.length * a.v_heads * 128
    state_count = a.sequences * a.v_heads * 128 * 128
    # Build the fixture on CPU. The ACU process then has H2D copies plus exactly
    # four or five FLA kernels, rather than unrelated fixture-generation kernels.
    q_index = torch.arange(q_count, dtype=torch.int64)
    v_index = torch.arange(v_count, dtype=torch.int64)
    s_index = torch.arange(state_count, dtype=torch.int64)
    q = (((q_index.remainder(7).to(torch.float32) - 3.0) / 64.0)
         .to(torch.bfloat16)
         .reshape(a.sequences, a.length, a.qk_heads, 128)
         .to(device))
    k = (((q_index.remainder(5).to(torch.float32) - 2.0) / 32.0)
         .to(torch.bfloat16)
         .reshape(a.sequences, a.length, a.qk_heads, 128)
         .to(device))
    v = (((v_index.remainder(9).to(torch.float32) - 4.0) / 64.0)
         .to(torch.bfloat16)
         .reshape(a.sequences, a.length, a.v_heads, 128)
         .to(device))
    initial = (((s_index.remainder(5).to(torch.float32) - 2.0) / 1024.0)
               .reshape(a.sequences, a.v_heads, 128, 128)
               .to(device))
    beta = torch.full(
        (a.sequences, a.length, a.v_heads), 0.5,
        dtype=torch.float32,
    ).to(device)

    # FLA multiplies raw g by RCP_LN2 while accumulating. This increment
    # therefore produces the exact same gamma_log2_cumsum=-local/64 consumed
    # directly by the actlizeLA ABI. The post-cumsum subject uploads that
    # cumulative tensor directly, just as the C ABI benchmark does.
    raw_increment = -(1.0 / 64.0) * math.log(2.0)
    g = torch.full(
        (a.sequences, a.length, a.v_heads), raw_increment,
        dtype=torch.float32,
    )
    g[:, ::64, :] = 0.0
    g = g.to(device)
    gamma_input = -(
        torch.arange(a.length).remainder(64).to(torch.float32) / 64.0
    ).view(1, a.length, 1).expand(a.sequences, -1, a.v_heads).contiguous().to(device)

    with torch.inference_mode():
        if a.scope == "full":
            gamma, output, _inverse, final_state, _initial, _g_input = chunk_gated_delta_rule_fwd(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                scale=0.5,
                initial_state=initial,
                output_final_state=True,
                state_v_first=False,
                chunk_size=64,
            )
        else:
            gamma = gamma_input
            w, u, _inverse = chunk_gated_delta_rule_fwd_intra(
                k=k,
                v=v,
                g=gamma,
                beta=beta,
                chunk_size=64,
            )
            h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
                k=k,
                w=w,
                u=u,
                g=gamma,
                initial_state=initial,
                output_final_state=True,
                state_v_first=False,
                chunk_size=64,
            )
            output = chunk_fwd_o(
                q=q,
                k=k,
                v=v_new,
                h=h,
                g=gamma,
                scale=0.5,
                state_v_first=False,
                chunk_size=64,
            )
        sync(a.device)

    gamma_cpu = gamma.detach().cpu()
    output_cpu = output.detach().cpu()
    final_state_cpu = final_state.detach().cpu()
    expected_gamma = -(
        torch.arange(a.length).remainder(64).to(torch.float32)
        / 64.0
    ).view(1, a.length, 1)
    gamma_bad = int(((gamma_cpu - expected_gamma).abs() > 1.0e-6).sum().item())
    output_nonfinite = int((~torch.isfinite(output_cpu)).sum().item())
    state_nonfinite = int((~torch.isfinite(final_state_cpu)).sum().item())
    print(
        "[FLA GDN ACU subject] shape=B1,T2048,H16,HV32,K128,V128,C64 "
        "device={} {} scope={} calls=1 kernels={} cumsum={} post_cumsum=4 "
        "gamma_bad={} output_nonfinite={} state_nonfinite={} "
        "output_sha256={} state_sha256={} {}".format(
            a.device,
            device_identity(device),
            a.scope,
            5 if a.scope == "full" else 4,
            1 if a.scope == "full" else 0,
            gamma_bad,
            output_nonfinite,
            state_nonfinite,
            payload_hash(output_cpu),
            payload_hash(final_state_cpu),
            "PASS" if gamma_bad == output_nonfinite == state_nonfinite == 0 else "FAIL",
        )
    )
    return 0 if gamma_bad == output_nonfinite == state_nonfinite == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
