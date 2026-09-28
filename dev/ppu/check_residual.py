#!/usr/bin/env python3
"""Bind residual layout/lifetime algebra to real code and separate dispatch."""
import argparse
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


def code(text):
    return re.sub(r"\s+", "", re.sub(r"//[^\n]*|/\*.*?\*/", "", text, flags=re.S))


def check(source, host, prepare, common):
    s, h, p, c = map(code, (source, host, prepare, common))
    # Ordered stages include both async completion and CTA publication. All
    # state/V producers own disjoint native C fragments (l020); cross-warp
    # consumers occur only after these boundaries; no inter-CTA dependency.
    tokens = (
        "for(intct=0;ct<p.shape.chunks();++ct)",
        "Key::stage(sm.k", "Inverse::stage(sm.inverse,ws.w+Plan::inverse_base(group),Chunk,Chunk)",
        "Value::stage(sm.value", "sm.snapshot[Snapshot::offset", "commit_wait();",
        "Snapshot::publish<Plan::Threads>", "floatkh[StateTile::ValueFragments][8]={}",
        "for(intk=0;k<Dim;k+=16)", "bf16_mma(kh[r],key,hs)",
        "floatconstdifference=float(sm.value[at])-factor[half]*kh[r][s]",
        "sm.residual[at]=BF16(row<valid?beta[half]*difference:0.0f)",
        "__syncthreads();", "floatvalue[StateTile::ValueFragments][8]={}",
        "for(intk=0;k<Chunk;k+=16)", "Value::load<true>(sm.residual",
        "Inverse::load(sm.inverse", "bf16_mma(value[r],inverse,residual)",
        "sm.value[at]=BF16(x)", "sm.scaled[at]=BF16(x*row_decay[r][StateGateRows::half(s)])",
        "__syncthreads();", "Value::publish<Plan::Threads>", "state[k][s]*=decay",
        "for(intr=0;r<Chunk;r+=16)", "Key::load<true>(sm.k", "bf16_mma(state[k],key,scaled)",
        "__syncthreads();", "if(final)", "sm.final_h[", "__syncthreads();", "state_publish_fp32<",
        "inverse_ws.snapshots=ws.w", "launch_split_inverse(p,inverse_ws,stream)",
        "gdn_wy_residual_state<<<grid,Plan::Threads,sizeof(Storage),stream>>>",
        "launch_aiu_output(p,ws,static_cast<BF16*>(output),stream)",
    )
    position = 0
    for token in tokens:
        at = s.find(token, position)
        if at < 0:
            raise AssertionError(f"residual algebra/lifetime/dispatch changed: {token}")
        position = at + len(token)
    if s.count("__syncthreads();") != 4 or s.count("commit_wait();") != 1:
        raise AssertionError("residual handoff inventory differs from conservative lifetime proof")
    if "cute::cp_async_fence();cute::cp_async_wait<0>();__syncthreads();" not in c:
        raise AssertionError("residual input handoff lost async completion")
    for token in ("Residual?Dim:Chunk", "Residual?torch::empty({0},q.options())",
                  "ifconstexpr(Residual)", "rc=gdn_wy_forward_residual(",
                  'm.def("residual",&forward<true>', 'm.def("forward",&forward<false>'):
        if token not in h:
            raise AssertionError(f"residual independent API/workspace missing: {token}")
    begin, end = p.index("intlaunch_split_inverse("), p.index("intlaunch_split_prepare(")
    if re.findall(r"(gdn_wy_split_\w+)<<<", p[begin:end]) != ["gdn_wy_split_prefix", "gdn_wy_split_solve"]:
        raise AssertionError("residual must skip W/U, not just rename the old launch chain")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    paths = ["csrc/gdn_chunk/" + f for f in ("gdn_wy_residual_ppu.cu", "gdn_wy_ops.cpp",
                                             "gdn_wy_split_prepare_ppu.cu", "gdn_wy_common.cuh")]
    texts = [(ROOT / path).read_text() for path in paths]
    check(*texts)
    if args.self_test:
        for name, index, old, new in (
                ("inverse-alias-snapshot", 0, "inverse_ws.snapshots = ws.w", "inverse_ws.snapshots = ws.snapshots"),
                ("wrong-inverse-pitch", 0, "Plan::inverse_base(group), Chunk, Chunk", "tile_offset(group), Chunk, Chunk"),
                ("drop-residual-handoff", 0, "__syncthreads();  // RESIDUAL_READY", "// RESIDUAL_READY"),
                ("drop-retirement", 0, "__syncthreads();  // RETIRE", "// RETIRE"),
                ("scaled-from-rounded-value", 0, "BF16(x * row_decay", "BF16(float(BF16(x)) * row_decay"),
                ("beta-omitted", 0, "beta[half] * difference", "difference"),
                ("inverse-underallocation", 1, "Residual ? Dim : Chunk", "Chunk"),
                ("residual-api-is-old", 1, '&forward<true>', '&forward<false>'),
                ("drop-async-wait", 3, "cute::cp_async_wait<0>();", "")):
            mutated = list(texts)
            if mutated[index].count(old) != 1:
                raise AssertionError(f"plant target not unique: {name}")
            mutated[index] = mutated[index].replace(old, new, 1)
            try:
                check(*mutated)
            except AssertionError:
                print(f"[residual source negative] {name} EXPECTED-RED/PASS")
            else:
                raise AssertionError(f"escaped residual negative: {name}")
    print("[residual source] API/rounding/paired-delivery/async+CTA-lifetimes PASS device=NOT_RUN")


if __name__ == "__main__":
    main()
