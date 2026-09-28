#!/usr/bin/env python3
"""Paired R/scaled layout only, against the SAME eight-warp geometry."""
import argparse
from collections import Counter
from pathlib import Path
from check_state_pipeline import code
from check_residual_blayout import native_body

ROOT = Path(__file__).resolve().parents[2]


def check_source(control, candidate, header):
    s = code(candidate)
    for old, new in (
        ("residual_warps8_blayout", "residual_warps8"),
        ("unsignedconstb_store_base=BIntermediate::producer_base(warp,lane);", ""),
        ("sm.residual[BIntermediate::producer_offset(b_store_base,r,s)]", "sm.residual[at]"),
        ("sm.scaled[BIntermediate::producer_offset(b_store_base,r,s)]", "sm.scaled[at]"),
        ("BIntermediate::load(sm.residual,", "Value::load<true>(sm.residual,"),
        ("BIntermediate::load(sm.scaled,", "Value::load<true>(sm.scaled,"),
    ):
        if old not in s:
            raise AssertionError(f"eight-warp B-layout seam missing: {old}")
        s = s.replace(old, new)
    if s != code(control):
        raise AssertionError("eight-warp B-layout changed beyond paired R/scaled writers/readers")
    h = code(header)
    for required in (
        "usingresidual_warps8::StateTile;", "usingresidual_warps8::Plan;",
        "usingresidual_warps8::Storage;",
        "structBIntermediate:residual_blayout::BIntermediate",
        "cube(StateTile::value_row(warp,0),StateTile::column(warp))",
        "returnmicrocube*256+(lane%4)*16+lane/4;",
        "StateTile::ValueFragments==1&&Plan::Threads==256",
    ):
        if required not in h:
            raise AssertionError(f"eight-warp B-layout owner contract missing: {required}")
    for forbidden in ("__shfl", "__syncthreads", "::stage(", "::publish("):
        if forbidden in h:
            raise AssertionError(f"eight-warp B-layout adds work: {forbidden}")


def check_native(isa):
    control, old_loops = native_body(isa, "gdn_wy_residual_warps8_stateE", 20)
    candidate, new_loops = native_body(isa, "gdn_wy_residual_warps8_blayout_stateE", 20)
    a, b = (Counter(line.split()[0] for line in seq) for seq in (control, candidate))
    for prefix in ("vmem.", "tsm.st.", "v.mma.", "v.exp2.", "v.mul.f32", "v.fma.f32",
                   "v.add.f32", "v.cnvt.bf16", "s.blksyn", "v.shuffle"):
        if {k:v for k,v in a.items() if k.startswith(prefix)} != {
                k:v for k,v in b.items() if k.startswith(prefix)}:
            raise AssertionError(f"eight-warp B-layout math/transfer/sync changed: {prefix}")
    for counts, expected in ((a, (12,24)), (b, (20,16))):
        got = tuple(counts[f"tsm.ld.swzl.b32x4.s0.t1.trans{t}"] for t in (0,1))
        if got != expected:
            raise AssertionError(f"eight-warp B-layout native orientations {got} != {expected}")
    for seq in (control, candidate):
        if sum("commit_group(0)" in line for line in seq) != 1:
            raise AssertionError("missing native async completion")
        if any("ivreg" in line or "shuffle" in line or "tsm.ld.ncom" in line for line in seq):
            raise AssertionError("unexpected indirect/remap/non-paired matrix delivery")
    if len(old_loops) != len(new_loops):
        raise AssertionError("eight-warp B-layout recurrence CFG changed")
    loops = []
    for old, new in zip(old_loops, new_loops):
        # Fixed counts at recurring backedges prevent moving math out of the
        # loop while keeping a superficially equal whole-body inventory.
        ca, cb = (Counter(x.split()[0] for x in seq) for seq in (old,new))
        for prefix in ("v.mma.", "v.exp2.", "tsm.st.", "vmem.", "s.blksyn"):
            if {k:v for k,v in ca.items() if k.startswith(prefix)} != {
                    k:v for k,v in cb.items() if k.startswith(prefix)}:
                raise AssertionError(f"recurrence work moved: {prefix}")
        loops.append(dict(control_sites=len(old), candidate_sites=len(new)))
    return dict(control_sites=len(control), candidate_sites=len(candidate),
                recurrence_backedges=loops, matrix_trans0=20, matrix_trans1=16,
                dynamic_instructions_and_BC="NOT_MEASURED")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--isa", type=Path)
    args = p.parse_args()
    d = ROOT / "csrc/gdn_chunk"
    texts = [(d/"gdn_wy_residual_warps8_ppu.cu").read_text(),
             (d/"gdn_wy_residual_warps8_blayout_ppu.cu").read_text(),
             (ROOT/"include/gdn_qsa/ppu/wy_residual_warps8_blayout.cuh").read_text()]
    check_source(*texts)
    if args.self_test:
        for label, index, old, new in (
            ("stale-scaled-writer",1,"sm.scaled[BIntermediate::producer_offset(b_store_base, r, s)]","sm.scaled[at]"),
            ("stale-residual-reader",1,"BIntermediate::load(sm.residual,","Value::load<true>(sm.residual,"),
            ("rounding",1,"BF16(x * row_decay","BF16(float(BF16(x)) * row_decay"),
            ("publication",1,"Value::publish<Plan::Threads>","Value::publish<128>"),
            ("retirement",1,"__syncthreads();  // RETIRE","/* missing */  // RETIRE"),
            ("owner",2,"cube(StateTile::value_row(warp, 0), StateTile::column(warp))","((warp/2)*4+warp%2)"),
        ):
            mutated = list(texts)
            if mutated[index].count(old) != 1:
                raise AssertionError(f"nonunique source plant: {label}")
            mutated[index] = mutated[index].replace(old,new,1)
            try: check_source(*mutated)
            except AssertionError: print(f"[warps8 B-layout source negative] {label} EXPECTED-RED/PASS")
            else: raise AssertionError(f"escaped source plant: {label}")
    if args.isa:
        isa = args.isa.read_text()
        print("[warps8 B-layout native]", check_native(isa))
        if args.self_test:
            from check_wy_binary import plant_in_kernel
            for label, old, new in (
                ("native-trans", "tsm.ld.swzl.b32x4.s0.t1.trans0", "tsm.ld.swzl.b32x4.s0.t1.trans1"),
                ("missing-mma", "v.mma.f32.bf16.m16n16k16", "REMOVED_MMA"),
                ("missing-wait", "commit_group(0)", "MISSING_WAIT"),
                ("missing-barrier", "s.blksyn.defer", "MISSING_BARRIER"),
            ):
                planted = plant_in_kernel(isa,"gdn_wy_residual_warps8_blayout_state",old,new)
                try: check_native(planted)
                except AssertionError: print(f"[warps8 B-layout native negative] {label} EXPECTED-RED/PASS")
                else: raise AssertionError(f"escaped native plant: {label}")
    print(f"[warps8 B-layout] source=PASS native={'PASS' if args.isa else 'NOT_RUN'} device=NOT_RUN")


if __name__ == "__main__":
    main()
