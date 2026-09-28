#!/usr/bin/env python3
"""Bind exact gate reuse to the real state body and native AIU instruction path."""
import argparse
from collections import Counter
from pathlib import Path
import re
from check_state_pipeline import code, block
from check_residual_blayout import native_body

ROOT = Path(__file__).resolve().parents[2]
CONTROL = "gdn_wy_residual_warps8_hvlayout_state"
SUBJECT = "gdn_wy_residual_gate_cache_state"


def replace_once(text, old, new):
    old, new = code(old), code(new)
    if text.count(old) != 1:
        raise AssertionError(f"missing/ambiguous coefficient seam: {old}")
    return text.replace(old, new, 1)


def check_source(control, subject, header, binding):
    old, new = code(control), code(subject)
    state = block(new, SUBJECT+"(").replace(SUBJECT, CONTROL)
    changes = (
        ("auto& shared = *reinterpret_cast<gate_cache::Storage*>(storage);"
         "auto& sm = shared.matrices; auto& coefficients = shared.coefficients;",
         "auto& sm = *reinterpret_cast<Storage*>(storage);"),
        ("coefficients.publish(tid, ws.gates[group * Chunk + tid], ws.gates[group * Chunk + valid - 1]);",
         "sm.gates[tid] = ws.gates[group * Chunk + tid];"),
        ("float row_decay[StateTile::ValueFragments][StateGateRows::Count];",
         "float const last = sm.gates[valid - 1]; float row_decay[StateTile::ValueFragments][StateGateRows::Count];"),
        ("factor[half] = coefficients.prefix[row];", "factor[half] = expf(sm.gates[row]);"),
        ("row_decay[r][half] = coefficients.relative[row];", "row_decay[r][half] = expf(last - sm.gates[row]);"),
        ("float const decay = coefficients.prefix[valid - 1];", "float const decay = expf(last);"),
    )
    for before, after in changes:
        state = replace_once(state, before, after)
    if state != block(old, CONTROL+"("):
        raise AssertionError("coefficient reuse changed arithmetic/layout/rounding/lifetime")
    h = code(header)
    if block(h, "CUTE_HOST_DEVICE void publish(") != code(
        "CUTE_HOST_DEVICE void publish(unsigned row, float log_prefix, float log_last) {"
        "prefix[row] = ::expf(log_prefix); relative[row] = ::expf(log_last - log_prefix); }"):
        raise AssertionError("coefficient formula changed: no reciprocal/fastmath")
    for token in ("sizeof(Storage)==46080", "offsetof(Storage,coefficients)==45568"):
        if token not in h: raise AssertionError("matrix offsets or shared extent changed")
    for token in (
        "return launch_hvlayout_output(p, ws, static_cast<BF16*>(output), stream);",
        "rc = launch_split_inverse(p, inverse_ws, stream);",
        "gdn_wy_residual_gate_cache_state<<<grid, Plan::Threads, sizeof(gate_cache::Storage), stream>>>",
        "hggcFuncSetAttribute(gdn_wy_residual_gate_cache_state, hggcFuncAttributeMaxDynamicSharedMemorySize, sizeof(gate_cache::Storage))",
    ):
        if code(token) not in new: raise AssertionError("gate cache launch/reused stages mismatch")
    b = code(binding)
    for token in ('Variant<=12&&(!Variant||Residual)',
                  'Variant==11?gdn_wy_forward_residual_solve_static:gdn_wy_forward_residual_gate_cache;',
                  'm.def("residual_gate_cache",&forward<true,12>,'):
        if b.count(token)!=1: raise AssertionError("exact gate-cache binding missing")


def check_native(isa):
    old, old_loops = native_body(isa, CONTROL+"E", 20)
    new, new_loops = native_body(isa, SUBJECT+"E", 20)
    if len(old_loops) != len(new_loops): raise AssertionError("recurrence denominator changed")
    # Matrix data path and synchronization may not be traded for coefficients.
    fixed = ("v.mma.", "vmem.aiu.", "tsm.ld.swzl.", "vmem.st.",
             "v.cnvt.bf16", "s.blksyn", "vmem.fence", "vmem.acp.")
    for a, b in [(old,new), *zip(old_loops,new_loops)]:
        ca, cb = (Counter(x.split()[0] for x in seq) for seq in (a,b))
        if {k:v for k,v in ca.items() if k.startswith(fixed)} != {
                k:v for k,v in cb.items() if k.startswith(fixed)}:
            raise AssertionError("gate cache changed matrix work/traffic or synchronization")
        if cb["v.exp2.f32"] != 2 or ca["v.exp2.f32"] != 5:
            raise AssertionError("expected two producer exp sites replacing five consumer sites")
        if any("ivreg" in x or "shuffle" in x or "tsm.ld.ncom" in x for x in b):
            raise AssertionError("gate cache introduced compatibility repair")
    return dict(control_sites=len(old), candidate_sites=len(new),
                exp_sites="5->2", logical_exp_lane_evaluations_per_chunk="1280->128",
                matrix_path="AIU.swzl+ld.swzl unchanged", device="NOT_RUN")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--isa",type=Path)
    a=p.parse_args()
    src=ROOT/"csrc/gdn_chunk"
    texts=[(src/"gdn_wy_residual_warps8_hvlayout_ppu.cu").read_text(),
           (src/"gdn_wy_residual_gate_cache_ppu.cu").read_text(),
           (ROOT/"include/gdn_qsa/ppu/wy_gate_coefficients.cuh").read_text(),
           (src/"gdn_wy_ops.cpp").read_text()]
    check_source(*texts)
    if a.self_test:
        for idx,old,new in (
            (1,"if (tid < Chunk)","if (tid < Chunk - 1)"),
            (1,"commit_wait();","/* lost publication */"),
            (1,"coefficients.prefix[valid - 1]","coefficients.prefix[0]"),
            (2,"::expf(log_last - log_prefix)","::expf(log_last) / ::expf(log_prefix)"),
            (1,"return launch_hvlayout_output(","return launch_hlayout_output("),
            (3,'&forward<true, 12>','&forward<true, 11>'),
        ):
            changed=list(texts)
            if changed[idx].count(old)!=1: raise AssertionError("negative target is not unique")
            changed[idx]=changed[idx].replace(old,new,1)
            try: check_source(*changed)
            except AssertionError: print("[gate cache source negative] EXPECTED-RED/PASS",old)
            else: raise AssertionError("source negative escaped")
    if a.isa:
        isa=a.isa.read_text()
        print("[gate cache native]",check_native(isa))
        if a.self_test:
            from check_wy_binary import plant_in_kernel
            for op in ("v.mma.f32.bf16.m16n16k16","v.exp2.f32","tsm.ld.swzl.b32x4.s0.t1.trans0","s.blksyn.defer"):
                try: check_native(plant_in_kernel(isa,SUBJECT,op,"MISSING"))
                except AssertionError: print("[gate cache native negative] EXPECTED-RED/PASS",op)
                else: raise AssertionError("native negative escaped")
    print("[gate cache] source=PASS native="+("PASS" if a.isa else "NOT_RUN")+" device=NOT_RUN")


if __name__=="__main__": main()
