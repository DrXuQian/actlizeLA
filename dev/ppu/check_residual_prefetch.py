#!/usr/bin/env python3
"""Scheduling-only proof: exact old arithmetic plus async lifetime/native CFG."""
import argparse
from pathlib import Path
import re
from check_state_pipeline import code, native_records, reachable

ROOT = Path(__file__).resolve().parents[2]
PROLOGUE = "Inverse::stage(sm.inverse,ws.w+Plan::inverse_base(p.shape.group(b,h,0)),Chunk,Chunk);cute::cp_async_fence();"
FUTURE = "intconstnext=Plan::next(ct,p.shape.chunks());if(next>=0){Inverse::stage(sm.inverse,ws.w+Plan::inverse_base(p.shape.group(b,h,next)),Chunk,Chunk);cute::cp_async_fence();}"
KEY = "Key::stage(sm.k,p.k+p.shape.input(b,first,qh,p.shape.q_heads),p.shape.q_heads*Dim,valid);"
CURRENT = "Inverse::stage(sm.inverse,ws.w+Plan::inverse_base(group),Chunk,Chunk);"


def check_source(control, subject):
    before, after = code(control), code(subject)
    if after.count(PROLOGUE) != 1 or after.count(FUTURE) != 1:
        raise AssertionError("prefetch prologue/next guard/pitch/commit changed")
    loop = after.index("for(intct=0;ct<p.shape.chunks();++ct)")
    future = after.index(FUTURE)
    boundary = "sm.scaled[at]=BF16(x*row_decay[r][StateGateRows::half(s)])"
    if boundary not in after:
        raise AssertionError("prefetch changed the FP32-to-BF16 scaledV boundary")
    ready = after.index(boundary)
    publication = after.index("Value::publish<Plan::Threads>")
    if not (after.index(PROLOGUE) < loop < ready < future < publication):
        raise AssertionError("prefetch moved outside its proved lifetime/overlap window")
    if after[ready:future].count("__syncthreads();") != 1:
        raise AssertionError("next P writer lacks current P reader retirement")
    after = after.replace(PROLOGUE, "").replace(FUTURE, "")
    after = after.replace(KEY, KEY + CURRENT)
    after = after.replace("gdn_wy_residual_prefetch_state", "gdn_wy_residual_state")
    after = after.replace("gdn_wy_forward_residual_prefetch", "gdn_wy_forward_residual")
    after = after.replace("wy::residual_prefetch", "wy::residual")
    after = after.replace("usingnamespaceresidual_prefetch;", "usingnamespaceresidual;")
    after = after.replace("wy_residual_prefetch.cuh", "wy_residual.cuh")
    if after != before:
        raise AssertionError("change exceeds registered inverse-prefetch moves: math/layout/launch changed")


def native_schedule(section):
    records, edges = native_records(section)
    mma = {pc for pc,op in records.items() if op.startswith("v.mma.f32.bf16.m16n16k16")}
    if len(mma) != 40:
        raise AssertionError("residual-prefetch arithmetic inventory changed")
    reverse = {pc:[] for pc in records}
    for pc,targets in edges.items():
        for target in targets: reverse[target].append(pc)
    start = min(mma)
    loop = reachable(start,edges) & reachable(start,reverse)
    if not mma <= loop:
        raise AssertionError("all40 MMAs must share the recurrent loop")
    copies = {pc for pc,op in records.items() if op.startswith("vmem.aiu.ld.tsm.l0")}
    waits = {pc for pc,op in records.items() if op.startswith("s.wait") and "commit_group(0)" in op}
    if len(copies-loop)!=1 or len(copies&loop)!=4 or len(waits)!=1 or not waits<=loop:
        raise AssertionError("P-prologue/KV+next-P/async-wait inventory differs")
    if sum(records[pc].startswith("s.blksyn.defer") for pc in loop)!=4:
        raise AssertionError("residual handoff count changed")

    def until_wait(copy):
        pending=[(target,0) for target in edges[copy]]
        seen,outcomes=set(),set()
        reached=False
        while pending:
            item=pending.pop()
            if item in seen: continue
            seen.add(item)
            pc,n=item
            if pc not in loop:
                # The syntactic final-chunk exit is not a next-P issue in
                # the source/Plan proof. Still require full update on that edge.
                outcomes.add(n); continue
            if pc in waits:
                reached=True;outcomes.add(n);continue
            n+=pc in mma
            if n>40: raise AssertionError("prefetch not drained before dependent work")
            pending.extend((target,n) for target in edges[pc])
        if not reached or not outcomes: raise AssertionError("async writer has no draining wait")
        return outcomes

    overlap={pc:until_wait(pc) for pc in copies&loop}
    current=[pc for pc,values in overlap.items() if values=={0}]
    future=[pc for pc,values in overlap.items() if values=={16}]
    if len(current)!=3 or len(future)!=1:
        raise AssertionError(f"next inverse not overlapped with16 UPDATE MMAs: {overlap}")
    return dict(records=records,edges=edges,loop=loop,waits=sorted(waits),
                future=future,current=current,overlap={hex(pc):sorted(n) for pc,n in overlap.items()})


def serialized_plant(isa):
    # Same opcode/operand multiset, but move the draining wait immediately
    # after next P's issue. This must fail even though counts are unchanged.
    from check_wy_binary import swap_native_instructions
    sections=isa.split("Disassembly of section ")
    indices=[i for i,s in enumerate(sections) if s.startswith(".text.kernel.") and
             "gdn_wy_residual_prefetch_state" in s.splitlines()[0]]
    if len(indices)!=1: raise AssertionError("prefetch plant lacks exact new body")
    i=indices[0]; schedule=native_schedule(sections[i])
    following=min(pc for pc in schedule["records"] if pc>schedule["future"][0])
    sections[i]=swap_native_instructions(sections[i],schedule["waits"][0],following)
    return "Disassembly of section ".join(sections)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    p.add_argument("--isa",type=Path)
    args=p.parse_args()
    directory=ROOT/"csrc/gdn_chunk"
    c=(directory/"gdn_wy_residual_ppu.cu").read_text()
    s=(directory/"gdn_wy_residual_prefetch_ppu.cu").read_text()
    check_source(c,s)
    if args.self_test:
        for name,old,new in (
            ("lost-next-commit","      cute::cp_async_fence();",""),
            ("early-overwrite","__syncthreads();  // VALUES_READY","// VALUES_READY"),
            ("stale-next-index","p.shape.group(b, h, next)","p.shape.group(b, h, ct)"),
            ("last-chunk-OOB","if (next >= 0)","if (true)"),
            ("lost-async-wait","commit_wait();","__syncthreads();"),
            ("lost-retire","__syncthreads();  // RETIRE","// RETIRE"),
            ("rounding-change","BF16(x * row_decay","BF16(float(BF16(x)) * row_decay"),
            ("reversed-K","r = 0; r < Chunk; r += 16","r = 48; r >= 0; r -= 16")):
            if s.count(old)!=1: raise AssertionError(f"nonunique source plant:{name}")
            try: check_source(c,s.replace(old,new,1))
            except AssertionError: print(f"[residual prefetch source negative] {name} EXPECTED-RED/PASS")
            else: raise AssertionError(f"escaped source negative:{name}")
    if args.isa:
        isa=args.isa.read_text()
        def section(text):
            return next(s for s in text.split("Disassembly of section ") if
                        s.startswith(".text.kernel.") and "gdn_wy_residual_prefetch_state" in s.splitlines()[0])
        print("[residual prefetch native]",native_schedule(section(isa))["overlap"])
        if args.self_test:
            try: native_schedule(section(serialized_plant(isa)))
            except AssertionError: print("[residual prefetch native negative] serialized-same-opcodes EXPECTED-RED/PASS")
            else: raise AssertionError("serialized prefetch escaped")
    print("[residual prefetch] arithmetic=CONTROL-IDENTICAL inverse-lifetime/source-bound PASS device=NOT_RUN")


if __name__=="__main__": main()
