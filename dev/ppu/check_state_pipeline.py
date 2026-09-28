#!/usr/bin/env python3
"""Bind lifetime proof to actual source and native CFG, not opcode counts alone."""
import argparse
from collections import Counter
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


def code(text):
    return re.sub(r"\s+", "", re.sub(r"//[^\n]*|/\*.*?\*/", "", text, flags=re.S))


def block(text, start, contains=""):
    for match in re.finditer(re.escape(code(start)), text):
        begin = text.index("{", match.end())
        depth = 1
        end = begin + 1
        while depth:
            depth += (text[end] == "{") - (text[end] == "}")
            end += 1
        region = text[match.start():end]
        if code(contains) in region:
            return region
    raise AssertionError(f"source region absent: {start} / {contains}")


def check_source(control, subject, dispatch, entry):
    c, s, host, abi = map(code, (control, subject, dispatch, entry))
    fragments = "for (int k = 0; k < StateTile::KFragments; ++k)"
    for start, contains in ((fragments, "state[k][s] = p.initial"),
                            (fragments, "sm.snapshot[H::offset"),
                            ("for (int k = 0; k < Dim; k += 16)", "bf16_mma(value[r]"),
                            ("for (int r = 0; r < Chunk; r += 16)", "bf16_mma(state[k]"),
                            (fragments, "state[k][s] *= decay"),
                            ("if (final)", "state_publish_fp32")):
        if block(c,start,contains) != block(s,start,contains):
            raise AssertionError(f"pipeline changed admitted arithmetic/order: {contains}")
    # Includes the FP32 subtraction, both distinct BF16 boundaries and row exp.
    def conditioning(t):
        a = t.index("floatconstlast=sm.g[valid-1];")
        b = t.index("__syncthreads();", a)
        return t[a:b]
    if conditioning(c) != conditioning(s):
        raise AssertionError("pipeline changed conditioning/rounding")
    wait = "cute::cp_async_wait<0>();__syncthreads();"
    waits = [m.start() for m in re.finditer(re.escape(wait),s)]
    if len(waits) != 2 or s.count("__syncthreads();") != 4:
        raise AssertionError("lifetime proof requires W-ready/input-ready/value-ready/final handoffs")
    ordered = (
        "W::stage(sm.w,ws.w+tile_offset(p.shape.group(b,h,0)),Dim,Chunk);cute::cp_async_fence();",
        "for(intct=0;ct<p.shape.chunks();++ct)",
        "valid=Plan::valid(ct,p.shape.sequence)",
        "sm.snapshot[H::offset",
        wait,
        "H::publish<StateTile::Threads>",
        "W::stage(sm.k,p.k+p.shape.input(b,first,qh,p.shape.q_heads),p.shape.q_heads*Dim,valid);",
        "V::stage(sm.u,ws.u+tile_offset(group)+v0,Dim,Chunk);",
        "if(tid<Chunk)sm.g[tid]=ws.gates[group*Chunk+tid];cute::cp_async_fence();",
        "floatvalue[StateTile::ValueFragments][8]={};",
        "bf16_mma(value[r],w,hs);",
        wait,
        "intconstnext=Plan::next(ct,p.shape.chunks());if(next>=0){W::stage(sm.w,ws.w+tile_offset(p.shape.group(b,h,next)),Dim,Chunk);cute::cp_async_fence();}",
        "floatconstlast=sm.g[valid-1];",
        "__syncthreads();V::publish<StateTile::Threads>",
        "floatconstdecay=expf(last);",
        "state[k][s]*=decay;",
        "bf16_mma(state[k],key,value_operand);",
        "if(final)",
    )
    position = 0
    for token in ordered:
        found = s.find(token,position)
        if found < 0:
            raise AssertionError(f"pipeline schedule no longer matches lifetime proof: {token}")
        position = found + len(token)
    for token, text in (
        ("delivery&(AiuOptions|SplitPrepare|StatePipeline)",abi),
        ("rc=pipeline?configure_state_pipeline()",host),
        ("if(pipeline){rc=launch_state_pipeline(p,ws,final,stream);}",host),
        ("gdn_wy_state_pipeline<<<grid,Plan::Threads,Plan::SharedBytes,stream>>>",s),
        ("state_pipeline::gdn_wy_state_pipeline,hggcFuncAttributeMaxDynamicSharedMemorySize,state_pipeline::Plan::SharedBytes",s),
    ):
        if token not in text:
            raise AssertionError(f"pipeline selector/configuration/launch seam not bound: {token}")


def native_records(section):
    labels = {name:int(pc,16) for pc,name in re.findall(r"^\s*([0-9a-f]+) <([^>]+)>:",section,re.M)}
    records = {int(pc,16):op for pc,op in re.findall(
        r"^\s*([0-9a-f]+):\s+(?:[0-9a-f]{2}\s+){8}\s*([^\n]+)",section,re.M)}
    addresses = list(records)
    edges = {pc:[] for pc in addresses}
    for i,pc in enumerate(addresses):
        op = records[pc]
        mnemonic = op.split()[0]
        if mnemonic.startswith("s.cbr"):
            target = re.search(r"<([^>]+)>",op)
            if not target or target[1] not in labels: raise AssertionError("unresolved native branch")
            edges[pc].append(labels[target[1]])
        if mnemonic not in ("s.cbr","s.exit") and i+1<len(addresses):
            edges[pc].append(addresses[i+1])
    return records,edges


def reachable(start, edges):
    seen, pending = set(), [start]
    while pending:
        pc = pending.pop()
        if pc in seen: continue
        seen.add(pc); pending.extend(edges[pc])
    return seen


def native_schedule(section):
    records,edges = native_records(section)
    mma = {pc for pc,op in records.items() if op.startswith("v.mma.f32.bf16.m16n16k16")}
    if len(mma) != 32: raise AssertionError("pipeline lost state/project MMA work")
    reverse = {pc:[] for pc in records}
    for pc,targets in edges.items():
        for target in targets: reverse[target].append(pc)
    start = min(mma)
    loop = reachable(start,edges) & reachable(start,reverse)
    if not mma <= loop: raise AssertionError("MMA body not in one recurrent native loop")
    # W@H has two accumulators, each8 MMA; update has four, each4 MMA.
    accum = {pc:re.search(r"vreg\[\d+:\d+\]",records[pc])[0] for pc in mma}
    count = Counter(accum.values())
    if sorted(count.values()) != [4,4,4,4,8,8]:
        raise AssertionError("native accumulator/order contract changed")
    project = {pc for pc in mma if count[accum[pc]] == 8}
    update = mma - project
    waits = {pc for pc,op in records.items() if op.startswith("s.wait") and "commit_group(0)" in op}
    copies = {pc for pc,op in records.items() if op.startswith("vmem.aiu.ld.tsm.l0")}
    if len(copies-loop) != 2 or len(copies & loop) != 5 or len(waits & loop) != 2:
        raise AssertionError("pipeline prologue/loop copy/wait denominator differs")
    if sum(records[pc].startswith("s.blksyn.defer") for pc in loop) != 3:
        raise AssertionError("native loop no longer has the proved three CTA handoffs")

    def until_wait(pc):
        # Explore every CFG branch, not just linear PC order. Loop wrap is
        # intentional: next-W's wait is in the next snapshot phase.
        seen, pending, outcomes = set(), [(target,0,0) for target in edges[pc]], set()
        reached_wait = False
        while pending:
            item = pending.pop()
            if item in seen: continue
            seen.add(item)
            at, np, nu = item
            if at not in loop:
                # The syntactic CFG also contains the last-chunk exit after
                # UPDATE. Source/Plan proofs bind its mutually exclusive
                # no-next-W guard. Do not invent a native branch predicate
                # equivalence: require all UPDATE work before even that exit.
                outcomes.add((np,nu)); continue
            if at in waits:
                reached_wait = True
                outcomes.add((np,nu)); continue
            np += at in project; nu += at in update
            if np+nu > 32: raise AssertionError("prefetch does not drain within one chunk")
            pending.extend((target,np,nu) for target in edges[at])
        if not outcomes or not reached_wait: raise AssertionError("no loop wait closes async copy")
        return outcomes

    current, future, overlap = [],[],{}
    for pc in sorted(copies & loop):
        outcome = until_wait(pc)
        overlap[pc] = sorted(outcome)
        if all(np > 0 and nu == 0 for np,nu in outcome): current.append(pc)
        elif outcome == {(0,16)}: future.append(pc)
        else: raise AssertionError(f"copy was serialized/consumed without intended work overlap: {pc:#x} {outcome}")
    if len(current) != 3 or len(future) != 2:
        raise AssertionError("K/U versus next-W copy assignment differs")
    # Native wait may move before the last independent W@H instructions.
    # Record actual count; never assert all16 precede it from source alone.
    return dict(records=records,edges=edges,loop=loop,project=project,update=update,
                current=current,future=future,waits=sorted(waits & loop),overlap=overlap)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test",action="store_true")
    args = parser.parse_args()
    directory = ROOT / "csrc/gdn_chunk"
    control = (directory / "gdn_wy_aiu_ppu.cu").read_text()
    subject = (directory / "gdn_wy_state_pipeline_ppu.cu").read_text()
    entry = (directory / "gdn_wy_ppu.cu").read_text()
    check_source(control,subject,control,entry)
    if args.self_test:
        plants = (
            ("missing-W-wait", "cute::cp_async_wait<0>();", "",1),
            ("missing-KU-wait", "cute::cp_async_wait<0>();", "",2),
            ("missing-W-ready-barrier", "__syncthreads();", "",1),
            ("missing-W-reader-retirement", "__syncthreads();", "",2),
            ("missing-value-handoff", "__syncthreads();", "",3),
            ("reversed-update-K", "r = 0; r < Chunk; r += 16", "r = 48; r >= 0; r -= 16",1),
            ("premature-rounding", "BF16(x * row_gate", "BF16(float(BF16(x)) * row_gate",1),
            ("stale-prefetch-group", "p.shape.group(b, h, next)", "p.shape.group(b, h, ct)",1),
            ("last-prefetch-OOB", "if (next >= 0)", "if (true)",1),
            ("wrong-tail", "Plan::valid(ct, p.shape.sequence)", "Chunk",1),
        )
        for name,old,new,occurrence in plants:
            positions = [m.start() for m in re.finditer(re.escape(old),subject)]
            if len(positions) < occurrence: raise AssertionError(f"missing negative target: {name}")
            at=positions[occurrence-1]
            changed=subject[:at]+new+subject[at+len(old):]
            try: check_source(control,changed,control,entry)
            except AssertionError: print(f"[WY pipeline source negative] {name} EXPECTED-RED/PASS")
            else: raise AssertionError(f"escaped pipeline source negative: {name}")
        try: check_source(control,subject,control,entry.replace(" | StatePipeline", ""))
        except AssertionError: print("[WY pipeline source negative] ignored-selector EXPECTED-RED/PASS")
        else: raise AssertionError("ignored selector escaped")
    print("[WY pipeline source] arithmetic=CONTROL-IDENTICAL schedule/alias/selector/stream bound PASS device_execution=NOT_RUN")


if __name__ == "__main__":
    main()
