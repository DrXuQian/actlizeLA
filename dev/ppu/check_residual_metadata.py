#!/usr/bin/env python3
"""Bind register-only lookahead, unchanged HV work and actual native overlap."""
import argparse
from collections import Counter
from pathlib import Path
import re
from check_state_pipeline import code, block, native_records, reachable
from check_residual_warps8_hvlayout import replace_one

ROOT = Path(__file__).resolve().parents[2]
STATE = 'gdn_wy_residual_warps8_metadata_state'
CONTROL = 'gdn_wy_residual_warps8_hvlayout_state'
PROLOGUE = 'autometadata=prefetch_metadata(p,ws,b,h,0,tid);'
FUTURE = 'if(MetadataPlan::has_next(p.shape,ct))metadata=prefetch_metadata(p,ws,b,h,ct+1,tid);'
PUBLISH = 'sm.gates[tid]=metadata.prefix;sm.beta[tid]=float(metadata.beta);'
OLD = ('sm.gates[tid]=ws.gates[group*Chunk+tid];sm.beta[tid]=tid<unsigned(valid)?'
       'float(p.beta[(int64_t(b)*p.shape.sequence+first+tid)*p.shape.value_heads+h]):0.0f;')


def check_source(control, subject, binding):
    c, s = map(code, (control, subject))
    helper = block(s, 'PrefetchedMetadataprefetch_metadata(')
    expected = ('{autoconstat=MetadataPlan::slot(p.shape,b,h,ct,tid);'
                'PrefetchedMetadataitem{0.0f,BF16::bitcast(0)};'
                'if(at.read_prefix)item.prefix=ws.gates[at.prefix];'
                'if(at.read_beta)item.beta=p.beta[at.beta];returnitem;}')
    if helper != ('PrefetchedMetadataprefetch_metadata(Inputsconst&p,Workspaceconst&ws,'
                  'intb,inth,intct,unsignedtid)'+expected):
        raise AssertionError('metadata loads must use the host-proved plan and preserve BF16 bits')
    state = block(s, STATE+'(').replace(STATE,CONTROL)
    if not (state.index(PROLOGUE) < state.index('for(intct=0;') < state.index(PUBLISH) <
            state.index('commit_wait();') < state.index('PublishedValue::publish<') <
            state.index(FUTURE) < state.index('floatconstdecay=expf(last);')):
        raise AssertionError('metadata issue/publication moved outside proved lifetime')
    state = replace_one(state, PROLOGUE, '')
    state = replace_one(state, FUTURE, '')
    state = replace_one(state, PUBLISH, OLD)
    if state != block(c, CONTROL+'('):
        raise AssertionError('metadata changed math/layout/grid/shared/retirement outside registered moves')
    launch = block(s, 'gdn_wy_forward_residual_warps8_metadata(')
    if ('rc=residual_warps8_hvlayout::configure_hvlayout_output();' not in launch or
        'returnresidual_warps8_hvlayout::launch_hvlayout_output(p,ws,static_cast<BF16*>(output),stream);' not in launch):
        raise AssertionError('metadata must reuse the exact existing HV output body')
    launch = launch.replace('residual_warps8_hvlayout::','').replace('warps8_metadata','warps8_hvlayout')
    if launch != block(c, 'gdn_wy_forward_residual_warps8_hvlayout('):
        raise AssertionError('metadata changed launcher/admission/other stages')
    b = code(binding)
    for required in ('Variant<=12&&(!Variant||Residual)',
                     'Variant==9?gdn_wy_forward_residual_warps8_hvlayout:Variant==10?gdn_wy_forward_residual_warps8_metadata:',
                     'm.def("residual_warps8_metadata",&forward<true,10>,'):
        if b.count(required) != 1:
            raise AssertionError('metadata Python/C++ selector not bound to exact variant10')


def section(isa, marker):
    found = [s for s in isa.split('Disassembly of section ') if
             s.startswith('.text.kernel.') and marker+'E' in s.splitlines()[0]]
    if len(found) != 1:
        raise AssertionError('native metadata/control body absent or ambiguous')
    return found[0]


def native_schedule(text):
    records, edges = native_records(text)
    mma = {pc for pc,op in records.items() if op.startswith('v.mma.f32.bf16.m16n16k16')}
    if len(mma) != 20:
        raise AssertionError('metadata arithmetic inventory changed')
    reverse = {pc:[] for pc in records}
    for pc, targets in edges.items():
        for target in targets: reverse[target].append(pc)
    start = min(mma)
    loop = reachable(start,edges) & reachable(start,reverse)
    if not mma <= loop:
        raise AssertionError('metadata MMA must remain in one recurrent SCC')
    accum = {pc:re.search(r'vreg\[\d+:\d+\]',records[pc])[0] for pc in mma}
    count = Counter(accum.values())
    if sorted(count.values()) != [4,4,12]:
        raise AssertionError('UPDATE versus KH/PR accumulator contract changed')
    update = {pc for pc in mma if count[accum[pc]]==4}
    loads = {pc for pc in loop if records[pc].split()[0] in ('vmem.ld.b32','vmem.ld.b16')}
    if Counter(records[pc].split()[0] for pc in loads) != {'vmem.ld.b32':1,'vmem.ld.b16':1}:
        raise AssertionError('loop must issue exactly one gate and one beta load per thread')
    waits = {pc for pc in loop if records[pc].startswith('s.wait') and 'vldcnt(' in records[pc]}
    if not waits:
        raise AssertionError('metadata global-load completion absent')
    overlap = {}
    for load in loads:
        seen, outcomes, drained = set(), set(), False
        pending = [(pc,0,0) for pc in edges[load]]
        while pending:
            item = pending.pop()
            if item in seen: continue
            seen.add(item)
            pc, updates, other = item
            if pc not in loop or pc in waits:
                # Tail exit is syntactically reachable, but source/host has_next
                # excludes the future load there. Require the full UPDATE even
                # on this edge; do not claim predicate equivalence from a CFG.
                outcomes.add((updates,other)); drained |= pc in waits; continue
            if pc == load:
                raise AssertionError('metadata reissued before a completion wait')
            updates += pc in update; other += pc in mma-update
            if updates+other>20:
                raise AssertionError('metadata wait not reached within one chunk')
            pending.extend((target,updates,other) for target in edges[pc])
        if not drained or outcomes != {(8,0)}:
            raise AssertionError(f'metadata lacks eight UPDATEs before wait: {load:#x} {outcomes}')
        overlap[hex(load)] = sorted(outcomes)
    if sum(records[pc].startswith('s.blksyn.defer') for pc in loop)!=4:
        raise AssertionError('metadata reader retirement changed')
    if sum(op.startswith('s.blksyn.defer') for op in records.values())!=5:
        raise AssertionError('metadata final-state publication changed')
    return dict(records=records,edges=edges,loop=loop,loads=loads,waits=waits,overlap=overlap)


def check_native(isa):
    new = native_schedule(section(isa,STATE))
    old_records, old_edges = native_records(section(isa,CONTROL))
    reverse = {pc:[] for pc in old_records}
    for pc, targets in old_edges.items():
        for target in targets: reverse[target].append(pc)
    start = min(pc for pc,op in old_records.items() if op.startswith('v.mma.'))
    old_loop = reachable(start,old_edges) & reachable(start,reverse)
    fixed = ('tsm.','vmem.','v.mma.','v.cnvt.bf16','v.exp2.','v.mul.f32','v.add.f32','v.fma.f32','s.blksyn')
    def inventory(records,loop):
        return Counter(records[pc].split()[0] for pc in loop if records[pc].startswith(fixed))
    if inventory(old_records,old_loop)!=inventory(new['records'],new['loop']):
        raise AssertionError('metadata changed loop useful work/transfers/shared accesses/barriers')
    expected=inventory(old_records,set(old_records))
    # Two additional STATIC issue sites are the prologue, not extra dynamic
    # loads: L031 proves prologue + chunks-1 lookaheads = chunks exactly.
    expected.update({'vmem.ld.b32':1,'vmem.ld.b16':1})
    if inventory(new['records'],set(new['records']))!=expected:
        raise AssertionError('metadata changed prologue/final publication work or transfers')
    return dict(control_sites=len(old_records),candidate_sites=len(new['records']),
                overlap=new['overlap'],recurring_work_and_traffic='IDENTICAL',device='NOT_RUN')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--self-test',action='store_true'); parser.add_argument('--isa',type=Path)
    args=parser.parse_args()
    src=ROOT/'csrc/gdn_chunk'
    control=(src/'gdn_wy_residual_warps8_hvlayout_ppu.cu').read_text()
    subject=(src/'gdn_wy_residual_warps8_metadata_ppu.cu').read_text()
    binding=(src/'gdn_wy_ops.cpp').read_text()
    check_source(control,subject,binding)
    if args.self_test:
        for label,old,new in (
            ('stale-chunk','ct + 1, tid','ct, tid'),
            ('last-OOB','MetadataPlan::has_next(p.shape, ct)','true'),
            ('no-prologue','auto metadata = prefetch_metadata(p, ws, b, h, 0, tid);','PrefetchedMetadata metadata{};'),
            ('wrong-field','item.beta = p.beta[at.beta]','item.beta = p.beta[at.beta + 1]'),
            ('lost-retire','__syncthreads();  // RETIRE','// RETIRE'),
            ('old-output','residual_warps8_hvlayout::launch_hvlayout_output(p, ws,','residual_warps8_hlayout::launch_hlayout_output(p, ws,'),
            ('rounded-state','state[k][s] *= decay','state[k][s] = float(BF16(state[k][s] * decay))'),
        ):
            if subject.count(old)!=1: raise AssertionError(f'negative seam not unique: {label}')
            try: check_source(control,subject.replace(old,new,1),binding)
            except (AssertionError,ValueError): print(f'[metadata source negative] {label} EXPECTED-RED/PASS')
            else: raise AssertionError(f'escaped source negative: {label}')
        for label,old,new in (
            ('old-variant','&forward<true, 10>','&forward<true, 9>'),
            ('old-launcher','Variant == 10 ? gdn_wy_forward_residual_warps8_metadata :','Variant == 10 ? gdn_wy_forward_residual_warps8_hvlayout :'),
        ):
            if binding.count(old)!=1: raise AssertionError('negative binding seam not unique')
            try: check_source(control,subject,binding.replace(old,new,1))
            except AssertionError: print(f'[metadata binding negative] {label} EXPECTED-RED/PASS')
            else: raise AssertionError(f'escaped binding negative: {label}')
    if args.isa:
        isa=args.isa.read_text(); print('[metadata native]',check_native(isa))
        if args.self_test:
            from check_wy_binary import plant_in_kernel, swap_native_instructions
            body=section(isa,STATE); schedule=native_schedule(body)
            for load in sorted(schedule['loads']):
                following=schedule['edges'][load][0]
                # A rotated loop may branch backward immediately after issue.
                # Follow that branch: swapping it would change the CFG, making
                # the negative a different defect instead of lost overlap.
                while schedule['records'][following].split()[0]=='s.cbr':
                    following=schedule['edges'][following][0]
                # Preserve the complete opcode/operand multiset, not just counts.
                bad=swap_native_instructions(body,min(schedule['waits']),following)
                bad_records,bad_edges=native_records(bad)
                if bad_edges!=schedule['edges'] or Counter(bad_records.values())!=Counter(schedule['records'].values()):
                    raise AssertionError('serialization negative changed CFG or instruction inventory')
                try: native_schedule(bad)
                except AssertionError as error:
                    if 'eight UPDATEs before wait' not in str(error): raise
                    print('[metadata native negative] serialized-same-opcodes+CFG EXPECTED-RED/PASS')
                else: raise AssertionError('serialized load escaped CFG admission')
            try: native_schedule(section(isa,CONTROL))
            except AssertionError: print('[metadata native negative] old-late-load EXPECTED-RED/PASS')
            else: raise AssertionError('old late-load schedule admitted as lookahead')
            for op in ('v.mma.f32.bf16.m16n16k16','vmem.aiu.ld.tsm.l0','s.blksyn.defer','vmem.st.b32x4'):
                try: check_native(plant_in_kernel(isa,STATE,op,'MISSING'))
                except AssertionError: print(f'[metadata native negative] {op} EXPECTED-RED/PASS')
                else: raise AssertionError('missing work/traffic escaped')
    print(f'[metadata] source=PASS native={"PASS" if args.isa else "NOT_RUN"} device=NOT_RUN')


if __name__=='__main__': main()
