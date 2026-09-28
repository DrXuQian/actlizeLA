#!/usr/bin/env python3
"""Bind input-V retirement, private Vnew publisher and the paired output reader."""
import argparse
from collections import Counter
from pathlib import Path
from check_state_pipeline import code, block
from check_residual_blayout import native_body

ROOT=Path(__file__).resolve().parents[2]
STATE='gdn_wy_residual_warps8_hvlayout_state'
OUTPUT='gdn_wy_residual_warps8_hvlayout_output'


def replace_one(text, old, new):
    if text.count(old)!=1:
        raise AssertionError(f'paired Vnew seam missing/ambiguous: {old}')
    return text.replace(old,new,1)


def check_source(control, subject, header):
    c,s,h=map(code,(control,subject,header))
    state=block(s,STATE+'(').replace(STATE,'gdn_wy_residual_warps8_hlayout_state')
    state=replace_one(state,'floatconstx=row<valid?value[r][s]:0.0f;',
        'unsignedconstat=Value::offset(row,StateTile::column(warp)+rc.col);floatconstx=row<valid?value[r][s]:0.0f;')
    state=replace_one(state,'sm.value[PublishedValue::producer_offset(b_store_base,r,s)]=BF16(x);',
                           'sm.value[at]=BF16(x);')
    state=replace_one(state,'PublishedValue::publish<Plan::Threads>(sm.value,ws.vnew+PublishedValue::workspace_offset(group,0,v0));',
                           'Value::publish<Plan::Threads>(sm.value,ws.vnew+tile_offset(group)+v0,Dim);')
    if state!=block(c,'gdn_wy_residual_warps8_hlayout_state('):
        raise AssertionError('Vnew changed input-V/H/K/math/rounding/retirement outside the paired publisher')
    out=block(s,OUTPUT+'(').replace(OUTPUT,'gdn_wy_residual_warps8_hlayout_output')
    out=replace_one(out,'usingV=OutputValue;usingOutput=aiu::Tile<Chunk,OutputTile::Panel>;',
                        'usingV=aiu::Tile<Chunk,OutputTile::Panel>;')
    out=replace_one(out,'V::stage(sm.v,ws.vnew+PublishedValue::workspace_offset(group,0,panel));',
                        'V::stage(sm.v,ws.vnew+tile_offset(group)+panel,Dim,Chunk);')
    out=replace_one(out,'V::load(sm.v,k,OutputTile::column(warp,c),v);',
                        'V::load<true>(sm.v,k,OutputTile::column(warp,c),v);')
    out=replace_one(out,'sm.stage.output[Output::offset(', 'sm.stage.output[V::offset(')
    out=replace_one(out,'Output::publish<OutputTile::Threads>(sm.stage.output,',
                        'V::publish<OutputTile::Threads>(sm.stage.output,')
    if out!=block(c,'gdn_wy_residual_warps8_hlayout_output('):
        raise AssertionError('Vnew changed H/math or the public-output exchange/publisher')
    launch=block(s,'gdn_wy_forward_residual_warps8_hvlayout(')
    if ('rc=configure_hvlayout_output();' not in launch or
        'returnlaunch_hvlayout_output(p,ws,static_cast<BF16*>(output),stream);' not in launch):
        raise AssertionError('new private Vnew must dispatch the paired output, never old-H output')
    launch=launch.replace('warps8_hvlayout','warps8_hlayout').replace('hvlayout_output','hlayout_output')
    if launch!=block(c,'gdn_wy_forward_residual_warps8_hlayout('):
        raise AssertionError('Vnew changed launcher/admission/workspace extent/other stages')
    for required in (
        'usingresidual_warps8_hlayout::Snapshot;', 'usingresidual_warps8_hlayout::Storage;',
        'usingresidual_warps8_hlayout::Value;', 'usingresidual_warps8_hlayout::BIntermediate;',
        'structPublishedValue:BIntermediate{', 'returntile_offset(group)+int64_t(v)*Chunk+t;',
        'usingPublication=StateVectorPlan<ValueTile,Chunk,2,Threads>;',
        'uint4constpacked=*reinterpret_cast<uint4const*>(shared+offset(t,v));',
        'cutlass::arch::global_store<uint4,16>(packed,global+int64_t(v)*Chunk+t,true);',
        'Physical::stage(shared,global,Chunk,OutputTile::Panel);',
        'Physical::load(shared,v,t,fragment);',
    ):
        if required not in h: raise AssertionError(f'Vnew layout contract absent: {required}')
    for forbidden in ('asm','__shfl','__syncthreads','BF16(','float('):
        if forbidden in h: raise AssertionError('layout introduced a repair/copy/rounding operation')
    for required in (
        f'hggcFuncSetAttribute({OUTPUT},hggcFuncAttributeMaxDynamicSharedMemorySize,sizeof(TiledOutputStorage))',
        f'{OUTPUT}<<<unsigned(p.shape.groups()),OutputTile::Threads,sizeof(TiledOutputStorage),stream>>>(p,ws,output);',
    ):
        if required not in s: raise AssertionError('new output geometry/attribute/launch absent')


def subset(counter,prefixes):
    return {k:v for k,v in counter.items() if k.startswith(prefixes)}


def check_binding(text):
    binding=code(text)
    for expected in (
        'Variant<=12&&(!Variant||Residual)',
        'Variant==8?gdn_wy_forward_residual_warps8_hlayout:Variant==9?gdn_wy_forward_residual_warps8_hvlayout:',
        'm.def("residual_warps8_hvlayout",&forward<true,9>,',
    ):
        if binding.count(expected)!=1:
            raise AssertionError('HV Python binding must select variant9 and its exact native launcher')


def check_native(isa):
    result={}
    for role,old_name,new_name,mmas,loop_mmas,orientations in (
        ('state','gdn_wy_residual_warps8_hlayout_state',STATE,20,20,((28,8),(28,8))),
        ('output','gdn_wy_residual_warps8_hlayout_output',OUTPUT,40,24,((52,8),(60,0))),
    ):
        old,old_loops=native_body(isa,old_name+'E',loop_mmas)
        new,new_loops=native_body(isa,new_name+'E',loop_mmas)
        if len(old_loops)!=len(new_loops): raise AssertionError('Vnew recurrence/panel backedges changed')
        for before,after in [(old,new),*zip(old_loops,new_loops)]:
            a,b=(Counter(x.split()[0] for x in seq) for seq in (before,after))
            fixed=('v.mma.','v.exp2.','v.mul.f32','v.fma.f32','v.add.f32','v.cnvt.bf16',
                   'tsm.st.','tsm.ld.b','vmem.st.','vmem.ld.','vmem.aiu.',
                   's.blksyn','vmem.fence','vmem.acp.')
            if subset(a,fixed)!=subset(b,fixed):
                raise AssertionError(f'Vnew {role} changed useful work/ordinary traffic/copies/retirement')
            if sum('commit_group(0)' in x for x in before)!=sum('commit_group(0)' in x for x in after):
                raise AssertionError('Vnew lost native async completion')
        for seq,want in ((old,orientations[0]),(new,orientations[1])):
            counts=Counter(x.split()[0] for x in seq)
            if counts['v.mma.f32.bf16.m16n16k16']!=mmas: raise AssertionError('Vnew lost MMA')
            if tuple(counts[f'tsm.ld.swzl.b32x4.s0.t1.trans{t}'] for t in (0,1))!=want:
                raise AssertionError(f'Vnew {role} reader absent/stale')
            if any('ivreg' in x or 'shuffle' in x or 'tsm.ld.ncom' in x for x in seq):
                raise AssertionError('Vnew introduced indirect slots/repair/another matrix reader')
            if counts['vmem.st.b16'] or not counts['vmem.st.b32x4']:
                raise AssertionError('Vnew lost vector publication')
        result[role]=dict(control_sites=len(old),candidate_sites=len(new),
                         control_loops=[len(x) for x in old_loops],candidate_loops=[len(x) for x in new_loops],
                         normal_trans_sites_control=orientations[0],normal_trans_sites_candidate=orientations[1])
    return result|dict(device_numerics_and_performance='NOT_MEASURED_BY_THIS_CHECK')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--isa',type=Path);args=parser.parse_args()
    src=ROOT/'csrc/gdn_chunk'
    texts=[(src/'gdn_wy_residual_warps8_hlayout_ppu.cu').read_text(),
           (src/'gdn_wy_residual_warps8_hvlayout_ppu.cu').read_text(),
           (ROOT/'include/gdn_qsa/ppu/wy_residual_warps8_hvlayout.cuh').read_text()]
    check_source(*texts)
    binding=(src/'gdn_wy_ops.cpp').read_text()
    check_binding(binding)
    if args.self_test:
        for label,old,new in (
            ('cpp-old-variant','&forward<true, 9>','&forward<true, 8>'),
            ('cpp-old-launcher','Variant == 9 ? gdn_wy_forward_residual_warps8_hvlayout :',
             'Variant == 9 ? gdn_wy_forward_residual_warps8_hlayout :'),
        ):
            if binding.count(old)!=1: raise AssertionError('binding negative target absent/ambiguous')
            try: check_binding(binding.replace(old,new,1))
            except AssertionError: print(f'[Vnew source negative] {label} EXPECTED-RED/PASS')
            else: raise AssertionError(f'escaped Vnew binding negative: {label}')
        for label,index,old,new in (
            ('stale-shared-writer',1,'PublishedValue::producer_offset(b_store_base, r, s)','Value::offset(row, StateTile::column(warp) + rc.col)'),
            ('old-publisher-address',1,'PublishedValue::workspace_offset(group, 0, v0)','tile_offset(group) + v0'),
            ('old-output-address',1,'PublishedValue::workspace_offset(group, 0, panel)','tile_offset(group) + panel'),
            ('old-output-reader',2,'Physical::load(shared, v, t, fragment);','Physical::load<true>(shared, t, v, fragment);'),
            ('wrong-pitch',2,'return tile_offset(group) + int64_t(v) * Chunk + t;','return tile_offset(group) + int64_t(v) * Dim + t;'),
            ('input-reader-uses-new-layout',1,'float(sm.value[at])','float(sm.value[PublishedValue::offset(row, StateTile::column(warp) + rc.col)])'),
            ('lost-input-retirement',1,'__syncthreads();  // RESIDUAL_READY','/* removed */  // RESIDUAL_READY'),
            ('old-output-dispatch',1,'return launch_hvlayout_output(p, ws,','return launch_hlayout_output(p, ws,'),
            ('public-output-transposed',1,'Output::offset(OutputTile::row(warp) + rc.row,\n                                  OutputTile::column(warp, c) + rc.col)',
                'Output::offset(OutputTile::column(warp, c) + rc.col, OutputTile::row(warp) + rc.row)'),
            ('rounding',1,'BF16(x * row_decay','BF16(float(BF16(x)) * row_decay'),
        ):
            mutated=list(texts)
            if mutated[index].count(old)!=1: raise AssertionError(f'nonunique Vnew plant: {label}')
            mutated[index]=mutated[index].replace(old,new,1)
            try: check_source(*mutated)
            except AssertionError: print(f'[Vnew source negative] {label} EXPECTED-RED/PASS')
            else: raise AssertionError(f'escaped Vnew source plant: {label}')
    if args.isa:
        isa=args.isa.read_text();print('[Vnew native]',check_native(isa))
        if args.self_test:
            from check_wy_binary import plant_in_kernel
            for label,marker,old,new in (
                ('stale-output-trans',OUTPUT,'tsm.ld.swzl.b32x4.s0.t1.trans0','tsm.ld.swzl.b32x4.s0.t1.trans1'),
                ('scalar-publication',STATE,'vmem.st.b32x4','vmem.st.b16'),
                ('missing-aiu',OUTPUT,'vmem.aiu.ld.tsm.l0','MISSING_AIU'),
                ('missing-input-wait',STATE,'commit_group(0)','MISSING_WAIT'),
                ('missing-barrier',STATE,'s.blksyn.defer','MISSING_BARRIER'),
                ('missing-mma',OUTPUT,'v.mma.f32.bf16.m16n16k16','MISSING_MMA'),
            ):
                try: check_native(plant_in_kernel(isa,marker,old,new))
                except AssertionError: print(f'[Vnew native negative] {label} EXPECTED-RED/PASS')
                else: raise AssertionError(f'escaped Vnew native plant: {label}')
    print(f'[Vnew layout] source=PASS native={"PASS" if args.isa else "NOT_RUN"} device=NOT_RUN')


if __name__=='__main__': main()
