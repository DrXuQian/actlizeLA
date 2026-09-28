#!/usr/bin/env python3
"""H has two consumers: bind shared writer AND private snapshot/output reader."""
import argparse
from collections import Counter
from pathlib import Path
from check_state_pipeline import code, block
from check_residual_blayout import native_body

ROOT = Path(__file__).resolve().parents[2]
STATE = 'gdn_wy_residual_warps8_hlayout_state'
OUTPUT = 'gdn_wy_residual_warps8_hlayout_output'


def check_source(control, output, subject, header):
    c, o, s, h = map(code, (control, output, subject, header))
    state = block(s, STATE + '(').replace(STATE, 'gdn_wy_residual_warps8_blayout_state')
    for old, new in (
        ('unsignedconsth_store_base=Snapshot::producer_base(warp,lane);', ''),
        ('Snapshot::producer_offset(h_store_base,k,s)',
         'Snapshot::offset(StateTile::k_row(warp,k)+rc.row,StateTile::column(warp)+rc.col)'),
        ('Snapshot::workspace_offset(group,0,v0)', 'state_offset(group)+v0'),
        ('Snapshot::load(', 'Snapshot::load<true>('),
    ):
        if state.count(old) != 1:
            raise AssertionError(f'paired H state seam missing/ambiguous: {old}')
        state = state.replace(old, new, 1)
    if state != block(c, 'gdn_wy_residual_warps8_blayout_state('):
        raise AssertionError('H experiment changed state beyond paired snapshot writer/readers')
    out = block(s, OUTPUT + '(').replace(OUTPUT, 'gdn_wy_aiu_output').replace('aiu::Tile<','Tile<')
    for old, new in (
        ('usingH=OutputSnapshot;', 'usingH=Tile<Dim,OutputTile::Panel>;'),
        ('H::stage(sm.stage.h,ws.snapshots+Snapshot::workspace_offset(group,0,panel));',
         'H::stage(sm.stage.h,ws.snapshots+state_offset(group)+panel,Dim,Dim);'),
        ('H::load(', 'H::load<true>('),
    ):
        if out.count(old) != 1:
            raise AssertionError(f'paired H output seam missing/ambiguous: {old}')
        out = out.replace(old,new,1)
    if out != block(o, 'gdn_wy_aiu_output('):
        raise AssertionError('H experiment changed output math/rounding/order or another plane')
    launch = block(s, 'gdn_wy_forward_residual_warps8_hlayout(')
    if ('rc=configure_hlayout_output();' not in launch or
        'returnlaunch_hlayout_output(p,ws,static_cast<BF16*>(output),stream);' not in launch):
        raise AssertionError('H private workspace requires the paired output launcher, not the old reader')
    launch = launch.replace('residual_warps8_hlayout','residual_warps8_blayout')
    launch = launch.replace('configure_hlayout_output','configure_aiu_output').replace('launch_hlayout_output','launch_aiu_output')
    if launch != block(c, 'gdn_wy_forward_residual_warps8_blayout('):
        raise AssertionError('H launcher changed admission/workspace/other kernel selection')
    for required in (
        'usingresidual_warps8_blayout::Storage;', 'usingresidual_warps8_blayout::StateTile;',
        'usingresidual_warps8_blayout::Key;', 'usingresidual_warps8_blayout::BIntermediate;',
        'returnstate_offset(group)+int64_t(v)*Dim+k;',
        'usingPublication=StateVectorPlan<ValueTile,Dim,2,Threads>;',
        'uint4constpacked=*reinterpret_cast<uint4const*>(shared+offset(k,v));',
        'cutlass::arch::global_store<uint4,16>(packed,global+int64_t(v)*stride+k,true);',
        'Physical::load(shared+cube(k,v)*256,0,0,fragment);',
        'Physical::stage(shared,global,Dim,OutputTile::Panel);',
        'Physical::load(shared,v,k,fragment);',
    ):
        if required not in h:
            raise AssertionError(f'paired H layout/publication contract absent: {required}')
    for forbidden in ('asm', '__shfl', '__syncthreads', 'float(', 'BF16('):
        if forbidden in h:
            raise AssertionError(f'layout contains unregistered repair/rounding: {forbidden}')
    for required in (
        f'hggcFuncSetAttribute({OUTPUT},hggcFuncAttributeMaxDynamicSharedMemorySize,sizeof(TiledOutputStorage))',
        f'{OUTPUT}<<<unsigned(p.shape.groups()),OutputTile::Threads,sizeof(TiledOutputStorage),stream>>>(p,ws,output);',
    ):
        if required not in s:
            raise AssertionError('new H output not configured/launched with the original geometry')


def subset(counts, prefixes):
    return {k:v for k,v in counts.items() if k.startswith(prefixes)}


def check_native(isa):
    result = {}
    for role, before, after, all_mmas, loop_mmas, expected in (
        ('state','gdn_wy_residual_warps8_blayout_state',STATE,20,20,((20,16),(28,8))),
        ('output','gdn_wy_aiu_output',OUTPUT,40,24,((36,24),(52,8))),
    ):
        old, old_loops = native_body(isa,before+'E',loop_mmas)
        new, new_loops = native_body(isa,after+'E',loop_mmas)
        if len(old_loops) != len(new_loops):
            raise AssertionError(f'H {role} recurrence/panel backedges differ')
        for a, b in [(old,new),*zip(old_loops,new_loops)]:
            ca, cb = (Counter(x.split()[0] for x in seq) for seq in (a,b))
            fixed = ('v.mma.', 'v.exp2.', 'v.mul.f32', 'v.fma.f32', 'v.add.f32',
                     'v.cnvt.bf16', 'tsm.st.', 's.blksyn', 'vmem.st.', 'vmem.ld.',
                     'tsm.ld.b', 'vmem.fence', 'vmem.acp.')
            if subset(ca,fixed) != subset(cb,fixed):
                raise AssertionError(f'H {role} changed useful math/ordinary traffic/retirement')
            aiu='vmem.aiu.ld.tsm.l0.t0.p0.s0.m0.2d.b16.kp1'
            if cb[aiu] != ca[aiu] + int(role=='output'):
                raise AssertionError(f'H {role} did not retain the priced AIU cube inventory')
            if sum('commit_group(0)' in x for x in a) != sum('commit_group(0)' in x for x in b):
                raise AssertionError(f'H {role} lost native async completion')
        for sequence, wanted in ((old,expected[0]),(new,expected[1])):
            counts=Counter(x.split()[0] for x in sequence)
            if counts['v.mma.f32.bf16.m16n16k16'] != all_mmas:
                raise AssertionError(f'H {role} lost native MMA')
            got=tuple(counts[f'tsm.ld.swzl.b32x4.s0.t1.trans{t}'] for t in (0,1))
            if got!=wanted:
                raise AssertionError(f'H {role} reader is stale/wrong orientation: {got}')
            if any('ivreg' in x or 'shuffle' in x or 'tsm.ld.ncom' in x for x in sequence):
                raise AssertionError(f'H {role} has an unregistered remap/reader')
            if counts['vmem.st.b16'] or not counts['vmem.st.b32x4']:
                raise AssertionError(f'H {role} publication was scalarized')
        result[role] = dict(control_sites=len(old),candidate_sites=len(new),
                           control_loops=[len(x) for x in old_loops],candidate_loops=[len(x) for x in new_loops],
                           matrix_loads_control=expected[0],matrix_loads_candidate=expected[1])
    return result | dict(device_latency_and_BC='NOT_MEASURED')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--isa',type=Path)
    args=parser.parse_args()
    d=ROOT/'csrc/gdn_chunk'
    texts=[(d/'gdn_wy_residual_warps8_blayout_ppu.cu').read_text(),
           (d/'gdn_wy_aiu_ppu.cu').read_text(),
           (d/'gdn_wy_residual_warps8_hlayout_ppu.cu').read_text(),
           (ROOT/'include/gdn_qsa/ppu/wy_residual_warps8_hlayout.cuh').read_text()]
    check_source(*texts)
    if args.self_test:
        for label,index,old,new in (
            ('stale-state-reader',2,'Snapshot::load(sm.snapshot,','residual::Snapshot::load<true>(sm.snapshot,'),
            ('stale-publisher-base',2,'Snapshot::workspace_offset(group, 0, v0)','state_offset(group) + v0'),
            ('stale-output-base',2,'Snapshot::workspace_offset(group, 0, panel)','state_offset(group) + panel'),
            ('stale-output-reader',3,'Physical::load(shared, v, k, fragment);','Physical::load<true>(shared, k, v, fragment);'),
            ('wrong-global-pitch',3,'global + int64_t(v) * stride + k','global + int64_t(k) * stride + v'),
            ('rounding',2,'BF16(x * row_decay','BF16(float(BF16(x)) * row_decay'),
            ('retirement',2,'__syncthreads();  // RETIRE','/* missing */  // RETIRE'),
            ('old-output-launch',2,'return launch_hlayout_output(p, ws,','return launch_aiu_output(p, ws,'),
        ):
            mutated=list(texts)
            if mutated[index].count(old)!=1:
                raise AssertionError(f'nonunique H source plant: {label}')
            mutated[index]=mutated[index].replace(old,new,1)
            try: check_source(*mutated)
            except AssertionError: print(f'[H layout source negative] {label} EXPECTED-RED/PASS')
            else: raise AssertionError(f'escaped H source plant: {label}')
    if args.isa:
        isa=args.isa.read_text()
        print('[H layout native]',check_native(isa))
        if args.self_test:
            from check_wy_binary import plant_in_kernel
            for label,marker,old,new in (
                ('state-trans',STATE,'tsm.ld.swzl.b32x4.s0.t1.trans0','tsm.ld.swzl.b32x4.s0.t1.trans1'),
                ('output-trans',OUTPUT,'tsm.ld.swzl.b32x4.s0.t1.trans0','tsm.ld.swzl.b32x4.s0.t1.trans1'),
                ('scalar-publication',STATE,'vmem.st.b32x4','vmem.st.b16'),
                ('missing-aiu',OUTPUT,'vmem.aiu.ld.tsm.l0','MISSING_AIU'),
                ('missing-wait',STATE,'commit_group(0)','MISSING_WAIT'),
                ('missing-barrier',OUTPUT,'s.blksyn.defer','MISSING_BARRIER'),
            ):
                try: check_native(plant_in_kernel(isa,marker,old,new))
                except AssertionError: print(f'[H layout native negative] {label} EXPECTED-RED/PASS')
                else: raise AssertionError(f'escaped H native plant: {label}')
    print(f'[H layout] source=PASS native={"PASS" if args.isa else "NOT_RUN"} device=NOT_RUN')


if __name__=='__main__':
    main()
