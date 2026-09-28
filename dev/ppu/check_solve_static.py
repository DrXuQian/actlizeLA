#!/usr/bin/env python3
"""Admit only static diagonal indexing, never a lower-precision inverse."""
import argparse
from collections import Counter
from pathlib import Path
import re
import subprocess
from check_state_pipeline import code, block, native_records
from check_residual_metadata import section
from check_residual_warps8_hvlayout import replace_one

ROOT = Path(__file__).resolve().parents[2]
CONTROL = 'gdn_wy_split_solve'
SUBJECT = 'gdn_wy_split_solve_static'


def check_source(control, subject, helper, hv, binding):
    c, s, h, v, b = map(code, (control, subject, helper, hv, binding))
    old = block(c, CONTROL+'(')
    new = block(s, SUBJECT+'(').replace(SUBJECT, CONTROL)
    start, end = old.index('floatcolumn[16];'), old.index('__syncthreads();', old.index('floatcolumn[16];'))
    seam = ('floatcolumn[16];unsignedconstdiagonal_base=warp*16*(Chunk+1);'
            'solve_static::diagonal(sm.lower+diagonal_base,sm.inverse+diagonal_base,lane,column);')
    if replace_one(new, seam, old[start:end]) != old:
        raise AssertionError('solve changed work/rounding/TF32/barriers/layout outside diagonal indexing')
    if block(c,'structSolveStorage') != block(s,'structSolveStorage'):
        raise AssertionError('solve storage/layout changed')
    expected_sub = ('subtract_row(float&x,floatconst*lower,floatconst(&column)[16]){'
                    'ifconstexpr(K<Row){x-=lower[Row*64+K]*column[K];'
                    'subtract_row<Row,K+1>(x,lower,column);}}')
    expected_diag = ('diagonal(floatconst*lower,float*inverse,unsignedlane,float(&column)[16]){'
                     'ifconstexpr(Row<16){floatx=Row==int(lane%16)?1.0f:0.0f;'
                     'subtract_row<Row>(x,lower,column);column[Row]=x;'
                     'if(lane<16)inverse[Row*64+lane]=x;'
                     'diagonal<Row+1>(lower,inverse,lane,column);}}')
    if (block(h,'subtract_row(') != expected_sub or block(h,'diagonal(') != expected_diag or
        'template<intRow,intK=0>' not in h or 'template<intRow=0>' not in h):
        raise AssertionError('compile-time helper changed index/order/term/owner/coverage')
    launch = block(s,'gdn_wy_forward_residual_solve_static(')
    # Require actual new bindings before normalizing to the unchanged HV ABI.
    for token in ('solve_static::configure();','solve_static::launch_inverse(p,inverse_ws,stream);',
                  'rc=configure_hvlayout_state();','rc=launch_hvlayout_state(p,ws,final,stream);'):
        if launch.count(token)!=1: raise AssertionError('candidate dispatch is stale or ambiguous')
    launch = launch.replace('gdn_wy_forward_residual_solve_static','gdn_wy_forward_residual_warps8_hvlayout')
    launch = replace_one(launch,'usingsolve_static::Key;','')
    launch = replace_one(launch,'solve_static::configure()','configure_split_prepare()')
    launch = replace_one(launch,'solve_static::launch_inverse(','launch_split_inverse(')
    launch = replace_one(launch,'configure_hvlayout_state()',
        'int(hggcFuncSetAttribute(gdn_wy_residual_warps8_hvlayout_state,'
        'hggcFuncAttributeMaxDynamicSharedMemorySize,sizeof(Storage)))')
    launch = replace_one(launch,'rc=launch_hvlayout_state(p,ws,final,stream);',
        'unsignedconstgrid=unsigned(int64_t(batch)*value_heads*(Dim/ValueTile));'
        'gdn_wy_residual_warps8_hvlayout_state<<<grid,Plan::Threads,sizeof(Storage),stream>>>(p,ws,final);'
        'rc=int(hggcGetLastError());')
    if launch != block(v,'gdn_wy_forward_residual_warps8_hvlayout('):
        raise AssertionError('candidate changed admission/workspace/other-stage ordering')
    required = (
        ('intconfigure()',s,'{returnint(hggcFuncSetAttribute(gdn_wy_split_solve_static,'
         'hggcFuncAttributeMaxDynamicSharedMemorySize,sizeof(SolveStorage)));}'),
        ('intlaunch_inverse(Inputs p, Workspace ws, gdn_arch::Stream stream)',s,
         '{intconstrc=launch_split_prefix(p,ws,stream);if(rc)returnrc;'
         'gdn_wy_split_solve_static<<<unsigned(p.shape.groups()),Plan::SolveThreads,'
         'sizeof(SolveStorage),stream>>>(p,ws);returnint(hggcGetLastError());}'),
        ('intlaunch_split_prefix(Inputs p, Workspace ws, gdn_arch::Stream stream)',c,
         '{usingnamespacesplit_prepare;gdn_wy_split_prefix<<<unsigned(p.shape.groups()),'
         'Plan::PrefixThreads,0,stream>>>(p,ws);returnint(hggcGetLastError());}'),
        ('intconfigure_hvlayout_state()',v,
         '{returnint(hggcFuncSetAttribute(gdn_wy_residual_warps8_hvlayout_state,'
         'hggcFuncAttributeMaxDynamicSharedMemorySize,sizeof(Storage)));}'),
        ('intlaunch_hvlayout_state(Inputs p, Workspace ws, float* final, gdn_arch::Stream stream)',v,
         '{unsignedconstgrid=unsigned(int64_t(p.shape.batch)*p.shape.value_heads*(Dim/ValueTile));'
         'gdn_wy_residual_warps8_hvlayout_state<<<grid,Plan::Threads,sizeof(Storage),stream>>>(p,ws,final);'
         'returnint(hggcGetLastError());}'),
    )
    for name,text,body in required:
        if block(text,name) != code(name)+body:
            raise AssertionError('same-symbol host reuse/launch contract changed: '+name)
    for token in ('Variant<=12&&(!Variant||Residual)',
        'Variant==10?gdn_wy_forward_residual_warps8_metadata:Variant==11?gdn_wy_forward_residual_solve_static:',
        'm.def("residual_solve_static",&forward<true,11>,'):
        if b.count(token)!=1: raise AssertionError('exact variant11 binding absent')


def diagonal_region(isa, marker):
    records, edges = native_records(section(isa,marker))
    barriers = [pc for pc,op in records.items() if op.startswith('s.blksyn.defer')]
    if len(barriers)!=5: raise AssertionError('solve barrier inventory changed')
    lo,hi=barriers[1:3]
    region={pc:op for pc,op in records.items() if lo<pc<hi}
    if any(op.startswith('v.mma.') for op in region.values()):
        raise AssertionError('diagonal region not bounded by KKT/inverse handoffs')
    return records,edges,region


def check_native(isa):
    old,_,before=diagonal_region(isa,CONTROL)
    new,edges,after=diagonal_region(isa,SUBJECT)
    if any('ivreg' in op for op in new.values()):
        raise AssertionError('static candidate retained indirect register reads')
    if any('pipe_flush' in op for op in after.values()):
        raise AssertionError('static diagonal retained indexed-loop pipe flush')
    if any(target<=pc for pc in after for target in edges[pc] if target in after):
        raise AssertionError('static diagonal retained a native backedge')
    cnt=Counter(op.split()[0] for op in after.values())
    words=sum(cnt[op]*width for op,width in [('tsm.ld.b32',1),('tsm.ld.b32x2',2),('tsm.ld.b32x4',4)])
    if words!=120 or cnt['v.fma.f32.rtte']!=120 or cnt['tsm.st.b32']!=16:
        raise AssertionError('diagonal must read120coefficients, apply120orderedFMAs and publish16rows')
    if any(not re.match(r'v\.fma\.f32\.rtte\s+vreg\d+,\s*!vreg\d+',op)
           for op in after.values() if op.startswith('v.fma.')):
        raise AssertionError('diagonal lost negative-coefficient round-to-nearest FMA')
    if cnt['v.add.f32'] or cnt['v.mul.f32']:
        raise AssertionError('diagonal FMA contraction changed')
    fixed=('v.mma.','v.exp2.','v.cnvt.','v.fma.f32','v.mul.f32','v.add.f32',
           'vmem.','s.blksyn','tsm.ld.swzl','tsm.st.')
    def outside(records,region):
        return Counter(op.split()[0] for pc,op in records.items() if pc not in region and op.startswith(fixed))
    if outside(old,before)!=outside(new,after):
        raise AssertionError('native non-diagonal precision/math/traffic/barriers changed')
    all_ops=Counter(op.split()[0] for op in new.values())
    if (all_ops['v.mma.f32.bf16.m16n16k16'],all_ops['v.mma.f32.tf32.m16n16k8'],
        all_ops['vmem.st.b32x4']) != (8,12,4):
        raise AssertionError('solve lost KKT/three-productTF32/vector publication')
    return dict(control_sites=len(old),candidate_sites=len(new),diagonal_sites=(len(before),len(after)),
                indirect_register_sites=(sum('ivreg' in op for op in before.values()),0),
                diagonal_coefficient_words=words,diagonal_ordered_fmas=120,device='NOT_RUN')


def main():
    p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true');p.add_argument('--isa',type=Path)
    p.add_argument('--host',type=Path,help='actual L032 executable, including omitted-context negative')
    args=p.parse_args();src=ROOT/'csrc/gdn_chunk'
    texts=[(src/'gdn_wy_split_prepare_ppu.cu').read_text(),(src/'gdn_wy_solve_static_ppu.cu').read_text(),
           (ROOT/'include/gdn_qsa/ppu/wy_solve_static.cuh').read_text(),
           (src/'gdn_wy_residual_warps8_hvlayout_ppu.cu').read_text(),(src/'gdn_wy_ops.cpp').read_text()]
    check_source(*texts)
    if args.self_test:
        for label,i,old,new in (
            ('wrong-column',2,'* column[K]','* column[K + 1]'),
            ('omitted-term',2,'K < Row','K + 1 < Row'),
            ('omitted-row',2,'Row < 16','Row < 15'),
            ('wrong-owner',2,'lane < 16','lane < 32'),
            ('wrong-warp-base',1,'warp * 16 * (Chunk + 1)','warp * 16 * Chunk'),
            ('wrong-solve-sign',1,'= -merged[s]','= merged[s]'),
            ('wrong-launch',1,'solve_static::launch_inverse(p, inverse_ws, stream)','launch_split_inverse(p, inverse_ws, stream)'),
            ('wrong-state',3,'gdn_wy_residual_warps8_hvlayout_state<<<grid, Plan::Threads','gdn_wy_residual_warps8_metadata_state<<<grid, Plan::Threads'),
            ('inverse-alias',1,'inverse_ws.snapshots = ws.w','inverse_ws.snapshots = ws.snapshots'),
            ('wrong-binding',4,'&forward<true, 11>','&forward<true, 9>'),
        ):
            if old not in texts[i]: raise AssertionError('missing negative seam:'+label)
            bad=texts.copy();bad[i]=bad[i].replace(old,new,1)
            try:check_source(*bad)
            except AssertionError:print('[solve static source negative]',label,'EXPECTED-RED/PASS')
            else:raise AssertionError('escaped source negative:'+label)
    if args.host:
        subprocess.run([str(args.host)],check=True)
        if args.self_test:
            bad=subprocess.run([str(args.host),'--omit-last-context'],capture_output=True,text=True)
            if bad.returncode==0 or 'coverage denominator incomplete' not in bad.stderr:
                raise AssertionError('missing one context escaped the actual host denominator')
            print('[solve static host negative] omitted-context EXPECTED-RED/PASS')
    if args.isa:
        isa=args.isa.read_text();print('[solve static native]',check_native(isa))
        if args.self_test:
            from check_wy_binary import plant_in_kernel
            for op in ('v.mma.f32.tf32.m16n16k8','vmem.st.b32x4','s.blksyn.defer'):
                try:check_native(plant_in_kernel(isa,SUBJECT,op,'MISSING'))
                except AssertionError:print('[solve static native negative]',op,'EXPECTED-RED/PASS')
                else:raise AssertionError('escaped native work negative')
            subject=section(isa,SUBJECT)
            _,_,region=diagonal_region(isa,SUBJECT)
            fma=next(op for op in region.values() if op.startswith('v.fma.f32'))
            bad=isa.replace(subject,replace_one(subject,fma,fma.replace('!','',1)))
            try:check_native(bad)
            except AssertionError as error:
                if 'negative-coefficient' not in str(error):raise
                print('[solve static native negative] positive-FMA EXPECTED-RED/PASS')
            else:raise AssertionError('wrong native FMA sign escaped')
            # Restore the actual old lowering under the new symbol: must reject
            # the exact indirect-index defect, not merely an absent symbol.
            bad=isa.replace(subject,section(isa,CONTROL).replace(CONTROL+'E',SUBJECT+'E'))
            try:check_native(bad)
            except AssertionError as error:
                if 'indirect register' not in str(error):raise
                print('[solve static native negative] old-indirect-body EXPECTED-RED/PASS')
            else:raise AssertionError('old lowering escaped')
    print('[solve static] source=PASS native='+('PASS' if args.isa else 'NOT_RUN')+' device=NOT_RUN')


if __name__=='__main__':main()
