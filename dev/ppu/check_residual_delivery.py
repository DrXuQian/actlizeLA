#!/usr/bin/env python3
"""Bind independent V16/register-prefetch arms to unchanged residual arithmetic."""
import argparse
from pathlib import Path
import re
from check_state_pipeline import block, code

ROOT=Path(__file__).resolve().parents[2]


def canonical(text, variant):
    s=code(text)
    for old,new in ((f"wy_residual_{variant}.cuh","wy_residual.cuh"),
                    (f"wy::residual_{variant}","wy::residual"),
                    (f"usingnamespaceresidual_{variant};","usingnamespaceresidual;"),
                    (f"gdn_wy_residual_{variant}_state","gdn_wy_residual_state"),
                    (f"gdn_wy_forward_residual_{variant}","gdn_wy_forward_residual")):
        s=s.replace(old,new)
    return s


def check_sources(control, operands, v16):
    c=code(control)
    v=canonical(v16,"v16").replace("Dim/residual_v16::ValueTile","Dim/ValueTile")
    if v!=c: raise AssertionError("V16 changes more than namespace/traits/grid")
    s=canonical(operands,"operands")
    # In each real helper call, same producer/transpose/coordinates/MMA order.
    # The independent host gate executes prefetch_atoms itself and checks both
    # slot lifetimes and complete ascending K coverage.
    replacements=(
      ("uint32_tkh_h", "floatconstlast", "for(intk=0;k<Dim;k+=16)", "bf16_mma(kh[r]",
       "uint32_t kh_h[2][4], kh_key[2][StateTile::ValueFragments][4]; prefetch_atoms<Dim/16>("
       "[&](auto atom,auto slot){ Snapshot::load<true>(sm.snapshot,int(atom)*16,StateTile::column(warp),kh_h[int(slot)]);"
       "CUTE_UNROLL for(int r=0;r<StateTile::ValueFragments;++r) Key::load(sm.k,StateTile::value_row(warp,r),int(atom)*16,kh_key[int(slot)][r]);},"
       "[&](auto,auto slot){CUTE_UNROLL for(int r=0;r<StateTile::ValueFragments;++r) bf16_mma(kh[r],kh_key[int(slot)][r],kh_h[int(slot)]);});"),
      ("uint32_tpr_r", "CUTE_UNROLLfor(intr=0;r<StateTile::ValueFragments", "for(intk=0;k<Chunk;k+=16)", "bf16_mma(value[r]",
       "uint32_t pr_r[2][4], pr_inverse[2][StateTile::ValueFragments][4]; prefetch_atoms<Chunk/16>("
       "[&](auto atom,auto slot){Value::load<true>(sm.residual,int(atom)*16,StateTile::column(warp),pr_r[int(slot)]);"
       "CUTE_UNROLL for(int r=0;r<StateTile::ValueFragments;++r) Inverse::load(sm.inverse,StateTile::value_row(warp,r),int(atom)*16,pr_inverse[int(slot)][r]);},"
       "[&](auto,auto slot){CUTE_UNROLL for(int r=0;r<StateTile::ValueFragments;++r) bf16_mma(value[r],pr_inverse[int(slot)][r],pr_r[int(slot)]);});"),
      ("uint32_tupdate_v", "__syncthreads();", "for(intr=0;r<Chunk;r+=16)", "bf16_mma(state[k]",
       "uint32_t update_v[2][4], update_k[2][StateTile::KFragments][4]; prefetch_atoms<Chunk/16>("
       "[&](auto atom,auto slot){Value::load<true>(sm.scaled,int(atom)*16,StateTile::column(warp),update_v[int(slot)]);"
       "CUTE_UNROLL for(int k=0;k<StateTile::KFragments;++k) Key::load<true>(sm.k,int(atom)*16,StateTile::k_row(warp,k),update_k[int(slot)][k]);},"
       "[&](auto,auto slot){CUTE_UNROLL for(int k=0;k<StateTile::KFragments;++k) bf16_mma(state[k],update_k[int(slot)][k],update_v[int(slot)]);});"))
    for start,end,loop,mma,expected in replacements:
        literal=code(expected)
        if s.count(literal)!=1: raise AssertionError(f"operand delivery/math changed: {start}")
        s=s.replace(literal,"CUTE_UNROLL"+block(c,loop,mma),1)
    if s!=c: raise AssertionError("operand change exceeds the three registered load/MMA regions")


def operand_profile(ops):
    loaded={};n=0;out=[]
    for i,line in enumerate(ops):
        if line.startswith('tsm.ld.swzl'):
            reg=re.search(r'vreg\[\d+:\d+\]',line)[0];loaded[reg]=(i,n)
        if line.startswith('v.mma.f32.bf16'):
            regs=re.findall(r'vreg\[\d+:\d+\]',line)[1:3]
            if len(regs)!=2 or any(reg not in loaded for reg in regs):
                raise AssertionError("MMA operand has no preceding native matrix load")
            out.append(tuple(n-loaded[reg][1] for reg in regs));n+=1
    if n!=40: raise AssertionError("operand profile does not cover all40 MMA sites")
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true');p.add_argument('--isa',type=Path)
    args=p.parse_args();d=ROOT/'csrc/gdn_chunk'
    c=(d/'gdn_wy_residual_ppu.cu').read_text()
    s=(d/'gdn_wy_residual_operands_ppu.cu').read_text()
    v=(d/'gdn_wy_residual_v16_ppu.cu').read_text()
    check_sources(c,s,v)
    if args.self_test:
        for name,which,old,new in (
            ('wrong-register-slot',1,'kh_key[int(slot)][r], kh_h[int(slot)]','kh_key[1-int(slot)][r], kh_h[int(slot)]'),
            ('missing-K-atom',1,'prefetch_atoms<Dim / 16>','prefetch_atoms<Dim / 16 - 1>'),
            ('missing-transpose',1,'Value::load<true>(sm.residual','Value::load<false>(sm.residual'),
            ('v16-wrong-grid',2,'Dim / residual_v16::ValueTile','Dim / 32'),
            ('v16-wrong-K-order',2,'k = 0; k < Dim; k += 16','k = 112; k >= 0; k -= 16')):
            texts=[c,s,v]
            if texts[which].count(old)!=1: raise AssertionError(f'plant not unique:{name}')
            texts[which]=texts[which].replace(old,new,1)
            try: check_sources(*texts)
            except AssertionError: print(f'[residual delivery negative] {name} EXPECTED-RED/PASS')
            else: raise AssertionError(f'escaped source plant:{name}')
    if args.isa:
        from check_wy_binary import kernel_sequences
        sequences=kernel_sequences(args.isa.read_text())
        for marker in ('gdn_wy_residual_stateE','gdn_wy_residual_operands_stateE'):
            ops=next(v for k,v in sequences.items() if marker in k)
            print('[residual native operand lookahead]',marker,operand_profile(ops))
        print('[residual operand interpretation] control already pipelines loads; changed KH/PR ordering is NOT a proven speedup')
    print('[residual delivery source] arithmetic/rounding/shared lifetimes preserved; real traits/template host gates required PASS device=NOT_RUN')


if __name__=='__main__': main()
