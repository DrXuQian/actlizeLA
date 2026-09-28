#!/usr/bin/env python3
"""B-oriented shared-layout seam: no new arithmetic/copy/barrier/remap."""
import argparse
from collections import Counter
from pathlib import Path
import re
from check_state_pipeline import code

ROOT = Path(__file__).resolve().parents[2]


def check_source(control, candidate, header):
    s = code(candidate)
    replacements = (
        ('wy_residual_blayout.cuh', 'wy_residual.cuh'),
        ('gdn_qsa::wy::residual_blayout', 'gdn_qsa::wy::residual'),
        ('usingnamespaceresidual_blayout;', 'usingnamespaceresidual;'),
        ('gdn_wy_residual_blayout_state', 'gdn_wy_residual_state'),
        ('gdn_wy_forward_residual_blayout', 'gdn_wy_forward_residual'),
        ('unsignedconstb_store_base=BIntermediate::producer_base(warp,lane);', ''),
        ('sm.residual[BIntermediate::producer_offset(b_store_base,r,s)]', 'sm.residual[at]'),
        ('sm.scaled[BIntermediate::producer_offset(b_store_base,r,s)]', 'sm.scaled[at]'),
        ('BIntermediate::load(sm.residual,', 'Value::load<true>(sm.residual,'),
        ('BIntermediate::load(sm.scaled,', 'Value::load<true>(sm.scaled,'),
    )
    for old, new in replacements:
        if old not in s:
            raise AssertionError(f'B-layout source seam missing: {old}')
        s = s.replace(old, new)
    if s != code(control):
        raise AssertionError('B-layout changes more than paired intermediate stores/loads')
    h = code(header)
    for required in ('usingPhysical=aiu::Tile<16,16>;',
                     'Physical::load(shared+cube(row,col)*256,0,0,fragment);',
                     'usingresidual::Storage;', 'usingresidual::Plan;',
                     'cute::Swizzle<1,3,3>', 'returnLayout{}(row,col);'):
        if required not in h:
            raise AssertionError(f'B-layout header contract missing: {required}')
    for forbidden in ('__shfl', '__syncthreads', 'remap<', '::stage(', '::publish('):
        if forbidden in h:
            raise AssertionError(f'B-layout added forbidden work: {forbidden}')


def native_body(isa, marker, mma_sites=40):
    sections = [s for s in isa.split('Disassembly of section ')
                if s.startswith('.text.kernel.') and marker in s.splitlines()[0]]
    if len(sections) != 1:
        raise AssertionError(f'exact B-layout native symbol missing/ambiguous: {marker}')
    s = sections[0]
    labels = {name: int(pc, 16) for pc, name in re.findall(r'^\s*([0-9a-f]+) <([^>]+)>:', s, re.M)}
    ops = [(int(pc, 16), text) for pc, text in re.findall(
        r'^\s*([0-9a-f]+):\s+(?:[0-9a-f]{2}\s+){8}\s*([^\n]+)', s, re.M)]
    loops = []
    for pc, text in ops:
        dest = re.search(r'<([^>]+)>', text)
        if text.startswith('s.cbr') and dest and labels[dest[1]] < pc:
            head = labels[dest[1]]
            body = [op for at, op in ops if head <= at <= pc]
            if sum(op.startswith('v.mma.f32.bf16') for op in body) == mma_sites:
                loops.append(body)
    if not loops or len({len(x) for x in loops}) != len(loops):
        raise AssertionError('residual recurrence native backedges not classified')
    return [text for _, text in ops], sorted(loops, key=len)


def check_native(isa):
    old, old_loops = native_body(isa, 'gdn_wy_residual_stateE')
    new, new_loops = native_body(isa, 'gdn_wy_residual_blayout_stateE')
    a, b = (Counter(x.split()[0] for x in seq) for seq in (old, new))
    # Same useful work and byte traffic; the only matrix-load change is8
    # static transposed sites becoming8 non-transposed native SWZL sites.
    for prefix in ('vmem.', 'tsm.st.', 'v.mma.', 'v.exp2.', 'v.mul.f32',
                   'v.fma.f32', 'v.add.f32', 'v.cnvt.bf16', 's.blksyn', 'v.shuffle'):
        if {k:v for k,v in a.items() if k.startswith(prefix)} != {
                k:v for k,v in b.items() if k.startswith(prefix)}:
            raise AssertionError(f'B-layout added/removed math, transfers or sync: {prefix}')
    for trans, count in ((0,32),(1,24)):
        if b[f'tsm.ld.swzl.b32x4.s0.t1.trans{trans}'] != count:
            raise AssertionError('B-layout matrix readers lost their proved orientation')
    if any('ivreg' in x or 'shuffle' in x or 'tsm.ld.ncom' in x for x in new):
        raise AssertionError('B-layout added an indirect remap or non-paired load')
    if len(old_loops) != len(new_loops):
        raise AssertionError('B-layout recurrence branch structure changed')
    summaries = []
    for control, candidate in zip(old_loops,new_loops):
        ca, cb = (Counter(x.split()[0] for x in seq) for seq in (control,candidate))
        addresses = ('v.madl.', 'v.shll.', 'v.shrl.', 'v.lop', 'v.or.', 'v.and.',
                     'v.add.i', 'v.add.co', 'v.add.ci', 'v.cnvt.u', 'v.mov.v2s')
        # Codegen cost is evidence, NOT a performance verdict. A user clarified
        # that no-added-work is a preference; full-call ACU decides promotion.
        summaries.append(dict(control_sites=len(control),candidate_sites=len(candidate),
            control_address_sites=sum(v for k,v in ca.items() if k.startswith(addresses)),
            candidate_address_sites=sum(v for k,v in cb.items() if k.startswith(addresses))))
    return dict(control_static_sites=len(old),candidate_static_sites=len(new),
                recurrence_backedges=summaries, dynamic_counts='NOT_MEASURED')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--isa',type=Path)
    args=p.parse_args()
    d=ROOT/'csrc/gdn_chunk'
    texts=[(d/'gdn_wy_residual_ppu.cu').read_text(),
           (d/'gdn_wy_residual_blayout_ppu.cu').read_text(),
           (ROOT/'include/gdn_qsa/ppu/wy_residual_blayout.cuh').read_text()]
    check_source(*texts)
    if args.self_test:
        for name,which,old,new in (
            ('stale-scaled-writer',1,'sm.scaled[BIntermediate::producer_offset(b_store_base,r,s)]','sm.scaled[at]'),
            ('wrong-residual-reader',1,'BIntermediate::load(sm.residual,','Value::load<true>(sm.residual,'),
            ('extra-barrier',1,'// VALUES_READY','__syncthreads(); // VALUES_READY'),
            ('wrong-rounding',1,'BF16(x * row_decay','BF16(float(BF16(x)) * row_decay'),
            ('wrong-native-orientation',2,'Physical::load(shared','Physical::load<true>(shared'),
            ('extra-remap',2,'Physical::load(shared','__syncthreads(); Physical::load(shared')):
            mutated=list(texts)
            if mutated[which].count(old)!=1: raise AssertionError(f'plant not unique: {name}')
            mutated[which]=mutated[which].replace(old,new,1)
            try: check_source(*mutated)
            except AssertionError: print(f'[B-layout source negative] {name} EXPECTED-RED/PASS')
            else: raise AssertionError(f'escaped source plant: {name}')
    if args.isa:
        isa=args.isa.read_text()
        print('[B-layout native]',check_native(isa))
        if args.self_test:
            # Native-text plants affect ONLY the candidate, not an old control.
            from check_wy_binary import plant_in_kernel
            for name,old,new in (
                ('native-transpose','tsm.ld.swzl.b32x4.s0.t1.trans0','tsm.ld.swzl.b32x4.s0.t1.trans1'),
                ('native-copy','vmem.aiu.ld.tsm.l0','vmem.aiu.ld.tsm.l1'),
                ('native-extra-remap','v.or.b32','v.shuffle.idx.b32')):
                changed=plant_in_kernel(isa,'gdn_wy_residual_blayout_state',old,new)
                try: check_native(changed)
                except AssertionError: print(f'[B-layout native negative] {name} EXPECTED-RED/PASS')
                else: raise AssertionError(f'escaped native plant: {name}')
    print('[B-layout] local seam PASS device numerics/BC/performance=NOT_RUN')


if __name__=='__main__': main()
