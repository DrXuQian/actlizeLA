#!/usr/bin/env python3
"""One UPDATE operand-schedule seam, with a native no-change negative."""
import argparse
from collections import Counter
from pathlib import Path
import re

from check_residual_blayout import native_body
from check_state_pipeline import block, code

ROOT = Path(__file__).resolve().parents[2]
CONTROL = 'gdn_wy_residual_warps8_blayout_stateE'
CANDIDATE = 'gdn_wy_residual_warps8_operands_stateE'


def check_source(control, candidate, header):
    c = code(control)
    s = code(candidate).replace('residual_warps8_operands', 'residual_warps8_blayout')
    replacement = code('''
    uint32_t update_v[2][4], update_k[2][StateTile::KFragments][4];
    prefetch_atoms<Chunk / 16>(
        [&](auto atom, auto slot) {
          BIntermediate::load(sm.scaled, int(atom) * 16, StateTile::column(warp), update_v[int(slot)]);
          CUTE_UNROLL
          for (int k = 0; k < StateTile::KFragments; ++k)
            Key::load<true>(sm.k, int(atom) * 16, StateTile::k_row(warp, k), update_k[int(slot)][k]);
        },
        [&](auto, auto slot) {
          CUTE_UNROLL
          for (int k = 0; k < StateTile::KFragments; ++k)
            bf16_mma(state[k], update_k[int(slot)][k], update_v[int(slot)]);
        });
    ''')
    if s.count(replacement) != 1:
        raise AssertionError('UPDATE operand coordinates/order/slots changed')
    s = s.replace(replacement, 'CUTE_UNROLL' + block(c, 'for(intr=0;r<Chunk;r+=16)', 'bf16_mma(state[k]'), 1)
    if s != c:
        raise AssertionError('change exceeds the registered UPDATE operand region')
    h = code(header)
    for name in ('StateTile', 'Plan', 'Storage', 'Key', 'Inverse', 'Snapshot', 'Value', 'BIntermediate'):
        if f'usingresidual_warps8_blayout::{name};' not in h:
            raise AssertionError(f'control layout/geometry alias missing: {name}')
    if 'usingresidual_operands::prefetch_atoms;' not in h:
        raise AssertionError('not using the host-proved production register-slot schedule')
    for forbidden in ('asm', '__shfl', '__syncthreads', '::stage(', '::publish(', 'template<'):
        if forbidden in h:
            raise AssertionError(f'unregistered delivery work: {forbidden}')


def register_set(operand):
    match = re.fullmatch(r'vreg\[(\d+):(\d+)\](?:\.reuse)?', operand.strip())
    if match:
        return tuple(range(int(match[1]), int(match[2]) + 1))
    match = re.fullmatch(r'vreg(\d+)(?:\.reuse)?', operand.strip())
    return (int(match[1]),) if match else ()


def operand_schedule(sequence):
    """Track physical word writes; classify phases by both native orientations.

    The actual source gate binds K order and coordinates. This separate gate
    measures emitted load lookahead/reuse, not hardware retirement cycles.
    No unbounded trace across recurrence backedges is inferred here.
    """
    loaded, rows, mma_count = {}, [], 0
    phases = {(False, True): 'KH', (False, False): 'PR', (True, False): 'UPDATE'}
    for pc, line in enumerate(sequence):
        opcode, operands = line.split(None, 1) if ' ' in line else (line, '')
        fields = operands.split(',')
        if opcode.startswith('v.mma.f32.bf16'):
            operands_regs = [register_set(x) for x in fields[1:3]]
            tags = []
            for regs in operands_regs:
                if len(regs) != 4 or any(r not in loaded for r in regs):
                    raise AssertionError('MMA input lacks a preceding intact native matrix load')
                tags_for_words = {loaded[r] for r in regs}
                if len(tags_for_words) != 1:
                    raise AssertionError('MMA input was partially overwritten after its matrix load')
                tags.append(next(iter(tags_for_words)))
            phase = phases.get(tuple(t[2] for t in tags))
            if phase is None:
                raise AssertionError('unexpected matrix-load pair/transpose')
            next_line = sequence[pc + 1] if pc + 1 < len(sequence) else ''
            next_regs = register_set(next_line.split(None, 1)[1].split(',')[0]) if next_line.startswith('tsm.ld.swzl') else ()
            tight = bool(set(next_regs).intersection(r for regs in operands_regs for r in regs))
            rows.append(dict(phase=phase, static_site=pc,
                             a_lead_mmas=mma_count-tags[0][1], b_lead_mmas=mma_count-tags[1][1],
                             a_lead_sites=pc-tags[0][0], b_lead_sites=pc-tags[1][0],
                             next_load_overwrites_input=tight))
            mma_count += 1
        writes = opcode.startswith(('v.', 'tsm.ld.', 'vmem.ld'))
        destination = register_set(fields[0]) if writes and fields else ()
        for reg in destination:
            loaded.pop(reg, None)
        if opcode.startswith('tsm.ld.swzl'):
            if len(destination) != 4 or not opcode.endswith(('trans0', 'trans1')):
                raise AssertionError('unproved matrix-load primitive')
            tag = (pc, mma_count, opcode.endswith('trans1'))
            for reg in destination:
                loaded[reg] = tag
    if Counter(row['phase'] for row in rows) != {'KH': 8, 'PR': 4, 'UPDATE': 8}:
        raise AssertionError('native MMA/phase coverage denominator changed')
    return rows


def check_native(isa):
    old, old_loops = native_body(isa, CONTROL, 20)
    new, new_loops = native_body(isa, CANDIDATE, 20)
    if len(old_loops) != len(new_loops):
        raise AssertionError('recurrence CFG/backedge denominator changed')
    for a, b in [(old, new), *zip(old_loops, new_loops)]:
        ca, cb = (Counter(x.split()[0] for x in seq) for seq in (a, b))
        for prefix in ('vmem.', 'tsm.ld.', 'tsm.st.', 'v.mma.', 'v.exp2.',
                       'v.mul.f32', 'v.fma.f32', 'v.add.f32', 'v.cnvt.bf16', 's.blksyn'):
            if {k:v for k,v in ca.items() if k.startswith(prefix)} != {k:v for k,v in cb.items() if k.startswith(prefix)}:
                raise AssertionError(f'native math/traffic/lifetime differs: {prefix}')
    for sequence in (old, new):
        if sum('commit_group(0)' in x for x in sequence) != 1:
            raise AssertionError('async input completion missing/duplicated')
        if any('ivreg' in x or 'shuffle' in x or 'tsm.ld.ncom' in x for x in sequence):
            raise AssertionError('unregistered indirect-register/remap/reader')
    before, after = operand_schedule(old), operand_schedule(new)
    a = [x['b_lead_mmas'] for x in before if x['phase'] == 'UPDATE']
    b = [x['b_lead_mmas'] for x in after if x['phase'] == 'UPDATE']
    if any(new < old for old,new in zip(a,b)) or sum(b) <= sum(a):
        raise AssertionError('UPDATE native operand lookahead did not improve; do not time a no-op experiment')
    tight_before = sum(x['next_load_overwrites_input'] for x in before if x['phase'] == 'UPDATE')
    tight_after = sum(x['next_load_overwrites_input'] for x in after if x['phase'] == 'UPDATE')
    if tight_after >= tight_before:
        raise AssertionError('targeted next-load MMA-source reuse was not reduced')
    return dict(control_static=len(old), candidate_static=len(new),
                control_loops=[len(x) for x in old_loops], candidate_loops=[len(x) for x in new_loops],
                update_b_lead_mmas_control=a, update_b_lead_mmas_candidate=b,
                update_immediate_reuse_control=tight_before, update_immediate_reuse_candidate=tight_after,
                remaining_other_phase_reuse=sum(x['next_load_overwrites_input'] for x in after if x['phase'] != 'UPDATE'),
                scope='static schedule proof; dynamic waits/latency NOT_MEASURED')


def plant_old_schedule(isa):
    sections = isa.split('Disassembly of section ')
    old = next(s for s in sections if s.startswith('.text.kernel.') and CONTROL in s.splitlines()[0])
    index = next(i for i,s in enumerate(sections) if s.startswith('.text.kernel.') and CANDIDATE in s.splitlines()[0])
    old_symbol = old.splitlines()[0].split()[0].removeprefix('.text.kernel.').rstrip(':')
    new_symbol = sections[index].splitlines()[0].split()[0].removeprefix('.text.kernel.').rstrip(':')
    sections[index] = old.replace(old_symbol, new_symbol)
    return 'Disassembly of section '.join(sections)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--isa', type=Path)
    args = parser.parse_args()
    d = ROOT / 'csrc/gdn_chunk'
    texts = [(d/'gdn_wy_residual_warps8_blayout_ppu.cu').read_text(),
             (d/'gdn_wy_residual_warps8_operands_ppu.cu').read_text(),
             (ROOT/'include/gdn_qsa/ppu/wy_residual_warps8_operands.cuh').read_text()]
    check_source(*texts)
    if args.self_test:
        for label, index, old, new in (
            ('wrong-slot',1,'bf16_mma(state[k], update_k[int(slot)][k], update_v[int(slot)]);',
             'bf16_mma(state[k], update_k[int(slot)][k], update_v[1-int(slot)]);'),
            ('missing-K',1,'prefetch_atoms<Chunk / 16>','prefetch_atoms<Chunk / 16 - 1>'),
            ('stale-reader',1,'BIntermediate::load(sm.scaled,','Value::load<true>(sm.scaled,'),
            ('wrong-rounding',1,'BF16(x * row_decay','BF16(float(BF16(x)) * row_decay'),
            ('missing-retire',1,'__syncthreads();  // RETIRE','/* missing */  // RETIRE'),
            ('changed-owner',2,'using residual_warps8_blayout::StateTile;','using residual::StateTile;'),
        ):
            mutated = list(texts)
            if mutated[index].count(old) != 1:
                raise AssertionError(f'nonunique source plant: {label}')
            mutated[index] = mutated[index].replace(old,new,1)
            try: check_source(*mutated)
            except AssertionError: print(f'[warps8 operands source negative] {label} EXPECTED-RED/PASS')
            else: raise AssertionError(f'escaped source negative: {label}')
    if args.isa:
        isa = args.isa.read_text()
        print('[warps8 operands native]', check_native(isa))
        if args.self_test:
            try: check_native(plant_old_schedule(isa))
            except AssertionError as exc:
                if 'lookahead did not improve' not in str(exc):
                    raise AssertionError(f'old-schedule plant failed for wrong reason: {exc}')
                print('[warps8 operands native negative] old-schedule-same-work EXPECTED-RED/PASS')
            else: raise AssertionError('old native schedule escaped')
            from check_wy_binary import plant_in_kernel
            for label, old, new in (('missing-mma','v.mma.f32.bf16','MISSING_MMA'),
                                    ('wrong-trans','tsm.ld.swzl.b32x4.s0.t1.trans0','tsm.ld.swzl.b32x4.s0.t1.trans1'),
                                    ('missing-wait','commit_group(0)','MISSING_WAIT'),
                                    ('missing-barrier','s.blksyn.defer','MISSING_BARRIER')):
                try: check_native(plant_in_kernel(isa,'gdn_wy_residual_warps8_operands_state',old,new))
                except AssertionError: print(f'[warps8 operands native negative] {label} EXPECTED-RED/PASS')
                else: raise AssertionError(f'escaped native plant: {label}')
    print(f'[warps8 operands] source=PASS native={"PASS" if args.isa else "NOT_RUN"} device=NOT_RUN')


if __name__ == '__main__':
    main()
