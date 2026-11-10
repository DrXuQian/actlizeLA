"""Attribution negatives; fixture instruction counts are not device evidence."""
from collections import Counter
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'dev/ppu/zw810_fla_kda')]
import analyze_native_work as native


def signature():
    instructions = [
        'v.and.b32 vreg79, vreg6, 0x7f800000',
        'v.cmp.ne.i32 vcc, sreg35, vreg79',
        's.lop.emsk sreg2, vcc # 0x8',
        's.xor.b32 sreg12, emsk, sreg2',
        'v.and.b32 vreg79, vreg6, 0x7fff',
        'v.cmp.gt.i16 vcc, vreg6, 0xffff',
        'v.cmp.eq.i32 sreg2, vreg79, 0x0',
        'v.and.b32 vreg79, vreg6.reuse, 0x18000',
        's.or.b32 sreg2, vcc, sreg2',
        'v.cmp.ne.i32 vcc, vreg79, 0x18000',
        'v.add.i32 vreg79, vreg6.reuse, 0x10000',
        's.and.b32 sreg2, vcc, sreg2',
        'v.csel.b32 vreg79, vreg6, vreg79, sreg2',
        's.lop.emsk sreg2, sreg12 # 0xe',
        's.xor.b32 emsk, emsk, sreg2',
    ]
    return [dict(Source=s, Address=hex(i*8), **{'Instructions Executed': '32'})
            for i, s in enumerate(instructions)]


def test_core_is_15_ops_not_all_integer_ops():
    rows = signature()
    rows.insert(1, dict(Source='tsm.st.b16 vreg1, [0x0 + vreg2 * 0x1]', Address='0xff0',
                        **{'Instructions Executed': '512'}))
    rows.insert(2, dict(Source='s.cnvt.i64.i32 sreg[16:17], sreg9', Address='0xff8',
                        **{'Instructions Executed': '128'}))
    sites, selected = native.software_bf16(rows)
    assert len(sites) == 1 and sites[0]['conversions_warp'] == 32
    assert sites[0]['core_instructions'] == 480
    assert len(selected) == 15 and 1 not in selected and 2 not in selected
    classes = Counter()
    for i, row in enumerate(rows):
        classes['software_bf16_core' if i in selected else native.category(row['Source'].split()[0])] += int(row['Instructions Executed'])
    assert sum(classes.values()) == 480 + 512 + 128
    assert classes['memory'] == 512 and classes['native_conversion'] == 128


@pytest.mark.parametrize('plant', ['round-mask', 'increment', 'missing-op', 'extra-op', 'different-count', 'no-close'])
def test_changed_signature_must_fail_not_be_attributed(plant):
    rows = signature()
    if plant == 'round-mask': rows[4]['Source'] = rows[4]['Source'].replace('0x7fff', '0x7ffe')
    elif plant == 'increment': rows[10]['Source'] = rows[10]['Source'].replace('0x10000', '0x10001')
    elif plant == 'missing-op': del rows[4]
    elif plant == 'extra-op': rows.insert(4, dict(rows[4], Address='0x100'))
    elif plant == 'different-count': rows[4]['Instructions Executed'] = '31'
    elif plant == 'no-close': rows.pop()
    with pytest.raises(ValueError): native.software_bf16(rows)


def test_native_bf16_is_not_the_software_chain():
    rows = [dict(Source='v.cnvt.bf16.f32.rtte vreg1, vreg2', Address='0x0',
                 **{'Instructions Executed': '32'})]
    assert native.software_bf16(rows) == ([], set())
    assert native.category(rows[0]['Source'].split()[0]) == 'native_conversion'


def test_work_scope_counts_wu_plus_state_and_keeps_fills():
    assert native.work_phase('fla', 'recompute_w_u_fwd_kernel') == 'state_equivalent'
    assert native.work_phase('fla', 'chunk_gated_delta_rule_fwd_kernel_h_blockdim64') == 'state_equivalent'
    assert native.work_phase('fla', 'vectorized_elementwise_kernel<FillFunctor<float>>') == 'other_included'
