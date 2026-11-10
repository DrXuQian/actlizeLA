"""Conversion contract and source-bound negatives; CPU tests are not PPU execution."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'dev/ppu'), str(ROOT / 'tools'), str(ROOT / 'tests')]
import generate_native_bf16 as gen
import native_bf16_device as numeric
import check_native_bf16 as gate
import gdn_kda_contract as contract


def test_host_rounding_contract_all_high16_boundaries():
    bits = numeric.corpus()
    assert bits.dtype == np.uint32 and len(bits) == 65536 * 7
    assert len(np.unique(bits[:65536*6])) == 65536*6
    # Retained source's independent mantissa/round/sticky formulation.
    b = bits.copy()
    finite = (b & 0x7f800000) != 0x7f800000
    round_bit = (b & 0x8000) != 0
    sticky = (b & 0x7fff) != 0
    mantissa = (b & 0x10000) != 0
    b[finite & round_bit & (sticky | mantissa)] += np.uint32(0x10000)
    nan = ~finite & ((b & 0x7fffff) != 0)
    b[nan] = 0x7fffffff
    expected = (b >> 16).astype(np.uint16)
    numeric.require_bits(numeric.oracle(bits), expected)
    special = np.array([0, 0x80000000, 0x7f800000, 0xff800000, 0x7f800001,
                        0xffc00000, 0x00008000, 0x00008001, 0x3f808000, 0x3f818000], dtype=np.uint32)
    assert numeric.oracle(special).tolist() == [0, 0x8000, 0x7f80, 0xff80, 0x7fff, 0x7fff, 0, 1, 0x3f80, 0x3f82]
    wrong = expected.copy(); wrong[0] ^= np.uint16(1)
    with pytest.raises(AssertionError, match='raw mismatch'):
        numeric.require_bits(wrong, expected)


def test_generator_has_one_authority_and_exact_seven_casts():
    gate.check_source()
    assert sum(gen.CASTS.values()) == 7
    for kind, filename in gen.SOURCES.items():
        source = (ROOT / 'csrc/gdn_chunk' / filename).read_text()
        generated = gen.generate(kind, source)
        assert generated.count('native_bf16::convert(') == gen.CASTS[kind]
        body = gen.definition(generated, f'gdn_wy_native_bf16_{kind}(Inputs')
        assert 'BF16(' not in body
        assert '__HGGCCC_VER_MAJOR__' not in generated
        # Delete one conversion site: a reduced denominator cannot pass.
        selected = gen.definition(source, gen.KERNELS[kind] + '(Inputs')
        corrupted = source.replace(selected, selected.replace('BF16(', 'UNCONVERTED(', 1), 1)
        with pytest.raises(ValueError, match='denominator'):
            gen.generate(kind, corrupted)


def test_cpp_definition_ignores_comment_and_string_braces():
    text = 'int x() { const char* s = "}"; /* } */ // }\n if (true) {} return 1; } trailing'
    assert gen.definition(text, 'int x()').endswith('return 1; }')
    with pytest.raises(ValueError): gen.definition(text + text, 'int x()')


def test_contract_keeps_prefix_and_requires_all_three_changed_kernels():
    names = contract.EXPERIMENTS['native-bf16']['kernels']
    assert names[0] == contract.CORE[0] and len(names) == 4
    rows = [dict(kernel=n, process='1', us=1.) for n in names]
    assert contract.inventory('native-bf16', rows)['status'] == 'ADMITTED'
    for changed in range(1, 4):
        bad = copy.deepcopy(rows); bad[changed]['kernel'] = contract.CORE[changed]
        assert contract.inventory('native-bf16', bad)['status'] == 'UNADJUDICATED'


def test_raw_bit_admission_is_not_replaceable_by_numeric_tolerance():
    # Intentionally otherwise incomplete; the relevant gate must reject
    # missing raw-bit proof after the existing numerical gate is complete.
    candidate = 'native-bf16'
    admission = dict(status='PASS', numeric_repeats=8, unchanged_public_default_bits='PASS',
                     arms={r:{} for r in contract.roles(candidate)}, candidate_kind=candidate)
    with pytest.raises(ValueError, match='raw-bit equality'):
        contract.validate_receipt(admission, {}, candidate)


def test_native_adapter_is_not_a_global_macro_workaround():
    text = (ROOT / 'include/gdn_qsa/ppu/native_bf16.cuh').read_text()
    assert '#define __HGGCCC' not in text
    assert 'ppu.cvt.rtte.bf16.f32' in text and 'bitcast(bits)' in text
    cmake = (ROOT / 'csrc/serving/CMakeLists.txt').read_text()
    assert 'option(ALA_PPU10_NATIVE_BF16_EXPERIMENT "Build a conversion-only PPU10 candidate; retain default" OFF)' in cmake
