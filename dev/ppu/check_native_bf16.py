#!/usr/bin/env python3
"""Native conversion-only compile gate, never a device/performance verdict."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

from check_wy_binary import kernel_sequences, plant_in_kernel
from generate_native_bf16 import ROOT, SOURCES, KERNELS, definition, generate


def check_source():
    for kind, filename in SOURCES.items():
        old = (ROOT / 'csrc/gdn_chunk' / filename).read_text()
        new = generate(kind, old)
        a = definition(old, KERNELS[kind] + '(Inputs')
        b = definition(new, f'gdn_wy_native_bf16_{kind}(Inputs')
        if (b.replace('native_bf16::convert(', 'BF16(')
                .replace(f'gdn_wy_native_bf16_{kind}(', KERNELS[kind] + '(') != a):
            raise AssertionError('algorithm changed outside the conversion seam')
    adapter = (ROOT / 'include/gdn_qsa/ppu/native_bf16.cuh').read_text()
    if ('ppu.cvt.rtte.bf16.f32' not in adapter or 'bitcast(bits)' not in adapter
            or re.search(r'#\s*define\s+__HG', adapter)):
        raise AssertionError('native scalar RTTE/bitcast contract changed')


def check_native(isa, resources):
    kernels = kernel_sequences(isa)
    def select(marker):
        names = [n for n in kernels if marker in n]
        if len(names) != 1: raise AssertionError('missing/ambiguous exact kernel: ' + marker)
        return names[0], kernels[names[0]]
    result = {}
    for stage, suffix in (('solve', 'ILj128E'), ('solve', 'ILj256E'), ('state', 'E'), ('output', 'E')):
        old_name, old = select(KERNELS[stage] + suffix)
        new_name, new = select(f'gdn_wy_native_bf16_{stage}' + suffix)
        a, b = (Counter(op.split()[0] for op in seq) for seq in (old, new))
        software = sum(bool(re.match(r'v\.and\.b32\s+.*?,\s*0x7f800000$', op)) for op in old)
        conversions = b['v.cnvt.bf16.f32.rtte'] + 2 * b['v.pcnvt.bf16x2']
        if not software or conversions != software or any('0x18000' in op for op in new):
            raise AssertionError('native BF16 must replace every software rounding site')
        fixed = ('v.mma.', 'v.fma.f32', 'v.mul.f32', 'v.add.f32', 'v.exp2.',
                 'vmem.aiu.', 'tsm.ld.', 'tsm.st.', 'vmem.ld.', 'vmem.st.',
                 's.blksyn', 'vmem.fence', 'vmem.acp.')
        if {k:v for k,v in a.items() if k.startswith(fixed)} != {k:v for k,v in b.items() if k.startswith(fixed)}:
            raise AssertionError('math/matrix staging/barrier work changed')
        if len(new) >= len(old) or any('ivreg' in op for op in new):
            raise AssertionError('native conversion did not remove the compatibility footprint')
        regs = []
        for name in (old_name, new_name):
            sections = [s for s in re.split(r'(?=^Func \d+ .*RESOURCE INFO:)', resources, flags=re.M)
                        if s.splitlines() and name in s.splitlines()[0]]
            if len(sections) != 1 or 'STACK SIZE:0' not in sections[0]:
                raise AssertionError('zero-stack resource proof missing')
            regs.append(int(re.search(r'vreg_number:(\d+)', sections[0])[1]))
        result[stage + suffix] = dict(static_instructions=[len(old), len(new)],
                                     software_to_native_sites=[software, conversions], registers=regs,
                                     native_stack=[0, 0], fixed_math_and_staging='PASS')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--isa', type=Path, required=True)
    parser.add_argument('--resources', type=Path, required=True)
    args = parser.parse_args()
    check_source()
    isa, resources = args.isa.read_text(), args.resources.read_text()
    result = check_native(isa, resources)
    for stage, suffix in (('solve', 'ILj128E'), ('solve', 'ILj256E'), ('state', 'E'), ('output', 'E')):
        marker = f'gdn_wy_native_bf16_{stage}' + suffix
        for opcode in ('v.cnvt.bf16.f32.rtte', 'v.mma.f32.bf16.m16n16k16', 'tsm.ld.swzl'):
            bad = plant_in_kernel(isa, marker, opcode, 'REMOVED')
            try: check_native(bad, resources)
            except AssertionError: pass
            else: raise AssertionError('missing-work negative escaped')
    print(json.dumps(dict(stages=result, scope='COMPILE_ONLY; device numerics/performance NOT_RUN',
                          negatives='removed conversion, MMA or SWZL load: EXPECTED_RED'), indent=2))


if __name__ == '__main__': main()
