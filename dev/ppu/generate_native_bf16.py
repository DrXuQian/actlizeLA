#!/usr/bin/env python3
"""Generate an opt-in conversion A/B from the retained algorithm authority.

No second hand-maintained kernel body, macro-renaming of a type/dependency, or
change to the default. Extract complete C++ definitions and replace only the
seven named scalar conversion sites plus candidate symbols/namespaces.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    'solve': 'gdn_wy_solve_static_ppu.cu',
    'state': 'gdn_wy_residual_full_chunk_ppu.cu',
    'output': 'gdn_wy_residual_warps8_hvlayout_ppu.cu',
}
KERNELS = {
    'solve': 'gdn_wy_split_solve_static',
    'state': 'gdn_wy_residual_full_chunk_state',
    'output': 'gdn_wy_residual_warps8_hvlayout_output',
}
CASTS = dict(solve=1, state=4, output=2)


def definition(text, marker):
    """Balanced definition, ignoring braces in comments and C++ literals."""
    matches = list(re.finditer(re.escape(marker), text))
    if len(matches) != 1:
        raise ValueError('missing/ambiguous definition: ' + marker)
    start = matches[0].start()
    opening = text.index('{', start)
    depth = 0
    tokens = re.finditer(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[{}]', text[opening:])
    for match in tokens:
        token = match.group()
        if token == '{': depth += 1
        elif token == '}':
            depth -= 1
            if not depth: return text[start:opening + match.end()]
    raise ValueError('unclosed definition: ' + marker)


def generate(kind, text):
    if kind not in KERNELS: raise ValueError('unknown native BF16 stage')
    old, new = KERNELS[kind], f'gdn_wy_native_bf16_{kind}'
    body = definition(text, old + '(Inputs')
    if body.count('BF16(') != CASTS[kind]:
        raise ValueError('scalar conversion denominator changed: ' + kind)
    candidate = body.replace('BF16(', 'native_bf16::convert(').replace(old + '(', new + '(')
    # Prove the reversible source seam before the native codegen gate.
    if candidate.replace('native_bf16::convert(', 'BF16(').replace(new + '(', old + '(') != body:
        raise ValueError('candidate changed outside conversion seam')
    headers = ('#include "gdn_wy_common.cuh"\n'
               '#include "gdn_qsa/ppu/native_bf16.cuh"\n'
               '#include "gdn_qsa/ppu/wy_aiu.cuh"\n'
               '#include "gdn_qsa/ppu/wy_full_chunk.hpp"\n')
    if kind == 'solve':
        headers += '#include "gdn_qsa/ppu/wy_solve_static.cuh"\n#include "gdn_qsa/ppu/wy_split_prepare.cuh"\n'
        preamble = ('using Key = aiu::Tile<Chunk, Dim>;\nusing split_prepare::Plan;\n'
                    + definition(text, 'struct SolveStorage') + ';\n'
                    'static_assert(sizeof(SolveStorage) == 49664);\n'
                    'template <unsigned Threads = Plan::SolveThreads>\n'
                    '__global__ void __launch_bounds__(Threads)\n')
        helpers = '\n'.join(definition(text, m) for m in ('int configure()', 'int launch_inverse(Inputs', 'int launch_inverse_default(Inputs'))
    elif kind == 'state':
        headers += '#include "gdn_wy_state_copy.cuh"\n#include "gdn_qsa/ppu/wy_gate_coefficients.cuh"\n'
        preamble = ('using namespace residual_warps8_hvlayout;\n'
                    'using residual_warps8_hvlayout::StateTile;\n'
                    '__global__ void __launch_bounds__(Plan::Threads)\n')
        helpers = '\n'.join(definition(text, m) for m in ('int configure_state()', 'int launch_state(Inputs'))
    else:
        headers += '#include "gdn_qsa/ppu/wy_residual_warps8_hvlayout.cuh"\n'
        preamble = ('using namespace residual_warps8_hvlayout;\n'
                    '__global__ void __launch_bounds__(OutputTile::Threads)\n')
        helpers = '\n'.join(definition(text, m) for m in ('int configure_hvlayout_output()', 'int launch_hvlayout_output(Inputs'))
    helpers = helpers.replace(old, new)
    return ('// Generated conversion-only candidate. Edit the retained authority, not this file.\n' + headers
            + f'namespace gdn_qsa::wy::native_bf16_{kind} {{\n' + preamble + candidate + '\n'
            + helpers + '\n}\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for kind, filename in SOURCES.items():
        authority = ROOT / 'csrc/gdn_chunk' / filename
        source = authority.read_text()
        result = generate(kind, source)
        output = args.out / f'native_bf16_{kind}.cu'
        output.write_text(result)
        manifest[kind] = dict(authority=str(authority.relative_to(ROOT)),
                              authority_sha256=hashlib.sha256(source.encode()).hexdigest(),
                              generated_sha256=hashlib.sha256(result.encode()).hexdigest(),
                              scalar_conversion_sites=CASTS[kind])
    (args.out / 'sources.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__': main()
