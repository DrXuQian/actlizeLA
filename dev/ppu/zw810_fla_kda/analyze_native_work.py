#!/usr/bin/env python3
"""Price native executed work from a validated offline KDA export; no GPU use.

BF16 attribution requires the full finite-rounding signature, not an opcode
guess. Counts are per-PC warp executions, never the aggregate PU metric.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'tools'))
from export_gdn_kda_details import compare_reimport, parse_raw, rule_counts, select_reports
from gdn_kda_contract import instruction_rows
from analyze_work import work_phase


BF16_FINITE = Counter({
    'v.and.b32': 3, 'v.cmp.ne.i32': 2, 's.lop.emsk': 2, 's.xor.b32': 2,
    'v.cmp.gt.i16': 1, 'v.cmp.eq.i32': 1, 's.or.b32': 1, 'v.add.i32': 1,
    's.and.b32': 1, 'v.csel.b32': 1,
})


def software_bf16(rows):
    """Conservative 15-instruction core, excluding shifts/stores/waits/literals.

The compiler interleaves other work. Count only signature instructions and
leave every other instruction in its original class. Exact core multiset,
rounding operands and execution counts are checked; a changed core fails.
    """
    sites, selected = [], set()
    for begin, row in enumerate(rows):
        source = row['Source']
        if not re.fullmatch(r'v\.and\.b32 vreg\d+, vreg\d+(?:\.reuse)?, 0x7f800000', source):
            continue
        end = next((i for i in range(begin, min(begin + 24, len(rows)))
                    if rows[i]['Source'].startswith('s.xor.b32 emsk,')), None)
        if end is None:
            raise ValueError('software BF16 signature lacks finite branch close')
        body = []
        for i in range(begin, end + 1):
            code = rows[i]['Source']
            op = code.split()[0]
            if op not in BF16_FINITE:
                continue
            body.append(i)
        if Counter(rows[i]['Source'].split()[0] for i in body) != BF16_FINITE:
            raise ValueError(f'software BF16 opcode signature changed at {row["Address"]}: '
                             + repr([rows[i]['Source'] for i in body]))
        text = '\n'.join(rows[i]['Source'] for i in body)
        required = (
            r'v\.and\.b32 .*?, 0x7fff$', r'v\.and\.b32 .*?, 0x18000$',
            r'v\.cmp\.gt\.i16 .*?, 0xffff$', r'v\.cmp\.ne\.i32 .*?, 0x18000$',
            r'v\.add\.i32 .*?, 0x10000$', r's\.lop\.emsk .*? # 0x8$',
            r's\.lop\.emsk .*? # 0xe$',
        )
        if any(not re.search(p, text, re.M) for p in required):
            raise ValueError('software BF16 bit/rounding operands changed')
        count = int(row['Instructions Executed'])
        if any(int(rows[i]['Instructions Executed']) != count for i in body):
            raise ValueError('BF16 finite-path execution count changed; price lanes separately')
        if selected.intersection(body):
            raise ValueError('overlapping software BF16 attribution')
        selected.update(body)
        sites.append(dict(first_pc=row['Address'], core_pcs=[rows[i]['Address'] for i in body],
                          conversions_warp=count, core_instructions=count * 15))
    return sites, selected


def category(op):
    if op.startswith('v.mma.'): return 'matrix'
    if 'cnvt' in op: return 'native_conversion'
    if op.startswith(('v.mov.', 's.mov.')): return 'move'
    if op.startswith(('tsm.', 'vmem.', 'smem.')):
        if any(x in op for x in ('fence', 'commit')): return 'synchronization'
        return 'memory'
    if op.startswith(('s.wait', 's.blksyn', 's.nop', 's.cbr', 's.exit', 's.lop.emsk')):
        return 'wait_control'
    if any(x in op for x in ('.f32', '.f16', '.bf16', '.tf32')): return 'scalar_float'
    return 'integer_mask_address_other'


def analyze(directory):
    summary = json.loads((directory / 'summary.json').read_text())
    captured_bytes = (directory / 'captured-summary.json').read_bytes()
    if (summary['status'] != 'OFFLINE_EXPORTED' or summary['errors'] or
            hashlib.sha256(captured_bytes).hexdigest() != summary['source_summary_sha256']):
        raise ValueError('incomplete or unbound native export')
    wanted = {(r['cell'], r['role'], r['repeat']): r for r in select_reports(json.loads(captured_bytes))}
    got = {(r['cell'], r['role'], r['repeat']): r for r in summary['exports']}
    if len(got) != 12 or len(summary['exports']) != 12 or got.keys() != wanted.keys():
        raise ValueError('changed diagnostic denominator')
    records = []
    for key, entry in got.items():
        if entry['report_sha256'] != wanted[key]['capture']['report_sha256']:
            raise ValueError('report binding changed')
        for name, digest in entry['files'].items():
            path = directory / name
            if not path.resolve().is_relative_to(directory.resolve()):
                raise ValueError('file escapes evidence directory')
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f'evidence hash changed: {name}')
        stem = directory / key[0] / f'{key[1]}-{key[2]}'
        old, new = (Path(str(stem) + suffix).read_text() for suffix in ('-captured.csv', '-raw.csv'))
        metric = summary['instruction_metric_id']
        comparison = compare_reimport(old, new, metric)
        if comparison['status'] != 'MATCH_AT_CAPTURE_PRECISION':
            raise ValueError('captured aggregate metadata changed')
        kernels = parse_raw(new)
        counts = instruction_rows(old, parse_raw(old), metric)
        native = Path(str(stem) + '-native.csv')
        actions = rule_counts(native, kernels, counts, metric)
        if actions != entry['actions']:
            raise ValueError('export summary differs from all native PC rows')
        groups = defaultdict(list)
        with native.open() as f:
            for row in csv.DictReader(f): groups[row['Range Id']].append(row)
        t, hv = map(int, re.fullmatch(r'T(\d+)-Hk16-Hv(\d+)-g-0\.1', key[0]).groups())
        chunk_heads = t // 64 * hv
        for kernel, count in zip(kernels, counts):
            rows = groups[kernel['id']]
            sites, selected = software_bf16(rows)
            classes = Counter()
            for i, row in enumerate(rows):
                label = 'software_bf16_core' if i in selected else category(row['Source'].split()[0])
                classes[label] += int(row['Instructions Executed'])
            action = actions[kernel['id']]
            if sum(classes.values()) != action['pc_warp_executions']:
                raise ValueError('instruction classification lost work')
            conv = sum(s['conversions_warp'] for s in sites)
            # Independent logical-cast denominator; not inferred from marker count.
            phase = work_phase(key[1], kernel['kernel'])
            if key[1] != 'fla':
                per_group = dict(prefix=0, solve=4096, state_equivalent=40960, output=12288, fills=0)[phase]
                if conv * 32 != per_group * chunk_heads:
                    raise ValueError('BF16 conversions differ from logical publication denominator')
            record = dict(cell=key[0], role=key[1], id=kernel['id'], phase=phase,
                          kernel=kernel['kernel'], us=kernel['us'], grid=kernel['grid'], block=kernel['block'],
                          chunk_heads=chunk_heads, aggregate=int(count['count']),
                          native=action['pc_warp_executions'], static_pcs=action['static_pcs'],
                          categories=dict(classes), opcodes=action['opcode_warp_counts'],
                          software_bf16_conversions_warp=conv, software_bf16_sites=sites,
                          source_lines_available=any(r['source_file_name'] not in ('', '<invalid>') for r in rows))
            records.append(record)
    return dict(scope='four geometries x three roles, g=-0.1, repeat0; native PC work, not new timing admission',
                source_sha=summary['captured_source_sha'], source_summary_sha256=summary['source_summary_sha256'],
                native_metric='Instructions Executed per native PC (warp)', aggregate_metric=summary['instruction_metric_id'],
                records=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--summary-out', type=Path)
    args = parser.parse_args()
    result = analyze(args.directory)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    if args.summary_out:
        compact = {k: v for k, v in result.items() if k != 'records'}
        compact['full_analysis_sha256'] = hashlib.sha256(args.out.read_bytes()).hexdigest()
        compact['records'] = [
            {**{k: v for k, v in r.items() if k not in ('opcodes', 'software_bf16_sites')},
             'mma': {k: v for k, v in r['opcodes'].items() if k.startswith('v.mma.')}}
            for r in result['records']]
        args.summary_out.write_text(json.dumps(compact, indent=2) + '\n')
    for r in result['records']:
        if r['cell'] == 'T2048-Hk16-Hv32-g-0.1':
            print(r['role'], r['phase'], 'native=', r['native'],
                  'software_bf16_core=', r['categories'].get('software_bf16_core', 0))


if __name__ == '__main__': main()
