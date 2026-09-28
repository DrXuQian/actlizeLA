#!/usr/bin/env python3
"""Re-extract S34/S35 source-alignment evidence with the existing nsys rule.

Run using the original Python/Torch environment. This is not a configuration
sweep: two coupled source profiles cannot establish optimal stage values.
"""
import argparse
import json
from pathlib import Path

from record_sm90_aux_campaign import digest, record
from record_sm90_inverse_campaign import stress
from record_sm90_role_campaign import load_module, verify_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--analyzer', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    analyzer = load_module(args.analyzer, 'registered_alignment_nsys')
    runs, checks = [], {}
    for name, label, count in [('fi-pipeline', 'S34', 41), ('fi-state', 'S35', 38)]:
        captures = [record(args.root, name+'-fi-weak', label, analyzer)]
        for suffix in ('fi-strong', 'qla-weak', 'qla-strong'):
            directory = args.root / f'{name}-{suffix}'
            if directory.exists():
                assert captures[0]['verdict_vs_control'] == 'CANDIDATE-WINS'
                captures.append(record(args.root, directory.name, label, analyzer))
        runs.extend(captures)
        directory = args.root / (name+'-build')
        manifest = json.loads((directory / 'build.json').read_text())
        binaries = list(directory.glob('_gdn_fused_sm90*.so'))
        assert len(binaries) == 1 and manifest['complete']
        assert manifest['target'] == 'cuda_sm90' and manifest['mode'] == 'native'
        assert digest(binaries[0]) == manifest['extension_sha256']
        native_log = (directory / 'device.log').read_text()
        assert 'C7512' not in native_log and 'C7510' not in native_log
        tests = args.root / (name+'-host-tests.log')
        test_log = tests.read_text()
        assert f'Ran {count} tests' in test_log and '\nOK\n' in test_log
        assert 'FAILED' not in test_log
        assert all(r['binary_sha256'] == manifest['extension_sha256'] for r in captures)
        checks[label] = dict(
            source=manifest['repository_revision'], binary_sha256=digest(binaries[0]),
            build_sha256=digest(directory/'build.json'), device_log_sha256=digest(directory/'device.log'),
            host_tests=dict(count=count, status='PASS', sha256=digest(tests)),
            numerical_cases=verify_cases(args.root, name), direct_stresses=stress(args.root, name))
    result = dict(scope='H800_SM90_NOT_NATIVE_PPU17', captures=len(runs),
                  complete_profiled_forwards=sum(r['complete_forwards'] for r in runs),
                  reanalysis='EXACT_SQLITE_REEXTRACTION', runs=runs, checks=checks,
                  config_sweep='NOT_DONE_TWO_SOURCE_PROFILES_ONLY',
                  routing='UNCHANGED', precision='UNCHANGED_PARENT_RAW_ADMISSION')
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('scope', 'captures', 'complete_profiled_forwards', 'config_sweep')}))


if __name__ == '__main__':
    main()
