#!/usr/bin/env python3
"""Bind local and measured CUDA images by four exact instruction streams.

cuobjdump --dump-sass input; not a dynamic instruction-count or speed gate.
Predicates, operand order and every instruction remain in the comparison.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


def kernels(path):
    found = {}
    for section in path.read_text().split("Function :")[1:]:
        symbol = section.splitlines()[0].strip()
        if "FlatKernelTmaWarpSpecializedKdaFwd" not in symbol:
            continue
        assert symbol not in found, "duplicate specialization"
        rows = re.findall(r"/\*[0-9a-f]+\*/\s*(.*?)\s*;\s*/\*", section)
        assert rows, "empty specialization"
        found[symbol] = rows
    assert len(found) == 4, "four exact device bodies required"
    return found


def compare(local, measured):
    assert len(local) == len(measured) == 4
    assert local.keys() == measured.keys(), "specialization denominator changed"
    assert local == measured, "actual instruction stream or operands changed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("local", type=Path)
    parser.add_argument("measured", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    local, measured = kernels(args.local), kernels(args.measured)
    compare(local, measured)
    symbol = next(iter(measured))
    plants = {
        "missing-specialization": {k: v for k, v in measured.items() if k != symbol},
        "changed-operand": measured | {symbol: [measured[symbol][0] + " CORRUPT", *measured[symbol][1:]]},
        "reordered": measured | {symbol: list(reversed(measured[symbol]))},
    }
    for name, plant in plants.items():
        try:
            compare(local, plant)
        except AssertionError:
            continue
        raise AssertionError(f"negative escaped: {name}")
    result = dict(status="PASS", scope="STATIC_NATIVE_IDENTITY_NOT_PERFORMANCE",
        bodies=len(local), instruction_sites=sum(map(len, local.values())),
        normalized_sha256=hashlib.sha256(json.dumps(local, sort_keys=True).encode()).hexdigest(),
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.local, args.measured)},
        negatives={name: "EXPECTED_RED" for name in plants})
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "source_sha256"}))


if __name__ == "__main__":
    main()
