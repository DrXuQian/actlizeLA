#!/usr/bin/env python3
"""Re-extract the closed S25/S26/S27 inventory with the original analyzer."""
import argparse
import importlib.util
import json
from pathlib import Path

from record_sm90_aux_campaign import record


RUNS = {
    "range-fi-weak": "S25",
    "staged-fi-weak": "S26",
    "range-flat-fi-weak": "S27",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--analyzer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("registered_analyzer", args.analyzer)
    analyzer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyzer)
    rows = [record(args.root, name, candidate, analyzer) for name, candidate in RUNS.items()]
    assert all(row["gate"] == -0.1 for row in rows)
    assert all(row["complete_forwards"] == 72 for row in rows)
    result = dict(
        scope="H800_SM90_NOT_NATIVE_PPU17",
        protocol="nsys_all_kernels_per_forward_12_interleaved_calls_per_role",
        reanalysis="EXACT_REEXTRACTION_ALL_3_CAPTURES",
        runs=rows, captures=len(rows), complete_forwards=216,
        incumbent="S24 bc3c154: not replaced by any candidate",
        missing="No strong-gate or QLA timing: all three failed weak parent promotion",
        invalid_build="range-local/range-build: __exp2f resolves host-only; only r2 is measured",
        native_ppu17="SKIP: native SDK/model unavailable; CUDA source-check is separate",
    )
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
