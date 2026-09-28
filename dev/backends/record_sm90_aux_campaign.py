#!/usr/bin/env python3
"""Re-extract the fixed S20--S24 campaign; no timing thresholds live here.

Use the original Python 3.12 and the hash-bound candidate-aware analyzer.
The profiler receipts/result files retain all samples, identities and gates.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3

RUNS = {
    "kk-first-fi-weak": "S20",
    "final-mask-fi-weak": "S21",
    "final-mask-fi-strong": "S21",
    "final-mask-qla-weak": "S21",
    "final-mask-qla-strong": "S21",
    "inplace-fi-weak-r3": "S22",
    "mask-inplace-fi-weak": "S23",
    "relative-fi-weak": "S24",
    "relative-fi-strong": "S24",
    "relative-qla-weak": "S24",
    "relative-qla-strong": "S24",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(root, name, candidate, analyzer):
    directory = root / name
    receipt = json.loads((directory / "receipt.json").read_text())
    result = json.loads((directory / "result.json").read_text())
    assert receipt["device_watch"]["errors"] == [], name
    assert receipt["candidate_requires_raw_bit"] is True, name
    assert all(row["repeat"] == "8/8 RAW-BIT" for row in receipt["admission"].values())
    assert all(row["calls"] == 12 for row in result["summary"].values())
    assert result["accounting"] == "ALL_CAPTURED_KERNELS_AND_MEMORY_ASSIGNED_EXACTLY_ONCE"
    for path, expected in result["evidence_sha256"].items():
        selected = Path(analyzer.__file__) if Path(path).name == "analyze_sm90_nsys.py" else directory / Path(path).name
        assert digest(selected) == expected, (name, path)
    with sqlite3.connect(f"file:{(directory / 'forward.sqlite').resolve()}?mode=ro", uri=True) as connection:
        recovered = analyzer.extract(connection, receipt)
    # JSON serializes integer histogram keys as strings. Compare the same
    # representation, without changing any float, sample, count or verdict.
    recovered = json.loads(json.dumps(recovered))
    original = {k: v for k, v in result.items() if k != "evidence_sha256"}
    assert recovered == original, (name, "SQLite re-extraction differs")
    summary = result["summary"]
    build = receipt["incumbent_builds"]["ours-candidate"]
    short = lambda value: {k: value[k] for k in ("median", "range")}
    return dict(candidate=candidate, capture=name, gate=result["gate"],
        source=build["repository_revision"], binary_sha256=build["extension_sha256"],
        candidate_us=short(summary["ours-candidate"]["kernel_sum_us"]),
        control_us=short(summary["ours-cuda"]["kernel_sum_us"]),
        references_us={k: short(v["kernel_sum_us"]) for k,v in summary.items() if not k.startswith("ours-")},
        verdict_vs_control=result["comparisons"]["ours-candidate"]["verdict"],
        candidate_vs_references=result["candidate_vs_references"],
        complete_forwards=len(result["forwards"]),
        admission=receipt["admission"]["ours-candidate"],
        record_sha256={p: digest(directory/p) for p in
                       ("receipt.json", "result.json", "forward.sqlite", "forward.nsys-rep")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--analyzer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("registered_analyzer", args.analyzer)
    analyzer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyzer)
    rows = [record(args.root, name, candidate, analyzer) for name,candidate in RUNS.items()]
    result = dict(scope="H800_SM90_NOT_NATIVE_PPU17",
        protocol="nsys_all_kernels_per_forward_12_interleaved_calls_per_role",
        reanalysis="EXACT_REEXTRACTION_ALL_11_CAPTURES", runs=rows,
        captures=len(rows), complete_forwards=sum(x["complete_forwards"] for x in rows),
        excluded=["inplace-fi-weak: foreign PID175958 before capture",
                  "inplace-fi-weak-r2: foreign PID176303 before capture"],
        invalid_build="relative-local: source changed during compile; relative-local-r2 replaces it")
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
