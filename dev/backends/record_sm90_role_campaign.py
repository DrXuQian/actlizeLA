#!/usr/bin/env python3
"""Re-extract D28/S29 evidence; S30 failed native admission, not timed.

Run with the original Python 3.12 and Torch. Raw timestamp intervals remain
diagnostic: they overlap across roles and include the probe's own overhead.
"""
import argparse
import importlib.util
import json
from pathlib import Path
from statistics import median

from record_sm90_aux_campaign import digest, record


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_cases(root, name):
    candidate = json.loads((root / f"{name}-cases/cases.json").read_text())
    parent = json.loads((root / "relative-cases/cases.json").read_text())
    assert candidate["denominator"] == candidate["completed"] == 14
    assert parent["denominator"] == parent["completed"] == 14
    assert len(candidate["cases"]) == len(parent["cases"]) == 14
    for got, want in zip(candidate["cases"], parent["cases"]):
        assert got["case"] == want["case"] and got["rc"] == want["rc"] == 0
        for field in ("input_sha256", "output_sha256", "errors"):
            assert got["result"][field] == want["result"][field], (name, got["case"], field)
        assert max(got["result"]["errors"]) < .02
    return dict(status="14/14 CPU_ORACLE_AND_PARENT_FINGERPRINTS_PASS",
                cases_sha256=digest(root / f"{name}-cases/cases.json"),
                parent_cases_sha256=digest(root / "relative-cases/cases.json"))


def verify_trace(root, name, analyzer):
    import torch

    directory = root / name
    receipt = json.loads((directory / "result.json").read_text())
    assert receipt["status"] == "PASS" and receipt["device_watch"]["errors"] == []
    assert receipt["repeat"] == "8/8 raw-equal-to-S24"
    assert receipt["timestamps_sha256"] == digest(directory / "timestamps.pt")
    timestamps = torch.load(directory / "timestamps.pt", weights_only=True)
    assert len(timestamps) == len(receipt["analyses"]) == 8
    recovered = [analyzer.analyze(row.flatten().tolist(), 32, 32) for row in timestamps]
    assert recovered == receipt["analyses"], "raw timestamps do not reproduce result"
    assert all(row["expected_stamps"] == 35840 for row in recovered)
    role_summary = {}
    for role in ("0", "1", "2"):
        first = recovered[0]["roles"][role]
        intervals = {
            label: median(row["roles"][role]["interval_sum_us"][label] for row in recovered)
            for label in first["interval_sum_us"]
        }
        role_summary[role] = dict(
            median_of_per_call_interval_medians_us=intervals,
            median_of_per_call_span_medians_us=median(row["roles"][role]["span_us"]["median"] for row in recovered),
            median_of_per_call_chunk_cadence_us=median(row["roles"][role]["steady_chunk_cadence_us"] for row in recovered),
        )
    return dict(gate=receipt["gate"], calls=8, stamps_per_call=35840,
                raw_reanalysis="EXACT", roles=role_summary,
                scope="INSTRUMENTED_INCLUSIVE_INTERVALS_NOT_ADDITIVE_ACROSS_ROLES",
                hashes={name: digest(directory / name) for name in ("result.json", "timestamps.pt")})


def verify_stresses(root):
    import torch

    results = []
    for magnitude in (8, 10000):
        paths = [root / f"{name}-stress{magnitude}/captured.pt" for name in ("warp-coords", "relative")]
        candidate, parent = [torch.load(path, weights_only=True) for path in paths]
        assert len(candidate) == len(parent) == 2
        assert all(torch.equal(x.view(torch.uint8), y.view(torch.uint8)) for x, y in zip(candidate, parent))
        receipt = json.loads((paths[0].parent / "result.json").read_text())
        assert max(receipt["errors"]) < .02
        results.append(dict(gate=-magnitude, status="DIRECT_BYTE_O_AND_STATE_PASS",
                            hashes={str(path.relative_to(root)): digest(path) for path in paths}))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--nsys-analyzer", type=Path, required=True)
    parser.add_argument("--role-analyzer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    nsys = load_module(args.nsys_analyzer, "registered_nsys")
    roles = load_module(args.role_analyzer, "registered_roles")
    captures = [record(args.root, name, candidate, nsys) for name, candidate in (
        ("trace-fi-weak", "D28"), ("trace-fi-strong", "D28"), ("warp-coords-fi-weak", "S29"))]
    for capture in captures:
        capture["use"] = "PROBE_PERTURBATION_ONLY" if capture["candidate"] == "D28" else "SPEED_SCREENING"
        if capture["candidate"] == "D28":
            capture["probe_overhead_fraction"] = capture["candidate_us"]["median"] / capture["control_us"]["median"] - 1
    builds = {}
    for name in ("trace-build-r2", "warp-coords-build", "aux-reg-build"):
        directory = args.root / name
        manifest = json.loads((directory / "build.json").read_text())
        binaries = list(directory.glob("_gdn_fused_sm90*.so"))
        assert len(binaries) == 1 and manifest["complete"]
        assert digest(binaries[0]) == manifest["extension_sha256"]
        builds[name] = dict(source=manifest["repository_revision"],
                            binary_sha256=manifest["extension_sha256"],
                            build_sha256=digest(directory / "build.json"),
                            device_log_sha256=digest(directory / "device.log"))
    register_log = (args.root / "aux-reg-build/device.log").read_text()
    assert register_log.count("(C7512)") == 4
    assert "wgmma.mma_async instructions are serialized due to insufficient register resources" in register_log
    result = dict(scope="H800_SM90_NOT_NATIVE_PPU17", captures=len(captures),
                  complete_profiled_forwards=sum(row["complete_forwards"] for row in captures),
                  runs=captures, builds=builds,
                  numerical_cases={name: verify_cases(args.root, name) for name in ("trace", "warp-coords")},
                  direct_stresses=verify_stresses(args.root),
                  diagnostics=[verify_trace(args.root, name, roles) for name in ("trace-weak", "trace-strong")],
                  rejected=dict(candidate="S30", reason="NATIVE_PROTOCOL_GATE_FAIL_C7512_ALL4",
                                device_numerics="NOT_RUN", performance="NOT_RUN"),
                  decision="RETAIN_S24_NO_NEW_PROMOTION_TARGET_NOT_MET")
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("scope", "captures", "complete_profiled_forwards", "decision")}, indent=2))


if __name__ == "__main__":
    main()
