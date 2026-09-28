#!/usr/bin/env python3
"""Re-extract the bounded S31/S32/S33 inventory without new timing thresholds.

Run in the original Python 3.12/Torch environment. Raw data, not copied console
summaries, bind all kernels, case denominators, actual binaries and verdicts.
"""
import argparse
import json
from pathlib import Path

from record_sm90_aux_campaign import digest, record
from record_sm90_role_campaign import load_module, verify_cases

EXPERIMENTS = (("normalized", "S31"), ("half2", "S32"), ("single-alpha", "S33"))


def stress(root, name):
    import torch
    result = []
    for gate in (8, 10000):
        paths = [root / f"{n}-stress{gate}/captured.pt" for n in (name, "relative")]
        got, want = [torch.load(path, weights_only=True, map_location="cpu") for path in paths]
        assert len(got) == len(want) == 2
        assert all(torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8))
                   for x, y in zip(got, want)), (name, gate)
        receipt = json.loads((paths[0].parent / "result.json").read_text())
        assert max(receipt["errors"]) < .02
        result.append(dict(gate=-gate, status="O_AND_STATE_DIRECT_BYTE_EQUAL",
                           hashes={str(p.relative_to(root)): digest(p) for p in paths}))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--analyzer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    analyzer = load_module(args.analyzer, "registered_inverse_nsys")
    captures = [record(args.root, f"{name}-fi-weak", label, analyzer)
                for name, label in EXPERIMENTS]
    checks = {}
    for name, label in EXPERIMENTS:
        directory = args.root / f"{name}-build"
        build = json.loads((directory / "build.json").read_text())
        binaries = list(directory.glob("_gdn_fused_sm90*.so"))
        assert len(binaries) == 1 and build["complete"]
        assert digest(binaries[0]) == build["extension_sha256"]
        assert build["target"] == "cuda_sm90" and build["mode"] == "native"
        codegen = (directory / "device.log").read_text()
        assert "C7512" not in codegen and "C7510" not in codegen
        tests = args.root / (f"{name}-host-tests.log" if name == "half2"
                             else f"{name}-host-tests-r2.log")
        text = tests.read_text()
        expected = 39 if name == "single-alpha" else 38
        assert f"Ran {expected} tests" in text and "\nOK\n" in text
        assert "FAILED" not in text
        checks[label] = dict(
            source=build["repository_revision"], binary_sha256=build["extension_sha256"],
            build_sha256=digest(directory / "build.json"), device_log_sha256=digest(directory / "device.log"),
            host_tests=dict(count=expected, status="PASS", log_sha256=digest(tests)),
            numerical_cases=verify_cases(args.root, name), direct_stresses=stress(args.root, name))
    assert [r["verdict_vs_control"] for r in captures] == ["CONTROL-WINS", "UNRESOLVED", "UNRESOLVED"]
    result = dict(scope="H800_SM90_NOT_NATIVE_PPU17", captures=len(captures),
                  complete_profiled_forwards=sum(r["complete_forwards"] for r in captures),
                  runs=captures, checks=checks,
                  decision="RETAIN_S24_NO_NEW_PROMOTION_TARGET_NOT_MET",
                  confirmations="strong_gate_and_QLA_NOT_RUN_no_candidate_won_screening")
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("scope", "captures", "complete_profiled_forwards", "decision")}, indent=2))


if __name__ == "__main__":
    main()
