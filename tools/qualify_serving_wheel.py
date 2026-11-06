#!/usr/bin/env python3
"""Installed wheel -> host contracts -> PPU/SM120 graph -> actual Flare gates.

No compiling, dependency installation or production-source PYTHONPATH. The
checkout supplies tests/oracles only. Reports distinguish PASS, NOT_RUN and FAIL.
"""

import argparse
import hashlib
import json
import os
import runpy
import shutil
import subprocess
import sys
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

_inventory = runpy.run_path(str(Path(__file__).with_name("serving_qualification.py")))
FLARE_TESTS, HOST_TESTS, CHECK_FILES = (_inventory[n] for n in ("FLARE_TESTS", "HOST_TESTS", "CHECK_FILES"))
public_default_complete = _inventory["public_default_complete"]


def junit_verdict(path, returncode, required):
    if not path.is_file():
        return dict(
            status="FAIL",
            reason="pytest did not produce its execution report",
            returncode=returncode,
        )
    cases = ET.parse(path).findall(".//testcase")
    skipped = sum(c.find("skipped") is not None for c in cases)
    failed = sum(
        c.find("failure") is not None or c.find("error") is not None for c in cases
    )
    passed = len(cases) - skipped - failed
    covered = {
        name
        for name in required
        for c in cases
        if name in c.get("classname", "").split(".")
        and all(c.find(x) is None for x in ("skipped", "failure", "error"))
    }
    status = (
        "FAIL"
        if failed or returncode not in (0, 5)
        else (
            "NOT_RUN" if skipped or not passed or covered != set(required) else "PASS"
        )
    )
    return dict(
        status=status,
        passed=passed,
        skipped=skipped,
        failed=failed,
        collected=len(cases),
        required_files=list(required),
        covered_files=sorted(covered),
        returncode=returncode,
        reasons=[
            c.find("skipped").get("message", "skipped")
            for c in cases
            if c.find("skipped") is not None
        ],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", required=True, choices=("ppu10", "ppu15", "cuda_sm120"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--flare",
        type=Path,
        help="existing Flare feat/qwen38-gguf tree, used as the test subject",
    )
    p.add_argument(
        "--blas",
        type=Path,
        default=Path(
            "/usr/local/PPU_SDK/CUDA_SDK/targets/x86_64-linux/lib/libcublas.so"
        ),
    )
    p.add_argument(
        "--host-only",
        action="store_true",
        help="explicitly leave all device/Flare gates NOT_RUN",
    )
    args = p.parse_args()
    if args.flare:
        args.flare = args.flare.resolve()
    args.blas = args.blas.resolve()
    assert sys.flags.isolated, "use python -I"
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    checks = out / "checks"
    checks.mkdir()
    for name in CHECK_FILES:
        shutil.copy2(root / "tests" / name, checks / name)
    for name in ("verify_installed_serving.py", "installed_serving_checks.py", "serving_qualification.py"):
        shutil.copy2(root / "tools" / name, checks / name)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    report = dict(
        status="FAIL",
        phases={},
        performance="DEFERRED_BY_USER_NOT_RUN",
        device_scope="OPERATOR_AND_COMPONENT_NOT_FULL_MODEL_RUST_REPLAY",
        tests_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in checks.iterdir()
        },
    )

    def run(label, cmd):
        with (out / f"{label}.log").open("w") as log:
            rc = subprocess.run(
                list(map(str, cmd)),
                cwd=out,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            ).returncode
        print(
            f"[installed serving] {label}: rc={rc}; log={out / (label + '.log')}",
            flush=True,
        )
        return rc

    def phase(name):
        output = out / (name + (".xml" if name in ("host", "flare") else ".json"))
        cmd = [
            sys.executable,
            "-I",
            checks / "installed_serving_checks.py",
            "--phase",
            name,
            "--target",
            args.target,
            "--tests",
            checks,
            "--output",
            output,
        ]
        if name == "flare":
            cmd += ["--flare", args.flare, "--blas", args.blas]
        rc = run(name, cmd)
        if name in ("host", "flare"):
            verdict = junit_verdict(
                output,
                rc,
                FLARE_TESTS
                if name == "flare"
                else HOST_TESTS,
            )
        elif output.is_file():
            data = json.loads(output.read_text())
            count = len(data.get("cases", []))
            verdict = dict(
                status="PASS"
                if rc == 0 and data.get("status") == "PASS" and count
                else ("NOT_RUN" if rc == 3 or data.get("status") == "SKIP" else "FAIL"),
                cases=count,
                returncode=rc,
                reason=data.get("reason"),
            )
            if name == "public_default" and verdict["status"] == "PASS" and not public_default_complete(data):
                verdict.update(status="FAIL", reason="incomplete installed public/prepared default denominator")
        else:
            verdict = dict(
                status="FAIL", reason="missing device execution evidence", returncode=rc
            )
        report["phases"][name] = verdict
        return verdict["status"]

    try:
        import actlize_la

        report["identity"] = actlize_la.get_library_info(args.target)
        rc = run(
            "installation",
            [
                sys.executable,
                "-I",
                checks / "verify_installed_serving.py",
                "--target",
                args.target,
                "--out",
                out / "installation",
            ],
        )
        report["phases"]["installation"] = dict(status="PASS" if rc == 0 else "FAIL")
        if rc or phase("host") != "PASS":
            return 1
        import torch

        report.update(torch=torch.__version__, python=sys.version)
        if args.host_only or not torch.cuda.is_available():
            for name in ("serving", "aux", "public_default", "native_graph", "flare"):
                report["phases"][name] = dict(
                    status="NOT_RUN", reason="host-only requested or no device"
                )
            report["status"] = "HOST_PASS_DEVICE_NOT_RUN"
            return 3
        from actlize_la.serving.device import current_device_identity

        report["device"] = current_device_identity(args.target)
        if phase("serving") != "PASS" or phase("aux") != "PASS":
            return 1
        if args.target == "ppu10":
            if phase("public_default") != "PASS":
                return 1
        else:
            report["phases"]["public_default"] = dict(
                status="NOT_APPLICABLE", reason="retained allocating wheel bridge is PPU1.0-only")
        rc = run("native_graph", [report["identity"]["native_graph_example"]])
        graph_log = (out / "native_graph.log").read_text()
        graph_ok = rc == 0 and "[FlashNext native] PASS replays=8 fixed-addresses kernel_nodes=1" in graph_log
        report["phases"]["native_graph"] = dict(
            status="PASS" if graph_ok else "FAIL", returncode=rc
        )
        if not graph_ok:
            return 1
        if not args.flare:
            report["phases"]["flare"] = dict(
                status="NOT_RUN", reason="Flare component tree not supplied"
            )
            report["status"] = "OPERATOR_PASS_FLARE_NOT_RUN"
            return 3
        if phase("flare") != "PASS":
            report["status"] = "NOT_ADMITTED"
            return 3 if report["phases"]["flare"]["status"] == "NOT_RUN" else 1
        report["status"] = "DEVICE_AND_COMPONENT_PASS_NOT_FULL_MODEL_ADMISSION"
        return 0
    except Exception as exc:
        report["error"] = repr(exc)
        raise
    finally:
        (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        # Logs/receipts only. No model, wheel, DSO, copied tests or corruption copies.
        with tarfile.open(out / "results.tar.gz", "w:gz") as archive:
            for path in sorted(out.iterdir()):
                if path.is_file() and path.suffix in (".json", ".xml", ".log"):
                    archive.add(path, arcname=path.name)
            if (out / "installation/verification.json").exists():
                archive.add(
                    out / "installation/verification.json", arcname="installation.json"
                )
        print(
            f"[installed serving] {report['status']}; upload={out / 'results.tar.gz'}",
            flush=True,
        )


if __name__ == "__main__":
    sys.exit(main())
