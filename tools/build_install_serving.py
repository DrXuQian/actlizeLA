#!/usr/bin/env python3
"""Build and qualify a serving wheel in an isolated, platform-Torch environment.

No performance run, package download, or modification of the system installation.
The existing build and installed qualification tools remain the authorities.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile


def built_wheel(build):
    receipt = json.loads((build / "result.json").read_text())
    path = Path(receipt["wheel"]).resolve(strict=True)
    if (not path.is_relative_to((build / "dist").resolve()) or
            not path.is_file() or path.suffix != ".whl" or receipt["source_dirty"] or
            hashlib.sha256(path.read_bytes()).hexdigest() != receipt["sha256"]):
        raise RuntimeError("wheel identity/location differs from clean build receipt")
    return path, receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sdk", type=Path, required=True)
    p.add_argument("--target", choices=("ppu10", "ppu15"), default="ppu10")
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--flare", type=Path)
    p.add_argument("--host-only", action="store_true")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.out.resolve()
    if args.jobs < 1 or not (args.sdk / "bin/hgcc").is_file():
        p.error("positive jobs and the matching PPU SDK with bin/hgcc are required")
    if args.flare and not args.flare.is_dir():
        p.error("--flare must name the existing framework checkout")
    if out.is_relative_to(root):
        p.error("build artifacts must be outside the source checkout")
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["LD_LIBRARY_PATH"] = str(args.sdk.resolve() / "lib") + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    env["CUDA_VISIBLE_DEVICES"] = env.get("DEVICE", env.get("CUDA_VISIBLE_DEVICES", "0"))
    report = dict(status="FAIL", target=args.target, performance="NOT_RUN", phases={})

    def run(name, argv):
        report["phases"][name] = dict(command=list(map(str, argv)), status="RUNNING")
        print(f"[serving release] {name}: log={out / (name + '.log')}", flush=True)
        with (out / (name + ".log")).open("w") as log:
            rc = subprocess.run(list(map(str, argv)), cwd=out, env=env,
                                stdout=log, stderr=subprocess.STDOUT).returncode
        report["phases"][name].update(returncode=rc, status="PASS" if rc == 0 else "NONZERO")
        if rc:
            print("\n".join((out / (name + ".log")).read_text().splitlines()[-50:]), flush=True)
        return rc

    try:
        build = out / "build"
        if run("build", [sys.executable, root / "tools/build_serving_wheel.py",
                         "--target", args.target, "--sdk", args.sdk.resolve(),
                         "--jobs", args.jobs, "--out", build]):
            return 1
        wheel, receipt = built_wheel(build)
        report["artifact"] = receipt
        if run("environment", [sys.executable, "-m", "venv", "--system-site-packages", out / "env"]):
            return 1
        python = out / "env/bin/python"
        if run("install", [python, "-m", "pip", "install", "--no-index", "--no-deps",
                           "--ignore-installed", wheel]):
            return 1
        cmd = [python, "-I", root / "tools/qualify_serving_wheel.py", "--target", args.target,
               "--out", out / "qualification"]
        if args.flare:
            cmd += ["--flare", args.flare.resolve()]
        if args.host_only:
            cmd += ["--host-only"]
        rc = run("qualification", cmd)
        summary = out / "qualification/summary.json"
        if not summary.is_file():
            raise RuntimeError("qualification did not produce a verdict")
        report["qualification"] = json.loads(summary.read_text())
        identity = report["qualification"].get("identity", {})
        if (identity.get("source_revision") != receipt["source_revision"] or
                identity.get("target") != args.target or
                not Path(identity.get("library_path", "")).is_relative_to(out / "env")):
            raise RuntimeError("qualification used a different source/target or non-private installation")
        report["status"] = report["qualification"]["status"]
        if rc == 0 and report["status"] != "DEVICE_AND_COMPONENT_PASS_NOT_FULL_MODEL_ADMISSION":
            report["status"] = "FAIL"
            raise RuntimeError("qualification exit zero without complete device/component evidence")
        print(f"[serving release] wheel={wheel}\n[serving release] installed_python={python}", flush=True)
        return rc
    except Exception as exc:
        report.update(status="FAIL", error=repr(exc))
        raise
    finally:
        (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        with tarfile.open(out / "results.tar.gz", "w:gz") as archive:
            for path in sorted(out.iterdir()):
                if path.is_file() and path.suffix in (".json", ".log"):
                    archive.add(path, arcname=path.name)
            evidence = out / "qualification/results.tar.gz"
            if evidence.is_file():
                archive.add(evidence, arcname="installed-results.tar.gz")
        print(f"[serving release] {report['status']}; upload={out / 'results.tar.gz'}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
