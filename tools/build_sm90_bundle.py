#!/usr/bin/env python3
"""Build/register all candidates for metadata-only automatic SM90 dispatch."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from actlize_la.sm90 import load_sm90
from actlize_la.sm90_auto import BUILD_IDENTITY_FIELDS, REGISTRATION_FILE, sha
from actlize_la.sm90_policy import policy, policy_digest


def assemble(directory):
    """Publish only a complete, same-source/compiler/dependency candidate set."""
    rows = {}
    common = None
    for name in policy()["configurations"]:
        member = directory / name
        if member.is_symlink() or not member.is_dir():
            raise ValueError(f"{name}: bundle member missing or symlinked")
        op = load_sm90(member)
        if op.configuration != name:
            raise ValueError(f"{name}: actual binary configuration mismatch")
        receipt_file = member / "build.json"
        receipt = json.loads(receipt_file.read_text())
        identity = tuple(receipt.get(k) for k in BUILD_IDENTITY_FIELDS)
        if not all(identity) or (common is not None and identity != common):
            raise ValueError("mixed SM90 compiler/source/dependency/Torch identities")
        common = identity
        rows[name] = dict(directory=name, receipt_sha256=sha(receipt_file))
    manifest = dict(schema=1, target="cuda_sm90", policy_sha256=policy_digest(), configurations=rows)
    temporary = directory / "bundle.pending.json"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(directory / "bundle.json")
    load_sm90(directory)  # Read the actual published contract, not an in-memory model.
    return directory / "bundle.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compiler", type=Path)
    parser.add_argument("--reuse-device", action="store_true")
    parser.add_argument("--assemble-only", action="store_true",
                        help="verify complete existing build directories; no compiler invocation")
    parser.add_argument("--register", action="store_true",
                        help="register this bundle for the installed source frontend")
    args = parser.parse_args()
    directory = args.out.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if not args.assemble_only:
        for name in policy()["configurations"]:
            command = [sys.executable, str(ROOT / "tools/build_gdn_sm90.py"),
                       "--target", "cuda_sm90", "--configuration", name,
                       "--out", str(directory / name)]
            if args.compiler:
                command += ["--compiler", str(args.compiler)]
            if args.reuse_device:
                command += ["--reuse-device"]
            subprocess.run(command, check=True)
    manifest = assemble(directory)
    if args.register:
        record = dict(schema=1, directory=str(directory), bundle_sha256=sha(manifest))
        temporary = REGISTRATION_FILE.with_suffix(".pending.json")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(REGISTRATION_FILE)
    print(f"[actlizeLA SM90 auto] PASS candidates={len(policy()['configurations'])} "
          f"registered={args.register} bundle={directory} device=NOT_RUN")


if __name__ == "__main__":
    main()
