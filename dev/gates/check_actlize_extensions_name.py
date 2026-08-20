#!/usr/bin/env python3
"""Fail closed when the retired extension include root reappears."""

from __future__ import annotations

import argparse
from pathlib import Path


LEGACY_ROOT = ("quactlize_" + "extensions").encode()
CANONICAL_ROOT = Path("include/actlize_extensions")
CANONICAL_HEADERS = (
    "cutlass/linear_attention/ppu_chunked_gdn_collective.cuh",
    "cutlass/linear_attention/ppu_chunked_gdn_kernel.cuh",
    "cutlass/linear_attention/ppu_chunked_gdn_resident_mma.cuh",
    "cutlass/linear_attention/ppu_chunked_gdn_types.hpp",
)
EXCLUDED_PARTS = frozenset(
    {".git", ".cache", "__pycache__", "build", "third_party"}
)


def scan(root: Path) -> list[Path]:
    violations: list[Path] = []
    for path in root.rglob("*"):
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if not path.is_file() or path.is_symlink():
            continue
        try:
            payload = path.read_bytes()
        except OSError:
            continue
        if LEGACY_ROOT in payload:
            violations.append(relative)
    return sorted(violations)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--require-canonical", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    failures: list[str] = []
    if not root.is_dir():
        failures.append(f"scan root is not a directory: {root}")

    if args.require_canonical and root.is_dir():
        canonical = root / CANONICAL_ROOT
        for relative in CANONICAL_HEADERS:
            if not (canonical / relative).is_file():
                failures.append(f"canonical header is missing: {CANONICAL_ROOT / relative}")
        retired = root / "include" / LEGACY_ROOT.decode()
        if retired.exists():
            failures.append(f"retired include root still exists: {retired.relative_to(root)}")

    violations = scan(root) if root.is_dir() else []
    failures.extend(
        f"legacy include-root token remains in {path}" for path in violations
    )

    if failures:
        for failure in failures:
            print(f"[actlize-extensions naming] FAIL: {failure}")
        return 1

    scope = "canonical+stale-scan" if args.require_canonical else "stale-scan"
    print(f"[actlize-extensions naming] PASS: scope={scope} root={root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
