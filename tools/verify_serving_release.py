#!/usr/bin/env python3
"""Verify prebuilt release assets against the checked-out source tag, off-device.

The publisher never builds or executes a kernel and never upgrades NOT_RUN to
PASS. Assets live on a separate LFS branch, not in the product source tree.
"""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import zipfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_assets(directory, tag, source_sha):
    path = directory / "release-manifest.json"
    if path.is_symlink():
        raise ValueError("symlink release manifest")
    data = json.loads(path.read_text())
    if (not re.fullmatch(r"v\d+\.\d+\.\d+", tag) or data.get("schema") != 1 or
            data.get("tag") != tag or data.get("source_revision") != source_sha or
            data.get("version") != tag[1:] or data.get("target") != "ppu10"):
        raise ValueError("release tag/source/version/target mismatch")
    if data.get("validation") != "HOST_QUALIFIED_DEVICE_NOT_RUN":
        raise ValueError("this publisher must preserve outstanding device qualification")
    assets = data.get("assets", {})
    if not isinstance(assets, dict) or set(assets) != {
            data.get("wheel"), "qualification-kit.tar.gz", "qualification.json", "host-evidence.tar.gz"}:
        raise ValueError("missing or extra release asset")
    for name, digest in assets.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError("invalid asset filename")
        item = directory / name
        if item.is_symlink() or not item.is_file() or sha(item) != digest:
            raise ValueError(f"missing/changed/LFS-unresolved asset: {name}")
    qualification = json.loads((directory / "qualification.json").read_text())
    if (qualification.get("status") != data["validation"] or
            qualification.get("source_revision") != source_sha or
            qualification.get("wheel_sha256") != assets[data["wheel"]]):
        raise ValueError("qualification is not bound to this source and wheel")
    return data


def unpack_wheel(wheel, destination):
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(wheel) as archive:
        names = set()
        for item in archive.infolist():
            path = PurePosixPath(item.filename)
            if (item.filename in names or path.is_absolute() or ".." in path.parts or
                    "\\" in item.filename or stat.S_ISLNK(item.external_attr >> 16)):
                raise ValueError("unsafe or duplicate wheel member")
            names.add(item.filename)
        archive.extractall(destination)
    packages = [p for p in destination.rglob("actlize_la")
                if p.is_dir() and (p / "__init__.py").is_file()]
    if len(packages) != 1:
        raise ValueError("wheel must contain exactly one package")
    return packages[0]


def verify(directory, source, scratch, tag, source_sha):
    data = checked_assets(directory, tag, source_sha)
    package = unpack_wheel(directory / data["wheel"], scratch)
    sys.path.insert(0, str(source))
    from actlize_la._library import _inspect
    info = _inspect(package, "ppu10")  # File-only, no Torch/DSO/device initialization.
    if (info["source_revision"] != source_sha or info["source_dirty"] or
            info["version"] != tag[1:] or info["manifest_sha256"] != data["manifest_sha256"]):
        raise ValueError("wheel belongs to a different/dirty source or manifest")
    manifest = json.loads(Path(info["manifest_path"]).read_text())
    for name, digest in manifest["resources"].items():
        if name.startswith("_native/"):
            continue
        path = source / "actlize_la" / name
        if not path.is_file() or path.is_symlink() or sha(path) != digest:
            raise ValueError(f"wheel/source binding differs: {name}")
    dependency = subprocess.check_output(
        ["git", "rev-parse", "HEAD:third_party/actlize"], cwd=source, text=True).strip()
    if info["source_identity"]["actlize_revision"] != dependency:
        raise ValueError("wheel used a different actlize dependency")
    files = [directory / name for name in sorted(data["assets"])] + [directory / "release-manifest.json"]
    sums = directory / "SHA256SUMS"
    sums.write_text("".join(f"{sha(path)}  {path.name}\n" for path in files))
    return files + [sums]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--scratch", type=Path, required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--source-sha", required=True)
    p.add_argument("--asset-list", type=Path, required=True)
    args = p.parse_args()
    source = args.source.resolve(strict=True)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if actual != args.source_sha:
        raise RuntimeError("checked-out source differs from requested tag SHA")
    files = verify(args.assets.resolve(strict=True), source, args.scratch.resolve(), args.tag, actual)
    args.asset_list.write_text("".join(str(path) + "\n" for path in files))
    print(f"[release] assets={len(files)} source={actual} host=QUALIFIED device=NOT_RUN")


if __name__ == "__main__":
    main()
