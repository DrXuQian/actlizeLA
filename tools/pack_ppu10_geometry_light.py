#!/usr/bin/env python3
"""Repack existing geometry evidence only: no build, device query or ACU launch."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile

from collect_ppu_gdn_acu import capture_arms
from sweep_ppu10_geometry import CONTROL, DELIVERIES, cells, check_comparison, load_shapes


def regular(root, relative):
    name = PurePosixPath(relative)
    if name.is_absolute() or ".." in name.parts or str(name) != relative:
        raise ValueError(f"unsafe evidence path: {relative}")
    path = root.joinpath(*name.parts)
    for part in (path, *path.parents):
        if part == root:
            break
        if part.is_symlink():
            raise ValueError(f"evidence symlink: {path}")
    if not path.is_file():
        raise ValueError(f"missing evidence: {path}")
    return path


def manifest(root, filename):
    values = {}
    for line in regular(root, filename).read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match or match[2] in values:
            raise ValueError(f"invalid/duplicate checksum entry: {filename}")
        name = PurePosixPath(match[2])
        if name.is_absolute() or ".." in name.parts or str(name) != match[2]:
            raise ValueError(f"unsafe checksum entry: {filename}")
        values[match[2]] = match[1]
    if not values:
        raise ValueError(f"empty checksum manifest: {filename}")
    return values


def checked(root, relative, hashes):
    data = regular(root, relative).read_bytes()
    if hashlib.sha256(data).hexdigest() != hashes.get(relative):
        raise ValueError(f"missing/changed capture checksum: {root / relative}")
    return data


def choose_run(argument, search_root=Path("/workspace")):
    if argument:
        run = argument.parent if argument.name == "geometry.tar.gz" else argument
        return run.resolve(strict=True)
    candidates = sorted(p for p in search_root.glob("actlizeLA-ppu10-geometry-*")
                        if p.is_dir() and (p / "geometry.SHA256SUMS").is_file())
    if len(candidates) != 1:
        listing = "\n".join(str(p) for p in candidates) or "(none)"
        raise ValueError("specify exactly one existing run directory; candidates:\n" + listing)
    return candidates[0].resolve()


def inventory(run):
    """Keep every arm, not only winners. Bytes are the original verified exports."""
    outer = manifest(run, "geometry.SHA256SUMS")
    required = {"sha.txt", "comparison.json", "geometry-shapes.json", "geometry-registration.md",
                "binaries.sha256", "residual-correctness.log", "geometry-edge.log", "geometry-native.log"}
    if not required <= outer.keys():
        raise ValueError("incomplete original geometry evidence manifest")
    files = {"original-geometry.SHA256SUMS": (run / "geometry.SHA256SUMS").read_bytes()}
    for name in outer:
        if "/" not in name and name.endswith((".txt", ".json", ".log", ".md", ".diff", ".sha256")):
            files[name] = checked(run, name, outer)
    measured_sha = files["sha.txt"].decode().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", measured_sha):
        raise ValueError("missing measured source SHA")
    shapes = load_shapes(run / "geometry-shapes.json")
    check_comparison(json.loads(files["comparison.json"]), shapes)
    identity = None
    reports = []
    total = 0
    for name, shape, gate in cells(shapes):
        bundle = run / name / "bundle"
        hashes = manifest(bundle, "SHA256SUMS")
        files[name + "/SHA256SUMS"] = (bundle / "SHA256SUMS").read_bytes()
        def keep(relative):
            data = checked(bundle, relative, hashes)
            files[name + "/" + relative] = data
            return data
        status = json.loads(keep("STATUS.json"))
        arms = capture_arms(bundle, "wy", DELIVERIES[0], CONTROL, DELIVERIES[1:])
        expected = {a.label: dict(role=a.role, wy_delivery=a.delivery,
                                 directory=str(a.directory.relative_to(bundle))) for a in arms}
        if (status.get("status") != "PASS" or status.get("errors") or
                status.get("capture_order") != [a.label for a in arms] or status.get("capture_arms") != expected):
            raise ValueError(f"incomplete/mismatched capture arms: {name}")
        if status.get("comparison_origin", {}).get("source_sha") != measured_sha:
            raise ValueError(f"mixed measured source SHA: {name}")
        # Small native resource/compiler/device records; no ISA or source tree.
        for relative in ("resources.txt", "hgcc-version.txt", "acu-version.txt", "device-before.txt",
                         "device-after.txt", "git-head.txt", "git-status.txt", "git-diff.txt",
                         "submodules.txt", "gdn_hgcc_arch.txt"):
            if relative in hashes:
                keep(relative)
        for arm in arms:
            prefix = arm.directory.relative_to(bundle)
            rel = lambda suffix: str(prefix / (arm.role + suffix))
            receipt = json.loads(keep(rel(".json")))
            keep(rel("-preflight.json"))
            if (receipt.get("status") != "PASS" or receipt.get("role") != arm.role or
                    receipt.get("wy_delivery") != arm.delivery or receipt.get("shape") != dict(**shape, K=128, V=128) or
                    receipt.get("gate") != gate or receipt.get("phase") != "subject" or
                    receipt.get("public_api_calls") != 1):
                raise ValueError(f"missing/wrong subject receipt: {name}/{arm.label}")
            current = tuple(receipt[k] for k in ("device", "torch", "extension_sha256", "library_sha256"))
            if identity is None:
                identity = current
            elif current != identity:
                raise ValueError(f"mixed device/runtime/binary: {name}/{arm.label}")
            for page in ("details", "raw"):
                if status.get("probes", {}).get(f"{arm.label}-{page}", {}).get("status") != "COLLECTED":
                    raise ValueError(f"missing successful existing export: {name}/{arm.label}/{page}; do not rerun kernels")
                if not keep(rel(f"-{page}.txt")).strip():
                    raise ValueError(f"empty counter export: {name}/{arm.label}/{page}")
            for suffix in ("-acu.log", "-acu.log.command", "-preflight.log", "-preflight.log.command",
                           "-details.txt.command", "-raw.txt.command"):
                if rel(suffix) in hashes:
                    keep(rel(suffix))
            base = str(prefix / f"{arm.role}-g{gate}.report")
            paths = [p for p in (base, base + ".acurep") if p in hashes]
            if len(paths) != 1:
                raise ValueError(f"missing/ambiguous original report digest: {name}/{arm.label}")
            report = regular(bundle, paths[0])
            if report.stat().st_size == 0:
                raise ValueError(f"empty original report: {name}/{arm.label}")
            reports.append(dict(cell=name, arm=arm.label, relative_report=paths[0],
                                sha256=hashes[paths[0]], bytes=report.stat().st_size,
                                digest_scope="original-capture-manifest; raw bytes not rehashed by this packer"))
            total += 1
    if total != 60 or len(reports) != 60:
        raise ValueError("geometry coverage must be 12 cells / 60 arms")
    index = dict(schema=1, measured_source_sha=measured_sha, cells=12, arms=total,
                 contents="complete details/raw text exports, receipts, identities; no metric filtering",
                 omitted="original reports, binaries, source trees, disassembly remain on box",
                 validation="selected files match original checksum manifests; NOT a new performance verdict",
                 reports=reports, created_utc=datetime.now(timezone.utc).isoformat(),
                 packer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    files["LIGHT_INDEX.json"] = (json.dumps(index, indent=2) + "\n").encode()
    return files


def pack(run, output=None):
    output = output or run / "geometry-light.tar.gz"
    if output.exists() or output.is_symlink():
        raise ValueError(f"preserve existing output: {output}")
    files = inventory(run)
    sums = "".join(f"{hashlib.sha256(data).hexdigest()}  {name}\n" for name, data in sorted(files.items()))
    files["LIGHT_SHA256SUMS"] = sums.encode()
    # Exclusive create. Never overwrite/delete original reports or an old pack.
    with output.open("xb") as stream, tarfile.open(fileobj=stream, mode="w:gz", compresslevel=9) as archive:
        for name, data in sorted(files.items()):
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            entry.mode = 0o644
            archive.addfile(entry, io.BytesIO(data))
    print(f"[geometry light] 12 cells / 60 arms; {output.stat().st_size / 2**20:.2f} MiB; UPLOAD={output}")
    print("[geometry light] originals unchanged; no compilation/profiling/device query executed")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?", type=Path, help="existing run directory or its geometry.tar.gz")
    parser.add_argument("--output", type=Path, help="new archive path; never overwritten")
    args = parser.parse_args()
    try:
        pack(choose_run(args.run), args.output)
    except (OSError, ValueError, KeyError) as exc:
        print(f"[geometry light] FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
