#!/usr/bin/env python3
"""Per-cell GDN prefill matrix over the hash-registered denominator.

Every registered cell is reported; unmeasured cells are NOT_RUN, never dropped.
A collected cell that is unregistered, measured twice, relabeled (its admitted
operands do not have the registered B/T/Hk/Hv/K/V/gate) or bound to another
device makes the summary FAIL. No geometric mean or best-cell number is an
acceptance result; only the per-cell registered target verdict counts.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

SHAPE = ("B", "T", "Hk", "Hv", "K", "V")
TARGET_STATUSES = ("TARGET_MET", "TARGET_NOT_MET", "UNRESOLVED", "UNADJUDICATED")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def matrix(manifest, expected_sha, runs, device_uuid=None):
    manifest = Path(manifest)
    digest = sha(manifest)
    if digest != expected_sha:
        raise ValueError("shape manifest differs from the registered denominator")
    cells = json.loads(manifest.read_text())["cells"]
    registered = {c["id"]: c for c in cells}
    if len(registered) != len(cells):
        raise ValueError("registered manifest has duplicate cell ids")
    table = {c["id"]: dict(cell=c, status="NOT_RUN") for c in cells}
    for path in runs:
        summary = json.loads(Path(path).read_text())
        if summary.get("manifest_sha256") not in (None, digest):
            raise ValueError(f"{path}: collected against a different manifest")
        for key, entry in summary.get("captures", {}).items():
            if key not in registered:
                raise ValueError(f"{path}: unregistered cell {key}")
            want = registered[key]
            spec, measured = entry.get("cell") or {}, entry.get("measured_shape") or {}
            if any(spec.get(k) != want[k] for k in (*SHAPE, "gate")):
                raise ValueError(f"{path}: {key} requested as {spec}, not the registered cell")
            if any(measured.get(k) != want[k] for k in SHAPE):
                raise ValueError(f"{path}: {key} admitted operands {measured} are not the registered shape")
            if device_uuid is not None and entry.get("device_uuid") != device_uuid:
                raise ValueError(f"{path}: {key} measured on another device {entry.get('device_uuid')}")
            if table[key]["status"] != "NOT_RUN":
                raise ValueError(f"{key}: measured more than once; choose explicitly, never the fastest")
            result = entry.get("summary")
            if result is None:
                table[key].update(status="UNADJUDICATED", run=str(path), reason="capture incomplete")
                continue
            if result.get("target_verdict") not in TARGET_STATUSES:
                raise ValueError(f"{key}: missing registered target verdict")
            rows = result["rows"]
            table[key].update(
                status=result["target_verdict"], run=str(path), repeats=result["repeats"],
                ratio=result["median_ratio_default_over_fla"],
                speedup=result["speedup_fla_over_default"],
                gap_to_target_us=result["gap_to_target_us"],
                core_verdict=result["core_verdict"],
                default=rows["default"], fla=rows["fla"], serving=rows["serving"],
                serving_transport_us=result.get("serving_transport_us"))
    counts = {s: 0 for s in (*TARGET_STATUSES, "NOT_RUN")}
    for row in table.values():
        counts[row["status"]] += 1
    phase1 = [c["id"] for c in cells if c["phase"] == 1]
    return dict(manifest_sha256=digest, denominator=len(cells), counts=counts,
                phase1_complete=all(table[i]["status"] != "NOT_RUN" for i in phase1),
                phase1_all_met=all(table[i]["status"] == "TARGET_MET" for i in phase1),
                all_met=all(row["status"] == "TARGET_MET" for row in table.values()),
                cells=table)


def markdown(result):
    lines = [f"Registered cells: {result['denominator']} (manifest {result['manifest_sha256'][:12]})",
             "Counts: " + ", ".join(f"{k}={v}" for k, v in result["counts"].items()), "",
             "| cell | status | ours median us [min,max] | FLA median us [min,max] | ours/FLA | speedup | gap to 0.80 (us) |",
             "|---|---|---|---|---:|---:|---:|"]
    for key, row in result["cells"].items():
        if row["status"] == "NOT_RUN" or "default" not in row:
            lines.append(f"| {key} | {row['status']} | | | | | |")
            continue
        d, f = row["default"], row["fla"]
        lines.append(f"| {key} | {row['status']} | {d['median_us']:.2f} [{d['min_us']:.2f},{d['max_us']:.2f}] | "
                     f"{f['median_us']:.2f} [{f['min_us']:.2f},{f['max_us']:.2f}] | {row['ratio']:.4f} | "
                     f"{row['speedup']:.4f}x | {row['gap_to_target_us']:+.2f} |")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--device-uuid")
    p.add_argument("--json", type=Path)
    p.add_argument("--markdown", type=Path)
    p.add_argument("summaries", type=Path, nargs="*")
    a = p.parse_args()
    try:
        result = matrix(a.manifest, a.manifest_sha256, a.summaries, a.device_uuid)
    except ValueError as error:
        print(f"[GDN matrix] FAIL: {error}", file=sys.stderr)
        return 1
    if a.json:
        a.json.write_text(json.dumps(result, indent=2) + "\n")
    text = markdown(result)
    if a.markdown:
        a.markdown.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
