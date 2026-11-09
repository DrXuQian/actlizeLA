#!/usr/bin/env python3
"""Offline additive phase budget from admitted eight-cell ACU bundles.

Accept the older default/serving/FLA bundle or the new default/PLA/FLA bundle.
Read reports as data only; never extract or execute uploaded Python code.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import collect_serving_prefill_acu as retained
import gdn_threeway_contract as threeway

PHASES = ("prefix", "solve", "state_equivalent", "output", "other_included")
METRICS = ("Registers Per Thread", "Dynamic Shared Memory Per Block",
           "Achieved Active Warps Per CU", "Executed Instructions", "CE Frequency")


def files(blob):
    result = {}
    total = 0
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as archive:
        for item in archive.getmembers():
            path = PurePosixPath(item.name)
            if (path.is_absolute() or ".." in path.parts or item.name in result
                    or not (item.isfile() or item.isdir()) or item.size > 128 * 1024 * 1024):
                raise ValueError(f"unsafe/duplicate archive member: {item.name}")
            if item.isfile():
                total += item.size
                if total > 512 * 1024 * 1024:
                    raise ValueError("archive exceeds bounded review size")
                result[item.name] = archive.extractfile(item).read()
    return result


def phase(role, kernel):
    names = ({"gdn_wy_split_prefix": "prefix", "gdn_wy_split_solve_static": "solve",
              "gdn_wy_residual_full_chunk_state": "state_equivalent",
              "gdn_wy_residual_warps8_hvlayout_output": "output"} if role == "default" else
             {"chunk_local_cumsum_scalar_kernel": "prefix", "chunk_gated_delta_rule_fwd_kkt_solve_kernel": "solve",
              "recompute_w_u_fwd_kernel": "state_equivalent", "chunk_gated_delta_rule_fwd_kernel_h": "state_equivalent",
              "chunk_fwd_kernel_o": "output"})
    matches = [value for name, value in names.items() if name in kernel]
    if len(matches) == 1:
        return matches[0]
    if role == "fla" and "FillFunctor<" in kernel:
        return "other_included"
    raise ValueError(f"unregistered phase: {kernel}")


def phase_times(role, kernels):
    times = dict.fromkeys(PHASES, 0.)
    for row in kernels:
        times[phase(role, row["kernel"])] += row["us"]
    if not math.isclose(sum(times.values()), sum(r["us"] for r in kernels), rel_tol=0, abs_tol=1e-8):
        raise ValueError("phase mapping lost work")
    return times


def analyze(blob):
    outer = files(blob)
    data = files(outer["acu/results.tar.gz"]) if "acu/results.tar.gz" in outer else outer
    summary = json.loads(data["summary.json"])
    if summary.get("status") != "COLLECTED" or summary.get("errors"):
        raise ValueError("bundle is not completely admitted/collected")
    expected = {c["id"] for c in threeway.cells()}
    if set(summary["captures"]) != expected:
        raise ValueError("missing/unexpected cell; keep the eight-cell denominator")
    new = "pla" in summary
    roles = threeway.ROLES if new else retained.ROLES
    validate = threeway.validate_receipt if new else retained.validate_receipts
    result = dict(archive_sha256=hashlib.sha256(blob).hexdigest(),
                  source_sha=summary.get("source_sha", summary.get("harness_sha")),
                  scope="admitted full-call ACU sums; additive phase budget uses MEANS, not sums of medians",
                  phases=PHASES, cells=[], clocks=[], source_observations={})
    clocks = set()
    for spec in threeway.cells():
        key = spec["id"]
        saved = summary["captures"][key]
        admission = json.loads(data[key + "/admission.json"])
        if (admission.get("shape") != threeway.geometry(spec) or admission.get("gate") != spec["gate"]
                or set(saved["captures"]) != set(roles)
                or {len(v) for v in saved["captures"].values()} not in ({3}, {6})):
            raise ValueError("registered shape/gate/role/repetition denominator changed")
        timings, resources = {}, {}
        for role in roles:
            group, counters = [], []
            captures = saved["captures"][role]
            if len(captures) not in (3, 6):
                raise ValueError("incomplete order cycle")
            for repeat, original in enumerate(captures):
                base = f"{key}/{role}-{repeat}"
                validate(admission, json.loads(data[base + ".json"]), role)
                text = data[base + ".csv"].decode()
                kernels = retained.parse_raw(text)
                measured = threeway.inventory(role, kernels) if new else retained.inventory(role, kernels)
                if (measured["status"] != "ADMITTED" or kernels != original["kernels"]
                        or not math.isclose(measured["total_us"], original["total_us"], rel_tol=0, abs_tol=1e-8)):
                    raise ValueError("CSV/receipt/inventory mismatch")
                if role not in ("default", "fla"):
                    continue
                group.append(phase_times(role, kernels))
                current = {}
                for row in csv.DictReader(text[text.index('"ID"'):].splitlines()):
                    metric = row["Metric Name"]
                    if metric in METRICS:
                        if metric == "CE Frequency":
                            if row["Metric Unit"] != "cycle/second":
                                raise ValueError("unregistered CE clock unit")
                            clocks.add(float(row["Metric Value"].replace(",", "")))
                        record = current.setdefault(row["ID"], dict(kernel=row["Kernel Name"],
                                                                    grid=row["Grid Size"], block=row["Block Size"]))
                        record[metric] = dict(value=row["Metric Value"], unit=row["Metric Unit"], section=row["Section Name"])
                counters.append(current)
            if role in ("default", "fla"):
                timings[role] = {p: statistics.mean(r[p] for r in group) for p in PHASES}
                resources[role] = counters
        delta = {p: timings["default"][p] - timings["fla"][p] for p in PHASES}
        result["cells"].append(dict(cell=key, phase_mean_us=timings, delta_mean_us=delta,
                                    total_delta_mean_us=sum(delta.values()), resources=resources))
    result["clocks"] = sorted(clocks)
    return result


def markdown(result):
    lines = ["# ZW810 additive phase budget", "", f"Archive SHA256: `{result['archive_sha256']}`", "",
             "Times below are **means** (us), ours minus FLA. Positive is excess; negative is a saving. "
             "Means make the phase differences additive. No per-issue stall ratio is treated as wall time.", "",
             "Our residual state is compared with FLA W/U + state. FLA fill kernels are retained in other.", "",
             "| cell | prefix | solve | state equivalent | output | other included | total gap |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in result["cells"]:
        numbers = [row["delta_mean_us"][p] for p in PHASES] + [row["total_delta_mean_us"]]
        lines.append("| " + row["cell"] + " | " + " | ".join(f"{v:+.3f}" for v in numbers) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.archive.read_bytes())
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "baseline.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.out / "baseline.md").write_text(markdown(result))
    with (args.out / "baseline.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["cell", *PHASES, "total_delta_mean_us"])
        for row in result["cells"]:
            writer.writerow([row["cell"], *[row["delta_mean_us"][p] for p in PHASES], row["total_delta_mean_us"]])
    print(markdown(result))


if __name__ == "__main__":
    main()
