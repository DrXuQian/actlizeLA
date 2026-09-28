#!/usr/bin/env python3
"""Account for every captured GPU kernel inside a registered forward range."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics

LIBRARY_ROLES = {
    "flashqla": ("ours-cuda", "ours-ppu-source-check", "flashqla-auto", "flashqla-no-cp"),
    "flashinfer": ("ours-cuda", "ours-ppu-source-check", "flashinfer-auto",
                   "flashinfer-no-cp", "flashinfer-auto-log-adapter"),
}


def union_ns(intervals):
    total = 0
    last = None
    for start, end in sorted(intervals):
        if end <= start:
            raise ValueError("non-positive GPU activity duration")
        total += end - max(start, last or start) if last is None or end > last else 0
        last = max(last or end, end)
    return total


def extract(connection, receipt):
    if receipt.get("status") != "CAPTURE_COMPLETE_AWAIT_NSYS_EXTRACTION":
        raise ValueError("capture receipt is not admitted")
    if receipt.get("device_watch", {}).get("errors"):
        raise ValueError("foreign work or monitor failure invalidates trace")
    expected = receipt["calls"]
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("empty/duplicate expected forward labels")
    family = receipt.get("comparison_family")
    if family is not None:
        if family not in LIBRARY_ROLES or receipt.get("samples") != 12:
            raise ValueError("unregistered comparison family/sample denominator")
        registered = {f"GDN_FORWARD|{role}|{sample:03d}"
                      for role in LIBRARY_ROLES[family] for sample in range(12)}
        if set(expected) != registered:
            raise ValueError("registered role/sample denominator mismatch")
    connection.row_factory = sqlite3.Row
    tables = {r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for name in ("StringIds", "NVTX_EVENTS", "CUPTI_ACTIVITY_KIND_KERNEL"):
        if name not in tables:
            raise ValueError(f"missing nsys evidence table {name}")
    strings = dict(connection.execute("SELECT id,value FROM StringIds"))
    ranges = {}
    for entry in connection.execute("SELECT * FROM NVTX_EVENTS"):
        row = dict(entry)
        label = row.get("text") or strings.get(row.get("textId"), "")
        if not label.startswith("GDN_FORWARD|"):
            continue
        if label in ranges or row.get("end") is None or row["end"] <= row["start"]:
            raise ValueError("duplicate/unclosed forward range")
        ranges[label] = {"label": label, "role": label.split("|")[1],
                         "start": row["start"], "end": row["end"], "kernels": [], "memory": []}
    if set(ranges) != set(expected):
        raise ValueError("forward denominator mismatch: missing/extra NVTX calls")

    def owner(row):
        matches = [v for v in ranges.values()
                   if v["start"] <= row["start"] and row["end"] <= v["end"]]
        if len(matches) != 1:
            raise ValueError(f"unassigned/ambiguous GPU activity: {row}")
        return matches[0]

    for entry in connection.execute("SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        row = dict(entry)
        name = strings.get(row.get("demangledName"), strings.get(row.get("shortName")))
        if not name:
            raise ValueError("GPU kernel has no decoded symbol")
        owner(row)["kernels"].append(row | {"name": name})
    for table in ("CUPTI_ACTIVITY_KIND_MEMSET", "CUPTI_ACTIVITY_KIND_MEMCPY"):
        if table in tables:
            for entry in connection.execute(f"SELECT * FROM {table}"):
                row = dict(entry)
                owner(row)["memory"].append(row | {"kind": table})

    roles = defaultdict(list)
    for label in expected:
        row = ranges[label]
        kernels = row["kernels"]
        if not kernels:
            raise ValueError(f"forward contains no GPU kernel: {label}")
        main = [k for k in kernels if "FlatKernelTmaWarpSpecializedKdaFwd" in k["name"]]
        require_fused = family is None or row["role"].startswith("ours-")
        if require_fused and len(main) != 1:
            raise ValueError("expected exactly one original/adapted C++ fused kernel per call")
        if row["role"].startswith("ours-") and len(kernels) != 1:
            raise ValueError("incumbent unexpectedly launched extra device kernels")
        all_activity = kernels + row["memory"]
        intervals = [(k["start"], k["end"]) for k in all_activity]
        span = max(x[1] for x in intervals) - min(x[0] for x in intervals)
        row.update(kernel_sum_us=sum(k["end"]-k["start"] for k in kernels)/1000,
                   memory_sum_us=sum(k["end"]-k["start"] for k in row["memory"])/1000,
                   gpu_span_us=span/1000, gpu_activity_union_us=union_ns(intervals)/1000,
                   gpu_gaps_us=(span-union_ns(intervals))/1000,
                   host_range_us=(row["end"]-row["start"])/1000)
        if require_fused:
            row["fused_kernel_us"] = (main[0]["end"]-main[0]["start"])/1000
        roles[row["role"]].append(row)
    summary = {}
    for role, calls in roles.items():
        metrics = {}
        for name in ("kernel_sum_us", "fused_kernel_us", "memory_sum_us", "gpu_span_us", "gpu_gaps_us"):
            if not all(name in c for c in calls):
                continue
            values = [c[name] for c in calls]
            metrics[name] = dict(median=statistics.median(values), range=[min(values), max(values)], samples=values)
        metrics["calls"] = len(calls)
        metrics["kernel_count_per_call"] = dict(Counter(len(c["kernels"]) for c in calls))
        per_symbol = defaultdict(list)
        for c in calls:
            for k in c["kernels"]:
                per_symbol[k["name"]].append((k["end"]-k["start"])/1000)
        metrics["symbols"] = {name: dict(count=len(vals), median_us=statistics.median(vals),
                                          total_us=sum(vals)) for name, vals in per_symbol.items()}
        summary[role] = metrics
    required_roles = LIBRARY_ROLES[family] if family else ("ours-cuda", "ours-ppu-source-check", "cula")
    if set(summary) != set(required_roles):
        raise ValueError("role denominator mismatch")
    comparisons = {}
    for role in ("ours-cuda", "ours-ppu-source-check") if family is None else required_roles[1:]:
        ours = summary[role]["kernel_sum_us"]
        control = "ours-cuda" if family else "cula"
        ref = summary[control]["kernel_sum_us"]
        verdict = (("CANDIDATE-WINS" if family else "OURS-WINS") if ours["range"][1] < ref["range"][0] else
                   ("CONTROL-WINS" if family else "CULA-WINS") if ref["range"][1] < ours["range"][0] else "UNRESOLVED")
        comparisons[role] = dict(control=control, control_over_candidate=ref["median"]/ours["median"],
                                verdict=verdict, criterion="disjoint-observed-kernel-sum-ranges")
        if family is None:
            comparisons[role]["cula_over_ours"] = ref["median"]/ours["median"]
    return dict(scope=receipt["scope"], gate=receipt["gate"], shape=receipt["shape"],
                input_sha256=receipt["input_sha256"], summary=summary,
                comparisons=comparisons, forwards=list(ranges.values()),
                accounting="ALL_CAPTURED_KERNELS_AND_MEMORY_ASSIGNED_EXACTLY_ONCE")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sqlite", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    with sqlite3.connect(f"file:{args.sqlite.resolve()}?mode=ro", uri=True) as connection:
        result = extract(connection, json.loads(args.receipt.read_text()))
    result["evidence_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in (args.sqlite, args.receipt, Path(__file__))}
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    for role, row in result["summary"].items():
        print(f"[nsys kernel sum] role={role} calls={row['calls']} "
              f"median_us={row['kernel_sum_us']['median']:.3f} range={row['kernel_sum_us']['range']} "
              f"fused_us={row.get('fused_kernel_us', {}).get('median', 'NA')} "
              f"memory_us={row['memory_sum_us']['median']:.3f} gaps_us={row['gpu_gaps_us']['median']:.3f}")
    print(json.dumps(result["comparisons"], indent=2))


if __name__ == "__main__":
    main()
