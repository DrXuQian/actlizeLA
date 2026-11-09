#!/usr/bin/env python3
"""Offline reader of an existing ACU report. No torch, profiler or GPU launch.

Use the Python report API shipped beside the capture's ACU executable. A
separate process contains native parser failures; never import uploaded code.
"""
import argparse
import gzip
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys

SASS_FIELDS = (
    "address", "function_id", "sass_code", "vector_live_registers", "scalar_live_registers",
    "warp_executed_count", "thread_executed_count", "achieved_access_cachelines",
    "theoretical_access_cachelines", "achieved_access_sectors", "achieved_access_data_bytes",
    "source_file_name", "source_file_line_number",
)
STALL_FIELDS = (
    "instruction_fetch", "compute_dependency", "memory_dependency", "memory_throttle",
    "not_selected", "pipe_busy", "sleeping", "sync", "salu_control", "simt_control",
    "sreg_read", "fence_sys", "amc", "others", "wait_pc", "total",
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def api_identity(home):
    root = Path(home).resolve(strict=True) / "extras/python/acu_report"
    if not (root / "__init__.py").is_file():
        raise ValueError(f"companion acu_report API unavailable: {root}; no device work attempted")
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in (".py", ".so"))
    if not any(p.suffix == ".so" for p in paths):
        raise ValueError("companion acu_report native reader missing")
    return dict(directory=str(root), files={str(p.relative_to(root)): sha(p) for p in paths})


def scalar(value):
    # Preserve a reported NaN/Inf explicitly, never replace absent data with 0.
    return str(value) if isinstance(value, float) and not math.isfinite(value) else value


def instances(metric):
    reader = {2: metric.as_string, 3: metric.as_double, 4: metric.as_double,
              5: metric.as_uint64, 6: metric.as_uint64}.get(metric.kind())
    n = metric.num_instances()
    if n and reader is None:
        raise ValueError(f"unsupported metric instance kind {metric.kind()}: {metric.name()}")
    return [scalar(reader(i)) for i in range(n)]


def metric_record(metric):
    if metric is None:
        raise ValueError("null metric from native report reader")
    values = instances(metric)
    result = dict(name=metric.name(), kind=metric.kind(), unit=metric.unit(), has_value=metric.has_value(),
                  value=scalar(metric.value()) if metric.has_value() else None, instances=values)
    if metric.has_correlation_ids():
        ids = metric.correlation_ids()
        if ids is None:
            raise ValueError("missing declared metric correlation IDs")
        result["correlation_ids"] = instances(ids)
        if len(result["correlation_ids"]) != len(values):
            raise ValueError("metric correlation denominator changed")
    return result


def action_record(action, range_index):
    metrics = {k: metric_record(v) for k,v in action.metrics().items()}
    native = action.sass()
    if not native:
        raise ValueError(f"no SASS in captured action: {action.name()}")
    instructions = []
    for pc, insn in sorted(native.items()):
        if insn is None:
            raise ValueError(f"missing SASS at {pc}")
        record = {name: getattr(insn, name) for name in SASS_FIELDS}
        if record["address"] != pc or not record["sass_code"]:
            raise ValueError("native PC/instruction binding missing")
        for name in ("stall", "not_issued_stall"):
            reason = getattr(insn, name)
            record[name] = {k: getattr(reason, k) for k in STALL_FIELDS}
        instructions.append(record)
    executed = sum(x["warp_executed_count"] for x in instructions)
    if executed <= 0:
        raise ValueError(f"no per-PC execution counts: {action.name()}")
    stall_samples = sum(x["stall"]["total"] for x in instructions)
    return dict(range_index=range_index, capture_id=action.capture_sequence_id(),
                name=action.name(), demangled=action.name(1), mangled=action.name(2),
                grid=action.grid_size(), block=action.block_size(), device=action.device(),
                workload_type=action.workload_type(), functions=action.functions(), metrics=metrics, sass=instructions,
                coverage=dict(static_pcs=len(instructions), warp_executions=executed,
                              source_bound_pcs=sum(bool(x["source_file_name"]) for x in instructions),
                              stall_samples=stall_samples,
                              stall_status="RECORDED" if stall_samples else "NONE_RECORDED_NOT_ZERO_LATENCY"))


def export_report(home, report, digest):
    report = Path(report).resolve(strict=True)
    if sha(report) != digest:
        raise ValueError("report SHA256 differs from the captured receipt")
    before = api_identity(home)
    root = Path(before["directory"])
    if "acu_report" in sys.modules:
        raise ValueError("run in a fresh process, not with a previously imported report API")
    sys.path.insert(0, str(root.parent))
    api = importlib.import_module("acu_report")
    if Path(api.__file__).resolve().parent != root:
        raise ValueError("loaded a different acu_report implementation")
    ctx = api.load_report(str(report))
    actions = []
    for i in range(ctx.num_ranges()):
        r = ctx.range_by_idx(i)
        for j in range(r.num_actions()):
            actions.append(action_record(r.action_by_idx(j), i))
    if not actions:
        raise ValueError("empty report, not a successful export")
    if sha(report) != digest or api_identity(home) != before:
        raise ValueError("report or reader changed during export")
    return dict(schema=1, scope="offline existing-report export; no device execution; not a new speed verdict",
                report_sha256=digest, api=before, actions=actions,
                notes=["PC warp executions and pu__inst_executed.sum are different metrics; do not equate them",
                       "stall fields are recorded samples, not disjoint wall-time components",
                       "missing source locations/zero samples remain explicit; no new line-info capture implied"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--acu-home", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = export_report(a.acu_home, a.report, a.sha256)
    with gzip.open(a.output, "xt", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    print(json.dumps(dict(actions=len(result["actions"]), report_sha256=a.sha256,
                          bytes=a.output.stat().st_size, status="OFFLINE_EXPORTED")))


if __name__ == "__main__":
    main()
