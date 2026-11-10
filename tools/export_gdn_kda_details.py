#!/usr/bin/env python3
"""Package native instructions/counters from EXISTING reports using PATH's acu.

The box path needs no acu_report Python API, SDK change, build or GPU execution.
"""
import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile

from acu_installation import check_profiler, profiler_layout
from collect_ppu_gdn_acu import DEFAULT_ACU, report_file, sha
from collect_serving_prefill_acu import parse_raw
from gdn_kda_contract import cells, complete, instruction_rows, inventory, roles, summarize, validate_receipt

ROOT = Path(__file__).resolve().parents[1]


def instruction_metric_id(layout):
    """Read the captured section authority; never guess the underlying counter."""
    path = Path(layout["sections"]) / "InstructionStatistics.section"
    if sha(path) != layout["section_sha256"][path.name]:
        raise ValueError("captured instruction section changed")
    names = []
    for block in re.findall(r"Metrics\s*\{([^{}]*)\}", path.read_text()):
        if re.findall(r'\bLabel:\s*"([^"]*)"', block) == ["Executed Instructions"]:
            names.extend(re.findall(r'\bName:\s*"([^"]*)"', block))
    if len(names) != 1:
        raise ValueError("captured Executed Instructions metric binding missing or ambiguous")
    return names[0]


def select_reports(summary):
    """Deterministic diagnostic subset; never select the fastest repeat."""
    candidate = summary.get("experiment")
    selected = roles(candidate)
    if (summary.get("status") != "COLLECTED" or summary.get("errors") or summary.get("cells") != cells()
            or complete(summary["captures"]) != "COLLECTED"):
        raise ValueError("incomplete registered eight-cell source bundle")
    reports = []
    for cell in cells():
        record = summary["captures"][cell["id"]]
        if summarize(record["captures"], candidate)["status"] != "ADMITTED":
            raise ValueError("missing or unadmitted source capture")
        if cell["gate"] != -.1:
            continue
        for role in selected:
            capture = record["captures"][role][0]
            digest = capture.get("report_sha256", "")
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("original report hash missing")
            reports.append(dict(cell=cell["id"], role=role, repeat=0, capture=capture))
    if len(reports) != 12:
        raise ValueError("diagnostic denominator must be four geometries x three roles")
    return reports


def source_counts(text, kernels):
    """Keep every PC; raw CLI columns remain in the accompanying CSV.

    In ACU's source CSV, Range Id identifies a captured action (not the API's
    context-range index). Function Id is a source-page ordinal or, for our
    embedded rule, the native function ID. Do not mix those namespaces. Bind
    actions to the checked raw page/report hash; retain all helper functions.
    """
    lines = text.splitlines()
    start = next((i for i,line in enumerate(lines) if line.startswith('"Range Id",')), None)
    if start is None:
        raise ValueError("ACU source page lacks the native PC table; metric rows are not native instructions")
    rows = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    required = {"Range Id", "Function Name", "Function Id", "Address", "Source",
                "Instructions Executed", "Thread Instructions Executed"}
    if not required.issubset(rows.fieldnames or ()):
        raise ValueError("ACU source page lacks required PC/execution fields")
    expected = {str(k["id"]): k["kernel"] for k in kernels}
    groups = {key: dict(kernel=name, static_pcs=0, pc_warp_executions=0,
                        pc_thread_executions=0, functions=set(), opcode_warp_counts={})
              for key,name in expected.items()}
    seen = set()
    for row in rows:
        action = row["Range Id"]
        if action not in groups:
            raise ValueError("unexpected source action; cannot drop extra work")
        pc = int(row["Address"], 16)
        key = (action, row["Function Id"], pc)
        code, function = row["Source"].strip(), row["Function Name"].strip()
        counts = [int(row[name].replace(",", "")) for name in
                  ("Instructions Executed", "Thread Instructions Executed")]
        if key in seen or not code or not function or any(n < 0 for n in counts):
            raise ValueError("duplicate/missing PC instruction or invalid execution count")
        seen.add(key)
        group = groups[action]
        group["static_pcs"] += 1
        group["pc_warp_executions"] += counts[0]
        group["pc_thread_executions"] += counts[1]
        group["functions"].add(function)
        # Exact source mnemonic, NOT the GUI's normalized opcode family.
        opcode = code.split()[0]
        group["opcode_warp_counts"][opcode] = group["opcode_warp_counts"].get(opcode, 0) + counts[0]
    for group in groups.values():
        if not group["static_pcs"] or group["pc_warp_executions"] <= 0:
            raise ValueError("missing native source/execution counts for a captured kernel")
        if not any(re.search(r"(?:^|::|\s)" + re.escape(name) + r"(?=[<(])", group["kernel"])
                   or name == group["kernel"] for name in group["functions"]):
            raise ValueError("source functions do not contain the captured kernel")
        group["functions"] = sorted(group["functions"])
    return groups


def rule_counts(path, kernels, instructions, metric_id):
    """A rule can fail with ACU rc0: require completion and original bindings."""
    marker = Path(str(path) + ".complete")
    if not marker.is_file():
        raise ValueError("ACU native rule did not complete; partial CSV/rc0 is not success")
    text = path.read_text()
    groups = source_counts(text, kernels)
    denominator = f"{len(kernels)},{sum(g['static_pcs'] for g in groups.values())}\n"
    if marker.read_text() != denominator:
        raise ValueError("native rule completion denominator differs from exported actions/PCs")
    expected = {str(k["id"]): (k, c) for k, c in zip(kernels, instructions)}
    if len(instructions) != len(kernels):
        raise ValueError("native rule instruction denominator changed")
    factors = {"nsecond": .001, "usecond": 1., "msecond": 1000., "second": 1000000.}
    shape = lambda s: tuple(int(x.strip()) for x in s.strip("() ").split(","))
    for row in csv.DictReader(io.StringIO(text)):
        k, c = expected[row["Range Id"]]
        if (row["Kernel Name"] != k["kernel"] or not row["Kernel Mangled Name"]
                or shape(row["Grid Size"]) != shape(k["grid"])
                or shape(row["Block Size"]) != shape(k["block"])):
            raise ValueError("native rule symbol/geometry differs from captured CSV")
        unit = row["Duration Unit"]
        if (unit not in factors or not math.isclose(float(row["Duration"]) * factors[unit],
                                                   k["us"], rel_tol=1e-9, abs_tol=1e-5)):
            raise ValueError("native rule duration differs from captured CSV")
        if (c["kernel"] != k["kernel"] or row["Instruction Metric"] != metric_id
                or row["Instruction Unit"] != "inst"
                or Decimal(row["Aggregate Instructions"]) != c["count"]):
            raise ValueError("native rule instruction count differs from captured CSV")
    return groups


def export_rule(prefix, report, destination, env, kernels, instructions, metric_id):
    rule = ROOT / "tools/acu_work_rule.py"
    command = [*prefix, "--import", str(report), "--stats-rule", str(rule),
               "--stats-rule-output", str(destination)]
    log = destination.with_suffix(".log")
    receipt = destination.with_suffix(".command.json")
    # Never reuse a completed result from another invocation.
    if any(p.exists() for p in (destination, Path(str(destination) + ".complete"), log, receipt)):
        raise ValueError("native export destination already exists")
    receipt.write_text(json.dumps(command) + "\n")
    with log.open("x") as output:
        process = subprocess.run(command, env=dict(env, ACTLIZE_ACU_INSTRUCTION_METRIC=metric_id),
                                 stdout=output, stderr=subprocess.STDOUT, timeout=180)
    if process.returncode or "Error parsing text-format HG.Profiler.ProfilerSection" in log.read_text(errors="replace"):
        raise RuntimeError(f"ACU offline native rule rc={process.returncode}; see {log}\n{log.read_text()[-3000:]}")
    try:
        return rule_counts(destination, kernels, instructions, metric_id)
    except Exception as error:
        raise ValueError(f"{error}; see {log}\n{log.read_text()[-3000:]}") from error


def export_page(prefix, report, page, destination, env):
    """Only --import; no executable-to-profile or device-query option exists here."""
    if page not in ("raw", "source", "details"):
        raise ValueError("unsupported offline report page")
    command = [*prefix, "--import", str(report), "--csv", "--page", page]
    log = destination.with_suffix(".log")
    destination.with_suffix(".command.json").write_text(json.dumps(command) + "\n")
    with destination.open("w") as output, log.open("w") as errors:
        process = subprocess.run(command, env=env, stdout=output, stderr=errors, timeout=180)
    if process.returncode:
        raise RuntimeError(f"ACU offline import rc={process.returncode}; see {log}\n{log.read_text()[-3000:]}")
    if (not destination.stat().st_size or "Error parsing text-format HG.Profiler.ProfilerSection"
            in destination.read_text(errors="replace") + log.read_text(errors="replace")):
        raise ValueError(f"empty/invalid ACU offline export: {destination}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True, help="completed KDA run with evidence/summary.json and acurep files")
    p.add_argument("--acu", type=Path, default=DEFAULT_ACU)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = (a.out or Path(f"/workspace/actlize-la-gdn-work-details-{stamp}-{os.getpid()}")).resolve()
    if not out.is_relative_to(Path("/workspace")) or out == Path("/workspace"):
        p.error("use a new dedicated /workspace output directory")
    out.mkdir(parents=True, exist_ok=False)
    status = dict(status="INCOMPLETE", created_utc=stamp, exports=[], errors=[], backend="offline-cli-stats-rule",
                  scope="offline opcode/PC/traffic evidence; no compilation, device query or GPU launch",
                  selection="all four geometries, g=-0.1, repeat0, all three roles; not fastest selection",
                  count_scope="source PC warp executions; do not equate to aggregate Executed Instructions",
                  source_locations="native PCs; absent optional diagnostics are blank, not zero",
                  function_id_namespace="native HgRules function ID, not source-page CLI ordinal")
    try:
        evidence = a.run.resolve(strict=True) / "evidence"
        source = evidence / "summary.json"
        summary = json.loads(source.read_text())
        selected = select_reports(summary)
        layout = profiler_layout(a.acu)
        captured = summary["profiler"]
        status.update(source_run=str(a.run.resolve()), source_summary_sha256=sha(source),
                      captured_source_sha=summary["source_sha"], candidate=summary["candidate"],
                      capture_profiler=captured, reader_profiler=layout, reader_matches_capture=(layout == captured),
                      reader_scope="offline decoding only; original timings/counters must match; "
                                   "reader-specific PC diagnostics are not mixed with capture-reader diagnostics",
                      exporter_files={n: sha(ROOT / "tools" / n) for n in
                                      ("export_gdn_kda_details.py", "acu_work_rule.py", "gdn_kda_contract.py",
                                       "acu_installation.py", "collect_ppu_gdn_acu.py")})
        (out / "captured-summary.json").write_bytes(source.read_bytes())
        # Read the original section's counter binding; do NOT execute its CLI.
        status["instruction_metric_id"] = instruction_metric_id(captured)
        prefix = [layout["executable"], "--section-folder", layout["sections"]]
        env = dict(os.environ, ASIGHT_HOME=layout["home"])
        print(f"[GDN work export] reader={layout['executable']} "
              f"sha256={layout['executable_sha256']} capture={captured['executable']} "
              f"same_reader={int(layout == captured)}; device_execution=NONE", flush=True)
        check_profiler(layout, env, prefix, out)
        for item in selected:
            cell, role, repeat = item["cell"], item["role"], item["repeat"]
            stem = f"{role}-{repeat}"
            directory = evidence / cell
            admission = json.loads((directory / "admission.json").read_text())
            subject = json.loads((directory / f"{stem}.json").read_text())
            validate_receipt(admission, subject, role)
            csv_text = (directory / f"{stem}.csv").read_text()
            observed = inventory(role, parse_raw(csv_text))
            if observed["status"] != "ADMITTED" or observed["kernels"] != item["capture"]["kernels"]:
                raise ValueError("original CSV no longer matches captured inventory")
            counts = instruction_rows(csv_text, observed["kernels"], status["instruction_metric_id"])
            report = report_file(directory / f"{stem}.report")
            if sha(report) != item["capture"]["report_sha256"]:
                raise ValueError("report SHA256 differs from captured receipt")
            target = out / cell
            target.mkdir(exist_ok=True)
            pages = {page: target / f"{stem}-{page}.csv" for page in ("raw", "details")}
            print(f"[GDN work export] {cell} {role} repeat0: ACU --import + embedded native rule; "
                  "external Python API not required", flush=True)
            for page, path in pages.items():
                export_page(prefix, report, page, path, env)
            raw = pages["raw"].read_text()
            decoded = parse_raw(raw)
            if (decoded != observed["kernels"]
                    or instruction_rows(raw, decoded, status["instruction_metric_id"]) != counts):
                raise ValueError("re-imported timing/instructions differ from captured CSV")
            pages["native"] = target / f"{stem}-native.csv"
            native = export_rule(prefix, report, pages["native"], env, decoded, counts, status["instruction_metric_id"])
            pages["complete"] = Path(str(pages["native"]) + ".complete")
            if sha(report) != item["capture"]["report_sha256"]:
                raise ValueError("original report changed during import")
            status["exports"].append(dict(cell=cell, role=role, repeat=repeat,
                files={str(path.relative_to(out)): sha(path) for path in pages.values()},
                report_sha256=item["capture"]["report_sha256"], actions=native))
        if profiler_layout(a.acu) != layout or sha(source) != status["source_summary_sha256"]:
            raise ValueError("capture manifest or offline ACU reader changed")
        if any(sha(ROOT / "tools" / name) != digest for name, digest in status["exporter_files"].items()):
            raise ValueError("exporter or native rule changed during export")
        if len(status["exports"]) != 12:
            raise ValueError("incomplete diagnostic export denominator")
        status["status"] = "OFFLINE_EXPORTED"
    except Exception as error:
        status["errors"].append(f"{type(error).__name__}: {error}")
        print(f"[GDN work export] INCOMPLETE: {status['errors'][-1]}", file=sys.stderr)
    finally:
        (out / "summary.json").write_text(json.dumps(status, indent=2) + "\n")
        paths = sorted(p for p in out.rglob("*") if p.is_file())
        with tarfile.open(out / "results.tar.gz", "w:gz") as archive:
            for path in paths:
                archive.add(path, arcname=str(path.relative_to(out)), recursive=False)
        print(f"[GDN work export] {status['status']}; upload={out / 'results.tar.gz'}; "
              f"bytes={(out / 'results.tar.gz').stat().st_size}; device_execution=NONE", flush=True)
    return 0 if status["status"] == "OFFLINE_EXPORTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
