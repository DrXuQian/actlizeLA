#!/usr/bin/env python3
"""Package native instructions/counters from EXISTING reports using the capture CLI.

The box path needs no acu_report Python API, SDK change, build or GPU execution.
"""
import argparse
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile

from acu_installation import profiler_layout
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
    context-range index). Function Id is a CLI ordinal, not the API's symbol
    hash. Preserve those namespaces; bind actions to the separately checked raw
    page and report hash. Helper functions are included, not silently filtered.
    """
    lines = text.splitlines()
    start = next((i for i,line in enumerate(lines) if line.startswith('"Range Id",')), None)
    if start is None:
        raise ValueError("ACU source page lacks the native PC table")
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
    status = dict(status="INCOMPLETE", created_utc=stamp, exports=[], errors=[], backend="capture-cli-import",
                  scope="offline opcode/PC/traffic evidence; no compilation, device query or GPU launch",
                  selection="all four geometries, g=-0.1, repeat0, all three roles; not fastest selection",
                  count_scope="source PC warp executions; do not equate to aggregate Executed Instructions",
                  source_locations="native PCs; CLI does not promise C++ source-line correlation")
    try:
        evidence = a.run.resolve(strict=True) / "evidence"
        source = evidence / "summary.json"
        summary = json.loads(source.read_text())
        selected = select_reports(summary)
        layout = profiler_layout(a.acu)
        if layout != summary["profiler"]:
            raise ValueError("selected ACU installation changed since capture; use the recorded installation")
        status.update(source_run=str(a.run.resolve()), source_summary_sha256=sha(source),
                      captured_source_sha=summary["source_sha"], candidate=summary["candidate"], profiler=layout,
                      exporter_files={n: sha(ROOT / "tools" / n) for n in
                                      ("export_gdn_kda_details.py", "gdn_kda_contract.py", "acu_installation.py")})
        (out / "captured-summary.json").write_bytes(source.read_bytes())
        status["instruction_metric_id"] = instruction_metric_id(layout)
        prefix = [layout["executable"], "--section-folder", layout["sections"]]
        env = dict(os.environ, ASIGHT_HOME=layout["home"])
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
            pages = {page: target / f"{stem}-{page}.csv" for page in ("raw", "source", "details")}
            print(f"[GDN work export] {cell} {role} repeat0: ACU --import only; Python API not required", flush=True)
            for page, path in pages.items():
                export_page(prefix, report, page, path, env)
            raw = pages["raw"].read_text()
            decoded = parse_raw(raw)
            if (decoded != observed["kernels"]
                    or instruction_rows(raw, decoded, status["instruction_metric_id"]) != counts):
                raise ValueError("re-imported timing/instructions differ from captured CSV")
            native = source_counts(pages["source"].read_text(), decoded)
            if sha(report) != item["capture"]["report_sha256"]:
                raise ValueError("original report changed during import")
            status["exports"].append(dict(cell=cell, role=role, repeat=repeat,
                files={str(path.relative_to(out)): sha(path) for path in pages.values()},
                report_sha256=item["capture"]["report_sha256"], actions=native))
        if profiler_layout(a.acu) != layout or sha(source) != status["source_summary_sha256"]:
            raise ValueError("capture manifest or ACU installation changed")
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
