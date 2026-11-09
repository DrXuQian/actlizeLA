#!/usr/bin/env python3
"""Package native instructions/counters from EXISTING KDA reports, without GPU work."""
import argparse
from datetime import datetime, timezone
import gzip
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile

from acu_installation import profiler_layout
from acu_report_details import api_identity, sha
from collect_ppu_gdn_acu import DEFAULT_ACU, report_file
from collect_serving_prefill_acu import parse_raw
from gdn_kda_contract import cells, complete, inventory, roles, summarize, validate_receipt

ROOT = Path(__file__).resolve().parents[1]


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


def shape(text):
    return tuple(int(x.strip()) for x in text.strip("() ").split(","))


def check_detail(detail, expected):
    if detail.get("report_sha256") != expected["report_sha256"]:
        raise ValueError("exported a different original report")
    actions = detail["actions"]
    if len(actions) != len(expected["kernels"]):
        raise ValueError("detail export dropped or added a kernel (including fills)")
    for action, original in zip(actions, expected["kernels"]):
        names = (action["name"], action["demangled"])
        if (original["kernel"] not in names or action["capture_id"] != int(original["id"])
                or shape(action["grid"]) != shape(original["grid"])
                or shape(action["block"]) != shape(original["block"])):
            raise ValueError("detail action/symbol/geometry differs from captured CSV")
        metric = action["metrics"].get("ppu__time_duration.sum")
        factors = {"nsecond": .001, "usecond": 1., "msecond": 1000., "second": 1000000.}
        if not metric or metric["unit"] not in factors or not metric["has_value"]:
            raise ValueError("missing native duration/known unit in report API")
        duration = float(metric["value"]) * factors[metric["unit"]]
        if not math.isclose(duration, original["us"], rel_tol=1e-9, abs_tol=1e-5):
            raise ValueError("native API duration differs from original CSV")
        if not action.get("sass") or action.get("coverage", {}).get("warp_executions", 0) <= 0:
            raise ValueError("missing SASS or execution counts, not a successful detail export")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True, help="completed KDA run containing evidence/summary.json and acurep files")
    p.add_argument("--acu", type=Path, default=DEFAULT_ACU)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = (a.out or Path(f"/workspace/actlize-la-gdn-work-details-{stamp}-{os.getpid()}")).resolve()
    if not out.is_relative_to(Path("/workspace")) or out == Path("/workspace"):
        p.error("use a new dedicated /workspace output directory")
    out.mkdir(parents=True, exist_ok=False)
    status = dict(status="INCOMPLETE", created_utc=stamp, exports=[], errors=[],
                  scope="offline opcode/PC/traffic evidence; no compilation, device query or GPU launch",
                  selection="all four geometries, g=-0.1, repeat0, all three roles; not fastest selection")
    try:
        evidence = a.run.resolve(strict=True) / "evidence"
        source = evidence / "summary.json"
        summary = json.loads(source.read_text())
        selected = select_reports(summary)
        layout = profiler_layout(a.acu)
        if layout != summary["profiler"]:
            raise ValueError("selected ACU installation changed since capture; use the recorded installation")
        reader = api_identity(layout["home"])
        status.update(source_run=str(a.run.resolve()), source_summary_sha256=sha(source),
                      captured_source_sha=summary["source_sha"], candidate=summary["candidate"],
                      profiler=layout, reader=reader,
                      exporter_files={n: sha(ROOT / "tools" / n) for n in
                                      ("export_gdn_kda_details.py", "acu_report_details.py")})
        (out / "captured-summary.json").write_bytes(source.read_bytes())
        env = dict(os.environ, ASIGHT_HOME=layout["home"])
        for item in selected:
            cell, role, repeat = item["cell"], item["role"], item["repeat"]
            stem = f"{role}-{repeat}"
            directory = evidence / cell
            admission = json.loads((directory / "admission.json").read_text())
            subject = json.loads((directory / f"{stem}.json").read_text())
            validate_receipt(admission, subject, role)
            observed = inventory(role, parse_raw((directory / f"{stem}.csv").read_text()))
            if (observed["status"] != "ADMITTED" or observed["kernels"] != item["capture"]["kernels"]):
                raise ValueError("original CSV no longer matches captured inventory")
            report = report_file(directory / f"{stem}.report")
            target = out / cell
            target.mkdir(exist_ok=True)
            output, log = target / f"{stem}.json.gz", target / f"{stem}.log"
            command = [sys.executable, str(ROOT / "tools/acu_report_details.py"),
                       "--acu-home", layout["home"], "--report", str(report),
                       "--sha256", item["capture"]["report_sha256"], "--output", str(output)]
            (target / f"{stem}.command.json").write_text(json.dumps(command) + "\n")
            print(f"[GDN work export] {cell} {role} repeat0: existing report only", flush=True)
            with log.open("w") as stream:
                process = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=180)
            if process.returncode:
                raise RuntimeError(f"offline reader rc={process.returncode}; see {log}\n{log.read_text()[-3000:]}")
            with gzip.open(output, "rt") as stream:
                detail = json.load(stream)
            check_detail(detail, item["capture"])
            if detail["api"] != reader:
                raise ValueError("report reader identity changed")
            status["exports"].append(dict(cell=cell, role=role, repeat=repeat,
                path=str(output.relative_to(out)), sha256=sha(output), report_sha256=detail["report_sha256"],
                actions=[dict(name=x["name"], coverage=x["coverage"]) for x in detail["actions"]]))
        if profiler_layout(a.acu) != layout or api_identity(layout["home"]) != reader or sha(source) != status["source_summary_sha256"]:
            raise ValueError("capture manifest or report reader changed")
        if len(status["exports"]) != 12:
            raise ValueError("incomplete diagnostic export denominator")
        status["status"] = "OFFLINE_EXPORTED"
    except Exception as error:
        status["errors"].append(f"{type(error).__name__}: {error}")
        print(f"[GDN work export] INCOMPLETE: {status['errors'][-1]}", file=sys.stderr)
    finally:
        (out / "summary.json").write_text(json.dumps(status, indent=2) + "\n")
        # Only our new text/compressed-text files; never operands, binaries or
        # original reports. Enumerate before opening the archive itself.
        paths = sorted(p for p in out.rglob("*") if p.is_file())
        with tarfile.open(out / "results.tar.gz", "w:gz") as archive:
            for path in paths:
                archive.add(path, arcname=str(path.relative_to(out)), recursive=False)
        print(f"[GDN work export] {status['status']}; upload={out / 'results.tar.gz'}; "
              f"bytes={(out / 'results.tar.gz').stat().st_size}; device_execution=NONE", flush=True)
    return 0 if status["status"] == "OFFLINE_EXPORTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
