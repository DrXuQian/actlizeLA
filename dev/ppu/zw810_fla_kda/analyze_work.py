#!/usr/bin/env python3
"""Audit executed-work counts before changing GDN parallelism.

Reads an admitted KDA archive as data, never executes uploaded source. Logical
matrix work is a separate C64/K128/V128 source model, NOT measured MMA counts.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics

from analyze_baseline import files, PHASES, phase
import collect_serving_prefill_acu as retained
import gdn_kda_contract as contract

METRIC = "Executed Instructions"
SECTION = "Instruction Statistics"


def logical_work():
    # Dense MACs include the zeros processed inside tensor-core tiles. BF16
    # work and TF32 inverse merges have DIFFERENT atom K extents; don't add
    # their native instruction counts using a single FLOP denominator.
    c, d = 64, 128
    return dict(
        scope="per complete C64 chunk and value head; K=V128; source model only",
        bf16_macs=dict(
            kkt_lower_10_blocks=10 * 16 * 16 * d,
            residual_state=dict(KH=c*d*d, PR=c*c*d, update=c*d*d),
            fla_wu_and_state=dict(W=c*c*d, U=c*c*d, WH=c*d*d, update=c*d*d),
            output=dict(QK=c*c*d, QH=c*d*d, AV=c*c*d)),
        inverse_merge_products=sum(4-gap for gap in range(1, 4)) +
                               sum((4-gap)*gap for gap in range(1, 4)),
        inverse_merge_macs=16 * 16**3,
        retained_tf32_multiplier=3, candidate_tf32_multiplier=1,
        excluded_from_mac_model="diagonal FP32 substitution, elementwise gates, conversion, address, delivery and control")


def work_phase(role, name):
    if role == "solve-tf32x1":
        if "gdn_wy_split_solve_tf32x1" in name:
            return "solve"
        role = "default"
    if role == "parallel-wu":
        if "gdn_wy_parallel_wu_" in name:
            return "state_equivalent"
        role = "default"
    return phase(role, name)


def instruction_rows(text, kernels):
    """Exactly one same-unit count for EVERY launch, including both FLA fills."""
    reader = csv.DictReader(io.StringIO(text[text.index('"ID"'):]))
    counts = {}
    for row in reader:
        if row["Metric Name"] != METRIC:
            continue
        key = (row["Process ID"], row["ID"])
        if key in counts or row["Section Name"] != SECTION or row["Metric Unit"] != "inst":
            raise ValueError("duplicate instruction metric or changed section/unit")
        count = float(row["Metric Value"].replace(",", ""))
        if not math.isfinite(count) or count <= 0 or not count.is_integer():
            raise ValueError("instruction count must be a positive integer")
        counts[key] = dict(kernel=row["Kernel Name"], count=int(count), grid=row["Grid Size"], block=row["Block Size"])
    expected = {(k["process"], str(k["id"])): k["kernel"] for k in kernels}
    if counts.keys() != expected.keys() or any(counts[k]["kernel"] != name for k, name in expected.items()):
        raise ValueError("instruction denominator does not cover the exact full-call inventory")
    return list(counts.values())


def analyze(blob):
    data = files(blob)
    summary = json.loads(data["summary.json"])
    candidate = summary.get("experiment")
    roles = contract.roles(candidate)
    if (summary.get("status") != "COLLECTED" or summary.get("errors") or summary.get("source_diff")
            or summary.get("cells") != contract.cells()
            or set(summary["captures"]) != {c["id"] for c in contract.cells()}):
        raise ValueError("require the complete unchanged eight-cell KDA denominator")
    result = dict(archive_sha256=hashlib.sha256(blob).hexdigest(), source_sha=summary["source_sha"],
                  candidate=candidate, libraries=dict(default=summary["input_run"]["origin"]["sha256"],
                                                     candidate=summary["candidate"]["sha256"]),
                  metric=dict(label=METRIC, section=SECTION, unit="inst",
                              note="not sass__inst_executed_per_opcode; no cross-metric subtraction"),
                  logical_matrix_work=logical_work(), cells=[])
    for cell in contract.cells():
        key = cell["id"]
        saved = summary["captures"][key]
        admission = json.loads(data[key + "/admission.json"])
        if admission["shape"] != contract.geometry(cell) or admission["gate"] != cell["gate"]:
            raise ValueError("shape/gate changed")
        if admission["libraries"] != result["libraries"]:
            raise ValueError("library identity changed")
        counts, rebuilt = {}, {}
        groups = cell["B"] * cell["Hv"] * (cell["T"] // 64)
        if cell["T"] % 64 or cell["K"] != 128 or cell["V"] != 128:
            raise ValueError("logical model is only full C64, K=V128")
        for role in roles:
            counts[role], rebuilt[role] = [], []
            for repeat, original in enumerate(saved["captures"][role]):
                base = f"{key}/{role}-{repeat}"
                contract.validate_receipt(admission, json.loads(data[base + ".json"]), role)
                text = data[base + ".csv"].decode()
                kernels = retained.parse_raw(text)
                observed = contract.inventory(role, kernels)
                if (observed["status"] != "ADMITTED" or kernels != original["kernels"]
                        or not math.isclose(observed["total_us"], original["total_us"], rel_tol=0, abs_tol=1e-8)):
                    raise ValueError("CSV timing/inventory changed")
                rebuilt[role].append(observed)
                rows = instruction_rows(text, kernels)
                phases = {p: sum(r["count"] for r in rows if work_phase(role, r["kernel"]) == p) for p in PHASES}
                if sum(phases.values()) != sum(r["count"] for r in rows):
                    raise ValueError("phase accounting dropped work")
                counts[role].append(dict(total=sum(phases.values()), phases=phases, kernels=rows))
        checked = contract.summarize(rebuilt, candidate)
        if checked["status"] != "ADMITTED":
            raise ValueError("incomplete numerical/timing inventory")
        means = {role: {p: statistics.mean(x["phases"][p] for x in samples) for p in PHASES}
                 for role, samples in counts.items()}
        ratios = {p: means[candidate][p]/means["fla"][p] if means["fla"][p] else None for p in PHASES}
        result["cells"].append(dict(cell=key, logical_chunk_heads=groups, counts=counts, mean_instructions=means,
                                    per_chunk_head={r: {p: v/groups for p,v in m.items()} for r,m in means.items()},
                                    candidate_to_fla=ratios))
    return result


def markdown(result):
    lines = ["# Executed-work audit", "", f"Archive: `{result['archive_sha256']}`; source `{result['source_sha']}`.", "",
             "Same ACU Executed Instructions metric, not a per-opcode sum. All repeats retained. "
             "State-equivalent includes FLA W/U. Both fill kernels remain in the total.", "",
             "| cell | solve ours/FLA | state-equivalent ours/FLA | output ours/FLA | total ours/FLA |",
             "|---|---:|---:|---:|---:|"]
    for c in result["cells"]:
        ours, fla = (c["mean_instructions"][r] for r in (result["candidate"], "fla"))
        values = [f"{ours[p]:,.0f}/{fla[p]:,.0f} ({ours[p]/fla[p]:.3f}x)" for p in ("solve", "state_equivalent", "output")]
        values.append(f"{sum(ours.values()):,.0f}/{sum(fla.values()):,.0f} ({sum(ours.values())/sum(fla.values()):.3f}x)")
        lines.append("| " + " | ".join([c["cell"], *values]) + " |")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    result = analyze(a.archive.read_bytes())
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / "work.json").write_text(json.dumps(result, indent=2) + "\n")
    (a.out / "work.md").write_text(markdown(result))
    print(markdown(result))


if __name__ == "__main__":
    main()
