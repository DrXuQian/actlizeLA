"""Bounded KDA inventories; existing numerical/ACU rules stay authoritative."""
import itertools
import csv
import io
import math
import statistics

from gdn_threeway_contract import cells, geometry
from collect_serving_prefill_acu import CORE, inventory as retained_inventory, needs_confirmation, target_verdict

ROLES = ("default", "parallel-wu", "fla")
CANDIDATE_KERNELS = CORE[:2] + ("gdn_wy_parallel_wu_materialize", "gdn_wy_parallel_wu_state") + CORE[3:]
ENTRY = "actlize_la_ppu10_parallel_wu_forward_prepared"
EXPERIMENTS = {
    "native-bf16": dict(entry="actlize_la_ppu10_native_bf16_forward_prepared",
                        configure="actlize_la_ppu10_native_bf16_configure",
                        arithmetic="retained-residual-TF32x3-native-BF16-same-rounding",
                        kernels=(CORE[0], "gdn_wy_native_bf16_solve", "gdn_wy_native_bf16_state", "gdn_wy_native_bf16_output"),
                        option="ALA_PPU10_NATIVE_BF16_EXPERIMENT", checker="check_native_bf16.py",
                        raw_bits=True, probe="native_bf16_device.py", sources=("dev/ppu/generate_native_bf16.py",),
                        generated_manifest="native_bf16/sources.json"),
    "parallel-wu": dict(entry=ENTRY, configure="actlize_la_ppu10_parallel_wu_configure",
                        arithmetic="materialized-WU-BF16", kernels=CANDIDATE_KERNELS,
                        option="ALA_PPU10_PARALLEL_WU_EXPERIMENT", checker="check_parallel_wu.py"),
    "solve-tf32x1": dict(entry="actlize_la_ppu10_solve_tf32x1_forward_prepared",
                         configure="actlize_la_ppu10_solve_tf32x1_configure",
                         arithmetic="residual-single-TF32-inverse",
                         kernels=(CORE[0], "gdn_wy_split_solve_tf32x1", *CORE[2:]),
                         option="ALA_PPU10_SOLVE_TF32X1_EXPERIMENT", checker="check_solve_tf32x1.py"),
}

INSTRUCTION_METRIC = "Executed Instructions"
INSTRUCTION_SECTION = "Instruction Statistics"


def instruction_rows(text, kernels, metric_id=None):
    """One same-unit count per captured launch, including EACH FLA fill."""
    reader = csv.DictReader(io.StringIO(text[text.index('"ID"'):]))
    long_format = "Metric Name" in (reader.fieldnames or ())
    if not long_format and (not metric_id or metric_id not in (reader.fieldnames or ())):
        raise ValueError("wide CSV requires the captured instruction metric binding")
    counts = {}
    for row in reader:
        if long_format and row["Metric Name"] != INSTRUCTION_METRIC:
            continue
        key = (row["Process ID"], row["ID"])
        if key in counts or (long_format and (row["Section Name"] != INSTRUCTION_SECTION or row["Metric Unit"] != "inst")):
            raise ValueError("duplicate instruction metric or changed section/unit")
        count = float(row["Metric Value" if long_format else metric_id].replace(",", ""))
        if not math.isfinite(count) or count <= 0 or not count.is_integer():
            raise ValueError("instruction count must be a positive integer")
        counts[key] = dict(kernel=row["Kernel Name"], count=int(count), grid=row["Grid Size"], block=row["Block Size"])
    expected = {(k["process"], str(k["id"])): k["kernel"] for k in kernels}
    if counts.keys() != expected.keys() or any(counts[k]["kernel"] != name for k, name in expected.items()):
        raise ValueError("instruction denominator does not cover the exact full-call inventory")
    return [counts[k] for k in expected]


def roles(candidate="parallel-wu"):
    if candidate not in EXPERIMENTS:
        raise ValueError("unknown KDA experiment")
    return ("default", candidate, "fla")


def inventory(role, kernels):
    if role in ("default", "fla"):
        return retained_inventory(role, kernels)
    if role not in EXPERIMENTS:
        raise ValueError("unknown KDA role")
    counts = {name: 0 for name in EXPERIMENTS[role]["kernels"]}
    unknown = []
    for row in kernels:
        matches = [name for name in counts if name in row["kernel"]]
        if len(matches) == 1:
            counts[matches[0]] += 1
        else:
            unknown.append(row["kernel"])
    valid = not unknown and all(v == 1 for v in counts.values()) and len({r["process"] for r in kernels}) == 1
    return dict(status="ADMITTED" if valid else "UNADJUDICATED", counts=counts, unknown=unknown,
                total_us=sum(r["us"] for r in kernels), kernels=kernels)


def summarize(captures, candidate="parallel-wu"):
    selected = roles(candidate)
    if set(captures) != set(selected) or {len(v) for v in captures.values()} not in ({3}, {6}):
        raise ValueError("incomplete three-role KDA order cycle")
    values = {r: [x["total_us"] for x in captures[r]] for r in selected}
    valid = all(x["status"] == "ADMITTED" for group in captures.values() for x in group)
    comparisons = {}
    for a, b in itertools.combinations(selected, 2):
        verdict = ("UNADJUDICATED" if not valid else a + "_WINS" if max(values[a]) < min(values[b])
                   else b + "_WINS" if max(values[b]) < min(values[a]) else "UNRESOLVED")
        comparisons[a + "_vs_" + b] = dict(verdict=verdict, ratio=statistics.median(values[a]) / statistics.median(values[b]))
    target, ratio = target_verdict(values[candidate], values["fla"], valid)
    confirm, spread = needs_confirmation(values, ratio)
    # An unresolved primary candidate/incumbent comparison gets a full second
    # order cycle too; retain every sample, never cherry-pick a fast repeat.
    confirm |= any(v["verdict"] == "UNRESOLVED" for v in comparisons.values())
    return dict(status="ADMITTED" if valid else "UNADJUDICATED", comparisons=comparisons,
                target_candidate_vs_fla=target, needs_confirmation=confirm, max_relative_spread=spread,
                rows={r: dict(median_us=statistics.median(v), min_us=min(v), max_us=max(v), samples_us=v)
                      for r, v in values.items()}, scope=("complete-core ACU kernel sums; all WU work included"
                      if candidate == "parallel-wu" else "complete-core ACU kernel sums; native BF16, unchanged rounding/TF32x3"
                      if candidate == "native-bf16" else "complete-core ACU kernel sums; single-TF32 inverse declared"))


def validate_receipt(admission, subject, role):
    candidates = set(admission.get("arms", {})) & set(EXPERIMENTS)
    if len(candidates) != 1:
        raise ValueError("missing/ambiguous numerical experiment")
    candidate = next(iter(candidates))
    selected = roles(candidate)
    if admission.get("candidate_kind", candidate) != candidate:
        raise ValueError("admission arithmetic identity disagrees with its arms")
    if (admission.get("status") != "PASS" or admission.get("numeric_repeats") != 8
            or admission.get("unchanged_public_default_bits") != "PASS"
            or set(admission.get("arms", {})) != set(selected) or role not in selected):
        raise ValueError("incomplete KDA numerical gate")
    if EXPERIMENTS[candidate].get("raw_bits") and admission.get("candidate_default_raw_bits") != "PASS":
        raise ValueError("native conversion candidate requires default raw-bit equality")
    if (subject.get("status") != "PASS" or subject.get("phase") != "subject" or subject.get("role") != role
            or subject.get("public_calls") != 1 or subject.get("warmup") != 0 or subject.get("verification") != "CPU"):
        raise ValueError("subject must be exactly one verified full core call")
    for name in ("device", "shape", "gate", "libraries", "operands_sha", "input_admission_sha", "canonical_input_sha"):
        if name not in admission or subject.get(name) != admission[name]:
            raise ValueError("subject identity changed: " + name)
    for name in ("input_sha", "output_sha", "route"):
        if subject.get(name) != admission["arms"][role][name]:
            raise ValueError("subject differs from numerical admission: " + name)
    if subject.get("candidate_kind") != admission.get("candidate_kind"):
        raise ValueError("subject changed the candidate arithmetic identity")


def complete(captures):
    if set(captures) != {c["id"] for c in cells()}:
        raise ValueError("do not reduce the registered eight-cell denominator")
    return "COLLECTED" if all(c.get("summary", {}).get("status") == "ADMITTED" for c in captures.values()) else "UNADJUDICATED"
