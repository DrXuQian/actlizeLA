"""Round-one inventory and receipts; existing numerical/ACU rules stay authoritative."""
import itertools
import statistics

from gdn_threeway_contract import cells, geometry
from collect_serving_prefill_acu import CORE, inventory as retained_inventory, needs_confirmation, target_verdict

ROLES = ("default", "parallel-wu", "fla")
CANDIDATE_KERNELS = CORE[:2] + ("gdn_wy_parallel_wu_materialize", "gdn_wy_parallel_wu_state") + CORE[3:]
ENTRY = "actlize_la_ppu10_parallel_wu_forward_prepared"


def inventory(role, kernels):
    if role in ("default", "fla"):
        return retained_inventory(role, kernels)
    if role != "parallel-wu":
        raise ValueError("unknown KDA role")
    counts = {name: 0 for name in CANDIDATE_KERNELS}
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


def summarize(captures):
    if set(captures) != set(ROLES) or {len(v) for v in captures.values()} not in ({3}, {6}):
        raise ValueError("incomplete three-role KDA order cycle")
    values = {r: [x["total_us"] for x in captures[r]] for r in ROLES}
    valid = all(x["status"] == "ADMITTED" for group in captures.values() for x in group)
    comparisons = {}
    for a, b in itertools.combinations(ROLES, 2):
        verdict = ("UNADJUDICATED" if not valid else a + "_WINS" if max(values[a]) < min(values[b])
                   else b + "_WINS" if max(values[b]) < min(values[a]) else "UNRESOLVED")
        comparisons[a + "_vs_" + b] = dict(verdict=verdict, ratio=statistics.median(values[a]) / statistics.median(values[b]))
    target, ratio = target_verdict(values["parallel-wu"], values["fla"], valid)
    confirm, spread = needs_confirmation(values, ratio)
    # An unresolved primary candidate/incumbent comparison gets a full second
    # order cycle too; retain every sample, never cherry-pick a fast repeat.
    confirm |= any(v["verdict"] == "UNRESOLVED" for v in comparisons.values())
    return dict(status="ADMITTED" if valid else "UNADJUDICATED", comparisons=comparisons,
                target_candidate_vs_fla=target, needs_confirmation=confirm, max_relative_spread=spread,
                rows={r: dict(median_us=statistics.median(v), min_us=min(v), max_us=max(v), samples_us=v)
                      for r, v in values.items()}, scope="complete-core ACU kernel sums; all WU work included")


def validate_receipt(admission, subject, role):
    if (admission.get("status") != "PASS" or admission.get("numeric_repeats") != 8
            or admission.get("unchanged_public_default_bits") != "PASS"
            or set(admission.get("arms", {})) != set(ROLES)):
        raise ValueError("incomplete KDA numerical gate")
    if (subject.get("status") != "PASS" or subject.get("phase") != "subject" or subject.get("role") != role
            or subject.get("public_calls") != 1 or subject.get("warmup") != 0 or subject.get("verification") != "CPU"):
        raise ValueError("subject must be exactly one verified full core call")
    for name in ("device", "shape", "gate", "libraries", "operands_sha", "input_admission_sha", "canonical_input_sha"):
        if name not in admission or subject.get(name) != admission[name]:
            raise ValueError("subject identity changed: " + name)
    for name in ("input_sha", "output_sha", "route"):
        if subject.get(name) != admission["arms"][role][name]:
            raise ValueError("subject differs from numerical admission: " + name)


def complete(captures):
    if set(captures) != {c["id"] for c in cells()}:
        raise ValueError("do not reduce the registered eight-cell denominator")
    return "COLLECTED" if all(c.get("summary", {}).get("status") == "ADMITTED" for c in captures.values()) else "UNADJUDICATED"
