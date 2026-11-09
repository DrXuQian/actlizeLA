"""Fixed ZW810 core comparison: shared evidence rules, not another kernel path."""
import itertools
import json
from pathlib import Path
import statistics
import subprocess

from collect_ppu_gdn_acu import sha
from collect_serving_prefill_acu import (
    inventory as existing_inventory, legacy_cells, needs_confirmation,
    read_run, target_verdict, validate_receipts as validate_previous_receipts,
)

PLA_URL = "https://github.com/t-head/pla.git"
PLA_SHA = "b8f72a2cd2be0770e0bac6eeff321533efc4e98a"
PLA_ACTLIZE_SHA = "334d6980a31b40cb97b86480768297bdeacbcba8"
ROLES = ("default", "pla", "fla")
CHUNKS = dict(default=64, pla=32, fla=64)
PLA_KERNELS = (
    # Template formatting is profiler-specific. C32 is separately bound to
    # pinned source + runtime hashes and the imported ops.CHUNK_SIZE value.
    "flashqla_cute::chunk_local_cumsum_cute_kernel<",
    "flashqla_kkt_tc01::kkt_solve_cute_kernel_tc01<",
    "flashqla_cute::fused_fwd_cp_cute_kernel<",
)


def cells():
    return legacy_cells([1], [2048, 8192], [32, 48], [-.1, -1.])


def geometry(cell):
    return {key: cell[key] for key in ("B", "T", "Hk", "Hv", "K", "V")}


def read_inputs(directory):
    """Bind all eight old numerical receipts/operands, never their old timings."""
    directory = Path(directory).resolve(strict=True)
    origin = read_run(directory / "build-run")
    inputs = {}
    identity = None
    for cell in cells():
        path = directory / "acu" / cell["id"]
        previous = json.loads((path / "admission.json").read_text())
        subject = json.loads((path / "default-0.json").read_text())
        validate_previous_receipts(previous, subject, "default")
        if (previous["shape"] != geometry(cell) or previous["gate"] != cell["gate"]
                or previous["library_sha"] != origin["sha256"]
                or sha(path / "operands.pt") != previous["operands_sha"]):
            raise ValueError(f"{cell['id']}: changed input, geometry or admitted binary")
        if identity is not None and previous["device"] != identity:
            raise ValueError("old input receipts mix devices/environments")
        identity = previous["device"]
        inputs[cell["id"]] = dict(directory=str(path), operands_sha=previous["operands_sha"],
                                  admission_sha=sha(path / "admission.json"))
    return dict(origin=origin, inputs=inputs, device=identity)


def git(directory, *args):
    return subprocess.check_output(["git", "-C", str(directory), *args], text=True).strip()


def check_pla_source(source):
    source = Path(source).resolve(strict=True)
    for path, expected in ((source, PLA_SHA), (source / "3rdparty/actlize", PLA_ACTLIZE_SHA)):
        if git(path, "rev-parse", "HEAD") != expected:
            raise ValueError(f"PLA source/submodule differs from pinned commit: {path}")
        if git(path, "status", "--porcelain", "--untracked-files=all"):
            raise ValueError(f"PLA reference must be unmodified: {path}")


def runtime_files(runtime):
    root = Path(runtime).resolve(strict=True)
    files = sorted(p for p in (root / "pla").rglob("*")
                   if p.is_file() and p.suffix in (".py", ".so"))
    if any(p.is_symlink() or not p.resolve().is_relative_to(root) for p in files):
        raise ValueError("PLA runtime contains external source/binary links")
    images = [p for p in files if p.suffix == ".so"]
    if len(images) != 1 or images[0].parent != root / "pla/prefill/flashqla":
        raise ValueError("require exactly one PLA prefill extension, not an installed fallback")
    if not (root / "pla/prefill/flashqla/ops.py").is_file():
        raise ValueError("missing PLA Python entry")
    return {str(p.relative_to(root)): sha(p) for p in files}


def record_pla(source, runtime):
    check_pla_source(source)
    files = runtime_files(runtime)
    for name, digest in files.items():
        if name.endswith(".py") and sha(Path(source) / name) != digest:
            raise ValueError(f"runtime Python file is not the pinned source: {name}")
    return dict(source=str(Path(source).resolve()), source_sha=PLA_SHA,
                actlize_sha=PLA_ACTLIZE_SHA, runtime=str(Path(runtime).resolve()), files=files,
                namespace="pla.prefill.flashqla", chunk=32,
                build="upstream setup.py; unmodified default flags; no global install")


def validate_pla(manifest):
    if (manifest.get("source_sha") != PLA_SHA or manifest.get("actlize_sha") != PLA_ACTLIZE_SHA
            or manifest.get("chunk") != CHUNKS["pla"]):
        raise ValueError("incorrect PLA reference identity")
    if runtime_files(manifest["runtime"]) != manifest["files"]:
        raise ValueError("PLA runtime source or binary changed")


def inventory(role, kernels):
    if role in ("default", "fla"):
        return existing_inventory(role, kernels)
    if role != "pla":
        raise ValueError("unknown three-way role")
    counts = {name: 0 for name in PLA_KERNELS}
    unknown = []
    for row in kernels:
        matches = [name for name in PLA_KERNELS if name in row["kernel"]]
        if len(matches) == 1:
            counts[matches[0]] += 1
        else:
            unknown.append(row["kernel"])
    valid = (not unknown and all(n == 1 for n in counts.values())
             and len({row["process"] for row in kernels}) == 1)
    # Unknown work is retained in the sum, but cannot produce a valid verdict.
    return dict(status="ADMITTED" if valid else "UNADJUDICATED", counts=counts,
                unknown=unknown, total_us=sum(row["us"] for row in kernels), kernels=kernels)


def summarize(captures):
    if set(captures) != set(ROLES) or {len(v) for v in captures.values()} not in ({3}, {6}):
        raise ValueError("incomplete three-role order cycle")
    rows = {role: [c["total_us"] for c in captures[role]] for role in ROLES}
    valid = all(c["status"] == "ADMITTED" for group in captures.values() for c in group)
    comparisons = {}
    for left, right in itertools.combinations(ROLES, 2):
        outcome = "UNADJUDICATED" if not valid else (
            f"{left.upper()}_WINS" if max(rows[left]) < min(rows[right]) else
            f"{right.upper()}_WINS" if max(rows[right]) < min(rows[left]) else "UNRESOLVED")
        comparisons[f"{left}_vs_{right}"] = dict(
            verdict=outcome, ratio=statistics.median(rows[left]) / statistics.median(rows[right]))
    target, ratio = target_verdict(rows["default"], rows["fla"], valid)
    confirm, spread = needs_confirmation(rows, ratio)
    return dict(status="ADMITTED" if valid else "UNADJUDICATED", comparisons=comparisons,
                rule="disjoint-observed-ACU-envelopes", target_default_vs_fla=target,
                chunks=CHUNKS, needs_confirmation=confirm, max_relative_spread=spread,
                rows={role: dict(median_us=statistics.median(v), min_us=min(v), max_us=max(v),
                                 samples_us=v) for role, v in rows.items()},
                scope="all-kernels-of-complete-core-call; ACU-not-uninstrumented-API")


def validate_receipt(admission, subject, role):
    if (admission.get("status") != "PASS" or admission.get("numeric_repeats") != 8
            or set(admission.get("arms", {})) != set(ROLES)):
        raise ValueError("incomplete three-core numerical gate")
    if (subject.get("status") != "PASS" or subject.get("phase") != "subject"
            or subject.get("role") != role or subject.get("public_calls") != 1
            or subject.get("warmup") != 0 or subject.get("verification") != "CPU"):
        raise ValueError("subject is not one verified public call")
    for name in ("device", "shape", "gate", "library_sha", "operands_sha", "input_admission_sha",
                 "canonical_input_sha", "pla", "chunks"):
        if name not in admission or subject.get(name) != admission[name]:
            raise ValueError(f"subject identity changed: {name}")
    expected = admission["arms"][role]
    for name in ("input_sha", "output_sha", "route"):
        if name not in expected or subject.get(name) != expected[name]:
            raise ValueError(f"{role}: subject differs from gate: {name}")


def complete(captures):
    expected = {c["id"] for c in cells()}
    if set(captures) != expected or any("summary" not in c for c in captures.values()):
        raise ValueError("missing registered cell; cannot reduce the eight-cell denominator")
    return ("COLLECTED" if all(c["summary"]["status"] == "ADMITTED" for c in captures.values())
            else "UNADJUDICATED")
