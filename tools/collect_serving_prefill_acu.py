#!/usr/bin/env python3
"""Bounded three-arm prefill capture; reuse a passed library, never compile."""
import argparse
from collections import deque
import csv
from datetime import datetime, timezone
import io
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tarfile

from collect_ppu_gdn_acu import DEFAULT_ACU, ROOT, report_file, run as run_process, sha

ROLES = ("serving", "default", "fla")
CORE = ("gdn_wy_split_prefix", "gdn_wy_split_solve_static",
        "gdn_wy_residual_full_chunk_state", "gdn_wy_residual_warps8_hvlayout_output")
TRANSPORT = ("validate_gdn", "prepare_gdn", "pack_prefill", "gather_state", "scatter_prefill")
FLA = ("chunk_local_cumsum_scalar_kernel", "chunk_gated_delta_rule_fwd_kkt_solve_kernel",
       "recompute_w_u_fwd_kernel", "chunk_gated_delta_rule_fwd_kernel_h", "chunk_fwd_kernel_o")
# Registered acceptance: time reduction of at least 20% (speedup >= 1.25x), not 1.20x.
TARGET_RATIO = 0.80
# Pre-registered confirmation rule (manifest/shapes.json): near-threshold or noisy
# three-sample cells get three more captures per role; every sample is kept.
CONFIRM_RATIO = (0.78, 0.82)
CONFIRM_SPREAD = 0.05
RETAINED_PAIR_HEADS = (32, 48, 64)
# Must equal benchmarks/profile_serving_prefill.py (host test enforces it).
BATCHES = (1, 2, 4)
LENGTHS = (512, 2048, 4096, 8192, 16384)
VALUE_HEADS = (32, 48, 64)
GATES = (-.1, -1.)


def run(command, log, env, **kwargs):
    try:
        return run_process(command, log, env, **kwargs)
    except Exception:
        if log.is_file():
            with log.open(errors="replace") as stream:
                print(f"[prefill ACU] failed command log: {log}\n" +
                      "".join(deque(stream, maxlen=60)), file=sys.stderr, flush=True)
        raise


def profiler_layout(acu):
    """A fixed executable must not inherit another installation's sections."""
    executable = Path(acu).resolve(strict=True)
    home = executable.parent.parent
    sections = home / "sections"
    files = sorted(sections.glob("*.section"))
    if not files:
        raise ValueError(f"selected ACU has no companion sections: {sections}")
    return dict(executable=str(executable), executable_sha256=sha(executable),
                home=str(home), sections=str(sections),
                section_sha256={p.name: sha(p) for p in files})


def validate_default_route(route, library, digest):
    expected = dict(api="gdn_forward", backend="ppu10", algorithm="residual",
                    delivery="full-chunk", native_entry="actlize_la_ppu10_forward_prepared",
                    native_provider="explicit-serving", output_final_state=True,
                    state_layout="B,Hv,K,V", final_state_dtype="fp32",
                    native_status="LOADED_FILES_VERIFIED_NOT_DEVICE_ADMITTED")
    if any(route.get(key) != value for key, value in expected.items()):
        raise ValueError("public default did not select the retained PPU1.0 serving core")
    images = route.get("native_images", [])
    if (len(images) != 1 or images[0].get("path") != str(Path(library).resolve())
            or images[0].get("sha256") != digest):
        raise ValueError("public default did not load the admitted library")


def read_run(directory):
    """Require the latest full gate's actual receipts, not just a folder name."""
    directory = directory.resolve(strict=True)
    binary = json.loads((directory / "binary.json").read_text())
    native = json.loads((directory / "results.json").read_text())
    aux = json.loads((directory / "aux-results.json").read_text())
    library = Path(binary["library"]).resolve(strict=True)
    if binary["target"] != "ppu10" or not library.is_relative_to(directory):
        raise ValueError("require a PPU1.0 library in this admitted run")
    digest = sha(library)
    if binary["sha256"] != digest or native.get("library_sha256") != digest or aux.get("binary_sha256") != digest:
        raise ValueError("passed receipts do not bind this exact library")
    for receipt in (native, aux):
        if receipt.get("status") != "PASS" or receipt.get("target") != "ppu10":
            raise ValueError("serving AND auxiliary device gates must have passed")
    pairs = [c for c in native["cases"] if c.get("case") == "ppu10-prefill-retained-pair"]
    expected = {(n, f"16:{h}") for n in (64, 65, 129, 8192) for h in RETAINED_PAIR_HEADS}
    if (len(pairs) != len(expected) or {(c["length"], c["heads"]) for c in pairs} != expected
            or any(c.get("output_state") != "RAW-BIT/PASS" for c in pairs)):
        raise ValueError("incomplete retained/prepared numerical denominator")
    checked = [c for c in aux["cases"] if c.get("case") == "gdn-checked-prefill"]
    if len(checked) != 1 or checked[0].get("checked_vs_unchecked") != "RAW-BIT/PASS":
        raise ValueError("missing corrected checked/unchecked prefill gate")
    if "[FlashNext native] PASS replays=8 fixed-addresses kernel_nodes=1" not in (
            directory / "native-graph.log").read_text():
        raise ValueError("native graph gate did not finish")
    return dict(directory=str(directory), library=str(library), sha256=digest,
                binary_source_sha=binary["source_sha"],
                device_receipts={name: sha(directory / name) for name in
                                 ("binary.json", "results.json", "aux-results.json", "native-graph.log")})


def select_run(explicit=None, workspace=Path("/workspace")):
    if explicit is not None:
        return read_run(explicit)
    valid, rejected = [], []
    for directory in sorted(workspace.glob("actlize-la-flare-ppu10-*")):
        try:
            valid.append(read_run(directory))
        except (ValueError, KeyError, OSError, TypeError) as error:
            rejected.append(f"{directory}: {error}")
    if len(valid) != 1:
        raise ValueError(f"expected exactly one passed serving run; found {len(valid)}. "
                         "Set --serving-run explicitly. " + json.dumps(dict(valid=valid, rejected=rejected)))
    return valid[0]


def parse_raw(text):
    """Native duration from wide counters or the site's long metric rows.

    SpeedOfLight.section labels ppu__time_duration.sum as Duration. Restrict
    that alias to its SOL section and require an explicit time unit. Never
    substitute active cycles, count each metric row as a launch, or drop work.
    """
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if '"ID"' in line
                  and '"Kernel Name"' in line and
                  ('"ppu__time_duration.sum"' in line or '"Metric Name"' in line)), None)
    if start is None:
        raise ValueError("ACU raw export lacks native ppu__time_duration.sum; no timing verdict")
    reader = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    if "Metric Name" in reader.fieldnames:
        return _parse_metric_rows(reader)
    result, seen = [], set()
    for row in reader:
        if not row.get("ID") or not row["ID"].strip().isdigit():
            raise ValueError("unexpected/non-numeric launch row in raw CSV")
        key = (row.get("Process ID"), row["ID"])
        if key in seen:
            raise ValueError("duplicate launch ID; cannot silently double-count replay")
        seen.add(key)
        duration = float(row["ppu__time_duration.sum"].replace(",", "")) / 1000
        if not math.isfinite(duration) or duration <= 0 or not row.get("Kernel Name"):
            raise ValueError("missing/invalid kernel duration")
        result.append(dict(id=row["ID"], process=row.get("Process ID"), kernel=row["Kernel Name"],
                           grid=row.get("Grid Size"), block=row.get("Block Size"), us=duration))
    if not result:
        raise ValueError("empty ACU capture")
    return result


def _parse_metric_rows(reader):
    units = {"nsecond": .001, "ns": .001, "usecond": 1., "us": 1.,
             "µsecond": 1., "µs": 1., "μs": 1., "msecond": 1000.,
             "ms": 1000., "second": 1e6, "s": 1e6}
    rows = {}
    metadata = ("Kernel Name", "Grid Size", "Block Size", "Device", "Context", "Stream")
    required = {"ID", "Process ID", *metadata, "Section Name", "Metric Name",
                "Metric Unit", "Metric Value"}
    if not required.issubset(reader.fieldnames):
        raise ValueError("incomplete long-form ACU duration schema")
    for row in reader:
        if None in row or any(row.get(field) is None for field in required):
            raise ValueError("malformed ACU metric row")
        if not row["ID"].isdigit() or not row["Process ID"].isdigit() or not row["Kernel Name"]:
            raise ValueError("unexpected/non-numeric launch row in raw CSV")
        key = (row["Process ID"], row["ID"])
        identity = tuple(row[field] for field in metadata)
        if key not in rows:
            rows[key] = dict(identity=identity, duration=None)
        launch = rows[key]
        if identity != launch["identity"]:
            raise ValueError("inconsistent metadata for ACU launch ID")
        native = row["Metric Name"] == "ppu__time_duration.sum"
        alias = row["Metric Name"] == "Duration" and row["Section Name"] == "PPU Speed Of Light Throughput"
        if native or alias:
            if launch["duration"] is not None:
                raise ValueError("duplicate duration for ACU launch ID")
            if row["Metric Unit"] not in units:
                raise ValueError("unknown duration unit in ACU metric rows")
            duration = float(row["Metric Value"].replace(",", "")) * units[row["Metric Unit"]]
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError("invalid kernel duration")
            launch["duration"] = duration
    if not rows or any(row["duration"] is None for row in rows.values()):
        raise ValueError("missing duration for one or more ACU launches")
    return [dict(id=key[1], process=key[0], kernel=row["identity"][0],
                 grid=row["identity"][1], block=row["identity"][2], us=row["duration"])
            for key, row in rows.items()]


def inventory(role, kernels):
    if role not in ROLES:
        raise ValueError("unknown capture role")
    required = CORE + TRANSPORT if role == "serving" else CORE if role == "default" else FLA
    counts = {name: 0 for name in required}
    unknown, helpers, core = [], 0., 0.
    for row in kernels:
        matches = [name for name in required if name in row["kernel"]]
        if len(matches) == 1:
            name = matches[0]
            counts[name] += 1
            if name in TRANSPORT:
                helpers += row["us"]
            else:
                core += row["us"]
        elif role == "fla" and "FillFunctor<" in row["kernel"]:
            # FLA-owned zero-fill IS part of its complete core call, not overhead
            # to subtract. Unknown additional kernels remain unadjudicated.
            core += row["us"]
        else:
            unknown.append(row["kernel"])
    valid = not unknown and all(count == 1 for count in counts.values())
    valid = valid and len({row["process"] for row in kernels}) == 1
    return dict(status="ADMITTED" if valid else "UNADJUDICATED", counts=counts, unknown=unknown,
                total_us=sum(row["us"] for row in kernels), transport_us=helpers,
                core_in_same_call_us=core, kernels=kernels)


def target_verdict(default, fla, valid):
    """Registered per-cell acceptance; never inferred from a single winner sample.

    TARGET_MET requires BOTH median(default) <= 0.80 * median(fla) AND disjoint
    envelopes. A passing median with overlapping envelopes stays UNRESOLVED; a
    failing median is TARGET_NOT_MET with its gap reported, never dropped.
    """
    if not valid:
        return "UNADJUDICATED", None
    ratio = statistics.median(default) / statistics.median(fla)
    if ratio > TARGET_RATIO:
        return "TARGET_NOT_MET", ratio
    return ("TARGET_MET" if max(default) < min(fla) else "UNRESOLVED"), ratio


def needs_confirmation(rows, ratio):
    spread = max((max(v) - min(v)) / statistics.median(v) for v in rows.values())
    near = ratio is not None and CONFIRM_RATIO[0] <= ratio <= CONFIRM_RATIO[1]
    return bool(near or spread > CONFIRM_SPREAD), spread


def summarize(captures, repeats):
    if repeats < 3 or repeats % 3:
        raise ValueError("need a complete three-role order cycle")
    for role in ROLES:
        if len(captures.get(role, [])) != repeats:
            raise ValueError("missing capture role/repetition")
    rows = {role: [c["total_us"] for c in captures[role]] for role in ROLES}
    valid = all(c["status"] == "ADMITTED" for role in ROLES for c in captures[role])
    if not valid:
        verdict = "UNADJUDICATED"
    elif max(rows["default"]) < min(rows["fla"]):
        verdict = "DEFAULT_CORE_WINS"
    elif max(rows["fla"]) < min(rows["default"]):
        verdict = "FLA_CORE_WINS"
    else:
        verdict = "UNRESOLVED"
    target, ratio = target_verdict(rows["default"], rows["fla"], valid)
    confirm, spread = needs_confirmation(rows, ratio)
    return dict(core_verdict=verdict, rule="disjoint-observed-ACU-envelopes",
                target_verdict=target, target_rule="median(default)<=0.80*median(fla) AND max(default)<min(fla)",
                median_ratio_default_over_fla=ratio,
                speedup_fla_over_default=(1 / ratio if ratio else None),
                gap_to_target_us=(statistics.median(rows["default"]) - TARGET_RATIO * statistics.median(rows["fla"])),
                needs_confirmation=confirm, max_relative_spread=spread, repeats=repeats,
                serving_vs_fla="NOT_COMPARABLE_RAW_SLOT_VS_PROCESSED_CORE",
                rows={role: dict(median_us=statistics.median(values), min_us=min(values), max_us=max(values),
                                 samples_us=values) for role, values in rows.items()},
                serving_transport_us=[c["transport_us"] for c in captures["serving"]],
                scope="ACU-profiled-kernel-sum-not-uninstrumented-API-latency")


def validate_receipts(admission, subject, role):
    if (admission.get("status") != "PASS" or admission.get("numeric_repeats") != 8
            or admission.get("serving_default_bits") != "PASS" or set(admission.get("arms", {})) != set(ROLES)):
        raise ValueError("incomplete three-arm numerical admission")
    if (subject.get("status") != "PASS" or subject.get("phase") != "subject"
            or subject.get("role") != role or subject.get("public_calls") != 1 or subject.get("warmup") != 0):
        raise ValueError("subject is not exactly one admitted public call")
    for key in ("device", "library_sha", "shape", "measured_shape", "gate", "operands_sha"):
        if key not in admission or subject.get(key) != admission[key]:
            raise ValueError(f"capture identity mismatch: {key}")
    for key in ("input_sha", "output_sha"):
        if not admission["arms"][role].get(key) or subject.get(key) != admission["arms"][role][key]:
            raise ValueError(f"capture differs from numerical admission: {key}")
    if (admission["arms"]["default"]["input_sha"] != admission["arms"]["fla"]["input_sha"]
            or admission["arms"]["serving"]["output_sha"] != admission["arms"]["default"]["output_sha"]):
        raise ValueError("lost same-input core or RAW-BIT serving/default binding")
    route = admission.get("default_route", {})
    images = route.get("native_images", [])
    if len(images) != 1:
        raise ValueError("missing default image admission")
    validate_default_route(route, images[0]["path"], admission["library_sha"])
    if role == "default" and subject.get("default_route") != route:
        raise ValueError("subject default route changed")


def core_input_run(directory, length=8192):
    """Reuse one successful core capture's operands, never regenerate inputs."""
    directory = Path(directory).resolve(strict=True)
    admission = json.loads((directory / "admission.json").read_text())
    expected = dict(B=1, T=length, Hk=16, Hv=48, K=128, V=128)
    if admission.get("shape") != expected or admission.get("gate") != -.1:
        raise ValueError("require the requested Flash-Next T/Hk16/Hv48/g=-0.1 admission")
    subject = json.loads((directory / "core.json").read_text())
    validate_receipts(admission, subject, "default")
    operands = directory / "operands.pt"
    library = Path(admission["default_route"]["native_images"][0]["path"]).resolve(strict=True)
    if sha(operands) != admission["operands_sha"] or sha(library) != admission["library_sha"]:
        raise ValueError("saved core operands or admitted library changed")
    return dict(directory=str(directory), library=str(library), shape=expected,
                operands_sha=admission["operands_sha"], library_sha=admission["library_sha"])


def select_core_input_run(explicit=None, *, length=8192, workspace=Path("/workspace")):
    if explicit is not None:
        return core_input_run(explicit, length)
    valid, rejected = [], []
    for directory in sorted(workspace.glob("flashnext-gdn-core-*")):
        try:
            valid.append(core_input_run(directory, length))
        except (ValueError, KeyError, OSError, TypeError) as error:
            rejected.append(f"{directory}: {error}")
    if len(valid) != 1:
        raise ValueError("need exactly one matching prior core input run; set INPUT_RUN explicitly. "
                         + json.dumps(dict(valid=valid, rejected=rejected)))
    return valid[0]


def small_archive(out):
    archive = out / "results.tar.gz"
    files = []
    for path in sorted(out.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"refuse to archive a source link: {path}")
        if path.is_file() and path.suffix in (".json", ".log", ".csv", ".command", ".py", ".txt"):
            files.append(path)
    with tarfile.open(archive, "w:gz", dereference=False) as tar:
        for path in files:
            tar.add(path, arcname=path.relative_to(out), recursive=False)
    return archive


def registered_cells(manifest, phase="1", ids=None, expected_sha=None):
    """Cells from the hash-registered matrix; an edited denominator is red."""
    manifest = Path(manifest).resolve(strict=True)
    digest = sha(manifest)
    if expected_sha is not None and digest != expected_sha:
        raise ValueError("registered shape manifest changed; a new denominator needs a new registration")
    cells = json.loads(manifest.read_text())["cells"]
    if ids:
        known = {c["id"] for c in cells}
        if set(ids) - known or len(ids) != len(set(ids)):
            raise ValueError("requested cells must be unique registered ids")
        cells = [c for c in cells if c["id"] in ids]
    elif phase != "all":
        cells = [c for c in cells if str(c["phase"]) == str(phase)]
    if not cells:
        raise ValueError("no registered cells selected")
    return [dict(id=c["id"], B=c["B"], T=c["T"], Hk=c["Hk"], Hv=c["Hv"], K=c["K"], V=c["V"], gate=c["gate"])
            for c in cells], digest


def legacy_cells(batches, lengths, value_heads, gates):
    cells = []
    for b, t, hv, gate in itertools.product(batches, lengths, value_heads, gates):
        key = f"T{t}-Hk16-Hv{hv}-g{gate}" if b == 1 else f"B{b}-T{t}-Hk16-Hv{hv}-g{gate}"
        cells.append(dict(id=key, B=b, T=t, Hk=16, Hv=hv, K=128, V=128, gate=gate))
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serving-run", type=Path, default=os.environ.get("SERVING_RUN"))
    parser.add_argument("--manifest", type=Path, help="hash-registered shape matrix (manifest/shapes.json)")
    parser.add_argument("--manifest-sha256", default=os.environ.get("MANIFEST_SHA256"))
    parser.add_argument("--manifest-phase", choices=("1", "2", "all"), default="1")
    parser.add_argument("--cells", nargs="+", help="explicit registered cell ids")
    parser.add_argument("--batches", type=int, nargs="+", choices=BATCHES, default=[1])
    parser.add_argument("--lengths", type=int, nargs="+", choices=LENGTHS, default=[2048, 8192])
    parser.add_argument("--value-heads", type=int, nargs="+", choices=VALUE_HEADS, default=[32])
    parser.add_argument("--gates", type=float, nargs="+", choices=GATES, default=[-.1])
    parser.add_argument("--repeats", type=int, choices=(3, 6), default=3)
    parser.add_argument("--no-confirm", action="store_true",
                        help="disable the pre-registered near-threshold/noise confirmation round")
    parser.add_argument("--acu", type=Path, default=os.environ.get("ACU"))
    parser.add_argument("--device", default=os.environ.get("DEVICE", "0"))
    args = parser.parse_args()
    if not args.device.isdigit():
        parser.error("device must be an explicit nonnegative physical ordinal")
    for values in (args.batches, args.lengths, args.value_heads, args.gates):
        if len(values) != len(set(values)):
            parser.error("duplicate shape/gate is not an independent cell")
    acu = Path(args.acu) if args.acu else DEFAULT_ACU
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(os.environ.get("OUT", f"/workspace/actlize-la-prefill-acu-{stamp}-{os.getpid()}"))
    out.mkdir(parents=True, exist_ok=False)
    sdk = Path(os.environ.get("PPU_SDK", os.environ.get("PPU_SDK_ROOT", "/usr/local/PPU_SDK")))
    env = os.environ | dict(CUDA_VISIBLE_DEVICES=args.device, PYTHONUNBUFFERED="1",
                            LD_LIBRARY_PATH=f"{sdk}/lib:{sdk}/lib64:" + os.environ.get("LD_LIBRARY_PATH", ""),
                            PYTHONPATH=":".join(filter(None, (str(ROOT), str(ROOT / "tests"),
                                os.environ.get("FLA_ROOT"), os.environ.get("PYTHONPATH")))))
    status = dict(status="INCOMPLETE", created_utc=stamp, captures={}, device=args.device,
                  intended_cells=[], repeats=args.repeats, errors=[], sdk=str(sdk), acu=str(acu),
                  target_ratio=TARGET_RATIO, confirm=dict(enabled=not args.no_confirm, ratio=CONFIRM_RATIO,
                                                          spread=CONFIRM_SPREAD))
    try:
        if args.manifest is not None:
            cells, status["manifest_sha256"] = registered_cells(
                args.manifest, args.manifest_phase, args.cells, args.manifest_sha256)
            status["manifest"] = str(args.manifest)
        elif args.cells:
            raise ValueError("--cells requires the registered --manifest")
        else:
            cells = legacy_cells(args.batches, args.lengths, args.value_heads, args.gates)
        status["intended_cells"] = cells
        if env.get("FLA_ROOT") and not (Path(env["FLA_ROOT"]) / "fla/__init__.py").is_file():
            raise ValueError("FLA_ROOT is not a valid FLA checkout; no silent installed-package fallback")
        origin = select_run(args.serving_run)
        status["binary_origin"] = origin
        if "GDN_QSA_WY_EXTENSION" in env:
            raise ValueError("unset GDN_QSA_WY_EXTENSION; measuring the serving-backed public default")
        if ("ACTLIZE_LA_SERVING_LIBRARY" in env and
                Path(env["ACTLIZE_LA_SERVING_LIBRARY"]).resolve() != Path(origin["library"])):
            raise ValueError("ACTLIZE_LA_SERVING_LIBRARY differs from the selected passed-run library")
        previous = out / "passed-run"
        previous.mkdir()
        for name in origin["device_receipts"]:
            shutil.copyfile(Path(origin["directory"]) / name, previous / name)
        status["harness_sha"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        status["source_diff"] = subprocess.check_output(["git", "diff", "HEAD", "--stat"], cwd=ROOT, text=True)
        if status["source_diff"]:
            raise ValueError("capture helpers must be committed before box measurement")
        status["profiler"] = profiler_layout(acu)
        status["inherited_asight_home"] = env.get("ASIGHT_HOME")
        env["ASIGHT_HOME"] = status["profiler"]["home"]
        section_flags = ["--section-folder", status["profiler"]["sections"]]
        status["acu_sha256"] = status["profiler"]["executable_sha256"]
        run([acu, "--version"], out / "acu-version.log", env, console=False)
        # Parse the unchanged full-set section schema BEFORE fixture/reference
        # work. A bad tool installation stays FAIL, never a reduced-metric PASS.
        run([acu, *section_flags, "--set", "full", "--list-sections"],
            out / "acu-section-check.log", env, console=False)
        smi = shutil.which("ppu-smi")
        if smi:
            status["driver_before"] = run([smi, "-i", args.device, "-q"], out / "device-before.log",
                                          env, optional=True, console=False, timeout=20)
        else:
            status["driver_before"] = dict(status="UNAVAILABLE", reason="ppu-smi not on PATH")
        for spec in cells:
            key = spec["id"]
            expected_shape = {name: spec[name] for name in ("B", "T", "Hk", "Hv", "K", "V")}
            cell = out / key
            cell.mkdir()
            common = [sys.executable, "-u", ROOT / "benchmarks/profile_serving_prefill.py",
                      "--library", origin["library"], "--directory", cell, "--batch", str(spec["B"]),
                      "--length", str(spec["T"]), "--value-heads", str(spec["Hv"]), "--gate", str(spec["gate"])]
            run([*common, "--phase", "admit", "--receipt", cell / "admission.json"],
                cell / "admission.log", env, console=False)
            admission = json.loads((cell / "admission.json").read_text())
            if admission.get("measured_shape") != expected_shape or admission.get("gate") != spec["gate"]:
                raise ValueError(f"{key}: admitted operands {admission.get('measured_shape')} / gate "
                                 f"{admission.get('gate')} are not the registered cell")
            print(f"[prefill ACU] {key}: numerical admission complete; profiling serially", flush=True)
            captures = {role: [] for role in ROLES}
            status["captures"][key] = dict(cell=spec, captures=captures,
                                           measured_shape=admission["measured_shape"],
                                           operands_sha=admission.get("operands_sha"),
                                           library_sha=admission.get("library_sha"),
                                           device_uuid=admission.get("device", {}).get("uuid"))

            def capture_round(first, stop):
                for repeat in range(first, stop):
                    shift = repeat % 3
                    roles = ROLES[shift:] + ROLES[:shift]
                    if repeat // 3:
                        roles = roles[::-1]
                    for role in roles:
                        base = cell / f"{role}-{repeat}.report"
                        command = [acu, *section_flags, "-f", "-o", base, "--set", "full", "--check-exit-code", "yes",
                                   *common, "--phase", "subject", "--role", role,
                                   "--receipt", cell / f"{role}-{repeat}.json"]
                        run(command, cell / f"{role}-{repeat}.log", env, console=False)
                        receipt = json.loads((cell / f"{role}-{repeat}.json").read_text())
                        validate_receipts(admission, receipt, role)
                        report = report_file(base)
                        exported = cell / f"{role}-{repeat}.csv"
                        # CSV is machine-consumed and packed; user need not paste it.
                        run([acu, *section_flags, "--import", report, "--csv", "--page", "raw"],
                            exported, env, console=False)
                        measured = inventory(role, parse_raw(exported.read_text()))
                        measured["report_sha256"] = sha(report)
                        captures[role].append(measured)
                        print(f"[prefill ACU] {key} role={role} repeat={repeat} "
                              f"kernel_sum={measured['total_us']:.3f}_us inventory={measured['status']}", flush=True)

            capture_round(0, args.repeats)
            summary = summarize(captures, args.repeats)
            if summary["needs_confirmation"] and args.repeats == 3 and not args.no_confirm:
                print(f"[prefill ACU] {key}: pre-registered confirmation round (3 more per role)", flush=True)
                capture_round(3, 6)
                summary = summarize(captures, 6)
                summary["confirmation_round"] = True
            status["captures"][key]["summary"] = summary
            print("[prefill ACU summary] " + json.dumps(dict(cell=key, **summary), sort_keys=True), flush=True)
        if sha(Path(origin["library"])) != origin["sha256"]:
            raise ValueError("reused admitted library changed during capture")
        if profiler_layout(acu) != status["profiler"]:
            raise ValueError("ACU executable or section files changed during capture")
        if (subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != status["harness_sha"]
                or subprocess.check_output(["git", "diff", "HEAD", "--stat"], cwd=ROOT, text=True)):
            raise ValueError("capture helper source changed during measurement")
        if smi:
            status["driver_after"] = run([smi, "-i", args.device, "-q"], out / "device-after.log",
                                         env, optional=True, console=False, timeout=20)
        status["status"] = ("COLLECTED" if all(c["summary"]["core_verdict"] != "UNADJUDICATED"
                                             for c in status["captures"].values()) else "UNADJUDICATED")
    except Exception as error:
        status["errors"].append(f"{type(error).__name__}: {error}")
        print(f"[prefill ACU] NOT_ADMITTED: {status['errors'][-1]}", file=sys.stderr)
    finally:
        (out / "summary.json").write_text(json.dumps(status, indent=2) + "\n")
        archive = small_archive(out)
        print(f"[prefill ACU] status={status['status']} artifacts={out} upload={archive}", flush=True)
        print("[prefill ACU] operand files and original .acurep stay on box; not included in small upload", flush=True)
    return 0 if status["status"] == "COLLECTED" else 3 if status["status"] == "UNADJUDICATED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
