#!/usr/bin/env python3
"""Build pinned PLA only, reuse admitted operands/core, capture three full calls."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys

from acu_installation import bind_profiler, check_profiler, profiler_layout
from collect_ppu_gdn_acu import DEFAULT_ACU, ROOT, report_file, sha
from collect_serving_prefill_acu import parse_raw, run, small_archive, source_fingerprint
from gdn_threeway_contract import (
    CHUNKS, PLA_URL, PLA_SHA, ROLES, cells, check_pla_source, complete,
    git, inventory, read_inputs, record_pla, summarize, validate_pla, validate_receipt,
)
from ppu_pla_host_cxx import include_dirs as host_include_dirs, compiler as host_compiler


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def reference_environment(env):
    overrides = sorted(name for name in env if name.startswith(("FF_", "PLA_")))
    if overrides:
        raise ValueError(f"unset reference overrides before default comparison: {overrides}")
    if "GDN_QSA_WY_EXTENSION" in env:
        raise ValueError("unset GDN_QSA_WY_EXTENSION; measure the admitted public default")


def reuse_failed_build(previous, selected, sdk, compiler_log, *, workspace=Path("/workspace")):
    previous = Path(previous).resolve(strict=True)
    if not previous.is_relative_to(workspace) or previous == workspace:
        raise ValueError("reuse must name a dedicated /workspace threeway run")
    old = json.loads((previous / "evidence/summary.json").read_text())
    if (old.get("status") != "INCOMPLETE" or old.get("captures") or old.get("pla")
            or old.get("input_run") != selected or Path(old["sdk"]).resolve() != sdk.resolve()):
        raise ValueError("reuse requires a build-failed run with identical input/binary/SDK, no captures")
    log = (previous / "evidence/pla-build.log").read_text(errors="replace")
    if "fatal error: cuda_runtime_api.h: No such file or directory" not in log:
        raise ValueError("reuse is restricted to this host-header build failure, not a generic resume")
    if (previous / "evidence/compiler.log").read_bytes() != compiler_log.read_bytes():
        raise ValueError("HGCC identity changed; do not reuse existing device objects")
    source, base = previous / "pla-source", previous / "pla-build"
    if any(p.is_symlink() or not p.is_dir() for p in (source, base)):
        raise ValueError("reuse source/build must be real directories in the failed run")
    check_pla_source(source)
    objects = sorted(base.rglob("*.o"))
    if any(p.is_symlink() or not p.resolve().is_relative_to(base) for p in objects):
        raise ValueError("reuse objects escape the failed build tree")
    return dict(run=str(previous), source=str(source), build_base=str(base),
                previous_summary_sha256=sha(previous / "evidence/summary.json"),
                objects_before={str(p): sha(p) for p in objects})


def build_pla(out, evidence, env, jobs, reuse=None):
    source = Path(reuse["source"]) if reuse else out / "pla-source"
    base = Path(reuse["build_base"]) if reuse else out / "pla-build"
    runtime = out / "pla-runtime"
    includes, cxx = host_include_dirs(env["PPU_SDK"]), host_compiler(env)
    if reuse:
        print(f"[PLA build] reuse={reuse['run']} existing_objects={len(reuse['objects_before'])} "
              "device_commands=UNCHANGED; ninja rebuilds only changed/missing targets", flush=True)
    # Fetch one immutable upstream revision and its recorded dependency, never
    # pip-install globally or compile actlizeLA a second time for this comparison.
    commands = (
        ["git", "init", source],
        ["git", "-C", source, "remote", "add", "origin", PLA_URL],
        ["git", "-C", source, "fetch", "--depth=1", "origin", PLA_SHA],
        ["git", "-C", source, "checkout", "--detach", PLA_SHA],
        ["git", "-C", source, "submodule", "update", "--init", "--depth=1", "3rdparty/actlize"],
    )
    if reuse is None:
        for index, command in enumerate(commands):
            run(command, evidence / f"pla-source-{index}.log", env, timeout=600)
    check_pla_source(source)
    build_env = dict(env, PLA_BUILD_NAMESPACE="pla.prefill.flashqla",
                     PPU_SDK_DIR=env["PPU_SDK"], PPU_ACTLIZE_DIR=str(source / "3rdparty/actlize"),
                     MAX_JOBS=str(jobs), HGCC_THREADS="1", ACTLIZE_LA_PLA_HOST_CXX=cxx,
                     CXX=str(ROOT / "tools/ppu_pla_host_cxx.py"))
    write(evidence / "pla-host-toolchain.json", dict(
        compiler=cxx, compiler_sha256=sha(cxx), adapter=build_env["CXX"],
        adapter_sha256=sha(build_env["CXX"]), include_dirs=list(map(str, includes)),
        cuda_runtime_api_sha256=sha(includes[0] / "cuda_runtime_api.h"), reuse=reuse))
    command = ["bash", "-c", 'cd -- "$1" && shift && exec "$@"', "pla-build", source,
               sys.executable, "setup.py", "build", "--build-base", base,
               "--build-lib", runtime]
    base.mkdir(parents=True, exist_ok=True)
    with (base / ".threeway-build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(command, evidence / "pla-build.log", build_env)
    manifest = record_pla(source, runtime)
    manifest["host_toolchain"] = json.loads((evidence / "pla-host-toolchain.json").read_text())
    if reuse:
        manifest["reuse"] = dict(reuse, unchanged_objects=[p for p, digest in reuse["objects_before"].items()
                                                          if Path(p).is_file() and sha(p) == digest])
        print(f"[PLA build] unchanged_reused_objects={len(manifest['reuse']['unchanged_objects'])}/"
              f"{len(reuse['objects_before'])}", flush=True)
    manifest["build_environment"] = {name: build_env[name] for name in
                                    ("PLA_BUILD_NAMESPACE", "PPU_SDK_DIR", "PPU_ACTLIZE_DIR", "MAX_JOBS", "HGCC_THREADS",
                                     "CXX", "ACTLIZE_LA_PLA_HOST_CXX")}
    manifest["compiler_sha256"] = sha(Path(env["PPU_SDK"]) / "bin/hgcc")
    write(evidence / "pla-build.json", manifest)
    return manifest


def measure_cell(cell, origin, saved, manifest_path, directory, command_prefix, env):
    common = [sys.executable, "-u", ROOT / "benchmarks/profile_gdn_threeway.py",
              "--library", origin["library"], "--inputs", saved["directory"],
              "--directory", directory, "--pla-manifest", manifest_path, "--cell", cell["id"]]
    run([*common, "--phase", "admit", "--receipt", directory / "admission.json"],
        directory / "admission.log", env, console=False)
    admission = json.loads((directory / "admission.json").read_text())
    if (admission["operands_sha"] != saved["operands_sha"]
            or admission["input_admission_sha"] != saved["admission_sha"]):
        raise ValueError("input changed between preflight and new numerical gate")
    captures = {role: [] for role in ROLES}
    print(f"[GDN threeway] {cell['id']}: all three numerical gates PASS; profiling serially", flush=True)

    def capture_round(first, stop):
        for repeat in range(first, stop):
            shift = repeat % 3
            order = ROLES[shift:] + ROLES[:shift]
            if repeat // 3:
                order = order[::-1]
            for role in order:
                base = directory / f"{role}-{repeat}.report"
                receipt_path = directory / f"{role}-{repeat}.json"
                run([*command_prefix, "-f", "-o", base, "--set", "full", "--check-exit-code", "yes",
                     *common, "--phase", "subject", "--role", role, "--receipt", receipt_path],
                    directory / f"{role}-{repeat}.log", env, console=False)
                validate_receipt(admission, json.loads(receipt_path.read_text()), role)
                report = report_file(base)
                exported = directory / f"{role}-{repeat}.csv"
                run([*command_prefix, "--import", report, "--csv", "--page", "raw"], exported, env, console=False)
                measured = inventory(role, parse_raw(exported.read_text()))
                measured["report_sha256"] = sha(report)
                captures[role].append(measured)
                write(directory / "captures.json", captures)
                print(f"[GDN threeway] {cell['id']} role={role} repeat={repeat} "
                      f"kernel_sum={measured['total_us']:.3f}_us inventory={measured['status']}", flush=True)

    capture_round(0, 3)
    result = summarize(captures)
    if result["needs_confirmation"]:
        print(f"[GDN threeway] {cell['id']}: registered noise/near-target confirmation; keep all samples", flush=True)
        capture_round(3, 6)
        result = summarize(captures)
    return dict(cell=cell, captures=captures, summary=result)


def table(records):
    lines = ["# ZW810 three-core ACU comparison", "",
             "Complete-call kernel sums (us), not uninstrumented API latency. "
             "Native chunks: default=64 / PLA=32 / FLA=64. All include final state.", "",
             "| cell | default median [min,max] | PLA median [min,max] | FLA median [min,max] | default vs PLA | default vs FLA | PLA vs FLA |",
             "|---|---:|---:|---:|---|---|---|"]
    for spec in cells():
        record = records.get(spec["id"], {})
        summary = record.get("summary", {})
        if not summary.get("rows"):
            lines.append(f"| {spec['id']} | NOT_RUN | NOT_RUN | NOT_RUN | UNADJUDICATED | UNADJUDICATED | UNADJUDICATED |")
            continue
        values = [f"{v['median_us']:.3f} [{v['min_us']:.3f},{v['max_us']:.3f}]"
                  for v in (summary["rows"][r] for r in ROLES)]
        verdicts = [summary["comparisons"][p]["verdict"] for p in
                    ("default_vs_pla", "default_vs_fla", "pla_vs_fla")]
        lines.append("| " + " | ".join([spec["id"], *values, *verdicts]) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-run", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--acu", type=Path, default=DEFAULT_ACU)
    parser.add_argument("--device", default="0")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--pla-build-run", type=Path, help="reuse only the failed host-header build; new evidence/runtime")
    args = parser.parse_args()
    if not args.device.isdigit() or args.jobs < 1:
        parser.error("device must be a physical ordinal and jobs must be positive")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = (args.out or Path(f"/workspace/actlize-la-gdn-threeway-{stamp}-{os.getpid()}")).resolve()
    if not out.is_relative_to(Path("/workspace")) or out == Path("/workspace"):
        parser.error("new artifacts must be in a dedicated /workspace subdirectory")
    out.mkdir(parents=True, exist_ok=False)
    evidence = out / "evidence"
    evidence.mkdir()
    status = dict(status="INCOMPLETE", created_utc=stamp, cells=cells(), chunks=CHUNKS, captures={}, errors=[],
                  scope="processed-core-only; no production routing change; no historical timing reused")
    try:
        reference_environment(os.environ)
        sdk = Path(os.environ.get("PPU_SDK", os.environ.get("PPU_SDK_ROOT", "/usr/local/PPU_SDK"))).resolve(strict=True)
        if not os.access(sdk / "bin/hgcc", os.X_OK):
            raise ValueError(f"missing executable hgcc in selected SDK: {sdk}")
        for name in ("scratch", "triton-cache"):
            (out / name).mkdir()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.device, PYTHONUNBUFFERED="1", PPU_SDK=str(sdk),
                   TMPDIR=str(out / "scratch"), TRITON_CACHE_DIR=str(out / "triton-cache"),
                   PATH=f"{sdk}/bin:{sdk}/CUDA_SDK/bin:" + os.environ.get("PATH", ""),
                   LD_LIBRARY_PATH=f"{sdk}/lib:{sdk}/lib64:{sdk}/CUDA_SDK/lib64:" + os.environ.get("LD_LIBRARY_PATH", ""),
                   PYTHONPATH=":".join(filter(None, (str(ROOT), str(ROOT / "tests"), str(ROOT / "tools"),
                                                     os.environ.get("FLA_ROOT"), os.environ.get("PYTHONPATH")))))
        if env.get("FLA_ROOT") and not (Path(env["FLA_ROOT"]) / "fla/__init__.py").is_file():
            raise ValueError("invalid FLA_ROOT; no silent installed fallback")
        status["source_sha"], status["source_files"] = git(ROOT, "rev-parse", "HEAD"), source_fingerprint()
        status["sdk"], status["physical_device"] = str(sdk), args.device
        status["source_diff"] = git(ROOT, "diff", "HEAD", "--stat")
        selected = read_inputs(args.input_run)
        status["input_run"] = selected
        write(evidence / "inputs.json", selected)
        passed = evidence / "passed-run"
        passed.mkdir()
        for name in selected["origin"]["device_receipts"]:
            shutil.copyfile(Path(selected["origin"]["directory"]) / name, passed / name)
        run([sys.executable, ROOT / "tools/check_ppu810_gdn_recheck.py", "--library", selected["origin"]["library"]],
            evidence / "device-and-routes.log", env)
        status["profiler"], env, command_prefix = bind_profiler(args.acu, env)
        check_profiler(status["profiler"], env, command_prefix, evidence, runner=run)
        run([sdk / "bin/hgcc", "--version"], evidence / "compiler.log", env)
        smi = shutil.which("ppu-smi", path=env["PATH"])
        if smi:
            status["driver_before"] = run([smi, "-i", args.device, "-q"], evidence / "device-before.log", env,
                                           optional=True, console=False, timeout=20)
        else:
            status["driver_before"] = dict(status="UNAVAILABLE", reason="ppu-smi not on selected SDK/PATH")
        reuse = (reuse_failed_build(args.pla_build_run, selected, sdk, evidence / "compiler.log")
                 if args.pla_build_run else None)
        manifest = build_pla(out, evidence, env, args.jobs, reuse)
        status["pla"] = manifest
        for cell in cells():
            key = cell["id"]
            directory = evidence / key
            directory.mkdir()
            shutil.copyfile(Path(selected["inputs"][key]["directory"]) / "admission.json",
                            directory / "input-admission.json")
            try:
                status["captures"][key] = measure_cell(cell, selected["origin"], selected["inputs"][key],
                                                      evidence / "pla-build.json", directory, command_prefix, env)
            except Exception as error:
                # Failed numerics do not get timed; other cells use new processes.
                message = f"{key}: {type(error).__name__}: {error}"
                status["errors"].append(message)
                status["captures"][key] = dict(cell=cell, error=message,
                                               summary=dict(status="UNADJUDICATED"))
                print(f"[GDN threeway] NOT_ADMITTED: {message}", file=sys.stderr, flush=True)
            write(evidence / "summary.json", status)
            (evidence / "comparison.txt").write_text(table(status["captures"]))
        if selected != read_inputs(args.input_run):
            raise ValueError("admitted binary/input receipts changed during collection")
        validate_pla(manifest)
        check_pla_source(manifest["source"])
        if (git(ROOT, "rev-parse", "HEAD") != status["source_sha"]
                or source_fingerprint() != status["source_files"]
                or profiler_layout(args.acu) != status["profiler"]):
            raise ValueError("harness or profiler changed during collection")
        if smi:
            status["driver_after"] = run([smi, "-i", args.device, "-q"], evidence / "device-after.log", env,
                                          optional=True, console=False, timeout=20)
        status["status"] = complete(status["captures"])
    except Exception as error:
        status["errors"].append(f"{type(error).__name__}: {error}")
        print(f"[GDN threeway] NOT_ADMITTED: {status['errors'][-1]}", file=sys.stderr, flush=True)
    finally:
        write(evidence / "summary.json", status)
        (evidence / "comparison.txt").write_text(table(status["captures"]))
        archive = small_archive(evidence)
        shutil.copyfile(archive, out / "results.tar.gz")
        print(table(status["captures"]), flush=True)
        print(f"[GDN threeway] status={status['status']} artifacts={out} upload={out / 'results.tar.gz'}", flush=True)
        print("[GDN threeway] binaries/operands/acurep stay on box; small upload includes logs, counters and receipts", flush=True)
    return 0 if status["status"] == "COLLECTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
