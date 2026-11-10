#!/usr/bin/env python3
"""User-operated KDA round: incumbent / declared candidate / FLA, fixed inventory."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

from acu_installation import bind_profiler, check_profiler, profiler_layout
from collect_ppu_gdn_acu import DEFAULT_ACU, ROOT, report_file, sha
from collect_serving_prefill_acu import parse_raw, run, small_archive, source_fingerprint
from collect_gdn_threeway_acu import reference_environment, write
from gdn_threeway_contract import git, read_inputs
from gdn_kda_contract import EXPERIMENTS, roles, cells, complete, inventory, summarize, validate_receipt


def source_identity():
    files = source_fingerprint()
    for policy in EXPERIMENTS.values():
        path = "dev/ppu/" + policy["checker"]
        files[path] = sha(ROOT / path)
        for path in policy.get("sources", ()):
            files[path] = sha(ROOT / path)
    for directory in ("include", "csrc", "cmake"):
        for p in sorted((ROOT / directory).rglob("*")):
            if p.is_file():
                files[str(p.relative_to(ROOT))] = sha(p)
    files["third_party/actlize:commit"] = git(ROOT / "third_party/actlize", "rev-parse", "HEAD")
    if git(ROOT / "third_party/actlize", "status", "--porcelain"):
        raise ValueError("actlize submodule has local changes")
    return files


def table(captures, candidate="parallel-wu"):
    lines = ["# ZW810 KDA: " + candidate, "", "Full-call ACU kernel sums, us; all work and final state included.", "",
             f"| cell | default median [min,max] | {candidate} median [min,max] | FLA median [min,max] | candidate/default | candidate/FLA | 20%-target |",
             "|---|---:|---:|---:|---|---|---|"]
    for cell in cells():
        summary = captures.get(cell["id"], {}).get("summary", {})
        if "rows" not in summary:
            lines.append(f"| {cell['id']} | NOT_RUN | NOT_RUN | NOT_RUN | UNADJUDICATED | UNADJUDICATED | UNADJUDICATED |")
            continue
        values = [f"{v['median_us']:.3f} [{v['min_us']:.3f},{v['max_us']:.3f}]" for v in
                  (summary["rows"][r] for r in roles(candidate))]
        verdicts = [summary["comparisons"][p]["verdict"] for p in ("default_vs_" + candidate, candidate + "_vs_fla")]
        lines.append("| " + " | ".join([cell["id"], *values, *verdicts, summary["target_candidate_vs_fla"]]) + " |")
    return "\n".join(lines) + "\n"


def measure(cell, selected, library, candidate_sha, directory, prefix, env, experiment="parallel-wu"):
    saved = selected["inputs"][cell["id"]]
    common = [sys.executable, "-u", ROOT / "benchmarks/profile_gdn_parallel_wu.py",
              "--library", selected["origin"]["library"], "--candidate", library,
              "--inputs", saved["directory"], "--directory", directory, "--cell", cell["id"],
              "--experiment", experiment]
    run([*common, "--phase", "admit", "--receipt", directory / "admission.json"], directory / "admission.log", env, console=False)
    admission = json.loads((directory / "admission.json").read_text())
    if (admission["operands_sha"] != saved["operands_sha"]
            or admission["input_admission_sha"] != saved["admission_sha"]
            or admission["libraries"] != dict(default=selected["origin"]["sha256"], candidate=candidate_sha)):
        raise ValueError("input/binary changed between preflight and admission")
    selected_roles = roles(experiment)
    captures = {role: [] for role in selected_roles}
    print(f"[GDN KDA] {cell['id']}: numerics passed; serial ACU capture", flush=True)

    def rounds(begin, end):
        for repeat in range(begin, end):
            shift = repeat % 3
            order = selected_roles[shift:] + selected_roles[:shift]
            if repeat // 3:
                order = order[::-1]
            for role in order:
                base, receipt = directory / f"{role}-{repeat}.report", directory / f"{role}-{repeat}.json"
                run([*prefix, "-f", "-o", base, "--set", "full", "--check-exit-code", "yes", *common,
                     "--phase", "subject", "--role", role, "--receipt", receipt], directory / f"{role}-{repeat}.log", env, console=False)
                validate_receipt(admission, json.loads(receipt.read_text()), role)
                report = report_file(base)
                exported = directory / f"{role}-{repeat}.csv"
                run([*prefix, "--import", report, "--csv", "--page", "raw"], exported, env, console=False)
                measured = inventory(role, parse_raw(exported.read_text()))
                measured["report_sha256"] = sha(report)
                captures[role].append(measured)
                write(directory / "captures.json", captures)
                print(f"[GDN KDA] {cell['id']} role={role} repeat={repeat} kernel_sum={measured['total_us']:.3f}_us {measured['status']}", flush=True)

    rounds(0, 3)
    result = summarize(captures, experiment)
    if result["needs_confirmation"]:
        rounds(3, 6)
        result = summarize(captures, experiment)
    return dict(cell=cell, captures=captures, summary=result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-run", type=Path, required=True)
    p.add_argument("--experiment", choices=EXPERIMENTS, default="parallel-wu")
    p.add_argument("--out", type=Path)
    p.add_argument("--device", default="0")
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--acu", type=Path, default=DEFAULT_ACU)
    a = p.parse_args()
    if not a.device.isdigit() or a.jobs < 1:
        p.error("physical device ordinal and positive jobs required")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    policy = EXPERIMENTS[a.experiment]
    out = (a.out or Path(f"/workspace/actlize-la-gdn-{a.experiment}-{stamp}-{os.getpid()}")).resolve()
    if not out.is_relative_to(Path("/workspace")) or out == Path("/workspace"):
        p.error("use a new dedicated /workspace directory")
    out.mkdir(parents=True, exist_ok=False)
    evidence = out / "evidence"
    evidence.mkdir()
    status = dict(status="INCOMPLETE", created_utc=stamp, cells=cells(), captures={}, errors=[],
                  experiment=a.experiment, arithmetic=policy["arithmetic"], scope="full-C64; no default promotion")
    try:
        reference_environment(os.environ)
        sdk = Path(os.environ.get("PPU_SDK", os.environ.get("PPU_SDK_ROOT", "/usr/local/PPU_SDK"))).resolve(strict=True)
        if not os.access(sdk / "bin/hgcc", os.X_OK):
            raise ValueError("hgcc missing from selected SDK")
        for name in ("scratch", "triton-cache"):
            (out / name).mkdir()
        env = dict(os.environ, PPU_SDK=str(sdk), CUDA_VISIBLE_DEVICES=a.device, PYTHONUNBUFFERED="1",
                   TMPDIR=str(out / "scratch"), TRITON_CACHE_DIR=str(out / "triton-cache"),
                   PATH=f"{sdk}/bin:{sdk}/CUDA_SDK/bin:" + os.environ.get("PATH", ""),
                   LD_LIBRARY_PATH=f"{sdk}/lib:{sdk}/lib64:{sdk}/CUDA_SDK/lib64:" + os.environ.get("LD_LIBRARY_PATH", ""),
                   PYTHONPATH=":".join(filter(None, (str(ROOT), str(ROOT / "tests"), str(ROOT / "tools"),
                                                     os.environ.get("FLA_ROOT"), os.environ.get("PYTHONPATH")))))
        if env.get("FLA_ROOT") and not (Path(env["FLA_ROOT"]) / "fla/__init__.py").is_file():
            raise ValueError("invalid explicit FLA_ROOT")
        status.update(source_sha=git(ROOT, "rev-parse", "HEAD"), source_files=source_identity(), sdk=str(sdk), physical_device=a.device)
        status["source_diff"] = git(ROOT, "diff", "HEAD", "--stat")
        if status["source_diff"]:
            raise ValueError("run the committed candidate, not a dirty source tree")
        selected = read_inputs(a.input_run)
        status["input_run"] = selected
        write(evidence / "inputs.json", selected)
        run([sys.executable, ROOT / "tools/check_ppu810_gdn_recheck.py", "--library", selected["origin"]["library"]],
            evidence / "device-and-routes.log", env)
        status["profiler"], env, prefix = bind_profiler(a.acu, env)
        check_profiler(status["profiler"], env, prefix, evidence, runner=run)
        run([sdk / "bin/hgcc", "--version"], evidence / "compiler.log", env)
        smi = shutil.which("ppu-smi", path=env["PATH"])
        if smi:
            run([smi, "-i", a.device, "-q"], evidence / "device-before.log", env, optional=True, console=False, timeout=20)
        build = out / "build"
        run(["cmake", "-S", ROOT / "csrc/serving", "-B", build, "-DALA_TARGET=ppu10",
             f"-DPPU_SDK_ROOT={sdk}", f"-D{policy['option']}=ON", "-DALA_BUILD_SERVING_EXAMPLE=OFF",
             "-DCMAKE_BUILD_TYPE=Release"], evidence / "configure.log", env)
        run(["cmake", "--build", build, f"-j{a.jobs}"], evidence / "build.log", env)
        library = build / "libactlize_la_serving.so"
        status["candidate"] = dict(library=str(library), sha256=sha(library), compiler_sha256=sha(sdk / "bin/hgcc"))
        shutil.copyfile(build / "CMakeCache.txt", evidence / "CMakeCache.txt")
        if policy.get("generated_manifest"):
            generated = build / policy["generated_manifest"]
            status["generated_sources"] = json.loads(generated.read_text())
            shutil.copyfile(generated, evidence / "generated-sources.json")
        dumper = sdk / "bin/hgobjdump"
        if not os.access(dumper, os.X_OK):
            raise ValueError("native codegen gate unavailable: hgobjdump missing from selected SDK")
        run([dumper, "--dump-isa", library], out / "native.isa", env, console=False)
        run([dumper, "--dump-resource-usage=all", library], out / "native.resources", env, console=False)
        run([sys.executable, ROOT / "dev/ppu" / policy["checker"], "--isa", out / "native.isa",
             "--resources", out / "native.resources"], evidence / "native-gate.log", env)
        if policy.get("probe"):
            receipt = evidence / "conversion-admission.json"
            run([sys.executable, ROOT / "tests" / policy["probe"], "--library", library,
                 "--receipt", receipt], evidence / "conversion-admission.log", env)
            status["conversion_admission"] = json.loads(receipt.read_text())
            if (status["conversion_admission"].get("status") != "PASS" or
                    status["conversion_admission"].get("library_sha256") != status["candidate"]["sha256"]):
                raise ValueError("native conversion device semantics not admitted")
        for cell in cells():
            key = cell["id"]
            directory = evidence / key
            directory.mkdir()
            shutil.copyfile(Path(selected["inputs"][key]["directory"]) / "admission.json", directory / "input-admission.json")
            try:
                status["captures"][key] = measure(cell, selected, library, status["candidate"]["sha256"], directory, prefix, env, a.experiment)
            except Exception as error:
                message = f"{key}: {type(error).__name__}: {error}"
                status["errors"].append(message)
                status["captures"][key] = dict(cell=cell, error=message, summary=dict(status="UNADJUDICATED"))
                print("[GDN KDA] NOT_ADMITTED: " + message, file=sys.stderr, flush=True)
            write(evidence / "summary.json", status)
            (evidence / "comparison.txt").write_text(table(status["captures"], a.experiment))
        if (read_inputs(a.input_run) != selected or sha(library) != status["candidate"]["sha256"]
                or git(ROOT, "rev-parse", "HEAD") != status["source_sha"]
                or source_identity() != status["source_files"] or profiler_layout(a.acu) != status["profiler"]):
            raise ValueError("source/binary/input/profiler identity changed during collection")
        if smi:
            run([smi, "-i", a.device, "-q"], evidence / "device-after.log", env, optional=True, console=False, timeout=20)
        status["status"] = complete(status["captures"])
    except Exception as error:
        status["errors"].append(f"{type(error).__name__}: {error}")
        print("[GDN KDA] NOT_ADMITTED: " + status["errors"][-1], file=sys.stderr, flush=True)
    finally:
        write(evidence / "summary.json", status)
        (evidence / "comparison.txt").write_text(table(status["captures"], a.experiment))
        archive = small_archive(evidence)
        shutil.copyfile(archive, out / "results.tar.gz")
        print(table(status["captures"], a.experiment), flush=True)
        print(f"[GDN KDA] status={status['status']} upload={out / 'results.tar.gz'}", flush=True)
        print("[GDN KDA] acurep/binaries/operands stay on box; COLLECTED is not a speed admission", flush=True)
    return 0 if status["status"] == "COLLECTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
