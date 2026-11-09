#!/usr/bin/env python3
"""Check one ACU installation without compiling or launching device code."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

from collect_ppu_gdn_acu import DEFAULT_ACU, run, sha


def profiler_layout(acu):
    """Bind executable and unchanged sections to the same resolved installation."""
    executable = Path(acu).resolve(strict=True)
    home = executable.parent.parent
    sections = home / "sections"
    files = sorted(sections.glob("*.section"))
    if not files:
        raise ValueError(f"selected ACU has no companion sections: {sections}")
    return dict(executable=str(executable), executable_sha256=sha(executable),
                home=str(home), sections=str(sections),
                section_sha256={p.name: sha(p) for p in files})


def bind_profiler(acu, env):
    installation = profiler_layout(acu)
    bound = dict(env, ASIGHT_HOME=installation["home"])
    command = [installation["executable"], "--section-folder", installation["sections"]]
    return installation, bound, command


def check_profiler(installation, env, command, out, runner=run, *, console=False):
    # Even --version may initialize the section parser. No bare invocation may
    # precede this check, including from a shell builder's identity collection.
    for args, name in ((["--version"], "acu-version.log"),
                       (["--set", "full", "--list-sections"], "acu-section-check.log")):
        log = out / name
        runner([*command, *args], log, env, console=console, timeout=30)
        if "Error parsing text-format HG.Profiler.ProfilerSection" in log.read_text(errors="replace"):
            raise RuntimeError(f"ACU section parser failed despite rc=0; see {log}")
    if profiler_layout(installation["executable"]) != installation:
        raise ValueError("ACU executable or section files changed during preflight")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acu", type=Path, default=os.environ.get("ACU", DEFAULT_ACU))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or Path(f"/workspace/actlize-la-acu-check-{stamp}-{os.getpid()}")
    out.mkdir(parents=True, exist_ok=False)
    receipt = dict(status="INCOMPLETE", created_utc=stamp,
                   inherited_asight_home=os.environ.get("ASIGHT_HOME"),
                   ld_library_path=os.environ.get("LD_LIBRARY_PATH"), acu=str(args.acu))
    try:
        installation, env, command = bind_profiler(args.acu, os.environ)
        receipt["profiler"] = installation
        print(f"[ACU installation] executable={installation['executable']} "
              f"ASIGHT_HOME={installation['home']} sections={installation['sections']}", flush=True)
        check_profiler(installation, env, command, out, console=True)
        receipt["status"] = "PASS"
        print("[ACU installation] PASS: version + unchanged full section set; device execution=NOT_RUN")
    except Exception as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        print(f"[ACU installation] FAIL: {receipt['error']}; no kernel was built or run. "
              "If MemoryMetricTable still fails with the printed companion sections, "
              "check this ACU installation and its loaded libraries; do not edit GDN or remove metrics.",
              file=sys.stderr)
    finally:
        (out / "profiler.json").write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"[ACU installation] artifacts={out}", flush=True)
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
