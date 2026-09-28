#!/usr/bin/env python3
"""Physical Hopper admission; each child still makes one target invocation.

Do not use the whole orchestrator as a one-call simulator input. Individual
commands are recorded and can be used separately with the simulation runner.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    ("full64", 1, 64, 1, 1, -.1, []),
    ("fp32-no-initial", 1, 65, 1, 2, -.1, ["--fp32-gate", "--vary-gate"]),
    ("tail1-initial", 2, 1, 1, 2, -.1, ["--initial", "--fp32-gate", "--vary-gate"]),
    ("tail31-gva", 2, 31, 2, 6, -.1, ["--vary-gate"]),
    ("half32-initial", 1, 32, 1, 2, -.1, ["--initial"]),
    ("tail63-distinct", 2, 63, 2, 4, -.1, ["--vary-gate"]),
    ("tail65-initial", 2, 65, 1, 2, -.1, ["--initial", "--fp32-gate", "--vary-gate"]),
    ("tail127", 1, 127, 2, 4, -.1, ["--vary-gate"]),
    ("full128-strong", 1, 128, 1, 1, -1., []),
    ("tail129-zero-decay", 2, 129, 2, 4, 0., ["--initial", "--fp32-gate"]),
    ("tail257-wrap", 1, 257, 2, 4, -1., ["--vary-gate"]),
    ("output-only", 2, 65, 1, 2, -.1, ["--initial", "--fp32-gate", "--vary-gate", "--output-only"]),
    ("target-weak", 1, 2048, 16, 32, -.1, []),
    ("target-strong-initial", 1, 2048, 16, 32, -1., ["--initial", "--fp32-gate"]),
]


def validate_cases(cases):
    names = [row[0] for row in cases]
    if len(names) != len(set(names)):
        raise ValueError("duplicate case in admission denominator")
    specializations = {( "--fp32-gate" in row[-1], "--initial" in row[-1]) for row in cases}
    if specializations != {(False, False), (False, True), (True, False), (True, True)}:
        raise ValueError("admission denominator omits a gate/state specialization")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--extension", type=Path, required=True)
    p.add_argument("--backend", choices=("cuda_sm90", "ppu17"), required=True)
    p.add_argument("--source-check", action="store_true")
    p.add_argument("--configuration", choices=("control","value64","value64-local-inverse","value128-paired"), default="control")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    validate_cases(CASES)
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for name, b, t, h, hv, g, extra in CASES:
        directory = args.out / name
        cmd = [sys.executable, str(ROOT / "tools/run_sm90_gdn.py"),
            "--extension", str(args.extension.resolve()), "--backend", args.backend,
            "--configuration", args.configuration,
            "--out", str(directory.resolve()), "--batch", str(b), "--length", str(t),
            "--q-heads", str(h), "--v-heads", str(hv), "--gate", str(g), *extra]
        if args.source_check:
            cmd.append("--source-check")
        with (args.out / (name + ".log")).open("w") as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=120)
        item = dict(case=name, command=cmd, rc=result.returncode)
        if result.returncode == 0:
            item["result"] = json.loads((directory / "result.json").read_text())
        results.append(item)
        (args.out / "cases.json").write_text(json.dumps(
            dict(denominator=len(CASES), completed=len(results), cases=results), indent=2) + "\n")
        if result.returncode:
            raise RuntimeError(f"{name} failed; {len(results)}/{len(CASES)} attempted; see {args.out / (name + '.log')}")
        print(f"[SM90 Hopper cases] {len(results)}/{len(CASES)} {name} "
              f"errors={item['result']['errors']} NUMERIC/PASS", flush=True)


if __name__ == "__main__":
    main()
