#!/usr/bin/env python3
"""Compare every actual native kernel, including instruction operands.

For mechanical refactors only. Equal names/counts are necessary but insufficient.
No timing/device correctness is inferred. Inputs must be same-toolchain dumps.
"""
import argparse
from pathlib import Path
import re


def kernels(text):
    found = {}
    for section in text.split("Disassembly of section ")[1:]:
        if not section.startswith(".text.kernel."):
            continue
        name = section.splitlines()[0].removeprefix(".text.kernel.").removesuffix(":")
        if name in found:
            raise AssertionError(f"duplicate kernel: {name}")
        instructions = re.findall(r"^\s*[0-9a-f]+:\s+(?:[0-9a-f]{2}\s+){8}\s*([^\n]+)", section, re.M)
        if not instructions:
            raise AssertionError(f"empty kernel: {name}")
        found[name] = instructions
    if not found:
        raise AssertionError("no native kernels; cannot classify preservation")
    return found


def compare(before, after):
    if before.keys() != after.keys():
        raise AssertionError(f"kernel denominator changed: {len(before)} -> {len(after)}")
    for name in before:
        if before[name] != after[name]:
            raise AssertionError(f"native instructions/operands changed: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    before, after = kernels(args.before.read_text()), kernels(args.after.read_text())
    compare(before, after)
    if args.self_test:
        name = next(iter(after))
        plants = {
            "missing-kernel": {k: v for k, v in after.items() if k != name},
            "changed-instruction-operand": after | {name: [after[name][0] + " CORRUPT", *after[name][1:]]},
            "same-count-wrong-order": after | {name: list(reversed(after[name]))},
        }
        for label, plant in plants.items():
            try:
                compare(before, plant)
            except AssertionError:
                print(f"[backend native negative] {label} EXPECTED-RED/PASS")
            else:
                raise AssertionError(f"negative escaped: {label}")
        try:
            kernels("Disassembly of section .text.kernel.empty:\n")
        except AssertionError:
            print("[backend native negative] empty-body EXPECTED-RED/PASS")
        else:
            raise AssertionError("empty native body escaped")
    print(f"[backend native preservation] PASS kernels={len(before)}/{len(after)} "
          f"instruction_sites={sum(map(len, before.values()))} operands=IDENTICAL device=NOT_RUN")


if __name__ == "__main__":
    main()
