#!/usr/bin/env python3
"""Compile-only KDA gate: the serial P@R must really disappear from native code."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

from check_residual_blayout import native_body
from check_wy_binary import kernel_sequences, plant_in_kernel


def check(isa, resources):
    result = dict(scope="local PPU1.0 native codegen; device correctness/timing NOT_RUN")
    for role, marker, mma, aiu in (
        ("control-state", "gdn_wy_residual_full_chunk_stateE", 20, 4),
        ("candidate-state", "gdn_wy_parallel_wu_stateE", 16, 5),
    ):
        body, loops = native_body(isa, marker, mma)
        if len(loops) != 1:
            raise AssertionError("unclassified recurrence loops")
        count = Counter(op.split()[0] for op in loops[0])
        if (sum(v for k,v in count.items() if k.startswith("vmem.aiu.")) != aiu
                or count["s.blksyn.defer"] != 4):
            raise AssertionError("state input-delivery/barrier contract changed")
        sections = [s for s in re.split(r"(?=^Func \d+ .*RESOURCE INFO:)", resources, flags=re.M)
                    if marker in s.splitlines()[0]]
        if len(sections) != 1 or "STACK SIZE:0" not in sections[0]:
            raise AssertionError("missing native resource/zero-stack evidence")
        regs = int(re.search(r"vreg_number:(\d+)", sections[0])[1])
        if any("ivreg" in op or "tsm.ld.ncom" in op for op in body):
            raise AssertionError("unexpected compatibility/remap path")
        result[role] = dict(static_sites=len(body), recurrence_sites=len(loops[0]),
                            recurrence_mma=mma, aiu_sites=aiu, registers=regs, native_stack=0)
    sequences = kernel_sequences(isa)
    wu = [ops for name,ops in sequences.items() if "gdn_wy_parallel_wu_materializeE" in name]
    if len(wu) != 1 or sum(op.startswith("v.mma.f32.bf16") for op in wu[0]) != 32:
        raise AssertionError("missing/changed chunk-parallel W/U work")
    sections = [s for s in re.split(r"(?=^Func \d+ .*RESOURCE INFO:)", resources, flags=re.M)
                if "gdn_wy_parallel_wu_materializeE" in s.splitlines()[0]]
    if len(sections) != 1 or "STACK SIZE:0" not in sections[0]:
        raise AssertionError("missing W/U native zero-stack evidence")
    result["materialize"] = dict(static_sites=len(wu[0]), mma_sites=32,
                                  registers=int(re.search(r"vreg_number:(\d+)", sections[0])[1]), native_stack=0)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--isa", required=True, type=Path)
    p.add_argument("--resources", required=True, type=Path)
    a = p.parse_args()
    isa, resources = a.isa.read_text(), a.resources.read_text()
    result = check(isa, resources)
    sections = isa.split("Disassembly of section ")
    old = next(s for s in sections if s.startswith(".text.kernel.") and "gdn_wy_residual_full_chunk_stateE" in s.splitlines()[0])
    at = next(i for i,s in enumerate(sections) if s.startswith(".text.kernel.") and "gdn_wy_parallel_wu_stateE" in s.splitlines()[0])
    sections[at] = old.replace("gdn_wy_residual_full_chunk_stateE", "gdn_wy_parallel_wu_stateE")
    try:
        check("Disassembly of section ".join(sections), resources)
    except AssertionError:
        pass
    else:
        raise AssertionError("retained P@R under candidate symbol escaped")
    for marker, opcode in (("gdn_wy_parallel_wu_stateE", "v.mma.f32.bf16.m16n16k16"),
                            ("gdn_wy_parallel_wu_materializeE", "v.mma.f32.bf16.m16n16k16"),
                            ("gdn_wy_parallel_wu_stateE", "vmem.aiu.ld.tsm")):
        mutated = plant_in_kernel(isa, marker, opcode, "MISSING")
        try:
            check(mutated, resources)
        except AssertionError:
            continue
        raise AssertionError("native work/delivery negative escaped")
    result["negatives"] = "retained P@R body and missing state MMA/materialize MMA/AIU EXPECTED_RED"
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
