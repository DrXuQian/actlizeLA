#!/usr/bin/env python3
"""Admit exactly the declared single-TF32 precision seam, not device numerics."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

from check_state_pipeline import block, code
from check_wy_binary import kernel_sequences, plant_in_kernel
from check_solve_static import diagonal_region

ROOT = Path(__file__).resolve().parents[2]
CONTROL = "gdn_wy_split_solve_static"
SUBJECT = "gdn_wy_split_solve_tf32x1"


def check_source(control, subject, original_launch):
    before, after, launch = map(code, (control, subject, original_launch))
    actual = block(after, SUBJECT + "(")
    if actual.count("single_tf32_product(") != 2:
        raise AssertionError("candidate did not bind both single-TF32 product sites")
    transformed = actual.replace(SUBJECT, CONTROL).replace("single_tf32_product(", "tf32_product(")
    if transformed != block(before, CONTROL + "("):
        raise AssertionError("solve changed outside the declared TF32-product seam")
    if block(after, "structSolveStorage") != block(before, "structSolveStorage"):
        raise AssertionError("solve storage changed")
    helper = ("single_tf32_product(float(&acc)[8],floatconst*a,intlda,floatconst*b,intldb){"
              "intconstlane=int(threadIdx.x)&31;CUTE_UNROLLfor(intk=0;k<16;k+=8){"
              "uint32_tah[4],bh[4];CUTE_UNROLLfor(ints=0;s<4;++s){autoconstrc=tf32_coord(lane,s);"
              "ah[s]=cutlass::tfloat32_t(a[rc.row*lda+rc.col+k]).raw();"
              "bh[s]=cutlass::tfloat32_t(b[(rc.col+k)*ldb+rc.row]).raw();}"
              "mma<cute::PPU0010_16x16x8_F32TF32TF32F32_TN>(acc,ah,bh);}}")
    if block(after, "single_tf32_product(") != helper:
        raise AssertionError("single-TF32 rounding/address/work contract changed")
    old = block(launch, "actlize_la_ppu10_forward_prepared(")
    new = block(after, "actlize_la_ppu10_solve_tf32x1_forward_prepared(")
    expected = old.replace("actlize_la_ppu10_forward_prepared", "actlize_la_ppu10_solve_tf32x1_forward_prepared")
    expected = expected.replace("sequence<=0||q_heads<=0", "sequence<=0||sequence%Chunk||q_heads<=0")
    expected = expected.replace("solve_static::launch_inverse_default", "solve_tf32x1::launch_inverse")
    expected = expected.replace("gate_cache::full_chunks(sequence)?full_chunk::launch_default_state(p,ws,final,stream):gate_cache::launch_state(p,ws,final,stream)",
                                "full_chunk::launch_default_state(p,ws,final,stream)")
    if new != expected:
        raise AssertionError("candidate wrapper changed default geometry/state/output/ABI")
    expected_launch = ("launch_inverse(Inputsp,Workspacews,gdn_arch::Streamstream){"
                       "if(intrc=launch_split_prefix(p,ws,stream))returnrc;"
                       "if(default_core_route(p.shape)!=CoreRoute::Retained)"
                       "gdn_wy_split_solve_tf32x1<256><<<unsigned(p.shape.groups()),256,sizeof(SolveStorage),stream>>>(p,ws);"
                       "elsegdn_wy_split_solve_tf32x1<128><<<unsigned(p.shape.groups()),128,sizeof(SolveStorage),stream>>>(p,ws);"
                       "returnint(hggcGetLastError());}")
    if block(after, "launch_inverse(") != expected_launch:
        raise AssertionError("solve thread/launch selection changed")


def check_native(isa, resources):
    kernels = kernel_sequences(isa)
    rows = {}
    for threads in (128, 256):
        def select(marker):
            names = [n for n in kernels if marker + f"ILj{threads}E" in n]
            if len(names) != 1:
                raise AssertionError("missing/ambiguous exact solve symbol")
            return names[0], kernels[names[0]]
        old_name, old = select(CONTROL)
        new_name, new = select(SUBJECT)
        a, b = (Counter(op.split()[0] for op in seq) for seq in (old, new))
        op = "v.mma.f32.tf32.m16n16k8"
        if (a[op], b[op]) != (12, 4):
            raise AssertionError("single-TF32 must replace12 with4 static MMA sites")
        fixed = ("v.mma.f32.bf16", "v.fma.f32", "v.exp2.", "vmem.aiu.", "vmem.st.", "s.blksyn")
        if ({k:v for k,v in a.items() if k.startswith(fixed)} !=
                {k:v for k,v in b.items() if k.startswith(fixed)}):
            raise AssertionError("KKT/FP32 diagonal/gates/publication/barriers changed")
        if any("ivreg" in op for op in new):
            raise AssertionError("unexpected indirect register lowering")
        # Isolate the actual diagonal between its two CTA barriers. Rounding,
        # 120 ordered negative FMAs and all16 rows are unchanged, not merely
        # a whole-body FMA count which could hide a compensating omission.
        _, _, diagonal = diagonal_region(isa, new_name.removesuffix("E"))
        diag = Counter(op.split()[0] for op in diagonal.values())
        words = sum(diag[k] * width for k,width in (("tsm.ld.b32",1),("tsm.ld.b32x2",2),("tsm.ld.b32x4",4)))
        if words != 120 or diag["v.fma.f32.rtte"] != 120 or diag["tsm.st.b32"] != 16:
            raise AssertionError("diagonal coefficient/FMA/row denominator changed")
        if any(not re.match(r"v\.fma\.f32\.rtte\s+vreg\d+,\s*!vreg\d+", op)
               for op in diagonal.values() if op.startswith("v.fma.")):
            raise AssertionError("diagonal subtraction changed")
        rr = []
        for name in (old_name, new_name):
            sections = [s for s in re.split(r"(?=^Func \d+ .*RESOURCE INFO:)", resources, flags=re.M)
                        if name in s.splitlines()[0]]
            if len(sections) != 1 or "STACK SIZE:0" not in sections[0]:
                raise AssertionError("missing native zero-stack proof")
            rr.append(int(re.search(r"vreg_number:(\d+)", sections[0])[1]))
        rows[str(threads)] = dict(static_sites=[len(old),len(new)], tf32_sites=[a[op],b[op]],
                                 registers=rr, native_stack=[0,0], diagonal_fmas=120)
    return rows


def source_texts():
    return tuple((ROOT / name).read_text() for name in (
        "csrc/gdn_chunk/gdn_wy_solve_static_ppu.cu", "csrc/gdn_chunk/gdn_wy_solve_tf32x1_ppu.cu",
        "csrc/backends/ppu10/launch.cu"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--isa", type=Path, required=True)
    p.add_argument("--resources", type=Path, required=True)
    a = p.parse_args()
    check_source(*source_texts())
    isa, resources = a.isa.read_text(), a.resources.read_text()
    result = check_native(isa, resources)
    for threads in (128, 256):
        marker = SUBJECT + f"ILj{threads}E"
        for opcode in ("v.mma.f32.tf32.m16n16k8", "v.mma.f32.bf16.m16n16k16", "vmem.st.b32x4"):
            try:
                check_native(plant_in_kernel(isa, marker, opcode, "MISSING"), resources)
            except AssertionError:
                pass
            else:
                raise AssertionError("missing-work negative escaped")
        sections = isa.split("Disassembly of section ")
        old = next(s for s in sections if s.startswith(".text.kernel.") and CONTROL + f"ILj{threads}E" in s.splitlines()[0])
        at = next(i for i,s in enumerate(sections) if s.startswith(".text.kernel.") and marker in s.splitlines()[0])
        sections[at] = old.replace("12solve_static25" + CONTROL, "12solve_tf32x125" + SUBJECT)
        try:
            check_native("Disassembly of section ".join(sections), resources)
        except AssertionError as exc:
            if "single-TF32" not in str(exc):
                raise
        else:
            raise AssertionError("retained triple-product body escaped")
    print(json.dumps(dict(scope="native compile proof; device numerics/performance NOT_RUN", solves=result,
                          negatives="missing TF32/BF16/store and retained triple-product body EXPECTED_RED"), indent=2))


if __name__ == "__main__":
    main()
