#!/usr/bin/env python3
"""Independent-row ownership only; preserve residual math/layout/lifetimes."""
import argparse
from collections import Counter
from pathlib import Path
from check_residual_delivery import canonical
from check_state_pipeline import code

ROOT = Path(__file__).resolve().parents[2]


def check_source(control, candidate):
    if canonical(candidate, "warps8") != code(control):
        raise AssertionError("eight-warps changed more than symbols and production owner traits")


def check_native(isa):
    from check_wy_binary import kernel_sequences
    sequences = kernel_sequences(isa)
    pairs = []
    for marker, expected in (("gdn_wy_residual_stateE", (4,40,9,5,24,32)),
                             ("gdn_wy_residual_warps8_stateE", (4,20,5,5,12,24))):
        bodies = [v for k,v in sequences.items() if marker in k]
        if len(bodies) != 1:
            raise AssertionError("eight-warps exact native body missing/ambiguous")
        seq = bodies[0]
        counts = Counter(line.split()[0] for line in seq)
        actual = tuple(counts[k] for k in (
            "vmem.aiu.ld.tsm.l0.t0.p0.s0.m0.2d.b16.kp1",
            "v.mma.f32.bf16.m16n16k16", "v.exp2.f32", "s.blksyn.defer",
            "tsm.ld.swzl.b32x4.s0.t1.trans0", "tsm.ld.swzl.b32x4.s0.t1.trans1"))
        if actual != expected:
            raise AssertionError(f"eight-warps native math/load/lifetime inventory: {actual} != {expected}")
        if sum("commit_group(0)" in line for line in seq) != 1:
            raise AssertionError("eight-warps lost async completion")
        if any("ivreg" in line or "shuffle" in line or
               line.startswith(("tsm.ld.ncom", "vmem.ld.tsm", "vmem.st.b16",
                                "vmem.aiu.ld.tsm.l1")) for line in seq):
            raise AssertionError("eight-warps has unexpected emulation/scalar publication")
        if "vmem.st.b32x4" not in counts:
            raise AssertionError("eight-warps lost vector publication")
        pairs.append(dict(marker=marker, static_instructions=len(seq), inventory=actual,
                          v2s=counts["v.mov.v2s"], waits=counts["s.wait"]))
    return pairs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--isa", type=Path)
    args = p.parse_args()
    d = ROOT/"csrc/gdn_chunk"
    control = (d/"gdn_wy_residual_ppu.cu").read_text()
    candidate = (d/"gdn_wy_residual_warps8_ppu.cu").read_text()
    check_source(control,candidate)
    if args.self_test:
        for label,old,new in (
            ("stale-launch-threads","<<<grid, Plan::Threads,","<<<grid, 128,"),
            ("stale-publication-threads","Value::publish<Plan::Threads>","Value::publish<128>"),
            ("missing-K-atom","k < Dim; k += 16","k < Dim - 16; k += 16"),
            ("changed-rounding","BF16(x * row_decay","BF16(float(BF16(x)) * row_decay"),
            ("missing-retirement","__syncthreads();  // RETIRE","/* missing */  // RETIRE"),
            ("duplicated-global-input","Key::stage(sm.k,","Key::stage(sm.k, p.k, Dim, Chunk); Key::stage(sm.k,")):
            if candidate.count(old)!=1:
                raise AssertionError(f"nonunique source plant:{label}")
            try:
                check_source(control,candidate.replace(old,new,1))
            except AssertionError:
                print(f"[warps8 source negative] {label} EXPECTED-RED/PASS")
            else:
                raise AssertionError(f"escaped source plant:{label}")
    if args.isa:
        isa=args.isa.read_text()
        print("[warps8 native]",check_native(isa))
        if args.self_test:
            from check_wy_binary import plant_in_kernel
            for label,old,new in (
                ("missing-native-mma","v.mma.f32.bf16.m16n16k16","REMOVED_MMA"),
                ("wrong-reader","tsm.ld.swzl","tsm.ld.ncom"),
                ("missing-wait","commit_group(0)","MISSING_WAIT"),
                ("missing-retirement","s.blksyn.defer","MISSING_BARRIER")):
                planted=plant_in_kernel(isa,"gdn_wy_residual_warps8_state",old,new)
                try:
                    check_native(planted)
                except AssertionError:
                    print(f"[warps8 native negative] {label} EXPECTED-RED/PASS")
                else:
                    raise AssertionError(f"escaped native plant:{label}")
    print(f"[warps8] source=PASS native={'PASS' if args.isa else 'NOT_RUN'}; "
          "device numerics/performance=NOT_RUN")


if __name__ == "__main__":
    main()
