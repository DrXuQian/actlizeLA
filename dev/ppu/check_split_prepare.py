#!/usr/bin/env python3
"""Bind the new structure to real source arithmetic/lifetimes, not a timing model.

l018 proves shipping helpers/ownership. This gate additionally compares the
complete FP32 diagonal+block solve to the frozen prepare_inverse authority,
and checks launch/publication order that host coordinate tests cannot execute.
It is not a substitute for RAW-BIT device admission.
"""
import argparse
from pathlib import Path
import re
from check_state_pipeline import block

ROOT = Path(__file__).resolve().parents[2]


def code(text):
    text = re.sub(r"//[^\n]*|/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"\s+", "", text)


def solve(text, end):
    text = text[text.index("float column[16];"):text.index(end)]
    # Signedness only affects provably nonnegative address arithmetic. These
    # two explicit casts are the only permitted source spelling differences.
    return code(text.replace("int(warp)", "warp").replace("int(lane % 16)", "(lane % 16)"))


def check(control, subject, launcher):
    if solve(control, "// All inverse-merge") != solve(subject, "// Temp readers"):
        raise AssertionError("FP32 solve/reduction/TF32 residual cadence differs from control")
    c, s, launch = code(control), code(subject), code(launcher)
    for expression in (
            "acc[s] * sm.beta[r] * expf(sm.prefix[r] - sm.prefix[c])",
            "sm.lower[r * Chunk + c] = r > c",
            "for (int k = 0; k < Dim; k += 16)"):
        if code(expression) not in c or code(expression) not in s:
            raise AssertionError(f"KKT arithmetic changed: {expression}")
    # Restrict to the original launch chain: an isolated solve now reuses
    # prefix through a second host-only entry, checked by check_solve_static.
    chain = block(s, 'intlaunch_split_inverse(') + block(s, 'intlaunch_split_prepare(')
    calls = re.findall(r"(gdn_wy_split_\w+)<<<", chain)
    if calls != ["gdn_wy_split_prefix", "gdn_wy_split_solve", "gdn_wy_split_wu"]:
        raise AssertionError("same-stream scratch producer/consumer launch order changed")
    for expression in (
            "gdn_wy_split_prefix<<<grid, Plan::PrefixThreads, 0, stream>>>",
            "gdn_wy_split_solve<<<grid, Plan::SolveThreads, sizeof(SolveStorage), stream>>>",
            "gdn_wy_split_wu<<<grid, Plan::WUThreads, sizeof(WUStorage), stream>>>",
            "sm.result[i] = BF16(i / Chunk >= i % Chunk ? sm.inverse[i] : 0.0f)",
            "ws.snapshots + Plan::inverse_base(group)",
            "Inverse::stage(sm.inverse, ws.snapshots + Plan::inverse_base(group), Chunk, Chunk)",
            "__syncthreads(); CUTE_UNROLL for (unsigned f = 0; f < Plan::WUFragments; ++f)",
            "__syncthreads(); Key::publish<Plan::WUThreads>(sm.key",
            "bf16_mma(w[f], a, bk); bf16_mma(u[f], a, bv)",
            "for (int k = 0; k < Chunk; k += 16)"):
        if code(expression) not in s:
            raise AssertionError(f"split prepare placement/lifetime not bound: {expression}")
    prepare_call = "rc=split?launch_split_prepare(p,ws,stream):launch_prepare_rows(p,ws,stream,PrepareRowsShared);"
    if prepare_call not in launch or launch.index(prepare_call) > launch.index("aiu::gdn_wy_aiu_state<<<"):
        raise AssertionError("selector ignored or snapshot scratch overwritten before WU completes")
    if "visit_aiu_options(options&AiuOptions" not in launch:
        raise AssertionError("AIU state/output selection no longer explicit")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    control = (ROOT / "csrc/gdn_chunk/gdn_wy_prepare.cuh").read_text()
    subject = (ROOT / "csrc/gdn_chunk/gdn_wy_split_prepare_ppu.cu").read_text()
    launcher = (ROOT / "csrc/gdn_chunk/gdn_wy_aiu_ppu.cu").read_text()
    check(control, subject, launcher)
    if args.self_test:
        for name, old, new in (
                ("inverse-sign", "= -merged[s]", "= merged[s]"),
                ("inverse-gap-order", "gap = 1; gap < 4; ++gap", "gap = 3; gap > 0; --gap"),
                ("unwritten-upper-inverse", "? sm.inverse[i] : 0.0f", "? sm.inverse[i] : sm.inverse[i]"),
                ("wrong-scratch-read-pitch", "Plan::inverse_base(group), Chunk, Chunk", "Plan::inverse_base(group), Dim, Chunk"),
                ("wrong-stream", "sizeof(WUStorage), stream>>>", "sizeof(WUStorage), 0>>>"),
                ("missing-reader-retirement", "__syncthreads();  // all key/value readers", "// all key/value readers"),
                ("missing-publish-handoff", "__syncthreads();\n  Key::publish", "Key::publish"),
                ("skip-WU-launch", "gdn_wy_split_wu<<<", "gdn_wy_split_solve<<<")):
            if subject.count(old) != 1:
                raise AssertionError(f"plant target not unique: {name}")
            try:
                check(control, subject.replace(old, new, 1), launcher)
            except AssertionError:
                print(f"[WY split source negative] {name} EXPECTED-RED/PASS")
            else:
                raise AssertionError(f"escaped source negative: {name}")
        try:
            check(control, subject, launcher.replace("rc = split ? launch_split_prepare", "rc = false ? launch_split_prepare"))
        except AssertionError:
            print("[WY split source negative] ignored-selector EXPECTED-RED/PASS")
        else:
            raise AssertionError("ignored split selector escaped")
    print("[WY split source] FP32 solve=CONTROL-IDENTICAL native helpers/stream-order/alias-handoffs PASS device_execution=NOT_RUN")


if __name__ == "__main__":
    main()
