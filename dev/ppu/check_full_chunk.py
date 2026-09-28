#!/usr/bin/env python3
"""Prove the full-C64 specialization without relabeling local codegen as speed."""
import argparse
from collections import Counter
from pathlib import Path
import re
from host_disassembly import read_host_disassembly

from check_state_pipeline import block, code

ROOT = Path(__file__).resolve().parents[2]
CONTROL = "gdn_wy_residual_gate_cache_state"
SUBJECT = "gdn_wy_residual_full_chunk_state"
ENTRY = "gdn_wy_forward_residual_full_chunk"
FALLBACK = "gdn_wy_forward_residual_gate_cache_solve_static"
ARGS = "q,k,v,g,beta,initial,output,final,inverse,snapshots,vnew,gates,batch,sequence,q_heads,value_heads,gate_fp32,stream"


def replace_once(text, old, new):
    old, new = code(old), code(new)
    if text.count(old) != 1:
        raise AssertionError("missing/ambiguous full-chunk seam: " + old)
    return text.replace(old, new, 1)


def check_source(control, subject, header, binding):
    old, new = code(control), code(subject)
    state = block(new, SUBJECT + "(").replace(SUBJECT, CONTROL, 1)
    # Restore only the four redundant tail conditions. Everything else,
    # including floating-point order, casts, layouts and barriers, must agree.
    for before, after in (
        ("valid = Chunk", "valid = Plan::valid(ct, p.shape.sequence)"),
        ("sm.beta[tid] = float(p.beta[(int64_t(b) * p.shape.sequence + first + tid) * p.shape.value_heads + h]);",
         "sm.beta[tid] = tid < unsigned(valid) ? float(p.beta[(int64_t(b) * p.shape.sequence + first + tid) * p.shape.value_heads + h]) : 0.0f;"),
        ("BF16(beta[half] * difference)", "BF16(row < valid ? beta[half] * difference : 0.0f)"),
        ("float const x = value[r][s];", "float const x = row < valid ? value[r][s] : 0.0f;"),
    ):
        state = replace_once(state, before, after)
    if state != block(old, CONTROL + "("):
        raise AssertionError("full chunk changed arithmetic/layout/traffic/lifetime")
    if block(code(header), "constexpr bool full_chunks(") != code(
        "constexpr bool full_chunks(int sequence) { return sequence > 0 && sequence % Chunk == 0; }"):
        raise AssertionError("full chunk accepts an invalid/tail sequence")
    body = block(new, ENTRY + "(")
    body = replace_once(body, "if (!gate_cache::full_chunks(sequence)) { return " + FALLBACK + "(" + ARGS + "); }", "")
    body = replace_once(body, "using full_chunk::" + SUBJECT + ";", "using gate_cache::" + CONTROL + ";")
    body = replace_once(body, "hggcFuncSetAttribute(" + SUBJECT + ",", "hggcFuncSetAttribute(" + CONTROL + ",")
    body = replace_once(body, SUBJECT + "<<<", CONTROL + "<<<")
    body = body.replace(ENTRY + "(", "forward_gate_cache(", 1)
    expected = block(old, "forward_gate_cache(")
    expected = replace_once(expected,
        "if constexpr (StaticSolve) rc = solve_static::configure(); else rc = configure_split_prepare();",
        "rc = solve_static::configure();")
    expected = replace_once(expected,
        "if constexpr (StaticSolve) rc = solve_static::launch_inverse(p, inverse_ws, stream); else rc = launch_split_inverse(p, inverse_ws, stream);",
        "rc = solve_static::launch_inverse(p, inverse_ws, stream);")
    if body != expected:
        raise AssertionError("full chunk changed admission/workspace/launch/stage composition")
    for token in (
        "Variant<=14&&(!Variant||Residual)",
        "Variant==13?gdn_wy_forward_residual_gate_cache_solve_static:gdn_wy_forward_residual_full_chunk;",
        'm.def("residual_full_chunk",&forward<true,14>,',
    ):
        if code(binding).count(token) != 1:
            raise AssertionError("full-chunk Python binding is not selected")


def check_native(isa):
    # Import late: the complete binary gate also calls this check.
    from check_wy_binary import kernel_sequences
    kernels = kernel_sequences(isa)
    def select(marker):
        matches = [ops for name, ops in kernels.items() if marker + "E" in name]
        if len(matches) != 1:
            raise AssertionError("missing/ambiguous state image: " + marker)
        return matches[0]
    old, new = select(CONTROL), select(SUBJECT)
    a, b = (Counter(op.split()[0] for op in seq) for seq in (old, new))
    fixed = ("v.mma.", "vmem.aiu.", "tsm.ld.", "vmem.ld.", "vmem.st.",
             "v.exp2.", "v.fma.f32", "v.mul.f32", "v.add.f32", "s.blksyn", "vmem.fence", "vmem.acp.")
    if {k:v for k,v in a.items() if k.startswith(fixed)} != {
            k:v for k,v in b.items() if k.startswith(fixed)}:
        raise AssertionError("full chunk altered useful math, matrix traffic or synchronization")
    if any("ivreg" in op or "shuffle" in op or "tsm.ld.ncom" in op for op in new):
        raise AssertionError("full chunk introduced compatibility repair")
    # Codegen hypothesis must actually occur; copying/renaming the old body
    # is not a valid experiment. Backedge spans may rotate: do not zip them.
    if (len(new) >= len(old) or b["s.min.i32"] or b["v.csel.b32"] or
            b["s.cbr.az"] >= a["s.cbr.az"]):
        raise AssertionError("full-chunk tail simplification was not emitted")
    return dict(control_sites=len(old), candidate_sites=len(new),
                matrix_path="AIU.swzl+ld.swzl unchanged", device="NOT_RUN")


def compare_controls(parent, candidate):
    from check_wy_binary import kernel_sequences
    old, new = kernel_sequences(parent), kernel_sequences(candidate)
    if len(old) != 37 or len(new) != 38:
        raise AssertionError("full-chunk denominator must be 37 controls + 1 candidate")
    extra = set(new) - set(old)
    if len(extra) != 1 or SUBJECT + "E" not in next(iter(extra)):
        raise AssertionError("wrong new image or missing old control")
    for name, ops in old.items():
        if new.get(name) != ops:
            raise AssertionError("control native instructions/operands changed: " + name)


def check_linked_host(host):
    from check_gate_cache_solve import functions, reachable_calls
    calls = reachable_calls(functions(host), ENTRY)
    for marker in (FALLBACK, SUBJECT + "(", "solve_static::configure(",
                   "solve_static::launch_inverse(", "launch_hvlayout_output("):
        if not any(marker in name for name in calls):
            raise AssertionError("linked full-chunk entry lost " + marker)
    return dict(full_state=True, static_solve=True, tail_fallback=FALLBACK, device="NOT_RUN")


def expect_red(label, check, *args):
    try:
        check(*args)
    except AssertionError:
        print("[full chunk negative]", label, "EXPECTED-RED/PASS")
    else:
        raise AssertionError("negative escaped: " + label)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--isa", type=Path)
    p.add_argument("--parent-isa", type=Path)
    p.add_argument("--library", type=Path)
    a = p.parse_args()
    src = ROOT / "csrc/gdn_chunk"
    texts = [(src / "gdn_wy_residual_gate_cache_ppu.cu").read_text(),
             (src / "gdn_wy_residual_full_chunk_ppu.cu").read_text(),
             (ROOT / "include/gdn_qsa/ppu/wy_full_chunk.hpp").read_text(),
             (src / "gdn_wy_ops.cpp").read_text()]
    check_source(*texts)
    if a.self_test:
        for label, index, old, new in (
            ("tail-eligible", 2, "sequence % Chunk == 0", "sequence % Chunk != 0"),
            ("missing-beta-row", 1, "if (tid < Chunk)", "if (tid < Chunk - 1)"),
            ("changed-rounding", 1, "BF16(beta[half] * difference)", "BF16(difference)"),
            ("lost-retirement", 1, "__syncthreads();  // RETIRE", "/* missing */ // RETIRE"),
            ("wrong-fallback", 1, "return " + FALLBACK + "(", "return gdn_wy_forward_residual_gate_cache("),
            ("wrong-launch", 1, SUBJECT + "<<<", CONTROL + "<<<"),
            ("wrong-binding", 3, "&forward<true, 14>", "&forward<true, 13>"),
        ):
            changed = list(texts)
            if changed[index].count(old) != 1:
                raise AssertionError("ambiguous negative: " + label)
            changed[index] = changed[index].replace(old, new, 1)
            expect_red(label, check_source, *changed)
    if a.isa:
        isa = a.isa.read_text()
        print("[full chunk native]", check_native(isa))
        if a.self_test:
            from check_wy_binary import plant_in_kernel
            for op in ("v.mma.f32.bf16.m16n16k16", "v.exp2.f32", "s.blksyn.defer",
                       "tsm.ld.swzl.b32x4.s0.t1.trans1"):
                expect_red("missing-" + op, check_native, plant_in_kernel(isa, SUBJECT, op, "MISSING"))
            sections = isa.split("Disassembly of section ")
            old_body = next(s for s in sections if s.startswith(".text.kernel.") and CONTROL + "E" in s.splitlines()[0])
            index = next(i for i,s in enumerate(sections) if s.startswith(".text.kernel.") and SUBJECT + "E" in s.splitlines()[0])
            name = sections[index].splitlines()[0]
            sections[index] = name + "\n" + old_body.split("\n", 1)[1]
            expect_red("renamed-old-body", check_native, "Disassembly of section ".join(sections))
        if a.parent_isa:
            parent = a.parent_isa.read_text()
            compare_controls(parent, isa)
            if a.self_test:
                expect_red("control-changed", compare_controls, parent,
                           isa.replace("v.mma.f32.bf16", "MISSING", 1))
                expect_red("short-denominator", compare_controls, parent,
                           isa[:isa.index("Disassembly of section ")])
            print("[full chunk controls] 37/37 native instruction+operand sequences IDENTICAL")
    elif a.parent_isa:
        p.error("--parent-isa requires --isa")
    if a.library:
        host = read_host_disassembly(a.library)
        print("[full chunk host calls]", check_linked_host(host))
        if a.self_test:
            expect_red("linked-new-entry-missing", check_linked_host, host.replace(ENTRY, "MISSING"))
            expect_red("linked-fallback-missing", check_linked_host, host.replace(FALLBACK, "MISSING"))
    print("[full chunk] source=PASS native=" + ("PASS" if a.isa else "NOT_RUN") + " device=NOT_RUN")


if __name__ == "__main__":
    main()
