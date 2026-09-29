#!/usr/bin/env python3
"""Bind mixed-tail source, native CFG and linked ABI to immutable controls."""
import argparse
from collections import Counter
from pathlib import Path
import re

from check_state_pipeline import block, code, native_records, reachable
from check_full_chunk import replace_once, expect_red, ARGS
from host_disassembly import read_host_disassembly

ROOT = Path(__file__).resolve().parents[2]
SUBJECT = "gdn_wy_residual_mixed_tail_state"
GENERIC = "gdn_wy_residual_gate_cache_state"
FULL = "gdn_wy_residual_full_chunk_state"
ENTRY = "gdn_wy_forward_residual_mixed_tail"
FALLBACK = "gdn_wy_forward_residual_full_chunk"
LOOP = "for (int ct = 0; ct < p.shape.chunks(); ++ct)"
CALL_ARGS = "p,ws,shared,b,h,qh,v0,tid,warp,lane,h_store_base,b_store_base,state,"


def contents(region):
    return region[region.index("{")+1:-1]


def check_source(generic, full, subject, composition, header, binding):
    generic, full, subject, composition, header, binding = map(
        code, (generic, full, subject, composition, header, binding))
    helper = contents(block(subject, "CUTE_DEVICE void step("))
    helper = replace_once(helper, "auto& sm = shared.matrices; auto& coefficients = shared.coefficients;", "")
    for is_full, control, marker in ((False, generic, GENERIC), (True, full, FULL)):
        actual = helper
        actual = replace_once(actual, "valid = Full ? Chunk : Plan::valid(ct, p.shape.sequence)",
                              "valid = Chunk" if is_full else "valid = Plan::valid(ct, p.shape.sequence)")
        if is_full:
            actual = replace_once(actual,
                "(Full || tid < unsigned(valid)) ? float(p.beta[(int64_t(b) * p.shape.sequence + first + tid) * p.shape.value_heads + h]) : 0.0f",
                "float(p.beta[(int64_t(b) * p.shape.sequence + first + tid) * p.shape.value_heads + h])")
            actual = replace_once(actual, "(Full || row < valid) ? beta[half] * difference : 0.0f", "beta[half] * difference")
            actual = replace_once(actual, "(Full || row < valid) ? value[r][s] : 0.0f", "value[r][s]")
        else:
            actual = actual.replace("(Full||tid<unsigned(valid))", "tid<unsigned(valid)")
            actual = actual.replace("(Full||row<valid)", "row<valid")
        expected = contents(block(block(control, marker + "("), LOOP))
        if actual != expected:
            raise AssertionError("mixed-tail iteration changes control arithmetic/layout/rounding: " + str(is_full))
    kernel = block(subject, SUBJECT + "(")
    sequence = ("Partitionconstpartition(p.shape.sequence);#pragmaunroll1"
        "for(intct=0;ct<partition.full_chunks;++ct){step<true>(" + CALL_ARGS + "ct);}"
        "step<false>(" + CALL_ARGS + "partition.full_chunks);")
    kernel = replace_once(kernel, sequence, "#pragma unroll 1 " + block(generic, LOOP))
    kernel = replace_once(kernel, "auto& sm = shared.matrices;",
                          "auto& sm = shared.matrices; auto& coefficients = shared.coefficients;")
    kernel = kernel.replace(SUBJECT + "(", GENERIC + "(", 1)
    if kernel != block(generic, GENERIC + "("):
        raise AssertionError("mixed-tail changed initialization, resident state, owners or publication")
    expected_header = code('''struct Partition {
      int full_chunks; int tail_rows;
      constexpr explicit Partition(int sequence)
        : full_chunks(sequence > 0 ? sequence / Chunk : 0),
          tail_rows(sequence > 0 ? sequence % Chunk : 0) {}
      constexpr bool eligible() const { return full_chunks > 0 && tail_rows > 0; }
    }''')
    if block(header, "struct Partition") != expected_header:
        raise AssertionError("mixed-tail partition admission changed")
    body = block(composition, ENTRY + "(")
    body = replace_once(body, "if (!mixed_tail::Partition(sequence).eligible()) { return " + FALLBACK + "(" + ARGS + "); }", "")
    parent = block(full, FALLBACK + "(")
    parent = replace_once(parent, block(parent, "if (!gate_cache::full_chunks(sequence))"), "")
    parent = replace_once(parent, "using full_chunk::" + FULL + ";", "")
    parent = replace_once(parent, "using namespace residual_warps8_hvlayout;", "using residual_warps8_hvlayout::Key;")
    parent = replace_once(parent, "int rc; rc = solve_static::configure();", "int rc = solve_static::configure();")
    parent = replace_once(parent,
        "rc = int(hggcFuncSetAttribute(" + FULL + ", hggcFuncAttributeMaxDynamicSharedMemorySize, sizeof(gate_cache::Storage)));",
        "rc = mixed_tail::configure_state();")
    parent = replace_once(parent, "rc = configure_hvlayout_output();", "rc = residual_warps8_hvlayout::configure_hvlayout_output();")
    parent = replace_once(parent,
        "unsigned const grid = unsigned(int64_t(batch) * value_heads * (Dim / ValueTile)); " + FULL + "<<<grid, Plan::Threads, sizeof(gate_cache::Storage), stream>>>(p, ws, final); rc = int(hggcGetLastError());",
        "rc = mixed_tail::launch_state(p, ws, final, stream);")
    parent = replace_once(parent, "return launch_hvlayout_output(", "return residual_warps8_hvlayout::launch_hvlayout_output(")
    if body.replace(ENTRY + "(", FALLBACK + "(", 1) != parent:
        raise AssertionError("mixed-tail host stages/admission/workspace differ")
    for text, token in ((subject, "hggcFuncSetAttribute(" + SUBJECT + ",hggcFuncAttributeMaxDynamicSharedMemorySize,sizeof(gate_cache::Storage))"),
        (subject, SUBJECT + "<<<grid,Plan::Threads,sizeof(gate_cache::Storage),stream>>>(p,ws,final)"),
        (subject, "unsignedconstgrid=unsigned(int64_t(p.shape.batch)*p.shape.value_heads*(Dim/ValueTile));"),
        (binding, 'm.def("residual_mixed_tail",&forward<true,14,5>,'),
        (binding, "FullStages==5?" + ENTRY + ":launch;"),
        (binding, "rc=selected(q.data_ptr()")):
        if text.count(token) != 1:
            raise AssertionError("mixed-tail ignored launch/resource/binding: " + token)


def check_native(isa):
    from check_wy_binary import kernel_sequences
    kernels = kernel_sequences(isa)
    section = next((s for s in isa.split("Disassembly of section ") if
                    s.startswith(".text.kernel.") and SUBJECT + "E" in s.splitlines()[0]), None)
    if section is None:
        raise AssertionError("mixed-tail native image missing")
    records, edges = native_records(section)
    reverse = {pc: [] for pc in records}
    for pc, successors in edges.items():
        for successor in successors:
            reverse[successor].append(pc)
    mma = {pc for pc, op in records.items() if op.startswith("v.mma.f32.bf16.m16n16k16")}
    components = {frozenset(reachable(pc, edges) & reachable(pc, reverse)) for pc in mma}
    loops = [c for c in components if len(c) > 1]
    if len(mma) != 40 or len(loops) != 1 or len(mma & loops[0]) != 20:
        raise AssertionError("require one20-MMA full loop and one20-MMA acyclic tail")
    loop = loops[0]
    tail = set(records) - loop
    fixed = ("v.mma.", "vmem.aiu.", "tsm.ld.swzl", "v.exp2.")
    selected = lambda ops: Counter(op.split()[0] for op in ops if op.startswith(fixed))
    control = next(ops for name, ops in kernels.items() if FULL + "E" in name)
    for region in (loop, tail):
        if selected(records[pc] for pc in region) != selected(control):
            raise AssertionError("mixed-tail per-iteration math/AIU/ld.swzl counts changed")
    if any(records[pc].startswith(("s.min.", "v.csel.")) for pc in loop):
        raise AssertionError("constant full loop retained dynamic tail predicates")
    if not any(records[pc].startswith("s.min.") for pc in tail):
        raise AssertionError("final tail lost runtime valid extent")
    if any("ivreg" in op or "shuffle" in op or "tsm.ld.ncom" in op for op in records.values()):
        raise AssertionError("mixed-tail added compatibility repair")
    if sum(records[pc].startswith("s.blksyn") for pc in loop) != 4 or sum(
            records[pc].startswith("s.blksyn") for pc in tail) != 5:
        raise AssertionError("mixed-tail lost4 per-step barriers + final publication barrier")
    # No path out of a tail MMA may return into the full-prefix component.
    if any(reachable(pc, edges) & loop for pc in mma - loop):
        raise AssertionError("tail re-enters the full-prefix recurrence")
    return dict(native_sites=len(records), prefix_sites=len(loop), prefix_mma_sites=20,
                tail_mma_sites=20, prefix_min_csel=0, matrix_path="AIU.swzl+ld.swzl", device="NOT_RUN")


def check_linked_host(host):
    from check_gate_cache_solve import functions, reachable_calls
    calls = reachable_calls(functions(host), ENTRY)
    for marker in (FALLBACK, "mixed_tail::configure_state(", "mixed_tail::launch_state(",
                   "solve_static::launch_inverse(", "launch_hvlayout_output("):
        if not any(marker in name for name in calls):
            raise AssertionError("linked mixed-tail entry lost " + marker)
    return dict(stages=4, unchanged_fallback=FALLBACK, device="NOT_RUN")


def check_linked_binding(host):
    from check_gate_cache_solve import functions, reachable_calls
    bodies = functions(host)
    entries = [n for n in bodies if re.search(r"::forward<true, 14u, 5u(?:, 0u)?>\(", n) and not n.endswith(" [clone .cold]")]
    if len(entries) != 1:
        raise AssertionError("mixed-tail Python specialization missing/ambiguous")
    targets = {n for n in reachable_calls(bodies, entries[0]) if n.startswith("gdn_wy_forward")}
    if targets != {ENTRY}:
        raise AssertionError("compiled mixed-tail binding did not select exact C ABI")
    return dict(binding=ENTRY, device="NOT_RUN")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    for name in ("isa", "library", "binding"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    names = ["csrc/gdn_chunk/" + name for name in (
        "gdn_wy_residual_gate_cache_ppu.cu", "gdn_wy_residual_full_chunk_ppu.cu",
        "gdn_wy_mixed_tail_state_ppu.cu", "gdn_wy_mixed_tail_ppu.cu")]
    names += ["include/gdn_qsa/ppu/wy_mixed_tail.hpp", "csrc/gdn_chunk/gdn_wy_ops.cpp"]
    texts = [(ROOT / name).read_text() for name in names]
    check_source(*texts)
    if args.self_test:
        for label, index, old, new in (
            ("skip-tail", 2, "  step<false>(", "  step<true>("),
            ("prefix-denominator", 2, "ct < partition.full_chunks", "ct < partition.full_chunks - 1"),
            ("repeat-tail", 2, "state, partition.full_chunks);", "state, partition.full_chunks - 1);"),
            ("reset-at-boundary", 2, "  step<false>(", "  state[0][0] = 0; step<false>("),
            ("tail-rounded", 2, "BF16(x * row_decay", "BF16(float(BF16(x)) * row_decay"),
            ("lost-retirement", 2, "__syncthreads();  // RETIRE", "/* missing */ // RETIRE"),
            ("tail-count", 4, "sequence % Chunk", "sequence % Chunk + 1"),
            ("wrong-alias", 3, "inverse_ws.snapshots = ws.w;", "inverse_ws.snapshots = ws.snapshots;"),
            ("old-body", 3, "rc = mixed_tail::launch_state(", "rc = full_chunk::launch_state("),
            ("wrong-binding", 5, "&forward<true, 14, 5>", "&forward<true, 14>"),
        ):
            plant = list(texts)
            if plant[index].count(old) != 1:
                raise AssertionError("ambiguous source negative: " + label)
            plant[index] = plant[index].replace(old, new, 1)
            expect_red(label, check_source, *plant)
    if args.isa:
        isa = args.isa.read_text()
        print("[mixed tail native]", check_native(isa))
        if args.self_test:
            from check_wy_binary import plant_in_kernel
            for opcode in ("v.mma.f32.bf16.m16n16k16", "vmem.aiu.ld", "tsm.ld.swzl", "s.blksyn", "s.min.i32"):
                expect_red("missing-" + opcode, check_native,
                           plant_in_kernel(isa, SUBJECT, opcode, "MISSING_OPCODE"))
    for path, label, check in ((args.library, "host", check_linked_host), (args.binding, "binding", check_linked_binding)):
        if path:
            disasm = read_host_disassembly(path)
            print("[mixed tail " + label + "]", check(disasm))
            if args.self_test:
                expect_red("missing-linked-" + label, check, disasm.replace(ENTRY, "MISSING_ENTRY"))
    print("[mixed tail] source=PASS native=" + ("PASS" if args.isa else "NOT_RUN") + " device=NOT_RUN")


if __name__ == "__main__":
    main()
