#!/usr/bin/env python3
"""Bound full-C64 solve/output edits; local proofs are not device speed claims."""
import argparse
from collections import Counter
from pathlib import Path
import re

from check_state_pipeline import block, code
from check_full_chunk import replace_once, expect_red, ARGS
from host_disassembly import read_host_disassembly

ROOT = Path(__file__).resolve().parents[2]
PAIRS = {
    "solve": ("gdn_wy_split_solve_static", "gdn_wy_full_chunk_solve"),
    "output": ("gdn_wy_residual_warps8_hvlayout_output", "gdn_wy_full_chunk_output"),
}
CHOICES = {"solve": ("true", "false"), "output": ("false", "true"), "both": ("true", "true")}
FALLBACK = "gdn_wy_forward_residual_full_chunk"


def check_source(old_solve, new_solve, old_output, new_output, composition, binding, state):
    for role, old, new in (("solve", old_solve, new_solve), ("output", old_output, new_output)):
        a, b = PAIRS[role]
        candidate = block(code(new), b + "(").replace(b + "(", a + "(", 1)
        candidate = replace_once(candidate, "valid = Chunk", "valid = min(Chunk, p.shape.sequence - first)")
        if role == "solve":
            candidate = replace_once(candidate, "first + int(tid)", "first + min(int(tid), valid - 1)")
            candidate = replace_once(candidate, "sm.beta[tid] = float(p.beta[at]);",
                "sm.beta[tid] = tid < unsigned(valid) ? float(p.beta[at]) : 0.0f;")
            if block(code(new), "struct SolveStorage") != block(code(old), "struct SolveStorage"):
                raise AssertionError("solve storage contract changed")
        else:
            candidate = replace_once(candidate, "row >= col ?", "row >= col && row < valid ?")
        if candidate != block(code(old), a + "("):
            raise AssertionError(role + " changed math/layout/causality/rounding/lifetime")
        resource = "SolveStorage" if role == "solve" else "TiledOutputStorage"
        threads = "Plan::SolveThreads" if role == "solve" else "OutputTile::Threads"
        for seam in (f"hggcFuncSetAttribute({b},hggcFuncAttributeMaxDynamicSharedMemorySize,sizeof({resource}))",
                     f"{b}<<<unsigned(p.shape.groups()),{threads},sizeof({resource}),stream>>>"):
            if code(new).count(seam) != 1:
                raise AssertionError(role + " wrong actual attribute/launch")
    text = code(composition)
    body = block(text, "forward_full_chunk_stages(")
    admission = code("if (!gate_cache::full_chunks(sequence)) { return " + FALLBACK + "(" + ARGS + "); }")
    if body.count(admission) != 1 or body.index(admission) > body.index("Inputs p".replace(" ", "")):
        raise AssertionError("tail fallback missing/late/wrong")
    shared_state = block(code(state), "configure_state(") + block(code(state), "launch_state(")
    for seam in (
        "batch <= 0 || sequence <= 0 || q_heads <= 0 || value_heads <= 0 || value_heads % q_heads || !Key::admitted_stride(int64_t(q_heads) * Dim) || !Key::admitted_stride(int64_t(value_heads) * Dim) || inverse == snapshots",
        "inverse_ws.snapshots = ws.w;",
        "if constexpr (FullSolve) rc = full_chunk_solve::configure(); else rc = solve_static::configure();",
        "if constexpr (FullOutput) rc = full_chunk_output::configure(); else rc = configure_hvlayout_output();",
        "if constexpr (FullSolve) rc = full_chunk_solve::launch_inverse(p, inverse_ws, stream); else rc = solve_static::launch_inverse(p, inverse_ws, stream);",
        "rc = full_chunk::configure_state();",
        "rc = full_chunk::launch_state(p, ws, final, stream);",
        "if constexpr (FullOutput) return full_chunk_output::launch_output(p, ws, static_cast<BF16*>(output), stream); else return launch_hvlayout_output(p, ws, static_cast<BF16*>(output), stream);",
    ):
        if body.count(code(seam)) != 1:
            raise AssertionError("wrong composition/lifetime seam: " + seam)
    ordering = [body.index(code(x)) for x in ("Workspace inverse_ws", "full_chunk_solve::launch_inverse(",
                "full_chunk::launch_state(", "full_chunk_output::launch_output(")]
    if ordering != sorted(ordering):
        raise AssertionError("prefix/solve/state/output dependency order changed")
    for suffix, flags in CHOICES.items():
        if text.count(code(f"GDN_FULL_CHUNK_ENTRY({FALLBACK}_{suffix}, {flags[0]}, {flags[1]})")) != 1:
            raise AssertionError("wrong native entry choice: " + suffix)
    compact = code(composition.replace("\\\n", ""))
    if compact.count(code("return gdn_qsa::wy::forward_full_chunk_stages<Solve, Output>(" + ARGS + ");")) != 1:
        raise AssertionError("C ABI arguments changed")
    bound = code(binding)
    for i, suffix in enumerate(CHOICES, 1):
        for seam in (f'FullStages=={i}?{FALLBACK}_{suffix}:',
                     f'm.def("residual_full_chunk_{suffix}",&forward<true,14,{i}>,'):
            if bound.count(seam) != 1:
                raise AssertionError("wrong Python/native stage choice: " + suffix)
    if "rc=selected(q.data_ptr()" not in bound or "FullStages<=6&&(!FullStages||(Residual&&Variant==14))" not in bound:
        raise AssertionError("stage option unused or unrestricted")
    for seam in (
        "hggcFuncSetAttribute(gdn_wy_residual_full_chunk_state,hggcFuncAttributeMaxDynamicSharedMemorySize,sizeof(gate_cache::Storage))",
        "unsigned const grid = unsigned(int64_t(p.shape.batch) * p.shape.value_heads * (Dim / ValueTile));",
        "gdn_wy_residual_full_chunk_state<<<grid, Plan::Threads, sizeof(gate_cache::Storage), stream>>>(p, ws, final);",
    ):
        if shared_state.count(code(seam)) != 1:
            raise AssertionError("shared full-state entry changed")


def check_native(isa):
    from check_wy_binary import kernel_sequences
    kernels = kernel_sequences(isa)
    results = {}
    for role, markers in PAIRS.items():
        sequences = []
        for marker in markers:
            matches = [ops for name, ops in kernels.items() if marker + "E" in name]
            if len(matches) != 1:
                raise AssertionError("missing/ambiguous stage image: " + marker)
            sequences.append(matches[0])
        old, new = sequences
        a, b = (Counter(op.split()[0] for op in seq) for seq in sequences)
        fixed = ("v.mma.", "vmem.aiu.", "tsm.ld.swzl", "vmem.st.", "v.exp2.",
                 "v.fma.f32", "v.mul.f32", "v.add.f32", "s.blksyn", "vmem.acp.", "vmem.fence")
        if {k:v for k,v in a.items() if k.startswith(fixed)} != {k:v for k,v in b.items() if k.startswith(fixed)}:
            raise AssertionError(role + " changed native math/matrix path/publication/synchronization")
        if any("ivreg" in op or "shuffle" in op or "tsm.ld.ncom" in op for op in new):
            raise AssertionError(role + " added indirect or compatibility delivery")
        if len(new) >= len(old) or b["s.min.i32"] >= a["s.min.i32"]:
            raise AssertionError(role + " full-chunk simplification did not materialize")
        results[role] = dict(control_sites=len(old), candidate_sites=len(new),
                             control_min=a["s.min.i32"], candidate_min=b["s.min.i32"], device="NOT_RUN")
    return results


def check_linked_host(host):
    from check_gate_cache_solve import functions, reachable_calls
    bodies = functions(host)
    # Inspect each specialization without following its tail fallback into the
    # control, which legitimately contains old solve/output calls.
    for suffix, flags in CHOICES.items():
        entry = FALLBACK + "_" + suffix
        calls = reachable_calls(bodies, entry)
        template = "forward_full_chunk_stages<" + ", ".join(flags) + ">"
        helpers = [name for name in calls if template in name]
        for helper in helpers:
            calls |= reachable_calls(bodies, helper)
        expected = (FALLBACK, "full_chunk::configure_state(", "full_chunk::launch_state(",
                    ("full_chunk_solve" if flags[0] == "true" else "solve_static") + "::configure(",
                    ("full_chunk_solve" if flags[0] == "true" else "solve_static") + "::launch_inverse(",
                    "full_chunk_output::configure(" if flags[1] == "true" else "configure_hvlayout_output(",
                    "full_chunk_output::launch_output(" if flags[1] == "true" else "launch_hvlayout_output(")
        for marker in expected:
            if not any(marker in name for name in calls):
                raise AssertionError(entry + " lost linked target " + marker)
        wrong = (("solve_static" if flags[0] == "true" else "full_chunk_solve") + "::launch_inverse(",
                 "launch_hvlayout_output(" if flags[1] == "true" else "full_chunk_output::launch_output(")
        if any(marker in name for marker in wrong for name in calls):
            raise AssertionError(entry + " reaches the wrong full-stage specialization")
    return dict(entries=3, control="full-chunk", tail="unchanged-control", device="NOT_RUN")


def check_linked_binding(host):
    from check_gate_cache_solve import functions, reachable_calls
    bodies = functions(host)
    for choice, suffix in enumerate(("", "_solve", "_output", "_both")):
        # Exact template instance, not the library's aggregate symbol list.
        marker = f"::forward<true, 14u, {choice}u>("
        entries = [name for name in bodies if marker in name and not name.endswith(" [clone .cold]")]
        if len(entries) != 1:
            raise AssertionError("missing/ambiguous compiled Python forward: " + marker)
        calls = reachable_calls(bodies, entries[0])
        targets = {name for name in calls if name.startswith("gdn_wy_forward")}
        if targets != {FALLBACK + suffix}:
            raise AssertionError("compiled Python stage selection wrong: " + marker)
    return dict(forward_specializations=4, exact_cabi_targets=True, device="NOT_RUN")


def compare_resources(parent, current):
    def records(text):
        pairs = re.findall(r"^Func \d+ (\S+) RESOURCE INFO:\n(.*?)(?=^Func \d+ \S+ RESOURCE INFO:|^ELF FILE |\Z)",
                           text, flags=re.S | re.M)
        if len({name for name, _ in pairs}) != len(pairs):
            raise AssertionError("duplicate resource image")
        return {name: body.strip() for name, body in pairs}
    old, new = records(parent), records(current)
    extra = set(new) - set(old)
    if len(old) != 38 or len(new) != 40 or len(extra) != 2 or not all(
            any(marker + "E" in name for name in extra) for _, marker in PAIRS.values()):
        raise AssertionError("resource denominator must be38 controls +2 new stages")
    for name, record in old.items():
        if new.get(name) != record:
            raise AssertionError("control resource/ABI changed: " + name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--isa", type=Path)
    p.add_argument("--library", type=Path)
    p.add_argument("--binding", type=Path)
    p.add_argument("--resources", type=Path)
    p.add_argument("--parent-resources", type=Path)
    a = p.parse_args()
    directory = ROOT / "csrc/gdn_chunk"
    names = ("gdn_wy_solve_static_ppu.cu", "gdn_wy_full_chunk_solve_ppu.cu",
             "gdn_wy_residual_warps8_hvlayout_ppu.cu", "gdn_wy_full_chunk_output_ppu.cu",
             "gdn_wy_full_chunk_stages_ppu.cu", "gdn_wy_ops.cpp", "gdn_wy_residual_full_chunk_ppu.cu")
    texts = [(directory / name).read_text() for name in names]
    check_source(*texts)
    if a.self_test:
        for label, index, old, new in (
            ("solve-wrong-beta", 1, "sm.beta[tid] = float(p.beta[at]);", "sm.beta[tid] = 0.0f;"),
            ("solve-wrong-rounding", 1, "tf32_product(merged,", "bf16_product(merged,"),
            ("output-lost-causal", 3, "row >= col ?", "row >= 0 ?"),
            ("output-missing-store", 3, "Output::publish<", "NO_STORE<"),
            ("output-lost-retirement", 3, "__syncthreads();  // the union", "/* removed */ // the union"),
            ("tail-admitted", 4, "if (!gate_cache::full_chunks(sequence))", "if (false)"),
            ("inverse-alias", 4, "inverse_ws.snapshots = ws.w;", "inverse_ws.snapshots = ws.snapshots;"),
            ("wrong-solve-launch", 4, "full_chunk_solve::launch_inverse(p,", "solve_static::launch_inverse(p,"),
            ("wrong-output-launch", 4, "full_chunk_output::launch_output(p,", "launch_hvlayout_output(p,"),
            ("ignored-choice", 5, "rc = selected(q.data_ptr()", "rc = launch(q.data_ptr()"),
            ("wrong-binding", 5, "&forward<true, 14, 3>", "&forward<true, 14, 1>"),
            ("wrong-shared-state-launch", 6,
             "gdn_wy_residual_full_chunk_state<<<grid, Plan::Threads, sizeof(gate_cache::Storage), stream>>>(p, ws, final);",
             "gdn_wy_residual_gate_cache_state<<<grid, Plan::Threads, sizeof(gate_cache::Storage), stream>>>(p, ws, final);"),
        ):
            if label == "wrong-shared-state-launch":
                plant = list(texts)
                helper = block(code(plant[index]), "launch_state(")
                wrong = replace_once(helper, old, new)
                plant[index] = replace_once(code(plant[index]), helper, wrong)
                expect_red(label, check_source, *plant)
                continue
            if texts[index].count(old) != 1:
                raise AssertionError("negative target ambiguous: " + label)
            plant = list(texts)
            plant[index] = plant[index].replace(old, new, 1)
            expect_red(label, check_source, *plant)
    if a.isa:
        isa = a.isa.read_text()
        print("[full stages native]", check_native(isa))
        if a.self_test:
            from check_wy_binary import plant_in_kernel
            for role, (old, new) in PAIRS.items():
                for opcode in ("v.mma.f32.bf16.m16n16k16", "s.blksyn.defer", "v.exp2.f32"):
                    expect_red(role + "-lost-" + opcode, check_native, plant_in_kernel(isa, new, opcode, "MISSING"))
                sections = isa.split("Disassembly of section ")
                parent = next(s for s in sections if s.startswith(".text.kernel.") and old + "E" in s.splitlines()[0])
                i = next(i for i,s in enumerate(sections) if s.startswith(".text.kernel.") and new + "E" in s.splitlines()[0])
                sections[i] = sections[i].splitlines()[0] + "\n" + parent.split("\n", 1)[1]
                expect_red(role + "-renamed-old-body", check_native, "Disassembly of section ".join(sections))
    if a.library:
        host = read_host_disassembly(a.library)
        print("[full stages linked host]", check_linked_host(host))
        if a.self_test:
            for suffix in CHOICES:
                expect_red("lost-entry-" + suffix, check_linked_host, host.replace(FALLBACK + "_" + suffix, "MISSING"))
    if a.binding:
        host = read_host_disassembly(a.binding)
        print("[full stages compiled Python binding]", check_linked_binding(host))
        if a.self_test:
            for suffix in CHOICES:
                expect_red("wrong-compiled-binding-" + suffix, check_linked_binding,
                           host.replace(FALLBACK + "_" + suffix, FALLBACK))
    if a.parent_resources and a.resources:
        parent, current = a.parent_resources.read_text(), a.resources.read_text()
        compare_resources(parent, current)
        if a.self_test:
            expect_red("control-resource-changed", compare_resources, parent,
                       current.replace("STACK SIZE:0", "STACK SIZE:32", 1))
            expect_red("resource-denominator", compare_resources, parent, "")
        print("[full stages resources] 38/38 native resource/ABI records IDENTICAL")
    elif a.parent_resources or a.resources:
        p.error("--resources and --parent-resources must be supplied together")
    print("[full stages] source=PASS native=" + ("PASS" if a.isa else "NOT_RUN") + " device=NOT_RUN")


if __name__ == "__main__":
    main()
