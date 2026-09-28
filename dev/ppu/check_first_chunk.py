#!/usr/bin/env python3
"""Bind the absent-initial specialization to its source, native guard and ABI."""
import argparse
from collections import Counter
from pathlib import Path
import re

from check_state_pipeline import block, code
from check_full_chunk import ARGS, replace_once, expect_red
from host_disassembly import read_host_disassembly

ROOT = Path(__file__).resolve().parents[2]
ENTRY = "gdn_wy_forward_residual_first_chunk"
FALLBACK = "gdn_wy_forward_residual_full_chunk"
PAIRS = {
    "state": ("gdn_wy_residual_full_chunk_state", "gdn_wy_residual_first_chunk_state"),
    "output": ("gdn_wy_residual_warps8_hvlayout_output", "gdn_wy_first_chunk_output"),
}


def check_source(old_state, new_state, old_output, new_output, composition, header, binding):
    for role, old, new in (("state", old_state, new_state), ("output", old_output, new_output)):
        before, after = PAIRS[role]
        control = block(code(old), before + "(")
        subject = block(code(new), after + "(").replace(after + "(", before + "(", 1)
        guarded = block(subject, "if (has_history(ct))")
        loop = block(guarded, "for (int k = 0; k < Dim; k += 16)")
        if guarded != code("if (has_history(ct)) { CUTE_UNROLL " + loop + " }"):
            raise AssertionError(role + " guard changed more than the history product")
        subject = replace_once(subject, guarded, "CUTE_UNROLL " + loop)
        if role == "state":
            init = block(control, "for (int k = 0; k < StateTile::KFragments; ++k)", "p.initial")
            subject = replace_once(subject, "float state[StateTile::KFragments][8] = {};",
                "float state[StateTile::KFragments][8]; CUTE_UNROLL " + init)
        if subject != control:
            raise AssertionError(role + " changed arithmetic/layout/rounding/publication/lifetime")
    if block(code(header), "constexpr bool eligible(") != code(
            "constexpr bool eligible(int sequence, bool has_initial) { return !has_initial && gate_cache::full_chunks(sequence); }"):
        raise AssertionError("absent-state/full-chunk admission changed")
    if block(code(header), "constexpr bool has_history(") != code(
            "constexpr bool has_history(int chunk) { return chunk != 0; }"):
        raise AssertionError("history predicate must omit only chunk zero")
    # Normalizing just the permitted guard and host helper spellings must
    # recover the complete admitted parent's C ABI. No second argument model.
    body = block(code(composition), ENTRY + "(")
    if body.index("if(!first_chunk::eligible(") > body.index("Inputsp{"):
        raise AssertionError("fallback must precede first-chunk inputs and launches")
    body = replace_once(body,
        "if (!first_chunk::eligible(sequence, initial != nullptr)) { return " + FALLBACK + "(" + ARGS + "); }", "")
    body = body.replace(ENTRY + "(", FALLBACK + "(", 1)
    parent = block(code(old_state), FALLBACK + "(")
    tail = "if (!gate_cache::full_chunks(sequence))"
    parent = replace_once(parent, block(parent, tail), "")
    parent = replace_once(parent, "using full_chunk::gdn_wy_residual_full_chunk_state;", "")
    parent = replace_once(parent, "using namespace residual_warps8_hvlayout;", "using residual_warps8_hvlayout::Key;")
    parent = replace_once(parent, "int rc; rc = solve_static::configure();", "int rc = solve_static::configure();")
    parent = replace_once(parent,
        "rc = int(hggcFuncSetAttribute(gdn_wy_residual_full_chunk_state, hggcFuncAttributeMaxDynamicSharedMemorySize, sizeof(gate_cache::Storage)));",
        "rc = first_chunk::configure_state();")
    parent = replace_once(parent, "rc = configure_hvlayout_output();", "rc = first_chunk::configure_output();")
    parent = replace_once(parent,
        "unsigned const grid = unsigned(int64_t(batch) * value_heads * (Dim / ValueTile)); gdn_wy_residual_full_chunk_state<<<grid, Plan::Threads, sizeof(gate_cache::Storage), stream>>>(p, ws, final); rc = int(hggcGetLastError());",
        "rc = first_chunk::launch_state(p, ws, final, stream);")
    parent = replace_once(parent, "return launch_hvlayout_output(p, ws, static_cast<BF16*>(output), stream);",
        "return first_chunk::launch_output(p, ws, static_cast<BF16*>(output), stream);")
    if parent != body:
        raise AssertionError("first-chunk host body differs beyond the registered fallback/helpers")
    for text in (body, parent):
        if "inverse_ws.snapshots=ws.w;" not in text:
            raise AssertionError("inverse/snapshot alias")
    # The new host composition uses configure/launch helpers; bind them to
    # actual resource and launch sites rather than equating different syntax.
    required = ("solve_static::configure();", "first_chunk::configure_state();",
                "first_chunk::configure_output();", "solve_static::launch_inverse(p,inverse_ws,stream);",
                "first_chunk::launch_state(p,ws,final,stream);",
                "first_chunk::launch_output(p,ws,static_cast<BF16*>(output),stream);")
    if any(body.count(x) != 1 for x in required):
        raise AssertionError("wrong first-chunk stage composition")
    if [body.index(x) for x in required] != sorted(body.index(x) for x in required):
        raise AssertionError("first-chunk dependency order changed")
    for seam in (
        "batch <= 0 || sequence <= 0 || q_heads <= 0 || value_heads <= 0 || value_heads % q_heads || !Key::admitted_stride(int64_t(q_heads) * Dim) || !Key::admitted_stride(int64_t(value_heads) * Dim) || inverse == snapshots",
        "Inputs p{static_cast<BF16 const*>(q), static_cast<BF16 const*>(k), static_cast<BF16 const*>(v), static_cast<BF16 const*>(beta), g, initial, gate_fp32, {batch, sequence, q_heads, value_heads}};",
        "Workspace ws{static_cast<BF16*>(inverse), nullptr, static_cast<BF16*>(snapshots), static_cast<BF16*>(vnew), gates};",
    ):
        if body.count(code(seam)) != 1:
            raise AssertionError("first-chunk admission/arguments changed")
    for role, text, resource, threads, grid in (
        ("state", new_state, "gate_cache::Storage", "Plan::Threads", "grid"),
        ("output", new_output, "TiledOutputStorage", "OutputTile::Threads", "unsigned(p.shape.groups())"),
    ):
        marker = PAIRS[role][1]
        for seam in (
            f"hggcFuncSetAttribute({marker},hggcFuncAttributeMaxDynamicSharedMemorySize,sizeof({resource}))",
            f"{marker}<<<{grid},{threads},sizeof({resource}),stream>>>(p,ws," + ("final);" if role == "state" else "output);"),
        ):
            if code(text).count(code(seam)) != 1:
                raise AssertionError(role + " native launch/resource seam changed")
    if code(new_state).count("unsignedconstgrid=unsigned(int64_t(p.shape.batch)*p.shape.value_heads*(Dim/ValueTile));") != 1:
        raise AssertionError("first-chunk state grid changed")
    bound = code(binding)
    for seam in ("FullStages<=4&&(!FullStages||(Residual&&Variant==14))",
                 "FullStages==4?gdn_wy_forward_residual_first_chunk:",
                 'm.def("residual_first_chunk",&forward<true,14,4>,', "rc=selected(q.data_ptr()"):
        if bound.count(seam) != 1:
            raise AssertionError("first-chunk binding missing or ignored: " + seam)


def history_region(section, mma_sites, load_sites):
    """Find the zero-predicate branch whose fallthrough is only the history product.

    State code is loop-rotated: both the zero branch and nonzero fallthrough's
    unconditional backedge converge at the same block. Output is a forward
    skip. Accept both, never assume numeric PC or compiler BB identifiers.
    """
    labels = {name: int(pc, 16) for pc, name in re.findall(r"^\s*([0-9a-f]+) <([^>]+)>:", section, re.M)}
    records = [(int(pc, 16), op) for pc, op in re.findall(
        r"^\s*([0-9a-f]+):\s+(?:[0-9a-f]{2}\s+){8}\s*([^\n]+)", section, re.M)]
    matches = []
    for i, (pc, op) in enumerate(records):
        branch = re.fullmatch(r"s.cbr.nz\s+(sreg\d+|scc),\s+\S+ <([^>]+)>", op)
        if not branch or branch[2] not in labels:
            continue
        target = labels[branch[2]]
        if target > pc:
            region = [(p, o) for p, o in records if pc < p < target]
        else:
            end = next((j for j in range(i + 1, len(records)) if records[j][1].startswith("s.cbr")), None)
            if end is None or not re.fullmatch(r"s.cbr\s+\S+ <" + re.escape(branch[2]) + ">", records[end][1]):
                continue
            region = records[i + 1:end]
        ops = [o for _, o in region]
        if sum(o.startswith("v.mma.f32.bf16.m16n16k16") for o in ops) != mma_sites:
            continue
        if sum(o.startswith("tsm.ld.swzl") for o in ops) != load_sites or any(
                o.startswith(("s.cbr", "s.blksyn", "vmem.st", "tsm.st", "vmem.aiu", "v.exp2")) for o in ops):
            continue
        # Trace the branch predicate to a real scalar compare with zero;
        # optional emsk AND only restricts lanes, it cannot invent a condition.
        predicate = branch[1]
        producer = None
        for _, prev in reversed(records[:i]):
            parts = prev.split(None, 1)
            if len(parts) != 2 or parts[1].split(",", 1)[0].strip() != predicate:
                continue
            masked = re.fullmatch(r"s.and.b32\s+" + predicate + r",\s+(sreg\d+|scc),\s+emsk", prev)
            if masked:
                predicate = masked[1]
                continue
            producer = re.fullmatch(r"s.cmp.eq.i32\s+" + predicate + r",\s+(sreg\d+),\s+0x0", prev)
            break
        if producer:
            matches.append(dict(branch_pc=pc, merge_pc=target, zero_register=producer[1],
                                omitted_mma_sites=mma_sites, omitted_shared_load_sites=load_sites))
    if len(matches) != 1:
        raise AssertionError(f"missing/ambiguous zero-history native guard: {matches}")
    return matches[0]


def check_native(isa):
    from check_wy_binary import kernel_sequences
    kernels = kernel_sequences(isa)
    result = {}
    for role, (old, new) in PAIRS.items():
        def select(marker):
            matches = [ops for name, ops in kernels.items() if marker + "E" in name]
            if len(matches) != 1:
                raise AssertionError("missing/ambiguous history image: " + marker)
            return matches[0]
        a, b = select(old), select(new)
        fixed = ("v.mma.", "vmem.aiu.", "tsm.ld.swzl", "vmem.st.", "s.blksyn", "v.exp2.", "vmem.acp.")
        counts = lambda ops: Counter(op.split()[0] for op in ops if op.startswith(fixed))
        if counts(a) != counts(b):
            raise AssertionError(role + " changed full history math/delivery/publication/barriers")
        if any("ivreg" in op or "shuffle" in op or "tsm.ld.ncom" in op for op in b):
            raise AssertionError(role + " added compatibility repair")
        section = next(s for s in isa.split("Disassembly of section ") if
                       s.startswith(".text.kernel.") and new + "E" in s.splitlines()[0])
        result[role] = history_region(section, 8 if role == "state" else 16, 16 if role == "state" else 24)
        result[role].update(control_sites=len(a), candidate_sites=len(b), device="NOT_RUN")
    return result


def check_linked_host(host):
    from check_gate_cache_solve import functions, reachable_calls
    calls = reachable_calls(functions(host), ENTRY)
    for marker in (FALLBACK, "solve_static::configure(", "solve_static::launch_inverse(",
                   "first_chunk::configure_state(", "first_chunk::launch_state(",
                   "first_chunk::configure_output(", "first_chunk::launch_output("):
        if not any(marker in name for name in calls):
            raise AssertionError("first-chunk linked host target missing: " + marker)
    return dict(stages=4, fallback=FALLBACK, device="NOT_RUN")


def check_linked_binding(host):
    from check_gate_cache_solve import functions, reachable_calls
    bodies = functions(host)
    entries = [name for name in bodies if "::forward<true, 14u, 4u>(" in name and not name.endswith(" [clone .cold]")]
    if len(entries) != 1:
        raise AssertionError("first-chunk compiled Python specialization missing/ambiguous")
    targets = {name for name in reachable_calls(bodies, entries[0]) if name.startswith("gdn_wy_forward")}
    if targets != {ENTRY}:
        raise AssertionError("compiled first-chunk selection is not the exact new C ABI")
    return dict(binding=ENTRY, device="NOT_RUN")


def compare_resources(parent, current):
    def records(text):
        pairs = re.findall(r"^Func \d+ (\S+) RESOURCE INFO:\n(.*?)(?=^Func \d+ \S+ RESOURCE INFO:|^ELF FILE |\Z)", text, re.S | re.M)
        if len({name for name, _ in pairs}) != len(pairs):
            raise AssertionError("duplicate resource image")
        return {name: body.strip() for name, body in pairs}
    old, new = records(parent), records(current)
    if len(old) != 40 or len(new) != 42 or any(new.get(n) != b for n, b in old.items()):
        raise AssertionError("all40 control resource/ABI records must remain identical")
    extra = set(new) - set(old)
    if len(extra) != 2 or not all(any(marker + "E" in n for n in extra) for _, marker in PAIRS.values()):
        raise AssertionError("wrong new resource inventory")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true")
    for name in ("isa", "library", "binding", "resources", "parent-resources"):
        p.add_argument("--" + name, type=Path)
    a = p.parse_args()
    names = ["csrc/gdn_chunk/" + n for n in (
        "gdn_wy_residual_full_chunk_ppu.cu", "gdn_wy_first_chunk_state_ppu.cu",
        "gdn_wy_residual_warps8_hvlayout_ppu.cu", "gdn_wy_first_chunk_output_ppu.cu",
        "gdn_wy_first_chunk_ppu.cu")]
    names += ["include/gdn_qsa/ppu/wy_first_chunk.hpp", "csrc/gdn_chunk/gdn_wy_ops.cpp"]
    texts = [(ROOT / name).read_text() for name in names]
    check_source(*texts)
    if a.self_test:
        for label, index, old, new in (
            ("skip-second-chunk", 5, "chunk != 0", "chunk > 1"),
            ("ignore-initial", 5, "!has_initial &&", "has_initial ||"),
            ("initial-fallback", 4, "initial != nullptr", "false"),
            ("tail-admitted", 5, "gate_cache::full_chunks(sequence)", "sequence > 0"),
            ("wrong-inverse-alias", 4, "inverse_ws.snapshots = ws.w;", "inverse_ws.snapshots = ws.snapshots;"),
            ("lost-snapshot-write", 1, "Snapshot::publish<", "NO_PUBLISH<"),
            ("uninitialized-qh", 3, "float acc[OutputTile::Fragments][8] = {};", "float acc[OutputTile::Fragments][8];"),
            ("missing-kh-guard", 1, "if (has_history(ct))", "if (true)"),
            ("missing-qh-guard", 3, "if (has_history(ct))", "if (true)"),
            ("wrong-binding", 6, "&forward<true, 14, 4>", "&forward<true, 14, 0>"),
        ):
            plant = list(texts)
            if plant[index].count(old) != 1:
                raise AssertionError("ambiguous first-chunk negative: " + label)
            plant[index] = plant[index].replace(old, new, 1)
            expect_red(label, check_source, *plant)
    if a.isa:
        isa = a.isa.read_text()
        print("[first chunk native]", check_native(isa))
        if a.self_test:
            from check_wy_binary import plant_in_kernel
            for role, (old, new) in PAIRS.items():
                sections = isa.split("Disassembly of section ")
                before = next(s for s in sections if s.startswith(".text.kernel.") and old + "E" in s.splitlines()[0])
                index = next(i for i, s in enumerate(sections) if s.startswith(".text.kernel.") and new + "E" in s.splitlines()[0])
                sections[index] = sections[index].splitlines()[0] + "\n" + before.split("\n", 1)[1]
                expect_red(role + "-renamed-old-body", check_native, "Disassembly of section ".join(sections))
                expect_red(role + "-missing-mma", check_native,
                           plant_in_kernel(isa, new, "v.mma.f32.bf16", "MISSING_MMA"))
                # Removing only this guard must fail even though all MMA and
                # memory instruction counts stay unchanged.
                section = isa.split("Disassembly of section ")[index]
                region = history_region(section, 8 if role == "state" else 16, 16 if role == "state" else 24)
                line = next(line for line in section.splitlines() if re.match(r"\s*" + format(region["branch_pc"], "x") + ":", line))
                expect_red(role + "-guard-inverted", check_native, isa.replace(line, line.replace("s.cbr.nz", "s.cbr.az"), 1))
    for path, label, check in ((a.library, "host", check_linked_host), (a.binding, "binding", check_linked_binding)):
        if path:
            host = read_host_disassembly(path)
            print("[first chunk " + label + "]", check(host))
            if a.self_test:
                expect_red("linked-" + label + "-missing", check, host.replace(ENTRY, "MISSING_ENTRY"))
    if a.parent_resources:
        if not a.resources:
            p.error("--parent-resources requires --resources")
        parent, current = a.parent_resources.read_text(), a.resources.read_text()
        compare_resources(parent, current)
        if a.self_test:
            expect_red("control-resource-changed", compare_resources, parent,
                       current.replace("STACK SIZE:0", "STACK SIZE:32", 1))
            expect_red("resource-denominator-short", compare_resources, parent,
                       current.replace(" RESOURCE INFO:", " MISSING_RESOURCE:", 1))
        print("[first chunk controls] 40/40 resource/ABI records IDENTICAL")
    print("[first chunk] source=PASS native=" + ("PASS" if a.isa else "NOT_RUN") + " device=NOT_RUN")


if __name__ == "__main__":
    main()
