#!/usr/bin/env python3
"""Admit a register-delivery seam, not a new inverse precision algorithm."""
import argparse
from collections import Counter
from pathlib import Path
import re
import subprocess

from check_state_pipeline import block, code
from check_full_chunk import replace_once, expect_red
from check_solve_static import diagonal_region
from host_disassembly import read_host_disassembly

ROOT = Path(__file__).resolve().parents[2]
CONTROL = "gdn_wy_split_solve_static"
SUBJECT = "gdn_wy_split_solve_register"
ENTRY = "gdn_wy_forward_residual_inverse_register"


def check_source(control, subject, helper, full, generic, binding):
    control, subject, helper, full, generic, binding = map(
        code, (control, subject, helper, full, generic, binding))
    old_seam = '''CUTE_UNROLL for (int s = 0; s < 8; ++s) {
      auto const rc = result_coord(lane, s);
      sm.temp[warp][rc.row * 16 + rc.col] = acc[s];
    } __syncwarp(); float merged[8] = {};
    tf32_product(merged, sm.inverse + br * 16 * Chunk + br * 16, Chunk, sm.temp[warp], 16);'''
    new_seam = '''float merged[8] = {};
    product(merged, sm.inverse + br * 16 * Chunk + br * 16, Chunk, acc);'''
    body = block(subject, SUBJECT + "(").replace(SUBJECT, CONTROL, 1)
    if replace_once(body, new_seam, old_seam) != block(control, CONTROL + "("):
        raise AssertionError("changed solve outside the private accumulator delivery seam")
    if block(subject, "struct SolveStorage") != block(control, "struct SolveStorage"):
        raise AssertionError("shared storage/capacity changed")
    expected = {
        "b_owner(": '''b_owner(unsigned lane, int word, int half) {
          auto const b = tf32_coord(int(lane), word);
          return gdn_qsa::ppu::result_owner(b.col + 8 * half, b.row);
        }''',
        "b_value(": '''b_value(unsigned lane, Exchange const& exchange) {
          static_assert(Half >= 0 && Half < 2 && Word >= 0 && Word < 4);
          constexpr int lo = b_owner(0, Word, Half).slot;
          auto const owner = b_owner(lane, Word, Half);
          float const low = exchange.template get<lo>(owner.lane);
          float const high = exchange.template get<lo + 1>(owner.lane);
          return (lane & 16u) ? high : low;
        }''',
        "product(": '''product(float (&acc)[8], float const* a, int lda, float const (&b)[8]) {
          unsigned const lane = unsigned(threadIdx.x) & 31u;
          WarpExchange const exchange{b};
          cute::for_each(cute::make_seq<2>{}, [&](auto half) {
            constexpr int Half = decltype(half)::value;
            uint32_t ah[4], al[4], bh[4], bl[4];
            cute::for_each(cute::make_seq<4>{}, [&](auto word) {
              constexpr int Word = decltype(word)::value;
              auto const rc = tf32_coord(int(lane), Word);
              float const av = a[rc.row * lda + rc.col + 8 * Half];
              float const bv = b_value<Half, Word>(lane, exchange);
              cutlass::tfloat32_t const at(av), bt(bv);
              ah[Word] = at.raw(); bh[Word] = bt.raw();
              al[Word] = cutlass::tfloat32_t(av - float(at)).raw();
              bl[Word] = cutlass::tfloat32_t(bv - float(bt)).raw();
            });
            using Op = cute::PPU0010_16x16x8_F32TF32TF32F32_TN;
            mma<Op>(acc, al, bh); mma<Op>(acc, ah, bl); mma<Op>(acc, ah, bh);
          });
        }''',
    }
    for marker, want in expected.items():
        if block(helper, marker) != code(want):
            raise AssertionError("register mapper/rounding/product order differs: " + marker)
    if "return__shfl_sync(0xffffffffu,values[Slot],peer);" not in helper:
        raise AssertionError("native warp exchange absent")
    for marker in ("int configure()", "int launch_inverse("):
        if block(subject, marker).replace(SUBJECT, CONTROL) != block(control, marker):
            raise AssertionError("register solve not bound to actual launch: " + marker)

    # Normalize the new host composition to the retained full-chunk entry.
    # Only solve and the existing full/generic state selection may differ.
    parent_name = "gdn_wy_forward_residual_full_chunk"
    parent = block(full, parent_name + "(")
    parent = replace_once(parent, block(parent, "if (!gate_cache::full_chunks(sequence))"), "")
    parent = replace_once(parent, "using namespace residual_warps8_hvlayout;", "using residual_warps8_hvlayout::Key;")
    parent = replace_once(parent, "using full_chunk::gdn_wy_residual_full_chunk_state;", "")
    parent = replace_once(parent, "int rc; rc = solve_static::configure();",
        "bool const full = gate_cache::full_chunks(sequence); int rc = inverse_register::configure();")
    parent = replace_once(parent,
        "rc = int(hggcFuncSetAttribute(gdn_wy_residual_full_chunk_state, hggcFuncAttributeMaxDynamicSharedMemorySize, sizeof(gate_cache::Storage)));",
        "rc = full ? full_chunk::configure_state() : gate_cache::configure_state();")
    parent = replace_once(parent, "rc = configure_hvlayout_output();", "rc = residual_warps8_hvlayout::configure_hvlayout_output();")
    parent = replace_once(parent, "solve_static::launch_inverse(", "inverse_register::launch_inverse(")
    parent = replace_once(parent,
        "unsigned const grid = unsigned(int64_t(batch) * value_heads * (Dim / ValueTile)); gdn_wy_residual_full_chunk_state<<<grid, Plan::Threads, sizeof(gate_cache::Storage), stream>>>(p, ws, final); rc = int(hggcGetLastError());",
        "rc = full ? full_chunk::launch_state(p, ws, final, stream) : gate_cache::launch_state(p, ws, final, stream);")
    parent = replace_once(parent, "return launch_hvlayout_output(", "return residual_warps8_hvlayout::launch_hvlayout_output(")
    if block(subject, ENTRY + "(").replace(ENTRY, parent_name, 1) != parent:
        raise AssertionError("composition changes admission/workspace/stages or bypasses new solve")
    # Added generic-state wrappers may only expose the already compiled kernel.
    wrapper = '''int launch_state(Inputs p, Workspace ws, float* final, gdn_arch::Stream stream) {
      unsigned const grid = unsigned(int64_t(p.shape.batch) * p.shape.value_heads * (Dim / ValueTile));
      gdn_wy_residual_gate_cache_state<<<grid, Plan::Threads, sizeof(Storage), stream>>>(p, ws, final);
      return int(hggcGetLastError());
    }'''
    if block(generic, "int launch_state(") != code(wrapper):
        raise AssertionError("generic state wrapper changed geometry/kernel")
    configure = '''int configure_state() {
      return int(hggcFuncSetAttribute(gdn_wy_residual_gate_cache_state,
          hggcFuncAttributeMaxDynamicSharedMemorySize, sizeof(Storage)));
    }'''
    if block(generic, "int configure_state(") != code(configure):
        raise AssertionError("generic state wrapper changed shared-memory admission")
    for token in ('m.def("residual_inverse_register",&forward<true,14,6>,',
                  "FullStages==6?" + ENTRY + ":", "rc=selected(q.data_ptr()"):
        if binding.count(token) != 1:
            raise AssertionError("exact inverse-register binding missing/ambiguous")


def check_native(isa):
    old, _, old_diag = diagonal_region(isa, CONTROL)
    new, edges, diag = diagonal_region(isa, SUBJECT)
    tally = lambda records: Counter(op.split()[0] for op in records.values())
    a, b, d = tally(old), tally(new), tally(diag)
    fixed = ("v.mma.", "v.cnvt.", "v.exp2.", "v.fma.f32", "vmem.", "tsm.ld.swzl", "s.blksyn")
    if Counter({k: v for k, v in a.items() if k.startswith(fixed)}) != Counter(
            {k: v for k, v in b.items() if k.startswith(fixed)}):
        raise AssertionError("inverse register lost/changed math, conversion, copies or barriers")
    if any("ivreg" in op or "tsm.ld.ncom" in op for op in new.values()):
        raise AssertionError("inverse register gained indirect/compatibility repair")
    if (b["v.mma.f32.bf16.m16n16k16"], b["v.mma.f32.tf32.m16n16k8"], b["vmem.st.b32x4"]) != (8, 12, 4):
        raise AssertionError("inverse register lost useful work/publication")
    if b["v.shuffle.idx.b32"] != 16 or a["v.shuffle.idx.b32"] != 0:
        raise AssertionError("require actual sixteen native register shuffles")
    if (a["tsm.st.b32"] - b["tsm.st.b32"], a["tsm.ld.b32"] - b["tsm.ld.b32"]) != (8, 8):
        raise AssertionError("private scratch round trip was not removed")
    # Scratch-address setup can move across these barriers when temp disappears.
    # Bind diagonal coefficient words / ordered arithmetic / stores, not unrelated
    # hoisted integer-address instructions to a linear-PC region.
    words = sum(d[op] * width for op, width in
                (("tsm.ld.b32", 1), ("tsm.ld.b32x2", 2), ("tsm.ld.b32x4", 4)))
    if words != 120 or d["tsm.st.b32"] != 16 or any(
            t <= pc for pc in diag for t in edges[pc] if t in diag):
        raise AssertionError("static diagonal changed coverage or regained backedge")
    if d["v.mul.f32"] or d["v.add.f32"]:
        raise AssertionError("static diagonal FMA contraction changed")
    fmas = [op for op in diag.values() if op.startswith("v.fma.f32.rtte")]
    if len(fmas) != 120 or any(not re.match(r"v\.fma\.f32\.rtte\s+vreg\d+,\s*!vreg\d+", op) for op in fmas):
        raise AssertionError("diagonal lost120 ordered negative-coefficient RTTE FMAs")
    # Bind register operands as well as opcode counts for each three-product
    # sequence. The Ahi and Bhi operands must recur in the third product.
    mma = [re.findall(r"vreg\[\d+:\d+\]", op) for op in new.values()
           if op.startswith("v.mma.f32.tf32.m16n16k8")]
    for first, second, third in zip(mma[::3], mma[1::3], mma[2::3]):
        if (any(len(row) != 4 for row in (first, second, third)) or
            not first[0] == second[0] == third[0] == first[3] == second[3] == third[3] or
            second[1] != third[1] or first[2] != third[2] or
            first[1] == second[1] or first[2] == second[2]):
            raise AssertionError("native high/residual product operands/order changed")
    return dict(control_sites=len(old), candidate_sites=len(new), scratch_store_delta=-8,
                scratch_load_delta=-8, shuffle_sites=16, diagonal_fmas=120,
                tf32_mma_sites=12, cta_barriers=5, device="NOT_RUN")


def check_linked_host(host):
    from check_gate_cache_solve import functions, reachable_calls
    calls = reachable_calls(functions(host), ENTRY)
    for marker in ("inverse_register::configure(", "inverse_register::launch_inverse(",
                   "full_chunk::launch_state(", "gate_cache::launch_state(", "launch_hvlayout_output("):
        if not any(marker in name for name in calls):
            raise AssertionError("linked inverse-register composition lost " + marker)
    if any("solve_static::" in name or name.startswith("gdn_wy_forward_residual_full_chunk") for name in calls):
        raise AssertionError("linked inverse-register entry can bypass new solve")
    return dict(stages=4, tail_solve="register", device="NOT_RUN")


def check_linked_binding(host):
    from check_gate_cache_solve import functions, reachable_calls
    bodies = functions(host)
    entries = [n for n in bodies if "::forward<true, 14u, 6u>(" in n and not n.endswith(" [clone .cold]")]
    if len(entries) != 1:
        raise AssertionError("inverse-register Python specialization missing/ambiguous")
    targets = {n for n in reachable_calls(bodies, entries[0]) if n.startswith("gdn_wy_forward")}
    if targets != {ENTRY}:
        raise AssertionError("compiled binding did not select inverse-register C ABI")
    return dict(binding=ENTRY, device="NOT_RUN")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true")
    for name in ("isa", "library", "binding", "host"):
        p.add_argument("--" + name, type=Path)
    args = p.parse_args()
    files = ("csrc/gdn_chunk/gdn_wy_solve_static_ppu.cu", "csrc/gdn_chunk/gdn_wy_inverse_register_ppu.cu",
             "include/gdn_qsa/ppu/wy_inverse_register.cuh", "csrc/gdn_chunk/gdn_wy_residual_full_chunk_ppu.cu",
             "csrc/gdn_chunk/gdn_wy_residual_gate_cache_ppu.cu", "csrc/gdn_chunk/gdn_wy_ops.cpp")
    texts = [(ROOT / name).read_text() for name in files]
    check_source(*texts)
    if args.self_test:
        for label, i, old, new in (
            ("wrong-slot", 2, "get<lo + 1>", "get<lo>"),
            ("wrong-lane", 2, "owner.lane", "owner.lane ^ 1u"),
            ("wrong-half", 2, "8 * half", "8 * (1 - half)"),
            ("changed-rounding", 2, "cutlass::tfloat32_t const at", "cutlass::bfloat16_t const at"),
            ("changed-product", 2, "mma<Op>(acc, al, bh)", "mma<Op>(acc, ah, bh)"),
            ("wrong-diagonal", 1, "warp * 16 * (Chunk + 1)", "warp * 16 * Chunk"),
            ("lost-barrier", 1, "    __syncthreads();", "    /* omitted barrier */"),
            ("old-solve", 1, "rc = inverse_register::launch_inverse(", "rc = solve_static::launch_inverse("),
            ("inverse-alias", 1, "inverse_ws.snapshots = ws.w;", "inverse_ws.snapshots = ws.snapshots;"),
            ("tail-silent-fallback", 1, ": gate_cache::launch_state(", ": full_chunk::launch_state("),
            ("wrong-tail-resource", 4, "sizeof(Storage)", "sizeof(Storage) - 128"),
            ("wrong-binding", 5, "&forward<true, 14, 6>", "&forward<true, 14>"),
        ):
            plant = texts.copy()
            if old not in plant[i]:
                raise AssertionError("negative seam absent: " + label)
            plant[i] = plant[i].replace(old, new, 1)
            expect_red(label, check_source, *plant)
    if args.host:
        subprocess.run([str(args.host.resolve())], check=True)
        if args.self_test:
            bad = subprocess.run([str(args.host.resolve()), "--omit-last-block"], capture_output=True, text=True)
            if bad.returncode == 0 or "coverage denominator incomplete" not in bad.stderr:
                raise AssertionError("omitted block escaped the actual host proof denominator")
            print("[inverse register negative] omitted-block EXPECTED-RED/PASS")
    if args.isa:
        isa = args.isa.read_text()
        print("[inverse register native]", check_native(isa))
        if args.self_test:
            from check_wy_binary import plant_in_kernel, swap_native_instructions
            from check_residual_metadata import section
            from check_state_pipeline import native_records
            for opcode in ("v.shuffle.idx.b32", "v.cnvt.tf32.f32", "v.mma.f32.tf32", "s.blksyn", "vmem.st.b32x4"):
                expect_red("missing-" + opcode, check_native,
                           plant_in_kernel(isa, SUBJECT, opcode, "MISSING_OPCODE"))
            body = section(isa, SUBJECT)
            records, _ = native_records(body)
            mmaps = [pc for pc, op in records.items() if op.startswith("v.mma.f32.tf32")]
            plant = isa.replace(body, swap_native_instructions(body, mmaps[6], mmaps[7]), 1)
            expect_red("product-order-same-opcode-counts", check_native, plant)
    for path, label, check in ((args.library, "host", check_linked_host), (args.binding, "binding", check_linked_binding)):
        if path:
            host = read_host_disassembly(path)
            print("[inverse register " + label + "]", check(host))
            if args.self_test:
                expect_red("missing-linked-" + label, check, host.replace(ENTRY, "MISSING_ENTRY"))
    print("[inverse register] source=PASS native=" + ("PASS" if args.isa else "NOT_RUN") + " device=NOT_RUN")


if __name__ == "__main__":
    main()
