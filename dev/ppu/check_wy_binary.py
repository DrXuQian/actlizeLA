#!/usr/bin/env python3
"""Fail-closed compile/resource gate; NOT a device numerical verdict."""
import argparse
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_state_pipeline import native_schedule
from check_residual_prefetch import native_schedule as residual_prefetch_schedule, serialized_plant
from check_residual_blayout import check_native as check_blayout_native
from check_residual_warps8 import check_native as check_warps8_native
from check_residual_warps8_blayout import check_native as check_warps8_blayout_native
from check_residual_warps8_operands import check_native as check_warps8_operands_native
from check_residual_warps8_hlayout import check_native as check_warps8_hlayout_native
from check_residual_warps8_hvlayout import check_native as check_warps8_hvlayout_native
from check_residual_metadata import check_native as check_metadata_native
from check_solve_static import check_native as check_solve_static_native
from check_gate_cache import check_native as check_gate_cache_native


def kernel_sequences(isa):
    result = {}
    for section in isa.split("Disassembly of section ")[1:]:
        if not section.startswith(".text.kernel."):
            continue
        name = section.splitlines()[0].removeprefix(".text.kernel.").removesuffix(":")
        result[name] = re.findall(r"^\s*[0-9a-f]+:\s+(?:[0-9a-f]{2}\s+){8}\s*([^\n]+)", section, re.M)
        if not result[name]:
            raise AssertionError(f"empty native instruction sequence: {name}")
    return result


def compare_controls(before, after):
    old, new = kernel_sequences(before), kernel_sequences(after)
    if len(old) != 36 or len(new) != 37:
        raise AssertionError("control comparison must cover all36 old and all37 current images")
    for name, sequence in old.items():
        if new.get(name) != sequence:
            raise AssertionError(f"admitted control native instructions changed: {name}")


def plant_in_kernel(isa, marker, old, new):
    """Mutate the new body only; an old-control failure cannot satisfy this test."""
    sections = isa.split("Disassembly of section ")
    selected = [i for i, section in enumerate(sections)
                if section.startswith(".text.kernel.") and marker in section.splitlines()[0]]
    if len(selected) != 1 or old not in sections[selected[0]]:
        raise AssertionError(f"negative did not locate its exact instruction target: {marker}")
    index = selected[0]
    sections[index] = sections[index].replace(old, new, 1)
    return "Disassembly of section ".join(sections)


def conditioning_region(section):
    """Locate actual native backedges between inverse MMA and W/U MMA.

    Exactly two BF16 shared stores distinguish K/V conditioning from the
    preceding one-store inverse conversion. No hard-coded PC/BB numbers.
    """
    labels = {name: int(pc, 16) for pc, name in re.findall(r"^\s*([0-9a-f]+) <([^>]+)>:", section, re.M)}
    records = [(int(pc, 16), op) for pc, op in re.findall(
        r"^\s*([0-9a-f]+):\s+(?:[0-9a-f]{2}\s+){8}\s*([^\n]+)", section, re.M)]
    tf = [pc for pc, op in records if op.startswith("v.mma.f32.tf32")]
    if not tf:
        raise AssertionError("prepare CFG lost inverse arithmetic")
    after = max(tf)
    wu = [pc for pc, op in records if pc > after and op.startswith("v.mma.f32.bf16")]
    if not wu:
        raise AssertionError("prepare CFG lost W/U arithmetic")
    before = min(wu)
    backedges, regions = [], {}
    for pc, op in records:
        target = re.search(r"<([^>]+)>", op)
        if not op.startswith("s.cbr") or not target:
            continue
        if target[1] not in labels:
            raise AssertionError("native branch target label absent")
        head = labels[target[1]]
        if head <= pc:
            backedges.append((head, pc))
        if after < head <= pc < before:
            body = [text for pos, text in records if head <= pos <= pc]
            if sum(text.startswith("tsm.st.b16") for text in body) == 2:
                regions[head] = max(pc, regions.get(head, pc))
    if len(regions) != 1:
        raise AssertionError(f"conditioning loop not uniquely identified: {regions}")
    head, tail = next(iter(regions.items()))
    repeated = [pc for pc, op in records if head <= pc <= tail and op.startswith("v.exp2.f32")]
    hoisted = [pc for pc, op in records if after < pc < head and op.startswith("v.exp2.f32")]
    return dict(head=head, tail=tail, wu=before, repeated=repeated, hoisted=hoisted,
                records=records, backedges=backedges)


def check_row_hoist(section, expected):
    region = conditioning_region(section)
    if region["repeated"] or len(region["hoisted"]) != expected:
        raise AssertionError("prepare row exponent is missing or still inside the conditioning loop")
    if any(lo <= pc <= hi for pc in region["hoisted"] for lo, hi in region["backedges"]):
        raise AssertionError("row exponent moved into a different native backedge loop")
    publication = [pc for pc, op in region["records"]
                   if max(region["hoisted"]) < pc < region["head"] and op.startswith("s.blksyn.defer")]
    shuffles = [pc for pc, op in region["records"]
                if region["head"] <= pc <= region["tail"] and op.startswith("v.shuffle.idx.b32")]
    if expected == 1:
        if len(publication) != 1 or shuffles:
            raise AssertionError("shared rows lack their pre-consumer publication barrier")
        stores = [pc for pc, op in region["records"]
                  if max(region["hoisted"]) < pc < publication[0] and op.startswith("tsm.st.b32")]
        if len(stores) != 1:
            raise AssertionError("shared row value is not stored before publication")
    elif publication or len(shuffles) != 1:
        raise AssertionError("warp rows lost indexed delivery or added a CTA publication barrier")
    region["publication"] = publication
    return region


def swap_native_instructions(section, first, second):
    """Validator-only ISA-text plant: move complete instructions, not mnemonics.

    The opcode/operand multiset is unchanged. This is a parser negative, NOT
    a modified device image; the displayed encoding bytes are not executed.
    """
    pattern = re.compile(r"^(\s*([0-9a-f]+):\s+(?:[0-9a-f]{2}\s+){8}\s*)([^\n]+)", re.M)
    instructions = {int(m[2], 16): m[3] for m in pattern.finditer(section)}
    if first == second or first not in instructions or second not in instructions:
        raise AssertionError("native instruction swap targets are absent or not distinct")
    replacements = {first: instructions[second], second: instructions[first]}
    return pattern.sub(lambda m: m[1] + replacements.get(int(m[2], 16), m[3]), section)


def plant_repeated_row_exp(isa):
    """Preserve the total exp count but move one back inside conditioning."""
    sections = isa.split("Disassembly of section ")
    matches = [i for i, s in enumerate(sections) if s.startswith(".text.kernel.") and
               "gdn_wy_rows_prepareILi1E" in s.splitlines()[0]]
    if len(matches) != 1:
        raise AssertionError("row-exp plant missing its exact target")
    i = matches[0]
    region = check_row_hoist(sections[i], 1)
    exp_pc = region["hoisted"][0]
    multiply = next(pc for pc, op in region["records"] if
                    region["head"] <= pc <= region["tail"] and op.startswith("v.mul.f32"))
    sections[i] = swap_native_instructions(sections[i], exp_pc, multiply)
    return "Disassembly of section ".join(sections)


def plant_late_row_barrier(isa):
    """Keep all seven barriers, but move cache publication after consumption."""
    sections = isa.split("Disassembly of section ")
    matches = [i for i, s in enumerate(sections) if s.startswith(".text.kernel.") and
               "gdn_wy_rows_prepareILi1E" in s.splitlines()[0]]
    if len(matches) != 1:
        raise AssertionError("publication plant missing its exact target")
    i = matches[0]
    region = check_row_hoist(sections[i], 1)
    late_nop = next(pc for pc, op in region["records"] if
                    region["tail"] < pc < region["wu"] and op.startswith("s.nop"))
    sections[i] = swap_native_instructions(sections[i], region["publication"][0], late_nop)
    return "Disassembly of section ".join(sections)


def plant_serialized_prefetch(isa):
    """Keep every opcode: move the loop W wait immediately after NEXT_W.

    The old three-hand-off count still agrees; only a CFG schedule check can
    reject the resulting loss of overlap. This is ISA-text validator input,
    not a runnable modified binary.
    """
    sections = isa.split("Disassembly of section ")
    matches = [i for i,s in enumerate(sections) if s.startswith(".text.kernel.") and
               "gdn_wy_state_pipelineE" in s.splitlines()[0]]
    if len(matches) != 1: raise AssertionError("prefetch plant exact symbol absent")
    i = matches[0]
    schedule = native_schedule(sections[i])
    # W wait is the first reachable commit_group wait after all16 UPDATEs;
    # CURRENT_INPUTS' wait lies after PROJECT and before the next-W copies.
    project = schedule["project"]
    begin_wait = min(schedule["waits"])
    if begin_wait > min(project): raise AssertionError("W wait anchor changed; inspect CFG")
    copy = max(schedule["future"])
    following = min(pc for pc in schedule["records"] if pc > copy)
    sections[i] = swap_native_instructions(sections[i],begin_wait,following)
    return "Disassembly of section ".join(sections)


def audit(isa, resources, symbols):
    check_blayout_native(isa)
    check_warps8_native(isa)
    check_warps8_blayout_native(isa)
    check_warps8_operands_native(isa)
    check_warps8_hlayout_native(isa)
    check_warps8_hvlayout_native(isa)
    check_metadata_native(isa)
    check_solve_static_native(isa)
    check_gate_cache_native(isa)
    funcs = re.findall(r"Func \d+ (\S+) RESOURCE INFO:\n(.*?)(?=Func \d+ \S+ RESOURCE INFO:|\Z)",
                       resources, flags=re.S)
    if len(funcs) != 37:
        raise AssertionError(f"WY image denominator must be36 controls +1 gate-cache image, got {len(funcs)}")
    rows = []
    mma_counts = {}
    for role, packed in ((role, packed) for role in ("prepare", "state", "output") for packed in (False, True)):
        matches = [(name, body) for name, body in funcs
                   if f"gdn_wy_{role}ILb{int(packed)}E" in name]
        if len(matches) != 1:
            raise AssertionError(f"expected exactly one WY {role}/{packed} image: {len(matches)}")
        name, body = matches[0]
        stack = int(re.search(r"STACK SIZE:(\d+)", body)[1])
        regs = int(re.search(r"vreg_number:(\d+)", body)[1])
        if stack:
            raise AssertionError(f"WY {role} spills: {stack} bytes")
        sections = isa.split("Disassembly of section ")
        own = [x for x in sections if x.splitlines() and name in x.splitlines()[0]]
        if len(own) != 1:
            raise AssertionError(f"missing exact ISA section: {role}")
        for opcode in ("v.mma.f32.bf16.m16n16k16", "tsm.ld.swzl", "vmem.ld.tsm"):
            if opcode not in own[0]:
                raise AssertionError(f"{role} lost {opcode}")
        if role == "prepare" and "v.mma.f32.tf32.m16n16k8" not in own[0]:
            raise AssertionError("FP32 block solve lost its TF32 products")
        if packed:
            if "vmem.st.b32x4" not in own[0]:
                raise AssertionError(f"{role} lost 16-byte vector global stores")
            if "vmem.st.b16" in own[0]:
                raise AssertionError(f"{role} retained scalar BF16 global stores")
            if role == "state" and "\tvmem.ld.b16\t" in own[0]:
                raise AssertionError("state retained scattered scalar BF16 U loads")
        counts = tuple(own[0].count(op) for op in (
            "v.mma.f32.bf16.m16n16k16", "v.mma.f32.tf32.m16n16k8"))
        if not packed:
            mma_counts[role] = counts
        elif counts != mma_counts[role]:
            raise AssertionError(f"{role} delivery change altered static MMA body counts")
        rows.append(dict(role=role, delivery="packed" if packed else "scalar", registers=regs, stack=stack))
    # New tilings deliberately change static loop bodies; do not compare
    # static MMA counts to a differently rolled scalar loop. Exact expected
    # bodies plus l009's per-output reduction order cover the new decomposition.
    for role, bf16, tf32 in (("prepare", 40, 12), ("state", 32, 0), ("output", 40, 0)):
        matches = [(name, body) for name, body in funcs if f"gdn_wy_tiled_{role}E" in name]
        if len(matches) != 1:
            raise AssertionError(f"expected exactly one tiled {role} image: {len(matches)}")
        name, body = matches[0]
        stack = int(re.search(r"STACK SIZE:(\d+)", body)[1])
        regs = int(re.search(r"vreg_number:(\d+)", body)[1])
        if stack:
            raise AssertionError(f"tiled {role} spills: {stack} bytes")
        own = [x for x in isa.split("Disassembly of section ")
               if x.splitlines() and name in x.splitlines()[0]]
        if len(own) != 1:
            raise AssertionError(f"missing exact tiled ISA section: {role}")
        body_isa = own[0]
        counts = tuple(body_isa.count(op) for op in (
            "v.mma.f32.bf16.m16n16k16", "v.mma.f32.tf32.m16n16k8"))
        if counts != (bf16, tf32):
            raise AssertionError(f"tiled {role} MMA body differs: {counts} != {(bf16, tf32)}")
        for opcode in ("tsm.ld.swzl", "vmem.ld.tsm", "vmem.st.b32x4"):
            if opcode not in body_isa:
                raise AssertionError(f"tiled {role} lost {opcode}")
        if "vmem.st.b16" in body_isa:
            raise AssertionError(f"tiled {role} scalar BF16 store regression")
        if role == "state" and ("v.shuffle" in body_isa or "\tvmem.ld.b16\t" in body_isa):
            raise AssertionError("tiled state retained register shuffle or scalar U load")
        rows.append(dict(role=role, delivery="tiled", registers=regs, stack=stack))
    sequences = kernel_sequences(isa)
    for address, gates in ((True, False), (False, True), (True, True)):
        matches = [(name, body) for name, body in funcs
                   if f"gdn_wy_state_abILb{int(address)}ELb{int(gates)}E" in name]
        if len(matches) != 1:
            raise AssertionError(f"missing/ambiguous state variant: address={address} gates={gates}")
        name, body = matches[0]
        stack = int(re.search(r"STACK SIZE:(\d+)", body)[1])
        regs = int(re.search(r"vreg_number:(\d+)", body)[1])
        if stack:
            raise AssertionError(f"state variant spills: {stack}")
        if name not in sequences:
            raise AssertionError(f"missing state variant native body: {name}")
        ops = [line.split()[0] for line in sequences[name]]
        copies = sum(op.startswith("vmem.ld.tsm") for op in ops)
        exponents = ops.count("v.exp2.f32")
        if copies != (18 if address else 3):
            raise AssertionError(f"state address body not emitted: cp.async sites={copies}")
        if exponents != (5 if gates else 17):
            raise AssertionError(f"state row reuse body not emitted: exponent sites={exponents}")
        if ops.count("v.mma.f32.bf16.m16n16k16") != 32 or any("mma.f32.tf32" in op for op in ops):
            raise AssertionError("state variant changed MMA arithmetic body")
        for required in ("vmem.st.b32x4", "tsm.ld.swzl"):
            if not any(op.startswith(required) for op in ops):
                raise AssertionError(f"state variant lost native/vector delivery: {required}")
        if any(op.startswith("v.shuffle") or op in ("vmem.st.b16", "vmem.ld.b16") for op in ops):
            raise AssertionError("state variant regressed to shuffle or scalar global BF16")
        rows.append(dict(role="state", address=address, row_reuse=gates, registers=regs,
                         stack=stack, static_instructions=len(ops), cp_async_sites=copies,
                         exponent_sites=exponents))
    for role, copies, bf16, tf32, exponents in (("prepare", 16, 16, 12, 9), ("output", 14, 40, 0, 18)):
        matches = [(name, body) for name, body in funcs if f"gdn_wy_address_{role}E" in name]
        if len(matches) != 1:
            raise AssertionError(f"missing/ambiguous address {role} image")
        name, body = matches[0]
        stack = int(re.search(r"STACK SIZE:(\d+)", body)[1])
        regs = int(re.search(r"vreg_number:(\d+)", body)[1])
        if stack or name not in sequences:
            raise AssertionError(f"address {role} spills or is missing its native body")
        ops = [line.split()[0] for line in sequences[name]]
        actual = (sum(op.startswith("vmem.ld.tsm") for op in ops),
                  ops.count("v.mma.f32.bf16.m16n16k16"), ops.count("v.mma.f32.tf32.m16n16k8"),
                  ops.count("v.exp2.f32"))
        if actual != (copies, bf16, tf32, exponents):
            raise AssertionError(f"address {role} copy/arithmetic body changed: {actual}")
        if not any(op.startswith("tsm.ld.swzl") for op in ops):
            raise AssertionError(f"address {role} lost its native matrix load")
        if role == "output" and ("vmem.st.b32x4" not in ops or "vmem.st.b16" in ops):
            raise AssertionError("address output lost vector publication")
        rows.append(dict(role=role, delivery="address", registers=regs, stack=stack,
                         static_instructions=len(ops), cp_async_sites=copies, exponent_sites=exponents,
                         bf16_mma_sites=bf16, tf32_mma_sites=tf32))
    address_section = next(s for s in isa.split("Disassembly of section ") if
                           s.startswith(".text.kernel.") and "gdn_wy_address_prepareE" in s.splitlines()[0])
    address_loop = conditioning_region(address_section)
    if len(address_loop["repeated"]) != 1 or address_loop["hoisted"]:
        raise AssertionError("prepare-address before is not the registered rolled exponential control")
    for mode in (1, 2):
        matches = [(name, body) for name, body in funcs if f"gdn_wy_rows_prepareILi{mode}E" in name]
        if len(matches) != 1:
            raise AssertionError("missing/ambiguous prepare-row image")
        name, body = matches[0]
        stack = int(re.search(r"STACK SIZE:(\d+)", body)[1])
        regs = int(re.search(r"vreg_number:(\d+)", body)[1])
        if stack or name not in sequences:
            raise AssertionError("prepare-row body missing or spilling")
        ops = [line.split()[0] for line in sequences[name]]
        actual = (sum(op.startswith("vmem.ld.tsm") for op in ops),
                  ops.count("v.mma.f32.bf16.m16n16k16"), ops.count("v.mma.f32.tf32.m16n16k8"),
                  ops.count("v.exp2.f32"), ops.count("s.blksyn.defer"), ops.count("v.shuffle.idx.b32"))
        expected = (16, 16, 12, 8 + mode, 7 if mode == 1 else 6, int(mode == 2))
        if actual != expected:
            raise AssertionError(f"prepare-row arithmetic/cache/sync body changed: {actual} != {expected}")
        section = next(s for s in isa.split("Disassembly of section ") if
                       s.startswith(".text.kernel.") and name in s.splitlines()[0])
        region = check_row_hoist(section, mode)
        rows.append(dict(role="prepare", delivery="row-shared" if mode == 1 else "row-warp",
                         registers=regs, stack=stack, static_instructions=len(ops),
                         conditioning_loop=f"{region['head']:#x}..{region['tail']:#x}",
                         loop_exponents=0, hoisted_exponent_pcs=[hex(pc) for pc in region['hoisted']],
                         cache_publication_pcs=[hex(pc) for pc in region['publication']],
                         cta_barrier_sites=actual[4], indexed_shuffle_sites=actual[5]))
    for role, copies, mmas, exponents, barriers in (("state", 5, 32, 5, 5), ("output", 6, 40, 18, 6)):
        matches = [(name, body) for name, body in funcs if f"gdn_wy_aiu_{role}E" in name]
        if len(matches) != 1:
            raise AssertionError(f"missing/ambiguous AIU {role} image")
        name, body = matches[0]
        stack = int(re.search(r"STACK SIZE:(\d+)", body)[1])
        regs = int(re.search(r"vreg_number:(\d+)", body)[1])
        if stack or name not in sequences:
            raise AssertionError(f"AIU {role} body absent or spilling")
        ops = [line.split()[0] for line in sequences[name]]
        # l0 is the native SWZL writer. A same-source true/false probe gives
        # l0/l1 respectively; do not count any AIU instruction as a matched pair.
        actual = (ops.count("vmem.aiu.ld.tsm.l0.t0.p0.s0.m0.2d.b16.kp1"),
                  ops.count("v.mma.f32.bf16.m16n16k16"), ops.count("v.exp2.f32"),
                  ops.count("s.blksyn.defer"))
        if actual != (copies, mmas, exponents, barriers):
            raise AssertionError(f"AIU {role} delivery/arithmetic/sync changed: {actual}")
        if any(op.startswith("vmem.ld.tsm") or op.startswith("vmem.aiu.ld.tsm.l1") for op in ops):
            raise AssertionError(f"AIU {role} retained manual copies or mismatched linear writer")
        if any(op.startswith("tsm.ld.ncom") for op in ops):
            raise AssertionError(f"AIU {role} has an unmatched NCOM consumer")
        if sum(op.startswith("tsm.ld.swzl") for op in ops) != (44 if role == "state" else 60):
            raise AssertionError(f"AIU {role} fragment-load coverage changed")
        for suffix in ("trans0", "trans1"):
            if f"tsm.ld.swzl.b32x4.s0.t1.{suffix}" not in ops:
                raise AssertionError(f"AIU {role} lost matching {suffix} SWZL reader")
        if "vmem.st.b32x4" not in ops or "vmem.st.b16" in ops:
            raise AssertionError(f"AIU {role} lost vector publication")
        rows.append(dict(role=role, delivery="aiu-paired", registers=regs, stack=stack,
                         static_instructions=len(ops), aiu_sites=copies, bf16_mma_sites=mmas,
                         v2s_sites=ops.count("v.mov.v2s"), cta_barrier_sites=barriers))
    # A compiled header or unchanged prepare kernel is not evidence that the
    # three new device bodies were emitted. Pin each body's arithmetic and
    # native delivery, not only the aggregate library mnemonic inventory.
    for role, copies, bf16, tf32, exps, barriers, stores in (
            ("prefix", 0, 0, 0, 0, 1, 0), ("solve", 2, 8, 12, 8, 5, 4),
            ("wu", 5, 32, 0, 1, 4, 8)):
        matches = [(name, body) for name, body in funcs if f"gdn_wy_split_{role}E" in name]
        if len(matches) != 1:
            raise AssertionError(f"missing/ambiguous split {role} image")
        name, body = matches[0]
        stack = int(re.search(r"STACK SIZE:(\d+)", body)[1])
        regs = int(re.search(r"vreg_number:(\d+)", body)[1])
        if stack or name not in sequences:
            raise AssertionError(f"split {role} absent or spilling")
        ops = [line.split()[0] for line in sequences[name]]
        actual = (ops.count("vmem.aiu.ld.tsm.l0.t0.p0.s0.m0.2d.b16.kp1"),
                  ops.count("v.mma.f32.bf16.m16n16k16"), ops.count("v.mma.f32.tf32.m16n16k8"),
                  ops.count("v.exp2.f32"), ops.count("s.blksyn.defer"), ops.count("vmem.st.b32x4"))
        if actual != (copies, bf16, tf32, exps, barriers, stores):
            raise AssertionError(f"split {role} native arithmetic/delivery/lifetime differs: {actual}")
        if any(op.startswith(("vmem.ld.tsm", "vmem.aiu.ld.tsm.l1", "tsm.ld.ncom")) for op in ops):
            raise AssertionError(f"split {role} unmatched operand path")
        if "vmem.st.b16" in ops:
            raise AssertionError(f"split {role} scalar BF16 publication")
        if role == "prefix" and (ops.count("v.shuffle.up.b32") != 5 or "vmem.st.b32" not in ops):
            raise AssertionError("split prefix lost original scan order/publication")
        if role != "prefix" and "tsm.ld.swzl.b32x4.s0.t1.trans0" not in ops:
            raise AssertionError(f"split {role} native SWZL consumer missing")
        if role == "wu" and "tsm.ld.swzl.b32x4.s0.t1.trans1" not in ops:
            raise AssertionError("split WU transposed operand consumer missing")
        rows.append(dict(role=role, delivery="split-prepare", registers=regs, stack=stack,
                         static_instructions=len(ops), aiu_sites=copies, bf16_mma_sites=bf16,
                         tf32_mma_sites=tf32, cta_barrier_sites=barriers, vector_store_sites=stores))
    matches = [(name,body) for name,body in funcs if "gdn_wy_state_pipelineE" in name]
    if len(matches) != 1: raise AssertionError("missing/ambiguous state pipeline image")
    name, body = matches[0]
    regs = int(re.search(r"vreg_number:(\d+)",body)[1])
    stack = int(re.search(r"STACK SIZE:(\d+)",body)[1])
    if stack or regs > 234 or name not in sequences:
        raise AssertionError("state pipeline spills, exceeds admitted register budget, or lacks native body")
    ops = [line.split()[0] for line in sequences[name]]
    if (ops.count("vmem.aiu.ld.tsm.l0.t0.p0.s0.m0.2d.b16.kp1"),
        ops.count("v.mma.f32.bf16.m16n16k16"),ops.count("v.exp2.f32"),
        ops.count("s.blksyn.defer"),sum(op.startswith("tsm.ld.swzl") for op in ops)) != (7,32,5,4,44):
        raise AssertionError("pipeline native delivery/arithmetic/lifetime inventory changed")
    if any(op.startswith(("vmem.ld.tsm","vmem.aiu.ld.tsm.l1","tsm.ld.ncom")) for op in ops):
        raise AssertionError("pipeline lost matched native producer/consumer")
    if "vmem.st.b32x4" not in ops or "vmem.st.b16" in ops:
        raise AssertionError("pipeline lost vector publication")
    section = next(s for s in isa.split("Disassembly of section ") if
                   s.startswith(".text.kernel.") and name in s.splitlines()[0])
    schedule = native_schedule(section)
    rows.append(dict(role="state",delivery="state-pipeline",registers=regs,stack=stack,
                     static_instructions=len(ops),aiu_sites=7,bf16_mma_sites=32,
                     v2s_sites=ops.count("v.mov.v2s"),cta_barrier_sites=4,
                     overlap_mma_before_commit_wait={hex(pc):work for pc,work in schedule["overlap"].items()}))
    matches = [(name, body) for name, body in funcs if "gdn_wy_residual_stateE" in name]
    if len(matches) != 1:
        raise AssertionError("residual algorithm body absent/ambiguous")
    name, body = matches[0]
    regs = int(re.search(r"vreg_number:(\d+)", body)[1])
    stack = int(re.search(r"STACK SIZE:(\d+)", body)[1])
    if stack or name not in sequences:
        raise AssertionError("residual state spills or lacks native code")
    ops = [line.split()[0] for line in sequences[name]]
    counts = (ops.count("vmem.aiu.ld.tsm.l0.t0.p0.s0.m0.2d.b16.kp1"),
              ops.count("v.mma.f32.bf16.m16n16k16"), ops.count("v.exp2.f32"),
              ops.count("s.blksyn.defer"), sum(op.startswith("tsm.ld.swzl") for op in ops))
    if counts != (4, 40, 9, 5, 56):
        raise AssertionError(f"residual native copy/math/lifetime changed: {counts}")
    if sum("commit_group(0)" in line for line in sequences[name]) != 1:
        raise AssertionError("residual async input completion missing or duplicated")
    if any(op.startswith(("vmem.ld.tsm", "vmem.aiu.ld.tsm.l1", "tsm.ld.ncom")) for op in ops):
        raise AssertionError("residual lost native AIU/SWZL pair")
    if "vmem.st.b32x4" not in ops or "vmem.st.b16" in ops:
        raise AssertionError("residual lost vector output publication")
    rows.append(dict(role="state", algorithm="residual", registers=regs, stack=stack,
                     static_instructions=len(ops), aiu_sites=4, bf16_mma_sites=40,
                     cta_barrier_sites=5, shared_bytes=45568, rounding="NEW-EXPLICIT"))
    for delivery, marker, expected, shared in (
            ("prefetch", "gdn_wy_residual_prefetch_stateE", (5,40,9,5,56), 45568),
            ("operands", "gdn_wy_residual_operands_stateE", (4,40,9,5,56), 45568),
            ("v16", "gdn_wy_residual_v16_stateE", (4,20,5,5,36), 35328),
            ("blayout", "gdn_wy_residual_blayout_stateE", (4,40,9,5,56), 45568),
            ("warps8", "gdn_wy_residual_warps8_stateE", (4,20,5,5,36), 45568),
            ("warps8-blayout", "gdn_wy_residual_warps8_blayout_stateE", (4,20,5,5,36), 45568),
            ("warps8-operands", "gdn_wy_residual_warps8_operands_stateE", (4,20,5,5,36), 45568),
            ("warps8-hlayout", "gdn_wy_residual_warps8_hlayout_stateE", (4,20,5,5,36), 45568),
            ("warps8-hvlayout", "gdn_wy_residual_warps8_hvlayout_stateE", (4,20,5,5,36), 45568),
            ("warps8-metadata", "gdn_wy_residual_warps8_metadata_stateE", (4,20,5,5,36), 45568)):
        matches=[(name,body) for name,body in funcs if marker in name]
        if len(matches)!=1: raise AssertionError(f"residual {delivery} image absent/ambiguous")
        name,body=matches[0]
        regs=int(re.search(r"vreg_number:(\d+)",body)[1])
        stack=int(re.search(r"STACK SIZE:(\d+)",body)[1])
        if stack or regs>256 or name not in sequences:
            raise AssertionError(f"residual {delivery} spills or lacks native body")
        if delivery=="warps8-metadata" and regs>122:
            raise AssertionError("metadata lookahead exceeded its HV register budget")
        ops=[line.split()[0] for line in sequences[name]]
        counts=(ops.count("vmem.aiu.ld.tsm.l0.t0.p0.s0.m0.2d.b16.kp1"),
                ops.count("v.mma.f32.bf16.m16n16k16"),ops.count("v.exp2.f32"),
                ops.count("s.blksyn.defer"),sum(op.startswith("tsm.ld.swzl") for op in ops))
        if counts!=expected or sum("commit_group(0)" in line for line in sequences[name])!=1:
            raise AssertionError(f"residual {delivery} native inventory differs: {counts}")
        if any(op.startswith(("vmem.ld.tsm","vmem.aiu.ld.tsm.l1","tsm.ld.ncom")) for op in ops):
            raise AssertionError("residual delivery lost matched AIU/SWZL pair")
        if any("ivreg" in line for line in sequences[name]):
            raise AssertionError("residual register-slot selection became indirect")
        if "vmem.st.b32x4" not in ops or "vmem.st.b16" in ops:
            raise AssertionError("residual delivery lost vector publication")
        row=dict(role="state",algorithm="residual",delivery=delivery,registers=regs,stack=stack,
                 static_instructions=len(ops),aiu_sites=counts[0],bf16_mma_sites=counts[1],
                 cta_barrier_sites=5,shared_bytes=shared,rounding="RESIDUAL-UNCHANGED")
        section=next(s for s in isa.split("Disassembly of section ") if
                     s.startswith(".text.kernel.") and name in s.splitlines()[0])
        if delivery=="prefetch": row["overlap"]=residual_prefetch_schedule(section)["overlap"]
        rows.append(row)
    for delivery in ("warps8-hlayout", "warps8-hvlayout"):
        matches=[(name,body) for name,body in funcs if ("gdn_wy_residual_"+delivery.replace("-","_")+"_outputE") in name]
        if len(matches)!=1: raise AssertionError("paired private output image absent/ambiguous")
        name,body=matches[0]
        regs=int(re.search(r"vreg_number:(\d+)",body)[1])
        stack=int(re.search(r"STACK SIZE:(\d+)",body)[1])
        if stack or regs>256 or name not in sequences:
            raise AssertionError("paired private output spills or lacks native body")
        ops=[line.split()[0] for line in sequences[name]]
        counts=(ops.count("vmem.aiu.ld.tsm.l0.t0.p0.s0.m0.2d.b16.kp1"),
                ops.count("v.mma.f32.bf16.m16n16k16"),ops.count("v.exp2.f32"),
                ops.count("s.blksyn.defer"),sum(op.startswith("tsm.ld.swzl") for op in ops))
        if counts!=(7,40,18,6,60) or "vmem.st.b32x4" not in ops or "vmem.st.b16" in ops:
            raise AssertionError(f"paired private output native work/publication differs: {counts}")
        rows.append(dict(role="output",algorithm="residual",delivery=delivery,registers=regs,
                         stack=stack,static_instructions=len(ops),aiu_sites=7,bf16_mma_sites=40,
                         cta_barrier_sites=6,shared_bytes=49408,rounding="RESIDUAL-UNCHANGED"))
    matches=[(name,body) for name,body in funcs if "gdn_wy_split_solve_staticE" in name]
    if len(matches)!=1: raise AssertionError("static solve resource image missing/ambiguous")
    name,body=matches[0]
    regs=int(re.search(r"vreg_number:(\d+)",body)[1])
    stack=int(re.search(r"STACK SIZE:(\d+)",body)[1])
    if stack or regs>84 or name not in sequences:
        raise AssertionError("static solve spills, exceeds84regs control budget, or lacks exact native body")
    rows.append(dict(role="solve",algorithm="residual",delivery="solve-static",registers=regs,
                     stack=stack,static_instructions=len(sequences[name]),shared_bytes=49664,
                     diagonal_ordered_fmas=120,tf32_mma_sites=12,indirect_register_reads=0))
    matches=[(name,body) for name,body in funcs if "gdn_wy_residual_gate_cache_stateE" in name]
    if len(matches)!=1: raise AssertionError("gate-cache resource image missing/ambiguous")
    name,body=matches[0]
    regs=int(re.search(r"vreg_number:(\d+)",body)[1])
    stack=int(re.search(r"STACK SIZE:(\d+)",body)[1])
    if stack or regs>256 or name not in sequences:
        raise AssertionError("gate-cache spills or lacks exact native body")
    rows.append(dict(role="state",algorithm="residual",delivery="gate-cache",
                     registers=regs,stack=stack,shared_bytes=46080,
                     static_instructions=len(sequences[name]),device="NOT_RUN"))
    for name in ("gdn_wy_forward_residual_gate_cache", "gdn_wy_forward", "gdn_wy_forward_delivery", "gdn_wy_forward_residual",
                 "gdn_wy_forward_residual_prefetch", "gdn_wy_forward_residual_operands", "gdn_wy_forward_residual_v16",
                 "gdn_wy_forward_residual_blayout", "gdn_wy_forward_residual_warps8",
                 "gdn_wy_forward_residual_warps8_blayout", "gdn_wy_forward_residual_warps8_operands",
                 "gdn_wy_forward_residual_warps8_hlayout", "gdn_wy_forward_residual_warps8_hvlayout",
                 "gdn_wy_forward_residual_warps8_metadata", "gdn_wy_forward_residual_solve_static"):
        if not re.search(rf"\b{name}$", symbols, re.M):
            raise AssertionError(f"WY launcher missing from linked library: {name}")
    for name in ("configure_tiled", "launch_tiled_prepare", "launch_tiled_state", "launch_tiled_output",
                 "configure_state_ab", "launch_state_ab", "configure_stage_address",
                 "launch_address_prepare", "launch_address_output", "configure_prepare_rows", "launch_prepare_rows",
                 "forward_aiu", "configure_split_prepare", "launch_split_prepare",
                 "configure_state_pipeline", "launch_state_pipeline", "launch_split_inverse", "launch_split_prefix",
                 "configure_aiu_output", "launch_aiu_output"):
        if not re.search(rf"\b_ZN7gdn_qsa2wy\d+{name}E\S*$", symbols, re.M):
            raise AssertionError(f"tiled cross-TU launcher missing from linked library: {name}")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("build", type=Path)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--baseline-isa", type=Path,
                   help="optional same-SDK parent build: require exact old instruction/operand sequences")
    args = p.parse_args()
    isa = (args.build / "gdn_wy_ppu.isa").read_text()
    resources = (args.build / "gdn_wy_ppu.resources").read_text()
    symbols = subprocess.check_output(["nm", "-D", "--defined-only", str(args.build / "libgdn_wy_ppu.so")], text=True)
    for row in audit(isa, resources, symbols):
        print(f"[WY binary] {row} compile-only/PASS")
    if args.self_test:
        for label, texts in (
            ("spill", (isa, resources.replace("STACK SIZE:0", "STACK SIZE:32", 1), symbols)),
            ("missing-tf32", (isa.replace("v.mma.f32.tf32.m16n16k8", "REMOVED"), resources, symbols)),
            ("missing-link", (isa, resources, symbols.replace("gdn_wy_forward", "REMOVED"))),
            ("scalar-store-regression", (isa.replace("vmem.st.b32x4", "vmem.st.b16"), resources, symbols)),
            ("missing-tiled-image", (isa.replace("gdn_wy_tiled_state", "MISSING_tiled_state"), resources, symbols)),
            ("missing-tiled-host-link", (isa, resources, symbols.replace("launch_tiled_state", "MISSING_tiled_state"))),
            ("tiled-shuffle-regression", (isa.replace("tsm.ld.swzl.b32x4.s0.t1.trans1", "v.shuffle.idx.b32"), resources, symbols)),
            ("missing-state-variant", (isa.replace("gdn_wy_state_abILb1ELb1E", "MISSING_state_variant"), resources, symbols)),
            ("missing-state-link", (isa, resources, symbols.replace("launch_state_ab", "MISSING_state_link"))),
            ("state-reuse-not-emitted", (isa.replace("v.exp2.f32", "MISSING_EXP"), resources, symbols)),
            ("state-copy-not-emitted", (isa.replace("vmem.ld.tsm.zfill.b32x4", "vmem.ld.WRONG.b32x4"), resources, symbols)),
            ("missing-address-prepare", (isa.replace("gdn_wy_address_prepare", "MISSING_address_prepare"), resources, symbols)),
            ("missing-address-output", (isa.replace("gdn_wy_address_output", "MISSING_address_output"), resources, symbols)),
            ("missing-stage-host-link", (isa, resources, symbols.replace("launch_address_output", "MISSING_address_output"))),
            ("address-prepare-lost-tf32", (plant_in_kernel(isa, "gdn_wy_address_prepare",
                "v.mma.f32.tf32.m16n16k8", "MISSING_TF32"), resources, symbols)),
            ("address-output-lost-copy", (plant_in_kernel(isa, "gdn_wy_address_output",
                "vmem.ld.tsm", "MISSING_COPY"), resources, symbols)),
            ("missing-row-image", (isa.replace("gdn_wy_rows_prepareILi1E", "MISSING_ROWS"), resources, symbols)),
            ("missing-row-host-link", (isa, resources, symbols.replace("launch_prepare_rows", "MISSING_ROWS"))),
            ("row-exp-back-in-loop-same-count", (plant_repeated_row_exp(isa), resources, symbols)),
            ("row-shared-late-publication-same-count", (plant_late_row_barrier(isa), resources, symbols)),
            ("missing-aiu-image", (isa.replace("gdn_wy_aiu_state", "MISSING_aiu_state"), resources, symbols)),
            ("missing-aiu-link", (isa, resources, symbols.replace("forward_aiu", "MISSING_aiu_link"))),
            ("aiu-linear-writer", (plant_in_kernel(isa, "gdn_wy_aiu_state",
                "vmem.aiu.ld.tsm.l0", "vmem.aiu.ld.tsm.l1"), resources, symbols)),
            ("aiu-wrong-reader", (plant_in_kernel(isa, "gdn_wy_aiu_output",
                "tsm.ld.swzl.b32x4.s0.t1.trans1", "tsm.ld.ncom.b32x4"), resources, symbols)),
            ("split-missing-link", (isa, resources, symbols.replace("launch_split_prepare", "MISSING_split"))),
            ("split-residual-missing", (plant_in_kernel(isa, "gdn_wy_split_solveE",
                "v.mma.f32.tf32.m16n16k8", "MISSING_RESIDUAL"), resources, symbols)),
            ("split-wu-scalar-store", (plant_in_kernel(isa, "gdn_wy_split_wu",
                "vmem.st.b32x4", "vmem.st.b16"), resources, symbols)),
            ("split-wu-wrong-reader", (plant_in_kernel(isa, "gdn_wy_split_wu",
                "tsm.ld.swzl.b32x4.s0.t1.trans1", "tsm.ld.ncom.b32x4"), resources, symbols)),
            ("split-wu-missing-retirement", (plant_in_kernel(isa, "gdn_wy_split_wu",
                "s.blksyn.defer", "MISSING_BARRIER"), resources, symbols)),
            ("split-prefix-lost-shuffle", (plant_in_kernel(isa, "gdn_wy_split_prefix",
                "v.shuffle.up.b32", "MISSING_SCAN"), resources, symbols)),
            ("pipeline-missing-image", (isa.replace("gdn_wy_state_pipelineE", "MISSING_pipelineE"),resources,symbols)),
            ("pipeline-missing-link", (isa,resources,symbols.replace("launch_state_pipeline","MISSING_pipeline"))),
            ("pipeline-wrong-writer", (plant_in_kernel(isa,"gdn_wy_state_pipeline",
                "vmem.aiu.ld.tsm.l0","vmem.aiu.ld.tsm.l1"),resources,symbols)),
            ("pipeline-lost-wait", (plant_in_kernel(isa,"gdn_wy_state_pipeline",
                "commit_group(0)","MISSING_WAIT"),resources,symbols)),
            ("pipeline-lost-retirement", (plant_in_kernel(isa,"gdn_wy_state_pipeline",
                "s.blksyn.defer","MISSING_BARRIER"),resources,symbols)),
            ("pipeline-serialized-same-opcodes", (plant_serialized_prefetch(isa),resources,symbols)),
            ("residual-removed-mma", (plant_in_kernel(isa,"gdn_wy_residual_state",
                "v.mma.f32.bf16.m16n16k16","MISSING_MMA"),resources,symbols)),
            ("residual-wrong-reader", (plant_in_kernel(isa,"gdn_wy_residual_state",
                "tsm.ld.swzl.b32x4.s0.t1.trans1","tsm.ld.ncom.b32x4"),resources,symbols)),
            ("residual-missing-barrier", (plant_in_kernel(isa,"gdn_wy_residual_state",
                "s.blksyn.defer","MISSING_BARRIER"),resources,symbols)),
            ("residual-missing-wait", (plant_in_kernel(isa,"gdn_wy_residual_state",
                "commit_group(0)","MISSING_WAIT"),resources,symbols)),
            ("prefetch-serialized-same-opcodes", (serialized_plant(isa),resources,symbols)),
            ("operands-missing-mma", (plant_in_kernel(isa,"gdn_wy_residual_operands_state",
                "v.mma.f32.bf16.m16n16k16","MISSING_MMA"),resources,symbols)),
            ("warps8-missing-mma", (plant_in_kernel(isa,"gdn_wy_residual_warps8_state",
                "v.mma.f32.bf16.m16n16k16","MISSING_MMA"),resources,symbols)),
            ("warps8-missing-image", (isa.replace("gdn_wy_residual_warps8_state","MISSING_WARPS8"),resources,symbols)),
            ("warps8-missing-link", (isa,resources,symbols.replace("gdn_wy_forward_residual_warps8","MISSING_WARPS8"))),
            ("warps8-blayout-missing-image", (isa.replace("gdn_wy_residual_warps8_blayout_state","MISSING_WARPS8_B"),resources,symbols)),
            ("warps8-blayout-missing-link", (isa,resources,symbols.replace("gdn_wy_forward_residual_warps8_blayout","MISSING_WARPS8_B"))),
            ("warps8-operands-missing-image", (isa.replace("gdn_wy_residual_warps8_operands_state","MISSING_WARPS8_OPERANDS"),resources,symbols)),
            ("warps8-operands-missing-link", (isa,resources,symbols.replace("gdn_wy_forward_residual_warps8_operands","MISSING_WARPS8_OPERANDS"))),
            ("hlayout-missing-state", (isa.replace("gdn_wy_residual_warps8_hlayout_state","MISSING_H_STATE"),resources,symbols)),
            ("hlayout-missing-output", (isa.replace("gdn_wy_residual_warps8_hlayout_output","MISSING_H_OUTPUT"),resources,symbols)),
            ("hlayout-missing-link", (isa,resources,symbols.replace("gdn_wy_forward_residual_warps8_hlayout","MISSING_H_LINK"))),
            ("hvlayout-missing-state", (isa.replace("gdn_wy_residual_warps8_hvlayout_state","MISSING_HV_STATE"),resources,symbols)),
            ("hvlayout-missing-output", (isa.replace("gdn_wy_residual_warps8_hvlayout_output","MISSING_HV_OUTPUT"),resources,symbols)),
            ("hvlayout-missing-link", (isa,resources,symbols.replace("gdn_wy_forward_residual_warps8_hvlayout","MISSING_HV_LINK"))),
            ("metadata-missing-state", (isa.replace("gdn_wy_residual_warps8_metadata_state","MISSING_METADATA_STATE"),resources,symbols)),
            ("metadata-missing-link", (isa,resources,symbols.replace("gdn_wy_forward_residual_warps8_metadata","MISSING_METADATA_LINK"))),
            ("solve-static-missing-image", (isa.replace("gdn_wy_split_solve_staticE","MISSING_STATIC_SOLVE"),resources,symbols)),
            ("solve-static-missing-link", (isa,resources,symbols.replace("gdn_wy_forward_residual_solve_static","MISSING_STATIC_SOLVE_LINK"))),
            ("v16-wrong-reader", (plant_in_kernel(isa,"gdn_wy_residual_v16_state",
                "tsm.ld.swzl","tsm.ld.ncom"),resources,symbols)),
        ):
            try:
                audit(*texts)
            except AssertionError:
                print(f"[WY binary negative] {label} EXPECTED-RED/PASS")
            else:
                raise AssertionError(f"escaped negative: {label}")
    if args.baseline_isa:
        before = args.baseline_isa.read_text()
        compare_controls(before, isa)
        if args.self_test:
            try:
                compare_controls(before, isa.replace("v.mma.f32.bf16", "CHANGED_CONTROL", 1))
            except AssertionError:
                print("[WY binary negative] changed-control EXPECTED-RED/PASS")
            else:
                raise AssertionError("control-comparison negative escaped")
        count = len(kernel_sequences(before))
        print(f"[WY binary controls] {count}/{count} native instruction+operand sequences IDENTICAL")
    print("[WY binary] PASS device_execution=NOT_RUN")


if __name__ == "__main__":
    main()
