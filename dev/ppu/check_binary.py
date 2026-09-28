#!/usr/bin/env python3
"""Compile evidence, not dynamic performance or a device correctness verdict."""
import argparse
from pathlib import Path
import re
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument("build", type=Path)
    args = p.parse_args()
    isa = (args.build / "gdn_qsa_ppu.isa").read_text()
    resources = (args.build / "gdn_qsa_ppu.resources").read_text()
    required_ops = ("v.mma.f32.bf16.m16n16k16", "v.mma.f16.f16.m16n16k16",
                    "tsm.ld.swzl", "vmem.ld.tsm", "vmem.acp.commit.grp")
    for op in required_ops:
        count = isa.count(op)
        if count == 0:
            raise AssertionError(f"native primitive missing: {op}")
        print(f"[PPU binary] opcode={op} static_sites={count} (not dynamic count)")
    seen = set()
    for m in re.finditer(r"Func \d+ (\S+) RESOURCE INFO:\n(.*?)(?=Func \d+ \S+ RESOURCE INFO:|\Z)",
                         resources, flags=re.S):
        name, body = m.groups()
        stack = int(re.search(r"STACK SIZE:(\d+)", body)[1])
        regs = int(re.search(r"vreg_number:(\d+)", body)[1])
        seen.add(name)
        # C32 was already experimental upstream and never selected by auto.
        # Retain it without passing its measured stack off as zero-spill C16.
        experimental = "gdn_recurrence_kernel32" in name or "ILi32ELi128" in name
        if stack and not experimental:
            raise AssertionError(f"production C16 kernel spills: {name}: {stack} B")
        scope = "C32-EXPERIMENTAL/NOT-PERFORMANCE-ADMITTED" if experimental else "C16-PRODUCTION"
        print(f"[PPU binary resource] symbol={name} regs={regs} stack={stack} scope={scope}")
    expected = ("gdn_recurrence_fused_kernel", "gdn_scan_stage1_reset_fused_kernel",
                "gdn_recurrence_kernel", "gdn_scan_stage1_kernel",
                "gdn_scan_stage2_kernel", "gdn_scan_stage2_blelloch_up_kernel",
                "gdn_scan_stage2_blelloch_down_kernel", "gdn_reset_check_kernel")
    if any(not any(key in name for name in seen) for key in expected):
        raise AssertionError("not all original production paths reached device codegen")
    symbols = subprocess.check_output(
        ["nm", "-D", "--defined-only", str(args.build / "libgdn_qsa_ppu.so")], text=True)
    for name in ("gdn_chunk_forward", "gdn_chunk_replay", "gdn_chunk_replay_fused",
                 "gdn_chunk_prepare_only", "gdn_scan_stage1", "gdn_scan_stage1_reset_fused",
                 "gdn_scan_stage2", "gdn_scan_stage2_blelloch", "gdn_reset_check"):
        if not re.search(rf"\b{re.escape(name)}$", symbols, re.M):
            raise AssertionError(f"missing linked launcher: {name}")
    bindings = list(args.build.glob("_gdn_chunk_ppu*.so"))
    if len(bindings) != 1:
        raise AssertionError(f"expected one linked PyTorch binding, found {bindings}")
    print(f"[PPU binary audit] PASS kernels={len(seen)} original_host_dispatch=LINKED "
          "production_C16_stack=0 device_execution=NOT_RUN")


if __name__ == "__main__":
    main()
