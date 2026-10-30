#!/usr/bin/env python3
"""Inspect native generation, never promote a compile into device admission."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


def inspect_dump(isa, resources):
    symbols = re.findall(r"^Func \d+: (\S+)", resources, re.M)
    expected = {"prefix_kernel": 1, "solve_kernel": 1, "state_kernel": 2, "output_kernel": 1}
    for name, count in expected.items():
        if sum(name in s for s in symbols) != count:
            raise ValueError(f"native body denominator: expected {count} {name}, got {symbols}")
    if len(symbols) != 5 or "PPU 1.5" not in resources or re.search(r"PPU 1\.[07]", resources):
        raise ValueError("expected exactly five native PPU1.5 bodies (full/tail state separate)")
    for opcode in ("v.mm.f32.bf16.m16n16k16", "v.mm.f32.tf32.m16n16k8", "vmem.aiu.ld.tsm", "tsm.ld.swzl"):
        if opcode not in isa:
            raise ValueError(f"missing native opcode: {opcode}")
    parts = re.split(r"Disassembly of section \.text\.kernel\.([^:\n]+):", isa)
    bodies = dict(zip(parts[1::2], parts[2::2]))
    for symbol in symbols:
        body = bodies.get(symbol, "")
        if not body.strip():
            raise ValueError(f"missing exact native body: {symbol}")
        required = [] if "prefix_kernel" in symbol else [
            "v.mm.f32.bf16.m16n16k16", "vmem.aiu.ld.tsm", "tsm.ld.swzl"]
        if "solve_kernel" in symbol:
            required.append("v.mm.f32.tf32.m16n16k8")
        for opcode in required:
            if opcode not in body:
                raise ValueError(f"missing native opcode {opcode} in exact body {symbol}")
    stacks = re.findall(r"STACK SIZE:(\d+)", resources)
    if len(stacks) != 5 or any(int(n) for n in stacks):
        raise ValueError(f"native stack/spill guard: {stacks}")
    if re.search(r"\bs\.trap\b|\bvmem\.(?:ld|st)\.local\b", isa):
        raise ValueError("native trap/local-memory fallback")
    return {"status": "PASS", "scope": "native-code-only", "kernels": symbols,
            "registers": [int(n) for n in re.findall(r"vreg_number:(\d+)",resources)],
            "stack_bytes": [int(n) for n in stacks], "device": "NOT_RUN"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sdk",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("binaries",type=Path,nargs="+")
    args = p.parse_args()
    tool = args.sdk / "bin/hgobjdump"
    if not tool.is_file():
        print(f"[ppu15 native] SKIP: hgobjdump missing: {tool}")
        return 77
    args.out.mkdir(parents=True,exist_ok=True)
    isa, resources, identities = [], [], []
    for index, path in enumerate(args.binaries):
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"missing/empty native image: {path}")
        identities.append({"path":str(path.resolve()),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
        for kind, flags, collection in (("isa",["--arch=ppu1.5","--dump-isa"],isa),
                                        ("resources",["--dump-resource-usage=all"],resources)):
            result = subprocess.run([str(tool),*flags,str(path)],text=True,capture_output=True)
            (args.out/f"{index}-{path.name}.{kind}").write_text(result.stdout+result.stderr)
            if result.returncode:
                raise ValueError(f"hgobjdump failed for {path}: {result.stderr}")
            collection.append(result.stdout)
    verdict = inspect_dump("\n".join(isa),"\n".join(resources))
    verdict["binaries"] = identities
    (args.out/"native.json").write_text(json.dumps(verdict,indent=2)+"\n")
    print(f"[ppu15 native] PASS bodies=5 regs={verdict['registers']} stack=0 device=NOT_RUN")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError,OSError) as error:
        print(f"[ppu15 native] FAIL: {error}",file=sys.stderr)
        sys.exit(1)
