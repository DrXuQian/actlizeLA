"""Resolve x86 host PLT calls from ELF relocations, not objdump label guesses."""
import re
import subprocess


def resolve_plt(host, relocations):
    slots = {}
    for line in relocations.splitlines():
        match = re.match(r"^([0-9a-f]+)\s+\S+\s+R_X86_64_(?:JUMP_SLOT|GLOB_DAT)\s+\S+\s+(.+) \+ 0$", line)
        if match:
            slot = int(match[1], 16)
            if slot in slots:
                raise AssertionError("duplicate host ELF relocation slot")
            slots[slot] = match[2]
    entries = {}
    for section in host.split("Disassembly of section ")[1:]:
        if section.splitlines()[0] not in (".plt:", ".plt.sec:", ".plt.got:"):
            continue
        entry = None
        for line in section.splitlines():
            match = re.match(r"\s*([0-9a-f]+):\s+endbr64\s*$", line)
            if match:
                entry = int(match[1], 16)
            jump = re.match(r"\s*([0-9a-f]+):\s+(?:bnd\s+)?jmp\s+\*.*#\s+([0-9a-f]+)\s+<", line)
            if not jump:
                continue
            pc, slot = int(jump[1], 16), int(jump[2], 16)
            if entry is not None and pc != entry + 4:
                raise AssertionError("unrecognized IBT PLT entry layout")
            if slot in slots:
                entries[pc if entry is None else entry] = slots[slot]
            elif entry is not None:
                raise AssertionError("IBT PLT entry lacks an exact ELF relocation")
            entry = None
    def resolve(match):
        pc = int(match[2], 16)
        if pc not in entries:
            return match[0]
        return match[1] + match[2] + " <" + entries[pc] + "@plt>"
    return re.sub(r"(\b(?:callq?|jmpq?)\s+)([0-9a-f]+)\s+<[^\n]+>", resolve, host)


def read_host_disassembly(library):
    host = subprocess.check_output(["objdump", "-d", "-C", "--no-show-raw-insn", str(library)], text=True)
    relocs = subprocess.check_output(["readelf", "-rW", str(library)], text=True)
    relocs = subprocess.check_output(["c++filt"], input=relocs, text=True)
    return resolve_plt(host, relocs)
