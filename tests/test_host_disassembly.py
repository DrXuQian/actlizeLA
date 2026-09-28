"""IBT symbol repair uses actual relocations and still rejects absent calls."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dev/ppu"))
from host_disassembly import resolve_plt
from check_gate_cache_solve import functions, reachable_calls

HOST = """Disassembly of section .plt.sec:
100: endbr64
104: jmp *0x1f6(%rip) # 300 <wrong_guessed_name>
Disassembly of section .text:
0000000000000200 <entry>:
 200: call 100 <irrelevant_data_label+0x30>
"""
RELOC = "0000000000000300 00001 R_X86_64_JUMP_SLOT 0000 actual_helper() + 0\n"


class HostDisassembly(unittest.TestCase):
    def test_ibt_target_comes_from_exact_got_relocation(self):
        resolved = resolve_plt(HOST, RELOC)
        self.assertEqual(reachable_calls(functions(resolved), "entry"), {"actual_helper()"})
        moved = resolve_plt(HOST, RELOC.replace("actual_helper()", "wrong_helper()"))
        self.assertNotIn("actual_helper()", reachable_calls(functions(moved), "entry"))

    def test_missing_relocation_and_changed_entry_fail(self):
        with self.assertRaises(AssertionError):
            resolve_plt(HOST, "")
        with self.assertRaises(AssertionError):
            resolve_plt(HOST.replace("104: jmp", "108: jmp"), RELOC)

    def test_legacy_plt_and_no_plt_keep_actual_direct_calls(self):
        legacy = HOST.replace(".plt.sec:", ".plt:").replace("100: endbr64\n104:", "100:")
        self.assertEqual(reachable_calls(functions(resolve_plt(legacy, RELOC)), "entry"), {"actual_helper()"})
        direct = "Disassembly of section .text:\n00000200 <entry>:\n 200: call 300 <direct()>\n"
        self.assertEqual(resolve_plt(direct, RELOC), direct)


if __name__ == "__main__":
    unittest.main()
