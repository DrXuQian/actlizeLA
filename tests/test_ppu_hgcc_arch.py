"""Run the real CMake selector against controlled HGGC dialects (no device)."""
import os
from pathlib import Path
import subprocess
import sys
import unittest
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
COMPILER = '''#!{python}
import os
from pathlib import Path
import sys
arch = next(x.split("=", 1)[1] for x in sys.argv if x.startswith("-arch="))
with Path(os.environ["HGCC_TEST_ATTEMPTS"]).open("a") as stream:
    stream.write(arch + "\\n")
mode = os.environ["HGCC_TEST_MODE"]
if mode == "broken":
    print("fatal error: hggc_fp16.h: No such file", file=sys.stderr)
    raise SystemExit(1)
if mode == "neither" or (mode == "legacy" and arch != "ppu001"):
    print("hgcc error: invalid value '" + arch + "' for option 'gpu-architecture=', valid values are: ppu001,ppu0015,all", file=sys.stderr)
    raise SystemExit(3)
if mode != "no-output":
    Path(sys.argv[sys.argv.index("-o") + 1]).write_bytes(b"" if mode == "empty" else b"TEST-ONLY-NOT-PPU-CODE")
'''


class HgccArchContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path("/workspace") / f"gdn-hgcc-arch-contract-{uuid4().hex}"
        cls.root.mkdir()

    def setUp(self):
        self.work = self.root / self._testMethodName
        self.work.mkdir()
        self.compiler = self.work / "hgcc-fixture"
        self.compiler.write_text(COMPILER.format(python=sys.executable))
        self.compiler.chmod(0o755)

    def configure(self, mode, arch="ppu_10"):
        script = self.work / "check.cmake"
        script.write_text(
            f'cmake_minimum_required(VERSION 3.19)\n'
            f'include("{ROOT / "cmake/GdnHgccArch.cmake"}")\n'
            f'gdn_hgcc_arch_flags(flags contract "{self.compiler}" '
            f'"-arch={arch}" -x hg -Xllvm -keep-original-option -DGDN_QSA_PPU=1)\n'
            'file(WRITE "selected-flags.txt" "${flags}")\n')
        attempts = self.work / f"attempts-{mode}-{arch}.txt"
        result = subprocess.run(["cmake", "-P", str(script)], cwd=self.work,
            env=os.environ | dict(HGCC_TEST_MODE=mode, HGCC_TEST_ATTEMPTS=str(attempts)),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60)
        return result, attempts.read_text().splitlines() if attempts.exists() else []

    def check_flags(self, selected):
        self.assertEqual((self.work / "selected-flags.txt").read_text().split(";"),
                         ["-x", "hg", "-Xllvm", "-keep-original-option", "-DGDN_QSA_PPU=1", f"-arch={selected}"])
        contract = (self.work / "gdn_hgcc_arch.txt").read_text()
        self.assertIn("logical=ppu0010\n", contract)
        self.assertIn(f"selected=-arch={selected}\n", contract)
        self.assertIn("compiler_sha256=", contract)

    def test_modern_preserves_old_spelling_and_all_other_flags(self):
        result, attempts = self.configure("modern")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(attempts, ["ppu_10"])
        self.check_flags("ppu_10")

    def test_box_error_falls_back_to_same_target_legacy_spelling(self):
        result, attempts = self.configure("legacy")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(attempts, ["ppu_10", "ppu001"])
        self.check_flags("ppu001")

    def test_unrelated_sdk_failure_never_changes_architecture(self):
        result, attempts = self.configure("broken")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("hggc_fp16.h", result.stdout)
        self.assertEqual(attempts, ["ppu_10"])

    def test_no_supported_spelling_is_red_not_all_or_ppu15(self):
        result, attempts = self.configure("neither")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(attempts, ["ppu_10", "ppu001"])
        self.assertFalse((self.work / "gdn_hgcc_arch.txt").exists())

    def test_success_without_new_nonempty_object_is_red(self):
        # A preceding success must not lend its old .o to either negative.
        result, _ = self.configure("modern")
        self.assertEqual(result.returncode, 0, result.stdout)
        for mode in ("no-output", "empty"):
            with self.subTest(mode=mode):
                result, attempts = self.configure(mode)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(attempts, ["ppu_10"])

    def test_reconfigure_detects_changed_dialect_and_updates_contract(self):
        result, _ = self.configure("modern")
        self.assertEqual(result.returncode, 0, result.stdout)
        contract = self.work / "gdn_hgcc_arch.txt"
        old_stat = contract.stat().st_mtime_ns
        result, _ = self.configure("modern")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(contract.stat().st_mtime_ns, old_stat)
        result, _ = self.configure("legacy")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.check_flags("ppu001")
        self.assertNotEqual(contract.stat().st_mtime_ns, old_stat)

    def test_wrong_logical_architecture_is_red_before_compile(self):
        result, attempts = self.configure("modern", "ppu_15")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(attempts, [])


if __name__ == "__main__":
    unittest.main()
