"""Frontend installation and explicit SM90 loading; no GPU required."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

from actlize_la import load_sm90
from actlize_la.gdn_sm90_interface import MATH_CONTRACT

ROOT = Path(__file__).resolve().parents[1]


class Installation(unittest.TestCase):
    def test_sm80_script_cannot_report_success_for_frontend_only(self):
        result = subprocess.run(["bash", str(ROOT / "scripts/build.sh")], cwd=ROOT,
                                env=os.environ | {"GDN_QSA_TARGET": "python"},
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertNotIn("build OK", result.stdout)

    def test_frontend_install_does_not_invoke_a_compiler(self):
        for env in ({}, {"GDN_QSA_TARGET": "python"}):
            with patch.dict(os.environ, env, clear=True), patch("setuptools.setup") as setup, patch(
                "runpy.run_path", wraps=runpy.run_path
            ) as runner:
                runner(str(ROOT / "setup.py"))
                setup.assert_called_once_with()
                self.assertEqual(runner.call_count, 1)

    def test_public_import_has_no_torch_or_old_namespace(self):
        command = ("import sys; from actlize_la import load_sm90,backend_inventory; "
                   "backend_inventory(); assert 'torch' not in sys.modules; "
                   "assert 'gdn_qsa_sm80' not in sys.modules")
        result = subprocess.run([sys.executable, "-c", command], cwd=ROOT,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((ROOT / "gdn_qsa_sm80").exists())

    def bundle(self, configuration="value64"):
        directory = Path("/workspace") / f"actlizeLA-install-test-{uuid4().hex}"
        directory.mkdir()
        binary = directory / "_gdn_fused_sm90.test.so"
        binary.write_bytes(b"host-only fake binary; loader is mocked")
        receipt = dict(complete=True, target="cuda_sm90", mode="native",
                       configuration=configuration,
                       extension="/old-machine/" + binary.name,
                       extension_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
        (directory / "build.json").write_text(json.dumps(receipt))
        module = SimpleNamespace(target="cuda_sm90", configuration=configuration,
                                 math_contract=MATH_CONTRACT,
                                 forward=Mock(return_value=("O", "H")))
        return directory, binary, receipt, module

    def test_all_named_builds_load_once_without_environment_dispatch(self):
        for configuration in ("control", "value64", "value64-local-inverse", "value128-paired"):
            directory, binary, _, module = self.bundle(configuration)
            with patch("actlize_la.sm90._load", return_value=module) as loader:
                forward = load_sm90(directory)
                self.assertEqual(forward.configuration, configuration)
                self.assertEqual(forward(1, 2, 3, 4, 5, initial_state="initial"), ("O", "H"))
                self.assertEqual(forward(1, 2, 3, 4, 5, output_final_state=False), ("O", None))
                loader.assert_called_once_with("_gdn_fused_sm90", str(binary),
                                               hashlib.sha256(binary.read_bytes()).hexdigest())
                self.assertEqual(module.forward.call_args_list[0].args, (1, 2, 3, 4, 5, "initial", True))

    def test_changed_missing_or_symlinked_binary_is_rejected_before_import(self):
        for plant in ("changed", "missing", "symlink"):
            directory, binary, _, module = self.bundle()
            if plant == "changed":
                binary.write_bytes(b"wrong binary")
            else:
                parked = directory / "parked.so"
                binary.rename(parked)
                if plant == "symlink":
                    binary.symlink_to(parked)
            with patch("actlize_la.sm90._load", return_value=module) as loader:
                with self.assertRaises(ValueError):
                    load_sm90(directory)
                loader.assert_not_called()

    def test_receipt_and_module_mismatches_are_red(self):
        for key, bad in (("complete", False), ("target", "ppu17"), ("mode", "source-check"),
                         ("configuration", "typo"), ("extension", ""), ("extension_sha256", "")):
            directory, _, receipt, module = self.bundle()
            receipt[key] = bad
            (directory / "build.json").write_text(json.dumps(receipt))
            with patch("actlize_la.sm90._load", return_value=module) as loader:
                with self.assertRaises(ValueError):
                    load_sm90(directory)
                loader.assert_not_called()
        for key, bad in (("target", "ppu17"), ("configuration", "control"), ("math_contract", "wrong")):
            directory, _, _, module = self.bundle()
            setattr(module, key, bad)
            with patch("actlize_la.sm90._load", return_value=module), self.assertRaises(ValueError):
                load_sm90(directory)
            module.forward.assert_not_called()


if __name__ == "__main__":
    unittest.main()
