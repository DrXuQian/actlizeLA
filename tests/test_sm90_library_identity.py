"""Exercise the real identity helper without loading Torch/CUDA in host CI."""
import ast
import hashlib
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
tree = ast.parse((ROOT / "benchmarks/profile_sm90_libraries.py").read_text())
helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "reference_binaries")
namespace = {"hashlib": hashlib}
exec(compile(ast.Module(body=[helper], type_ignores=[]), "identity-helper", "exec"), namespace)


class Identity(unittest.TestCase):
    def exercise(self, cuda_elf):
        module = ModuleType("flashinfer.gdn_kernels.delta_rule_dsl.custom_compile_cache")
        runtime = ModuleType("cutlass.base_dsl.jit_executor")
        host = b"\x7fELF" + bytes(14) + (62).to_bytes(2, "little") + bytes(44)
        cuda = b"\x7fELF" + bytes(14) + (190).to_bytes(2, "little") + bytes(44)
        compiled = MagicMock(kernel_info={}, function_name="test_kernel")

        def dump(prefix):
            self.assertEqual(prefix, "gdn_reference")
            return host + (cuda if cuda_elf else b"")

        compiled.dump_to_object.side_effect = dump
        module._in_mem_compile_cache = {"kernel": compiled}
        runtime.walk_module_and_get_cubin_data = MagicMock()
        with patch.dict(sys.modules, {module.__name__: module, runtime.__name__: runtime}):
            return namespace["reference_binaries"]("flashinfer", MagicMock())

    def test_tvm_ffi_embedded_cuda_image_is_bound(self):
        result = self.exercise(True)
        self.assertEqual(result[0]["cuda_elf_offsets"], [64])
        self.assertEqual(result[0]["kind"], "JIT_OBJECT_WITH_EMBEDDED_CUDA_ELF")

    def test_host_only_object_is_red(self):
        with self.assertRaisesRegex(RuntimeError, "no CUDA ELF"):
            self.exercise(False)


if __name__ == "__main__":
    unittest.main()
