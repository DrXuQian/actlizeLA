import importlib.util
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import unittest
from unittest.mock import Mock,patch
from uuid import uuid4
from actlize_la.gdn_sm90_interface import gdn_chunk_sm90,MATH_CONTRACT
from actlize_la import gdn_forward

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("sm90_builder",ROOT/"tools/build_gdn_sm90.py")
builder=importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class Configurations(unittest.TestCase):
    def test_compiler_and_public_names_match(self):
        header=(ROOT/"csrc/backends/sm90/configuration.cuh").read_text()
        for number,name in enumerate(builder.CONFIGURATIONS):
            self.assertIn('"'+name+'"',header)
            flags=builder.flags("cuda_sm90","native",name)
            self.assertIn(f"-DGDN_SM90_CONFIGURATION={number}",flags)
            self.assertEqual("-DNDEBUG" in flags,name!="control")
        with self.assertRaises(ValueError): builder.flags("cuda_sm90","native","typo")

    def test_every_configuration_is_identity_bound(self):
        for name in builder.CONFIGURATIONS:
            module=SimpleNamespace(target="cuda_sm90",math_contract=MATH_CONTRACT,
                                   configuration=name,forward=Mock(return_value=("O","H")))
            with patch("actlize_la.gdn_sm90_interface._load",return_value=module),patch(
                    "actlize_la.gdn_sm90_interface._explicit_path",return_value="binary.so"):
                self.assertEqual(gdn_chunk_sm90(1,2,3,4,5,backend="cuda_sm90",configuration=name),("O","H"))
                for wrong in (*[n for n in builder.CONFIGURATIONS if n!=name],"typo"):
                    with self.assertRaises((ValueError,RuntimeError)):
                        gdn_chunk_sm90(1,2,3,4,5,backend="cuda_sm90",configuration=wrong)
                module.forward.assert_called_once()

    def test_unidentified_old_experiment_is_not_control(self):
        module=SimpleNamespace(target="cuda_sm90",math_contract=MATH_CONTRACT,forward=Mock())
        with patch("actlize_la.gdn_sm90_interface._load",return_value=module),patch(
                "actlize_la.gdn_sm90_interface._explicit_path",return_value="binary.so"):
            with self.assertRaisesRegex(RuntimeError,"configuration"):
                gdn_chunk_sm90(1,2,3,4,5,backend="cuda_sm90")
        module.forward.assert_not_called()

    def test_configuration_does_not_leak_into_ppu10(self):
        with self.assertRaisesRegex(ValueError,"another backend"):
            gdn_forward(1,2,3,4,5,algorithm="residual",backend="ppu10",configuration="value64")

    def test_cmake_forwards_every_configuration_and_explicit_dependency(self):
        root = Path("/workspace") / f"gdn-sm90-config-contract-{uuid4().hex}"
        for name in builder.CONFIGURATIONS:
            out = root / name
            result = subprocess.run([
                "cmake", "-G", "Unix Makefiles", "-S", str(ROOT), "-B", str(out),
                "-DBUILD_TESTING=OFF", "-DGDN_QSA_BUILD_PPU_TESTS=OFF",
                "-DGDN_QSA_TARGET=cuda_sm90", f"-DGDN_SM90_CONFIGURATION={name}",
                "-DGDN_SM90_CUTLASS_ROOT=/explicit/cutlass/source",
                f"-DPython3_EXECUTABLE={sys.executable}",
            ], capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            command = (out / "CMakeFiles/gdn_fused_sm90.dir/build.make").read_text()
            self.assertIn(f"--configuration {name}", command)
            self.assertIn("--cutlass-root /explicit/cutlass/source", command)
            self.assertFalse((out / "actlize").exists())


if __name__=="__main__": unittest.main()
