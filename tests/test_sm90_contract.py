"""Fail-closed dispatch/build/async boundary checks, without a GPU."""
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from actlize_la import gdn_forward,backend_inventory
from actlize_la.gdn_sm90_interface import gdn_chunk_sm90,MATH_CONTRACT


class Contracts(unittest.TestCase):
    def test_algorithm_dispatch_is_explicit_and_complete(self):
        for target in ("cuda_sm90","ppu17"):
            call=Mock(return_value=("O","H"))
            with patch("actlize_la.gdn_interface.import_module",return_value=SimpleNamespace(gdn_chunk_sm90=call)):
                self.assertEqual(gdn_forward(1,2,3,4,5,initial_state=6,algorithm="fused_sm90",backend=target),("O","H"))
                call.assert_called_once_with(1,2,3,4,5,initial_state=6,output_final_state=True,backend=target)
        with self.assertRaisesRegex(ValueError,"explicit"):
            gdn_forward(1,2,3,4,5,algorithm="fused_sm90")

    def test_binary_identity_never_borrows_legacy_or_simulation(self):
        for identity in ("ppu10","cuda_sm90","ppu17-source-check"):
            module=SimpleNamespace(target=identity,math_contract=MATH_CONTRACT,forward=Mock())
            with patch("actlize_la.gdn_sm90_interface._explicit_path",return_value="binary.so"),patch(
                "actlize_la.gdn_sm90_interface._load",return_value=module),self.assertRaisesRegex(RuntimeError,"identity"):
                gdn_chunk_sm90(1,2,3,4,5,backend="ppu17")
            module.forward.assert_not_called()

    def test_output_only_is_not_a_different_math_implementation(self):
        module=SimpleNamespace(target="cuda_sm90",math_contract=MATH_CONTRACT,configuration="control",forward=Mock(return_value=("O",None)))
        with patch("actlize_la.gdn_sm90_interface._explicit_path",return_value="binary.so"),patch(
            "actlize_la.gdn_sm90_interface._load",return_value=module):
            self.assertEqual(gdn_chunk_sm90(1,2,3,4,5,initial_state=6,backend="cuda_sm90",output_final_state=False),("O",None))
            module.forward.assert_called_once_with(1,2,3,4,5,6,False)

    def test_catalog_does_not_claim_device_admission(self):
        j=backend_inventory()
        self.assertEqual(j['algorithms']['fused_sm90']['device_admission'],'UNVERIFIED')
        self.assertEqual(j['algorithms']['original']['targets'],['cuda_sm80','ppu10'])

    def test_ppu_build_dependency_and_flags(self):
        spec=importlib.util.spec_from_file_location("sm90_build",ROOT/"tools/build_gdn_sm90.py")
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        with self.assertRaisesRegex(ValueError,"explicit PPU_CUTLASS_ROOT"):
            module.dependency("ppu17",None)
        with self.assertRaisesRegex(ValueError,"PPU CUTLASS 3.6.0"):
            module.dependency("ppu17",ROOT/"third_party/cutlass")
        for mode in ("native","source-check"):
            f=module.flags("ppu17",mode)
            self.assertIn("-DACOMPUTE_VERSION=10700",f)
            self.assertIn("-gencode=arch=compute_90a,code=sm_90a",f)
            self.assertFalse(any("-D__CUDA_ARCH__" in x for x in f))
            self.assertEqual("-DGDN_SM90_SOURCE_CHECK=1" in f,mode=="source-check")
        # Explicit reuse cannot hide a stale object, changed source or target.
        directory=Path("/workspace")/f"gdn-sm90-reuse-{uuid4().hex}"
        directory.mkdir()
        obj=directory/"object.o";obj.write_bytes(b"synthetic-test-object")
        identity=dict(target="cuda_sm90",source_sha256={"a":"a"},dependency_tree_sha256="header-tree")
        prior=identity|{"device_object_sha256":module.sha(obj)}
        module.admit_reuse(prior,identity,obj)
        for plant in (identity|{"target":"ppu17"},identity|{"source_sha256":{"a":"b"}}):
            with self.assertRaisesRegex(ValueError,"changed"): module.admit_reuse(prior,plant,obj)
        obj.write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError,"hash mismatch"):module.admit_reuse(prior,identity,obj)

    def test_store_retirement_precedes_stage_reuse(self):
        text=(ROOT/"csrc/backends/sm90/cula/kda/sm90/collective/store_tma.hpp").read_text()
        def valid(src):
            self.assertLess(src.index("tma_store_arrive();"),src.index("tma_store_wait<0>();"))
            self.assertLess(src.index("tma_store_wait<0>();"),src.index("pipeline_.consumer_release(src_pipe)"))
            self.assertNotIn("tensormap_replace_global_dim",src)
        valid(text)
        for plant in (text.replace("cute::tma_store_wait<0>();",""),
                      text.replace("cute::tma_store_wait<0>();","pipeline_.consumer_release(src_pipe);\n cute::tma_store_wait<0>();")):
            with self.assertRaises((AssertionError,ValueError)): valid(plant)

    def test_scalar_gate_is_not_tma_byte_counted(self):
        src=(ROOT/"csrc/backends/sm90/cula/kda/sm90/kernel/kernel_kda_fwd.hpp").read_text()
        self.assertIn("alpha_pipeline_params.producer_arv_count = cutlass::NumThreadsPerWarp",src)
        self.assertIn("static constexpr int AlphaConsumers = StateThreads+AuxThreads+32;",src)
        self.assertIn("alpha_pipeline_params.consumer_arv_count = AlphaConsumers;",src)
        self.assertIn("static constexpr int StateThreads = 128*NumStateMmaWarpGroups;",src)
        self.assertIn("static constexpr int AuxThreads = 128*NumAuxMmaWarpGroups;",src)
        self.assertNotIn("alpha_pipeline_params.transaction_bytes",src)

    def test_new_build_graph_and_single_launch_runner(self):
        launch=(ROOT/"csrc/backends/sm90/launch.cu").read_text()
        binding=(ROOT/"csrc/backends/sm90/bindings.cpp").read_text()
        app=(ROOT/"tools/run_sm90_gdn.py").read_text()
        self.assertEqual(launch.count("op.run(stream)"),1)
        self.assertEqual(binding.count("gdn::sm90::launch("),1)
        self.assertEqual(app.count("got=gdn_chunk_sm90("),1)
        for forbidden in (".contiguous()",".zero_()","zeros_like","cudaMemset","gdn_target.cuh"):
            self.assertNotIn(forbidden,binding)
        self.assertNotIn("sm80",(ROOT/"cmake/backends/sm90.cmake").read_text().lower().split("\n",1)[1])


if __name__=="__main__": unittest.main()
