"""Geometry inventory, actual runner plumbing and fail-closed negatives; no GPU."""
from contextlib import redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import sweep_ppu10_geometry as sweep
import test_ppu_gdn_acu_contract as old
from actlize_la import gdn_residual_interface as api


class GeometryContract(unittest.TestCase):
    def directory(self):
        path=Path("/workspace")/("actlizeLA-geometry-contract-"+uuid4().hex)
        path.mkdir()
        return path.resolve()

    def test_inventory_and_actual_cli_cover_every_cell(self):
        shapes=sweep.load_shapes(ROOT/"dev/ppu/geometry_shapes.json")
        expected={(1,2048,16,32),(1,2048,8,16),(2,2048,16,32),
                  (1,2048,16,64),(1,2049,16,32),(1,2111,16,32)}
        self.assertEqual({tuple(s[k] for k in ("B","S","Hk","Hv")) for s in shapes},expected)
        self.assertEqual(len(list(sweep.cells(shapes))),12)
        for _,s,g in sweep.cells(shapes):
            cmd=sweep.command(Path("/workspace/fake-run"),s,g)
            for option,value in (("--batch",s["B"]),("--sequence",s["S"]),("--q-heads",s["Hk"]),
                                 ("--value-heads",s["Hv"]),("--gate",g)):
                self.assertEqual(cmd[cmd.index(option)+1],str(value))
            self.assertEqual(cmd[cmd.index("--wy-extra-deliveries")+1:cmd.index("--batch")],list(sweep.DELIVERIES[1:]))

    def test_missing_duplicate_or_substituted_admission_cell_is_red(self):
        shapes=sweep.load_shapes(ROOT/"dev/ppu/geometry_shapes.json")
        record=dict(cases=[dict(shape=dict(**s,K=128,V=128),g=g,
                      arms={"wy-"+d:{} for d in (*sweep.DELIVERIES,sweep.CONTROL)}) for _,s,g in sweep.cells(shapes)])
        sweep.check_comparison(record,shapes)
        for wrong in (record["cases"][:-1],record["cases"][:-1]+record["cases"][:1]):
            with self.assertRaises(ValueError): sweep.check_comparison(dict(cases=wrong),shapes)
        wrong=copy.deepcopy(record);wrong["cases"][0]["shape"]["B"]=99
        with self.assertRaises(ValueError): sweep.check_comparison(wrong,shapes)
        wrong=copy.deepcopy(record);del wrong["cases"][0]["arms"]["wy-"+sweep.DELIVERIES[-1]]
        with self.assertRaises(ValueError): sweep.check_comparison(wrong,shapes)

    def test_explicit_geometry_no_hidden_fallback_or_default_change(self):
        inputs=[Mock() for _ in range(5)]
        for x in inputs:x.contiguous.return_value=x
        backend=SimpleNamespace(**{n:Mock(return_value=(1,2)) for n in api.RESIDUAL_ENTRYPOINTS.values()})
        with patch.object(api,"_backend",return_value=backend):
            for profile in sweep.DELIVERIES:
                delivery=api.RESIDUAL_VARIANTS[profile];entry=api.RESIDUAL_ENTRYPOINTS[delivery]
                self.assertEqual(api.gdn_chunk_residual(*inputs,delivery=delivery),(1,2))
                getattr(backend,entry).assert_called_once_with(*inputs,None,True)
                self.assertEqual(api.RESIDUAL_CONTROLS[profile],sweep.CONTROL)
                delattr(backend,entry)
                with self.assertRaises(AttributeError):api.gdn_chunk_residual(*inputs,delivery=delivery)
            backend.residual.assert_not_called()
            api.gdn_chunk_residual(*inputs);backend.residual.assert_called_once()

    def test_state_source_and_constructive_negatives(self):
        r=subprocess.run([sys.executable,str(ROOT/"dev/ppu/check_geometry.py"),"--self-test"],text=True,capture_output=True)
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        self.assertEqual(r.stdout.count("EXPECTED-RED/PASS"),4)

    def test_every_tail_reaches_every_geometry_and_oracle(self):
        import test_ppu_geometry_backend as gate
        with patch.object(sys,"argv",["gate","--extension",__file__]),patch.dict(os.environ), \
             patch.object(gate,"admit") as admit,patch.object(gate.torch.cuda,"set_device"), \
             patch.object(gate.torch.cuda,"get_device_properties",return_value=SimpleNamespace(name="PPU")), \
             patch.object(gate.torch,"set_num_threads"),redirect_stdout(io.StringIO()):
            gate.main()
        self.assertEqual(admit.call_count,256)
        for n in range(65,129):
            for g in (-.1,-1.):
                for initial in (False,True):
                    admit.assert_any_call((1,n,1,2),g,initial,deliveries=["full-chunk","geometry-v32-w8","geometry-v32-w4","geometry-v16-w4"])

    def test_actual_five_arm_collector_does_not_ignore_extra_or_shape(self):
        helper=old.ACUContract()
        for plant in (None,"extra-wrong-shape","extra-changed-output","extra-stale-library"):
            with self.subTest(plant=plant):
                status,commands,_,_=helper.run_mock_capture(self.directory(),control=sweep.CONTROL,
                    subject_delivery=sweep.DELIVERIES[0],extras=sweep.DELIVERIES[1:],
                    batch=2,q_heads=8,value_heads=32,sequence=2111,plant=plant)
                if plant:
                    self.assertEqual(status["status"],"INCOMPLETE",status)
                else:
                    self.assertEqual(status["status"],"PASS",status["errors"])
                    subjects=[c for c in commands if "--set" in c]
                    self.assertEqual(len(subjects),5)
                    self.assertEqual([c[c.index("--wy-delivery")+1] for c in subjects],
                                     [sweep.CONTROL,*sweep.DELIVERIES,sweep.DELIVERIES[0]])
                    self.assertEqual(sum(c[c.index("--role")+1]=="fla" for c in subjects),1)
                    self.assertTrue(all("--kernel-name" not in c and "--launch-count" not in c for c in subjects))

    def test_duplicate_extra_or_different_control_rejected(self):
        for extras in ((sweep.DELIVERIES[0],), (sweep.DELIVERIES[1],)*2, ("residual-v16",)):
            with self.assertRaises(ValueError):
                old.collect.capture_arms(Path("/unused"),"wy",sweep.DELIVERIES[0],sweep.CONTROL,extras)

    def test_legacy_unshaped_admission_cannot_admit_other_batch(self):
        helper=old.ACUContract()
        status,_,_,_=helper.run_mock_capture(self.directory(),control=sweep.CONTROL,
            subject_delivery=sweep.DELIVERIES[0],batch=2,plant="missing-case-shape")
        self.assertEqual(status["status"],"INCOMPLETE")


if __name__=="__main__":unittest.main()
