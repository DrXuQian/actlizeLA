"""Single upload includes every registered cell and binds the archived receipts."""
import copy
import io
import json
from pathlib import Path
import sys
import tarfile
import unittest
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
import sweep_ppu10_geometry as sweep


class GeometryPack(unittest.TestCase):
    def setUp(self):
        self.run=(Path("/workspace")/("actlizeLA-geometry-pack-"+uuid4().hex))
        self.run.mkdir();self.run=self.run.resolve()
        self.shapes=sweep.load_shapes(ROOT/"dev/ppu/geometry_shapes.json")
        for name in ("source.diff","submodules.txt","binaries.sha256","comparison.json","comparison.log",
                     "residual-correctness.log","geometry-edge.log","geometry-native.log","l040_wy_geometry.log",
                     "geometry-registration.md","codegen.log"):
            (self.run/name).write_text("SYNTHETIC/NOT-DEVICE-EVIDENCE\n")
        (self.run/"sha.txt").write_text("d"*40+"\n")
        (self.run/"geometry-shapes.json").write_text(json.dumps(self.shapes))
        for name,shape,gate in sweep.cells(self.shapes):
            bundle=self.run/name/"bundle";bundle.mkdir(parents=True)
            order=["wy-control","wy",*["wy-"+d for d in sweep.DELIVERIES[1:]],"fla"]
            status=dict(status="PASS",errors=[],capture_order=order,comparison_origin=dict(source_sha="d"*40),
                        capture_arms={})
            for label,delivery in zip(order,(sweep.CONTROL,*sweep.DELIVERIES,sweep.DELIVERIES[0])):
                role="fla" if label=="fla" else "wy"
                directory="." if label in ("wy","fla") else label
                status["capture_arms"][label]=dict(role=role,directory=directory,wy_delivery=delivery)
                (bundle/directory).mkdir(exist_ok=True)
                receipt=dict(shape=dict(**shape,K=128,V=128),gate=gate,device=dict(uuid="fixture",cu=72),
                             torch="fixture",extension_sha256="binding",library_sha256="library")
                (bundle/directory/(role+".json")).write_text(json.dumps(receipt))
            (bundle/"STATUS.json").write_text(json.dumps(status))
            archive=self.run/name/(name+".tar.gz")
            with tarfile.open(archive,"w:gz") as tar:
                for p in bundle.rglob("*.json"):tar.add(p,arcname=name+"/"+str(p.relative_to(bundle)),recursive=False)
            archive.with_suffix(".gz.sha256").write_text(f"{sweep.sha(archive)}  {archive.name}\n")

    def test_complete_12_cell_pack_and_no_overwrite(self):
        archive=sweep.pack(self.run,self.shapes)
        saved=sweep.sha(archive)
        with tarfile.open(archive,"r:gz") as tar:
            self.assertEqual(sum(n.endswith(".tar.gz") for n in tar.getnames()),12)
            lines=tar.extractfile("geometry.SHA256SUMS").read().decode().splitlines()
            self.assertEqual(len(lines),25)
        with self.assertRaises(ValueError):sweep.pack(self.run,self.shapes)
        self.assertEqual(sweep.sha(archive),saved)

    def test_missing_capture_red(self):
        name,_,_=list(sweep.cells(self.shapes))[-1]
        path=self.run/name/(name+".tar.gz")
        path.rename(path.with_suffix(".held"))
        with self.assertRaises((ValueError,OSError)):sweep.pack(self.run,self.shapes)
        self.assertFalse((self.run/"geometry.tar.gz").exists())

    def test_changed_archived_receipt_is_red(self):
        name,_,_=next(sweep.cells(self.shapes))
        path=self.run/name/"bundle/wy.json"
        record=json.loads(path.read_text());record["unarchived-field"]=True;path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError,"archived receipt differs"):sweep.pack(self.run,self.shapes)

    def test_wrong_device_is_red(self):
        name,_,_=next(sweep.cells(self.shapes))
        path=self.run/name/"bundle/wy.json"
        record=json.loads(path.read_text());record["device"]["uuid"]="other";path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError,"mixed device"):sweep.pack(self.run,self.shapes)

    def test_missing_arm_and_incomplete_status_red(self):
        name,_,_=next(sweep.cells(self.shapes))
        path=self.run/name/"bundle/STATUS.json";original=json.loads(path.read_text())
        for field,value in (("status","INCOMPLETE"),("capture_order",original["capture_order"][:-1])):
            changed=copy.deepcopy(original);changed[field]=value;path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,"incomplete/missing"):sweep.pack(self.run,self.shapes)
        changed=copy.deepcopy(original);del changed["capture_arms"]["wy"]
        path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError,"incomplete/missing"):sweep.pack(self.run,self.shapes)


if __name__=="__main__":unittest.main()
