import copy
import importlib.util
from pathlib import Path
import sqlite3
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("nsys_accounting", ROOT / "tools/analyze_sm90_nsys.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Accounting(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(":memory:")
        self.db.executescript("""
        CREATE TABLE StringIds(id INTEGER,value TEXT);
        INSERT INTO StringIds VALUES(1,'FlatKernelTmaWarpSpecializedKdaFwd'),(2,'gate_cumsum');
        CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,text TEXT,textId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,demangledName INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET(start INTEGER,end INTEGER);
        """)
        labels = []
        for i, role in enumerate(("ours-cuda", "ours-ppu-source-check", "cula")):
            label = f"GDN_FORWARD|{role}|000"
            labels.append(label)
            start = i*100000
            self.db.execute("INSERT INTO NVTX_EVENTS VALUES(?,?,?,NULL)", (start,start+90000,label))
            self.db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,1)",(start+20000,start+80000))
            if role == "cula":
                self.db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,2)",(start+1000,start+5000))
                self.db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMSET VALUES(?,?)",(start+6000,start+7000))
        self.receipt = dict(status="CAPTURE_COMPLETE_AWAIT_NSYS_EXTRACTION", calls=labels,
                            scope="H800_CONTROL_NOT_NATIVE_PPU17", shape=[1,2048,16,32,128],
                            gate=-.1, input_sha256="synthetic", device_watch={"errors":[]})

    def tearDown(self):
        self.db.close()

    def test_all_kernels_not_only_fused_are_counted(self):
        data = module.extract(self.db,self.receipt)
        self.assertEqual(data["summary"]["cula"]["kernel_sum_us"]["median"],64)
        self.assertEqual(data["summary"]["cula"]["fused_kernel_us"]["median"],60)
        self.assertEqual(data["summary"]["cula"]["memory_sum_us"]["median"],1)
        self.assertEqual(data["summary"]["cula"]["gpu_gaps_us"]["median"],14)

    def test_missing_call_is_red(self):
        self.db.execute("DELETE FROM NVTX_EVENTS WHERE text LIKE '%cula%'")
        with self.assertRaisesRegex(ValueError,"denominator"):
            module.extract(self.db,self.receipt)

    def test_duplicate_call_is_red(self):
        self.db.execute("INSERT INTO NVTX_EVENTS SELECT * FROM NVTX_EVENTS LIMIT 1")
        with self.assertRaisesRegex(ValueError,"duplicate"):
            module.extract(self.db,self.receipt)

    def test_unassigned_kernel_is_red(self):
        self.db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(400000,410000,2)")
        with self.assertRaisesRegex(ValueError,"unassigned"):
            module.extract(self.db,self.receipt)

    def test_foreign_work_is_red(self):
        receipt = copy.deepcopy(self.receipt)
        receipt["device_watch"]["errors"] = ["foreign GPU PID"]
        with self.assertRaisesRegex(ValueError,"foreign"):
            module.extract(self.db,receipt)

    def test_no_kernel_cannot_become_zero_cost(self):
        self.db.execute("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start<100000")
        with self.assertRaisesRegex(ValueError,"no GPU kernel"):
            module.extract(self.db,self.receipt)

    def test_reducing_manifest_denominator_does_not_hide_a_call(self):
        receipt = copy.deepcopy(self.receipt)
        receipt["calls"] = receipt["calls"][:-1]
        with self.assertRaisesRegex(ValueError,"denominator"):
            module.extract(self.db,receipt)

    def test_helper_kernel_is_not_silently_dropped(self):
        self.db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(205001,205501,2)")
        result = module.extract(self.db,self.receipt)
        self.assertEqual(result["summary"]["cula"]["kernel_sum_us"]["median"],64.5)

    def test_overlap_is_union_not_sum(self):
        self.assertEqual(module.union_ns([(0,10),(5,15),(20,23)]),18)


class LibraryAccounting(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(":memory:")
        self.db.executescript("""
        CREATE TABLE StringIds(id INTEGER,value TEXT);
        INSERT INTO StringIds VALUES(1,'FlatKernelTmaWarpSpecializedKdaFwd'),(2,'prefix'),(3,'cp_main');
        CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,text TEXT,textId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,demangledName INTEGER);
        """)
        labels = []
        for sample in range(12):
            for role in module.LIBRARY_ROLES["flashqla"]:
                start = len(labels)*100000
                label = f"GDN_FORWARD|{role}|{sample:03d}"
                labels.append(label)
                self.db.execute("INSERT INTO NVTX_EVENTS VALUES(?,?,?,NULL)", (start,start+90000,label))
                if role.startswith("ours-"):
                    kernels = [(start+10000, start+60000, 1)]
                else:
                    kernels = [(start+1000, start+2000, 2), (start+10000, start+80000, 3)]
                self.db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?)", kernels)
        self.receipt = dict(status="CAPTURE_COMPLETE_AWAIT_NSYS_EXTRACTION", calls=labels,
                            comparison_family="flashqla", samples=12, scope="H800_CONTROL_NOT_NATIVE_PPU17",
                            shape=[1,2048,16,32,128], gate=-.1, input_sha256="synthetic", device_watch={"errors":[]})

    def tearDown(self):
        self.db.close()

    def test_multiple_reference_kernels_are_all_counted(self):
        result = module.extract(self.db,self.receipt)
        self.assertEqual(result["summary"]["flashqla-auto"]["kernel_sum_us"]["median"],71)
        self.assertNotIn("fused_kernel_us", result["summary"]["flashqla-auto"])
        self.assertEqual(len(result["forwards"]),48)

    def test_omitting_same_call_from_trace_and_receipt_still_fails(self):
        label = self.receipt["calls"].pop()
        row = self.db.execute("SELECT start,end FROM NVTX_EVENTS WHERE text=?",(label,)).fetchone()
        self.db.execute("DELETE FROM NVTX_EVENTS WHERE text=?",(label,))
        self.db.execute("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start>=? AND end<=?", row)
        with self.assertRaisesRegex(ValueError,"denominator"):
            module.extract(self.db,self.receipt)

    def test_extra_reference_helper_changes_sum(self):
        self.db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(202001,202501,2)")
        result = module.extract(self.db,self.receipt)
        self.assertEqual(result["summary"]["flashqla-auto"]["kernel_sum_us"]["samples"][0],71.5)

    def test_foreign_task_invalidates_all_roles(self):
        self.receipt["device_watch"]["errors"] = ["foreign PID"]
        with self.assertRaisesRegex(ValueError,"foreign"):
            module.extract(self.db,self.receipt)


if __name__ == "__main__":
    unittest.main()
