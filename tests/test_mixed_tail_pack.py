"""One upload, both gates and admission receipts; no incomplete green archive."""
import hashlib
import io
from pathlib import Path
import subprocess
import tarfile
import unittest
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]


class MixedTailPack(unittest.TestCase):
    def setUp(self):
        self.run_dir = Path("/workspace") / ("actlizeLA-mixed-pack-" + uuid4().hex)
        self.run_dir.mkdir()
        self.files = ["sha.txt", "binaries.sha256", "comparison.json", "residual-correctness.log",
                      "mixed-tail-edge.log", "mixed-tail-native.log", "l037_wy_mixed_tail.log"]
        for name in self.files:
            (self.run_dir / name).write_text("fixture\n")
        (self.run_dir / "sha.txt").write_text("8bbd3dc220367933add1cc5d606e0fa9e54231a0\n")
        for sequence, gate in ((s, g) for s in ("2049", "2111") for g in ("-1.0", "-0.1")):
            name = "acu-mixed-s" + sequence + "-g" + gate
            (self.run_dir / name).mkdir()
            member = f"{name}/{name}.tar.gz"
            with tarfile.open(self.run_dir / member, "w:gz") as archive:
                info, content = tarfile.TarInfo("evidence.txt"), gate.encode()
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
            self.files.append(member)

    def pack(self):
        return subprocess.run(["bash", str(ROOT / "tools/pack_ppu10_mixed_tail.sh"), str(self.run_dir)],
                              capture_output=True, text=True)

    def test_one_tar_preserves_both_captures_receipts_and_hashes(self):
        run = self.pack()
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertIn("captures=4 measurement_sha=8bbd3dc", run.stdout)
        with tarfile.open(self.run_dir / "mixed-tail.tar.gz") as archive:
            self.assertEqual(set(archive.getnames()), {*self.files, "mixed-tail.SHA256SUMS"})
            lines = archive.extractfile("mixed-tail.SHA256SUMS").read().decode().splitlines()
            self.assertEqual(len(lines), 11)
            for line in lines:
                digest, name = line.split("  ", 1)
                data = archive.extractfile(name).read()
                self.assertEqual(data, (self.run_dir / name).read_bytes())
                self.assertEqual(hashlib.sha256(data).hexdigest(), digest)

    def test_missing_capture_is_red(self):
        path = self.run_dir / self.files[-1]
        path.rename(path.with_suffix(".held"))
        self.assertNotEqual(self.pack().returncode, 0)
        self.assertFalse((self.run_dir / "mixed-tail.tar.gz").exists())

    def test_missing_edge_admission_is_red(self):
        (self.run_dir / "mixed-tail-edge.log").write_text("")
        self.assertNotEqual(self.pack().returncode, 0)
        self.assertFalse((self.run_dir / "mixed-tail.tar.gz").exists())

    def test_archive_not_overwritten(self):
        self.assertEqual(self.pack().returncode, 0)
        path = self.run_dir / "mixed-tail.tar.gz"
        old = path.read_bytes()
        self.assertNotEqual(self.pack().returncode, 0)
        self.assertEqual(path.read_bytes(), old)

    def test_invalid_sha_not_replaced_with_checkout(self):
        (self.run_dir / "sha.txt").write_text("unknown\n")
        self.assertNotEqual(self.pack().returncode, 0)
        self.assertFalse((self.run_dir / "mixed-tail.tar.gz").exists())


if __name__ == "__main__":
    unittest.main()

