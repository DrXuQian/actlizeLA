"""Both captures, measured identity and all admission receipts in one tar."""
import hashlib
import io
from pathlib import Path
import subprocess
import tarfile
import unittest
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]


class InverseRegisterPack(unittest.TestCase):
    def setUp(self):
        self.directory = Path("/workspace") / ("actlizeLA-inverse-pack-" + uuid4().hex)
        self.directory.mkdir()
        self.files = ["sha.txt", "binaries.sha256", "comparison.json", "residual-correctness.log",
                      "inverse-register-edge.log", "inverse-register-native.log", "l038_wy_inverse_register.log"]
        for name in self.files:
            (self.directory / name).write_text("fixture\n")
        (self.directory / "sha.txt").write_text("2bde8331b05471293dd834cf51d57ac7e21fd496\n")
        for gate in ("-1.0", "-0.1"):
            name = "acu-inverse-g" + gate
            (self.directory / name).mkdir()
            member = f"{name}/{name}.tar.gz"
            with tarfile.open(self.directory / member, "w:gz") as archive:
                info, content = tarfile.TarInfo("evidence.txt"), gate.encode()
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
            self.files.append(member)

    def pack(self):
        return subprocess.run(["bash", str(ROOT / "tools/pack_ppu10_inverse_register.sh"), str(self.directory)],
                              capture_output=True, text=True)

    def test_one_tar_hashes_both_captures_and_all_receipts(self):
        run = self.pack()
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertIn("captures=2 measurement_sha=2bde833", run.stdout)
        with tarfile.open(self.directory / "inverse-register.tar.gz") as archive:
            self.assertEqual(set(archive.getnames()), {*self.files, "inverse-register.SHA256SUMS"})
            lines = archive.extractfile("inverse-register.SHA256SUMS").read().decode().splitlines()
            self.assertEqual(len(lines), 9)
            for line in lines:
                digest, name = line.split("  ", 1)
                data = archive.extractfile(name).read()
                self.assertEqual(data, (self.directory / name).read_bytes())
                self.assertEqual(hashlib.sha256(data).hexdigest(), digest)

    def test_missing_capture_is_red(self):
        path = self.directory / self.files[-1]
        path.rename(path.with_suffix(".held"))
        self.assertNotEqual(self.pack().returncode, 0)
        self.assertFalse((self.directory / "inverse-register.tar.gz").exists())

    def test_empty_edge_admission_is_red(self):
        (self.directory / "inverse-register-edge.log").write_text("")
        self.assertNotEqual(self.pack().returncode, 0)

    def test_archive_preserved_no_overwrite(self):
        self.assertEqual(self.pack().returncode, 0)
        path = self.directory / "inverse-register.tar.gz"
        saved = path.read_bytes()
        self.assertNotEqual(self.pack().returncode, 0)
        self.assertEqual(path.read_bytes(), saved)

    def test_invalid_sha_not_replaced_with_checkout(self):
        (self.directory / "sha.txt").write_text("unknown\n")
        self.assertNotEqual(self.pack().returncode, 0)


if __name__ == "__main__":
    unittest.main()
