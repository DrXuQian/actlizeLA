import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("profile_sm90_cula", ROOT / "benchmarks/profile_sm90_cula.py")
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


class Identity(unittest.TestCase):
    def test_wrong_binary_target_mode_or_incomplete_build_is_red(self):
        path = MagicMock()
        manifest = {"complete": True, "extension_sha256": "binary", "target": "cuda_sm90", "mode": "native"}
        source = path.parent.__truediv__.return_value
        with patch.object(profile, "sha", return_value="binary"):
            source.read_text.return_value = json.dumps(manifest)
            self.assertEqual(profile.build_receipt(path, "cuda_sm90", "native"), manifest)
            for delta in ({"complete": False}, {"extension_sha256": "other"},
                          {"target": "ppu17"}, {"mode": "source-check"}):
                source.read_text.return_value = json.dumps(manifest | delta)
                with self.assertRaisesRegex(RuntimeError,"identity mismatch"):
                    profile.build_receipt(path,"cuda_sm90","native")


if __name__ == "__main__":
    unittest.main()
