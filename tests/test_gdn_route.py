"""Default-route execution, image identity and independent negative controls."""
import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

from actlize_la import gdn_forward, get_gdn_route_info
from actlize_la import gdn_residual_interface as residual
from actlize_la.backends import loading, registry


class TensorMetadata:
    dtype = "torch.bfloat16"
    device = "cuda:0"

    def __init__(self, *shape):
        self.shape = shape

    def stride(self):
        result, pitch = [], 1
        for extent in reversed(self.shape):
            result.append(pitch)
            pitch *= extent
        return tuple(reversed(result))

    def contiguous(self):
        return self

    def cpu(self):
        raise AssertionError("route inspection must not read tensor data")


def inputs(length=8192, hv=32):
    q = TensorMetadata(1, length, 16, 128)
    v = TensorMetadata(1, length, hv, 128)
    gate = TensorMetadata(1, length, hv)
    return q, q, v, gate, gate


class Route(unittest.TestCase):
    def assert_default_bound(self, args, **options):
        native = SimpleNamespace(residual=Mock(side_effect=AssertionError("old default")),
                                 residual_full_chunk=Mock(return_value=("O", "H")))
        state = TensorMetadata(1, args[2].shape[2], 128, 128)
        with patch.object(residual, "_backend", return_value=native), patch(
                "actlize_la.device.device_profile", side_effect=AssertionError("no PPU query")):
            info = get_gdn_route_info(*args, initial_state=state, backend="ppu10",
                                      load_binary=False, **options)
            self.assertEqual(gdn_forward(*args, initial_state=state, backend="ppu10", **options), ("O", "H"))
        # Independent expected entry, not derived from the implementation's map.
        self.assertEqual((info["algorithm"], info["delivery"], info["native_entry"]),
                         ("residual", "full-chunk", "residual_full_chunk"))
        native.residual_full_chunk.assert_called_once_with(*args, state, True)
        native.residual.assert_not_called()
        self.assertEqual(info["final_state_dtype"], "fp32")
        self.assertEqual(info["native_status"], "NOT_LOADED")
        self.assertEqual(info["device_execution"], "NOT_RUN")
        self.assertEqual(info["inputs"]["q"]["shape"], list(args[0].shape))
        self.assertEqual([Path(row["path"]).name for row in info["python_sources_on_disk"]],
                         ["gdn_interface.py", "gdn_residual_interface.py", "targets.json"])
        json.dumps(info)

    def test_all_declared_prefill_and_tail_geometries_use_retained_default(self):
        for length in (1, 63, 64, 65, 127, 128, 129, 2048, 8192):
            for hv in (32, 48):
                with self.subTest(length=length, hv=hv):
                    self.assert_default_bound(inputs(length, hv))

    def test_perfmodel_and_explicit_residual_share_execution_policy(self):
        self.assert_default_bound(inputs(), algorithm="residual")
        self.assert_default_bound(inputs(), mode="perfmodel", sm_count=20)

    def test_reverting_default_to_scalar_is_detected(self):
        catalog = copy.deepcopy(registry._catalog())
        catalog["algorithms"]["residual"]["target_entries"]["ppu10"]["default_delivery"] = "scalar"
        with patch.object(registry, "_catalog", return_value=catalog):
            with self.assertRaisesRegex(AssertionError, "old default"):
                self.assert_default_bound(inputs())

    def test_explicit_original_reports_bf16_and_unobserved_reset_dispatch(self):
        with patch.dict(os.environ, {"GDN_QSA_PPU_EXTENSION": "/unused.so"}):
            info = get_gdn_route_info(*inputs(), backend="ppu10", algorithm="original", load_binary=False)
        self.assertEqual(info["native_entry"], "forward_gdn_chunk_auto")
        self.assertEqual(info["final_state_dtype"], "bf16")
        self.assertEqual(info["internal_scan_reset_selection"], "DATA_DEPENDENT_NOT_OBSERVED")

    def test_missing_default_symbol_is_error_not_scalar_fallback(self):
        native = SimpleNamespace(residual=Mock())
        with patch.object(residual, "_backend", return_value=native):
            with self.assertRaisesRegex(RuntimeError, "missing.*residual_full_chunk"):
                get_gdn_route_info(*inputs(), backend="ppu10")
            with self.assertRaisesRegex(RuntimeError, "residual_full_chunk"):
                gdn_forward(*inputs(), backend="ppu10")
        native.residual.assert_not_called()

    def test_inspection_does_not_execute_the_native_entry(self):
        native = SimpleNamespace(residual_full_chunk=Mock(side_effect=AssertionError("no launch")))
        images = [{"path": "/bound.so", "sha256": "test-only"}]
        with patch.object(residual, "_backend", return_value=native), patch(
                "actlize_la.gdn_route.loaded_image_info", return_value=images) as inspect:
            info = get_gdn_route_info(*inputs(), backend="ppu10")
        inspect.assert_called_once_with(native, "libgdn_wy_ppu.so")
        native.residual_full_chunk.assert_not_called()
        self.assertEqual(info["native_images"], images)
        self.assertEqual(info["device_execution"], "NOT_RUN")

    def test_forward_hot_path_never_runs_image_inspection(self):
        native = SimpleNamespace(residual_full_chunk=Mock(return_value=("O", "H")))
        with patch.object(residual, "_backend", return_value=native), patch(
                "actlize_la.gdn_route.loaded_image_info", side_effect=AssertionError("hot path I/O")):
            for _ in range(8):
                self.assertEqual(gdn_forward(*inputs(), backend="ppu10"), ("O", "H"))

    def test_serving_provider_inspects_native_policy_with_actual_gate_dtype(self):
        import torch
        from actlize_la._gdn_ppu10 import PPU10Backend
        native = object.__new__(PPU10Backend)
        query = Mock()
        native.library = SimpleNamespace(lib=SimpleNamespace(actlize_la_ppu10_core_route_v2=query))
        native.provider, native.manifest_sha256 = "explicit-serving", None
        native.image_info = Mock(return_value=[dict(path="/built.so", sha256="fixture")])
        native.residual_full_chunk = Mock(side_effect=AssertionError("inspection executed kernel"))
        for dtype, selected in ((torch.float32, 3), (torch.bfloat16, 0)):
            args = list(inputs(length=2048))
            args[3] = TensorMetadata(1, 2048, 32)
            args[3].dtype = dtype
            query.return_value = selected
            with patch.object(residual, "_backend", return_value=native):
                info = get_gdn_route_info(*args, backend="ppu10")
            self.assertEqual(info["core_route"]["id"], selected)
            self.assertEqual(info["native_entry"], "actlize_la_ppu10_forward_prepared")
            query.assert_called_with(1, 2048, 16, 32, dtype == torch.float32)
        native.residual_full_chunk.assert_not_called()


class Images(unittest.TestCase):
    def setUp(self):
        self.directory = Path("/workspace") / f"actlize-la-route-images-{uuid4().hex}"
        self.directory.mkdir()
        self.binding = self.directory / "_gdn_wy_ppu.fixture.so"
        self.device = self.directory / "libgdn_wy_ppu.so"
        self.binding.write_bytes(b"binding fixture, never executed")
        self.device.write_bytes(b"device fixture, never executed")
        self.module = SimpleNamespace()

    def tearDown(self):
        loading.clear_backend_cache()

    def remember(self, paths=None):
        with patch.object(loading, "_mapped_ppu_libraries", return_value=paths or [str(self.device)]):
            loading._remember_image(self.module, loading._file_image(self.binding))

    def inspect(self, paths=None):
        with patch.object(loading, "_mapped_ppu_libraries", return_value=paths or [str(self.device)]):
            return loading.loaded_image_info(self.module, "libgdn_wy_ppu.so")

    def test_hash_both_binding_and_actually_loaded_kernel_library(self):
        self.remember()
        rows = self.inspect()
        self.assertEqual([r["path"] for r in rows], [str(self.binding), str(self.device)])
        for row in rows:
            self.assertEqual(row["sha256"], hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest())

    def test_replaced_binding_or_device_image_is_red_even_at_same_path(self):
        for path in (self.binding, self.device):
            self.remember()
            path.write_bytes(b"replacement file: not the loaded image")
            with self.assertRaisesRegex(RuntimeError, "changed after load"):
                self.inspect()

    def test_same_named_second_kernel_library_is_ambiguous_not_silently_selected(self):
        extra = self.directory / "old"
        extra.mkdir()
        image = extra / self.device.name
        image.write_bytes(b"older loaded device library")
        self.remember()
        with self.assertRaisesRegex(RuntimeError, "exactly one.*candidates"):
            self.inspect([str(self.device), str(image)])

    def test_unknown_loaded_module_cannot_be_reported_verified(self):
        with self.assertRaisesRegex(RuntimeError, "no load-time"):
            self.inspect()

    def test_deleted_and_escaped_mapped_paths(self):
        path = str(self.directory / "with space" / self.device.name)
        row = "000000-001000 r-xp 000000 00:01 1 " + path.replace(" ", "\\040")
        with patch.object(Path, "read_text", return_value=row):
            self.assertEqual(loading._mapped_ppu_libraries(), [path])
        with patch.object(Path, "read_text", return_value=row + " (deleted)"):
            with self.assertRaisesRegex(RuntimeError, "deleted"):
                loading._mapped_ppu_libraries()


if __name__ == "__main__":
    unittest.main()
