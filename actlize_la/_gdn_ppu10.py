"""Allocating public GDN adapter for the complete wheel's retained native core.

No conversion through raw logits, VK slots, or a legacy pybind extension.
The prepared serving API remains the allocation-free execution interface.
"""

from functools import lru_cache
from pathlib import Path

from .backends.loading import _file_image


class PPU10Backend:
    def __init__(self, path=None):
        from ._library import get_library_info
        from .serving import NativeLibrary
        from .serving.gdn import _require_native_prefill

        info = get_library_info("ppu10") if path is None else None
        selected = info["library_path"] if info else path
        self.image = _file_image(selected)
        self.library = NativeLibrary(selected, target="ppu10")
        _require_native_prefill(self.library)
        self.manifest_sha256 = info["manifest_sha256"] if info else None
        self.provider = "packaged-serving" if info else "explicit-serving"
        self.image_info()

    def image_info(self):
        from ._library import get_library_info

        info = get_library_info("ppu10") if self.manifest_sha256 else None
        if (_file_image(self.library.path) != self.image or
                (info is not None and info["manifest_sha256"] != self.manifest_sha256)):
            raise RuntimeError("packaged GDN image changed after load; restart the process")
        return [dict(self.image)]

    def residual_full_chunk(self, q, k, v, g, beta, initial, output_final_state):
        import torch
        from .serving.gdn_core import PreparedGDNCore, gdn_core_workspace_specs, _geometry

        if q.ndim != 4 or v.ndim != 4:
            raise ValueError("GDN inputs must have rank four")
        b, t, hk, hv = _geometry(q, k, v, g, beta, initial, v, None)
        if not q.is_cuda:
            raise ValueError("PPU1.0 GDN requires CUDA-compatible device tensors")
        specs = gdn_core_workspace_specs(b, t, hk, hv)
        output = torch.empty_like(v)
        final = (torch.empty((b, hv, 128, 128), device=q.device, dtype=torch.float32)
                 if output_final_state else None)
        work = {name: torch.empty(shape, dtype=dtype, device=q.device)
                for name, (shape, dtype) in specs.items()}
        plan = PreparedGDNCore(self.library, q=q, k=k, v=v, g=g, beta=beta,
                               initial_state=initial, final_state=final,
                               output=output, workspace=work)
        plan()
        return output, final


@lru_cache(maxsize=1)
def packaged_backend():
    return PPU10Backend()


@lru_cache(maxsize=None)
def explicit_backend(path, cwd):
    if not path or not path.strip():
        raise ValueError("ACTLIZE_LA_SERVING_LIBRARY must name a built PPU1.0 library")
    return PPU10Backend((Path(cwd) / path).resolve(strict=True))
