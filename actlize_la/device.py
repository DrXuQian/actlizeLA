"""Measured device metadata or explicit model inputs, never an implicit fallback."""
from dataclasses import dataclass
from functools import lru_cache

from .backends.registry import require_backend


@dataclass(frozen=True)
class DeviceProfile:
    name: str
    sm_count: int
    capability: tuple | None
    source: str = "measured"


def validate_profile_options(mode, backend, sm_count):
    if mode not in ("device", "perfmodel"):
        raise ValueError("mode must be 'device' or 'perfmodel'")
    if mode == "device":
        if sm_count is not None:
            raise ValueError("sm_count is a model input; it requires mode='perfmodel'")
        return
    if backend is None:
        raise ValueError("perfmodel requires an explicit backend; no hardware discovery")
    require_backend(backend)
    if type(sm_count) is not int or sm_count <= 0:
        raise ValueError("perfmodel requires an explicit positive integer sm_count")


@lru_cache(maxsize=None)
def device_profile(index):
    """Hardware query, cached only for measured profiles."""
    import torch
    props = torch.cuda.get_device_properties(index)
    return DeviceProfile(props.name, props.multi_processor_count, (props.major, props.minor))


def get_device_profile(index=None, *, mode="device", backend=None, sm_count=None):
    """In perfmodel mode return configured metadata without importing Torch.

    PPU generation is specified by backend, not a fabricated CUDA capability.
    The configured SM count is metadata, not a device partition or grid override.
    No default model size, device query, timing or profiler invocation is hidden.
    """
    validate_profile_options(mode, backend, sm_count)
    if mode == "perfmodel":
        capability = {"cuda_sm80": (8, 0), "cuda_sm90": (9, 0)}.get(backend)
        return DeviceProfile(backend, sm_count, capability, source="configured")
    return device_profile(index)
