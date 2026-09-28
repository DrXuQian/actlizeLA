"""Metadata-only SM90 selection; H800 evidence is not a PPU/H100 claim."""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path

POLICY_FILE = Path(__file__).with_suffix(".json")


@lru_cache(maxsize=1)
def policy():
    data = json.loads(POLICY_FILE.read_text())
    if data.get("schema") != 1 or data.get("default") not in data.get("configurations", ()):
        raise ValueError("invalid SM90 policy")
    rows = data["rows"]
    keys = [tuple(r["shape"]) for r in rows]
    if len(set(keys)) != len(keys) or any(r["configuration"] not in data["configurations"] for r in rows):
        raise ValueError("duplicate or unknown SM90 policy row")
    return data


def policy_digest():
    return hashlib.sha256(POLICY_FILE.read_bytes()).hexdigest()


@dataclass(frozen=True)
class DeviceProfile:
    name: str
    sm_count: int
    capability: tuple


@dataclass(frozen=True)
class Selection:
    configuration: str
    basis: str
    policy_id: str


@lru_cache(maxsize=None)
def device_profile(index):
    import torch
    props = torch.cuda.get_device_properties(index)
    return DeviceProfile(props.name, props.multi_processor_count, (props.major, props.minor))


def tensor_backend(q):
    device = getattr(q, "device", None)
    if device is None or device.type != "cuda":
        raise ValueError("automatic GDN requires a CUDA/PPU device tensor")
    profile = device_profile(device.index)
    if "PPU" in profile.name.upper():
        raise ValueError("PPU requires an explicit backend; do not guess its hardware generation")
    if profile.capability == (9, 0):
        return "cuda_sm90"
    if profile.capability[0] == 8:
        return "cuda_sm80"
    raise ValueError("automatic GDN target unsupported; select an admitted backend explicitly")


def select(shape, profile):
    """shape=(B,T,Hk,Hv,K,V); no gate values, tensor copies or GPU launches."""
    data = policy()
    if len(shape) != 6 or any(not isinstance(x, int) or x <= 0 for x in shape):
        raise ValueError("SM90 selector requires positive B,T,Hk,Hv,K,V")
    if shape[3] % shape[2] or tuple(shape[4:]) != tuple(data["head_dimensions"]):
        raise ValueError("SM90 requires integral GVA and K=V=128")
    expected = data["device"]
    if profile.capability != tuple(expected["capability"]) or "PPU" in profile.name.upper():
        raise ValueError("SM90 shape policy cannot be applied to another execution target")
    if expected["name_contains"] not in profile.name or profile.sm_count != expected["sm_count"]:
        return Selection(data["default"], "unmeasured-device-default", data["id"])
    for row in data["rows"]:
        if tuple(row["shape"]) == tuple(shape[:4]):
            return Selection(row["configuration"], "h800-measured-shape", data["id"])
    return Selection(data["default"], "unmeasured-shape-default", data["id"])


def selection_for(q, v):
    if q.ndim != 4 or v.ndim != 4 or tuple(q.shape[:2]) != tuple(v.shape[:2]):
        raise ValueError("SM90 requires q/v [B,T,H,D] with matching batch and sequence")
    if q.device.type != "cuda" or q.device != v.device:
        raise ValueError("SM90 q/v must be on the same CUDA device")
    shape = tuple(int(x) for x in (*q.shape[:3], v.shape[2], q.shape[3], v.shape[3]))
    return select(shape, device_profile(q.device.index))
