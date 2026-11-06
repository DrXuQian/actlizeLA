"""Preparation/test-only device admission; never queried by captured run()."""

import argparse
import json
import re


def validate_device_target(target, name, capability=None):
    """Fail on mismatches/unknown PPU families; never retarget a binary.

    PPU1.0 (public ISA ppu001, HGGC -arch=ppu_10/ppu001) covers ZW810 and
    ZW610 (and their E variants); ZW890 is PPU1.5. Per-SKU bandwidth/CU
    differences still require their own measurements, never inherited ones.
    """
    family = re.search(r"ZW(810|610|890)(?![0-9])", name.upper())
    measured = {"810": "ppu10", "610": "ppu10", "890": "ppu15"}.get(family[1]) if family else None
    if target in ("ppu10", "ppu15"):
        if measured != target:
            hint = (
                f"; use TARGET={measured}"
                if measured
                else "; unsupported/unidentified PPU family"
            )
            raise ValueError(
                f"device/target mismatch: name={name!r}, target={target}{hint}"
            )
    elif target == "cuda_sm120":
        if measured is not None or "PPU" in name.upper() or capability != (12, 0):
            raise ValueError(
                f"SM120 target requires native SM120 NVIDIA, got {name!r} capability={capability}"
            )
    else:
        raise ValueError(f"unknown serving target: {target}")


def current_device_identity(target):
    import torch

    index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    capability = (
        torch.cuda.get_device_capability(index) if target == "cuda_sm120" else None
    )
    validate_device_target(target, props.name, capability)
    return dict(
        target=target,
        device=index,
        name=props.name,
        cu=props.multi_processor_count,
        torch=torch.__version__,
        runtime=torch.version.cuda,
    )


def main():
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target", required=True, choices=("ppu10", "ppu15", "cuda_sm120")
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print(
            "[serving device admission] SKIP: no device; nothing built or launched",
            flush=True,
        )
        return 3
    identity = current_device_identity(args.target)
    print(
        "[serving device admission] " + json.dumps(identity) + " MATCH/PASS", flush=True
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
