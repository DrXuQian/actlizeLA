"""Admission/route receipt for the bounded ZW810 GDN recheck; never time kernels."""
import argparse
import json
import re
from pathlib import Path


def validate_810(identity):
    from actlize_la.serving.device import validate_device_target
    validate_device_target("ppu10", identity["name"])
    if re.search(r"ZW810(?![0-9])", identity["name"].upper()) is None:
        raise ValueError(f"ZW810 recheck requires ZW810, got {identity['name']!r}")


def expected_route(length, value_heads):
    if length not in (2048, 8192) or value_heads not in (32, 48):
        raise ValueError("outside the eight-cell ZW810 recheck")
    return 1 if (length, value_heads) == (2048, 32) else 0


def check_route(route, length, value_heads):
    wanted = expected_route(length, value_heads)
    name = "solve8" if wanted else "retained"
    if route.get("id") != wanted or route.get("name") != name:
        raise ValueError(f"unexpected core route T{length}/Hv{value_heads}: {route}; expected {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path)
    args = parser.parse_args()
    import torch
    from actlize_la.serving.device import current_device_identity
    if not torch.cuda.is_available():
        print("[ZW810 recheck] SKIP: no device; no measurement", flush=True)
        return 3
    identity = current_device_identity("ppu10")
    validate_810(identity)
    identity["historical_72cu_match"] = identity["cu"] == 72
    print("[ZW810 recheck device] " + json.dumps(identity, sort_keys=True), flush=True)
    if args.library:
        from actlize_la.serving import NativeLibrary
        from actlize_la.serving.gdn_core import gdn_core_route
        library = NativeLibrary(args.library, target="ppu10")
        for length in (2048, 8192):
            for hv in (32, 48):
                route = gdn_core_route(library, 1, length, 16, hv)
                check_route(route, length, hv)
                print("[ZW810 recheck route] " + json.dumps(
                    dict(B=1, T=length, Hk=16, Hv=hv, **route), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
